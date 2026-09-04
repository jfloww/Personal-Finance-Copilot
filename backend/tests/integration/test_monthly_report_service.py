"""A month, assembled from what is actually stored."""

from __future__ import annotations

import hashlib
import uuid
from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

import seed_demo
from offerdelta.application.reports.monthly import available_months, monthly_report
from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.application.transactions.enter_transaction import ManualEntry, enter_transaction
from offerdelta.domain.common.evidence import Evidence
from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.parsing import normalise_description
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    ImportBatchRepository,
    TransactionRepository,
    UserRepository,
)
from offerdelta.records.transactions import TransactionRecord
from tests.integration.conftest import requires_database

pytestmark = requires_database

THRESHOLD = Decimal("0.600")
KEY = "chase-checking-5718"


def _seed(scope: TenantScope, rows: list[tuple[str, str]]) -> list[uuid.UUID]:
    """Register an account and hand-enter each `(iso_date, amount)` row.

    Goes through `enter_transaction`, exactly as a person typing rows in
    would, rather than writing `TransactionRow`s directly - so these tests
    exercise the same path real data takes.
    """
    AccountRepository(scope).register("Chase Checking 5718")
    ids: list[uuid.UUID] = []
    for index, (iso_date, amount) in enumerate(rows):
        outcome = enter_transaction(
            scope,
            ManualEntry(
                account_key=KEY,
                posted_on=date.fromisoformat(iso_date),
                description=f"ROW {index}",
                amount=Money.parse(amount),
                repeat=False,
            ),
        )
        assert outcome.transaction_id is not None
        ids.append(outcome.transaction_id)
    return ids


def _seed_snapshot_batch(
    scope: TenantScope, window: tuple[date, date], rows: list[tuple[str, str]]
) -> list[uuid.UUID]:
    """Commit a snapshot import declaring `window`, with `rows` filed under it.

    So that completeness has a real `import_batches` row to read: an account,
    a batch opened in snapshot mode with the given window, and the rows
    written against that batch's id - not just transactions with no batch
    behind them, which `enter_transaction` alone would leave.
    """
    account = AccountRepository(scope).register("Chase Checking 5718")
    checksum = hashlib.sha256(repr((window, rows)).encode()).hexdigest()
    batch, _created = ImportBatchRepository(scope).open(
        account.id,
        source_file="snapshot.csv",
        source_sha256=checksum,
        mode="snapshot",
        window_start=window[0],
        window_end=window[1],
        row_count=len(rows),
    )

    seen: dict[tuple[date, str, str], int] = {}
    records: list[TransactionRecord] = []
    for iso_date, amount in rows:
        posted_on = date.fromisoformat(iso_date)
        description = f"SNAPSHOT ROW {iso_date}"
        merchant = normalise_description(description)
        key = (posted_on, merchant, amount)
        seen[key] = seen.get(key, 0) + 1
        records.append(
            TransactionRecord(
                account_id=account.id,
                posted_on=posted_on,
                description=description,
                normalised_merchant=merchant,
                amount=Money.parse(amount),
                external_id=None,
                occurrence=seen[key],
                provenance=None,
            )
        )

    result = TransactionRepository(scope).add_many(records, batch_id=batch.id)
    return list(result.imported_ids)


def _seed_incremental_batch(scope: TenantScope, window: tuple[date, date]) -> None:
    """Commit an incremental batch that happens to declare a full-month window.

    Nothing at the schema level stops an incremental batch from carrying
    `window_start`/`window_end` - only `ck_import_batches_snapshot_window`
    requires them for snapshot mode - but an incremental batch claims only
    that activity was appended, never that a window is whole. Completeness
    must never read this declaration, so no transactions need to exist under
    it for this helper's purpose.
    """
    account = AccountRepository(scope).register(f"Incremental {uuid.uuid4().hex[:8]}")
    checksum = hashlib.sha256(repr(("incremental", window)).encode()).hexdigest()
    ImportBatchRepository(scope).open(
        account.id,
        source_file="incremental.csv",
        source_sha256=checksum,
        mode="incremental",
        window_start=window[0],
        window_end=window[1],
        row_count=0,
    )


def test_the_root_equals_the_sum_of_every_stored_row_for_that_month(
    scope: TenantScope,
) -> None:
    ids = _seed(
        scope,
        [("2026-03-01", "5000.00"), ("2026-03-02", "-12.34"), ("2026-03-03", "-9.99")],
    )
    repo = TransactionRepository(scope)
    repo.confirm_label(ids[0], "INCOME")
    repo.record_suggestion(
        ids[1], label="LIVING_DINING", source="llm", confidence=Decimal("0.9"), by="x"
    )
    # ids[2] stays unclassified on purpose.

    report = monthly_report(scope, 2026, 3, threshold=THRESHOLD)

    assert report.tree.amount == Money.parse("4977.67")
    assert report.coverage.rows == 3
    assert report.coverage.classified == 2


def test_an_abstained_row_does_not_count_as_classified(scope: TenantScope) -> None:
    """Fix 3: `classified` must agree with where the tree files the row.

    `'UNKNOWN'` is a non-null `effective_label`, so counting any non-null
    label as classified would say this row is classified while
    `build_monthly_report` files it under
    `unclassified/examined_no_answer` - the exact disagreement this pins
    against. It also matches this repository's one definition of coverage,
    `evaluation.metrics.ClassificationReport.coverage`.
    """
    ids = _seed(scope, [("2026-03-01", "5000.00"), ("2026-03-02", "-9.99")])
    repo = TransactionRepository(scope)
    repo.confirm_label(ids[0], "INCOME")
    repo.record_suggestion(
        ids[1], label="UNKNOWN", source="llm", confidence=Decimal("0.000"), by="x"
    )

    report = monthly_report(scope, 2026, 3, threshold=THRESHOLD)

    assert report.coverage.rows == 2
    assert report.coverage.classified == 1

    unclassified = next(child for child in report.tree.children if child.code == "unclassified")
    examined_no_answer = next(
        child for child in unclassified.children if child.code == "examined_no_answer"
    )
    assert examined_no_answer.amount == Money.parse("-9.99")


def test_awaiting_review_agrees_with_the_repository_for_the_same_threshold(
    scope: TenantScope,
) -> None:
    ids = _seed(scope, [("2026-03-01", "-1.00"), ("2026-03-02", "-2.00")])
    repo = TransactionRepository(scope)
    repo.record_suggestion(
        ids[0], label="LIVING_DINING", source="llm", confidence=Decimal("0.95"), by="x"
    )
    repo.record_suggestion(
        ids[1], label="LIVING_DINING", source="llm", confidence=Decimal("0.20"), by="x"
    )

    report = monthly_report(scope, 2026, 3, threshold=THRESHOLD)

    assert report.coverage.awaiting_review == len(repo.awaiting_review(THRESHOLD))
    assert report.coverage.awaiting_review == 1


def test_a_month_no_declared_window_covers_is_partial(scope: TenantScope) -> None:
    _seed(scope, [("2026-03-05", "-1.00")])
    months = {(m.year, m.month): m for m in available_months(scope, threshold=THRESHOLD)}
    assert months[(2026, 3)].complete is False


def test_a_month_a_snapshot_window_covers_completely_is_complete(
    scope: TenantScope,
) -> None:
    _seed_snapshot_batch(
        scope, window=(date(2026, 3, 1), date(2026, 3, 31)), rows=[("2026-03-05", "-1.00")]
    )
    months = {(m.year, m.month): m for m in available_months(scope, threshold=THRESHOLD)}
    assert months[(2026, 3)].complete is True


def test_an_incremental_batch_spanning_the_month_never_makes_it_complete(
    scope: TenantScope,
) -> None:
    """The distinction the import path was built around: appended, not whole."""
    _seed(scope, [("2026-03-05", "-1.00")])
    _seed_incremental_batch(scope, window=(date(2026, 3, 1), date(2026, 3, 31)))

    months = {(m.year, m.month): m for m in available_months(scope, threshold=THRESHOLD)}

    assert months[(2026, 3)].complete is False


def test_awaiting_review_agrees_between_available_months_and_monthly_report(
    scope: TenantScope,
) -> None:
    """The property Fix 1 exists for: one threshold, one meaning, everywhere.

    `ids[0]` is confidently suggested (0.95, above `THRESHOLD`) and never
    confirmed - the exact shape that used to disagree between the two
    functions when `available_months` counted every unconfirmed row instead
    of evaluating the same threshold `monthly_report` was given.
    """
    ids = _seed(scope, [("2026-03-01", "-1.00"), ("2026-03-02", "-2.00")])
    repo = TransactionRepository(scope)
    repo.record_suggestion(
        ids[0], label="LIVING_DINING", source="llm", confidence=Decimal("0.95"), by="x"
    )
    # ids[1] stays untouched: no suggestion, no confirmation.

    listed = {(m.year, m.month): m for m in available_months(scope, threshold=THRESHOLD)}
    report = monthly_report(scope, 2026, 3, threshold=THRESHOLD)

    assert listed[(2026, 3)].awaiting_review == report.coverage.awaiting_review == 1


def test_one_tenant_never_sees_another_tenants_month(
    scope: TenantScope, other_scope: TenantScope
) -> None:
    _seed(scope, [("2026-03-01", "-1.00")])

    mine = monthly_report(scope, 2026, 3, threshold=THRESHOLD)
    theirs = monthly_report(other_scope, 2026, 3, threshold=THRESHOLD)

    assert mine.coverage.rows == 1
    assert theirs.coverage.rows == 0
    assert theirs.tree.amount == Money.zero()


# --------------------------------------------------------- evidence, over real stored rows


def test_confirming_a_suggested_row_turns_the_root_user_confirmed(scope: TenantScope) -> None:
    """Fix 4: the evidence chain, proved over rows that went through the
    repository - not a unit test that hand-feeds `confirmed=` to
    `ClassifiedTransaction` and never touches `record_suggestion` or
    `confirm_label` at all.

    Before this test existed, `_to_classified` could have hardcoded
    `confirmed=True` and every test in the suite would still have passed.
    """
    ids = _seed(scope, [("2026-03-01", "-12.34")])
    repo = TransactionRepository(scope)
    repo.record_suggestion(
        ids[0], label="LIVING_DINING", source="llm", confidence=Decimal("0.910"), by="x"
    )

    suggested = monthly_report(scope, 2026, 3, threshold=THRESHOLD)
    assert suggested.tree.evidence is Evidence.ASSUMED

    repo.confirm_label(ids[0], "LIVING_DINING")

    confirmed = monthly_report(scope, 2026, 3, threshold=THRESHOLD)
    assert confirmed.tree.evidence is Evidence.USER_CONFIRMED


def test_the_seeded_demo_tenant_builds_a_real_tree(session: Session) -> None:
    """The gap the last task deferred: no test built the actual tree over
    `seed_demo`'s own fixtures - every fixture the domain-layer tests use is
    hand-built instead. This runs the real seeding function (not a copy of
    its data) and hands the result to the real report builder, so a change
    that breaks either one - or the root-equals-every-row invariant Fix 1
    added, over a larger and more realistic row set than the unit tests use
    - would fail here.
    """
    tenant = seed_demo._TENANTS[0]
    seed_demo._seed_tenant(session, tenant, "throwaway-demo-password")

    stored_user = UserRepository(session).by_email(tenant.email)
    assert stored_user is not None
    scope = TenantScope(
        session=session, user=AuthenticatedUser(id=stored_user.id, email=stored_user.email)
    )

    report = monthly_report(scope, 2026, 1, threshold=THRESHOLD)

    january = [
        seed
        for seed in tenant.transactions
        if (seed.posted_on.year, seed.posted_on.month) == (2026, 1)
    ]
    expected_total = Money.zero()
    for seed in january:
        expected_total = expected_total + Money.parse(seed.amount)
    expected_classified = sum(1 for seed in january if seed.label is not None)

    assert report.tree.amount == expected_total
    assert report.coverage.rows == len(january)
    assert report.coverage.classified == expected_classified
    # One January row is deliberately left unconfirmed (see the module
    # docstring), so the demo's own root is honestly ASSUMED, not SOURCED or
    # USER_CONFIRMED - it is not the empty, childless case Fix 7 covers.
    assert report.tree.evidence is Evidence.ASSUMED
