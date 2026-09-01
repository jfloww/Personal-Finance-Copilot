"""Imported transactions against real PostgreSQL.

These tests verify the claims that belong to PostgreSQL itself: exact NUMERIC
round trips, JSON lineage, and the multiplicity-aware unique constraint
created by Alembic.

The shared integration fixture wraps every test in an outer transaction and
rolls it back, so verification leaves no bank rows, and no accounts, behind.
"""

from __future__ import annotations

import uuid
from datetime import date

import pytest

from offerdelta.application.scope import TenantScope
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    TransactionRepository,
)
from offerdelta.records.transactions import Provenance, TransactionRecord
from tests.integration.conftest import requires_database

pytestmark = requires_database


def _account_name(prefix: str) -> str:
    # Unique per test run so a live shared database can never collide with a
    # name a previous run happened to register.
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


def _account(scope: TenantScope, prefix: str = "integration-checking") -> uuid.UUID:
    return AccountRepository(scope).register(_account_name(prefix)).id


def _record(
    account_id: uuid.UUID,
    *,
    posted_on: date = date(2026, 8, 17),
    merchant: str = "BLUE BOTTLE",
    amount: str = "-4.50",
    occurrence: int = 1,
    external_id: str | None = None,
    provenance: Provenance | None = None,
) -> TransactionRecord:
    return TransactionRecord(
        account_id=account_id,
        posted_on=posted_on,
        description=merchant,
        normalised_merchant=merchant,
        amount=Money.parse(amount),
        external_id=external_id,
        occurrence=occurrence,
        provenance=provenance,
    )


def test_a_record_round_trips_exactly_with_its_provenance(scope: TenantScope) -> None:
    account_id = _account(scope)
    provenance = Provenance(
        source_file="statement.csv", source_line=2, raw_cells={"Amount": "-4.50"}
    )
    record = _record(account_id, provenance=provenance)

    result = TransactionRepository(scope).add_many([record])
    stored = TransactionRepository(scope).get(result.imported_ids[0])

    assert stored is not None
    assert stored.amount == Money.parse("-4.50")
    assert stored.account_id == account_id
    assert stored.source_file == "statement.csv"
    assert stored.source_line == 2
    assert stored.raw_cells == {"Amount": "-4.50"}


def test_a_second_identical_record_is_reported_already_stored_not_written(
    scope: TenantScope,
) -> None:
    account_id = _account(scope)
    record = _record(account_id)
    repo = TransactionRepository(scope)

    first = repo.add_many([record])
    second = repo.add_many([record])

    assert first.imported_count == 1
    assert second.imported_count == 0
    assert second.already_stored_count == 1
    assert repo.count(account_id=account_id) == 1


def test_two_accounts_do_not_share_identities(scope: TenantScope) -> None:
    repo = TransactionRepository(scope)
    first_account = _account(scope, "account-one")
    second_account = _account(scope, "account-two")

    # Same date, merchant, and amount in both accounts: without account_id in
    # the fingerprint this would collide and the second import would be
    # reported as already stored.
    first = repo.add_many([_record(first_account)])
    second = repo.add_many([_record(second_account)])

    assert first.imported_count == 1
    assert second.imported_count == 1
    assert second.already_stored_count == 0
    assert repo.count(account_id=first_account) == 1
    assert repo.count(account_id=second_account) == 1


def test_count_is_correct(scope: TenantScope) -> None:
    """Per account, and summed across the accounts this tenant owns.

    `count()` with no account means "all of mine", not "all of everyone's",
    so `before_total` is this tenant's baseline; the cross-tenant half of
    that claim is asserted in `test_tenant_isolation.py`.
    """
    repo = TransactionRepository(scope)
    account_id = _account(scope)
    other_account = _account(scope, "other-account")
    before_total = repo.count()

    repo.add_many(
        [
            _record(account_id, occurrence=1),
            _record(account_id, merchant="NETFLIX", occurrence=1),
        ]
    )
    repo.add_many([_record(other_account)])

    assert repo.count(account_id=account_id) == 2
    assert repo.count(account_id=other_account) == 1
    assert repo.count() == before_total + 3


def test_a_mixed_account_batch_is_refused(scope: TenantScope) -> None:
    """Dedup below is scoped to one account_id.

    A batch spanning two accounts would check every record but the first
    against the wrong account's stored history, silently missing real
    duplicates in every account after the first. Refusing up front is the
    fix, not letting that happen and hoping the unique constraint saves us.
    """
    first_account = _account(scope, "account-one")
    second_account = _account(scope, "account-two")
    repo = TransactionRepository(scope)

    with pytest.raises(ValidationError, match="one account at a time"):
        repo.add_many(
            [
                _record(first_account),
                _record(second_account, merchant="NETFLIX", amount="-15.99"),
            ]
        )


def test_a_repeated_external_id_is_reported_already_stored_despite_different_details(
    scope: TenantScope,
) -> None:
    """external_id is authoritative for dedup, in both directions.

    Different date, merchant, and amount than the stored row: the fingerprint
    would not match it. Only the external_id does, which is the point - the
    bank's own id is trusted over content when both are present.
    """
    account_id = _account(scope)
    repo = TransactionRepository(scope)
    first = repo.add_many([_record(account_id, external_id="bank-ext-1")])

    second = repo.add_many(
        [
            _record(
                account_id,
                posted_on=date(2026, 9, 1),
                merchant="DIFFERENT MERCHANT",
                amount="-99.99",
                external_id="bank-ext-1",
            )
        ]
    )

    assert first.imported_count == 1
    assert second.imported_count == 0
    assert second.already_stored_count == 1
    assert repo.count(account_id=account_id) == 1


def test_a_new_external_id_is_written_even_when_content_matches_a_stored_row(
    scope: TenantScope,
) -> None:
    """The case that matters most: the bank says it's a different transaction.

    Same date, merchant, and amount as the stored row - content alone cannot
    tell these two charges apart, which is exactly why the occurrence counter
    exists for content-only imports. Here the bank's own id is what says this
    is genuinely the second charge, and occurrence is incremented the same way
    a correctly behaving caller increments it for any repeat: the fingerprint
    is shared identity, occurrence is the position within it. (The database
    enforces this: (account_id, fingerprint, occurrence) is unique regardless
    of external_id, so two rows can never share all three - confirmed against
    the live database, not assumed.)
    """
    account_id = _account(scope)
    repo = TransactionRepository(scope)
    first = repo.add_many([_record(account_id, occurrence=1, external_id="bank-ext-1")])

    second = repo.add_many([_record(account_id, occurrence=2, external_id="bank-ext-2")])

    assert first.imported_count == 1
    assert second.imported_count == 1
    assert second.already_stored_count == 0
    assert repo.count(account_id=account_id) == 2


def test_the_same_external_id_under_a_different_account_is_written(scope: TenantScope) -> None:
    repo = TransactionRepository(scope)
    first_account = _account(scope, "account-one")
    second_account = _account(scope, "account-two")

    first = repo.add_many([_record(first_account, external_id="bank-ext-1")])
    second = repo.add_many([_record(second_account, external_id="bank-ext-1")])

    assert first.imported_count == 1
    assert second.imported_count == 1
    assert second.already_stored_count == 0
    assert repo.count(account_id=first_account) == 1
    assert repo.count(account_id=second_account) == 1


def test_a_mixed_batch_dedupes_each_record_by_its_own_rule(scope: TenantScope) -> None:
    account_id = _account(scope)
    repo = TransactionRepository(scope)

    repo.add_many(
        [
            _record(account_id, external_id="bank-ext-1"),
            _record(account_id, merchant="NETFLIX", amount="-15.99", occurrence=1),
        ]
    )

    batch = [
        # Same external_id as a stored row: dedupe by external_id, even
        # though its content also happens to match.
        _record(account_id, external_id="bank-ext-1"),
        # No external_id, same content as the stored NETFLIX row: dedupe by
        # fingerprint + occurrence.
        _record(account_id, merchant="NETFLIX", amount="-15.99", occurrence=1),
        # No external_id, genuinely new content: written.
        _record(account_id, merchant="SPOTIFY", amount="-9.99", occurrence=1),
        # New external_id, genuinely new content: written.
        _record(account_id, merchant="AMAZON", amount="-25.00", external_id="bank-ext-2"),
    ]
    result = repo.add_many(batch)

    assert result.attempted_count == 4
    assert result.imported_count == 2
    assert result.already_stored_count == 2
    assert repo.count(account_id=account_id) == 4


def test_an_id_less_record_sharing_a_fingerprint_keeps_its_planned_occurrence(
    scope: TenantScope,
) -> None:
    """The offset built for an id-carrying record must not leak onto an id-less one.

    A record with no external_id is identified by (fingerprint, occurrence)
    alone - occurrence *is* its identity. Here the batch's id-less record
    shares a fingerprint with an already-stored row (so the offset dict gets
    a real, non-zero entry for that fingerprint) and lands on a genuinely new
    occurrence of its own. If the offset built for the id-carrying record
    sharing that fingerprint leaked onto it too, it would be stored at the
    wrong occurrence, opening a gap: a later, genuinely-new charge could then
    land on the skipped occurrence and be silently written again.
    """
    account_id = _account(scope)
    repo = TransactionRepository(scope)

    # Seed occurrence=1 for this fingerprint with no external_id, e.g. a
    # prior snapshot-style row.
    repo.add_many([_record(account_id, occurrence=1)])

    batch = [
        # Id-carrying, unrelated content: present only so the offset
        # machinery activates for this batch at all (it is built whenever
        # any record in the batch carries an external_id).
        _record(
            account_id,
            merchant="NETFLIX",
            amount="-15.99",
            occurrence=1,
            external_id="bank-ext-1",
        ),
        # Id-less, SAME fingerprint as the seeded row above, but a genuinely
        # new charge planned at occurrence=2 (not already stored, since only
        # occurrence=1 exists for this fingerprint). Its stored occurrence
        # must be exactly 2, not 2 plus the offset computed for this
        # fingerprint from the seeded row.
        _record(account_id, occurrence=2),
    ]
    result = repo.add_many(batch)

    assert result.imported_count == 2
    stored = [repo.get(identifier) for identifier in result.imported_ids]
    id_less = next(s for s in stored if s is not None and s.normalised_merchant == "BLUE BOTTLE")

    assert id_less.occurrence == 2
