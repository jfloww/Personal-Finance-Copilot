# Auditable Monthly Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give imported transactions a stored classification, then build a monthly report whose root is the sum of every row in the window, so completeness is structural rather than checked.

**Architecture:** A classification is stored in two layers — the machine's `suggested_label` with its provenance, and the person's `confirmed_label`, which re-classification never touches. A local CLI runs rules first and Claude second and writes suggestions; the deployed service only reads them. The report is a `DerivationNode` tree rooted at every row, so a dropped or double-counted row means no report at all, and `Evidence` propagation makes an unreviewed row visible all the way to the root.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, Alembic, PostgreSQL, Anthropic API, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-monthly-report-design.md`

## Global Constraints

- **Domain stays standard-library only.** `offerdelta.domain` must not import fastapi, pydantic, sqlalchemy, alembic, httpx, or requests.
- **Layers:** `offerdelta.api` → `offerdelta.application` → `offerdelta.domain`. A fifth contract forbids `offerdelta.api` importing `offerdelta.infrastructure.postgres.repositories`; routes reach data through use cases.
- **mypy is `strict = true` with `disallow_any_explicit = true`** over `src` and `tests`. No bare `Any`.
- **ruff line-length is 100. All imports at module top level** — `PLC0415` is enforced and CI runs `ruff check`.
- **No `__init__.py` in test directories.**
- **Four file lists must name the same set:** `PY_FILES` in the repo-root `Makefile`, mypy `files` in `backend/pyproject.toml`, and **both** ruff invocations in `.github/workflows/ci.yml`. A new root script also needs an entry in ruff's `known-first-party`.
- **CI supplies both `CONNECTION_STRING` and `JWT_SECRET`** and fails rather than skips when either is missing.
- **404, never 403,** for another tenant's resource.
- **Published evaluation figures must not move.** V2's numbers depend on how the LLM request is constructed; Task 2 changes an input type and is guarded by a payload-equality test for exactly this reason.
- **Coverage is not accuracy.** The report states how many rows it could account for. It must never present that as a statement about label correctness.
- **The full gate is `make check`** — `lint`, `types`, `arch`, `test`.
- **Never write to the default database from a task.** It holds 742 real bank transactions. Migrations against it are run by the controller.

---

### Task 1: Classification columns

Seven nullable columns and one migration. Nothing is backfilled: existing rows are unclassified, which is what they are.

**Files:**
- Modify: `backend/src/offerdelta/infrastructure/postgres/models.py`
- Create: `backend/migrations/versions/<generated>_transaction_classification.py`
- Create: `backend/tests/integration/test_classification_migration.py`

**Interfaces:**
- Consumes: nothing
- Produces: `TransactionRow.suggested_label`, `.suggested_source`, `.suggested_confidence`, `.suggested_by`, `.suggested_at`, `.confirmed_label`, `.confirmed_at`

- [ ] **Step 1: Write the failing migration test**

Create `backend/tests/integration/test_classification_migration.py`. Follow the shape of `test_users_migration.py`, which already drives Alembic against the `scratch_schema` fixture — read it first and reuse its `_upgrade_to` helper verbatim rather than inventing a second one.

```python
"""Classification columns.

Every column is nullable and nothing is backfilled: a row that has never
been categorised is unclassified, and saying so is more honest than
inventing a label for it at migration time.
"""

from __future__ import annotations

import uuid

from sqlalchemy import Engine, text

from tests.integration.conftest import requires_database

pytestmark = requires_database

_EXPECTED = {
    "suggested_label",
    "suggested_source",
    "suggested_confidence",
    "suggested_by",
    "suggested_at",
    "confirmed_label",
    "confirmed_at",
}


def test_the_columns_exist_and_are_all_nullable(engine: Engine, scratch_schema: str) -> None:
    _upgrade_to(engine, scratch_schema, "head")

    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{scratch_schema}"'))
        rows = conn.execute(
            text(
                "SELECT column_name, is_nullable FROM information_schema.columns "
                "WHERE table_schema = :schema AND table_name = 'transactions'"
            ),
            {"schema": scratch_schema},
        ).all()

    present = {name: nullable for name, nullable in rows}
    assert _EXPECTED <= set(present)
    assert all(present[name] == "YES" for name in _EXPECTED)


def test_existing_rows_are_left_unclassified(engine: Engine, scratch_schema: str) -> None:
    """No backfill. An unlabelled row must not acquire a label from a migration."""
    _upgrade_to(engine, scratch_schema, "b275cd06fc0b")

    user_id, account_id = uuid.uuid4(), uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{scratch_schema}"'))
        conn.execute(
            text(
                "INSERT INTO users (id, email, display_name, is_active, created_at) "
                "VALUES (:id, 'a@example.test', 'A', true, now())"
            ),
            {"id": user_id},
        )
        conn.execute(
            text(
                "INSERT INTO accounts (id, user_id, key, display_name, created_at) "
                "VALUES (:id, :user_id, 'chase-checking-5718', 'Chase', now())"
            ),
            {"id": account_id, "user_id": user_id},
        )
        conn.execute(
            text(
                "INSERT INTO transactions (id, imported_at, account_id, posted_on, "
                "description, normalised_merchant, currency, amount, fingerprint, "
                "fingerprint_version, occurrence) VALUES (:id, now(), :account_id, "
                "'2026-03-01', 'BLUE BOTTLE', 'BLUE BOTTLE', 'USD', -12.34, "
                "'0000000000000000000000000000000a', 1, 1)"
            ),
            {"id": uuid.uuid4(), "account_id": account_id},
        )

    _upgrade_to(engine, scratch_schema, "head")

    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{scratch_schema}"'))
        labels = conn.execute(
            text("SELECT suggested_label, confirmed_label FROM transactions")
        ).one()

    assert labels == (None, None)
```

Copy `_upgrade_to` from `tests/integration/test_users_migration.py` into this module (top-level import block, no function-level imports).

- [ ] **Step 2: Run it and watch it fail**

Run: `cd backend && uv run pytest tests/integration/test_classification_migration.py -v`
Expected: FAIL — the columns do not exist.

- [ ] **Step 3: Add the columns to the model**

In `models.py`, inside `TransactionRow`, after `raw_cells`:

```python
    #: What a categoriser proposed. Overwritten freely by re-classification.
    #: NULL means never examined; the literal 'UNKNOWN' means examined and
    #: declined, which is abstention and a different fact entirely.
    suggested_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    suggested_source: Mapped[str | None] = mapped_column(String(16), nullable=True)
    suggested_confidence: Mapped[Decimal | None] = mapped_column(Numeric(4, 3), nullable=True)

    #: Which ruleset or which model-and-prompt produced it, so a report can
    #: answer what made its numbers.
    suggested_by: Mapped[str | None] = mapped_column(String(120), nullable=True)
    suggested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: A person's decision. Re-classification never writes here, so erasing
    #: one is not something a caller has to remember not to do.
    confirmed_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confirmed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
```

Add `Numeric` to the SQLAlchemy imports if it is not already there.

- [ ] **Step 4: Generate and fill the migration**

Run: `cd backend && uv run alembic revision -m "transaction classification"`

`down_revision` must be `"b275cd06fc0b"`. The body is seven `add_column` calls and seven `drop_column` calls in reverse. There is no data step at all:

```python
def upgrade() -> None:
    """Give a transaction somewhere to record what it was classified as.

    Every column is nullable and nothing is backfilled. A row nobody has
    categorised is unclassified, and a migration that invented a label for
    it would be asserting something no categoriser ever said.
    """
    op.add_column("transactions", sa.Column("suggested_label", sa.String(64), nullable=True))
    op.add_column("transactions", sa.Column("suggested_source", sa.String(16), nullable=True))
    op.add_column(
        "transactions", sa.Column("suggested_confidence", sa.Numeric(4, 3), nullable=True)
    )
    op.add_column("transactions", sa.Column("suggested_by", sa.String(120), nullable=True))
    op.add_column(
        "transactions", sa.Column("suggested_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("transactions", sa.Column("confirmed_label", sa.String(64), nullable=True))
    op.add_column(
        "transactions", sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    for column in (
        "confirmed_at",
        "confirmed_label",
        "suggested_at",
        "suggested_by",
        "suggested_confidence",
        "suggested_source",
        "suggested_label",
    ):
        op.drop_column("transactions", column)
```

- [ ] **Step 5: Run the tests, the gate, and commit**

Run: `cd backend && uv run pytest tests/integration/test_classification_migration.py -v`, then the full CI list (`ruff check`, `ruff format --check`, `mypy`, `lint-imports`, `pytest`).

**Do not run `alembic upgrade head` against the default database.** The controller does that.

```bash
git add backend/src/offerdelta/infrastructure/postgres/models.py backend/migrations/versions/ backend/tests/integration/test_classification_migration.py
git commit -m "Schema: where a transaction records what it was classified as"
```

---

### Task 2: A narrow categoriser input

Both categorisers read exactly four fields off their input. The protocol nevertheless takes `LabelledTransaction`, which carries a gold label — an evaluation type that production code has no business constructing. Extract the projection.

**Files:**
- Create: `backend/src/offerdelta/domain/transactions/view.py`
- Modify: `backend/src/offerdelta/evaluation/dataset.py`
- Modify: `backend/src/offerdelta/evaluation/categorisers.py`
- Modify: `backend/src/offerdelta/evaluation/llm_categoriser.py`
- Modify: `backend/src/offerdelta/evaluation/rule_baseline.py`
- Create: `backend/tests/unit/domain/transactions/test_view.py`
- Create: `backend/tests/unit/llm/test_request_is_unchanged.py`

**Interfaces:**
- Consumes: nothing
- Produces: `TransactionView(normalised_merchant: str, raw_description: str, amount: Money, account_type: str)` in `offerdelta.domain.transactions.view`; `LabelledTransaction.view -> TransactionView`; `Categoriser.predict(view: TransactionView) -> Prediction` and `.predict_many(views: Sequence[TransactionView]) -> list[Prediction]`

- [ ] **Step 1: Write the payload-equality guard first**

This is the test that makes the refactor safe. V2's published figures depend on how the request is built; if the bytes sent to the model change, those numbers stop describing the code. Create `backend/tests/unit/llm/test_request_is_unchanged.py`:

```python
"""The refactor must not change what the model receives.

Published V2 figures describe requests built a particular way. This asserts
the tool payload for a given record is exactly what it was before the input
type narrowed - so the numbers keep describing the code that produced them.
"""

from __future__ import annotations

from datetime import date

from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.view import TransactionView
from offerdelta.evaluation.dataset import LabelledTransaction

RECORD = LabelledTransaction(
    transaction_id="chase-checking-5718:abc:1",
    posted_on=date(2026, 3, 1),
    raw_description="SQ *BLUE BOTTLE #417",
    normalised_merchant="SQ BLUE BOTTLE",
    amount=Money.parse("-6.75"),
    account_type="checking",
    source="chase",
    bank_format="chase-checking-v1",
    primary_label="LIVING_DINING",
)

#: The four values the categorisers actually read, pinned literally.
EXPECTED = {
    "merchant": "SQ BLUE BOTTLE",
    "raw_description": "SQ *BLUE BOTTLE #417",
    "amount": "-6.75",
    "account_type": "checking",
}


def test_the_view_carries_exactly_the_four_fields() -> None:
    view = RECORD.view
    assert view.normalised_merchant == EXPECTED["merchant"]
    assert view.raw_description == EXPECTED["raw_description"]
    assert str(view.amount.amount) == EXPECTED["amount"]
    assert view.account_type == EXPECTED["account_type"]


def test_a_view_built_by_hand_is_indistinguishable_from_one_off_a_record() -> None:
    """Production builds views from stored rows; evaluation builds them from
    labelled records. A categoriser must not be able to tell which it got."""
    by_hand = TransactionView(
        normalised_merchant="SQ BLUE BOTTLE",
        raw_description="SQ *BLUE BOTTLE #417",
        amount=Money.parse("-6.75"),
        account_type="checking",
    )
    assert by_hand == RECORD.view
```

Then find the test in `backend/tests/unit/llm/` that asserts on the LLM request body (grep for `tool` or `input` in that directory) and confirm it still passes untouched after this task. Name it in your report.

- [ ] **Step 2: Run it and watch it fail**

Run: `cd backend && uv run pytest tests/unit/llm/test_request_is_unchanged.py -v`
Expected: FAIL — `offerdelta.domain.transactions.view` does not exist.

- [ ] **Step 3: Write the view**

Create `backend/src/offerdelta/domain/transactions/view.py`:

```python
"""What a categoriser is allowed to see.

Deliberately four fields. A categoriser that could read a balance, a
neighbouring row, or a gold label would be scored on information the
running system does not have, and the score would not transfer.

It also exists so production code never has to construct a
`LabelledTransaction`: that type carries a human's answer, which a
categoriser must never be handed.
"""

from __future__ import annotations

from dataclasses import dataclass

from offerdelta.domain.common.money import Money


@dataclass(frozen=True)
class TransactionView:
    """The projection every categoriser predicts from."""

    normalised_merchant: str
    raw_description: str
    amount: Money
    account_type: str
```

- [ ] **Step 4: Add `.view` to `LabelledTransaction` and narrow the protocol**

In `dataset.py`, add to `LabelledTransaction`:

```python
    @property
    def view(self) -> TransactionView:
        """What a categoriser sees. Never the label."""
        return TransactionView(
            normalised_merchant=self.normalised_merchant,
            raw_description=self.raw_description,
            amount=self.amount,
            account_type=self.account_type,
        )
```

In `categorisers.py`, change the protocol to take `TransactionView`. In `llm_categoriser.py` and `rule_baseline.py`, change `predict`/`predict_many` signatures to accept `TransactionView` and read the fields directly off it — the four attribute names are identical, so the bodies barely move. `rule_baseline`'s *fitting* function still takes `LabelledTransaction`, because fitting genuinely needs the gold label; only prediction narrows.

Update every call site: `run_evaluation.py` and anything under `src/offerdelta/evaluation/` that calls `predict` or `predict_many` now passes `record.view`.

- [ ] **Step 5: Prove the figures did not move**

Run the whole evaluation-related test suite and quote the result:
`cd backend && uv run pytest tests/unit/evaluation tests/unit/llm -v`

Then run the full gate. If any test that pins an LLM request body fails, **stop and report** — that means the payload changed and the refactor is not safe as written.

- [ ] **Step 6: Commit**

```bash
git add backend/src/offerdelta/domain/transactions/view.py backend/src/offerdelta/evaluation backend/tests
git commit -m "Categorisers predict from a view, not from a labelled record"
```

---

### Task 3: Reading and writing a classification

Tenant-scoped repository methods. Nothing here calls a model.

**Files:**
- Modify: `backend/src/offerdelta/infrastructure/postgres/repositories.py`
- Create: `backend/tests/integration/test_classification_repository.py`

**Interfaces:**
- Consumes: `TenantScope`, `TransactionView` (Task 2), the columns from Task 1
- Produces, on `TransactionRepository`:
  - `unclassified(limit: int | None = None) -> list[StoredTransaction]`
  - `record_suggestion(transaction_id: uuid.UUID, *, label: str, source: str, confidence: Decimal, by: str, now: datetime | None = None) -> None`
  - `confirm_label(transaction_id: uuid.UUID, label: str, *, now: datetime | None = None) -> None`
  - `for_month(year: int, month: int) -> list[StoredTransaction]`
  - `awaiting_review(threshold: Decimal, *, month: tuple[int, int] | None = None) -> list[StoredTransaction]`
- `StoredTransaction` gains `suggested_label`, `suggested_source`, `suggested_confidence`, `suggested_by`, `confirmed_label`, and a property `effective_label: str | None` returning `confirmed_label or suggested_label`.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/integration/test_classification_repository.py`:

```python
"""Storing what a categoriser said, and what a person said instead."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.application.transactions.enter_transaction import ManualEntry, enter_transaction
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    TransactionRepository,
    UserRepository,
)
from tests.integration.conftest import requires_database

pytestmark = requires_database

KEY = "chase-checking-5718"


def _scope(session: Session, email: str) -> TenantScope:
    stored = UserRepository(session).create(email, f"Owner {email}")
    return TenantScope(session=session, user=AuthenticatedUser(id=stored.id, email=stored.email))


def _one_transaction(scope: TenantScope, description: str = "BLUE BOTTLE") -> uuid.UUID:
    AccountRepository(scope).register("Chase Checking 5718")
    outcome = enter_transaction(
        scope,
        ManualEntry(
            account_key=KEY,
            posted_on=date(2026, 3, 1),
            description=description,
            amount=Money.parse("-12.34"),
            repeat=False,
        ),
    )
    assert outcome.transaction_id is not None
    return outcome.transaction_id


def test_a_new_transaction_is_unclassified(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    _one_transaction(scope)

    pending = TransactionRepository(scope).unclassified()
    assert len(pending) == 1
    assert pending[0].suggested_label is None
    assert pending[0].effective_label is None


def test_a_suggestion_is_recorded_with_its_provenance(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)
    repo = TransactionRepository(scope)

    repo.record_suggestion(
        txn_id,
        label="LIVING_DINING",
        source="llm",
        confidence=Decimal("0.910"),
        by="claude-haiku-4-5:categorise/v3",
    )

    stored = repo.for_month(2026, 3)[0]
    assert stored.suggested_label == "LIVING_DINING"
    assert stored.suggested_source == "llm"
    assert stored.suggested_confidence == Decimal("0.910")
    assert stored.suggested_by == "claude-haiku-4-5:categorise/v3"
    assert stored.effective_label == "LIVING_DINING"


def test_a_confirmation_outranks_a_suggestion(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)
    repo = TransactionRepository(scope)

    repo.record_suggestion(
        txn_id, label="LIVING_OTHER", source="llm", confidence=Decimal("0.400"), by="x"
    )
    repo.confirm_label(txn_id, "LIVING_DINING")

    stored = repo.for_month(2026, 3)[0]
    assert stored.confirmed_label == "LIVING_DINING"
    assert stored.suggested_label == "LIVING_OTHER"
    assert stored.effective_label == "LIVING_DINING"


def test_a_later_suggestion_does_not_erase_a_confirmation(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)
    repo = TransactionRepository(scope)

    repo.confirm_label(txn_id, "LIVING_DINING")
    repo.record_suggestion(
        txn_id, label="TRANSFER", source="llm", confidence=Decimal("0.990"), by="x"
    )

    stored = repo.for_month(2026, 3)[0]
    assert stored.effective_label == "LIVING_DINING"


def test_a_label_outside_the_space_is_refused(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)

    with pytest.raises(ValidationError, match="not a label"):
        TransactionRepository(scope).confirm_label(txn_id, "COFFEE")


def test_unknown_is_a_valid_label_and_is_not_the_same_as_never_examined(
    session: Session,
) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)
    repo = TransactionRepository(scope)

    repo.record_suggestion(
        txn_id, label="UNKNOWN", source="llm", confidence=Decimal("0.000"), by="x"
    )

    assert repo.unclassified() == []
    assert repo.for_month(2026, 3)[0].suggested_label == "UNKNOWN"


def test_awaiting_review_takes_low_confidence_and_unknown_and_never_examined(
    session: Session,
) -> None:
    scope = _scope(session, "a@example.test")
    repo = TransactionRepository(scope)
    AccountRepository(scope).register("Chase Checking 5718")

    ids = []
    for day, description in ((1, "CONFIDENT"), (2, "UNSURE"), (3, "ABSTAINED"), (4, "UNSEEN")):
        outcome = enter_transaction(
            scope,
            ManualEntry(
                account_key=KEY,
                posted_on=date(2026, 3, day),
                description=description,
                amount=Money.parse("-1.00"),
                repeat=False,
            ),
        )
        assert outcome.transaction_id is not None
        ids.append(outcome.transaction_id)

    repo.record_suggestion(
        ids[0], label="LIVING_DINING", source="llm", confidence=Decimal("0.950"), by="x"
    )
    repo.record_suggestion(
        ids[1], label="LIVING_DINING", source="llm", confidence=Decimal("0.400"), by="x"
    )
    repo.record_suggestion(
        ids[2], label="UNKNOWN", source="llm", confidence=Decimal("0.000"), by="x"
    )

    queued = repo.awaiting_review(Decimal("0.600"))
    assert {t.description for t in queued} == {"UNSURE", "ABSTAINED", "UNSEEN"}
    assert [t.posted_on.day for t in queued] == [4, 3, 2], "most recent first"


def test_confirming_takes_a_row_out_of_the_queue(session: Session) -> None:
    scope = _scope(session, "a@example.test")
    txn_id = _one_transaction(scope)
    repo = TransactionRepository(scope)

    repo.record_suggestion(
        txn_id, label="LIVING_OTHER", source="llm", confidence=Decimal("0.100"), by="x"
    )
    assert len(repo.awaiting_review(Decimal("0.600"))) == 1

    repo.confirm_label(txn_id, "LIVING_DINING")
    assert repo.awaiting_review(Decimal("0.600")) == []


def test_one_tenant_cannot_classify_another_tenants_transaction(session: Session) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    txn_id = _one_transaction(a)

    with pytest.raises(ValidationError, match="no transaction"):
        TransactionRepository(b).confirm_label(txn_id, "LIVING_DINING")


def test_one_tenant_never_sees_another_tenants_month(session: Session) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    _one_transaction(a)

    assert TransactionRepository(a).for_month(2026, 3) != []
    assert TransactionRepository(b).for_month(2026, 3) == []
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cd backend && uv run pytest tests/integration/test_classification_repository.py -v`
Expected: FAIL — the methods do not exist.

- [ ] **Step 3: Implement**

Add the six fields and the `effective_label` property to `StoredTransaction`, and the five methods to `TransactionRepository`. Every query filters through the tenant the way the existing ones do — join `AccountRow` and filter `AccountRow.user_id == self._user_id`. Reuse the existing `_require_own_account` shape for a new `_require_own_transaction(transaction_id)` that raises `ValidationError(f"no transaction {transaction_id}")`; `record_suggestion` and `confirm_label` both call it first.

Validate labels against `LABEL_SPACE` from `offerdelta.evaluation.labels`, raising `ValidationError(f"{label!r} is not a label in this taxonomy")`.

`awaiting_review` returns rows where `confirmed_label IS NULL` and (`suggested_label IS NULL` or `suggested_label = 'UNKNOWN'` or `suggested_confidence < threshold`), ordered by `posted_on` descending. The optional `month` narrows it.

**Note the layering.** `offerdelta.infrastructure` importing `offerdelta.evaluation.labels` is new. Run `uv run lint-imports` and confirm it stays at 5 kept. If a contract objects, move `LABEL_SPACE` into `offerdelta.domain` (it is derived from `CostCategory`, which already lives there) and report that you did.

- [ ] **Step 4: Run the tests and the full gate, then commit**

```bash
git add backend/src/offerdelta/infrastructure/postgres/repositories.py backend/tests/integration/test_classification_repository.py
git commit -m "Classification: suggestions, confirmations, and the queue between them"
```

---

### Task 4: The `categorise.py` CLI

**Files:**
- Create: `backend/categorise.py`
- Create: `backend/tests/unit/test_categorise_cli.py`
- Modify: repo-root `Makefile`, `backend/pyproject.toml` (mypy `files` **and** ruff `known-first-party`), `.github/workflows/ci.yml` (both ruff lines)

**Interfaces:**
- Consumes: `TransactionRepository.unclassified` / `.record_suggestion` (Task 3), `TransactionView` (Task 2), the existing rule baseline and `LLMCategoriser`
- Produces: `build_parser() -> argparse.ArgumentParser`, `main(argv: list[str]) -> int`, `estimate_cost(rows: int) -> Decimal`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/unit/test_categorise_cli.py`, modelled on `tests/unit/test_users_cli.py`. Cover:

```python
def test_user_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_reclassify_is_off_by_default() -> None:
    args = build_parser().parse_args(["--user", "a@example.test"])
    assert args.reclassify is False


def test_an_unknown_argument_is_an_error() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--user", "a@example.test", "--force"])


def test_the_estimate_is_stated_before_anything_is_spent(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No --yes means the run stops after telling you the price."""
    recorded: list[tuple[uuid.UUID, str]] = []

    class _FakeRepo:
        def __init__(self, scope: object) -> None:
            pass

        def unclassified(self, limit: int | None = None) -> list[object]:
            return [_row("SOME MERCHANT") for _ in range(10)]

        def record_suggestion(self, transaction_id: uuid.UUID, **kw: object) -> None:
            recorded.append((transaction_id, str(kw["label"])))

    monkeypatch.setattr(categorise, "TransactionRepository", _FakeRepo)
    monkeypatch.setattr(categorise, "_open_scope", lambda email: _fake_scope())

    exit_code = categorise.main(["--user", "a@example.test"])

    out = capsys.readouterr().out
    assert exit_code == 2
    assert "$" in out, "the estimate must name a price"
    assert "10" in out, "and how many rows it covers"
    assert recorded == [], "nothing may be written without --yes"


def test_rules_run_before_the_model_and_shrink_what_it_is_asked(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The free baseline answers what it can; only the rest costs anything."""
    asked_of_model: list[str] = []

    class _FakeLLM:
        name = "fake-llm"

        def predict_many(self, views: list[object]) -> list[object]:
            asked_of_model.extend(getattr(v, "normalised_merchant", "") for v in views)
            return [_prediction("LIVING_OTHER", "0.5") for _ in views]

    monkeypatch.setattr(categorise, "_llm_categoriser", lambda: _FakeLLM())
    monkeypatch.setattr(categorise, "_open_scope", lambda email: _fake_scope())
    monkeypatch.setattr(categorise, "TransactionRepository", _rule_answerable_repo())

    categorise.main(["--user", "a@example.test", "--yes"])

    assert "STARBUCKS" not in asked_of_model, "the rules already answered this one"
```

Write the four small helpers this needs — `_row`, `_prediction`, `_fake_scope`, `_rule_answerable_repo` — at the top of the module, in the style `tests/unit/test_users_cli.py` uses for its fakes. `_rule_answerable_repo` returns a fake repository whose `unclassified()` yields one row the keyword rules answer (a well-known merchant such as `STARBUCKS`) and one they do not.

Name `_open_scope` and `_llm_categoriser` as module-level functions in `categorise.py` precisely so these two seams exist; a CLI that builds its dependencies inline cannot be tested without a database and a key.

- [ ] **Step 2: Run them and watch them fail**

Run: `cd backend && uv run pytest tests/unit/test_categorise_cli.py -v`
Expected: FAIL — `categorise` module not found.

- [ ] **Step 3: Write `categorise.py`**

```python
"""Give stored transactions a label.

Separate from `transactions.py` on purpose. This tool needs the LLM client
and an API key; the import tool deliberately does not, and keeping that
dependency off the import path is why classification is not done at import
time. The same line is worth holding one level down.

Rules run first and the model runs on what is left, which is the order the
evaluation uses and for the same reason: a free deterministic baseline
answers about a fifth of rows, and only the remainder costs anything.
"""
```

Structure it as `build_parser()` / `main(argv)` mirroring `users.py`, including its `_report` error handling — the promise that a DSN never reaches a printed error applies here too. The flow:

1. Resolve the user and build a `TenantScope` (as `transactions.py` does).
2. `rows = repo.unclassified(limit=args.limit)`, or all rows when `--reclassify`.
3. Run the rule baseline over every row's `TransactionView`; record each non-abstaining prediction with `source="rules"`.
4. Count what is left, print `f"{n} rows need the model; estimated ${estimate_cost(n)}"`, and stop with exit 2 unless `--yes`.
5. Run `LLMCategoriser` over the remainder, recording each with `source="llm"` and `by=f"{model}:{prompt_version}"`.
6. Print a summary: rules answered N, model answered M, abstained K.

`estimate_cost` uses the published per-row figure of `$0.001913` for the selected prompt. Put that constant in one place with a comment naming where it came from.

- [ ] **Step 4: Add `categorise.py` to all four lists plus `known-first-party`**

The four must name the same set. Verify with:

```bash
cd .. && grep -n "categorise.py" Makefile backend/pyproject.toml .github/workflows/ci.yml
```

Expected: one hit in `Makefile`, two in `pyproject.toml` (mypy `files` and `known-first-party`), two in `ci.yml`.

- [ ] **Step 5: Run the tests and the full gate, then commit**

```bash
git add backend/categorise.py backend/tests/unit/test_categorise_cli.py Makefile backend/pyproject.toml .github/workflows/ci.yml
git commit -m "CLI: label stored transactions, rules first and the model after"
```

---

### Task 5: The review threshold, measured

The queue's threshold is a number that must come from evidence, not from taste.

**Files:**
- Create: `backend/sweep_threshold.py`
- Create: `backend/tests/unit/test_sweep_threshold.py`
- Create: `docs/eval/review-threshold.json`
- Modify: the four file lists plus `known-first-party`

**Interfaces:**
- Consumes: `offerdelta.evaluation.predictions` (existing saved predictions)
- Produces: `sweep(predictions, thresholds) -> list[ThresholdPoint]` where `ThresholdPoint` has `threshold: Decimal`, `coverage: Decimal`, `accuracy_when_answered: Decimal`, `queued: int`

- [ ] **Step 1: Write the failing test**

```python
"""Choosing where to stop trusting the model.

The threshold is picked on the development split and reported once on the
frozen benchmark - the same separation that selected prompt v3. A number
chosen by looking at the benchmark would be a number the benchmark can no
longer check.
"""

from decimal import Decimal

from sweep_threshold import ThresholdPoint, sweep


def test_a_threshold_of_zero_queues_nothing() -> None:
    points = sweep(_predictions(), [Decimal("0.0")])
    assert points[0].queued == 0
    assert points[0].coverage == Decimal("1")


def test_raising_the_threshold_queues_more_and_never_fewer() -> None:
    points = sweep(_predictions(), [Decimal("0.0"), Decimal("0.5"), Decimal("0.9")])
    queued = [p.queued for p in points]
    assert queued == sorted(queued)


def test_accuracy_among_answered_rows_rises_with_the_threshold() -> None:
    """The whole point: what is left after queueing should be more trustworthy."""
    points = sweep(_predictions(), [Decimal("0.0"), Decimal("0.8")])
    assert points[1].accuracy_when_answered >= points[0].accuracy_when_answered
```

Write `_predictions()` as a small hand-built list mixing correct high-confidence rows, wrong low-confidence rows, and one abstention, so each assertion has a reason to hold. Do not read a real prediction file in a unit test.

- [ ] **Step 2: Run it, watch it fail, implement `sweep`, run it again**

- [ ] **Step 3: Run the sweep for real and write the artifact**

```bash
cd backend && uv run python sweep_threshold.py \
  data/eval/predictions/development-categorise-v3.jsonl \
  --json ../docs/eval/review-threshold.json
```

The artifact records the curve, the chosen threshold, and one sentence saying it was chosen on development. Then measure that threshold once on `holdout-categorise-v3.jsonl` and record the result in the same file under a separate key. **Do not choose by looking at the holdout number.**

- [ ] **Step 4: Add to the four lists, run the gate, commit**

```bash
git add backend/sweep_threshold.py backend/tests/unit/test_sweep_threshold.py docs/eval/review-threshold.json Makefile backend/pyproject.toml .github/workflows/ci.yml
git commit -m "Eval: where to stop trusting the model, chosen on development"
```

---

### Task 6: Move `DerivationNode` to `domain/common`

Mechanical, and it stands alone so the move is reviewable without the report on top of it.

**Files:**
- Move: `backend/src/offerdelta/domain/comparisons/derivation.py` → `backend/src/offerdelta/domain/common/derivation.py`
- Modify: every importer
- Move: the corresponding test module to match

**Interfaces:**
- Produces: `offerdelta.domain.common.derivation.DerivationNode`, unchanged in every other respect

- [ ] **Step 1: Find every importer**

```bash
cd backend && grep -rn "comparisons.derivation\|comparisons import derivation" src tests
```

- [ ] **Step 2: Move the file and update imports**

Use `git mv` so history follows. Change nothing inside the file except its module docstring, which should say why it lives in `common`: a derivation tree is not a comparison concept, and both the comparison engine and the monthly report build one.

- [ ] **Step 3: Run the full suite and `lint-imports`, then commit**

The suite must be unchanged in count. Any change means the move was not mechanical.

```bash
git add -A && git commit -m "DerivationNode is a common type, not a comparison one"
```

---

### Task 7: The monthly report tree

The heart of it. Pure domain: no session, no infrastructure.

**Files:**
- Create: `backend/src/offerdelta/domain/reports/__init__.py`
- Create: `backend/src/offerdelta/domain/reports/monthly.py`
- Create: `backend/tests/unit/domain/reports/test_monthly.py`

**Interfaces:**
- Consumes: `DerivationNode` (Task 6), `Money`, `Evidence`, `PeriodKind`, `CostCategory`, `TransactionKind`
- Produces:
  - `ClassifiedTransaction(posted_on: date, description: str, amount: Money, label: str | None, confirmed: bool)`
  - `build_monthly_report(year: int, month: int, transactions: Sequence[ClassifiedTransaction]) -> DerivationNode`

- [ ] **Step 1: Write the failing tests**

```python
"""The monthly tree.

Rooted at the sum of every row rather than at net cash flow, so that a
dropped or double-counted row makes the report impossible to build rather
than quietly wrong.
"""

from __future__ import annotations

from datetime import date

import pytest

from offerdelta.domain.common.derivation import DerivationNode
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.evidence import Evidence
from offerdelta.domain.common.money import Money
from offerdelta.domain.reports.monthly import ClassifiedTransaction, build_monthly_report


def _txn(day: int, amount: str, label: str | None, *, confirmed: bool = False) -> ClassifiedTransaction:
    return ClassifiedTransaction(
        posted_on=date(2026, 3, day),
        description=f"ROW {day}",
        amount=Money.parse(amount),
        label=label,
        confirmed=confirmed,
    )


def _branch(root: DerivationNode, code: str) -> DerivationNode:
    for child in root.children:
        if child.code == code:
            return child
    raise AssertionError(f"no branch {code!r} in {[c.code for c in root.children]}")


def test_the_root_is_the_sum_of_every_row() -> None:
    rows = [
        _txn(1, "5000.00", "INCOME", confirmed=True),
        _txn(2, "-12.34", "LIVING_DINING", confirmed=True),
        _txn(3, "-500.00", "TRANSFER", confirmed=True),
        _txn(4, "-9.99", None),
    ]
    root = build_monthly_report(2026, 3, rows)
    assert root.amount == Money.parse("4477.67")


def test_transfers_are_a_branch_and_not_an_exclusion() -> None:
    """Excluding them would leave a balanced tree around a misclassified one."""
    rows = [_txn(1, "-500.00", "TRANSFER", confirmed=True)]
    root = build_monthly_report(2026, 3, rows)
    assert _branch(root, "transfers").amount == Money.parse("-500.00")
    assert root.amount == Money.parse("-500.00")


def test_never_examined_and_examined_but_unknown_are_separate_leaves() -> None:
    rows = [_txn(1, "-1.00", None), _txn(2, "-2.00", "UNKNOWN")]
    unclassified = _branch(build_monthly_report(2026, 3, rows), "unclassified")
    by_code = {child.code: child.amount for child in unclassified.children}
    assert by_code == {
        "never_examined": Money.parse("-1.00"),
        "examined_no_answer": Money.parse("-2.00"),
    }


def test_a_month_of_confirmed_rows_has_a_confirmed_root() -> None:
    rows = [_txn(1, "-1.00", "LIVING_DINING", confirmed=True)]
    assert build_monthly_report(2026, 3, rows).evidence is Evidence.USER_CONFIRMED


def test_one_unreviewed_row_makes_the_whole_root_assumed() -> None:
    rows = [
        _txn(1, "-1.00", "LIVING_DINING", confirmed=True),
        _txn(2, "-2.00", "LIVING_GROCERY", confirmed=False),
    ]
    assert build_monthly_report(2026, 3, rows).evidence is Evidence.ASSUMED


def test_spending_is_broken_down_by_category() -> None:
    rows = [
        _txn(1, "-10.00", "LIVING_DINING", confirmed=True),
        _txn(2, "-5.00", "LIVING_DINING", confirmed=True),
        _txn(3, "-20.00", "LIVING_GROCERY", confirmed=True),
    ]
    spending = _branch(build_monthly_report(2026, 3, rows), "spending")
    by_code = {child.code: child.amount for child in spending.children}
    assert by_code == {
        "LIVING_DINING": Money.parse("-15.00"),
        "LIVING_GROCERY": Money.parse("-20.00"),
    }


def test_a_row_outside_the_month_is_refused() -> None:
    stray = ClassifiedTransaction(
        posted_on=date(2026, 4, 1),
        description="APRIL ROW",
        amount=Money.parse("-1.00"),
        label="LIVING_DINING",
        confirmed=True,
    )
    with pytest.raises(ValidationError, match="2026-03"):
        build_monthly_report(2026, 3, [stray])


def test_an_empty_month_is_a_zero_root_not_an_error() -> None:
    root = build_monthly_report(2026, 3, [])
    assert root.amount == Money.zero()


def test_no_row_is_silently_dropped_by_the_grouping() -> None:
    """The failure the root-at-every-row shape exists to make impossible.

    A label that falls through every branch would vanish from the tree and
    the totals would still look plausible. One row per branch, and the root
    must equal their sum - if grouping loses one, this fails.
    """
    rows = [
        _txn(1, "5000.00", "INCOME", confirmed=True),
        _txn(2, "-10.00", "LIVING_DINING", confirmed=True),
        _txn(3, "25.00", "REFUND", confirmed=True),
        _txn(4, "-500.00", "TRANSFER", confirmed=True),
        _txn(5, "-1.00", None),
        _txn(6, "-2.00", "UNKNOWN"),
    ]
    root = build_monthly_report(2026, 3, rows)

    leaves = [node for node in root.walk() if not node.children]
    total = Money.zero()
    for leaf in leaves:
        total = total + leaf.amount

    assert len(leaves) == len(rows), "every row appears exactly once as a leaf"
    assert total == root.amount
    assert root.amount == Money.parse("4512.00")
    assert {child.code for child in root.children} == {
        "income",
        "spending",
        "refunds",
        "transfers",
        "unclassified",
    }
```

`DerivationNode.walk()` exists (`domain/common/derivation.py` after Task 6) and yields the node and its descendants. Counting leaves rather than nodes keeps the assertion about the property — one row, one leaf — instead of about the tree's intermediate shape, which is free to change.

- [ ] **Step 2: Run them and watch them fail**

- [ ] **Step 3: Implement `build_monthly_report`**

Group rows into five branches by their label: `income`, `spending` (further grouped by `CostCategory`), `refunds`, `transfers`, `unclassified` (two leaves — `never_examined` for `label is None`, `examined_no_answer` for `"UNKNOWN"`). Every branch is a `DerivationNode` whose children sum to it; the root sums the five. Use `PeriodKind.MONTHLY` throughout.

Leaf evidence is `Evidence.USER_CONFIRMED` when `confirmed` is true and `Evidence.ASSUMED` otherwise; unclassified leaves are `Evidence.ASSUMED`. Branch evidence is the weakest of its children — `domain/comparisons/derivation_builder.py` already has a `_weakest` helper; move it beside `DerivationNode` in `domain/common/derivation.py` rather than writing a second copy, and update its existing caller.

Refuse a row whose `posted_on` is outside the requested month with `ValidationError(f"{txn.posted_on} is not in 2026-03")` (built from the arguments, not hard-coded).

- [ ] **Step 4: Run the tests and the full gate, then commit**

```bash
git add backend/src/offerdelta/domain/reports backend/src/offerdelta/domain/common/derivation.py backend/src/offerdelta/domain/comparisons/derivation_builder.py backend/tests/unit/domain/reports
git commit -m "Reports: a monthly tree rooted at every row"
```

---

### Task 8: Assembling a report from stored rows

The application-layer use case that turns stored rows into the tree, plus the month index.

**Files:**
- Create: `backend/src/offerdelta/application/reports/__init__.py`
- Create: `backend/src/offerdelta/application/reports/monthly.py`
- Create: `backend/tests/integration/test_monthly_report_service.py`

**Interfaces:**
- Consumes: `TransactionRepository.for_month` (Task 3), `build_monthly_report` (Task 7), `ImportBatchRepository`
- Produces:
  - `MonthCoverage(year: int, month: int, complete: bool, rows: int, classified: int, awaiting_review: int)`
  - `available_months(scope: TenantScope) -> list[MonthCoverage]`
  - `monthly_report(scope: TenantScope, year: int, month: int, *, threshold: Decimal) -> MonthlyReport` where `MonthlyReport` carries `tree: DerivationNode` and `coverage: MonthCoverage`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/integration/test_monthly_report_service.py`. Reuse the `_scope` helper shape from `test_classification_repository.py`.

```python
"""A month, assembled from what is actually stored."""

from __future__ import annotations

from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from offerdelta.application.reports.monthly import available_months, monthly_report
from offerdelta.application.scope import TenantScope
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.repositories import TransactionRepository
from tests.integration.conftest import requires_database

pytestmark = requires_database

THRESHOLD = Decimal("0.600")


def test_the_root_equals_the_sum_of_every_stored_row_for_that_month(
    scope: TenantScope,
) -> None:
    ids = _seed(scope, [("2026-03-01", "5000.00"), ("2026-03-02", "-12.34"),
                        ("2026-03-03", "-9.99")])
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
    months = {(m.year, m.month): m for m in available_months(scope)}
    assert months[(2026, 3)].complete is False


def test_a_month_a_snapshot_window_covers_completely_is_complete(
    scope: TenantScope,
) -> None:
    _seed_snapshot_batch(scope, window=(date(2026, 3, 1), date(2026, 3, 31)),
                         rows=[("2026-03-05", "-1.00")])
    months = {(m.year, m.month): m for m in available_months(scope)}
    assert months[(2026, 3)].complete is True


def test_one_tenant_never_sees_another_tenants_month(
    scope: TenantScope, other_scope: TenantScope
) -> None:
    _seed(scope, [("2026-03-01", "-1.00")])

    mine = monthly_report(scope, 2026, 3, threshold=THRESHOLD)
    theirs = monthly_report(other_scope, 2026, 3, threshold=THRESHOLD)

    assert mine.coverage.rows == 1
    assert theirs.coverage.rows == 0
    assert theirs.tree.amount == Money.zero()
```

Write `_seed(scope, rows)` — registers an account and enters each `(iso_date, amount)` through `enter_transaction`, returning the transaction ids — and `_seed_snapshot_batch(scope, window, rows)`, which commits a snapshot import declaring that window so completeness has something to read. Add an `other_scope` fixture beside the existing `scope` fixture in `tests/integration/conftest.py`, building a second tenant the same way.

- [ ] **Step 2: Run them, watch them fail, implement, run again**

Completeness is decided from `import_batches`: a month is complete when the union of `[window_start, window_end]` across that tenant's snapshot batches covers every day of the month. Incremental batches never make a month complete, because they claim only that activity was appended, not that a window is whole.

- [ ] **Step 3: Run the full gate and commit**

```bash
git add backend/src/offerdelta/application/reports backend/tests/integration/test_monthly_report_service.py
git commit -m "Reports: assembling a month from stored rows, with its coverage"
```

---

### Task 9: The four routes

**Files:**
- Modify: `backend/src/offerdelta/api/main.py`
- Modify: `backend/src/offerdelta/api/schemas.py`
- Create: `backend/tests/integration/test_reports_api.py`
- Modify: `backend/tests/contract/test_public_demo.py`

**Interfaces:**
- Consumes: `available_months`, `monthly_report` (Task 8), `TransactionRepository.confirm_label` and `.awaiting_review` (Task 3), `_scope` (Phase 0)
- Produces: the four routes in the spec

- [ ] **Step 1: Write the failing tests**

```
GET  /v1/reports/months
GET  /v1/reports/monthly/{YYYY-MM}
GET  /v1/review-queue          ?month=YYYY-MM optional
POST /v1/transactions/{id}/label
```

Cover: each route 401s without a token; a month with no data returns an empty-but-valid tree; a malformed month string is 422; `POST .../label` with a label outside `LABEL_SPACE` is 422; **`POST .../label` naming another tenant's transaction id is 404**; the review queue contains no other tenant's rows; and the report's amounts are decimal strings, not JSON numbers.

Reuse the `client`, `token_a` and `account_key_of_b` fixtures from `tests/integration/conftest.py`.

- [ ] **Step 2: Implement, reusing what exists**

`DerivationNodeSchema.of()` already serialises a tree for `/v1/demo/derivation` — use it rather than writing a second serialiser. Routes depend on `_scope`. `api` must not import `offerdelta.infrastructure.postgres.repositories`; go through the Task 8 use cases, adding a thin application function for the label write and the queue read if one is missing.

- [ ] **Step 3: Extend the public-surface containment test**

In `tests/contract/test_public_demo.py`, add the four routes to whatever test defines the allowed public surface, asserting each requires authentication — the same way `/v1/auth/token` was pinned in Phase 0.

- [ ] **Step 4: Run the full gate and commit**

```bash
git add backend/src/offerdelta/api backend/tests
git commit -m "API: the monthly report and the review queue, both tenant-scoped"
```

---

### Task 10: Labels for the seeded tenants

The deployment shows the product for the first time.

**Files:**
- Modify: `backend/seed_demo.py`
- Modify: `backend/tests/unit/test_seed_demo.py`

**Interfaces:**
- Consumes: `TransactionRepository.confirm_label` (Task 3)
- Produces: seeded transactions that carry confirmed labels

- [ ] **Step 1: Write the failing test**

```python
def test_every_seeded_transaction_carries_a_confirmed_label() -> None:
    """Synthetic data is honestly USER_CONFIRMED: a person wrote it."""
```

Assert through the existing fake repositories that each seeded transaction is confirmed, and that the labels used are all in `LABEL_SPACE`. Add a test that the seeded set spans at least two complete months, so the deployed demo can show a month-over-month comparison rather than a single month.

- [ ] **Step 2: Extend the seed**

Give each synthetic tenant transactions across two full months with confirmed labels covering income, several spending categories, a transfer, and a refund — enough that the report tree has every branch populated. Keep the script idempotent and keep its single-commit shape.

- [ ] **Step 3: Run the full gate and commit**

```bash
git add backend/seed_demo.py backend/tests/unit/test_seed_demo.py
git commit -m "Seed: two labelled months, so the demo shows a report"
```

---

## Done when

- `make check` is green: 5 import-linter contracts, strict mypy, the full suite.
- CI is green on the branch, with both `CONNECTION_STRING` and `JWT_SECRET` supplied.
- A month's report root equals the sum of every stored row in that month, and a builder that drops one raises instead of returning.
- A month containing one unreviewed row has an `ASSUMED` root; confirming it turns the root `USER_CONFIRMED`.
- `NULL` and `'UNKNOWN'` appear as separate leaves.
- The review threshold in `docs/eval/review-threshold.json` was chosen on the development split and measured once on the benchmark.
- No route reaches a report or the queue without a tenant, and another tenant's transaction id answers 404.
- The evaluation figures published for V2 are unchanged, and the test pinning the LLM request body still passes.
