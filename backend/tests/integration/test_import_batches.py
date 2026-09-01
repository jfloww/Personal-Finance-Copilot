from __future__ import annotations

import uuid
from datetime import date

from offerdelta.application.scope import TenantScope
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    ImportBatchRepository,
    StoredBatch,
)
from tests.integration.conftest import requires_database

pytestmark = requires_database

CHECKSUM = "a" * 64


def _open(
    scope: TenantScope, account_id: uuid.UUID, checksum: str = CHECKSUM
) -> tuple[StoredBatch, bool]:
    return ImportBatchRepository(scope).open(
        account_id,
        source_file="aug.csv",
        source_sha256=checksum,
        mode="snapshot",
        window_start=date(2026, 8, 1),
        window_end=date(2026, 8, 31),
        row_count=400,
    )


def test_opening_a_new_batch_reports_it_as_created(scope: TenantScope) -> None:
    account = AccountRepository(scope).register("Checking")
    _batch, created = _open(scope, account.id)
    assert created is True


def test_an_identical_file_returns_the_original_batch(scope: TenantScope) -> None:
    """The one unambiguous form of 'already imported'."""
    account = AccountRepository(scope).register("Checking")
    first, created_first = _open(scope, account.id)
    second, created_second = _open(scope, account.id)

    assert created_first is True
    assert created_second is False
    assert second.id == first.id
    assert second.imported_at == first.imported_at


def test_a_different_file_opens_a_new_batch(scope: TenantScope) -> None:
    account = AccountRepository(scope).register("Checking")
    first, _ = _open(scope, account.id)
    second, created = _open(scope, account.id, checksum="b" * 64)

    assert created is True
    assert second.id != first.id


def test_the_same_file_in_another_account_is_a_new_batch(scope: TenantScope) -> None:
    repo = AccountRepository(scope)
    checking = repo.register("Checking")
    savings = repo.register("Savings")

    _first, _ = _open(scope, checking.id)
    _second, created = _open(scope, savings.id)
    assert created is True


def test_a_losing_racer_gets_the_existing_batch_not_a_crash(scope: TenantScope) -> None:
    """`open()` does a SELECT, then an INSERT, with a gap between the two.

    Two callers racing the same `(account_id, source_sha256)` can both pass
    the SELECT before either has inserted, so only one INSERT can win the
    unique constraint - the loser is documented to get the existing batch
    back with `created=False`, not a traceback.

    A genuine concurrent test would need two overlapping live transactions
    with precise timing between the SELECT and the INSERT on each side; that
    is impractical to construct reliably here, since every test in this
    suite shares one connection and is rolled back afterwards, and there is
    no second connection available to race against. So this instead calls
    `_insert` - the repository's own insert-and-recover step - directly and
    twice for the same key, with no existence check in front of either call.
    That is exactly what a losing racer's code path does after its own SELECT
    has already (wrongly) reported "nothing there": the second `_insert` call
    below violates the real unique constraint for real, and the real
    `IntegrityError` handler has to recover it, not a stand-in for either.
    """
    account = AccountRepository(scope).register("Checking")
    repo = ImportBatchRepository(scope)

    def _insert() -> tuple[StoredBatch, bool]:
        return repo._insert(
            account.id,
            source_file="aug.csv",
            source_sha256=CHECKSUM,
            mode="snapshot",
            window_start=date(2026, 8, 1),
            window_end=date(2026, 8, 31),
            row_count=400,
            now=None,
        )

    winner, created_winner = _insert()
    loser, created_loser = _insert()

    assert created_winner is True
    assert created_loser is False
    assert loser.id == winner.id
    assert loser.source_sha256 == CHECKSUM
