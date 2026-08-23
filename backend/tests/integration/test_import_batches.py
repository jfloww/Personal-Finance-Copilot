from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy.orm import Session

from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    ImportBatchRepository,
    StoredBatch,
)
from tests.integration.conftest import requires_database

pytestmark = requires_database

CHECKSUM = "a" * 64


def _open(
    session: Session, account_id: uuid.UUID, checksum: str = CHECKSUM
) -> tuple[StoredBatch, bool]:
    return ImportBatchRepository(session).open(
        account_id,
        source_file="aug.csv",
        source_sha256=checksum,
        mode="snapshot",
        window_start=date(2026, 8, 1),
        window_end=date(2026, 8, 31),
        row_count=400,
    )


def test_opening_a_new_batch_reports_it_as_created(session: Session) -> None:
    account = AccountRepository(session).register("Checking")
    _batch, created = _open(session, account.id)
    assert created is True


def test_an_identical_file_returns_the_original_batch(session: Session) -> None:
    """The one unambiguous form of 'already imported'."""
    account = AccountRepository(session).register("Checking")
    first, created_first = _open(session, account.id)
    second, created_second = _open(session, account.id)

    assert created_first is True
    assert created_second is False
    assert second.id == first.id
    assert second.imported_at == first.imported_at


def test_a_different_file_opens_a_new_batch(session: Session) -> None:
    account = AccountRepository(session).register("Checking")
    first, _ = _open(session, account.id)
    second, created = _open(session, account.id, checksum="b" * 64)

    assert created is True
    assert second.id != first.id


def test_the_same_file_in_another_account_is_a_new_batch(session: Session) -> None:
    repo = AccountRepository(session)
    checking = repo.register("Checking")
    savings = repo.register("Savings")

    _first, _ = _open(session, checking.id)
    _second, created = _open(session, savings.id)
    assert created is True
