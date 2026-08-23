"""Snapshot semantics, asserted on real PostgreSQL."""

from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.fingerprint import compute_fingerprint
from offerdelta.infrastructure.postgres.models import TransactionRow
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    TransactionImportResult,
    TransactionRepository,
)
from offerdelta.ingest.commit import ImportMode, ImportWindow, plan_records
from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.preview import preview_csv
from tests.integration.conftest import requires_database

pytestmark = requires_database

HEADER = "Date,Description,Amount\n"
AUGUST = ImportWindow(start=date(2026, 8, 1), end=date(2026, 8, 31))


def _import(
    session: Session,
    account_id: uuid.UUID,
    tmp_path: Path,
    body: str,
    name: str = "aug.csv",
) -> TransactionImportResult:
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="utf-8")
    preview = preview_csv(path, date_order=DateOrder.ISO)
    records = plan_records(preview, account_id=account_id, mode=ImportMode.SNAPSHOT, window=AUGUST)
    return TransactionRepository(session).add_many(records)


def test_the_same_charge_written_two_ways_is_one_transaction(
    session: Session, tmp_path: Path
) -> None:
    """-4.50 and -4.5 are the same coffee. This was the headline bug."""
    account = AccountRepository(session).register("Checking")
    repo = TransactionRepository(session)

    _import(session, account.id, tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n", "a.csv")
    second = _import(session, account.id, tmp_path, "2026-08-17,BLUE BOTTLE,-4.5\n", "b.csv")

    assert second.imported_count == 0
    assert second.already_stored_count == 1
    assert repo.count(account_id=account.id) == 1


def test_re_importing_an_identical_window_writes_nothing(session: Session, tmp_path: Path) -> None:
    account = AccountRepository(session).register("Checking")
    repo = TransactionRepository(session)
    body = "2026-08-17,BLUE BOTTLE,-4.50\n2026-08-18,TRANSIT,-2.75\n"

    _import(session, account.id, tmp_path, body, "a.csv")
    second = _import(session, account.id, tmp_path, body, "b.csv")

    assert second.imported_count == 0
    assert repo.count(account_id=account.id) == 2


def test_two_identical_charges_on_one_day_both_persist(session: Session, tmp_path: Path) -> None:
    """Two coffees are two coffees. Deduplicating them deletes real money."""
    account = AccountRepository(session).register("Checking")
    repo = TransactionRepository(session)

    _import(
        session,
        account.id,
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-4.50\n2026-08-17,BLUE BOTTLE,-4.50\n",
    )
    assert repo.count(account_id=account.id) == 2


def test_a_later_window_containing_a_third_repeat_adds_exactly_one(
    session: Session, tmp_path: Path
) -> None:
    """The case an unconditional max-occurrence offset would have doubled."""
    account = AccountRepository(session).register("Checking")
    repo = TransactionRepository(session)

    _import(
        session,
        account.id,
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-4.50\n2026-08-17,BLUE BOTTLE,-4.50\n",
        "first.csv",
    )
    result = _import(
        session,
        account.id,
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-4.50\n"
        "2026-08-17,BLUE BOTTLE,-4.50\n"
        "2026-08-17,BLUE BOTTLE,-4.50\n",
        "second.csv",
    )

    assert result.imported_count == 1
    assert result.already_stored_count == 2
    assert repo.count(account_id=account.id) == 3


def test_two_accounts_do_not_share_identities(session: Session, tmp_path: Path) -> None:
    repo = AccountRepository(session)
    checking = repo.register("Checking")
    savings = repo.register("Savings")
    body = "2026-08-17,BLUE BOTTLE,-4.50\n"

    _import(session, checking.id, tmp_path, body, "a.csv")
    result = _import(session, savings.id, tmp_path, body, "b.csv")

    assert result.imported_count == 1


def test_every_stored_fingerprint_recomputes_from_its_own_row(
    session: Session, tmp_path: Path
) -> None:
    """Reproducibility is a property the suite proves, not a claim."""
    account = AccountRepository(session).register("Checking")
    _import(
        session,
        account.id,
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-4.50\n2026-08-18,TRANSIT,-2.75\n",
    )

    for row in session.scalars(select(TransactionRow)).all():
        recomputed = compute_fingerprint(
            account_id=row.account_id,
            posted_on=row.posted_on,
            normalised_merchant=row.normalised_merchant,
            amount=Money(row.amount, row.currency),
        )
        assert recomputed == row.fingerprint
