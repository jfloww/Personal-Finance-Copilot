from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from offerdelta.application.transactions.import_transactions import (
    ImportRequest,
    import_csv,
)
from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.postgres.repositories import AccountRepository
from offerdelta.ingest.commit import ImportMode, ImportWindow
from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import ColumnMapping
from tests.integration.conftest import requires_database

pytestmark = requires_database

HEADER = "Date,Description,Amount\n"
AUGUST = ImportWindow(start=date(2026, 8, 1), end=date(2026, 8, 31))


def _request(path: Path, key: str = "checking") -> ImportRequest:
    return ImportRequest(
        path=path,
        account_key=key,
        mode=ImportMode.SNAPSHOT,
        window=AUGUST,
        mapping=None,
        date_order=DateOrder.ISO,
    )


def _file(tmp_path: Path, body: str, name: str = "aug.csv") -> Path:
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_an_unregistered_account_is_refused(session: Session, tmp_path: Path) -> None:
    path = _file(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")
    with pytest.raises(ValidationError, match="no account"):
        import_csv(session, _request(path))


def test_the_error_lists_known_accounts(session: Session, tmp_path: Path) -> None:
    AccountRepository(session).register("Chase Checking")
    path = _file(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")
    with pytest.raises(ValidationError, match="chase-checking"):
        import_csv(session, _request(path, key="checking"))


def test_a_successful_import_reports_the_batch(session: Session, tmp_path: Path) -> None:
    AccountRepository(session).register("Checking")
    path = _file(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")

    outcome = import_csv(session, _request(path))

    assert outcome.created is True
    assert outcome.result.imported_count == 1
    assert outcome.batch.mode == "snapshot"
    assert outcome.batch.window_start == date(2026, 8, 1)


def test_the_identical_file_is_a_batch_level_no_op(session: Session, tmp_path: Path) -> None:
    AccountRepository(session).register("Checking")
    path = _file(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")

    first = import_csv(session, _request(path))
    second = import_csv(session, _request(path))

    assert first.created is True
    assert second.created is False
    assert second.result.imported_count == 0
    assert second.batch.id == first.batch.id


def test_snapshot_mode_refuses_a_mapped_external_id(session: Session, tmp_path: Path) -> None:
    """Carry-forward 2.

    `add_many` decides its occurrence offset purely on `external_id is not
    None`, and `plan_records` does not forbid SNAPSHOT with an id column
    mapped. If it slipped through, a genuinely-new external id whose
    fingerprint collides with a stored non-id row would be silently written
    at an offset occurrence instead of raising - snapshot mode is supposed to
    judge duplicates by content alone. The service is the first real caller
    that supplies both a mode and a mapping together, so it is where this has
    to be refused.
    """
    AccountRepository(session).register("Checking")
    path = tmp_path / "aug.csv"
    path.write_text(
        "Date,Description,Amount,Ref\n2026-08-17,BLUE BOTTLE,-4.50,TXN-1\n",
        encoding="utf-8",
    )
    mapping = ColumnMapping(
        date="Date", description="Description", amount="Amount", external_id="Ref"
    )
    request = ImportRequest(
        path=path,
        account_key="checking",
        mode=ImportMode.SNAPSHOT,
        window=AUGUST,
        mapping=mapping,
        date_order=DateOrder.ISO,
    )

    with pytest.raises(ValidationError, match="snapshot mode identifies rows by content"):
        import_csv(session, request)
