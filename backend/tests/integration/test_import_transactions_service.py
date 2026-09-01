from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from offerdelta.application.scope import TenantScope
from offerdelta.application.transactions.import_transactions import (
    ImportRequest,
    import_csv,
)
from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    TransactionRepository,
)
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


def test_an_unregistered_account_is_refused(scope: TenantScope, tmp_path: Path) -> None:
    path = _file(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")
    with pytest.raises(ValidationError, match="no account"):
        import_csv(scope, _request(path))


def test_the_error_lists_known_accounts(scope: TenantScope, tmp_path: Path) -> None:
    AccountRepository(scope).register("Chase Checking")
    path = _file(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")
    with pytest.raises(ValidationError, match="chase-checking"):
        import_csv(scope, _request(path, key="checking"))


def test_a_successful_import_reports_the_batch(scope: TenantScope, tmp_path: Path) -> None:
    AccountRepository(scope).register("Checking")
    path = _file(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")

    outcome = import_csv(scope, _request(path))

    assert outcome.created is True
    assert outcome.result.imported_count == 1
    assert outcome.batch.mode == "snapshot"
    assert outcome.batch.window_start == date(2026, 8, 1)


def test_the_identical_file_is_a_batch_level_no_op(scope: TenantScope, tmp_path: Path) -> None:
    AccountRepository(scope).register("Checking")
    path = _file(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")

    first = import_csv(scope, _request(path))
    second = import_csv(scope, _request(path))

    assert first.created is True
    assert second.created is False
    assert second.result.imported_count == 0
    assert second.batch.id == first.batch.id


def test_a_file_that_changes_mid_read_is_refused(
    scope: TenantScope, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batch checksum must describe the bytes actually parsed.

    `import_csv` now hashes the file both before and after `preview_csv`
    reads it and refuses on a mismatch. Simulated here by making the two
    calls to `file_sha256` return different values, standing in for the file
    changing mid-read: if the digest recorded a *different* file than the one
    whose rows were just written, a later import of the real file with that
    real digest would be treated as an already-imported no-op - a batch that
    silently never happened, with no report.
    """
    AccountRepository(scope).register("Checking")
    path = _file(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")

    digests = iter(["digest-before-the-change", "digest-after-the-change"])
    monkeypatch.setattr(
        "offerdelta.application.transactions.import_transactions.file_sha256",
        lambda _path: next(digests),
    )

    with pytest.raises(ValidationError, match="changed while it was being read"):
        import_csv(scope, _request(path))

    account = AccountRepository(scope).by_key("checking")
    assert account is not None
    assert TransactionRepository(scope).count(account_id=account.id) == 0


def test_snapshot_mode_refuses_a_mapped_external_id(scope: TenantScope, tmp_path: Path) -> None:
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
    AccountRepository(scope).register("Checking")
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
        import_csv(scope, request)


def test_incremental_import_is_refused_after_a_snapshot_import_on_the_same_account(
    scope: TenantScope, tmp_path: Path
) -> None:
    """Carry-forward 1: mixing modes on one account silently duplicates a real charge.

    Snapshot rows are stored with external_id = NULL, which makes them
    invisible to the incremental dedupe check (`existing_external` only
    collects non-NULL ids); the occurrence offset then steps past those same
    rows too, so the unique constraint does not catch it either. One real
    charge, imported once as a snapshot and once incrementally, ends up
    stored twice with a clean report. Refusing to mix modes on one account is
    the fix - identity is judged differently in each mode, so switching modes
    on the same account can duplicate or drop charges either direction.
    """
    account = AccountRepository(scope).register("Checking")
    snapshot_path = _file(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")
    snapshot_outcome = import_csv(scope, _request(snapshot_path))
    assert snapshot_outcome.result.imported_count == 1

    incremental_path = tmp_path / "aug-incremental.csv"
    incremental_path.write_text(
        "Date,Description,Amount,Ref\n2026-08-17,BLUE BOTTLE,-4.50,TXN-1\n",
        encoding="utf-8",
    )
    mapping = ColumnMapping(
        date="Date", description="Description", amount="Amount", external_id="Ref"
    )
    incremental_request = ImportRequest(
        path=incremental_path,
        account_key="checking",
        mode=ImportMode.INCREMENTAL,
        window=AUGUST,
        mapping=mapping,
        date_order=DateOrder.ISO,
    )

    with pytest.raises(ValidationError, match="snapshot"):
        import_csv(scope, incremental_request)

    assert TransactionRepository(scope).count(account_id=account.id) == 1
