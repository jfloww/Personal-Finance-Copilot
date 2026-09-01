"""Incremental semantics, which exist only when the bank supplies an id."""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest

from offerdelta.application.scope import TenantScope
from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    TransactionImportResult,
    TransactionRepository,
)
from offerdelta.ingest.commit import ImportMode, plan_records
from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import ColumnMapping
from offerdelta.ingest.preview import preview_csv
from tests.integration.conftest import requires_database

pytestmark = requires_database

HEADER = "TxnId,Date,Description,Amount\n"
MAPPING = ColumnMapping(
    date="Date", description="Description", amount="Amount", external_id="TxnId"
)

#: Planning validates the mode before touching the database, so the refusal
#: test needs an account id but never a real account.
ANY_ACCOUNT = uuid.UUID("33333333-3333-3333-3333-333333333333")


def _import(
    scope: TenantScope, account_id: uuid.UUID, tmp_path: Path, body: str, name: str
) -> TransactionImportResult:
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="utf-8")
    preview = preview_csv(path, mapping=MAPPING, date_order=DateOrder.ISO)
    records = plan_records(preview, account_id=account_id, mode=ImportMode.INCREMENTAL, window=None)
    return TransactionRepository(scope).add_many(records)


def test_a_genuine_third_repeat_is_stored(scope: TenantScope, tmp_path: Path) -> None:
    """The case that silently lost money under file-local numbering."""
    account = AccountRepository(scope).register("Checking")
    repo = TransactionRepository(scope)

    _import(
        scope,
        account.id,
        tmp_path,
        "T1,2026-08-17,BLUE BOTTLE,-4.50\nT2,2026-08-17,BLUE BOTTLE,-4.50\n",
        "first.csv",
    )
    result = _import(scope, account.id, tmp_path, "T3,2026-08-17,BLUE BOTTLE,-4.50\n", "second.csv")

    assert result.imported_count == 1
    assert repo.count(account_id=account.id) == 3


def test_a_re_sent_id_is_skipped(scope: TenantScope, tmp_path: Path) -> None:
    account = AccountRepository(scope).register("Checking")
    repo = TransactionRepository(scope)

    _import(scope, account.id, tmp_path, "T1,2026-08-17,BLUE BOTTLE,-4.50\n", "a.csv")
    result = _import(
        scope,
        account.id,
        tmp_path,
        "T1,2026-08-17,BLUE BOTTLE,-4.50\nT2,2026-08-18,TRANSIT,-2.75\n",
        "b.csv",
    )

    assert result.imported_count == 1
    assert result.already_stored_count == 1
    assert repo.count(account_id=account.id) == 2


def test_incremental_is_refused_without_an_id_column(tmp_path: Path) -> None:
    """No database needed: the refusal happens during planning."""
    path = tmp_path / "no-id.csv"
    path.write_text("Date,Description,Amount\n2026-08-17,BLUE BOTTLE,-4.50\n", encoding="utf-8")
    preview = preview_csv(path, date_order=DateOrder.ISO)

    with pytest.raises(ValidationError, match="transaction id"):
        plan_records(
            preview,
            account_id=ANY_ACCOUNT,
            mode=ImportMode.INCREMENTAL,
            window=None,
        )


def test_a_new_external_id_is_written_despite_a_fingerprint_collision(
    scope: TenantScope, tmp_path: Path
) -> None:
    """Proves external_id, not fingerprint, decides identity.

    Same date, merchant, and amount as the stored row, so fingerprint +
    file-local occurrence would both land on occurrence 1 and read as
    "already stored". Only a different external_id tells these two rows
    apart, so this is the one case an implementation that stores
    external_id as an inert column - without actually branching on it -
    cannot pass: it would either raise IntegrityError (no offset) or
    report the second row as a duplicate (fingerprint used for identity).
    """
    account = AccountRepository(scope).register("Checking")
    repo = TransactionRepository(scope)

    _import(scope, account.id, tmp_path, "T1,2026-08-17,BLUE BOTTLE,-4.50\n", "a.csv")
    result = _import(scope, account.id, tmp_path, "T2,2026-08-17,BLUE BOTTLE,-4.50\n", "b.csv")

    assert result.imported_count == 1
    assert repo.count(account_id=account.id) == 2
