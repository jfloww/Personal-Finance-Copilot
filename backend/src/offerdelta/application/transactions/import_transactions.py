"""Importing transactions, orchestrated in one place.

CSV ingest is one input adapter. Manual entry will be another, and it needs the
same sequence — resolve the account, open a batch, build records, write, report
— minus the parsing. Putting that sequence here is what stops it being written
twice, or a form reaching into the CSV pipeline for something it does not need.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    ImportBatchRepository,
    StoredBatch,
    TransactionImportResult,
    TransactionRepository,
)
from offerdelta.ingest.checksum import file_sha256
from offerdelta.ingest.commit import ImportMode, ImportWindow, plan_records
from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import ColumnMapping
from offerdelta.ingest.preview import preview_csv


@dataclass(frozen=True)
class ImportRequest:
    """Everything one import needs, stated explicitly."""

    path: Path
    account_key: str
    mode: ImportMode
    window: ImportWindow | None
    mapping: ColumnMapping | None = None
    date_order: DateOrder | None = None


@dataclass(frozen=True)
class ImportOutcome:
    """What happened, in full."""

    batch: StoredBatch
    created: bool
    result: TransactionImportResult


def import_csv(session: Session, request: ImportRequest) -> ImportOutcome:
    """Resolve, plan, and write one CSV import."""
    # Snapshot mode judges duplicates by content: a fingerprint plus a
    # per-file occurrence number. `add_many` only takes that path when a
    # record's `external_id` is None; a mapped id column would flip every
    # record in this import onto the external_id path instead, silently
    # changing how duplicates are judged. Refused here, before any I/O,
    # because this is a property of the request itself, not of the account
    # or the file's contents.
    if (
        request.mode is ImportMode.SNAPSHOT
        and request.mapping is not None
        and request.mapping.external_id is not None
    ):
        raise ValidationError(
            "snapshot mode identifies rows by content, not by id; a mapped "
            f"external_id column ({request.mapping.external_id!r}) would change how "
            "duplicates are judged. Use incremental mode if the file's ids should "
            "decide identity, or map this file without external_id for a snapshot "
            "import."
        )

    accounts = AccountRepository(session)
    account = accounts.by_key(request.account_key)
    if account is None:
        known = ", ".join(a.key for a in accounts.all()) or "none registered yet"
        raise ValidationError(
            f"no account {request.account_key!r}. Known accounts: {known}. "
            f"Register one with: transactions.py accounts add <display name>"
        )

    preview = preview_csv(request.path, mapping=request.mapping, date_order=request.date_order)
    records = plan_records(preview, account_id=account.id, mode=request.mode, window=request.window)

    batch, created = ImportBatchRepository(session).open(
        account.id,
        source_file=request.path.name,
        source_sha256=file_sha256(request.path),
        mode=str(request.mode),
        window_start=request.window.start if request.window else None,
        window_end=request.window.end if request.window else None,
        row_count=len(records),
    )
    if not created:
        # Byte-identical file. The one unambiguous "already imported".
        return ImportOutcome(
            batch=batch,
            created=False,
            result=TransactionImportResult(
                attempted_count=len(records), imported_ids=(), already_stored=()
            ),
        )

    result = TransactionRepository(session).add_many(records, batch_id=batch.id)
    return ImportOutcome(batch=batch, created=True, result=result)
