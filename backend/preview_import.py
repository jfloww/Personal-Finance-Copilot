"""Preview a bank export, then optionally commit that exact preview.

    uv run python preview_import.py path/to/export.csv
    uv run python preview_import.py export.csv --day-first
    uv run python preview_import.py export.csv --map date=Posted,description=Details,amount=Value
    uv run python preview_import.py export.csv --commit --account=checking

Previewing is always the default and always happens first. ``--commit`` is the
explicit write boundary; it requires ``--account`` and a preview with no row
errors. Run the preview-only form first, because the cheapest moment to catch a
day-first/month-first mistake is before four hundred rows carry it.

Exits non-zero when nothing could be parsed.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

from sqlalchemy.exc import SQLAlchemyError

from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.postgres.engine import unit_of_work
from offerdelta.infrastructure.postgres.records import Provenance, TransactionRecord
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    TransactionRepository,
)
from offerdelta.ingest.commit import ImportPlan, plan_import
from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import ColumnMapping
from offerdelta.ingest.preview import ImportPreview, preview_csv

#: argv[0] is the script itself, so a path means at least two entries.
_MIN_ARGS = 2

_ORDERS = {
    "--day-first": DateOrder.DAY_FIRST,
    "--month-first": DateOrder.MONTH_FIRST,
    "--iso": DateOrder.ISO,
}


def _parse_map(spec: str) -> ColumnMapping:
    pairs = dict(part.split("=", 1) for part in spec.split(",") if "=" in part)
    return ColumnMapping(
        date=pairs.get("date", ""),
        description=pairs.get("description", ""),
        merchant=pairs.get("merchant"),
        amount=pairs.get("amount"),
        debit=pairs.get("debit"),
        credit=pairs.get("credit"),
    )


def _records(plan: ImportPlan, account_id: uuid.UUID) -> list[TransactionRecord]:
    return [
        TransactionRecord(
            account_id=account_id,
            posted_on=planned.row.posted_on,
            description=planned.row.description,
            normalised_merchant=planned.row.normalised_merchant,
            amount=planned.row.amount,
            external_id=None,
            occurrence=planned.occurrence,
            provenance=Provenance(
                source_file=plan.source_file,
                source_line=planned.row.line,
                raw_cells=planned.row.raw,
            ),
        )
        for planned in plan.rows
    ]


def _commit(preview: ImportPreview, account: str | None) -> int:
    if account is None:
        print("\nnot committed: --commit requires --account=NAME")
        return 2

    try:
        plan = plan_import(preview, account=account)
    except ValidationError as error:
        print(f"\nnot committed: {error}")
        return 1

    if not get_settings().database_available:
        print("\nnot committed: CONNECTION_STRING is not set")
        return 1

    try:
        with unit_of_work() as session:
            accounts = AccountRepository(session)
            stored_account = accounts.by_key(plan.account) or accounts.register(plan.account)
            records = _records(plan, stored_account.id)
            result = TransactionRepository(session).add_many(records)
    except (RuntimeError, ValidationError) as error:
        print(f"\nnot committed: {error}")
        return 1
    except SQLAlchemyError:
        # SQLAlchemy exceptions can include connection details. The unit of
        # work already rolled back; keep the CLI useful without echoing a DSN.
        print("\nnot committed: the database write failed; no rows were committed")
        return 1

    print(f"\ncommitted {result.imported_count} of {result.attempted_count} rows")
    if result.already_stored:
        lines = ", ".join(str(row.source_line) for row in result.already_stored)
        print(f"already stored {result.already_stored_count}: source lines {lines}")
    return 0


def main(argv: list[str]) -> int:
    if len(argv) < _MIN_ARGS:
        print(__doc__)
        return 2

    path = Path(argv[1])
    order = next((_ORDERS[a] for a in argv[2:] if a in _ORDERS), None)
    mapping = next(
        (_parse_map(a.removeprefix("--map=")) for a in argv[2:] if a.startswith("--map=")),
        None,
    )
    should_commit = "--commit" in argv[2:]
    account = next(
        (a.removeprefix("--account=") for a in argv[2:] if a.startswith("--account=")),
        None,
    )

    try:
        preview = preview_csv(path, mapping=mapping, date_order=order)
    except ValidationError as error:
        print(error)
        return 1

    print(preview.render())
    if not should_commit:
        return 0 if preview.importable else 1
    return _commit(preview, account)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
