"""Preview a bank export. Committing lives in transactions.py now.

    uv run python preview_import.py path/to/export.csv
    uv run python preview_import.py export.csv --day-first
    uv run python preview_import.py export.csv --map date=Posted,description=Details,amount=Value

Previewing only. ``--commit`` is refused outright: this script's only route to
a window was deriving one from the file's own min/max dates, which can never
reject an out-of-range row — a guarantee that does not hold is worse than
none. Use ``transactions.py`` (Task 11), which commits behind a declared
``--from``/``--to`` window and an explicit ``--yes``.

Exits non-zero when nothing could be parsed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from offerdelta.domain.common.errors import ValidationError
from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import ColumnMapping
from offerdelta.ingest.preview import preview_csv

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

    try:
        preview = preview_csv(path, mapping=mapping, date_order=order)
    except ValidationError as error:
        print(error)
        return 1

    print(preview.render())

    if should_commit:
        print(
            "\ncommitting is disabled in this script.\n"
            "Its import window is derived from the file's own rows, so it can never\n"
            "reject an out-of-range row - a guarantee that does not hold is worse than\n"
            "none. Use transactions.py (Task 11), which commits behind a declared\n"
            "--from/--to window and an explicit --yes."
        )
        return 2

    return 0 if preview.importable else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
