"""Preview, commit, and account registration.

Two properties this script is responsible for, both of which the previous one
got wrong.

**Unknown arguments are an error.** The old hand-rolled scan discarded anything
it did not recognise, so `--dayfirst` (a missing hyphen) vanished and four
hundred rows committed with an eleven-month date error. argparse rejects
unknown arguments by default, and nothing here uses `parse_known_args`.

**Preview never falls through into a write.** The old script rendered ten of
four hundred rows and then committed in the same non-interactive invocation,
which made the preview decorative. `commit` is a separate subcommand that
prints a summary, not a preview, and requires `--yes` or an interactive
confirmation.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from sqlalchemy.orm import Session

from offerdelta.application.transactions.import_transactions import ImportRequest, import_csv
from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.postgres.engine import get_engine
from offerdelta.infrastructure.postgres.repositories import AccountRepository
from offerdelta.ingest.commit import ImportMode, ImportWindow
from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import ColumnMapping
from offerdelta.ingest.preview import preview_csv


def _mapping(raw: str) -> ColumnMapping:
    """Parse `field:Column,field:Column,...` into a mapping.

    Wired in as argparse's `type=` for `--map`, so argparse itself catches a
    bad value (either shape here, or a `ColumnMapping` that rejects its own
    fields) and turns it into a clean usage error, not a traceback.
    """
    fields: dict[str, str] = {}
    for pair in raw.split(","):
        name, _, column = pair.partition(":")
        if not name or not column:
            raise argparse.ArgumentTypeError(f"--map entries look like field:Column, got {pair!r}")
        fields[name.strip()] = column.strip()
    try:
        return ColumnMapping(**fields)
    except (ValidationError, TypeError) as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="transactions.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    preview = sub.add_parser("preview", help="show what an import would do; writes nothing")
    preview.add_argument("file", type=Path)
    preview.add_argument("--map", dest="mapping", type=_mapping, default=None)
    preview.add_argument(
        "--dates", dest="dates", choices=[o.value for o in DateOrder], default=None
    )

    commit = sub.add_parser("commit", help="write an import; requires --yes")
    commit.add_argument("file", type=Path)
    commit.add_argument("--account", required=True)
    commit.add_argument("--mode", required=True, choices=[m.value for m in ImportMode])
    commit.add_argument("--from", dest="window_start", type=date.fromisoformat, default=None)
    commit.add_argument("--to", dest="window_end", type=date.fromisoformat, default=None)
    commit.add_argument("--map", dest="mapping", type=_mapping, default=None)
    commit.add_argument("--dates", dest="dates", choices=[o.value for o in DateOrder], default=None)
    commit.add_argument("--yes", action="store_true", help="confirm the write")

    accounts = sub.add_parser("accounts", help="register and list accounts")
    accounts_sub = accounts.add_subparsers(dest="accounts_command", required=True)
    add = accounts_sub.add_parser("add")
    add.add_argument("display_name")
    accounts_sub.add_parser("list")

    return parser


def _confirm(summary: str) -> bool:
    """Refuse unless a human actually types yes.

    `isatty()` is not trustworthy everywhere — on some platforms and shells it
    reports a terminal even when stdin is redirected from /dev/null — so EOF is
    treated as a refusal rather than allowed to raise. A write path must fail
    closed when it cannot ask.
    """
    print(summary)
    if not sys.stdin.isatty():
        print("refusing to write without --yes (stdin is not a terminal)")
        return False
    try:
        answer = input("type 'yes' to write: ")
    except EOFError:
        print("refusing to write without --yes (no input available)")
        return False
    return answer.strip().lower() == "yes"


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    try:
        if args.command == "preview":
            preview = preview_csv(
                args.file,
                mapping=args.mapping,
                date_order=DateOrder(args.dates) if args.dates else None,
            )
            print(preview.render())
            return 0 if preview.importable else 1

        if args.command == "accounts":
            with Session(get_engine()) as session:
                repo = AccountRepository(session)
                if args.accounts_command == "add":
                    account = repo.register(args.display_name)
                    session.commit()
                    print(f"created {account.key}  ({account.display_name})")
                else:
                    for account in repo.all():
                        print(f"{account.key:<24}{account.display_name}")
                return 0

        window = None
        if args.window_start is not None and args.window_end is not None:
            window = ImportWindow(start=args.window_start, end=args.window_end)

        summary = f"{args.file.name} -> account {args.account}, mode {args.mode}" + (
            f", window {args.window_start} to {args.window_end}" if window else ""
        )
        if not args.yes and not _confirm(summary):
            return 2

        with Session(get_engine()) as session:
            outcome = import_csv(
                session,
                ImportRequest(
                    path=args.file,
                    account_key=args.account,
                    mode=ImportMode(args.mode),
                    window=window,
                    mapping=args.mapping,
                    date_order=DateOrder(args.dates) if args.dates else None,
                ),
            )
            session.commit()

        if not outcome.created:
            print(f"already imported: identical file, batch {outcome.batch.id}")
            return 0
        print(f"committed {outcome.result.imported_count} of {outcome.result.attempted_count} rows")
        if outcome.result.already_stored_count:
            lines = ", ".join(str(a.source_line) for a in outcome.result.already_stored[:20])
            print(f"already stored {outcome.result.already_stored_count}: source lines {lines}")
        return 0

    except ValidationError as error:
        print(error)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
