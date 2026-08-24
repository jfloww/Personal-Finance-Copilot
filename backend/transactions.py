"""Preview, commit, and account registration.

Two properties this script is responsible for, both of which the previous one
got wrong.

**Unknown arguments are an error.** The old hand-rolled scan discarded anything
it did not recognise, so `--dayfirst` (a missing hyphen) vanished and four
hundred rows committed with an eleven-month date error. argparse rejects
unknown arguments by default, and nothing here uses `parse_known_args`.

**Preview never falls through into a write.** The old script rendered ten of
four hundred rows and then committed in the same non-interactive invocation,
which made the preview decorative. `commit` is a separate subcommand: it
always prints a one-line summary of the file, account, mode, and any window
*before* doing anything else -- including on `--yes`, where nobody is there
to read a prompt but the run should still leave a record of what it did --
then requires `--yes` or an interactive confirmation to proceed. The summary
carries no row count: getting one would mean parsing the file a second time,
and a second read can silently disagree with the first if the file changes
between them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Final

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from offerdelta.application.transactions.enter_transaction import ManualEntry, enter_transaction
from offerdelta.application.transactions.import_transactions import ImportRequest, import_csv
from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.transactions.parsing import parse_amount
from offerdelta.infrastructure.postgres.engine import get_engine
from offerdelta.infrastructure.postgres.repositories import AccountRepository
from offerdelta.ingest.commit import ImportMode, ImportWindow
from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import AmountSign, ColumnMapping
from offerdelta.ingest.preview import preview_csv

#: The `--map` flag names source columns only. `amount_sign` is a convention,
#: not a column, so it gets its own flag rather than hiding inside this one.
_MAPPABLE: Final = frozenset(
    {"date", "description", "merchant", "external_id", "amount", "debit", "credit"}
)


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
        key = name.strip()
        if key not in _MAPPABLE:
            raise argparse.ArgumentTypeError(
                f"--map takes column names for {', '.join(sorted(_MAPPABLE))}; "
                f"got {key!r}. The sign convention has its own flag, --sign."
            )
        fields[key] = column.strip()
    try:
        return ColumnMapping(**fields)  # type: ignore[arg-type]
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
    preview.add_argument("--sign", choices=[s.value for s in AmountSign], default=None)

    commit = sub.add_parser("commit", help="write an import; requires --yes")
    commit.add_argument("file", type=Path)
    commit.add_argument("--account", required=True)
    commit.add_argument("--mode", required=True, choices=[m.value for m in ImportMode])
    commit.add_argument("--from", dest="window_start", type=date.fromisoformat, default=None)
    commit.add_argument("--to", dest="window_end", type=date.fromisoformat, default=None)
    commit.add_argument("--map", dest="mapping", type=_mapping, default=None)
    commit.add_argument("--dates", dest="dates", choices=[o.value for o in DateOrder], default=None)
    commit.add_argument(
        "--sign",
        choices=[s.value for s in AmountSign],
        default=None,
        help="which sign an outflow carries; Amex exports need outflow-positive",
    )
    commit.add_argument("--yes", action="store_true", help="confirm the write")

    add = sub.add_parser("add", help="enter one transaction by hand; requires --yes")
    add.add_argument("--account", required=True)
    add.add_argument("--date", dest="posted_on", required=True, type=date.fromisoformat)
    add.add_argument("--description", required=True)
    add.add_argument("--amount", required=True, help="signed; negative is money out")
    add.add_argument(
        "--repeat",
        action="store_true",
        help="this really is another identical charge, not a re-entry of one already stored",
    )
    add.add_argument("--yes", action="store_true", help="confirm the write")

    accounts = sub.add_parser("accounts", help="register and list accounts")
    accounts_sub = accounts.add_subparsers(dest="accounts_command", required=True)
    add = accounts_sub.add_parser("add")
    add.add_argument("display_name")
    accounts_sub.add_parser("list")

    return parser


def _confirm() -> bool:
    """Ask whether to proceed. The caller has already printed what for.

    `isatty()` is not trustworthy everywhere — on some platforms and shells it
    reports a terminal even when stdin is redirected from /dev/null — so EOF is
    treated as a refusal rather than allowed to raise. A write path must fail
    closed when it cannot ask.
    """
    if not sys.stdin.isatty():
        print("refusing to write without --yes (stdin is not a terminal)")
        return False
    try:
        answer = input("type 'yes' to write: ")
    except EOFError:
        print("refusing to write without --yes (no input available)")
        return False
    return answer.strip().lower() == "yes"


def _report(error: ValidationError | RuntimeError | SQLAlchemyError) -> int:
    """Say what went wrong without echoing anything we have not vetted.

    `config.py` promises the connection string is never logged, echoed, or put
    into an error message. A SQLAlchemy exception breaks that promise if
    printed: it can carry the DSN and the SQL that was running, real amounts
    and descriptions included. So only the redacted host is named, and nothing
    from the exception itself.
    """
    if isinstance(error, SQLAlchemyError):
        host = get_settings().redacted_dsn
        print(f"database error while reaching {host}; the operation did not complete")
    elif isinstance(error, RuntimeError) and get_settings().database_available:
        # The RuntimeError we expect is get_engine() finding CONNECTION_STRING
        # unset, whose message names no host and no SQL. Any other RuntimeError
        # came from somewhere we have not reasoned about, so it is not echoed.
        print("the operation did not complete; an unexpected internal error occurred")
    else:
        # A ValidationError's message is always ours, as is the unset-DSN one.
        print(error)
    return 1


def _add(args: argparse.Namespace) -> int:
    """Write one hand-entered transaction behind the same gate as an import."""
    entry = ManualEntry(
        account_key=args.account,
        posted_on=args.posted_on,
        description=args.description,
        amount=parse_amount(args.amount),
        repeat=args.repeat,
    )

    summary = (
        f"{entry.posted_on} {entry.description} {entry.amount.amount:.2f} "
        f"-> account {entry.account_key}" + (" (repeat)" if entry.repeat else "")
    )
    print(summary)
    if not args.yes and not _confirm():
        return 2

    with Session(get_engine()) as session:
        outcome = enter_transaction(session, entry)
        session.commit()

    if not outcome.stored:
        print(
            f"already stored: {outcome.already_stored_count} identical "
            f"transaction(s) exist. Pass --repeat if there really was another one."
        )
        return 1

    print(f"stored occurrence {outcome.occurrence} ({outcome.transaction_id})")
    return 0


def _accounts(args: argparse.Namespace) -> int:
    """Register an account, or list the ones that exist.

    Registration is deliberately its own command: an import refuses an unknown
    account rather than creating one, because a typo that auto-creates is the
    original bug wearing a different hat.
    """
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


def _preview(args: argparse.Namespace) -> int:
    """Show what an import would do. Writes nothing, ever."""
    preview = preview_csv(
        args.file,
        mapping=args.mapping,
        date_order=DateOrder(args.dates) if args.dates else None,
        amount_sign=AmountSign(args.sign) if args.sign else None,
    )
    print(preview.render())
    return 0 if preview.importable else 1


def _commit(args: argparse.Namespace) -> int:
    """Write an import, behind the summary and the confirmation gate."""
    window = None
    if args.window_start is not None and args.window_end is not None:
        window = ImportWindow(start=args.window_start, end=args.window_end)

    summary = f"{args.file.name} -> account {args.account}, mode {args.mode}" + (
        f", window {args.window_start} to {args.window_end}" if window else ""
    )
    print(summary)
    if not args.yes and not _confirm():
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
                amount_sign=AmountSign(args.sign) if args.sign else None,
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


def main(argv: list[str]) -> int:
    """Parse, dispatch, and turn any expected failure into an exit code.

    Each subcommand is its own function so this stays a dispatcher: the branch
    count here tracks the number of commands, not the work any of them does.
    """
    args = build_parser().parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "preview": _preview,
        "add": _add,
        "accounts": _accounts,
        "commit": _commit,
    }

    try:
        return handlers[args.command](args)
    except (ValidationError, RuntimeError, SQLAlchemyError) as error:
        return _report(error)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
