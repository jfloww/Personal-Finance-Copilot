"""User lifecycle.

There is no registration endpoint, so this is the only way an account comes
into existence. That is the point: an invite-only system whose invitations are
issued by a person with database access has no signup surface to attack.

`set-password` takes the password from OFFERDELTA_NEW_PASSWORD rather than
argv, because an argument lands in shell history and in `ps` output.

Unknown arguments are an error, for the same reason `transactions.py` refuses
them: a silently discarded typo is worse than a rejected one.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.postgres.engine import get_engine
from offerdelta.infrastructure.postgres.repositories import UserRepository

#: Read at `set-password` time, never accepted as an argument. An argument
#: lands in shell history and in `ps` output for the life of the process; an
#: environment variable set inline on the same command line (`VAR=value cmd`)
#: appears in neither.
_PASSWORD_ENV: str = "OFFERDELTA_NEW_PASSWORD"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="users.py", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="create a user; the only way one comes into existence")
    create.add_argument("--email", required=True)
    create.add_argument("--display-name", required=True)

    set_password = sub.add_parser(
        "set-password",
        help=f"set a user's password from ${_PASSWORD_ENV}, never from an argument",
    )
    set_password.add_argument("--email", required=True)

    sub.add_parser("list", help="list every user; never prints a password hash")

    deactivate = sub.add_parser("deactivate", help="deactivate a user so they can no longer log in")
    deactivate.add_argument("--email", required=True)

    return parser


def _report(error: ValidationError | RuntimeError | SQLAlchemyError) -> int:
    """Say what went wrong without echoing anything we have not vetted.

    `config.py` promises the connection string is never logged, echoed, or put
    into an error message. A SQLAlchemy exception breaks that promise if
    printed: it can carry the DSN and the SQL that was running. So only the
    redacted host is named, and nothing from the exception itself.
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


def _create(args: argparse.Namespace) -> int:
    """Create the user. The only entry point that can - there is no signup route."""
    with Session(get_engine()) as session:
        user = UserRepository(session).create(args.email, args.display_name)
        session.commit()
    print(f"created {user.email} ({user.display_name})")
    return 0


def _set_password(args: argparse.Namespace) -> int:
    """Take the new password from the environment, never from argv.

    Checked before the database is even reached, so a missing variable fails
    fast and never opens a connection it does not need.
    """
    password = os.environ.get(_PASSWORD_ENV)
    if not password:
        print(f"{_PASSWORD_ENV} is not set; refusing to accept a password on the command line")
        return 2

    with Session(get_engine()) as session:
        UserRepository(session).set_password(args.email, password)
        session.commit()
    print(f"password set for {args.email}")
    return 0


def _list(_args: argparse.Namespace) -> int:
    """Print who exists without ever printing what proves who they are.

    `has_password` is a boolean precisely so this can say "password set"
    without the hash ever passing through a print statement.
    """
    with Session(get_engine()) as session:
        for user in UserRepository(session).all():
            active = "active" if user.is_active else "inactive"
            password = "password set" if user.has_password else "no password"
            print(f"{user.email:<32}{user.display_name:<24}{active:<10}{password}")
    return 0


def _deactivate(args: argparse.Namespace) -> int:
    """Deactivate the user; their existing password hash is left in place.

    Deactivation only flips `is_active`, so re-activating (were that ever
    added) would not require a new password to be set.
    """
    with Session(get_engine()) as session:
        UserRepository(session).deactivate(args.email)
        session.commit()
    print(f"deactivated {args.email}")
    return 0


def main(argv: list[str]) -> int:
    """Parse, dispatch, and turn any expected failure into an exit code.

    Each subcommand is its own function so this stays a dispatcher: the branch
    count here tracks the number of commands, not the work any of them does.
    """
    args = build_parser().parse_args(argv)
    handlers: dict[str, Callable[[argparse.Namespace], int]] = {
        "create": _create,
        "set-password": _set_password,
        "list": _list,
        "deactivate": _deactivate,
    }

    try:
        return handlers[args.command](args)
    except (ValidationError, RuntimeError, SQLAlchemyError) as error:
        return _report(error)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
