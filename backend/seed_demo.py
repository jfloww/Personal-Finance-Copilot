"""Two synthetic tenants for a deployed database.

**Real bank data stays local.** Everything this script writes is invented: two
addresses that end `@example.test` so nobody can mistake them for real people,
one obviously-synthetic account each, and a handful of transactions against
merchants named "Synthetic ..." so a deployed database can never be confused
with the local one holding real history.

**Two tenants, not one.** With a single tenant the deployment is single-tenant
in practice, and nothing about isolation is exercised by its existence.

**Passwords come from the environment, never from argv**, for the same reason
`users.py set-password` reads `OFFERDELTA_NEW_PASSWORD` rather than taking a
flag: an argument lands in shell history and in `ps` output for the life of
the process, while an environment variable set inline on the same command line
(`VAR=value cmd`) appears in neither. `build_parser` declares no `--password`
flag at all, so there is no argument shape a caller could even try to smuggle
one through, and both variables are read and checked before any database
connection is opened.

**Idempotent, by checking rather than by hoping.** Every write but one is
guarded by a read: a user is created only if `by_email` does not find one, an
account only if `by_key` does not, and every transaction is entered with
`repeat=False` so `enter_transaction`'s own fingerprint dedup - already proven
in `tests/integration/test_enter_transaction.py` - reports an existing row
rather than writing a second one. Running this script twice against the same
database therefore creates nothing the second time. The transaction dates are
fixed calendar dates rather than computed from "today": a fingerprint includes
the posted date, so if this seeded on `date.today()` instead, running it again
a day later would mint an entirely new set of "duplicate" transactions instead
of recognising the ones already there.

**The password is the exception, and is set on every run.** A read-guarded
password would mean a re-run with a changed `OFFERDELTA_DEMO_PASSWORD` or
`OFFERDELTA_OTHER_PASSWORD` silently kept whatever was set the first time and
still reported success - the operator would have no way to tell a rotation
worked short of trying to log in with it. Setting it unconditionally keeps the
credential in sync with the environment on every run while leaving the user
row, the account, and the transactions untouched if they already exist.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import date
from typing import Final

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.application.transactions.enter_transaction import ManualEntry, enter_transaction
from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.engine import get_engine
from offerdelta.infrastructure.postgres.repositories import AccountRepository, UserRepository

_DEMO_PASSWORD_ENV: Final = "OFFERDELTA_DEMO_PASSWORD"
_OTHER_PASSWORD_ENV: Final = "OFFERDELTA_OTHER_PASSWORD"

_DEMO_EMAIL: Final = "demo@example.test"
_OTHER_EMAIL: Final = "other@example.test"

#: Both addresses end `@example.test` so nobody can mistake them for a real
#: person's account - the reservation `.test` TLD exists exactly for this.
SYNTHETIC_TENANTS: Final[tuple[str, str]] = (_DEMO_EMAIL, _OTHER_EMAIL)


@dataclass(frozen=True)
class _SeedTransaction:
    """One invented transaction. `amount` is a decimal string for `Money.parse`."""

    posted_on: date
    description: str
    amount: str


@dataclass(frozen=True)
class _Tenant:
    """Everything needed to seed one synthetic tenant, and nothing that isn't."""

    email: str
    display_name: str
    password_env: str
    account_display_name: str
    transactions: tuple[_SeedTransaction, ...]


#: `account_display_name` is chosen so `canonical_account_key` produces exactly
#: `demo-checking-0001` / `other-checking-0002` - obviously-synthetic keys a
#: reviewer of the deployed database can recognise on sight.
_TENANTS: Final[tuple[_Tenant, ...]] = (
    _Tenant(
        email=_DEMO_EMAIL,
        display_name="Demo Tenant",
        password_env=_DEMO_PASSWORD_ENV,
        account_display_name="Demo Checking 0001",
        transactions=(
            _SeedTransaction(date(2026, 1, 5), "Synthetic Grocery Co", "-54.12"),
            _SeedTransaction(date(2026, 1, 12), "Synthetic Coffee Roasters", "-4.75"),
            _SeedTransaction(date(2026, 1, 19), "Synthetic Streaming Co", "-12.99"),
            _SeedTransaction(date(2026, 1, 26), "Synthetic Payroll Inc", "980.00"),
        ),
    ),
    _Tenant(
        email=_OTHER_EMAIL,
        display_name="Other Tenant",
        password_env=_OTHER_PASSWORD_ENV,
        account_display_name="Other Checking 0002",
        transactions=(
            _SeedTransaction(date(2026, 1, 6), "Synthetic Hardware Store", "-32.40"),
            _SeedTransaction(date(2026, 1, 13), "Synthetic Gym Membership", "-29.00"),
            _SeedTransaction(date(2026, 1, 20), "Synthetic Bookstore", "-18.50"),
            _SeedTransaction(date(2026, 1, 27), "Synthetic Payroll Inc", "860.00"),
        ),
    ),
)


def build_parser() -> argparse.ArgumentParser:
    """A parser with no flags of its own.

    Deliberately empty rather than omitted: it still rejects any argument this
    script does not recognise (there is no `parse_known_args` anywhere here),
    which is what makes "no `--password` flag exists" a fact `--help` and a
    failing `parse_args` can both demonstrate, not just a claim in a docstring.
    """
    return argparse.ArgumentParser(prog="seed_demo.py", description=__doc__)


def _report(error: ValidationError | RuntimeError | SQLAlchemyError) -> int:
    """Say what went wrong without echoing anything we have not vetted.

    `config.py` promises the connection string is never logged, echoed, or put
    into an error message. A SQLAlchemy exception breaks that promise if
    printed: it can carry the DSN and the data it was writing. So only the
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


def _read_passwords() -> dict[str, str] | None:
    """Read both passwords before anything else runs.

    Returns `None` and prints which variable is missing rather than raising,
    so the caller can return a clean exit code without opening a database
    connection first - checked here, before `main` ever calls `get_engine()`.
    """
    passwords: dict[str, str] = {}
    for tenant in _TENANTS:
        password = os.environ.get(tenant.password_env)
        if not password:
            print(
                f"{tenant.password_env} is not set; refusing to seed {tenant.email} "
                f"without a password supplied through the environment"
            )
            return None
        passwords[tenant.email] = password
    return passwords


def _seed_tenant(session: Session, tenant: _Tenant, password: str) -> None:
    """Create the tenant, its account, and its transactions if not already there.

    Every step but the password reads before it writes: `UserRepository.by_email`
    and `AccountRepository.by_key` are checked first, and every transaction is
    entered with `repeat=False`. That is what makes a second run against a
    database this already populated a no-op instead of a duplicate-riddled
    retry - see the module docstring. The password is set unconditionally, on
    every run, so rotating the environment variable actually rotates the
    stored credential instead of being silently ignored for a user this
    script did not create today.
    """
    users = UserRepository(session)
    stored_user = users.by_email(tenant.email)
    if stored_user is None:
        stored_user = users.create(tenant.email, tenant.display_name)
    users.set_password(tenant.email, password)

    scope = TenantScope(
        session=session,
        user=AuthenticatedUser(id=stored_user.id, email=stored_user.email),
    )
    accounts = AccountRepository(scope)
    account = accounts.by_key(tenant.account_display_name)
    if account is None:
        account = accounts.register(tenant.account_display_name)

    for seed in tenant.transactions:
        enter_transaction(
            scope,
            ManualEntry(
                account_key=account.key,
                posted_on=seed.posted_on,
                description=seed.description,
                amount=Money.parse(seed.amount),
                repeat=False,
            ),
        )


def main(argv: list[str]) -> int:
    """Parse, check both passwords, seed both tenants, and turn any expected
    failure into an exit code - the same shape as `users.py` and
    `transactions.py`.
    """
    build_parser().parse_args(argv)

    passwords = _read_passwords()
    if passwords is None:
        return 2

    try:
        with Session(get_engine()) as session:
            for tenant in _TENANTS:
                _seed_tenant(session, tenant, passwords[tenant.email])
            session.commit()
    except (ValidationError, RuntimeError, SQLAlchemyError) as error:
        return _report(error)

    print(f"seeded {len(_TENANTS)} synthetic tenant(s): {', '.join(SYNTHETIC_TENANTS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
