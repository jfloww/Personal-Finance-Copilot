"""Two synthetic tenants for a deployed database.

**Real bank data stays local.** Everything this script writes is invented: two
addresses that end `@example.test` so nobody can mistake them for real people,
one obviously-synthetic account each, and a handful of transactions against
merchants named "Synthetic ..." so a deployed database can never be confused
with the local one holding real history.

**Two tenants, not one.** With a single tenant the deployment is single-tenant
in practice, and nothing about isolation is exercised by its existence.

**Two months, every branch populated, honestly confirmed.** The monthly
report always emits five branches - income, spending, refunds, transfers,
unclassified - even at zero, so a demo where three of them stay zero shows
less than it could. Each tenant gets income, several spending categories, a
transfer, and a refund, spread across two calendar months, so the deployed
demo can show a month-over-month comparison instead of one static screen.
Every label handed to `confirm_label` comes from `offerdelta.evaluation.labels
.LABEL_SPACE` - the same taxonomy the report tree groups by - so a typo here
would fail loudly at write time instead of seeding a row no report can place.
Labels are *confirmed*, never suggested: synthetic data is honestly
`USER_CONFIRMED`, because a person wrote every row, and confirming it is what
makes the demo tree render `USER_CONFIRMED` evidence rather than `ASSUMED`,
which would misrepresent data nothing here ever guessed at.

A small number of rows in each tenant's first month are left without a
confirmed label on purpose, so the review queue is not empty either - a demo
where the queue has nothing in it cannot show what the queue is for.

**A month reports as complete only if a snapshot window says so.**
`available_months` (`offerdelta.application.reports.monthly`) reads
completeness from a declared `mode="snapshot"` import window, never from row
density - see that module's docstring. Manually entered transactions carry no
batch at all, so without an explicit declaration every seeded month would show
as partial no matter how many rows it held. This script declares one snapshot
window per tenant per seeded month, covering that month's full calendar span,
so the deployed demo shows two complete months rather than two partial ones.

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
rather than writing a second one. A snapshot window is declared unconditionally
every run, but `ImportBatchRepository.open` guards itself on
`(account_id, source_sha256)`: the checksum here is a deterministic digest of
the account key and the month rather than a file hash, so re-declaring the
same month's window on a second run finds the row already there instead of
inserting a duplicate. Label confirmation is guarded by this script, not by
`confirm_label` itself - it has no built-in check, unlike the two calls above
- so before confirming anything this script re-reads the month's stored rows
and skips any that already carry a `confirmed_label`. Without that guard, a
second run would re-stamp `confirmed_at` on every seeded row with a fresh
timestamp, silently rewriting when a person supposedly confirmed it. Running
this script twice against the same database therefore creates nothing the
second time, and confirms nothing a second time either. The transaction dates
are fixed calendar dates rather than computed from "today": a fingerprint
includes the posted date, so if this seeded on `date.today()` instead, running
it again a day later would mint an entirely new set of "duplicate"
transactions instead of recognising the ones already there.

**The password is the exception, and is set on every run.** A read-guarded
password would mean a re-run with a changed `OFFERDELTA_DEMO_PASSWORD` or
`OFFERDELTA_OTHER_PASSWORD` silently kept whatever was set the first time and
still reported success - the operator would have no way to tell a rotation
worked short of trying to log in with it. Setting it unconditionally keeps the
credential in sync with the environment on every run while leaving the user
row, the account, the transactions, the snapshot windows, and the
confirmations untouched if they already exist.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from calendar import monthrange
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
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    ImportBatchRepository,
    StoredAccount,
    TransactionRepository,
    UserRepository,
)
from offerdelta.ingest.commit import ImportMode

_DEMO_PASSWORD_ENV: Final = "OFFERDELTA_DEMO_PASSWORD"
_OTHER_PASSWORD_ENV: Final = "OFFERDELTA_OTHER_PASSWORD"

_DEMO_EMAIL: Final = "demo@example.test"
_OTHER_EMAIL: Final = "other@example.test"

#: Both addresses end `@example.test` so nobody can mistake them for a real
#: person's account - the reservation `.test` TLD exists exactly for this.
SYNTHETIC_TENANTS: Final[tuple[str, str]] = (_DEMO_EMAIL, _OTHER_EMAIL)


@dataclass(frozen=True)
class _SeedTransaction:
    """One invented transaction. `amount` is a decimal string for `Money.parse`.

    `label` is a value from `offerdelta.evaluation.labels.LABEL_SPACE` for a
    row this seed confirms, or `None` for the few rows deliberately left for
    the review queue - see the module docstring. Stated explicitly at every
    call site rather than defaulted, so omitting it is never an accident.
    """

    posted_on: date
    description: str
    amount: str
    label: str | None


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
#:
#: Each tenant spans January and February 2026. January carries one
#: deliberately unconfirmed row so the review queue is not empty; February
#: has none, showing the queue's empty, healthy state on the same deployment.
_TENANTS: Final[tuple[_Tenant, ...]] = (
    _Tenant(
        email=_DEMO_EMAIL,
        display_name="Demo Tenant",
        password_env=_DEMO_PASSWORD_ENV,
        account_display_name="Demo Checking 0001",
        transactions=(
            _SeedTransaction(date(2026, 1, 5), "Synthetic Grocery Co", "-54.12", "LIVING_GROCERY"),
            _SeedTransaction(
                date(2026, 1, 8), "Synthetic Coffee Roasters", "-4.75", "LIVING_DINING"
            ),
            _SeedTransaction(
                date(2026, 1, 12), "Synthetic Electric Utility", "-85.30", "HOUSING_UTILITIES"
            ),
            _SeedTransaction(date(2026, 1, 15), "Synthetic Fuel Station", "-42.00", "COMMUTE_FUEL"),
            _SeedTransaction(
                date(2026, 1, 19), "Synthetic Streaming Co", "-12.99", "LIVING_SUBSCRIPTIONS"
            ),
            _SeedTransaction(
                date(2026, 1, 22), "Synthetic Transfer To Savings", "-300.00", "TRANSFER"
            ),
            _SeedTransaction(date(2026, 1, 26), "Synthetic Payroll Inc", "980.00", "INCOME"),
            _SeedTransaction(date(2026, 1, 29), "Synthetic Bookstore Refund", "18.50", "REFUND"),
            _SeedTransaction(date(2026, 1, 30), "Synthetic Unreviewed Purchase", "-9.99", None),
            _SeedTransaction(date(2026, 2, 5), "Synthetic Grocery Co", "-58.40", "LIVING_GROCERY"),
            _SeedTransaction(
                date(2026, 2, 8), "Synthetic Coffee Roasters", "-5.10", "LIVING_DINING"
            ),
            _SeedTransaction(
                date(2026, 2, 12), "Synthetic Electric Utility", "-79.00", "HOUSING_UTILITIES"
            ),
            _SeedTransaction(date(2026, 2, 15), "Synthetic Fuel Station", "-39.75", "COMMUTE_FUEL"),
            _SeedTransaction(
                date(2026, 2, 19), "Synthetic Streaming Co", "-12.99", "LIVING_SUBSCRIPTIONS"
            ),
            _SeedTransaction(
                date(2026, 2, 22), "Synthetic Transfer To Savings", "-300.00", "TRANSFER"
            ),
            _SeedTransaction(date(2026, 2, 26), "Synthetic Payroll Inc", "980.00", "INCOME"),
            _SeedTransaction(date(2026, 2, 28), "Synthetic Gym Membership", "-29.00", "LIVING_GYM"),
        ),
    ),
    _Tenant(
        email=_OTHER_EMAIL,
        display_name="Other Tenant",
        password_env=_OTHER_PASSWORD_ENV,
        account_display_name="Other Checking 0002",
        transactions=(
            _SeedTransaction(
                date(2026, 1, 6), "Synthetic Hardware Store", "-32.40", "LIVING_OTHER"
            ),
            _SeedTransaction(date(2026, 1, 9), "Synthetic Cafe Bistro", "-6.25", "LIVING_DINING"),
            _SeedTransaction(date(2026, 1, 13), "Synthetic Gym Membership", "-29.00", "LIVING_GYM"),
            _SeedTransaction(
                date(2026, 1, 16), "Synthetic Transit Pass", "-75.00", "COMMUTE_TRANSIT_FARE"
            ),
            _SeedTransaction(
                date(2026, 1, 20), "Synthetic Bookstore", "-18.50", "LIVING_EDUCATION"
            ),
            _SeedTransaction(
                date(2026, 1, 23), "Synthetic Transfer To Savings", "-200.00", "TRANSFER"
            ),
            _SeedTransaction(date(2026, 1, 27), "Synthetic Payroll Inc", "860.00", "INCOME"),
            _SeedTransaction(date(2026, 1, 30), "Synthetic Hardware Return", "12.40", "REFUND"),
            _SeedTransaction(date(2026, 1, 31), "Synthetic Unreviewed Purchase", "-7.25", None),
            _SeedTransaction(
                date(2026, 2, 6), "Synthetic Hardware Store", "-28.10", "LIVING_OTHER"
            ),
            _SeedTransaction(date(2026, 2, 9), "Synthetic Cafe Bistro", "-6.80", "LIVING_DINING"),
            _SeedTransaction(date(2026, 2, 13), "Synthetic Gym Membership", "-29.00", "LIVING_GYM"),
            _SeedTransaction(
                date(2026, 2, 16), "Synthetic Transit Pass", "-75.00", "COMMUTE_TRANSIT_FARE"
            ),
            _SeedTransaction(
                date(2026, 2, 20), "Synthetic Mobile Phone Co", "-45.00", "LIVING_PHONE"
            ),
            _SeedTransaction(
                date(2026, 2, 23), "Synthetic Transfer To Savings", "-200.00", "TRANSFER"
            ),
            _SeedTransaction(date(2026, 2, 27), "Synthetic Payroll Inc", "860.00", "INCOME"),
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


def _months_of(transactions: tuple[_SeedTransaction, ...]) -> list[tuple[int, int]]:
    """Every distinct `(year, month)` this tenant's seed rows touch, sorted."""
    return sorted({(seed.posted_on.year, seed.posted_on.month) for seed in transactions})


def _declare_month_complete(
    scope: TenantScope, account: StoredAccount, year: int, month: int, row_count: int
) -> None:
    """Declare a synthetic snapshot window covering this month in full.

    Without this, `available_months` would report every seeded month as
    partial regardless of how many rows it held: manually entered
    transactions carry no batch at all, and completeness is read only from a
    declared `mode="snapshot"` window - see the module docstring.

    `source_sha256` is not a file hash - nothing was actually parsed - but a
    deterministic digest of the account key and the month. That is what makes
    `ImportBatchRepository.open`'s own `(account_id, source_sha256)` check
    recognise a second declaration of the same month as the same one, rather
    than inserting a second window that would still cover it correctly but
    would leave two rows behind for what is, in truth, one fact.
    """
    window_start = date(year, month, 1)
    window_end = date(year, month, monthrange(year, month)[1])
    digest = hashlib.sha256(
        f"seed-demo-snapshot:{account.key}:{year:04d}-{month:02d}".encode()
    ).hexdigest()
    ImportBatchRepository(scope).open(
        account.id,
        source_file=f"synthetic-{account.key}-{year:04d}-{month:02d}.snapshot",
        source_sha256=digest,
        mode=str(ImportMode.SNAPSHOT),
        window_start=window_start,
        window_end=window_end,
        row_count=row_count,
    )


def _confirm_labels(scope: TenantScope, tenant: _Tenant, months: list[tuple[int, int]]) -> None:
    """Confirm every seed row that carries a label, and none that does not.

    Reads each month's stored rows back once and matches on `(posted_on,
    description)`, unique within a tenant's month by construction, rather than
    trusting the `EntryOutcome` from entering: a repeat run reports an
    existing row without its id, so the id has to come from somewhere that
    works whether the row was just written or was already there.

    A row already carrying `confirmed_label` is skipped. `confirm_label` has
    no guard of its own - see the module docstring - so this check is what
    keeps a second run from re-stamping `confirmed_at` on every seeded row.
    """
    transactions = TransactionRepository(scope)
    stored_by_key = {
        (row.posted_on, row.description): row
        for year, month in months
        for row in transactions.for_month(year, month)
    }
    for seed in tenant.transactions:
        if seed.label is None:
            continue
        row = stored_by_key[(seed.posted_on, seed.description)]
        if row.confirmed_label is None:
            transactions.confirm_label(row.id, seed.label)


def _seed_tenant(session: Session, tenant: _Tenant, password: str) -> None:
    """Create the tenant, its account, its transactions, and their labels.

    Every step but the password and the confirmations reads before it writes:
    `UserRepository.by_email` and `AccountRepository.by_key` are checked
    first, every transaction is entered with `repeat=False`, and every
    snapshot window is declared through `ImportBatchRepository.open`, which
    guards itself. That is what makes a second run against a database this
    already populated a no-op instead of a duplicate-riddled retry - see the
    module docstring. The password is set unconditionally, on every run, so
    rotating the environment variable actually rotates the stored credential
    instead of being silently ignored for a user this script did not create
    today.
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

    months = _months_of(tenant.transactions)
    for year, month in months:
        row_count = sum(
            1
            for seed in tenant.transactions
            if (seed.posted_on.year, seed.posted_on.month) == (year, month)
        )
        _declare_month_complete(scope, account, year, month, row_count)

    _confirm_labels(scope, tenant, months)


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
