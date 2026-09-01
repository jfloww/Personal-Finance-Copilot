"""No transaction text reaches a log.

`import_batches` stores a file name and a checksum, never the bytes, so this
is very nearly true already - but it is true by accident, and an accident is
not a guarantee. This is the test that makes it a rule: if a future change
adds a debug line that prints a transaction's description or amount, this
test is the one that is supposed to catch it.

That guarantee depends on log capture actually being live, which is not
something to assume. `migrations/env.py` calls Alembic's `fileConfig`, whose
`disable_existing_loggers` default is True - it silently disables every
logger that already existed and is not one of the three names Alembic
configures. In a pytest session that includes the migration tests, every
application logger imported before that point (which, by the time any test
body runs, is effectively all of them) can be switched off for the rest of
the session. A negative assertion - "this text never appears" - passes
trivially under that condition: nothing is captured because nothing can be,
not because nothing leaked. `test_user_repository.py`'s
`test_a_corrupted_password_hash_fails_authentication_without_raising` hit
this first and worked around it locally. `migrations/env.py` now passes
`disable_existing_loggers=False`, which removes the mechanism, but this test
does not lean on that alone: the canary below proves capture is live on this
run, so the negative assertions that follow it mean what they say rather than
passing by accident a second time.
"""

from __future__ import annotations

import logging
from datetime import date

import pytest

from offerdelta.application.scope import TenantScope
from offerdelta.application.transactions.enter_transaction import (
    ManualEntry,
    enter_transaction,
)
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres import repositories as repositories_module
from offerdelta.infrastructure.postgres.repositories import AccountRepository
from tests.integration.conftest import requires_database

pytestmark = requires_database

SECRET_MERCHANT = "ZZQQ UNIQUE MERCHANT STRING"
SECRET_AMOUNT = "-1234.56"

#: Logged through the same logger object `enter_transaction`'s own write path
#: (`TransactionRepository`, in the same module) would use if it ever grew a
#: debug line - so this canary is disabled by exactly the same condition that
#: would silence a real leak, and proves nothing about a logger a leak would
#: not actually use.
_CANARY = "canary: log capture is live for test_no_description_or_amount_reaches_the_logs"


def test_no_description_or_amount_reaches_the_logs(
    scope: TenantScope, caplog: pytest.LogCaptureFixture
) -> None:
    AccountRepository(scope).register("Chase Checking 5718")

    with caplog.at_level(logging.DEBUG):
        repositories_module.logger.debug(_CANARY)
        enter_transaction(
            scope,
            ManualEntry(
                account_key="chase-checking-5718",
                posted_on=date(2026, 3, 1),
                description=SECRET_MERCHANT,
                amount=Money.parse(SECRET_AMOUNT),
                repeat=False,
            ),
        )

    captured = caplog.text

    # Self-defending: if this fails, the logger the write path shares is
    # disabled (see the module docstring) and the two assertions below would
    # hold trivially rather than for the reason they claim to. Fix the
    # capture before trusting the silence, not after.
    assert _CANARY in captured, (
        "log capture is not working - the negative assertions below would "
        "pass regardless of whether a leak occurred, and prove nothing"
    )

    assert SECRET_MERCHANT not in captured
    assert "1234.56" not in captured
