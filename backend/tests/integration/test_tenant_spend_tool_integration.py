"""A real-data tool cannot cross tenants or write during a read."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

from sqlalchemy.orm import Session

from offerdelta.application.agent.tenant_tools import build_tenant_spend_registry
from offerdelta.application.scope import TenantScope
from offerdelta.application.transactions.enter_transaction import ManualEntry, enter_transaction
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.repositories import AccountRepository, TransactionRepository
from tests.integration.conftest import requires_database

pytestmark = requires_database


def _seed(scope: TenantScope, account_key: str, amount: str, when: date) -> uuid.UUID:
    outcome = enter_transaction(
        scope,
        ManualEntry(
            account_key=account_key,
            posted_on=when,
            description="STORE",
            amount=Money.parse(amount),
            repeat=False,
        ),
    )
    assert outcome.transaction_id is not None
    TransactionRepository(scope).confirm_label(outcome.transaction_id, "LIVING_DINING")
    return outcome.transaction_id


def test_registry_is_bound_to_one_tenant_and_does_not_write(
    scope: TenantScope, other_scope: TenantScope, session: Session
) -> None:
    account = AccountRepository(scope).register("Checking")
    other_account = AccountRepository(other_scope).register("Checking")
    previous_id = _seed(scope, account.key, "-2.00", date(2026, 2, 5))
    current_id = _seed(scope, account.key, "-5.00", date(2026, 3, 5))
    other_id = _seed(other_scope, other_account.key, "-99.00", date(2026, 3, 5))
    session.flush()

    registry = build_tenant_spend_registry(scope)
    result = registry.call("explain_spend_change", {"month": "2026-03"})
    assert result.ok
    currencies = result.payload["currencies"]
    assert isinstance(currencies, list)
    usd = currencies[0]
    assert isinstance(usd, dict)
    assert usd["delta"] == "3.00"
    drivers = usd["drivers"]
    assert isinstance(drivers, list)
    assert len(drivers) == 1
    driver = drivers[0]
    assert isinstance(driver, dict)
    assert driver["previous_transaction_ids"] == [str(previous_id)]
    assert driver["current_transaction_ids"] == [str(current_id)]
    assert str(other_id) not in str(result.payload)
    assert not session.new
    assert not session.dirty
    assert not session.deleted

    other_result = build_tenant_spend_registry(other_scope).call(
        "explain_spend_change", {"month": "2026-03"}
    )
    assert other_result.ok
    other_currencies = other_result.payload["currencies"]
    assert isinstance(other_currencies, list)
    other_usd = other_currencies[0]
    assert isinstance(other_usd, dict)
    assert other_usd["delta"] == "99.00"
    assert Decimal(str(other_usd["delta"])) != Decimal(str(usd["delta"]))
