"""Manual entry, through the service and through the endpoint.

Both surfaces share one service, so the rules are asserted once here against a
real database and the endpoint tests cover only what HTTP adds: status codes,
the wire format, and the session boundary.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from offerdelta.api.main import _scope, _session, app
from offerdelta.application.scope import TenantScope
from offerdelta.application.transactions.enter_transaction import (
    ManualEntry,
    enter_transaction,
)
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.fingerprint import compute_fingerprint
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    TransactionRepository,
)
from tests.integration.conftest import requires_database

pytestmark = requires_database


def _entry(key: str, *, description: str = "Blue Bottle", repeat: bool = False) -> ManualEntry:
    return ManualEntry(
        account_key=key,
        posted_on=date(2026, 8, 17),
        description=description,
        amount=Money.parse("-4.50"),
        repeat=repeat,
    )


# ---------------------------------------------------------------- the service


def test_an_entry_is_stored(scope: TenantScope) -> None:
    account = AccountRepository(scope).register("Checking")
    outcome = enter_transaction(scope, _entry(account.key))

    assert outcome.stored is True
    assert outcome.occurrence == 1
    assert TransactionRepository(scope).count(account_id=account.id) == 1


def test_an_unregistered_account_is_refused(scope: TenantScope) -> None:
    with pytest.raises(ValidationError, match="no account"):
        enter_transaction(scope, _entry("nope"))


def test_a_blank_description_is_refused(scope: TenantScope) -> None:
    account = AccountRepository(scope).register("Checking")
    with pytest.raises(ValidationError, match="description"):
        enter_transaction(scope, _entry(account.key, description="   "))


def test_entering_the_same_thing_twice_writes_once(scope: TenantScope) -> None:
    """A form cannot tell two coffees from one typed twice. Refuse by default."""
    account = AccountRepository(scope).register("Checking")
    enter_transaction(scope, _entry(account.key))
    second = enter_transaction(scope, _entry(account.key))

    assert second.stored is False
    assert second.already_stored_count == 1
    assert TransactionRepository(scope).count(account_id=account.id) == 1


def test_repeat_asserts_the_second_one_is_real(scope: TenantScope) -> None:
    """The one piece of information only the person has."""
    account = AccountRepository(scope).register("Checking")
    enter_transaction(scope, _entry(account.key))
    second = enter_transaction(scope, _entry(account.key, repeat=True))

    assert second.stored is True
    assert second.occurrence == 2
    assert TransactionRepository(scope).count(account_id=account.id) == 2


def test_a_third_repeat_keeps_counting(scope: TenantScope) -> None:
    account = AccountRepository(scope).register("Checking")
    enter_transaction(scope, _entry(account.key))
    enter_transaction(scope, _entry(account.key, repeat=True))
    third = enter_transaction(scope, _entry(account.key, repeat=True))

    assert third.occurrence == 3
    assert TransactionRepository(scope).count(account_id=account.id) == 3


def test_two_accounts_do_not_collide(scope: TenantScope) -> None:
    repo = AccountRepository(scope)
    checking = repo.register("Checking")
    savings = repo.register("Savings")

    enter_transaction(scope, _entry(checking.key))
    other = enter_transaction(scope, _entry(savings.key))

    assert other.stored is True
    assert other.occurrence == 1


def test_a_manual_entry_carries_no_provenance(scope: TenantScope) -> None:
    """No file, no line, no raw cells - the whole reason the plan was dropped."""
    account = AccountRepository(scope).register("Checking")
    outcome = enter_transaction(scope, _entry(account.key))

    assert outcome.transaction_id is not None
    stored = TransactionRepository(scope).get(outcome.transaction_id)
    assert stored is not None
    assert stored.source_file is None
    assert stored.source_line is None
    assert stored.raw_cells is None


def test_the_stored_fingerprint_recomputes_from_its_own_row(scope: TenantScope) -> None:
    account = AccountRepository(scope).register("Checking")
    outcome = enter_transaction(scope, _entry(account.key))
    assert outcome.transaction_id is not None
    stored = TransactionRepository(scope).get(outcome.transaction_id)

    assert stored is not None
    assert (
        compute_fingerprint(
            account_id=stored.account_id,
            posted_on=stored.posted_on,
            normalised_merchant=stored.normalised_merchant,
            amount=stored.amount,
        )
        == stored.fingerprint
    )


# ---------------------------------------------------------------- the endpoint


@pytest.fixture
def client(session: Session, scope: TenantScope) -> Iterator[TestClient]:
    """The app, wired to the rolled-back test session and this test's tenant.

    `_scope` is overridden directly rather than exercised through a real
    bearer token: these tests are about the HTTP wire format and the session
    boundary, not the identity dependency itself, which `test_auth_api.py`
    covers. Depending on `scope` here - the same fixture the test bodies use
    to register accounts - keeps the account a test registers and the tenant
    the request runs as identical, rather than two different phantom users.
    """
    app.dependency_overrides[_session] = lambda: session
    app.dependency_overrides[_scope] = lambda: scope
    yield TestClient(app)
    app.dependency_overrides.clear()


def _body(key: str, **overrides: object) -> dict[str, object]:
    body: dict[str, object] = {
        "account_key": key,
        "posted_on": "2026-08-17",
        "description": "Blue Bottle",
        "amount": "-4.50",
    }
    body.update(overrides)
    return body


def test_posting_a_transaction_returns_201(client: TestClient, scope: TenantScope) -> None:
    account = AccountRepository(scope).register("Checking")
    response = client.post("/v1/transactions", json=_body(account.key))

    assert response.status_code == 201
    assert response.json()["occurrence"] == 1


def test_the_response_never_serialises_money_as_a_number(
    client: TestClient, scope: TenantScope
) -> None:
    """The module's rule: a browser parsing 4217.33 as a number loses exactness.

    The status check is not decoration: without it an error body satisfies
    this test trivially, since `{"detail": "..."}` also contains no float.
    """
    account = AccountRepository(scope).register("Checking")
    response = client.post("/v1/transactions", json=_body(account.key))

    assert response.status_code == 201
    for value in response.json().values():
        assert not isinstance(value, float)


def test_an_unknown_account_is_404(client: TestClient) -> None:
    response = client.post("/v1/transactions", json=_body("nope"))
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_a_duplicate_is_409_not_a_silent_success(client: TestClient, scope: TenantScope) -> None:
    account = AccountRepository(scope).register("Checking")
    client.post("/v1/transactions", json=_body(account.key))
    response = client.post("/v1/transactions", json=_body(account.key))

    assert response.status_code == 409
    assert "repeat=true" in response.json()["detail"]


def test_repeat_true_stores_the_second_one(client: TestClient, scope: TenantScope) -> None:
    account = AccountRepository(scope).register("Checking")
    client.post("/v1/transactions", json=_body(account.key))
    response = client.post("/v1/transactions", json=_body(account.key, repeat=True))

    assert response.status_code == 201
    assert response.json()["occurrence"] == 2


def test_an_unparseable_amount_is_422(client: TestClient, scope: TenantScope) -> None:
    account = AccountRepository(scope).register("Checking")
    response = client.post("/v1/transactions", json=_body(account.key, amount="not money"))
    assert response.status_code == 422


def test_a_blank_description_is_422_not_404(client: TestClient, scope: TenantScope) -> None:
    """`min_length=1` alone lets `"   "` through - whitespace is not blank.

    A registered account, deliberately: if a blank description ever reached
    `enter_transaction` again, its `ValidationError` would 404 through the
    same handler `test_an_unknown_account_is_404` exercises. A real account
    makes sure this test would catch that regression rather than passing for
    the wrong reason.
    """
    account = AccountRepository(scope).register("Checking")
    response = client.post("/v1/transactions", json=_body(account.key, description="   "))
    assert response.status_code == 422


def test_a_missing_field_is_422(client: TestClient) -> None:
    response = client.post("/v1/transactions", json={"account_key": "checking"})
    assert response.status_code == 422
