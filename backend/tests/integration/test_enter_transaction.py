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

from offerdelta.api.main import _session, app
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


def test_an_entry_is_stored(session: Session) -> None:
    account = AccountRepository(session).register("Checking")
    outcome = enter_transaction(session, _entry(account.key))

    assert outcome.stored is True
    assert outcome.occurrence == 1
    assert TransactionRepository(session).count(account_id=account.id) == 1


def test_an_unregistered_account_is_refused(session: Session) -> None:
    with pytest.raises(ValidationError, match="no account"):
        enter_transaction(session, _entry("nope"))


def test_a_blank_description_is_refused(session: Session) -> None:
    account = AccountRepository(session).register("Checking")
    with pytest.raises(ValidationError, match="description"):
        enter_transaction(session, _entry(account.key, description="   "))


def test_entering_the_same_thing_twice_writes_once(session: Session) -> None:
    """A form cannot tell two coffees from one typed twice. Refuse by default."""
    account = AccountRepository(session).register("Checking")
    enter_transaction(session, _entry(account.key))
    second = enter_transaction(session, _entry(account.key))

    assert second.stored is False
    assert second.already_stored_count == 1
    assert TransactionRepository(session).count(account_id=account.id) == 1


def test_repeat_asserts_the_second_one_is_real(session: Session) -> None:
    """The one piece of information only the person has."""
    account = AccountRepository(session).register("Checking")
    enter_transaction(session, _entry(account.key))
    second = enter_transaction(session, _entry(account.key, repeat=True))

    assert second.stored is True
    assert second.occurrence == 2
    assert TransactionRepository(session).count(account_id=account.id) == 2


def test_a_third_repeat_keeps_counting(session: Session) -> None:
    account = AccountRepository(session).register("Checking")
    enter_transaction(session, _entry(account.key))
    enter_transaction(session, _entry(account.key, repeat=True))
    third = enter_transaction(session, _entry(account.key, repeat=True))

    assert third.occurrence == 3
    assert TransactionRepository(session).count(account_id=account.id) == 3


def test_two_accounts_do_not_collide(session: Session) -> None:
    repo = AccountRepository(session)
    checking = repo.register("Checking")
    savings = repo.register("Savings")

    enter_transaction(session, _entry(checking.key))
    other = enter_transaction(session, _entry(savings.key))

    assert other.stored is True
    assert other.occurrence == 1


def test_a_manual_entry_carries_no_provenance(session: Session) -> None:
    """No file, no line, no raw cells - the whole reason the plan was dropped."""
    account = AccountRepository(session).register("Checking")
    outcome = enter_transaction(session, _entry(account.key))

    assert outcome.transaction_id is not None
    stored = TransactionRepository(session).get(outcome.transaction_id)
    assert stored is not None
    assert stored.source_file is None
    assert stored.source_line is None
    assert stored.raw_cells is None


def test_the_stored_fingerprint_recomputes_from_its_own_row(session: Session) -> None:
    account = AccountRepository(session).register("Checking")
    outcome = enter_transaction(session, _entry(account.key))
    assert outcome.transaction_id is not None
    stored = TransactionRepository(session).get(outcome.transaction_id)

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
def client(session: Session) -> Iterator[TestClient]:
    """The app, wired to the rolled-back test session rather than a real one."""
    app.dependency_overrides[_session] = lambda: session
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


def test_posting_a_transaction_returns_201(client: TestClient, session: Session) -> None:
    account = AccountRepository(session).register("Checking")
    response = client.post("/v1/transactions", json=_body(account.key))

    assert response.status_code == 201
    assert response.json()["occurrence"] == 1


def test_the_response_never_serialises_money_as_a_number(
    client: TestClient, session: Session
) -> None:
    """The module's rule: a browser parsing 4217.33 as a number loses exactness."""
    account = AccountRepository(session).register("Checking")
    response = client.post("/v1/transactions", json=_body(account.key))

    for value in response.json().values():
        assert not isinstance(value, float)


def test_an_unknown_account_is_404(client: TestClient) -> None:
    response = client.post("/v1/transactions", json=_body("nope"))
    assert response.status_code == 404
    assert "nope" in response.json()["detail"]


def test_a_duplicate_is_409_not_a_silent_success(client: TestClient, session: Session) -> None:
    account = AccountRepository(session).register("Checking")
    client.post("/v1/transactions", json=_body(account.key))
    response = client.post("/v1/transactions", json=_body(account.key))

    assert response.status_code == 409
    assert "repeat=true" in response.json()["detail"]


def test_repeat_true_stores_the_second_one(client: TestClient, session: Session) -> None:
    account = AccountRepository(session).register("Checking")
    client.post("/v1/transactions", json=_body(account.key))
    response = client.post("/v1/transactions", json=_body(account.key, repeat=True))

    assert response.status_code == 201
    assert response.json()["occurrence"] == 2


def test_an_unparseable_amount_is_422(client: TestClient, session: Session) -> None:
    account = AccountRepository(session).register("Checking")
    response = client.post("/v1/transactions", json=_body(account.key, amount="not money"))
    assert response.status_code == 422


def test_a_missing_field_is_422(client: TestClient) -> None:
    response = client.post("/v1/transactions", json={"account_key": "checking"})
    assert response.status_code == 422
