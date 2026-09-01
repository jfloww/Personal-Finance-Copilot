"""The boundary as a caller sees it."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from offerdelta.config import get_settings
from offerdelta.infrastructure.postgres.repositories import UserRepository
from tests.integration.conftest import requires_auth, requires_database

#: Both database and a signing key are needed for real: this file mints and
#: verifies actual tokens rather than stubbing `_scope` the way
#: `test_enter_transaction.py` does. CI sets both, so this file runs there;
#: see `requires_auth`'s comment for why a checkout without `JWT_SECRET`
#: skips it instead of failing it.
pytestmark = [requires_database, requires_auth]


def test_a_protected_route_without_a_token_is_401(client: TestClient) -> None:
    response = client.post("/v1/transactions", json={})
    assert response.status_code == 401


def test_a_protected_route_with_a_malformed_token_is_401(client: TestClient) -> None:
    response = client.post(
        "/v1/transactions", json={}, headers={"Authorization": "Bearer not.a.token"}
    )
    assert response.status_code == 401


def test_unknown_email_and_wrong_password_are_indistinguishable(
    client: TestClient, existing_user_email: str
) -> None:
    unknown = client.post(
        "/v1/auth/token",
        json={"email": "nobody@example.test", "password": "whatever"},
    )
    wrong = client.post(
        "/v1/auth/token",
        json={"email": existing_user_email, "password": "not the password"},
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json()


def test_a_good_password_returns_a_token(
    client: TestClient, existing_user_email: str, existing_user_password: str
) -> None:
    response = client.post(
        "/v1/auth/token",
        json={"email": existing_user_email, "password": existing_user_password},
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_a_deactivated_user_is_401(
    client: TestClient, deactivated_user_email: str, existing_user_password: str
) -> None:
    response = client.post(
        "/v1/auth/token",
        json={"email": deactivated_user_email, "password": existing_user_password},
    )
    assert response.status_code == 401


def test_deactivating_a_user_invalidates_their_existing_token(
    client: TestClient,
    session: Session,
    existing_user_email: str,
    existing_user_password: str,
) -> None:
    """The user row is reloaded on every request, not trusted from the token.

    A token minted before deactivation must stop working on the very next
    request, not whenever it happens to expire on its own - up to an hour
    later for this service's tokens.
    """
    login = client.post(
        "/v1/auth/token",
        json={"email": existing_user_email, "password": existing_user_password},
    )
    assert login.status_code == 200, login.text
    token = login.json()["access_token"]

    UserRepository(session).deactivate(existing_user_email)

    response = client.post(
        "/v1/transactions",
        headers={"Authorization": f"Bearer {token}"},
        json={},
    )
    assert response.status_code == 401


def test_the_sixth_failure_is_429(client: TestClient, existing_user_email: str) -> None:
    for _ in range(5):
        client.post(
            "/v1/auth/token",
            json={"email": existing_user_email, "password": "wrong"},
        )
    response = client.post(
        "/v1/auth/token", json={"email": existing_user_email, "password": "wrong"}
    )
    assert response.status_code == 429


def test_a_whitespace_padded_address_shares_the_victims_rate_limit_budget(
    client: TestClient, existing_user_email: str
) -> None:
    """The bypass `normalise_email` closes: the repository matched addresses
    after `.strip().lower()` while the limiter keyed on `.lower()` alone, so a
    leading or trailing space authenticated against the real row while opening
    a fresh limiter bucket for free. Five failures against the bare address
    followed by one against a whitespace-padded variant must land in the same
    bucket and get 429, not a sixth attempt's worth of allowance."""
    for _ in range(5):
        client.post(
            "/v1/auth/token",
            json={"email": existing_user_email, "password": "wrong"},
        )
    response = client.post(
        "/v1/auth/token",
        json={"email": f" {existing_user_email} ", "password": "wrong"},
    )
    assert response.status_code == 429


# ---------------------------------------------------------------- 503 without a secret
#
# `_scope` and `issue_access_token` both read `get_settings().jwt_secret`
# live rather than trusting `_AUTH_CONFIGURED`, the module-level constant
# `include_in_schema` uses - so both are reachable through the same "clear
# the cache, unset the variable" trick this file's own `requires_auth` mark
# would otherwise make unnecessary to exercise here.


def test_a_protected_route_is_503_without_a_signing_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_scope` checks the secret before it even looks at the Authorization
    header, so this must return 503 - not 401 - with no header supplied."""
    monkeypatch.delenv("JWT_SECRET", raising=False)
    get_settings.cache_clear()
    try:
        response = client.post("/v1/transactions", json={})
    finally:
        get_settings.cache_clear()
    assert response.status_code == 503


def test_the_token_endpoint_is_503_without_a_signing_key(
    client: TestClient,
    existing_user_email: str,
    existing_user_password: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`issue_access_token` checks the secret only after a successful password
    match, so a wrong password would return 401 first and never reach this
    branch - the credentials here must be real."""
    monkeypatch.delenv("JWT_SECRET", raising=False)
    get_settings.cache_clear()
    try:
        response = client.post(
            "/v1/auth/token",
            json={"email": existing_user_email, "password": existing_user_password},
        )
    finally:
        get_settings.cache_clear()
    assert response.status_code == 503


def test_one_tenant_gets_404_naming_another_tenants_account(
    client: TestClient, token_a: str, account_key_of_b: str
) -> None:
    """404, not 403: a 403 would confirm the account exists."""
    response = client.post(
        "/v1/transactions",
        headers={"Authorization": f"Bearer {token_a}"},
        json={
            "account_key": account_key_of_b,
            "posted_on": "2026-03-01",
            "description": "BLUE BOTTLE COFFEE",
            "amount": "-12.34",
        },
    )
    assert response.status_code == 404
