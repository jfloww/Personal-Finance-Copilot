"""The HTTP surface over the monthly report and the review queue.

Every route here depends on `_scope`, so every test either goes in without a
token and expects 401, or authenticates as one tenant (`token_a`) and proves
it never sees, and never reaches, another tenant's rows (`other_scope`). The
service-level behaviour behind these routes - completeness, coverage, the
threshold - is already covered in `test_monthly_report_service.py`; this file
is about status codes, the wire format, and the tenant boundary HTTP adds on
top of that.
"""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from offerdelta.application.reports.review import REVIEW_THRESHOLD
from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.application.transactions.enter_transaction import ManualEntry, enter_transaction
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    TransactionRepository,
    UserRepository,
)
from tests.integration.conftest import requires_auth, requires_database

pytestmark = [requires_database, requires_auth]

KEY = "chase-checking-5718"


@pytest.fixture
def scope_a(session: Session, existing_user_email: str) -> TenantScope:
    """The tenant `token_a` authenticates as, rebuilt as a `TenantScope`.

    `token_a` only carries a bearer token; seeding the rows these tests read
    goes straight through a repository rather than one HTTP POST per row, so
    this looks the same user back up by `existing_user_email` - the address
    the token was actually issued for - rather than minting a third, unrelated
    tenant that would make every "does token_a see its own data" assertion
    trivially true for the wrong reason.
    """
    stored = UserRepository(session).by_email(existing_user_email)
    assert stored is not None
    return TenantScope(session=session, user=AuthenticatedUser(id=stored.id, email=stored.email))


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _seed(scope: TenantScope, rows: list[tuple[str, str]]) -> list[uuid.UUID]:
    """Register an account and hand-enter each `(iso_date, amount)` row.

    Goes through `enter_transaction`, as `test_monthly_report_service.py`'s
    identically-named helper does, so these rows arrive the same way real
    data would rather than as a hand-built `TransactionRow`.
    """
    AccountRepository(scope).register("Chase Checking 5718")
    ids: list[uuid.UUID] = []
    for index, (iso_date, amount) in enumerate(rows):
        outcome = enter_transaction(
            scope,
            ManualEntry(
                account_key=KEY,
                posted_on=date.fromisoformat(iso_date),
                description=f"ROW {index}",
                amount=Money.parse(amount),
                repeat=False,
            ),
        )
        assert outcome.transaction_id is not None
        ids.append(outcome.transaction_id)
    return ids


def _no_floats(value: object) -> None:
    """Every amount in the payload must be a string - see `schemas.py`'s module docstring."""
    assert not isinstance(value, float)
    if isinstance(value, dict):
        for v in value.values():
            _no_floats(v)
    elif isinstance(value, list):
        for v in value:
            _no_floats(v)


# ---------------------------------------------------------------- 401 without a token


def test_reports_months_requires_a_token(client: TestClient) -> None:
    assert client.get("/v1/reports/months").status_code == 401


def test_reports_monthly_requires_a_token(client: TestClient) -> None:
    assert client.get("/v1/reports/monthly/2026-03").status_code == 401


def test_observed_debits_requires_a_token(client: TestClient) -> None:
    assert client.get("/v1/reports/observed-debits/2026-03").status_code == 401


def test_review_queue_requires_a_token(client: TestClient) -> None:
    assert client.get("/v1/review-queue").status_code == 401


def test_confirm_label_requires_a_token(client: TestClient) -> None:
    response = client.post(
        f"/v1/transactions/{uuid.uuid4()}/label", json={"label": "LIVING_DINING"}
    )
    assert response.status_code == 401


# ---------------------------------------------------------------- GET /v1/reports/months


def test_months_lists_a_month_with_rows_as_partial(
    client: TestClient, token_a: str, scope_a: TenantScope
) -> None:
    _seed(scope_a, [("2026-03-05", "-9.99")])
    response = client.get("/v1/reports/months", headers=_auth(token_a))
    assert response.status_code == 200
    months = {(row["year"], row["month"]): row for row in response.json()}
    assert months[(2026, 3)]["complete"] is False
    assert months[(2026, 3)]["rows"] == 1


def test_months_never_lists_another_tenants_month(
    client: TestClient, token_a: str, scope_a: TenantScope, other_scope: TenantScope
) -> None:
    _seed(scope_a, [("2026-03-05", "-9.99")])
    _seed(other_scope, [("2026-04-01", "-1.00")])
    response = client.get("/v1/reports/months", headers=_auth(token_a))
    assert response.status_code == 200
    listed = {(row["year"], row["month"]) for row in response.json()}
    assert listed == {(2026, 3)}


# ---------------------------------------------------------------- GET /v1/reports/monthly/{month}


def test_a_month_with_no_data_is_an_empty_but_valid_tree(client: TestClient, token_a: str) -> None:
    response = client.get("/v1/reports/monthly/2026-01", headers=_auth(token_a))
    assert response.status_code == 200
    body = response.json()
    assert body["tree"]["amount"] == "0"
    assert body["coverage"]["rows"] == 0
    assert body["coverage"]["complete"] is False


@pytest.mark.parametrize("month", ["not-a-month", "2026-13", "2026-3", "0000-00", "9999-99"])
def test_a_malformed_month_is_422(client: TestClient, token_a: str, month: str) -> None:
    """Every value here is one path segment, so it reaches `_parse_month` and
    is refused there. A value containing a `/` (`2026/03`) or an empty
    segment never reaches the route at all - Starlette's own routing 404s it
    before FastAPI resolves a single dependency - so neither belongs in a
    list about what *this handler* refuses."""
    response = client.get(f"/v1/reports/monthly/{month}", headers=_auth(token_a))
    assert response.status_code == 422


def test_the_reports_amounts_are_decimal_strings_not_numbers(
    client: TestClient, token_a: str, scope_a: TenantScope
) -> None:
    """The status check is not decoration: an error body has no float either,
    so without it this would pass trivially on a broken request."""
    _seed(scope_a, [("2026-03-01", "1234.56"), ("2026-03-02", "-12.34")])
    response = client.get("/v1/reports/monthly/2026-03", headers=_auth(token_a))
    assert response.status_code == 200
    body = response.json()
    _no_floats(body)
    assert body["tree"]["amount"] == "1222.22"
    assert isinstance(body["tree"]["amount"], str)


def test_a_tenant_never_sees_another_tenants_rows_in_their_report(
    client: TestClient, token_a: str, scope_a: TenantScope, other_scope: TenantScope
) -> None:
    _seed(scope_a, [("2026-03-01", "-5.00")])
    _seed(other_scope, [("2026-03-02", "-999.00")])

    response = client.get("/v1/reports/monthly/2026-03", headers=_auth(token_a))
    assert response.status_code == 200
    body = response.json()
    assert body["coverage"]["rows"] == 1
    assert body["tree"]["amount"] == "-5.00"


def test_observed_debits_compares_only_this_tenants_rows(
    client: TestClient, token_a: str, scope_a: TenantScope, other_scope: TenantScope
) -> None:
    _seed(scope_a, [("2026-02-05", "-10.10"), ("2026-03-05", "-12.35")])
    _seed(other_scope, [("2026-03-05", "-999.00")])

    response = client.get("/v1/reports/observed-debits/2026-03", headers=_auth(token_a))
    assert response.status_code == 200
    body = response.json()
    _no_floats(body)
    assert body["previous_coverage"]["rows"] == 1
    assert body["current_coverage"]["rows"] == 1
    assert body["previous_coverage"]["complete"] is False
    assert body["current_coverage"]["complete"] is False
    assert body["currencies"][0]["previous"] == "10.10"
    assert body["currencies"][0]["current"] == "12.35"
    assert body["currencies"][0]["delta"] == "2.25"
    assert "transfers" in body["caveat"]


@pytest.mark.parametrize("month", ["not-a-month", "2026-13", "0001-01"])
def test_observed_debits_rejects_invalid_or_uncomparable_month(
    client: TestClient, token_a: str, month: str
) -> None:
    response = client.get(f"/v1/reports/observed-debits/{month}", headers=_auth(token_a))
    assert response.status_code == 422


# ---------------------------------------------------------------- GET /v1/review-queue


def test_the_review_queue_lists_an_unclassified_row(
    client: TestClient, token_a: str, scope_a: TenantScope
) -> None:
    _seed(scope_a, [("2026-03-01", "-5.00")])
    response = client.get("/v1/review-queue", headers=_auth(token_a))
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["amount"] == "-5.00"
    assert body[0]["suggested_label"] is None


def test_the_review_queue_filters_by_month(
    client: TestClient, token_a: str, scope_a: TenantScope
) -> None:
    _seed(scope_a, [("2026-03-01", "-5.00"), ("2026-04-01", "-9.00")])
    response = client.get("/v1/review-queue", headers=_auth(token_a), params={"month": "2026-03"})
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["amount"] == "-5.00"


def test_the_review_queue_malformed_month_is_422(client: TestClient, token_a: str) -> None:
    response = client.get(
        "/v1/review-queue", headers=_auth(token_a), params={"month": "not-a-month"}
    )
    assert response.status_code == 422


def test_the_review_queue_never_contains_another_tenants_rows(
    client: TestClient, token_a: str, scope_a: TenantScope, other_scope: TenantScope
) -> None:
    _seed(scope_a, [("2026-03-01", "-5.00")])
    _seed(other_scope, [("2026-03-02", "-999.00")])

    response = client.get("/v1/review-queue", headers=_auth(token_a))
    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["amount"] == "-5.00"


def test_a_confirmed_row_never_leaves_confirmed_via_the_queue(
    client: TestClient, token_a: str, scope_a: TenantScope
) -> None:
    """A person's own confirmed row - above threshold or below - is settled,
    not awaiting review; `TransactionRepository.confirm_label` is what this
    is proving reaches the queue route correctly."""
    ids = _seed(scope_a, [("2026-03-01", "-5.00")])
    TransactionRepository(scope_a).confirm_label(ids[0], "LIVING_DINING")

    response = client.get("/v1/review-queue", headers=_auth(token_a))
    assert response.status_code == 200
    assert response.json() == []


# ---------------------------------------------------------------- POST /v1/transactions/{id}/label


def test_confirming_a_label_returns_204(
    client: TestClient, token_a: str, scope_a: TenantScope
) -> None:
    ids = _seed(scope_a, [("2026-03-01", "-5.00")])
    response = client.post(
        f"/v1/transactions/{ids[0]}/label",
        json={"label": "LIVING_DINING"},
        headers=_auth(token_a),
    )
    assert response.status_code == 204


def test_a_confirmed_label_is_recorded(
    client: TestClient, token_a: str, scope_a: TenantScope
) -> None:
    ids = _seed(scope_a, [("2026-03-01", "-5.00")])
    client.post(
        f"/v1/transactions/{ids[0]}/label",
        json={"label": "LIVING_DINING"},
        headers=_auth(token_a),
    )
    stored = TransactionRepository(scope_a).get(ids[0])
    assert stored is not None
    assert stored.confirmed_label == "LIVING_DINING"


def test_confirming_a_label_removes_the_row_from_the_queue(
    client: TestClient, token_a: str, scope_a: TenantScope
) -> None:
    ids = _seed(scope_a, [("2026-03-01", "-5.00")])
    assert len(client.get("/v1/review-queue", headers=_auth(token_a)).json()) == 1

    response = client.post(
        f"/v1/transactions/{ids[0]}/label",
        json={"label": "LIVING_DINING"},
        headers=_auth(token_a),
    )
    assert response.status_code == 204

    assert client.get("/v1/review-queue", headers=_auth(token_a)).json() == []


def test_a_label_outside_the_taxonomy_is_422(
    client: TestClient, token_a: str, scope_a: TenantScope
) -> None:
    ids = _seed(scope_a, [("2026-03-01", "-5.00")])
    response = client.post(
        f"/v1/transactions/{ids[0]}/label",
        json={"label": "NOT_A_REAL_LABEL"},
        headers=_auth(token_a),
    )
    assert response.status_code == 422


def test_confirming_unknown_is_422(client: TestClient, token_a: str, scope_a: TenantScope) -> None:
    """Fix 5: `UNKNOWN` is a valid *suggestion* but not a valid *confirmation*.

    Confirming it would leave the row in the review queue forever - there is
    no unconfirm route - and `_group_by_label` evidences that leaf `ASSUMED`
    regardless of `confirmed`, so the month's root could never turn
    `USER_CONFIRMED`. Declining to label a row is done by leaving it alone.
    """
    ids = _seed(scope_a, [("2026-03-01", "-5.00")])
    response = client.post(
        f"/v1/transactions/{ids[0]}/label",
        json={"label": "UNKNOWN"},
        headers=_auth(token_a),
    )
    assert response.status_code == 422


def test_confirming_another_tenants_transaction_is_404_not_403(
    client: TestClient, token_a: str, other_scope: TenantScope
) -> None:
    """404, not 403: a 403 would confirm the row exists."""
    ids = _seed(other_scope, [("2026-03-01", "-5.00")])
    response = client.post(
        f"/v1/transactions/{ids[0]}/label",
        json={"label": "LIVING_DINING"},
        headers=_auth(token_a),
    )
    assert response.status_code == 404


def test_the_404_for_another_tenants_transaction_names_only_the_id(
    client: TestClient, token_a: str, other_scope: TenantScope
) -> None:
    """The message may repeat the id the caller supplied - that told them
    nothing they did not already know - but must not describe the row: no
    amount, no description, nothing that exists only because it was read
    from the other tenant's stored data."""
    ids = _seed(other_scope, [("2026-03-01", "-5.00")])
    response = client.post(
        f"/v1/transactions/{ids[0]}/label",
        json={"label": "LIVING_DINING"},
        headers=_auth(token_a),
    )
    detail = response.json()["detail"]
    assert "ROW 0" not in detail
    assert "5.00" not in detail
    assert str(ids[0]) in detail


def test_confirming_an_unknown_transaction_is_404(client: TestClient, token_a: str) -> None:
    response = client.post(
        f"/v1/transactions/{uuid.uuid4()}/label",
        json={"label": "LIVING_DINING"},
        headers=_auth(token_a),
    )
    assert response.status_code == 404


def test_an_invalid_label_never_reaches_a_valid_row_in_another_tenants_account(
    client: TestClient, token_a: str, other_scope: TenantScope
) -> None:
    """Bad label and wrong tenant, together: the wire-boundary check on the
    label must fire before the tenancy check does, so this is 422 - not the
    404 it would be if the invalid label were allowed through to
    `confirm_label` and happened to be checked second there."""
    ids = _seed(other_scope, [("2026-03-01", "-5.00")])
    response = client.post(
        f"/v1/transactions/{ids[0]}/label",
        json={"label": "NOT_A_REAL_LABEL"},
        headers=_auth(token_a),
    )
    assert response.status_code == 422


# ---------------------------------------------------------------- REVIEW_THRESHOLD


def test_review_threshold_is_the_value_chosen_on_development_and_measured_on_holdout() -> None:
    """Pinned so a future change to the constant is a deliberate edit, not a
    typo that silently changes who lands in the queue."""
    assert Decimal("0.80") == REVIEW_THRESHOLD
