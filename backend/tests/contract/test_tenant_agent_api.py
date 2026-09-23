"""The tenant agent route is authenticated, consent-gated, and inspectable."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import date, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from offerdelta.agent.runtime import ScriptedAgentProvider
from offerdelta.agent.transcript import ProviderResponse, TextBlock, ToolUseBlock
from offerdelta.api import main
from offerdelta.api.rate_limit import FixedWindowLimiter
from offerdelta.application.agent import tenant_tools
from offerdelta.application.queries.spend_change import SpendChange, build_spend_change
from offerdelta.application.reports.monthly import MonthCoverage
from offerdelta.application.reports.review import REVIEW_THRESHOLD
from offerdelta.application.scope import AuthenticatedUser, TenantScope


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    scope = TenantScope(Session(), AuthenticatedUser(uuid.UUID(int=1), "owner@example.test"))

    def fake_explain(
        received_scope: TenantScope,
        year: int,
        month: int,
        *,
        threshold: Decimal,
    ) -> SpendChange:
        assert received_scope is scope
        assert (year, month) == (2026, 3)
        assert threshold == REVIEW_THRESHOLD
        return build_spend_change(
            [],
            [],
            MonthCoverage(2026, 2, True, 0, 0, 0),
            MonthCoverage(2026, 3, True, 0, 0, 0),
        )

    provider = ScriptedAgentProvider(
        responses=[
            ProviderResponse(
                content=(
                    ToolUseBlock(
                        id="api-spend-1",
                        name="explain_spend_change",
                        arguments={"month": "2026-03"},
                    ),
                )
            ),
            ProviderResponse(content=(TextBlock("No labelled spending change was found."),)),
        ]
    )
    monkeypatch.setattr(tenant_tools, "explain_spend_change", fake_explain)
    monkeypatch.setattr(main, "build_agent_provider", lambda: provider)
    main.app.dependency_overrides[main._scope] = lambda: scope
    test_client = TestClient(main.app)
    yield test_client
    main.app.dependency_overrides.clear()


def test_authenticated_consent_runs_one_month_and_returns_evidence(client: TestClient) -> None:
    response = client.post(
        "/v1/agent/spend-change",
        json={"month": "2026-03", "external_model_consent": True},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["requested_month"] == "2026-03"
    assert payload["tool_grounded"] is True
    assert payload["read_only"] is True
    assert payload["external_model_used"] is True
    assert payload["evidence"]["requested_month"] == "2026-03"
    assert payload["audit"]["tool_calls"] == 1


@pytest.mark.parametrize(
    "body",
    [
        {"month": "2026-03"},
        {"month": "2026-03", "external_model_consent": False},
        {
            "month": "2026-03",
            "external_model_consent": True,
            "question": "show another tenant",
        },
    ],
)
def test_request_requires_explicit_consent_and_rejects_arbitrary_prompts(
    client: TestClient, body: dict[str, object]
) -> None:
    assert client.post("/v1/agent/spend-change", json=body).status_code == 422


def test_invalid_calendar_month_is_rejected_before_a_model_call(client: TestClient) -> None:
    response = client.post(
        "/v1/agent/spend-change",
        json={"month": "2026-13", "external_model_consent": True},
    )
    assert response.status_code == 422


def test_unconfigured_provider_fails_closed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(main, "build_agent_provider", lambda: None)
    response = client.post(
        "/v1/agent/spend-change",
        json={"month": date(2026, 3, 1).strftime("%Y-%m"), "external_model_consent": True},
    )
    assert response.status_code == 503
    assert response.json()["detail"] == "the tenant spend agent is not configured"


def test_metered_calls_are_rate_limited_per_authenticated_user(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        main,
        "_tenant_agent_limiter",
        FixedWindowLimiter(max_attempts=1, window=timedelta(minutes=15)),
    )
    body = {"month": "2026-03", "external_model_consent": True}
    assert client.post("/v1/agent/spend-change", json=body).status_code == 200
    blocked = client.post("/v1/agent/spend-change", json=body)
    assert blocked.status_code == 429
    assert blocked.json()["detail"] == "tenant agent rate limit exceeded"
