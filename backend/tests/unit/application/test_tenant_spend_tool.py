"""The real-data tool is scope-bound, bounded, and runtime-compatible."""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import date
from decimal import Decimal

import pytest
from sqlalchemy.orm import Session

from offerdelta.agent.runtime import AgentRuntime, InProcessToolSource, ScriptedAgentProvider
from offerdelta.agent.tools.operations import build_operations_registry
from offerdelta.agent.tools.registry import ToolRegistry
from offerdelta.agent.transcript import ProviderResponse, TextBlock, ToolUseBlock
from offerdelta.application.agent import tenant_tools
from offerdelta.application.queries.spend_change import SpendChange, SpendEntry, build_spend_change
from offerdelta.application.reports.monthly import MonthCoverage
from offerdelta.application.reports.review import REVIEW_THRESHOLD
from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.domain.common.money import Money


def _comparison() -> SpendChange:
    rows = [
        SpendEntry(
            transaction_id=uuid.UUID(int=merchant * 10 + occurrence),
            posted_on=date(2026, 3, 5),
            merchant=f"MERCHANT {merchant}",
            amount=Money.parse(f"-{merchant}.00"),
            label="LIVING_DINING",
            confirmed=True,
        )
        for merchant in range(1, 8)
        for occurrence in range(7 if merchant == 7 else 1)
    ]
    return build_spend_change(
        [],
        rows,
        MonthCoverage(2026, 2, False, 0, 0, 0),
        MonthCoverage(2026, 3, False, len(rows), len(rows), 0),
    )


@pytest.fixture
def registry(monkeypatch: pytest.MonkeyPatch) -> Iterator[ToolRegistry]:
    comparison = _comparison()

    def fake_explain(
        scope: TenantScope, year: int, month: int, *, threshold: Decimal
    ) -> SpendChange:
        assert scope.user.email == "owner@example.test"
        assert (year, month) == (2026, 3)
        assert threshold == REVIEW_THRESHOLD
        return comparison

    monkeypatch.setattr(tenant_tools, "explain_spend_change", fake_explain)
    with Session() as session:
        scope = TenantScope(
            session=session,
            user=AuthenticatedUser(uuid.UUID(int=1), "owner@example.test"),
        )
        yield tenant_tools.build_tenant_spend_registry(scope)


def test_tool_has_only_month_input_and_rejects_tenant_override(
    registry: ToolRegistry,
) -> None:
    assert registry.names == ("explain_spend_change",)
    assert "explain_spend_change" not in build_operations_registry().names
    schema = registry.tools[0].input_schema
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["month"]
    properties = schema["properties"]
    assert isinstance(properties, dict)
    assert set(properties) == {"month"}

    for arguments in (
        {"month": "2026-03", "user_id": "someone-else"},
        {"month": "2026-03", "account_id": "someone-else"},
        {"month": "2026-13"},
        {"month": "0001-01"},
        {"month": "2026-03-01"},
    ):
        result = registry.call("explain_spend_change", arguments)
        assert not result.ok
        assert result.payload == {}


def test_payload_is_bounded_and_still_reconciles(registry: ToolRegistry) -> None:
    result = registry.call("explain_spend_change", {"month": "2026-03"})
    assert result.ok
    payload = result.payload
    assert payload["read_only"] is True
    assert payload["provisional"] is True
    assert payload["requested_month"] == "2026-03"
    assert payload["previous_month"] == "2026-02"
    currencies = payload["currencies"]
    assert isinstance(currencies, list)
    usd = currencies[0]
    assert isinstance(usd, dict)
    assert usd["currency"] == "USD"
    assert usd["delta"] == "70.00"
    assert usd["other_delta"] == "3.00"
    assert usd["omitted_merchants"] == 2
    drivers = usd["drivers"]
    assert isinstance(drivers, list)
    assert len(drivers) == 5
    top = drivers[0]
    assert isinstance(top, dict)
    assert top["merchant"] == "MERCHANT 7"
    ids = top["current_transaction_ids"]
    assert isinstance(ids, list)
    assert len(ids) == 5
    assert top["current_evidence_omitted"] == 2
    assert sum(Decimal(str(row["delta"])) for row in drivers if isinstance(row, dict)) + Decimal(
        str(usd["other_delta"])
    ) == Decimal(str(usd["delta"]))


def test_scripted_runtime_can_call_the_tenant_tool(registry: ToolRegistry) -> None:
    provider = ScriptedAgentProvider(
        responses=[
            ProviderResponse(
                content=(
                    ToolUseBlock(
                        id="tenant-call-1",
                        name="explain_spend_change",
                        arguments={"month": "2026-03"},
                    ),
                )
            ),
            ProviderResponse(content=(TextBlock("The comparison is provisional."),)),
        ]
    )
    run = AgentRuntime(provider, InProcessToolSource(registry)).run(
        "How did my labelled spending change in March?"
    )
    assert run.stopped_reason == "completed"
    assert len(run.tool_calls) == 1
    assert run.tool_calls[0].result.ok
    assert run.tool_calls[0].result.payload["provisional"] is True


def test_registry_can_be_bound_to_one_month(
    registry: ToolRegistry,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = TenantScope(
        session=Session(),
        user=AuthenticatedUser(uuid.UUID(int=2), "owner@example.test"),
    )
    calls = 0

    def fake_explain(
        _scope: TenantScope, _year: int, _month: int, *, threshold: Decimal
    ) -> SpendChange:
        nonlocal calls
        calls += 1
        assert threshold == REVIEW_THRESHOLD
        return _comparison()

    monkeypatch.setattr(tenant_tools, "explain_spend_change", fake_explain)
    bound = tenant_tools.build_tenant_spend_registry(scope, allowed_month="2026-03")
    assert not bound.call("explain_spend_change", {"month": "2026-02"}).ok
    assert calls == 0
    assert bound.call("explain_spend_change", {"month": "2026-03"}).ok
    assert calls == 1

    assert registry.call("explain_spend_change", {"month": "2026-03"}).ok
