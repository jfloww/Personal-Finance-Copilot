"""Tenant agent answers need successful evidence from the bound tool."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.orm import Session

from offerdelta.agent.runtime import ScriptedAgentProvider
from offerdelta.agent.tools.registry import Tool, ToolRegistry, ToolResult
from offerdelta.agent.transcript import ProviderResponse, TextBlock, ToolUseBlock
from offerdelta.application.agent import tenant_runtime
from offerdelta.application.scope import AuthenticatedUser, TenantScope


@pytest.fixture
def scope() -> TenantScope:
    return TenantScope(
        session=Session(),
        user=AuthenticatedUser(uuid.UUID(int=1), "owner@example.test"),
    )


def _registry(month: str) -> ToolRegistry:
    return ToolRegistry(
        (
            Tool(
                name="explain_spend_change",
                description="Test-only read-only spend evidence.",
                input_schema={
                    "type": "object",
                    "properties": {"month": {"type": "string", "enum": [month]}},
                    "required": ["month"],
                    "additionalProperties": False,
                },
                call=lambda _arguments: ToolResult.success(
                    {"requested_month": month, "delta": "12.34", "read_only": True}
                ),
            ),
        )
    )


def test_successful_tool_result_is_returned_beside_model_text(
    scope: TenantScope,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tenant_runtime,
        "build_tenant_spend_registry",
        lambda _scope, *, allowed_month: _registry(allowed_month),
    )
    provider = ScriptedAgentProvider(
        responses=[
            ProviderResponse(
                content=(
                    ToolUseBlock(
                        id="spend-1",
                        name="explain_spend_change",
                        arguments={"month": "2026-03"},
                    ),
                )
            ),
            ProviderResponse(content=(TextBlock("March increased by 12.34 USD."),)),
        ]
    )
    outcome = tenant_runtime.run_tenant_spend_agent(scope, "2026-03", provider)
    payload = outcome.public_payload(model=provider.model)

    assert outcome.tool_grounded
    assert payload["answer"] == "March increased by 12.34 USD."
    assert payload["evidence"] == {
        "requested_month": "2026-03",
        "delta": "12.34",
        "read_only": True,
    }
    assert payload["read_only"] is True
    assert payload["external_model_used"] is True
    assert payload["audit"] == {
        "model_calls": 2,
        "tool_calls": 1,
        "input_tokens": 0,
        "output_tokens": 0,
        "stopped_reason": "completed",
    }


def test_model_text_without_a_successful_tool_call_is_replaced(
    scope: TenantScope,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tenant_runtime,
        "build_tenant_spend_registry",
        lambda _scope, *, allowed_month: _registry(allowed_month),
    )
    provider = ScriptedAgentProvider(
        responses=[ProviderResponse(content=(TextBlock("I guessed that spending increased."),))]
    )
    outcome = tenant_runtime.run_tenant_spend_agent(scope, "2026-03", provider)
    payload = outcome.public_payload(model=provider.model)

    assert not outcome.tool_grounded
    assert payload["answer"] == tenant_runtime.UNGROUNDED_ANSWER
    assert payload["evidence"] is None


def test_call_for_another_month_cannot_ground_the_answer(
    scope: TenantScope,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tenant_runtime,
        "build_tenant_spend_registry",
        lambda _scope, *, allowed_month: _registry(allowed_month),
    )
    provider = ScriptedAgentProvider(
        responses=[
            ProviderResponse(
                content=(
                    ToolUseBlock(
                        id="wrong-month",
                        name="explain_spend_change",
                        arguments={"month": "2026-02"},
                    ),
                )
            ),
            ProviderResponse(content=(TextBlock("February changed."),)),
        ]
    )
    outcome = tenant_runtime.run_tenant_spend_agent(scope, "2026-03", provider)

    assert not outcome.tool_grounded
    assert not outcome.run.tool_calls[0].result.ok
    assert (
        outcome.public_payload(model=provider.model)["answer"] == tenant_runtime.UNGROUNDED_ANSWER
    )
