from __future__ import annotations

from offerdelta.agent.runtime import (
    PROVIDER_FAILURE_TEXT,
    TURN_LIMIT_TEXT,
    AgentRuntime,
    InProcessToolSource,
    ScriptedAgentProvider,
)
from offerdelta.agent.tools.definitions import build_tool_registry
from offerdelta.agent.transcript import ProviderResponse, TextBlock, ToolUseBlock

PAIR: dict[str, object] = {
    "current": "auburn_current",
    "candidate": "new_jersey_candidate",
}


def test_agent_executes_a_tool_then_records_a_grounded_answer() -> None:
    provider = ScriptedAgentProvider(
        responses=[
            ProviderResponse(
                content=(ToolUseBlock(id="call-1", name="break_even", arguments=PAIR),),
                input_tokens=20,
                output_tokens=5,
            ),
            ProviderResponse(
                content=(TextBlock("It first and stably breaks even in month 1."),),
                input_tokens=40,
                output_tokens=10,
            ),
        ]
    )

    run = AgentRuntime(provider, InProcessToolSource(build_tool_registry())).run(
        "When does it break even?"
    )

    assert run.final_text == "It first and stably breaks even in month 1."
    assert run.model_calls == 2
    assert run.total_tokens == 75
    assert len(run.tool_calls) == 1
    assert run.tool_calls[0].result.ok
    assert run.tool_calls[0].result.payload["first_crossing_month"] == 1
    assert len(run.messages) == 4


def test_unknown_tool_result_is_returned_to_the_provider() -> None:
    provider = ScriptedAgentProvider(
        responses=[
            ProviderResponse(content=(ToolUseBlock(id="call-1", name="not_a_tool", arguments={}),)),
            ProviderResponse(content=(TextBlock("I cannot answer with these tools."),)),
        ]
    )

    run = AgentRuntime(provider, InProcessToolSource(build_tool_registry())).run("Do it")

    assert not run.tool_calls[0].result.ok
    assert run.tool_calls[0].result.error is not None
    assert "unknown tool" in run.tool_calls[0].result.error


def test_provider_failure_abstains_without_exposing_the_exception() -> None:
    provider = ScriptedAgentProvider(responses=[])

    run = AgentRuntime(provider, InProcessToolSource(build_tool_registry())).run("Compare them")

    assert run.final_text == PROVIDER_FAILURE_TEXT
    assert run.stopped_reason == "provider_failure"


def test_turn_limit_abstains() -> None:
    provider = ScriptedAgentProvider(
        responses=[
            ProviderResponse(
                content=(ToolUseBlock(id="call-1", name="break_even", arguments=PAIR),)
            )
        ]
    )

    run = AgentRuntime(
        provider,
        InProcessToolSource(build_tool_registry()),
        max_model_turns=1,
    ).run("When?")

    assert run.final_text == TURN_LIMIT_TEXT
    assert run.stopped_reason == "turn_limit"


def test_duplicate_tool_call_ids_are_rejected() -> None:
    provider = ScriptedAgentProvider(
        responses=[
            ProviderResponse(
                content=(
                    ToolUseBlock(id="same", name="break_even", arguments=PAIR),
                    ToolUseBlock(id="same", name="equivalent_salary", arguments=PAIR),
                )
            ),
            ProviderResponse(content=(TextBlock("The second call was rejected."),)),
        ]
    )

    run = AgentRuntime(provider, InProcessToolSource(build_tool_registry())).run("Both")

    assert len(run.tool_calls) == 2
    assert run.tool_calls[0].result.ok
    assert not run.tool_calls[1].result.ok
    assert run.tool_calls[1].result.error == "tool call ids must be non-empty and unique"
    tool_results = provider.requests[1][-1].content
    assert len(tool_results) == 2
