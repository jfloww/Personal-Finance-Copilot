from __future__ import annotations

import json

import pytest

from offerdelta.agent.tools.definitions import build_tool_registry
from offerdelta.agent.tools.registry import ToolResult
from offerdelta.agent.transcript import (
    AgentMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from offerdelta.infrastructure.llm.anthropic_agent import (
    AnthropicAgentConfig,
    AnthropicAgentProvider,
)
from offerdelta.infrastructure.llm.errors import MalformedResponseError
from offerdelta.infrastructure.llm.retry import FakeClock
from offerdelta.infrastructure.llm.transport import FakeTransport, json_response


def _response(content: list[dict[str, object]], *, stop_reason: str = "end_turn") -> str:
    return json.dumps(
        {
            "content": content,
            "stop_reason": stop_reason,
            "usage": {"input_tokens": 123, "output_tokens": 45},
        }
    )


def test_agent_provider_sends_registry_schemas_and_parses_tool_use() -> None:
    transport = FakeTransport(
        [
            json_response(
                200,
                _response(
                    [
                        {"type": "text", "text": "I will check."},
                        {
                            "type": "tool_use",
                            "id": "tool-1",
                            "name": "break_even",
                            "input": {
                                "current": "auburn_current",
                                "candidate": "new_jersey_candidate",
                            },
                        },
                    ],
                    stop_reason="tool_use",
                ),
            )
        ]
    )
    provider = AnthropicAgentProvider(
        AnthropicAgentConfig(api_key="test-key"),
        transport=transport,
        clock=FakeClock(),
    )

    response = provider.respond(
        system_prompt="Use tools.",
        messages=[AgentMessage(role="user", content=(TextBlock("When?"),))],
        tools=build_tool_registry().tools,
    )

    assert response.input_tokens == 123
    assert response.output_tokens == 45
    assert isinstance(response.content[0], TextBlock)
    assert isinstance(response.content[1], ToolUseBlock)
    request = json.loads(transport.requests[0].body)
    assert len(request["tools"]) == 6
    break_even = next(tool for tool in request["tools"] if tool["name"] == "break_even")
    assert break_even["input_schema"]["additionalProperties"] is False
    assert "temperature" not in request
    assert "test-key" not in transport.requests[0].body.decode()


def test_tool_results_are_mirrored_back_with_error_status() -> None:
    transport = FakeTransport(
        [json_response(200, _response([{"type": "text", "text": "Cannot answer."}]))]
    )
    provider = AnthropicAgentProvider(AnthropicAgentConfig(api_key="test-key"), transport=transport)
    messages = [
        AgentMessage(
            role="assistant",
            content=(ToolUseBlock(id="tool-1", name="break_even", arguments={}),),
        ),
        AgentMessage(
            role="user",
            content=(
                ToolResultBlock(
                    tool_use_id="tool-1",
                    result=ToolResult.failure("provider unavailable"),
                ),
            ),
        ),
    ]

    provider.respond(
        system_prompt="Use tools.", messages=messages, tools=build_tool_registry().tools
    )

    request = json.loads(transport.requests[0].body)
    block = request["messages"][1]["content"][0]
    assert block["type"] == "tool_result"
    assert block["is_error"] is True
    assert "provider unavailable" in block["content"]


def test_truncated_or_unusable_agent_responses_are_rejected() -> None:
    truncated = AnthropicAgentProvider(
        AnthropicAgentConfig(api_key="test-key"),
        transport=FakeTransport([json_response(200, _response([], stop_reason="max_tokens"))]),
    )
    empty = AnthropicAgentProvider(
        AnthropicAgentConfig(api_key="test-key"),
        transport=FakeTransport([json_response(200, _response([]))]),
    )

    for provider in (truncated, empty):
        with pytest.raises(MalformedResponseError):
            provider.respond(
                system_prompt="Use tools.",
                messages=[AgentMessage(role="user", content=(TextBlock("Question"),))],
                tools=build_tool_registry().tools,
            )


def test_agent_config_redacts_the_key() -> None:
    config = AnthropicAgentConfig(api_key="super-secret")

    assert "super-secret" not in repr(config)
    assert "<redacted>" in repr(config)
