"""Anthropic Messages adapter for the transport-neutral single-agent loop."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final

from offerdelta.agent.tools.registry import Tool
from offerdelta.agent.transcript import (
    AgentMessage,
    ProviderResponse,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.llm.anthropic import API_VERSION, DEFAULT_BASE_URL
from offerdelta.infrastructure.llm.errors import LLMError, MalformedResponseError, classify_status
from offerdelta.infrastructure.llm.retry import Clock, RetryPolicy, SystemClock, call_with_retry
from offerdelta.infrastructure.llm.transport import (
    HttpRequest,
    HttpResponse,
    Transport,
    UrllibTransport,
)

DEFAULT_AGENT_MODEL: Final = "claude-opus-5"
DEFAULT_AGENT_MAX_TOKENS: Final = 2048
_OK: Final = 200


@dataclass(frozen=True)
class AnthropicAgentConfig:
    api_key: str
    model: str = DEFAULT_AGENT_MODEL
    base_url: str = DEFAULT_BASE_URL
    max_tokens: int = DEFAULT_AGENT_MAX_TOKENS
    timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValidationError("ANTHROPIC_API_KEY is required for a live agent run")
        if self.max_tokens < 1:
            raise ValidationError("agent max_tokens must be positive")

    @property
    def messages_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/messages"

    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def __repr__(self) -> str:
        return (
            f"AnthropicAgentConfig(model={self.model!r}, base_url={self.base_url!r}, "
            "api_key='<redacted>')"
        )


@dataclass
class AnthropicAgentProvider:
    config: AnthropicAgentConfig
    transport: Transport = field(default_factory=UrllibTransport)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    clock: Clock = field(default_factory=SystemClock)
    retries: int = 0

    @property
    def model(self) -> str:
        return self.config.model

    def respond(
        self,
        *,
        system_prompt: str,
        messages: Sequence[AgentMessage],
        tools: Sequence[Tool],
    ) -> ProviderResponse:
        body = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "system": system_prompt,
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in tools
            ],
            "messages": [_message(message) for message in messages],
        }
        request = HttpRequest(
            url=self.config.messages_url,
            body=json.dumps(body).encode("utf-8"),
            headers=self.config.headers(),
            timeout_s=self.config.timeout_s,
        )
        started = self.clock.monotonic()

        def attempt() -> HttpResponse:
            response = self.transport.send(request)
            if response.status != _OK:
                raise classify_status(
                    response.status,
                    body=response.text(),
                    retry_after=response.headers.get("Retry-After"),
                )
            return response

        def note_retry(_attempt: int, _error: LLMError, _delay: float) -> None:
            self.retries += 1

        response = call_with_retry(
            attempt,
            policy=self.retry_policy,
            clock=self.clock,
            on_retry=note_retry,
        )
        elapsed_ms = int((self.clock.monotonic() - started) * 1000)
        return _parse(response, elapsed_ms, self.config.max_tokens)


def _message(message: AgentMessage) -> dict[str, object]:
    blocks: list[dict[str, object]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            blocks.append({"type": "text", "text": block.text})
        elif isinstance(block, ToolUseBlock):
            blocks.append(
                {
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.arguments,
                }
            )
        elif isinstance(block, ToolResultBlock):
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.tool_use_id,
                    "content": json.dumps(block.result.as_json()),
                    "is_error": not block.result.ok,
                }
            )
    return {"role": message.role, "content": blocks}


def _parse(response: HttpResponse, elapsed_ms: int, max_tokens: int) -> ProviderResponse:
    try:
        payload = json.loads(response.text())
    except json.JSONDecodeError as error:
        raise MalformedResponseError(f"agent response is not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise MalformedResponseError("agent response JSON is not an object")
    if payload.get("stop_reason") == "max_tokens":
        raise MalformedResponseError(
            f"agent response hit max_tokens ({max_tokens}) before completing"
        )

    raw_content = payload.get("content")
    if not isinstance(raw_content, list):
        raise MalformedResponseError("agent response carries no content blocks")

    content: list[TextBlock | ToolUseBlock] = []
    for raw in raw_content:
        if not isinstance(raw, dict):
            continue
        if raw.get("type") == "text" and isinstance(raw.get("text"), str):
            content.append(TextBlock(raw["text"]))
        elif raw.get("type") == "tool_use":
            identifier = raw.get("id")
            name = raw.get("name")
            arguments = raw.get("input")
            if not isinstance(identifier, str) or not isinstance(name, str):
                raise MalformedResponseError("tool use has no string id or name")
            if not isinstance(arguments, dict) or not all(
                isinstance(key, str) for key in arguments
            ):
                raise MalformedResponseError("tool use input is not an object")
            content.append(ToolUseBlock(id=identifier, name=name, arguments=arguments))

    if not content:
        raise MalformedResponseError("agent response has no usable text or tool call")

    usage = payload.get("usage")
    usage_object = usage if isinstance(usage, dict) else {}
    return ProviderResponse(
        content=tuple(content),
        input_tokens=_non_negative_int(usage_object.get("input_tokens")),
        output_tokens=_non_negative_int(usage_object.get("output_tokens")),
        latency_ms=max(0, elapsed_ms),
    )


def _non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
