"""Typed messages and the complete audit record of one agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from offerdelta.agent.tools.registry import JsonValue, ToolResult


@dataclass(frozen=True)
class TextBlock:
    text: str


@dataclass(frozen=True)
class ToolUseBlock:
    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ToolResultBlock:
    tool_use_id: str
    result: ToolResult


type MessageBlock = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True)
class AgentMessage:
    role: Literal["user", "assistant"]
    content: tuple[MessageBlock, ...]


@dataclass(frozen=True)
class ProviderResponse:
    """One model turn, before any requested tool is executed."""

    content: tuple[TextBlock | ToolUseBlock, ...]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


@dataclass(frozen=True)
class ToolCallRecord:
    id: str
    name: str
    arguments: dict[str, JsonValue]
    result: ToolResult
    latency_ms: int


@dataclass(frozen=True)
class AgentRun:
    question: str
    final_text: str
    messages: tuple[AgentMessage, ...]
    tool_calls: tuple[ToolCallRecord, ...]
    model_calls: int
    input_tokens: int
    output_tokens: int
    model_latencies_ms: tuple[int, ...] = field(default=())
    stopped_reason: Literal["completed", "provider_failure", "turn_limit"] = "completed"

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def public_summary(self) -> dict[str, JsonValue]:
        """Aggregate-safe run facts; deliberately excludes messages and arguments."""
        return {
            "model_calls": self.model_calls,
            "tool_calls": len(self.tool_calls),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stopped_reason": self.stopped_reason,
        }
