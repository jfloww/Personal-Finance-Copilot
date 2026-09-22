"""Fault injection at the tool boundary, where recovery can be measured."""

from __future__ import annotations

from enum import StrEnum

from offerdelta.agent.tools.registry import Tool, ToolRegistry, ToolResult


class FaultMode(StrEnum):
    ERROR = "error"
    EMPTY = "empty"
    REFUSE = "refuse"


def inject_fault(registry: ToolRegistry, tool_name: str, mode: FaultMode) -> ToolRegistry:
    original = registry.get(tool_name)
    if original is None:
        raise ValueError(f"cannot inject a fault into unknown tool {tool_name!r}")

    def fail(_arguments: object) -> ToolResult:
        if mode is FaultMode.EMPTY:
            return ToolResult.success({})
        if mode is FaultMode.REFUSE:
            return ToolResult.failure("the requested input is outside the supported range")
        return ToolResult.failure("injected tool failure")

    replacement = Tool(
        name=original.name,
        description=original.description,
        input_schema=original.input_schema,
        call=fail,
    )
    return registry.replacing(replacement)
