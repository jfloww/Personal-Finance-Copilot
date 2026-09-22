"""A strict, transport-neutral registry for model-callable tools.

The registry owns the schemas. MCP, an in-process agent, and the evaluation
harness all consume these exact objects; none regenerate a schema from a
function signature. That makes schema drift testable instead of conventional.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from offerdelta.domain.common.errors import ValidationError

type JsonValue = str | int | bool | list[JsonValue] | dict[str, JsonValue] | None
type ToolCall = Callable[[Mapping[str, object]], ToolResult]

_TOOL_NAME = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True)
class ToolResult:
    """A result safe to return to a model or serialize over MCP."""

    ok: bool
    payload: dict[str, JsonValue]
    error: str | None = None

    def __post_init__(self) -> None:
        if self.ok and self.error is not None:
            raise ValidationError("a successful tool result cannot carry an error")
        if not self.ok and not self.error:
            raise ValidationError("a failed tool result needs an error the model can act on")

    @classmethod
    def success(cls, payload: dict[str, JsonValue]) -> ToolResult:
        return cls(ok=True, payload=payload)

    @classmethod
    def failure(cls, error: str) -> ToolResult:
        return cls(ok=False, payload={}, error=error)

    def as_json(self) -> dict[str, JsonValue]:
        return {"ok": self.ok, "payload": self.payload, "error": self.error}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, object]
    call: ToolCall

    def __post_init__(self) -> None:
        if not _TOOL_NAME.fullmatch(self.name):
            raise ValidationError(f"invalid tool name {self.name!r}")
        if not self.description.strip():
            raise ValidationError(f"tool {self.name!r} needs a description")
        if self.input_schema.get("type") != "object":
            raise ValidationError(f"tool {self.name!r} schema root must be an object")
        if self.input_schema.get("additionalProperties") is not False:
            raise ValidationError(f"tool {self.name!r} must set additionalProperties=false")


class ToolRegistry:
    """Validated lookup and execution over an immutable tool set."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        by_name: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in by_name:
                raise ValidationError(f"duplicate tool name {tool.name!r}")
            by_name[tool.name] = tool
        if not by_name:
            raise ValidationError("a tool registry cannot be empty")
        self._by_name = by_name

    @property
    def tools(self) -> tuple[Tool, ...]:
        return tuple(self._by_name[name] for name in sorted(self._by_name))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self.tools)

    def get(self, name: str) -> Tool | None:
        return self._by_name.get(name)

    def call(self, name: str, arguments: Mapping[str, object]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            known = ", ".join(self.names)
            return ToolResult.failure(f"unknown tool {name!r}; available tools: {known}")

        errors = _validate_object(arguments, tool.input_schema)
        if errors:
            return ToolResult.failure("invalid arguments: " + "; ".join(errors))

        try:
            return tool.call(arguments)
        except ValidationError as error:
            return ToolResult.failure(str(error))
        except (KeyError, TypeError, ValueError) as error:
            return ToolResult.failure(f"invalid arguments: {error}")
        except Exception:  # pragma: no cover - defensive boundary
            # Unexpected exceptions stay out of the model context and logs can
            # retain the traceback at the adapter boundary. A model cannot fix
            # an internal traceback, and exposing it leaks implementation detail.
            return ToolResult.failure("tool execution failed")

    def replacing(self, replacement: Tool) -> ToolRegistry:
        if replacement.name not in self._by_name:
            raise ValidationError(f"cannot replace unknown tool {replacement.name!r}")
        return ToolRegistry(
            replacement if tool.name == replacement.name else tool for tool in self.tools
        )


def _validate_object(arguments: Mapping[str, object], schema: Mapping[str, object]) -> list[str]:
    properties = schema.get("properties")
    required = schema.get("required")
    if not isinstance(properties, dict) or not isinstance(required, list):
        return ["tool schema is malformed"]

    errors: list[str] = []
    allowed = set(properties)
    unknown = sorted(set(arguments) - allowed)
    if unknown:
        errors.append(f"unexpected fields: {', '.join(unknown)}")

    for name in required:
        if isinstance(name, str) and name not in arguments:
            errors.append(f"missing required field {name!r}")

    for name, value in arguments.items():
        property_schema = properties.get(name)
        if isinstance(property_schema, dict):
            errors.extend(_validate_value(name, value, property_schema))
    return errors


def _validate_value(name: str, value: object, schema: Mapping[str, object]) -> list[str]:
    if variants := schema.get("anyOf"):
        if isinstance(variants, list) and any(
            isinstance(variant, dict) and not _validate_value(name, value, variant)
            for variant in variants
        ):
            return []
        return [f"field {name!r} does not match any allowed type"]

    expected = schema.get("type")
    valid_type = (
        (expected == "string" and isinstance(value, str))
        or (expected == "integer" and isinstance(value, int) and not isinstance(value, bool))
        or (expected == "boolean" and isinstance(value, bool))
        or (expected == "null" and value is None)
    )
    if not valid_type:
        return [f"field {name!r} must be {expected}"]

    if (enum := schema.get("enum")) and isinstance(enum, list) and value not in enum:
        return [f"field {name!r} must be one of {enum}"]

    errors: list[str] = []
    if isinstance(value, int) and not isinstance(value, bool):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if isinstance(minimum, int) and value < minimum:
            errors.append(f"field {name!r} must be at least {minimum}")
        if isinstance(maximum, int) and value > maximum:
            errors.append(f"field {name!r} must be at most {maximum}")
    if isinstance(value, str):
        minimum_length = schema.get("minLength")
        if isinstance(minimum_length, int) and len(value) < minimum_length:
            errors.append(f"field {name!r} is too short")
    return errors
