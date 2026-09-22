from __future__ import annotations

import pytest

from offerdelta.agent.tools.definitions import TARGET_METRIC, build_tool_registry
from offerdelta.agent.tools.registry import Tool, ToolRegistry, ToolResult
from offerdelta.domain.common.errors import ValidationError

PAIR = {"current": "auburn_current", "candidate": "new_jersey_candidate"}


def test_registry_exposes_the_six_explicit_tools() -> None:
    registry = build_tool_registry()

    assert registry.names == (
        "break_even",
        "compare_offers",
        "equivalent_salary",
        "explain_component",
        "list_profiles",
        "negotiation_gap",
    )
    assert all(tool.input_schema["additionalProperties"] is False for tool in registry.tools)


def test_registry_rejects_duplicate_names() -> None:
    tool = Tool(
        name="duplicate",
        description="A valid description.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        call=lambda _arguments: ToolResult.success({}),
    )

    with pytest.raises(ValidationError, match="duplicate tool name"):
        ToolRegistry((tool, tool))


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"current": "auburn_current"}, "missing required field 'candidate'"),
        ({**PAIR, "surprise": True}, "unexpected fields: surprise"),
        ({"current": 7, "candidate": "new_jersey_candidate"}, "must be string"),
        ({"current": "missing", "candidate": "new_jersey_candidate"}, "must be one of"),
    ],
)
def test_registry_validates_before_domain_code(arguments: dict[str, object], message: str) -> None:
    result = build_tool_registry().call("break_even", arguments)

    assert not result.ok
    assert result.error is not None
    assert message in result.error


def test_unknown_tool_is_a_model_readable_failure() -> None:
    result = build_tool_registry().call("delete_everything", {})

    assert not result.ok
    assert result.error is not None
    assert "unknown tool" in result.error
    assert "available tools" in result.error


def test_list_profiles_returns_only_public_keys_and_labels() -> None:
    result = build_tool_registry().call("list_profiles", {})

    assert result.ok
    assert result.payload == {
        "profiles": [
            {"key": "auburn_current", "label": "Current - Auburn, AL"},
            {
                "key": "new_jersey_candidate",
                "label": "Candidate - Jersey City, NJ (works NYC)",
            },
        ]
    }


def test_comparison_returns_exact_strings_and_derivations() -> None:
    result = build_tool_registry().call(
        "compare_offers",
        {**PAIR, "horizon_months": 12, "move_date": "2026-07-01"},
    )

    assert result.ok
    assert result.payload["cash_delta"] == "34231.2000000"
    assert result.payload["currency"] == "USD"
    assert result.payload["horizon_months"] == 12
    assert isinstance(result.payload["current_derivation"], dict)
    assert not _contains_float(result.payload)


def test_comparison_rejects_bad_dates_and_same_profile() -> None:
    registry = build_tool_registry()

    bad_date = registry.call(
        "compare_offers", {**PAIR, "horizon_months": 12, "move_date": "07/01/2026"}
    )
    same = registry.call(
        "break_even",
        {"current": "auburn_current", "candidate": "auburn_current"},
    )

    assert not bad_date.ok
    assert bad_date.error is not None
    assert "YYYY-MM-DD" in bad_date.error
    assert not same.ok
    assert same.error == "current and candidate profiles must be different"


def test_solver_and_explanation_tools_return_checked_results() -> None:
    registry = build_tool_registry()

    break_even = registry.call("break_even", PAIR)
    equivalent = registry.call("equivalent_salary", PAIR)
    negotiation = registry.call("negotiation_gap", {**PAIR, "target": TARGET_METRIC})
    housing = registry.call("explain_component", {**PAIR, "component": "housing"})

    assert break_even.payload["stable_break_even_month"] == 1
    assert equivalent.payload["converged"] is True
    assert equivalent.payload["equivalent_salary"] == "88560.60028076171875"
    assert negotiation.payload["needs_negotiation"] is False
    assert isinstance(housing.payload["current"], dict)
    assert isinstance(housing.payload["candidate"], dict)


def test_missing_component_is_an_abstention_not_a_guess() -> None:
    result = build_tool_registry().call("explain_component", {**PAIR, "component": "stock_price"})

    assert not result.ok
    assert result.error == "component 'stock_price' does not exist in this comparison"


def test_tool_result_invariants() -> None:
    with pytest.raises(ValidationError, match="successful"):
        ToolResult(ok=True, payload={}, error="wrong")
    with pytest.raises(ValidationError, match="needs an error"):
        ToolResult(ok=False, payload={})


def _contains_float(value: object) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_contains_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_float(item) for item in value)
    return False
