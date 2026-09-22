"""The six read-only tools exposed to both the agent and MCP clients."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Final

from offerdelta.agent.tools.registry import JsonValue, Tool, ToolRegistry, ToolResult
from offerdelta.application.queries.demo_profiles import PROFILE_KEYS, list_demo_profiles
from offerdelta.application.queries.get_demo_comparison import (
    HORIZON_MONTHS,
    MOVE_DATE,
    ComparisonView,
    get_demo_comparison,
)
from offerdelta.domain.common.derivation import DerivationNode
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.solvers.negotiation_gap import NegotiationGapResult

TARGET_METRIC: Final = "first_year_disposable_cash"

_PROFILE_PROPERTY: Final[dict[str, object]] = {
    "type": "string",
    "enum": list(PROFILE_KEYS),
}
_PAIR_PROPERTIES: Final[dict[str, object]] = {
    "current": _PROFILE_PROPERTY,
    "candidate": _PROFILE_PROPERTY,
}


def _schema(properties: Mapping[str, object], required: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": required,
        "additionalProperties": False,
    }


def _string(arguments: Mapping[str, object], name: str) -> str:
    value = arguments[name]
    if not isinstance(value, str):  # registry validation makes this defensive
        raise ValidationError(f"{name} must be a string")
    return value


def _integer(arguments: Mapping[str, object], name: str) -> int:
    value = arguments[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValidationError(f"{name} must be an integer")
    return value


def _date(arguments: Mapping[str, object], name: str) -> date | None:
    value = arguments[name]
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValidationError(f"{name} must be an ISO date or null")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValidationError(f"{name} must be an ISO date in YYYY-MM-DD form") from error


def _comparison(arguments: Mapping[str, object], *, configurable: bool = False) -> ComparisonView:
    horizon = _integer(arguments, "horizon_months") if configurable else HORIZON_MONTHS
    move_date = _date(arguments, "move_date") if configurable else MOVE_DATE
    return get_demo_comparison(
        current_key=_string(arguments, "current"),
        candidate_key=_string(arguments, "candidate"),
        horizon_months=horizon,
        move_date=move_date,
    )


def _list_profiles(_arguments: Mapping[str, object]) -> ToolResult:
    profiles: list[JsonValue] = [
        {"key": profile.key, "label": profile.label} for profile in list_demo_profiles()
    ]
    return ToolResult.success({"profiles": profiles})


def _compare_offers(arguments: Mapping[str, object]) -> ToolResult:
    return ToolResult.success(_comparison_payload(_comparison(arguments, configurable=True)))


def _explain_component(arguments: Mapping[str, object]) -> ToolResult:
    view = _comparison(arguments)
    component = _string(arguments, "component")
    current = _find_node(view.current_derivation, component)
    candidate = _find_node(view.candidate_derivation, component)
    if current is None and candidate is None:
        raise ValidationError(f"component {component!r} does not exist in this comparison")
    return ToolResult.success(
        {
            "component": component,
            "current": None if current is None else _derivation(current),
            "candidate": None if candidate is None else _derivation(candidate),
        }
    )


def _break_even(arguments: Mapping[str, object]) -> ToolResult:
    result = _comparison(arguments).break_even
    return ToolResult.success(
        {
            "metric": str(result.metric),
            "horizon_months": result.horizon_months,
            "first_crossing_month": result.first_crossing_month,
            "stable_break_even_month": result.stable_break_even_month,
        }
    )


def _equivalent_salary(arguments: Mapping[str, object]) -> ToolResult:
    view = _comparison(arguments)
    result = view.equivalent_salary
    if result is None:
        return ToolResult.failure(view.equivalent_salary_error or "no equivalent salary found")
    return ToolResult.success(
        {
            "equivalent_salary": str(result.equivalent_salary.amount),
            "currency": result.equivalent_salary.currency,
            "target_metric": result.target_metric,
            "residual": str(result.residual.amount),
            "iterations": result.iterations,
            "converged": result.converged,
            "monotonicity_verified": result.monotonicity_verified,
            "tax_model": result.tax_model_name,
            "calibration_distance_percent": str(result.calibration_distance.as_percent()),
            "is_far_from_calibration": result.is_far_from_calibration,
        }
    )


def _negotiation_gap(arguments: Mapping[str, object]) -> ToolResult:
    target = _string(arguments, "target")
    if target != TARGET_METRIC:
        raise ValidationError(f"unsupported negotiation target {target!r}")
    view = _comparison(arguments)
    result = view.negotiation
    if result is None:
        return ToolResult.failure(view.negotiation_error or "no negotiation result found")
    return ToolResult.success(_negotiation(result))


def _pair_schema(extra: Mapping[str, object] | None = None) -> dict[str, object]:
    properties = dict(_PAIR_PROPERTIES)
    properties.update(extra or {})
    return _schema(properties, list(properties))


def build_tool_registry() -> ToolRegistry:
    """Build a fresh registry so tests can replace tools without global state."""
    return ToolRegistry(
        (
            Tool(
                name="list_profiles",
                description=(
                    "List the closed set of public demo employment profiles. "
                    "Returns stable keys and display labels, never private financial data."
                ),
                input_schema=_schema({}, []),
                call=_list_profiles,
            ),
            Tool(
                name="compare_offers",
                description=(
                    "Compare two demo employment profiles over a projection horizon. "
                    "Returns exact decimal strings, component deltas, cumulative cash, "
                    "and full derivation trees."
                ),
                input_schema=_pair_schema(
                    {
                        "horizon_months": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 120,
                        },
                        "move_date": {
                            "anyOf": [
                                {"type": "string", "format": "date"},
                                {"type": "null"},
                            ]
                        },
                    }
                ),
                call=_compare_offers,
            ),
            Tool(
                name="explain_component",
                description=(
                    "Return the derivation tree for one component code on both sides "
                    "of the default twelve-month comparison."
                ),
                input_schema=_pair_schema({"component": {"type": "string", "minLength": 1}}),
                call=_explain_component,
            ),
            Tool(
                name="break_even",
                description=(
                    "Find the first and stable cash break-even months for the default "
                    "twelve-month comparison."
                ),
                input_schema=_pair_schema(),
                call=_break_even,
            ),
            Tool(
                name="equivalent_salary",
                description=(
                    "Find the candidate base salary that matches the current profile's "
                    "first-year disposable cash."
                ),
                input_schema=_pair_schema(),
                call=_equivalent_salary,
            ),
            Tool(
                name="negotiation_gap",
                description=(
                    "Measure the first-year disposable-cash gap and evaluate individual "
                    "salary, signing-bonus, relocation, and remote-work levers."
                ),
                input_schema=_pair_schema({"target": {"type": "string", "enum": [TARGET_METRIC]}}),
                call=_negotiation_gap,
            ),
        )
    )


def _find_node(root: DerivationNode, code: str) -> DerivationNode | None:
    return next((node for node in root.walk() if node.code == code), None)


def _derivation(node: DerivationNode) -> dict[str, JsonValue]:
    children: list[JsonValue] = [_derivation(child) for child in node.children]
    return {
        "code": node.code,
        "label": node.label,
        "amount": str(node.amount.amount),
        "currency": node.amount.currency,
        "period": str(node.period),
        "formula": node.formula,
        "evidence": str(node.evidence),
        "children": children,
    }


def _comparison_payload(view: ComparisonView) -> dict[str, JsonValue]:
    comparison = view.comparison
    components: list[JsonValue] = [
        {
            "code": component.code,
            "label": component.label,
            "current": str(component.current_cash.amount),
            "candidate": str(component.candidate_cash.amount),
            "delta": str(component.delta.amount),
        }
        for component in comparison.component_deltas
    ]
    cumulative: list[JsonValue] = [str(value.amount) for value in comparison.cumulative_cash_delta]
    return {
        "current_label": view.current_label,
        "candidate_label": view.candidate_label,
        "horizon_months": view.horizon_months,
        "currency": comparison.cash_delta.currency,
        "cash_delta": str(comparison.cash_delta.amount),
        "wealth_delta": str(comparison.wealth_delta.amount),
        "time_delta_hours": str(comparison.time_delta_hours),
        "cumulative_cash_delta": cumulative,
        "component_deltas": components,
        "current_derivation": _derivation(view.current_derivation),
        "candidate_derivation": _derivation(view.candidate_derivation),
    }


def _negotiation(result: NegotiationGapResult) -> dict[str, JsonValue]:
    options: list[JsonValue] = [
        {
            "lever": str(option.lever),
            "feasible": option.feasible,
            "note": option.note,
            "required_amount": (
                None if option.required_amount is None else str(option.required_amount.amount)
            ),
            "required_days": (None if option.required_days is None else str(option.required_days)),
        }
        for option in result.options
    ]
    return {
        "gap": str(result.gap.amount),
        "currency": result.gap.currency,
        "needs_negotiation": result.needs_negotiation,
        "options": options,
    }
