"""Tool-selection and exact-argument scoring against a computed gold trace."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal

from offerdelta.agent.tools.registry import JsonValue
from offerdelta.agent.transcript import AgentRun

_PLACES = Decimal("0.0001")


@dataclass(frozen=True)
class GoldCall:
    name: str
    arguments: dict[str, JsonValue]


@dataclass(frozen=True)
class TraceScore:
    expected_calls: int
    actual_calls: int
    correctly_selected: int
    exact_arguments: int

    @property
    def selection_precision(self) -> Decimal:
        return _ratio(self.correctly_selected, self.actual_calls)

    @property
    def selection_recall(self) -> Decimal:
        return _ratio(self.correctly_selected, self.expected_calls)

    @property
    def argument_accuracy(self) -> Decimal | None:
        if self.correctly_selected == 0:
            return None
        return _ratio(self.exact_arguments, self.correctly_selected)


def score_trace(run: AgentRun, gold: tuple[GoldCall, ...]) -> TraceScore:
    expected_names = Counter(call.name for call in gold)
    actual_names = Counter(call.name for call in run.tool_calls)
    correctly_selected = sum(
        min(count, actual_names[name]) for name, count in expected_names.items()
    )

    unmatched_actual = list(run.tool_calls)
    exact = 0
    for expected in gold:
        match = next(
            (
                call
                for call in unmatched_actual
                if call.name == expected.name and call.arguments == expected.arguments
            ),
            None,
        )
        if match is not None:
            exact += 1
            unmatched_actual.remove(match)

    return TraceScore(
        expected_calls=len(gold),
        actual_calls=len(run.tool_calls),
        correctly_selected=correctly_selected,
        exact_arguments=exact,
    )


def _ratio(numerator: int, denominator: int) -> Decimal:
    if denominator == 0:
        return Decimal(1) if numerator == 0 else Decimal(0)
    return (Decimal(numerator) / Decimal(denominator)).quantize(_PLACES)
