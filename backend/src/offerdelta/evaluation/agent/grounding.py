"""Account for every numeric token in an agent's final answer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal, InvalidOperation, localcontext

from offerdelta.agent.tools.registry import JsonValue
from offerdelta.agent.transcript import AgentRun

_NUMBER = re.compile(
    r"(?<![\w.])(?:\$\s*)?(?P<number>-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)%?"
    r"(?!\w|\.\d|,\d)"
)


@dataclass(frozen=True)
class NumericToken:
    raw: str
    value: Decimal
    decimal_places: int


@dataclass(frozen=True)
class GroundingReport:
    total_numbers: int
    grounded_numbers: int
    fabrications: tuple[str, ...]

    @property
    def answers_with_fabrication(self) -> int:
        return int(bool(self.fabrications))


def score_grounding(run: AgentRun) -> GroundingReport:
    """Score final-answer numbers against the question and successful tools."""
    answer = extract_numbers(run.final_text)
    sources = extract_numbers(run.question)
    for call in run.tool_calls:
        if call.result.ok:
            sources.extend(_payload_numbers(call.result.payload))

    grounded = 0
    fabricated: list[str] = []
    for token in answer:
        if any(_matches(token, source) for source in sources):
            grounded += 1
        else:
            fabricated.append(token.raw)
    return GroundingReport(
        total_numbers=len(answer),
        grounded_numbers=grounded,
        fabrications=tuple(fabricated),
    )


def extract_numbers(text: str) -> list[NumericToken]:
    tokens: list[NumericToken] = []
    for match in _NUMBER.finditer(text):
        raw_number = match.group("number")
        try:
            value = Decimal(raw_number.replace(",", ""))
        except InvalidOperation:  # pragma: no cover - regex admits decimals only
            continue
        fraction = raw_number.rpartition(".")[2] if "." in raw_number else ""
        tokens.append(
            NumericToken(
                raw=match.group(0),
                value=value,
                decimal_places=len(fraction),
            )
        )
    return tokens


def _payload_numbers(payload: dict[str, JsonValue]) -> list[NumericToken]:
    numbers: list[NumericToken] = []
    for value in payload.values():
        if isinstance(value, str):
            numbers.extend(extract_numbers(value))
        elif isinstance(value, int) and not isinstance(value, bool):
            numbers.append(NumericToken(raw=str(value), value=Decimal(value), decimal_places=0))
        elif isinstance(value, list):
            for item in value:
                numbers.extend(_value_numbers(item))
        elif isinstance(value, dict):
            numbers.extend(_payload_numbers(value))
    return numbers


def _value_numbers(value: JsonValue) -> list[NumericToken]:
    if isinstance(value, str):
        return extract_numbers(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return [NumericToken(raw=str(value), value=Decimal(value), decimal_places=0)]
    if isinstance(value, list):
        return [number for item in value for number in _value_numbers(item)]
    if isinstance(value, dict):
        return _payload_numbers(value)
    return []


def _matches(answer: NumericToken, source: NumericToken) -> bool:
    quantum = Decimal(1).scaleb(-answer.decimal_places)
    # Solver outputs can legitimately carry more digits than Decimal's default
    # context precision. Grounding is a comparison, not arithmetic worth
    # rounding through that unrelated global ceiling, so size the local
    # context to the token actually being checked.
    with localcontext() as context:
        context.prec = max(
            50,
            len(source.value.as_tuple().digits) + answer.decimal_places + 2,
        )
        return source.value.quantize(quantum, rounding=ROUND_HALF_EVEN) == answer.value
