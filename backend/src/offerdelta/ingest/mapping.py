"""Working out which column is which.

Bank exports agree on nothing. The same field is `Description`, `Details`,
`Memo`, `Narrative`, `Payee` or `Transaction`; money out is a negative `Amount`
in one file and a positive `Debit` in the next.

Detection therefore uses two signals and requires both to agree where it can:
the **header name** against a list of aliases, and the **values themselves**. A
column called `date` full of merchant names is not a date column, and a header
guess that the values contradict is dropped rather than trusted.

Uncertainty is reported, never resolved by picking. `detect_mapping` returns
what it found *and* what it could not settle, and the importer refuses to run
until an explicit mapping fills the gaps. A wrong column here corrupts every
figure downstream in a way no later check can catch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from offerdelta.domain.common.errors import ValidationError
from offerdelta.ingest.dates import looks_like_dates
from offerdelta.ingest.numbers import looks_like_amounts

#: Header aliases, lowercased and stripped of punctuation before matching.
#:
#: **Order is preference.** Every Chase export carries both a transaction
#: date and a post date, so refusing to choose would make the most common
#: export in the world unimportable. The transaction date wins because it is
#: when the purchase happened. The choice is reported, never hidden.
_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "date": (
        "date",
        "transaction date",
        "trans date",
        "posted date",
        "post date",
        "posting date",
        "date posted",
        "value date",
        "completed date",
    ),
    "description": (
        "description",
        "details",
        "memo",
        "narrative",
        "transaction",
        "particulars",
        "reference",
        "notes",
    ),
    "merchant": ("merchant", "payee", "vendor", "name", "counterparty", "beneficiary"),
    "amount": ("amount", "value", "transaction amount", "amt"),
    "debit": (
        "debit",
        "debits",
        "withdrawal",
        "withdrawals",
        "money out",
        "paid out",
        "charge",
        "charges",
        "outflow",
    ),
    "credit": (
        "credit",
        "credits",
        "deposit",
        "deposits",
        "money in",
        "paid in",
        "inflow",
    ),
}


def _canonical(header: str) -> str:
    cleaned = "".join(c if c.isalnum() or c.isspace() else " " for c in header.lower())
    return " ".join(cleaned.split())


class AmountSign(StrEnum):
    """Which sign an outflow carries in a signed amount column.

    Most banks write money leaving the account as negative. American Express
    does the opposite: a charge is positive and a payment or credit is
    negative. Taking a signed column at face value therefore reads every Amex
    charge as income and every payment as spending - both totals wrong, in
    opposite directions, and both entirely plausible-looking.

    There is no reliable way to detect this from the numbers alone, because a
    month of mostly-refunds is a real thing. So the convention is declared per
    import rather than guessed.
    """

    #: Money out is negative. Chase, Capital One, and most banks.
    OUTFLOW_NEGATIVE = "outflow-negative"

    #: Money out is positive. American Express.
    OUTFLOW_POSITIVE = "outflow-positive"


@dataclass(frozen=True)
class ColumnMapping:
    """Which source column supplies each canonical field.

    Either `amount` carries a signed value, or `debit` and `credit` carry
    magnitudes in separate columns. Both shapes are common and the importer
    normalises them to one signed amount.
    """

    date: str
    description: str
    merchant: str | None = None

    #: The bank's own transaction id, when the export carries one. Its presence
    #: is what makes an incremental import possible at all.
    external_id: str | None = None
    amount: str | None = None
    debit: str | None = None
    credit: str | None = None

    #: Only meaningful with a signed `amount` column: split debit/credit
    #: columns carry magnitudes, so their direction is structural.
    amount_sign: AmountSign = AmountSign.OUTFLOW_NEGATIVE

    def __post_init__(self) -> None:
        if not self.date:
            raise ValidationError("a mapping needs a date column")
        if not self.description:
            raise ValidationError("a mapping needs a description column")

        has_signed = bool(self.amount)
        has_split = bool(self.debit or self.credit)

        if not has_signed and not has_split:
            raise ValidationError(
                "a mapping needs either an amount column or a debit/credit pair; "
                "without one there is no money in the file"
            )
        if has_signed and has_split:
            raise ValidationError(
                "a mapping has both an amount column and a debit/credit pair. "
                "Reading both would double every row; choose one."
            )

    @property
    def uses_split_columns(self) -> bool:
        return not self.amount

    def source_columns(self) -> tuple[str, ...]:
        named = (
            self.date,
            self.description,
            self.merchant,
            self.external_id,
            self.amount,
            self.debit,
            self.credit,
        )
        return tuple(column for column in named if column)


@dataclass(frozen=True)
class MappingDetection:
    """What detection found, and what it could not settle."""

    mapping: ColumnMapping | None
    matched: dict[str, str] = field(default_factory=dict)

    #: Fields where the header suggested a column but the values disagreed.
    uncertain: tuple[str, ...] = ()

    #: Choices made between several plausible columns. Reported so a reader
    #: can overrule them, never silently applied.
    resolved: tuple[str, ...] = ()

    #: Why a mapping could not be built at all.
    problems: tuple[str, ...] = ()

    @property
    def confident(self) -> bool:
        return self.mapping is not None and not self.uncertain

    def render(self) -> str:
        lines = []
        if self.mapping is None:
            lines.append("could not map the file automatically")
        else:
            lines.append("detected mapping")
            for canonical, column in sorted(self.matched.items()):
                lines.append(f"    {canonical:<12} <- {column}")
        for problem in self.problems:
            lines.append(f"  problem: {problem}")
        for choice in self.resolved:
            lines.append(f"  resolved: {choice}")
        for field_name in self.uncertain:
            lines.append(f"  uncertain: {field_name}")
        return "\n".join(lines)


def _confirm_against_values(
    matched: dict[str, str], sample: dict[str, list[str]], uncertain: list[str]
) -> None:
    """Drop a header guess the values contradict.

    A column called `date` full of merchant names is not a date column, and
    trusting the name would corrupt every row.
    """
    date_column = matched.get("date")
    if date_column and not looks_like_dates(sample.get(date_column, [])):
        uncertain.append(f"date: column {date_column!r} does not contain dates")
        matched.pop("date")

    for money_field in ("amount", "debit", "credit"):
        column = matched.get(money_field)
        if column and not looks_like_amounts(sample.get(column, [])):
            uncertain.append(f"{money_field}: column {column!r} does not contain amounts")
            matched.pop(money_field)


def _fill_missing_date(
    matched: dict[str, str],
    headers: list[str],
    sample: dict[str, list[str]],
    uncertain: list[str],
) -> None:
    """Find a date column nobody named, by looking at what the values hold."""
    if "date" in matched:
        return
    holders = [header for header in headers if looks_like_dates(sample.get(header, []))]
    if not holders:
        return
    matched["date"] = holders[0]
    if len(holders) > 1:
        uncertain.append(
            f"date: several unnamed columns hold dates ({', '.join(holders)}); "
            f"took the first, confirm it is right"
        )


def detect_mapping(headers: list[str], sample: dict[str, list[str]]) -> MappingDetection:
    """Guess the mapping from headers, confirmed against the values.

    `sample` maps each header to some of its values. A header guess the values
    contradict is dropped: a column called `date` full of merchant names is not
    a date column, and trusting the name would corrupt every row.
    """
    matched: dict[str, str] = {}
    uncertain: list[str] = []
    resolved: list[str] = []

    for canonical, aliases in _ALIASES.items():
        candidates = sorted(
            (aliases.index(_canonical(header)), header)
            for header in headers
            if _canonical(header) in aliases
        )
        if not candidates:
            continue

        chosen = candidates[0][1]
        matched[canonical] = chosen

        if len(candidates) > 1:
            rejected = ", ".join(header for _, header in candidates[1:])
            resolved.append(f"{canonical}: chose {chosen!r} over {rejected} by preference")

    _confirm_against_values(matched, sample, uncertain)
    _fill_missing_date(matched, headers, sample, uncertain)

    problems: list[str] = []
    if "date" not in matched:
        problems.append("no date column found")
    if "description" not in matched and "merchant" not in matched:
        problems.append("no description or merchant column found")
    if not any(field_name in matched for field_name in ("amount", "debit", "credit")):
        problems.append("no amount, debit or credit column found")

    if problems:
        return MappingDetection(
            mapping=None,
            matched=matched,
            uncertain=tuple(uncertain),
            resolved=tuple(resolved),
            problems=tuple(problems),
        )

    # A merchant column can stand in for a missing description and vice versa.
    description = matched.get("description") or matched["merchant"]

    try:
        mapping = ColumnMapping(
            date=matched["date"],
            description=description,
            merchant=matched.get("merchant"),
            amount=matched.get("amount"),
            debit=matched.get("debit"),
            credit=matched.get("credit"),
        )
    except ValidationError as error:
        return MappingDetection(
            mapping=None,
            matched=matched,
            uncertain=tuple(uncertain),
            resolved=tuple(resolved),
            problems=(str(error),),
        )

    return MappingDetection(
        mapping=mapping,
        matched=matched,
        uncertain=tuple(uncertain),
        resolved=tuple(resolved),
    )
