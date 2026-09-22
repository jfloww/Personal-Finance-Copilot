"""Bounded synthetic CSV fixtures for the public operations showcase.

The browser chooses a bundled case; it never accepts a visitor's financial CSV.
The parser is reusable and tested with unseen rows, so the investigation is
computed from data rather than from case-specific answer literals.
"""

from __future__ import annotations

import csv
from datetime import date, time
from decimal import Decimal, InvalidOperation
from io import StringIO
from pathlib import Path
from typing import Final

from offerdelta.agent.tools.operations import TRANSACTIONS, DemoTransaction
from offerdelta.domain.common.errors import ValidationError

HEADER: Final = ["id", "posted_on", "posted_time", "merchant", "category", "amount", "receipt"]
MAX_CSV_BYTES: Final = 16_384
MAX_TRANSACTIONS: Final = 100
MAX_ID_LENGTH: Final = 64
MAX_TEXT_LENGTH: Final = 80
CONTROL_CODE_LIMIT: Final = 32
MONTH_COUNT: Final = 2
ALTERNATE_SCENARIO: Final = "alternate_billing_review"


def _valid_text(value: str, maximum: int) -> bool:
    return (
        bool(value.strip())
        and len(value) <= maximum
        and all(ord(char) >= CONTROL_CODE_LIMIT for char in value)
    )


def _parse_row(columns: list[str], identifiers: set[str], line_number: int) -> DemoTransaction:
    if len(columns) != len(HEADER):
        raise ValidationError(f"synthetic CSV line {line_number} has the wrong field count")
    identifier, posted_on, posted_time, merchant, category, amount, receipt = columns
    if not _valid_text(identifier, MAX_ID_LENGTH) or identifier in identifiers:
        raise ValidationError(f"synthetic CSV line {line_number} has an invalid ID")
    if not _valid_text(merchant, MAX_TEXT_LENGTH) or not _valid_text(category, MAX_TEXT_LENGTH):
        raise ValidationError(f"synthetic CSV line {line_number} has invalid text")
    day = date.fromisoformat(posted_on)
    clock = time.fromisoformat(posted_time)
    if clock.tzinfo is not None or clock.second or clock.microsecond:
        raise ValidationError(f"synthetic CSV line {line_number} needs a local HH:MM time")
    charge = Decimal(amount)
    if (
        not charge.is_finite()
        or charge <= 0
        or charge > Decimal("1000000.00")
        or charge.quantize(Decimal("0.01")) != charge
    ):
        raise ValidationError(f"synthetic CSV line {line_number} has an invalid amount")
    if receipt not in ("true", "false"):
        raise ValidationError(f"synthetic CSV line {line_number} has an invalid receipt flag")
    identifiers.add(identifier)
    return DemoTransaction(
        id=identifier,
        month=day.strftime("%Y-%m"),
        day=day.day,
        minute=clock.hour * 60 + clock.minute,
        merchant=merchant.strip(),
        category=category.strip(),
        amount=charge,
        receipt=receipt == "true",
    )


def parse_synthetic_csv(raw: str) -> tuple[DemoTransaction, ...]:
    """Parse an in-memory, two-month sample with strict limits and exact money."""
    if len(raw.encode("utf-8")) > MAX_CSV_BYTES:
        raise ValidationError("synthetic CSV exceeds the 16 KiB demo limit")
    try:
        reader = csv.reader(StringIO(raw, newline=""), strict=True)
        if next(reader, None) != HEADER:
            raise ValidationError("synthetic CSV header does not match the demo schema")
        rows: list[DemoTransaction] = []
        identifiers: set[str] = set()
        for line_number, columns in enumerate(reader, 2):
            if len(rows) >= MAX_TRANSACTIONS:
                raise ValidationError("synthetic CSV exceeds 100 transactions")
            rows.append(_parse_row(columns, identifiers, line_number))
    except (csv.Error, ValueError, InvalidOperation) as error:
        raise ValidationError("synthetic CSV contains an invalid date, time, or amount") from error
    if len({row.month for row in rows}) != MONTH_COUNT:
        raise ValidationError("synthetic CSV must contain exactly two months")
    return tuple(rows)


def load_synthetic_case(scenario: str) -> tuple[DemoTransaction, ...]:
    """Load only committed samples, never a user-provided path."""
    if scenario == "august_software_exceptions":
        return TRANSACTIONS
    if scenario == ALTERNATE_SCENARIO:
        sample = Path(__file__).parent / "data" / "alternate_billing_review.csv"
        return parse_synthetic_csv(sample.read_text(encoding="utf-8"))
    raise ValidationError("unknown synthetic investigation scenario")
