"""A bounded read-only tool registry bound to one authenticated tenant.

Construct this registry per request. Never cache it or publish it through the
synthetic demo/MCP server: its closure holds a database session and user scope.
The model can choose a month, but cannot choose a tenant or account.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from typing import Final

from offerdelta.agent.tools.registry import JsonValue, Tool, ToolRegistry, ToolResult
from offerdelta.application.queries.spend_change import (
    CurrencySpendChange,
    SpendChange,
    SpendPeriod,
    explain_spend_change,
)
from offerdelta.application.reports.monthly import MonthCoverage
from offerdelta.application.reports.review import REVIEW_THRESHOLD
from offerdelta.application.scope import TenantScope
from offerdelta.domain.common.errors import ValidationError

_MONTH: Final = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})")
_MAX_DRIVERS: Final = 5
_MAX_EVIDENCE_IDS: Final = 5
_CAVEAT: Final = (
    "Labelled net spending includes provisional suggestions. An incomplete import, "
    "unclassified rows, inconsistent labels/signs, or suggested labels make the "
    "comparison provisional. Transfers are excluded and refunds reduce spending."
)


def build_tenant_spend_registry(scope: TenantScope) -> ToolRegistry:
    """Create one tool over this request's scope; no tenant identifier is accepted."""

    def call(arguments: Mapping[str, object]) -> ToolResult:
        value = arguments.get("month")
        if not isinstance(value, str):
            raise ValidationError("month must be a YYYY-MM string")
        year, month = _parse_month(value)
        comparison = explain_spend_change(scope, year, month, threshold=REVIEW_THRESHOLD)
        return ToolResult.success(_payload(value, comparison))

    return ToolRegistry(
        (
            Tool(
                name="explain_spend_change",
                description=(
                    "Read this authenticated tenant's labelled net-spend change for a month. "
                    "Returns exact amounts, coverage, top merchant drivers, and bounded evidence. "
                    "Provisional results are not a verified spending verdict. Never writes data."
                ),
                input_schema={
                    "type": "object",
                    "properties": {"month": {"type": "string", "minLength": 7}},
                    "required": ["month"],
                    "additionalProperties": False,
                },
                call=call,
            ),
        )
    )


def _parse_month(value: str) -> tuple[int, int]:
    match = _MONTH.fullmatch(value)
    if match is None:
        raise ValidationError(f"{value!r} is not a YYYY-MM month")
    year, month = int(match["year"]), int(match["month"])
    try:
        date(year, month, 1)
    except ValueError as error:
        raise ValidationError(f"{value!r} is not a valid month") from error
    if (year, month) == (1, 1):
        raise ValidationError("0001-01 has no preceding calendar month")
    return year, month


def _payload(requested_month: str, comparison: SpendChange) -> dict[str, JsonValue]:
    provisional = (
        not comparison.previous_coverage.complete
        or not comparison.current_coverage.complete
        or any(
            period.unclassified_rows or period.suggested_rows or period.inconsistent_rows
            for group in comparison.currencies
            for period in (group.previous, group.current)
        )
    )
    currencies: list[JsonValue] = [_currency_payload(group) for group in comparison.currencies]
    return {
        "requested_month": requested_month,
        "previous_month": (
            f"{comparison.previous_coverage.year:04d}-{comparison.previous_coverage.month:02d}"
        ),
        "previous_coverage": _coverage_payload(comparison.previous_coverage),
        "current_coverage": _coverage_payload(comparison.current_coverage),
        "currencies": currencies,
        "provisional": provisional,
        "caveat": _CAVEAT,
        "read_only": True,
    }


def _coverage_payload(coverage: MonthCoverage) -> dict[str, JsonValue]:
    return {
        "complete": coverage.complete,
        "rows": coverage.rows,
        "classified": coverage.classified,
        "awaiting_review": coverage.awaiting_review,
    }


def _period_payload(period: SpendPeriod) -> dict[str, JsonValue]:
    return {
        "gross_spending": str(period.gross_spending),
        "refunds": str(period.refunds),
        "net_spending": str(period.net_spending),
        "unclassified_debits": str(period.unclassified_debits),
        "transfer_debits": str(period.transfer_debits),
        "inconsistent_debits": str(period.inconsistent_debits),
        "unclassified_rows": period.unclassified_rows,
        "suggested_rows": period.suggested_rows,
        "inconsistent_rows": period.inconsistent_rows,
    }


def _currency_payload(group: CurrencySpendChange) -> dict[str, JsonValue]:
    drivers: list[JsonValue] = []
    included_delta = Decimal(0)
    for row in group.merchants[:_MAX_DRIVERS]:
        included_delta += row.delta
        drivers.append(
            {
                "merchant": row.merchant,
                "previous": str(row.previous),
                "current": str(row.current),
                "delta": str(row.delta),
                "previous_transaction_ids": [
                    str(item.transaction_id) for item in row.previous_evidence[:_MAX_EVIDENCE_IDS]
                ],
                "current_transaction_ids": [
                    str(item.transaction_id) for item in row.current_evidence[:_MAX_EVIDENCE_IDS]
                ],
                "previous_evidence_omitted": max(0, len(row.previous_evidence) - _MAX_EVIDENCE_IDS),
                "current_evidence_omitted": max(0, len(row.current_evidence) - _MAX_EVIDENCE_IDS),
            }
        )
    return {
        "currency": group.currency,
        "previous": _period_payload(group.previous),
        "current": _period_payload(group.current),
        "delta": str(group.delta),
        "drivers": drivers,
        "other_delta": str(group.delta - included_delta),
        "omitted_merchants": len(group.merchants) - len(drivers),
    }
