"""Read-only month-over-month comparison of a tenant's observed bank debits.

Negative bank amounts are debits, not necessarily consumption: transfers and
uncategorised rows can be present. No policy decision or duplicate-charge
claim is made from this evidence alone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal

from offerdelta.application.reports.monthly import MonthCoverage, load_month
from offerdelta.application.scope import TenantScope
from offerdelta.domain.common.money import Money


@dataclass(frozen=True)
class DebitEntry:
    merchant: str
    amount: Money


@dataclass(frozen=True)
class MerchantDebitDelta:
    merchant: str
    previous: Decimal
    current: Decimal
    delta: Decimal


@dataclass(frozen=True)
class CurrencyDebitComparison:
    currency: str
    previous: Decimal
    current: Decimal
    delta: Decimal
    merchants: tuple[MerchantDebitDelta, ...]


@dataclass(frozen=True)
class ObservedDebitComparison:
    previous_coverage: MonthCoverage
    current_coverage: MonthCoverage
    currencies: tuple[CurrencyDebitComparison, ...]


def compare_observed_debits(
    scope: TenantScope, year: int, month: int, *, threshold: Decimal
) -> ObservedDebitComparison:
    """Compare the requested month to the immediately preceding calendar month."""
    current_start = date(year, month, 1)
    previous_day = current_start - timedelta(days=1)
    previous = load_month(scope, previous_day.year, previous_day.month, threshold=threshold)
    current = load_month(scope, year, month, threshold=threshold)
    return build_observed_debit_comparison(
        [DebitEntry(row.normalised_merchant, row.amount) for row in previous.rows],
        [DebitEntry(row.normalised_merchant, row.amount) for row in current.rows],
        previous.coverage,
        current.coverage,
    )


def build_observed_debit_comparison(
    previous_entries: Sequence[DebitEntry],
    current_entries: Sequence[DebitEntry],
    previous_coverage: MonthCoverage,
    current_coverage: MonthCoverage,
) -> ObservedDebitComparison:
    """Group exact debit magnitudes by currency and merchant; never mix currencies."""
    previous = _debit_totals(previous_entries)
    current = _debit_totals(current_entries)
    currencies: list[CurrencyDebitComparison] = []
    for currency in sorted(set(previous) | set(current)):
        before = previous.get(currency, {})
        after = current.get(currency, {})
        merchants = [
            MerchantDebitDelta(
                merchant=name,
                previous=before.get(name, Decimal(0)),
                current=after.get(name, Decimal(0)),
                delta=after.get(name, Decimal(0)) - before.get(name, Decimal(0)),
            )
            for name in set(before) | set(after)
        ]
        merchants.sort(key=lambda row: (-abs(row.delta), row.merchant))
        previous_total = sum(before.values(), Decimal(0))
        current_total = sum(after.values(), Decimal(0))
        currencies.append(
            CurrencyDebitComparison(
                currency=currency,
                previous=previous_total,
                current=current_total,
                delta=current_total - previous_total,
                merchants=tuple(merchants),
            )
        )
    return ObservedDebitComparison(previous_coverage, current_coverage, tuple(currencies))


def _debit_totals(entries: Sequence[DebitEntry]) -> dict[str, dict[str, Decimal]]:
    totals: dict[str, dict[str, Decimal]] = defaultdict(lambda: defaultdict(Decimal))
    for entry in entries:
        if entry.amount.amount < 0:
            totals[entry.amount.currency][entry.merchant] += -entry.amount.amount
    return totals
