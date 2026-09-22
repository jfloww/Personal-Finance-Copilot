"""Exact arithmetic and honest grouping for observed debit comparisons."""

from decimal import Decimal

from offerdelta.application.queries.observed_debits import (
    DebitEntry,
    build_observed_debit_comparison,
)
from offerdelta.application.reports.monthly import MonthCoverage
from offerdelta.domain.common.money import Money


def _coverage(year: int, month: int) -> MonthCoverage:
    return MonthCoverage(year, month, False, 0, 0, 0)


def _entry(merchant: str, amount: str, currency: str = "USD") -> DebitEntry:
    return DebitEntry(merchant, Money.parse(amount, currency))


def test_debits_are_exact_and_inflows_are_not_subtracted_from_them() -> None:
    comparison = build_observed_debit_comparison(
        [_entry("STORE", "-0.10"), _entry("STORE", "-0.20"), _entry("PAYROLL", "100")],
        [_entry("STORE", "-0.45"), _entry("OTHER", "-1.00"), _entry("REFUND", "0.05")],
        _coverage(2026, 2),
        _coverage(2026, 3),
    )

    usd = comparison.currencies[0]
    assert usd.currency == "USD"
    assert usd.previous == Decimal("0.30")
    assert usd.current == Decimal("1.45")
    assert usd.delta == Decimal("1.15")
    assert [(row.merchant, row.delta) for row in usd.merchants] == [
        ("OTHER", Decimal("1.00")),
        ("STORE", Decimal("0.15")),
    ]
    assert sum((row.delta for row in usd.merchants), Decimal(0)) == usd.delta


def test_currencies_are_kept_separate_even_when_merchant_names_match() -> None:
    comparison = build_observed_debit_comparison(
        [_entry("STORE", "-2.00", "USD"), _entry("STORE", "-3.00", "EUR")],
        [_entry("STORE", "-5.00", "USD"), _entry("STORE", "-1.00", "EUR")],
        _coverage(2026, 2),
        _coverage(2026, 3),
    )

    assert [(group.currency, group.delta) for group in comparison.currencies] == [
        ("EUR", Decimal("-2.00")),
        ("USD", Decimal("3.00")),
    ]


def test_no_debits_has_no_currency_groups() -> None:
    comparison = build_observed_debit_comparison(
        [_entry("PAYROLL", "100")],
        [],
        _coverage(2026, 2),
        _coverage(2026, 3),
    )
    assert comparison.currencies == ()
