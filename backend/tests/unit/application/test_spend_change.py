"""A spend explanation must reconcile while disclosing excluded and provisional rows."""

import uuid
from datetime import date
from decimal import Decimal

import pytest

from offerdelta.application.queries.spend_change import SpendEntry, build_spend_change
from offerdelta.application.reports.monthly import MonthCoverage
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money


def _coverage(year: int, month: int) -> MonthCoverage:
    return MonthCoverage(year, month, False, 0, 0, 0)


def _entry(
    number: int,
    merchant: str,
    amount: str,
    label: str | None,
    *,
    confirmed: bool = True,
    currency: str = "USD",
) -> SpendEntry:
    return SpendEntry(
        uuid.UUID(int=number),
        date(2026, 3, 5),
        merchant,
        Money.parse(amount, currency),
        label,
        confirmed,
    )


def test_refunds_reduce_spend_and_every_driver_has_transaction_evidence() -> None:
    comparison = build_spend_change(
        [
            _entry(1, "STORE", "-10.00", "LIVING_DINING"),
            _entry(2, "STORE", "2.00", "REFUND"),
        ],
        [
            _entry(3, "STORE", "-20.00", "LIVING_DINING"),
            _entry(4, "STORE", "5.00", "REFUND", confirmed=False),
            _entry(5, "OTHER", "-3.00", "LIVING_GROCERY", confirmed=False),
            _entry(6, "TRANSFER", "-100.00", "TRANSFER"),
            _entry(7, "UNKNOWN", "-7.00", None, confirmed=False),
            _entry(8, "BAD INCOME", "-4.00", "INCOME"),
            _entry(9, "PAYROLL", "1000.00", "INCOME"),
        ],
        _coverage(2026, 2),
        _coverage(2026, 3),
    )

    usd = comparison.currencies[0]
    assert usd.previous.gross_spending == Decimal("10.00")
    assert usd.previous.refunds == Decimal("2.00")
    assert usd.previous.net_spending == Decimal("8.00")
    assert usd.current.gross_spending == Decimal("23.00")
    assert usd.current.refunds == Decimal("5.00")
    assert usd.current.net_spending == Decimal("18.00")
    assert usd.current.transfer_debits == Decimal("100.00")
    assert usd.current.unclassified_debits == Decimal("7.00")
    assert usd.current.inconsistent_debits == Decimal("4.00")
    assert usd.current.suggested_rows == 2
    assert usd.current.unclassified_rows == 1
    assert usd.current.inconsistent_rows == 1
    assert usd.delta == Decimal("10.00")
    assert [(row.merchant, row.delta) for row in usd.merchants] == [
        ("STORE", Decimal("7.00")),
        ("OTHER", Decimal("3.00")),
    ]
    assert sum((row.delta for row in usd.merchants), Decimal(0)) == usd.delta
    store = usd.merchants[0]
    assert {item.transaction_id for item in store.previous_evidence} == {
        uuid.UUID(int=1),
        uuid.UUID(int=2),
    }
    assert store.current_evidence[1].contribution == Decimal("-5.00")
    assert store.current_evidence[1].confirmed is False


def test_currencies_never_combine_and_refund_only_month_can_be_negative() -> None:
    comparison = build_spend_change(
        [_entry(1, "STORE", "-2.00", "LIVING_DINING", currency="USD")],
        [
            _entry(2, "STORE", "3.00", "REFUND", currency="USD"),
            _entry(3, "STORE", "-4.00", "LIVING_DINING", currency="EUR"),
        ],
        _coverage(2026, 2),
        _coverage(2026, 3),
    )
    assert [(group.currency, group.delta) for group in comparison.currencies] == [
        ("EUR", Decimal("4.00")),
        ("USD", Decimal("-5.00")),
    ]
    assert comparison.currencies[1].current.net_spending == Decimal("-3.00")


def test_unknown_label_is_refused_instead_of_disappearing() -> None:
    with pytest.raises(ValidationError, match="not a recognised transaction label"):
        build_spend_change(
            [],
            [_entry(1, "STORE", "-2.00", "NOT_A_LABEL")],
            _coverage(2026, 2),
            _coverage(2026, 3),
        )
