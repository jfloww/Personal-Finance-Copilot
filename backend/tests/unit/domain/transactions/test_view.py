"""The projection a categoriser is allowed to see.

Deliberately four fields, nothing more - a categoriser scored on a balance or
a neighbouring row would be scored on information the running system does not
have.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.view import TransactionView


def test_it_carries_exactly_the_four_fields() -> None:
    view = TransactionView(
        normalised_merchant="BLUE BOTTLE",
        raw_description="SQ *BLUE BOTTLE #417",
        amount=Money.parse("-6.75"),
        account_type="checking",
    )
    assert view.normalised_merchant == "BLUE BOTTLE"
    assert view.raw_description == "SQ *BLUE BOTTLE #417"
    assert view.amount == Money.parse("-6.75")
    assert view.account_type == "checking"


def test_it_is_frozen() -> None:
    """A categoriser must not be able to mutate its own input mid-prediction."""
    view = TransactionView(
        normalised_merchant="BLUE BOTTLE",
        raw_description="SQ *BLUE BOTTLE #417",
        amount=Money.parse("-6.75"),
        account_type="checking",
    )
    with pytest.raises(FrozenInstanceError):
        view.normalised_merchant = "SOMEWHERE ELSE"  # type: ignore[misc]


def test_two_views_built_from_the_same_values_are_equal() -> None:
    """Value equality is what lets a hand-built view and one built from a
    stored row stand in for each other."""
    first = TransactionView(
        normalised_merchant="BLUE BOTTLE",
        raw_description="SQ *BLUE BOTTLE #417",
        amount=Money.parse("-6.75"),
        account_type="checking",
    )
    second = TransactionView(
        normalised_merchant="BLUE BOTTLE",
        raw_description="SQ *BLUE BOTTLE #417",
        amount=Money.parse("-6.75"),
        account_type="checking",
    )
    assert first == second


def test_a_different_amount_makes_two_views_unequal() -> None:
    same_merchant = {
        "normalised_merchant": "BLUE BOTTLE",
        "raw_description": "SQ *BLUE BOTTLE #417",
        "account_type": "checking",
    }
    cheaper = TransactionView(amount=Money.parse("-6.75"), **same_merchant)
    pricier = TransactionView(amount=Money.parse("-6.76"), **same_merchant)
    assert cheaper != pricier


def test_it_holds_a_real_money_value() -> None:
    view = TransactionView(
        normalised_merchant="EMPLOYER",
        raw_description="PAYROLL",
        amount=Money(Decimal("100.00")),
        account_type="checking",
    )
    assert view.amount.amount == Decimal("100.00")
    assert view.amount.currency == "USD"
