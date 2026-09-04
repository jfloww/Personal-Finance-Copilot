"""The refactor must not change what the model receives.

Published V2 figures describe requests built a particular way. This asserts
the tool payload for a given record is exactly what it was before the input
type narrowed - so the numbers keep describing the code that produced them.
"""

from __future__ import annotations

from datetime import date

from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.view import TransactionView
from offerdelta.evaluation.dataset import LabelledTransaction

RECORD = LabelledTransaction(
    transaction_id="chase-checking-5718:abc:1",
    posted_on=date(2026, 3, 1),
    raw_description="SQ *BLUE BOTTLE #417",
    normalised_merchant="SQ BLUE BOTTLE",
    amount=Money.parse("-6.75"),
    account_type="checking",
    source="chase",
    bank_format="chase-checking-v1",
    primary_label="LIVING_DINING",
)

#: The four values the categorisers actually read, pinned literally.
EXPECTED = {
    "merchant": "SQ BLUE BOTTLE",
    "raw_description": "SQ *BLUE BOTTLE #417",
    "amount": "-6.75",
    "account_type": "checking",
}


def test_the_view_carries_exactly_the_four_fields() -> None:
    view = RECORD.view
    assert view.normalised_merchant == EXPECTED["merchant"]
    assert view.raw_description == EXPECTED["raw_description"]
    assert str(view.amount.amount) == EXPECTED["amount"]
    assert view.account_type == EXPECTED["account_type"]


def test_a_view_built_by_hand_is_indistinguishable_from_one_off_a_record() -> None:
    """Production builds views from stored rows; evaluation builds them from
    labelled records. A categoriser must not be able to tell which it got."""
    by_hand = TransactionView(
        normalised_merchant="SQ BLUE BOTTLE",
        raw_description="SQ *BLUE BOTTLE #417",
        amount=Money.parse("-6.75"),
        account_type="checking",
    )
    assert by_hand == RECORD.view
