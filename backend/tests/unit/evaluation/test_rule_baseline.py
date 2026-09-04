"""The rule-based baseline.

This is the number the LLM has to beat. Without it, "the model scores 0.91" is
not a result — it might be beating 0.40, or losing to 0.93 at a hundredth of the
cost and latency.

The rules are deliberately simple and deliberately abstain. A rule set that
guesses on every merchant it has never seen would score better on accuracy and
be worse in every way that matters, because the hybrid routes on exactly the
abstentions.

Rules are fitted from the development split only. Fitting them on the holdout
would be the leak the merchant-disjoint split exists to prevent, arriving by a
different route.
"""

from datetime import date
from decimal import Decimal

import pytest

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.evaluation.dataset import LabelledDataset, LabelledTransaction
from offerdelta.evaluation.labels import ABSTAIN
from offerdelta.evaluation.rule_baseline import RuleBaseline, fit_rules

GROCERY = "LIVING_GROCERY"
DINING = "LIVING_DINING"
SUBS = "LIVING_SUBSCRIPTIONS"
INCOME = "INCOME"
TRANSFER = "TRANSFER"


def _record(
    txn_id: str,
    merchant: str,
    label: str = DINING,
    amount: str = "-4.50",
    description: str | None = None,
) -> LabelledTransaction:
    return LabelledTransaction(
        transaction_id=txn_id,
        posted_on=date(2026, 8, 1),
        raw_description=description or merchant,
        normalised_merchant=merchant,
        amount=Money.parse(amount),
        account_type="credit_card",
        source="test",
        bank_format="test_csv",
        primary_label=label,
    )


TRAINING = LabelledDataset(
    dataset_version="dev",
    records=(
        _record("1", "NETFLIX", SUBS, "-15.99"),
        _record("2", "NETFLIX", SUBS, "-15.99"),
        _record("3", "BLUE BOTTLE", DINING),
        _record("4", "BLUE BOTTLE", DINING),
        _record("5", "WHOLE FOODS", GROCERY, "-88.10"),
        _record("6", "PAYROLL DEPOSIT", INCOME, "3000.00"),
        _record("7", "TRANSFER TO SAVINGS", TRANSFER, "-500.00"),
    ),
)


# --- Fitting ---------------------------------------------------------------


def test_a_known_merchant_is_predicted_from_the_rules() -> None:
    rules = fit_rules(TRAINING)
    assert rules.predict(_record("x", "NETFLIX", SUBS, "-15.99").view).label == SUBS


def test_an_unseen_merchant_causes_abstention() -> None:
    # The behaviour that makes the hybrid worthwhile. Guessing here would
    # inflate accuracy and remove the signal the router needs.
    rules = fit_rules(TRAINING)
    assert rules.predict(_record("x", "SOME NEW CAFE").view).label == ABSTAIN


def test_abstention_carries_zero_confidence() -> None:
    rules = fit_rules(TRAINING)
    assert rules.predict(_record("x", "SOME NEW CAFE").view).confidence == Decimal(0)


def test_a_prediction_explains_itself() -> None:
    # A wrong answer with no reason is only wrong; one with a reason is
    # diagnosable.
    rules = fit_rules(TRAINING)
    assert rules.predict(_record("x", "NETFLIX", SUBS, "-15.99").view).reason


def test_the_majority_label_wins_a_conflicted_merchant() -> None:
    # Real data disagrees with itself. Two GROCERY and one DINING for the same
    # merchant should yield GROCERY, not a coin flip.
    conflicted = LabelledDataset(
        dataset_version="dev",
        records=(
            _record("1", "TARGET", GROCERY, "-40.00"),
            _record("2", "TARGET", GROCERY, "-52.00"),
            _record("3", "TARGET", DINING, "-12.00"),
        ),
    )
    assert fit_rules(conflicted).predict(_record("x", "TARGET").view).label == GROCERY


def test_confidence_reflects_how_consistent_the_merchant_was() -> None:
    # A merchant labelled the same way four times is a safer rule than one
    # labelled two ways, and the router needs to see that difference.
    mixed = LabelledDataset(
        dataset_version="dev",
        records=(
            _record("1", "TARGET", GROCERY, "-40.00"),
            _record("2", "TARGET", DINING, "-12.00"),
            _record("3", "COSTCO", GROCERY, "-120.00"),
            _record("4", "COSTCO", GROCERY, "-95.00"),
        ),
    )
    rules = fit_rules(mixed)
    consistent = rules.predict(_record("x", "COSTCO").view).confidence
    conflicted = rules.predict(_record("y", "TARGET").view).confidence
    assert consistent > conflicted


def test_fitting_needs_data() -> None:
    with pytest.raises(ValidationError, match="at least one"):
        fit_rules(LabelledDataset(dataset_version="dev", records=()))


# --- Sign heuristics -------------------------------------------------------


def test_an_unseen_positive_amount_is_guessed_as_income() -> None:
    # Weak but genuinely informative: money arriving is rarely spending. Held
    # at low confidence so the hybrid can override it.
    rules = fit_rules(TRAINING)
    prediction = rules.predict(_record("x", "UNKNOWN EMPLOYER", INCOME, "2500.00").view)
    assert prediction.label == INCOME
    assert prediction.confidence < Decimal("0.6")


def test_an_unseen_transfer_keyword_is_recognised() -> None:
    # TRANSFER is the label whose absence causes double counting, so it earns a
    # keyword rule even in a baseline this simple.
    rules = fit_rules(TRAINING)
    prediction = rules.predict(_record("x", "TRANSFER TO BROKERAGE", TRANSFER, "-1000.00").view)
    assert prediction.label == TRANSFER


def test_a_keyword_rule_does_not_override_a_known_merchant() -> None:
    # Exact merchant knowledge beats a keyword guess.
    training = LabelledDataset(
        dataset_version="dev",
        records=(
            _record("1", "TRANSFERWISE", DINING, "-20.00"),
            _record("2", "TRANSFERWISE", DINING, "-25.00"),
        ),
    )
    assert fit_rules(training).predict(_record("x", "TRANSFERWISE").view).label == DINING


# --- Batch and identity ----------------------------------------------------


def test_predict_many_matches_predict_one_by_one() -> None:
    rules = fit_rules(TRAINING)
    views = [_record("x", "NETFLIX", SUBS, "-15.99").view, _record("y", "NOPE").view]
    assert [p.label for p in rules.predict_many(views)] == [
        rules.predict(views[0]).label,
        rules.predict(views[1]).label,
    ]


def test_the_baseline_names_itself() -> None:
    # The report has three columns and must be able to label them.
    assert fit_rules(TRAINING).name == "rules"


def test_the_baseline_reports_how_many_merchants_it_learned() -> None:
    # Coverage of the rule table is the honest measure of a memorising system.
    assert fit_rules(TRAINING).merchant_count == 5


def test_the_baseline_is_reusable_across_records() -> None:
    rules = fit_rules(TRAINING)
    first = rules.predict(_record("x", "NETFLIX", SUBS, "-15.99").view).label
    rules.predict(_record("y", "WHOLE FOODS", GROCERY, "-88.10").view)
    assert rules.predict(_record("z", "NETFLIX", SUBS, "-15.99").view).label == first


def test_the_type_is_exported() -> None:
    assert RuleBaseline is not None
