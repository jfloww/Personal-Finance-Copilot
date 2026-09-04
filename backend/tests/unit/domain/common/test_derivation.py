"""Derivation nodes.

The derivation tree is the demo: every figure the product shows can be expanded
to reveal the inputs, formula, and provenance that produced it.

A node with children must equal the sum of those children. Child amounts are
signed, so income is positive and costs are negative and the whole thing is one
addition. That invariant is the seed of the monthly cash-flow reconciliation
check the engine will enforce in milestone 3.
"""

import pytest

from offerdelta.domain.common.derivation import DerivationNode, _weakest
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.evidence import Evidence
from offerdelta.domain.common.money import Money
from offerdelta.domain.common.periods import PeriodKind


def _leaf(code: str, amount: str) -> DerivationNode:
    return DerivationNode(
        code=code,
        label=code.replace("_", " ").title(),
        amount=Money.parse(amount),
        period=PeriodKind.MONTHLY,
        formula="given",
        evidence=Evidence.USER_CONFIRMED,
    )


def test_a_leaf_carries_its_amount() -> None:
    assert _leaf("rent", "-1800.00").amount == Money.parse("-1800.00")


def test_a_leaf_has_no_children() -> None:
    assert _leaf("rent", "-1800.00").children == ()


def test_a_parent_equals_the_sum_of_its_children() -> None:
    total = DerivationNode(
        code="disposable_cash",
        label="Monthly disposable cash",
        amount=Money.parse("1200.00"),
        period=PeriodKind.MONTHLY,
        formula="net_pay - rent",
        evidence=Evidence.ASSUMED,
        children=(_leaf("net_pay", "3000.00"), _leaf("rent", "-1800.00")),
    )
    assert total.amount == Money.parse("1200.00")


def test_a_parent_that_disagrees_with_its_children_is_rejected() -> None:
    # This is the whole point: a derivation that does not add up is a bug in the
    # calculation, and it should be impossible to render one.
    with pytest.raises(ValidationError, match="does not equal the sum"):
        DerivationNode(
            code="disposable_cash",
            label="Monthly disposable cash",
            amount=Money.parse("9999.00"),
            period=PeriodKind.MONTHLY,
            formula="net_pay - rent",
            evidence=Evidence.ASSUMED,
            children=(_leaf("net_pay", "3000.00"), _leaf("rent", "-1800.00")),
        )


def test_children_must_share_the_parents_period() -> None:
    annual_child = DerivationNode(
        code="bonus",
        label="Bonus",
        amount=Money.parse("1000.00"),
        period=PeriodKind.ANNUAL,
        formula="given",
        evidence=Evidence.ASSUMED,
    )
    with pytest.raises(ValidationError, match="period"):
        DerivationNode(
            code="total",
            label="Total",
            amount=Money.parse("1000.00"),
            period=PeriodKind.MONTHLY,
            formula="sum",
            evidence=Evidence.ASSUMED,
            children=(annual_child,),
        )


def test_nodes_nest_to_any_depth() -> None:
    housing = DerivationNode(
        code="housing",
        label="Housing",
        amount=Money.parse("-1950.00"),
        period=PeriodKind.MONTHLY,
        formula="rent + utilities",
        evidence=Evidence.ASSUMED,
        children=(_leaf("rent", "-1800.00"), _leaf("utilities", "-150.00")),
    )
    root = DerivationNode(
        code="disposable_cash",
        label="Monthly disposable cash",
        amount=Money.parse("1050.00"),
        period=PeriodKind.MONTHLY,
        formula="net_pay + housing",
        evidence=Evidence.ASSUMED,
        children=(_leaf("net_pay", "3000.00"), housing),
    )
    assert root.children[1].children[0].code == "rent"


def test_walking_yields_every_node_once() -> None:
    root = DerivationNode(
        code="total",
        label="Total",
        amount=Money.parse("1200.00"),
        period=PeriodKind.MONTHLY,
        formula="a + b",
        evidence=Evidence.ASSUMED,
        children=(_leaf("net_pay", "3000.00"), _leaf("rent", "-1800.00")),
    )
    assert [node.code for node in root.walk()] == ["total", "net_pay", "rent"]


def test_is_immutable() -> None:
    node = _leaf("rent", "-1800.00")
    with pytest.raises(AttributeError):
        node.amount = Money.zero()  # type: ignore[misc]


def test_weakest_of_no_children_is_assumed_not_sourced() -> None:
    """A childless branch has no child evidence to be weakest of.

    `SOURCED` - "taken from a versioned public dataset" - is the strongest
    claim `Evidence` can make, not a safe default for "nothing to go on"; a
    fully-reviewed month's empty branches, and an empty month's whole root,
    would otherwise claim data provenance they do not have - the property
    Fix 7 exists for.
    """
    assert _weakest([]) is Evidence.ASSUMED


def test_weakest_of_only_sourced_children_is_still_sourced() -> None:
    """The empty-list default must not swallow a real, non-empty case."""
    assert _weakest([Evidence.SOURCED, Evidence.SOURCED]) is Evidence.SOURCED
