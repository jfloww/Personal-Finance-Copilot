"""The walking skeleton's calculation.

Deliberately the smallest real calculation that produces a derivation tree:
monthly net pay less housing, commute, and living costs. It exists so the
deployed skeleton renders a figure that came from domain code rather than a
literal, which is what makes the deployment worth proving.

Two things replace this shortly:

- Milestone 2 replaces the loose arguments with cost items carrying their own
  category, owning calculator, period, and evidence.
- Milestone 3 replaces the function with the composed calculator engine and the
  full monthly cash-flow reconciliation invariant.

Net pay is taken as given here, which mirrors the phase-1 net-pay override: the
tax engine does not arrive until phase 2.
"""

from __future__ import annotations

from offerdelta.domain.common.derivation import DerivationNode
from offerdelta.domain.common.evidence import Evidence
from offerdelta.domain.common.money import Money
from offerdelta.domain.common.periods import PeriodKind


def _cost(code: str, label: str, amount: Money) -> DerivationNode:
    """A cost, recorded as a negative amount so the tree is one addition."""
    return DerivationNode(
        code=code,
        label=label,
        amount=-amount,
        period=PeriodKind.MONTHLY,
        formula=f"{code} as entered",
        evidence=Evidence.ASSUMED,
        children=(),
    )


def monthly_disposable_cash(
    *,
    net_pay: Money,
    rent: Money,
    utilities: Money,
    commute: Money,
    living: Money,
) -> DerivationNode:
    """Compute monthly disposable cash and explain how.

    Costs are supplied as positive amounts and recorded as negative nodes, so
    every parent in the returned tree is the plain sum of its children.
    """
    housing = DerivationNode(
        code="housing",
        label="Housing",
        amount=-(rent + utilities),
        period=PeriodKind.MONTHLY,
        formula="rent + utilities",
        evidence=Evidence.DERIVED,
        children=(
            _cost("rent", "Rent", rent),
            _cost("utilities", "Utilities", utilities),
        ),
    )

    take_home = DerivationNode(
        code="net_pay",
        label="Monthly take-home pay",
        amount=net_pay,
        period=PeriodKind.MONTHLY,
        formula="verified net pay from a paystub",
        evidence=Evidence.USER_CONFIRMED,
        children=(),
    )

    children = (
        take_home,
        housing,
        _cost("commute", "Commute", commute),
        _cost("living", "Recurring living costs", living),
    )

    total = Money.zero(net_pay.currency)
    for child in children:
        total = total + child.amount

    return DerivationNode(
        code="monthly_disposable_cash",
        label="Monthly disposable cash",
        amount=total,
        period=PeriodKind.MONTHLY,
        formula="net_pay - housing - commute - living",
        evidence=Evidence.DERIVED,
        children=children,
    )
