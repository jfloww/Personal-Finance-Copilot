"""Turning calculated impacts into a derivation tree.

The demo feature: every figure expands into the inputs, formula, and provenance
that produced it.

The tree is built on a first-year basis so every branch shares one period.
Monthly impacts are annualised; one-time impacts stand as they are, because an
event does not recur. Mixing periods would make the parent-equals-children
invariant meaningless, and that invariant is the only thing guaranteeing the
explanation matches the number it explains.

Wealth-only impacts are excluded from a cash tree. The employer match is real
money, but it is not cash, and including it would make the tree disagree with
the total it claims to explain.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Final

from offerdelta.domain.common.derivation import DerivationNode, _weakest
from offerdelta.domain.common.evidence import Evidence
from offerdelta.domain.common.money import Money
from offerdelta.domain.common.periods import PeriodKind
from offerdelta.domain.comparisons.engine import CalculationResult
from offerdelta.domain.comparisons.impacts import CostImpact
from offerdelta.domain.costs.categories import CalculatorName

_MONTHS_PER_YEAR: Final = 12

_BRANCH_LABELS: Final[dict[CalculatorName, str]] = {
    CalculatorName.HOUSING: "Housing",
    CalculatorName.HEALTH: "Health and benefits",
    CalculatorName.COMMUTE: "Commute",
    CalculatorName.LIVING: "Income and living",
    CalculatorName.RELOCATION: "Relocation",
}


def _first_year_amount(impact: CostImpact) -> Money:
    """The impact's contribution to one year of cash."""
    if impact.period is PeriodKind.MONTHLY:
        return impact.cash_amount * _MONTHS_PER_YEAR
    return impact.cash_amount


def build_derivation(result: CalculationResult, *, label: str) -> DerivationNode:
    """Assemble a first-year cash derivation from a calculated result."""
    grouped: dict[CalculatorName, list[CostImpact]] = defaultdict(list)
    for impact in result.impacts:
        if impact.cash_amount.is_zero():
            continue  # wealth and time belong to their own trees
        if impact.period is PeriodKind.HORIZON_CUMULATIVE:
            continue
        grouped[impact.produced_by].append(impact)

    branches: list[DerivationNode] = []
    for name in CalculatorName:
        impacts = grouped.get(name)
        if not impacts:
            continue

        leaves = tuple(_leaf(impact) for impact in impacts)
        subtotal = Money.zero()
        for leaf in leaves:
            subtotal = subtotal + leaf.amount

        branches.append(
            DerivationNode(
                code=name.value.lower(),
                label=_BRANCH_LABELS[name],
                amount=subtotal,
                period=PeriodKind.ANNUAL,
                formula=f"sum of {len(leaves)} {name.value.lower()} components",
                evidence=_weakest([leaf.evidence for leaf in leaves]),
                children=leaves,
            )
        )

    total = Money.zero()
    for branch in branches:
        total = total + branch.amount

    return DerivationNode(
        code="first_year_disposable_cash",
        label=label,
        amount=total,
        period=PeriodKind.ANNUAL,
        formula="income - housing - health - commute - living - relocation",
        evidence=Evidence.DERIVED,
        children=tuple(branches),
    )


def _leaf(impact: CostImpact) -> DerivationNode:
    detail = ", ".join(f"{ref.label}={ref.value}" for ref in impact.inputs)
    formula = impact.formula_id if not detail else f"{impact.formula_id} ({detail})"
    return DerivationNode(
        code=impact.code,
        label=impact.label,
        amount=_first_year_amount(impact),
        period=PeriodKind.ANNUAL,
        formula=formula,
        evidence=impact.evidence,
        children=(),
    )
