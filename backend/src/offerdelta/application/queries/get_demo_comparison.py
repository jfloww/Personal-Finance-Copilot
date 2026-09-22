"""The demo comparison: Auburn against a New Jersey offer.

Assembles everything the engine and solvers can say about one pair of profiles.
Milestone 5 replaces the fixtures with stored profiles; the shape of what comes
back does not change.

Solvers are allowed to fail. An equivalent salary may not exist inside the
search range, and reporting that honestly is better than widening the bounds
until some number appears.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from decimal import Decimal
from typing import Final

from offerdelta.application.queries.demo_profiles import load_demo_profile
from offerdelta.demo.profiles import AUBURN_CURRENT, NEW_JERSEY_CANDIDATE, ComparisonSide
from offerdelta.domain.common.dates import DateRange
from offerdelta.domain.common.derivation import DerivationNode
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.domain.common.percentage import Percentage
from offerdelta.domain.common.periods import PeriodicAmount, PeriodKind
from offerdelta.domain.comparisons.comparison import ComparisonResult, compare
from offerdelta.domain.comparisons.context import CalculationContext
from offerdelta.domain.comparisons.derivation_builder import build_derivation
from offerdelta.domain.comparisons.pre_move import inherit_costs_until_move
from offerdelta.domain.solvers.break_even import BreakEvenResult, solve_break_even
from offerdelta.domain.solvers.equivalent_salary import (
    EquivalentSalaryResult,
    SolverBounds,
    solve_equivalent_salary,
)
from offerdelta.domain.solvers.negotiation_gap import (
    NegotiationGapResult,
    solve_negotiation_gap,
)
from offerdelta.domain.taxes.override_model import NetPayOverrideTaxModel

HORIZON_START: Final = date(2026, 1, 1)
HORIZON_MONTHS: Final = 12
MOVE_DATE: Final = date(2026, 7, 1)

#: Placeholder until a verified paystub replaces it. Stated here rather than
#: buried in a fixture so it is obvious what is being assumed.
ASSUMED_NET_SHARE: Final = Decimal("0.74")
ASSUMED_MARGINAL_RATE: Final = "32"

SEARCH_BOUNDS: Final = SolverBounds(
    lower=Money.parse("40000.00"),
    upper=Money.parse("400000.00"),
    tolerance=Money.parse("1.00"),
)


@dataclass(frozen=True)
class ComparisonView:
    """Everything the API needs to render one comparison."""

    current_label: str
    candidate_label: str
    horizon_months: int

    comparison: ComparisonResult
    current_derivation: DerivationNode
    candidate_derivation: DerivationNode

    break_even: BreakEvenResult
    equivalent_salary: EquivalentSalaryResult | None
    equivalent_salary_error: str | None
    negotiation: NegotiationGapResult | None
    negotiation_error: str | None


def _context(side: ComparisonSide, horizon_months: int) -> CalculationContext:
    """Build a calculation context from a comparison side.

    The tax model is calibrated on the profile's own salary. Milestone 5
    replaces the assumed net share with the stored net-pay override.
    """
    employment = side.employment
    salary = employment.compensation.base_salary
    return CalculationContext(
        employment=employment,
        costs=side.costs,
        household=side.household,
        tax_model=NetPayOverrideTaxModel(
            observed_gross=PeriodicAmount(salary, PeriodKind.ANNUAL),
            observed_net=PeriodicAmount(salary * ASSUMED_NET_SHARE, PeriodKind.ANNUAL),
            marginal_rate=Percentage.from_percent(ASSUMED_MARGINAL_RATE),
        ),
        horizon=DateRange.of_months(HORIZON_START, horizon_months),
    )


def get_demo_comparison(
    *,
    current_key: str = AUBURN_CURRENT,
    candidate_key: str = NEW_JERSEY_CANDIDATE,
    horizon_months: int = HORIZON_MONTHS,
    move_date: date | None = MOVE_DATE,
) -> ComparisonView:
    """Run the full Auburn-to-New-Jersey comparison.

    `move_date` may be None to see the comparison without inherited pre-move
    costs — which is what the engine did before that gap was closed, and worth
    being able to reproduce rather than only describe.
    """
    if horizon_months < 1:
        raise ValidationError(
            f"a comparison horizon must span at least one month, got {horizon_months}"
        )
    if current_key == candidate_key:
        raise ValidationError("current and candidate profiles must be different")

    current = _context(load_demo_profile(current_key), horizon_months)
    candidate = _context(load_demo_profile(candidate_key), horizon_months)

    result = compare(current=current, candidate=candidate, move_date=move_date)

    # The solvers see the same inherited-cost candidate the comparison used, so
    # their answers describe the profile the user is actually looking at.
    solved_candidate = (
        candidate if move_date is None else _with_inherited_costs(current, candidate, move_date)
    )

    equivalent, equivalent_error = _try_equivalent_salary(current, solved_candidate)
    negotiation, negotiation_error = _try_negotiation(current, solved_candidate)

    return ComparisonView(
        current_label=current.employment.label,
        candidate_label=candidate.employment.label,
        horizon_months=horizon_months,
        comparison=result,
        current_derivation=build_derivation(
            result.current, label=f"{current.employment.label} — first-year cash"
        ),
        candidate_derivation=build_derivation(
            result.candidate, label=f"{candidate.employment.label} — first-year cash"
        ),
        break_even=solve_break_even(result.cumulative_cash_delta),
        equivalent_salary=equivalent,
        equivalent_salary_error=equivalent_error,
        negotiation=negotiation,
        negotiation_error=negotiation_error,
    )


def _with_inherited_costs(
    current: CalculationContext, candidate: CalculationContext, move_date: date
) -> CalculationContext:
    return replace(
        candidate,
        costs=inherit_costs_until_move(
            current=current.costs, candidate=candidate.costs, move_date=move_date
        ),
    )


def _try_equivalent_salary(
    current: CalculationContext, candidate: CalculationContext
) -> tuple[EquivalentSalaryResult | None, str | None]:
    try:
        return (
            solve_equivalent_salary(current=current, candidate=candidate, bounds=SEARCH_BOUNDS),
            None,
        )
    except ValidationError as error:
        return None, str(error)


def _try_negotiation(
    current: CalculationContext, candidate: CalculationContext
) -> tuple[NegotiationGapResult | None, str | None]:
    try:
        return (
            solve_negotiation_gap(current=current, candidate=candidate, bounds=SEARCH_BOUNDS),
            None,
        )
    except ValidationError as error:
        return None, str(error)
