"""The two reference profiles: Auburn current job, New Jersey candidate offer.

These drive the golden tests, the demo, and eventually the end-to-end flow.

**Every figure here is currently a placeholder marked ASSUMED.** None of it is
real data. Blueprint section 23 requires assumed values to stay visually
distinct from confirmed ones precisely so a demo built on guesses can never be
mistaken for a real comparison. Swapping in real numbers changes these
constants and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from offerdelta.domain.common.evidence import Evidence
from offerdelta.domain.common.location import Location
from offerdelta.domain.common.money import Money
from offerdelta.domain.common.percentage import Percentage
from offerdelta.domain.common.periods import PayFrequency, PeriodicAmount, PeriodKind
from offerdelta.domain.costs.categories import CashFlowType, CostCategory
from offerdelta.domain.costs.household import HouseholdProfile
from offerdelta.domain.costs.items import CostItem, CostProfile
from offerdelta.domain.employment.benefits import Benefits, RetirementMatch
from offerdelta.domain.employment.compensation import (
    Compensation,
    EquityGrant,
    SigningBonus,
    TargetBonus,
)
from offerdelta.domain.employment.overrides import NetPayOverride
from offerdelta.domain.employment.profile import EmploymentProfile
from offerdelta.domain.employment.value_objects import FilingStatus
from offerdelta.domain.employment.work_schedule import WorkSchedule

TAX_YEAR = 2026
START = date(2026, 1, 1)
MOVE_DATE = date(2026, 7, 1)


@dataclass(frozen=True)
class ComparisonSide:
    """One side of a comparison: the job, its costs, and who shares them."""

    employment: EmploymentProfile
    costs: CostProfile
    household: HouseholdProfile


def _recurring(
    category: CostCategory,
    monthly: str,
    evidence: Evidence = Evidence.ASSUMED,
    effective: date = START,
) -> CostItem:
    return CostItem(
        category=category,
        amount=PeriodicAmount(Money.parse(monthly), PeriodKind.MONTHLY),
        cash_flow_type=CashFlowType.RECURRING_CASH,
        effective_date=effective,
        evidence=evidence,
    )


def _one_time(category: CostCategory, amount: str, effective: date = MOVE_DATE) -> CostItem:
    return CostItem(
        category=category,
        amount=PeriodicAmount(Money.parse(amount), PeriodKind.ONE_TIME),
        cash_flow_type=CashFlowType.ONE_TIME_CASH,
        effective_date=effective,
        evidence=Evidence.ASSUMED,
    )


def auburn_current() -> ComparisonSide:
    """The current job in Auburn, Alabama.

    Single jurisdiction: lives and works in the same state, so the hard
    multi-state rules do not apply here — they apply to the candidate.
    """
    employment = EmploymentProfile(
        label="Current - Auburn, AL",
        work_location=Location(state="AL", locality="Auburn"),
        residence=Location(state="AL", locality="Auburn"),
        compensation=Compensation(
            base_salary=Money.parse("78000.00"),
            pay_frequency=PayFrequency.BIWEEKLY,
            expected_annual_raise=Percentage.from_percent("3"),
        ),
        benefits=Benefits(
            employee_health_premium=PeriodicAmount(Money.parse("165.00"), PeriodKind.MONTHLY),
            employer_hsa_contribution=PeriodicAmount(Money.parse("750.00"), PeriodKind.ANNUAL),
            employee_hsa_fsa_contribution=PeriodicAmount(Money.parse("1440.00"), PeriodKind.ANNUAL),
            retirement_match=RetirementMatch(
                match_rate=Percentage.from_percent("100"),
                match_limit_rate=Percentage.from_percent("4"),
                vesting_months=24,
            ),
            employee_contribution_rate=Percentage.from_percent("6"),
            pto_days=15,
            paid_holidays=10,
        ),
        schedule=WorkSchedule(
            weekly_work_hours=Decimal("40"),
            annual_working_weeks=Decimal("48"),
            onsite_days_per_week=Decimal("3"),
            one_way_commute_minutes=Decimal("15"),
        ),
        tax_year=TAX_YEAR,
        filing_status=FilingStatus.SINGLE,
    )

    # A verified paystub figure would carry Evidence.USER_CONFIRMED; this one is
    # a placeholder until a real one replaces it.
    override = NetPayOverride(
        observed_net_pay=PeriodicAmount(Money.parse("4820.00"), PeriodKind.MONTHLY),
        basis=employment.override_basis(),
        captured_at=date(2026, 8, 1),
    )
    employment = EmploymentProfile(
        label=employment.label,
        work_location=employment.work_location,
        residence=employment.residence,
        compensation=employment.compensation,
        benefits=employment.benefits,
        schedule=employment.schedule,
        tax_year=employment.tax_year,
        filing_status=employment.filing_status,
        net_pay_override=override,
    )

    costs = CostProfile(
        items=(
            _recurring(CostCategory.HOUSING_RENT_OR_MORTGAGE, "1150.00"),
            _recurring(CostCategory.HOUSING_UTILITIES, "145.00"),
            _recurring(CostCategory.HOUSING_INTERNET, "60.00"),
            _recurring(CostCategory.LIVING_INSURANCE, "18.00"),
            _recurring(CostCategory.HEALTH_OUT_OF_POCKET, "70.00"),
            _recurring(CostCategory.COMMUTE_FUEL, "95.00"),
            _recurring(CostCategory.COMMUTE_VEHICLE_WEAR, "45.00"),
            _recurring(CostCategory.LIVING_GROCERY, "420.00"),
            _recurring(CostCategory.LIVING_DINING, "180.00"),
            _recurring(CostCategory.LIVING_PHONE, "55.00"),
            _recurring(CostCategory.LIVING_VEHICLE_FIXED, "125.00"),
            _recurring(CostCategory.LIVING_GYM, "35.00"),
            _recurring(CostCategory.LIVING_SUBSCRIPTIONS, "45.00"),
            _recurring(CostCategory.LIVING_ENTERTAINMENT, "90.00"),
            _recurring(CostCategory.LIVING_TRAVEL, "150.00"),
            _recurring(CostCategory.LIVING_OTHER, "80.00"),
        )
    )

    return ComparisonSide(
        employment=employment,
        costs=costs,
        household=HouseholdProfile.solo(),
    )


def new_jersey_candidate() -> ComparisonSide:
    """A candidate offer in New Jersey, commuting into New York City.

    The interesting case. Residence and work fall in different states, and there
    is no NY-NJ reciprocal agreement, so this profile cannot be priced by one
    state's rules alone. NYC personal income tax does not apply — that reaches
    NYC residents only, not commuters.
    """
    employment = EmploymentProfile(
        label="Candidate - Jersey City, NJ (works NYC)",
        work_location=Location(state="NY", locality="New York City"),
        residence=Location(state="NJ", locality="Jersey City"),
        compensation=Compensation(
            base_salary=Money.parse("135000.00"),
            pay_frequency=PayFrequency.BIWEEKLY,
            signing_bonus=SigningBonus(amount=Money.parse("15000.00"), repayment_months=12),
            target_bonus=TargetBonus(
                rate=Percentage.from_percent("12"),
                probability=Percentage.from_percent("70"),
            ),
            equity=EquityGrant(
                total_value=Money.parse("96000.00"),
                vesting_months=48,
                cliff_months=12,
                estimated_withholding_rate=Percentage.from_percent("30"),
            ),
            relocation_reimbursement=Money.parse("8000.00"),
            expected_annual_raise=Percentage.from_percent("4"),
        ),
        benefits=Benefits(
            employee_health_premium=PeriodicAmount(Money.parse("210.00"), PeriodKind.MONTHLY),
            employer_hsa_contribution=PeriodicAmount(Money.parse("1000.00"), PeriodKind.ANNUAL),
            employee_hsa_fsa_contribution=PeriodicAmount(Money.parse("1440.00"), PeriodKind.ANNUAL),
            retirement_match=RetirementMatch(
                match_rate=Percentage.from_percent("50"),
                match_limit_rate=Percentage.from_percent("6"),
                vesting_months=36,
            ),
            employee_contribution_rate=Percentage.from_percent("6"),
            pto_days=20,
            paid_holidays=11,
        ),
        schedule=WorkSchedule(
            weekly_work_hours=Decimal("45"),
            annual_working_weeks=Decimal("48"),
            onsite_days_per_week=Decimal("3"),
            one_way_commute_minutes=Decimal("55"),
        ),
        tax_year=TAX_YEAR,
        filing_status=FilingStatus.SINGLE,
    )

    costs = CostProfile(
        items=(
            _recurring(CostCategory.HOUSING_RENT_OR_MORTGAGE, "2850.00", effective=MOVE_DATE),
            _recurring(CostCategory.HOUSING_UTILITIES, "165.00", effective=MOVE_DATE),
            _recurring(CostCategory.HOUSING_INTERNET, "70.00", effective=MOVE_DATE),
            _recurring(CostCategory.LIVING_INSURANCE, "22.00", effective=MOVE_DATE),
            _recurring(CostCategory.HEALTH_OUT_OF_POCKET, "85.00", effective=MOVE_DATE),
            _recurring(CostCategory.COMMUTE_TRANSIT_FARE, "182.00", effective=MOVE_DATE),
            _recurring(CostCategory.LIVING_GROCERY, "520.00", effective=MOVE_DATE),
            _recurring(CostCategory.LIVING_DINING, "320.00", effective=MOVE_DATE),
            _recurring(CostCategory.LIVING_PHONE, "55.00", effective=MOVE_DATE),
            _recurring(CostCategory.LIVING_VEHICLE_FIXED, "0.00", effective=MOVE_DATE),
            _recurring(CostCategory.LIVING_GYM, "85.00", effective=MOVE_DATE),
            _recurring(CostCategory.LIVING_SUBSCRIPTIONS, "45.00", effective=MOVE_DATE),
            _recurring(CostCategory.LIVING_ENTERTAINMENT, "180.00", effective=MOVE_DATE),
            _recurring(CostCategory.LIVING_TRAVEL, "200.00", effective=MOVE_DATE),
            _recurring(CostCategory.LIVING_OTHER, "110.00", effective=MOVE_DATE),
            _one_time(CostCategory.RELOCATION_MOVE, "3200.00"),
            _one_time(CostCategory.RELOCATION_DEPOSIT, "5700.00"),
            _one_time(CostCategory.RELOCATION_BROKER_FEE, "4275.00"),
            _one_time(CostCategory.RELOCATION_FURNISHING, "2500.00"),
        )
    )

    return ComparisonSide(
        employment=employment,
        costs=costs,
        household=HouseholdProfile.solo(),
    )
