"""Cost categories and single ownership.

The highest-priority correctness fix in the design. Insurance, medical, and
vehicle costs each plausibly belong to two calculators, and any overlap
subtracts the same dollar twice — producing a total that is wrong but entirely
plausible on screen.

Every category is owned by exactly one calculator, and the owners together must
partition this enum: total coverage, no overlap. The check runs when the engine
is assembled, so adding a category without an owner fails immediately rather
than silently dropping a cost.

The rule that resolves the hard cases: **a cost belongs to COMMUTE only if it
would fall to zero when `onsite_days_per_week` is zero.** Fuel and tolls vanish;
auto insurance and registration do not, so they are living costs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from enum import StrEnum

from offerdelta.domain.common.errors import ValidationError


class CalculatorName(StrEnum):
    """The calculators that consume cost items."""

    HOUSING = "HOUSING"
    HEALTH = "HEALTH"
    COMMUTE = "COMMUTE"
    LIVING = "LIVING"
    RELOCATION = "RELOCATION"


class CashFlowType(StrEnum):
    """Which track a cost affects.

    Keeping these apart is what stops the product collapsing into one
    misleading score: cash, wealth, and time are reported separately.
    """

    RECURRING_CASH = "RECURRING_CASH"
    ONE_TIME_CASH = "ONE_TIME_CASH"
    NON_CASH_WEALTH = "NON_CASH_WEALTH"
    TIME = "TIME"


class CostCategory(StrEnum):
    """A closed set. There is deliberately no generic bucket.

    No `INSURANCE` and no `MEDICAL`: a generic category is precisely what lets
    the same dollar be claimed by two calculators. Insurance routes to housing,
    health, or vehicle depending on what it insures; medical routes to health.
    """

    HOUSING_RENT_OR_MORTGAGE = "HOUSING_RENT_OR_MORTGAGE"
    HOUSING_UTILITIES = "HOUSING_UTILITIES"
    HOUSING_INTERNET = "HOUSING_INTERNET"
    HOUSING_RENTERS_INSURANCE = "HOUSING_RENTERS_INSURANCE"
    HOUSING_PARKING_RESIDENTIAL = "HOUSING_PARKING_RESIDENTIAL"

    HEALTH_PREMIUM = "HEALTH_PREMIUM"
    HEALTH_OUT_OF_POCKET = "HEALTH_OUT_OF_POCKET"

    COMMUTE_TRANSIT_FARE = "COMMUTE_TRANSIT_FARE"
    COMMUTE_FUEL = "COMMUTE_FUEL"
    COMMUTE_TOLLS = "COMMUTE_TOLLS"
    COMMUTE_PARKING_WORK = "COMMUTE_PARKING_WORK"
    COMMUTE_VEHICLE_WEAR = "COMMUTE_VEHICLE_WEAR"

    LIVING_GROCERY = "LIVING_GROCERY"
    LIVING_DINING = "LIVING_DINING"
    LIVING_PHONE = "LIVING_PHONE"
    LIVING_VEHICLE_FIXED = "LIVING_VEHICLE_FIXED"
    LIVING_GYM = "LIVING_GYM"
    LIVING_SUBSCRIPTIONS = "LIVING_SUBSCRIPTIONS"
    LIVING_ENTERTAINMENT = "LIVING_ENTERTAINMENT"
    LIVING_TRAVEL = "LIVING_TRAVEL"
    LIVING_OTHER = "LIVING_OTHER"

    #: Appended rather than slotted in beside the other LIVING members:
    #: annotator codes are derived from declaration order, so inserting
    #: earlier would silently redefine a code someone is already using.
    LIVING_CLOTHING = "LIVING_CLOTHING"

    RELOCATION_MOVE = "RELOCATION_MOVE"
    RELOCATION_DEPOSIT = "RELOCATION_DEPOSIT"
    RELOCATION_BROKER_FEE = "RELOCATION_BROKER_FEE"
    RELOCATION_LEASE_BREAK = "RELOCATION_LEASE_BREAK"
    RELOCATION_FURNISHING = "RELOCATION_FURNISHING"


CATEGORY_OWNER: Mapping[CostCategory, CalculatorName] = {
    CostCategory.HOUSING_RENT_OR_MORTGAGE: CalculatorName.HOUSING,
    CostCategory.HOUSING_UTILITIES: CalculatorName.HOUSING,
    CostCategory.HOUSING_INTERNET: CalculatorName.HOUSING,
    CostCategory.HOUSING_RENTERS_INSURANCE: CalculatorName.HOUSING,
    CostCategory.HOUSING_PARKING_RESIDENTIAL: CalculatorName.HOUSING,
    CostCategory.HEALTH_PREMIUM: CalculatorName.HEALTH,
    CostCategory.HEALTH_OUT_OF_POCKET: CalculatorName.HEALTH,
    CostCategory.COMMUTE_TRANSIT_FARE: CalculatorName.COMMUTE,
    CostCategory.COMMUTE_FUEL: CalculatorName.COMMUTE,
    CostCategory.COMMUTE_TOLLS: CalculatorName.COMMUTE,
    CostCategory.COMMUTE_PARKING_WORK: CalculatorName.COMMUTE,
    CostCategory.COMMUTE_VEHICLE_WEAR: CalculatorName.COMMUTE,
    CostCategory.LIVING_GROCERY: CalculatorName.LIVING,
    CostCategory.LIVING_DINING: CalculatorName.LIVING,
    CostCategory.LIVING_PHONE: CalculatorName.LIVING,
    CostCategory.LIVING_VEHICLE_FIXED: CalculatorName.LIVING,
    CostCategory.LIVING_GYM: CalculatorName.LIVING,
    CostCategory.LIVING_SUBSCRIPTIONS: CalculatorName.LIVING,
    CostCategory.LIVING_ENTERTAINMENT: CalculatorName.LIVING,
    CostCategory.LIVING_TRAVEL: CalculatorName.LIVING,
    CostCategory.LIVING_OTHER: CalculatorName.LIVING,
    CostCategory.LIVING_CLOTHING: CalculatorName.LIVING,
    CostCategory.RELOCATION_MOVE: CalculatorName.RELOCATION,
    CostCategory.RELOCATION_DEPOSIT: CalculatorName.RELOCATION,
    CostCategory.RELOCATION_BROKER_FEE: CalculatorName.RELOCATION,
    CostCategory.RELOCATION_LEASE_BREAK: CalculatorName.RELOCATION,
    CostCategory.RELOCATION_FURNISHING: CalculatorName.RELOCATION,
}


def owner_of(category: CostCategory) -> CalculatorName:
    """The single calculator permitted to consume this category."""
    try:
        return CATEGORY_OWNER[category]
    except KeyError:
        raise ValidationError(
            f"{category} has no owning calculator; every category must be claimed by exactly one"
        ) from None


def categories_owned_by(owner: CalculatorName) -> frozenset[CostCategory]:
    return frozenset(category for category, name in CATEGORY_OWNER.items() if name is owner)


def assert_categories_partitioned(claimed: Iterable[frozenset[CostCategory]]) -> None:
    """Verify a set of calculators covers every category exactly once.

    Called when the engine is assembled. A gap silently drops a cost from the
    total; an overlap subtracts it twice. Both are invisible in the output,
    which is why this is checked rather than trusted.
    """
    seen: set[CostCategory] = set()
    duplicated: set[CostCategory] = set()

    for group in claimed:
        duplicated |= seen & group
        seen |= group

    if duplicated:
        names = sorted(category.value for category in duplicated)
        raise ValidationError(f"categories claimed by more than one calculator: {names}")

    missing = set(CostCategory) - seen
    if missing:
        names = sorted(category.value for category in missing)
        raise ValidationError(f"categories not claimed by any calculator: {names}")
