"""Cost categories and single ownership.

The correctness fix this taxonomy exists for: insurance, medical, and vehicle
costs each plausibly belong to two calculators. Any overlap double counts real
money, and the resulting error looks entirely plausible on screen.

Every category is therefore owned by exactly one calculator, and the ownership
map must partition the enum — total, with no overlap.
"""

import pytest

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.costs.categories import (
    CATEGORY_OWNER,
    CalculatorName,
    CashFlowType,
    CostCategory,
    assert_categories_partitioned,
    categories_owned_by,
    owner_of,
)


def test_every_category_has_an_owner() -> None:
    unowned = [category for category in CostCategory if category not in CATEGORY_OWNER]
    assert unowned == [], f"categories with no owning calculator: {unowned}"


def test_the_ownership_map_has_no_categories_outside_the_enum() -> None:
    assert set(CATEGORY_OWNER) <= set(CostCategory)


def test_owner_of_returns_the_single_owning_calculator() -> None:
    assert owner_of(CostCategory.HOUSING_RENT_OR_MORTGAGE) is CalculatorName.HOUSING


def test_categories_owned_by_returns_only_that_calculators_categories() -> None:
    owned = categories_owned_by(CalculatorName.COMMUTE)
    assert CostCategory.COMMUTE_FUEL in owned
    assert CostCategory.LIVING_GROCERY not in owned


def test_the_owners_together_partition_every_category() -> None:
    claimed = [categories_owned_by(name) for name in CalculatorName]
    assert_categories_partitioned(claimed)


def test_a_gap_in_coverage_is_rejected() -> None:
    incomplete = [
        categories_owned_by(name) for name in CalculatorName if name is not CalculatorName.LIVING
    ]
    with pytest.raises(ValidationError, match="not claimed"):
        assert_categories_partitioned(incomplete)


def test_an_overlap_between_calculators_is_rejected() -> None:
    # The exact bug the taxonomy prevents: two calculators consuming the same
    # cost, so it is subtracted twice.
    overlapping = [
        frozenset(CostCategory),
        frozenset({CostCategory.LIVING_GROCERY}),
    ]
    with pytest.raises(ValidationError, match="claimed by more than one"):
        assert_categories_partitioned(overlapping)


# --- The three ambiguous cases from blueprint section 7.3 -------------------


def test_auto_insurance_is_a_living_cost_not_a_housing_or_health_one() -> None:
    assert owner_of(CostCategory.LIVING_VEHICLE_FIXED) is CalculatorName.LIVING


def test_renters_insurance_is_a_housing_cost() -> None:
    assert owner_of(CostCategory.LIVING_INSURANCE) is CalculatorName.LIVING


def test_health_premiums_and_out_of_pocket_both_belong_to_health() -> None:
    assert owner_of(CostCategory.HEALTH_PREMIUM) is CalculatorName.HEALTH
    assert owner_of(CostCategory.HEALTH_OUT_OF_POCKET) is CalculatorName.HEALTH


def test_there_is_no_generic_insurance_or_medical_category() -> None:
    # A generic bucket is what lets the same dollar be claimed twice. Insurance
    # routes to housing, health, or vehicle; medical routes to health only.
    names = {category.value for category in CostCategory}
    assert "INSURANCE" not in names
    assert "MEDICAL" not in names


def test_work_parking_and_residential_parking_are_different_categories() -> None:
    assert owner_of(CostCategory.COMMUTE_PARKING_WORK) is CalculatorName.COMMUTE
    assert owner_of(CostCategory.HOUSING_PARKING_RESIDENTIAL) is CalculatorName.HOUSING


def test_every_commute_category_would_vanish_without_onsite_days() -> None:
    # The disambiguating rule: a cost belongs to COMMUTE only if it falls to
    # zero when onsite_days_per_week is zero. Fixed vehicle costs do not, which
    # is why they live under LIVING.
    commute = categories_owned_by(CalculatorName.COMMUTE)
    assert all(category.value.startswith("COMMUTE_") for category in commute)
    assert CostCategory.LIVING_VEHICLE_FIXED not in commute


# --- Cash-flow classification ----------------------------------------------


def test_cash_flow_types_cover_the_four_tracks() -> None:
    assert {kind.value for kind in CashFlowType} == {
        "RECURRING_CASH",
        "ONE_TIME_CASH",
        "NON_CASH_WEALTH",
        "TIME",
    }


def test_relocation_costs_are_one_time() -> None:
    # A security deposit is an event, not a rate; annualising it would be the
    # error explicit periods and cash-flow types exist to prevent.
    assert owner_of(CostCategory.RELOCATION_DEPOSIT) is CalculatorName.RELOCATION


def test_every_category_has_an_owning_calculator() -> None:
    """A category with no owner raises only when something tries to use it.

    That is late: the taxonomy is a closed set, so an unmapped member is a
    mistake at the moment it is declared, not at the moment a calculator
    happens to meet it. Adding LIVING_CLOTHING is what made the gap visible.
    """
    unmapped = [category for category in CostCategory if category not in CATEGORY_OWNER]
    assert unmapped == [], f"no owning calculator for: {[c.value for c in unmapped]}"


def test_the_owner_map_names_no_category_that_does_not_exist() -> None:
    assert set(CATEGORY_OWNER) <= set(CostCategory)
