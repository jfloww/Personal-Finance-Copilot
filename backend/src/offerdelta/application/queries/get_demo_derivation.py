"""The walking skeleton's use case.

Supplies a hardcoded Auburn profile to the domain calculation and returns the
resulting derivation tree. Milestone 2 replaces the hardcoded figures with a
stored profile; milestone 5 replaces this query with one that reads from
PostgreSQL.

Every figure here is an assumption standing in for real data, which is exactly
why the tree marks provenance on each node.
"""

from __future__ import annotations

from offerdelta.domain.common.derivation import DerivationNode
from offerdelta.domain.common.money import Money
from offerdelta.domain.comparisons.simple_cash import monthly_disposable_cash


def get_demo_derivation() -> DerivationNode:
    """Compute the demo profile's monthly disposable cash, with its derivation."""
    return monthly_disposable_cash(
        net_pay=Money.parse("4820.00"),
        rent=Money.parse("1150.00"),
        utilities=Money.parse("185.00"),
        commute=Money.parse("240.00"),
        living=Money.parse("910.00"),
    )
