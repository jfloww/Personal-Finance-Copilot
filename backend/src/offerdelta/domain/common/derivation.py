"""Derivation trees.

Every figure the product shows can be expanded to reveal the inputs, formula,
and provenance behind it. This is the strongest demo feature in the project and
the reason a viewer can trust the numbers, so it is a first-class domain type
rather than a presentation concern.

A derivation tree is not a comparison concept: both the comparison engine and
the monthly report build one. It imports only Money, Evidence, PeriodKind and
the shared error type.

A node with children must equal the sum of those children. Child amounts are
signed — income positive, costs negative — so the whole tree is one addition.
That invariant is the seed of the monthly cash-flow reconciliation check the
engine enforces in milestone 3.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.evidence import Evidence
from offerdelta.domain.common.money import Money
from offerdelta.domain.common.periods import PeriodKind


@dataclass(frozen=True)
class DerivationNode:
    """One step in the explanation of a calculated figure."""

    code: str
    label: str
    amount: Money
    period: PeriodKind
    formula: str
    evidence: Evidence
    children: tuple[DerivationNode, ...] = field(default=())

    def __post_init__(self) -> None:
        if not self.children:
            return

        for child in self.children:
            if child.period is not self.period:
                raise ValidationError(
                    f"derivation node {self.code!r} has period {self.period} but "
                    f"child {child.code!r} has period {child.period}"
                )

        total = Money.zero(self.amount.currency)
        for child in self.children:
            total = total + child.amount
        if total != self.amount:
            raise ValidationError(
                f"derivation node {self.code!r} does not equal the sum of its "
                f"children: stated {self.amount}, children sum to {total}"
            )

    def walk(self) -> Iterator[DerivationNode]:
        """Yield this node then every descendant, depth first."""
        yield self
        for child in self.children:
            yield from child.walk()


def _weakest(evidence: Sequence[Evidence]) -> Evidence:
    """A branch is only as well-evidenced as its least-supported child.

    Taking the strongest would let one confirmed figure make a branch of
    guesses look sourced. Shared by every tree builder — the comparison
    engine's derivation and the monthly report both call this rather than
    each keeping its own copy, which would let the two definitions drift.

    A childless branch has no child evidence to be weakest of, and `SOURCED`
    - "taken from a versioned public dataset" - is the strongest claim this
    type can make, not the safe default for "nothing to go on". `ASSUMED` is:
    a branch with nothing behind it is exactly what that value is for.
    """
    for level in (Evidence.ASSUMED, Evidence.DERIVED, Evidence.USER_CONFIRMED):
        if level in evidence:
            return level
    return Evidence.SOURCED if evidence else Evidence.ASSUMED
