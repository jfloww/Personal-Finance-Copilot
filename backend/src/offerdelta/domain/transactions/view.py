"""What a categoriser is allowed to see.

Deliberately four fields. A categoriser that could read a balance, a
neighbouring row, or a gold label would be scored on information the
running system does not have, and the score would not transfer.

It also exists so production code never has to construct a
`LabelledTransaction`: that type carries a human's answer, which a
categoriser must never be handed.
"""

from __future__ import annotations

from dataclasses import dataclass

from offerdelta.domain.common.money import Money


@dataclass(frozen=True)
class TransactionView:
    """The projection every categoriser predicts from."""

    normalised_merchant: str
    raw_description: str
    amount: Money
    account_type: str
