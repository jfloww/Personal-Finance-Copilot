"""Categorisers and the port they share.

Three systems get compared on one frozen dataset, so all three implement one
interface and the harness cannot accidentally treat them differently.

Every categoriser returns a `Prediction` rather than a bare label, carrying
confidence and a reason. Confidence is what makes the hybrid possible — routing
by it is the whole idea — and the reason is what makes a wrong answer
diagnosable rather than merely wrong.

`Categoriser.name` is recorded in the report. A comparison that cannot say which
system produced which column is not a comparison.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from offerdelta.domain.transactions.view import TransactionView
from offerdelta.evaluation.labels import ABSTAIN, LABEL_SPACE


@dataclass(frozen=True)
class Prediction:
    """One categorisation, with enough context to route and to diagnose it."""

    label: str
    confidence: Decimal
    reason: str

    def __post_init__(self) -> None:
        if self.label not in LABEL_SPACE:
            raise ValueError(f"{self.label!r} is not in the label space")
        if not Decimal(0) <= self.confidence <= Decimal(1):
            raise ValueError(f"confidence must be between 0 and 1, got {self.confidence}")

    @property
    def abstained(self) -> bool:
        return self.label == ABSTAIN

    @classmethod
    def abstain(cls, reason: str) -> Prediction:
        """Declining to answer. Confidence is zero by construction."""
        return cls(label=ABSTAIN, confidence=Decimal(0), reason=reason)


class Categoriser(Protocol):
    """Anything that can label a transaction."""

    @property
    def name(self) -> str: ...

    def predict(self, view: TransactionView) -> Prediction: ...

    def predict_many(self, views: Sequence[TransactionView]) -> list[Prediction]: ...
