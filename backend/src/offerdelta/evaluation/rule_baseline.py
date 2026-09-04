"""The rule-based baseline.

This is the number the LLM has to beat, and building it first is the difference
between a result and a press release. "The model scores 0.91" says nothing until
it sits beside a baseline — it might be beating 0.40, or losing to 0.93 at a
hundredth of the cost and latency.

Two design choices matter more than the rules themselves.

**It abstains rather than guesses.** A rule set that guesses on every unseen
merchant would post a higher accuracy and be worse in every way that counts,
because the hybrid routes precisely on abstentions. Guessing destroys the signal.

**It is fitted on the development split only.** Fitting on the holdout would be
the leak the merchant-disjoint split exists to prevent, arriving by a different
door.

The rules are deliberately unsophisticated: remember what each merchant was
labelled, fall back to two weak signals, otherwise say UNKNOWN. That is the
point. A baseline that is clever is a baseline nobody can reason about.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.transactions.view import TransactionView
from offerdelta.evaluation.categorisers import Prediction
from offerdelta.evaluation.dataset import LabelledDataset

#: Descriptions containing these are almost always movement between the user's
#: own accounts. TRANSFER earns a keyword rule even in a baseline this simple,
#: because it is the label whose absence causes double counting.
_TRANSFER_KEYWORDS: Final = ("TRANSFER", "XFER", "TFR TO", "TFR FROM", "ZELLE")

#: Deliberately low. A positive amount usually means money arriving, but the
#: hybrid must be able to overrule it without a fight.
_SIGN_CONFIDENCE: Final = Decimal("0.5")
_KEYWORD_CONFIDENCE: Final = Decimal("0.7")


@dataclass(frozen=True)
class RuleBaseline:
    """A merchant lookup table with two weak fallbacks."""

    #: merchant -> (label, share of that merchant's rows carrying it)
    merchant_labels: dict[str, tuple[str, Decimal]]

    @property
    def name(self) -> str:
        return "rules"

    @property
    def merchant_count(self) -> int:
        """How many merchants the table memorised.

        The honest measure of a system that works by memorising: coverage of
        the table, stated rather than implied by an accuracy figure.
        """
        return len(self.merchant_labels)

    def predict(self, view: TransactionView) -> Prediction:
        merchant = view.normalised_merchant

        # Exact merchant knowledge beats any keyword guess.
        if merchant in self.merchant_labels:
            label, consistency = self.merchant_labels[merchant]
            return Prediction(
                label=label,
                confidence=consistency,
                reason=(
                    f"merchant {merchant!r} was labelled {label} in "
                    f"{consistency:.0%} of development rows"
                ),
            )

        haystack = f"{view.raw_description} {merchant}".upper()
        for keyword in _TRANSFER_KEYWORDS:
            if keyword in haystack:
                return Prediction(
                    label="TRANSFER",
                    confidence=_KEYWORD_CONFIDENCE,
                    reason=f"description contains {keyword!r}",
                )

        if view.amount.amount > 0:
            return Prediction(
                label="INCOME",
                confidence=_SIGN_CONFIDENCE,
                reason="amount is positive, so money arrived",
            )

        return Prediction.abstain(f"merchant {merchant!r} was not seen in the development split")

    def predict_many(self, views: Sequence[TransactionView]) -> list[Prediction]:
        return [self.predict(view) for view in views]


def fit_rules(dataset: LabelledDataset) -> RuleBaseline:
    """Learn the merchant table from a development split.

    Where a merchant carries more than one label — real data disagrees with
    itself — the majority wins, and the share supporting it becomes the
    confidence. A merchant labelled one way four times is a safer rule than one
    labelled two ways, and the hybrid router needs to see that difference.
    """
    if not dataset.records:
        raise ValidationError("fitting rules needs at least one labelled record")

    by_merchant: dict[str, Counter[str]] = defaultdict(Counter)
    for record in dataset.records:
        by_merchant[record.normalised_merchant][record.gold_label] += 1

    table: dict[str, tuple[str, Decimal]] = {}
    for merchant, counts in by_merchant.items():
        total = sum(counts.values())
        # Sorted for determinism: an exact tie must not depend on dict ordering.
        label, hits = max(sorted(counts.items()), key=lambda item: item[1])
        table[merchant] = (label, Decimal(hits) / Decimal(total))

    return RuleBaseline(merchant_labels=table)
