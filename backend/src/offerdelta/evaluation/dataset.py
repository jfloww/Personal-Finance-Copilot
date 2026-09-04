"""The labelled evaluation dataset.

The benchmark is only worth the discipline in its schema, and this one carries
four decisions that a looser schema would quietly lose.

**Adjudication is a third field, not an overwrite.** When two annotators
disagree and a human resolves it, both original labels stay. Overwriting the
primary would make inter-annotator agreement unmeasurable afterwards, and
agreement is one of the headline numbers — a benchmark whose own annotators
agreed only 70% of the time has a ceiling that every model score must be read
against.

**Ambiguity is represented, not resolved.** AMAZON is genuinely groceries or
shopping depending on the basket. Forcing one gold label there manufactures
errors that teach nothing about the model, so an ambiguous row carries a set of
acceptable labels and is reported as its own stratum.

**Abstention is separate from error.** A model saying UNKNOWN has not made a
mistake; it has declined to make one. Folding that into the error rate punishes
exactly the behaviour worth rewarding, so it is reported as coverage.

**The dataset is provably frozen.** Comparing three systems on "the same data"
is a claim. The content checksum makes it a fact, and it ignores record order
so two runs over the same rows in a different sequence still agree.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Final

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.view import TransactionView
from offerdelta.evaluation.labels import LABEL_SPACE

#: Bumped when the record shape changes. A stored result names the schema that
#: produced it, so an old report is never silently reinterpreted under new rules.
SCHEMA_VERSION: Final = "1.0"

#: An ambiguous row that lists one acceptable label is just a labelled row
#: with an extra claim attached, so two is the floor.
_MIN_ACCEPTABLE_LABELS: Final = 2


@dataclass(frozen=True)
class LabelledTransaction:
    """One human-labelled transaction."""

    transaction_id: str
    posted_on: date
    raw_description: str
    normalised_merchant: str
    amount: Money
    account_type: str

    #: Where the row came from, and which export format produced it. Any claim
    #: about generalising across banks depends on being able to slice by these.
    source: str
    bank_format: str

    primary_label: str

    #: Present only for the stratified double-annotated subset.
    secondary_label: str | None = None

    #: Set when a disagreement was resolved. Never replaces the two above.
    adjudicated_label: str | None = None

    #: Non-empty only for ambiguous rows. Any prediction inside it is correct.
    acceptable_labels: frozenset[str] = field(default_factory=frozenset)

    #: No single right answer exists for this row.
    ambiguous: bool = False

    #: The annotator was unsure. Distinct from ambiguity, which is a property
    #: of the transaction rather than of the annotator.
    needs_review: bool = False

    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.transaction_id.strip():
            raise ValidationError("a labelled transaction needs an id")
        if not self.normalised_merchant.strip():
            raise ValidationError(
                "a labelled transaction needs a normalised merchant; it is the "
                "split key, and without it leakage between rule data and "
                "evaluation data cannot be detected"
            )

        for name, label in (
            ("primary_label", self.primary_label),
            ("secondary_label", self.secondary_label),
            ("adjudicated_label", self.adjudicated_label),
        ):
            if label is not None and label not in LABEL_SPACE:
                raise ValidationError(f"{name} {label!r} is not in the label space")

        for label in self.acceptable_labels:
            if label not in LABEL_SPACE:
                raise ValidationError(f"acceptable label {label!r} is not in the label space")

        if self.ambiguous:
            if len(self.acceptable_labels) < _MIN_ACCEPTABLE_LABELS:
                raise ValidationError(
                    f"{self.transaction_id}: an ambiguous row needs at least two "
                    f"acceptable labels, otherwise the ambiguity claim has "
                    f"nothing behind it"
                )
            if self.gold_label not in self.acceptable_labels:
                raise ValidationError(
                    f"{self.transaction_id}: the gold label {self.gold_label!r} is "
                    f"not among the acceptable labels, so the row would score "
                    f"its own answer wrong"
                )

    @property
    def gold_label(self) -> str:
        """The label to score against: adjudicated where one exists."""
        return self.adjudicated_label or self.primary_label

    @property
    def view(self) -> TransactionView:
        """What a categoriser sees. Never the label."""
        return TransactionView(
            normalised_merchant=self.normalised_merchant,
            raw_description=self.raw_description,
            amount=self.amount,
            account_type=self.account_type,
        )

    @property
    def has_second_annotation(self) -> bool:
        return self.secondary_label is not None

    @property
    def annotators_agree(self) -> bool | None:
        """None when only one annotator saw this row."""
        if self.secondary_label is None:
            return None
        return self.primary_label == self.secondary_label

    def is_correct(self, predicted: str) -> bool:
        """Whether a prediction counts as right for this row.

        Ambiguous rows accept any label in their acceptable set; everything
        else accepts only the gold label.
        """
        if self.ambiguous:
            return predicted in self.acceptable_labels
        return predicted == self.gold_label


@dataclass(frozen=True)
class LabelledDataset:
    """A frozen, versioned set of labelled transactions."""

    dataset_version: str
    records: tuple[LabelledTransaction, ...]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # A generator passed here would be consumed by the duplicate check
        # below and read as empty ever after — a benchmark silently scoring
        # zero rows. The annotation says tuple; this makes it true.
        if not isinstance(self.records, tuple):
            raise ValidationError(
                f"records must be a tuple, got {type(self.records).__name__}; a "
                f"generator would be exhausted on first read and leave the "
                f"dataset silently empty"
            )
        if not self.records:
            raise ValidationError("a dataset needs at least one record")

        seen: set[str] = set()
        for record in self.records:
            if record.transaction_id in seen:
                raise ValidationError(f"duplicate transaction id {record.transaction_id!r}")
            seen.add(record.transaction_id)

    @property
    def checksum(self) -> str:
        """A content digest that proves two runs used the same data.

        Order-independent: the rows are sorted before hashing, so evaluating
        the same dataset in a different sequence still produces the same value.
        """
        lines = sorted(
            "\x1f".join(
                (
                    record.transaction_id,
                    record.posted_on.isoformat(),
                    record.raw_description,
                    record.normalised_merchant,
                    str(record.amount.amount),
                    record.amount.currency,
                    record.account_type,
                    record.source,
                    record.bank_format,
                    record.primary_label,
                    record.secondary_label or "",
                    record.adjudicated_label or "",
                    ",".join(sorted(record.acceptable_labels)),
                    str(record.ambiguous),
                    str(record.needs_review),
                )
            )
            for record in self.records
        )
        payload = "\x1e".join([self.schema_version, self.dataset_version, *lines])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def merchants(self) -> frozenset[str]:
        return frozenset(record.normalised_merchant for record in self.records)

    @property
    def double_annotated(self) -> tuple[LabelledTransaction, ...]:
        return tuple(r for r in self.records if r.has_second_annotation)

    @property
    def ambiguous(self) -> tuple[LabelledTransaction, ...]:
        return tuple(r for r in self.records if r.ambiguous)

    @property
    def unambiguous(self) -> tuple[LabelledTransaction, ...]:
        return tuple(r for r in self.records if not r.ambiguous)

    @property
    def needs_review(self) -> tuple[LabelledTransaction, ...]:
        return tuple(r for r in self.records if r.needs_review)

    def __len__(self) -> int:
        return len(self.records)
