"""Running several systems over one frozen dataset and reporting the comparison.

Four guarantees hold the report together.

**The labels cannot move.** The dataset checksum is taken before the run and
verified after it. If any system mutated a record — directly, or by holding a
reference to a mutable field — the report refuses to be produced rather than
publishing numbers scored against labels that changed underneath them. Frozen
dataclasses make this unlikely; checking makes it provable.

**Every system sees the same rows in the same order.** They are scored on one
holdout, and the checksum in the header lets a reader confirm two reports really
did use the same data.

**Ambiguous rows are a separate stratum, not an exclusion.** They stay in the
overall figures, where their acceptable-label sets apply, and are reported again
on their own. Dropping them would flatter every system by removing the hardest
rows; hiding the split would make the overall number impossible to interpret.

**Nothing here knows what a model is.** Systems are asked for predictions and,
optionally, for usage. A system with no cost reports none, and the report says
so rather than printing zeros that look like a measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.transactions.view import TransactionView
from offerdelta.evaluation.categorisers import Prediction
from offerdelta.evaluation.dataset import LabelledDataset, LabelledTransaction
from offerdelta.evaluation.metrics import ClassificationReport, cohens_kappa, score
from offerdelta.evaluation.usage import ReportsUsage, Usage


class EvaluatableSystem(Protocol):
    """Anything the harness can score. Deliberately narrow."""

    @property
    def name(self) -> str: ...

    def predict_many(self, views: Sequence[TransactionView]) -> list[Prediction]: ...


@dataclass(frozen=True)
class AgreementSummary:
    """How much the annotators agreed, and how much of that was luck."""

    double_annotated: int
    agreed: int
    raw_agreement: Decimal | None
    kappa: Decimal | None

    def render(self) -> str:
        if not self.double_annotated:
            return "  annotators: no double-annotated rows, agreement unmeasurable"
        kappa = "undefined" if self.kappa is None else str(self.kappa)
        return (
            f"  annotators: {self.agreed}/{self.double_annotated} agreed "
            f"({self.raw_agreement}), Cohen's kappa {kappa}"
        )


@dataclass(frozen=True)
class SystemResult:
    """One system's performance across the strata."""

    name: str
    overall: ClassificationReport
    unambiguous: ClassificationReport
    ambiguous: ClassificationReport | None
    usage: Usage | None

    #: Rows the two annotators agreed on, and rows they did not. A difficulty
    #: stratum, not an ambiguity claim: disagreement usually means one annotator
    #: was wrong, and this project refuses to read it as legitimate ambiguity.
    #: What it does measure is whether the rows humans found hard are the same
    #: rows the model finds hard. `None` when nothing was double-annotated.
    annotators_agreed: ClassificationReport | None = None
    annotators_disagreed: ClassificationReport | None = None

    #: Every prediction, in dataset order. Kept so a scoring policy can change
    #: without paying for inference again, and so failure analysis has something
    #: to read. Never published: these are per-row and stay on the machine that
    #: ran them.
    predictions: tuple[Prediction, ...] = ()

    def render(self, *, input_price: Decimal | None, output_price: Decimal | None) -> str:
        lines = [f"--- {self.name}", self.overall.render()]

        lines.append("")
        lines.append("  strata")
        lines.append(
            f"    unambiguous ({self.unambiguous.total:>4} rows)  "
            f"macro F1 {self.unambiguous.macro_f1}  "
            f"weighted {self.unambiguous.weighted_f1}"
        )
        if self.ambiguous is None:
            lines.append(
                "    ambiguous   (   0 rows)  none authored in this holdout - "
                "ambiguity is written down during adjudication, never inferred"
            )
        else:
            lines.append(
                f"    ambiguous   ({self.ambiguous.total:>4} rows)  "
                f"macro F1 {self.ambiguous.macro_f1}  "
                f"weighted {self.ambiguous.weighted_f1}"
            )

        if self.annotators_agreed is not None and self.annotators_disagreed is not None:
            lines.append("")
            lines.append("  by annotator agreement (difficulty, not ambiguity)")
            for label, report in (
                ("agreed   ", self.annotators_agreed),
                ("disagreed", self.annotators_disagreed),
            ):
                lines.append(
                    f"    {label} ({report.total:>4} rows)  "
                    f"accuracy {report.accuracy}  macro F1 {report.macro_f1}"
                )

        lines.append("")
        if self.usage is None:
            lines.append("  cost: none - this system makes no external calls")
        else:
            usage = self.usage
            cost = usage.cost(input_per_million=input_price, output_per_million=output_price)
            priced = "not priced" if cost is None else f"{cost:.4f}"
            per_row = (
                "n/a"
                if cost is None or self.overall.total == 0
                else f"{cost / Decimal(self.overall.total):.6f}"
            )
            lines.append(
                f"  tokens: {usage.input_tokens} in, {usage.output_tokens} out, "
                f"{usage.total_tokens} total over {usage.calls} calls"
            )
            rates = (
                "unpriced"
                if input_price is None or output_price is None
                else f"at {input_price}/{output_price} per Mtok"
            )
            lines.append(f"  cost: total {priced}, per row {per_row} ({rates})")
            lines.append(
                f"  latency: mean {usage.mean_latency_ms}ms  "
                f"p50 {usage.p50_latency_ms}ms  p95 {usage.p95_latency_ms}ms"
            )
            if usage.failures or usage.rejected_outputs:
                lines.append(
                    f"  provider failures {usage.failures}, "
                    f"outputs rejected as outside the taxonomy {usage.rejected_outputs}"
                )
        return "\n".join(lines)


@dataclass(frozen=True)
class EvaluationReport:
    """Several systems, one frozen dataset, side by side."""

    dataset_version: str
    schema_version: str
    checksum: str
    rows: int
    merchants: int
    ambiguous_rows: int
    agreement: AgreementSummary
    systems: tuple[SystemResult, ...]
    input_price_per_million: Decimal | None = None
    output_price_per_million: Decimal | None = None

    def best_by_macro_f1(self) -> SystemResult:
        return max(self.systems, key=lambda s: s.overall.macro_f1)

    def _scoring_policy(self) -> str:
        """What "correct" meant in this run, said once rather than assumed.

        Accuracy here is acceptable-label accuracy: an ambiguous row counts a
        prediction correct if it is anywhere in that row's authored set. With no
        ambiguous rows the two measures coincide, and saying so is better than
        letting a reader guess which one they are looking at.
        """
        if self.ambiguous_rows:
            return (
                f"  scoring: acceptable-label accuracy - {self.ambiguous_rows} "
                f"ambiguous rows count any label in their authored set"
            )
        return (
            "  scoring: acceptable-label accuracy, which equals exact-match "
            "here - no row carries an authored acceptable set"
        )

    def render(self) -> str:
        lines = [
            "EVALUATION REPORT",
            f"  dataset {self.dataset_version} (schema {self.schema_version})",
            f"  checksum {self.checksum[:16]}...",
            f"  {self.rows} rows, {self.merchants} merchants, {self.ambiguous_rows} ambiguous",
            self.agreement.render(),
            self._scoring_policy(),
            "",
        ]
        for result in self.systems:
            lines.append(
                result.render(
                    input_price=self.input_price_per_million,
                    output_price=self.output_price_per_million,
                )
            )
            lines.append("")

        lines.append("SUMMARY")
        lines.append(f"  {'system':<34}{'macro F1':>10}{'weighted':>10}{'coverage':>10}")
        for result in self.systems:
            lines.append(
                f"  {result.name:<34}{result.overall.macro_f1:>10}"
                f"{result.overall.weighted_f1:>10}{result.overall.coverage:>10}"
            )
        return "\n".join(lines)


def evaluate(
    dataset: LabelledDataset,
    systems: Sequence[EvaluatableSystem],
    *,
    input_price_per_million: Decimal | None = None,
    output_price_per_million: Decimal | None = None,
) -> EvaluationReport:
    """Score every system on the same frozen holdout.

    Raises if the dataset changed during the run. A benchmark whose labels can
    move is not a benchmark, and the only way to know they did not is to check.
    """
    if not systems:
        raise ValidationError("evaluation needs at least one system to score")

    records = dataset.records
    before = dataset.checksum

    results = tuple(_run(system, records) for system in systems)

    after = dataset.checksum
    if before != after:
        raise ValidationError(
            "the dataset changed during evaluation: checksum was "
            f"{before[:16]} and is now {after[:16]}. Results scored against "
            "labels that moved are meaningless and will not be reported."
        )

    return EvaluationReport(
        dataset_version=dataset.dataset_version,
        schema_version=dataset.schema_version,
        checksum=before,
        rows=len(dataset),
        merchants=len(dataset.merchants),
        ambiguous_rows=len(dataset.ambiguous),
        agreement=summarise_agreement(dataset),
        systems=results,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
    )


def _run(system: EvaluatableSystem, records: tuple[LabelledTransaction, ...]) -> SystemResult:
    predictions = system.predict_many([record.view for record in records])
    if len(predictions) != len(records):
        raise ValidationError(
            f"{system.name} returned {len(predictions)} predictions for "
            f"{len(records)} rows; a partial run cannot be scored"
        )

    labels = [prediction.label for prediction in predictions]

    return SystemResult(
        name=system.name,
        overall=_score_subset(records, labels, range(len(records))),
        unambiguous=_score_subset(
            records,
            labels,
            [i for i, record in enumerate(records) if not record.ambiguous],
        ),
        ambiguous=_score_optional(
            records,
            labels,
            [i for i, record in enumerate(records) if record.ambiguous],
        ),
        annotators_agreed=_score_optional(
            records,
            labels,
            [i for i, record in enumerate(records) if record.annotators_agree is True],
        ),
        annotators_disagreed=_score_optional(
            records,
            labels,
            [i for i, record in enumerate(records) if record.annotators_agree is False],
        ),
        predictions=tuple(predictions),
        usage=system.usage() if isinstance(system, ReportsUsage) else None,
    )


def _score_subset(
    records: tuple[LabelledTransaction, ...],
    labels: list[str],
    indices: Sequence[int] | range,
) -> ClassificationReport:
    """Score a stratum that must contain rows."""
    chosen = list(indices)
    if not chosen:
        raise ValidationError("cannot score an empty subset")

    pairs = [(records[i].gold_label, labels[i]) for i in chosen]

    # Acceptable-label sets travel with the row, re-indexed to the subset, so a
    # stratum scores ambiguity exactly as the overall run does.
    acceptable = {
        position: records[i].acceptable_labels
        for position, i in enumerate(chosen)
        if records[i].ambiguous
    }
    return score(pairs, acceptable=acceptable)


def _score_optional(
    records: tuple[LabelledTransaction, ...],
    labels: list[str],
    indices: Sequence[int],
) -> ClassificationReport | None:
    """Score a stratum that may legitimately be empty.

    A holdout with no ambiguous rows is a normal outcome, and reporting None
    says so more honestly than a report over zero rows.
    """
    return _score_subset(records, labels, indices) if indices else None


def summarise_agreement(dataset: LabelledDataset) -> AgreementSummary:
    """Annotator agreement, computed from the labels alone.

    Independent of every system: this is a property of the benchmark, and it is
    the ceiling each system's score has to be read against.
    """
    double = dataset.double_annotated
    if not double:
        return AgreementSummary(0, 0, None, None)

    pairs = [
        (record.primary_label, record.secondary_label)
        for record in double
        if record.secondary_label is not None
    ]
    agreed = sum(1 for a, b in pairs if a == b)
    raw = (Decimal(agreed) / Decimal(len(pairs))).quantize(Decimal("0.0001"))

    return AgreementSummary(
        double_annotated=len(pairs),
        agreed=agreed,
        raw_agreement=raw,
        kappa=cohens_kappa(pairs),
    )
