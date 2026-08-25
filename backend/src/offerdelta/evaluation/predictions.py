"""Writing down what each system actually predicted, row by row.

The report renders aggregates and throws the predictions away. That is fine
until the first question a result provokes, which is always *which rows*, and
answering it by running the model again costs money to learn something the last
run already knew.

So a run can record its predictions. Three things become possible:

- **Failure analysis** reads them instead of re-running.
- **A scoring policy can change** - a new acceptable-label set, a different
  abstention rule - and the old predictions can be re-scored for free.
- **Two runs can be diffed** at the row level, which is the only way to see
  that a prompt change moved twelve rows rather than one net point.

**These files never leave the machine.** A row here is a real transaction's id
next to what a model thought of it, which is precisely what the public artifact
exists to avoid publishing. They are written under `backend/data/`, which the
repository denies by default, and the public results generator reads the
rendered reports rather than these.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Final

from offerdelta.domain.common.errors import ValidationError
from offerdelta.evaluation.categorisers import Prediction
from offerdelta.evaluation.dataset import LabelledDataset, LabelledTransaction

#: Bumped when the record shape changes, so a reader can refuse a file it does
#: not understand rather than silently missing a field.
PREDICTIONS_SCHEMA: Final = "predictions/1"


@dataclass(frozen=True)
class RecordedPrediction:
    """One system's answer for one row, as written to disk."""

    transaction_id: str
    system: str
    predicted: str
    confidence: Decimal
    abstained: bool
    reason: str
    gold: str
    acceptable: frozenset[str]
    annotators_agreed: bool | None

    @property
    def correct(self) -> bool:
        """Scored the same way the report scores: acceptable set, else gold."""
        if self.abstained:
            return False
        if self.acceptable:
            return self.predicted in self.acceptable
        return self.predicted == self.gold

    def to_json(self) -> dict[str, object]:
        return {
            "transaction_id": self.transaction_id,
            "system": self.system,
            "predicted": self.predicted,
            "confidence": str(self.confidence),
            "abstained": self.abstained,
            "reason": self.reason,
            "gold": self.gold,
            "acceptable": sorted(self.acceptable),
            "annotators_agreed": self.annotators_agreed,
        }

    @classmethod
    def from_json(cls, row: dict[str, object]) -> RecordedPrediction:
        return cls(
            transaction_id=str(row["transaction_id"]),
            system=str(row["system"]),
            predicted=str(row["predicted"]),
            confidence=Decimal(str(row["confidence"])),
            abstained=bool(row["abstained"]),
            reason=str(row["reason"]),
            gold=str(row["gold"]),
            acceptable=frozenset(str(v) for v in _as_list(row["acceptable"])),
            annotators_agreed=(
                None if row["annotators_agreed"] is None else bool(row["annotators_agreed"])
            ),
        )


def _as_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValidationError(f"expected a list, got {type(value).__name__}")
    return value


@dataclass(frozen=True)
class PredictedSystem:
    """One system's name and its predictions, in dataset order."""

    name: str
    predictions: Sequence[Prediction]


def _pairs(
    dataset: LabelledDataset, system: PredictedSystem
) -> Iterator[tuple[LabelledTransaction, Prediction]]:
    if len(system.predictions) != len(dataset.records):
        raise ValidationError(
            f"{system.name} has {len(system.predictions)} predictions for "
            f"{len(dataset.records)} rows; a partial run is not worth recording"
        )
    yield from zip(dataset.records, system.predictions, strict=True)


def write_predictions(
    path: Path,
    dataset: LabelledDataset,
    systems: Sequence[PredictedSystem],
    *,
    header: dict[str, object] | None = None,
) -> int:
    """Write one JSONL file holding every system's answer for every row.

    The first line is a header naming the dataset and its checksum, so a
    predictions file can never be silently re-scored against different labels
    than the ones it was produced under.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "schema": PREDICTIONS_SCHEMA,
                    "dataset_version": dataset.dataset_version,
                    "dataset_checksum": dataset.checksum,
                    "rows": len(dataset.records),
                    "systems": [system.name for system in systems],
                    **(header or {}),
                }
            )
            + "\n"
        )
        for system in systems:
            for record, prediction in _pairs(dataset, system):
                recorded = RecordedPrediction(
                    transaction_id=record.transaction_id,
                    system=system.name,
                    predicted=prediction.label,
                    confidence=prediction.confidence,
                    abstained=prediction.abstained,
                    reason=prediction.reason,
                    gold=record.gold_label,
                    acceptable=record.acceptable_labels,
                    annotators_agreed=record.annotators_agree,
                )
                handle.write(json.dumps(recorded.to_json()) + "\n")
                written += 1

    return written


def read_predictions(path: Path) -> tuple[dict[str, object], list[RecordedPrediction]]:
    """The header and every recorded prediction, in the order written."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValidationError(f"{path} is empty")

    header = json.loads(lines[0])
    if not isinstance(header, dict):
        raise ValidationError(f"{path}: first line is not a header object")
    if header.get("schema") != PREDICTIONS_SCHEMA:
        raise ValidationError(
            f"{path}: schema {header.get('schema')!r} is not {PREDICTIONS_SCHEMA!r}; "
            f"refusing to read a file whose shape this code does not know"
        )

    return header, [RecordedPrediction.from_json(json.loads(line)) for line in lines[1:] if line]
