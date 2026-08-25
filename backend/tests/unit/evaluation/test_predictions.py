"""Recording what each system predicted, so a result can be questioned later.

The point of writing predictions down is that the next question - *which rows?*
- can be answered without paying for inference again. That only holds if the
file round-trips exactly and refuses to be read against the wrong labels, which
is what these pin.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.evaluation.categorisers import Prediction
from offerdelta.evaluation.dataset import LabelledDataset, LabelledTransaction
from offerdelta.evaluation.predictions import (
    PREDICTIONS_SCHEMA,
    PredictedSystem,
    RecordedPrediction,
    read_predictions,
    write_predictions,
)


def _record(
    transaction_id: str = "acct-1",
    primary: str = "LIVING_DINING",
    secondary: str | None = None,
    acceptable: frozenset[str] = frozenset(),
    ambiguous: bool = False,
) -> LabelledTransaction:
    return LabelledTransaction(
        transaction_id=transaction_id,
        posted_on=date(2026, 1, 1),
        raw_description="A DESCRIPTION",
        normalised_merchant=f"MERCHANT {transaction_id}",
        amount=Money(Decimal("-10.00")),
        account_type="CREDIT",
        source="test",
        bank_format="test",
        primary_label=primary,
        secondary_label=secondary,
        acceptable_labels=acceptable,
        ambiguous=ambiguous,
    )


def _dataset(*records: LabelledTransaction) -> LabelledDataset:
    return LabelledDataset(dataset_version="test", records=records)


def _prediction(label: str = "LIVING_DINING", confidence: str = "0.9") -> Prediction:
    return Prediction(label=label, confidence=Decimal(confidence), reason="because")


def test_predictions_round_trip(tmp_path: Path) -> None:
    dataset = _dataset(_record("a"), _record("b", primary="TRANSFER"))
    written = write_predictions(
        tmp_path / "p.jsonl",
        dataset,
        [PredictedSystem("llm:test", [_prediction(), _prediction("TRANSFER", "0.4")])],
    )
    assert written == 2

    header, rows = read_predictions(tmp_path / "p.jsonl")
    assert header["schema"] == PREDICTIONS_SCHEMA
    assert header["dataset_checksum"] == dataset.checksum
    assert [r.transaction_id for r in rows] == ["a", "b"]
    assert rows[1].confidence == Decimal("0.4")
    assert rows[1].predicted == "TRANSFER"


def test_the_header_carries_the_checksum_the_rows_were_scored_against(tmp_path: Path) -> None:
    """Without it, a predictions file could be silently re-scored against labels
    that moved after it was written - which is the exact failure the dataset
    checksum exists to prevent inside a single run."""
    dataset = _dataset(_record("a"))
    write_predictions(tmp_path / "p.jsonl", dataset, [PredictedSystem("llm:test", [_prediction()])])

    header, _ = read_predictions(tmp_path / "p.jsonl")
    assert header["dataset_checksum"] == dataset.checksum
    assert header["rows"] == 1


def test_a_partial_run_is_refused(tmp_path: Path) -> None:
    dataset = _dataset(_record("a"), _record("b"))
    with pytest.raises(ValidationError, match="partial run"):
        write_predictions(
            tmp_path / "p.jsonl", dataset, [PredictedSystem("llm:test", [_prediction()])]
        )


def test_an_unknown_schema_is_refused(tmp_path: Path) -> None:
    """Reading a file whose shape this code does not know would silently drop
    whatever field it did not expect."""
    path = tmp_path / "p.jsonl"
    path.write_text('{"schema": "predictions/99"}\n', encoding="utf-8")
    with pytest.raises(ValidationError, match="predictions/99"):
        read_predictions(path)


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValidationError, match="empty"):
        read_predictions(path)


def test_a_recorded_prediction_scores_itself_the_way_the_report_does() -> None:
    exact = RecordedPrediction(
        transaction_id="a",
        system="llm:test",
        predicted="LIVING_DINING",
        confidence=Decimal("0.9"),
        abstained=False,
        reason="",
        gold="LIVING_DINING",
        acceptable=frozenset(),
        annotators_agreed=True,
    )
    assert exact.correct

    wrong = RecordedPrediction(**{**vars(exact), "predicted": "TRANSFER"})
    assert not wrong.correct


def test_an_acceptable_label_counts_as_correct() -> None:
    """The whole reason the acceptable set exists. A row with no single right
    answer must not manufacture an error for a defensible one."""
    row = RecordedPrediction(
        transaction_id="a",
        system="llm:test",
        predicted="LIVING_GROCERY",
        confidence=Decimal("0.6"),
        abstained=False,
        reason="",
        gold="LIVING_DINING",
        acceptable=frozenset({"LIVING_DINING", "LIVING_GROCERY"}),
        annotators_agreed=False,
    )
    assert row.correct


def test_an_abstention_is_never_correct() -> None:
    """Declining is not an error, but it is not a right answer either. The
    report counts it as coverage; here it must not count as a hit."""
    row = RecordedPrediction(
        transaction_id="a",
        system="llm:test",
        predicted="UNKNOWN",
        confidence=Decimal(0),
        abstained=True,
        reason="declined",
        gold="UNKNOWN",
        acceptable=frozenset(),
        annotators_agreed=None,
    )
    assert not row.correct


def test_annotator_agreement_survives_the_round_trip(tmp_path: Path) -> None:
    """None, True and False are three different facts: no second annotator, two
    who agreed, and two who did not. Collapsing None into False would invent a
    disagreement that never happened."""
    dataset = _dataset(
        _record("a"),
        _record("b", secondary="LIVING_DINING"),
        _record("c", secondary="TRANSFER"),
    )
    write_predictions(
        tmp_path / "p.jsonl",
        dataset,
        [PredictedSystem("llm:test", [_prediction()] * 3)],
    )

    _, rows = read_predictions(tmp_path / "p.jsonl")
    assert [r.annotators_agreed for r in rows] == [None, True, False]
