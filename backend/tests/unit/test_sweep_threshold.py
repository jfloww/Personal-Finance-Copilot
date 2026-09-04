"""Choosing where to stop trusting the model.

The threshold is picked on the development split and reported once on the
frozen benchmark - the same separation that selected prompt v3. A number
chosen by looking at the benchmark would be a number the benchmark can no
longer check.
"""

from __future__ import annotations

from decimal import Decimal

from offerdelta.evaluation.predictions import RecordedPrediction
from sweep_threshold import ThresholdPoint, choose_threshold, sweep


def _row(
    transaction_id: str,
    *,
    confidence: str,
    predicted: str,
    gold: str,
    abstained: bool = False,
) -> RecordedPrediction:
    """One recorded row, built by hand rather than read from a real predictions file.

    A unit test that depended on `data/eval/predictions/*.jsonl` could not run in
    CI: that directory is git-denied because a row there is a real transaction id
    next to what a model thought of it. Every field the test cares about
    (confidence, correctness, abstention) is set explicitly here instead.
    """
    return RecordedPrediction(
        transaction_id=transaction_id,
        system="llm:test-model",
        predicted=predicted,
        confidence=Decimal(confidence),
        abstained=abstained,
        reason="fixture row",
        gold=gold,
        acceptable=frozenset(),
        annotators_agreed=True,
    )


def _predictions() -> list[RecordedPrediction]:
    """Correct high-confidence rows, wrong low-confidence rows, one abstention.

    Mixed so each test below has a concrete reason to hold rather than passing
    by accident: correct rows only clear a high bar, wrong rows only clear a
    low one, and the abstention carries confidence zero by construction (see
    `Prediction.abstain`), so it never counts as "answered" once a threshold is
    above zero.
    """
    return [
        _row("t-1", confidence="0.95", predicted="LIVING_GROCERY", gold="LIVING_GROCERY"),
        _row("t-2", confidence="0.90", predicted="LIVING_DINING", gold="LIVING_DINING"),
        _row("t-3", confidence="0.85", predicted="TRANSFER", gold="TRANSFER"),
        _row("t-4", confidence="0.40", predicted="REFUND", gold="TRANSFER"),
        _row("t-5", confidence="0.30", predicted="LIVING_OTHER", gold="COMMUTE_TRANSIT_FARE"),
        _row("t-6", confidence="0.00", predicted="UNKNOWN", gold="LIVING_DINING", abstained=True),
    ]


def test_a_threshold_of_zero_queues_nothing() -> None:
    points = sweep(_predictions(), [Decimal("0.0")])
    assert points[0].queued == 0
    assert points[0].coverage == Decimal("1")


def test_raising_the_threshold_queues_more_and_never_fewer() -> None:
    points = sweep(_predictions(), [Decimal("0.0"), Decimal("0.5"), Decimal("0.9")])
    queued = [p.queued for p in points]
    assert queued == sorted(queued)


def test_accuracy_among_answered_rows_rises_with_the_threshold() -> None:
    """The whole point: what is left after queueing should be more trustworthy."""
    points = sweep(_predictions(), [Decimal("0.0"), Decimal("0.8")])
    assert points[1].accuracy_when_answered >= points[0].accuracy_when_answered


def test_sweep_returns_one_point_per_threshold_in_the_order_given() -> None:
    thresholds = [Decimal("0.9"), Decimal("0.0"), Decimal("0.5")]
    points = sweep(_predictions(), thresholds)
    assert [p.threshold for p in points] == thresholds
    assert all(isinstance(p, ThresholdPoint) for p in points)


def test_choose_threshold_prefers_a_point_that_actually_separates_right_from_wrong() -> None:
    """Picking 1.0 (queue everything) would trivially maximise accuracy-when-answered.

    `choose_threshold` must not fall into that trap: 1.0 answers nothing, so it
    cannot be what "evidence" recommends. The candidate grid below includes it
    on purpose, and the fixture's three wrong-or-abstained rows all sit below
    0.5 while its three correct rows all sit at 0.85 or above, so a threshold
    around there should score better than the extremes.
    """
    thresholds = [Decimal("0.0"), Decimal("0.5"), Decimal("0.8"), Decimal("1.0")]
    chosen = choose_threshold(_predictions(), thresholds)
    assert chosen not in (Decimal("0.0"), Decimal("1.0"))


def test_uninformative_confidence_does_not_favour_a_higher_threshold() -> None:
    """Self-review: does the sweep measure separation, or does it just reward
    raising the bar regardless of whether confidence means anything?

    Built so every confidence level carries exactly one correct and one wrong
    row. No threshold separates right from wrong any better than any other, so
    accuracy-when-answered must sit at exactly one half everywhere it is
    defined, and the chosen threshold must not be forced to the top of the
    grid - which is what a naive "higher threshold looks better" rule would
    always produce, on any data, informative or not.
    """
    levels = ["0.2", "0.4", "0.6", "0.8"]
    predictions = []
    for level in levels:
        predictions.append(
            _row(f"right-{level}", confidence=level, predicted="TRANSFER", gold="TRANSFER")
        )
        predictions.append(
            _row(f"wrong-{level}", confidence=level, predicted="TRANSFER", gold="REFUND")
        )

    thresholds = [Decimal("0.0"), Decimal("0.2"), Decimal("0.4"), Decimal("0.6"), Decimal("0.8")]
    points = sweep(predictions, thresholds)
    for point in points[:-1]:  # the top threshold answers nothing; skip the vacuous case
        assert point.accuracy_when_answered == Decimal("0.5000")

    chosen = choose_threshold(predictions, thresholds)
    assert chosen != max(thresholds)
