"""Choosing where to stop trusting the model.

    uv run python sweep_threshold.py \\
      data/eval/predictions/development-categorise-v3.jsonl \\
      --json ../docs/eval/review-threshold.json

`awaiting_review`'s `threshold` argument (Task 3) decides which suggested rows
a person sees and which ones a report treats as settled. That number has to
come from evidence, not from a value that merely looked reasonable: this
repository already rejected prompt v2 after measurement and adopted v3 only
because several metrics moved together on the development split before the
frozen benchmark was ever touched. A threshold picked by feel would be the one
unmeasured number in an otherwise measured system.

**The split discipline is the whole point.** `sweep` runs over
`development-categorise-v3.jsonl` and a candidate threshold is picked from
that curve alone, by `choose_threshold`. Only afterwards is the chosen number
measured - once - against `holdout-categorise-v3.jsonl`. Nothing in this file
ever lets the holdout numbers feed back into the choice: `main` builds the
development curve and the pick before it even opens the holdout file, and
`choose_threshold` never receives holdout predictions as an argument.

**Why Youden's J, not a hand-picked accuracy target.** A rule like "the
lowest threshold where accuracy-when-answered clears 90%" bakes in a target
nobody has justified, and "the highest available threshold" always wins that
kind of rule, informative confidence or not - which defeats the purpose of
sweeping a curve at all instead of just reading off its last point. Youden's J
(true-positive rate plus true-negative rate, minus one, treating "queued" as a
prediction of "wrong") is the standard way to pick a cutoff from a curve like
this: it rewards a threshold only for separating right answers from wrong
ones, and scores zero at both extremes - queue nothing, or queue everything -
because neither extreme uses the confidence signal to decide anything. See
`test_uninformative_confidence_does_not_favour_a_higher_threshold` for the
check that this really does score noise as flat rather than as "raise the
bar".

Row-level predictions never leave this file's own process. They are read from
`backend/data/eval/predictions/`, which `.gitignore` denies by default because
a row there is a real transaction id next to what a model thought of it; the
JSON this writes carries the swept curve (threshold, coverage, accuracy,
queued counts) and nothing that could identify a row.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Final

from offerdelta.domain.common.errors import ValidationError
from offerdelta.evaluation.predictions import RecordedPrediction, read_predictions

#: Enough places that a ratio survives rendering without pretending to more
#: precision than 265 development rows (or 135 holdout ones) can support.
#: Matches `offerdelta.evaluation.metrics`, which reports the same way.
_PLACES: Final = Decimal("0.0001")

#: Every confidence value either categoriser has ever recorded is a multiple
#: of this (see the distinct values in `data/eval/predictions/*.jsonl`); a
#: finer grid would only add points where the curve cannot move.
_GRID_STEP: Final = Decimal("0.05")

#: 0.00 through 1.00 inclusive, so "queue nothing" and "queue everything" both
#: appear on the curve as checkable endpoints, not just the interior.
DEFAULT_THRESHOLDS: Final[tuple[Decimal, ...]] = tuple(
    (Decimal(step) * _GRID_STEP).quantize(_GRID_STEP) for step in range(21)
)


def _ratio(numerator: int, denominator: int) -> Decimal:
    """Zero when there is nothing to divide by.

    A threshold that queues every row has an empty answered set; reporting
    zero for its accuracy is a fact about that threshold, not a crash the
    caller has to guard against separately. Quantized on both branches, so a
    reader of the artifact never sees a bare "0" next to "0.7434" and wonders
    whether the difference means something.
    """
    if denominator == 0:
        return Decimal(0).quantize(_PLACES)
    return (Decimal(numerator) / Decimal(denominator)).quantize(_PLACES)


@dataclass(frozen=True)
class ThresholdPoint:
    """One candidate threshold's cost and benefit, over one fixed set of predictions.

    `queued` and `coverage` are two views of the same split (they always sum
    to the row count / to one) and are both kept because a report reads one
    as a workload and the other as a rate.
    """

    threshold: Decimal
    coverage: Decimal
    accuracy_when_answered: Decimal
    queued: int

    def to_json(self) -> dict[str, object]:
        return {
            "threshold": str(self.threshold),
            "coverage": str(self.coverage),
            "accuracy_when_answered": str(self.accuracy_when_answered),
            "queued": self.queued,
        }


def sweep(
    predictions: Sequence[RecordedPrediction], thresholds: Sequence[Decimal]
) -> list[ThresholdPoint]:
    """One `ThresholdPoint` per threshold, over one fixed set of predictions.

    A row is "answered" at a threshold when `confidence >= threshold` - the
    same comparison `awaiting_review` runs in the other direction
    (`suggested_confidence < threshold` puts a row in the queue), so a
    threshold this function reports as queueing zero rows is a threshold
    `awaiting_review` would also queue zero rows at, on the same data.
    Correctness is read from `RecordedPrediction.correct`, which already
    scores an abstained row as wrong and an accepted-label row as right by
    the same rule the published report uses - this does not re-derive that
    judgement, only slices it by confidence.

    Raises `ValidationError` on an empty `predictions` sequence: a curve with
    no rows behind it would report 100% coverage over nothing, which is not a
    measurement of anything.
    """
    if not predictions:
        raise ValidationError(
            "sweep needs at least one prediction; a curve over zero rows "
            "would report perfect coverage over nothing"
        )

    total = len(predictions)
    points = []
    for threshold in thresholds:
        answered = [row for row in predictions if row.confidence >= threshold]
        correct = sum(1 for row in answered if row.correct)
        points.append(
            ThresholdPoint(
                threshold=threshold,
                coverage=_ratio(len(answered), total),
                accuracy_when_answered=_ratio(correct, len(answered)),
                queued=total - len(answered),
            )
        )
    return points


def choose_threshold(
    predictions: Sequence[RecordedPrediction], thresholds: Sequence[Decimal]
) -> Decimal:
    """The development-only pick: the threshold that best separates right from wrong.

    Scored by Youden's J - `P(queued | wrong) + P(trusted | correct) - 1` -
    the standard cutoff-selection statistic for exactly this shape of
    problem, maximised over `thresholds`. It answers "how much does this
    threshold's confidence bar actually track correctness", not "how high is
    accuracy after filtering", so it cannot be maximised by simply raising the
    bar: both queue-nothing and queue-everything score zero, because neither
    uses the confidence signal to separate anything. Ties are broken toward
    the first threshold reaching the maximum in `thresholds`' given order (a
    lower one, if the grid is ascending), which prefers keeping more rows out
    of the queue when two candidates separate equally well.

    Raises `ValidationError` when `predictions` has no wrong rows, no correct
    rows, or `thresholds` is empty: with one outcome absent, or nothing to
    compare, "separates right from wrong" has nothing to measure.
    """
    if not thresholds:
        raise ValidationError("choose_threshold needs at least one candidate threshold")

    total_wrong = sum(1 for row in predictions if not row.correct)
    total_correct = len(predictions) - total_wrong
    if total_wrong == 0 or total_correct == 0:
        raise ValidationError(
            "choose_threshold needs both right and wrong rows to measure "
            "separation; a split with only one outcome cannot tell one "
            "threshold apart from another"
        )

    best_threshold = thresholds[0]
    best_j: Decimal | None = None
    for threshold in thresholds:
        wrong_queued = sum(
            1 for row in predictions if not row.correct and row.confidence < threshold
        )
        correct_trusted = sum(
            1 for row in predictions if row.correct and row.confidence >= threshold
        )
        j = Decimal(wrong_queued) / total_wrong + Decimal(correct_trusted) / total_correct - 1
        if best_j is None or j > best_j:
            best_j = j
            best_threshold = threshold
    return best_threshold


def _llm_predictions(path: Path) -> tuple[dict[str, object], list[RecordedPrediction]]:
    """This file's rows for the one LLM system it recorded.

    `awaiting_review`'s threshold only ever compares against
    `suggested_confidence` written by a categoriser call - the rule tier
    answers with confidence 1 by construction (see `RuleBaseline`) and is not
    a candidate for this kind of thresholding. Predictions files record every
    system side by side for comparison; this keeps only the one whose
    confidence is worth sweeping.
    """
    header, rows = read_predictions(path)
    llm_name = next((row.system for row in rows if row.system.startswith("llm:")), None)
    if llm_name is None:
        raise ValidationError(f"{path}: no llm: system recorded; nothing to threshold")
    return header, [row for row in rows if row.system == llm_name]


def _derive_holdout_path(development_path: Path) -> Path:
    """The frozen counterpart of a development predictions file, by file naming.

    `data/eval/predictions/` names its files `development-<run>.jsonl` and
    `holdout-<run>.jsonl` in pairs (see `analyse_failures.py`'s own
    `DEFAULT_DATASET` convention for the sibling case). Deriving the name
    keeps `main`'s example invocation to a single path; `--holdout` overrides
    it for a `development_path` that does not follow the convention.
    """
    if "development" not in development_path.name:
        raise ValidationError(
            f"{development_path} does not look like a development predictions "
            f"file (expected 'development' in the name); pass --holdout explicitly"
        )
    return development_path.with_name(development_path.name.replace("development", "holdout", 1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sweep_threshold.py", description=__doc__)
    parser.add_argument("predictions", type=Path, help="development predictions JSONL")
    parser.add_argument(
        "--holdout",
        type=Path,
        default=None,
        help="frozen benchmark predictions JSONL, measured once and never used to choose",
    )
    parser.add_argument("--json", type=Path, default=None, help="write the artifact here")
    return parser


def _render_curve(points: Sequence[ThresholdPoint]) -> str:
    lines = [f"  {'threshold':>9}{'coverage':>10}{'acc.':>8}{'queued':>8}"]
    for point in points:
        lines.append(
            f"  {point.threshold!s:>9}{point.coverage!s:>10}"
            f"{point.accuracy_when_answered!s:>8}{point.queued:>8}"
        )
    return "\n".join(lines)


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)

    if not args.predictions.exists():
        print(f"{args.predictions} does not exist.")
        print("Produce one with: uv run python run_evaluation.py --live --save-predictions")
        return 1

    dev_header, dev_predictions = _llm_predictions(args.predictions)
    if dev_header.get("split") != "development":
        print(
            f"{args.predictions}: split is {dev_header.get('split')!r}, not "
            "'development'; refusing to choose a threshold on anything else"
        )
        return 1

    points = sweep(dev_predictions, list(DEFAULT_THRESHOLDS))
    chosen = choose_threshold(dev_predictions, list(DEFAULT_THRESHOLDS))
    chosen_point = next(point for point in points if point.threshold == chosen)

    print(f"DEVELOPMENT CURVE  ({len(dev_predictions)} rows, {args.predictions})")
    print(_render_curve(points))
    print(
        f"\nchosen threshold: {chosen}  (coverage {chosen_point.coverage}, "
        f"accuracy when answered {chosen_point.accuracy_when_answered}, "
        f"queued {chosen_point.queued})"
    )
    print("chosen by Youden's J over the development curve above; the holdout is not consulted.")

    result: dict[str, object] = {
        "schema": "review-threshold/1",
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "distribution_caveat": (
            "This curve was swept over LLM predictions made with each row's real "
            "account_type and no rule tier ahead of the model, while the deployed "
            "categorise.py pipeline runs the rule tier unfitted and sends every row "
            'to the model with account_type="unknown" instead, so this curve does '
            "not describe the input distribution the deployed pipeline actually "
            "produces; the chosen threshold below is kept as-is rather than "
            "re-picked against it, since it was chosen once, on the development "
            "split, and re-picking now would void that."
        ),
        "development": {
            "source": args.predictions.as_posix(),
            "split": dev_header.get("split"),
            "rows": len(dev_predictions),
            "curve": [point.to_json() for point in points],
            "chosen_threshold": str(chosen),
            "chosen_by": (
                "Youden's J statistic, maximised over the curve above: the "
                "threshold that best separates rows the model got right from "
                "rows it got wrong, using only this development split. J "
                "weighs a needless queue entry and a silently wrong accept as "
                "equally costly; this threshold governs a human review queue "
                "where they may not be."
            ),
            "note": (
                "Chosen on the development split alone. The benchmark below is "
                "measured once, after this choice, and never used to make it."
            ),
        },
    }

    holdout_path = (
        args.holdout if args.holdout is not None else _derive_holdout_path(args.predictions)
    )
    if not holdout_path.exists():
        if args.json is not None:
            # Writing a JSON artifact under `--json` without this key is worse
            # than not writing one: a reader would have no way to tell "not
            # measured yet" from "measured and the file just omits it", and
            # the whole point of this file is that the holdout number is
            # recorded, not merely computed once and mentioned in a chat log.
            print(
                f"\n{holdout_path} does not exist; refusing to write an artifact "
                "that silently omits the once-only holdout measurement. Pass "
                "--holdout to point at the right file, or drop --json to only "
                "print the development curve."
            )
            return 1
        print(f"\n{holdout_path} does not exist; skipping the holdout measurement.")
    else:
        holdout_header, holdout_predictions = _llm_predictions(holdout_path)
        if holdout_header.get("split") != "holdout":
            # The mirror of the development check above. Without it, a
            # mistyped or misdirected --holdout path could write its numbers
            # into the artifact's `holdout` key while actually being
            # development data (or anything else) - silently mislabelling a
            # section whose entire value is that a reader can trust which
            # split produced it.
            print(
                f"\n{holdout_path}: split is {holdout_header.get('split')!r}, "
                "not 'holdout'; refusing to label these numbers as the frozen "
                "benchmark when they are not"
            )
            return 1
        holdout_point = sweep(holdout_predictions, [chosen])[0]
        baseline_point = sweep(holdout_predictions, [Decimal("0")])[0]
        print(
            f"\nHOLDOUT  ({len(holdout_predictions)} rows, {holdout_path}), "
            f"measured once at threshold {chosen}"
        )
        print(
            f"  coverage {holdout_point.coverage}  "
            f"accuracy when answered {holdout_point.accuracy_when_answered}  "
            f"queued {holdout_point.queued}  "
            f"(unfiltered accuracy {baseline_point.accuracy_when_answered})"
        )
        result["holdout"] = {
            "source": holdout_path.as_posix(),
            "split": holdout_header.get("split"),
            "rows": len(holdout_predictions),
            "measured_at_threshold": str(chosen),
            "coverage": str(holdout_point.coverage),
            "accuracy_when_answered": str(holdout_point.accuracy_when_answered),
            "queued": holdout_point.queued,
            "unfiltered_accuracy": str(baseline_point.accuracy_when_answered),
            "note": (
                "Measured once, after the threshold above was chosen on "
                "development. This split was never swept and never chose the "
                "number it is measuring."
            ),
        }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"\nartifact -> {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
