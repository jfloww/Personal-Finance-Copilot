"""Turn a run's recorded predictions into a failure analysis worth acting on.

    uv run python analyse_failures.py data/eval/predictions/development-categorise-v1.jsonl

An aggregate score says a system is wrong 34% of the time. It does not say
whether those are one broken category, a systematic confusion between two
neighbouring ones, or rows a human annotator also got wrong - and those three
call for completely different fixes. This reads the row-level predictions and
sorts the failures into buckets that each imply a different action.

**Two outputs, deliberately different.** The console report is for the person
who owns the data: it can name merchants, because they already know them. The
JSON is for publication and carries counts, label pairs, and structural facts
only. Nothing that reaches the JSON can name a merchant, an amount, a date, or
a transaction.

Structural facts are the compromise that keeps the published half useful. "The
model was wrong on 9 rows whose description was under 15 characters" is a real
finding about insufficient context, and it says nothing about where anyone
shops.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Final

from offerdelta.evaluation.predictions import RecordedPrediction, read_predictions

DEFAULT_DATASET: Final = Path("data/eval/transactions.csv")

#: Above this, the model said it was sure. A wrong answer here is worse than a
#: wrong answer at 0.4: it is the one a reviewer would not think to check.
CONFIDENT: Final = Decimal("0.80")

#: Below this, the model was already unsure. A wrong answer here is one where
#: abstaining would have cost coverage and saved an error.
UNSURE: Final = Decimal("0.60")

#: Descriptions shorter than this carry little more than a merchant token. Where
#: the model is wrong on these, more prompt engineering is not the fix - the row
#: genuinely does not say enough.
THIN_DESCRIPTION: Final = 15


@dataclass(frozen=True)
class RowFacts:
    """What can be said about a row without quoting it."""

    transaction_id: str
    description_length: int
    word_count: int
    digit_share: float
    merchant_label_count: int

    @property
    def thin(self) -> bool:
        return self.description_length < THIN_DESCRIPTION

    @property
    def polysemous_merchant(self) -> bool:
        """This merchant carries more than one gold label in the dataset.

        Not a model failure. A merchant that is genuinely two things cannot be
        resolved from the description alone, by a model or by anyone.
        """
        return self.merchant_label_count > 1


def load_facts(path: Path) -> dict[str, RowFacts]:
    """Structural facts per row, read from the labelled CSV and never quoted."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))

    labels_per_merchant: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        merchant = _merchant_key(row["description"])
        labels_per_merchant[merchant].add(row["final_label"] or row["annotator_a_label"])

    facts = {}
    for row in rows:
        description = row["description"]
        digits = sum(1 for c in description if c.isdigit())
        facts[row["transaction_id"]] = RowFacts(
            transaction_id=row["transaction_id"],
            description_length=len(description.strip()),
            word_count=len(description.split()),
            digit_share=round(digits / len(description), 3) if description else 0.0,
            merchant_label_count=len(labels_per_merchant[_merchant_key(description)]),
        )
    return facts


def _merchant_key(description: str) -> str:
    """A crude leading-token merchant key, only ever used to group rows."""
    return " ".join(description.upper().split()[:2])


def redact(description: str) -> str:
    """A description's shape, with its content removed.

    Letters become X, digits become 9, punctuation and spacing survive. Enough
    to tell `XXXXXX #999` from `XXXXXXX XXXXXXXX XX 99999` - which is the
    difference between a merchant with a store number and a full payment
    memo - and not enough to name anybody.
    """
    out = []
    for char in description.strip():
        if char.isdigit():
            out.append("9")
        elif char.isalpha():
            out.append("X")
        else:
            out.append(char)
    return "".join(out)


@dataclass(frozen=True)
class Bucket:
    """One failure mode: how many rows, and which label pairs."""

    name: str
    what_it_means: str
    rows: tuple[RecordedPrediction, ...]

    def confusions(self, limit: int = 5) -> list[tuple[str, int]]:
        pairs = Counter(f"{r.gold} -> {r.predicted}" for r in self.rows if not r.abstained)
        return pairs.most_common(limit)

    def to_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "what_it_means": self.what_it_means,
            "rows": len(self.rows),
            "top_confusions": [{"pair": pair, "count": count} for pair, count in self.confusions()],
        }


def bucket_failures(
    by_system: dict[str, dict[str, RecordedPrediction]],
    facts: dict[str, RowFacts],
    llm_name: str,
    rules_name: str,
) -> list[Bucket]:
    """Sort the model's failures into modes that each imply a different fix."""
    llm = by_system[llm_name]
    rules = by_system.get(rules_name, {})

    wrong = [p for p in llm.values() if not p.correct and not p.abstained]

    def has(
        predicate: Callable[[RecordedPrediction], bool], rows: list[RecordedPrediction]
    ) -> tuple[RecordedPrediction, ...]:
        return tuple(r for r in rows if predicate(r))

    return [
        Bucket(
            "confident_but_wrong",
            "The model was sure and wrong. The failure a reviewer would not "
            "think to check, and the one that most needs the confidence signal "
            "to mean something.",
            has(lambda r: r.confidence >= CONFIDENT, wrong),
        ),
        Bucket(
            "abstention_would_have_been_better",
            "The model was already unsure and answered anyway. Abstaining would "
            "have cost coverage and saved an error, which is the trade the "
            "routing threshold exists to make.",
            has(lambda r: r.confidence < UNSURE, wrong),
        ),
        Bucket(
            "rules_right_model_wrong",
            "The deterministic baseline got these and the model did not. Every "
            "row here is an argument for the hybrid, and against replacing the "
            "rules with the model.",
            has(
                lambda r: (
                    r.transaction_id in rules
                    and rules[r.transaction_id].correct
                    and not rules[r.transaction_id].abstained
                ),
                wrong,
            ),
        ),
        Bucket(
            "model_right_rules_wrong",
            "What the model is actually buying: rows the baseline could not "
            "answer or answered wrongly. Counted over correct model rows, not "
            "failures.",
            tuple(
                p
                for p in llm.values()
                if p.correct
                and (p.transaction_id not in rules or not rules[p.transaction_id].correct)
            ),
        ),
        Bucket(
            "thin_description",
            f"Wrong on rows whose description is under {THIN_DESCRIPTION} "
            "characters. Not a prompt problem: the row does not carry enough to "
            "decide, and a better prompt cannot add context that is absent.",
            has(lambda r: facts[r.transaction_id].thin, wrong),
        ),
        Bucket(
            "polysemous_merchant",
            "Wrong on a merchant that carries more than one gold label in this "
            "dataset. Genuinely two things; no description-only system resolves "
            "these reliably.",
            has(lambda r: facts[r.transaction_id].polysemous_merchant, wrong),
        ),
        Bucket(
            "annotators_disagreed_too",
            "Wrong on rows where the two human annotators also disagreed. The "
            "model is failing where people find it hard, which bounds how much "
            "of this is fixable by prompting.",
            has(lambda r: r.annotators_agreed is False, wrong),
        ),
        Bucket(
            "abstained",
            "Declined to answer. Not an error, but it is lost coverage and it "
            "belongs in the same picture.",
            tuple(p for p in llm.values() if p.abstained),
        ),
    ]


def summarise(
    header: dict[str, object],
    by_system: dict[str, dict[str, RecordedPrediction]],
    buckets: list[Bucket],
    llm_name: str,
) -> dict[str, object]:
    """The publishable half: counts, label pairs, and nothing else."""
    llm = by_system[llm_name]
    answered = [p for p in llm.values() if not p.abstained]
    wrong = [p for p in answered if not p.correct]

    return {
        "split": header.get("split", "unknown"),
        "prompt_version": header.get("prompt_version"),
        "model": header.get("model"),
        "rows": len(llm),
        "answered": len(answered),
        "correct": sum(1 for p in answered if p.correct),
        "wrong": len(wrong),
        "abstained": len(llm) - len(answered),
        "mean_confidence_when_right": _mean([p.confidence for p in answered if p.correct]),
        "mean_confidence_when_wrong": _mean([p.confidence for p in wrong]),
        "failure_modes": [bucket.to_json() for bucket in buckets],
        "note": (
            "Counts and taxonomy label pairs only. No merchant, amount, date, "
            "or transaction appears here or can be recovered from it."
        ),
    }


def _mean(values: list[Decimal]) -> float | None:
    if not values:
        return None
    return round(float(sum(values) / len(values)), 4)


def render(
    header: dict[str, object],
    buckets: list[Bucket],
    facts: dict[str, RowFacts],
    descriptions: dict[str, str],
    examples: int,
) -> str:
    """The console report. Redacted, so it is safe to paste into an issue."""
    lines = [
        "FAILURE ANALYSIS",
        f"  split {header.get('split', 'unknown')}, prompt {header.get('prompt_version')}, "
        f"model {header.get('model')}",
        f"  dataset {header.get('dataset_version')} checksum "
        f"{str(header.get('dataset_checksum', ''))[:16]}...",
        "",
    ]

    for bucket in buckets:
        lines.append(f"--- {bucket.name}  ({len(bucket.rows)} rows)")
        lines.append(f"    {bucket.what_it_means}")
        if confusions := bucket.confusions():
            lines.append("    most frequent:")
            for pair, count in confusions:
                lines.append(f"      {pair}: {count}")
        for row in bucket.rows[:examples]:
            fact = facts[row.transaction_id]
            shape = redact(descriptions.get(row.transaction_id, ""))
            lines.append(
                f"      e.g. {shape[:44]:<44} conf {row.confidence} len {fact.description_length}"
            )
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="analyse_failures.py", description=__doc__)
    parser.add_argument("predictions", type=Path, help="a JSONL file from --save-predictions")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json", type=Path, default=None, help="write the sanitized summary here")
    parser.add_argument("--examples", type=int, default=3, help="redacted examples per bucket")
    args = parser.parse_args(argv)

    if not args.predictions.exists():
        print(f"{args.predictions} does not exist.")
        print("Produce one with: uv run python run_evaluation.py --live --save-predictions")
        return 1

    header, recorded = read_predictions(args.predictions)

    by_system: dict[str, dict[str, RecordedPrediction]] = defaultdict(dict)
    for row in recorded:
        by_system[row.system][row.transaction_id] = row

    llm_name = next((name for name in by_system if name.startswith("llm:")), None)
    if llm_name is None:
        print(f"no llm system in {args.predictions}; nothing to analyse")
        return 1
    rules_name = next((name for name in by_system if name == "rules"), "rules")

    facts = load_facts(args.dataset)
    missing = {r.transaction_id for r in recorded} - set(facts)
    if missing:
        print(f"{len(missing)} predicted rows are absent from {args.dataset}.")
        print("The predictions and the dataset disagree; refusing to analyse.")
        return 1

    descriptions = {
        row["transaction_id"]: row["description"]
        for row in csv.DictReader(args.dataset.open(encoding="utf-8"))
    }

    buckets = bucket_failures(by_system, facts, llm_name, rules_name)
    print(render(header, buckets, facts, descriptions, args.examples))

    if args.json:
        summary = summarise(header, by_system, buckets, llm_name)
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"sanitized summary -> {args.json}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
