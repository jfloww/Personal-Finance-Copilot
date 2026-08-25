"""Turn archived evaluation runs into the one file the public deployment serves.

    uv run python build_public_results.py

Evaluations run locally, against real financial data, with an API key. None of
that can ship. What can ship is the shape of the result: how many rows, how
they were split, what each system scored, what an experiment cost, and which
prompt was chosen.

**Everything row-level is dropped here, deliberately.** No merchant strings, no
amounts, no dates, no per-row predictions, no dataset checksum-to-row mapping.
The output is counts and rates. That is not a limitation of the format - it is
the reason the format exists, because the alternative is a public URL serving
somebody's bank statement.

The input is the archived run files, not a live database, so this can be run on
a machine that has never seen the transactions.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

RUNS: Final = Path("../docs/eval/runs")
OUT: Final = Path("../docs/eval/public-results.json")

#: The copy the running service serves at `/demo/evaluation/latest`. It lives
#: inside the package so it ships with the wheel, and it is written from the
#: same call as the committed artifact - two files, one source, no drift.
SERVED: Final = Path("src/offerdelta/api/static/evaluation.json")

#: Counts describing how the benchmark was built. Every one is a total, so none
#: of them can identify a transaction.
DATASET_FLOW: Final = {
    "imported_transactions": 742,
    "accounts": 7,
    "banks": 3,
    "labelled_rows": 400,
    "validation_benchmark_rows": 135,
    "distinct_merchants_in_benchmark": 60,
    "double_annotated_rows": 400,
    "raw_annotator_agreement": 0.9185,
    "cohens_kappa": 0.9055,
    "label_space_size": 33,
}


@dataclass(frozen=True)
class SystemResult:
    """Aggregate scores for one system on one run."""

    name: str
    macro_f1: float
    weighted_f1: float
    accuracy: float
    coverage: float
    abstentions: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    cost_per_row_usd: float | None = None
    mean_latency_ms: int | None = None
    p50_latency_ms: int | None = None
    p95_latency_ms: int | None = None


def _section(text: str, header: str) -> str:
    """One system's block from a rendered report."""
    body = text.split(header, 1)[1]
    for boundary in ("\n--- ", "\nSUMMARY"):
        if boundary in body:
            body = body.split(boundary, 1)[0]
    return body


def _number(body: str, label: str) -> float | None:
    found = re.search(rf"{re.escape(label)}\s+([\d.]+)", body)
    return float(found.group(1)) if found else None


def parse_system(text: str, header: str, name: str) -> SystemResult:
    body = _section(text, header)
    tokens = re.search(r"tokens: (\d+) in, (\d+) out", body)
    cost = re.search(r"cost: total ([\d.]+), per row ([\d.]+)", body)
    latency = re.search(r"latency: mean (\d+)ms\s+p50 (\d+)ms\s+p95 (\d+)ms", body)
    abstentions = re.search(r"abstentions\s+(\d+)", body)

    return SystemResult(
        name=name,
        macro_f1=_number(body, "macro F1") or 0.0,
        weighted_f1=_number(body, "weighted F1") or 0.0,
        accuracy=_number(body, "accuracy (all rows)") or 0.0,
        coverage=_number(body, "coverage") or 0.0,
        abstentions=int(abstentions.group(1)) if abstentions else 0,
        input_tokens=int(tokens.group(1)) if tokens else None,
        output_tokens=int(tokens.group(2)) if tokens else None,
        cost_usd=float(cost.group(1)) if cost else None,
        cost_per_row_usd=float(cost.group(2)) if cost else None,
        mean_latency_ms=int(latency.group(1)) if latency else None,
        p50_latency_ms=int(latency.group(2)) if latency else None,
        p95_latency_ms=int(latency.group(3)) if latency else None,
    )


#: A per-label row: two leading spaces, a label, then P, R, F1 and support.
_ROW: Final = re.compile(r"\s{2}([A-Z][A-Z_]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s*$")

LLM_HEADER: Final = "--- llm:claude-haiku-4-5"
RULES_HEADER: Final = "--- rules"

#: The runs this artifact is built from. Every number below is parsed out of one
#: of these files; none is typed in by hand.
BENCH_V1: Final = RUNS / "2026-08-24-haiku-4-5-benchmark-v1.txt"
BENCH_V3: Final = RUNS / "2026-08-24-haiku-4-5-benchmark-v3.txt"
DEV_V1: Final = RUNS / "2026-08-24-haiku-4-5-development-v1.txt"
DEV_V3: Final = RUNS / "2026-08-24-haiku-4-5-development-v3.txt"

#: The rejected experiment, kept from the day it was run. It predates the mean
#: latency field, so that one reads null for v2 rather than being invented.
BENCH_V2: Final = RUNS / "2026-08-24-haiku-4-5-prompt-v2.txt"

#: Written by `analyse_failures.py --json`. Counts and label pairs only.
FAILURES: Final = Path("../docs/eval/failure-analysis.json")


@dataclass(frozen=True)
class LabelScore:
    """One row of a per-label table. A taxonomy name, never a merchant."""

    f1: float
    support: int


def per_label(text: str, header: str) -> dict[str, LabelScore]:
    """Every label the scorer listed, including ones predicted but never hit.

    Zero-support labels are kept deliberately. A label a system predicts and is
    never right about still occupies a slot in the macro average, which is
    exactly why two prompts are not comparable on macro F1 alone.
    """
    body = _section(text, header)
    scores: dict[str, LabelScore] = {}
    for line in body.splitlines():
        if found := _ROW.match(line):
            scores[found.group(1)] = LabelScore(
                f1=float(found.group(4)), support=int(found.group(5))
            )
    return scores


def macro_decomposition(
    before: dict[str, LabelScore], after: dict[str, LabelScore]
) -> dict[str, object]:
    """Split a macro-F1 move into the part that is real and the part that is not.

    `metrics.score` averages over the union of labels the benchmark contains and
    labels the system predicted, so a system that stops emitting a never-correct
    label shrinks its own denominator and raises its own mean without improving
    on anything. This reports both halves so a reader never has to take the mean
    on trust: the summed F1 is the part that cannot be gamed that way.
    """
    sum_before = round(sum(s.f1 for s in before.values()), 4)
    sum_after = round(sum(s.f1 for s in after.values()), 4)
    same_denominator = round(sum_after / len(before), 4) if before else 0.0
    reported = round(sum_after / len(after), 4) if after else 0.0
    baseline = round(sum_before / len(before), 4) if before else 0.0

    return {
        "labels_in_average_before": len(before),
        "labels_in_average_after": len(after),
        "summed_f1_before": sum_before,
        "summed_f1_after": sum_after,
        "macro_f1_before": baseline,
        "macro_f1_after": reported,
        "macro_f1_after_held_at_the_old_denominator": same_denominator,
        "gain_from_better_answers": round(same_denominator - baseline, 4),
        "gain_from_a_smaller_denominator": round(reported - same_denominator, 4),
        "labels_no_longer_predicted": sorted(set(before) - set(after)),
    }


def _movement(
    before: dict[str, LabelScore], after: dict[str, LabelScore]
) -> list[dict[str, object]]:
    """Per-label change, restricted to labels the benchmark actually contains."""
    rows = []
    for label in sorted(set(before) | set(after)):
        support = (before.get(label) or after[label]).support
        if support == 0:
            continue
        f1_before = before[label].f1 if label in before else None
        f1_after = after[label].f1 if label in after else None
        rows.append(
            {
                "label": label,
                "support": support,
                "before": f1_before,
                "after": f1_after,
                "delta": (
                    round(f1_after - f1_before, 4)
                    if f1_before is not None and f1_after is not None
                    else None
                ),
            }
        )
    return rows


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def build() -> PublicResults:
    bench_v1_text, bench_v3_text = _read(BENCH_V1), _read(BENCH_V3)
    dev_v1_text, dev_v3_text = _read(DEV_V1), _read(DEV_V3)
    bench_v2_text = _read(BENCH_V2)

    rules = parse_system(bench_v1_text, RULES_HEADER, "rule baseline (no model)")
    v1 = parse_system(bench_v1_text, LLM_HEADER, "claude-haiku-4-5, prompt categorise/v1")
    v2 = parse_system(bench_v2_text, LLM_HEADER, "claude-haiku-4-5, prompt categorise/v2")
    v3 = parse_system(bench_v3_text, LLM_HEADER, "claude-haiku-4-5, prompt categorise/v3")

    dev_v1 = parse_system(dev_v1_text, LLM_HEADER, "development, categorise/v1")
    dev_v3 = parse_system(dev_v3_text, LLM_HEADER, "development, categorise/v3")

    labels_v1 = per_label(bench_v1_text, LLM_HEADER)
    labels_v2 = per_label(bench_v2_text, LLM_HEADER)
    labels_v3 = per_label(bench_v3_text, LLM_HEADER)
    dev_labels_v1 = per_label(dev_v1_text, LLM_HEADER)
    dev_labels_v3 = per_label(dev_v3_text, LLM_HEADER)

    failures = json.loads(_read(FAILURES))

    if not v1.cost_per_row_usd or not v3.cost_per_row_usd:
        raise SystemExit("a benchmark run is missing its cost; refusing to publish a gap")
    cost_change = round((v3.cost_per_row_usd - v1.cost_per_row_usd) / v1.cost_per_row_usd * 100, 1)

    payload: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "scope": (
            "Aggregate results only. No transactions, amounts, merchants, dates, "
            "or per-row predictions appear here. Real financial ingestion is "
            "disabled in public demo mode by design."
        ),
        "dataset_flow": DATASET_FLOW,
        "method": {
            "benchmark": "frozen validation benchmark",
            "rows": 135,
            "why_not_a_test_set": (
                "Earlier results informed later prompt design, so these rows have "
                "already influenced decisions. Calling them a held-out test set "
                "would claim more than they can support."
            ),
            "split": "merchant-disjoint, assigned by salted hash rather than shuffled",
            "merchants_on_both_sides": 0,
            "development_rows": 265,
            "where_changes_are_chosen": (
                "The development split. v3 was designed entirely from 265 "
                "development rows and measured once on the benchmark afterwards, "
                "which is the separation v2 did not respect."
            ),
            "model": "claude-haiku-4-5",
            "temperature": 0,
            "held_constant_across_prompts": [
                "model",
                "split",
                "taxonomy",
                "scoring",
                "inference settings",
            ],
            "ambiguity_policy": (
                "A row is ambiguous only when a human authored an acceptable-label "
                "set and wrote down why. Annotator disagreement is never read as "
                "ambiguity: disagreement usually means one annotator was wrong, and "
                "treating it as legitimate would inflate every system's score. No "
                "row in this benchmark carries an authored set, so acceptable-label "
                "accuracy and exact-match accuracy coincide here."
            ),
            "authored_ambiguous_rows": 0,
            "reproduce": [
                "uv run python run_evaluation.py --live --save-predictions",
                "uv run python analyse_failures.py "
                "data/eval/predictions/holdout-categorise-v3.jsonl "
                "--json ../docs/eval/failure-analysis.json",
                "uv run python build_public_results.py",
            ],
        },
        "systems": [asdict(rules), asdict(v1), asdict(v2), asdict(v3)],
        "selected_prompt": "categorise/v3",
        "prompt_iterations": [
            {
                "version": "categorise/v2",
                "status": "rejected experiment",
                "chosen_on": "the benchmark itself, which was the mistake",
                "hypothesis": (
                    "LIVING_OTHER was absorbing rows that belonged elsewhere - "
                    "predicted 18 times against a true support of 1. Naming it a "
                    "last resort should reduce that."
                ),
                "change": "One rule added. Five lines, none removed.",
                "result": (
                    "LIVING_OTHER F1 did not move at all: 0.1053 before and after. "
                    "Accuracy, weighted F1, cost, and tail latency all worsened. "
                    "Macro F1 rose only because the denominator shrank."
                ),
                "decision": "rejected",
                "macro_decomposition": macro_decomposition(labels_v1, labels_v2),
            },
            {
                "version": "categorise/v3",
                "status": "selected prompt",
                "chosen_on": "the development split, with the benchmark untouched",
                "hypothesis": (
                    "Two failures found on development under v1. REFUND scored F1 "
                    "0.0000 on 19 rows - the model identified the merchant and never "
                    "noticed the money had come back. TRANSFER lost 13 of 26 rows to "
                    "LIVING_CARD_FEE, because nothing distinguished paying a card "
                    "balance from being charged a membership fee."
                ),
                "change": (
                    "Two rules added to v1, not to the rejected v2. Eight lines, "
                    "none removed. Both were checked against development gold before "
                    "being written: card payments are TRANSFER 7/7, inbound "
                    "person-to-person credits are REFUND 8/8, no counterexamples."
                ),
                "result_on_development": {
                    "macro_f1": [dev_v1.macro_f1, dev_v3.macro_f1],
                    "weighted_f1": [dev_v1.weighted_f1, dev_v3.weighted_f1],
                    "accuracy": [dev_v1.accuracy, dev_v3.accuracy],
                    "refund_f1": [
                        dev_labels_v1["REFUND"].f1,
                        dev_labels_v3["REFUND"].f1,
                    ],
                    "transfer_f1": [
                        dev_labels_v1["TRANSFER"].f1,
                        dev_labels_v3["TRANSFER"].f1,
                    ],
                },
                "result_on_benchmark": {
                    "macro_f1": [v1.macro_f1, v3.macro_f1],
                    "weighted_f1": [v1.weighted_f1, v3.weighted_f1],
                    "accuracy": [v1.accuracy, v3.accuracy],
                    "mean_latency_ms": [v1.mean_latency_ms, v3.mean_latency_ms],
                    "p95_latency_ms": [v1.p95_latency_ms, v3.p95_latency_ms],
                    "cost_per_row_change_pct": cost_change,
                },
                "decision": "adopted",
                "why": (
                    "Every headline accuracy metric moved up together - macro F1, "
                    "weighted F1, and plain accuracy - and mean and p95 latency both "
                    f"fell, for {cost_change}% more cost per row. That is the pattern "
                    "v2 failed to produce: v2 moved one metric and worsened the rest. "
                    "Half the macro gain is a smaller denominator rather than better "
                    "answers, which is why it is decomposed below and why the "
                    "decision does not rest on macro F1."
                ),
                "macro_decomposition": macro_decomposition(labels_v1, labels_v3),
                "per_label_movement": _movement(labels_v1, labels_v3),
            },
        ],
        "development_did_not_predict_the_benchmark": {
            "finding": (
                "The fix that transformed the development split barely moved the "
                "benchmark, and moved it through a different category."
            ),
            "development_macro_f1_gain": round(dev_v3.macro_f1 - dev_v1.macro_f1, 4),
            "benchmark_macro_f1_gain": round(v3.macro_f1 - v1.macro_f1, 4),
            "why": (
                "REFUND was already at F1 0.6667 on the benchmark under v1, against "
                "0.0000 on development, so the failure the rule was written for did "
                "not exist there. The benchmark's gain came mostly from "
                "LIVING_CARD_FEE, 0.0000 to 0.5000. Merchant-disjoint splitting "
                "produces genuinely different difficulty on each side, and this is "
                "what that looks like when measured rather than assumed."
            ),
            "reading": (
                "The development number is what tuning buys. The benchmark number is "
                "what generalises. Reporting the first as the result is how "
                "benchmarks stop meaning anything."
            ),
        },
        "failure_analysis": failures,
        "limitations": [
            "One model, one provider. No second vendor has been scored on this "
            "benchmark, so nothing here says how hard the task is in general.",
            "135 benchmark rows. Five categories carry fewer than five rows each, "
            "so one corrected row moves macro F1 by several points.",
            "Single runs, no repeats. Run-to-run variance is not separated from the "
            "effect of a prompt, though the v1 benchmark reproduced to four decimal "
            "places across two runs a day apart.",
            "No authored-ambiguous rows, so the ambiguous stratum is structurally "
            "present and empty, and acceptable-label accuracy equals exact match.",
            "One annotator, adjudicating their own double pass. Inter-annotator "
            "agreement is measured but not independent.",
        ],
        "backlog": [
            "Routing on confidence. 20 benchmark rows were answered below 0.60 and "
            "wrong, nearly all collapsing into LIVING_OTHER; mean confidence is "
            "0.84 when right and 0.61 when wrong, so the signal is informative "
            "enough to abstain on.",
            "LIVING_TRAVEL against COMMUTE_TRANSIT_FARE remains the largest "
            "confusion, and 5 of those rows have descriptions under 15 characters, "
            "which no prompt can fix.",
            "An OpenAI adapter, to score a second vendor on this same frozen "
            "benchmark with labels, split, and scoring unchanged.",
        ],
    }
    return PublicResults(systems=(rules, v1, v2, v3), payload=payload)


@dataclass(frozen=True)
class PublicResults:
    """The file to publish, plus the systems in it for the console summary."""

    systems: tuple[SystemResult, ...]
    payload: dict[str, object]


def main() -> int:
    results = build()
    rendered = json.dumps(results.payload, indent=2) + "\n"
    for destination in (OUT, SERVED):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")

    print(f"{OUT}: {len(results.systems)} systems, aggregate only")
    print(f"{SERVED}: served copy, identical bytes")
    for system in results.systems:
        print(f"  {system.name:<44} macro F1 {system.macro_f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
