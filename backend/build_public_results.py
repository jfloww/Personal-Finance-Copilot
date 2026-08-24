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
    latency = re.search(r"latency: p50 (\d+)ms\s+p95 (\d+)ms", body)
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
        p50_latency_ms=int(latency.group(1)) if latency else None,
        p95_latency_ms=int(latency.group(2)) if latency else None,
    )


#: A per-label row: two leading spaces, a label, then P, R, F1 and support.
_ROW: Final = re.compile(r"\s{2}([A-Z][A-Z_]+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\d+)\s*$")

LLM_HEADER: Final = "--- llm:claude-haiku-4-5"
RULES_HEADER: Final = "--- rules"


@dataclass(frozen=True)
class LabelScore:
    """One row of a per-label table. A taxonomy name, never a merchant."""

    f1: float
    support: int


@dataclass(frozen=True)
class LabelMovement:
    """What one label did between the two prompts."""

    label: str
    support: int
    v1: float | None
    v2: float | None
    delta: float | None


@dataclass(frozen=True)
class PublicResults:
    """The file to publish, plus the systems in it for the console summary."""

    systems: tuple[SystemResult, ...]
    payload: dict[str, object]


def per_label(text: str, header: str) -> dict[str, LabelScore]:
    """Every label the scorer listed, including ones it predicted but never hit.

    Zero-support labels are kept deliberately. They are not noise: a label a
    system predicts and is never right about still occupies a slot in the macro
    average, which is exactly why the two prompts are not comparable on macro
    F1 alone.
    """
    body = _section(text, header)
    scores: dict[str, LabelScore] = {}
    for line in body.splitlines():
        if found := _ROW.match(line):
            scores[found.group(1)] = LabelScore(
                f1=float(found.group(4)), support=int(found.group(5))
            )
    return scores


def macro_denominator(scores: dict[str, LabelScore]) -> dict[str, object]:
    """What the macro average was divided by, and what it summed.

    `metrics.score` builds its label set from the union of labels the benchmark
    contains and labels the system predicted, so a system that stops emitting a
    never-correct label shrinks its own denominator. Published next to the mean
    because the mean on its own does not show it.
    """
    return {
        "labels_in_average": len(scores),
        "labels_never_correct": sorted(name for name, s in scores.items() if s.f1 == 0.0),
        "sum_of_f1": round(sum(s.f1 for s in scores.values()), 4),
    }


def compare(before: dict[str, LabelScore], after: dict[str, LabelScore]) -> list[LabelMovement]:
    """Per-label movement, with `None` where a label exists on one side only."""
    return [
        LabelMovement(
            label=label,
            support=(before.get(label) or after[label]).support,
            v1=before[label].f1 if label in before else None,
            v2=after[label].f1 if label in after else None,
            delta=(
                round(after[label].f1 - before[label].f1, 4)
                if label in before and label in after
                else None
            ),
        )
        for label in sorted(set(before) | set(after))
    ]


def build() -> PublicResults:
    v1_text = (RUNS / "2026-08-24-haiku-4-5-prompt-v1.txt").read_text(encoding="utf-8")
    v2_text = (RUNS / "2026-08-24-haiku-4-5-prompt-v2.txt").read_text(encoding="utf-8")

    rules = parse_system(v1_text, RULES_HEADER, "rule baseline (no model)")
    v1 = parse_system(v1_text, LLM_HEADER, "claude-haiku-4-5, prompt categorise/v1")
    v2 = parse_system(v2_text, LLM_HEADER, "claude-haiku-4-5, prompt categorise/v2")

    before = per_label(v1_text, LLM_HEADER)
    after = per_label(v2_text, LLM_HEADER)
    movement = compare(before, after)
    regressed = [m.label for m in movement if m.delta is not None and m.delta < 0]
    improved = [m.label for m in movement if m.delta is not None and m.delta > 0]

    if not (v1.cost_usd and v2.cost_usd and v1.p95_latency_ms and v2.p95_latency_ms):
        raise SystemExit("archived runs are missing cost or latency; refusing to publish a gap")

    cost_gap = round(float(v2.cost_usd - v1.cost_usd) / float(v2.cost_usd) * 100, 1)
    p95_gap = v2.p95_latency_ms - v1.p95_latency_ms
    sum_before = round(sum(s.f1 for s in before.values()), 4)
    sum_after = round(sum(s.f1 for s in after.values()), 4)

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
                "The v1 results informed the design of v2, so these rows have "
                "already influenced a decision. Calling them a held-out test set "
                "would claim more than they can still support."
            ),
            "split": "merchant-disjoint, assigned by salted hash rather than shuffled",
            "merchants_on_both_sides": 0,
            "model": "claude-haiku-4-5",
            "held_constant_across_prompts": [
                "model",
                "split",
                "taxonomy",
                "scoring",
                "inference settings",
            ],
        },
        "systems": [asdict(rules), asdict(v1), asdict(v2)],
        "prompt_experiment": {
            "status": "rejected experiment",
            "hypothesis": (
                "LIVING_OTHER was absorbing rows that belonged elsewhere: it was "
                "predicted 18 times against a true support of 1. Naming it a last "
                "resort rather than a default should reduce that."
            ),
            "controlled_change": (
                "One rule added to the system prompt - five lines added, none "
                "removed, nothing else touched. Same 135 rows, same model, same "
                "split, same taxonomy, same scoring."
            ),
            "outcome": {
                "target_unmoved": (
                    "LIVING_OTHER F1 was 0.1053 before and 0.1053 after. The rule "
                    "did not touch the failure it was written for."
                ),
                "categories_regressed": regressed,
                "categories_improved": improved,
                "v1_is_cheaper_by_pct": cost_gap,
                "v1_p95_latency_lower_by_ms": p95_gap,
            },
            "why_macro_f1_rose_anyway": (
                "Macro F1 divides by the number of labels in play, and that set is "
                "the union of labels the benchmark contains and labels the system "
                "predicts. v2 stopped predicting LIVING_GYM, which is never correct "
                f"here, so its denominator fell from {len(before)} to {len(after)}. "
                f"The summed per-label F1 dropped, from {sum_before} to {sum_after} "
                "- the mean rose because the divisor shrank, not because the system "
                "got better at any category."
            ),
            "macro_denominator": {
                "v1": macro_denominator(before),
                "v2": macro_denominator(after),
            },
            "decision": "rejected",
            "selected_prompt": "categorise/v1",
            "reasoning": (
                "Macro F1 is not a misleading metric - it is the one that stops a "
                "model hiding a rare class it ignores, which is why it is the "
                "headline here. Deciding on macro F1 alone is what misleads. It "
                "moved +0.0120 while accuracy, weighted F1, cost, and tail latency "
                "all moved the wrong way, four categories regressed against two "
                "improvements, and the rule never touched the failure it targeted. "
                "Read together, those say reject."
            ),
            "caveat": (
                "A single A/B with no repeated runs, so run-to-run variance is not "
                "separated from the effect of the prompt."
            ),
            "per_label_movement": [asdict(m) for m in movement],
        },
        "backlog": [
            "LIVING_TRAVEL is still mistaken for COMMUTE_TRANSIT_FARE more often "
            "than anything else. A v3 on that boundary is deferred, not dropped.",
            "An OpenAI adapter, to score a second vendor's model on this same "
            "frozen benchmark with the labels, split, and scoring unchanged.",
        ],
    }
    return PublicResults(systems=(rules, v1, v2), payload=payload)


def main() -> int:
    results = build()
    rendered = json.dumps(results.payload, indent=2) + "\n"
    for destination in (OUT, SERVED):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")

    print(f"{OUT}: {len(results.systems)} systems, aggregate only")
    print(f"{SERVED}: served copy, identical bytes")
    for system in results.systems:
        print(f"  {system.name:<42} macro F1 {system.macro_f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
