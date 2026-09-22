"""A tiny authored regression set for policy retrieval, not a production benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Final

from offerdelta.policy.retrieval import PolicyRetriever, default_policy_retriever

type EvaluationValue = str | int | bool | list[EvaluationValue] | dict[str, EvaluationValue] | None


@dataclass(frozen=True)
class RetrievalCase:
    case_id: str
    query: str
    relevant_chunk_id: str | None


CASES: Final[tuple[RetrievalCase, ...]] = (
    RetrievalCase(
        "high-value-receipt",
        "Does a $4,850 purchase need a receipt and human review?",
        "POL-PROC-04@demo-2#2.1",
    ),
    RetrievalCase(
        "duplicate-reversal",
        "Can we reverse two duplicate merchant charges posted minutes apart?",
        "POL-PROC-04@demo-2#2.2",
    ),
    RetrievalCase(
        "renewal-variance",
        "What evidence is needed when a software subscription renewal increased?",
        "POL-PROC-04@demo-2#3.1",
    ),
    RetrievalCase(
        "reimbursement-deadline",
        "How long does an employee have to submit a reimbursement?",
        "POL-EXP-02@demo-1#1.4",
    ),
    RetrievalCase(
        "travel-exception",
        "What must be attached for a hotel or airfare exception?",
        "POL-EXP-02@demo-1#2.3",
    ),
    RetrievalCase(
        "unmatched-close",
        "Who owns an unmatched bank transaction during month-end close?",
        "POL-CLOSE-03@demo-1#4.2",
    ),
    RetrievalCase(
        "evidence-retention",
        "How many years should reconciliation evidence be retained?",
        "POL-CLOSE-03@demo-1#5.1",
    ),
    RetrievalCase("out-of-domain", "What is the parental leave allowance?", None),
)


def _rate(numerator: Decimal, denominator: int) -> str:
    if denominator == 0:
        return "0.0000"
    return f"{numerator / Decimal(denominator):.4f}"


def evaluate_policy_retrieval(
    retriever: PolicyRetriever | None = None,
) -> dict[str, EvaluationValue]:
    """Measure ranking and abstention on the committed synthetic cases."""
    selected = retriever or default_policy_retriever()
    answerable = [case for case in CASES if case.relevant_chunk_id is not None]
    unanswerable = [case for case in CASES if case.relevant_chunk_id is None]
    top_one_hits = 0
    top_three_hits = 0
    reciprocal_rank = Decimal(0)
    correct_abstentions = 0
    rows: list[EvaluationValue] = []
    for case in CASES:
        matches = selected.search(case.query, limit=3)
        retrieved = [match.chunk.chunk_id for match in matches]
        retrieved_values: list[EvaluationValue] = list(retrieved)
        rank: int | None = None
        if case.relevant_chunk_id is None:
            correct_abstentions += int(not retrieved)
        elif case.relevant_chunk_id in retrieved:
            rank = retrieved.index(case.relevant_chunk_id) + 1
            top_three_hits += 1
            top_one_hits += int(rank == 1)
            reciprocal_rank += Decimal(1) / Decimal(rank)
        rows.append(
            {
                "case_id": case.case_id,
                "query": case.query,
                "relevant_chunk_id": case.relevant_chunk_id,
                "retrieved_chunk_ids": retrieved_values,
                "relevant_rank": rank,
                "passed": (not retrieved if case.relevant_chunk_id is None else rank == 1),
            }
        )
    return {
        "dataset": "synthetic-policy-retrieval-regression-v1",
        "interpretation": "authored synthetic smoke set; not live-traffic accuracy",
        "retrieval": "bm25_v1",
        "corpus_version": selected.corpus_version,
        "query_count": len(CASES),
        "answerable_queries": len(answerable),
        "unanswerable_queries": len(unanswerable),
        "metrics": {
            "recall_at_1": _rate(Decimal(top_one_hits), len(answerable)),
            "recall_at_3": _rate(Decimal(top_three_hits), len(answerable)),
            "mean_reciprocal_rank": _rate(reciprocal_rank, len(answerable)),
            "unanswerable_abstention_accuracy": _rate(
                Decimal(correct_abstentions), len(unanswerable)
            ),
        },
        "cases": rows,
    }
