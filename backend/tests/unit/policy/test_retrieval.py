from __future__ import annotations

from offerdelta.policy.corpus import load_policy_corpus
from offerdelta.policy.evaluation import CASES, evaluate_policy_retrieval
from offerdelta.policy.retrieval import BM25PolicyRetriever


def test_versioned_corpus_has_unique_citable_sections() -> None:
    chunks = load_policy_corpus()
    assert len(chunks) == 7
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert {chunk.corpus_version for chunk in chunks} == {"synthetic-finops-2026-09-01"}
    assert all(chunk.source.endswith(f"#{chunk.section}") for chunk in chunks)


def test_bm25_search_is_deterministic_and_abstains_without_overlap() -> None:
    retriever = BM25PolicyRetriever(load_policy_corpus())
    first = retriever.search("itemized receipt purchase above 3000", limit=3)
    second = retriever.search("itemized receipt purchase above 3000", limit=3)
    assert first == second
    assert first[0].chunk.chunk_id == "POL-PROC-04@demo-2#2.1"
    assert first[0].score > 0
    assert retriever.search("parental leave allowance", limit=3) == ()


def test_committed_retrieval_set_is_a_passing_regression_not_a_live_benchmark() -> None:
    report = evaluate_policy_retrieval()
    metrics = report["metrics"]
    assert isinstance(metrics, dict)
    assert len(CASES) == 8
    assert report["interpretation"] == "authored synthetic smoke set; not live-traffic accuracy"
    assert metrics == {
        "recall_at_1": "1.0000",
        "recall_at_3": "1.0000",
        "mean_reciprocal_rank": "1.0000",
        "unanswerable_abstention_accuracy": "1.0000",
    }
