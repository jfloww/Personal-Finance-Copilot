"""Synthetic investigations are reproducible, bounded, and genuinely read-only."""

from __future__ import annotations

from decimal import Decimal

import pytest

from offerdelta.agent.tools.operations import TRANSACTIONS, build_operations_registry
from offerdelta.application.queries.operations_demo import investigate_demo
from offerdelta.demo.operations_cases import load_synthetic_case, parse_synthetic_csv
from offerdelta.domain.common.errors import ValidationError


def test_spend_reconciles_and_duplicate_is_a_candidate() -> None:
    report = investigate_demo()
    summary = report["summary"]
    assert isinstance(summary, dict)
    assert summary["prior_spend"] == "12000.00"
    assert summary["total_spend"] == "24500.00"
    assert summary["change"] == "12500.00"
    drivers = summary["drivers"]
    assert isinstance(drivers, list)
    assert sum(Decimal(str(item["change"])) for item in drivers if isinstance(item, dict)) == (
        Decimal(str(summary["change"]))
    )
    assert summary["transaction_count"] == 8
    assert report["duplicate"] == {
        "transaction_ids": ["TX-9824-A1", "TX-9825-B2"],
        "merchant": "Amazon Web Services",
        "amount_each": "4850.00",
        "currency": "USD",
        "minutes_apart": 4,
        "assessment": "possible_duplicate_needs_review",
    }


def test_proposing_review_never_changes_the_ledger_or_review_queue() -> None:
    before = TRANSACTIONS
    registry = build_operations_registry()
    arguments = {"transaction_id": "TX-9825-B2", "reason": "possible_duplicate"}
    first = registry.call("propose_review_case", arguments)
    second = registry.call("propose_review_case", arguments)
    assert first == second
    assert first.ok
    assert first.payload["status"] == "proposal_only"
    assert first.payload["ledger_mutated"] is False
    assert first.payload["review_queue_mutated"] is False
    assert TRANSACTIONS is before


def test_policy_has_citation_and_unmatched_query_has_no_fabricated_citation() -> None:
    registry = build_operations_registry()
    hit = registry.call("retrieve_policy", {"query": "purchase receipt review"})
    assert hit.ok
    citations = hit.payload["citations"]
    assert isinstance(citations, list)
    assert isinstance(citations[0], dict)
    assert citations[0]["chunk_id"] == "POL-PROC-04@demo-2#2.1"
    assert citations[0]["source"] == "synthetic://policies/POL-PROC-04@demo-2#2.1"
    assert citations[0]["rank"] == 1
    assert Decimal(str(citations[0]["score"])) > 0
    assert hit.payload["retrieval"] == "bm25_v1"
    assert hit.payload["corpus_version"] == "synthetic-finops-2026-09-01"
    miss = registry.call("retrieve_policy", {"query": "unrelated"})
    assert miss.payload["citations"] == []


def test_policy_retrieval_ranks_distinct_sections_and_honours_limit() -> None:
    registry = build_operations_registry()
    duplicate = registry.call(
        "retrieve_policy", {"query": "reverse duplicate merchant charges", "limit": 1}
    )
    renewal = registry.call(
        "retrieve_policy", {"query": "software subscription renewal variance", "limit": 1}
    )
    assert duplicate.payload["retrieved_count"] == 1
    duplicate_citations = duplicate.payload["citations"]
    renewal_citations = renewal.payload["citations"]
    assert isinstance(duplicate_citations, list)
    assert isinstance(duplicate_citations[0], dict)
    assert isinstance(renewal_citations, list)
    assert isinstance(renewal_citations[0], dict)
    assert duplicate_citations[0]["chunk_id"] == "POL-PROC-04@demo-2#2.2"
    assert renewal_citations[0]["chunk_id"] == "POL-PROC-04@demo-2#3.1"
    assert not registry.call("retrieve_policy", {"query": "receipt", "limit": 0}).ok


def test_tools_reject_unscoped_and_invalid_requests() -> None:
    registry = build_operations_registry()
    assert len(registry.tools) == 6
    assert not registry.call("search_transactions", {"month": "2026-08", "tenant_id": "other"}).ok
    assert not registry.call("search_transactions", {"month": "2026-09"}).ok
    assert not registry.call(
        "propose_review_case", {"transaction_id": "missing", "reason": "possible_duplicate"}
    ).ok
    assert not registry.call(
        "propose_review_case", {"transaction_id": "AUG-08", "reason": "possible_duplicate"}
    ).ok
    assert not registry.call("get_transaction_context", {"transaction_id": "missing"}).ok


def test_trace_comes_from_actual_registry_calls() -> None:
    registry = build_operations_registry()
    report = investigate_demo()
    trace = report["trace"]
    assert isinstance(trace, list)
    assert all(isinstance(step, dict) for step in trace)
    steps = [step for step in trace if isinstance(step, dict)]
    assert [step["tool"] for step in steps] == [
        "search_transactions",
        "summarize_spend",
        "detect_duplicates",
        "retrieve_policy",
        "get_transaction_context",
        "propose_review_case",
    ]
    for step in steps:
        name, arguments = step["tool"], step["arguments"]
        assert isinstance(name, str)
        assert isinstance(arguments, dict)
        assert step["result"] == registry.call(name, arguments).as_json()
    assert report["mode"] == "scripted_demo_not_live_model"


def test_alternate_csv_reconciles_to_a_different_result() -> None:
    report = investigate_demo("alternate_billing_review")
    summary = report["summary"]
    assert isinstance(summary, dict)
    assert summary["prior_spend"] == "5000.00"
    assert summary["total_spend"] == "10500.00"
    assert summary["change"] == "5500.00"
    assert summary["drivers"] == [
        {"merchant": "Observability Cloud", "change": "3500.00"},
        {"merchant": "Compliance Renewal", "change": "2000.00"},
    ]
    assert report["duplicate"] != investigate_demo()["duplicate"]
    assert report["proposal"] == {
        "proposal_id": "DEMO-REVIEW-ALT-AUG-02",
        "transaction_id": "ALT-AUG-02",
        "reason": "possible_duplicate",
        "status": "proposal_only",
        "ledger_mutated": False,
        "review_queue_mutated": False,
        "requires_human_approval": True,
    }
    trace = report["trace"]
    assert isinstance(trace, list)
    assert len(trace) == 6


def test_csv_parser_and_registry_respond_to_unseen_synthetic_rows() -> None:
    raw = (
        "id,posted_on,posted_time,merchant,category,amount,receipt\n"
        "A,2026-07-01,09:00,Vendor,Software,10.00,true\n"
        "B,2026-08-01,09:00,Vendor,Software,20.00,true\n"
    )
    transactions = parse_synthetic_csv(raw)
    registry = build_operations_registry(transactions)
    assert registry.call("summarize_spend", {"month": "2026-08"}).payload["change"] == "10.00"
    assert registry.call("detect_duplicates", {"month": "2026-08"}).payload["pairs"] == []


@pytest.mark.parametrize(
    "bad_row",
    [
        "A,2026-08-01,09:00,Vendor,Software,1.001,true",
        "A,2026-08-01,09:00,Vendor,Software,NaN,true",
        "A,2026-08-01,09:00,Vendor,Software,10.00,maybe",
        "A,2026-08-01,09:00,Vendor,Software,10.00,true,extra",
    ],
)
def test_synthetic_csv_rejects_invalid_rows(bad_row: str) -> None:
    raw = (
        "id,posted_on,posted_time,merchant,category,amount,receipt\n"
        "J,2026-07-01,09:00,Vendor,Software,10.00,true\n"
        f"{bad_row}\n"
    )
    with pytest.raises(ValidationError):
        parse_synthetic_csv(raw)


def test_synthetic_case_loader_never_reads_a_caller_selected_path() -> None:
    assert len(load_synthetic_case("alternate_billing_review")) == 6
    with pytest.raises(ValidationError):
        load_synthetic_case("../../private/bank.csv")
