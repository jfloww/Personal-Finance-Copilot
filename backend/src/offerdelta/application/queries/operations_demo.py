"""One reproducible investigation; scripted orchestration is not model inference."""

from __future__ import annotations

from decimal import Decimal

from offerdelta.agent.tools.operations import build_operations_registry
from offerdelta.agent.tools.registry import JsonValue, ToolResult
from offerdelta.demo.operations_cases import load_synthetic_case

SCENARIO = "august_software_exceptions"
PAIR_ID_COUNT = 2
QUESTION = (
    "Why did August software spend increase? Find duplicates and policy review triggers "
    "and prepare a review proposal without changing the ledger."
)


def _require(result: ToolResult) -> dict[str, JsonValue]:
    if not result.ok:
        raise RuntimeError("a built-in synthetic investigation tool failed")
    return result.payload


def investigate_demo(scenario: str = SCENARIO) -> dict[str, JsonValue]:
    """Compute a case from synthetic rows; no account, model, or network access."""
    transactions = load_synthetic_case(scenario)
    months = sorted({item.month for item in transactions})
    previous, month = months
    registry = build_operations_registry(transactions)
    steps: list[tuple[str, dict[str, JsonValue]]] = [
        ("search_transactions", {"month": month}),
        ("summarize_spend", {"month": month}),
        ("detect_duplicates", {"month": month}),
        ("retrieve_policy", {"query": "purchase receipt duplicate review policy"}),
    ]
    trace: list[JsonValue] = []
    results: dict[str, dict[str, JsonValue]] = {}
    for name, arguments in steps:
        result = registry.call(name, arguments)
        results[name] = _require(result)
        trace.append({"tool": name, "arguments": arguments, "result": result.as_json()})

    spend = results["summarize_spend"]
    change = Decimal(str(spend["change"]))
    duplicate = results["detect_duplicates"]
    pairs = duplicate["pairs"]
    if not isinstance(pairs, list):
        raise RuntimeError("the synthetic duplicate result is malformed")
    candidate = pairs[0] if pairs and isinstance(pairs[0], dict) else None
    duplicate_amount = (
        Decimal(str(candidate["amount_each"])) if candidate is not None else Decimal("0.00")
    )

    totals: dict[str, Decimal] = {}
    for item in transactions:
        signed_amount = item.amount if item.month == month else -item.amount
        totals[item.merchant] = totals.get(item.merchant, Decimal("0.00")) + signed_amount
    drivers: list[JsonValue] = [
        {"merchant": merchant, "change": str(amount)}
        for merchant, amount in sorted(totals.items(), key=lambda item: (-abs(item[1]), item[0]))
        if amount
    ]
    if sum(totals.values(), Decimal("0.00")) != change:
        raise RuntimeError("the synthetic spend drivers do not reconcile")
    proposal: dict[str, JsonValue] | None = None
    if candidate is not None:
        identifiers = candidate.get("transaction_ids")
        if not isinstance(identifiers, list) or len(identifiers) != PAIR_ID_COUNT:
            raise RuntimeError("the synthetic duplicate candidate is malformed")
        identifier = identifiers[1]
        if not isinstance(identifier, str):
            raise RuntimeError("the synthetic duplicate ID is malformed")
        followups: list[tuple[str, dict[str, JsonValue]]] = [
            ("get_transaction_context", {"transaction_id": identifier}),
            ("propose_review_case", {"transaction_id": identifier, "reason": "possible_duplicate"}),
        ]
        for name, followup_args in followups:
            result = registry.call(name, followup_args)
            results[name] = _require(result)
            trace.append({"tool": name, "arguments": followup_args, "result": result.as_json()})
        proposal = results["propose_review_case"]

    policy = results["retrieve_policy"]["citations"]
    return {
        "scenario": scenario,
        "mode": "scripted_demo_not_live_model",
        "synthetic": True,
        "question": QUESTION,
        "summary": {
            "month": month,
            "prior_month": previous,
            "currency": "USD",
            "total_spend": spend["total_spend"],
            "prior_spend": spend["prior_spend"],
            "change": str(change),
            "possible_duplicate": str(duplicate_amount),
            "drivers": drivers,
            "transaction_count": results["search_transactions"]["results_count"],
        },
        "duplicate": candidate,
        "policy_citations": policy,
        "proposal": proposal,
        "ledger_mutated": False,
        "review_queue_mutated": False,
        "trace": trace,
    }
