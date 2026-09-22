"""Read-only transaction investigations over an explicitly synthetic ledger."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from functools import partial
from typing import Final, cast

from offerdelta.agent.tools.registry import JsonValue, Tool, ToolRegistry, ToolResult
from offerdelta.domain.common.errors import ValidationError
from offerdelta.policy.corpus import PolicyChunk, load_policy_corpus
from offerdelta.policy.retrieval import PolicyRetriever, default_policy_retriever


@dataclass(frozen=True)
class DemoTransaction:
    id: str
    month: str
    day: int
    minute: int
    merchant: str
    category: str
    amount: Decimal
    receipt: bool = True


# Six July charges recur in August. Exactly two additional August charges
# account for the $12,500.00 increase: renewal $7,650 + duplicate $4,850.
_BASE: Final = (
    ("Amazon Web Services", "Cloud Infrastructure", "4850.00"),
    ("Linear", "Software", "2000.00"),
    ("Figma", "Software", "1500.00"),
    ("Datadog", "Software", "1200.00"),
    ("GitHub", "Software", "950.00"),
    ("Notion", "Software", "1500.00"),
)

TRANSACTIONS: Final[tuple[DemoTransaction, ...]] = (
    tuple(
        DemoTransaction(
            f"JUL-{index:02}", "2026-07", 14, index * 10, name, category, Decimal(amount)
        )
        for index, (name, category, amount) in enumerate(_BASE, 1)
    )
    + tuple(
        DemoTransaction(
            "TX-9824-A1" if index == 1 else f"AUG-{index:02}",
            "2026-08",
            14,
            index * 10,
            name,
            category,
            Decimal(amount),
        )
        for index, (name, category, amount) in enumerate(_BASE, 1)
    )
    + (
        DemoTransaction(
            "TX-9825-B2",
            "2026-08",
            14,
            14,
            "Amazon Web Services",
            "Cloud Infrastructure",
            Decimal("4850.00"),
            False,
        ),
        DemoTransaction(
            "AUG-08",
            "2026-08",
            20,
            0,
            "Security Platform Renewal",
            "Software",
            Decimal("7650.00"),
        ),
    )
)

DUPLICATE_WINDOW_MINUTES: Final = 5

PURCHASE_POLICY_ID: Final = "POL-PROC-04"
PURCHASE_POLICY_SECTION: Final = "2.1"


def _schema(properties: dict[str, object], required: list[str]) -> dict[str, object]:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


ID: Final[dict[str, object]] = {"type": "string", "minLength": 1}


def _transaction(item: DemoTransaction) -> dict[str, JsonValue]:
    return {
        "id": item.id,
        "month": item.month,
        "day": item.day,
        "merchant": item.merchant,
        "category": item.category,
        "amount": str(item.amount),
        "currency": "USD",
        "has_receipt": item.receipt,
    }


def _search(args: Mapping[str, object], transactions: tuple[DemoTransaction, ...]) -> ToolResult:
    month = args["month"]
    results: list[JsonValue] = [_transaction(t) for t in transactions if t.month == month]
    return ToolResult.success(
        {"month": str(month), "results_count": len(results), "transactions": results}
    )


def _summarize(args: Mapping[str, object], transactions: tuple[DemoTransaction, ...]) -> ToolResult:
    month = args["month"]
    total = sum((t.amount for t in transactions if t.month == month), Decimal(0))
    months = sorted({t.month for t in transactions})
    position = months.index(str(month))
    previous = months[position - 1] if position > 0 else None
    prior = (
        sum((t.amount for t in transactions if t.month == previous), Decimal(0))
        if previous
        else None
    )
    return ToolResult.success(
        {
            "month": str(month),
            "currency": "USD",
            "total_spend": str(total),
            "prior_month": previous,
            "prior_spend": str(prior) if prior is not None else None,
            "change": str(total - prior) if prior is not None else None,
        }
    )


def _duplicates(
    args: Mapping[str, object], transactions: tuple[DemoTransaction, ...]
) -> ToolResult:
    month = args["month"]
    rows = [t for t in transactions if t.month == month]
    pairs: list[JsonValue] = []
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if (
                left.merchant == right.merchant
                and left.amount == right.amount
                and left.day == right.day
                and 0 < abs(left.minute - right.minute) <= DUPLICATE_WINDOW_MINUTES
            ):
                pairs.append(
                    {
                        "transaction_ids": [left.id, right.id],
                        "merchant": left.merchant,
                        "amount_each": str(left.amount),
                        "currency": "USD",
                        "minutes_apart": abs(left.minute - right.minute),
                        "assessment": "possible_duplicate_needs_review",
                    }
                )
    return ToolResult.success({"month": str(month), "pairs": pairs, "count": len(pairs)})


def _policy(args: Mapping[str, object], *, retriever: PolicyRetriever) -> ToolResult:
    query = str(args["query"])
    limit_value = args.get("limit", 3)
    if isinstance(limit_value, bool) or not isinstance(limit_value, int):
        raise ValidationError("limit must be an integer")
    matches = retriever.search(query, limit=limit_value)
    citations: list[JsonValue] = [
        cast(JsonValue, match.citation(rank)) for rank, match in enumerate(matches, 1)
    ]
    return ToolResult.success(
        {
            "query": query,
            "citations": citations,
            "retrieved_count": len(citations),
            "retrieval": "bm25_v1",
            "corpus_version": retriever.corpus_version,
        }
    )


def _purchase_review_policy() -> tuple[PolicyChunk, str]:
    chunk = next(
        (
            item
            for item in load_policy_corpus()
            if item.policy_id == PURCHASE_POLICY_ID and item.section == PURCHASE_POLICY_SECTION
        ),
        None,
    )
    if chunk is None or chunk.threshold is None:
        raise ValidationError("synthetic purchase review policy is unavailable")
    return chunk, chunk.threshold


def _context(args: Mapping[str, object], transactions: tuple[DemoTransaction, ...]) -> ToolResult:
    identifier = args["transaction_id"]
    item = next((t for t in transactions if t.id == identifier), None)
    if item is None:
        return ToolResult.failure("transaction not found in the synthetic demo ledger")
    policy, threshold = _purchase_review_policy()
    return ToolResult.success(
        {
            "transaction": _transaction(item),
            "policy_id": policy.policy_id,
            "policy_version": policy.version,
            "policy_section": policy.section,
            "over_policy_threshold": item.amount > Decimal(threshold),
        }
    )


def _proposal(args: Mapping[str, object], transactions: tuple[DemoTransaction, ...]) -> ToolResult:
    identifier = args["transaction_id"]
    item = next((t for t in transactions if t.id == identifier), None)
    if item is None:
        return ToolResult.failure("transaction not found in the synthetic demo ledger")
    if item.month != max(t.month for t in transactions):
        return ToolResult.failure("review proposals are limited to the latest demo month")
    if args["reason"] != "possible_duplicate":
        raise ValidationError("unsupported review reason")
    pairs = _duplicates({"month": item.month}, transactions).payload["pairs"]
    if not isinstance(pairs, list) or not any(_pair_contains(pair, identifier) for pair in pairs):
        return ToolResult.failure("no duplicate candidate supports this review proposal")
    return ToolResult.success(
        {
            "proposal_id": f"DEMO-REVIEW-{item.id}",
            "transaction_id": item.id,
            "reason": "possible_duplicate",
            "status": "proposal_only",
            "ledger_mutated": False,
            "review_queue_mutated": False,
            "requires_human_approval": True,
        }
    )


def _pair_contains(pair: JsonValue, identifier: object) -> bool:
    if not isinstance(pair, dict):
        return False
    ids = pair.get("transaction_ids")
    return isinstance(ids, list) and identifier in ids


def build_operations_registry(
    transactions: tuple[DemoTransaction, ...] = TRANSACTIONS,
    policy_retriever: PolicyRetriever | None = None,
) -> ToolRegistry:
    """Six closed-world investigation tools bound to one immutable synthetic ledger."""
    if not transactions:
        raise ValidationError("an operations tool registry needs transactions")
    selected_retriever = policy_retriever or default_policy_retriever()
    month: dict[str, object] = {"type": "string", "enum": sorted({t.month for t in transactions})}
    return ToolRegistry(
        (
            Tool(
                "search_transactions",
                "Search a fixed synthetic monthly transaction ledger.",
                _schema({"month": month}, ["month"]),
                partial(_search, transactions=transactions),
            ),
            Tool(
                "summarize_spend",
                "Sum synthetic spend using exact Decimal amounts.",
                _schema({"month": month}, ["month"]),
                partial(_summarize, transactions=transactions),
            ),
            Tool(
                "detect_duplicates",
                "Find same-merchant same-amount charges within five minutes; "
                "flags candidates, not proof.",
                _schema({"month": month}, ["month"]),
                partial(_duplicates, transactions=transactions),
            ),
            Tool(
                "retrieve_policy",
                "BM25 retrieval over versioned synthetic policy sections with ranked citations.",
                _schema(
                    {
                        "query": ID,
                        "limit": {"type": "integer", "minimum": 1, "maximum": 5},
                    },
                    ["query"],
                ),
                partial(_policy, retriever=selected_retriever),
            ),
            Tool(
                "get_transaction_context",
                "Read one synthetic transaction and its policy threshold.",
                _schema({"transaction_id": ID}, ["transaction_id"]),
                partial(_context, transactions=transactions),
            ),
            Tool(
                "propose_review_case",
                "Compute an unpersisted review proposal; never writes a ledger or review queue.",
                _schema(
                    {
                        "transaction_id": ID,
                        "reason": {"type": "string", "enum": ["possible_duplicate"]},
                    },
                    ["transaction_id", "reason"],
                ),
                partial(_proposal, transactions=transactions),
            ),
        )
    )
