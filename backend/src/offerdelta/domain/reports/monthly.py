"""The monthly tree.

Rooted at the sum of every row rather than at net cash flow, so that a
dropped or double-counted row makes the report impossible to build rather
than quietly wrong. That is what makes the report auditable without any
balance data, which no bank export in this project provides.

Transfers get their own branch rather than being excluded. Excluding them
would mean the root is no longer the sum of everything, and a transfer
misclassified as spending would leave a tree that still balances - losing
exactly the error this structure exists to catch.

Net cash flow and savings rate are derived views of this tree, computed by a
caller. They are not the root.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from offerdelta.domain.common.derivation import DerivationNode, _weakest
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.evidence import Evidence
from offerdelta.domain.common.money import Money
from offerdelta.domain.common.periods import PeriodKind
from offerdelta.domain.costs.categories import CostCategory
from offerdelta.domain.transactions.entities import TransactionKind

#: Explicit abstention: a row a categoriser examined and declined to answer.
#: Mirrors `offerdelta.evaluation.labels.ABSTAIN`, restated here rather than
#: imported because the domain layer may depend on the standard library only
#: and `offerdelta.evaluation` does not.
_ABSTAIN_LABEL = "UNKNOWN"

_SPENDING_LABELS: frozenset[str] = frozenset(category.value for category in CostCategory)

_NEVER_EXAMINED_CODE = "never_examined"
_EXAMINED_NO_ANSWER_CODE = "examined_no_answer"


@dataclass(frozen=True)
class ClassifiedTransaction:
    """One imported row together with whatever classification it currently has.

    `transaction_id` is the row's identity one layer up (`StoredTransaction.id`
    in the persistence layer). It becomes the leaf's `code` in the tree this
    module builds, so that a figure can expand back to the transaction that
    produced it even after the tree has been serialised and the caller's
    original, ordered list of rows is long gone - a positional index would
    only resolve against that exact list, which is useless to a frontend.

    `label` is `None` for a row nobody has looked at yet, the sentinel
    `"UNKNOWN"` for one a categoriser examined and declined to answer, a
    `TransactionKind` value (`INCOME`, `TRANSFER`, `REFUND`) for what the row
    *is* when it is not a cost, or a `CostCategory` value for a spending row.

    `confirmed` distinguishes a user-verified label from a machine
    suggestion; it is meaningless without a label and is ignored for the two
    unclassified cases, so an unreviewed guess cannot be made to look
    confirmed by a stray flag.
    """

    transaction_id: uuid.UUID
    posted_on: date
    description: str
    amount: Money
    label: str | None
    confirmed: bool


@dataclass
class _Groups:
    """The rows for one month, sorted into every branch of the tree.

    A plain accumulator, not part of the public interface: `_group_by_label`
    fills it in one pass and `_assemble_branches` reads it in another, so
    that each of those two functions stays short enough to review on its own.
    """

    income: list[DerivationNode]
    refunds: list[DerivationNode]
    transfers: list[DerivationNode]
    never_examined: list[DerivationNode]
    examined_no_answer: list[DerivationNode]
    spending_by_category: dict[str, list[DerivationNode]]


def build_monthly_report(
    year: int, month: int, transactions: Sequence[ClassifiedTransaction]
) -> DerivationNode:
    """Build the report tree for one calendar month.

    The root's amount is computed directly from `transactions`, independently
    of the branches below it - it is never derived from them. `DerivationNode`
    then refuses to construct a node whose children do not sum to its stated
    amount, so the root's independent total and the branches' total have to
    agree or the report does not build. That is what makes a row the grouping
    drops, or counts twice, a build failure instead of a total that only
    checks itself: `_group_by_label` could reproduce either mistake and
    nothing downstream of it would notice, because every branch and the old
    root were both computed from its own output. A row posted outside the
    requested month is refused for the same auditability reason - it would
    otherwise inflate or deflate this month's total without a trace.
    """
    month_name = f"{year:04d}-{month:02d}"
    _require_within_month(transactions, year, month, month_name)
    branches = _assemble_branches(_group_by_label(transactions))

    return DerivationNode(
        code="monthly_report",
        label=f"Monthly report for {month_name}",
        amount=_sum_transactions(transactions),
        period=PeriodKind.MONTHLY,
        formula="income + spending + refunds + transfers + unclassified",
        # An empty branch's own evidence is `ASSUMED` (see `_weakest`'s
        # empty-list fallback) so it never claims data it does not have when
        # shown on its own - but it must not then drag a fully-reviewed
        # month's root down to `ASSUMED` just because an unrelated category
        # happened to be empty. `if branch.children` excludes it here for the
        # same reason `_assemble_branches` keeps it at zero rather than
        # omitting it: "$0.00" and "no branch at all" must mean the same
        # thing, for evidence exactly as much as for amount.
        evidence=_weakest([branch.evidence for branch in branches if branch.children]),
        children=tuple(branches),
    )


def _require_within_month(
    transactions: Sequence[ClassifiedTransaction], year: int, month: int, month_name: str
) -> None:
    for txn in transactions:
        if txn.posted_on.year != year or txn.posted_on.month != month:
            raise ValidationError(f"{txn.posted_on} is not in {month_name}")


def _group_by_label(transactions: Sequence[ClassifiedTransaction]) -> _Groups:
    """Sort every row into exactly one list, or refuse it.

    An unrecognised label falls through every `elif` below; raising in the
    `else` is what turns "a label the grouping does not handle" into a build
    failure instead of a row that quietly never reaches the tree.
    """
    groups = _Groups(
        income=[],
        refunds=[],
        transfers=[],
        never_examined=[],
        examined_no_answer=[],
        spending_by_category=defaultdict(list),
    )
    for txn in transactions:
        if txn.label is None:
            groups.never_examined.append(_leaf(txn, Evidence.ASSUMED))
        elif txn.label == _ABSTAIN_LABEL:
            groups.examined_no_answer.append(_leaf(txn, Evidence.ASSUMED))
        elif txn.label == TransactionKind.INCOME.value:
            groups.income.append(_leaf(txn, _evidence_of(txn)))
        elif txn.label == TransactionKind.TRANSFER.value:
            groups.transfers.append(_leaf(txn, _evidence_of(txn)))
        elif txn.label == TransactionKind.REFUND.value:
            groups.refunds.append(_leaf(txn, _evidence_of(txn)))
        elif txn.label in _SPENDING_LABELS:
            groups.spending_by_category[txn.label].append(_leaf(txn, _evidence_of(txn)))
        else:
            raise ValidationError(f"{txn.label!r} is not a recognised transaction label")
    return groups


def _assemble_branches(groups: _Groups) -> list[DerivationNode]:
    """The five headline branches, always present and always in this order.

    `income`, `spending`, `refunds`, `transfers` and `unclassified` are built
    unconditionally, at zero when the month has no rows for them. A reader
    comparing this report across months needs "$0.00" and "no branch at all"
    to mean the same thing, and a caller needs the same five codes to be
    reachable on every root regardless of how the month went - the month
    with nothing left unclassified is the success case, and it must not be
    the one where `unclassified` disappears.
    """
    return [
        _group("income", "Income", groups.income),
        _spending_branch(groups.spending_by_category),
        _group("refunds", "Refunds", groups.refunds),
        _group("transfers", "Transfers", groups.transfers),
        _unclassified_branch(groups),
    ]


def _unclassified_branch(groups: _Groups) -> DerivationNode:
    """Two leaves under one branch - never examined, and examined but unanswered.

    The branch itself is always present, at zero when both are empty, so the
    tree can state "0 awaiting review" rather than omit the fact. The two
    children stay conditional: they are detail about *why* something is
    unclassified, not the headline fact that anything is, so an empty one
    carries no information the parent's zero does not already have.
    """
    children: list[DerivationNode] = []
    if groups.never_examined:
        children.append(_group(_NEVER_EXAMINED_CODE, "Never examined", groups.never_examined))
    if groups.examined_no_answer:
        children.append(
            _group(_EXAMINED_NO_ANSWER_CODE, "Examined, no answer", groups.examined_no_answer)
        )
    return DerivationNode(
        code="unclassified",
        label="Unclassified",
        amount=_sum(children),
        period=PeriodKind.MONTHLY,
        formula="never examined + examined but unanswered",
        evidence=_weakest([child.evidence for child in children]),
        children=tuple(children),
    )


def _evidence_of(txn: ClassifiedTransaction) -> Evidence:
    """A confirmed label is sourced from the user; a suggestion is a guess."""
    return Evidence.USER_CONFIRMED if txn.confirmed else Evidence.ASSUMED


def _leaf(txn: ClassifiedTransaction, evidence: Evidence) -> DerivationNode:
    """One row, one leaf - the property the whole tree shape exists to enforce.

    The leaf's code is the transaction's own id rather than a position in
    some list, so a figure in a serialised tree still expands to the row
    that produced it.
    """
    return DerivationNode(
        code=str(txn.transaction_id),
        label=txn.description,
        amount=txn.amount,
        period=PeriodKind.MONTHLY,
        formula="imported transaction",
        evidence=evidence,
        children=(),
    )


def _group(code: str, label: str, leaves: list[DerivationNode]) -> DerivationNode:
    """Sum a list of leaves into a named branch, evidenced by their weakest."""
    return DerivationNode(
        code=code,
        label=label,
        amount=_sum(leaves),
        period=PeriodKind.MONTHLY,
        formula=f"sum of {len(leaves)} transaction(s)",
        evidence=_weakest([leaf.evidence for leaf in leaves]),
        children=tuple(leaves),
    )


def _spending_branch(by_category: dict[str, list[DerivationNode]]) -> DerivationNode:
    """Spending, further grouped by `CostCategory` rather than left flat.

    The branch itself is always present, at zero when the month has no
    spending. Its category children stay conditional: they are detail about
    *where* spending went, not the headline fact of how much - a branch for
    every unused category out of the full taxonomy would bury the ones that
    matter, and an empty one carries no information the parent's zero does
    not already have.
    """
    category_nodes = tuple(
        _group(category, _title(category), leaves)
        for category, leaves in sorted(by_category.items())
    )
    return DerivationNode(
        code="spending",
        label="Spending",
        amount=_sum(category_nodes),
        period=PeriodKind.MONTHLY,
        formula=f"sum of {len(category_nodes)} categories",
        evidence=_weakest([node.evidence for node in category_nodes]),
        children=category_nodes,
    )


def _title(category_code: str) -> str:
    return category_code.replace("_", " ").title()


def _sum(nodes: Sequence[DerivationNode]) -> Money:
    total = Money.zero()
    for node in nodes:
        total = total + node.amount
    return total


def _sum_transactions(transactions: Sequence[ClassifiedTransaction]) -> Money:
    """The root's amount, computed straight from the input rows.

    Deliberately not routed through `_group_by_label` or the branches it
    produces - see `build_monthly_report`'s docstring for why the root must
    reach its total by a path independent of the grouping it is meant to
    check.
    """
    total = Money.zero()
    for txn in transactions:
        total = total + txn.amount
    return total
