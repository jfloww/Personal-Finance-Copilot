"""The monthly tree.

Rooted at the sum of every row rather than at net cash flow, so that a
dropped or double-counted row makes the report impossible to build rather
than quietly wrong.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import date

import pytest

from offerdelta.domain.common.derivation import DerivationNode
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.evidence import Evidence
from offerdelta.domain.common.money import Money
from offerdelta.domain.reports import monthly as monthly_module
from offerdelta.domain.reports.monthly import ClassifiedTransaction, build_monthly_report


def _txn(
    day: int, amount: str, label: str | None, *, confirmed: bool = False
) -> ClassifiedTransaction:
    return ClassifiedTransaction(
        transaction_id=uuid.uuid4(),
        posted_on=date(2026, 3, day),
        description=f"ROW {day}",
        amount=Money.parse(amount),
        label=label,
        confirmed=confirmed,
    )


def _branch(root: DerivationNode, code: str) -> DerivationNode:
    for child in root.children:
        if child.code == code:
            return child
    raise AssertionError(f"no branch {code!r} in {[c.code for c in root.children]}")


def test_the_root_is_the_sum_of_every_row() -> None:
    rows = [
        _txn(1, "5000.00", "INCOME", confirmed=True),
        _txn(2, "-12.34", "LIVING_DINING", confirmed=True),
        _txn(3, "-500.00", "TRANSFER", confirmed=True),
        _txn(4, "-9.99", None),
    ]
    root = build_monthly_report(2026, 3, rows)
    assert root.amount == Money.parse("4477.67")


def test_transfers_are_a_branch_and_not_an_exclusion() -> None:
    """Excluding them would leave a balanced tree around a misclassified one."""
    rows = [_txn(1, "-500.00", "TRANSFER", confirmed=True)]
    root = build_monthly_report(2026, 3, rows)
    assert _branch(root, "transfers").amount == Money.parse("-500.00")
    assert root.amount == Money.parse("-500.00")


def test_never_examined_and_examined_but_unknown_are_separate_leaves() -> None:
    rows = [_txn(1, "-1.00", None), _txn(2, "-2.00", "UNKNOWN")]
    unclassified = _branch(build_monthly_report(2026, 3, rows), "unclassified")
    by_code = {child.code: child.amount for child in unclassified.children}
    assert by_code == {
        "never_examined": Money.parse("-1.00"),
        "examined_no_answer": Money.parse("-2.00"),
    }


def test_a_month_of_confirmed_rows_has_a_confirmed_root() -> None:
    rows = [_txn(1, "-1.00", "LIVING_DINING", confirmed=True)]
    assert build_monthly_report(2026, 3, rows).evidence is Evidence.USER_CONFIRMED


def test_one_unreviewed_row_makes_the_whole_root_assumed() -> None:
    rows = [
        _txn(1, "-1.00", "LIVING_DINING", confirmed=True),
        _txn(2, "-2.00", "LIVING_GROCERY", confirmed=False),
    ]
    assert build_monthly_report(2026, 3, rows).evidence is Evidence.ASSUMED


def test_spending_is_broken_down_by_category() -> None:
    rows = [
        _txn(1, "-10.00", "LIVING_DINING", confirmed=True),
        _txn(2, "-5.00", "LIVING_DINING", confirmed=True),
        _txn(3, "-20.00", "LIVING_GROCERY", confirmed=True),
    ]
    spending = _branch(build_monthly_report(2026, 3, rows), "spending")
    by_code = {child.code: child.amount for child in spending.children}
    assert by_code == {
        "LIVING_DINING": Money.parse("-15.00"),
        "LIVING_GROCERY": Money.parse("-20.00"),
    }


def test_a_row_outside_the_month_is_refused() -> None:
    stray = ClassifiedTransaction(
        transaction_id=uuid.uuid4(),
        posted_on=date(2026, 4, 1),
        description="APRIL ROW",
        amount=Money.parse("-1.00"),
        label="LIVING_DINING",
        confirmed=True,
    )
    with pytest.raises(ValidationError, match="2026-03"):
        build_monthly_report(2026, 3, [stray])


def test_an_empty_branch_is_assumed_not_sourced() -> None:
    """A month with no income claims no provenance for its zero, not the
    strongest provenance `Evidence` has - the property Fix 7 exists for.
    """
    rows = [_txn(1, "-10.00", "LIVING_DINING", confirmed=True)]
    root = build_monthly_report(2026, 3, rows)
    assert _branch(root, "income").evidence is Evidence.ASSUMED


def test_empty_branches_do_not_drag_a_confirmed_root_to_assumed() -> None:
    """The interaction Fix 7 has to get right: an empty branch's own
    evidence is honestly `ASSUMED` (see the test above), but it must not
    then make a month where every actual row is confirmed report as
    `ASSUMED` overall just because income, refunds, transfers and
    unclassified all happen to be empty this month. "$0.00" and "no branch
    at all" mean the same thing for evidence, exactly as they already do
    for amount.
    """
    rows = [_txn(1, "-10.00", "LIVING_DINING", confirmed=True)]
    root = build_monthly_report(2026, 3, rows)
    assert root.evidence is Evidence.USER_CONFIRMED


def test_an_empty_month_is_a_zero_root_not_an_error() -> None:
    """The purest form of the success case the always-five-branches contract exists for.

    A month where every row has been reviewed - here, a month with no rows
    at all - must still say "0 awaiting review" rather than have the
    `unclassified` branch vanish because it happens to agree with zero.
    """
    root = build_monthly_report(2026, 3, [])
    assert root.amount == Money.zero()
    assert root.evidence is Evidence.ASSUMED, "no rows means no provenance to claim, not SOURCED"
    by_code = {child.code: child.amount for child in root.children}
    assert by_code == {
        "income": Money.zero(),
        "spending": Money.zero(),
        "refunds": Money.zero(),
        "transfers": Money.zero(),
        "unclassified": Money.zero(),
    }


def test_a_month_of_only_income_still_has_all_five_branches() -> None:
    """Pins the contract: every branch is reachable on every root, not just a
    populated one - four of the five are zero here, and still present.
    """
    rows = [_txn(1, "5000.00", "INCOME", confirmed=True)]
    root = build_monthly_report(2026, 3, rows)
    by_code = {child.code: child.amount for child in root.children}
    assert by_code == {
        "income": Money.parse("5000.00"),
        "spending": Money.zero(),
        "refunds": Money.zero(),
        "transfers": Money.zero(),
        "unclassified": Money.zero(),
    }


def test_no_row_is_silently_dropped_by_the_grouping() -> None:
    """The failure the root-at-every-row shape exists to make impossible.

    A label that falls through every branch would vanish from the tree and
    the totals would still look plausible. One row per branch, and the root
    must equal their sum - if grouping loses one, this fails.
    """
    rows = [
        _txn(1, "5000.00", "INCOME", confirmed=True),
        _txn(2, "-10.00", "LIVING_DINING", confirmed=True),
        _txn(3, "25.00", "REFUND", confirmed=True),
        _txn(4, "-500.00", "TRANSFER", confirmed=True),
        _txn(5, "-1.00", None),
        _txn(6, "-2.00", "UNKNOWN"),
    ]
    root = build_monthly_report(2026, 3, rows)

    leaves = [node for node in root.walk() if not node.children]
    total = Money.zero()
    for leaf in leaves:
        total = total + leaf.amount

    assert len(leaves) == len(rows), "every row appears exactly once as a leaf"
    assert total == root.amount
    assert root.amount == Money.parse("4512.00")
    assert {child.code for child in root.children} == {
        "income",
        "spending",
        "refunds",
        "transfers",
        "unclassified",
    }


# --------------------------------------------------------- the root checks the grouping

#: Fixed so both regression tests below tamper with the same honest total:
#: income 5000.00, dining -10.00, refund 25.00 sum to 5015.00. Dropping the
#: refund row leaves 4990.00; doubling the income row makes it 10015.00 - the
#: exact figures a happy, wrong tree produced before this guard existed.
_GUARD_ROWS = [
    _txn(1, "5000.00", "INCOME", confirmed=True),
    _txn(2, "-10.00", "LIVING_DINING", confirmed=True),
    _txn(3, "25.00", "REFUND", confirmed=True),
]


def test_a_row_the_grouping_drops_makes_the_report_impossible_to_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pins Fix 1: before the root was computed independently of the
    branches, this exact tampering built a happy tree at 4990.00 instead of
    the honest 5015.00 - no exception either. `_group_by_label` is patched
    rather than handed a bad label, so this proves the *root*, not `else`
    branch in `_group_by_label`, is what catches a silently dropped row.
    """
    real_group_by_label = monthly_module._group_by_label
    dropped_id = _GUARD_ROWS[2].transaction_id

    def _dropping_a_row(
        transactions: Sequence[ClassifiedTransaction],
    ) -> monthly_module._Groups:
        return real_group_by_label([t for t in transactions if t.transaction_id != dropped_id])

    monkeypatch.setattr(monthly_module, "_group_by_label", _dropping_a_row)

    with pytest.raises(ValidationError, match="does not equal the sum"):
        build_monthly_report(2026, 3, _GUARD_ROWS)


def test_a_row_the_grouping_doubles_makes_the_report_impossible_to_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The mirror of the dropped-row regression above: before this guard,
    the same tampering built a happy tree at 10015.00 instead of 5015.00.
    """
    real_group_by_label = monthly_module._group_by_label

    def _doubling_a_row(
        transactions: Sequence[ClassifiedTransaction],
    ) -> monthly_module._Groups:
        return real_group_by_label([*transactions, _GUARD_ROWS[0]])

    monkeypatch.setattr(monthly_module, "_group_by_label", _doubling_a_row)

    with pytest.raises(ValidationError, match="does not equal the sum"):
        build_monthly_report(2026, 3, _GUARD_ROWS)
