"""Turning stored rows into a month's report, and indexing what months exist.

The application-layer glue between persistence and the pure domain tree
builder (`offerdelta.domain.reports.monthly.build_monthly_report`): read this
tenant's rows for a month, map each one into a `ClassifiedTransaction`, and
hand the list to the builder. Every row `for_month` returns is mapped, none
filtered out here - the builder's root-equals-sum-of-every-row property only
holds if this module never quietly drops one first.

This module also answers the question a report picker has to ask before it
can request any one month: which months has this tenant got, and is each one
whole. That is `available_months`, and "whole" is a narrower claim than "has
rows" - see `MonthCoverage` and `_snapshot_windows`.
"""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from typing import Final

from sqlalchemy import select

from offerdelta.application.scope import TenantScope
from offerdelta.domain.common.derivation import DerivationNode
from offerdelta.domain.reports.monthly import ClassifiedTransaction, build_monthly_report
from offerdelta.evaluation.labels import is_abstention
from offerdelta.infrastructure.postgres.models import AccountRow, ImportBatchRow, TransactionRow
from offerdelta.infrastructure.postgres.repositories import (
    StoredTransaction,
    TransactionRepository,
)

#: The mode value `ImportBatchRow` requires, per its `ck_import_batches_mode`
#: check constraint. Completeness is decided from this exact declaration -
#: never from row density, gaps, or counts - because an incremental batch
#: claims only that activity was appended, not that a window is whole.
_SNAPSHOT_MODE: Final = "snapshot"


@dataclass(frozen=True)
class MonthCoverage:
    """How much of one calendar month this tenant's stored rows account for.

    `complete` is a narrower claim than "has rows": it is whether a snapshot
    import declared, in `import_batches`, that its window covered the
    *entire* month - never inferred from how many rows happen to be there,
    because a thin month and a genuinely partial one look identical by row
    count alone. `rows` and `classified` come from the same set of stored
    rows; `classified` counts those with an `effective_label` (a suggestion
    or a confirmation, either one) that is not an abstention, independent of
    any confidence bar. `'UNKNOWN'` is a categoriser's answer, not the
    absence of one - `is_abstention` excludes it here for the same reason
    `build_monthly_report` files it under `unclassified/examined_no_answer`
    rather than under a headline branch: counting it as classified would let
    this figure disagree with where the report's own tree puts the row, and
    would contradict `evaluation.metrics.ClassificationReport.coverage`'s
    `(total - abstentions) / total`, the one definition of coverage this
    repository has.

    `awaiting_review` always means the same thing wherever this type is
    produced: `TransactionRepository.awaiting_review` evaluated at one
    caller-supplied threshold, for this month. Both `available_months` and
    `monthly_report` require that threshold as an argument for exactly this
    reason - a coverage figure computed at a different, undisclosed bar
    would read as the same fact while meaning a different one.
    """

    year: int
    month: int
    complete: bool
    rows: int
    classified: int
    awaiting_review: int


@dataclass(frozen=True)
class MonthlyReport:
    """One month's tree, plus how much of the month it could account for."""

    tree: DerivationNode
    coverage: MonthCoverage


@dataclass(frozen=True)
class LoadedMonth:
    """One tenant-scoped row set and coverage calculated from those same rows."""

    rows: list[StoredTransaction]
    coverage: MonthCoverage


def available_months(scope: TenantScope, *, threshold: Decimal) -> list[MonthCoverage]:
    """Every month this tenant has at least one stored row for, oldest first.

    A month with rows but no snapshot window covering it is still listed -
    as partial, never omitted - because leaving it out would hide exactly
    the gap this feature exists to surface.

    `threshold` is required, not defaulted: a caller who lists months and
    then opens one via `monthly_report` must pass the same value to both, or
    `awaiting_review` means two different things for what looks like the
    same month - see `MonthCoverage`.
    """
    windows = _snapshot_windows(scope)
    repo = TransactionRepository(scope)
    months = sorted(_months_with_rows(scope))
    return [
        _coverage(
            scope, year, month, repo.for_month(year, month), windows=windows, threshold=threshold
        )
        for year, month in months
    ]


def monthly_report(
    scope: TenantScope, year: int, month: int, *, threshold: Decimal
) -> MonthlyReport:
    """Build one month's tree from what is actually stored, plus its coverage.

    The tree is rooted at the sum of every stored row for the month - see
    `build_monthly_report` - so a row this function failed to map would be a
    row the root silently stopped accounting for. Every `StoredTransaction`
    `for_month` returns is mapped, none filtered, which is what keeps that
    property true.

    `for_month` is queried exactly once, here, and the same list is reused
    to build both the tree and the coverage: a second, independent query
    would open a window where a concurrent write between the two SELECTs
    could make `coverage.rows` disagree with the tree's own root sum.
    """
    loaded = load_month(scope, year, month, threshold=threshold)
    tree = build_monthly_report(year, month, [_to_classified(row) for row in loaded.rows])
    return MonthlyReport(tree=tree, coverage=loaded.coverage)


def load_month(scope: TenantScope, year: int, month: int, *, threshold: Decimal) -> LoadedMonth:
    """Fetch a month's rows once, then calculate coverage over precisely that set."""
    stored = TransactionRepository(scope).for_month(year, month)
    windows = _snapshot_windows(scope)
    coverage = _coverage(scope, year, month, stored, windows=windows, threshold=threshold)
    return LoadedMonth(rows=stored, coverage=coverage)


def _to_classified(row: StoredTransaction) -> ClassifiedTransaction:
    """A person's word over a model's guess - `effective_label` already decides that."""
    return ClassifiedTransaction(
        transaction_id=row.id,
        posted_on=row.posted_on,
        description=row.description,
        amount=row.amount,
        label=row.effective_label,
        confirmed=row.confirmed_label is not None,
    )


def _coverage(
    scope: TenantScope,
    year: int,
    month: int,
    stored: list[StoredTransaction],
    *,
    windows: list[tuple[date, date]],
    threshold: Decimal,
) -> MonthCoverage:
    """Build one month's `MonthCoverage` from a list already fetched by the caller.

    Takes `stored` rather than fetching it again: both call sites (`available_months`
    and `monthly_report`) already have their own `for_month` result in hand, and a
    second, independent query here would risk `rows`/`classified` disagreeing with
    whatever the caller built from its own copy of the same month.
    """
    classified = sum(
        1
        for row in stored
        if row.effective_label is not None and not is_abstention(row.effective_label)
    )
    awaiting = TransactionRepository(scope).awaiting_review(threshold, month=(year, month))
    return MonthCoverage(
        year=year,
        month=month,
        complete=_month_is_covered(year, month, windows),
        rows=len(stored),
        classified=classified,
        awaiting_review=len(awaiting),
    )


def _months_with_rows(scope: TenantScope) -> set[tuple[int, int]]:
    """Every `(year, month)` this tenant has at least one transaction posted in."""
    posted_on_dates = scope.session.scalars(
        select(TransactionRow.posted_on)
        .join(AccountRow, AccountRow.id == TransactionRow.account_id)
        .where(AccountRow.user_id == scope.user.id)
    ).all()
    return {(posted_on.year, posted_on.month) for posted_on in posted_on_dates}


def _snapshot_windows(scope: TenantScope) -> list[tuple[date, date]]:
    """This tenant's declared snapshot windows, across every account.

    Filters on `mode == "snapshot"` explicitly rather than on whether a
    window happens to be present: an incremental batch claims only that
    activity was appended, never that a window is whole, so its window (if
    it carries one at all) must never contribute to completeness. That is
    the one distinction this whole function exists to hold.
    """
    rows = scope.session.execute(
        select(ImportBatchRow.window_start, ImportBatchRow.window_end)
        .join(AccountRow, AccountRow.id == ImportBatchRow.account_id)
        .where(
            AccountRow.user_id == scope.user.id,
            ImportBatchRow.mode == _SNAPSHOT_MODE,
        )
    ).all()
    # `ck_import_batches_snapshot_window` guarantees both columns are set on
    # every row a snapshot-mode filter can return; the None checks below are
    # what tell mypy that, not a runtime possibility this code expects.
    return [(start, end) for start, end in rows if start is not None and end is not None]


def _month_is_covered(year: int, month: int, windows: list[tuple[date, date]]) -> bool:
    month_start = date(year, month, 1)
    month_end = date(year, month, monthrange(year, month)[1])
    merged = _merge_windows(windows)
    return any(start <= month_start and end >= month_end for start, end in merged)


def _merge_windows(windows: list[tuple[date, date]]) -> list[tuple[date, date]]:
    """Collapse overlapping or adjoining windows into their maximal spans.

    Two snapshots covering the 1st-15th and the 16th-31st leave no day
    uncovered even though neither alone spans the month; only the merged
    union can answer "is every day covered", which is the claim `complete`
    actually makes.
    """
    merged: list[tuple[date, date]] = []
    for start, end in sorted(windows):
        if merged and start <= merged[-1][1] + timedelta(days=1):
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged
