"""Explain labelled net-spend changes using tenant-scoped, inspectable rows.

The calculation is deterministic. Suggested labels are included but disclosed;
unclassified rows and transfers never silently become spending. Refunds reduce
net spending and retain their own transaction evidence.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from decimal import Decimal

from offerdelta.application.reports.monthly import MonthCoverage, load_month
from offerdelta.application.scope import TenantScope
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.domain.costs.categories import CostCategory
from offerdelta.domain.transactions.entities import TransactionKind
from offerdelta.infrastructure.postgres.repositories import StoredTransaction

_SPENDING_LABELS = frozenset(category.value for category in CostCategory)


@dataclass(frozen=True)
class SpendEntry:
    transaction_id: uuid.UUID
    posted_on: date
    merchant: str
    amount: Money
    label: str | None
    confirmed: bool


@dataclass(frozen=True)
class SpendEvidence:
    transaction_id: uuid.UUID
    posted_on: date
    contribution: Decimal
    label: str
    confirmed: bool


@dataclass(frozen=True)
class SpendPeriod:
    gross_spending: Decimal
    refunds: Decimal
    net_spending: Decimal
    unclassified_debits: Decimal
    transfer_debits: Decimal
    inconsistent_debits: Decimal
    unclassified_rows: int
    suggested_rows: int
    inconsistent_rows: int


@dataclass(frozen=True)
class MerchantSpendDelta:
    merchant: str
    previous: Decimal
    current: Decimal
    delta: Decimal
    previous_evidence: tuple[SpendEvidence, ...]
    current_evidence: tuple[SpendEvidence, ...]


@dataclass(frozen=True)
class CurrencySpendChange:
    currency: str
    previous: SpendPeriod
    current: SpendPeriod
    delta: Decimal
    merchants: tuple[MerchantSpendDelta, ...]


@dataclass(frozen=True)
class SpendChange:
    previous_coverage: MonthCoverage
    current_coverage: MonthCoverage
    currencies: tuple[CurrencySpendChange, ...]


@dataclass
class _CurrencyAccumulator:
    gross_spending: Decimal = Decimal(0)
    refunds: Decimal = Decimal(0)
    unclassified_debits: Decimal = Decimal(0)
    transfer_debits: Decimal = Decimal(0)
    inconsistent_debits: Decimal = Decimal(0)
    unclassified_rows: int = 0
    suggested_rows: int = 0
    inconsistent_rows: int = 0
    merchants: dict[str, list[SpendEvidence]] = field(default_factory=lambda: defaultdict(list))

    def period(self) -> SpendPeriod:
        return SpendPeriod(
            gross_spending=self.gross_spending,
            refunds=self.refunds,
            net_spending=self.gross_spending - self.refunds,
            unclassified_debits=self.unclassified_debits,
            transfer_debits=self.transfer_debits,
            inconsistent_debits=self.inconsistent_debits,
            unclassified_rows=self.unclassified_rows,
            suggested_rows=self.suggested_rows,
            inconsistent_rows=self.inconsistent_rows,
        )


def explain_spend_change(
    scope: TenantScope, year: int, month: int, *, threshold: Decimal
) -> SpendChange:
    """Compare this tenant's labelled spending with the preceding calendar month."""
    current_start = date(year, month, 1)
    if (year, month) == (1, 1):
        raise ValidationError("0001-01 has no preceding calendar month")
    previous_day = current_start - timedelta(days=1)
    previous = load_month(scope, previous_day.year, previous_day.month, threshold=threshold)
    current = load_month(scope, year, month, threshold=threshold)
    return build_spend_change(
        [_entry(row) for row in previous.rows],
        [_entry(row) for row in current.rows],
        previous.coverage,
        current.coverage,
    )


def _entry(row: StoredTransaction) -> SpendEntry:
    return SpendEntry(
        transaction_id=row.id,
        posted_on=row.posted_on,
        merchant=row.normalised_merchant,
        amount=row.amount,
        label=row.effective_label,
        confirmed=row.confirmed_label is not None,
    )


def build_spend_change(
    previous_entries: Sequence[SpendEntry],
    current_entries: Sequence[SpendEntry],
    previous_coverage: MonthCoverage,
    current_coverage: MonthCoverage,
) -> SpendChange:
    """Keep currencies separate and reconcile every merchant delta to net spend."""
    previous = _accumulate(previous_entries)
    current = _accumulate(current_entries)
    groups: list[CurrencySpendChange] = []
    for currency in sorted(set(previous) | set(current)):
        before = previous.get(currency, _CurrencyAccumulator())
        after = current.get(currency, _CurrencyAccumulator())
        merchants = []
        for name in set(before.merchants) | set(after.merchants):
            before_evidence = tuple(before.merchants.get(name, ()))
            after_evidence = tuple(after.merchants.get(name, ()))
            before_amount = sum((row.contribution for row in before_evidence), Decimal(0))
            after_amount = sum((row.contribution for row in after_evidence), Decimal(0))
            merchants.append(
                MerchantSpendDelta(
                    merchant=name,
                    previous=before_amount,
                    current=after_amount,
                    delta=after_amount - before_amount,
                    previous_evidence=before_evidence,
                    current_evidence=after_evidence,
                )
            )
        merchants.sort(key=lambda row: (-abs(row.delta), row.merchant))
        before_period, after_period = before.period(), after.period()
        delta = after_period.net_spending - before_period.net_spending
        if sum((row.delta for row in merchants), Decimal(0)) != delta:
            raise ValidationError("merchant drivers do not reconcile to net spending")
        groups.append(
            CurrencySpendChange(currency, before_period, after_period, delta, tuple(merchants))
        )
    return SpendChange(previous_coverage, current_coverage, tuple(groups))


def _accumulate(entries: Sequence[SpendEntry]) -> dict[str, _CurrencyAccumulator]:
    groups: dict[str, _CurrencyAccumulator] = defaultdict(_CurrencyAccumulator)
    for entry in entries:
        group = groups[entry.amount.currency]
        amount = entry.amount.amount
        label = entry.label
        if label is not None and label != "UNKNOWN" and not entry.confirmed:
            group.suggested_rows += 1
        if label is None or label == "UNKNOWN":
            group.unclassified_rows += 1
            if amount < 0:
                group.unclassified_debits += -amount
        elif label == TransactionKind.TRANSFER.value:
            if amount < 0:
                group.transfer_debits += -amount
        elif label in _SPENDING_LABELS and amount < 0:
            group.gross_spending += -amount
            _record(group, entry, -amount)
        elif label == TransactionKind.REFUND.value and amount > 0:
            group.refunds += amount
            _record(group, entry, -amount)
        elif label == TransactionKind.INCOME.value and amount >= 0:
            continue
        elif label in _SPENDING_LABELS | {
            TransactionKind.REFUND.value,
            TransactionKind.INCOME.value,
        }:
            group.inconsistent_rows += 1
            if amount < 0:
                group.inconsistent_debits += -amount
        else:
            raise ValidationError(f"{label!r} is not a recognised transaction label")
    return groups


def _record(group: _CurrencyAccumulator, entry: SpendEntry, contribution: Decimal) -> None:
    assert entry.label is not None
    group.merchants[entry.merchant].append(
        SpendEvidence(
            entry.transaction_id, entry.posted_on, contribution, entry.label, entry.confirmed
        )
    )
