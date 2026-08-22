"""Repositories.

Domain-oriented persistence, not a thin wrapper over the ORM. There is a `save`
and there are reads; there is deliberately no `update`, because a completed
comparison run is immutable. Writing the same run twice raises rather than
overwriting — an audit record you can quietly replace is not an audit record.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from offerdelta.application.queries.get_demo_comparison import ComparisonView
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.domain.common.rounding import CURRENCY_DISPLAY
from offerdelta.domain.transactions.accounts import canonical_account_key
from offerdelta.domain.transactions.fingerprint import (
    FINGERPRINT_VERSION,
    compute_fingerprint,
)
from offerdelta.infrastructure.postgres.models import (
    AccountRow,
    ComparisonRunRow,
    ResultComponentRow,
    TransactionRow,
)
from offerdelta.ingest.commit import ImportPlan


@dataclass(frozen=True)
class StoredRun:
    """A run as it came back from the database."""

    id: uuid.UUID
    created_at: datetime
    engine_version: str
    rounding_policy: str
    current_label: str
    candidate_label: str
    horizon_months: int
    cash_delta: Money
    wealth_delta: Money
    reconciled: bool
    component_count: int


@dataclass(frozen=True)
class StoredTransaction:
    """An imported bank row as it came back from storage."""

    id: uuid.UUID
    imported_at: datetime
    account_id: uuid.UUID
    posted_on: date
    description: str
    normalised_merchant: str
    amount: Money
    fingerprint: str
    occurrence: int
    source_file: str | None
    source_line: int | None
    raw_cells: dict[str, str] | None


@dataclass(frozen=True)
class AlreadyStoredTransaction:
    """A source row that matched one already stored for this account."""

    source_line: int
    fingerprint: str
    occurrence: int


@dataclass(frozen=True)
class TransactionImportResult:
    """The complete, non-silent outcome of a transaction import."""

    attempted_count: int
    imported_ids: tuple[uuid.UUID, ...]
    already_stored: tuple[AlreadyStoredTransaction, ...]

    @property
    def imported_count(self) -> int:
        return len(self.imported_ids)

    @property
    def already_stored_count(self) -> int:
        return len(self.already_stored)


def _quantised(amount: Money) -> Money:
    """Persistence is a rounding boundary; the policy is recorded alongside."""
    return amount.quantize(CURRENCY_DISPLAY)


def _resolve_account(session: Session, name: str, *, now: datetime) -> uuid.UUID:
    """Find the account this import belongs to, registering it on first sight.

    Interim: Task 6 gives account registration an explicit step of its own.
    Until then the foreign key still has to point at a real row, and minting a
    fresh account per import would defeat the very constraint it feeds.
    """
    key = canonical_account_key(name)
    existing = session.scalar(select(AccountRow.id).where(AccountRow.key == key))
    if existing is not None:
        return existing

    identifier = uuid.uuid4()
    session.add(AccountRow(id=identifier, key=key, display_name=name.strip(), created_at=now))
    session.flush()
    return identifier


class ComparisonRunRepository:
    """Stores and retrieves completed comparison runs."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def save(
        self,
        view: ComparisonView,
        *,
        engine_version: str,
        request_fingerprint: str,
        run_id: uuid.UUID | None = None,
        now: datetime | None = None,
    ) -> uuid.UUID:
        """Write a run and its breakdown in one transaction.

        Raises if the run already exists. Overwriting would mean a figure
        someone acted on could change afterwards without trace.
        """
        comparison = view.comparison
        reconciled = all(
            month.residual.is_zero()
            for side in (comparison.current, comparison.candidate)
            for month in side.months
        )
        if not reconciled:
            # Belt and braces: the engine already refuses to return an
            # unbalanced result, so reaching here means something upstream
            # changed. Storing it would put a number nobody should trust into
            # the permanent record.
            raise ValidationError("refusing to store a run whose months do not reconcile")

        identifier = run_id or uuid.uuid4()

        # Checked explicitly for a clear error, and the primary key still backs
        # it up: this read cannot see a row a concurrent writer has not
        # committed, so the constraint below is the guarantee and this is the
        # message.
        if self._session.get(ComparisonRunRow, identifier) is not None:
            raise ValidationError(
                f"comparison run {identifier} already exists; completed runs are immutable"
            )

        row = ComparisonRunRow(
            id=identifier,
            created_at=now or datetime.now(UTC),
            engine_version=engine_version,
            rounding_policy=CURRENCY_DISPLAY.name,
            current_label=view.current_label,
            candidate_label=view.candidate_label,
            horizon_months=view.horizon_months,
            move_date=None,
            request_fingerprint=request_fingerprint,
            currency=comparison.cash_delta.currency,
            cash_delta=_quantised(comparison.cash_delta).amount,
            wealth_delta=_quantised(comparison.wealth_delta).amount,
            time_delta_hours=comparison.time_delta_hours,
            reconciled=reconciled,
            components=[
                ResultComponentRow(
                    id=uuid.uuid4(),
                    position=position,
                    code=component.code,
                    label=component.label,
                    currency=component.delta.currency,
                    current_cash=_quantised(component.current_cash).amount,
                    candidate_cash=_quantised(component.candidate_cash).amount,
                    delta=_quantised(component.delta).amount,
                )
                for position, component in enumerate(comparison.component_deltas)
            ],
        )

        self._session.add(row)
        try:
            self._session.flush()
        except IntegrityError as error:
            raise ValidationError(
                f"comparison run {identifier} already exists; completed runs are immutable"
            ) from error
        return identifier

    def get(self, run_id: uuid.UUID) -> StoredRun | None:
        row = self._session.get(ComparisonRunRow, run_id)
        return None if row is None else _to_stored(row)

    def recent(self, limit: int = 20) -> list[StoredRun]:
        rows = self._session.scalars(
            select(ComparisonRunRow).order_by(ComparisonRunRow.created_at.desc()).limit(limit)
        ).all()
        return [_to_stored(row) for row in rows]

    def count(self) -> int:
        return len(self._session.scalars(select(ComparisonRunRow.id)).all())


class TransactionRepository:
    """Stores inspected bank rows without collapsing real duplicate charges."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def import_plan(
        self,
        plan: ImportPlan,
        *,
        now: datetime | None = None,
    ) -> TransactionImportResult:
        """Write new occurrence identities and report every stored match.

        The unique constraint is the final concurrency guard. The read first is
        what turns an ordinary re-import into a useful result instead of an
        exception; a genuinely concurrent import still fails the whole unit of
        work rather than leaving a partial batch behind.
        """
        # Interim: Task 6 replaces this method with a record-based write path
        # that carries a batch and an external id. What it must not do in the
        # meantime is stamp a fingerprint_version onto a hash no version of
        # the fingerprint ever produced, so identity comes from the domain
        # function the column is named after.
        imported_at = now or datetime.now(UTC)
        account_id = _resolve_account(self._session, plan.account, now=imported_at)
        keys = [
            compute_fingerprint(
                account_id=account_id,
                posted_on=planned.row.posted_on,
                normalised_merchant=planned.row.normalised_merchant,
                amount=planned.row.amount,
            )
            for planned in plan.rows
        ]
        existing = set(
            self._session.execute(
                select(TransactionRow.fingerprint, TransactionRow.occurrence).where(
                    TransactionRow.account_id == account_id,
                    TransactionRow.fingerprint.in_(set(keys)),
                )
            ).all()
        )

        imported_ids: list[uuid.UUID] = []
        already_stored: list[AlreadyStoredTransaction] = []

        for key, planned in zip(keys, plan.rows, strict=True):
            row = planned.row
            identity = (key, planned.occurrence)
            if identity in existing:
                already_stored.append(
                    AlreadyStoredTransaction(
                        source_line=row.line,
                        fingerprint=key,
                        occurrence=planned.occurrence,
                    )
                )
                continue

            identifier = uuid.uuid4()
            imported_ids.append(identifier)
            self._session.add(
                TransactionRow(
                    id=identifier,
                    imported_at=imported_at,
                    account_id=account_id,
                    batch_id=None,
                    posted_on=row.posted_on,
                    description=row.description,
                    normalised_merchant=row.normalised_merchant,
                    currency=row.amount.currency,
                    amount=_quantised(row.amount).amount,
                    external_id=None,
                    fingerprint=key,
                    fingerprint_version=FINGERPRINT_VERSION,
                    occurrence=planned.occurrence,
                    source_file=plan.source_file,
                    source_line=row.line,
                    raw_cells=dict(row.raw),
                )
            )

        if imported_ids:
            try:
                self._session.flush()
            except IntegrityError as error:
                raise ValidationError(
                    "transaction import conflicted with another import; retry so stored "
                    "duplicates can be reported safely"
                ) from error

        return TransactionImportResult(
            attempted_count=len(plan.rows),
            imported_ids=tuple(imported_ids),
            already_stored=tuple(already_stored),
        )

    def get(self, transaction_id: uuid.UUID) -> StoredTransaction | None:
        row = self._session.get(TransactionRow, transaction_id)
        return None if row is None else _to_stored_transaction(row)

    def recent(self, *, account_id: uuid.UUID, limit: int = 20) -> list[StoredTransaction]:
        rows = self._session.scalars(
            select(TransactionRow)
            .where(TransactionRow.account_id == account_id)
            .order_by(TransactionRow.posted_on.desc(), TransactionRow.id)
            .limit(limit)
        ).all()
        return [_to_stored_transaction(row) for row in rows]

    def count(self, *, account_id: uuid.UUID | None = None) -> int:
        statement = select(TransactionRow.id)
        if account_id is not None:
            statement = statement.where(TransactionRow.account_id == account_id)
        return len(self._session.scalars(statement).all())


def _to_stored(row: ComparisonRunRow) -> StoredRun:
    return StoredRun(
        id=row.id,
        created_at=row.created_at,
        engine_version=row.engine_version,
        rounding_policy=row.rounding_policy,
        current_label=row.current_label,
        candidate_label=row.candidate_label,
        horizon_months=row.horizon_months,
        cash_delta=Money(row.cash_delta, row.currency),
        wealth_delta=Money(row.wealth_delta, row.currency),
        reconciled=row.reconciled,
        component_count=len(row.components),
    )


def _to_stored_transaction(row: TransactionRow) -> StoredTransaction:
    return StoredTransaction(
        id=row.id,
        imported_at=row.imported_at,
        account_id=row.account_id,
        posted_on=row.posted_on,
        description=row.description,
        normalised_merchant=row.normalised_merchant,
        amount=Money(row.amount, row.currency),
        fingerprint=row.fingerprint,
        occurrence=row.occurrence,
        source_file=row.source_file,
        source_line=row.source_line,
        raw_cells=None if row.raw_cells is None else dict(row.raw_cells),
    )
