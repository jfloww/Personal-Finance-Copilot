"""Repositories.

Domain-oriented persistence, not a thin wrapper over the ORM. There is a `save`
and there are reads; there is deliberately no `update`, because a completed
comparison run is immutable. Writing the same run twice raises rather than
overwriting — an audit record you can quietly replace is not an audit record.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy import func, select
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
    ImportBatchRow,
    ResultComponentRow,
    TransactionRow,
)
from offerdelta.infrastructure.postgres.records import TransactionRecord


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
class StoredAccount:
    """A registered account."""

    id: uuid.UUID
    key: str
    display_name: str
    created_at: datetime


@dataclass(frozen=True)
class StoredBatch:
    """One recorded import of one file."""

    id: uuid.UUID
    account_id: uuid.UUID
    source_file: str
    source_sha256: str
    mode: str
    window_start: date | None
    window_end: date | None
    row_count: int
    imported_at: datetime


@dataclass(frozen=True)
class StoredTransaction:
    """An imported bank row as it came back from storage."""

    id: uuid.UUID
    imported_at: datetime
    account_id: uuid.UUID
    batch_id: uuid.UUID | None
    posted_on: date
    description: str
    normalised_merchant: str
    amount: Money
    external_id: str | None
    fingerprint: str
    fingerprint_version: int
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


class AccountRepository:
    """Accounts exist because somebody registered them, never by accident.

    An import against an unknown account is refused rather than auto-creating
    one: auto-creation relocates the original bug instead of fixing it, since a
    typo still silently produces a second parallel account.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def register(self, display_name: str, *, now: datetime | None = None) -> StoredAccount:
        key = canonical_account_key(display_name)
        if self.by_key(key) is not None:
            raise ValidationError(f"account {key!r} is already registered")
        row = AccountRow(
            id=uuid.uuid4(),
            key=key,
            display_name=display_name.strip(),
            created_at=now or datetime.now(UTC),
        )
        self._session.add(row)
        try:
            self._session.flush()
        except IntegrityError as error:
            raise ValidationError(f"account {key!r} is already registered") from error
        return _to_stored_account(row)

    def by_key(self, key: str) -> StoredAccount | None:
        row = self._session.scalars(
            select(AccountRow).where(AccountRow.key == canonical_account_key(key))
        ).one_or_none()
        return None if row is None else _to_stored_account(row)

    def all(self) -> list[StoredAccount]:
        rows = self._session.scalars(select(AccountRow).order_by(AccountRow.key)).all()
        return [_to_stored_account(row) for row in rows]


def _to_stored_account(row: AccountRow) -> StoredAccount:
    return StoredAccount(
        id=row.id, key=row.key, display_name=row.display_name, created_at=row.created_at
    )


class ImportBatchRepository:
    """Batches make a re-import of the same bytes provably a no-op."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def open(
        self,
        account_id: uuid.UUID,
        *,
        source_file: str,
        source_sha256: str,
        mode: str,
        window_start: date | None,
        window_end: date | None,
        row_count: int,
        now: datetime | None = None,
    ) -> tuple[StoredBatch, bool]:
        """Return the batch and whether it was newly created.

        Two callers racing the same `(account_id, source_sha256)` can both
        pass the SELECT below before either has inserted; `_insert` is what
        survives that.
        """
        existing = self._existing(account_id, source_sha256)
        if existing is not None:
            return _to_stored_batch(existing), False

        return self._insert(
            account_id,
            source_file=source_file,
            source_sha256=source_sha256,
            mode=mode,
            window_start=window_start,
            window_end=window_end,
            row_count=row_count,
            now=now,
        )

    def _insert(
        self,
        account_id: uuid.UUID,
        *,
        source_file: str,
        source_sha256: str,
        mode: str,
        window_start: date | None,
        window_end: date | None,
        row_count: int,
        now: datetime | None,
    ) -> tuple[StoredBatch, bool]:
        """Create the row, or recover if another writer already has.

        Called with no existence check of its own, so this is also the
        losing side of the race `open()` exists to survive: two callers can
        both reach here for the same `(account_id, source_sha256)` after each
        passed its own SELECT, and the unique constraint then lets only one
        INSERT through. The loser recovers here instead of surfacing the raw
        `IntegrityError`, so a race still ends in the documented "existing
        batch, `created=False`" rather than a crash.
        """
        row = ImportBatchRow(
            id=uuid.uuid4(),
            account_id=account_id,
            source_file=source_file,
            source_sha256=source_sha256,
            mode=mode,
            window_start=window_start,
            window_end=window_end,
            row_count=row_count,
            imported_at=now or datetime.now(UTC),
        )
        try:
            # Scoped to a SAVEPOINT so a losing racer only unwinds this
            # INSERT, not whatever else the caller's session may hold
            # pending - the caller's unit of work is theirs to roll back,
            # not this repository's to guess at.
            with self._session.begin_nested():
                self._session.add(row)
                self._session.flush()
        except IntegrityError:
            existing = self._existing(account_id, source_sha256)
            if existing is None:
                # The constraint fired for a reason other than the race this
                # handles; the caller's own bug is more useful than a
                # swallowed exception.
                raise
            return _to_stored_batch(existing), False
        return _to_stored_batch(row), True

    def _existing(self, account_id: uuid.UUID, source_sha256: str) -> ImportBatchRow | None:
        return self._session.scalars(
            select(ImportBatchRow).where(
                ImportBatchRow.account_id == account_id,
                ImportBatchRow.source_sha256 == source_sha256,
            )
        ).one_or_none()


def _to_stored_batch(row: ImportBatchRow) -> StoredBatch:
    return StoredBatch(
        id=row.id,
        account_id=row.account_id,
        source_file=row.source_file,
        source_sha256=row.source_sha256,
        mode=row.mode,
        window_start=row.window_start,
        window_end=row.window_end,
        row_count=row.row_count,
        imported_at=row.imported_at,
    )


class TransactionRepository:
    """Stores inspected bank rows without collapsing real duplicate charges."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def add_many(
        self,
        records: Sequence[TransactionRecord],
        *,
        batch_id: uuid.UUID | None = None,
        now: datetime | None = None,
    ) -> TransactionImportResult:
        """Write new identities and report every stored match.

        The read-first is what turns an ordinary re-import into a useful report
        instead of an exception; the unique constraint remains the final
        concurrency guard.

        Every record must belong to the same account. Deduplication below reads
        existing fingerprints and external ids scoped to one account_id, so a
        mixed batch would check records from every account but the first
        against the wrong account's history and silently miss real duplicates.
        """
        if not records:
            return TransactionImportResult(attempted_count=0, imported_ids=(), already_stored=())

        accounts = {record.account_id for record in records}
        if len(accounts) > 1:
            raise ValidationError(
                f"add_many writes one account at a time; got {len(accounts)}. "
                f"Deduplication is scoped to a single account, so a mixed batch "
                f"would silently miss duplicates in all but the first."
            )

        account_id = records[0].account_id
        fingerprints = {self._fingerprint(record) for record in records}
        existing = set(
            self._session.execute(
                select(TransactionRow.fingerprint, TransactionRow.occurrence).where(
                    TransactionRow.account_id == account_id,
                    TransactionRow.fingerprint.in_(fingerprints),
                )
            ).all()
        )
        existing_external = set(
            self._session.scalars(
                select(TransactionRow.external_id).where(
                    TransactionRow.account_id == account_id,
                    TransactionRow.external_id.is_not(None),
                )
            ).all()
        )

        # Incremental records carry a bank-supplied external_id, which is what
        # actually decides identity below. Their occurrence is only numbered
        # per file (starting at 1 again), so a genuinely new row can land on
        # an occurrence already taken by a previously stored row with the
        # same fingerprint and trip the unique constraint even though it is
        # not a duplicate. Offsetting by the highest occurrence already
        # stored for that fingerprint is safe only because external_id, not
        # occurrence, is doing the deduplication here - occurrence only has
        # to satisfy the constraint, not carry identity. Snapshot batches
        # never carry an external_id, so this offset never applies to them.
        offsets: dict[str, int] = {}
        if any(record.external_id is not None for record in records):
            rows = self._session.execute(
                select(
                    TransactionRow.fingerprint,
                    func.max(TransactionRow.occurrence),
                )
                .where(
                    TransactionRow.account_id == account_id,
                    TransactionRow.fingerprint.in_(fingerprints),
                )
                .group_by(TransactionRow.fingerprint)
            ).all()
            offsets = {fingerprint: highest for fingerprint, highest in rows}  # noqa: C416

        imported_ids: list[uuid.UUID] = []
        already_stored: list[AlreadyStoredTransaction] = []
        imported_at = now or datetime.now(UTC)

        for record in records:
            fingerprint = self._fingerprint(record)
            if record.external_id is not None:
                seen = record.external_id in existing_external
            else:
                seen = (fingerprint, record.occurrence) in existing

            if seen:
                already_stored.append(
                    AlreadyStoredTransaction(
                        source_line=record.provenance.source_line if record.provenance else 0,
                        fingerprint=fingerprint,
                        occurrence=record.occurrence,
                    )
                )
                continue

            occurrence = record.occurrence + offsets.get(fingerprint, 0)
            identifier = uuid.uuid4()
            imported_ids.append(identifier)
            quantised = _quantised(record.amount)
            self._session.add(
                TransactionRow(
                    id=identifier,
                    imported_at=imported_at,
                    account_id=record.account_id,
                    batch_id=batch_id,
                    posted_on=record.posted_on,
                    description=record.description,
                    normalised_merchant=record.normalised_merchant,
                    currency=quantised.currency,
                    amount=quantised.amount,
                    external_id=record.external_id,
                    fingerprint=fingerprint,
                    fingerprint_version=FINGERPRINT_VERSION,
                    occurrence=occurrence,
                    source_file=record.provenance.source_file if record.provenance else None,
                    source_line=record.provenance.source_line if record.provenance else None,
                    raw_cells=dict(record.provenance.raw_cells) if record.provenance else None,
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
            attempted_count=len(records),
            imported_ids=tuple(imported_ids),
            already_stored=tuple(already_stored),
        )

    @staticmethod
    def _fingerprint(record: TransactionRecord) -> str:
        return compute_fingerprint(
            account_id=record.account_id,
            posted_on=record.posted_on,
            normalised_merchant=record.normalised_merchant,
            amount=record.amount,
        )

    def get(self, transaction_id: uuid.UUID) -> StoredTransaction | None:
        row = self._session.get(TransactionRow, transaction_id)
        return None if row is None else _to_stored_transaction(row)

    def count(self, *, account_id: uuid.UUID | None = None) -> int:
        statement = select(func.count()).select_from(TransactionRow)
        if account_id is not None:
            statement = statement.where(TransactionRow.account_id == account_id)
        return self._session.scalars(statement).one()


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
        batch_id=row.batch_id,
        posted_on=row.posted_on,
        description=row.description,
        normalised_merchant=row.normalised_merchant,
        amount=Money(row.amount, row.currency),
        external_id=row.external_id,
        fingerprint=row.fingerprint,
        fingerprint_version=row.fingerprint_version,
        occurrence=row.occurrence,
        source_file=row.source_file,
        source_line=row.source_line,
        raw_cells=dict(row.raw_cells) if row.raw_cells is not None else None,
    )
