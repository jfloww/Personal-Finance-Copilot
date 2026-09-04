"""Repositories.

Domain-oriented persistence, not a thin wrapper over the ORM. There is a `save`
and there are reads; there is deliberately no `update`, because a completed
comparison run is immutable. Writing the same run twice raises rather than
overwriting — an audit record you can quietly replace is not an audit record.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Final

from argon2.exceptions import InvalidHashError
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from offerdelta.application.queries.get_demo_comparison import ComparisonView
from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.domain.common.rounding import CURRENCY_DISPLAY
from offerdelta.domain.transactions.accounts import canonical_account_key
from offerdelta.domain.transactions.fingerprint import (
    FINGERPRINT_VERSION,
    compute_fingerprint,
)
from offerdelta.domain.users.identity import normalise_email
from offerdelta.evaluation.labels import ABSTAIN, LABEL_SPACE
from offerdelta.infrastructure.auth.passwords import hash_password, verify_password
from offerdelta.infrastructure.postgres.models import (
    AccountRow,
    ComparisonRunRow,
    ImportBatchRow,
    ResultComponentRow,
    TransactionRow,
    UserRow,
)
from offerdelta.records.transactions import TransactionRecord

logger = logging.getLogger(__name__)


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

    #: What a categoriser proposed, and who or what proposed it. NULL means
    #: never examined; the literal 'UNKNOWN' means examined and declined -
    #: see `TransactionRow` for why those are not the same fact.
    suggested_label: str | None
    suggested_source: str | None
    suggested_confidence: Decimal | None
    suggested_by: str | None

    #: A person's decision. Outranks `suggested_label` structurally, via
    #: `effective_label`, rather than by a caller remembering to check it.
    confirmed_label: str | None

    @property
    def effective_label(self) -> str | None:
        """The label a report should use: a person's word over a model's guess."""
        return self.confirmed_label or self.suggested_label


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


@dataclass(frozen=True)
class StoredUser:
    """A user as it came back from the database.

    `has_password` rather than the hash itself: nothing outside
    `UserRepository` needs the raw hash, and a type that cannot carry it
    cannot leak it into a log line or a response by accident.
    """

    id: uuid.UUID
    email: str
    display_name: str
    is_active: bool
    has_password: bool


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


class _ScopedRepository:
    """Everything a repository needs to answer for exactly one tenant.

    The constructor takes a `TenantScope` where it used to take a `Session`,
    so a query without a tenant is not something a caller can forget to
    write - it is something they cannot express. Subclasses still have to
    put `user_id` in each `WHERE`; what this removes is the case where the
    identity was never in scope to filter on at all.
    """

    def __init__(self, scope: TenantScope) -> None:
        self._scope = scope
        self._session = scope.session

    @property
    def _user_id(self) -> uuid.UUID:
        return self._scope.user.id

    def _require_own_account(self, account_id: uuid.UUID) -> None:
        """An account id from a caller is only usable if this tenant owns it.

        The alternative is chain of custody - the id came from a scoped
        `AccountRepository`, so it must be safe. That is true today and it is
        an argument about call sites; it stops being true the first time a
        route accepts an account id from a request body. The application
        layer is the only thing enforcing tenancy here, so the property must
        not rest on an argument. The cost is one indexed primary-key lookup.

        Deliberately the same message for "no such account" and "somebody
        else's account": telling the caller which one it is confirms the
        existence of a row they are not allowed to see.
        """
        owned = self._session.scalars(
            select(AccountRow.id).where(
                AccountRow.id == account_id,
                AccountRow.user_id == self._user_id,
            )
        ).one_or_none()
        if owned is None:
            raise ValidationError(f"no account {account_id}")

    def _require_own_batch(self, batch_id: uuid.UUID) -> None:
        """A batch id from a caller is only usable if this tenant owns it.

        Mirrors `_require_own_account` exactly, for the same reason:
        `TransactionRepository.add_many` writes `batch_id` onto every row it
        inserts, and the foreign key alone is satisfied by *any* batch, so
        without this a caller who owns the account could still tag its rows
        onto somebody else's import history. No entry point supplies a batch
        id today - "nothing calls it that way" is the exact argument this
        whole phase refuses to rest on, so the check exists before a caller
        does rather than after one is found.
        """
        owned = self._session.scalars(
            select(ImportBatchRow.id)
            .join(AccountRow, AccountRow.id == ImportBatchRow.account_id)
            .where(
                ImportBatchRow.id == batch_id,
                AccountRow.user_id == self._user_id,
            )
        ).one_or_none()
        if owned is None:
            raise ValidationError(f"no batch {batch_id}")

    def _require_own_transaction(self, transaction_id: uuid.UUID) -> TransactionRow:
        """A transaction id from a caller is only usable if this tenant owns it.

        Mirrors `_require_own_account` exactly, for the same reason:
        `record_suggestion` and `confirm_label` both write onto a row named
        by an id from outside this repository, and a primary key alone
        answers for any tenant's row, not just this one's. Without this, an
        id that reached a route from a request body could confirm a label
        onto - or read the provenance of - somebody else's transaction.

        Returns the row itself, fetched through this tenant-filtered query,
        rather than just confirming it exists. A caller that mutates it
        afterward then never needs a second, unscoped `session.get` by bare
        primary key - which would only be safe again by chain-of-custody
        reasoning ("nothing runs between the check and the fetch"), the
        exact argument this whole layer refuses to rest on.

        Deliberately the same message for "no such transaction" and
        "somebody else's transaction", for the same reason
        `_require_own_account` gives: telling the caller which one it is
        confirms the existence of a row they are not allowed to see.
        """
        owned = self._session.scalars(
            select(TransactionRow)
            .join(AccountRow, AccountRow.id == TransactionRow.account_id)
            .where(
                TransactionRow.id == transaction_id,
                AccountRow.user_id == self._user_id,
            )
        ).one_or_none()
        if owned is None:
            raise ValidationError(f"no transaction {transaction_id}")
        return owned


class AccountRepository(_ScopedRepository):
    """Accounts exist because somebody registered them, never by accident.

    An import against an unknown account is refused rather than auto-creating
    one: auto-creation relocates the original bug instead of fixing it, since a
    typo still silently produces a second parallel account.

    Every read here filters on the scope's user. `key` is unique per user
    rather than globally (`uq_accounts_user_key`), so an unfiltered read of a
    key would return whichever tenant's row the database happened to hand
    back first - and `one_or_none` would raise once two tenants had both
    registered the same bank.
    """

    def register(self, display_name: str, *, now: datetime | None = None) -> StoredAccount:
        key = canonical_account_key(display_name)
        if self.by_key(key) is not None:
            raise ValidationError(f"account {key!r} is already registered")
        row = AccountRow(
            id=uuid.uuid4(),
            user_id=self._user_id,
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
            select(AccountRow).where(
                AccountRow.user_id == self._user_id,
                AccountRow.key == canonical_account_key(key),
            )
        ).one_or_none()
        return None if row is None else _to_stored_account(row)

    def all(self) -> list[StoredAccount]:
        rows = self._session.scalars(
            select(AccountRow).where(AccountRow.user_id == self._user_id).order_by(AccountRow.key)
        ).all()
        return [_to_stored_account(row) for row in rows]


def _to_stored_account(row: AccountRow) -> StoredAccount:
    return StoredAccount(
        id=row.id, key=row.key, display_name=row.display_name, created_at=row.created_at
    )


class ImportBatchRepository(_ScopedRepository):
    """Batches make a re-import of the same bytes provably a no-op."""

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

        The account id arrives from the caller, so ownership is verified here
        before anything is read or written against it - see
        `_require_own_account`. Without that, an id from a request body would
        open a batch, and then transactions, inside somebody else's account.

        Two callers racing the same `(account_id, source_sha256)` can both
        pass the SELECT below before either has inserted; `_insert` is what
        survives that.
        """
        self._require_own_account(account_id)

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

        Checks ownership itself rather than inheriting `open`'s check. This
        is the only method in the layer that writes a row against an
        `account_id` it was handed, and the foreign key alone is satisfied by
        *any* account, so without this a call here would file a batch - and
        then, through `batch_id`, transactions - inside somebody else's
        account. "`open` is the only caller and it already checked" is an
        argument about call sites, which is the argument this whole change
        exists to stop relying on: one exception is the difference between
        "we scope our queries" and "an unscoped query cannot be written".
        The cost is one indexed lookup, once per import.

        Called with no *existence* check of its own, so this is also the
        losing side of the race `open()` exists to survive: two callers can
        both reach here for the same `(account_id, source_sha256)` after each
        passed its own SELECT, and the unique constraint then lets only one
        INSERT through. The loser recovers here instead of surfacing the raw
        `IntegrityError`, so a race still ends in the documented "existing
        batch, `created=False`" rather than a crash.
        """
        self._require_own_account(account_id)

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
        """The tenant filter is repeated here rather than inherited from `open`.

        `open` has already refused a foreign account id, so the join is
        redundant on that path - and it is the only thing standing between
        this method and another tenant's batch on any path added later. A
        query that is safe only because of what its caller checked is a query
        whose safety is not written down anywhere it can be read.
        """
        return self._session.scalars(
            select(ImportBatchRow)
            .join(AccountRow, AccountRow.id == ImportBatchRow.account_id)
            .where(
                ImportBatchRow.account_id == account_id,
                AccountRow.user_id == self._user_id,
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


class TransactionRepository(_ScopedRepository):
    """Stores inspected bank rows without collapsing real duplicate charges."""

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

        The account id comes off the records, which come from the caller, so
        ownership is verified before any of it is trusted - see
        `_require_own_account`. Every read below then repeats the tenant
        filter through a join to `accounts`, so no query here depends on that
        check having happened first.

        `batch_id` is the same kind of vector and gets the same check, via
        `_require_own_batch`: it is written onto every inserted row with no
        constraint of its own to stop it naming somebody else's import
        history, so an id from a caller that owns the account but not the
        batch must still be refused rather than silently accepted.

        The caller must also not mix identity schemes on one account. Records
        with an ``external_id`` are deduplicated against stored external ids
        only, and rows written without one are invisible to that check - so
        importing a charge incrementally that was already stored from a
        snapshot writes it twice, silently. ``import_csv`` refuses a mode change
        per account for exactly this reason; a caller reaching this method
        directly inherits the obligation, not the protection.
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
        self._require_own_account(account_id)
        if batch_id is not None:
            self._require_own_batch(batch_id)

        fingerprints = {self._fingerprint(record) for record in records}
        existing = set(
            self._session.execute(
                select(TransactionRow.fingerprint, TransactionRow.occurrence)
                .join(AccountRow, AccountRow.id == TransactionRow.account_id)
                .where(
                    TransactionRow.account_id == account_id,
                    AccountRow.user_id == self._user_id,
                    TransactionRow.fingerprint.in_(fingerprints),
                )
            ).all()
        )
        existing_external = set(
            self._session.scalars(
                select(TransactionRow.external_id)
                .join(AccountRow, AccountRow.id == TransactionRow.account_id)
                .where(
                    TransactionRow.account_id == account_id,
                    AccountRow.user_id == self._user_id,
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
                .join(AccountRow, AccountRow.id == TransactionRow.account_id)
                .where(
                    TransactionRow.account_id == account_id,
                    AccountRow.user_id == self._user_id,
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

            # Offsetting only makes sense for a record whose identity is its
            # external_id: occurrence there only has to satisfy the unique
            # constraint. For an id-less record, occurrence *is* its identity
            # (paired with fingerprint), so shifting it would store the wrong
            # identity outright, opening a gap a later real charge could fall
            # into and be written a second time.
            occurrence = record.occurrence + (
                offsets.get(fingerprint, 0) if record.external_id is not None else 0
            )
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
        """`None` for another tenant's row, exactly as for one that does not exist.

        This was `session.get(TransactionRow, id)`, which answers for any row
        in the table: a primary key is an address, not an authorisation, and
        an id that reached a URL would have been enough to read somebody
        else's charge. Not-found and not-yours are the same answer here so
        that the response cannot be used to confirm a row exists.
        """
        row = self._session.scalars(
            select(TransactionRow)
            .join(AccountRow, AccountRow.id == TransactionRow.account_id)
            .where(
                TransactionRow.id == transaction_id,
                AccountRow.user_id == self._user_id,
            )
        ).one_or_none()
        return None if row is None else _to_stored_transaction(row)

    def count(self, *, account_id: uuid.UUID | None = None) -> int:
        """How many of *this tenant's* transactions, not how many exist.

        The no-account form is the easy one to leave untenanted: it returns a
        plausible number either way, so nothing about the result says it was
        counting the whole table.
        """
        statement = (
            select(func.count())
            .select_from(TransactionRow)
            .join(AccountRow, AccountRow.id == TransactionRow.account_id)
            .where(AccountRow.user_id == self._user_id)
        )
        if account_id is not None:
            statement = statement.where(TransactionRow.account_id == account_id)
        return self._session.scalars(statement).one()

    def unclassified(self, limit: int | None = None) -> list[StoredTransaction]:
        """Rows a categoriser has never looked at - not rows it declined.

        Filters on ``suggested_label IS NULL`` alone. A row a categoriser
        already examined and abstained on carries ``suggested_label =
        'UNKNOWN'``, which is a different fact - "looked and declined" - and
        belongs in `awaiting_review`, not back in a queue meant for rows
        nobody has run a model over yet. Collapsing the two here would mean
        an abstained row gets re-classified every run forever, which is the
        opposite of what abstention is for.
        """
        statement = (
            select(TransactionRow)
            .join(AccountRow, AccountRow.id == TransactionRow.account_id)
            .where(
                AccountRow.user_id == self._user_id,
                TransactionRow.suggested_label.is_(None),
            )
            .order_by(TransactionRow.posted_on, TransactionRow.id)
        )
        if limit is not None:
            statement = statement.limit(limit)
        rows = self._session.scalars(statement).all()
        return [_to_stored_transaction(row) for row in rows]

    def record_suggestion(
        self,
        transaction_id: uuid.UUID,
        *,
        label: str,
        source: str,
        confidence: Decimal,
        by: str,
        now: datetime | None = None,
    ) -> None:
        """Write a categoriser's guess, without ever touching a person's decision.

        `transaction_id` arrives from a caller, so ownership is checked first
        via `_require_own_transaction` - the same reason every other method
        here that takes a bare id checks it before writing or reading through
        it.

        Re-classification calls this freely, including over a row a person
        already confirmed: `confirmed_label` is a separate column that only
        `confirm_label` ever writes, so overwriting `suggested_label` here
        can never erase a confirmation. That is what makes "a person's
        decision survives re-classification" structural rather than a rule
        every caller has to remember.
        """
        row = self._require_own_transaction(transaction_id)
        self._require_label(label)
        row.suggested_label = label
        row.suggested_source = source
        row.suggested_confidence = confidence
        row.suggested_by = by
        row.suggested_at = now or datetime.now(UTC)
        self._session.flush()

    def confirm_label(
        self, transaction_id: uuid.UUID, label: str, *, now: datetime | None = None
    ) -> None:
        """Record a person's decision. Never writes `suggested_label`.

        `transaction_id` arrives from a caller, so ownership is checked
        first via `_require_own_transaction`, exactly as in
        `record_suggestion`.

        Writing only `confirmed_label` (never `suggested_label`) is what
        lets `record_suggestion` be called again later, by a later
        re-classification run, without a caller on either side having to
        coordinate to protect this row - see `record_suggestion`.
        """
        row = self._require_own_transaction(transaction_id)
        self._require_label(label)
        row.confirmed_label = label
        row.confirmed_at = now or datetime.now(UTC)
        self._session.flush()

    def for_month(self, year: int, month: int) -> list[StoredTransaction]:
        """This tenant's transactions posted in one calendar month."""
        start, end = _month_bounds(year, month)
        rows = self._session.scalars(
            select(TransactionRow)
            .join(AccountRow, AccountRow.id == TransactionRow.account_id)
            .where(
                AccountRow.user_id == self._user_id,
                TransactionRow.posted_on >= start,
                TransactionRow.posted_on < end,
            )
            .order_by(TransactionRow.posted_on, TransactionRow.id)
        ).all()
        return [_to_stored_transaction(row) for row in rows]

    def awaiting_review(
        self, threshold: Decimal, *, month: tuple[int, int] | None = None
    ) -> list[StoredTransaction]:
        """Rows nobody has confirmed and no categoriser confidently resolved.

        A row belongs here when `confirmed_label IS NULL` - a person's
        decision always takes a row out of the queue, see `confirm_label` -
        and, on the suggestion side, any of three separate facts hold:
        never examined (`suggested_label IS NULL`), examined and declined
        (`suggested_label = 'UNKNOWN'`), or examined and unsure
        (`suggested_confidence < threshold`). These are kept as an `OR`
        rather than folded into one condition because a report built on top
        of this (a later task) shows them as distinct leaves - collapsing
        them here would throw away the distinction before it ever reaches
        that report.

        Ordered by `posted_on` descending: the most recent uncertain
        spending is what a person reviewing the queue wants to see first.
        `id` breaks ties on the same date, as `unclassified` and `for_month`
        already do - without it, two rows sharing a date could come back in
        either order on different calls, and a person paging through the
        queue could see one row twice and never see the other.
        """
        conditions = [
            AccountRow.user_id == self._user_id,
            TransactionRow.confirmed_label.is_(None),
            or_(
                TransactionRow.suggested_label.is_(None),
                TransactionRow.suggested_label == ABSTAIN,
                TransactionRow.suggested_confidence < threshold,
            ),
        ]
        if month is not None:
            year, mon = month
            start, end = _month_bounds(year, mon)
            conditions.append(TransactionRow.posted_on >= start)
            conditions.append(TransactionRow.posted_on < end)
        rows = self._session.scalars(
            select(TransactionRow)
            .join(AccountRow, AccountRow.id == TransactionRow.account_id)
            .where(*conditions)
            .order_by(TransactionRow.posted_on.desc(), TransactionRow.id)
        ).all()
        return [_to_stored_transaction(row) for row in rows]

    @staticmethod
    def _require_label(label: str) -> None:
        """A label a categoriser or a person supplies must be one the taxonomy knows.

        Guards against a typo or a stale model output writing a string into
        `suggested_label` or `confirmed_label` that no report downstream
        (grouped by `CostCategory`, see `offerdelta.evaluation.labels`) can
        ever match - a silently unreportable row rather than a loud failure
        at the point the bad label was about to be stored.
        """
        if label not in LABEL_SPACE:
            raise ValidationError(f"{label!r} is not a label in this taxonomy")


_DECEMBER: Final = 12


def _month_bounds(year: int, month: int) -> tuple[date, date]:
    """The half-open `[start, end)` range of calendar dates for one month."""
    start = date(year, month, 1)
    end = date(year + 1, 1, 1) if month == _DECEMBER else date(year, month + 1, 1)
    return start, end


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
        suggested_label=row.suggested_label,
        suggested_source=row.suggested_source,
        suggested_confidence=row.suggested_confidence,
        suggested_by=row.suggested_by,
        confirmed_label=row.confirmed_label,
    )


class UserRepository:
    """The one repository that is not tenant-scoped.

    It operates on the tenants themselves, so there is no outer tenant to
    scope it to. Everything else takes a `TenantScope`; this takes a bare
    `Session`, deliberately, because there is no user to scope the query to
    before this repository has answered who is asking.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(self, email: str, display_name: str, *, now: datetime | None = None) -> StoredUser:
        address = normalise_email(email)
        if self.by_email(address) is not None:
            raise ValidationError(f"user {address!r} already exists")
        row = UserRow(
            id=uuid.uuid4(),
            email=address,
            password_hash=None,
            display_name=display_name.strip(),
            is_active=True,
            created_at=now or datetime.now(UTC),
        )
        self._session.add(row)
        try:
            self._session.flush()
        except IntegrityError as error:
            # The read-then-write above already checks for this; this is the
            # concurrent-writer case that check cannot see, not the common
            # path - see `ImportBatchRepository._insert` for the same shape.
            raise ValidationError(f"user {address!r} already exists") from error
        return _to_stored_user(row)

    def by_email(self, email: str) -> StoredUser | None:
        row = self._row(email)
        return None if row is None else _to_stored_user(row)

    def by_id(self, user_id: uuid.UUID) -> StoredUser | None:
        row = self._session.get(UserRow, user_id)
        return None if row is None else _to_stored_user(row)

    def all(self) -> list[StoredUser]:
        rows = self._session.scalars(select(UserRow).order_by(UserRow.email)).all()
        return [_to_stored_user(row) for row in rows]

    def set_password(self, email: str, plain: str) -> None:
        row = self._require(email)
        row.password_hash = hash_password(plain)
        self._session.flush()

    def deactivate(self, email: str) -> None:
        row = self._require(email)
        row.is_active = False
        self._session.flush()

    def authenticate(self, email: str, plain: str) -> AuthenticatedUser | None:
        """`None` for every failure: unknown address, wrong password, or deactivated.

        `verify_password` runs a real verification even when there is no row
        or no stored hash, so an unknown address does not answer faster than a
        known one - see its docstring for why that matters.

        A stored hash that is not valid argon2 at all (as opposed to simply
        not matching) makes `verify_password` raise `InvalidHashError`. That
        can only happen for a row that exists and carries a non-null,
        corrupted `password_hash` - `password_hash IS NULL` already means
        "cannot log in" and never reaches the real verifier. A caller
        mistyping their password can neither cause nor fix a corrupted row,
        so this is caught here rather than in `verify_password`: logged with
        the user id (never the email - this repository is careful about what
        identifying data reaches a log) so the corruption is not silent, but
        still resolved to the same `None` every other failure returns. Letting
        it escape uncaught would answer 500 for that one address while every
        other failure answers the same way as a wrong password - a response
        shape that is itself an enumeration signal, the exact thing
        `verify_password`'s constant-work comparison exists to prevent.
        """
        row = self._row(email)
        hashed = row.password_hash if row is not None else None
        try:
            verified = verify_password(plain, hashed)
        except InvalidHashError:
            if row is not None:
                logger.error(
                    "authenticate: password_hash for user %s is not a valid argon2 hash",
                    row.id,
                )
            return None
        if not verified:
            return None
        if row is None or not row.is_active:
            return None
        return AuthenticatedUser(id=row.id, email=row.email)

    def _row(self, email: str) -> UserRow | None:
        return self._session.scalars(
            select(UserRow).where(UserRow.email == normalise_email(email))
        ).one_or_none()

    def _require(self, email: str) -> UserRow:
        row = self._row(email)
        if row is None:
            raise ValidationError(f"no user {normalise_email(email)!r}")
        return row


def _to_stored_user(row: UserRow) -> StoredUser:
    return StoredUser(
        id=row.id,
        email=row.email,
        display_name=row.display_name,
        is_active=row.is_active,
        has_password=row.password_hash is not None,
    )
