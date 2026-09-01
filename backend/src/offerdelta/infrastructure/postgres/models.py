"""Relational mapping.

Two rules govern this schema.

**Money is NUMERIC, never a floating-point type.** In this database
`0.1 + 0.2 = 0.3` is true for NUMERIC and false for `float8`; the whole product
rests on the first being the case.

**Persistence is a rounding boundary.** The engine keeps full precision through
every intermediate step, but a stored result is the figure a person was shown,
so amounts are quantised on the way in and the policy that did it is recorded
on the row. A run therefore says not just what the number was but how it was
presented.

Completed runs are immutable. The repository offers no update path, and the
primary key makes a second write of the same run fail rather than overwrite the
first.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Final

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

#: Eighteen digits with two decimal places. Comfortably beyond any salary, and
#: exact — NUMERIC stores decimal digits rather than approximating them.
MONEY = Numeric(18, 2)

#: Hours carry a finer scale than money: a commute is measured in minutes.
HOURS = Numeric(12, 4)


class Base(DeclarativeBase):
    pass


class ComparisonRunRow(Base):
    """One completed calculation, immutable once written."""

    __tablename__ = "comparison_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    #: Which version of the calculation rules produced this. A stored figure is
    #: meaningless without it: the same inputs under different rules are a
    #: different answer, and pretending otherwise is how an audit trail lies.
    engine_version: Mapped[str] = mapped_column(String(64))

    #: Names the policy that quantised every amount below.
    rounding_policy: Mapped[str] = mapped_column(String(64))

    current_label: Mapped[str] = mapped_column(String(200))
    candidate_label: Mapped[str] = mapped_column(String(200))

    horizon_months: Mapped[int] = mapped_column(Integer)
    move_date: Mapped[date | None] = mapped_column(Date, nullable=True)

    #: Digest of the request that produced this run, so an identical request can
    #: be recognised without re-running the engine.
    request_fingerprint: Mapped[str] = mapped_column(String(64), index=True)

    currency: Mapped[str] = mapped_column(String(3))
    cash_delta: Mapped[Decimal] = mapped_column(MONEY)
    wealth_delta: Mapped[Decimal] = mapped_column(MONEY)
    time_delta_hours: Mapped[Decimal] = mapped_column(HOURS)

    #: Whether every projected month balanced. The engine refuses to return an
    #: unbalanced result, so this is always true on write — stored because a
    #: claim nobody recorded is a claim nobody can audit later.
    reconciled: Mapped[bool] = mapped_column(Boolean)

    components: Mapped[list[ResultComponentRow]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="ResultComponentRow.position",
    )

    __table_args__ = (Index("ix_comparison_runs_created_at", "created_at"),)


class ResultComponentRow(Base):
    """One line of a run's breakdown."""

    __tablename__ = "result_components"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("comparison_runs.id", ondelete="CASCADE"), index=True
    )

    #: Preserves the engine's ordering, which is by size of impact. Without it a
    #: reload would come back in whatever order the database chose.
    position: Mapped[int] = mapped_column(Integer)

    code: Mapped[str] = mapped_column(String(120))
    label: Mapped[str] = mapped_column(String(200))

    currency: Mapped[str] = mapped_column(String(3))
    current_cash: Mapped[Decimal] = mapped_column(MONEY)
    candidate_cash: Mapped[Decimal] = mapped_column(MONEY)
    delta: Mapped[Decimal] = mapped_column(MONEY)

    run: Mapped[ComparisonRunRow] = relationship(back_populates="components")

    __table_args__ = (Index("ix_result_components_run_position", "run_id", "position"),)


#: Fixed so the migration is deterministic and the row is recognisable. Only
#: ever created when there are orphan accounts to adopt.
PLACEHOLDER_USER_ID: Final = uuid.UUID("00000000-0000-4000-8000-000000000001")


class UserRow(Base):
    """Somebody who can hold accounts.

    `password_hash` is nullable, and NULL means *this user cannot log in*. The
    migration uses that to adopt existing rows without inventing a credential,
    and `users set-password` is how a real one arrives.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    display_name: Mapped[str] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class AccountRow(Base):
    """An account the user has deliberately registered.

    The canonical `key` is what every constraint sees; `display_name` is what a
    person reads. Keeping both means normalisation can be strict without
    turning "Chase Checking" into "chase-checking" on a report.

    `key` is unique only within its owner (`uq_accounts_user_key`), not
    globally: a bank account's natural key has the same shape for everybody,
    so `chase-checking-5718` must be free for every user to pick.
    """

    __tablename__ = "accounts"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    key: Mapped[str] = mapped_column(String(100))
    display_name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_accounts_user_key"),)


class ImportBatchRow(Base):
    """One import of one file.

    `source_sha256` is what makes a byte-identical re-import a provable no-op
    rather than an inference. The declared window is what makes "this is a
    complete snapshot" a checkable claim.
    """

    __tablename__ = "import_batches"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    account_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("accounts.id"))
    source_file: Mapped[str] = mapped_column(String(255))
    source_sha256: Mapped[str] = mapped_column(String(64))
    mode: Mapped[str] = mapped_column(String(16))
    window_start: Mapped[date | None] = mapped_column(Date, nullable=True)
    window_end: Mapped[date | None] = mapped_column(Date, nullable=True)
    row_count: Mapped[int] = mapped_column(Integer)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint(
            "mode IN ('snapshot', 'incremental')",
            name="ck_import_batches_mode",
        ),
        CheckConstraint(
            "mode <> 'snapshot' OR (window_start IS NOT NULL AND window_end IS NOT NULL)",
            name="ck_import_batches_snapshot_window",
        ),
        CheckConstraint(
            "window_start IS NULL OR window_start <= window_end",
            name="ck_import_batches_window_ordered",
        ),
        UniqueConstraint(
            "account_id",
            "source_sha256",
            name="uq_import_batches_account_checksum",
        ),
    )


class TransactionRow(Base):
    """One imported bank row."""

    __tablename__ = "transactions"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    account_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("accounts.id"))
    batch_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("import_batches.id"), nullable=True
    )

    posted_on: Mapped[date] = mapped_column(Date)
    description: Mapped[str] = mapped_column(Text)
    normalised_merchant: Mapped[str] = mapped_column(Text)

    currency: Mapped[str] = mapped_column(String(3))
    amount: Mapped[Decimal] = mapped_column(MONEY)

    #: The bank's own id when the export carries one. Authoritative for dedupe.
    external_id: Mapped[str | None] = mapped_column(String(200), nullable=True)

    fingerprint: Mapped[str] = mapped_column(String(32))
    fingerprint_version: Mapped[int] = mapped_column(SmallInteger)
    occurrence: Mapped[int] = mapped_column(Integer)

    #: Absent for manual entry, which has no file behind it.
    source_file: Mapped[str | None] = mapped_column(String(255), nullable=True)
    source_line: Mapped[int | None] = mapped_column(Integer, nullable=True)
    raw_cells: Mapped[dict[str, str] | None] = mapped_column(JSONB, nullable=True)

    __table_args__ = (
        CheckConstraint("occurrence > 0", name="ck_transactions_occurrence_positive"),
        CheckConstraint(
            "source_line IS NULL OR source_line > 1",
            name="ck_transactions_source_line_after_header",
        ),
        UniqueConstraint(
            "account_id",
            "fingerprint",
            "occurrence",
            name="uq_transactions_account_fingerprint_occurrence",
        ),
        Index(
            "uq_transactions_account_external_id",
            "account_id",
            "external_id",
            unique=True,
            postgresql_where=text("external_id IS NOT NULL"),
        ),
        Index("ix_transactions_account_posted_on", "account_id", "posted_on"),
    )


class SchemaNote(Base):
    """A single row recording what this schema is for.

    Useful when someone opens the database cold and needs to know whether they
    are looking at demo data or something real.
    """

    __tablename__ = "schema_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    note: Mapped[str] = mapped_column(Text)
