"""rebuild transaction storage

Revision ID: 3dfb104aabcb
Revises: a6128d6e4f20
Create Date: 2026-08-22 17:57:53.280983

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "3dfb104aabcb"
down_revision: str | Sequence[str] | None = "a6128d6e4f20"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Rebuild transaction storage on a corrected identity model.

    Destructive by design: stored fingerprints predate versioning, hashed an
    unquantised amount, and keyed on free-text accounts, so they cannot be
    recomputed or migrated. Rebuilding is the honest option — but a migration
    written for an empty table has to say so when the table is not empty,
    because whoever runs it months from now will not remember that assumption.
    """
    bind = op.get_bind()
    existing = bind.execute(sa.text("SELECT count(*) FROM transactions")).scalar_one()
    if existing:
        raise RuntimeError(
            f"transactions holds {existing} row(s). This revision rebuilds the "
            f"table and cannot preserve them: stored fingerprints predate "
            f"versioning and cannot be recomputed from their own rows. Back up, "
            f"TRUNCATE deliberately, then re-run."
        )

    op.drop_table("transactions")

    op.create_table(
        "accounts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("key", sa.String(length=100), nullable=False),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_accounts"),
        sa.UniqueConstraint("key", name="uq_accounts_key"),
    )

    op.create_table(
        "import_batches",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("account_id", sa.Uuid(), nullable=False),
        sa.Column("source_file", sa.String(length=255), nullable=False),
        sa.Column("source_sha256", sa.String(length=64), nullable=False),
        sa.Column("mode", sa.String(length=16), nullable=False),
        sa.Column("window_start", sa.Date(), nullable=True),
        sa.Column("window_end", sa.Date(), nullable=True),
        sa.Column("row_count", sa.Integer(), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("mode IN ('snapshot', 'incremental')", name="ck_import_batches_mode"),
        sa.CheckConstraint(
            "mode <> 'snapshot' OR (window_start IS NOT NULL AND window_end IS NOT NULL)",
            name="ck_import_batches_snapshot_window",
        ),
        sa.CheckConstraint(
            "window_start IS NULL OR window_start <= window_end",
            name="ck_import_batches_window_ordered",
        ),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], name="fk_import_batches_account"),
        sa.PrimaryKeyConstraint("id", name="pk_import_batches"),
        sa.UniqueConstraint(
            "account_id", "source_sha256", name="uq_import_batches_account_checksum"
        ),
    )

    op.create_table(
        "transactions",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("imported_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("account_id", sa.Uuid(), nullable=False),
        sa.Column("batch_id", sa.Uuid(), nullable=True),
        sa.Column("posted_on", sa.Date(), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("normalised_merchant", sa.Text(), nullable=False),
        sa.Column("currency", sa.String(length=3), nullable=False),
        sa.Column("amount", sa.Numeric(precision=18, scale=2), nullable=False),
        sa.Column("external_id", sa.String(length=200), nullable=True),
        sa.Column("fingerprint", sa.String(length=32), nullable=False),
        sa.Column("fingerprint_version", sa.SmallInteger(), nullable=False),
        sa.Column("occurrence", sa.Integer(), nullable=False),
        sa.Column("source_file", sa.String(length=255), nullable=True),
        sa.Column("source_line", sa.Integer(), nullable=True),
        sa.Column("raw_cells", postgresql.JSONB(), nullable=True),
        sa.CheckConstraint("occurrence > 0", name="ck_transactions_occurrence_positive"),
        sa.CheckConstraint(
            "source_line IS NULL OR source_line > 1",
            name="ck_transactions_source_line_after_header",
        ),
        sa.ForeignKeyConstraint(["account_id"], ["accounts.id"], name="fk_transactions_account"),
        sa.ForeignKeyConstraint(["batch_id"], ["import_batches.id"], name="fk_transactions_batch"),
        sa.PrimaryKeyConstraint("id", name="pk_transactions"),
        sa.UniqueConstraint(
            "account_id",
            "fingerprint",
            "occurrence",
            name="uq_transactions_account_fingerprint_occurrence",
        ),
    )
    op.create_index(
        "uq_transactions_account_external_id",
        "transactions",
        ["account_id", "external_id"],
        unique=True,
        postgresql_where=sa.text("external_id IS NOT NULL"),
    )
    op.create_index(
        "ix_transactions_account_posted_on", "transactions", ["account_id", "posted_on"]
    )


def downgrade() -> None:
    """Deliberately irreversible.

    A truthful downgrade would have to reconstruct v0 fingerprints, which is
    the exact thing that cannot be done: they hashed an unquantised amount
    against a free-text account, so they are not recomputable from any row this
    schema stores. Recreating the old tables empty would leave a database that
    looks downgraded and has silently lost every transaction. Refusing is the
    honest option.
    """
    raise NotImplementedError(
        "a6128d6e4f20 -> this revision is one-way: v0 fingerprints cannot be "
        "reconstructed. Restore from a backup instead."
    )
