"""transaction classification

Revision ID: 9dc812bfd7ed
Revises: b275cd06fc0b
Create Date: 2026-09-02 08:07:23.577115

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9dc812bfd7ed"
down_revision: str | Sequence[str] | None = "b275cd06fc0b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Give a transaction somewhere to record what it was classified as.

    Every column is nullable and nothing is backfilled. A row nobody has
    categorised is unclassified, and a migration that invented a label for
    it would be asserting something no categoriser ever said.
    """
    op.add_column("transactions", sa.Column("suggested_label", sa.String(64), nullable=True))
    op.add_column("transactions", sa.Column("suggested_source", sa.String(16), nullable=True))
    op.add_column(
        "transactions", sa.Column("suggested_confidence", sa.Numeric(4, 3), nullable=True)
    )
    op.add_column("transactions", sa.Column("suggested_by", sa.String(120), nullable=True))
    op.add_column(
        "transactions", sa.Column("suggested_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("transactions", sa.Column("confirmed_label", sa.String(64), nullable=True))
    op.add_column(
        "transactions", sa.Column("confirmed_at", sa.DateTime(timezone=True), nullable=True)
    )


def downgrade() -> None:
    for column in (
        "confirmed_at",
        "confirmed_label",
        "suggested_at",
        "suggested_by",
        "suggested_confidence",
        "suggested_source",
        "suggested_label",
    ):
        op.drop_column("transactions", column)
