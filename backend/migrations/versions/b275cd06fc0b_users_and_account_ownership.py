"""users and account ownership

Revision ID: b275cd06fc0b
Revises: 3dfb104aabcb
Create Date: 2026-08-31 14:22:21.527124

"""

import uuid
from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b275cd06fc0b"
down_revision: str | Sequence[str] | None = "3dfb104aabcb"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: Mirrors offerdelta.infrastructure.postgres.models.PLACEHOLDER_USER_ID,
#: inlined rather than imported. A migration is a frozen historical record;
#: importing application code would couple it to a module that keeps
#: changing, and this file must still run correctly after that module does.
#: test_users_migration.py asserts against the real constant, which is what
#: catches the two ever drifting apart.
_PLACEHOLDER_USER_ID = uuid.UUID("00000000-0000-4000-8000-000000000001")


def upgrade() -> None:
    """Give every account an owner, and make the account key per-owner.

    Destructive: `accounts.key` was globally unique and becomes unique only
    within `(user_id, key)`, because a bank account's natural key
    ("chase-checking-5718") has the same shape for everybody and cannot stay
    a global namespace once there is more than one owner.
    """
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_users"),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    # Nullable first: the column has to exist before it can be filled.
    op.add_column("accounts", sa.Column("user_id", sa.Uuid(), nullable=True))

    connection = op.get_bind()
    orphans = connection.execute(
        sa.text("SELECT count(*) FROM accounts WHERE user_id IS NULL")
    ).scalar_one()

    # Only when there is something to adopt. An empty database - CI, and
    # Render on first deploy - must not gain a user row nobody asked for.
    if orphans:
        connection.execute(
            sa.text(
                "INSERT INTO users (id, email, password_hash, display_name, "
                "is_active, created_at) VALUES (:id, :email, NULL, :name, true, now())"
            ),
            {
                "id": _PLACEHOLDER_USER_ID,
                "email": "placeholder@localhost.invalid",
                "name": "Placeholder owner (set a password to claim)",
            },
        )
        connection.execute(
            sa.text("UPDATE accounts SET user_id = :id WHERE user_id IS NULL"),
            {"id": _PLACEHOLDER_USER_ID},
        )

    op.alter_column("accounts", "user_id", nullable=False)
    op.create_foreign_key("fk_accounts_user_id", "accounts", "users", ["user_id"], ["id"])

    # The destructive change: a bank account key is the same shape for
    # everybody, so it can only be unique within one owner. The existing
    # constraint is named uq_accounts_key (confirmed against the live
    # database via sqlalchemy's inspector), not the generic accounts_key_key
    # a bare `unique=True` column might suggest.
    op.drop_constraint("uq_accounts_key", "accounts", type_="unique")
    op.create_unique_constraint("uq_accounts_user_key", "accounts", ["user_id", "key"])


def downgrade() -> None:
    """Reverse of upgrade: drops ownership and restores the global key.

    Restoring `uq_accounts_key` will itself fail if two accounts belonging to
    different users now share a key - the exact state this migration exists
    to allow. That failure is correct: a global unique constraint cannot
    honestly coexist with the per-owner data upgrade() intentionally created,
    so an operator hitting it must resolve the collision by hand rather than
    have this migration silently pick a survivor.
    """
    op.drop_constraint("uq_accounts_user_key", "accounts", type_="unique")
    op.create_unique_constraint("uq_accounts_key", "accounts", ["key"])
    op.drop_constraint("fk_accounts_user_id", "accounts", type_="foreignkey")
    op.drop_column("accounts", "user_id")
    op.drop_table("users")
