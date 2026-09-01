"""The one step that is awkward to undo.

Two cases matter: a database with existing accounts must end up with exactly
one placeholder owner, and an empty database must gain no user row at all.
CI and a fresh Render deploy are the empty case, and inventing a user there
would ship a row nobody asked for.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text
from sqlalchemy.exc import IntegrityError

from offerdelta.infrastructure.postgres.models import PLACEHOLDER_USER_ID
from tests.integration.conftest import requires_database

pytestmark = requires_database

#: alembic.ini lives at the backend root, two directories above this package.
#: Resolved from `__file__` rather than left as a bare "alembic.ini" so this
#: test does not depend on the working directory pytest was launched from —
#: matching the working pattern in test_migration_rebuild.py.
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"


def _upgrade_to(engine: Engine, schema: str, revision: str) -> None:
    """Run one upgrade inside `schema`, on its own transaction.

    Handing Alembic a live connection (`env.py`'s documented hook) is what
    confines the run to the throwaway schema; left to itself `env.py` builds
    an engine from settings and would migrate the shared database.

    `SET LOCAL` rather than `SET`: a session-scoped search_path would ride
    the pooled connection back into the rest of the suite still pointing at
    a schema this fixture has already dropped, once the transaction below
    commits and the connection returns to the pool. This mirrors the
    `_upgrade` helper in test_migration_rebuild.py, which carries the same
    warning.
    """
    config = Config(str(_ALEMBIC_INI))
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        config.attributes["connection"] = conn
        command.upgrade(config, revision)


def test_existing_accounts_gain_one_placeholder_owner(engine: Engine, scratch_schema: str) -> None:
    _upgrade_to(engine, scratch_schema, "3dfb104aabcb")

    account_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{scratch_schema}"'))
        conn.execute(
            text(
                "INSERT INTO accounts (id, key, display_name, created_at) "
                "VALUES (:id, 'chase-checking-5718', 'Chase Checking', now())"
            ),
            {"id": account_id},
        )

    _upgrade_to(engine, scratch_schema, "head")

    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{scratch_schema}"'))
        users = conn.execute(text("SELECT id, password_hash FROM users")).all()
        owner = conn.execute(
            text("SELECT user_id FROM accounts WHERE id = :id"), {"id": account_id}
        ).scalar_one()

    assert len(users) == 1
    assert users[0][0] == PLACEHOLDER_USER_ID
    assert users[0][1] is None, "the placeholder must not be able to log in"
    assert owner == PLACEHOLDER_USER_ID


def test_an_empty_database_gains_no_user(engine: Engine, scratch_schema: str) -> None:
    _upgrade_to(engine, scratch_schema, "head")

    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{scratch_schema}"'))
        count = conn.execute(text("SELECT count(*) FROM users")).scalar_one()

    assert count == 0


def test_two_users_may_hold_the_same_account_key(engine: Engine, scratch_schema: str) -> None:
    """The point of the destructive change."""
    _upgrade_to(engine, scratch_schema, "head")

    first, second = uuid.uuid4(), uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{scratch_schema}"'))
        for user_id, email in ((first, "a@example.test"), (second, "b@example.test")):
            conn.execute(
                text(
                    "INSERT INTO users (id, email, display_name, is_active, created_at) "
                    "VALUES (:id, :email, 'Someone', true, now())"
                ),
                {"id": user_id, "email": email},
            )
            conn.execute(
                text(
                    "INSERT INTO accounts (id, user_id, key, display_name, created_at) "
                    "VALUES (:id, :user_id, 'chase-checking-5718', 'Chase', now())"
                ),
                {"id": uuid.uuid4(), "user_id": user_id},
            )

        held = conn.execute(
            text("SELECT count(*) FROM accounts WHERE key = 'chase-checking-5718'")
        ).scalar_one()

    assert held == 2


def _insert_duplicate_key_account(engine: Engine, schema: str, user_id: uuid.UUID) -> None:
    """Second insert of the same (user_id, key) pair, for the raises() block below.

    Pulled into its own call so the `pytest.raises` block holds one simple
    statement rather than a `SET LOCAL` plus an insert (ruff PT012).
    """
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        conn.execute(
            text(
                "INSERT INTO accounts (id, user_id, key, display_name, created_at) "
                "VALUES (:id, :user_id, 'chase-checking-5718', 'Chase', now())"
            ),
            {"id": uuid.uuid4(), "user_id": user_id},
        )


def test_the_same_key_twice_for_one_user_is_still_refused(
    engine: Engine, scratch_schema: str
) -> None:
    _upgrade_to(engine, scratch_schema, "head")

    user_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{scratch_schema}"'))
        conn.execute(
            text(
                "INSERT INTO users (id, email, display_name, is_active, created_at) "
                "VALUES (:id, 'a@example.test', 'Someone', true, now())"
            ),
            {"id": user_id},
        )
        conn.execute(
            text(
                "INSERT INTO accounts (id, user_id, key, display_name, created_at) "
                "VALUES (:id, :user_id, 'chase-checking-5718', 'Chase', now())"
            ),
            {"id": uuid.uuid4(), "user_id": user_id},
        )

    with pytest.raises(IntegrityError):
        _insert_duplicate_key_account(engine, scratch_schema, user_id)
