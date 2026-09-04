"""Classification columns.

Every column is nullable and nothing is backfilled: a row that has never
been categorised is unclassified, and saying so is more honest than
inventing a label for it at migration time.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, text

from tests.integration.conftest import requires_database

pytestmark = requires_database

#: alembic.ini lives at the backend root, two directories above this package.
#: Resolved from `__file__` rather than left as a bare "alembic.ini" so this
#: test does not depend on the working directory pytest was launched from —
#: matching the working pattern in test_migration_rebuild.py.
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"

_EXPECTED = {
    "suggested_label",
    "suggested_source",
    "suggested_confidence",
    "suggested_by",
    "suggested_at",
    "confirmed_label",
    "confirmed_at",
}


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


def test_the_columns_exist_and_are_all_nullable(engine: Engine, scratch_schema: str) -> None:
    _upgrade_to(engine, scratch_schema, "head")

    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{scratch_schema}"'))
        rows = conn.execute(
            text(
                "SELECT column_name, is_nullable FROM information_schema.columns "
                "WHERE table_schema = :schema AND table_name = 'transactions'"
            ),
            {"schema": scratch_schema},
        ).all()

    present = dict(rows)  # type: ignore[arg-type,var-annotated]
    assert set(present) >= _EXPECTED
    assert all(present[name] == "YES" for name in _EXPECTED)


def test_existing_rows_are_left_unclassified(engine: Engine, scratch_schema: str) -> None:
    """No backfill. An unlabelled row must not acquire a label from a migration."""
    _upgrade_to(engine, scratch_schema, "b275cd06fc0b")

    user_id, account_id = uuid.uuid4(), uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{scratch_schema}"'))
        conn.execute(
            text(
                "INSERT INTO users (id, email, display_name, is_active, created_at) "
                "VALUES (:id, 'a@example.test', 'A', true, now())"
            ),
            {"id": user_id},
        )
        conn.execute(
            text(
                "INSERT INTO accounts (id, user_id, key, display_name, created_at) "
                "VALUES (:id, :user_id, 'chase-checking-5718', 'Chase', now())"
            ),
            {"id": account_id, "user_id": user_id},
        )
        conn.execute(
            text(
                "INSERT INTO transactions (id, imported_at, account_id, posted_on, "
                "description, normalised_merchant, currency, amount, fingerprint, "
                "fingerprint_version, occurrence) VALUES (:id, now(), :account_id, "
                "'2026-03-01', 'BLUE BOTTLE', 'BLUE BOTTLE', 'USD', -12.34, "
                "'0000000000000000000000000000000a', 1, 1)"
            ),
            {"id": uuid.uuid4(), "account_id": account_id},
        )

    _upgrade_to(engine, scratch_schema, "head")

    with engine.begin() as conn:
        conn.execute(text(f'SET LOCAL search_path TO "{scratch_schema}"'))
        row = conn.execute(
            text(
                "SELECT suggested_label, suggested_source, suggested_confidence, "
                "suggested_by, suggested_at, confirmed_label, confirmed_at "
                "FROM transactions"
            )
        ).one()

    assert row == (None, None, None, None, None, None, None)
