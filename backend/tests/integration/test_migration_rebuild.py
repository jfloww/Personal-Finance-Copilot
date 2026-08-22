"""The migration is destructive by design, so its guard gets a real test.

These are also the only tests that run `alembic upgrade` from nothing. The
shared database is permanently at head, so the from-scratch path can only be
exercised inside a schema created for the purpose and dropped afterwards.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from sqlalchemy.engine import Connection, Engine

from tests.integration.conftest import requires_database

pytestmark = requires_database

#: alembic.ini lives at the backend root, two directories above this package.
_ALEMBIC_INI = Path(__file__).resolve().parents[2] / "alembic.ini"

#: The last revision before the rebuild: the state a real database is in.
_BEFORE_REBUILD = "a6128d6e4f20"


def _upgrade(connection: Connection, schema: str, revision: str) -> None:
    """Run one upgrade inside `schema`, on the caller's own transaction.

    Handing Alembic the connection is what confines the whole upgrade to the
    throwaway schema; left to itself `env.py` builds an engine from settings
    and would migrate the shared database.

    `SET LOCAL` rather than `SET`: a session-scoped search_path would ride the
    pooled connection back into the rest of the suite still pointing at a
    schema this fixture has already dropped.
    """
    connection.execute(sa.text(f'SET LOCAL search_path TO "{schema}"'))
    config = Config(str(_ALEMBIC_INI))
    config.attributes["connection"] = connection
    command.upgrade(config, revision)


def test_upgrade_from_nothing_creates_all_three_tables(engine: Engine, scratch_schema: str) -> None:
    with engine.begin() as conn:
        _upgrade(conn, scratch_schema, "head")
        tables = set(sa.inspect(conn).get_table_names(schema=scratch_schema))

    assert {"accounts", "import_batches", "transactions"} <= tables


def test_raw_cells_is_jsonb_not_json(engine: Engine, scratch_schema: str) -> None:
    """json has no equality operator, so DISTINCT over transactions errors."""
    with engine.begin() as conn:
        _upgrade(conn, scratch_schema, "head")
        kind = conn.execute(
            sa.text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_schema = :s AND table_name = 'transactions' "
                "AND column_name = 'raw_cells'"
            ),
            {"s": scratch_schema},
        ).scalar_one()

    assert kind == "jsonb"


def test_upgrade_aborts_when_transactions_holds_rows(engine: Engine, scratch_schema: str) -> None:
    """A destructive migration must fail loudly, not quietly delete money."""
    with engine.begin() as conn:
        _upgrade(conn, scratch_schema, _BEFORE_REBUILD)
        conn.execute(
            sa.text(
                "INSERT INTO transactions (id, imported_at, account, posted_on, "
                "description, normalised_merchant, currency, amount, fingerprint, "
                "occurrence, source_file, source_line, raw_cells) VALUES "
                "(:id, now(), 'checking', '2026-08-17', 'BLUE BOTTLE', 'BLUE BOTTLE', "
                "'USD', -4.50, 'abc', 1, 'aug.csv', 2, '{}')"
            ),
            {"id": str(uuid.uuid4())},
        )

    with pytest.raises(RuntimeError, match="holds 1 row"), engine.begin() as conn:
        _upgrade(conn, scratch_schema, "head")
