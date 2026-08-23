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


def _constraints(connection: Connection, schema: str) -> dict[str, tuple[str, str]]:
    """conname -> (contype, definition) for every constraint in `schema`.

    `uq_transactions_account_external_id` is deliberately absent from this
    map: it is a partial unique *index* (`op.create_index(..., unique=True,
    postgresql_where=...)`), not a table constraint, so PostgreSQL records it
    in `pg_indexes`, not `pg_constraint`. See `_indexes` below.
    """
    rows = connection.execute(
        sa.text(
            "SELECT c.conname, c.contype, pg_get_constraintdef(c.oid) "
            "FROM pg_constraint c "
            "JOIN pg_namespace n ON n.oid = c.connamespace "
            "WHERE n.nspname = :schema"
        ),
        {"schema": schema},
    ).all()
    return {name: (contype, definition) for name, contype, definition in rows}


def _indexes(connection: Connection, schema: str, table: str) -> dict[str, str]:
    """indexname -> indexdef for every index on `table` in `schema`."""
    rows = connection.execute(
        sa.text(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE schemaname = :schema AND tablename = :table"
        ),
        {"schema": schema, "table": table},
    ).all()
    return dict(tuple(row) for row in rows)


def test_head_revision_installs_the_database_level_guards(
    engine: Engine, scratch_schema: str
) -> None:
    """The identity and integrity guarantees this branch relies on must be
    enforced by PostgreSQL itself, not merely by application code — that is
    the entire reason `test_upgrade_aborts_when_transactions_holds_rows`
    above insists the rebuild is destructive rather than silently lossy.

    A deleted test (`test_the_multiplicity_aware_identity_is_a_database_
    constraint`) used to query `pg_constraint` directly to prove this; nothing
    replaced it. Without this test, every guard below — including the
    composite uniqueness `add_many`'s own docstring calls "the final
    concurrency guard" — can be deleted from the migration and the suite
    stays green, because nothing else queries the catalog.
    """
    with engine.begin() as conn:
        _upgrade(conn, scratch_schema, "head")
        constraints = _constraints(conn, scratch_schema)
        tx_indexes = _indexes(conn, scratch_schema, "transactions")

    # The final concurrency guard `add_many` relies on: one row per
    # (account, fingerprint, occurrence) triple, enforced by PostgreSQL
    # itself rather than by application code racing a SELECT-then-INSERT.
    contype, definition = constraints["uq_transactions_account_fingerprint_occurrence"]
    assert contype == "u"
    assert definition == "UNIQUE (account_id, fingerprint, occurrence)"

    contype, definition = constraints["ck_transactions_occurrence_positive"]
    assert contype == "c"
    assert definition == "CHECK ((occurrence > 0))"

    contype, definition = constraints["ck_transactions_source_line_after_header"]
    assert contype == "c"
    assert definition == "CHECK (((source_line IS NULL) OR (source_line > 1)))"

    contype, mode_def = constraints["ck_import_batches_mode"]
    assert contype == "c"
    assert "'snapshot'" in mode_def
    assert "'incremental'" in mode_def

    contype, window_def = constraints["ck_import_batches_snapshot_window"]
    assert contype == "c"
    assert "window_start IS NOT NULL" in window_def
    assert "window_end IS NOT NULL" in window_def

    contype, definition = constraints["uq_import_batches_account_checksum"]
    assert contype == "u"
    assert definition == "UNIQUE (account_id, source_sha256)"

    # A partial unique index, not a table constraint (see `_constraints`), so
    # it lives in pg_indexes. Asserting the WHERE predicate -- not merely
    # that an index of this name exists -- is the point: without it, two
    # transactions that both lack an external_id (the common case) could
    # never coexist in the same account.
    external_id_index = tx_indexes["uq_transactions_account_external_id"]
    assert "CREATE UNIQUE INDEX" in external_id_index
    assert "WHERE (external_id IS NOT NULL)" in external_id_index

    # Redundant indexes the rebuild is supposed to have retired: the
    # composite unique above already indexes fingerprint, and imported_at is
    # not used in any predicate that needs its own index.
    assert "ix_transactions_imported_at" not in tx_indexes
    assert not any("(fingerprint)" in indexdef for indexdef in tx_indexes.values()), (
        "a standalone index on fingerprint is redundant with "
        "uq_transactions_account_fingerprint_occurrence, which already indexes it"
    )


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
