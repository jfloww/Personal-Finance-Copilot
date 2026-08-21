"""Database fixtures.

Every test runs inside a transaction that is rolled back afterwards, so the
suite can point at a real PostgreSQL instance without leaving anything behind.
That matters here: the target is a live Neon database, not a throwaway
container, and a test that litters is a test nobody runs twice.

The whole module skips when CONNECTION_STRING is unset. CI has no secret, and a
suite that fails there for want of one teaches people to ignore red builds.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from sqlalchemy import Connection, Engine, text
from sqlalchemy.orm import Session

from offerdelta.config import get_settings
from offerdelta.infrastructure.postgres.engine import get_engine

requires_database = pytest.mark.skipif(
    not get_settings().database_available,
    reason="CONNECTION_STRING is not set; database tests need a live PostgreSQL",
)


@pytest.fixture(scope="session")
def engine() -> Engine:
    return get_engine()


@pytest.fixture
def connection(engine: Engine) -> Iterator[Connection]:
    conn = engine.connect()
    transaction = conn.begin()
    try:
        yield conn
    finally:
        # Unconditional: even a passing test leaves nothing committed.
        transaction.rollback()
        conn.close()


@pytest.fixture
def session(connection: Connection) -> Iterator[Session]:
    """A session joined to the outer transaction, so its commits are undone."""
    with Session(
        bind=connection, expire_on_commit=False, join_transaction_mode="create_savepoint"
    ) as s:
        yield s


@pytest.fixture
def scratch_schema(engine: Engine) -> Iterator[str]:
    """An empty schema that exists only for this test.

    Migrations need a database at a known revision, which the shared one is
    not. Creating a throwaway schema gives each migration test a pristine
    namespace without a container, and works identically against Neon and the
    CI service.
    """
    name = f"scratch_{uuid.uuid4().hex[:12]}"
    with engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{name}"'))
    try:
        yield name
    finally:
        with engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{name}" CASCADE'))
