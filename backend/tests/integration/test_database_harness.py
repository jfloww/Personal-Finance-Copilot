"""The harness itself is load-bearing, so it gets its own test."""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.engine import Engine

from tests.integration.conftest import requires_database

pytestmark = requires_database


def test_scratch_schema_exists_during_the_test(engine: Engine, scratch_schema: str) -> None:
    with engine.connect() as conn:
        found = conn.execute(
            sa.text("SELECT 1 FROM information_schema.schemata WHERE schema_name = :n"),
            {"n": scratch_schema},
        ).scalar_one_or_none()
    assert found == 1


def test_scratch_schema_is_isolated(engine: Engine, scratch_schema: str) -> None:
    with engine.begin() as conn:
        conn.execute(sa.text(f'CREATE TABLE "{scratch_schema}".probe (id integer)'))
        conn.execute(sa.text(f'INSERT INTO "{scratch_schema}".probe VALUES (1)'))
        count = conn.execute(sa.text(f'SELECT count(*) FROM "{scratch_schema}".probe')).scalar_one()
    assert count == 1
