"""Database connections.

Neon puts the database behind a pooler that can drop an idle connection at any
time, so `pool_pre_ping` is on: a stale connection is discovered and replaced
before a query rather than surfacing as a mystery failure mid-request.
"""

from __future__ import annotations

from functools import lru_cache

from sqlalchemy import Engine, create_engine

from offerdelta.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(
        settings.sqlalchemy_dsn,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args={"connect_timeout": 15},
    )
