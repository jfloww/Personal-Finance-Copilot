"""Alembic environment.

The URL comes from the application's own settings rather than alembic.ini, so
the connection string lives in exactly one place and never sits in a file that
git tracks.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import Connection, engine_from_config, pool

from offerdelta.config import get_settings
from offerdelta.infrastructure.postgres.models import Base

config = context.config

if config.config_file_name is not None:
    # `disable_existing_loggers` defaults to True, which does not mean "leave
    # everything else alone" - it means every logger that already exists and
    # is not named in `alembic.ini`'s `[loggers]` section gets `.disabled`
    # set, silently. In a test process that means any application logger
    # imported before this runs - which, by the time any test body executes,
    # is all of them - stops emitting for the rest of the session. A test
    # asserting that some text never reaches a log would then pass whether or
    # not that is true, because nothing reaches the log either way. Alembic
    # owns three logger names here; it has no business switching off the
    # ones it does not.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _url() -> str:
    return get_settings().sqlalchemy_dsn


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        # Catches a column whose type drifts from the model — a NUMERIC
        # quietly becoming a float is exactly the change that must never
        # pass unnoticed here.
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    # A caller may supply its own connection, which is Alembic's documented
    # hook for running migrations programmatically. The migration tests use it
    # to confine a whole upgrade to a throwaway schema: without it this
    # function builds an engine from settings and there is no way to exercise
    # the from-scratch path except against the real database.
    supplied = config.attributes.get("connection")
    if supplied is not None:
        _run(supplied)
        return

    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _url()

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        _run(connection)


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
