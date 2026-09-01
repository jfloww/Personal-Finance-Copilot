"""Users, and the one query that decides who a caller is."""

from __future__ import annotations

import logging

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.postgres import repositories as repositories_module
from offerdelta.infrastructure.postgres.models import UserRow
from offerdelta.infrastructure.postgres.repositories import UserRepository
from tests.integration.conftest import requires_database

pytestmark = requires_database


def test_a_created_user_cannot_log_in_until_a_password_is_set(session: Session) -> None:
    repo = UserRepository(session)
    created = repo.create("a@example.test", "Person A")
    assert created.has_password is False
    assert repo.authenticate("a@example.test", "anything") is None


def test_authenticate_returns_the_user_after_a_password_is_set(session: Session) -> None:
    repo = UserRepository(session)
    repo.create("a@example.test", "Person A")
    repo.set_password("a@example.test", "a good long password")

    who = repo.authenticate("a@example.test", "a good long password")
    assert who is not None
    assert who.email == "a@example.test"


def test_authenticate_rejects_a_wrong_password(session: Session) -> None:
    repo = UserRepository(session)
    repo.create("a@example.test", "Person A")
    repo.set_password("a@example.test", "a good long password")
    assert repo.authenticate("a@example.test", "a good long passwerd") is None


def test_authenticate_rejects_an_unknown_email(session: Session) -> None:
    assert UserRepository(session).authenticate("nobody@example.test", "x") is None


def test_a_deactivated_user_cannot_authenticate(session: Session) -> None:
    repo = UserRepository(session)
    repo.create("a@example.test", "Person A")
    repo.set_password("a@example.test", "a good long password")
    repo.deactivate("a@example.test")
    assert repo.authenticate("a@example.test", "a good long password") is None


def test_emails_are_matched_case_insensitively(session: Session) -> None:
    repo = UserRepository(session)
    repo.create("Person.A@Example.test", "Person A")
    repo.set_password("person.a@example.test", "a good long password")
    assert repo.authenticate("PERSON.A@EXAMPLE.TEST", "a good long password") is not None


def test_a_duplicate_email_is_refused(session: Session) -> None:
    repo = UserRepository(session)
    repo.create("a@example.test", "Person A")
    with pytest.raises(ValidationError):
        repo.create("A@Example.test", "Person A again")


def test_a_corrupted_password_hash_fails_authentication_without_raising(
    session: Session, caplog: pytest.LogCaptureFixture
) -> None:
    """A malformed `password_hash` is data corruption, not a login failure.

    `UserRepository`'s own interface never writes an invalid hash, so this
    reaches through the ORM the way real corruption would arrive - a bad
    migration, a manual edit - and confirms `authenticate` still resolves to
    `None` instead of letting `InvalidHashError` escape. A caller-visible
    500 for this one address while every other failure answers identically
    would itself be an enumeration signal, so the log line is the only place
    this is allowed to be distinguishable - and it must name the user id, not
    the address, since this repository does not put identifying data other
    than the id in a log.

    If `test_migration_rebuild.py` or `test_users_migration.py` ran earlier in
    this session, Alembic's `env.py` has already called
    `logging.config.fileConfig`, whose default `disable_existing_loggers=True`
    disables every logger that existed at that point - including this
    module's, imported long before this test runs. That is a real,
    process-global side effect of a sibling test file, not something this
    test owns; it is undone here rather than left to depend on suite order.
    """
    repositories_module.logger.disabled = False
    repo = UserRepository(session)
    created = repo.create("a@example.test", "Person A")
    row = session.scalars(select(UserRow).where(UserRow.email == "a@example.test")).one()
    row.password_hash = "not-a-valid-argon2-hash"
    session.flush()

    with caplog.at_level(logging.ERROR):
        result = repo.authenticate("a@example.test", "anything")

    assert result is None
    errors = [record for record in caplog.records if record.levelno == logging.ERROR]
    assert len(errors) == 1
    assert str(created.id) in errors[0].getMessage()
    assert "a@example.test" not in errors[0].getMessage()
