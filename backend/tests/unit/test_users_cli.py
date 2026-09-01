"""The CLI is the only way a user is created. There is no registration route."""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy.exc import SQLAlchemyError

from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.postgres import engine as pg_engine
from offerdelta.infrastructure.postgres.repositories import StoredUser
from users import build_parser, main


def test_create_requires_an_email_and_a_display_name() -> None:
    args = build_parser().parse_args(
        ["create", "--email", "a@example.test", "--display-name", "Person A"]
    )
    assert args.command == "create"
    assert args.email == "a@example.test"


def test_set_password_reads_the_password_from_the_environment_not_the_argv() -> None:
    """A password on the command line lands in shell history and in ps output."""
    args = build_parser().parse_args(["set-password", "--email", "a@example.test"])
    assert args.command == "set-password"
    assert not hasattr(args, "password")


def test_an_unknown_argument_is_an_error() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["create", "--email", "a@example.test", "--admin"])


def test_create_rejects_an_unknown_flag_even_with_every_required_one_present() -> None:
    """Distinct from the case above: every required field is here, so only the
    unrecognised flag itself can be the reason this exits non-zero."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["create", "--email", "a@example.test", "--display-name", "Person A", "--admin"]
        )


def test_set_password_rejects_an_unknown_flag() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["set-password", "--email", "a@example.test", "--force"])


def test_list_takes_no_arguments() -> None:
    args = build_parser().parse_args(["list"])
    assert args.command == "list"


def test_deactivate_requires_an_email() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["deactivate"])


# ---------------------------------------------------------------- set-password


def test_set_password_exits_2_when_the_environment_variable_is_unset(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No database call should even be attempted: the check comes first."""
    monkeypatch.delenv("OFFERDELTA_NEW_PASSWORD", raising=False)
    code = main(["set-password", "--email", "a@example.test"])
    out = capsys.readouterr().out
    assert code == 2
    assert "OFFERDELTA_NEW_PASSWORD" in out


def test_set_password_never_prints_the_password_even_on_success(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The happy path is exactly where a careless `print(args)` would leak it."""
    captured: dict[str, str] = {}

    class _FakeRepo:
        def __init__(self, _session: object) -> None:
            pass

        def set_password(self, email: str, plain: str) -> None:
            captured["email"] = email
            captured["plain"] = plain

    monkeypatch.setenv("OFFERDELTA_NEW_PASSWORD", "correct horse battery staple")
    monkeypatch.setattr("users.get_engine", lambda: None)
    monkeypatch.setattr("users.Session", lambda _engine: _NullSession())
    monkeypatch.setattr("users.UserRepository", _FakeRepo)

    code = main(["set-password", "--email", "a@example.test"])
    out = capsys.readouterr().out

    assert code == 0
    assert captured == {"email": "a@example.test", "plain": "correct horse battery staple"}
    assert "correct horse battery staple" not in out


# ---------------------------------------------------------------- list


def test_list_prints_status_but_never_a_password_hash(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeRepo:
        def __init__(self, _session: object) -> None:
            pass

        def all(self) -> list[StoredUser]:
            return [
                StoredUser(
                    id=uuid.uuid4(),
                    email="a@example.test",
                    display_name="Person A",
                    is_active=True,
                    has_password=True,
                ),
                StoredUser(
                    id=uuid.uuid4(),
                    email="b@example.test",
                    display_name="Person B",
                    is_active=False,
                    has_password=False,
                ),
            ]

    monkeypatch.setattr("users.get_engine", lambda: None)
    monkeypatch.setattr("users.Session", lambda _engine: _NullSession())
    monkeypatch.setattr("users.UserRepository", _FakeRepo)

    code = main(["list"])
    out = capsys.readouterr().out

    assert code == 0
    assert "a@example.test" in out
    assert "b@example.test" in out
    assert "argon2" not in out
    assert "password_hash" not in out


# ---------------------------------------------------------------- error handling


def test_create_reports_a_validation_error_without_a_traceback(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class _FakeRepo:
        def __init__(self, _session: object) -> None:
            pass

        def create(self, email: str, _display_name: str) -> StoredUser:
            raise ValidationError(f"user {email!r} already exists")

    monkeypatch.setattr("users.get_engine", lambda: None)
    monkeypatch.setattr("users.Session", lambda _engine: _NullSession())
    monkeypatch.setattr("users.UserRepository", _FakeRepo)

    code = main(["create", "--email", "a@example.test", "--display-name", "Person A"])
    out = capsys.readouterr().out

    assert code == 1
    assert "already exists" in out
    assert "Traceback" not in out


def test_a_database_error_never_echoes_credentials_or_the_raw_message(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config.py` promises the connection string is never logged, echoed, or put
    into an error message. A raw SQLAlchemy exception can carry the DSN and the
    SQL that was running, so only the redacted host may reach the terminal.
    """

    class _FakeRepo:
        def __init__(self, _session: object) -> None:
            pass

        def all(self) -> list[StoredUser]:
            raise SQLAlchemyError(
                'connection to server failed: password authentication failed for user "hunter2"'
            )

    monkeypatch.setenv("CONNECTION_STRING", "postgresql://user:hunter2@dbhost/offerdelta")
    get_settings.cache_clear()
    monkeypatch.setattr("users.get_engine", lambda: None)
    monkeypatch.setattr("users.Session", lambda _engine: _NullSession())
    monkeypatch.setattr("users.UserRepository", _FakeRepo)
    try:
        code = main(["list"])
        out = capsys.readouterr().out
    finally:
        get_settings.cache_clear()

    assert code == 1
    assert "hunter2" not in out
    assert "password authentication failed" not in out


def test_list_reports_a_missing_connection_string_without_a_traceback(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`get_engine()` raises `RuntimeError` when `CONNECTION_STRING` is unset;
    `main()` must turn that into a clean message and a non-zero exit, not an
    uncaught traceback, exactly as `transactions.py` does.
    """
    monkeypatch.setenv("CONNECTION_STRING", "")
    get_settings.cache_clear()
    pg_engine.get_engine.cache_clear()
    try:
        code = main(["list"])
        out = capsys.readouterr().out
    finally:
        get_settings.cache_clear()
        pg_engine.get_engine.cache_clear()

    assert code == 1
    assert "CONNECTION_STRING" in out
    assert "Traceback" not in out


class _NullSession:
    """Enough Session for the CLI's `with` block; the repository is faked out."""

    def __enter__(self) -> _NullSession:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def commit(self) -> None:
        return None
