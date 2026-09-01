"""Two tenants, not one.

With a single tenant the deployment is single-tenant in practice, and nothing
about isolation is exercised by its existence.

The database tests below fake every repository rather than opening a real
`Session`: the one Postgres this checkout can reach already holds 742 real
bank transactions, and this suite must never be the thing that writes a
synthetic row next to them. The fakes mirror the real repositories' contract
closely enough to prove what matters - existence is checked before every
write except the password, so a second run creates nothing and still rotates
a changed credential - without touching that database.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

import pytest
from sqlalchemy.exc import SQLAlchemyError

from offerdelta.application.scope import TenantScope
from offerdelta.application.transactions.enter_transaction import EntryOutcome, ManualEntry
from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.transactions.accounts import canonical_account_key
from offerdelta.infrastructure.postgres import engine as pg_engine
from offerdelta.infrastructure.postgres.repositories import StoredAccount, StoredUser
from seed_demo import SYNTHETIC_TENANTS, build_parser, main

_DEMO_PASSWORD: Final = "OFFERDELTA_DEMO_PASSWORD"
_OTHER_PASSWORD: Final = "OFFERDELTA_OTHER_PASSWORD"


def test_there_are_two_tenants() -> None:
    assert len(SYNTHETIC_TENANTS) == 2


def test_the_addresses_are_obviously_not_real() -> None:
    assert all(email.endswith("@example.test") for email in SYNTHETIC_TENANTS)


def test_an_unknown_argument_is_an_error() -> None:
    """No flag exists on this parser at all - see the password tests below for why."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--admin"])


def test_there_is_no_password_flag_to_reach() -> None:
    """The only way a password could land in argv, shell history, or `ps` output."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--password", "hunter2"])


# ---------------------------------------------------------------- passwords


def test_refuses_when_the_demo_password_is_unset(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(_DEMO_PASSWORD, raising=False)
    monkeypatch.setenv(_OTHER_PASSWORD, "a good long passphrase for other")

    code = main([])
    out = capsys.readouterr().out

    assert code != 0
    assert _DEMO_PASSWORD in out


def test_refuses_when_the_other_password_is_unset(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_DEMO_PASSWORD, "a good long passphrase for demo")
    monkeypatch.delenv(_OTHER_PASSWORD, raising=False)

    code = main([])
    out = capsys.readouterr().out

    assert code != 0
    assert _OTHER_PASSWORD in out


def test_missing_password_never_opens_a_database_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The check comes before `get_engine()`, exactly like `users.py set-password`."""
    monkeypatch.delenv(_DEMO_PASSWORD, raising=False)
    monkeypatch.delenv(_OTHER_PASSWORD, raising=False)

    def _boom() -> None:
        raise AssertionError("get_engine() must not be called before both passwords are checked")

    monkeypatch.setattr("seed_demo.get_engine", _boom)

    code = main([])
    assert code != 0


# ---------------------------------------------------------------- idempotence


class _FakeDatabase:
    """An in-memory stand-in for just enough of Postgres to prove idempotence.

    Shared across two `main()` calls within one test, so a bug that would
    re-create a user, re-register an account, or double-enter a transaction on
    a second run shows up as growth in these collections - not as an assertion
    about intentions.
    """

    def __init__(self) -> None:
        self.users: dict[str, StoredUser] = {}
        self.accounts: dict[tuple[uuid.UUID, str], StoredAccount] = {}
        self.entered: set[tuple[uuid.UUID, str, str, str]] = set()
        self.create_calls = 0
        self.set_password_calls = 0
        self.register_calls = 0
        self.entered_calls: list[tuple[str, bool]] = []
        #: The last password `set_password` was actually called with, per
        #: address - what proves a rotation reached the store rather than
        #: merely that some call count went up.
        self.passwords: dict[str, str] = {}

    @property
    def stored_transaction_count(self) -> int:
        return len(self.entered)


def _fake_user_repository(db: _FakeDatabase) -> type:
    class _FakeUserRepository:
        """Mirrors `UserRepository`: `create` raises on a duplicate email."""

        def __init__(self, _session: object) -> None:
            pass

        def by_email(self, email: str) -> StoredUser | None:
            return db.users.get(email)

        def create(self, email: str, display_name: str) -> StoredUser:
            if email in db.users:
                raise ValidationError(f"user {email!r} already exists")
            user = StoredUser(
                id=uuid.uuid4(),
                email=email,
                display_name=display_name,
                is_active=True,
                has_password=False,
            )
            db.users[email] = user
            db.create_calls += 1
            return user

        def set_password(self, email: str, plain: str) -> None:
            db.set_password_calls += 1
            db.passwords[email] = plain

    return _FakeUserRepository


def _fake_account_repository(db: _FakeDatabase) -> type:
    class _FakeAccountRepository:
        """Mirrors `AccountRepository`: keyed per user, `register` raises on a duplicate."""

        def __init__(self, scope: TenantScope) -> None:
            self._user_id = scope.user.id

        def by_key(self, key: str) -> StoredAccount | None:
            return db.accounts.get((self._user_id, canonical_account_key(key)))

        def register(self, display_name: str) -> StoredAccount:
            key = canonical_account_key(display_name)
            if (self._user_id, key) in db.accounts:
                raise ValidationError(f"account {key!r} is already registered")
            account = StoredAccount(
                id=uuid.uuid4(),
                key=key,
                display_name=display_name.strip(),
                created_at=datetime.now(UTC),
            )
            db.accounts[(self._user_id, key)] = account
            db.register_calls += 1
            return account

    return _FakeAccountRepository


def _fake_enter_transaction(
    db: _FakeDatabase,
) -> Callable[[TenantScope, ManualEntry], EntryOutcome]:
    """Mirrors `enter_transaction`'s dedup: an identical entry reports, not writes."""

    def _enter(scope: TenantScope, entry: ManualEntry) -> EntryOutcome:
        db.entered_calls.append((entry.description, entry.repeat))
        identity = (
            scope.user.id,
            entry.account_key,
            entry.description,
            str(entry.amount.amount),
        )
        if identity in db.entered:
            return EntryOutcome(
                stored=False,
                transaction_id=None,
                fingerprint="fake",
                occurrence=1,
                already_stored_count=1,
            )
        db.entered.add(identity)
        return EntryOutcome(
            stored=True, transaction_id=uuid.uuid4(), fingerprint="fake", occurrence=1
        )

    return _enter


class _NullSession:
    """Enough `Session` for the CLI's `with` block; the repositories are faked out."""

    def __enter__(self) -> _NullSession:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def commit(self) -> None:
        return None


def _patch_fakes(monkeypatch: pytest.MonkeyPatch, db: _FakeDatabase) -> None:
    monkeypatch.setenv(_DEMO_PASSWORD, "a good long passphrase for demo")
    monkeypatch.setenv(_OTHER_PASSWORD, "a good long passphrase for other")
    monkeypatch.setattr("seed_demo.get_engine", lambda: None)
    monkeypatch.setattr("seed_demo.Session", lambda _engine: _NullSession())
    monkeypatch.setattr("seed_demo.UserRepository", _fake_user_repository(db))
    monkeypatch.setattr("seed_demo.AccountRepository", _fake_account_repository(db))
    monkeypatch.setattr("seed_demo.enter_transaction", _fake_enter_transaction(db))


def test_the_first_run_creates_both_tenants(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    code = main([])

    assert code == 0
    assert set(db.users) == set(SYNTHETIC_TENANTS)
    assert db.create_calls == 2
    assert db.set_password_calls == 2
    assert db.register_calls == 2
    assert db.stored_transaction_count > 0


def test_every_entry_is_written_with_repeat_false(monkeypatch: pytest.MonkeyPatch) -> None:
    """`repeat=True` is the one flag that would defeat `enter_transaction`'s own
    dedup - if this script ever passed it, a second run would double every row."""
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    main([])

    assert db.entered_calls  # something was actually entered
    assert all(repeat is False for _description, repeat in db.entered_calls)


def test_running_main_twice_creates_nothing_the_second_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    first_code = main([])
    after_first = (len(db.users), len(db.accounts), db.stored_transaction_count)

    second_code = main([])
    after_second = (len(db.users), len(db.accounts), db.stored_transaction_count)

    assert first_code == 0
    assert second_code == 0
    assert after_first == after_second
    # The user, the account, and the transactions are only ever created once.
    assert db.create_calls == 2
    assert db.register_calls == 2
    # The password is the one write that is not read-guarded - see the module
    # docstring - so it fires again on the second run, once per tenant.
    assert db.set_password_calls == 4


def test_a_tenant_whose_user_already_exists_is_not_recreated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user created out of band (or by a previous partial run) is not
    recreated - but does still get its password (re-)applied, exactly like a
    tenant this run created itself. See the module docstring for why."""
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)
    db.users[SYNTHETIC_TENANTS[0]] = StoredUser(
        id=uuid.uuid4(),
        email=SYNTHETIC_TENANTS[0],
        display_name="Demo Tenant",
        is_active=True,
        has_password=True,
    )

    code = main([])

    assert code == 0
    assert db.create_calls == 1  # only the other tenant
    assert db.set_password_calls == 2  # both addresses, including the existing one


def test_a_changed_password_reaches_a_user_that_already_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bug this closes: a re-run with a rotated `OFFERDELTA_DEMO_PASSWORD`
    must not silently leave a pre-existing user's stored password untouched."""
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    main([])
    assert db.passwords[SYNTHETIC_TENANTS[0]] == "a good long passphrase for demo"

    monkeypatch.setenv(_DEMO_PASSWORD, "a completely different passphrase for demo")
    code = main([])

    assert code == 0
    assert db.passwords[SYNTHETIC_TENANTS[0]] == "a completely different passphrase for demo"


# ---------------------------------------------------------------- error handling


def test_a_database_error_never_echoes_credentials_or_the_raw_message(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config.py` promises the connection string is never logged, echoed, or put
    into an error message. A raw SQLAlchemy exception can carry the DSN and the
    data it was writing, so only the redacted host may reach the terminal.
    """

    class _BoomUserRepository:
        def __init__(self, _session: object) -> None:
            pass

        def by_email(self, _email: str) -> StoredUser | None:
            raise SQLAlchemyError(
                'connection to server failed: password authentication failed for user "hunter2"'
            )

    monkeypatch.setenv(_DEMO_PASSWORD, "a good long passphrase for demo")
    monkeypatch.setenv(_OTHER_PASSWORD, "a good long passphrase for other")
    monkeypatch.setenv("CONNECTION_STRING", "postgresql://user:hunter2@dbhost/offerdelta")
    get_settings.cache_clear()
    monkeypatch.setattr("seed_demo.get_engine", lambda: None)
    monkeypatch.setattr("seed_demo.Session", lambda _engine: _NullSession())
    monkeypatch.setattr("seed_demo.UserRepository", _BoomUserRepository)
    try:
        code = main([])
        out = capsys.readouterr().out
    finally:
        get_settings.cache_clear()

    assert code == 1
    assert "hunter2" not in out
    assert "password authentication failed" not in out


def test_reports_a_missing_connection_string_without_a_traceback(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`get_engine()` raises `RuntimeError` when `CONNECTION_STRING` is unset;
    `main()` must turn that into a clean message and a non-zero exit, exactly as
    `users.py` and `transactions.py` do."""
    monkeypatch.setenv(_DEMO_PASSWORD, "a good long passphrase for demo")
    monkeypatch.setenv(_OTHER_PASSWORD, "a good long passphrase for other")
    monkeypatch.setenv("CONNECTION_STRING", "")
    get_settings.cache_clear()
    pg_engine.get_engine.cache_clear()
    try:
        code = main([])
        out = capsys.readouterr().out
    finally:
        get_settings.cache_clear()
        pg_engine.get_engine.cache_clear()

    assert code == 1
    assert "CONNECTION_STRING" in out
    assert "Traceback" not in out


def test_a_validation_error_is_reported_without_a_traceback(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BoomUserRepository:
        def __init__(self, _session: object) -> None:
            pass

        def by_email(self, email: str) -> StoredUser | None:
            raise ValidationError(f"synthetic failure for {email!r}")

    monkeypatch.setenv(_DEMO_PASSWORD, "a good long passphrase for demo")
    monkeypatch.setenv(_OTHER_PASSWORD, "a good long passphrase for other")
    monkeypatch.setattr("seed_demo.get_engine", lambda: None)
    monkeypatch.setattr("seed_demo.Session", lambda _engine: _NullSession())
    monkeypatch.setattr("seed_demo.UserRepository", _BoomUserRepository)

    code = main([])
    out = capsys.readouterr().out

    assert code == 1
    assert "synthetic failure" in out
    assert "Traceback" not in out
