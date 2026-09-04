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
from calendar import monthrange
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Final

import pytest
from sqlalchemy.exc import SQLAlchemyError

from offerdelta.application.scope import TenantScope
from offerdelta.application.transactions.enter_transaction import EntryOutcome, ManualEntry
from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.accounts import canonical_account_key
from offerdelta.evaluation.labels import LABEL_SPACE
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


@dataclass
class _FakeTransaction:
    """One row as the fake `TransactionRepository` stores it.

    Carries enough for `for_month` and `confirm_label` to answer honestly -
    whose row it is, when it posted, what it says, and whether a person has
    confirmed it - without any of the columns this suite has no reason to
    fake (fingerprints, occurrences, source files).
    """

    id: uuid.UUID
    user_id: uuid.UUID
    posted_on: date
    description: str
    amount: Money
    confirmed_label: str | None = None


@dataclass
class _FakeBatch:
    """One declared import window, keyed the way the real table is: per account and checksum."""

    id: uuid.UUID
    account_id: uuid.UUID
    mode: str
    window_start: date | None
    window_end: date | None
    row_count: int


class _FakeDatabase:
    """An in-memory stand-in for just enough of Postgres to prove idempotence.

    Shared across two `main()` calls within one test, so a bug that would
    re-create a user, re-register an account, double-enter a transaction,
    re-declare a snapshot window, or re-confirm a label on a second run shows
    up as growth in these collections - not as an assertion about intentions.
    """

    def __init__(self) -> None:
        self.users: dict[str, StoredUser] = {}
        self.accounts: dict[tuple[uuid.UUID, str], StoredAccount] = {}
        #: Identity mirrors what a real fingerprint distinguishes - account,
        #: posted date, description, and amount - mapped to the id assigned
        #: when the row was first entered, so a second `enter_transaction`
        #: call for the same identity reports rather than writes.
        self.entered: dict[tuple[uuid.UUID, str, date, str, str], uuid.UUID] = {}
        self.transactions: dict[uuid.UUID, _FakeTransaction] = {}
        #: Keyed exactly as `uq_import_batches_account_checksum` is: a second
        #: declaration for the same account and checksum must find this row
        #: rather than insert another one.
        self.batches: dict[tuple[uuid.UUID, str], _FakeBatch] = {}
        self.create_calls = 0
        self.set_password_calls = 0
        self.register_calls = 0
        self.entered_calls: list[tuple[str, bool]] = []
        self.confirm_calls: list[tuple[uuid.UUID, str]] = []
        self.open_batch_calls = 0
        #: The last password `set_password` was actually called with, per
        #: address - what proves a rotation reached the store rather than
        #: merely that some call count went up.
        self.passwords: dict[str, str] = {}

    @property
    def stored_transaction_count(self) -> int:
        return len(self.transactions)


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
    """Mirrors `enter_transaction`'s dedup: an identical entry reports, not writes.

    The identity includes `posted_on`, matching what `compute_fingerprint`
    actually distinguishes - a fake that omitted it would treat two genuinely
    different transactions (the same merchant and amount, a month apart) as
    duplicates of each other.
    """

    def _enter(scope: TenantScope, entry: ManualEntry) -> EntryOutcome:
        db.entered_calls.append((entry.description, entry.repeat))
        identity = (
            scope.user.id,
            entry.account_key,
            entry.posted_on,
            entry.description,
            str(entry.amount.amount),
        )
        existing_id = db.entered.get(identity)
        if existing_id is not None:
            return EntryOutcome(
                stored=False,
                transaction_id=None,
                fingerprint="fake",
                occurrence=1,
                already_stored_count=1,
            )
        transaction_id = uuid.uuid4()
        db.entered[identity] = transaction_id
        db.transactions[transaction_id] = _FakeTransaction(
            id=transaction_id,
            user_id=scope.user.id,
            posted_on=entry.posted_on,
            description=entry.description,
            amount=entry.amount,
        )
        return EntryOutcome(
            stored=True, transaction_id=transaction_id, fingerprint="fake", occurrence=1
        )

    return _enter


def _fake_transaction_repository(db: _FakeDatabase) -> type:
    class _FakeTransactionRepository:
        """Mirrors the two `TransactionRepository` methods this script calls."""

        def __init__(self, scope: TenantScope) -> None:
            self._user_id = scope.user.id

        def for_month(self, year: int, month: int) -> list[_FakeTransaction]:
            return [
                txn
                for txn in db.transactions.values()
                if txn.user_id == self._user_id
                and txn.posted_on.year == year
                and txn.posted_on.month == month
            ]

        def confirm_label(
            self,
            transaction_id: uuid.UUID,
            label: str,
            *,
            now: datetime | None = None,  # noqa: ARG002 - real signature, fake tracks no time
        ) -> None:
            if label not in LABEL_SPACE:
                raise ValidationError(f"{label!r} is not a label in this taxonomy")
            txn = db.transactions.get(transaction_id)
            if txn is None or txn.user_id != self._user_id:
                raise ValidationError(f"no such transaction {transaction_id}")
            txn.confirmed_label = label
            db.confirm_calls.append((transaction_id, label))

    return _FakeTransactionRepository


def _fake_import_batch_repository(db: _FakeDatabase) -> type:
    class _FakeImportBatchRepository:
        """Mirrors `ImportBatchRepository.open`: guarded on `(account_id, source_sha256)`."""

        def __init__(self, scope: TenantScope) -> None:
            self._user_id = scope.user.id

        def open(
            self,
            account_id: uuid.UUID,
            *,
            source_file: str,  # noqa: ARG002 - real signature, fake keys on the checksum alone
            source_sha256: str,
            mode: str,
            window_start: date | None,
            window_end: date | None,
            row_count: int,
            now: datetime | None = None,  # noqa: ARG002 - real signature, fake tracks no time
        ) -> tuple[_FakeBatch, bool]:
            db.open_batch_calls += 1
            key = (account_id, source_sha256)
            existing = db.batches.get(key)
            if existing is not None:
                return existing, False
            batch = _FakeBatch(
                id=uuid.uuid4(),
                account_id=account_id,
                mode=mode,
                window_start=window_start,
                window_end=window_end,
                row_count=row_count,
            )
            db.batches[key] = batch
            return batch, True

    return _FakeImportBatchRepository


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
    monkeypatch.setattr("seed_demo.TransactionRepository", _fake_transaction_repository(db))
    monkeypatch.setattr("seed_demo.ImportBatchRepository", _fake_import_batch_repository(db))


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


def test_every_seeded_description_says_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    """No seeded row may be mistaken for real bank activity.

    Checked against what was actually entered, not against the literal
    source in this file, so a row added later without the same convention
    fails this test instead of quietly blending into a deployed database
    that also holds 742 real transactions.
    """
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    main([])

    assert db.transactions
    assert all(txn.description.startswith("Synthetic ") for txn in db.transactions.values())


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
    after_first = (
        len(db.users),
        len(db.accounts),
        db.stored_transaction_count,
        len(db.batches),
    )

    second_code = main([])
    after_second = (
        len(db.users),
        len(db.accounts),
        db.stored_transaction_count,
        len(db.batches),
    )

    assert first_code == 0
    assert second_code == 0
    assert after_first == after_second
    # The user, the account, the transactions, and the snapshot windows are
    # only ever created once.
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


# ---------------------------------------------------------------- labels


def test_every_confirmed_label_is_in_the_label_space(monkeypatch: pytest.MonkeyPatch) -> None:
    """Synthetic data is honestly USER_CONFIRMED: a person wrote it - and every
    label a person could have written is one the taxonomy actually knows.

    Asserted against the real `LABEL_SPACE`, not against a copy of it, so a
    typo in a seeded label - or a category the taxonomy has since dropped -
    fails this test instead of seeding a row no report can place.
    """
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    main([])

    confirmed = [txn for txn in db.transactions.values() if txn.confirmed_label is not None]
    assert confirmed  # something was actually confirmed
    assert all(txn.confirmed_label in LABEL_SPACE for txn in confirmed)


def test_confirming_is_the_only_way_a_label_is_ever_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every confirmed label was actually written through `confirm_label`,
    not assigned directly - proving the fake and the script agree on how a
    label reaches a row, not just on the end state."""
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    main([])

    confirmed_ids = {txn.id for txn in db.transactions.values() if txn.confirmed_label is not None}
    called_ids = {transaction_id for transaction_id, _label in db.confirm_calls}
    assert confirmed_ids == called_ids


def test_the_review_queue_is_not_left_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """A demo where the queue has nothing in it cannot show what the queue is
    for - see the module docstring. Some seeded rows are deliberately left
    without a confirmed label."""
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    main([])

    unconfirmed = [txn for txn in db.transactions.values() if txn.confirmed_label is None]
    assert unconfirmed
    # Deliberately a small minority, not an oversight that swallowed most of
    # the seed: most of what was seeded is decided, some is left to review.
    assert len(unconfirmed) < db.stored_transaction_count / 2


def test_running_main_twice_does_not_double_confirm(monkeypatch: pytest.MonkeyPatch) -> None:
    """`confirm_label` has no guard of its own - see the module docstring - so
    this script must supply one. Without it, a second run would call
    `confirm_label` again for every already-confirmed row."""
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    main([])
    confirmed_after_first = len(db.confirm_calls)

    main([])
    confirmed_after_second = len(db.confirm_calls)

    assert confirmed_after_first > 0
    assert confirmed_after_second == confirmed_after_first


# ---------------------------------------------------------------- months


def _months_per_tenant(db: _FakeDatabase) -> dict[uuid.UUID, set[tuple[int, int]]]:
    months: dict[uuid.UUID, set[tuple[int, int]]] = defaultdict(set)
    for txn in db.transactions.values():
        months[txn.user_id].add((txn.posted_on.year, txn.posted_on.month))
    return months


def test_each_tenant_spans_at_least_two_calendar_months(monkeypatch: pytest.MonkeyPatch) -> None:
    """So the deployed demo can show a month-over-month comparison rather than
    a single static screen."""
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    main([])

    months = _months_per_tenant(db)
    assert len(months) == 2  # one entry per tenant
    for user_id, spanned in months.items():
        assert len(spanned) >= 2, f"tenant {user_id} only spans {spanned}"


def test_every_seeded_month_is_declared_as_a_complete_snapshot_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`available_months` reads completeness from a declared snapshot window,
    never from row density - see `application/reports/monthly.py`. Every
    month this script seeds transactions for must therefore also carry a
    `mode="snapshot"` batch whose window spans that month's full calendar
    range, or the deployed demo would show it as partial regardless of how
    much was actually seeded.
    """
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    main([])

    account_ids_by_user = defaultdict(set)
    for (user_id, _key), account in db.accounts.items():
        account_ids_by_user[user_id].add(account.id)

    for user_id, spanned in _months_per_tenant(db).items():
        (account_id,) = account_ids_by_user[user_id]
        declared = {
            (batch.window_start, batch.window_end)
            for batch in db.batches.values()
            if batch.account_id == account_id and batch.mode == "snapshot"
        }
        for year, month in spanned:
            start = date(year, month, 1)
            end = date(year, month, monthrange(year, month)[1])
            assert (start, end) in declared, (
                f"no complete snapshot window declared for {year}-{month:02d} "
                f"(account {account_id})"
            )


def test_running_main_twice_declares_no_extra_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    db = _FakeDatabase()
    _patch_fakes(monkeypatch, db)

    main([])
    batches_after_first = len(db.batches)
    calls_after_first = db.open_batch_calls

    main([])
    batches_after_second = len(db.batches)
    calls_after_second = db.open_batch_calls

    assert batches_after_first > 0
    assert batches_after_second == batches_after_first
    # `open` is called unconditionally every run - it guards itself - so the
    # call count still doubles even though nothing new is created.
    assert calls_after_second == calls_after_first * 2


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
