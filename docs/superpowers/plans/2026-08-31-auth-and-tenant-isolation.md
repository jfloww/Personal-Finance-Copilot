# Authentication and Tenant Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put identity on every request and a tenant on every query, so one user cannot read or write another user's financial data, and prove it in CI.

**Architecture:** A `users` table becomes the tenant root through `accounts.user_id`; `import_batches` and `transactions` inherit tenancy by foreign key. Repositories take a `TenantScope` (session + authenticated user) where they take a bare `Session` today, so a query without a tenant cannot be expressed. A JWT bearer token identifies the caller; the user row is loaded per request so deactivation is immediate.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy 2, Alembic, PostgreSQL, `argon2-cffi`, `PyJWT`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-31-auth-and-tenant-isolation-design.md`

## Global Constraints

- **Domain stays standard-library only.** `offerdelta.domain` must not import fastapi, pydantic, sqlalchemy, alembic, httpx, or requests. Nothing in this plan adds code to `offerdelta.domain`.
- **Layers:** `offerdelta.api` → `offerdelta.application` → `offerdelta.domain`. Never the reverse.
- **mypy is `strict = true` with `disallow_any_explicit = true`.** Every function needs annotations. No bare `Any`.
- **ruff line-length is 100.**
- **Lint and format cover one list, shared with CI:** `$(PY_FILES)` in the `Makefile`. New files under `src/` and `tests/` are already covered by the `src` and `tests` entries.
- **CI has PostgreSQL** (service container, `CONNECTION_STRING` set at job level, `alembic upgrade head` before the suite). Integration tests run in CI.
- **Secrets are never logged, echoed, or put in an error message.** This now includes `JWT_SECRET`.
- **404, never 403, for another tenant's resource.** A 403 confirms the resource exists.
- **The full gate is `make check`** — `lint`, `types`, `arch`, `test`.

---

### Task 1: `JWT_SECRET` setting and `auth_available`

Follows the existing `database_available` / `llm_available` pattern so CI stays green without a secret.

**Files:**
- Modify: `backend/src/offerdelta/config.py`
- Test: `backend/tests/unit/test_config.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Settings.jwt_secret: str | None`, `Settings.auth_available: bool`

- [ ] **Step 1: Write the failing test**

Append to `backend/tests/unit/test_config.py`:

```python
def test_auth_is_unavailable_without_a_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("JWT_SECRET", raising=False)
    get_settings.cache_clear()
    assert get_settings().auth_available is False


def test_auth_is_available_with_a_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    get_settings.cache_clear()
    assert get_settings().auth_available is True
    get_settings.cache_clear()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_config.py -k auth -v`
Expected: FAIL with `AttributeError: 'Settings' object has no attribute 'auth_available'`

- [ ] **Step 3: Write minimal implementation**

In `config.py`, beside `anthropic_api_key`:

```python
    #: Signing key for access tokens. Absent in CI by design, exactly like the
    #: Anthropic key: a suite that needs a secret to go green teaches people to
    #: ignore red builds. Absent means authentication is switched off and the
    #: routes that need it disappear rather than half-work.
    jwt_secret: str | None = Field(default=None, alias="JWT_SECRET")
```

And beside `llm_available`:

```python
    @property
    def auth_available(self) -> bool:
        return self.jwt_secret is not None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/unit/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Confirm the secret cannot leak**

Run: `cd backend && grep -n "jwt_secret" src/offerdelta/config.py`
Expected: `jwt_secret` appears only in the field definition and `auth_available`. It must not appear in `redacted_dsn` output, `__repr__`, or any log line.

- [ ] **Step 6: Commit**

```bash
git add backend/src/offerdelta/config.py backend/tests/unit/test_config.py
git commit -m "Settings: JWT_SECRET, absent by default like the Anthropic key"
```

---

### Task 2: Password hashing

**Files:**
- Create: `backend/src/offerdelta/infrastructure/auth/__init__.py`
- Create: `backend/src/offerdelta/infrastructure/auth/passwords.py`
- Create: `backend/tests/unit/auth/__init__.py`
- Create: `backend/tests/unit/auth/test_passwords.py`
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Consumes: nothing
- Produces: `hash_password(plain: str) -> str`, `verify_password(plain: str, hashed: str | None) -> bool`

- [ ] **Step 1: Add the dependency**

In `backend/pyproject.toml`, under `dependencies`, add:

```toml
    "argon2-cffi>=23.1",
```

Run: `cd backend && uv sync`

- [ ] **Step 2: Write the failing test**

Create `backend/tests/unit/auth/__init__.py` (empty) and `backend/tests/unit/auth/test_passwords.py`:

```python
"""Password hashing.

argon2id over bcrypt because bcrypt silently truncates at 72 bytes: two
different long passwords can hash the same, and nothing tells anybody.
"""

from __future__ import annotations

from offerdelta.infrastructure.auth.passwords import hash_password, verify_password


def test_a_hash_verifies_against_its_own_password() -> None:
    hashed = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", hashed) is True


def test_a_hash_rejects_a_different_password() -> None:
    hashed = hash_password("correct horse battery staple")
    assert verify_password("Correct horse battery staple", hashed) is False


def test_the_same_password_hashes_differently_each_time() -> None:
    assert hash_password("same input") != hash_password("same input")


def test_long_passwords_are_not_truncated() -> None:
    base = "a" * 100
    hashed = hash_password(base + "one")
    assert verify_password(base + "two", hashed) is False


def test_a_null_hash_never_verifies() -> None:
    """`password_hash IS NULL` means the user cannot log in at all."""
    assert verify_password("anything", None) is False


def test_verifying_against_null_still_does_the_work() -> None:
    """A missing user must not be faster than a wrong password."""
    import time

    hashed = hash_password("real password")

    start = time.perf_counter()
    verify_password("guess", hashed)
    with_hash = time.perf_counter() - start

    start = time.perf_counter()
    verify_password("guess", None)
    without_hash = time.perf_counter() - start

    # Generous: this asserts the dummy path runs a real verification at all,
    # not that the two are equal. Equality would be flaky on shared CI.
    assert without_hash > with_hash / 10
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/auth/test_passwords.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'offerdelta.infrastructure.auth'`

- [ ] **Step 4: Write the implementation**

Create `backend/src/offerdelta/infrastructure/auth/__init__.py` (empty) and `passwords.py`:

```python
"""Password hashing.

argon2id rather than bcrypt. bcrypt truncates silently at 72 bytes, which
turns two different long passwords into the same hash and tells nobody.

`verify_password` accepts a `None` hash and still performs a verification
against a throwaway hash before returning False. A user who does not exist
must not answer faster than one who does, because the difference is how an
attacker enumerates addresses.
"""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

_hasher = PasswordHasher()

#: Verified against when there is no real hash, purely to spend the time.
_DUMMY_HASH = _hasher.hash("dummy password for constant-work verification")


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str | None) -> bool:
    target = hashed if hashed is not None else _DUMMY_HASH
    try:
        _hasher.verify(target, plain)
    except (VerifyMismatchError, VerificationError):
        return False
    return hashed is not None
```

- [ ] **Step 5: Run the tests**

Run: `cd backend && uv run pytest tests/unit/auth/test_passwords.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/offerdelta/infrastructure/auth/ backend/tests/unit/auth/
git commit -m "Auth: argon2id password hashing that costs the same when the user does not exist"
```

---

### Task 3: Token issuing and verification

**Files:**
- Create: `backend/src/offerdelta/infrastructure/auth/tokens.py`
- Create: `backend/tests/unit/auth/test_tokens.py`
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Consumes: `Settings.jwt_secret` (Task 1)
- Produces:
  - `TOKEN_TTL: Final[timedelta]` (one hour)
  - `issue_token(user_id: uuid.UUID, *, secret: str, now: datetime | None = None) -> str`
  - `decode_token(token: str, *, secret: str, now: datetime | None = None) -> uuid.UUID | None` — `None` for every rejection: expired, wrong signature, malformed, missing or unparseable `sub`.

- [ ] **Step 1: Add the dependency**

In `backend/pyproject.toml`, under `dependencies`, add:

```toml
    "pyjwt>=2.9",
```

Run: `cd backend && uv sync`

- [ ] **Step 2: Write the failing test**

Create `backend/tests/unit/auth/test_tokens.py`:

```python
"""Access tokens.

Every rejection returns None rather than a reason. The caller turns all of
them into the same 401, so a distinguishable reason would only ever leak.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from offerdelta.infrastructure.auth.tokens import TOKEN_TTL, decode_token, issue_token

SECRET = "test secret, at least thirty-two characters long"
OTHER_SECRET = "a different secret, also long enough for use"
NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)


def test_a_token_round_trips_to_its_subject() -> None:
    user_id = uuid.uuid4()
    token = issue_token(user_id, secret=SECRET, now=NOW)
    assert decode_token(token, secret=SECRET, now=NOW) == user_id


def test_a_token_is_valid_just_before_it_expires() -> None:
    user_id = uuid.uuid4()
    token = issue_token(user_id, secret=SECRET, now=NOW)
    just_inside = NOW + TOKEN_TTL - timedelta(seconds=1)
    assert decode_token(token, secret=SECRET, now=just_inside) == user_id


def test_an_expired_token_is_rejected() -> None:
    token = issue_token(uuid.uuid4(), secret=SECRET, now=NOW)
    past_expiry = NOW + TOKEN_TTL + timedelta(seconds=1)
    assert decode_token(token, secret=SECRET, now=past_expiry) is None


def test_a_token_signed_with_another_secret_is_rejected() -> None:
    token = issue_token(uuid.uuid4(), secret=OTHER_SECRET, now=NOW)
    assert decode_token(token, secret=SECRET, now=NOW) is None


def test_a_malformed_token_is_rejected() -> None:
    assert decode_token("not.a.token", secret=SECRET, now=NOW) is None


def test_an_empty_token_is_rejected() -> None:
    assert decode_token("", secret=SECRET, now=NOW) is None


def test_the_ttl_is_one_hour() -> None:
    assert TOKEN_TTL == timedelta(hours=1)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/auth/test_tokens.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'offerdelta.infrastructure.auth.tokens'`

- [ ] **Step 4: Write the implementation**

Create `backend/src/offerdelta/infrastructure/auth/tokens.py`:

```python
"""Access tokens.

HS256, one hour, three claims. There are no roles to encode, and a claim that
exists is a claim something will eventually trust.

There is no refresh token. A refresh token is only meaningful next to a
revocation store; without one it is a second secret with a longer life and no
compensating power. Re-authenticating is one POST.

Every failure returns None. The caller turns all of them into the same 401, so
distinguishing them here could only ever leak.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Final

import jwt

#: Short enough that a leaked token stops working the same afternoon, long
#: enough that a person is not re-authenticating mid-task.
TOKEN_TTL: Final = timedelta(hours=1)

_ALGORITHM: Final = "HS256"


def issue_token(user_id: uuid.UUID, *, secret: str, now: datetime | None = None) -> str:
    issued_at = now or datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + TOKEN_TTL).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def decode_token(token: str, *, secret: str, now: datetime | None = None) -> uuid.UUID | None:
    at = now or datetime.now(UTC)
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[_ALGORITHM],
            options={"require": ["sub", "exp", "iat"]},
        )
    except jwt.PyJWTError:
        return None

    expires_at = claims.get("exp")
    if not isinstance(expires_at, int) or at.timestamp() >= expires_at:
        return None

    subject = claims.get("sub")
    if not isinstance(subject, str):
        return None
    try:
        return uuid.UUID(subject)
    except ValueError:
        return None
```

Note: `jwt.decode` checks `exp` against the real clock, which is why the
explicit `now` comparison follows it — the tests drive time by parameter, and
the parameter must be what decides.

- [ ] **Step 5: Run the tests**

Run: `cd backend && uv run pytest tests/unit/auth/test_tokens.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/src/offerdelta/infrastructure/auth/tokens.py backend/tests/unit/auth/test_tokens.py
git commit -m "Auth: HS256 access tokens, one hour, every rejection identical"
```

---

### Task 4: `users` table and the `accounts.user_id` migration

The only destructive change in the plan. `accounts.key` loses its global
uniqueness for `(user_id, key)`, because the natural key of a bank account is
the same shape for everybody.

**Files:**
- Modify: `backend/src/offerdelta/infrastructure/postgres/models.py`
- Create: `backend/migrations/versions/<generated>_users_and_account_ownership.py`
- Create: `backend/tests/integration/test_users_migration.py`

**Interfaces:**
- Consumes: nothing
- Produces: `UserRow` (`id`, `email`, `password_hash`, `display_name`, `is_active`, `created_at`), `AccountRow.user_id`, `PLACEHOLDER_USER_ID: Final[uuid.UUID]`

- [ ] **Step 1: Write the failing migration test**

Create `backend/tests/integration/test_users_migration.py`:

```python
"""The one step that is awkward to undo.

Two cases matter: a database with existing accounts must end up with exactly
one placeholder owner, and an empty database must gain no user row at all.
CI and a fresh Render deploy are the empty case, and inventing a user there
would ship a row nobody asked for.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import Engine, text

from offerdelta.infrastructure.postgres.models import PLACEHOLDER_USER_ID
from tests.integration.conftest import requires_database

pytestmark = requires_database


def _upgrade_to(engine: Engine, schema: str, revision: str) -> None:
    from alembic import command
    from alembic.config import Config

    config = Config("alembic.ini")
    config.set_main_option("version_locations", "migrations/versions")
    config.attributes["connection_schema"] = schema
    with engine.begin() as conn:
        conn.execute(text(f'SET search_path TO "{schema}"'))
        config.attributes["connection"] = conn
        command.upgrade(config, revision)


def test_existing_accounts_gain_one_placeholder_owner(
    engine: Engine, scratch_schema: str
) -> None:
    _upgrade_to(engine, scratch_schema, "3dfb104aabcb")

    account_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(text(f'SET search_path TO "{scratch_schema}"'))
        conn.execute(
            text(
                "INSERT INTO accounts (id, key, display_name, created_at) "
                "VALUES (:id, 'chase-checking-5718', 'Chase Checking', now())"
            ),
            {"id": account_id},
        )

    _upgrade_to(engine, scratch_schema, "head")

    with engine.begin() as conn:
        conn.execute(text(f'SET search_path TO "{scratch_schema}"'))
        users = conn.execute(text("SELECT id, password_hash FROM users")).all()
        owner = conn.execute(
            text("SELECT user_id FROM accounts WHERE id = :id"), {"id": account_id}
        ).scalar_one()

    assert len(users) == 1
    assert users[0][0] == PLACEHOLDER_USER_ID
    assert users[0][1] is None, "the placeholder must not be able to log in"
    assert owner == PLACEHOLDER_USER_ID


def test_an_empty_database_gains_no_user(engine: Engine, scratch_schema: str) -> None:
    _upgrade_to(engine, scratch_schema, "head")

    with engine.begin() as conn:
        conn.execute(text(f'SET search_path TO "{scratch_schema}"'))
        count = conn.execute(text("SELECT count(*) FROM users")).scalar_one()

    assert count == 0


def test_two_users_may_hold_the_same_account_key(
    engine: Engine, scratch_schema: str
) -> None:
    """The point of the destructive change."""
    _upgrade_to(engine, scratch_schema, "head")

    first, second = uuid.uuid4(), uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(text(f'SET search_path TO "{scratch_schema}"'))
        for user_id, email in ((first, "a@example.test"), (second, "b@example.test")):
            conn.execute(
                text(
                    "INSERT INTO users (id, email, display_name, is_active, created_at) "
                    "VALUES (:id, :email, 'Someone', true, now())"
                ),
                {"id": user_id, "email": email},
            )
            conn.execute(
                text(
                    "INSERT INTO accounts (id, user_id, key, display_name, created_at) "
                    "VALUES (:id, :user_id, 'chase-checking-5718', 'Chase', now())"
                ),
                {"id": uuid.uuid4(), "user_id": user_id},
            )

        held = conn.execute(
            text("SELECT count(*) FROM accounts WHERE key = 'chase-checking-5718'")
        ).scalar_one()

    assert held == 2


def test_the_same_key_twice_for_one_user_is_still_refused(
    engine: Engine, scratch_schema: str
) -> None:
    from sqlalchemy.exc import IntegrityError

    _upgrade_to(engine, scratch_schema, "head")

    user_id = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(text(f'SET search_path TO "{scratch_schema}"'))
        conn.execute(
            text(
                "INSERT INTO users (id, email, display_name, is_active, created_at) "
                "VALUES (:id, 'a@example.test', 'Someone', true, now())"
            ),
            {"id": user_id},
        )
        conn.execute(
            text(
                "INSERT INTO accounts (id, user_id, key, display_name, created_at) "
                "VALUES (:id, :user_id, 'chase-checking-5718', 'Chase', now())"
            ),
            {"id": uuid.uuid4(), "user_id": user_id},
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(text(f'SET search_path TO "{scratch_schema}"'))
            conn.execute(
                text(
                    "INSERT INTO accounts (id, user_id, key, display_name, created_at) "
                    "VALUES (:id, :user_id, 'chase-checking-5718', 'Chase', now())"
                ),
                {"id": uuid.uuid4(), "user_id": user_id},
            )
```

If `_upgrade_to` does not drive Alembic correctly against a named schema in
this repo's configuration, adapt it to match `backend/alembic.ini` and
`backend/migrations/env.py` — the assertions are the contract, not the helper.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/integration/test_users_migration.py -v`
Expected: FAIL with `ImportError: cannot import name 'PLACEHOLDER_USER_ID'`

- [ ] **Step 3: Add the models**

In `models.py`, above `AccountRow`:

```python
#: Fixed so the migration is deterministic and the row is recognisable. Only
#: ever created when there are orphan accounts to adopt.
PLACEHOLDER_USER_ID: Final = uuid.UUID("00000000-0000-4000-8000-000000000001")


class UserRow(Base):
    """Somebody who can hold accounts.

    `password_hash` is nullable, and NULL means *this user cannot log in*. The
    migration uses that to adopt existing rows without inventing a credential,
    and `users set-password` is how a real one arrives.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True)
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    display_name: Mapped[str] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
```

Change `AccountRow`: add the column, drop `unique=True` from `key`, add the
composite constraint.

```python
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id"))
    key: Mapped[str] = mapped_column(String(100))
    display_name: Mapped[str] = mapped_column(String(200))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        UniqueConstraint("user_id", "key", name="uq_accounts_user_key"),
    )
```

Add `Boolean` and `Final` to the imports if they are not already there.

- [ ] **Step 4: Write the migration**

Run: `cd backend && uv run alembic revision -m "users and account ownership"`

Then fill the generated file. `down_revision` must be `"3dfb104aabcb"`.

```python
def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.Text(), nullable=True),
        sa.Column("display_name", sa.String(length=200), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("email", name="uq_users_email"),
    )

    # Nullable first: the column has to exist before it can be filled.
    op.add_column("accounts", sa.Column("user_id", sa.Uuid(), nullable=True))

    connection = op.get_bind()
    orphans = connection.execute(
        sa.text("SELECT count(*) FROM accounts WHERE user_id IS NULL")
    ).scalar_one()

    # Only when there is something to adopt. An empty database - CI, and Render
    # on first deploy - must not gain a user row nobody asked for.
    if orphans:
        connection.execute(
            sa.text(
                "INSERT INTO users (id, email, password_hash, display_name, "
                "is_active, created_at) VALUES (:id, :email, NULL, :name, true, now())"
            ),
            {
                "id": PLACEHOLDER_USER_ID,
                "email": "placeholder@localhost.invalid",
                "name": "Placeholder owner (set a password to claim)",
            },
        )
        connection.execute(
            sa.text("UPDATE accounts SET user_id = :id WHERE user_id IS NULL"),
            {"id": PLACEHOLDER_USER_ID},
        )

    op.alter_column("accounts", "user_id", nullable=False)
    op.create_foreign_key(
        "fk_accounts_user_id", "accounts", "users", ["user_id"], ["id"]
    )

    # The destructive change: a bank account key is the same shape for
    # everybody, so it can only be unique within one owner.
    op.drop_constraint("accounts_key_key", "accounts", type_="unique")
    op.create_unique_constraint("uq_accounts_user_key", "accounts", ["user_id", "key"])


def downgrade() -> None:
    op.drop_constraint("uq_accounts_user_key", "accounts", type_="unique")
    op.create_unique_constraint("accounts_key_key", "accounts", ["key"])
    op.drop_constraint("fk_accounts_user_id", "accounts", type_="foreignkey")
    op.drop_column("accounts", "user_id")
    op.drop_table("users")
```

Import `PLACEHOLDER_USER_ID` at the top:
`from offerdelta.infrastructure.postgres.models import PLACEHOLDER_USER_ID`

If the existing unique constraint on `accounts.key` has a different name, find
it first and use that name:

```bash
cd backend && uv run python -c "
from sqlalchemy import inspect
from offerdelta.infrastructure.postgres.engine import get_engine
print([c['name'] for c in inspect(get_engine()).get_unique_constraints('accounts')])
"
```

- [ ] **Step 5: Run the migration tests**

Run: `cd backend && uv run pytest tests/integration/test_users_migration.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Apply it to the local database and confirm the real backfill**

Run: `cd backend && uv run alembic upgrade head`
Run:
```bash
cd backend && uv run python -c "
from sqlalchemy import text
from offerdelta.infrastructure.postgres.engine import get_engine
with get_engine().connect() as c:
    print('users:', c.execute(text('select count(*) from users')).scalar_one())
    print('accounts without owner:', c.execute(text('select count(*) from accounts where user_id is null')).scalar_one())
    print('accounts:', c.execute(text('select count(*) from accounts')).scalar_one())
"
```
Expected: `users: 1`, `accounts without owner: 0`, `accounts: 7`

- [ ] **Step 7: Commit**

```bash
git add backend/src/offerdelta/infrastructure/postgres/models.py backend/migrations/versions/ backend/tests/integration/test_users_migration.py
git commit -m "Schema: users, account ownership, and a per-owner account key"
```

---

### Task 5: `TenantScope` and `UserRepository`

`UserRepository` is deliberately **not** tenant-scoped: it operates on the
users themselves, which is the one place tenancy cannot apply.

**Files:**
- Create: `backend/src/offerdelta/application/scope.py`
- Modify: `backend/src/offerdelta/infrastructure/postgres/repositories.py`
- Create: `backend/tests/integration/test_user_repository.py`

**Interfaces:**
- Consumes: `UserRow`, `hash_password`, `verify_password`
- Produces:
  - `AuthenticatedUser(id: uuid.UUID, email: str)` — frozen dataclass
  - `TenantScope(session: Session, user: AuthenticatedUser)` — frozen dataclass
  - `StoredUser(id, email, display_name, is_active, has_password)` — frozen dataclass
  - `UserRepository(session)` with `create(email, display_name, *, now=None) -> StoredUser`, `by_email(email) -> StoredUser | None`, `by_id(user_id) -> StoredUser | None`, `set_password(email, plain) -> None`, `deactivate(email) -> None`, `all() -> list[StoredUser]`, `authenticate(email, plain) -> AuthenticatedUser | None`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/integration/test_user_repository.py`:

```python
"""Users, and the one query that decides who a caller is."""

from __future__ import annotations

import pytest
from sqlalchemy.orm import Session

from offerdelta.domain.common.errors import ValidationError
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/integration/test_user_repository.py -v`
Expected: FAIL with `ImportError: cannot import name 'UserRepository'`

- [ ] **Step 3: Write the scope types**

Create `backend/src/offerdelta/application/scope.py`:

```python
"""Who is asking, and the session they are asking through.

A repository takes one of these where it used to take a bare `Session`, so a
query without a tenant is not something a caller can forget to write - it is
something they cannot express.

`AuthenticatedUser` is the shape after the checks have passed, which is why it
carries no `is_active`: an inactive user never becomes one. Holding a
`TenantScope` means identity was established, not that it still needs
verifying.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session


@dataclass(frozen=True)
class AuthenticatedUser:
    id: uuid.UUID
    email: str


@dataclass(frozen=True)
class TenantScope:
    session: Session
    user: AuthenticatedUser
```

- [ ] **Step 4: Write `UserRepository`**

In `repositories.py`, add near the other repositories:

```python
@dataclass(frozen=True)
class StoredUser:
    id: uuid.UUID
    email: str
    display_name: str
    is_active: bool
    has_password: bool


def _normalise_email(raw: str) -> str:
    """Addresses are compared lowercased, so one person is one row."""
    return raw.strip().lower()


class UserRepository:
    """The one repository that is not tenant-scoped.

    It operates on the tenants themselves, so there is no outer tenant to scope
    it to. Everything else in this module takes a `TenantScope`.
    """

    def __init__(self, session: Session) -> None:
        self._session = session

    def create(
        self, email: str, display_name: str, *, now: datetime | None = None
    ) -> StoredUser:
        address = _normalise_email(email)
        if self.by_email(address) is not None:
            raise ValidationError(f"user {address!r} already exists")
        row = UserRow(
            id=uuid.uuid4(),
            email=address,
            password_hash=None,
            display_name=display_name.strip(),
            is_active=True,
            created_at=now or datetime.now(UTC),
        )
        self._session.add(row)
        try:
            self._session.flush()
        except IntegrityError as error:
            raise ValidationError(f"user {address!r} already exists") from error
        return _to_stored_user(row)

    def by_email(self, email: str) -> StoredUser | None:
        row = self._row(email)
        return None if row is None else _to_stored_user(row)

    def by_id(self, user_id: uuid.UUID) -> StoredUser | None:
        row = self._session.get(UserRow, user_id)
        return None if row is None else _to_stored_user(row)

    def all(self) -> list[StoredUser]:
        rows = self._session.scalars(select(UserRow).order_by(UserRow.email)).all()
        return [_to_stored_user(row) for row in rows]

    def set_password(self, email: str, plain: str) -> None:
        row = self._require(email)
        row.password_hash = hash_password(plain)
        self._session.flush()

    def deactivate(self, email: str) -> None:
        row = self._require(email)
        row.is_active = False
        self._session.flush()

    def authenticate(self, email: str, plain: str) -> AuthenticatedUser | None:
        """None for every failure: unknown, wrong password, or deactivated.

        `verify_password` runs a real verification even when there is no row,
        so an unknown address does not answer faster than a known one.
        """
        row = self._row(email)
        hashed = row.password_hash if row is not None else None
        if not verify_password(plain, hashed):
            return None
        if row is None or not row.is_active:
            return None
        return AuthenticatedUser(id=row.id, email=row.email)

    def _row(self, email: str) -> UserRow | None:
        return self._session.scalars(
            select(UserRow).where(UserRow.email == _normalise_email(email))
        ).one_or_none()

    def _require(self, email: str) -> UserRow:
        row = self._row(email)
        if row is None:
            raise ValidationError(f"no user {_normalise_email(email)!r}")
        return row


def _to_stored_user(row: UserRow) -> StoredUser:
    return StoredUser(
        id=row.id,
        email=row.email,
        display_name=row.display_name,
        is_active=row.is_active,
        has_password=row.password_hash is not None,
    )
```

Add the imports this needs: `UserRow` from `.models`, `AuthenticatedUser` from
`offerdelta.application.scope`, and `hash_password` / `verify_password` from
`offerdelta.infrastructure.auth.passwords`.

- [ ] **Step 5: Run the tests**

Run: `cd backend && uv run pytest tests/integration/test_user_repository.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/src/offerdelta/application/scope.py backend/src/offerdelta/infrastructure/postgres/repositories.py backend/tests/integration/test_user_repository.py
git commit -m "Users: repository, and the TenantScope every other repository will take"
```

---

### Task 6: Repositories and use cases take a `TenantScope`

The task that actually creates the isolation. It touches many call sites at
once because a half-scoped repository layer is worse than either end state.

**Files:**
- Modify: `backend/src/offerdelta/infrastructure/postgres/repositories.py`
- Modify: `backend/src/offerdelta/application/transactions/enter_transaction.py`
- Modify: `backend/src/offerdelta/application/transactions/import_transactions.py`
- Modify: `backend/src/offerdelta/api/main.py`
- Modify: `backend/transactions.py`
- Modify: existing integration tests that construct repositories
- Create: `backend/tests/integration/test_tenant_isolation.py`

**Interfaces:**
- Consumes: `TenantScope`, `AuthenticatedUser` (Task 5)
- Produces:
  - `AccountRepository(scope: TenantScope)`, `ImportBatchRepository(scope: TenantScope)`, `TransactionRepository(scope: TenantScope)`
  - `enter_transaction(scope: TenantScope, entry: ManualEntry) -> EntryOutcome`
  - `import_csv(scope: TenantScope, request: ImportRequest) -> ImportOutcome`

- [ ] **Step 1: Write the failing isolation test**

Create `backend/tests/integration/test_tenant_isolation.py`:

```python
"""What this whole phase is for.

Two users, each with a valid identity. Nothing either does may be visible to
the other, and the same account key and the same transaction fingerprint must
be able to exist on both sides at once.
"""

from __future__ import annotations

import uuid
from datetime import date

from sqlalchemy.orm import Session

from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.application.transactions.enter_transaction import (
    ManualEntry,
    enter_transaction,
)
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    UserRepository,
)
from tests.integration.conftest import requires_database

pytestmark = requires_database

KEY = "chase-checking-5718"


def _scope(session: Session, email: str) -> TenantScope:
    stored = UserRepository(session).create(email, f"Owner of {email}")
    return TenantScope(
        session=session, user=AuthenticatedUser(id=stored.id, email=stored.email)
    )


def _entry(amount: str = "-12.34") -> ManualEntry:
    return ManualEntry(
        account_key=KEY,
        posted_on=date(2026, 3, 1),
        description="BLUE BOTTLE COFFEE",
        amount=Money.parse(amount),
        repeat=False,
    )


def test_one_tenant_cannot_read_another_tenants_account(session: Session) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")

    AccountRepository(a).register("Chase Checking 5718")

    assert AccountRepository(a).by_key(KEY) is not None
    assert AccountRepository(b).by_key(KEY) is None


def test_listing_accounts_shows_only_your_own(session: Session) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")

    AccountRepository(a).register("Chase Checking 5718")
    AccountRepository(b).register("Amex Gold 1006")

    assert [x.key for x in AccountRepository(a).all()] == [KEY]
    assert [x.key for x in AccountRepository(b).all()] == ["amex-gold-1006"]


def test_both_tenants_may_register_the_same_account_key(session: Session) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")

    AccountRepository(a).register("Chase Checking 5718")
    AccountRepository(b).register("Chase Checking 5718")

    # Bound to locals rather than chained: `by_key` returns an Optional, and
    # mypy runs over tests too.
    held_by_a = AccountRepository(a).by_key(KEY)
    held_by_b = AccountRepository(b).by_key(KEY)
    assert held_by_a is not None
    assert held_by_b is not None
    assert held_by_a.id != held_by_b.id


def test_entering_a_transaction_into_another_tenants_account_is_refused(
    session: Session,
) -> None:
    from offerdelta.domain.common.errors import ValidationError
    import pytest

    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    AccountRepository(a).register("Chase Checking 5718")

    with pytest.raises(ValidationError):
        enter_transaction(b, _entry())


def test_opening_a_batch_against_another_tenants_account_is_refused(
    session: Session,
) -> None:
    """The import path, guarded at the repository rather than by call site."""
    from offerdelta.domain.common.errors import ValidationError
    from offerdelta.infrastructure.postgres.repositories import ImportBatchRepository
    import pytest

    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    account_of_a = AccountRepository(a).register("Chase Checking 5718")

    with pytest.raises(ValidationError):
        ImportBatchRepository(b).open(
            account_of_a.id,
            source_file="statement.csv",
            source_sha256="0" * 64,
            mode="snapshot",
            window_start=date(2026, 3, 1),
            window_end=date(2026, 3, 31),
            row_count=1,
        )


def test_the_same_transaction_may_exist_in_both_tenants(session: Session) -> None:
    a = _scope(session, "a@example.test")
    b = _scope(session, "b@example.test")
    AccountRepository(a).register("Chase Checking 5718")
    AccountRepository(b).register("Chase Checking 5718")

    first = enter_transaction(a, _entry())
    second = enter_transaction(b, _entry())

    assert first.stored is True
    assert second.stored is True, "deduplication must be per tenant, not global"
    assert first.fingerprint == second.fingerprint
    assert first.transaction_id != second.transaction_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/integration/test_tenant_isolation.py -v`
Expected: FAIL — `AccountRepository.__init__` still takes a `Session`

- [ ] **Step 3: Change the repository constructors and scope every query**

In `repositories.py`:

```python
class AccountRepository:
    def __init__(self, scope: TenantScope) -> None:
        self._scope = scope
        self._session = scope.session

    def register(self, display_name: str, *, now: datetime | None = None) -> StoredAccount:
        key = canonical_account_key(display_name)
        if self.by_key(key) is not None:
            raise ValidationError(f"account {key!r} is already registered")
        row = AccountRow(
            id=uuid.uuid4(),
            user_id=self._scope.user.id,
            key=key,
            display_name=display_name.strip(),
            created_at=now or datetime.now(UTC),
        )
        ...

    def by_key(self, key: str) -> StoredAccount | None:
        row = self._session.scalars(
            select(AccountRow).where(
                AccountRow.user_id == self._scope.user.id,
                AccountRow.key == canonical_account_key(key),
            )
        ).one_or_none()
        return None if row is None else _to_stored_account(row)

    def all(self) -> list[StoredAccount]:
        rows = self._session.scalars(
            select(AccountRow)
            .where(AccountRow.user_id == self._scope.user.id)
            .order_by(AccountRow.key)
        ).all()
        return [_to_stored_account(row) for row in rows]
```

`ImportBatchRepository` and `TransactionRepository` take the scope the same
way. **Every query in them that filters on `account_id` gains a join to
`accounts` filtered by the scope's user:**

```python
    def _owned(self, account_id: uuid.UUID) -> bool:
        """An account id is only usable if it belongs to this tenant.

        The alternative is chain of custody - the id came from a scoped
        AccountRepository, so it must be safe. That is true today and is an
        argument about call sites; it stops being true the first time a route
        accepts an account id from a request body.
        """
        return (
            self._session.scalars(
                select(AccountRow.id).where(
                    AccountRow.id == account_id,
                    AccountRow.user_id == self._scope.user.id,
                )
            ).one_or_none()
            is not None
        )
```

Call `_owned` at the top of `ImportBatchRepository.open` and
`TransactionRepository.add_many`, raising
`ValidationError(f"no account {account_id}")` when it returns False. Apply the
same `AccountRow.user_id` filter to the `select` statements inside `add_many`
that look for existing fingerprints and external ids, by joining through
`AccountRow`.

- [ ] **Step 4: Change the use case signatures**

`enter_transaction.py`:

```python
def enter_transaction(scope: TenantScope, entry: ManualEntry) -> EntryOutcome:
    """Write one hand-entered transaction, or report that it is already there."""
    session = scope.session
    ...
    accounts = AccountRepository(scope)
```

Everywhere the body used `session`, it now uses `scope.session` through that
local. `TransactionRepository(session)` becomes `TransactionRepository(scope)`.

`import_transactions.py`: the same change to `import_csv(scope, request)`, with
`AccountRepository(scope)`, `ImportBatchRepository(scope)`, and
`TransactionRepository(scope)`.

- [ ] **Step 5: Update the two call sites so the suite can run**

In `api/main.py`, the `create_transaction` handler cannot work without an
identity. Until Task 8 gives it one, make it fail loudly rather than silently
untenanted — temporarily raise `HTTPException(501)` and mark its existing test
`xfail(reason="awaiting the auth dependency in Task 8")`.

In `transactions.py`, add a module-level helper and use it in `_add` and
`_accounts`:

```python
def _scope_for(session: Session, email: str) -> TenantScope:
    stored = UserRepository(session).by_email(email)
    if stored is None:
        raise ValidationError(f"no user {email!r}. Create one: users.py create --email ...")
    return TenantScope(
        session=session, user=AuthenticatedUser(id=stored.id, email=stored.email)
    )
```

Task 7 adds the `--user` argument that supplies `email`. For this step, read it
from `args.user` and add the argument to the `add`, `accounts`, and `commit`
subparsers as `required=True`.

- [ ] **Step 6: Update existing repository tests**

`tests/integration/test_transaction_repository.py`,
`test_import_transactions_service.py`, and `test_enter_transaction.py`
construct repositories and call use cases with a `Session`. Give each a scope
built the way `test_tenant_isolation.py` does. Add a shared fixture to
`tests/integration/conftest.py`:

```python
@pytest.fixture
def scope(session: Session) -> TenantScope:
    """A tenant to run against, so tests never construct an untenanted query."""
    from offerdelta.application.scope import AuthenticatedUser, TenantScope
    from offerdelta.infrastructure.postgres.repositories import UserRepository

    stored = UserRepository(session).create("fixture@example.test", "Fixture Owner")
    return TenantScope(
        session=session, user=AuthenticatedUser(id=stored.id, email=stored.email)
    )
```

- [ ] **Step 7: Run the whole suite**

Run: `cd backend && uv run pytest -q`
Expected: PASS, with the one `xfail` from Step 5.

- [ ] **Step 8: Run types and architecture**

Run: `cd backend && uv run mypy && uv run lint-imports`
Expected: `Success` and `Contracts: 4 kept, 0 broken.`

- [ ] **Step 9: Commit**

```bash
git add backend/src backend/transactions.py backend/tests
git commit -m "Tenancy: repositories take a scope, so an untenanted query cannot be written"
```

---

### Task 7: User lifecycle on the CLI

**Files:**
- Create: `backend/users.py`
- Modify: `backend/Makefile` (add `users.py` to `PY_FILES`)
- Modify: `backend/pyproject.toml` (add `users.py` to mypy `files`)
- Modify: `.github/workflows/ci.yml` — add `users.py` to both ruff invocations
- Create: `backend/tests/unit/test_users_cli.py`

**Interfaces:**
- Consumes: `UserRepository` (Task 5)
- Produces: `users.py` with subcommands `create`, `set-password`, `list`, `deactivate`; `build_parser() -> argparse.ArgumentParser`; `main(argv: list[str]) -> int`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/unit/test_users_cli.py`, modelled on the existing
`tests/unit/test_transactions_cli.py`:

```python
"""The CLI is the only way a user is created. There is no registration route."""

from __future__ import annotations

import pytest

from users import build_parser


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/unit/test_users_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'users'`

- [ ] **Step 3: Write `users.py`**

Mirror `transactions.py`: `build_parser()`, a `main(argv)` that dispatches, and
the same error handling that never echoes a DSN.

```python
"""User lifecycle.

There is no registration endpoint, so this is the only way an account comes
into existence. That is the point: an invite-only system whose invitations are
issued by a person with database access has no signup surface to attack.

`set-password` takes the password from OFFERDELTA_NEW_PASSWORD rather than
argv, because an argument lands in shell history and in `ps` output.
"""
```

`set-password` reads `os.environ["OFFERDELTA_NEW_PASSWORD"]` and exits 2 with a
clear message when it is unset. `list` prints email, display name, active, and
whether a password is set — never the hash.

- [ ] **Step 4: Run the tests**

Run: `cd backend && uv run pytest tests/unit/test_users_cli.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Add `users.py` to every file list**

`Makefile` `PY_FILES`, mypy `files` in `pyproject.toml`, and **both** ruff
commands in `.github/workflows/ci.yml`. The Makefile and the workflow must
still name the same set — that is what Task 0 of this repository's history
established and what a green `make check` now depends on.

- [ ] **Step 6: Claim the placeholder user on the local database**

```bash
cd backend && OFFERDELTA_NEW_PASSWORD='<a real password>' \
  uv run python users.py set-password --email placeholder@localhost.invalid
```

Then rename it to a real address by creating the intended user and moving the
accounts, or simply keep the placeholder address locally — it never leaves the
machine. Record which you did.

- [ ] **Step 7: Run the gate and commit**

Run: `cd backend && uv run ruff format $(PY_FILES) && uv run ruff check $(PY_FILES) && uv run mypy`

```bash
git add backend/users.py backend/tests/unit/test_users_cli.py backend/Makefile backend/pyproject.toml .github/workflows/ci.yml
git commit -m "CLI: user lifecycle, the only way an account is created"
```

---

### Task 8: The auth endpoint, the `_scope` dependency, and rate limiting

**Files:**
- Create: `backend/src/offerdelta/api/rate_limit.py`
- Modify: `backend/src/offerdelta/api/main.py`
- Modify: `backend/src/offerdelta/api/schemas.py`
- Create: `backend/tests/integration/test_auth_api.py`
- Create: `backend/tests/unit/test_rate_limit.py`

**Interfaces:**
- Consumes: `UserRepository.authenticate`, `issue_token`, `decode_token`, `TenantScope`
- Produces: `POST /v1/auth/token`; `_scope()` FastAPI dependency yielding `TenantScope`; `FixedWindowLimiter(max_attempts: int, window: timedelta)` with `check(key: str, now: datetime | None = None) -> bool`

- [ ] **Step 1: Write the failing rate-limit test**

Create `backend/tests/unit/test_rate_limit.py`:

```python
"""Five failures per address per fifteen minutes.

In-process and fixed-window: it does not survive a restart and means nothing
across instances. On one free-tier instance it works, and saying so here is
better than implying a guarantee that is not there.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from offerdelta.api.rate_limit import FixedWindowLimiter

NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)


def test_the_first_five_attempts_are_allowed() -> None:
    limiter = FixedWindowLimiter(max_attempts=5, window=timedelta(minutes=15))
    assert [limiter.check("a@example.test", NOW) for _ in range(5)] == [True] * 5


def test_the_sixth_attempt_is_refused() -> None:
    limiter = FixedWindowLimiter(max_attempts=5, window=timedelta(minutes=15))
    for _ in range(5):
        limiter.check("a@example.test", NOW)
    assert limiter.check("a@example.test", NOW) is False


def test_the_window_reopens() -> None:
    limiter = FixedWindowLimiter(max_attempts=5, window=timedelta(minutes=15))
    for _ in range(5):
        limiter.check("a@example.test", NOW)
    later = NOW + timedelta(minutes=15, seconds=1)
    assert limiter.check("a@example.test", later) is True


def test_addresses_are_counted_separately() -> None:
    limiter = FixedWindowLimiter(max_attempts=5, window=timedelta(minutes=15))
    for _ in range(5):
        limiter.check("a@example.test", NOW)
    assert limiter.check("b@example.test", NOW) is True
```

- [ ] **Step 2: Run it, watch it fail, implement `FixedWindowLimiter`, run it again**

Run: `cd backend && uv run pytest tests/unit/test_rate_limit.py -v`
Expected first: FAIL (`ModuleNotFoundError`). Expected after: PASS (4 tests).

- [ ] **Step 3: Write the failing API test**

Create `backend/tests/integration/test_auth_api.py`:

```python
"""The boundary as a caller sees it."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.integration.conftest import requires_database

pytestmark = requires_database


def test_a_protected_route_without_a_token_is_401(client: TestClient) -> None:
    response = client.post("/v1/transactions", json={})
    assert response.status_code == 401


def test_a_protected_route_with_a_malformed_token_is_401(client: TestClient) -> None:
    response = client.post(
        "/v1/transactions", json={}, headers={"Authorization": "Bearer not.a.token"}
    )
    assert response.status_code == 401


def test_unknown_email_and_wrong_password_are_indistinguishable(
    client: TestClient, existing_user_email: str
) -> None:
    unknown = client.post(
        "/v1/auth/token",
        json={"email": "nobody@example.test", "password": "whatever"},
    )
    wrong = client.post(
        "/v1/auth/token",
        json={"email": existing_user_email, "password": "not the password"},
    )
    assert unknown.status_code == wrong.status_code == 401
    assert unknown.json() == wrong.json()


def test_a_good_password_returns_a_token(
    client: TestClient, existing_user_email: str, existing_user_password: str
) -> None:
    response = client.post(
        "/v1/auth/token",
        json={"email": existing_user_email, "password": existing_user_password},
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_a_deactivated_user_is_401(client: TestClient, deactivated_user_email: str) -> None:
    response = client.post(
        "/v1/auth/token",
        json={"email": deactivated_user_email, "password": "a good long password"},
    )
    assert response.status_code == 401


def test_the_sixth_failure_is_429(client: TestClient, existing_user_email: str) -> None:
    for _ in range(5):
        client.post(
            "/v1/auth/token",
            json={"email": existing_user_email, "password": "wrong"},
        )
    response = client.post(
        "/v1/auth/token", json={"email": existing_user_email, "password": "wrong"}
    )
    assert response.status_code == 429


def test_one_tenant_gets_404_naming_another_tenants_account(
    client: TestClient, token_a: str, account_key_of_b: str
) -> None:
    """404, not 403: a 403 would confirm the account exists."""
    response = client.post(
        "/v1/transactions",
        headers={"Authorization": f"Bearer {token_a}"},
        json={
            "account_key": account_key_of_b,
            "posted_on": "2026-03-01",
            "description": "BLUE BOTTLE COFFEE",
            "amount": "-12.34",
        },
    )
    assert response.status_code == 404
```

Add these fixtures to `tests/integration/conftest.py`. Overriding `_session` is
enough: `_scope` depends on it, so the whole chain runs against the rolled-back
test session.

```python
PASSWORD = "a good long password for tests"


@pytest.fixture
def client(session: Session) -> Iterator[TestClient]:
    """The app, wired to the transaction this test will roll back."""
    from offerdelta.api.main import _session as session_dependency
    from offerdelta.api.main import app

    app.dependency_overrides[session_dependency] = lambda: session
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def existing_user_password() -> str:
    return PASSWORD


@pytest.fixture
def existing_user_email(session: Session) -> str:
    from offerdelta.infrastructure.postgres.repositories import UserRepository

    repo = UserRepository(session)
    repo.create("a@example.test", "Person A")
    repo.set_password("a@example.test", PASSWORD)
    return "a@example.test"


@pytest.fixture
def deactivated_user_email(session: Session) -> str:
    from offerdelta.infrastructure.postgres.repositories import UserRepository

    repo = UserRepository(session)
    repo.create("gone@example.test", "Departed")
    repo.set_password("gone@example.test", PASSWORD)
    repo.deactivate("gone@example.test")
    return "gone@example.test"


@pytest.fixture
def token_a(client: TestClient, existing_user_email: str) -> str:
    response = client.post(
        "/v1/auth/token", json={"email": existing_user_email, "password": PASSWORD}
    )
    assert response.status_code == 200, response.text
    token: str = response.json()["access_token"]
    return token


@pytest.fixture
def account_key_of_b(session: Session) -> str:
    """An account belonging to somebody who is not A."""
    from offerdelta.application.scope import AuthenticatedUser, TenantScope
    from offerdelta.infrastructure.postgres.repositories import (
        AccountRepository,
        UserRepository,
    )

    stored = UserRepository(session).create("b@example.test", "Person B")
    scope = TenantScope(
        session=session, user=AuthenticatedUser(id=stored.id, email=stored.email)
    )
    return AccountRepository(scope).register("Chase Checking 5718").key
```

`TestClient` and `Iterator` need importing at the top of the conftest;
`tests/contract/test_api_money_serialisation.py` already uses `TestClient(app)`
and is the pattern to follow.

Change the deactivated-user test to use the shared constant rather than
repeating the literal:

```python
def test_a_deactivated_user_is_401(
    client: TestClient, deactivated_user_email: str, existing_user_password: str
) -> None:
    response = client.post(
        "/v1/auth/token",
        json={"email": deactivated_user_email, "password": existing_user_password},
    )
    assert response.status_code == 401
```

- [ ] **Step 4: Implement the endpoint and the dependency**

In `api/main.py`, add the schemas and:

```python
_login_limiter = FixedWindowLimiter(max_attempts=5, window=timedelta(minutes=15))


@app.post("/v1/auth/token", include_in_schema=_AUTH_CONFIGURED, response_model=TokenSchema)
def issue_access_token(
    body: LoginSchema, session: Annotated[Session, Depends(_session)]
) -> TokenSchema:
    """One endpoint, one answer shape.

    Unknown address and wrong password return the same status and the same
    body, so this cannot be used to learn who has an account.
    """
    if not _login_limiter.check(body.email.lower()):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many attempts")

    who = UserRepository(session).authenticate(body.email, body.password)
    if who is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")

    secret = get_settings().jwt_secret
    if secret is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "authentication is not configured")
    return TokenSchema(access_token=issue_token(who.id, secret=secret))


def _scope(
    session: Annotated[Session, Depends(_session)],
    credentials: Annotated[str | None, Header(alias="Authorization")] = None,
) -> TenantScope:
    """Identity first, then data. Both, or neither."""
    secret = get_settings().jwt_secret
    if secret is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "authentication is not configured")
    if credentials is None or not credentials.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")

    user_id = decode_token(credentials.removeprefix("Bearer "), secret=secret)
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")

    # Loaded every request, so deactivation takes effect now rather than when
    # the token happens to expire.
    stored = UserRepository(session).by_id(user_id)
    if stored is None or not stored.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")

    return TenantScope(
        session=session, user=AuthenticatedUser(id=stored.id, email=stored.email)
    )
```

Change `create_transaction` to depend on `_scope` instead of `_session`, call
`enter_transaction(scope, entry)`, and remove the temporary 501 and the `xfail`
from Task 6.

- [ ] **Step 5: Run the tests**

Run: `cd backend && uv run pytest tests/integration/test_auth_api.py tests/unit/test_rate_limit.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/src/offerdelta/api backend/tests
git commit -m "API: bearer identity on every data route, and one login endpoint"
```

---

### Task 9: The contract, the log test, and the stale comment

**Files:**
- Modify: `backend/pyproject.toml` (import-linter)
- Create: `backend/tests/integration/test_no_pii_in_logs.py`
- Modify: `backend/tests/integration/conftest.py` (docstring)

**Interfaces:**
- Consumes: everything above
- Produces: a fifth import-linter contract

- [ ] **Step 1: Add the contract**

In `pyproject.toml`:

```toml
[[tool.importlinter.contracts]]
# Routes reach data through use cases. A route that could construct a
# repository could construct one without a scope, and the scope is the only
# thing keeping one tenant out of another's rows.
name = "API does not reach repositories directly"
type = "forbidden"
source_modules = ["offerdelta.api"]
forbidden_modules = ["offerdelta.infrastructure.postgres.repositories"]
```

- [ ] **Step 2: Run it and watch it fail**

Run: `cd backend && uv run lint-imports`
Expected: BROKEN — Task 8 imported `UserRepository` into `api/main.py`.

- [ ] **Step 3: Move the two uses behind the application layer**

Create `offerdelta/application/auth.py` with `authenticate(session, email, password) -> AuthenticatedUser | None` and `load_active_user(session, user_id) -> AuthenticatedUser | None`, both wrapping `UserRepository`. The API imports those.

- [ ] **Step 4: Run it again**

Run: `cd backend && uv run lint-imports`
Expected: `Contracts: 5 kept, 0 broken.`

- [ ] **Step 5: Write the PII log test**

Create `backend/tests/integration/test_no_pii_in_logs.py`:

```python
"""No transaction text reaches a log.

`import_batches` stores a file name and a checksum, never the bytes, so this
is very nearly true already - but it is true by accident, and an accident is
not a guarantee. This is the test that makes it a rule.
"""

from __future__ import annotations

import logging
from datetime import date

import pytest
from sqlalchemy.orm import Session

from offerdelta.application.scope import TenantScope
from offerdelta.application.transactions.enter_transaction import (
    ManualEntry,
    enter_transaction,
)
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.repositories import AccountRepository
from tests.integration.conftest import requires_database

pytestmark = requires_database

SECRET_MERCHANT = "ZZQQ UNIQUE MERCHANT STRING"
SECRET_AMOUNT = "-1234.56"


def test_no_description_or_amount_reaches_the_logs(
    scope: TenantScope, caplog: pytest.LogCaptureFixture
) -> None:
    AccountRepository(scope).register("Chase Checking 5718")

    with caplog.at_level(logging.DEBUG):
        enter_transaction(
            scope,
            ManualEntry(
                account_key="chase-checking-5718",
                posted_on=date(2026, 3, 1),
                description=SECRET_MERCHANT,
                amount=Money.parse(SECRET_AMOUNT),
                repeat=False,
            ),
        )

    captured = caplog.text
    assert SECRET_MERCHANT not in captured
    assert "1234.56" not in captured
```

- [ ] **Step 6: Fix the stale comment**

In `tests/integration/conftest.py`, the module docstring says CI has no
database. The workflow provisions PostgreSQL and sets `CONNECTION_STRING` at
job level, so it does. Replace that sentence with:

```
The whole module skips when CONNECTION_STRING is unset, which is what makes a
local checkout without a database still able to run the suite. CI does set it,
against a service container, so these tests run there.
```

- [ ] **Step 7: Run the full gate**

Run: `cd backend && uv run ruff format --check $(PY_FILES) && uv run ruff check $(PY_FILES) && uv run mypy && uv run lint-imports && uv run pytest -q`
Expected: all green, 5 contracts kept.

- [ ] **Step 8: Commit**

```bash
git add backend/pyproject.toml backend/src/offerdelta/application/auth.py backend/src/offerdelta/api backend/tests
git commit -m "Contract: routes cannot reach a repository, and PII cannot reach a log"
```

---

### Task 10: Deploy with two synthetic tenants

**Files:**
- Modify: `render.yaml`
- Create: `backend/seed_demo.py`
- Modify: `backend/Makefile`, `backend/pyproject.toml`, `.github/workflows/ci.yml` (add `seed_demo.py` to the file lists)
- Create: `backend/tests/unit/test_seed_demo.py`

**Interfaces:**
- Consumes: `UserRepository`, `AccountRepository`, `enter_transaction`
- Produces: `seed_demo.py` with `main(argv: list[str]) -> int`; `SYNTHETIC_TENANTS: Final[tuple[str, str]]`

- [ ] **Step 1: Declare the secrets in the blueprint**

In `render.yaml`, under the service's `envVars`:

```yaml
    envVars:
      - key: CONNECTION_STRING
        sync: false
      - key: JWT_SECRET
        sync: false
```

`sync: false` states the requirement without carrying the value.

- [ ] **Step 2: Write the failing seed test**

Create `backend/tests/unit/test_seed_demo.py`:

```python
"""Two tenants, not one.

With a single tenant the deployment is single-tenant in practice, and nothing
about isolation is exercised by its existence.
"""

from __future__ import annotations

from seed_demo import SYNTHETIC_TENANTS


def test_there_are_two_tenants() -> None:
    assert len(SYNTHETIC_TENANTS) == 2


def test_the_addresses_are_obviously_not_real() -> None:
    assert all(email.endswith("@example.test") for email in SYNTHETIC_TENANTS)
```

- [ ] **Step 3: Write `seed_demo.py`**

It creates two users with `@example.test` addresses, sets their passwords from
`OFFERDELTA_DEMO_PASSWORD` and `OFFERDELTA_OTHER_PASSWORD`, registers one
obviously-synthetic account each (`demo-checking-0001`, `other-checking-0002`),
and enters a handful of invented transactions per tenant. It refuses to run
when either password variable is unset, and it is idempotent: running it twice
creates nothing the second time.

- [ ] **Step 4: Run the tests, add the file to the lists, run the gate**

Run: `cd backend && uv run pytest tests/unit/test_seed_demo.py -v`
Then add `seed_demo.py` to `PY_FILES`, mypy `files`, and both ruff commands in
the workflow, and run the full gate.

- [ ] **Step 5: Provision and deploy**

Attach a PostgreSQL database to the Render service, set `CONNECTION_STRING` and
a freshly generated `JWT_SECRET` in the dashboard, and deploy. Then:

```bash
# against the deployed database, from the local machine
cd backend && CONNECTION_STRING='<render database url>' uv run alembic upgrade head
cd backend && CONNECTION_STRING='<render database url>' \
  OFFERDELTA_DEMO_PASSWORD='<...>' OFFERDELTA_OTHER_PASSWORD='<...>' \
  uv run python seed_demo.py
```

- [ ] **Step 6: Verify the deployed boundary**

```bash
curl -s -o /dev/null -w '%{http_code}\n' https://offerdelta.onrender.com/v1/health/ready
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://offerdelta.onrender.com/v1/transactions -d '{}'
curl -s https://offerdelta.onrender.com/v1/auth/token \
  -H 'content-type: application/json' \
  -d '{"email":"demo@example.test","password":"<...>"}'
curl -s https://offerdelta.onrender.com/openapi.json | grep -c '"/v1/auth/token"'
```

Expected: `200`; `401`; a token; `1`. Confirm no real merchant string appears
anywhere in the deployed database:

```bash
CONNECTION_STRING='<render database url>' uv run python -c "
from sqlalchemy import text
from offerdelta.infrastructure.postgres.engine import get_engine
with get_engine().connect() as c:
    print(c.execute(text('select count(*) from users')).scalar_one(), 'users')
    print(c.execute(text('select count(*) from transactions')).scalar_one(), 'transactions')
    print([r[0] for r in c.execute(text('select distinct key from accounts'))])
"
```
Expected: 2 users, a small transaction count, and only the two synthetic
account keys.

- [ ] **Step 7: Commit**

```bash
git add render.yaml backend/seed_demo.py backend/tests/unit/test_seed_demo.py backend/Makefile backend/pyproject.toml .github/workflows/ci.yml
git commit -m "Deploy: authenticated service, two synthetic tenants, no real data"
```

---

## Done when

- `make check` is green: 5 import-linter contracts, strict mypy, the full suite.
- CI is green on `main`.
- The cross-tenant table in §7 of the spec is covered by passing tests.
- `https://offerdelta.onrender.com/v1/auth/token` issues a token for a
  synthetic tenant, and every data route without one answers 401.
- The deployed database holds two synthetic tenants and no real transaction.
- The local database still holds 742 real transactions, now owned by a user
  with a password.
