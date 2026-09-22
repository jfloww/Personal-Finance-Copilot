"""Database fixtures.

Every test runs inside a transaction that is rolled back afterwards. The suite
only accepts an explicitly configured TEST_DATABASE_URL; it never inherits the
application's CONNECTION_STRING or backend/.env credential.

The whole module skips when TEST_DATABASE_URL is unset, which is what makes a
local checkout without a database still able to run the suite. CI does set it,
against a disposable service container, and sets JWT_SECRET too, so these tests
run there.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import Connection, Engine, text
from sqlalchemy.orm import Session

from offerdelta.api.main import _session as _session_dependency
from offerdelta.api.main import app
from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.config import get_settings
from offerdelta.infrastructure.postgres.engine import get_engine
from offerdelta.infrastructure.postgres.repositories import AccountRepository, UserRepository

#: Every prerequisite `_require` found missing while `CI` was set. Collected
#: rather than failed on the spot: raising mid-import here would abort while
#: pytest is still loading conftests, which surfaces as a raw traceback
#: instead of a normal red build. `pytest_collection_finish` below turns this
#: into one clean, unmissable failure once collection has actually finished.
_missing_in_ci: list[str] = []


def _require(available: bool, reason: str) -> pytest.MarkDecorator:
    """Skip on a checkout that lacks the prerequisite; fail outright in CI.

    A `skipif` alone is how this branch already lost coverage once: drop
    `TEST_DATABASE_URL` or `JWT_SECRET` from `ci.yml` and every test guarded
    by it quietly disappears, and a build with fewer tests than yesterday
    still goes green. `CI` is the variable every CI runner sets and a
    developer's shell does not, so the same missing prerequisite that skips
    locally - no database, no signing key, nothing to test against - is
    promoted to a hard failure there instead, where "fewer tests" must read
    as red.
    """
    if not available and os.environ.get("CI"):
        _missing_in_ci.append(reason)
    return pytest.mark.skipif(not available, reason=reason)


def pytest_collection_finish(session: pytest.Session) -> None:  # noqa: ARG001
    """Turn a missing CI prerequisite into a session-ending failure.

    Runs once collection is done, so every other file still collects
    normally; this only stops the run from proceeding to a suite that
    silently dropped the isolation table or the auth tests. `session` is
    unused here but required: pytest matches hook implementations against
    `pytest_collection_finish`'s fixed signature by parameter name.
    """
    if _missing_in_ci:
        pytest.exit("; ".join(_missing_in_ci), returncode=1)


requires_database = _require(
    bool(os.environ.get("TEST_DATABASE_URL")),
    reason="TEST_DATABASE_URL is not set; database tests need a disposable PostgreSQL",
)

#: The auth tests mint and verify real tokens, so they need a real signing
#: key. CI sets a throwaway `JWT_SECRET` - an arbitrary self-issued string
#: with no external validity, unlike the Anthropic key - specifically so this
#: mark is a no-op there. It still guards a local checkout without one: a
#: suite that needs a secret to go green teaches people to ignore red builds,
#: so absence skips locally and fails in CI - see `_require`.
requires_auth = _require(
    get_settings().auth_available,
    reason="JWT_SECRET is not set; auth tests need a signing key",
)

#: Every fixture below that creates a user needs an address. A constant one
#: was tried first and rejected: this suite runs against a live shared
#: database, so a fixed address collides with any row a previous interrupted
#: run left behind, and - separately - the login endpoint's rate limiter is
#: an in-process singleton that outlives any one test's rolled-back
#: transaction, so a fixed address lets one test's failed logins lock out a
#: later test that reuses it. A unique address per call makes both
#: structurally impossible rather than relying on test order.
PASSWORD = "a good long password for tests"


def _unique_email(label: str) -> str:
    return f"{label}-{uuid.uuid4().hex[:12]}@example.test"


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
def scope(session: Session) -> TenantScope:
    """A tenant to run against, so no test can construct an untenanted query.

    Every repository takes one of these now, so a test that wants a
    repository has to say whose data it is looking at - the same obligation
    the application has.

    The address is unique per test so interrupted or concurrently running test
    sessions cannot collide even on a shared test service.
    """
    address = f"fixture-{uuid.uuid4().hex[:12]}@example.test"
    stored = UserRepository(session).create(address, "Fixture Owner")
    return TenantScope(session=session, user=AuthenticatedUser(id=stored.id, email=stored.email))


@pytest.fixture
def other_scope(session: Session) -> TenantScope:
    """A second tenant, built the same way as `scope`, to prove isolation.

    Distinct from `scope` on the same session and the same rolled-back
    transaction, so a test can prove one tenant never sees a row or a month
    that belongs to the other.
    """
    address = f"fixture-{uuid.uuid4().hex[:12]}@example.test"
    stored = UserRepository(session).create(address, "Other Fixture Owner")
    return TenantScope(session=session, user=AuthenticatedUser(id=stored.id, email=stored.email))


@pytest.fixture
def client(session: Session) -> Iterator[TestClient]:
    """The app, wired to the transaction this test will roll back.

    Only `_session` is overridden: `_scope` depends on it, so the whole
    identity chain - token decode, the user reload, the tenant it builds -
    runs for real against this rolled-back session rather than being stubbed
    out here too.
    """
    app.dependency_overrides[_session_dependency] = lambda: session
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def existing_user_password() -> str:
    return PASSWORD


@pytest.fixture
def existing_user_email(session: Session) -> str:
    address = _unique_email("a")
    repo = UserRepository(session)
    repo.create(address, "Person A")
    repo.set_password(address, PASSWORD)
    return address


@pytest.fixture
def deactivated_user_email(session: Session) -> str:
    address = _unique_email("gone")
    repo = UserRepository(session)
    repo.create(address, "Departed")
    repo.set_password(address, PASSWORD)
    repo.deactivate(address)
    return address


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
    stored = UserRepository(session).create(_unique_email("b"), "Person B")
    scope = TenantScope(session=session, user=AuthenticatedUser(id=stored.id, email=stored.email))
    return AccountRepository(scope).register("Chase Checking 5718").key


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
