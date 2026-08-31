# Authentication and tenant isolation — design

**Date:** 2026-08-31
**Status:** approved for planning
**Scope:** Phase 0. The deployment boundary that everything after it depends on.
Changes no domain calculation and no evaluation artifact.

---

## 1. What this is

Today the public deployment is safe because it has no database. Real financial
ingestion is not deployed, so no unauthenticated request can reach a
transaction. That property is real, and it was the right call, but it was
achieved by leaving the feature out.

The next phase is a monthly spending report over imported transactions. That
report *is* the transaction path. The property can no longer come from absence,
so it has to come from a boundary: identity on every request, a tenant on every
query, and a test that proves one user cannot read another's rows.

This document specifies that boundary. It is deliberately the whole phase. No
report, no analysis, no agent — those are Phase 1 and later, and they are
easier to build correctly on top of an identity that already exists than to
retrofit around one that does not.

### The claim being made

**One user cannot read or write another user's financial data, and that is
enforced by code the CI runs rather than by care.**

Section 7 is how the claim becomes falsifiable.

---

## 2. Non-goals

- **No self-serve registration.** Accounts are created by CLI. No email
  verification, no password reset, no invitation tokens. These are commodity
  flows that prove nothing and expand the attack surface of a personal project.
- **No role or permission system.** Every authenticated user has exactly the
  same rights over exactly their own data. Read-only demo access is achieved by
  what is exposed, not by a flag (§6).
- **No refresh tokens.** See §8.
- **No PostgreSQL row-level security.** See §8.
- **No file upload endpoint.** CSV import stays a local CLI operation in this
  phase.
- **No change to the evaluation artifacts.** The labelled dataset lives in CSV
  under a git-denied path and is keyed by a derived `transaction_id`, not by a
  database row. Nothing in this phase touches it.
- **No change to the demo routes.** `/demo/*`, `/v1/demo/*`, `/v1/comparisons`,
  health, version, and the evaluation JSON compute from in-memory profiles or
  static files. They stay unauthenticated and untenanted because they hold
  nobody's data.

---

## 3. Schema and migration

### Tenancy lives on `accounts`, and only there

```
users                    (new)
  id, email UNIQUE, password_hash NULL-able, display_name,
  is_active, created_at

accounts
  + user_id FK NOT NULL
  ~ key UNIQUE  ->  UNIQUE (user_id, key)

import_batches, transactions
  unchanged - tenancy inherited through account_id
```

`import_batches` and `transactions` already constrain on `account_id`
(`uq_import_batches_account_checksum`, `uq_transactions_account_fingerprint_occurrence`,
`uq_transactions_account_external_id`). Making the account the tenant root means
those constraints keep working unchanged and every row reaches its owner by
foreign key.

Denormalising `user_id` onto `transactions` would remove a join. It would also
create a second copy of the ownership fact, and two copies can disagree. A
transaction whose `user_id` disagrees with its account's owner does not fail
loudly — it produces a report that is quietly wrong for two people at once.
One owner column cannot contradict itself.

### The one destructive change

`accounts.key` is globally unique today. Two users cannot both register
`chase-checking-5718`, which is not a hypothetical: the natural key for a bank
account is the same shape for everybody. It becomes `UNIQUE (user_id, key)`.

This is the only structural change in the migration and the only part that is
awkward to reverse.

### `comparison_runs` is left alone

`ComparisonRunRepository` has integration tests and no production caller. The
table has no writer and no rows. Adding a `NOT NULL` foreign key to it now
would be a guess about a feature that does not exist. It gets a tenant when it
gets a writer.

### Backfilling the existing rows

Standard three steps: add nullable, backfill, set `NOT NULL`.

The backfill creates a placeholder user with a fixed UUID **only when orphan
accounts exist**, and assigns them to it. On an empty database — CI, and Render
on first deploy — no user row is created at all.

`password_hash IS NULL` means *this user cannot log in*. The migration therefore
never invents a credential. A real password is set afterwards, out of band:

```
users set-password --email <address>
```

The local database has seven accounts and 742 transactions spanning
2026-01-02 to 2026-08-21. They all belong to one person, so one placeholder is
the whole backfill.

---

## 4. Authentication

### Credentials

argon2id, via `argon2-cffi`. Chosen over bcrypt because it does not silently
truncate at 72 bytes and because it is memory-hard.

### Token

JWT, HS256, signed with a new `JWT_SECRET` setting. Claims are `sub` (user id),
`iat`, and `exp`. Nothing else: there are no roles to encode, and a claim that
exists is a claim that will eventually be trusted.

Access token only, one hour. There is no refresh token (§8).

### The user is loaded on every request

Stateless verification would mean `is_active = false` has no effect until the
token expires. An authenticated request already opens a database session to do
its work, so reading one more row costs effectively nothing, and buys two
things: deactivation takes effect immediately, and the scope carries a real
user rather than a bare UUID decoded from a string the client supplied.

### One endpoint

```
POST /v1/auth/token      email + password -> access token
```

There is no registration route. User lifecycle is CLI:

```
users create --email --display-name     # created without a password; cannot log in
users set-password --email
users list
users deactivate --email
```

### Enumeration and abuse

An unknown email still runs a dummy verification so the failing paths do the
same work, and unknown-email and wrong-password return the same status and the
same body. A caller cannot use the login endpoint to learn who has an account.

Login failures are rate limited by an in-process fixed-window counter: **five
failures per email address per fifteen minutes**, after which the endpoint
returns 429 until the window closes. A success does not reset the window early;
the counter is on the address, not the session.

**This counter does not survive a restart and is meaningless across multiple
instances.** On a single free-tier instance it works; the limitation is written
here rather than implied away, and it is the first thing to revisit if the
service is ever scaled out.

### Absent secret disables authentication

Following the existing `database_available` / `llm_available` pattern,
`auth_available` is false when `JWT_SECRET` is unset. The auth route disappears
from the schema and protected routes return 503. CI stays green without holding
a secret, which is the same discipline the LLM client already follows.

---

## 5. The TenantScope seam

```python
# offerdelta/application/scope.py
@dataclass(frozen=True)
class AuthenticatedUser:
    id: uuid.UUID
    email: str

@dataclass(frozen=True)
class TenantScope:
    session: Session
    user: AuthenticatedUser
```

Both types live in `offerdelta/application/scope.py`. `AuthenticatedUser` is
the shape after the checks have already passed, which is why it carries no
`is_active`: an inactive user never becomes one. A `TenantScope` in hand means
identity was established, not that it still needs verifying.

Repositories take a `TenantScope` where they take a `Session` today. Use cases
take one too: `enter_transaction(scope, entry)`,
`import_transactions(scope, ...)`.

### Every query scopes explicitly

The cheaper alternative is chain of custody: an `account_id` always came from a
tenant-scoped `AccountRepository`, so anything hanging off it is already safe.
That is true today. It is also an argument about call sites, and it stops being
true the first time a route accepts an account id from a request body.

The application layer is the only layer enforcing tenancy here, so the property
must not rest on an argument. Transaction and batch queries join to `accounts`
filtered by the scope's user. The cost is one join.

### Both entry points, one rule

```
API   Depends(_scope)  ->  bearer token -> decode -> load user
                           -> 401 if missing, invalid, expired, or inactive
                           -> 503 if no database
                           -> Session -> TenantScope

CLI   --user <email>   ->  load user -> Session -> TenantScope
```

`--user` is **required** on every command that touches tenant data. There is no
default and no ambient current-user. The question "which tenant did I just
import 700 rows into" should not be answerable only by looking afterwards.

### A contract, not a convention

A new import-linter contract forbids `offerdelta.api` from importing
`offerdelta.infrastructure.postgres.repositories`. Routes reach data through
use cases; a route cannot construct a repository, and therefore cannot
construct one without a scope. This uses the tool the repository already runs
in CI rather than adding a new one.

---

## 6. Deployment boundary and public surface

Render receives a database connection and `JWT_SECRET`, declared in
`render.yaml` as `sync: false` entries so the blueprint documents the
requirement without holding the value.

**Real bank data stays local.** The deployed database is seeded with two
synthetic tenants. Two rather than one: with a single tenant the deployment is
single-tenant in practice and nothing about isolation is exercised by its
existence. Demo credentials are handed out for one of them.

### Read-only demo without a permission system

A `can_write` flag would be the first row of a permission table, and the second
row always follows. The exposed surface does the same job:

- There is no upload endpoint in this phase.
- CSV import is CLI-only and local.
- `POST /v1/transactions` (one manual entry) is exposed, tenant-scoped.

A demo user adding one invented transaction to their own tenant is harmless and
demonstrates the boundary working. The risk worth avoiding was a reviewer
putting a real bank statement on a free-tier box, and that requires an upload
endpoint that does not exist.

### PII

`import_batches` stores a source file *name* and a SHA-256 checksum. No raw
statement bytes are persisted anywhere in the system. The retention policy is
therefore almost entirely a no-op already — but it is true by accident rather
than by rule, so §7 adds the test that makes it a rule.

Account keys such as `chase-checking-5718` keep their last four digits. The key
is user-chosen, displayed to its owner, and low risk alone. Seeded tenants use
obviously synthetic keys.

---

## 7. What proves isolation

The deliverable of this phase is not a login form. It is this table, with two
users each holding a valid token:

| Scenario | Expected |
|---|---|
| B reads A's account by key | 404 |
| B lists accounts | only B's |
| B enters a transaction naming A's account key | 404 |
| B imports into A's account key | 404 |
| A and B each register the **same** account key | both succeed |
| A and B each store a transaction with the **same** fingerprint | both stored |

The last two rows are what prove the `UNIQUE (user_id, key)` change landed and
that deduplication is per-tenant rather than global.

**404, not 403.** A 403 confirms the resource exists and belongs to somebody
else, which is itself a disclosure. Somebody else's row and no row at all must
be indistinguishable to the caller.

### Authentication failure modes

Missing header, malformed token, wrong signature, expired token, and
deactivated user all return 401. Unknown email and wrong password return an
identical status and body, and the test asserts the dummy-verification path
ran. Timing is not asserted: timing assertions are flaky and would fail for
reasons unrelated to the property.

Repeated failures return 429.

### The migration gets its own test

It is the only step that is awkward to undo. Two cases: a database with orphan
accounts gains exactly one placeholder user and satisfies `NOT NULL`; an empty
database gains no user row at all.

### PII does not reach logs

A log-capturing test runs an import and asserts that no description string and
no amount appears in the captured output.

### Two guarantees that never run

Constructing a repository without a scope is a type error under the existing
strict mypy configuration. A route importing a repository is an import-linter
contract violation. Neither is a runtime test; both make the wrong code
impossible to write rather than possible to catch.

### This runs in CI

The workflow provisions a PostgreSQL service, sets `CONNECTION_STRING` at job
level, and runs `alembic upgrade head` before the suite. Integration tests
therefore execute in CI, and the isolation table above becomes a condition of a
green build rather than a claim in a README.

The comment at the top of `tests/integration/conftest.py` still says CI has no
database. That was true once and is not true now; it is corrected as part of
this work.

---

## 8. Deferred, with reasons

**Refresh tokens.** A refresh token is only meaningful with a revocation store;
without one it is a second, longer-lived secret and nothing else. Re-logging in
is one POST on an invite-only system. Revisit when there is a browser session
to keep alive.

**PostgreSQL row-level security.** RLS defends against a developer forgetting a
`WHERE` clause. Constructor-enforced scoping removes that failure mode earlier,
by making the untenanted query impossible to express. What RLS would add on top
is protection against raw SQL bypassing the repositories, at the cost of a
`SET LOCAL` on every transaction through a connection pooler — and a separate
proof that no pooled path leaks. That trade does not pay at this size. It
becomes worth revisiting if raw SQL enters the request path or a second service
gains database access.

**Roles and permissions.** Nothing in this phase needs a second kind of user.

**Tenanted `comparison_runs`.** Deferred until the table has a writer.

---

## 9. What this unblocks

With identity on every request and a tenant on every query, Phase 1 can build
the auditable monthly report against real imported transactions without
re-opening any of these questions. The report assembles existing domain
functions — `total_income`, `total_spending`, `net_cash_flow`,
`detect_recurring` — which are built and tested today and have no caller
outside their own modules.
