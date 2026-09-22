"""Public synthetic investigations, historical demonstrations, and protected APIs."""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Final

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from offerdelta.agent.tools.registry import JsonValue
from offerdelta.api.presenters import present_comparison
from offerdelta.api.rate_limit import FixedWindowLimiter
from offerdelta.api.schemas import (
    ComparisonRequest,
    ComparisonSchema,
    DemoInvestigationRequest,
    DerivationNodeSchema,
    HealthSchema,
    LabelConfirmationSchema,
    LoginSchema,
    MonthCoverageSchema,
    MonthlyReportSchema,
    ObservedDebitComparisonSchema,
    ReadinessSchema,
    ReviewQueueRowSchema,
    TokenSchema,
    TransactionEntrySchema,
    TransactionStoredSchema,
    VersionSchema,
)
from offerdelta.application.auth import authenticate, load_active_user
from offerdelta.application.idempotency import IdempotencyOutcome, IdempotencyService
from offerdelta.application.queries.get_demo_comparison import get_demo_comparison
from offerdelta.application.queries.get_demo_derivation import get_demo_derivation
from offerdelta.application.queries.observed_debits import compare_observed_debits
from offerdelta.application.queries.operations_demo import investigate_demo
from offerdelta.application.reports.monthly import available_months, monthly_report
from offerdelta.application.reports.review import REVIEW_THRESHOLD, confirm, queue
from offerdelta.application.scope import TenantScope
from offerdelta.application.transactions.enter_transaction import (
    ManualEntry,
    enter_transaction,
)
from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.transactions.fingerprint import FINGERPRINT_VERSION
from offerdelta.domain.transactions.parsing import parse_amount
from offerdelta.domain.users.identity import normalise_email
from offerdelta.infrastructure.auth.tokens import decode_token, issue_token
from offerdelta.infrastructure.memory.idempotency import InMemoryIdempotencyStore
from offerdelta.infrastructure.postgres.engine import get_engine

#: Bumped whenever a calculation rule changes. Every result will reference it
#: once results are persisted, so a stored figure stays reproducible.
ENGINE_VERSION: Final = "0.1.0-skeleton"

_STATIC = Path(__file__).parent / "static"

#: Aggregate evaluation results, generated from the archived local runs by
#: `build_public_results.py`. It lives in the package so it ships with the
#: wheel, and it holds counts and rates only - the runs it summarises read real
#: statements and an API key, and neither can leave the machine they ran on.
_EVALUATION = _STATIC / "evaluation.json"

#: Process-local, so it guards a single instance. Honest for one container and
#: inadequate for two, which is why it sits behind a port — DynamoDB with
#: conditional writes replaces it when the async path arrives.
_idempotency = IdempotencyService(InMemoryIdempotencyStore())

app = FastAPI(
    title="MyFinSecretary",
    summary="Synthetic read-only transaction investigations and approval-gated design",
    description=(
        "**Public demo mode: the agent demonstration uses synthetic in-memory transactions only.** "
        "It executes a fixed scripted tool sequence, not a live model. Policy lookup is "
        "deterministic keyword search, not vector RAG. Review proposals do not mutate a ledger "
        "or queue. Database-backed routes, when configured, are separately authenticated.\n\n"
        "`/demo/evaluation/latest` publishes aggregate results from evaluation "
        "runs performed locally for the legacy categorisation research. It carries counts and "
        "rates only - no transactions, amounts, merchants, dates, or per-row "
        "predictions."
    ),
    docs_url="/docs",
)


def _package_version() -> str:
    try:
        return version("offerdelta")
    except PackageNotFoundError:  # pragma: no cover - only when running from source
        return "unknown"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Product-first public landing page; the old evaluation remains reachable."""
    return FileResponse(_STATIC / "agent.html")


@app.get("/demo/evaluation", include_in_schema=False)
def evaluation_page() -> FileResponse:
    """The historical categorisation evaluation showcase."""
    return FileResponse(_STATIC / "index.html")


@app.get("/demo/comparison", include_in_schema=False)
def comparison_page() -> FileResponse:
    """The deterministic engine, rendered with its full derivation.

    Reads `/v1/demo/comparison`. Every amount stays a decimal string all the way
    into the DOM - the page formats by string surgery rather than coercing to a
    double, so what it prints is what the engine computed.
    """
    return FileResponse(_STATIC / "comparison.html")


@app.get("/demo/agent", include_in_schema=False)
def operations_workbench() -> FileResponse:
    """Public transaction-operations investigation over synthetic data only."""
    return FileResponse(_STATIC / "agent.html")


@app.get("/v1/demo/agent/investigation")
def demo_investigation() -> dict[str, JsonValue]:
    """Scripted, reproducible investigation. Not a live model result or ledger write."""
    return investigate_demo()


@app.post("/v1/demo/agent/run")
def run_demo_investigation(_request: DemoInvestigationRequest) -> dict[str, JsonValue]:
    """Run a bundled read-only scenario, never an unrestricted live-model prompt."""
    return investigate_demo(_request.scenario)


@lru_cache(maxsize=1)
def _evaluation_payload() -> str:
    """The published results, read once and served byte for byte.

    Verbatim rather than parsed and re-serialised: the file is a generated
    artifact committed to the repository, and round-tripping it here would
    create a second place for the numbers to differ from it.
    """
    return _EVALUATION.read_text(encoding="utf-8")


@app.get("/demo/evaluation/latest")
def evaluation_results() -> Response:
    """Aggregate results for the frozen validation benchmark.

    Dataset counts, per-system scores, the prompt experiment and its rejection,
    and per-label movement. Counts and rates only: the runs behind these numbers
    read real bank statements and used a model API key, and neither is deployed.

    "Frozen validation benchmark" rather than test set, deliberately. The v1
    results informed the design of v2, so these rows have already influenced a
    decision and can no longer be described as held out.
    """
    try:
        payload = _evaluation_payload()
    except OSError as error:  # pragma: no cover - the artifact is committed
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "evaluation results have not been generated for this build; "
            "run build_public_results.py",
        ) from error
    return Response(content=payload, media_type="application/json")


#: The transaction endpoints need a database. Without one they can only
#: return 500, so they are not registered at all rather than advertised in the
#: public schema and then failing - a documented endpoint that cannot work is
#: worse than an absent one.
_DATABASE_CONFIGURED: Final = get_settings().database_available

#: Without a signing key nothing can be authenticated. The login route hides
#: from the schema exactly like the transaction routes hide without a
#: database, and every protected route answers 503 rather than pretending a
#: token could ever be valid.
_AUTH_CONFIGURED: Final = get_settings().auth_available

#: Five failed logins per address per fifteen minutes. Process-local: see
#: `FixedWindowLimiter`'s docstring for what that does and does not guarantee.
_login_limiter = FixedWindowLimiter(max_attempts=5, window=timedelta(minutes=15))


@app.get("/v1/health/live", response_model=HealthSchema)
def live() -> HealthSchema:
    """The process is running."""
    return HealthSchema(status="live")


def _database_state() -> str:
    """What the database is actually doing, checked rather than assumed."""
    if not get_settings().database_available:
        return "unconfigured"
    try:
        with get_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
    except SQLAlchemyError:
        # The DSN is never echoed, here least of all: this response is public.
        return "unreachable"
    return "connected"


@app.get("/v1/health/ready", response_model=ReadinessSchema)
def ready() -> ReadinessSchema:
    """The service can serve traffic, and says what it cannot serve.

    Deliberately still "ready" without a database. The demo endpoints are
    in-memory and work regardless, and this path is the platform health check -
    failing it would take a working demo offline to report a missing database.
    The `database` field carries that news instead.
    """
    return ReadinessSchema(status="ready", database=_database_state())


@app.get("/v1/version", response_model=VersionSchema)
def service_version() -> VersionSchema:
    return VersionSchema(
        service="offerdelta",
        version=_package_version(),
        engine=ENGINE_VERSION,
    )


@app.get("/v1/demo/derivation", response_model=DerivationNodeSchema)
def demo_derivation() -> DerivationNodeSchema:
    """Monthly disposable cash for the demo profile, with its full derivation.

    All amounts are decimal strings. Do not parse them as JavaScript numbers.
    """
    return DerivationNodeSchema.of(get_demo_derivation())


@app.get("/v1/demo/comparison", response_model=ComparisonSchema)
def demo_comparison() -> ComparisonSchema:
    """The full Auburn-to-New-Jersey comparison.

    Component deltas, both derivation trees, the cumulative series, and all
    three solvers. Every amount is a decimal string.

    `reconciled` reports whether every projected month balanced on both sides.
    The engine refuses to return an unbalanced result, so it is always true —
    it is surfaced so a reader can see the guarantee rather than trust it.
    """
    return present_comparison(get_demo_comparison())


@app.post("/v1/comparisons", status_code=201, response_model=ComparisonSchema)
def run_comparison(
    request: ComparisonRequest,
    response: Response,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> Response:
    """Run a comparison over the demo profiles.

    Guarded by the standard idempotency contract. A retry with the same key and
    the same body replays the original response byte for byte and sets
    `Idempotent-Replay: true`; the same key with a different body is a `409`,
    because reusing a key for different content is a client bug that should
    surface rather than silently return the first answer.
    """
    body = request.model_dump(mode="json")
    now = datetime.now(UTC)

    outcome = _idempotency.begin(key=idempotency_key, body=body, now=now)

    if outcome.kind is IdempotencyOutcome.Kind.CONFLICT:
        raise HTTPException(status_code=409, detail=outcome.reason)

    if outcome.kind is IdempotencyOutcome.Kind.REPLAY:
        return Response(
            content=outcome.response,
            status_code=200,
            media_type="application/json",
            headers={"Idempotent-Replay": "true"},
        )

    try:
        view = get_demo_comparison(
            horizon_months=request.horizon_months,
            move_date=request.move_date,
        )
        payload = present_comparison(view).model_dump_json()
    except ValidationError as error:
        # Release the key: the caller's reason for retrying is that this did not
        # finish, and holding it would make a corrected retry impossible.
        if idempotency_key is not None:
            _idempotency.abandon(key=idempotency_key)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception:
        if idempotency_key is not None:
            _idempotency.abandon(key=idempotency_key)
        raise

    if idempotency_key is not None:
        _idempotency.complete(key=idempotency_key, response=payload)

    response.status_code = 201
    return Response(content=payload, status_code=201, media_type="application/json")


def _session() -> Iterator[Session]:
    """One request, one transaction.

    Committed only when the handler returns. A handler that raises leaves
    nothing behind, which is the property the whole import path is built on and
    the reason manual entry does not get its own weaker rule.
    """
    if not _DATABASE_CONFIGURED:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "transaction storage is not configured on this deployment; "
            "the demo endpoints are unaffected",
        )
    with Session(get_engine()) as session:
        yield session
        session.commit()


@app.post("/v1/auth/token", include_in_schema=_AUTH_CONFIGURED, response_model=TokenSchema)
def issue_access_token(
    body: LoginSchema, session: Annotated[Session, Depends(_session)]
) -> TokenSchema:
    """One endpoint, one answer shape.

    Unknown address and wrong password return the same status and the same
    body, so this cannot be used to learn who has an account. The rate-limit
    check runs before authentication and counts every call, not only failed
    ones, so it cannot itself be used to probe whether an address exists.

    The limiter is keyed on `normalise_email`, the same function
    `UserRepository` matches addresses on - not a second, ad hoc `.lower()`
    here. Two normalisations that can drift apart is exactly what let
    " victim@x.test" authenticate against the real row while opening a fresh
    rate-limit budget: the repository stripped the space and the limiter did
    not, so the two calls disagreed about which address they had just seen.
    """
    if not _login_limiter.check(normalise_email(body.email)):
        raise HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "too many attempts")

    who = authenticate(session, body.email, body.password)
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
    """Identity first, then data. Both, or neither.

    The user is reloaded through `load_active_user` on every call rather than
    trusted from the token's claims, so a deactivation takes effect on the
    very next request instead of whenever that request's token happens to
    expire.
    """
    secret = get_settings().jwt_secret
    if secret is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "authentication is not configured")
    if credentials is None or not credentials.startswith("Bearer "):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")

    user_id = decode_token(credentials.removeprefix("Bearer "), secret=secret)
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")

    who = load_active_user(session, user_id)
    if who is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")

    return TenantScope(session=session, user=who)


@app.post(
    "/v1/transactions",
    include_in_schema=_DATABASE_CONFIGURED,
    response_model=TransactionStoredSchema,
    status_code=status.HTTP_201_CREATED,
)
def create_transaction(
    body: TransactionEntrySchema, scope: Annotated[TenantScope, Depends(_scope)]
) -> TransactionStoredSchema:
    """Enter one transaction by hand.

    409 rather than a silent success when a matching transaction already
    exists: the caller asked for something that is already true, and telling
    them so is the difference between a duplicate they can see and one they
    cannot. `repeat` is how they say they meant it.
    """
    try:
        amount = parse_amount(body.amount)
    except ValidationError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    entry = ManualEntry(
        account_key=body.account_key,
        posted_on=body.posted_on,
        description=body.description,
        amount=amount,
        repeat=body.repeat,
    )

    try:
        outcome = enter_transaction(scope, entry)
    except ValidationError as error:
        # The only ValidationError this path raises is an unknown account, and
        # its message names the registered ones - scoped to this caller's own
        # accounts, never another tenant's. 404, not 403: naming the resource
        # as forbidden would itself confirm it exists.
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error

    if not outcome.stored:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            f"{outcome.already_stored_count} identical transaction(s) already stored; "
            f"set repeat=true if there really was another one",
        )

    return TransactionStoredSchema(
        transaction_id=str(outcome.transaction_id),
        fingerprint=outcome.fingerprint,
        fingerprint_version=FINGERPRINT_VERSION,
        occurrence=outcome.occurrence,
    )


#: `YYYY-MM` and nothing else. `date.fromisoformat` alone would also accept
#: `2026-03-01` and silently take its first two fields, which would make
#: `/v1/reports/monthly/2026-03-01` behave like `2026-03` instead of 422ing -
#: the exact kind of malformed input this route has to refuse rather than
#: quietly reinterpret.
_MONTH_PATTERN: Final = re.compile(r"(?P<year>\d{4})-(?P<month>\d{2})")


def _parse_month(value: str) -> tuple[int, int]:
    """Parse a `YYYY-MM` route value, refusing anything that is not exactly that shape.

    Raises `ValidationError` - mapped to 422 by every caller below - for a
    string the pattern does not match and, separately, for one that matches
    but names a month that does not exist (`2026-13`): `date(year, month, 1)`
    is what actually proves the month is real, since the regex alone accepts
    `99-99` as two two-digit groups.
    """
    match = _MONTH_PATTERN.fullmatch(value)
    if match is None:
        raise ValidationError(f"{value!r} is not a YYYY-MM month")
    year, month = int(match["year"]), int(match["month"])
    try:
        date(year, month, 1)
    except ValueError as error:
        raise ValidationError(f"{value!r} is not a valid month") from error
    return year, month


@app.get(
    "/v1/reports/months",
    include_in_schema=_DATABASE_CONFIGURED,
    response_model=list[MonthCoverageSchema],
)
def report_months(scope: Annotated[TenantScope, Depends(_scope)]) -> list[MonthCoverageSchema]:
    """Every month this tenant has stored a row for, each marked complete or partial.

    Scoped by `_scope` alone: `available_months` takes a `TenantScope`, and
    every query it runs filters on that tenant's own accounts - there is no
    month id or account list a caller could substitute to reach another
    tenant's data. `REVIEW_THRESHOLD` is passed explicitly rather than left
    to a default so this list's `awaiting_review` always means the same bar
    `/v1/review-queue` uses - see `MonthCoverage`'s docstring for why the two
    functions require it rather than defaulting it independently.
    """
    return [
        MonthCoverageSchema.of(coverage)
        for coverage in available_months(scope, threshold=REVIEW_THRESHOLD)
    ]


@app.get(
    "/v1/reports/monthly/{month}",
    include_in_schema=_DATABASE_CONFIGURED,
    response_model=MonthlyReportSchema,
)
def report_monthly(
    month: str, scope: Annotated[TenantScope, Depends(_scope)]
) -> MonthlyReportSchema:
    """One month's tree, with its coverage.

    A month with no stored rows is not an error: `monthly_report` still
    builds a tree, rooted at zero, so a tenant who has imported nothing this
    month gets a valid empty report rather than a 404. `month` is parsed
    before any query runs, so a malformed value is a 422 from bad input
    rather than a query silently answering for whatever `_parse_month` would
    have rejected.
    """
    try:
        year, mon = _parse_month(month)
    except ValidationError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    return MonthlyReportSchema.of(monthly_report(scope, year, mon, threshold=REVIEW_THRESHOLD))


@app.get(
    "/v1/reports/observed-debits/{month}",
    include_in_schema=_DATABASE_CONFIGURED,
    response_model=ObservedDebitComparisonSchema,
)
def report_observed_debits(
    month: str, scope: Annotated[TenantScope, Depends(_scope)]
) -> ObservedDebitComparisonSchema:
    """Compare this tenant's raw debit magnitudes with the preceding month.

    Months need not be complete for the evidence to be visible, but their
    coverage is returned so clients cannot silently present a partial-month
    comparison as a definitive change in spending.
    """
    try:
        year, mon = _parse_month(month)
        if (year, mon) == (1, 1):
            raise ValidationError("0001-01 has no preceding calendar month")
    except ValidationError as error:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error
    return ObservedDebitComparisonSchema.of(
        compare_observed_debits(scope, year, mon, threshold=REVIEW_THRESHOLD)
    )


@app.get(
    "/v1/review-queue",
    include_in_schema=_DATABASE_CONFIGURED,
    response_model=list[ReviewQueueRowSchema],
)
def review_queue(
    scope: Annotated[TenantScope, Depends(_scope)], month: str | None = None
) -> list[ReviewQueueRowSchema]:
    """Rows nobody has confirmed and no categoriser confidently resolved.

    `month`, when given, narrows to one calendar month and is parsed with the
    same `_parse_month` `/v1/reports/monthly/{month}` uses, so a malformed
    value is 422 here too rather than silently matching nothing. Scoped by
    `_scope`: `queue` (`offerdelta.application.reports.review.queue`) takes a
    `TenantScope` and reaches `TransactionRepository.awaiting_review`, which
    filters on this tenant's own accounts on every path - there is no request
    shape that reaches another tenant's rows.
    """
    parsed_month: tuple[int, int] | None = None
    if month is not None:
        try:
            parsed_month = _parse_month(month)
        except ValidationError as error:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(error)) from error

    rows = queue(scope, REVIEW_THRESHOLD, month=parsed_month)
    return [
        ReviewQueueRowSchema(
            transaction_id=str(row.id),
            posted_on=row.posted_on,
            description=row.description,
            amount=str(row.amount.amount),
            suggested_label=row.suggested_label,
            suggested_confidence=(
                None if row.suggested_confidence is None else str(row.suggested_confidence)
            ),
        )
        for row in rows
    ]


@app.post(
    "/v1/transactions/{transaction_id}/label",
    include_in_schema=_DATABASE_CONFIGURED,
    status_code=status.HTTP_204_NO_CONTENT,
)
def confirm_transaction_label(
    transaction_id: uuid.UUID,
    body: LabelConfirmationSchema,
    scope: Annotated[TenantScope, Depends(_scope)],
) -> None:
    """Record a person's decision on one transaction.

    404, never 403, for a `transaction_id` naming another tenant's row (or no
    row at all): a 403 would itself confirm the row exists. `body.label` is
    already checked against the taxonomy at the wire boundary - see
    `LabelConfirmationSchema` - so the only `ValidationError` `confirm` can
    raise by the time it reaches here is the tenancy one, and this handler
    does not need to tell the two apart by inspecting the message.
    """
    try:
        confirm(scope, transaction_id, body.label)
    except ValidationError as error:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(error)) from error
