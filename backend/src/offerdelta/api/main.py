"""The walking skeleton's HTTP surface.

One calculated figure, its full derivation, and the health endpoints a hosted
service needs. Deliberately small: this exists to prove the deployment path and
the money-serialisation boundary while both are still trivial to debug.

The real API arrives in milestone 5.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Final

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import FileResponse
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from offerdelta.api.presenters import present_comparison
from offerdelta.api.schemas import (
    ComparisonRequest,
    ComparisonSchema,
    DerivationNodeSchema,
    HealthSchema,
    ReadinessSchema,
    TransactionEntrySchema,
    TransactionStoredSchema,
    VersionSchema,
)
from offerdelta.application.idempotency import IdempotencyOutcome, IdempotencyService
from offerdelta.application.queries.get_demo_comparison import get_demo_comparison
from offerdelta.application.queries.get_demo_derivation import get_demo_derivation
from offerdelta.application.transactions.enter_transaction import (
    ManualEntry,
    enter_transaction,
)
from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.transactions.fingerprint import FINGERPRINT_VERSION
from offerdelta.domain.transactions.parsing import parse_amount
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
    title="Personal Finance Copilot",
    summary="Deterministic personal-finance engine with AI kept outside the calculation boundary",
    description=(
        "**This deployment runs in public demo mode.** Real financial ingestion "
        "is disabled here by design: no database is attached, the transaction "
        "endpoints are not registered, and no model API key is present. The "
        "demo routes below compute from fixed in-memory profiles.\n\n"
        "`/demo/evaluation/latest` publishes aggregate results from evaluation "
        "runs performed locally against real statements. It carries counts and "
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
    """The evaluation showcase: what was measured, and which prompt it selected."""
    return FileResponse(_STATIC / "index.html")


@app.get("/demo/comparison", include_in_schema=False)
def comparison_page() -> FileResponse:
    """The deterministic engine, rendered with its full derivation.

    Reads `/v1/demo/comparison`. Every amount stays a decimal string all the way
    into the DOM - the page formats by string surgery rather than coercing to a
    double, so what it prints is what the engine computed.
    """
    return FileResponse(_STATIC / "comparison.html")


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


@app.post(
    "/v1/transactions",
    include_in_schema=_DATABASE_CONFIGURED,
    response_model=TransactionStoredSchema,
    status_code=status.HTTP_201_CREATED,
)
def create_transaction(
    body: TransactionEntrySchema, session: Annotated[Session, Depends(_session)]
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
        outcome = enter_transaction(session, entry)
    except ValidationError as error:
        # The only ValidationError this path raises is an unknown account, and
        # its message names the registered ones.
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
