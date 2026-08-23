"""The walking skeleton's HTTP surface.

One calculated figure, its full derivation, and the health endpoints a hosted
service needs. Deliberately small: this exists to prove the deployment path and
the money-serialisation boundary while both are still trivial to debug.

The real API arrives in milestone 5.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Annotated, Final

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from offerdelta.api.presenters import present_comparison
from offerdelta.api.schemas import (
    ComparisonRequest,
    ComparisonSchema,
    DerivationNodeSchema,
    HealthSchema,
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
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.transactions.fingerprint import FINGERPRINT_VERSION
from offerdelta.domain.transactions.parsing import parse_amount
from offerdelta.infrastructure.memory.idempotency import InMemoryIdempotencyStore
from offerdelta.infrastructure.postgres.engine import get_engine

#: Bumped whenever a calculation rule changes. Every result will reference it
#: once results are persisted, so a stored figure stays reproducible.
ENGINE_VERSION: Final = "0.1.0-skeleton"

_STATIC = Path(__file__).parent / "static"

#: Process-local, so it guards a single instance. Honest for one container and
#: inadequate for two, which is why it sits behind a port — DynamoDB with
#: conditional writes replaces it when the async path arrives.
_idempotency = IdempotencyService(InMemoryIdempotencyStore())

app = FastAPI(
    title="Personal Finance Copilot",
    summary="Deterministic personal-finance engine with AI kept outside the calculation boundary",
    docs_url="/docs",
)


def _package_version() -> str:
    try:
        return version("offerdelta")
    except PackageNotFoundError:  # pragma: no cover - only when running from source
        return "unknown"


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(_STATIC / "index.html")


@app.get("/v1/health/live", response_model=HealthSchema)
def live() -> HealthSchema:
    """The process is running."""
    return HealthSchema(status="live")


@app.get("/v1/health/ready", response_model=HealthSchema)
def ready() -> HealthSchema:
    """The service can serve traffic.

    Once PostgreSQL arrives in milestone 5 this checks the connection; for now
    readiness and liveness are the same thing.
    """
    return HealthSchema(status="ready")


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
    with Session(get_engine()) as session:
        yield session
        session.commit()


@app.post(
    "/v1/transactions",
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
