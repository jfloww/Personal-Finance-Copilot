"""The CLI that labels stored transactions. Rules first, the model after."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import cast

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import categorise
from categorise import build_parser, main
from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.config import get_settings
from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.evaluation.categorisers import Prediction
from offerdelta.infrastructure.postgres.repositories import StoredTransaction


def _row(
    merchant: str,
    *,
    amount: str = "-4.50",
    description: str | None = None,
) -> StoredTransaction:
    """A `StoredTransaction` with every field a real row would carry.

    `amount` defaults negative and `description` defaults to the merchant
    itself, carrying no transfer keyword - so by default this row is one the
    rule baseline abstains on, which is what most of these tests want.
    `_rule_answerable_repo` below overrides both to build the one row rules
    should actually answer.
    """
    return StoredTransaction(
        id=uuid.uuid4(),
        imported_at=datetime(2026, 3, 1, tzinfo=UTC),
        account_id=uuid.uuid4(),
        batch_id=None,
        posted_on=date(2026, 3, 1),
        description=description or merchant,
        normalised_merchant=merchant,
        amount=Money.parse(amount),
        external_id=None,
        fingerprint="fingerprint",
        fingerprint_version=1,
        occurrence=1,
        source_file=None,
        source_line=None,
        raw_cells=None,
        suggested_label=None,
        suggested_source=None,
        suggested_confidence=None,
        suggested_by=None,
        confirmed_label=None,
    )


def _prediction(label: str, confidence: str) -> Prediction:
    return Prediction(label=label, confidence=Decimal(confidence), reason="test stub")


class _NullSession:
    """Enough of `Session` for `categorise.main`: commit and close, no-ops.

    Mirrors the `_NullSession` in `tests/unit/test_transactions_cli.py` -
    `_open_scope` is patched out entirely in these tests, so nothing here
    ever reaches a real engine or a real query.
    """

    def commit(self) -> None:
        return None

    def close(self) -> None:
        return None


def _fake_scope() -> TenantScope:
    return TenantScope(
        session=cast(Session, _NullSession()),
        user=AuthenticatedUser(id=uuid.uuid4(), email="a@example.test"),
    )


class _RuleAnswerableRepo:
    """One row the rule baseline answers, one it does not.

    `categorise._rule_baseline` builds an unfitted `RuleBaseline` (no private
    dataset to fit from - see its docstring), so the only rows it can answer
    without escalating are ones its two built-in heuristics catch: a
    transfer keyword, or a positive amount read as income. STARBUCKS is given
    a positive amount here (a refund is a realistic reason for a coffee shop
    charge to run positive) so it is answered by the sign heuristic without
    needing a merchant table. SOME NEW CAFE keeps `_row`'s defaults - a
    negative amount and no keyword - so it reaches the model.
    """

    def __init__(self, _scope: object) -> None:
        pass

    def unclassified(self, limit: int | None = None) -> list[StoredTransaction]:  # noqa: ARG002
        # `limit` must keep this exact name: `categorise._rows_to_classify`
        # calls `unclassified(limit=...)` by keyword, so a fake standing in
        # for the real repository has to accept the same keyword to be
        # call-compatible - even though this fake never reads it.
        return [
            _row("STARBUCKS", amount="4.25", description="STARBUCKS REFUND"),
            _row("SOME NEW CAFE"),
        ]

    def record_suggestion(self, _transaction_id: uuid.UUID, **_kw: object) -> None:
        return None


def _rule_answerable_repo() -> type[_RuleAnswerableRepo]:
    return _RuleAnswerableRepo


# ---------------------------------------------------------------- argument parsing


def test_user_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_reclassify_is_off_by_default() -> None:
    args = build_parser().parse_args(["--user", "a@example.test"])
    assert args.reclassify is False


def test_an_unknown_argument_is_an_error() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--user", "a@example.test", "--force"])


# ---------------------------------------------------------------- the spending gate


def test_the_estimate_is_stated_before_anything_is_spent(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No --yes means the run stops after telling you the price - and writes
    nothing, not even a rule-answered row that costs nothing to label.

    Nine of the ten rows are rule-unanswerable by construction; the tenth
    (STARBUCKS, a positive amount) is one the rule baseline *does* answer for
    free. If a free answer ever slipped past the --yes gate and got written
    early, this is the row that would prove it - a fixture where every row is
    unanswerable could never catch that regression.
    """
    recorded: list[tuple[uuid.UUID, str]] = []

    class _FakeRepo:
        def __init__(self, scope: object) -> None:
            pass

        def unclassified(self, limit: int | None = None) -> list[StoredTransaction]:  # noqa: ARG002
            return [
                *(_row("SOME MERCHANT") for _ in range(9)),
                _row("STARBUCKS", amount="4.25", description="STARBUCKS REFUND"),
            ]

        def record_suggestion(self, transaction_id: uuid.UUID, **kw: object) -> None:
            recorded.append((transaction_id, str(kw["label"])))

    monkeypatch.setattr(categorise, "TransactionRepository", _FakeRepo)
    monkeypatch.setattr(categorise, "_open_scope", lambda _email: _fake_scope())

    exit_code = categorise.main(["--user", "a@example.test"])

    out = capsys.readouterr().out
    assert exit_code == 2
    assert "$" in out, "the estimate must name a price"
    assert "9" in out, "one of the ten rows was rule-answerable, so nine remain"
    assert recorded == [], "nothing may be written without --yes, not even a free rule answer"


def test_rules_run_before_the_model_and_shrink_what_it_is_asked(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The free baseline answers what it can; only the rest costs anything."""
    asked_of_model: list[str] = []

    class _FakeLLM:
        name = "fake-llm"

        def predict_many(self, views: list[object]) -> list[object]:
            asked_of_model.extend(getattr(v, "normalised_merchant", "") for v in views)
            return [_prediction("LIVING_OTHER", "0.5") for _ in views]

    monkeypatch.setattr(categorise, "_llm_categoriser", _FakeLLM)
    monkeypatch.setattr(categorise, "_open_scope", lambda _email: _fake_scope())
    monkeypatch.setattr(categorise, "TransactionRepository", _rule_answerable_repo())

    categorise.main(["--user", "a@example.test", "--yes"])

    capsys.readouterr()
    assert "STARBUCKS" not in asked_of_model, "the rules already answered this one"


# ---------------------------------------------------------------- recording an abstention


def test_a_model_abstention_is_recorded_as_unknown(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fix 2: the model's abstention must reach the database, not just stdout.

    Before this, `still_unanswered` was only ever counted and printed - no
    writer of `suggested_label = 'UNKNOWN'` existed anywhere in production.
    """
    recorded: list[tuple[uuid.UUID, str, str, Decimal]] = []

    class _FakeRepo:
        def __init__(self, _scope: object) -> None:
            pass

        def unclassified(self, limit: int | None = None) -> list[StoredTransaction]:  # noqa: ARG002
            return [_row("SOME NEW CAFE")]

        def record_suggestion(self, transaction_id: uuid.UUID, **kw: object) -> None:
            confidence = cast(Decimal, kw["confidence"])
            recorded.append((transaction_id, str(kw["label"]), str(kw["source"]), confidence))

    class _AbstainingLLM:
        name = "fake-llm:v1"

        def predict_many(self, views: list[object]) -> list[object]:
            return [Prediction.abstain(reason="ambiguous") for _ in views]

    monkeypatch.setattr(categorise, "TransactionRepository", _FakeRepo)
    monkeypatch.setattr(categorise, "_llm_categoriser", _AbstainingLLM)
    monkeypatch.setattr(categorise, "_open_scope", lambda _email: _fake_scope())

    exit_code = categorise.main(["--user", "a@example.test", "--yes"])

    capsys.readouterr()
    assert exit_code == 0
    assert len(recorded) == 1
    _transaction_id, label, source, confidence = recorded[0]
    assert label == "UNKNOWN"
    assert source == "llm"
    assert confidence == Decimal(0)


def test_an_abstained_row_is_not_resent_to_the_model_on_a_later_run(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row the model already declined to answer must not be paid for again.

    The fake repository here tracks `suggested_label` the way the real one
    does, so `unclassified()` on the second run reflects what
    `record_suggestion` wrote on the first - the same behaviour
    `TransactionRepository.unclassified` documents, and that
    `test_unknown_is_a_valid_label_and_is_not_the_same_as_never_examined`
    (in `tests/integration/test_classification_repository.py`) pins against a
    real database.
    """
    row = _row("SOME NEW CAFE")
    suggested: dict[uuid.UUID, str] = {}
    asked_of_model: list[uuid.UUID] = []

    class _StatefulRepo:
        def __init__(self, _scope: object) -> None:
            pass

        def unclassified(self, limit: int | None = None) -> list[StoredTransaction]:  # noqa: ARG002
            return [] if row.id in suggested else [row]

        def record_suggestion(self, transaction_id: uuid.UUID, **kw: object) -> None:
            suggested[transaction_id] = str(kw["label"])

    class _AbstainingLLM:
        name = "fake-llm:v1"

        def predict_many(self, views: list[object]) -> list[object]:
            asked_of_model.append(row.id)
            return [Prediction.abstain(reason="ambiguous") for _ in views]

    monkeypatch.setattr(categorise, "TransactionRepository", _StatefulRepo)
    monkeypatch.setattr(categorise, "_llm_categoriser", _AbstainingLLM)
    monkeypatch.setattr(categorise, "_open_scope", lambda _email: _fake_scope())

    first = categorise.main(["--user", "a@example.test", "--yes"])
    second = categorise.main(["--user", "a@example.test", "--yes"])
    capsys.readouterr()

    assert first == 0
    assert second == 0
    assert suggested[row.id] == "UNKNOWN"
    assert asked_of_model == [row.id], "the second run must not re-send the abstained row"


# ---------------------------------------------------------------- error handling


def test_an_unknown_user_reports_a_validation_error_without_a_traceback(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise(_email: str) -> TenantScope:
        raise ValidationError("no user 'a@example.test'; create it first with users.py create")

    monkeypatch.setattr(categorise, "_open_scope", _raise)

    code = main(["--user", "a@example.test"])
    out = capsys.readouterr().out

    assert code == 1
    assert "no user" in out
    assert "Traceback" not in out


def test_a_database_error_reports_the_redacted_host_without_the_dsn(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config.py` promises the connection string never reaches an error message.

    `_report`'s `SQLAlchemyError` branch is what keeps that promise for this
    tool specifically - a raw SQLAlchemy exception can carry the DSN and the
    SQL that was running, real amounts and descriptions included. This
    exercises the branch directly, the same way
    `test_a_database_error_never_echoes_credentials_or_the_raw_message` does
    for `users.py`, rather than trusting it by inspection alone.
    """

    def _raise(_email: str) -> TenantScope:
        raise SQLAlchemyError(
            'connection to server failed: password authentication failed for user "hunter2"'
        )

    monkeypatch.setenv("CONNECTION_STRING", "postgresql://user:hunter2@dbhost/offerdelta")
    get_settings.cache_clear()
    monkeypatch.setattr(categorise, "_open_scope", _raise)
    try:
        code = main(["--user", "a@example.test"])
        out = capsys.readouterr().out
    finally:
        get_settings.cache_clear()

    assert code == 1
    assert "dbhost" in out, "the redacted host is allowed to appear"
    assert "hunter2" not in out
    assert "password authentication failed" not in out


def test_an_unexpected_runtime_error_is_reported_generically(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unrecognised `RuntimeError` might carry anything - `_report`'s
    generic message, not the exception's own text, is what may reach the
    terminal. `database_available` must be true for this branch: the one
    `RuntimeError` `_report` prints verbatim is the specific, message-free
    "CONNECTION_STRING is not set" one, which only fires when it is not.
    """

    def _raise(_email: str) -> TenantScope:
        raise RuntimeError("some internal detail nobody vetted")

    monkeypatch.setenv("CONNECTION_STRING", "postgresql://user:hunter2@dbhost/offerdelta")
    get_settings.cache_clear()
    monkeypatch.setattr(categorise, "_open_scope", _raise)
    try:
        code = main(["--user", "a@example.test"])
        out = capsys.readouterr().out
    finally:
        get_settings.cache_clear()

    assert code == 1
    assert "some internal detail" not in out
    assert "unexpected internal error" in out
