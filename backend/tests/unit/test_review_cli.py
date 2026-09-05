"""The CLI that works the review queue. One question per merchant, by default."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from decimal import Decimal
from typing import cast

import pytest
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import review
from offerdelta.application.scope import AuthenticatedUser, TenantScope
from offerdelta.domain.common.money import Money
from offerdelta.infrastructure.postgres.repositories import StoredTransaction
from review import build_parser, group_by_merchant, main, resolve_label, suggested_default


def _row(
    merchant: str,
    *,
    amount: str = "-4.50",
    suggested: str | None = None,
    confidence: str = "0.60",
    posted_on: date = date(2026, 3, 1),
) -> StoredTransaction:
    """A queue row: never confirmed, and by default never examined either.

    `suggested=None` is the "never examined" leg of `awaiting_review`;
    passing `"UNKNOWN"` is the "examined and declined" leg; passing a real
    label with a confidence below the threshold is the third.
    """
    return StoredTransaction(
        id=uuid.uuid4(),
        imported_at=datetime(2026, 3, 1, tzinfo=UTC),
        account_id=uuid.uuid4(),
        batch_id=None,
        posted_on=posted_on,
        description=merchant,
        normalised_merchant=merchant,
        amount=Money.parse(amount),
        external_id=None,
        fingerprint="fingerprint",
        fingerprint_version=1,
        occurrence=1,
        source_file=None,
        source_line=None,
        raw_cells=None,
        suggested_label=suggested,
        suggested_source=None if suggested is None else "llm",
        suggested_confidence=None if suggested is None else Decimal(confidence),
        suggested_by=None if suggested is None else "fake-llm:v1",
        confirmed_label=None,
    )


class _RecordingSession:
    """Commits, in the order they arrived, into the same log the writes use.

    Per-merchant commits are only worth anything if each lands before the
    next merchant is asked about, so a commit *count* is not enough - it
    cannot tell that apart from every commit arriving at the end.
    """

    def __init__(self, log: list[str], *, fail_on: int | None = None) -> None:
        self._log = log
        self._fail_on = fail_on
        self.commits = 0

    def commit(self) -> None:
        self.commits += 1
        if self.commits == self._fail_on:
            raise SQLAlchemyError("connection closed by the pooler")
        self._log.append("commit")

    def close(self) -> None:
        return None


def _fake_scope(session: object | None = None) -> TenantScope:
    return TenantScope(
        session=cast(Session, session if session is not None else _RecordingSession([])),
        user=AuthenticatedUser(id=uuid.uuid4(), email="a@example.test"),
    )


class _Harness:
    """A queue, scripted answers, and a log of everything that was written.

    `install` replaces the three seams `review.py` exposes for exactly this:
    the database (`_open_scope`), the person (`_prompt`), and the terminal
    check (`_is_a_terminal`). Nothing here reaches a real engine.
    """

    def __init__(self, rows: list[StoredTransaction], answers: list[str]) -> None:
        self.rows = rows
        self.answers = list(answers)
        self.log: list[str] = []
        self.confirmed: list[tuple[str, str]] = []
        self.names = {row.id: row.normalised_merchant for row in rows}
        self.session = _RecordingSession(self.log)

    def install(self, monkeypatch: pytest.MonkeyPatch, *, terminal: bool = True) -> None:
        answers, rows = self.answers, self.rows

        def _prompt(_message: str) -> str:
            if not answers:
                raise EOFError
            return answers.pop(0)

        def _confirm(_scope: object, transaction_id: uuid.UUID, label: str) -> None:
            self.confirmed.append((self.names[transaction_id], label))
            self.log.append(f"{self.names[transaction_id]}={label}")

        monkeypatch.setattr(review, "_is_a_terminal", lambda: terminal)
        monkeypatch.setattr(review, "_prompt", _prompt)
        monkeypatch.setattr(review, "queue", lambda _scope, _threshold, month=None: rows)  # noqa: ARG005
        monkeypatch.setattr(review, "confirm", _confirm)
        monkeypatch.setattr(review, "_open_scope", lambda _email: _fake_scope(self.session))

    def run(self) -> int:
        return main(["work", "--user", "a@example.test"])


# ---------------------------------------------------------------- argument parsing


def test_a_subcommand_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_user_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["work"])


def test_a_month_is_parsed_into_a_year_and_a_month() -> None:
    args = build_parser().parse_args(["list", "--user", "a@example.test", "--month", "2026-03"])
    assert args.month == (2026, 3)


@pytest.mark.parametrize("bad", ["2026", "2026-13", "26-03", "march", "2026-00"])
def test_an_impossible_month_is_a_usage_error(bad: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["list", "--user", "a@example.test", "--month", bad])


# ---------------------------------------------------------------- resolving a label


def test_a_full_label_resolves_to_itself() -> None:
    assert resolve_label("living_dining") == ("LIVING_DINING", ())


def test_a_unique_substring_is_enough() -> None:
    """Every label is `AREA_THING`, so substring beats prefix: `din` has to work."""
    assert resolve_label("din") == ("LIVING_DINING", ())


def test_an_ambiguous_entry_resolves_to_nothing_and_returns_the_candidates() -> None:
    label, candidates = resolve_label("housing")
    assert label is None
    assert len(candidates) > 1
    assert all("HOUSING" in candidate for candidate in candidates)


def test_an_unmatched_entry_returns_no_candidates() -> None:
    assert resolve_label("zzz") == (None, ())


# ---------------------------------------------------------------- what Enter accepts


def test_enter_accepts_a_suggestion_the_whole_group_agrees_on() -> None:
    rows = [_row("STARBUCKS", suggested="LIVING_DINING") for _ in range(3)]
    assert suggested_default(rows) == "LIVING_DINING"


def test_enter_stands_for_nothing_when_the_model_disagreed_with_itself() -> None:
    """Taking the commonest would hide the disagreement behind one keystroke."""
    rows = [_row("AMAZON", suggested="LIVING_GROCERY"), _row("AMAZON", suggested="LIVING_CLOTHING")]
    assert suggested_default(rows) is None


def test_enter_stands_for_nothing_on_rows_the_model_never_answered() -> None:
    assert suggested_default([_row("A"), _row("B", suggested="UNKNOWN")]) is None


def test_a_row_with_no_suggestion_will_not_take_a_bare_enter(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enter must not become "whatever was on screen" where nothing was proposed."""
    harness = _Harness([_row("SOME NEW CAFE")], answers=["", "din"])
    harness.install(monkeypatch)

    exit_code = harness.run()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "nothing to accept here" in out
    assert harness.confirmed == [("SOME NEW CAFE", "LIVING_DINING")]


# ---------------------------------------------------------------- grouping


def test_rows_group_by_merchant_in_the_order_the_queue_returned_them() -> None:
    rows = [_row("B"), _row("A"), _row("B")]
    assert list(group_by_merchant(rows)) == ["B", "A"]
    assert len(group_by_merchant(rows)["B"]) == 2


def test_one_answer_settles_every_row_of_a_merchant(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole reason the queue is walked by merchant rather than by row."""
    rows = [_row("STARBUCKS", suggested="LIVING_DINING") for _ in range(4)]
    harness = _Harness(rows, answers=["", "y"])
    harness.install(monkeypatch)

    exit_code = harness.run()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert harness.confirmed == [("STARBUCKS", "LIVING_DINING")] * 4
    assert "confirmed 4 rows" in out


def test_a_single_row_merchant_is_never_asked_whether_to_apply_to_all(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness([_row("ONE OFF", suggested="LIVING_OTHER")], answers=[""])
    harness.install(monkeypatch)

    exit_code = harness.run()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "apply to all" not in out
    assert harness.confirmed == [("ONE OFF", "LIVING_OTHER")]


def test_split_asks_each_row_of_a_group_separately(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """AMAZON is groceries one week and clothing the next; the group must break."""
    rows = [_row("AMAZON"), _row("AMAZON")]
    harness = _Harness(rows, answers=["groc", "s", "", "cloth"])
    harness.install(monkeypatch)

    exit_code = harness.run()

    capsys.readouterr()
    assert exit_code == 0
    # The group's own answer becomes the default for a row the model said
    # nothing about, so the bare Enter above is LIVING_GROCERY, not an error.
    assert harness.confirmed == [("AMAZON", "LIVING_GROCERY"), ("AMAZON", "LIVING_CLOTHING")]


# ---------------------------------------------------------------- writing


def test_each_merchant_is_committed_before_the_next_is_asked_about(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The connection is idle while a person thinks, so it must not stay checked out.

    Asserted as an ordered log rather than a commit count: "write everything,
    then commit twice" would satisfy a count and keep the defect.
    """
    rows = [
        _row("STARBUCKS", suggested="LIVING_DINING"),
        _row("STARBUCKS", suggested="LIVING_DINING"),
        _row("KROGER", suggested="LIVING_GROCERY"),
    ]
    harness = _Harness(rows, answers=["", "y", ""])
    harness.install(monkeypatch)

    exit_code = harness.run()

    capsys.readouterr()
    assert exit_code == 0
    assert harness.log == [
        "STARBUCKS=LIVING_DINING",
        "STARBUCKS=LIVING_DINING",
        "commit",
        "KROGER=LIVING_GROCERY",
        "commit",
    ]


def test_quitting_keeps_what_was_already_decided(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An hour of answers must survive stopping, or nobody starts the queue twice."""
    rows = [_row("STARBUCKS", suggested="LIVING_DINING"), _row("KROGER")]
    harness = _Harness(rows, answers=["", "q"])
    harness.install(monkeypatch)

    exit_code = harness.run()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert harness.confirmed == [("STARBUCKS", "LIVING_DINING")]
    assert harness.log == ["STARBUCKS=LIVING_DINING", "commit", "commit"]
    assert "confirmed 1 rows" in out


def test_running_out_of_answers_ends_the_session_rather_than_raising(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """EOF is a person walking away, not a crash - and it keeps their work."""
    rows = [_row("STARBUCKS", suggested="LIVING_DINING"), _row("KROGER")]
    harness = _Harness(rows, answers=[""])
    harness.install(monkeypatch)

    exit_code = harness.run()

    capsys.readouterr()
    assert exit_code == 0
    assert harness.confirmed == [("STARBUCKS", "LIVING_DINING")]


def test_an_ambiguous_answer_writes_nothing_and_asks_again(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness([_row("SOME NEW CAFE")], answers=["housing", "rent"])
    harness.install(monkeypatch)

    exit_code = harness.run()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "could be" in out
    assert harness.confirmed == [("SOME NEW CAFE", "HOUSING_RENT_OR_MORTGAGE")]


def test_a_queue_with_nothing_in_it_says_so(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    harness = _Harness([], answers=[])
    harness.install(monkeypatch)

    exit_code = harness.run()

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "empty" in out


# ---------------------------------------------------------------- refusing to guess


def test_work_refuses_without_a_terminal_and_writes_nothing(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """This command writes a person's decisions; without a person there are none."""
    harness = _Harness([_row("STARBUCKS", suggested="LIVING_DINING")], answers=[""])
    harness.install(monkeypatch, terminal=False)

    exit_code = harness.run()

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "terminal" in out
    assert harness.confirmed == []


def test_list_writes_nothing_and_needs_no_terminal(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [_row("STARBUCKS", amount="-4.50"), _row("STARBUCKS", amount="-5.50")]
    harness = _Harness(rows, answers=[])
    harness.install(monkeypatch, terminal=False)

    exit_code = main(["list", "--user", "a@example.test"])

    out = capsys.readouterr().out
    assert exit_code == 0
    assert "STARBUCKS" in out
    assert "-10.00" in out, "the group's total is what makes the listing worth reading"
    assert "1 merchants" in out
    assert harness.confirmed == []


# ---------------------------------------------------------------- error handling


def test_a_database_error_reports_the_redacted_host_without_the_dsn(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """`config.py` promises the connection string never reaches an error message."""

    def _raise(_email: str) -> TenantScope:
        raise SQLAlchemyError(
            'connection to server failed: password authentication failed for user "hunter2"'
        )

    monkeypatch.setattr(review, "_is_a_terminal", lambda: True)
    monkeypatch.setattr(review, "_open_scope", _raise)

    exit_code = main(["work", "--user", "a@example.test"])

    out = capsys.readouterr().out
    assert exit_code == 1
    assert "hunter2" not in out
    assert "password authentication failed" not in out
