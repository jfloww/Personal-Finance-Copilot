"""The annotator's rules, and the loop driven end to end.

The interactive loop takes its input function as an argument, so these tests
feed it a scripted list of keystrokes and assert on the file it writes. Nothing
about the labelling logic is mocked - only the keyboard.
"""

from __future__ import annotations

import csv
from collections.abc import Callable
from pathlib import Path

import pytest

from annotate import (
    CODES,
    Row,
    Session,
    annotate,
    build_memory,
    load_bank_csv,
    load_dataset,
    merge,
    resolve,
    save,
)
from offerdelta.domain.common.errors import ValidationError
from offerdelta.evaluation.csv_loader import REQUIRED_COLUMNS
from offerdelta.evaluation.labels import ABSTAIN, LABEL_SPACE
from offerdelta.ingest.dates import DateOrder

HEADER = "Date,Description,Amount\n"


def _statement(tmp_path: Path, body: str, name: str = "aug.csv") -> Path:
    path = tmp_path / name
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def _scripted(answers: list[str]) -> Callable[[str], str]:
    """A keyboard that types exactly these things, then gives up."""
    remaining = list(answers)

    def ask(_prompt: str) -> str:
        if not remaining:
            raise EOFError
        return remaining.pop(0)

    return ask


def _row(description: str = "BLUE BOTTLE", merchant: str = "BLUE BOTTLE") -> Row:
    return Row(
        transaction_id="t1",
        description=description,
        amount="-4.50",
        posted_on="2026-08-17",
        normalised_merchant=merchant,
    )


# ---------------------------------------------------------------- the code map


def test_every_label_has_exactly_one_code() -> None:
    """A label with no code cannot be chosen without typing it in full."""
    assert set(CODES.values()) == LABEL_SPACE
    assert len(CODES) == len(LABEL_SPACE)


def test_codes_are_unique() -> None:
    assert len(set(CODES)) == len(CODES)


def test_the_non_spending_labels_are_reachable() -> None:
    """A real statement is full of transfers; TRANSFER must be one keystroke."""
    for label in ("INCOME", "TRANSFER", "REFUND"):
        assert label in CODES.values()


def test_abstention_is_reachable() -> None:
    assert CODES["u"] == ABSTAIN


# ---------------------------------------------------------------- resolve


def test_a_code_resolves() -> None:
    assert resolve("l2") == CODES["l2"]


def test_a_code_is_case_insensitive() -> None:
    assert resolve("L2") == CODES["l2"]


def test_a_full_label_resolves() -> None:
    assert resolve("LIVING_GROCERY") == "LIVING_GROCERY"


def test_a_unique_prefix_resolves() -> None:
    assert resolve("groc") == "LIVING_GROCERY"


def test_an_ambiguous_fragment_resolves_to_nothing() -> None:
    """HOUSING_ matches five labels. Guessing one would be a silent wrong label."""
    assert resolve("HOUSING") is None


def test_nonsense_resolves_to_nothing() -> None:
    assert resolve("zzzz") is None
    assert resolve("") is None


# ---------------------------------------------------------------- memory


def test_memory_is_built_from_the_annotators_own_column() -> None:
    rows = [
        Row("t1", "A", "-1.00", "2026-08-17", "COFFEE", annotator_a_label="LIVING_DINING"),
        Row("t2", "B", "-2.00", "2026-08-17", "TRANSIT", annotator_b_label="COMMUTE_TRANSIT_FARE"),
    ]
    assert build_memory(rows, "a") == {"COFFEE": "LIVING_DINING"}
    assert build_memory(rows, "b") == {"TRANSIT": "COMMUTE_TRANSIT_FARE"}


def test_b_never_inherits_a_suggestions() -> None:
    """The dataset README requires B to label without seeing A. Enforce it here."""
    rows = [Row("t1", "A", "-1.00", "2026-08-17", "COFFEE", annotator_a_label="LIVING_DINING")]
    assert build_memory(rows, "b") == {}


# ---------------------------------------------------------------- the loop


def test_a_label_is_recorded(tmp_path: Path) -> None:
    session = Session(rows=[_row()], annotator="a", dataset=tmp_path / "d.csv")
    assert annotate(session, _scripted(["l2"])) == "finished"
    assert session.rows[0].annotator_a_label == CODES["l2"]


def test_enter_accepts_the_remembered_label(tmp_path: Path) -> None:
    """The whole point: the fortieth coffee is one keystroke."""
    rows = [_row(), Row("t2", "BLUE BOTTLE 2", "-4.50", "2026-08-18", "BLUE BOTTLE")]
    session = Session(rows=rows, annotator="a", dataset=tmp_path / "d.csv")

    assert annotate(session, _scripted(["l2", ""])) == "finished"
    assert rows[0].annotator_a_label == rows[1].annotator_a_label == CODES["l2"]


def test_enter_with_no_suggestion_reprompts(tmp_path: Path) -> None:
    """An empty answer must never become a label by accident."""
    session = Session(rows=[_row()], annotator="a", dataset=tmp_path / "d.csv")
    assert annotate(session, _scripted(["", "l2"])) == "finished"
    assert session.rows[0].annotator_a_label == CODES["l2"]


def test_an_unrecognised_entry_reprompts(tmp_path: Path) -> None:
    session = Session(rows=[_row()], annotator="a", dataset=tmp_path / "d.csv")
    assert annotate(session, _scripted(["zzzz", "l2"])) == "finished"
    assert session.rows[0].annotator_a_label == CODES["l2"]


def test_skip_leaves_the_row_unlabelled(tmp_path: Path) -> None:
    session = Session(rows=[_row()], annotator="a", dataset=tmp_path / "d.csv")
    assert annotate(session, _scripted(["s"])) == "finished"
    assert session.rows[0].annotator_a_label == ""


def test_quit_stops_without_labelling(tmp_path: Path) -> None:
    session = Session(rows=[_row()], annotator="a", dataset=tmp_path / "d.csv")
    assert annotate(session, _scripted(["q"])) == "quit"
    assert session.rows[0].annotator_a_label == ""


def test_undo_clears_the_previous_label(tmp_path: Path) -> None:
    rows = [_row(), Row("t2", "TRANSIT", "-2.75", "2026-08-18", "TRANSIT")]
    session = Session(rows=rows, annotator="a", dataset=tmp_path / "d.csv")

    assert annotate(session, _scripted(["l2", "u", "l1", "c1"])) == "finished"
    assert rows[0].annotator_a_label == CODES["l1"]
    assert rows[1].annotator_a_label == CODES["c1"]


# ---------------------------------------------------------------- ambiguity


def test_ambiguity_records_both_labels_and_the_note(tmp_path: Path) -> None:
    session = Session(rows=[_row()], annotator="a", dataset=tmp_path / "d.csv")
    answers = ["?", "l1 l2", "a corner shop that is half cafe", "l2"]

    assert annotate(session, _scripted(answers)) == "finished"
    row = session.rows[0]
    assert row.acceptable_labels == "|".join(sorted([CODES["l1"], CODES["l2"]]))
    assert row.ambiguity_note == "a corner shop that is half cafe"
    assert row.annotator_a_label == CODES["l2"]


def test_ambiguity_without_a_note_is_cancelled(tmp_path: Path) -> None:
    """The README requires the note. A row cannot be ambiguous without a reason."""
    session = Session(rows=[_row()], annotator="a", dataset=tmp_path / "d.csv")

    assert annotate(session, _scripted(["?", "l1 l2", "  ", "l2"])) == "finished"
    assert session.rows[0].acceptable_labels == ""
    assert session.rows[0].annotator_a_label == CODES["l2"]


def test_one_acceptable_label_is_not_ambiguity(tmp_path: Path) -> None:
    session = Session(rows=[_row()], annotator="a", dataset=tmp_path / "d.csv")

    assert annotate(session, _scripted(["?", "l2", "l2"])) == "finished"
    assert session.rows[0].acceptable_labels == ""


# ---------------------------------------------------------------- files


def test_loading_a_statement_uses_the_importers_parser(tmp_path: Path) -> None:
    path = _statement(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")
    rows = load_bank_csv(path, mapping=None, date_order=DateOrder.ISO)

    assert len(rows) == 1
    assert rows[0].description == "BLUE BOTTLE"
    assert rows[0].amount == "-4.50"
    assert rows[0].posted_on == "2026-08-17"


def test_a_statement_the_importer_would_refuse_is_refused_here(tmp_path: Path) -> None:
    """Annotating rows the importer will later reject wastes the annotation."""
    path = _statement(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50,EXTRA\n")
    with pytest.raises(ValidationError, match="could not be parsed"):
        load_bank_csv(path, mapping=None, date_order=DateOrder.ISO)


def test_transaction_ids_are_stable_across_runs(tmp_path: Path) -> None:
    path = _statement(tmp_path, "2026-08-17,A,-1.00\n2026-08-18,B,-2.00\n")
    first = load_bank_csv(path, mapping=None, date_order=DateOrder.ISO)
    second = load_bank_csv(path, mapping=None, date_order=DateOrder.ISO)

    assert [r.transaction_id for r in first] == [r.transaction_id for r in second]


def test_merge_keeps_existing_labels() -> None:
    existing = [Row("aug-0002", "A", "-1.00", "2026-08-17", "A", annotator_a_label="LIVING_OTHER")]
    incoming = [
        Row("aug-0002", "A", "-1.00", "2026-08-17", "A"),
        Row("aug-0003", "B", "-2", "", "B"),
    ]

    merged = merge(existing, incoming)
    assert len(merged) == 2
    assert merged[0].annotator_a_label == "LIVING_OTHER"


def test_a_saved_file_round_trips(tmp_path: Path) -> None:
    dataset = tmp_path / "d.csv"
    rows = [_row()]
    rows[0].annotator_a_label = "LIVING_DINING"
    save(rows, dataset)

    reloaded = load_dataset(dataset)
    assert reloaded[0].annotator_a_label == "LIVING_DINING"
    assert reloaded[0].transaction_id == "t1"


def test_a_saved_file_has_the_columns_the_loader_requires(tmp_path: Path) -> None:
    """The dataset is only useful if validate_dataset.py accepts it."""
    dataset = tmp_path / "d.csv"
    rows = [_row()]
    rows[0].annotator_a_label = "LIVING_DINING"
    save(rows, dataset)

    with dataset.open(encoding="utf-8-sig", newline="") as handle:
        headers = next(csv.reader(handle))
    assert set(REQUIRED_COLUMNS) <= set(headers)


def test_resuming_only_offers_unlabelled_rows(tmp_path: Path) -> None:
    rows = [
        Row("t1", "A", "-1.00", "2026-08-17", "A", annotator_a_label="LIVING_OTHER"),
        Row("t2", "B", "-2.00", "2026-08-18", "B"),
    ]
    session = Session(rows=rows, annotator="a", dataset=tmp_path / "d.csv")

    assert len(session.pending) == 1
    assert session.pending[0].transaction_id == "t2"
    assert session.done_count == 1


# ---------------------------------------------------------------- adjudication


def _disputed() -> Row:
    return Row(
        "t1",
        "CORNER DELI",
        "-11.20",
        "2026-08-22",
        "CORNER DELI",
        annotator_a_label="LIVING_DINING",
        annotator_b_label="LIVING_GROCERY",
    )


def _agreed() -> Row:
    return Row(
        "t2",
        "NETFLIX",
        "-15.99",
        "2026-08-23",
        "NETFLIX",
        annotator_a_label="LIVING_SUBSCRIPTIONS",
        annotator_b_label="LIVING_SUBSCRIPTIONS",
    )


def test_a_row_is_disputed_only_when_both_labelled_and_differ() -> None:
    assert _disputed().disputed is True
    assert _agreed().disputed is False
    # One annotator has not reached it yet: not a dispute, just unfinished.
    assert Row("t3", "A", "-1", "", "A", annotator_a_label="LIVING_OTHER").disputed is False


def test_adjudication_only_visits_disputed_rows(tmp_path: Path) -> None:
    session = Session(rows=[_agreed(), _disputed()], annotator="final", dataset=tmp_path / "d.csv")
    assert [row.transaction_id for row in session.pending] == ["t1"]


def test_adjudication_records_the_final_label(tmp_path: Path) -> None:
    """Without this the dataset silently adopts A and the disagreement vanishes."""
    rows = [_agreed(), _disputed()]
    session = Session(rows=rows, annotator="final", dataset=tmp_path / "d.csv")

    assert annotate(session, _scripted(["l1"])) == "finished"
    assert rows[1].final_label == CODES["l1"]
    assert rows[0].final_label == ""


def test_an_agreed_row_never_gets_a_final_label(tmp_path: Path) -> None:
    session = Session(rows=[_agreed()], annotator="final", dataset=tmp_path / "d.csv")
    assert annotate(session, _scripted([])) == "finished"
    assert session.rows[0].final_label == ""


def test_adjudication_is_idempotent(tmp_path: Path) -> None:
    row = _disputed()
    row.final_label = "LIVING_GROCERY"
    session = Session(rows=[row], annotator="final", dataset=tmp_path / "d.csv")
    assert session.pending == []


def test_a_dead_terminal_stops_instead_of_spinning(tmp_path: Path) -> None:
    """isatty() reports a terminal on this platform even at EOF, so the
    not-a-terminal guard passes and every prompt returns "". Without a limit
    the loop reprompts forever and only Ctrl+C ends it."""
    rows = [_row(), Row("t2", "OTHER", "-1.00", "2026-08-18", "OTHER")]
    session = Session(rows=rows, annotator="a", dataset=tmp_path / "d.csv")

    calls = 0

    def dead(_prompt: str) -> str:
        nonlocal calls
        calls += 1
        if calls > 50:
            raise AssertionError("still spinning")
        return ""

    assert annotate(session, dead) == "no-input"
    assert calls <= 5
    assert rows[0].annotator_a_label == ""


def test_a_typo_does_not_end_the_session(tmp_path: Path) -> None:
    """Three empties in a row means gone; two mistakes means a person."""
    session = Session(rows=[_row()], annotator="a", dataset=tmp_path / "d.csv")

    assert annotate(session, _scripted(["", "", "l2"])) == "finished"
    assert session.rows[0].annotator_a_label == CODES["l2"]


def test_a_real_answer_resets_the_streak(tmp_path: Path) -> None:
    rows = [_row(), Row("t2", "OTHER", "-1.00", "2026-08-18", "OTHER")]
    session = Session(rows=rows, annotator="a", dataset=tmp_path / "d.csv")

    assert annotate(session, _scripted(["", "", "l2", "", "", "c1"])) == "finished"
    assert rows[1].annotator_a_label == CODES["c1"]
