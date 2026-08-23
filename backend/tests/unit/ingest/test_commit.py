from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import pytest

from offerdelta.domain.common.errors import ValidationError
from offerdelta.ingest.commit import ImportMode, ImportWindow, plan_records
from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import ColumnMapping
from offerdelta.ingest.preview import ImportPreview, preview_csv

ACCOUNT = uuid.UUID("11111111-1111-1111-1111-111111111111")
AUGUST = ImportWindow(start=date(2026, 8, 1), end=date(2026, 8, 31))
HEADER = "Date,Description,Amount\n"


def _preview(tmp_path: Path, body: str) -> ImportPreview:
    path = tmp_path / "aug.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return preview_csv(path, date_order=DateOrder.ISO)


def test_snapshot_numbers_repeats_within_the_file(tmp_path: Path) -> None:
    preview = _preview(
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-4.50\n2026-08-17,BLUE BOTTLE,-4.50\n",
    )
    records = plan_records(preview, account_id=ACCOUNT, mode=ImportMode.SNAPSHOT, window=AUGUST)
    assert [r.occurrence for r in records] == [1, 2]


def test_snapshot_requires_a_window(tmp_path: Path) -> None:
    preview = _preview(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")
    with pytest.raises(ValidationError, match="window"):
        plan_records(preview, account_id=ACCOUNT, mode=ImportMode.SNAPSHOT, window=None)


def test_a_row_outside_the_window_is_refused(tmp_path: Path) -> None:
    """A file wider than the declared range is not the snapshot you said."""
    preview = _preview(
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-4.50\n2026-09-02,LATE CHARGE,-9.00\n",
    )
    with pytest.raises(ValidationError, match="outside the declared window"):
        plan_records(preview, account_id=ACCOUNT, mode=ImportMode.SNAPSHOT, window=AUGUST)


def test_the_offending_line_is_named(tmp_path: Path) -> None:
    preview = _preview(
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-4.50\n2026-09-02,LATE CHARGE,-9.00\n",
    )
    with pytest.raises(ValidationError, match="line 3"):
        plan_records(preview, account_id=ACCOUNT, mode=ImportMode.SNAPSHOT, window=AUGUST)


def test_incremental_without_an_external_id_is_refused(tmp_path: Path) -> None:
    """The ambiguity is unresolvable without a stable id, so refuse it."""
    preview = _preview(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")
    with pytest.raises(ValidationError, match="transaction id"):
        plan_records(preview, account_id=ACCOUNT, mode=ImportMode.INCREMENTAL, window=None)


def _preview_with_ids(tmp_path: Path, body: str) -> ImportPreview:
    """A preview whose mapping names an id column.

    Auto-detection never guesses `external_id` — there is no alias list for
    it, unlike date/description/amount — so exercising incremental mode needs
    an explicit `ColumnMapping`.
    """
    header = "Date,Description,Amount,TransactionId\n"
    path = tmp_path / "aug.csv"
    path.write_text(header + body, encoding="utf-8")
    mapping = ColumnMapping(
        date="Date", description="Description", amount="Amount", external_id="TransactionId"
    )
    return preview_csv(path, mapping=mapping, date_order=DateOrder.ISO)


def test_incremental_refuses_a_row_with_a_blank_id_cell(tmp_path: Path) -> None:
    """A sparse id column is a realistic export shape, not an edge case.

    The column-level check above only proves the mapping *names* an id
    column; it says nothing about whether every row actually carries one. A
    row with a blank cell there has exactly the ambiguity incremental mode
    exists to refuse, so it must be refused too, not silently downgraded to
    fingerprint-based matching.
    """
    preview = _preview_with_ids(
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-4.50,TXN-1\n2026-08-18,TRANSIT,-2.75,\n",
    )
    with pytest.raises(ValidationError, match="line 3"):
        plan_records(preview, account_id=ACCOUNT, mode=ImportMode.INCREMENTAL, window=None)


def test_incremental_with_every_row_id_populated_succeeds(tmp_path: Path) -> None:
    preview = _preview_with_ids(
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-4.50,TXN-1\n2026-08-18,TRANSIT,-2.75,TXN-2\n",
    )
    records = plan_records(preview, account_id=ACCOUNT, mode=ImportMode.INCREMENTAL, window=None)
    assert [r.external_id for r in records] == ["TXN-1", "TXN-2"]
    assert all(r.external_id is not None for r in records)


def test_records_carry_provenance(tmp_path: Path) -> None:
    preview = _preview(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")
    records = plan_records(preview, account_id=ACCOUNT, mode=ImportMode.SNAPSHOT, window=AUGUST)
    assert records[0].provenance is not None
    assert records[0].provenance.source_file == "aug.csv"
    assert records[0].provenance.source_line == 2


def test_a_preview_with_errors_is_refused(tmp_path: Path) -> None:
    # A lone "not-a-date" row can't demonstrate this: with only one sample
    # value, the date column fails detection entirely and the preview comes
    # back with `mapping=None`, not a resolved mapping plus a row error. Three
    # good dates plus one bad amount keeps detection confident (the header
    # column still parses as dates 100% of the time) while still producing
    # exactly the row-level error this test is about.
    preview = _preview(
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-4.50\n"
        "2026-08-18,NETFLIX,OOPS\n"
        "2026-08-19,SHELL,-52.10\n"
        "2026-08-20,COSTCO,-120.00\n",
    )
    assert preview.mapping is not None
    assert preview.errors
    with pytest.raises(ValidationError, match="have errors"):
        plan_records(preview, account_id=ACCOUNT, mode=ImportMode.SNAPSHOT, window=AUGUST)


def test_amount_grouping_agrees_with_the_stored_fingerprint(tmp_path: Path) -> None:
    """0.125 and 0.13 must never collide the way `-4.50`/`-4.5` once did.

    `compute_fingerprint` quantises through CURRENCY_DISPLAY, which rounds
    ROUND_HALF_UP: both 0.125 and 0.13 become 0.13, so both rows share one
    fingerprint and must be numbered occurrences 1 and 2 of it. Grouping with
    plain `f"{amount:.2f}"` instead rounds ROUND_HALF_EVEN, where 0.125 becomes
    0.12 — a different key from 0.13's 0.13 — so both rows would be numbered
    occurrence 1 of two different keys while sharing one real fingerprint.
    `add_many` would then try to insert (fingerprint, 1) twice and the second
    write would trip the unique constraint, surfacing as "conflicted with
    another import; retry" — a message retrying can never fix, since the same
    mismatched plan is rebuilt every time.
    """
    preview = _preview(
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-0.125\n2026-08-17,BLUE BOTTLE,-0.13\n",
    )
    records = plan_records(preview, account_id=ACCOUNT, mode=ImportMode.SNAPSHOT, window=AUGUST)
    assert [r.occurrence for r in records] == [1, 2]


def test_an_empty_preview_is_refused(tmp_path: Path) -> None:
    # A header-only file leaves detection with an empty sample, and
    # `looks_like_dates([])` is False, so auto-detection alone would come back
    # with `mapping=None` rather than a resolved mapping over zero rows. An
    # explicit mapping bypasses detection so the "no parsed rows" branch in
    # `plan_records` is the one that actually fires.
    path = tmp_path / "aug.csv"
    path.write_text(HEADER, encoding="utf-8")
    mapping = ColumnMapping(date="Date", description="Description", amount="Amount")
    preview = preview_csv(path, mapping=mapping, date_order=DateOrder.ISO)
    assert preview.mapping is not None
    assert not preview.rows
    with pytest.raises(ValidationError, match="no parsed rows"):
        plan_records(preview, account_id=ACCOUNT, mode=ImportMode.SNAPSHOT, window=AUGUST)
