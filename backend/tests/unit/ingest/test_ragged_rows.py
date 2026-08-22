from __future__ import annotations

from pathlib import Path

from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import ColumnMapping
from offerdelta.ingest.preview import preview_csv

HEADER = "Date,Description,Amount\n"


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "statement.csv"
    path.write_text(HEADER + body, encoding="utf-8")
    return path


def test_a_row_with_surplus_cells_is_an_error_not_a_parsed_row(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50,EXTRA1,EXTRA2\n")
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert len(preview.rows) == 0
    assert len(preview.errors) == 1
    assert "extra" in preview.errors[0].reason.lower()


def test_a_short_row_is_an_error_not_a_parsed_row(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-08-17,BLUE BOTTLE\n")
    # An explicit mapping, because this file has only one data row and it is
    # the one under test: with no other row to confirm "Amount" holds money,
    # detection itself would refuse the file before row-shape validation ever
    # runs. That is a property of column detection, not of ragged-row
    # handling, so it is sidestepped here rather than left to obscure what
    # this test is actually checking.
    preview = preview_csv(
        path,
        mapping=ColumnMapping(date="Date", description="Description", amount="Amount"),
        date_order=DateOrder.ISO,
    )

    assert len(preview.rows) == 0
    assert len(preview.errors) == 1
    assert "missing" in preview.errors[0].reason.lower()


def test_no_row_is_lost_when_ragged(tmp_path: Path) -> None:
    """The preview's central promise: parsed + failed == source rows."""
    path = _write(
        tmp_path,
        "2026-08-17,BLUE BOTTLE,-4.50\n2026-08-18,RAGGED,-1.00,EXTRA\n2026-08-19,SHORT\n",
    )
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert preview.total_rows == 3
    assert len(preview.rows) + len(preview.errors) == 3


def test_an_embedded_newline_does_not_desynchronise_the_line_number(tmp_path: Path) -> None:
    """A quoted newline consumes two physical lines; the next row must know."""
    path = _write(
        tmp_path,
        '2026-08-17,"MEMO\nSECOND LINE",-4.50\n2026-08-18,BLUE BOTTLE,-3.00\n',
    )
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert len(preview.rows) == 2
    # header is line 1; first record starts at line 2 and spans lines 2-3
    assert preview.rows[0].line == 2
    # so the second record starts at line 4, not line 3
    assert preview.rows[1].line == 4


def test_line_numbers_are_physical_for_ordinary_files(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "2026-08-17,A,-1.00\n2026-08-18,B,-2.00\n2026-08-19,C,-3.00\n",
    )
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert [row.line for row in preview.rows] == [2, 3, 4]
