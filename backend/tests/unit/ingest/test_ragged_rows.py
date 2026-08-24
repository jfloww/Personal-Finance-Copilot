from __future__ import annotations

from pathlib import Path

from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import AmountSign, ColumnMapping
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


def test_blank_separator_lines_do_not_desynchronise_line_numbers(tmp_path: Path) -> None:
    """`csv.DictReader` silently skips blank rows; every skip must still count.

    Bank exports routinely carry blank separator or trailing lines. Without
    accounting for the skip, each blank line shifts every later row's
    recorded line number backwards by one — the audit trail then points at
    the wrong physical row.
    """
    path = _write(
        tmp_path,
        "\n2026-08-17,BLUE BOTTLE,-4.50\n\n\n2026-08-18,TRANSIT,-2.75\n",
    )
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert len(preview.rows) == 2
    # header is line 1; line 2 is blank; the first record is line 3
    assert preview.rows[0].line == 3
    # lines 4 and 5 are blank; the second record is line 6
    assert preview.rows[1].line == 6


def test_blank_lines_and_an_embedded_newline_compose_correctly(tmp_path: Path) -> None:
    """Both desynchronising forces at once, so neither fix can regress the other."""
    path = _write(
        tmp_path,
        '\n2026-08-17,"MEMO\nSECOND LINE",-4.50\n\n2026-08-19,BLUE BOTTLE,-3.00\n',
    )
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert len(preview.rows) == 2
    # header=1, blank=2, the quoted record starts at line 3 and spans 3-4
    assert preview.rows[0].line == 3
    # line 5 is blank; the next record starts at line 6
    assert preview.rows[1].line == 6


# ------------------------------------------------- rows that are not transactions


def test_a_row_empty_in_every_mapped_column_is_not_a_transaction(tmp_path: Path) -> None:
    """Real exports interleave blank rows among real ones."""
    path = _write(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n,,\n2026-08-18,TRANSIT,-2.75\n")
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert len(preview.rows) == 2
    assert len(preview.errors) == 0
    assert preview.blank_rows == 1


def test_such_rows_are_excluded_from_the_row_count(tmp_path: Path) -> None:
    """parsed + failed == total_rows must still hold, or the invariant lies."""
    path = _write(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n,,\n,,\n")
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert preview.total_rows == 1
    assert len(preview.rows) + len(preview.errors) == preview.total_rows


def test_content_in_an_unmapped_column_does_not_make_it_a_transaction(
    tmp_path: Path,
) -> None:
    """The Chase case: 18 rows carrying a literal "1" in a column nothing reads."""
    path = tmp_path / "chase.csv"
    path.write_text(
        "Date,Description,Amount,Check or Slip #\n2026-08-17,BLUE BOTTLE,-4.50,\n,,,1\n",
        encoding="utf-8",
    )
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert len(preview.rows) == 1
    assert len(preview.errors) == 0
    assert preview.blank_rows == 1


def test_a_partially_filled_row_is_still_an_error(tmp_path: Path) -> None:
    """The safety property: a row with real content must never be skipped."""
    path = _write(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n,BLUE BOTTLE,-9.99\n")
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert len(preview.rows) == 1
    assert len(preview.errors) == 1
    assert preview.blank_rows == 0


def test_an_amount_with_no_date_is_still_an_error(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n,,-9.99\n")
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert len(preview.errors) == 1
    assert preview.blank_rows == 0


def test_the_count_is_reported_not_hidden(tmp_path: Path) -> None:
    """Setting rows aside silently is what this importer exists not to do."""
    path = _write(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n,,\n")
    rendered = preview_csv(path, date_order=DateOrder.ISO).render()

    assert "1 source row(s) held nothing in any mapped column" in rendered


def test_line_numbers_survive_skipped_rows(tmp_path: Path) -> None:
    """A skipped row still consumed a physical line; provenance must not shift."""
    path = _write(tmp_path, "2026-08-17,A,-1.00\n,,\n2026-08-19,B,-2.00\n")
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert [row.line for row in preview.rows] == [2, 4]


# ------------------------------------------------- the preview names what it uses


def test_a_supplied_mapping_is_named_in_the_preview(tmp_path: Path) -> None:
    """Showing only the detected guess would name one column while reading another."""
    path = tmp_path / "two-dates.csv"
    path.write_text(
        "Transaction Date,Posted Date,Description,Amount\n"
        "2025-12-31,2026-01-02,BLUE BOTTLE,-4.50\n",
        encoding="utf-8",
    )
    override = ColumnMapping(date="Posted Date", description="Description", amount="Amount")
    rendered = preview_csv(path, mapping=override, date_order=DateOrder.ISO).render()

    assert "supplied mapping overrides detection" in rendered
    assert "Posted Date" in rendered


def test_no_override_notice_when_the_mapping_matches_detection(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")
    rendered = preview_csv(path, date_order=DateOrder.ISO).render()

    assert "supplied mapping overrides detection" not in rendered


def test_the_override_changes_what_is_parsed(tmp_path: Path) -> None:
    """The notice must reflect reality, not just print a claim."""
    path = tmp_path / "two-dates.csv"
    path.write_text(
        "Transaction Date,Posted Date,Description,Amount\n"
        "2025-12-31,2026-01-02,BLUE BOTTLE,-4.50\n",
        encoding="utf-8",
    )
    override = ColumnMapping(date="Posted Date", description="Description", amount="Amount")
    preview = preview_csv(path, mapping=override, date_order=DateOrder.ISO)

    assert preview.rows[0].posted_on.isoformat() == "2026-01-02"


# ------------------------------------------------- sign convention


def test_a_signed_column_is_read_money_out_negative_by_default(tmp_path: Path) -> None:
    path = _write(tmp_path, "2026-08-17,BLUE BOTTLE,-4.50\n")
    preview = preview_csv(path, date_order=DateOrder.ISO)

    assert preview.rows[0].amount.amount < 0


def test_outflow_positive_inverts_the_column(tmp_path: Path) -> None:
    """Amex writes a charge positive; read at face value it becomes income."""
    path = _write(tmp_path, "2026-08-17,BLUE BOTTLE,4.50\n2026-08-18,PAYMENT,-100.00\n")
    preview = preview_csv(path, date_order=DateOrder.ISO, amount_sign=AmountSign.OUTFLOW_POSITIVE)

    assert preview.rows[0].amount.amount < 0  # the charge is money out
    assert preview.rows[1].amount.amount > 0  # the payment is money in


def test_the_sign_applies_to_a_detected_mapping(tmp_path: Path) -> None:
    """Headers detect fine for Amex; only the convention needs stating."""
    path = _write(tmp_path, "2026-08-17,BLUE BOTTLE,4.50\n")
    preview = preview_csv(path, date_order=DateOrder.ISO, amount_sign=AmountSign.OUTFLOW_POSITIVE)

    assert preview.mapping is not None
    assert preview.mapping.amount_sign is AmountSign.OUTFLOW_POSITIVE
    assert preview.rows[0].amount.amount < 0


def test_a_mostly_positive_signed_file_is_flagged(tmp_path: Path) -> None:
    """The check that would have caught 245 inverted Amex rows."""
    body = "".join(f"2026-08-{d:02d},MERCHANT {d},{d}.00\n" for d in range(1, 11))
    path = _write(tmp_path, body)
    rendered = preview_csv(path, date_order=DateOrder.ISO).render()

    assert "read as money IN" in rendered
    assert "outflow-positive" in rendered


def test_a_mostly_negative_file_is_not_flagged(tmp_path: Path) -> None:
    body = "".join(f"2026-08-{d:02d},MERCHANT {d},-{d}.00\n" for d in range(1, 11))
    path = _write(tmp_path, body)

    assert "read as money IN" not in preview_csv(path, date_order=DateOrder.ISO).render()


def test_no_warning_once_the_convention_is_declared(tmp_path: Path) -> None:
    body = "".join(f"2026-08-{d:02d},MERCHANT {d},{d}.00\n" for d in range(1, 11))
    path = _write(tmp_path, body)
    rendered = preview_csv(
        path, date_order=DateOrder.ISO, amount_sign=AmountSign.OUTFLOW_POSITIVE
    ).render()

    assert "read as money IN" not in rendered


def test_split_debit_credit_ignores_the_sign_flag(tmp_path: Path) -> None:
    """Split columns carry magnitudes; their direction is structural."""
    path = tmp_path / "split.csv"
    path.write_text(
        "Date,Description,Debit,Credit\n2026-08-17,BLUE BOTTLE,4.50,\n", encoding="utf-8"
    )
    preview = preview_csv(path, date_order=DateOrder.ISO, amount_sign=AmountSign.OUTFLOW_POSITIVE)

    assert preview.rows[0].amount.amount < 0
