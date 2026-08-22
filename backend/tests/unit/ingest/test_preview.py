"""Importing a bank export: detection, normalisation, and the preview.

Written against the shapes real exports actually take — signed amount columns,
split debit/credit columns, European date order, headers nobody agreed on — and
against the ways they go wrong.

Fixtures are triple-quoted rather than inline escapes, so what a test feeds the
importer looks like what a bank actually writes. They also carry several rows:
detection confirms a header guess against the values, and a one-row file cannot
demonstrate anything about a column.

The rule the tests keep returning to: nothing is silently dropped. A row that
cannot be parsed is reported with its line number, and the counts have to add up
to the file's length.
"""

from datetime import date
from decimal import Decimal
from pathlib import Path

import pytest

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.ingest.dates import DateOrder
from offerdelta.ingest.mapping import ColumnMapping, detect_mapping
from offerdelta.ingest.preview import preview_csv


def _write(tmp_path: Path, text: str, name: str = "export.csv") -> Path:
    path = tmp_path / name
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


CHASE = """
Transaction Date,Description,Amount
08/17/2026,SQ *BLUE BOTTLE 4412,-4.50
08/18/2026,NETFLIX.COM,-15.99
08/20/2026,ACME CORP DIRECT DEP,3210.44
"""

MONZO = """
Date,Name,Money Out,Money In
17/08/2026,Blue Bottle,4.50,
18/08/2026,Netflix,15.99,
20/08/2026,Acme Corp,,3210.44
"""

#: A good file with one unparseable amount, so detection still recognises the
#: column and the bad row surfaces as a row error rather than a column problem.
ONE_BAD_AMOUNT = """
Date,Description,Amount
2026-08-17,BLUE BOTTLE,-4.50
2026-08-18,NETFLIX,OOPS
2026-08-19,SHELL,-52.10
2026-08-20,COSTCO,-120.00
"""

BOTH_DEBIT_AND_CREDIT = """
Date,Description,Debit,Credit
2026-08-17,BLUE BOTTLE,4.50,
2026-08-18,ACME,,3210.44
2026-08-19,CONFUSED,4.50,10.00
"""

NEITHER_DEBIT_NOR_CREDIT = """
Date,Description,Debit,Credit
2026-08-17,BLUE BOTTLE,4.50,
2026-08-18,ACME,,3210.44
2026-08-19,MYSTERY,,
"""

SIGNED_DEBIT = """
Date,Description,Debit,Credit
2026-08-17,BLUE BOTTLE,-4.50,
2026-08-18,NETFLIX,-15.99,
2026-08-19,ACME,,3210.44
"""

AMBIGUOUS_DATES = """
Date,Description,Amount
03/04/2026,BLUE BOTTLE,-4.50
05/06/2026,NETFLIX,-15.99
"""

MIXED_FAILURES = """
Date,Description,Amount
2026-08-17,BLUE BOTTLE,-4.50
PENDING,NETFLIX,-15.99
2026-08-19,,-1.00
2026-08-20,SHELL,-52.10
"""

TWO_IDENTICAL = """
Date,Description,Amount
2026-08-17,BLUE BOTTLE,-4.50
2026-08-17,BLUE BOTTLE,-4.50
2026-08-18,NETFLIX,-15.99
"""


# --- Detection -------------------------------------------------------------


def test_a_signed_amount_export_is_detected(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, CHASE))
    assert preview.mapping is not None
    assert preview.mapping.amount == "Amount"
    assert preview.mapping.uses_split_columns is False


def test_a_split_debit_credit_export_is_detected(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, MONZO))
    assert preview.mapping is not None
    assert preview.mapping.debit == "Money Out"
    assert preview.mapping.credit == "Money In"
    assert preview.mapping.uses_split_columns is True


def test_header_aliases_are_recognised(tmp_path: Path) -> None:
    odd = """
Posting Date,Narrative,Value
2026-08-17,BLUE BOTTLE,-4.50
2026-08-18,NETFLIX,-15.99
"""
    preview = preview_csv(_write(tmp_path, odd))
    assert preview.mapping is not None
    assert preview.mapping.date == "Posting Date"
    assert preview.mapping.description == "Narrative"


def test_a_header_the_values_contradict_is_not_trusted(tmp_path: Path) -> None:
    # A column called "date" full of merchant names is not a date column, and
    # accepting it on its name would corrupt every row.
    bad = """
Date,Amount
BLUE BOTTLE,-4.50
NETFLIX,-15.99
"""
    preview = preview_csv(_write(tmp_path, bad))
    assert preview.mapping is None
    assert any("date" in problem for problem in preview.detection.problems)


def test_a_reference_number_column_is_not_mistaken_for_an_amount() -> None:
    # Long unsigned digit runs parse as numbers and are not money.
    detection = detect_mapping(
        ["Date", "Description", "Amount"],
        {
            "Date": ["2026-08-17"],
            "Description": ["BLUE BOTTLE"],
            "Amount": ["100200300400", "100200300401"],
        },
    )
    assert detection.mapping is None


def test_an_unnamed_date_column_is_found_by_its_values(tmp_path: Path) -> None:
    odd = """
When,Description,Amount
2026-08-17,BLUE BOTTLE,-4.50
2026-08-18,NETFLIX,-15.99
"""
    preview = preview_csv(_write(tmp_path, odd))
    assert preview.mapping is not None
    assert preview.mapping.date == "When"


def test_a_choice_between_plausible_columns_is_reported_as_resolved() -> None:
    # Two date columns is the normal case, not a failure. The importer picks by
    # preference and says so, rather than refusing a file every bank produces.
    detection = detect_mapping(
        ["Date", "Posted Date", "Description", "Amount"],
        {
            "Date": ["2026-08-17", "2026-08-18"],
            "Posted Date": ["2026-08-18", "2026-08-19"],
            "Description": ["BLUE BOTTLE", "NETFLIX"],
            "Amount": ["-4.50", "-15.99"],
        },
    )
    assert detection.mapping is not None
    assert detection.resolved


def test_a_header_contradicted_by_its_values_is_still_uncertain() -> None:
    # Preference resolves which of several plausible columns to use. It does
    # not paper over a column whose contents disagree with its name.
    detection = detect_mapping(
        ["Date", "Description", "Amount"],
        {
            "Date": ["2026-08-17", "2026-08-18"],
            "Description": ["BLUE BOTTLE", "NETFLIX"],
            "Amount": ["BLUE BOTTLE", "NETFLIX"],
        },
    )
    assert detection.uncertain


def test_a_file_with_no_money_column_is_refused(tmp_path: Path) -> None:
    bad = """
Date,Description
2026-08-17,BLUE BOTTLE
2026-08-18,NETFLIX
"""
    preview = preview_csv(_write(tmp_path, bad))
    assert preview.mapping is None
    assert any("amount" in problem for problem in preview.detection.problems)


# --- Explicit mapping ------------------------------------------------------


def test_an_explicit_mapping_overrides_detection(tmp_path: Path) -> None:
    odd = """
col_a,col_b,col_c
2026-08-17,BLUE BOTTLE,-4.50
2026-08-18,NETFLIX,-15.99
"""
    preview = preview_csv(
        _write(tmp_path, odd),
        mapping=ColumnMapping(date="col_a", description="col_b", amount="col_c"),
    )
    assert preview.rows[0].amount == Money.parse("-4.50")


def test_a_mapping_naming_a_missing_column_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="not in"):
        preview_csv(
            _write(tmp_path, CHASE),
            mapping=ColumnMapping(date="Nope", description="Description", amount="Amount"),
        )


def test_a_mapping_with_both_shapes_is_refused() -> None:
    # Reading a signed amount and a debit column would double every row.
    with pytest.raises(ValidationError, match="choose one"):
        ColumnMapping(date="d", description="x", amount="a", debit="b")


def test_a_mapping_with_no_money_at_all_is_refused() -> None:
    with pytest.raises(ValidationError, match="no money in the file"):
        ColumnMapping(date="d", description="x")


# --- Normalisation ---------------------------------------------------------


def test_a_signed_amount_is_kept_as_is(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, CHASE))
    assert preview.rows[0].amount == Money.parse("-4.50")
    assert preview.rows[2].amount == Money.parse("3210.44")


def test_a_debit_becomes_negative(tmp_path: Path) -> None:
    # Split columns hold magnitudes; money out is negative everywhere else in
    # this codebase, so the debit is negated on the way in.
    preview = preview_csv(_write(tmp_path, MONZO))
    assert preview.rows[0].amount == Money.parse("-4.50")


def test_a_credit_stays_positive(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, MONZO))
    assert preview.rows[2].amount == Money.parse("3210.44")


def test_an_already_signed_debit_is_not_negated_twice(tmp_path: Path) -> None:
    # Some exports sign the debit column. Negating a negative would turn a
    # payment into income.
    preview = preview_csv(_write(tmp_path, SIGNED_DEBIT))
    assert preview.rows[0].amount == Money.parse("-4.50")


def test_us_and_european_orders_both_resolve_to_the_same_day(tmp_path: Path) -> None:
    # The same date written under two conventions, each detected from its own
    # column. Getting this wrong would shift a transaction by months.
    us = preview_csv(_write(tmp_path, CHASE))
    eu = preview_csv(_write(tmp_path, MONZO, name="eu.csv"))
    assert us.rows[0].posted_on == date(2026, 8, 17)
    assert eu.rows[0].posted_on == date(2026, 8, 17)


def test_an_ambiguous_date_column_parses_nothing_until_told(tmp_path: Path) -> None:
    # Every value reads both ways. Guessing would be invisible downstream.
    preview = preview_csv(_write(tmp_path, AMBIGUOUS_DATES))
    assert preview.date_order is DateOrder.AMBIGUOUS
    assert len(preview.errors) == 2
    assert not preview.rows


def test_an_explicit_date_order_resolves_it(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, AMBIGUOUS_DATES), date_order=DateOrder.DAY_FIRST)
    assert preview.rows[0].posted_on == date(2026, 4, 3)


def test_the_merchant_is_normalised(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, CHASE))
    assert preview.rows[0].normalised_merchant == "BLUE BOTTLE"


# --- Nothing is dropped ----------------------------------------------------


def test_parsed_and_failed_account_for_every_row(tmp_path: Path) -> None:
    # The invariant. An importer that loses rows cannot be trusted with money.
    preview = preview_csv(_write(tmp_path, MIXED_FAILURES))
    assert preview.total_rows == 4
    assert len(preview.rows) + len(preview.errors) == 4


def test_a_bad_row_is_reported_with_its_line_number(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, ONE_BAD_AMOUNT))
    assert [error.line for error in preview.errors] == [3]


def test_a_bad_row_keeps_its_original_values(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, ONE_BAD_AMOUNT))
    assert preview.errors[0].raw["Amount"] == "OOPS"


def test_a_row_with_both_debit_and_credit_is_an_error(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, BOTH_DEBIT_AND_CREDIT))
    assert any("cannot be money out and money in" in error.reason for error in preview.errors)


def test_a_row_with_neither_debit_nor_credit_is_an_error(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, NEITHER_DEBIT_NOR_CREDIT))
    assert any("neither debit nor credit" in error.reason for error in preview.errors)


# --- Traceability ----------------------------------------------------------


def test_every_parsed_row_keeps_its_original_cells(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, CHASE))
    assert preview.rows[0].raw["Description"] == "SQ *BLUE BOTTLE 4412"
    assert preview.rows[0].raw["Amount"] == "-4.50"


def test_the_original_is_kept_even_when_normalisation_changed_it(
    tmp_path: Path,
) -> None:
    preview = preview_csv(_write(tmp_path, CHASE))
    row = preview.rows[0]
    assert row.normalised_merchant != row.raw["Description"]


# --- Duplicate detection ----------------------------------------------------


def test_duplicate_grouping_ignores_row_position(tmp_path: Path) -> None:
    # A duplicate that moved position is still a duplicate: the two identical
    # rows below are not adjacent, and must still be grouped together.
    scattered = """
Date,Description,Amount
2026-08-17,BLUE BOTTLE,-4.50
2026-08-18,NETFLIX,-15.99
2026-08-17,BLUE BOTTLE,-4.50
"""
    preview = preview_csv(_write(tmp_path, scattered))
    groups = preview.duplicate_groups
    assert len(groups) == 1
    _, rows = groups[0]
    assert len(rows) == 2


def test_a_different_amount_is_not_grouped_as_a_duplicate(tmp_path: Path) -> None:
    distinct_amounts = """
Date,Description,Amount
2026-08-17,BLUE BOTTLE,-4.50
2026-08-17,BLUE BOTTLE,-5.50
"""
    preview = preview_csv(_write(tmp_path, distinct_amounts))
    assert preview.duplicate_groups == []


def test_merchant_variants_are_grouped_as_duplicates(tmp_path: Path) -> None:
    # The same charge exported twice with different processor noise is one
    # transaction, and dedupe has to see that.
    variants = """
Date,Description,Amount
2026-08-17,SQ *BLUE BOTTLE 4412,-4.50
2026-08-17,TST* BLUE BOTTLE,-4.50
"""
    preview = preview_csv(_write(tmp_path, variants))
    assert len(preview.duplicate_groups) == 1


def test_duplicates_are_reported_not_removed(tmp_path: Path) -> None:
    # Two identical coffees on one day are a real pair. Collapsing them would
    # delete money that was actually spent.
    preview = preview_csv(_write(tmp_path, TWO_IDENTICAL))
    assert len(preview.rows) == 3
    assert len(preview.duplicate_groups) == 1


def test_distinct_rows_produce_no_duplicate_groups(tmp_path: Path) -> None:
    assert preview_csv(_write(tmp_path, CHASE)).duplicate_groups == []


# --- The preview itself ----------------------------------------------------


def test_a_preview_writes_nothing(tmp_path: Path) -> None:
    path = _write(tmp_path, CHASE)
    before = path.read_text(encoding="utf-8")
    preview_csv(path)
    assert path.read_text(encoding="utf-8") == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["export.csv"]


def test_a_preview_renders_as_text(tmp_path: Path) -> None:
    rendered = preview_csv(_write(tmp_path, CHASE)).render()
    assert "parsed 3" in rendered
    assert "BLUE BOTTLE" in rendered


def test_the_render_is_ascii_safe(tmp_path: Path) -> None:
    # A preview that cannot be printed on a default Windows console is useless
    # at exactly the moment somebody needs to read it.
    preview_csv(_write(tmp_path, MONZO)).render().encode("cp1252")


def test_the_render_shows_duplicates_without_removing_them(tmp_path: Path) -> None:
    rendered = preview_csv(_write(tmp_path, TWO_IDENTICAL)).render()
    assert "possible duplicates" in rendered
    assert "not removed" in rendered


def test_a_missing_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="no file at"):
        preview_csv(tmp_path / "nope.csv")


def test_a_headerless_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValidationError, match="no header"):
        preview_csv(path)


def test_a_spreadsheet_byte_order_mark_is_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "bom.csv"
    path.write_text("﻿" + CHASE.strip() + "\n", encoding="utf-8")
    preview = preview_csv(path)
    assert preview.mapping is not None
    assert len(preview.rows) == 3


def test_the_preview_reports_the_detected_order(tmp_path: Path) -> None:
    assert preview_csv(_write(tmp_path, MONZO)).date_order is DateOrder.DAY_FIRST


def test_amounts_are_decimal_not_float(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, CHASE))
    assert isinstance(preview.rows[0].amount.amount, Decimal)


# --- Preference between plausible columns ----------------------------------

CHASE_FULL = """
Transaction Date,Post Date,Description,Category,Type,Amount,Memo
08/17/2026,08/18/2026,SQ *BLUE BOTTLE 4412,Food & Drink,Sale,-4.50,
08/18/2026,08/19/2026,NETFLIX.COM,Entertainment,Sale,-15.99,
08/19/2026,08/20/2026,SHELL OIL 4471,Gas,Sale,-52.10,
08/20/2026,08/21/2026,ACME CORP DIRECT DEP,,Payment,3210.44,
"""


def test_a_real_chase_export_imports(tmp_path: Path) -> None:
    # Every Chase export carries both a transaction date and a post date.
    # Refusing to choose would make the most common export unimportable.
    preview = preview_csv(_write(tmp_path, CHASE_FULL))
    assert preview.mapping is not None
    assert len(preview.rows) == 4


def test_the_transaction_date_beats_the_post_date(tmp_path: Path) -> None:
    # When the purchase happened, not when the bank got round to it.
    preview = preview_csv(_write(tmp_path, CHASE_FULL))
    assert preview.mapping is not None
    assert preview.mapping.date == "Transaction Date"


def test_the_description_beats_the_memo(tmp_path: Path) -> None:
    preview = preview_csv(_write(tmp_path, CHASE_FULL))
    assert preview.mapping is not None
    assert preview.mapping.description == "Description"


def test_a_preference_choice_is_reported_not_hidden(tmp_path: Path) -> None:
    # The principle survives: the importer chooses, and says that it chose.
    preview = preview_csv(_write(tmp_path, CHASE_FULL))
    assert any("Post Date" in choice for choice in preview.detection.resolved)


def test_a_reported_choice_appears_in_the_render(tmp_path: Path) -> None:
    rendered = preview_csv(_write(tmp_path, CHASE_FULL)).render()
    assert "resolved:" in rendered


def test_an_explicit_mapping_still_overrides_a_preference(tmp_path: Path) -> None:
    # Reporting the choice is only useful if it can be overruled.
    preview = preview_csv(
        _write(tmp_path, CHASE_FULL),
        mapping=ColumnMapping(date="Post Date", description="Description", amount="Amount"),
    )
    assert preview.rows[0].posted_on == date(2026, 8, 18)
