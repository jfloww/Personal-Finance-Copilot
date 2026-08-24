"""Previewing an import before committing to it.

Nothing here writes anything. The point is to show what *would* be imported so a
person can look at it first, because the cheapest moment to catch a
day-first/month-first mistake is before four hundred rows carry it.

Three rules govern the output.

**Nothing is silently dropped.** Every source row becomes either a parsed row or
a row error, and the preview asserts that the two counts add up to the file's
length. A row that fails validation is reported with its line number and reason,
never skipped.

**Originals are preserved.** Each parsed row keeps the raw cell values it came
from, so any figure can be traced back to the text that produced it without
re-reading the file.

**Duplicates are reported, not removed.** Two identical coffees on one day are a
legitimate pair, and an importer that silently deduplicated them would delete
real money. The preview groups rows sharing the same visible content and leaves
the decision to a person.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Final, cast

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.parsing import normalise_description, parse_amount
from offerdelta.ingest.dates import DateOrder, detect_date_order, parse_date
from offerdelta.ingest.mapping import (
    AmountSign,
    ColumnMapping,
    MappingDetection,
    detect_mapping,
)

#: How many values per column to sample when detecting the mapping and the date
#: order. Enough to find a decisive day-above-twelve without reading a whole
#: year of statements.
SAMPLE_SIZE: Final = 200

#: csv.DictReader puts surplus cells under a key and missing cells under a
#: value. Both defaults are `None`, which violates `dict[str, str]` and
#: serialises to a JSON key of "null". Explicit sentinels make a ragged row
#: detectable instead of silently corrupting the stored provenance.
_RESTKEY: Final = "__surplus__"
_RESTVAL: Final = "\x00__missing__"


@dataclass(frozen=True)
class ParsedRow:
    """One source row, normalised, with its original still attached."""

    line: int
    posted_on: date
    description: str
    normalised_merchant: str

    #: Signed: negative is money out, whichever shape the file used.
    amount: Money

    #: Every original cell, so a figure can be traced to the text behind it.
    raw: dict[str, str]


@dataclass(frozen=True)
class RowError:
    """A source row that could not be parsed, and why."""

    line: int
    reason: str
    raw: dict[str, str]


@dataclass(frozen=True)
class ImportPreview:
    """What an import would produce, before anything is written."""

    path: str
    headers: tuple[str, ...]
    detection: MappingDetection
    mapping: ColumnMapping | None
    date_order: DateOrder
    total_rows: int

    #: Source rows that were not transactions at all: every column the mapping
    #: uses was empty. Counted separately from `total_rows` so the parsed-plus-
    #: failed invariant still holds, and reported by `render` so setting them
    #: aside is visible rather than silent.
    blank_rows: int = 0
    rows: tuple[ParsedRow, ...] = ()
    errors: tuple[RowError, ...] = ()

    def __post_init__(self) -> None:
        if self.mapping is not None and len(self.rows) + len(self.errors) != self.total_rows:
            raise ValidationError(
                f"{len(self.rows)} parsed and {len(self.errors)} failed do not "
                f"account for {self.total_rows} source rows; an importer that "
                f"loses rows cannot be trusted with money"
            )

    @property
    def importable(self) -> bool:
        return self.mapping is not None and bool(self.rows)

    @property
    def duplicate_groups(self) -> list[tuple[str, list[ParsedRow]]]:
        """Rows that look identical, reported rather than removed.

        Grouped on the visible content rather than a stored fingerprint: this
        is a preview, and it has no account to compute a real identity against.
        """
        grouped: dict[str, list[ParsedRow]] = defaultdict(list)
        for row in self.rows:
            key = f"{row.posted_on.isoformat()}|{row.normalised_merchant}|{row.amount.amount:.2f}"
            grouped[key].append(row)
        return sorted(
            ((key, rows) for key, rows in grouped.items() if len(rows) > 1),
            key=lambda item: item[1][0].line,
        )

    def _sign_warning(self) -> list[str]:
        """Say so when a signed column looks like it runs the other way.

        Most rows on a card or current account are money leaving it. A signed
        file that is mostly positive under the default convention is therefore
        suspicious - American Express writes charges positive, and read at face
        value every charge becomes income and every payment becomes spending.

        This is a warning and not a refusal because a month of mostly refunds,
        or a savings account taking deposits, is a real thing. The numbers
        cannot settle it; only the person who downloaded the file can.
        """
        if self.mapping is None or not self.mapping.amount or not self.rows:
            return []
        if self.mapping.amount_sign is not AmountSign.OUTFLOW_NEGATIVE:
            return []

        positive = sum(1 for row in self.rows if row.amount.amount > 0)
        if positive * 2 <= len(self.rows):
            return []

        return [
            f"  {positive} of {len(self.rows)} rows are positive, i.e. read as money IN.",
            "  If this is an American Express export, charges are written positive and "
            "this is backwards:",
            "  re-run with --sign=outflow-positive. Every total depends on getting this right.",
        ]

    def _override_lines(self) -> list[str]:
        """Name the columns actually in use when they differ from detection.

        `detection` reports what the headers suggested. A caller-supplied
        mapping overrides it, and a preview that shows only the guess would
        name one column while the import reads another - in the one place whose
        whole job is showing what is about to happen.
        """
        if self.mapping is None or self.detection.mapping is None:
            return []

        detected = self.detection.mapping
        differences = [
            (field, getattr(detected, field), getattr(self.mapping, field))
            for field in (
                "date",
                "description",
                "merchant",
                "external_id",
                "amount",
                "debit",
                "credit",
            )
            if getattr(detected, field) != getattr(self.mapping, field)
        ]
        if not differences:
            return []

        lines = ["  supplied mapping overrides detection"]
        lines.extend(
            f"    {field:<12} <- {new!r} (detected {old!r})" for field, old, new in differences
        )
        return lines

    def render(self, limit: int = 10) -> str:
        lines = [f"{self.path}: {self.total_rows} rows", ""]
        lines.append(self.detection.render())
        lines.extend(self._override_lines())
        lines.append(f"  date order   {self.date_order}")

        if self.date_order is DateOrder.AMBIGUOUS and self.mapping is not None:
            lines.append(
                "  the date column reads both ways; state the order explicitly "
                "rather than risk an eleven-month error"
            )

        if self.mapping is None:
            lines.append("\nnothing parsed: supply an explicit mapping")
            return "\n".join(lines)

        lines.append("")
        lines.append(f"parsed {len(self.rows)}, failed {len(self.errors)}")
        lines.extend(self._sign_warning())
        if self.blank_rows:
            lines.append(
                f"  {self.blank_rows} source row(s) held nothing in any mapped column "
                f"and were not counted as transactions"
            )

        if self.rows:
            lines.append("")
            lines.append(f"  {'line':>5}  {'date':<12}{'merchant':<26}{'amount':>12}")
            for row in self.rows[:limit]:
                lines.append(
                    f"  {row.line:>5}  {row.posted_on.isoformat():<12}"
                    f"{row.normalised_merchant[:24]:<26}{row.amount.amount:>12}"
                )
            if len(self.rows) > limit:
                lines.append(f"  ... {len(self.rows) - limit} more")

        if self.errors:
            lines.append("")
            lines.append(f"errors ({len(self.errors)})")
            for error in self.errors[:limit]:
                lines.append(f"  line {error.line}: {error.reason}")
            if len(self.errors) > limit:
                lines.append(f"  ... {len(self.errors) - limit} more")

        if groups := self.duplicate_groups:
            lines.append("")
            lines.append(f"possible duplicates ({len(groups)} groups)")
            for _, rows in groups[:limit]:
                positions = ", ".join(str(row.line) for row in rows)
                lines.append(
                    f"  lines {positions}: {rows[0].normalised_merchant} "
                    f"{rows[0].amount.amount} on {rows[0].posted_on}"
                )
            lines.append(
                "  reported, not removed - two identical charges on one day can both be real"
            )

        return "\n".join(lines)


def preview_csv(
    path: Path | str,
    *,
    mapping: ColumnMapping | None = None,
    date_order: DateOrder | None = None,
    amount_sign: AmountSign | None = None,
) -> ImportPreview:
    """Parse a file and report what an import would produce.

    Writes nothing. Supply `mapping` when detection is uncertain, and
    `date_order` when the column reads both ways.
    """
    path = Path(path)
    if not path.exists():
        raise ValidationError(f"no file at {path}")

    with path.open(encoding="utf-8-sig", newline="") as handle:
        # A raw `csv.reader`, not `csv.DictReader`: `DictReader.__next__` skips
        # blank physical lines *inside itself*, looping past them before it
        # ever returns — which hides how many lines were skipped from anyone
        # tracking `line_num` from the outside. Every skip has to stay visible
        # here, or the next real row's recorded line drifts backwards by one
        # for each blank line already consumed.
        reader = csv.reader(handle)
        headers = tuple(next(reader, ()))
        # `next` forces the header read, so line_num now points at the last
        # physical line the header occupied — usually 1.
        previous_line = reader.line_num
        # `str | list[str]`, not `str`: a surplus cell is recorded under
        # `_RESTKEY` as a list, which is exactly the shape `_reject_ragged`
        # and `_displayable` below are written to detect.
        numbered: list[tuple[int, dict[str, str | list[str]]]] = []
        for raw_row in reader:
            if raw_row == []:
                # A genuinely blank physical line. Update the baseline and
                # move on without recording anything — same as
                # `DictReader.__next__` — but here the skip is visible, so it
                # is accounted for before the *next* row's line is computed.
                previous_line = reader.line_num
                continue
            numbered.append((previous_line + 1, _zip_row(headers, raw_row)))
            previous_line = reader.line_num

    rows = [record for _, record in numbered]

    if not headers:
        raise ValidationError(f"{path.name} has no header row")

    # `row.get(header) or ""` alone would let the missing-cell sentinel through:
    # it is a non-empty string, so a short row would otherwise poison the
    # sample that mapping and date-order detection are confirmed against.
    sample = {
        header: [_sample_value(row.get(header)) for row in rows[:SAMPLE_SIZE]] for header in headers
    }

    detection = detect_mapping(list(headers), sample)
    resolved = mapping or detection.mapping
    if resolved is not None and amount_sign is not None:
        # Detection reads headers, which say nothing about which way the signs
        # run. Applying the convention here means a file whose columns detect
        # correctly - as Amex does - needs only the sign stated, not a full
        # hand-written mapping.
        resolved = replace(resolved, amount_sign=amount_sign)

    if resolved is None:
        return ImportPreview(
            path=str(path),
            headers=headers,
            detection=detection,
            mapping=None,
            date_order=DateOrder.AMBIGUOUS,
            total_rows=len(rows),
        )

    missing = [c for c in resolved.source_columns() if c not in headers]
    if missing:
        raise ValidationError(
            f"the mapping names columns that are not in {path.name}: {', '.join(missing)}"
        )

    order = date_order or detect_date_order(sample.get(resolved.date, []))

    parsed: list[ParsedRow] = []
    errors: list[RowError] = []
    blank = 0
    for line, row in numbered:
        if _is_not_a_transaction(row, resolved):
            blank += 1
            continue
        try:
            _reject_ragged(row)
            # `_reject_ragged` raised if `row` held a surplus list or a missing
            # sentinel, so every value here is a plain cell.
            parsed.append(_to_row(cast(dict[str, str], row), line, resolved, order))
        except ValidationError as error:
            errors.append(RowError(line=line, reason=str(error), raw=_displayable(row)))

    return ImportPreview(
        path=str(path),
        headers=headers,
        detection=detection,
        mapping=resolved,
        date_order=order,
        total_rows=len(rows) - blank,
        blank_rows=blank,
        rows=tuple(parsed),
        errors=tuple(errors),
    )


def _is_not_a_transaction(row: dict[str, str | list[str]], mapping: ColumnMapping) -> bool:
    """True when every column the mapping uses is empty.

    Real exports carry rows that are not transactions: a Chase checking export
    in this repository's own test data holds 188 rows where all seven fields
    are empty and 18 more carrying a literal "1" in a column the mapping never
    reads - 206 of 334, interleaved rather than trailing, so they cannot simply
    be chopped off the end.

    Refusing the whole file over them is correct but useless, and skipping any
    row with *a* blank cell would silently drop real transactions. So the test
    is deliberately narrow: not one mapped column has anything in it, which
    means there is no transaction here to lose. A row with a date but no
    amount, or an amount but no date, is still an error and still reported.
    """
    for column in mapping.source_columns():
        value = row.get(column)
        if isinstance(value, str) and value.strip() and value != _RESTVAL:
            return False
    return True


def _sample_value(value: str | list[str] | None) -> str:
    """A cell's value for detection purposes: blank if absent or missing.

    The missing-cell sentinel is deliberately non-empty so a ragged row is
    never mistaken for a blank one — but that same non-emptiness would corrupt
    the sample that column and date-order detection confirm their guesses
    against, so it is normalised back to blank here. A list only ever appears
    under `_RESTKEY`, never under a real header, but the type is shared with
    `numbered`'s rows, so it is handled the same way: not a usable sample.
    """
    if value is None or isinstance(value, list) or value == _RESTVAL:
        return ""
    return value


def _zip_row(headers: tuple[str, ...], row: list[str]) -> dict[str, str | list[str]]:
    """Pair a raw row with the header, exactly as `csv.DictReader` would.

    Reimplemented rather than reused: `DictReader.__next__` is where the
    blank-line skipping this module needs to see happens, so the header-zip
    and restkey/restval logic it also does has to move out here with it.
    """
    # `strict=False`: a mismatched length is exactly what a ragged row is —
    # handled below via `_RESTKEY`/`_RESTVAL`, not an error at zip time.
    record: dict[str, str | list[str]] = dict(zip(headers, row, strict=False))
    if len(row) > len(headers):
        record[_RESTKEY] = row[len(headers) :]
    elif len(row) < len(headers):
        for header in headers[len(row) :]:
            record[header] = _RESTVAL
    return record


def _reject_ragged(row: dict[str, str | list[str]]) -> None:
    """A row that does not match the header is refused, not repaired.

    Guessing which column a surplus cell belongs to is exactly the kind of
    silent decision that puts a wrong number in front of someone.
    """
    if _RESTKEY in row:
        surplus = row[_RESTKEY]
        count = len(surplus) if isinstance(surplus, list) else 1
        raise ValidationError(
            f"the row has {count} extra cell(s) beyond the header; "
            f"the file does not match its own columns"
        )
    missing = [key for key, value in row.items() if value == _RESTVAL]
    if missing:
        raise ValidationError(f"the row is missing cell(s) for: {', '.join(sorted(missing))}")


def _displayable(row: dict[str, str | list[str]]) -> dict[str, str]:
    """Raw cells with the sentinels made readable for the error report."""
    out: dict[str, str] = {}
    for key, value in row.items():
        # A list only ever appears under `_RESTKEY`; narrowing on its shape
        # rather than the key name keeps `value` provably `str` below.
        if isinstance(value, list):
            out[key] = ", ".join(value)
        else:
            out[key] = "" if value == _RESTVAL else value
    return out


def _to_row(row: dict[str, str], line: int, mapping: ColumnMapping, order: DateOrder) -> ParsedRow:
    posted_on = parse_date(row.get(mapping.date) or "", order)

    description = (row.get(mapping.description) or "").strip()
    if not description and mapping.merchant:
        description = (row.get(mapping.merchant) or "").strip()
    if not description:
        raise ValidationError("description is blank")

    amount = _amount(row, mapping)

    return ParsedRow(
        line=line,
        posted_on=posted_on,
        description=description,
        normalised_merchant=normalise_description(description),
        amount=amount,
        raw=dict(row),
    )


def _amount(row: dict[str, str], mapping: ColumnMapping) -> Money:
    """Normalise either shape to one signed amount.

    A signed `amount` column is read under the declared convention. Split
    `debit`/`credit` columns hold magnitudes, so the debit is negated — the
    convention everywhere else in this codebase is that money out is negative.
    """
    if mapping.amount:
        signed = parse_amount((row.get(mapping.amount) or "").strip())
        if mapping.amount_sign is AmountSign.OUTFLOW_POSITIVE:
            return -signed
        return signed

    debit_text = (row.get(mapping.debit) or "").strip() if mapping.debit else ""
    credit_text = (row.get(mapping.credit) or "").strip() if mapping.credit else ""

    if debit_text and credit_text:
        raise ValidationError(
            "both debit and credit are filled in; one row cannot be money out and money in at once"
        )
    if not debit_text and not credit_text:
        raise ValidationError("neither debit nor credit is filled in")

    if debit_text:
        magnitude = parse_amount(debit_text)
        # Some exports already sign the debit column. Negating a negative would
        # turn a payment into income.
        return magnitude if magnitude.amount < 0 else -magnitude
    return parse_amount(credit_text)
