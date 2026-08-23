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
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final, cast

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.parsing import normalise_description, parse_amount
from offerdelta.ingest.dates import DateOrder, detect_date_order, parse_date
from offerdelta.ingest.mapping import ColumnMapping, MappingDetection, detect_mapping

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

    def render(self, limit: int = 10) -> str:
        lines = [f"{self.path}: {self.total_rows} rows", ""]
        lines.append(self.detection.render())
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
    for line, row in numbered:
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
        total_rows=len(rows),
        rows=tuple(parsed),
        errors=tuple(errors),
    )


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

    A signed `amount` column is taken as-is. Split `debit`/`credit` columns hold
    magnitudes, so the debit is negated — the convention everywhere else in this
    codebase is that money out is negative.
    """
    if mapping.amount:
        return parse_amount((row.get(mapping.amount) or "").strip())

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
