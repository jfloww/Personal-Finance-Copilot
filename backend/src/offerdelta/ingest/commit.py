"""Turning an inspected preview into records that are safe to write.

Planning stays separate from writing: the preview remains a pure, read-only
description of the file, and only a caller that explicitly asks gets records
back.

## The ambiguity this module refuses to guess at

A fingerprint plus a per-file occurrence number cannot distinguish a genuine
third identical charge from one already stored. Suppose two `BLUE BOTTLE -4.50`
charges on 2026-08-17 are already persisted, and a new file contains exactly
one. Two readings are equally consistent with the file:

1. It is a **full-window snapshot** overlapping August, so that charge is
   occurrence 1 and is already stored. Writing it duplicates real money.
2. It is an **append-only incremental export**, so that charge is a genuine
   third coffee. Refusing it loses real money.

The deciding information is not in the file — it is a fact about how the file
was produced. So the mode is declared by the caller, never inferred, and
incremental mode refuses to run without a stable bank transaction id rather
than silently picking a reading.
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path
from typing import Final

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.rounding import CURRENCY_DISPLAY
from offerdelta.infrastructure.postgres.records import Provenance, TransactionRecord
from offerdelta.ingest.preview import ImportPreview

#: How many offending lines to name before summarising the rest as "and N more".
_MAX_LINES_SHOWN: Final = 10


class ImportMode(StrEnum):
    """How the file was produced. Declared, never detected."""

    #: A complete window. Occurrences are numbered per file.
    SNAPSHOT = "snapshot"

    #: Only activity not previously exported. Requires an external id.
    INCREMENTAL = "incremental"


@dataclass(frozen=True)
class ImportWindow:
    """The date range a snapshot claims to cover completely."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValidationError(
                f"an import window starts before it ends; got {self.start} to {self.end}"
            )

    def covers(self, day: date) -> bool:
        return self.start <= day <= self.end


def plan_records(
    preview: ImportPreview,
    *,
    account_id: uuid.UUID,
    mode: ImportMode,
    window: ImportWindow | None,
) -> tuple[TransactionRecord, ...]:
    """Validate a preview and turn it into records ready to persist.

    A preview with even one bad row is refused outright. Committing only the
    valid subset would break the preview's central promise that nothing
    disappears silently.
    """
    if preview.mapping is None:
        raise ValidationError("cannot commit an import whose column mapping is unresolved")
    if preview.errors:
        raise ValidationError(
            f"cannot commit while {len(preview.errors)} source row(s) have errors; "
            "fix them and preview the file again"
        )
    if not preview.rows:
        raise ValidationError("cannot commit an import with no parsed rows")

    has_external_id = bool(preview.mapping.external_id)

    if mode is ImportMode.INCREMENTAL and not has_external_id:
        raise ValidationError(
            "an incremental import cannot tell a new repeat charge from one "
            "already stored, because the deciding fact is not in the file. "
            "Supply the bank's transaction id column with --map=external_id:<column>, "
            "or re-export a full window and use --mode=snapshot."
        )

    if mode is ImportMode.SNAPSHOT:
        if window is None:
            raise ValidationError("a snapshot import needs a declared window; pass --from and --to")
        outside = [row.line for row in preview.rows if not window.covers(row.posted_on)]
        if outside:
            shown = ", ".join(f"line {line}" for line in outside[:_MAX_LINES_SHOWN])
            more = (
                ""
                if len(outside) <= _MAX_LINES_SHOWN
                else f" and {len(outside) - _MAX_LINES_SHOWN} more"
            )
            raise ValidationError(
                f"{len(outside)} row(s) fall outside the declared window "
                f"{window.start} to {window.end}: {shown}{more}. The file is not "
                f"the snapshot it was declared to be."
            )

    source_file = Path(preview.path).name
    seen: dict[tuple[date, str, str], int] = defaultdict(int)
    records: list[TransactionRecord] = []

    for row in preview.rows:
        # Quantised exactly as `compute_fingerprint` quantises — through
        # CURRENCY_DISPLAY (ROUND_HALF_UP), with the same negative-zero
        # normalisation — so the occurrence-grouping key can never disagree
        # with the fingerprint the repository stores. Grouping on plain
        # `f"{amount:.2f}"` instead (ROUND_HALF_EVEN, no negative-zero fix)
        # let two rows land in different occurrence buckets that share one
        # fingerprint: the second write then trips the unique constraint and
        # surfaces as "conflicted with another import; retry" — a message
        # retrying can never resolve, since the same mismatched plan would be
        # rebuilt every time.
        quantised_amount = row.amount.quantize(CURRENCY_DISPLAY).amount
        normalised_amount = quantised_amount if quantised_amount else abs(quantised_amount)
        key = (row.posted_on, row.normalised_merchant, f"{normalised_amount:.2f}")
        seen[key] += 1
        external_id = None
        if preview.mapping.external_id:
            external_id = (row.raw.get(preview.mapping.external_id) or "").strip() or None
        records.append(
            TransactionRecord(
                account_id=account_id,
                posted_on=row.posted_on,
                description=row.description,
                normalised_merchant=row.normalised_merchant,
                amount=row.amount,
                external_id=external_id,
                occurrence=seen[key],
                provenance=Provenance(
                    source_file=source_file,
                    source_line=row.line,
                    raw_cells=dict(row.raw),
                ),
            )
        )

    return tuple(records)
