"""Turning an inspected preview into an explicit import plan.

Planning is deliberately separate from writing. The preview remains a pure,
read-only description of the source file; only a caller that explicitly asks
for a plan can hand rows to a repository.

The occurrence number solves an important ambiguity in the fingerprint. Two
identical coffees on one day may both be real, so a fingerprint cannot be
unique by itself. Numbering repeats per account preserves both charges while
still making a re-import of the same file a no-op.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from offerdelta.domain.common.errors import ValidationError
from offerdelta.ingest.preview import ImportPreview, ParsedRow


@dataclass(frozen=True)
class PlannedRow:
    """A parsed row with its stable position among identical transactions."""

    row: ParsedRow
    occurrence: int


@dataclass(frozen=True)
class ImportPlan:
    """Rows that are safe to write as one transaction."""

    account: str
    source_file: str
    rows: tuple[PlannedRow, ...]


def plan_import(preview: ImportPreview, *, account: str) -> ImportPlan:
    """Validate a preview and assign multiplicity-aware identities.

    A preview with even one bad row is refused. Committing only the valid
    subset would violate the preview's central promise that nothing disappears
    silently; the user should correct or deliberately remove the bad source row
    and preview again.
    """
    account = account.strip()
    if not account:
        raise ValidationError("an import needs an account name")
    if preview.mapping is None:
        raise ValidationError("cannot commit an import whose column mapping is unresolved")
    if preview.errors:
        raise ValidationError(
            f"cannot commit while {len(preview.errors)} source row(s) have errors; "
            "fix them and preview the file again"
        )
    if not preview.rows:
        raise ValidationError("cannot commit an import with no parsed rows")

    seen: dict[str, int] = defaultdict(int)
    planned: list[PlannedRow] = []
    for row in preview.rows:
        # Content-based identity, not a stored fingerprint: preview has no
        # account to compute a real one against. Mirrors
        # ImportPreview.duplicate_groups' grouping key.
        key = f"{row.posted_on.isoformat()}|{row.normalised_merchant}|{row.amount.amount:.2f}"
        seen[key] += 1
        planned.append(PlannedRow(row=row, occurrence=seen[key]))

    return ImportPlan(
        account=account,
        # A local directory is not part of transaction provenance and may
        # expose a username. The bank-provided filename is sufficient.
        source_file=Path(preview.path).name,
        rows=tuple(planned),
    )
