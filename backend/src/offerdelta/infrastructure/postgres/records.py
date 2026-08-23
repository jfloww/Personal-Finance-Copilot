"""What the repository accepts.

Deliberately not the domain `Transaction`: that entity refuses to exist without
a `kind`, and a SPENDING one without a category, because an uncategorised
outflow vanishes from every total. An imported bank row has neither until
something classifies it. Loosening a real safety invariant to make persistence
tidier would be the wrong trade, so imported rows are persistence records until
they are classified.

Deliberately not the ingest `ImportPlan` either: a manually entered transaction
has no CSV, no parsed row, and no source line, and should not have to fabricate
a preview to reach storage.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from offerdelta.domain.common.money import Money


@dataclass(frozen=True)
class Provenance:
    """Where a stored transaction came from. Absent for manual entry."""

    source_file: str
    source_line: int
    raw_cells: dict[str, str]


@dataclass(frozen=True)
class TransactionRecord:
    """One transaction, ready to persist, from any source.

    The fingerprint is deliberately absent: the repository derives it from
    these fields, which is what keeps a stored fingerprint reproducible from
    its own row rather than dependent on whoever built the record.
    """

    account_id: uuid.UUID
    posted_on: date
    description: str
    normalised_merchant: str
    amount: Money
    external_id: str | None
    occurrence: int
    provenance: Provenance | None
