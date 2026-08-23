"""Entering one transaction by hand.

The second input adapter, and the reason the repository stopped taking an
`ImportPlan`: a typed-in transaction has no CSV, no parsed row, and no source
line, and should not have to fabricate a preview to reach storage.

## Why a repeat has to be asserted

An import knows how many identical charges a file contained, so it can number
them. A form knows nothing: two coffees on one day and the same coffee typed in
twice are the same keystrokes. Guessing either way is wrong in a way the user
cannot see — silently storing both hides a typo, silently storing one loses
real money.

So the default refuses. An entry matching something already stored is reported,
not written, and `repeat=True` is how a person says "I know, there really were
two." That is the one piece of information only they have.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.fingerprint import compute_fingerprint
from offerdelta.domain.transactions.parsing import normalise_description
from offerdelta.infrastructure.postgres.models import TransactionRow
from offerdelta.infrastructure.postgres.repositories import (
    AccountRepository,
    TransactionRepository,
)
from offerdelta.records.transactions import TransactionRecord


@dataclass(frozen=True)
class ManualEntry:
    """One transaction as a person typed it."""

    account_key: str
    posted_on: date
    description: str
    amount: Money
    repeat: bool = False


@dataclass(frozen=True)
class EntryOutcome:
    """What happened, including the case where nothing was written."""

    stored: bool
    transaction_id: uuid.UUID | None
    fingerprint: str
    occurrence: int

    #: Set when `stored` is False: how many identical rows already exist.
    already_stored_count: int = 0


def enter_transaction(session: Session, entry: ManualEntry) -> EntryOutcome:
    """Write one hand-entered transaction, or report that it is already there."""
    description = entry.description.strip()
    if not description:
        raise ValidationError("a transaction needs a description")

    accounts = AccountRepository(session)
    account = accounts.by_key(entry.account_key)
    if account is None:
        known = ", ".join(a.key for a in accounts.all()) or "none registered yet"
        raise ValidationError(
            f"no account {entry.account_key!r}. Known accounts: {known}. "
            f"Register one with: transactions.py accounts add <display name>"
        )

    normalised_merchant = normalise_description(description)
    fingerprint = compute_fingerprint(
        account_id=account.id,
        posted_on=entry.posted_on,
        normalised_merchant=normalised_merchant,
        amount=entry.amount,
    )

    highest = session.scalars(
        select(func.max(TransactionRow.occurrence)).where(
            TransactionRow.account_id == account.id,
            TransactionRow.fingerprint == fingerprint,
        )
    ).one()
    existing = highest or 0

    if existing and not entry.repeat:
        return EntryOutcome(
            stored=False,
            transaction_id=None,
            fingerprint=fingerprint,
            occurrence=existing,
            already_stored_count=existing,
        )

    occurrence = existing + 1
    record = TransactionRecord(
        account_id=account.id,
        posted_on=entry.posted_on,
        description=description,
        normalised_merchant=normalised_merchant,
        amount=entry.amount,
        external_id=None,
        occurrence=occurrence,
        provenance=None,
    )

    result = TransactionRepository(session).add_many([record])
    if not result.imported_ids:
        # Only reachable if something was written between the count above and
        # this write. Reporting it beats claiming a row that is not ours.
        return EntryOutcome(
            stored=False,
            transaction_id=None,
            fingerprint=fingerprint,
            occurrence=occurrence,
            already_stored_count=result.already_stored_count,
        )

    return EntryOutcome(
        stored=True,
        transaction_id=result.imported_ids[0],
        fingerprint=fingerprint,
        occurrence=occurrence,
    )
