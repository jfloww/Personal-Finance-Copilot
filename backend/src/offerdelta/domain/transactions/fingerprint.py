"""The identity that decides what is a duplicate and what is new money.

Two rules make this value trustworthy, and both were violated by the first
implementation.

**Every input is a persisted column.** Nothing derived-and-discarded enters the
hash, so a stored fingerprint can always be recomputed from its own row. A
fingerprint you cannot reproduce is one you can never rebuild or audit.

**The amount is quantised before hashing.** Persistence rounds to two places,
so hashing the unrounded value produced a fingerprint that disagreed with the
row it was stored beside: `-4.50` and `-4.5` hashed differently while landing
on the same stored amount, and the same charge was written twice.

`normalised_merchant` is the output of an evolving heuristic, so the version is
stamped on every row. Changing `normalise_description` without bumping
`FINGERPRINT_VERSION` silently re-imports every affected transaction.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import date
from typing import Final

from offerdelta.domain.common.money import Money
from offerdelta.domain.common.rounding import CURRENCY_DISPLAY

#: Bump whenever any input to the payload changes meaning — including a change
#: to `normalise_description`, whose output is hashed here.
FINGERPRINT_VERSION: Final[int] = 1

#: ASCII unit separator. Vanishingly rare in bank descriptions, and joining on
#: it stops "A" + "2026-08-17" from colliding with a merchant literally named
#: "A\x1f2026-08-17".
_DELIMITER: Final = "\x1f"

_HEX_LENGTH: Final = 32


def compute_fingerprint(
    *,
    account_id: uuid.UUID,
    posted_on: date,
    normalised_merchant: str,
    amount: Money,
) -> str:
    """A stable identity for one transaction within one account.

    Deliberately excludes the line number: a duplicate that moved position in
    the file is still a duplicate.
    """
    quantised = amount.quantize(CURRENCY_DISPLAY)
    payload = _DELIMITER.join(
        (
            str(account_id),
            posted_on.isoformat(),
            normalised_merchant,
            f"{quantised.amount:.2f}",
            quantised.currency,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:_HEX_LENGTH]
