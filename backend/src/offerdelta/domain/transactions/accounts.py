"""Account identity.

An account was previously free text compared after a bare `.strip()`, so
`--account=Checking` in August and `--account=checking` in September built two
parallel copies of the same statement and reported both as clean imports. Every
total silently doubled.

One function owns the normalisation, and the canonical key it returns is what
the unique constraint sees. The text the user typed is preserved separately for
display — normalisation is lossy, and "chase-checking" is not what anyone wants
to read on a report.
"""

from __future__ import annotations

import re
from typing import Final

from offerdelta.domain.common.errors import ValidationError

_SEPARATORS: Final = re.compile(r"[^a-z0-9]+")


def canonical_account_key(raw: str) -> str:
    """Casefold, collapse every run of punctuation or space to one hyphen.

    Idempotent: applying it to its own output changes nothing.
    """
    key = _SEPARATORS.sub("-", raw.strip().casefold()).strip("-")
    if not key:
        raise ValidationError(f"an account name needs at least one letter or digit, got {raw!r}")
    return key
