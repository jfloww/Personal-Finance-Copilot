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
import unicodedata
from typing import Final

from offerdelta.domain.common.errors import ValidationError

_COLLAPSE: Final = re.compile(r"-+")


def canonical_account_key(raw: str) -> str:
    """Normalise, casefold, and reduce every run of non-alphanumerics to one hyphen.

    NFKC first so that a composed "é" and a decomposed "e" + combining accent
    canonicalise to the same key — otherwise two identical-looking names would
    be two accounts, which is the bug this function exists to prevent.

    Classification is Unicode-aware rather than ASCII-only: an accented or
    non-Latin name must keep its letters. Folding them away made "Café Checking"
    and "Caf Checking" the same account, and rejected a name written entirely in
    a non-Latin script as empty.
    """
    folded = unicodedata.normalize("NFKC", raw).strip().casefold()
    key = _COLLAPSE.sub("-", "".join(c if c.isalnum() else "-" for c in folded)).strip("-")
    if not key:
        raise ValidationError(f"an account name needs at least one letter or digit, got {raw!r}")
    return key
