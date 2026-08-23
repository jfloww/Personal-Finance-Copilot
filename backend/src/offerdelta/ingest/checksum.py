"""File identity.

A byte-identical re-import is the one case where "already imported" is a fact
rather than an inference from content, so it is worth recording exactly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Final

_CHUNK: Final = 1 << 20


def file_sha256(path: Path) -> str:
    """Hex digest of the file's bytes, read in chunks so size does not matter."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
