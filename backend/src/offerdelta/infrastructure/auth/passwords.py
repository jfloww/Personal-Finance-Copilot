"""Password hashing.

argon2id rather than bcrypt. bcrypt truncates silently at 72 bytes, which
turns two different long passwords into the same hash and tells nobody.

`verify_password` accepts a `None` hash and still performs a verification
against a throwaway hash before returning False. A user who does not exist
must not answer faster than one who does, because the difference is how an
attacker enumerates addresses.
"""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

_hasher = PasswordHasher()

#: Verified against when there is no real hash, purely to spend the time.
_DUMMY_HASH = _hasher.hash("dummy password for constant-work verification")


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str | None) -> bool:
    target = hashed if hashed is not None else _DUMMY_HASH
    try:
        _hasher.verify(target, plain)
    except (VerifyMismatchError, VerificationError):
        return False
    return hashed is not None
