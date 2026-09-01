"""Password hashing.

argon2id over bcrypt because bcrypt silently truncates at 72 bytes: two
different long passwords can hash the same, and nothing tells anybody.
"""

from __future__ import annotations

from unittest.mock import patch

from argon2 import PasswordHasher

from offerdelta.infrastructure.auth import passwords as passwords_module
from offerdelta.infrastructure.auth.passwords import hash_password, verify_password


def test_a_hash_verifies_against_its_own_password() -> None:
    hashed = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", hashed) is True


def test_a_hash_rejects_a_different_password() -> None:
    hashed = hash_password("correct horse battery staple")
    assert verify_password("Correct horse battery staple", hashed) is False


def test_the_same_password_hashes_differently_each_time() -> None:
    assert hash_password("same input") != hash_password("same input")


def test_long_passwords_are_not_truncated() -> None:
    base = "a" * 100
    hashed = hash_password(base + "one")
    assert verify_password(base + "two", hashed) is False


def test_a_null_hash_never_verifies() -> None:
    """`password_hash IS NULL` means the user cannot log in at all."""
    assert verify_password("anything", None) is False


def test_verifying_against_null_still_does_the_work() -> None:
    """A missing user must not be faster than a wrong password.

    Proved deterministically rather than by timing: a stopwatch assertion is
    flaky for reasons that have nothing to do with the property (load on a
    shared CI runner, a slow first call before anything is warmed up) - see
    spec §7, "Timing is not asserted." A spy on `PasswordHasher.verify` shows
    both paths call it exactly once, which is the actual claim: the dummy
    path performs a real verification rather than short-circuiting.

    Patched on the class rather than `_hasher` itself: `PasswordHasher` is a
    C extension type whose `verify` attribute is read-only per instance, so
    only a class-level patch can intercept it. `wraps` is bound to `_hasher`
    specifically so the spy still calls through to the one real hasher this
    module uses.
    """
    hashed = hash_password("real password")

    with patch.object(PasswordHasher, "verify", wraps=passwords_module._hasher.verify) as spy:
        verify_password("guess", hashed)
        verify_password("guess", None)

    assert spy.call_count == 2
