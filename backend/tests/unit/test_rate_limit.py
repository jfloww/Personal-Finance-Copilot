"""Five failures per address per fifteen minutes.

In-process and fixed-window: it does not survive a restart and means nothing
across instances. On one free-tier instance it works, and saying so here is
better than implying a guarantee that is not there.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from offerdelta.api.rate_limit import FixedWindowLimiter

NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)


def test_the_first_five_attempts_are_allowed() -> None:
    limiter = FixedWindowLimiter(max_attempts=5, window=timedelta(minutes=15))
    assert [limiter.check("a@example.test", NOW) for _ in range(5)] == [True] * 5


def test_the_sixth_attempt_is_refused() -> None:
    limiter = FixedWindowLimiter(max_attempts=5, window=timedelta(minutes=15))
    for _ in range(5):
        limiter.check("a@example.test", NOW)
    assert limiter.check("a@example.test", NOW) is False


def test_the_window_reopens() -> None:
    limiter = FixedWindowLimiter(max_attempts=5, window=timedelta(minutes=15))
    for _ in range(5):
        limiter.check("a@example.test", NOW)
    later = NOW + timedelta(minutes=15, seconds=1)
    assert limiter.check("a@example.test", later) is True


def test_addresses_are_counted_separately() -> None:
    limiter = FixedWindowLimiter(max_attempts=5, window=timedelta(minutes=15))
    for _ in range(5):
        limiter.check("a@example.test", NOW)
    assert limiter.check("b@example.test", NOW) is True


def test_an_expired_windows_entry_is_actually_removed_not_just_ignored() -> None:
    """The keys are attacker-controlled - a new address, or before that was
    fixed, a whitespace variant of an old one - opens a new dictionary entry
    for free. Proving the window is *evicted* rather than merely stale-but-
    present is the only way to know the dictionary does not grow forever on a
    long-lived process; a limiter that just ignored the old count while
    leaving the entry in place would pass every test above and still leak."""
    limiter = FixedWindowLimiter(max_attempts=5, window=timedelta(minutes=15))
    limiter.check("a@example.test", NOW)
    assert "a@example.test" in limiter._windows

    later = NOW + timedelta(minutes=15, seconds=1)
    limiter.check("b@example.test", later)

    assert "a@example.test" not in limiter._windows
