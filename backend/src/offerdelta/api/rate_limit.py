"""A fixed-window limiter for login attempts.

Five failures per email address per fifteen minutes, then refused until the
window closes - the values the login endpoint configures this with, not a
property of the class itself. In-process and fixed-window: it does not
survive a restart and means nothing across instances. On one free-tier
instance it works, and saying so here is better than implying a guarantee
that is not there.

Fixed rather than sliding: a caller landing right on the boundary can get a
short burst of extra attempts as the window rolls over. Accepted for a login
endpoint - a sliding window or a token bucket is real code this
single-instance limiter does not need yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta


@dataclass
class _Window:
    started_at: datetime
    count: int


class FixedWindowLimiter:
    """Counts calls per key inside a fixed time window.

    One dictionary, no lock - it matches `InMemoryIdempotencyStore`, which
    accepts the same "single process, single instance" honesty rather than
    reaching for machinery a walking-skeleton-stage deployment does not need.
    """

    def __init__(self, max_attempts: int, window: timedelta) -> None:
        self._max_attempts = max_attempts
        self._window = window
        self._windows: dict[str, _Window] = {}

    def check(self, key: str, now: datetime | None = None) -> bool:
        """Record one call for `key` and report whether it was allowed.

        Every call counts against the window, not only failed ones - the
        caller decides what counts as an attempt by when it calls this. A
        window opens on its first call and only ever expires on its own; nothing
        resets it early, so a call that succeeds still spends one of the
        address's slots for the window it landed in.
        """
        at = now if now is not None else datetime.now(UTC)
        self._evict_expired(at)
        window = self._windows.get(key)
        if window is None or at >= window.started_at + self._window:
            window = _Window(started_at=at, count=0)
            self._windows[key] = window
        window.count += 1
        return window.count <= self._max_attempts

    def _evict_expired(self, at: datetime) -> None:
        """Drop every window whose time has already run out.

        The keys are email addresses a caller supplies, so they are
        attacker-controlled: varying the address (or, before that was fixed,
        just its whitespace) opens a new entry for free. Without this, the
        dictionary grows by one entry per distinct address for the life of
        the process - a slow, unbounded leak rather than a crash, which is
        exactly the kind of thing nothing notices until the container falls
        over. Swept here, on every insert, rather than on a timer: eviction
        needs no background task and no dependency beyond the calls this
        limiter already receives.
        """
        expired = [
            key for key, window in self._windows.items() if at >= window.started_at + self._window
        ]
        for key in expired:
            del self._windows[key]
