"""Access tokens.

Every rejection returns None rather than a reason. The caller turns all of
them into the same 401, so a distinguishable reason would only ever leak.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from offerdelta.infrastructure.auth.tokens import TOKEN_TTL, decode_token, issue_token

SECRET = "test secret, at least thirty-two characters long"
OTHER_SECRET = "a different secret, also long enough for use"
NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)


def test_a_token_round_trips_to_its_subject() -> None:
    user_id = uuid.uuid4()
    token = issue_token(user_id, secret=SECRET, now=NOW)
    assert decode_token(token, secret=SECRET, now=NOW) == user_id


def test_a_token_is_valid_just_before_it_expires() -> None:
    user_id = uuid.uuid4()
    token = issue_token(user_id, secret=SECRET, now=NOW)
    just_inside = NOW + TOKEN_TTL - timedelta(seconds=1)
    assert decode_token(token, secret=SECRET, now=just_inside) == user_id


def test_an_expired_token_is_rejected() -> None:
    token = issue_token(uuid.uuid4(), secret=SECRET, now=NOW)
    past_expiry = NOW + TOKEN_TTL + timedelta(seconds=1)
    assert decode_token(token, secret=SECRET, now=past_expiry) is None


def test_a_token_signed_with_another_secret_is_rejected() -> None:
    token = issue_token(uuid.uuid4(), secret=OTHER_SECRET, now=NOW)
    assert decode_token(token, secret=SECRET, now=NOW) is None


def test_a_malformed_token_is_rejected() -> None:
    assert decode_token("not.a.token", secret=SECRET, now=NOW) is None


def test_an_empty_token_is_rejected() -> None:
    assert decode_token("", secret=SECRET, now=NOW) is None


def test_the_ttl_is_one_hour() -> None:
    assert TOKEN_TTL.total_seconds() == 3600
