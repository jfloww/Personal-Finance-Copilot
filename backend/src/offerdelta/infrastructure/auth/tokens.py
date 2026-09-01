"""Access tokens.

HS256, one hour, three claims. There are no roles to encode, and a claim that
exists is a claim something will eventually trust.

There is no refresh token. A refresh token is only meaningful next to a
revocation store; without one it is a second secret with a longer life and no
compensating power. Re-authenticating is one POST.

Every failure returns None. The caller turns all of them into the same 401, so
distinguishing them here could only ever leak.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Final

import jwt

#: Short enough that a leaked token stops working the same afternoon, long
#: enough that a person is not re-authenticating mid-task.
TOKEN_TTL: Final = timedelta(hours=1)

_ALGORITHM: Final = "HS256"


def issue_token(user_id: uuid.UUID, *, secret: str, now: datetime | None = None) -> str:
    issued_at = now or datetime.now(UTC)
    payload = {
        "sub": str(user_id),
        "iat": int(issued_at.timestamp()),
        "exp": int((issued_at + TOKEN_TTL).timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=_ALGORITHM)


def decode_token(token: str, *, secret: str, now: datetime | None = None) -> uuid.UUID | None:
    at = now or datetime.now(UTC)
    try:
        claims = jwt.decode(
            token,
            secret,
            algorithms=[_ALGORITHM],
            options={"require": ["sub", "exp", "iat"], "verify_exp": False},
        )
    except jwt.PyJWTError:
        return None

    expires_at = claims.get("exp")
    if not isinstance(expires_at, int) or at.timestamp() >= expires_at:
        return None

    subject = claims.get("sub")
    if not isinstance(subject, str):
        return None
    try:
        return uuid.UUID(subject)
    except ValueError:
        return None
