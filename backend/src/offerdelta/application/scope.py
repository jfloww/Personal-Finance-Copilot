"""Who is asking, and the session they are asking through.

A repository takes one of these where it used to take a bare `Session`, so a
query without a tenant is not something a caller can forget to write - it is
something they cannot express.

`AuthenticatedUser` is the shape after the checks have passed, which is why it
carries no `is_active`: an inactive user never becomes one. Holding a
`TenantScope` means identity was established, not that it still needs
verifying.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy.orm import Session


@dataclass(frozen=True)
class AuthenticatedUser:
    id: uuid.UUID
    email: str


@dataclass(frozen=True)
class TenantScope:
    session: Session
    user: AuthenticatedUser
