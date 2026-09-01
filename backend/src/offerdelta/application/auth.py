"""Authentication, one layer above where it is actually stored.

`api/main.py` needs a user by email-and-password and a user by id, and is not
allowed to reach `UserRepository` directly: Task 9 adds the import-linter
contract that forbids exactly that import from the API layer. This module is
where the calls live instead, so the boundary already holds before the rule
that pins it exists - writing the API layer straight against the repository
now would land a layering violation that only Task 9 would notice.
"""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from offerdelta.application.scope import AuthenticatedUser
from offerdelta.infrastructure.postgres.repositories import UserRepository


def authenticate(session: Session, email: str, password: str) -> AuthenticatedUser | None:
    """`None` for every failure: unknown address, wrong password, or deactivated.

    A pass-through to `UserRepository.authenticate`, which is where the
    constant-work comparison actually lives - this module adds no logic of
    its own, only the layer boundary.
    """
    return UserRepository(session).authenticate(email, password)


def load_active_user(session: Session, user_id: uuid.UUID) -> AuthenticatedUser | None:
    """The user behind a token, or `None` if they can no longer act.

    Meant to be called on every request rather than trusted from the token's
    claims: a deactivated user's already-issued token would otherwise keep
    working until it expires on its own, up to an hour after the account was
    turned off.
    """
    stored = UserRepository(session).by_id(user_id)
    if stored is None or not stored.is_active:
        return None
    return AuthenticatedUser(id=stored.id, email=stored.email)
