from __future__ import annotations

import pytest

from offerdelta.application.scope import TenantScope
from offerdelta.domain.common.errors import ValidationError
from offerdelta.infrastructure.postgres.repositories import AccountRepository
from tests.integration.conftest import requires_database

pytestmark = requires_database


def test_registering_returns_a_canonical_key(scope: TenantScope) -> None:
    account = AccountRepository(scope).register("Chase Checking")
    assert account.key == "chase-checking"
    assert account.display_name == "Chase Checking"


def test_case_variants_resolve_to_the_same_account(scope: TenantScope) -> None:
    """The original bug: Checking and checking were two accounts."""
    repo = AccountRepository(scope)
    registered = repo.register("Checking")

    for spelling in ("checking", "Checking", "CHECKING", "  Checking  "):
        found = repo.by_key(spelling)
        assert found is not None
        assert found.id == registered.id


def test_registering_the_same_account_twice_is_refused(scope: TenantScope) -> None:
    repo = AccountRepository(scope)
    repo.register("Checking")
    with pytest.raises(ValidationError, match="already registered"):
        repo.register("checking")


def test_an_unregistered_account_is_not_found(scope: TenantScope) -> None:
    assert AccountRepository(scope).by_key("nonexistent") is None
