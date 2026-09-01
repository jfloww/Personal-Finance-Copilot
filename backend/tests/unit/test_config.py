"""Runtime configuration settings and their availability flags.

Tests the Settings class, ensuring secrets are optional and
availability properties report whether each feature can be enabled.
"""

from __future__ import annotations

import pytest

from offerdelta.config import get_settings


def test_auth_is_unavailable_without_a_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("JWT_SECRET", raising=False)
    get_settings.cache_clear()
    try:
        assert get_settings().auth_available is False
    finally:
        get_settings.cache_clear()


def test_auth_is_available_with_a_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("JWT_SECRET", "x" * 32)
    get_settings.cache_clear()
    try:
        assert get_settings().auth_available is True
    finally:
        get_settings.cache_clear()


def test_auth_is_unavailable_with_an_empty_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("JWT_SECRET", "")
    get_settings.cache_clear()
    try:
        assert get_settings().auth_available is False
    finally:
        get_settings.cache_clear()
