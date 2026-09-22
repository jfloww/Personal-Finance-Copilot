"""Shared pytest configuration."""

from __future__ import annotations

import os

import pytest

# Tests never inherit the application's local database credential from
# backend/.env. Database-backed tests must opt into an explicitly named test
# database; an ordinary local checkout therefore cannot touch a shared Neon
# database merely because the application is configured to use one.
os.environ["CONNECTION_STRING"] = os.environ.get("TEST_DATABASE_URL", "")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite golden fixture files instead of asserting against them.",
    )
