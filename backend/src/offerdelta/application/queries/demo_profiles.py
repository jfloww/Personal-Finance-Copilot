"""The public profile catalogue used by the demo and read-only agent tools."""

from __future__ import annotations

from dataclasses import dataclass

from offerdelta.demo.profiles import (
    AUBURN_CURRENT,
    DEMO_PROFILE_FACTORIES,
    NEW_JERSEY_CANDIDATE,
    ComparisonSide,
    demo_profile,
)
from offerdelta.domain.common.errors import ValidationError


@dataclass(frozen=True)
class DemoProfileSummary:
    key: str
    label: str


PROFILE_KEYS = (AUBURN_CURRENT, NEW_JERSEY_CANDIDATE)


def list_demo_profiles() -> tuple[DemoProfileSummary, ...]:
    """List stable keys without exposing the profile's financial inputs."""
    return tuple(
        DemoProfileSummary(key=key, label=demo_profile(key).employment.label)
        for key in sorted(DEMO_PROFILE_FACTORIES)
    )


def load_demo_profile(key: str) -> ComparisonSide:
    """Load a profile while mapping fixture lookup failures to a domain error."""
    try:
        return demo_profile(key)
    except ValueError as error:
        raise ValidationError(str(error)) from error
