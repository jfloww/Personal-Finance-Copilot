from __future__ import annotations

import uuid
from datetime import date

import pytest

from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.fingerprint import FINGERPRINT_VERSION, compute_fingerprint

ACCOUNT = uuid.UUID("11111111-1111-1111-1111-111111111111")
OTHER = uuid.UUID("22222222-2222-2222-2222-222222222222")


def _fp(**overrides: object) -> str:
    kwargs: dict[str, object] = {
        "account_id": ACCOUNT,
        "posted_on": date(2026, 8, 17),
        "normalised_merchant": "BLUE BOTTLE",
        "amount": Money.parse("-4.50"),
    }
    kwargs.update(overrides)
    return compute_fingerprint(**kwargs)  # type: ignore[arg-type]


def test_trailing_zeros_do_not_change_the_fingerprint() -> None:
    """The defect this module exists to fix: -4.50 and -4.5 are one charge."""
    assert _fp(amount=Money.parse("-4.50")) == _fp(amount=Money.parse("-4.5"))


def test_sub_cent_differences_collapse_after_quantisation() -> None:
    assert _fp(amount=Money.parse("-4.504")) == _fp(amount=Money.parse("-4.501"))


def test_it_is_32_hex_characters() -> None:
    value = _fp()
    assert len(value) == 32
    assert all(c in "0123456789abcdef" for c in value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", OTHER),
        ("posted_on", date(2026, 8, 18)),
        ("normalised_merchant", "BLUE BOTTLE COFFEE"),
        ("amount", Money.parse("-4.51")),
    ],
)
def test_every_input_changes_the_fingerprint(field: str, value: object) -> None:
    assert _fp(**{field: value}) != _fp()


def test_the_delimiter_cannot_be_forged() -> None:
    """A merchant containing the separator must not collide with a real split."""
    left = _fp(normalised_merchant="A\x1f2026-08-17")
    right = _fp(normalised_merchant="A")
    assert left != right


def test_version_is_exported() -> None:
    assert FINGERPRINT_VERSION == 1
