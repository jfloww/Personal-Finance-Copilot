from __future__ import annotations

import pytest

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.transactions.accounts import canonical_account_key


@pytest.mark.parametrize(
    "raw",
    ["checking", "Checking", "CHECKING", "  checking  ", "cHeCkInG"],
)
def test_case_and_padding_collapse_to_one_key(raw: str) -> None:
    """The bug: --account=Checking and --account=checking were two accounts."""
    assert canonical_account_key(raw) == "checking"


def test_words_join_with_hyphens() -> None:
    assert canonical_account_key("Chase Checking") == "chase-checking"
    assert canonical_account_key("Chase   Checking") == "chase-checking"


def test_punctuation_becomes_a_separator() -> None:
    assert canonical_account_key("Amex (Gold)") == "amex-gold"
    assert canonical_account_key("Chase - Checking") == "chase-checking"


def test_digits_survive() -> None:
    assert canonical_account_key("Checking 1234") == "checking-1234"


@pytest.mark.parametrize("raw", ["", "   ", "!!!", "---"])
def test_a_key_with_no_content_is_refused(raw: str) -> None:
    with pytest.raises(ValidationError, match="account name"):
        canonical_account_key(raw)


def test_it_is_idempotent() -> None:
    once = canonical_account_key("Chase Checking")
    assert canonical_account_key(once) == once


def test_an_accent_is_not_folded_away() -> None:
    """Café Checking and Caf Checking are different accounts."""
    assert canonical_account_key("Café Checking") != canonical_account_key("Caf Checking")
    assert canonical_account_key("Café Checking") == "café-checking"


def test_a_non_latin_name_is_accepted() -> None:
    """A name in another script is a real name, not an empty one."""
    assert canonical_account_key("저축") == "저축"


def test_composed_and_decomposed_forms_agree() -> None:
    """NFKC: the same name typed two ways is one account."""
    composed = "Café"  # é as one codepoint
    decomposed = "Café"  # e + combining acute
    assert canonical_account_key(composed) == canonical_account_key(decomposed)


def test_non_latin_case_variants_still_collapse() -> None:
    assert canonical_account_key("СБЕР") == canonical_account_key("сбер")  # noqa: RUF001
