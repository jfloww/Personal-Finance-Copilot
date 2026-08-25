"""Sorting failures into modes, and never quoting a transaction while doing it.

Two things are being protected here. The buckets have to mean what their names
say, because each one implies a different fix and a miscounted bucket sends the
next change in the wrong direction. And the sanitized output has to stay
sanitized: it is the half that gets published, and a merchant reaching it is the
failure this whole layer exists to prevent.
"""

from __future__ import annotations

from decimal import Decimal

from analyse_failures import (
    CONFIDENT,
    THIN_DESCRIPTION,
    UNSURE,
    Bucket,
    RowFacts,
    bucket_failures,
    redact,
    summarise,
)

from offerdelta.evaluation.predictions import RecordedPrediction


def _p(
    transaction_id: str,
    predicted: str,
    gold: str,
    confidence: str = "0.9",
    *,
    system: str = "llm:test",
    abstained: bool = False,
    agreed: bool | None = True,
) -> RecordedPrediction:
    return RecordedPrediction(
        transaction_id=transaction_id,
        system=system,
        predicted=predicted,
        confidence=Decimal(confidence),
        abstained=abstained,
        reason="",
        gold=gold,
        acceptable=frozenset(),
        annotators_agreed=agreed,
    )


def _facts(transaction_id: str, length: int = 40, merchant_labels: int = 1) -> RowFacts:
    return RowFacts(
        transaction_id=transaction_id,
        description_length=length,
        word_count=4,
        digit_share=0.0,
        merchant_label_count=merchant_labels,
    )


def _buckets(
    llm: list[RecordedPrediction],
    rules: list[RecordedPrediction] | None = None,
    facts: dict[str, RowFacts] | None = None,
) -> dict[str, Bucket]:
    by_system = {"llm:test": {p.transaction_id: p for p in llm}}
    if rules is not None:
        by_system["rules"] = {p.transaction_id: p for p in rules}
    resolved = facts or {p.transaction_id: _facts(p.transaction_id) for p in llm}
    return {b.name: b for b in bucket_failures(by_system, resolved, "llm:test", "rules")}


# --- Redaction --------------------------------------------------------------


def test_redaction_removes_every_letter_and_digit() -> None:
    shape = redact("KROGER #631 PURCHASE")
    assert "KROGER" not in shape
    assert "631" not in shape
    assert shape == "XXXXXX #999 XXXXXXXX"


def test_redaction_keeps_the_shape_that_makes_it_useful() -> None:
    """The whole reason for masking rather than dropping: a store number and a
    long payment memo are different failure stories, and the shape says which."""
    assert redact("SQ *COFFEE") == "XX *XXXXXX"
    assert redact("ACH PMT 20260101 REF 99") == "XXX XXX 99999999 XXX 99"


def test_redaction_never_lengthens_or_shortens_a_description() -> None:
    for text in ("A", "SHORT ONE", "a much longer description 12345 with digits"):
        assert len(redact(text)) == len(text.strip())


# --- Buckets ----------------------------------------------------------------


def test_a_confident_wrong_answer_is_counted_as_confident() -> None:
    buckets = _buckets([_p("a", "TRANSFER", "REFUND", str(CONFIDENT))])
    assert len(buckets["confident_but_wrong"].rows) == 1
    assert len(buckets["abstention_would_have_been_better"].rows) == 0


def test_an_unsure_wrong_answer_is_counted_as_one_that_should_have_abstained() -> None:
    below = str(UNSURE - Decimal("0.01"))
    buckets = _buckets([_p("a", "TRANSFER", "REFUND", below)])
    assert len(buckets["abstention_would_have_been_better"].rows) == 1
    assert len(buckets["confident_but_wrong"].rows) == 0


def test_a_correct_answer_is_in_no_failure_bucket_however_confident() -> None:
    buckets = _buckets([_p("a", "REFUND", "REFUND", "0.99")])
    for name in ("confident_but_wrong", "abstention_would_have_been_better", "thin_description"):
        assert len(buckets[name].rows) == 0, name


def test_an_abstention_is_not_counted_as_a_wrong_answer() -> None:
    """It is lost coverage, not an error. Counting it as both would double-count
    the only behaviour the system explicitly wants to reward."""
    buckets = _buckets([_p("a", "UNKNOWN", "REFUND", "0", abstained=True)])
    assert len(buckets["abstained"].rows) == 1
    assert len(buckets["confident_but_wrong"].rows) == 0
    assert len(buckets["abstention_would_have_been_better"].rows) == 0


def test_the_baseline_being_right_is_only_counted_when_it_actually_answered() -> None:
    """An abstaining rule did not 'get it right'. Counting it as such would
    argue for a hybrid on evidence that does not exist."""
    llm = [_p("a", "TRANSFER", "REFUND"), _p("b", "TRANSFER", "REFUND")]
    rules = [
        _p("a", "REFUND", "REFUND", system="rules"),
        _p("b", "UNKNOWN", "REFUND", "0", system="rules", abstained=True),
    ]
    buckets = _buckets(llm, rules)
    assert len(buckets["rules_right_model_wrong"].rows) == 1


def test_what_the_model_buys_is_counted_over_its_correct_rows() -> None:
    llm = [_p("a", "REFUND", "REFUND"), _p("b", "REFUND", "REFUND")]
    rules = [
        _p("a", "TRANSFER", "REFUND", system="rules"),
        _p("b", "REFUND", "REFUND", system="rules"),
    ]
    buckets = _buckets(llm, rules)
    assert len(buckets["model_right_rules_wrong"].rows) == 1


def test_a_thin_description_is_measured_from_the_row_not_the_prediction() -> None:
    facts = {
        "a": _facts("a", length=THIN_DESCRIPTION - 1),
        "b": _facts("b", length=THIN_DESCRIPTION),
    }
    buckets = _buckets([_p("a", "TRANSFER", "REFUND"), _p("b", "TRANSFER", "REFUND")], facts=facts)
    assert len(buckets["thin_description"].rows) == 1


def test_a_merchant_with_one_gold_label_is_not_called_polysemous() -> None:
    facts = {"a": _facts("a", merchant_labels=1), "b": _facts("b", merchant_labels=2)}
    buckets = _buckets([_p("a", "TRANSFER", "REFUND"), _p("b", "TRANSFER", "REFUND")], facts=facts)
    assert len(buckets["polysemous_merchant"].rows) == 1


def test_rows_without_a_second_annotator_are_not_counted_as_disagreements() -> None:
    rows = [
        _p("a", "TRANSFER", "REFUND", agreed=None),
        _p("b", "TRANSFER", "REFUND", agreed=True),
        _p("c", "TRANSFER", "REFUND", agreed=False),
    ]
    buckets = _buckets(rows)
    assert len(buckets["annotators_disagreed_too"].rows) == 1


# --- The published half -----------------------------------------------------


def test_the_summary_carries_counts_and_label_pairs_only() -> None:
    """Anything else in here would be on a public page."""
    llm = [_p("a", "TRANSFER", "REFUND"), _p("b", "REFUND", "REFUND")]
    by_system = {"llm:test": {p.transaction_id: p for p in llm}}
    facts = {p.transaction_id: _facts(p.transaction_id) for p in llm}
    buckets = bucket_failures(by_system, facts, "llm:test", "rules")

    summary = summarise({"split": "holdout"}, by_system, buckets, "llm:test")

    assert summary["rows"] == 2
    assert summary["correct"] == 1
    assert summary["wrong"] == 1

    modes = summary["failure_modes"]
    assert isinstance(modes, list)
    for mode in modes:
        assert isinstance(mode, dict)
        assert set(mode) == {"name", "what_it_means", "rows", "top_confusions"}
        for confusion in mode["top_confusions"]:
            # A label pair, never a description.
            assert " -> " in confusion["pair"]
            assert confusion["pair"].replace(" -> ", "").replace("_", "").isalpha()


def test_confidence_is_summarised_separately_for_right_and_wrong_answers() -> None:
    """The two numbers only mean something apart. Equal means the confidence
    signal carries no information and routing on it would be theatre."""
    llm = [
        _p("a", "REFUND", "REFUND", "0.9"),
        _p("b", "TRANSFER", "REFUND", "0.5"),
    ]
    by_system = {"llm:test": {p.transaction_id: p for p in llm}}
    facts = {p.transaction_id: _facts(p.transaction_id) for p in llm}
    buckets = bucket_failures(by_system, facts, "llm:test", "rules")

    summary = summarise({}, by_system, buckets, "llm:test")
    assert summary["mean_confidence_when_right"] == 0.9
    assert summary["mean_confidence_when_wrong"] == 0.5


def test_confidence_means_are_none_rather_than_zero_when_there_is_nothing_to_average() -> None:
    """Zero would read as a measured, perfectly-uncalibrated model."""
    llm = [_p("a", "REFUND", "REFUND", "0.9")]
    by_system = {"llm:test": {p.transaction_id: p for p in llm}}
    facts = {"a": _facts("a")}
    buckets = bucket_failures(by_system, facts, "llm:test", "rules")

    summary = summarise({}, by_system, buckets, "llm:test")
    assert summary["mean_confidence_when_wrong"] is None
