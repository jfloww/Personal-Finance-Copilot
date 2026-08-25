"""What a run cost, and refusing to guess when we do not know.

`Usage` counts calls, tokens, and latency without knowing what a model is. The
price table is what turns that into money, and it is the piece most able to
report a confident wrong number - so these tests exist to pin it.
"""

from __future__ import annotations

from decimal import Decimal

from run_evaluation import PRICES, prices_for

from offerdelta.evaluation.usage import Usage


def test_a_model_is_priced_at_its_own_rate() -> None:
    """A single pair of constants priced every model identically, so a cheaper
    model was reported costing what a dearer one would have."""
    assert prices_for("claude-haiku-4-5") == (Decimal(1), Decimal(5))
    assert prices_for("claude-sonnet-5") == (Decimal(3), Decimal(15))


def test_an_unknown_model_is_not_priced_at_all() -> None:
    """A wrong cost is worse than no cost: it reads as measured. A provider
    added later must be given rates deliberately, never inherit someone else's."""
    assert prices_for("gpt-5.6-luna") == (None, None)
    assert prices_for("stand-in(no-key)") == (None, None)


def test_every_priced_model_has_both_rates() -> None:
    for model, (input_price, output_price) in PRICES.items():
        assert input_price > 0, model
        assert output_price > 0, model


def test_output_is_dearer_than_input_for_every_model() -> None:
    """True of every model this project can reach. A pair the other way round
    is a transposed entry, which would under-report every run."""
    for model, (input_price, output_price) in PRICES.items():
        assert output_price > input_price, model


def test_cost_uses_both_token_directions() -> None:
    usage = Usage(calls=1, input_tokens=1_000_000, output_tokens=1_000_000)
    assert usage.cost(input_per_million=Decimal(1), output_per_million=Decimal(5)) == Decimal(6)


def test_cost_is_none_without_rates() -> None:
    """The honest answer when the model is not in the table."""
    usage = Usage(calls=1, input_tokens=1000, output_tokens=100)
    assert usage.cost(input_per_million=None, output_per_million=None) is None


def test_tokens_are_counted_separately() -> None:
    usage = Usage(calls=2, input_tokens=900, output_tokens=80)
    assert usage.input_tokens == 900
    assert usage.output_tokens == 80
    assert usage.total_tokens == 980


def test_mean_latency_is_reported_alongside_the_percentiles() -> None:
    """A mean is what a total-runtime estimate needs and what people expect. It
    is also the one a single slow call drags, so it never travels alone."""
    usage = Usage(calls=4, latencies_ms=(100, 200, 300, 1000))
    assert usage.mean_latency_ms == 400
    assert usage.p50_latency_ms == 200


def test_mean_latency_is_none_when_nothing_was_timed() -> None:
    """Zero would read as an instantaneous system rather than an unmeasured one."""
    assert Usage(calls=0).mean_latency_ms is None


def test_the_mean_is_rounded_rather_than_truncated() -> None:
    # 32/3 = 10.67. Truncation would report 10 and quietly understate every
    # latency in the report by up to a millisecond per call.
    assert Usage(calls=3, latencies_ms=(10, 11, 11)).mean_latency_ms == 11
