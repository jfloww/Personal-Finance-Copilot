"""The new client driving the existing categoriser, with nothing mocked between.

Every other test in this package stops at the provider boundary. These run the
whole path - HTTP response, parsing, taxonomy validation, prediction - because
that seam is where a working client and a working harness can still fail to fit
together.

The seam also carries the guarantee the whole design rests on: whatever a model
is persuaded to say, only a label already in the taxonomy reaches a number
anyone sees.
"""

from __future__ import annotations

import json
from datetime import date
from decimal import Decimal

from offerdelta.domain.common.money import Money
from offerdelta.domain.transactions.view import TransactionView
from offerdelta.evaluation.dataset import LabelledTransaction
from offerdelta.evaluation.labels import ABSTAIN, LABEL_SPACE
from offerdelta.evaluation.llm_categoriser import LLMCategoriser
from offerdelta.infrastructure.llm.anthropic import AnthropicConfig, AnthropicProvider
from offerdelta.infrastructure.llm.retry import FakeClock, RetryPolicy
from offerdelta.infrastructure.llm.transport import FakeTransport, json_response

FAST = RetryPolicy(max_attempts=2, base_delay_s=1.0, max_delay_s=1.0, jitter=False)


def _tool_call(label: str, confidence: float = 0.9) -> str:
    return json.dumps(
        {
            "stop_reason": "tool_use",
            "content": [
                {
                    "type": "tool_use",
                    "name": "categorise_transaction",
                    "input": {"label": label, "confidence": confidence, "reason": "because"},
                }
            ],
            "usage": {"input_tokens": 400, "output_tokens": 30},
        }
    )


def _categoriser(*responses: object) -> tuple[LLMCategoriser, FakeTransport]:
    transport = FakeTransport(responses=list(responses))  # type: ignore[arg-type]
    provider = AnthropicProvider(
        config=AnthropicConfig(api_key="sk-test"),
        transport=transport,
        retry_policy=FAST,
        clock=FakeClock(),
    )
    return LLMCategoriser(provider=provider), transport


def _record() -> TransactionView:
    return LabelledTransaction(
        transaction_id="t1",
        posted_on=date(2026, 8, 1),
        raw_description="BLUE BOTTLE COFFEE #417",
        normalised_merchant="BLUE BOTTLE",
        amount=Money.parse("-4.50"),
        account_type="credit_card",
        source="test",
        bank_format="test_csv",
        primary_label="LIVING_DINING",
    ).view


def test_a_valid_label_reaches_the_prediction() -> None:
    categoriser, _ = _categoriser(json_response(200, _tool_call("LIVING_DINING")))

    prediction = categoriser.predict(_record())

    assert prediction.label == "LIVING_DINING"
    assert prediction.confidence == Decimal("0.9")
    assert not prediction.abstained


def test_a_label_outside_the_taxonomy_becomes_an_abstention() -> None:
    """The structural defence, end to end.

    A compromised or confused model returning `TOTALLY_MADE_UP` must not put a
    new category into anyone's budget. It becomes an abstention, which shows up
    in coverage rather than in the error rate.
    """
    categoriser, _ = _categoriser(json_response(200, _tool_call("TOTALLY_MADE_UP")))

    prediction = categoriser.predict(_record())

    assert prediction.abstained
    assert "TOTALLY_MADE_UP" in prediction.reason
    assert categoriser.rejected_labels == ["TOTALLY_MADE_UP"]


def test_an_outage_becomes_an_abstention_rather_than_a_crash() -> None:
    # An outage is not a licence to guess, and it must not take down a batch.
    categoriser, transport = _categoriser(
        json_response(503, "upstream down"),
        json_response(503, "upstream down"),
    )

    prediction = categoriser.predict(_record())

    assert prediction.abstained
    assert categoriser.provider_failures == 1
    assert len(transport.requests) == 2  # it did retry first


def test_a_rejected_key_becomes_an_abstention_too() -> None:
    # Wrong, but the harness must survive it and report it rather than crash
    # halfway through a 400-row benchmark.
    categoriser, transport = _categoriser(json_response(401, '{"error":"bad key"}'))

    prediction = categoriser.predict(_record())

    assert prediction.abstained
    assert categoriser.provider_failures == 1
    assert len(transport.requests) == 1


def test_token_usage_flows_into_the_reports_vocabulary() -> None:
    # Cost per transaction is reported beside F1, so it has to arrive here.
    categoriser, _ = _categoriser(
        json_response(200, _tool_call("LIVING_DINING")),
        json_response(200, _tool_call("LIVING_GROCERY")),
    )

    categoriser.predict(_record())
    categoriser.predict(_record())
    usage = categoriser.usage()

    assert usage.calls == 2
    assert usage.input_tokens == 800
    assert usage.output_tokens == 60
    assert usage.failures == 0


def test_a_failed_call_is_counted_but_not_billed_as_a_successful_one() -> None:
    categoriser, _ = _categoriser(
        json_response(200, _tool_call("LIVING_DINING")),
        json_response(401, "bad key"),
    )

    categoriser.predict(_record())
    categoriser.predict(_record())
    usage = categoriser.usage()

    assert usage.calls == 1
    assert usage.failures == 1


def test_every_label_in_the_taxonomy_survives_the_round_trip() -> None:
    """No label is mangled between the wire and the prediction.

    Cheap to assert and it would catch a normalisation slip - a stray `.lower()`
    or `.strip()` - across the whole space rather than the one label a fixture
    happens to use.
    """
    for label in sorted(LABEL_SPACE):
        categoriser, _ = _categoriser(json_response(200, _tool_call(label)))

        prediction = categoriser.predict(_record())

        assert prediction.label == label, f"{label} did not survive"
        # UNKNOWN is excluded because it is not an ordinary label: see below.
        if label != ABSTAIN:
            assert not prediction.abstained


def test_the_model_can_decline_by_choosing_unknown() -> None:
    """`UNKNOWN` is in the enum the model is given, and it is also the abstention.

    That is deliberate rather than an overlap to tidy away. It gives the model a
    way to say "I do not know" inside the schema, instead of forcing it to pick
    the least-wrong category and inflating confident errors - and the abstention
    lands in coverage, where a reviewer will see it.
    """
    categoriser, _ = _categoriser(json_response(200, _tool_call(ABSTAIN)))

    prediction = categoriser.predict(_record())

    assert prediction.abstained
    # Not a rejection: the model used the taxonomy correctly.
    assert categoriser.rejected_labels == []
