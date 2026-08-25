"""The comparison report.

Run entirely against scripted providers, so it needs neither the real dataset
nor an API key. The properties proved here are the ones that make the published
numbers worth anything:

- the frozen labels cannot move during a run, and the report refuses to exist
  if they did;
- ambiguous rows are reported separately *and* counted in the overall figures;
- the report knows about usage, not about models, so a system with no cost
  reports none rather than zeros that look measured.
"""

import contextlib
from datetime import date
from decimal import Decimal

import pytest

from offerdelta.domain.common.errors import ValidationError
from offerdelta.domain.common.money import Money
from offerdelta.evaluation.categorisers import Prediction
from offerdelta.evaluation.dataset import LabelledDataset, LabelledTransaction
from offerdelta.evaluation.llm_categoriser import HybridCategoriser, LLMCategoriser
from offerdelta.evaluation.providers import LLMResponse, ScriptedProvider
from offerdelta.evaluation.report import evaluate, summarise_agreement
from offerdelta.evaluation.rule_baseline import fit_rules

GROCERY = "LIVING_GROCERY"
DINING = "LIVING_DINING"
OTHER = "LIVING_OTHER"


def _record(
    txn_id: str,
    merchant: str,
    primary: str = DINING,
    *,
    secondary: str | None = None,
    final: str | None = None,
    acceptable: frozenset[str] = frozenset(),
) -> LabelledTransaction:
    return LabelledTransaction(
        transaction_id=txn_id,
        posted_on=date(2026, 8, 1),
        raw_description=merchant,
        normalised_merchant=merchant,
        amount=Money.parse("-10.00"),
        account_type="credit_card",
        source="test",
        bank_format="test_csv",
        primary_label=primary,
        secondary_label=secondary,
        adjudicated_label=final,
        acceptable_labels=acceptable,
        ambiguous=bool(acceptable),
        notes="ambiguous basket" if acceptable else None,
    )


HOLDOUT = LabelledDataset(
    dataset_version="2026.08.1+holdout",
    records=(
        _record("h1", "BLUE BOTTLE", DINING, secondary=DINING),
        _record("h2", "WHOLE FOODS", GROCERY, secondary=GROCERY),
        _record("h3", "SWEETGREEN", DINING, secondary=GROCERY, final=DINING),
        _record(
            "h4",
            "AMAZON",
            GROCERY,
            acceptable=frozenset({GROCERY, OTHER}),
        ),
    ),
)


class _Always:
    """A system that answers the same label every time."""

    def __init__(self, label: str, name: str = "always") -> None:
        self._label = label
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def predict_many(self, records):  # type: ignore[no-untyped-def]
        return [
            Prediction(label=self._label, confidence=Decimal("1"), reason="fixed") for _ in records
        ]


class _Mutator:
    """A badly behaved system that tries to rewrite the labels it is scored on."""

    @property
    def name(self) -> str:
        return "mutator"

    def predict_many(self, records):  # type: ignore[no-untyped-def]
        for record in records:
            with contextlib.suppress(Exception):
                object.__setattr__(record, "primary_label", GROCERY)
        return [Prediction(label=GROCERY, confidence=Decimal("1"), reason="fixed") for _ in records]


class _Short:
    """Returns fewer predictions than rows."""

    @property
    def name(self) -> str:
        return "short"

    def predict_many(self, records):  # type: ignore[no-untyped-def]  # noqa: ARG002
        # Deliberately ignores the rows: returning one prediction for many is
        # exactly the partial run the harness must refuse.
        return [Prediction(label=DINING, confidence=Decimal("1"), reason="x")]


# --- Frozen labels ---------------------------------------------------------


def test_a_system_that_mutates_the_dataset_is_caught() -> None:
    # The guarantee the whole report rests on. Frozen dataclasses make this
    # unlikely; checking the checksum makes it provable.
    with pytest.raises(ValidationError, match="changed during evaluation"):
        evaluate(HOLDOUT, [_Mutator()])


def test_the_report_records_the_checksum_it_ran_against() -> None:
    report = evaluate(HOLDOUT, [_Always(DINING)])
    assert report.checksum == HOLDOUT.checksum


def test_two_runs_over_the_same_data_report_the_same_checksum() -> None:
    # How a reader confirms two reports really used the same rows.
    a = evaluate(HOLDOUT, [_Always(DINING)])
    b = evaluate(HOLDOUT, [_Always(GROCERY)])
    assert a.checksum == b.checksum


def test_a_partial_run_is_refused() -> None:
    with pytest.raises(ValidationError, match="partial run"):
        evaluate(HOLDOUT, [_Short()])


def test_evaluating_no_systems_is_refused() -> None:
    with pytest.raises(ValidationError, match="at least one system"):
        evaluate(HOLDOUT, [])


# --- Strata ----------------------------------------------------------------


def test_ambiguous_rows_are_reported_separately() -> None:
    report = evaluate(HOLDOUT, [_Always(DINING)])
    result = report.systems[0]
    assert result.ambiguous is not None
    assert result.ambiguous.total == 1


def test_ambiguous_rows_remain_in_the_overall_figures() -> None:
    # Dropping them would flatter every system by removing the hardest rows.
    report = evaluate(HOLDOUT, [_Always(DINING)])
    result = report.systems[0]
    assert result.ambiguous is not None
    assert result.overall.total == len(HOLDOUT)
    assert result.overall.total == result.unambiguous.total + result.ambiguous.total


def test_an_acceptable_label_is_honoured_in_the_overall_score() -> None:
    # AMAZON accepts GROCERY or OTHER; answering OTHER everywhere should score
    # that row correct rather than wrong.
    report = evaluate(HOLDOUT, [_Always(OTHER)])
    assert report.systems[0].ambiguous is not None
    assert report.systems[0].ambiguous.accuracy == Decimal(1)


def test_a_holdout_without_ambiguous_rows_reports_none() -> None:
    plain = LabelledDataset(dataset_version="v1", records=(_record("p1", "BLUE BOTTLE"),))
    report = evaluate(plain, [_Always(DINING)])
    assert report.systems[0].ambiguous is None


def test_the_strata_use_the_same_acceptable_sets_as_the_overall_run() -> None:
    report = evaluate(HOLDOUT, [_Always(OTHER)])
    result = report.systems[0]
    assert result.ambiguous is not None
    # The ambiguous row is correct under its acceptable set in both views.
    assert result.ambiguous.correct == 1


# --- Agreement -------------------------------------------------------------


def test_agreement_is_computed_from_the_labels_alone() -> None:
    # A property of the benchmark, not of any system — and the ceiling every
    # score must be read against.
    summary = summarise_agreement(HOLDOUT)
    assert summary.double_annotated == 3
    assert summary.agreed == 2


def test_agreement_survives_adjudication() -> None:
    # h3 was adjudicated, and the original disagreement is still visible.
    summary = summarise_agreement(HOLDOUT)
    assert summary.raw_agreement is not None
    assert summary.raw_agreement < Decimal(1)


def test_kappa_is_reported_alongside_raw_agreement() -> None:
    summary = summarise_agreement(HOLDOUT)
    assert summary.kappa is not None or summary.double_annotated == 0


def test_a_dataset_with_no_second_annotator_reports_no_agreement() -> None:
    plain = LabelledDataset(dataset_version="v1", records=(_record("p1", "X"),))
    summary = summarise_agreement(plain)
    assert summary.double_annotated == 0
    assert summary.kappa is None


# --- Usage -----------------------------------------------------------------


def _response(label: str) -> LLMResponse:
    return LLMResponse(
        label=label,
        confidence=Decimal("0.9"),
        reason="scripted",
        input_tokens=100,
        output_tokens=10,
        latency_ms=250,
    )


def test_a_system_with_no_cost_reports_no_usage() -> None:
    # Zeros would look like a measurement. None says there was nothing to spend.
    rules = fit_rules(HOLDOUT)
    report = evaluate(HOLDOUT, [rules])
    assert report.systems[0].usage is None


def test_a_model_backed_system_reports_its_usage() -> None:
    llm = LLMCategoriser(ScriptedProvider(default=_response(DINING)))
    report = evaluate(HOLDOUT, [llm])
    usage = report.systems[0].usage
    assert usage is not None
    assert usage.calls == len(HOLDOUT)
    assert usage.total_tokens == 110 * len(HOLDOUT)


def test_latency_percentiles_are_reported() -> None:
    llm = LLMCategoriser(ScriptedProvider(default=_response(DINING)))
    usage = evaluate(HOLDOUT, [llm]).systems[0].usage
    assert usage is not None
    assert usage.p95_latency_ms == 250


def test_cost_is_none_when_no_prices_are_given() -> None:
    # An unpriced run and a free run are different facts.
    llm = LLMCategoriser(ScriptedProvider(default=_response(DINING)))
    usage = evaluate(HOLDOUT, [llm]).systems[0].usage
    assert usage is not None
    assert usage.cost(input_per_million=None, output_per_million=None) is None


def test_cost_is_computed_when_prices_are_given() -> None:
    llm = LLMCategoriser(ScriptedProvider(default=_response(DINING)))
    usage = evaluate(HOLDOUT, [llm]).systems[0].usage
    assert usage is not None
    # 400 input + 40 output tokens at $3 / $15 per million.
    cost = usage.cost(input_per_million=Decimal(3), output_per_million=Decimal(15))
    assert cost == Decimal("0.0018")


def test_the_hybrid_reports_only_the_calls_it_actually_made() -> None:
    # Delegating usage to the model means the escalation rate and the token
    # count cannot disagree about how often it ran.
    provider = ScriptedProvider(default=_response(DINING))
    hybrid = HybridCategoriser(fit_rules(HOLDOUT), LLMCategoriser(provider))
    usage = evaluate(HOLDOUT, [hybrid]).systems[0].usage
    assert usage is not None
    assert usage.calls == len(provider.calls)


# --- The comparison --------------------------------------------------------


def test_several_systems_are_scored_on_one_dataset() -> None:
    report = evaluate(
        HOLDOUT,
        [
            fit_rules(HOLDOUT),
            LLMCategoriser(ScriptedProvider(default=_response(DINING))),
            HybridCategoriser(
                fit_rules(HOLDOUT),
                LLMCategoriser(ScriptedProvider(default=_response(DINING))),
            ),
        ],
    )
    assert len(report.systems) == 3
    assert len({s.name for s in report.systems}) == 3


def test_the_best_system_can_be_identified() -> None:
    report = evaluate(HOLDOUT, [_Always(DINING, "weak"), _Always(GROCERY, "other")])
    assert report.best_by_macro_f1().name in {"weak", "other"}


def test_abstention_shows_as_coverage_not_as_error() -> None:
    class _Abstainer:
        name = "abstainer"

        def predict_many(self, records):  # type: ignore[no-untyped-def]
            return [Prediction.abstain("never sure") for _ in records]

    report = evaluate(HOLDOUT, [_Abstainer()])
    assert report.systems[0].overall.coverage == Decimal(0)
    assert report.systems[0].overall.accuracy_when_answered is None


def test_the_report_renders_as_text() -> None:
    report = evaluate(
        HOLDOUT,
        [fit_rules(HOLDOUT), LLMCategoriser(ScriptedProvider(default=_response(DINING)))],
        input_price_per_million=Decimal(3),
        output_price_per_million=Decimal(15),
    )
    rendered = report.render()
    assert "EVALUATION REPORT" in rendered
    assert "SUMMARY" in rendered
    assert "macro F1" in rendered
    assert "ambiguous" in rendered
    assert "Cohen's kappa" in rendered


def test_the_rendered_report_names_the_dataset_and_checksum() -> None:
    rendered = evaluate(HOLDOUT, [_Always(DINING)]).render()
    assert HOLDOUT.dataset_version in rendered
    assert HOLDOUT.checksum[:16] in rendered


def test_the_rendered_report_says_when_a_system_has_no_cost() -> None:
    rendered = evaluate(HOLDOUT, [fit_rules(HOLDOUT)]).render()
    assert "makes no external calls" in rendered


# --- Annotator agreement as a difficulty stratum ---------------------------


def test_rows_are_stratified_by_whether_the_annotators_agreed() -> None:
    """A difficulty signal, not an ambiguity claim. It answers whether the rows
    people found hard are the same rows the model finds hard - which bounds how
    much of a failure is reachable by prompting at all."""
    result = evaluate(HOLDOUT, [_Always(DINING)]).systems[0]

    assert result.annotators_agreed is not None
    assert result.annotators_disagreed is not None
    # h1 and h2 were agreed on; h3 was not. h4 has no second annotator.
    assert result.annotators_agreed.total == 2
    assert result.annotators_disagreed.total == 1


def test_a_row_with_no_second_annotator_is_in_neither_agreement_stratum() -> None:
    """Absent is not the same as disagreed. Counting it as either would invent a
    fact about an annotation that never happened."""
    result = evaluate(HOLDOUT, [_Always(DINING)]).systems[0]

    assert result.annotators_agreed is not None
    assert result.annotators_disagreed is not None
    counted = result.annotators_agreed.total + result.annotators_disagreed.total
    assert counted == len(HOLDOUT) - 1


def test_the_agreement_strata_do_not_claim_the_rows_are_ambiguous() -> None:
    """The two are reported separately on purpose: this project refuses to read
    disagreement as ambiguity, and the report must not blur them."""
    rendered = (
        evaluate(HOLDOUT, [_Always(DINING)]).systems[0].render(input_price=None, output_price=None)
    )
    assert "difficulty, not ambiguity" in rendered


# --- Predictions are retained ----------------------------------------------


def test_predictions_are_kept_so_a_run_can_be_re_scored_without_re_running() -> None:
    result = evaluate(HOLDOUT, [_Always(DINING)]).systems[0]

    assert len(result.predictions) == len(HOLDOUT)
    assert {p.label for p in result.predictions} == {DINING}


# --- The scoring policy is stated ------------------------------------------


def test_the_report_says_what_correct_meant() -> None:
    """Accuracy here is acceptable-label accuracy. With ambiguous rows present
    the two differ, and a reader must not have to guess which one they see."""
    rendered = evaluate(HOLDOUT, [_Always(DINING)]).render()
    assert "acceptable-label accuracy" in rendered
    assert "1 ambiguous rows count any label" in rendered


def test_the_report_says_when_the_two_accuracies_coincide() -> None:
    unambiguous = LabelledDataset(
        dataset_version="no-ambiguity",
        records=(_record("u1", "BLUE BOTTLE", DINING),),
    )
    rendered = evaluate(unambiguous, [_Always(DINING)]).render()
    assert "equals exact-match" in rendered
