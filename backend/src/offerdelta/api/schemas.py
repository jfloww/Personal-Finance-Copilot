"""Wire formats.

Every monetary value crosses the boundary as a **string**, never a JSON number.

JavaScript has one numeric type, an IEEE 754 double, so `4217.33` becomes an
approximation the instant a browser parses it as a number. Rendering alone
usually rounds back correctly and hides the problem; the first client-side
subtotal exposes it, in a product whose entire premise is that its numbers can
be trusted.

Typing the field as `str` makes emitting a number structurally impossible,
which is a stronger guarantee than configuring a serializer that a future
library version might change.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from offerdelta.application.queries.observed_debits import ObservedDebitComparison
from offerdelta.application.queries.spend_change import (
    CurrencySpendChange,
    MerchantSpendDelta,
    SpendChange,
    SpendEvidence,
    SpendPeriod,
)
from offerdelta.application.reports.monthly import MonthCoverage, MonthlyReport
from offerdelta.domain.common.derivation import DerivationNode
from offerdelta.evaluation.labels import ABSTAIN, LABEL_SPACE


class DemoInvestigationRequest(BaseModel):
    """Only bundled synthetic CSV scenarios are exposed; arbitrary queries are not accepted."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    scenario: Literal["august_software_exceptions", "alternate_billing_review"]


class DerivationNodeSchema(BaseModel):
    """One step in the explanation of a calculated figure."""

    model_config = ConfigDict(frozen=True)

    code: str
    label: str
    amount: str = Field(description="Exact decimal string. Never parse this as a number.")
    currency: str
    period: str
    formula: str
    evidence: str
    children: tuple[DerivationNodeSchema, ...] = ()

    @classmethod
    def of(cls, node: DerivationNode) -> DerivationNodeSchema:
        return cls(
            code=node.code,
            label=node.label,
            amount=str(node.amount.amount),
            currency=node.amount.currency,
            period=str(node.period),
            formula=node.formula,
            evidence=str(node.evidence),
            children=tuple(cls.of(child) for child in node.children),
        )


class ComparisonRequest(BaseModel):
    """What to compare, and over how long."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon_months: int = Field(default=12, ge=1, le=120, description="Whole months to project.")

    #: Null runs the comparison without inheriting pre-move costs — the
    #: behaviour before that gap was closed, kept reproducible rather than only
    #: described.
    move_date: date | None = Field(default=date(2026, 7, 1))


class ComponentDeltaSchema(BaseModel):
    """One line of the breakdown. All amounts annualised, all as strings."""

    code: str
    label: str
    current: str
    candidate: str
    delta: str


class BreakEvenSchema(BaseModel):
    """Two months, because one is misleading when an offer front-loads cash."""

    metric: str
    horizon_months: int
    first_crossing_month: int | None
    stable_break_even_month: int | None


class EquivalentSalarySchema(BaseModel):
    equivalent_salary: str
    target_metric: str
    tax_model: str
    calibration_distance_percent: str
    is_far_from_calibration: bool
    converged: bool
    iterations: int


class NegotiationOptionSchema(BaseModel):
    lever: str
    feasible: bool
    note: str
    required_amount: str | None = None
    required_days: str | None = None


class NegotiationSchema(BaseModel):
    gap: str
    needs_negotiation: bool
    options: tuple[NegotiationOptionSchema, ...]


class ComparisonSchema(BaseModel):
    """A full comparison, ready to render."""

    current_label: str
    candidate_label: str
    horizon_months: int
    currency: str

    cash_delta: str
    wealth_delta: str
    time_delta_hours: str

    cumulative_cash_delta: tuple[str, ...]
    component_deltas: tuple[ComponentDeltaSchema, ...]

    current_derivation: DerivationNodeSchema
    candidate_derivation: DerivationNodeSchema

    break_even: BreakEvenSchema
    equivalent_salary: EquivalentSalarySchema | None = None
    equivalent_salary_error: str | None = None
    negotiation: NegotiationSchema | None = None
    negotiation_error: str | None = None

    #: True when every projected month on both sides balanced. The engine
    #: refuses to return an unbalanced result, so this is always true — it is
    #: surfaced so a reader can see the guarantee rather than take it on trust.
    reconciled: bool


class VersionSchema(BaseModel):
    service: str
    version: str
    engine: str


class HealthSchema(BaseModel):
    """The process is alive. Deliberately says nothing else.

    Liveness does not check the database, so it does not report on one -
    a field carrying an unchecked default would be a small lie in the one
    place a monitoring system trusts absolutely.
    """

    status: str


class ReadinessSchema(BaseModel):
    """The service can serve traffic, and what it cannot serve.

    `status` stays "ready" while the demo endpoints work: that is what the
    platform health check asks, and taking a working demo offline would be a
    strange way to report a missing database. `database` carries that news
    instead - a probe reporting "ready" while silent about an absent database
    asserts something untrue.
    """

    status: str
    database: str


class LoginSchema(BaseModel):
    """Credentials as they arrive on the wire."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    email: str = Field(min_length=1)
    password: str = Field(min_length=1)


class TokenSchema(BaseModel):
    """One bearer access token."""

    model_config = ConfigDict(frozen=True)

    access_token: str


class TransactionEntrySchema(BaseModel):
    """One hand-entered transaction, as it arrives on the wire.

    `amount` is a string for the reason at the top of this module: a browser
    parsing `-4.50` as a JSON number turns an exact decimal into an IEEE 754
    approximation, and this is a product whose premise is that its numbers can
    be trusted. It is parsed with the same `parse_amount` the CSV importer
    uses, so `$1,234.56` and `(45.00)` mean here what they mean there.
    """

    model_config = ConfigDict(frozen=True)

    account_key: str = Field(min_length=1)
    posted_on: date
    description: str = Field(min_length=1)
    amount: str = Field(description="Signed decimal string. Negative is money out.")
    repeat: bool = Field(
        default=False,
        description=(
            "Assert that this really is another identical charge rather than a "
            "re-entry of one already stored. Without it, a matching transaction "
            "is reported and nothing is written."
        ),
    )

    @field_validator("description")
    @classmethod
    def _description_is_not_blank(cls, value: str) -> str:
        """Reject whitespace-only text here, at the wire boundary.

        `min_length=1` alone lets `"   "` through - it is one character or
        more, just none of them meaningful. Stripping and re-checking here
        means a blank description is a 422 from bad input, not a
        `ValidationError` that reaches `enter_transaction` and gets mapped to
        404 alongside an unknown account - the only failure that handler's
        `except ValidationError` is meant to answer for.
        """
        stripped = value.strip()
        if not stripped:
            raise ValueError("a transaction needs a description")
        return stripped


class TransactionStoredSchema(BaseModel):
    """What was written."""

    model_config = ConfigDict(frozen=True)

    transaction_id: str
    fingerprint: str
    fingerprint_version: int
    occurrence: int


class MonthCoverageSchema(BaseModel):
    """How much of one calendar month this tenant's stored rows account for.

    Mirrors `offerdelta.application.reports.monthly.MonthCoverage` field for
    field - see that type's docstring for what `complete` and
    `awaiting_review` do and do not claim.
    """

    model_config = ConfigDict(frozen=True)

    year: int
    month: int
    complete: bool
    rows: int
    classified: int
    awaiting_review: int

    @classmethod
    def of(cls, coverage: MonthCoverage) -> MonthCoverageSchema:
        return cls(
            year=coverage.year,
            month=coverage.month,
            complete=coverage.complete,
            rows=coverage.rows,
            classified=coverage.classified,
            awaiting_review=coverage.awaiting_review,
        )


class MonthlyReportSchema(BaseModel):
    """One month's tree, plus how much of the month it could account for.

    `tree` reuses `DerivationNodeSchema` - the same serialiser
    `/v1/demo/derivation` uses - rather than a second one, so the "amounts
    are decimal strings, never numbers" guarantee and its contract test cover
    this tree too, without restating either.
    """

    model_config = ConfigDict(frozen=True)

    tree: DerivationNodeSchema
    coverage: MonthCoverageSchema

    @classmethod
    def of(cls, report: MonthlyReport) -> MonthlyReportSchema:
        return cls(
            tree=DerivationNodeSchema.of(report.tree),
            coverage=MonthCoverageSchema.of(report.coverage),
        )


class MerchantDebitDeltaSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    merchant: str
    previous: str
    current: str
    delta: str


class CurrencyDebitComparisonSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str
    previous: str
    current: str
    delta: str
    merchants: list[MerchantDebitDeltaSchema]


class ObservedDebitComparisonSchema(BaseModel):
    """Raw debit evidence, not classified spending or a duplicate-charge verdict."""

    model_config = ConfigDict(frozen=True)

    previous_coverage: MonthCoverageSchema
    current_coverage: MonthCoverageSchema
    currencies: list[CurrencyDebitComparisonSchema]
    caveat: str = (
        "Observed debits may include transfers and unclassified transactions. "
        "Compare only months whose coverage.complete is true."
    )

    @classmethod
    def of(cls, comparison: ObservedDebitComparison) -> ObservedDebitComparisonSchema:
        return cls(
            previous_coverage=MonthCoverageSchema.of(comparison.previous_coverage),
            current_coverage=MonthCoverageSchema.of(comparison.current_coverage),
            currencies=[
                CurrencyDebitComparisonSchema(
                    currency=group.currency,
                    previous=str(group.previous),
                    current=str(group.current),
                    delta=str(group.delta),
                    merchants=[
                        MerchantDebitDeltaSchema(
                            merchant=row.merchant,
                            previous=str(row.previous),
                            current=str(row.current),
                            delta=str(row.delta),
                        )
                        for row in group.merchants
                    ],
                )
                for group in comparison.currencies
            ],
        )


class SpendEvidenceSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    transaction_id: str
    posted_on: date
    contribution: str = Field(description="Signed exact decimal contribution to net spending.")
    label: str
    confirmed: bool

    @classmethod
    def of(cls, row: SpendEvidence) -> SpendEvidenceSchema:
        return cls(
            transaction_id=str(row.transaction_id),
            posted_on=row.posted_on,
            contribution=str(row.contribution),
            label=row.label,
            confirmed=row.confirmed,
        )


class SpendPeriodSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    gross_spending: str
    refunds: str
    net_spending: str
    unclassified_debits: str
    transfer_debits: str
    inconsistent_debits: str
    unclassified_rows: int
    suggested_rows: int
    inconsistent_rows: int

    @classmethod
    def of(cls, period: SpendPeriod) -> SpendPeriodSchema:
        return cls(
            gross_spending=str(period.gross_spending),
            refunds=str(period.refunds),
            net_spending=str(period.net_spending),
            unclassified_debits=str(period.unclassified_debits),
            transfer_debits=str(period.transfer_debits),
            inconsistent_debits=str(period.inconsistent_debits),
            unclassified_rows=period.unclassified_rows,
            suggested_rows=period.suggested_rows,
            inconsistent_rows=period.inconsistent_rows,
        )


class MerchantSpendDeltaSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    merchant: str
    previous: str
    current: str
    delta: str
    previous_evidence: list[SpendEvidenceSchema]
    current_evidence: list[SpendEvidenceSchema]

    @classmethod
    def of(cls, row: MerchantSpendDelta) -> MerchantSpendDeltaSchema:
        return cls(
            merchant=row.merchant,
            previous=str(row.previous),
            current=str(row.current),
            delta=str(row.delta),
            previous_evidence=[SpendEvidenceSchema.of(item) for item in row.previous_evidence],
            current_evidence=[SpendEvidenceSchema.of(item) for item in row.current_evidence],
        )


class CurrencySpendChangeSchema(BaseModel):
    model_config = ConfigDict(frozen=True)

    currency: str
    previous: SpendPeriodSchema
    current: SpendPeriodSchema
    delta: str
    merchants: list[MerchantSpendDeltaSchema]

    @classmethod
    def of(cls, group: CurrencySpendChange) -> CurrencySpendChangeSchema:
        return cls(
            currency=group.currency,
            previous=SpendPeriodSchema.of(group.previous),
            current=SpendPeriodSchema.of(group.current),
            delta=str(group.delta),
            merchants=[MerchantSpendDeltaSchema.of(row) for row in group.merchants],
        )


class SpendChangeSchema(BaseModel):
    """An auditable labelled-spend comparison, never a model-generated verdict."""

    model_config = ConfigDict(frozen=True)

    previous_coverage: MonthCoverageSchema
    current_coverage: MonthCoverageSchema
    currencies: list[CurrencySpendChangeSchema]
    caveat: str = (
        "Net spending uses category-labelled debits minus labelled refunds. "
        "Suggested labels are provisional; inspect suggested_rows, unclassified_rows, "
        "inconsistent_rows, and both coverage.complete flags before interpreting the change."
    )

    @classmethod
    def of(cls, comparison: SpendChange) -> SpendChangeSchema:
        return cls(
            previous_coverage=MonthCoverageSchema.of(comparison.previous_coverage),
            current_coverage=MonthCoverageSchema.of(comparison.current_coverage),
            currencies=[CurrencySpendChangeSchema.of(group) for group in comparison.currencies],
        )


class ReviewQueueRowSchema(BaseModel):
    """One stored row awaiting a person's decision.

    Deliberately narrower than the full `StoredTransaction` a repository
    returns: no account id, batch id, or raw source cells - a review queue
    exists to collect a label, not to double as an import audit trail, and
    every field kept here is one this route has to be able to justify handing
    back to whichever tenant is asking.
    """

    model_config = ConfigDict(frozen=True)

    transaction_id: str
    posted_on: date
    description: str
    amount: str = Field(description="Exact decimal string. Never parse this as a number.")
    suggested_label: str | None
    suggested_confidence: str | None = Field(
        default=None,
        description="Exact decimal string when present. Never parse this as a number.",
    )


class LabelConfirmationSchema(BaseModel):
    """A confirmed label, as it arrives on the wire.

    `label` is checked against the taxonomy here, at the wire boundary,
    rather than left for `TransactionRepository.confirm_label` to reject.
    That repository method raises the same `ValidationError` - and the same
    message - for an unknown label as it does for a transaction id naming
    another tenant's row, because the two must read identically to a caller
    probing for one tenant's data through another's session. A route that let
    an invalid label reach that method would have to guess, from a shared
    message, whether to answer 422 or 404; rejecting it here means the only
    `ValidationError` left for the route to catch is the tenancy one, so it
    can map that to 404 without ambiguity. Compare
    `TransactionEntrySchema._description_is_not_blank`, which exists for the
    same reason on the same route family.

    `ABSTAIN` ("UNKNOWN") is in `LABEL_SPACE` - a categoriser needs to be
    able to say it - but a person cannot *confirm* an abstention: there is no
    unconfirm route, so a row confirmed as `UNKNOWN` would leave the review
    queue forever, and `_group_by_label` gives that leaf `Evidence.ASSUMED`
    regardless of `confirmed`, so the month's root could never turn
    `USER_CONFIRMED` either. Declining to label a row is done by leaving it
    alone, not by parking it here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str = Field(min_length=1)

    @field_validator("label")
    @classmethod
    def _label_is_in_the_taxonomy(cls, value: str) -> str:
        if value not in LABEL_SPACE:
            raise ValueError(f"{value!r} is not a label in this taxonomy")
        if value == ABSTAIN:
            raise ValueError(f"{value!r} cannot be confirmed; leave the row unreviewed instead")
        return value
