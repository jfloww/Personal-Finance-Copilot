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

from pydantic import BaseModel, ConfigDict, Field

from offerdelta.domain.comparisons.derivation import DerivationNode


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


class TransactionStoredSchema(BaseModel):
    """What was written."""

    model_config = ConfigDict(frozen=True)

    transaction_id: str
    fingerprint: str
    fingerprint_version: int
    occurrence: int
