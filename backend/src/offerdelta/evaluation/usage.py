"""What a system spent producing its answers.

Kept deliberately free of any provider's vocabulary. The report knows about
calls, tokens and latency; it does not know what a model is. A system that has
no cost — the rule baseline — simply reports nothing, and the report says so
rather than printing zeros that look like a measurement.

Cost per transaction sits beside F1 because a system that wins by two points at
forty times the cost has not won, and a report that omits the price cannot say
that.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol, runtime_checkable

#: Tokens are priced per million by every provider worth naming.
TOKENS_PER_PRICE_UNIT = Decimal(1_000_000)


@dataclass(frozen=True)
class Usage:
    """Calls, tokens, and latency for one system over one run."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latencies_ms: tuple[int, ...] = ()

    #: Provider errors that became abstentions rather than crashes.
    failures: int = 0

    #: Answers discarded for naming a label outside the taxonomy. The count
    #: that makes an injection attempt visible instead of silently becoming an
    #: abstention.
    rejected_outputs: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def mean_latency_ms(self) -> int | None:
        """The average wait per call.

        Reported beside p50 and p95 rather than instead of them. A mean is the
        number people expect and the one a total-runtime estimate needs; it is
        also the one a single slow call can drag, which is why it never travels
        alone here.
        """
        if not self.latencies_ms:
            return None
        return round(sum(self.latencies_ms) / len(self.latencies_ms))

    @property
    def p50_latency_ms(self) -> int | None:
        return self._percentile(50)

    @property
    def p95_latency_ms(self) -> int | None:
        """The number that decides whether this is usable interactively.

        A mean hides the tail, and the tail is what a person waiting on a page
        actually experiences.
        """
        return self._percentile(95)

    def _percentile(self, percentile: int) -> int | None:
        if not self.latencies_ms:
            return None
        ordered = sorted(self.latencies_ms)
        # Nearest-rank: no interpolation, so the value reported is one that was
        # actually observed rather than one between two observations.
        rank = max(1, (percentile * len(ordered) + 99) // 100)
        return ordered[min(rank, len(ordered)) - 1]

    def cost(
        self, *, input_per_million: Decimal | None, output_per_million: Decimal | None
    ) -> Decimal | None:
        """Total spend, or None when no prices were supplied.

        None rather than zero: an unpriced run and a free run are different
        facts, and printing 0.00 for the first would be a lie.
        """
        if input_per_million is None or output_per_million is None:
            return None
        return (
            Decimal(self.input_tokens) * input_per_million
            + Decimal(self.output_tokens) * output_per_million
        ) / TOKENS_PER_PRICE_UNIT


@runtime_checkable
class ReportsUsage(Protocol):
    """A system that can account for what it spent.

    Optional by design: the rule baseline implements nothing and the harness
    records no usage for it, which is the honest outcome.
    """

    def usage(self) -> Usage: ...
