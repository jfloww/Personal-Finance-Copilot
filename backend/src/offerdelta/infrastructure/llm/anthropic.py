"""The Anthropic adapter: a real provider behind the port the harness already uses.

Everything above this file - the categoriser, the hybrid, the metrics, the
report - was written against `LLMProvider` and does not change because this
exists. That was the point of building the port first.

Three decisions are worth stating.

**Structured output through tool use, not prose parsing.** The request pins
`tool_choice` to a single tool whose schema enumerates the valid labels, so a
free-text answer is not one of the shapes the response can take. Parsing prose
for a category means writing a parser for every way a model might phrase
"probably groceries", and being wrong about a case nobody thought of. Making
the API refuse the shape is cheaper and stricter than validating it afterwards.

**Temperature zero.** This is classification, not writing. The same transaction
should get the same label on Tuesday as it did on Monday, or the benchmark is
measuring sampling noise as well as capability. It does not make the model
deterministic - nothing does, across model updates - which is why the model
string and prompt version are recorded with every result.

**The key is never logged, echoed, or repr'd.** It arrives from the environment
and goes into one header. `__repr__` is overridden because a dataclass would
otherwise print it the first time this object appears in a traceback.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Final

from offerdelta.domain.common.errors import ValidationError
from offerdelta.evaluation.providers import LLMRequest, LLMResponse
from offerdelta.infrastructure.llm.errors import (
    LLMError,
    MalformedResponseError,
    classify_status,
)
from offerdelta.infrastructure.llm.prompts import (
    PROMPT_VERSION,
    SYSTEM_PROMPTS,
    TOOL_NAME,
    build_tool_schema,
    render_transaction,
)
from offerdelta.infrastructure.llm.retry import (
    Clock,
    RetryPolicy,
    SystemClock,
    call_with_retry,
)
from offerdelta.infrastructure.llm.transport import (
    HttpRequest,
    HttpResponse,
    Transport,
    UrllibTransport,
)

#: Sonnet is the default because this is high-volume, low-difficulty
#: classification: the work is recognising merchants, not reasoning. Opus costs
#: materially more for a task where the ceiling is set by ambiguous merchants
#: rather than model capability - and the harness reports cost per transaction
#: alongside F1 precisely so that claim can be checked rather than believed.
DEFAULT_MODEL: Final = "claude-sonnet-5"

DEFAULT_BASE_URL: Final = "https://api.anthropic.com"

#: The dated API contract, not a model version. Pinned so a future change to
#: the wire format cannot alter this client's behaviour without a code change.
API_VERSION: Final = "2023-06-01"

#: One tool call with three short fields. Generous enough that a long `reason`
#: cannot truncate the label, small enough to bound the cost of a runaway.
DEFAULT_MAX_TOKENS: Final = 512

_OK: Final = 200


@dataclass(frozen=True)
class AnthropicConfig:
    """Everything the client needs, with the key kept out of every printout."""

    api_key: str
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    max_tokens: int = DEFAULT_MAX_TOKENS

    #: Zero for classification. See the module docstring.
    temperature: float = 0.0

    #: Ceiling for a single attempt. The retry budget is separate and longer.
    timeout_s: float = 30.0

    #: Which recorded prompt to send. Defaults to the selected one; an
    #: evaluation can name an older version to reproduce an archived score, or
    #: a candidate to measure one against the same rows.
    prompt_version: str = PROMPT_VERSION

    @property
    def system_prompt(self) -> str:
        return SYSTEM_PROMPTS[self.prompt_version]

    def __post_init__(self) -> None:
        if self.prompt_version not in SYSTEM_PROMPTS:
            known = ", ".join(sorted(SYSTEM_PROMPTS))
            raise ValidationError(
                f"unknown prompt version {self.prompt_version!r}; recorded "
                f"versions are {known}. Scores are only comparable within a "
                f"version, so an unrecognised one is a typo, not a new prompt."
            )
        if not self.api_key or not self.api_key.strip():
            raise ValidationError(
                "no API key: set ANTHROPIC_API_KEY in the environment. It is "
                "deliberately not defaulted - a client that silently runs "
                "keyless fails at the worst moment instead of at startup."
            )
        if self.max_tokens < 1:
            raise ValidationError("max_tokens must be positive")

    @property
    def messages_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/messages"

    def headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def __repr__(self) -> str:
        """Redacted. A key that reaches a traceback reaches a log aggregator."""
        return (
            f"AnthropicConfig(model={self.model!r}, base_url={self.base_url!r}, "
            f"api_key='<redacted>')"
        )


@dataclass
class AnthropicProvider:
    """Classifies one transaction per call, with retries and token accounting."""

    config: AnthropicConfig
    transport: Transport = field(default_factory=UrllibTransport)
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    clock: Clock = field(default_factory=SystemClock)

    #: Retries actually performed, so a run can report that it hit rate limits
    #: rather than silently taking four times as long as expected.
    retries: int = 0

    @property
    def model(self) -> str:
        return self.config.model

    @property
    def prompt_version(self) -> str:
        """Recorded with results, so a score names the prompt that produced it."""
        return self.config.prompt_version

    def classify(self, request: LLMRequest) -> LLMResponse:
        """Ask the model, retrying only what is worth retrying.

        Latency is measured across the whole call including any retries, because
        that is what the caller actually waits. A p95 that excluded backoff
        would describe a system nobody is running.
        """
        body = self._build_body(request)
        http = HttpRequest(
            url=self.config.messages_url,
            body=json.dumps(body).encode("utf-8"),
            headers=self.config.headers(),
            timeout_s=self.config.timeout_s,
        )

        started = self.clock.monotonic()

        def attempt() -> HttpResponse:
            response = self.transport.send(http)
            if response.status != _OK:
                raise classify_status(
                    response.status,
                    body=response.text(),
                    retry_after=response.headers.get("Retry-After"),
                )
            return response

        def note_retry(_attempt: int, _error: LLMError, _delay: float) -> None:
            self.retries += 1

        response = call_with_retry(
            attempt,
            policy=self.retry_policy,
            clock=self.clock,
            on_retry=note_retry,
        )

        elapsed_ms = int((self.clock.monotonic() - started) * 1000)
        return self._parse(response, elapsed_ms)

    def _build_body(self, request: LLMRequest) -> dict[str, object]:
        return {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "system": self.config.system_prompt,
            "tools": [build_tool_schema(request.allowed_labels)],
            # Pinning the choice is what makes the output structured. Without
            # it the model may answer in prose, and then this is a parser.
            "tool_choice": {"type": "tool", "name": TOOL_NAME},
            "messages": [
                {
                    "role": "user",
                    "content": render_transaction(
                        merchant=request.merchant,
                        raw_description=request.raw_description,
                        amount=request.amount,
                        account_type=request.account_type,
                    ),
                }
            ],
        }

    def _parse(self, response: HttpResponse, elapsed_ms: int) -> LLMResponse:
        try:
            payload = json.loads(response.text())
        except json.JSONDecodeError as error:
            raise MalformedResponseError(
                f"the API returned a body that is not JSON: {error}"
            ) from error

        if not isinstance(payload, dict):
            raise MalformedResponseError("the API returned JSON that is not an object")

        usage = payload.get("usage") or {}
        input_tokens = _int(usage.get("input_tokens"))
        output_tokens = _int(usage.get("output_tokens"))

        if payload.get("stop_reason") == "max_tokens":
            # The tool call was cut off mid-write. Its arguments may parse and
            # still be wrong, which is worse than failing, so refuse here.
            raise MalformedResponseError(
                f"the response hit max_tokens ({self.config.max_tokens}) before the "
                f"tool call finished; the arguments cannot be trusted"
            )

        block = _find_tool_use(payload.get("content"))
        if block is None:
            raise MalformedResponseError(
                f"no {TOOL_NAME!r} tool call in the response "
                f"(stop_reason={payload.get('stop_reason')!r})"
            )

        arguments = block.get("input")
        if not isinstance(arguments, dict):
            raise MalformedResponseError("the tool call carried no arguments object")

        label = arguments.get("label")
        if not isinstance(label, str) or not label:
            raise MalformedResponseError(f"the tool call returned no usable label: {label!r}")

        return LLMResponse(
            label=label,
            confidence=_confidence(arguments.get("confidence")),
            reason=str(arguments.get("reason") or "").strip(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=elapsed_ms,
        )


def _find_tool_use(content: object) -> dict[str, object] | None:
    """Locate our tool call among the response blocks.

    A response can carry several blocks, and matching on the name rather than
    taking the first `tool_use` means a future second tool cannot quietly become
    the answer.
    """
    if not isinstance(content, list):
        return None
    for block in content:
        if (
            isinstance(block, dict)
            and block.get("type") == "tool_use"
            and block.get("name") == TOOL_NAME
        ):
            return block
    return None


def _confidence(value: object) -> Decimal:
    """Read the confidence as an exact decimal, clamped to [0, 1].

    Via `str`, never `Decimal(float)`. JSON numbers arrive as binary floats, and
    `Decimal(0.7)` is 0.6999999999999999555910790149937383830547332763671875,
    which then propagates into a routing threshold comparison. The rest of this
    codebase refuses float money for the same reason.

    An unreadable confidence becomes zero rather than an error: the label is
    still useful, and a zero routes it for review, which is the safe reading of
    a model that could not say how sure it was.
    """
    if isinstance(value, bool) or value is None:
        return Decimal(0)
    try:
        confidence = Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal(0)
    return max(Decimal(0), min(Decimal(1), confidence))


def _int(value: object) -> int:
    """Token counts, defensively. A missing count is zero, not a crash."""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return max(0, value)
