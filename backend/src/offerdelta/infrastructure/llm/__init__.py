"""A production LLM client: transport, error taxonomy, retry, structured output.

Assembled so the interesting failures are testable without a key:

    from offerdelta.infrastructure.llm import (
        AnthropicConfig, AnthropicProvider, FakeTransport, FakeClock,
    )

    provider = AnthropicProvider(
        config=AnthropicConfig(api_key="test"),
        transport=FakeTransport(responses=[...]),
        clock=FakeClock(),
    )

The real one differs only in which transport and clock it is handed.
"""

from offerdelta.infrastructure.llm.anthropic import (
    API_VERSION,
    DEFAULT_MODEL,
    AnthropicConfig,
    AnthropicProvider,
)
from offerdelta.infrastructure.llm.errors import (
    AuthenticationError,
    FatalError,
    InvalidRequestError,
    LLMError,
    MalformedResponseError,
    ModelNotFoundError,
    OverloadedError,
    PermissionDeniedError,
    RateLimitError,
    RequestTooLargeError,
    RetryableError,
    RetryBudgetExhaustedError,
    ServiceUnavailableError,
    TransportError,
    classify_status,
    parse_retry_after,
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
    FakeClock,
    RetryPolicy,
    SystemClock,
    call_with_retry,
)
from offerdelta.infrastructure.llm.transport import (
    FakeTransport,
    Headers,
    HttpRequest,
    HttpResponse,
    Transport,
    UrllibTransport,
    json_response,
)

__all__ = [
    "API_VERSION",
    "DEFAULT_MODEL",
    "PROMPT_VERSION",
    "SYSTEM_PROMPTS",
    "TOOL_NAME",
    "AnthropicConfig",
    "AnthropicProvider",
    "AuthenticationError",
    "Clock",
    "FakeClock",
    "FakeTransport",
    "FatalError",
    "Headers",
    "HttpRequest",
    "HttpResponse",
    "InvalidRequestError",
    "LLMError",
    "MalformedResponseError",
    "ModelNotFoundError",
    "OverloadedError",
    "PermissionDeniedError",
    "RateLimitError",
    "RequestTooLargeError",
    "RetryBudgetExhaustedError",
    "RetryPolicy",
    "RetryableError",
    "ServiceUnavailableError",
    "SystemClock",
    "Transport",
    "TransportError",
    "UrllibTransport",
    "build_tool_schema",
    "call_with_retry",
    "classify_status",
    "json_response",
    "parse_retry_after",
    "render_transaction",
]
