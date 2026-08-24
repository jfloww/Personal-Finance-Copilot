"""Building a provider from the environment.

Separate from `anthropic.py` so the client itself stays independent of how
configuration reaches it: the adapter takes a config object, and only this file
knows that one of the ways to get it is an environment variable.

`build_provider` returns `None` rather than raising when no key is set. Every
caller so far - the evaluation runner, the report - has a useful thing to do
without a model, which is to run the rule baseline and say plainly that the LLM
row is missing. Raising here would turn "no key today" into a crash in a script
that had other work to do.
"""

from __future__ import annotations

from offerdelta.config import Settings, get_settings
from offerdelta.infrastructure.llm.anthropic import (
    DEFAULT_MODEL,
    AnthropicConfig,
    AnthropicProvider,
)
from offerdelta.infrastructure.llm.prompts import PROMPT_VERSION


def build_provider(
    settings: Settings | None = None, *, prompt_version: str = PROMPT_VERSION
) -> AnthropicProvider | None:
    """A live provider, or `None` when no key is configured.

    `prompt_version` names which recorded prompt to send. It is a keyword rather
    than a setting because it belongs to one run, not to the environment: an
    evaluation reproducing an archived score and the API serving traffic can
    disagree about it and both be right.
    """
    settings = settings or get_settings()

    if not settings.anthropic_api_key:
        return None

    config = AnthropicConfig(
        api_key=settings.anthropic_api_key,
        model=settings.anthropic_model or DEFAULT_MODEL,
        prompt_version=prompt_version,
    )
    return AnthropicProvider(config=config)
