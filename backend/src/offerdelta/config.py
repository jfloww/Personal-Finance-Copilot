"""Runtime configuration.

Read from the environment, with a local `.env` for development. The connection
string is a secret: it is never logged, never echoed, and never included in an
error message. `Settings.redacted_dsn` exists so diagnostics can say which host
they reached without leaking the credentials to reach it.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Everything the service needs from its environment."""

    model_config = SettingsConfigDict(
        env_file=_BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    #: PostgreSQL DSN. CI sets this at job level against a disposable service
    #: container, so it is not absent there - a checkout without a `.env` is
    #: the case that is actually missing it. Every database-backed test skips
    #: rather than fails when it is missing, for that case: a green suite must
    #: not depend on a secret that only exists on one machine.
    connection_string: str | None = Field(default=None, alias="CONNECTION_STRING")

    #: Anthropic API key. Absent in CI by design: the whole LLM client is
    #: tested through an injected transport, so a green suite never needs a
    #: key and no key is ever spent proving the retry loop works.
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")

    #: Overridable per environment so a cheaper model can be pinned for a bulk
    #: run without editing code. Left as None to take the client's default.
    anthropic_model: str | None = Field(default=None, alias="ANTHROPIC_MODEL")

    #: Kept separate because agent tool selection and bulk categorisation are
    #: different workloads and should not silently share a cost decision.
    agent_model: str | None = Field(default=None, alias="AGENT_MODEL")

    #: Signing key for access tokens. Unlike the Anthropic key above, this is
    #: not an external credential that costs money and cannot be faked - it is
    #: an arbitrary self-issued string, the same kind of thing as the
    #: hardcoded `CONNECTION_STRING` the CI workflow sets beside it. CI sets a
    #: throwaway value so the auth suite runs there for real. Locally, absence
    #: still means authentication is switched off and the routes that need it
    #: disappear rather than half-work - that discipline does not depend on
    #: whether CI happens to hold a value.
    jwt_secret: str | None = Field(default=None, alias="JWT_SECRET")

    @property
    def database_available(self) -> bool:
        return bool(self.connection_string)

    @property
    def llm_available(self) -> bool:
        return bool(self.anthropic_api_key)

    @property
    def auth_available(self) -> bool:
        return bool(self.jwt_secret)

    @property
    def sqlalchemy_dsn(self) -> str:
        """The DSN with a driver SQLAlchemy 2 understands.

        Neon hands out `postgresql://…`; SQLAlchemy needs the driver named
        explicitly or it reaches for psycopg2, which is not installed.
        """
        if not self.connection_string:
            raise RuntimeError(
                "CONNECTION_STRING is not set; add it to backend/.env or the environment"
            )
        dsn = self.connection_string
        if dsn.startswith("postgresql://"):
            return dsn.replace("postgresql://", "postgresql+psycopg://", 1)
        if dsn.startswith("postgres://"):
            return dsn.replace("postgres://", "postgresql+psycopg://", 1)
        return dsn

    @property
    def redacted_dsn(self) -> str:
        """Host and database only — safe to log, print, or put in an error."""
        if not self.connection_string:
            return "<unset>"
        parts = urlsplit(self.connection_string)
        host = parts.hostname or "?"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
