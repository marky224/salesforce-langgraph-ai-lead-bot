"""
Application configuration for the AI Sales Lead Bot.

Centralises all environment variable loading via ``pydantic-settings`` and
provides a factory function (``get_llm``) that returns the correct LangChain
chat model based on the ``LLM_PROVIDER`` env var.

Supported providers:
- **anthropic** — Claude via ``langchain-anthropic``
- **openai** — GPT-4o via ``langchain-openai``
- **groq** — Llama / Mixtral via ``langchain-groq``
- **xai** — Grok via OpenAI-compatible endpoint

Usage::

    from app.config import settings, get_llm

    llm = get_llm()                   # Returns configured ChatModel
    print(settings.sf_instance_url)   # Salesforce instance URL
"""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache
from typing import Any, Optional

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class LLMProvider(str, Enum):
    """Supported LLM providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    GROQ = "groq"
    XAI = "xai"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and ``.env`` file.

    All secrets use ``SecretStr`` so they never appear in logs, repr, or
    serialisation output.  Access the raw value via ``.get_secret_value()``.
    """

    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM configuration ------------------------------------------------

    llm_provider: LLMProvider = Field(
        default=LLMProvider.ANTHROPIC,
        description="Which LLM backend to use: anthropic | openai | groq | xai",
    )
    llm_model: Optional[str] = Field(
        default=None,
        description=(
            "Override the default model for the chosen provider. "
            "If not set, a sensible default is selected per provider."
        ),
    )
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for the LLM.",
    )

    # Provider API keys (only the active provider's key is required)
    anthropic_api_key: Optional[SecretStr] = None
    openai_api_key: Optional[SecretStr] = None
    groq_api_key: Optional[SecretStr] = None
    xai_api_key: Optional[SecretStr] = None

    # --- Salesforce --------------------------------------------------------

    sf_instance_url: str = Field(
        default="https://login.salesforce.com",
        description="Salesforce instance URL",
    )
    sf_client_id: Optional[SecretStr] = Field(
        default=None, description="Connected App consumer key"
    )
    sf_client_secret: Optional[SecretStr] = Field(
        default=None, description="Connected App consumer secret"
    )
    sf_username: Optional[str] = Field(
        default=None, description="Salesforce integration user"
    )
    sf_password: Optional[SecretStr] = Field(
        default=None, description="Salesforce password"
    )
    sf_security_token: Optional[SecretStr] = Field(
        default=None, description="Salesforce security token"
    )

    # --- Server / deployment -----------------------------------------------

    backend_url: str = Field(
        default="http://localhost:8000",
        description="Public backend URL (used by frontend widget)",
    )
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:5173",
        description="Allowed CORS origins (comma-separated string)",
    )
    log_level: str = Field(
        default="INFO",
        description="Python logging level",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Application version string",
    )

    # --- Computed helpers --------------------------------------------------

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse the comma-separated ``cors_origins`` string into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # --- Convenience -------------------------------------------------------

    @property
    def active_api_key(self) -> SecretStr:
        """Return the API key for the currently configured provider."""
        key_map: dict[LLMProvider, Optional[SecretStr]] = {
            LLMProvider.ANTHROPIC: self.anthropic_api_key,
            LLMProvider.OPENAI: self.openai_api_key,
            LLMProvider.GROQ: self.groq_api_key,
            LLMProvider.XAI: self.xai_api_key,
        }
        key = key_map.get(self.llm_provider)
        if key is None:
            raise ValueError(
                f"No API key configured for provider '{self.llm_provider.value}'. "
                f"Set the {self.llm_provider.value.upper()}_API_KEY environment variable."
            )
        return key

    @property
    def salesforce_configured(self) -> bool:
        """True if minimum Salesforce credentials are present."""
        return bool(
            self.sf_client_id
            and self.sf_client_secret
            and self.sf_username
            and self.sf_password
        )


# ---------------------------------------------------------------------------
# Singleton settings instance
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return the cached application settings instance.

    Uses ``lru_cache`` so the ``.env`` file is only read once.
    """
    return Settings()


# Convenience alias — importable as ``from app.config import settings``
settings = get_settings()


# ---------------------------------------------------------------------------
# Default model names per provider
# ---------------------------------------------------------------------------

_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
    LLMProvider.XAI: "grok-3",
}


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm(
    provider: LLMProvider | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> Any:
    """
    Instantiate and return a LangChain chat model for the configured provider.

    Parameters
    ----------
    provider : LLMProvider | None
        Override the provider from settings.  Defaults to
        ``settings.llm_provider``.
    model : str | None
        Override the model name.  Defaults to ``settings.llm_model`` or
        the provider's default.
    temperature : float | None
        Override the sampling temperature.  Defaults to
        ``settings.llm_temperature``.

    Returns
    -------
    BaseChatModel
        A LangChain chat model instance ready for ``.ainvoke()``.

    Raises
    ------
    ValueError
        If the provider is unrecognised or the required API key is missing.
    ImportError
        If the provider's LangChain integration package is not installed.
    """
    _settings = get_settings()
    provider = provider or _settings.llm_provider
    model = model or _settings.llm_model or _DEFAULT_MODELS[provider]
    temperature = temperature if temperature is not None else _settings.llm_temperature

    logger.info(
        "Initialising LLM: provider=%s  model=%s  temperature=%.2f",
        provider.value,
        model,
        temperature,
    )

    if provider == LLMProvider.ANTHROPIC:
        return _build_anthropic(model, temperature, _settings)

    if provider == LLMProvider.OPENAI:
        return _build_openai(model, temperature, _settings)

    if provider == LLMProvider.GROQ:
        return _build_groq(model, temperature, _settings)

    if provider == LLMProvider.XAI:
        return _build_xai(model, temperature, _settings)

    raise ValueError(f"Unsupported LLM provider: {provider}")


# ---------------------------------------------------------------------------
# Provider-specific builders
# ---------------------------------------------------------------------------

def _build_anthropic(model: str, temperature: float, s: Settings) -> Any:
    """Build a ChatAnthropic instance."""
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise ImportError(
            "langchain-anthropic is required for the Anthropic provider. "
            "Install it with: pip install langchain-anthropic"
        ) from exc

    if s.anthropic_api_key is None:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")

    return ChatAnthropic(
        model=model,
        temperature=temperature,
        anthropic_api_key=s.anthropic_api_key.get_secret_value(),
        max_tokens=1024,
    )


def _build_openai(model: str, temperature: float, s: Settings) -> Any:
    """Build a ChatOpenAI instance."""
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for the OpenAI provider. "
            "Install it with: pip install langchain-openai"
        ) from exc

    if s.openai_api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=s.openai_api_key.get_secret_value(),
    )


def _build_groq(model: str, temperature: float, s: Settings) -> Any:
    """Build a ChatGroq instance."""
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:
        raise ImportError(
            "langchain-groq is required for the Groq provider. "
            "Install it with: pip install langchain-groq"
        ) from exc

    if s.groq_api_key is None:
        raise ValueError("GROQ_API_KEY environment variable is not set.")

    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=s.groq_api_key.get_secret_value(),
    )


def _build_xai(model: str, temperature: float, s: Settings) -> Any:
    """
    Build a chat model for xAI (Grok).

    xAI exposes an OpenAI-compatible API, so we use ``ChatOpenAI`` with
    a custom ``base_url`` pointing to the xAI endpoint.
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for the xAI provider (OpenAI-compatible). "
            "Install it with: pip install langchain-openai"
        ) from exc

    if s.xai_api_key is None:
        raise ValueError("XAI_API_KEY environment variable is not set.")

    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=s.xai_api_key.get_secret_value(),
        base_url="https://api.x.ai/v1",
    )


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def configure_logging() -> None:
    """
    Set up structured logging for the application.

    Call once at startup from ``server.py``.
    """
    log_level = get_settings().log_level.upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Quieten noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("Logging configured at %s level", log_level)
