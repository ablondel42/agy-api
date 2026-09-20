"""Application configuration loaded from environment variables."""
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # AGY CLI
    agy_binary: str = Field(
        default="agy",
        validation_alias=AliasChoices("agy_binary", "AGY_BINARY", "AGY_AGY_BINARY"),
    )
    agy_default_agent: str = Field(
        default="default",
        validation_alias=AliasChoices("agy_default_agent", "AGY_DEFAULT_AGENT", "AGY_AGY_DEFAULT_AGENT"),
    )
    agy_default_model: str = Field(
        default="Gemini 3.8 Flash",
        validation_alias=AliasChoices("agy_default_model", "AGY_DEFAULT_MODEL", "AGY_AGY_DEFAULT_MODEL"),
    )
    agy_default_reflection: str = Field(
        default="high",  # "low", "medium", "high"
        validation_alias=AliasChoices("agy_default_reflection", "AGY_DEFAULT_REFLECTION", "AGY_AGY_DEFAULT_REFLECTION"),
    )
    agy_default_timeout: int = Field(
        default=120,
        validation_alias=AliasChoices("agy_default_timeout", "AGY_DEFAULT_TIMEOUT", "AGY_AGY_DEFAULT_TIMEOUT"),
    )
    agy_default_workspace: str = Field(
        default=".",
        validation_alias=AliasChoices("agy_default_workspace", "AGY_DEFAULT_WORKSPACE", "AGY_AGY_DEFAULT_WORKSPACE"),
    )
    agy_agents_dir: str = Field(
        default=".agents/agents",
        validation_alias=AliasChoices("agy_agents_dir", "AGY_AGENTS_DIR", "AGY_AGY_AGENTS_DIR"),
    )
    agy_default_mode: str | None = Field(
        default=None,
        validation_alias=AliasChoices("agy_default_mode", "AGY_DEFAULT_MODE", "AGY_AGY_DEFAULT_MODE"),
    )
    agy_default_sandbox: bool = Field(
        default=False,
        validation_alias=AliasChoices("agy_default_sandbox", "AGY_DEFAULT_SANDBOX", "AGY_AGY_DEFAULT_SANDBOX"),
    )
    agy_default_project: str | None = Field(
        default=None,
        validation_alias=AliasChoices("agy_default_project", "AGY_DEFAULT_PROJECT", "AGY_AGY_DEFAULT_PROJECT"),
    )
    agy_dangerously_skip_permissions: bool = Field(
        default=False,
        description="Deprecated: permissions bypass is not allowed",
    )
    agy_interactive_timeout: int = Field(
        default=180,
        validation_alias=AliasChoices("agy_interactive_timeout", "AGY_INTERACTIVE_TIMEOUT", "AGY_AGY_INTERACTIVE_TIMEOUT"),
    )
    agy_log_dir: str = Field(
        default="log",
        validation_alias=AliasChoices("agy_log_dir", "AGY_LOG_DIR", "AGY_AGY_LOG_DIR"),
    )
    agy_app_data_dir: str | None = Field(
        default=None,
        validation_alias=AliasChoices("agy_app_data_dir", "AGY_APP_DATA_DIR", "AGY_AGY_APP_DATA_DIR"),
    )
    model_cache_ttl: int = 300  # seconds

    # Session Pool
    max_sessions: int = 10
    session_idle_timeout: int = 600  # seconds

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # "json" or "text"
    log_timezone: str | None = None  # None = host local timezone, e.g. "Europe/Paris"

    # Auth (future)
    # api_key: str | None = None

    model_config = {"env_file": [".env", "dev.env"], "env_prefix": "AGY_", "extra": "ignore"}


settings = Settings()
