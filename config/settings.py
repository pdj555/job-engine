"""Config."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")
    perplexity_api_key: str = Field(default="", alias="PERPLEXITY_API_KEY")
    fast_model: str = "gpt-4o-mini"

    # Hermes Agent brain — Nous Research hermes-agent's OpenAI-compatible server.
    hermes_base_url: str = Field(default="http://127.0.0.1:8642/v1", alias="HERMES_BASE_URL")
    hermes_api_key: str = Field(default="change-me-local-dev", alias="HERMES_API_KEY")
    hermes_model: str = Field(default="hermes-agent", alias="HERMES_MODEL")
    hermes_timeout: float = Field(default=120.0, alias="HERMES_TIMEOUT")

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
