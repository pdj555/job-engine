"""Configuration management."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # API Keys
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    brave_api_key: str = Field(default="", alias="BRAVE_API_KEY")
    perplexity_api_key: str = Field(default="", alias="PERPLEXITY_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # Models
    embedding_model: str = "text-embedding-3-small"
    reasoning_model: str = "gpt-4o"
    fast_model: str = "gpt-4o-mini"

    # Vector DB
    chroma_persist_dir: str = "./data/chroma"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
