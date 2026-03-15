"""Configuration settings for the FastAPI application."""

import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Configuration
    api_title: str = "Ignitio AI Tutor API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"

    # CORS Configuration
    cors_origins: str = "*"  # Comma-separated origins or "*" for all

    # Session Configuration
    session_max_age_minutes: int = 60  # Sessions expire after 1 hour
    max_sessions: int = 1000  # Maximum concurrent sessions

    # Environment
    environment: str = os.getenv("ENVIRONMENT", "development")

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins from comma-separated string."""
        if self.cors_origins == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",")]

    model_config = {
        "env_prefix": "API_",
        "env_file": ".env",
        "extra": "ignore",  # Ignore extra env vars not defined in model
    }


# Global settings instance
settings = Settings()