"""Configuration settings for LegalTrace acquisition module."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration settings backed by environment variables."""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Indian Kanoon API Credentials and Endpoints
    INDIANKANOON_API_TOKEN: Optional[str] = Field(default=None, description="Official API authentication token")
    INDIANKANOON_API_BASE: str = Field(default="https://api.indiankanoon.org", description="API base URL")
    INDIANKANOON_BASE_URL: str = Field(default="https://indiankanoon.org", description="Web portal base URL")

    # Acquisition Strategy Mode (defaults to scrape_only as scraper is primary)
    MODE: Literal["api", "hybrid", "scrape_only"] = Field(
        default="scrape_only",
        description="Operating mode: scrape_only (primary scraper), hybrid, or api"
    )
    SCRAPER_ENGINE: Literal["scrapling", "httpx"] = Field(
        default="scrapling",
        description="HTML scraper engine: scrapling (dynamic stealth browser) or httpx (direct HTTP)"
    )

    # Budget & Cost Control
    MAX_API_CALLS_PER_RUN: int = Field(default=5000, description="Max allowed API calls per execution")
    ENABLE_HTML_FALLBACK: bool = Field(default=True, description="Allow fallback to HTML scraper")

    # Concurrency & Multithreading (default 1 for polite single-stream scraping)
    MAX_WORKERS: int = Field(default=1, ge=1, le=16, description="Number of concurrent worker threads")

    # Rate Limiting & Politeness (Scraping - strictly obeying robots.txt Crawl-delay: 3+)
    SCRAPE_MIN_DELAY_SECONDS: float = Field(default=5.0, description="Minimum delay between scrape requests")
    SCRAPE_MAX_DELAY_SECONDS: float = Field(default=8.0, description="Maximum jittered delay between scrape requests")
    API_RATE_LIMIT_RPS: float = Field(default=2.0, description="Requests per second ceiling for API calls")
    ADAPTIVE_BACKOFF_SECONDS: float = Field(default=15.0, description="Global cool-down delay when throttled")

    # Ethical User-Agent & Optional Browser Session
    USER_AGENT: str = Field(
        default="LegalTraceAcademicBot/0.1 (+contact: academic-research@legaltrace.org; purpose: academic research corpus, non-commercial)",
        description="User-Agent string identifying academic research bot"
    )
    INDIANKANOON_COOKIES: Optional[str] = Field(
        default=None,
        description="Optional custom cookies (e.g. session / clearance tokens) for authenticated or challenged requests"
    )

    @field_validator("USER_AGENT", mode="before")
    @classmethod
    def clean_user_agent(cls, v: Optional[str]) -> str:
        if v and isinstance(v, str):
            cleaned = v.strip().strip("'\"").strip()
            return cleaned or "LegalTraceAcademicBot/0.1"
        return "LegalTraceAcademicBot/0.1"

    @field_validator("INDIANKANOON_COOKIES", mode="before")
    @classmethod
    def clean_cookies(cls, v: Optional[str]) -> Optional[str]:
        if v and isinstance(v, str):
            cleaned = v.strip().strip("'\"").strip()
            return cleaned or None
        return v

    # Target Scope
    TARGET_COURTS: list[str] = Field(
        default=["Delhi High Court", "Supreme Court of India"],
        description="Target courts to acquire"
    )
    TARGET_YEARS: list[int] = Field(default=[2023], description="Target years to acquire")
    SAMPLE_LIMIT: int = Field(default=1000, description="Maximum total documents to ingest in a run")

    # Storage & Persistence
    INTAKE_OUTPUT_PATH: Path = Field(default=Path("./raw-store"), description="Directory where raw documents are written")
    DB_PATH: Path = Field(default=Path("./raw-store/job_state/legaltrace_ingest.sqlite"), description="SQLite database path")

    # Network & Retries
    HTTP_TIMEOUT_SECONDS: float = Field(default=30.0, description="HTTP request timeout in seconds")
    MAX_RETRIES: int = Field(default=3, description="Maximum retry attempts on transient network errors")
    BACKOFF_FACTOR: float = Field(default=2.0, description="Exponential backoff factor for retries")


def get_settings() -> Settings:
    """Retrieve application settings instance."""
    return Settings()
