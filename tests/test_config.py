"""Unit tests for configuration and settings."""

from pathlib import Path
import pytest
from ingestion_source.config import Settings, get_settings


def test_default_settings():
    settings = Settings()
    assert settings.INDIANKANOON_API_BASE == "https://api.indiankanoon.org"
    assert settings.INDIANKANOON_BASE_URL == "https://indiankanoon.org"
    assert settings.MODE in ("api", "hybrid", "scrape_only")
    assert settings.MAX_API_CALLS_PER_RUN == 5000
    assert settings.SCRAPE_MIN_DELAY_SECONDS >= 1.0
    assert isinstance(settings.INTAKE_OUTPUT_PATH, Path)


def test_custom_settings(tmp_path: Path):
    settings = Settings(
        INDIANKANOON_API_TOKEN="custom_token",
        MODE="hybrid",
        MAX_API_CALLS_PER_RUN=250,
        ENABLE_HTML_FALLBACK=True,
        INTAKE_OUTPUT_PATH=tmp_path / "custom_intake",
    )
    assert settings.INDIANKANOON_API_TOKEN == "custom_token"
    assert settings.MODE == "hybrid"
    assert settings.MAX_API_CALLS_PER_RUN == 250
    assert settings.ENABLE_HTML_FALLBACK is True
    assert settings.INTAKE_OUTPUT_PATH == tmp_path / "custom_intake"


def test_user_agent_and_cookies_clean_validator():
    settings = Settings(
        USER_AGENT="'Mozilla/5.0 CustomBot/1.0'",
        INDIANKANOON_COOKIES='"csrftoken=val1; cf_clearance=val2"',
    )
    assert settings.USER_AGENT == "Mozilla/5.0 CustomBot/1.0"
    assert settings.INDIANKANOON_COOKIES == "csrftoken=val1; cf_clearance=val2"

