"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from ingestion_source.config import Settings
from ingestion_source.models import DocRef

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def mock_robots_txt() -> str:
    return (FIXTURES_DIR / "mock_robots.txt").read_text(encoding="utf-8")


@pytest.fixture
def mock_judgment_html() -> str:
    return (FIXTURES_DIR / "mock_judgment.html").read_text(encoding="utf-8")


@pytest.fixture
def mock_api_doc_json() -> dict:
    return json.loads((FIXTURES_DIR / "mock_api_doc.json").read_text(encoding="utf-8"))


@pytest.fixture
def mock_api_search_json() -> dict:
    return json.loads((FIXTURES_DIR / "mock_api_search.json").read_text(encoding="utf-8"))


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        INDIANKANOON_API_TOKEN="test_token_xyz",
        INDIANKANOON_API_BASE="https://api.indiankanoon.org",
        INDIANKANOON_BASE_URL="https://indiankanoon.org",
        MODE="api",
        MAX_API_CALLS_PER_RUN=100,
        ENABLE_HTML_FALLBACK=True,
        SCRAPE_MIN_DELAY_SECONDS=0.01,
        SCRAPE_MAX_DELAY_SECONDS=0.02,
        API_RATE_LIMIT_RPS=100.0,
        TARGET_COURTS=["Delhi High Court"],
        TARGET_YEARS=[2023],
        SAMPLE_LIMIT=50,
        INTAKE_OUTPUT_PATH=tmp_path / "raw-store",
        DB_PATH=tmp_path / "raw-store" / "job_state" / "test.sqlite",
        SCRAPER_ENGINE="httpx",
        HTTP_TIMEOUT_SECONDS=5.0,
        MAX_RETRIES=2,
        BACKOFF_FACTOR=1.1,
    )


@pytest.fixture
def sample_doc_ref() -> DocRef:
    return DocRef(
        doc_id="1940428",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        month=1,
        title="Union Of India vs M/S Bright Power Projects",
        source_url="https://indiankanoon.org/doc/1940428/",
        doctype="judgment",
    )
