"""Unit tests for TokenBucketLimiter and RobotsTxtManager."""

import time
import pytest
import httpx
import respx

from ingestion_source.rate_limiter import (
    RobotsDisallowedException,
    RobotsTxtManager,
    TokenBucketLimiter,
)


def test_token_bucket_limiter_rate():
    limiter = TokenBucketLimiter(rate_rps=50.0, capacity=5.0, min_delay_seconds=0.0)
    slept = limiter.acquire()
    assert slept >= 0.0


def test_token_bucket_limiter_min_delay():
    limiter = TokenBucketLimiter(rate_rps=100.0, capacity=10.0, min_delay_seconds=0.05, max_delay_seconds=0.06)
    limiter.acquire()
    start = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.04


def test_token_bucket_limiter_adaptive_backoff():
    limiter = TokenBucketLimiter(rate_rps=10.0, capacity=2.0, min_delay_seconds=0.01, backoff_penalty_seconds=0.1)
    assert limiter.is_cooling_down() is False

    penalty = limiter.trigger_backoff(penalty_seconds=0.08)
    assert penalty == 0.08
    assert limiter.is_cooling_down() is True

    start = time.monotonic()
    limiter.acquire()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.05
    assert limiter.is_cooling_down() is False


def test_robots_txt_manager_with_content(mock_robots_txt: str):
    mgr = RobotsTxtManager(user_agent="LegalTraceAcademicBot/0.1")
    mgr.set_content(mock_robots_txt)

    assert mgr.is_allowed("https://indiankanoon.org/doc/12345/") is True
    assert mgr.is_allowed("https://indiankanoon.org/browse/delhi/2023/") is True
    assert mgr.is_allowed("https://indiankanoon.org/search?formInput=test") is False
    assert mgr.is_allowed("https://indiankanoon.org/oauth/login") is False

    assert mgr.get_crawl_delay() == 3.0


def test_robots_txt_manager_assert_allowed(mock_robots_txt: str):
    mgr = RobotsTxtManager(user_agent="LegalTraceAcademicBot/0.1")
    mgr.set_content(mock_robots_txt)

    mgr.assert_allowed("https://indiankanoon.org/doc/12345/")

    with pytest.raises(RobotsDisallowedException):
        mgr.assert_allowed("https://indiankanoon.org/search?q=test")


@respx.mock
def test_robots_txt_manager_fetch_http(mock_robots_txt: str):
    respx.get("https://indiankanoon.org/robots.txt").mock(
        return_value=httpx.Response(200, text=mock_robots_txt)
    )

    mgr = RobotsTxtManager(base_url="https://indiankanoon.org")
    mgr.fetch_rules()
    assert mgr.is_allowed("https://indiankanoon.org/doc/999/") is True
    assert mgr.is_allowed("https://indiankanoon.org/search") is False
