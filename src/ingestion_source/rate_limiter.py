"""Rate limiting and robots.txt compliance engine for LegalTrace acquisition."""

from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse
import urllib.robotparser

import httpx

logger = logging.getLogger(__name__)


class RateLimiterException(Exception):
    """Base exception for rate limiter and compliance errors."""
    pass


class RobotsDisallowedException(RateLimiterException):
    """Raised when an operation attempts to crawl a path disallowed by robots.txt."""
    def __init__(self, url: str, user_agent: str) -> None:
        self.url = url
        self.user_agent = user_agent
        super().__init__(f"Path disallowed by robots.txt for User-Agent '{user_agent}': {url}")


class TokenBucketLimiter:
    """Thread-safe rate limiter with adaptive backoff, circuit breaking, and global pacing."""

    def __init__(
        self,
        rate_rps: float = 1.0,
        capacity: float = 1.0,
        min_delay_seconds: float = 0.0,
        max_delay_seconds: float = 0.0,
        backoff_penalty_seconds: float = 15.0,
    ) -> None:
        """Initialize thread-safe token bucket rate limiter.

        Args:
            rate_rps: Fill rate in tokens per second.
            capacity: Maximum bucket capacity.
            min_delay_seconds: Explicit minimum delay floor between consecutive acquisitions.
            max_delay_seconds: Maximum delay ceiling (adds random jitter between min and max).
            backoff_penalty_seconds: Duration to pause all threads when throttled/challenged.
        """
        self.rate_rps = max(0.01, rate_rps)
        self.capacity = max(1.0, capacity)
        self.base_min_delay = max(0.0, min_delay_seconds)
        self.base_max_delay = max(self.base_min_delay, max_delay_seconds)
        self.backoff_penalty_seconds = max(1.0, backoff_penalty_seconds)

        self._current_min_delay = self.base_min_delay
        self._current_max_delay = self.base_max_delay
        self._cool_down_until = 0.0
        self._consecutive_successes = 0

        self._tokens = self.capacity
        self._last_refill = time.monotonic()
        self._last_acquired = 0.0
        self._lock = threading.Lock()

    def trigger_backoff(self, penalty_seconds: Optional[float] = None) -> float:
        """Trigger global cool-down across all threads when 429/challenge is encountered."""
        with self._lock:
            penalty = penalty_seconds or self.backoff_penalty_seconds
            now = time.monotonic()
            self._cool_down_until = max(self._cool_down_until, now + penalty)
            # Increase delay temporarily
            self._current_min_delay = min(15.0, max(self._current_min_delay * 1.5, 4.0))
            self._current_max_delay = max(self._current_min_delay + 2.0, self._current_max_delay * 1.5)
            self._consecutive_successes = 0
            logger.warning(
                "Global rate limiter backoff triggered. Pausing all threads for %.1fs. Adjusted min delay: %.1fs",
                penalty,
                self._current_min_delay,
            )
            return penalty

    def report_success(self) -> None:
        """Record successful request to gradually restore normal rate limits."""
        with self._lock:
            self._consecutive_successes += 1
            if self._consecutive_successes >= 10:
                # Slowly decay back to base delays
                self._current_min_delay = max(self.base_min_delay, self._current_min_delay * 0.9)
                self._current_max_delay = max(self.base_max_delay, self._current_max_delay * 0.9)

    def is_cooling_down(self) -> bool:
        """Check if limiter is currently in cool-down state."""
        with self._lock:
            return time.monotonic() < self._cool_down_until

    def _refill(self, now: float) -> None:
        elapsed = now - self._last_refill
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate_rps)
        self._last_refill = now

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until requested tokens are available and minimum delay is satisfied.

        Returns:
            Total sleep duration in seconds.
        """
        total_slept = 0.0
        with self._lock:
            # 1. Enforce global cool-down if backoff was triggered
            now = time.monotonic()
            if now < self._cool_down_until:
                wait_time = self._cool_down_until - now
                time.sleep(wait_time)
                total_slept += wait_time
                self._cool_down_until = 0.0
                now = time.monotonic()

            # 2. Check interval delay with jitter across threads
            if self._current_min_delay > 0.0 and self._last_acquired > 0.0:
                elapsed_since_last = now - self._last_acquired
                target_interval = self._current_min_delay
                if self._current_max_delay > self._current_min_delay:
                    target_interval += random.uniform(0.0, self._current_max_delay - self._current_min_delay)

                required_wait = target_interval - elapsed_since_last
                if required_wait > 0.0:
                    time.sleep(required_wait)
                    total_slept += required_wait
                    now = time.monotonic()

            # 3. Enforce token bucket capacity
            while True:
                self._refill(now)
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    self._last_acquired = time.monotonic()
                    break

                deficit = tokens - self._tokens
                sleep_needed = deficit / self.rate_rps
                time.sleep(sleep_needed)
                total_slept += sleep_needed
                now = time.monotonic()

        return total_slept


class RobotsTxtManager:
    """Manages downloading, parsing, caching, and compliance checks for robots.txt."""

    def __init__(
        self,
        base_url: str = "https://indiankanoon.org",
        user_agent: str = "LegalTraceAcademicBot/0.1",
        cache_dir: Optional[Path] = None,
        http_client: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.robots_url = urljoin(self.base_url + "/", "robots.txt")
        self.user_agent = user_agent
        self.cache_dir = cache_dir
        self._http_client = http_client
        self._parser: Optional[urllib.robotparser.RobotFileParser] = None
        self._fetched_at: Optional[float] = None
        self._raw_content: Optional[str] = None
        self._lock = threading.Lock()

    def set_content(self, robots_txt_content: str) -> None:
        """Manually supply robots.txt content (useful for fixtures and tests)."""
        with self._lock:
            self._raw_content = robots_txt_content
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(robots_txt_content.splitlines())
            self._parser = parser
            self._fetched_at = time.time()

    def fetch_rules(self, timeout_seconds: float = 15.0) -> None:
        """Fetch and parse robots.txt from the remote source or disk cache."""
        with self._lock:
            if self._parser is not None:
                return

            # Check disk cache if available
            if self.cache_dir:
                cache_file = self.cache_dir / "robots.txt"
                if cache_file.exists():
                    try:
                        content = cache_file.read_text(encoding="utf-8")
                        parser = urllib.robotparser.RobotFileParser()
                        parser.parse(content.splitlines())
                        self._parser = parser
                        self._raw_content = content
                        self._fetched_at = cache_file.stat().st_mtime
                        logger.info("Loaded robots.txt from cache: %s", cache_file)
                        return
                    except Exception as e:
                        logger.warning("Failed to read robots.txt cache: %s", e)

            # Fetch via HTTP
            content = ""
            client = self._http_client or httpx.Client(
                headers={"User-Agent": self.user_agent},
                timeout=timeout_seconds,
                follow_redirects=True,
            )
            should_close = self._http_client is None
            try:
                logger.info("Fetching robots.txt from %s", self.robots_url)
                resp = client.get(self.robots_url)
                if resp.status_code == 200:
                    content = resp.text
                elif resp.status_code in (404, 410):
                    # If robots.txt doesn't exist, allow all
                    content = "User-agent: *\nAllow: /"
                else:
                    logger.warning("Non-200 status fetching robots.txt: %s. Defaulting to polite allow.", resp.status_code)
                    content = "User-agent: *\nDisallow: /search\n"
            except Exception as e:
                logger.error("Error fetching robots.txt: %s. Using safe fallback rules.", e)
                content = "User-agent: *\nDisallow: /search\n"
            finally:
                if should_close:
                    client.close()

            parser = urllib.robotparser.RobotFileParser()
            parser.parse(content.splitlines())
            self._parser = parser
            self._raw_content = content
            self._fetched_at = time.time()

            # Write to disk cache
            if self.cache_dir:
                try:
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = self.cache_dir / "robots.txt"
                    cache_file.write_text(content, encoding="utf-8")
                except Exception as e:
                    logger.warning("Failed to save robots.txt cache: %s", e)

    def is_allowed(self, url: str) -> bool:
        """Check if target URL is allowed by robots.txt for our User-Agent."""
        if self._parser is None:
            self.fetch_rules()

        assert self._parser is not None
        # Extract path component
        parsed = urlparse(url)
        path = parsed.path
        if parsed.query:
            path = f"{path}?{parsed.query}"

        # Test with our user agent first, then wildcard fallback
        allowed = self._parser.can_fetch(self.user_agent, url)
        if allowed:
            allowed = self._parser.can_fetch("*", url)
        return allowed

    def get_crawl_delay(self) -> Optional[float]:
        """Get crawl delay specified in robots.txt for our User-Agent or default."""
        if self._parser is None:
            self.fetch_rules()

        assert self._parser is not None
        try:
            delay = self._parser.crawl_delay(self.user_agent)
            if delay is None:
                delay = self._parser.crawl_delay("*")
            return float(delay) if delay is not None else None
        except Exception:
            return None

    def assert_allowed(self, url: str) -> None:
        """Raise RobotsDisallowedException if URL is not allowed."""
        if not self.is_allowed(url):
            raise RobotsDisallowedException(url, self.user_agent)
