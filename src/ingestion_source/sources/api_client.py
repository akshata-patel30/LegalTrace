"""Official Indian Kanoon API client implementation."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Iterator, Optional
from urllib.parse import urljoin

import httpx

from ingestion_source.config import Settings, get_settings
from ingestion_source.models import (
    DocRef,
    FetchFailureType,
    FetchResult,
    SourceType,
)
from ingestion_source.rate_limiter import TokenBucketLimiter

logger = logging.getLogger(__name__)


class IndianKanoonAPIClient:
    """Client for official api.indiankanoon.org endpoints."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        http_client: Optional[httpx.Client] = None,
        rate_limiter: Optional[TokenBucketLimiter] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.api_base = self.settings.INDIANKANOON_API_BASE.rstrip("/")
        self.token = self.settings.INDIANKANOON_API_TOKEN
        self.max_calls = self.settings.MAX_API_CALLS_PER_RUN
        self.api_calls_count = 0

        # Rate limiter
        self.rate_limiter = rate_limiter or TokenBucketLimiter(
            rate_rps=self.settings.API_RATE_LIMIT_RPS,
            capacity=2.0,
            min_delay_seconds=0.2,
        )

        # HTTP client
        headers = {
            "User-Agent": self.settings.USER_AGENT,
            "Accept": "application/json",
        }
        if self.token:
            headers["Authorization"] = f"Token {self.token}"

        self._client = http_client or httpx.Client(
            base_url=self.api_base,
            headers=headers,
            timeout=self.settings.HTTP_TIMEOUT_SECONDS,
        )
        self._owns_client = http_client is None

    def get_source_type(self) -> SourceType:
        return SourceType.API

    def _increment_call_count(self) -> None:
        self.api_calls_count += 1

    def _check_budget(self) -> Optional[str]:
        if self.api_calls_count >= self.max_calls:
            return f"API call budget limit ({self.max_calls}) reached."
        return None

    def search(
        self,
        query: str,
        pagenum: int = 0,
        max_pages: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Search Indian Kanoon API for judgments."""
        budget_err = self._check_budget()
        if budget_err:
            logger.warning(budget_err)
            return []

        self.rate_limiter.acquire()
        self._increment_call_count()

        params = {"formInput": query, "pagenum": pagenum}
        try:
            resp = self._client.post("/search/", params=params)
            resp.raise_for_status()
            data = resp.json()
            docs = data.get("docs", [])
            return docs if isinstance(docs, list) else []
        except httpx.HTTPStatusError as e:
            logger.error("API search HTTP error %s: %s", e.response.status_code, e)
            return []
        except Exception as e:
            logger.error("API search failed: %s", e)
            return []

    def discover(
        self,
        court: str,
        year: int,
        month: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Iterator[DocRef]:
        """Discover judgments via API search matching court and year."""
        # Convert court name to query term
        court_query = court.lower().replace(" high court", "").replace("of india", "").strip()
        query = f"doctypes:judgments court:{court_query} year:{year}"
        if month:
            query += f" month:{month}"

        page = 0
        yielded = 0
        court_slug = court.lower().replace(" ", "_")

        while True:
            budget_err = self._check_budget()
            if budget_err:
                logger.warning("Discovery stopped: %s", budget_err)
                break

            docs = self.search(query, pagenum=page)
            if not docs:
                break

            for item in docs:
                tid = str(item.get("tid", item.get("doc_id", "")))
                if not tid:
                    continue

                title = item.get("title", f"Document {tid}")
                url = f"https://indiankanoon.org/doc/{tid}/"
                doctype = item.get("doctype", "judgment")

                ref = DocRef(
                    doc_id=tid,
                    court=court,
                    court_slug=court_slug,
                    year=year,
                    month=month,
                    title=title,
                    source_url=url,
                    doctype=doctype,
                )
                yield ref
                yielded += 1
                if limit and yielded >= limit:
                    return

            page += 1
            if len(docs) < 10:  # standard page size is 10 on Indian Kanoon
                break

    def fetch(self, ref: DocRef) -> FetchResult:
        """Fetch full document content from API."""
        budget_err = self._check_budget()
        if budget_err:
            return FetchResult(
                ref=ref,
                success=False,
                source=SourceType.API,
                failure_type=FetchFailureType.BUDGET_EXCEEDED,
                error_message=budget_err,
            )

        if not self.token:
            return FetchResult(
                ref=ref,
                success=False,
                source=SourceType.API,
                failure_type=FetchFailureType.FORBIDDEN,
                error_message="INDIANKANOON_API_TOKEN is not configured.",
            )

        self.rate_limiter.acquire()
        self._increment_call_count()

        start_time = time.perf_counter()
        doc_endpoint = f"/doc/{ref.doc_id}/"

        try:
            resp = self._client.post(doc_endpoint)
            latency = (time.perf_counter() - start_time) * 1000.0

            if resp.status_code == 200:
                data = resp.json()
                raw_text = data.get("doc", "")
                return FetchResult(
                    ref=ref,
                    success=True,
                    source=SourceType.API,
                    raw_content=raw_text if isinstance(raw_text, str) else json.dumps(raw_text),
                    raw_json=data,
                    status_code=200,
                    latency_ms=latency,
                )
            elif resp.status_code == 404:
                return FetchResult(
                    ref=ref,
                    success=False,
                    source=SourceType.API,
                    status_code=404,
                    failure_type=FetchFailureType.NOT_FOUND,
                    error_message=f"Document {ref.doc_id} not found on Indian Kanoon API.",
                    latency_ms=latency,
                )
            elif resp.status_code == 429:
                return FetchResult(
                    ref=ref,
                    success=False,
                    source=SourceType.API,
                    status_code=429,
                    failure_type=FetchFailureType.RATE_LIMITED,
                    error_message="Rate limit exceeded (429) on Indian Kanoon API.",
                    latency_ms=latency,
                )
            elif resp.status_code in (401, 403):
                return FetchResult(
                    ref=ref,
                    success=False,
                    source=SourceType.API,
                    status_code=resp.status_code,
                    failure_type=FetchFailureType.FORBIDDEN,
                    error_message=f"Authentication or permission error ({resp.status_code}): {resp.text}",
                    latency_ms=latency,
                )
            else:
                return FetchResult(
                    ref=ref,
                    success=False,
                    source=SourceType.API,
                    status_code=resp.status_code,
                    failure_type=FetchFailureType.TRANSIENT,
                    error_message=f"Unexpected status code {resp.status_code}: {resp.text[:200]}",
                    latency_ms=latency,
                )

        except (httpx.TimeoutException, httpx.ConnectError, httpx.NetworkError) as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            return FetchResult(
                ref=ref,
                success=False,
                source=SourceType.API,
                failure_type=FetchFailureType.TRANSIENT,
                error_message=f"Network error during API fetch: {type(e).__name__} - {e}",
                latency_ms=latency,
            )
        except Exception as e:
            latency = (time.perf_counter() - start_time) * 1000.0
            return FetchResult(
                ref=ref,
                success=False,
                source=SourceType.API,
                failure_type=FetchFailureType.UNKNOWN,
                error_message=f"Unhandled error fetching doc {ref.doc_id}: {e}",
                latency_ms=latency,
            )

    def close(self) -> None:
        if self._owns_client and self._client:
            self._client.close()
