"""Resumable multithreaded job orchestrator and pipeline execution state machine."""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sqlite3
import threading
import time
from typing import Callable, Optional
import uuid

from ingestion_source.config import Settings, get_settings
from ingestion_source.dedupe import DedupeStore
from ingestion_source.discovery import normalize_court_slug
from ingestion_source.models import (
    DocRef,
    FetchFailureType,
    FetchResult,
    IngestionStatus,
    JobItem,
    JobItemStatus,
    JobReport,
    JobState,
    SourceType,
)
from ingestion_source.normalizer import JudgmentNormalizer
from ingestion_source.rate_limiter import RobotsTxtManager, TokenBucketLimiter
from ingestion_source.sources.api_client import IndianKanoonAPIClient
from ingestion_source.sources.base import DocumentSource
from ingestion_source.sources.html_fallback import IndianKanoonHTMLFallback
from ingestion_source.storage import StorageWriter

logger = logging.getLogger(__name__)


class JobOrchestrator:
    """Manages parallel worker pools, job persistence, retries, deduplication, and resumes."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        db_path: Optional[Path] = None,
        rate_limiter: Optional[TokenBucketLimiter] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.db_path = db_path or self.settings.DB_PATH
        self.dedupe_store = DedupeStore(self.db_path)
        self.storage_writer = StorageWriter(self.settings.INTAKE_OUTPUT_PATH)
        self._db_lock = threading.Lock()

        # Shared rate limiter across all worker threads
        self.rate_limiter = rate_limiter or TokenBucketLimiter(
            rate_rps=1.0 / max(0.5, self.settings.SCRAPE_MIN_DELAY_SECONDS),
            capacity=1.0,
            min_delay_seconds=self.settings.SCRAPE_MIN_DELAY_SECONDS,
            max_delay_seconds=self.settings.SCRAPE_MAX_DELAY_SECONDS,
            backoff_penalty_seconds=self.settings.ADAPTIVE_BACKOFF_SECONDS,
        )

        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=60000;")
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._db_lock, self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_runs (
                    job_id TEXT PRIMARY KEY,
                    mode TEXT,
                    status TEXT,
                    started_at TEXT,
                    updated_at TEXT,
                    total_discovered INTEGER DEFAULT 0,
                    fetched_count INTEGER DEFAULT 0,
                    stored_count INTEGER DEFAULT 0,
                    skipped_count INTEGER DEFAULT 0,
                    failed_count INTEGER DEFAULT 0,
                    api_calls_count INTEGER DEFAULT 0,
                    config_json TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS job_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT,
                    doc_id TEXT,
                    court TEXT,
                    court_slug TEXT,
                    year INTEGER,
                    month INTEGER,
                    title TEXT,
                    source_url TEXT,
                    status TEXT DEFAULT 'pending',
                    failure_type TEXT,
                    error_message TEXT,
                    attempts INTEGER DEFAULT 0,
                    last_attempt_at TEXT,
                    UNIQUE(job_id, doc_id)
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_job_items_queue ON job_items(job_id, status);")
            conn.commit()

    def create_job(self, refs: list[DocRef], mode: Optional[str] = None) -> str:
        """Initialize a new job run in SQLite with discovered references."""
        job_id = f"job_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job_mode = mode or self.settings.MODE
        now_str = datetime.now(timezone.utc).isoformat()

        with self._db_lock, self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO job_runs (
                    job_id, mode, status, started_at, updated_at, total_discovered, config_json
                ) VALUES (?, ?, 'running', ?, ?, ?, ?)
                """,
                (
                    job_id,
                    job_mode,
                    now_str,
                    now_str,
                    len(refs),
                    json.dumps(self.settings.model_dump(mode="json"), default=str),
                ),
            )

            for ref in refs:
                slug = ref.court_slug or normalize_court_slug(ref.court)
                conn.execute(
                    """
                    INSERT OR IGNORE INTO job_items (
                        job_id, doc_id, court, court_slug, year, month, title, source_url, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')
                    """,
                    (
                        job_id,
                        ref.doc_id,
                        ref.court,
                        slug,
                        ref.year,
                        ref.month,
                        ref.title,
                        ref.source_url,
                    ),
                )
            conn.commit()

        logger.info("Initialized job %s with %d items (mode: %s)", job_id, len(refs), job_mode)
        return job_id

    def execute_job(
        self,
        job_id: str,
        limit: Optional[int] = None,
        workers: Optional[int] = None,
        on_progress: Optional[Callable[[DocRef, str], None]] = None,
    ) -> JobReport:
        """Execute or resume an ingestion job using parallel worker threads."""
        job_row = self._get_job_row(job_id)
        if not job_row:
            raise ValueError(f"Job {job_id} does not exist.")

        mode = job_row["mode"]
        num_workers = max(1, workers or self.settings.MAX_WORKERS)

        # Clients share the coordinated rate limiter
        api_client = IndianKanoonAPIClient(settings=self.settings, rate_limiter=self.rate_limiter)
        html_client = IndianKanoonHTMLFallback(settings=self.settings, rate_limiter=self.rate_limiter)

        items = self._get_runnable_items(job_id, limit=limit)
        logger.info("Executing job %s with %d runnable items across %d worker threads", job_id, len(items), num_workers)

        stop_requested = threading.Event()

        def _worker_task(item: dict) -> None:
            if stop_requested.is_set():
                return

            ref = DocRef(
                doc_id=item["doc_id"],
                court=item["court"],
                court_slug=item["court_slug"],
                year=item["year"],
                month=item["month"],
                title=item["title"],
                source_url=item["source_url"],
            )

            # 1. Thread-safe pre-fetch dedupe claim
            if not self.dedupe_store.claim_doc_id(ref.doc_id):
                logger.info("Doc %s already ingested or in-flight; skipping.", ref.doc_id)
                self._update_item_status(
                    job_id, ref.doc_id, JobItemStatus.SKIPPED, error="Already ingested doc_id"
                )
                self._increment_job_counter(job_id, "skipped_count")
                if on_progress:
                    on_progress(ref, "skipped")
                return

            try:
                # 2. Rate-limited fetch via configured mode with retry and polite backoff
                max_retries = max(1, self.settings.MAX_RETRIES)
                fetch_res = None
                for attempt in range(1, max_retries + 1):
                    fetch_res = self._fetch_with_mode(mode, ref, api_client, html_client)
                    if fetch_res.success:
                        break

                    if attempt < max_retries and fetch_res.failure_type in (
                        FetchFailureType.RATE_LIMITED,
                        FetchFailureType.FORBIDDEN,
                        FetchFailureType.CHALLENGE_DETECTED,
                        FetchFailureType.TRANSIENT,
                    ):
                        pause_sec = max(5.0, (self.settings.BACKOFF_FACTOR ** attempt) * self.settings.SCRAPE_MIN_DELAY_SECONDS)
                        logger.warning(
                            "Doc %s received %s on attempt %d/%d. Pausing for %.1fs before retry...",
                            ref.doc_id,
                            fetch_res.failure_type,
                            attempt,
                            max_retries,
                            pause_sec,
                        )
                        if on_progress:
                            on_progress(ref, "retrying")
                        time.sleep(pause_sec)
                    else:
                        break

                if not fetch_res or not fetch_res.success:
                    logger.warning("Fetch failed for %s: %s", ref.doc_id, fetch_res.error_message if fetch_res else "No response")
                    self.dedupe_store.release_doc_claim(ref.doc_id)
                    self._update_item_status(
                        job_id,
                        ref.doc_id,
                        JobItemStatus.FAILED,
                        failure_type=fetch_res.failure_type if fetch_res else FetchFailureType.UNKNOWN,
                        error=fetch_res.error_message if fetch_res else "Fetch failed",
                    )
                    self._increment_job_counter(job_id, "failed_count")
                    if on_progress:
                        on_progress(ref, "failed")
                    return

                # 3. Content Normalization
                judgment = JudgmentNormalizer.normalize(fetch_res)

                # 4. Post-fetch Content Hash Deduplication
                if self.dedupe_store.is_hash_seen(judgment.content_hash):
                    logger.info("Content hash for doc %s already seen; skipping duplicate.", ref.doc_id)
                    self.dedupe_store.release_doc_claim(ref.doc_id)
                    self._update_item_status(
                        job_id, ref.doc_id, JobItemStatus.SKIPPED, error="Duplicate content hash"
                    )
                    self._increment_job_counter(job_id, "skipped_count")
                    if on_progress:
                        on_progress(ref, "skipped")
                    return

                # 5. Thread-safe storage write & manifest append
                self.storage_writer.write_judgment(judgment, fetch_result=fetch_res)
                self.dedupe_store.record_document(judgment)

                # 6. Mark Completed
                self._update_item_status(job_id, ref.doc_id, JobItemStatus.COMPLETED)
                self._increment_job_counter(job_id, "stored_count")
                self._increment_job_counter(job_id, "fetched_count")

                if on_progress:
                    on_progress(ref, "stored")

            except Exception as e:
                logger.error("Unhandled error processing doc %s: %s", ref.doc_id, e)
                self.dedupe_store.release_doc_claim(ref.doc_id)
                self._update_item_status(
                    job_id,
                    ref.doc_id,
                    JobItemStatus.FAILED,
                    failure_type=FetchFailureType.UNKNOWN,
                    error=str(e),
                )
                self._increment_job_counter(job_id, "failed_count")
                if on_progress:
                    on_progress(ref, "failed")

        try:
            if num_workers == 1 or len(items) <= 1:
                for item in items:
                    _worker_task(item)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(_worker_task, item) for item in items]
                    for future in concurrent.futures.as_completed(futures):
                        future.result()

            # Mark complete if all runnable items processed
            remaining = self._get_runnable_items(job_id, limit=1)
            if not remaining:
                self._update_job_status(job_id, "completed")
            else:
                self._update_job_status(job_id, "partial")

        except KeyboardInterrupt:
            logger.warning("Job %s interrupted by user; state is preserved for resume.", job_id)
            stop_requested.set()
            self._update_job_status(job_id, "paused")
        finally:
            total_api_calls = api_client.api_calls_count
            if total_api_calls > 0:
                with self._db_lock, self._get_connection() as conn:
                    conn.execute(
                        "UPDATE job_runs SET api_calls_count = api_calls_count + ? WHERE job_id = ?",
                        (total_api_calls, job_id),
                    )
                    conn.commit()

            api_client.close()
            html_client.close()

        return self.get_report(job_id)

    def _fetch_with_mode(
        self,
        mode: str,
        ref: DocRef,
        api_client: IndianKanoonAPIClient,
        html_client: IndianKanoonHTMLFallback,
    ) -> FetchResult:
        """Fetch document respecting selected mode and retry policies."""
        if mode == "scrape_only":
            return self._fetch_with_retry(html_client, ref)
        elif mode == "hybrid":
            # Try API first if token present
            if self.settings.INDIANKANOON_API_TOKEN:
                res = self._fetch_with_retry(api_client, ref)
                if res.success:
                    return res
                logger.info("API fetch for %s unsuccessful (%s); falling back to polite scraper.", ref.doc_id, res.failure_type)

            return self._fetch_with_retry(html_client, ref)
        else:  # mode == "api"
            return self._fetch_with_retry(api_client, ref)

    def _fetch_with_retry(self, source: DocumentSource, ref: DocRef) -> FetchResult:
        """Execute fetch with retry and exponential backoff on transient errors."""
        max_attempts = self.settings.MAX_RETRIES
        backoff = self.settings.BACKOFF_FACTOR
        attempt = 0

        while attempt < max_attempts:
            attempt += 1
            res = source.fetch(ref)
            if res.success:
                return res

            # Non-retryable errors
            if res.failure_type in (
                FetchFailureType.NOT_FOUND,
                FetchFailureType.DISALLOWED_BY_ROBOTS,
                FetchFailureType.BUDGET_EXCEEDED,
                FetchFailureType.STRUCTURAL_DRIFT,
                FetchFailureType.FORBIDDEN,
                FetchFailureType.CHALLENGE_DETECTED,
            ):
                return res

            if attempt < max_attempts:
                sleep_time = backoff ** attempt
                logger.info("Transient error fetching %s; retrying in %.1fs (attempt %d/%d)", ref.doc_id, sleep_time, attempt, max_attempts)
                time.sleep(sleep_time)

        return res

    def _get_job_row(self, job_id: str) -> Optional[dict]:
        with self._db_lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM job_runs WHERE job_id = ?", (job_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def _get_runnable_items(self, job_id: str, limit: Optional[int] = None) -> list[dict]:
        query = """
            SELECT * FROM job_items
            WHERE job_id = ? AND status IN ('pending', 'failed')
            ORDER BY id ASC
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        with self._db_lock, self._get_connection() as conn:
            cur = conn.execute(query, (job_id,))
            return [dict(r) for r in cur.fetchall()]

    def _update_item_status(
        self,
        job_id: str,
        doc_id: str,
        status: JobItemStatus,
        failure_type: Optional[FetchFailureType] = None,
        error: Optional[str] = None,
    ) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db_lock, self._get_connection() as conn:
            conn.execute(
                """
                UPDATE job_items
                SET status = ?, failure_type = ?, error_message = ?, attempts = attempts + 1, last_attempt_at = ?
                WHERE job_id = ? AND doc_id = ?
                """,
                (
                    status.value,
                    failure_type.value if failure_type else None,
                    error,
                    now_str,
                    job_id,
                    doc_id,
                ),
            )
            conn.commit()

    def _increment_job_counter(self, job_id: str, column: str) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db_lock, self._get_connection() as conn:
            conn.execute(
                f"UPDATE job_runs SET {column} = {column} + 1, updated_at = ? WHERE job_id = ?",
                (now_str, job_id),
            )
            conn.commit()

    def _update_job_status(self, job_id: str, status: str) -> None:
        now_str = datetime.now(timezone.utc).isoformat()
        with self._db_lock, self._get_connection() as conn:
            conn.execute(
                "UPDATE job_runs SET status = ?, updated_at = ? WHERE job_id = ?",
                (status, now_str, job_id),
            )
            conn.commit()

    def get_report(self, job_id: str) -> JobReport:
        """Generate structured summary report for a job ID."""
        row = self._get_job_row(job_id)
        if not row:
            raise ValueError(f"Job {job_id} does not exist.")

        started = datetime.fromisoformat(row["started_at"])
        updated = datetime.fromisoformat(row["updated_at"])
        duration = max(0.0, (updated - started).total_seconds())

        error_summary: dict[str, int] = {}
        with self._db_lock, self._get_connection() as conn:
            cur = conn.execute(
                """
                SELECT failure_type, COUNT(*) as cnt
                FROM job_items
                WHERE job_id = ? AND status = 'failed' AND failure_type IS NOT NULL
                GROUP BY failure_type
                """,
                (job_id,),
            )
            for r in cur.fetchall():
                error_summary[r["failure_type"]] = int(r["cnt"])

        return JobReport(
            job_id=job_id,
            status=row["status"],
            mode=row["mode"],
            started_at=started,
            updated_at=updated,
            duration_seconds=duration,
            total_discovered=row["total_discovered"],
            stored_count=row["stored_count"],
            skipped_count=row["skipped_count"],
            failed_count=row["failed_count"],
            api_calls_count=row["api_calls_count"],
            error_summary=error_summary,
        )
