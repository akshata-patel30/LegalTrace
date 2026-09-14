"""SQLite-backed thread-safe deduplication store for doc_id and SHA-256 content hashes."""

from __future__ import annotations

import logging
from pathlib import Path
import sqlite3
import threading
from typing import Optional, Set

from ingestion_source.models import IngestionStatus, RawJudgment

logger = logging.getLogger(__name__)


class DedupeStore:
    """Thread-safe persistent store ensuring pre-fetch and post-fetch deduplication."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._in_flight_doc_ids: Set[str] = set()
        self._seen_doc_ids: Set[str] = set()
        self._seen_content_hashes: Set[str] = set()
        self._init_db()
        self._warm_cache()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA busy_timeout=60000;")
        return conn

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS seen_docs (
                    doc_id TEXT PRIMARY KEY,
                    content_hash TEXT,
                    court TEXT,
                    year INTEGER,
                    source TEXT,
                    status TEXT,
                    title TEXT,
                    fetched_at TEXT
                );
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS content_hashes (
                    content_hash TEXT PRIMARY KEY,
                    doc_id TEXT,
                    first_seen_at TEXT
                );
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_docs_hash ON seen_docs(content_hash);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_seen_docs_status ON seen_docs(status);")
            conn.commit()

    def _warm_cache(self) -> None:
        """Pre-populate in-memory sets from SQLite for microsecond-fast thread-safe dedupe checks."""
        with self._lock, self._get_connection() as conn:
            cur_docs = conn.execute(
                "SELECT doc_id FROM seen_docs WHERE status = ?",
                (IngestionStatus.STORED.value,),
            )
            self._seen_doc_ids = {row[0] for row in cur_docs.fetchall()}

            cur_hashes = conn.execute("SELECT content_hash FROM content_hashes")
            self._seen_content_hashes = {row[0] for row in cur_hashes.fetchall()}
        logger.debug("Dedupe cache warmed: %d docs, %d hashes", len(self._seen_doc_ids), len(self._seen_content_hashes))

    def claim_doc_id(self, doc_id: str) -> bool:
        """Atomically test and claim a doc_id to prevent parallel threads from fetching the same doc."""
        with self._lock:
            if doc_id in self._seen_doc_ids or doc_id in self._in_flight_doc_ids:
                return False
            self._in_flight_doc_ids.add(doc_id)
            return True

    def release_doc_claim(self, doc_id: str) -> None:
        """Release an in-flight claim if processing fails or is skipped."""
        with self._lock:
            self._in_flight_doc_ids.discard(doc_id)

    def is_doc_id_seen(self, doc_id: str) -> bool:
        """Check if a doc_id has already been successfully stored."""
        with self._lock:
            if doc_id in self._seen_doc_ids:
                return True

        # Fallback check SQLite in case written by external process
        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                "SELECT 1 FROM seen_docs WHERE doc_id = ? AND status = ?",
                (doc_id, IngestionStatus.STORED.value),
            )
            seen = cur.fetchone() is not None
            if seen:
                self._seen_doc_ids.add(doc_id)
            return seen

    def is_hash_seen(self, content_hash: str) -> bool:
        """Check if a normalized content hash has already been stored."""
        if not content_hash:
            return False
        with self._lock:
            if content_hash in self._seen_content_hashes:
                return True

        with self._lock, self._get_connection() as conn:
            cur = conn.execute(
                "SELECT 1 FROM content_hashes WHERE content_hash = ?",
                (content_hash,),
            )
            seen = cur.fetchone() is not None
            if seen:
                self._seen_content_hashes.add(content_hash)
            return seen

    def get_doc_by_id(self, doc_id: str) -> Optional[dict]:
        """Fetch metadata dictionary for a previously recorded doc_id."""
        with self._lock, self._get_connection() as conn:
            cur = conn.execute("SELECT * FROM seen_docs WHERE doc_id = ?", (doc_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def record_document(self, judgment: RawJudgment) -> None:
        """Insert or update judgment ingestion record and content hash in SQLite and memory cache."""
        year = judgment.date.year if judgment.date else None
        fetched_at_str = judgment.fetched_at.isoformat()

        with self._lock:
            self._in_flight_doc_ids.discard(judgment.doc_id)
            if judgment.ingestion_status == IngestionStatus.STORED:
                self._seen_doc_ids.add(judgment.doc_id)
                if judgment.content_hash:
                    self._seen_content_hashes.add(judgment.content_hash)

            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO seen_docs (
                        doc_id, content_hash, court, year, source, status, title, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        judgment.doc_id,
                        judgment.content_hash,
                        judgment.court,
                        year,
                        judgment.source.value,
                        judgment.ingestion_status.value,
                        judgment.title,
                        fetched_at_str,
                    ),
                )
                if judgment.content_hash and judgment.ingestion_status == IngestionStatus.STORED:
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO content_hashes (content_hash, doc_id, first_seen_at)
                        VALUES (?, ?, ?)
                        """,
                        (judgment.content_hash, judgment.doc_id, fetched_at_str),
                    )
                conn.commit()

    def get_stored_count(self) -> int:
        """Return total count of successfully stored documents."""
        with self._lock:
            return len(self._seen_doc_ids)
