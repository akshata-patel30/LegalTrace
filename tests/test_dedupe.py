"""Unit tests for DedupeStore."""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from ingestion_source.dedupe import DedupeStore
from ingestion_source.models import IngestionStatus, RawJudgment, SourceType


def test_dedupe_store_operations(tmp_path: Path):
    db_path = tmp_path / "test_dedupe.sqlite"
    store = DedupeStore(db_path)

    assert store.is_doc_id_seen("1001") is False
    assert store.is_hash_seen("hash_abc") is False

    judgment = RawJudgment(
        doc_id="1001",
        source=SourceType.API,
        source_url="https://indiankanoon.org/doc/1001/",
        fetched_at=datetime.now(timezone.utc),
        content_hash="hash_abc",
        court="Delhi High Court",
        title="Case A vs Case B",
        ingestion_status=IngestionStatus.STORED,
    )

    store.record_document(judgment)

    assert store.is_doc_id_seen("1001") is True
    assert store.is_hash_seen("hash_abc") is True
    assert store.get_stored_count() == 1

    # Record another doc with same hash (duplicate text)
    judgment_dup = RawJudgment(
        doc_id="1002",
        source=SourceType.SCRAPE,
        source_url="https://indiankanoon.org/doc/1002/",
        fetched_at=datetime.now(timezone.utc),
        content_hash="hash_abc",
        court="Delhi High Court",
        title="Duplicate Case",
        ingestion_status=IngestionStatus.STORED,
    )
    store.record_document(judgment_dup)

    assert store.is_doc_id_seen("1002") is True
    assert store.get_stored_count() == 2


def test_dedupe_atomic_claiming(tmp_path: Path):
    db_path = tmp_path / "test_claim.sqlite"
    store = DedupeStore(db_path)

    # First claim should succeed
    assert store.claim_doc_id("5001") is True
    # Concurrent second claim on same doc_id must fail
    assert store.claim_doc_id("5001") is False

    # After releasing, claiming should succeed again
    store.release_doc_claim("5001")
    assert store.claim_doc_id("5001") is True

    # Once stored, claim should fail permanently
    judgment = RawJudgment(
        doc_id="5001",
        source=SourceType.SCRAPE,
        source_url="https://indiankanoon.org/doc/5001/",
        fetched_at=datetime.now(timezone.utc),
        content_hash="hash_5001",
        court="Delhi High Court",
        title="Claim Case",
        ingestion_status=IngestionStatus.STORED,
    )
    store.record_document(judgment)
    assert store.claim_doc_id("5001") is False
