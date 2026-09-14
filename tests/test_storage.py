"""Unit tests for StorageWriter."""

from datetime import date, datetime, timezone
import json
from pathlib import Path
import pytest

from ingestion_source.models import (
    DocRef,
    FetchResult,
    IngestionStatus,
    RawJudgment,
    SourceType,
)
from ingestion_source.storage import StorageWriter


def test_storage_writer_api_doc(tmp_path: Path, sample_doc_ref: DocRef, mock_api_doc_json: dict):
    base_dir = tmp_path / "raw-store"
    writer = StorageWriter(base_dir)

    judgment = RawJudgment(
        doc_id="1940428",
        source=SourceType.API,
        source_url=sample_doc_ref.source_url,
        fetched_at=datetime.now(timezone.utc),
        content_hash="mock_hash_123",
        court="Delhi High Court",
        court_slug="delhi",
        title=sample_doc_ref.title or "",
        date=date(2023, 1, 16),
        bench=["Yashwant Varma"],
        full_text="Hon'ble Mr. Justice Yashwant Varma\nFull judgment text...",
    )

    fetch_res = FetchResult(
        ref=sample_doc_ref,
        success=True,
        source=SourceType.API,
        raw_json=mock_api_doc_json,
        raw_content=mock_api_doc_json["doc"],
        status_code=200,
    )

    stored = writer.write_judgment(judgment, fetch_result=fetch_res)

    assert stored.ingestion_status == IngestionStatus.STORED
    assert stored.full_text_path == "delhi/2023/1940428/full_text.txt"
    assert stored.json_raw_path == "delhi/2023/1940428/raw.json"

    # Verify physical files
    doc_dir = base_dir / "delhi" / "2023" / "1940428"
    assert (doc_dir / "full_text.txt").exists()
    assert (doc_dir / "metadata.json").exists()
    assert (doc_dir / "judgment.json").exists()
    assert (doc_dir / "raw.json").exists()

    # Verify judgment.json content
    judgment_json_data = json.loads((doc_dir / "judgment.json").read_text(encoding="utf-8"))
    assert judgment_json_data["doc_id"] == "1940428"
    assert judgment_json_data["court"] == "Delhi High Court"
    assert "bench" in judgment_json_data
    assert "citation_metrics" in judgment_json_data
    assert "paragraphs" in judgment_json_data

    # Verify manifest.jsonl
    manifest_file = base_dir / "manifest.jsonl"
    assert manifest_file.exists()
    lines = manifest_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    entry = json.loads(lines[0])
    assert entry["doc_id"] == "1940428"
    assert entry["court_slug"] == "delhi"
    assert entry["content_hash"] == "mock_hash_123"
    assert entry["judgment_json_path"] == "delhi/2023/1940428/judgment.json"


def test_storage_restructure_stored_documents(tmp_path: Path, mock_judgment_html: str):
    base_dir = tmp_path / "raw-store"
    doc_dir = base_dir / "delhi" / "2023" / "5555"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "raw.html").write_text(mock_judgment_html, encoding="utf-8")

    writer = StorageWriter(base_dir)
    count = writer.restructure_stored_documents()

    assert count == 1
    assert (doc_dir / "judgment.json").exists()
    assert (doc_dir / "metadata.json").exists()
    assert (doc_dir / "full_text.txt").exists()

    structured = json.loads((doc_dir / "judgment.json").read_text(encoding="utf-8"))
    assert structured["doc_id"] == "5555"
    assert len(structured["paragraphs"]) > 0

    # Test export_consolidated_jsonl
    export_file = tmp_path / "export.jsonl"
    exp_count = writer.export_consolidated_jsonl(export_file)
    assert exp_count == 1
    assert export_file.exists()
    exp_lines = export_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(exp_lines) == 1
    loaded = json.loads(exp_lines[0])
    assert loaded["doc_id"] == "5555"


