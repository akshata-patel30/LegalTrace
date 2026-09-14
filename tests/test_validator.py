"""Unit tests for DocumentValidator (Step 2: Document Validation & Integrity QA)."""

import json
from pathlib import Path
import pytest

from ingestion_source.normalizer import compute_sha256_hash
from ingestion_source.validator import DocumentValidator


@pytest.fixture
def validator() -> DocumentValidator:
    return DocumentValidator()


def test_validator_valid_document(tmp_path: Path, validator: DocumentValidator) -> None:
    doc_dir = tmp_path / "12345"
    doc_dir.mkdir()

    text = "This is a valid judgment from the Delhi High Court. " * 12
    (doc_dir / "full_text.txt").write_text(text, encoding="utf-8")
    c_hash = compute_sha256_hash(text)

    meta = {
        "doc_id": "12345",
        "court": "Delhi High Court",
        "title": "State vs John Doe",
        "date": "2023-05-15",
        "bench": ["Justice A", "Justice B"],
        "parties": {"petitioner": "State", "respondent": "John Doe"},
        "case_numbers": ["CRL.A. 1/2023"],
    }
    (doc_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    res = validator.validate_document(doc_dir, expected_hash=c_hash)
    assert res.is_valid is True
    assert res.content_hash_matched is True
    assert res.quality_score == 100.0
    assert len(res.issues) == 0


def test_validator_missing_files(tmp_path: Path, validator: DocumentValidator) -> None:
    doc_dir = tmp_path / "99999"
    doc_dir.mkdir()

    res = validator.validate_document(doc_dir)
    assert res.is_valid is False
    assert res.quality_score == 0.0
    assert any("full_text.txt" in iss.field for iss in res.issues)
    assert any("metadata.json" in iss.field for iss in res.issues)


def test_validator_hash_mismatch(tmp_path: Path, validator: DocumentValidator) -> None:
    doc_dir = tmp_path / "55555"
    doc_dir.mkdir()

    text = "Judgment content that will not match the expected hash. " * 5
    (doc_dir / "full_text.txt").write_text(text, encoding="utf-8")
    meta = {
        "doc_id": "55555",
        "court": "Delhi High Court",
        "title": "A vs B",
    }
    (doc_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    res = validator.validate_document(doc_dir, expected_hash="wrong_sha256_hash_here")
    assert res.is_valid is False
    assert res.content_hash_matched is False
    assert any("hash mismatch" in iss.message for iss in res.issues)


def test_validator_short_text_warning(tmp_path: Path, validator: DocumentValidator) -> None:
    doc_dir = tmp_path / "77777"
    doc_dir.mkdir()

    # Between 100 and 500 chars -> Warning
    text = "Short procedural order directing notice to be issued to the respondent. " * 3
    (doc_dir / "full_text.txt").write_text(text, encoding="utf-8")
    meta = {
        "doc_id": "77777",
        "court": "Delhi High Court",
        "title": "Notice Order",
    }
    (doc_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    res = validator.validate_document(doc_dir)
    assert res.is_valid is True
    assert any("brief" in iss.message for iss in res.issues)


def test_validator_store_batch(tmp_path: Path, validator: DocumentValidator) -> None:
    court_dir = tmp_path / "delhi" / "2023" / "101"
    court_dir.mkdir(parents=True)

    text = "Detailed court judgment analyzing criminal jurisprudence in depth. " * 10
    (court_dir / "full_text.txt").write_text(text, encoding="utf-8")
    c_hash = compute_sha256_hash(text)
    meta = {
        "doc_id": "101",
        "court": "Delhi High Court",
        "title": "Sample Case",
    }
    (court_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    # Write manifest
    manifest_line = {"doc_id": "101", "content_hash": c_hash}
    (tmp_path / "manifest.jsonl").write_text(json.dumps(manifest_line) + "\n", encoding="utf-8")

    report = validator.validate_store(tmp_path)
    assert report.total_documents == 1
    assert report.valid_documents == 1
    assert report.invalid_documents == 0
    assert report.total_errors == 0
