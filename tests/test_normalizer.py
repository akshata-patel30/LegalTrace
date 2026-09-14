"""Unit tests for JudgmentNormalizer."""

from datetime import date
import pytest

from ingestion_source.models import DocRef, FetchResult, SourceType
from ingestion_source.normalizer import (
    JudgmentNormalizer,
    compute_sha256_hash,
    extract_bench_from_text,
    parse_indian_date,
)


def test_parse_indian_date():
    assert parse_indian_date("16 January, 2023") == date(2023, 1, 16)
    assert parse_indian_date("16 Jan 2023") == date(2023, 1, 16)
    assert parse_indian_date("2023-01-16") == date(2023, 1, 16)
    assert parse_indian_date("16-01-2023") == date(2023, 1, 16)
    assert parse_indian_date("1st February, 2023") == date(2023, 2, 1)
    assert parse_indian_date(None) is None
    assert parse_indian_date("Invalid string") is None


def test_extract_bench():
    text = "Bench: Justice Yashwant Varma, Justice D.K. Sharma\nJudgment follows..."
    bench = extract_bench_from_text(text)
    assert len(bench) >= 1
    assert any("Yashwant Varma" in b for b in bench)


def test_normalize_api(sample_doc_ref: DocRef, mock_api_doc_json: dict):
    fetch_result = FetchResult(
        ref=sample_doc_ref,
        success=True,
        source=SourceType.API,
        raw_json=mock_api_doc_json,
        raw_content=mock_api_doc_json["doc"],
        status_code=200,
    )

    judgment = JudgmentNormalizer.normalize(fetch_result)

    assert judgment.doc_id == "1940428"
    assert judgment.court == "Delhi High Court"
    assert judgment.date == date(2023, 1, 16)
    assert judgment.bench == ["Yashwant Varma"]
    assert judgment.content_hash != ""
    assert "Hon'ble Mr. Justice Yashwant Varma" in judgment.full_text


def test_normalize_html(sample_doc_ref: DocRef, mock_judgment_html: str):
    fetch_result = FetchResult(
        ref=sample_doc_ref,
        success=True,
        source=SourceType.SCRAPE,
        raw_content=mock_judgment_html,
        status_code=200,
    )

    judgment = JudgmentNormalizer.normalize(fetch_result)

    assert judgment.doc_id == "1940428"
    assert judgment.court == "Delhi High Court"
    assert judgment.content_hash != ""
    assert "arbitral award" in judgment.full_text.lower()
    assert judgment.structured is not None
    assert len(judgment.structured.paragraphs) >= 3
    assert judgment.structured.parties is not None
    assert judgment.structured.parties.petitioner == "Union Of India"
    assert judgment.structured.parties.respondent == "M/S Bright Power Projects"


def test_parse_parties():
    from ingestion_source.normalizer import parse_parties

    p1 = parse_parties("Lt Foods Limited vs Saraswati Trading Company on 30 January, 2023")
    assert p1 is not None
    assert p1.petitioner == "Lt Foods Limited"
    assert p1.respondent == "Saraswati Trading Company"

    p2 = parse_parties("Union of India v. ABC Corp")
    assert p2 is not None
    assert p2.petitioner == "Union of India"
    assert p2.respondent == "ABC Corp"


def test_extract_neutral_citation():
    from ingestion_source.normalizer import extract_neutral_citation

    nc1 = extract_neutral_citation("Some header\nNeutral Citation Number : 2023/DHC/000684\nMore text")
    assert nc1 == "2023/DHC/000684"


def test_sha256_hash_determinism():
    text1 = "This is a judgment text with   spaces."
    text2 = "This is a judgment text with spaces."
    assert compute_sha256_hash(text1) == compute_sha256_hash(text2)
