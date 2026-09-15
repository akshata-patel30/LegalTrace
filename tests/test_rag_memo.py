"""Unit tests for LegalMemoGenerator automated court brief and memo generator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from ingestion_source.models import RhetoricalRole, SearchResponse, SearchResultItem
from ingestion_source.rag.memo_generator import LegalDocument, LegalMemoGenerator
from ingestion_source.rag.shepardizer import LegalShepardizer
from ingestion_source.rag.synthesizer import LegalRAGSynthesizer


@pytest.fixture
def mock_raw_store(tmp_path: Path) -> Path:
    """Create a mock raw-store directory for document 9999."""
    doc_dir = tmp_path / "delhi" / "2023" / "9999"
    doc_dir.mkdir(parents=True, exist_ok=True)

    cleaned = {
        "doc_id": "9999",
        "title": "State of NCT of Delhi vs. Raman Sharma",
        "court": "Delhi High Court",
        "disposition": "allowed",
        "header": {
            "decision_date": "2023-04-10",
            "bench": ["Hon'ble Justice S. K. Roy"],
            "docket_numbers": ["CRL.M.C. 550/2023"],
            "petitioner": "State of NCT of Delhi",
            "respondent": "Raman Sharma",
        },
    }
    (doc_dir / "cleaned.json").write_text(json.dumps(cleaned), encoding="utf-8")

    headnote = {
        "catchwords_line": "CRIMINAL LAW — SECTION 482 CRPC — QUASHING OF FIR",
        "brief_facts": "The parties entered into an amicable settlement.",
        "issues": ["Whether FIR under Section 420 IPC can be quashed upon settlement."],
        "ratio_decidendi": "Inherent powers can be invoked to quash compoundable and commercial offences upon settlement.",
        "operative_holding": "FIR No. 101/2022 and all consequential proceedings are hereby QUASHED.",
    }
    (doc_dir / "headnote.json").write_text(json.dumps(headnote), encoding="utf-8")

    enriched = {
        "canonical_judges": ["S. K. Roy"],
        "statutes": [{"act_name": "Code of Criminal Procedure, 1973", "sections": ["Section 482"]}],
        "citations": [{"citation_text": "AIR 1978 SC 597"}],
    }
    (doc_dir / "enriched.json").write_text(json.dumps(enriched), encoding="utf-8")

    return tmp_path


@pytest.fixture
def mock_search_engine():
    """Mock search engine returning search hit for bail/quashing."""
    engine = MagicMock()
    hits = [
        SearchResultItem(
            rank=1,
            score=0.88,
            dense_score=0.85,
            bm25_score=11.2,
            chunk_id="ch_1",
            doc_id="9999",
            court="Delhi High Court",
            year=2023,
            case_title="State vs. Raman Sharma",
            rhetorical_role=RhetoricalRole.ANALYSIS,
            text_snippet="The Supreme Court has repeatedly held that Section 482 CrPC is intended to prevent abuse of the process of any court.",
            text_with_context="[Context] The Supreme Court has repeatedly held that Section 482 CrPC is intended to prevent abuse of the process of any court.",
            statute_mentions=["Section 482 CrPC"],
            citations=["AIR 1978 SC 597"],
        )
    ]
    engine.search.return_value = SearchResponse(
        query="Section 482 CrPC",
        mode="hybrid",
        total_indexed=1,
        latency_ms=3.0,
        results=hits,
    )
    return engine


def test_generate_case_brief(mock_raw_store: Path):
    """Verify appellate case brief formatting from stored judgment artifacts."""
    gen = LegalMemoGenerator(raw_store_dir=mock_raw_store)
    brief = gen.generate_case_brief(doc_id="9999", court="delhi", year=2023)

    assert isinstance(brief, LegalDocument)
    assert brief.doc_type == "case_brief"
    assert "State of NCT of Delhi vs. Raman Sharma" in brief.title
    assert "APPELLATE CHAMBER CASE BRIEF" in brief.content
    assert "SECTION 482 CRPC" in brief.content
    assert "Hon'ble Justice S. K. Roy" in brief.content
    assert "FIR No. 101/2022 and all consequential proceedings are hereby QUASHED" in brief.content


def test_generate_legal_memorandum(mock_search_engine):
    """Verify formal legal memorandum structure with evidence table."""
    gen = LegalMemoGenerator()
    memo = gen.generate_legal_memorandum(
        query="Scope of interference with commercial disputes under Section 482 CrPC",
        engine=mock_search_engine,
    )

    assert isinstance(memo, LegalDocument)
    assert memo.doc_type == "legal_memorandum"
    assert "LEGAL RESEARCH MEMORANDUM" in memo.content
    assert "QUESTION PRESENTED / ISSUE" in memo.content
    assert "EVIDENCE AND TABLE OF AUTHORITIES" in memo.content
    assert len(memo.citations) > 0


def test_generate_bail_application(mock_search_engine):
    """Verify formal court pleading generation for Section 439 CrPC bail application."""
    gen = LegalMemoGenerator()
    bail_draft = gen.generate_bail_application(
        applicant_name="Vikramaditya Mehta",
        fir_no="45/2023",
        police_station="Hauz Khas",
        sections_invoked=["Section 420 IPC", "Section 120B IPC"],
        custody_date="15th January 2023",
        facts_summary="The Applicant was arrested on alleged commercial irregularities.",
        engine=mock_search_engine,
    )

    assert isinstance(bail_draft, LegalDocument)
    assert bail_draft.doc_type == "bail_application"
    assert "IN THE HIGH COURT OF DELHI AT NEW DELHI" in bail_draft.content
    assert "VIKRAMADITYA MEHTA" in bail_draft.content
    assert "SECTION 439 OF THE CODE OF CRIMINAL PROCEDURE" in bail_draft.content
    assert "FIR NO. 45/2023" in bail_draft.content
    assert "HAUZ KHAS" in bail_draft.content
