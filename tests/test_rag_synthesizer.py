"""Unit tests for LegalRAGSynthesizer grounded answer generation."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ingestion_source.config import Settings
from ingestion_source.models import RhetoricalRole, SearchResponse, SearchResultItem
from ingestion_source.rag.planner import LegalQueryPlanner
from ingestion_source.rag.shepardizer import LegalShepardizer
from ingestion_source.rag.synthesizer import (
    CitedPassage,
    GroundedAnswer,
    LegalRAGSynthesizer,
)


@pytest.fixture
def mock_search_engine():
    """Mock search engine returning realistic legal SearchResultItems."""
    engine = MagicMock()
    hits = [
        SearchResultItem(
            rank=1,
            score=0.89,
            dense_score=0.85,
            bm25_score=12.4,
            chunk_id="chunk_1940428_c001",
            doc_id="1940428",
            court="Delhi High Court",
            year=2023,
            case_title="D.S. Rashmi Ranjan vs Chairman, JEE",
            rhetorical_role=RhetoricalRole.ANALYSIS,
            text_snippet="Under the Persons with Disabilities Act, 40 percent minimum disability is mandated.",
            text_with_context="[Context] Under the Persons with Disabilities Act, 40 percent minimum disability is mandated for statutory reservation in medical admissions.",
            statute_mentions=["Persons with Disabilities Act 1995"],
            citations=["AIR 1978 SC 597"],
        ),
        SearchResultItem(
            rank=2,
            score=0.78,
            dense_score=0.74,
            bm25_score=9.8,
            chunk_id="chunk_1940428_c002",
            doc_id="1940428",
            court="Delhi High Court",
            year=2023,
            case_title="D.S. Rashmi Ranjan vs Chairman, JEE",
            rhetorical_role=RhetoricalRole.OPERATIVE,
            text_snippet="In such view of the matter, we find no merit in the contention of the petitioner and the writ petition is dismissed.",
            text_with_context="[Context] The writ petition is dismissed.",
            statute_mentions=[],
            citations=[],
        ),
    ]
    resp = SearchResponse(
        query="disability reservation MBBS",
        mode="hybrid",
        total_indexed=100,
        latency_ms=5.0,
        results=hits,
    )
    engine.search.return_value = resp
    return engine


def test_synthesizer_extractive_fallback(mock_search_engine):
    """Verify deterministic extractive synthesis outputs inline citations and correct schema."""
    settings = Settings(GROQ_API_KEY=None)
    synthesizer = LegalRAGSynthesizer(settings=settings, search_engine=mock_search_engine)

    answer = synthesizer.query("What is the minimum disability threshold for medical course reservation?")

    assert isinstance(answer, GroundedAnswer)
    assert len(answer.citations) == 2
    assert "[1]" in answer.answer_text or "[2]" in answer.answer_text
    assert answer.confidence > 0.5
    assert answer.model_used == "extractive_grounded_fallback"
    assert answer.total_chunks_examined >= 2
    assert "D.S. Rashmi Ranjan" in answer.citations[0].case_title


def test_synthesizer_groq_mocked(mock_search_engine):
    """Verify Groq API integration when GROQ_API_KEY is configured."""
    settings = Settings(GROQ_API_KEY="test_groq_key", RAG_LLM_MODEL="llama-3.1-70b-versatile")
    synthesizer = LegalRAGSynthesizer(settings=settings, search_engine=mock_search_engine)

    mock_llm_reply = (
        "Under Section 39 of the Persons with Disabilities Act, 1995, a minimum of 40% disability "
        "certified by an authorized medical board is mandatory for reservation benefits in educational "
        "institutions including medical colleges [1]. Consequently, an applicant certified with only 20% "
        "disability is ineligible, warranting dismissal of the writ petition [2]."
    )

    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = mock_llm_reply
        mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])
        mock_groq_cls.return_value = mock_client

        answer = synthesizer.query("What is the disability percentage needed for MBBS reservation?")

        assert "40% disability" in answer.answer_text
        assert "[1]" in answer.answer_text
        assert answer.model_used.startswith("groq:")
        assert answer.confidence >= 0.5


def test_synthesizer_empty_evidence():
    """Verify clean handling when search engine returns 0 hits."""
    empty_engine = MagicMock()
    empty_engine.search.return_value = SearchResponse(
        query="xyz empty",
        mode="hybrid",
        total_indexed=0,
        latency_ms=1.0,
        results=[],
    )

    synthesizer = LegalRAGSynthesizer(settings=Settings(GROQ_API_KEY=None), search_engine=empty_engine)
    answer = synthesizer.query("Completely non-existent legal concept 999")

    assert answer.confidence == 0.0
    assert len(answer.citations) == 0
    assert "Insufficient evidence" in answer.answer_text
