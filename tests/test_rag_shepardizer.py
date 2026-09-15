"""Unit tests for LegalShepardizer precedent treatment and good-law verification."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from ingestion_source.rag.shepardizer import (
    LawStatus,
    LegalShepardizer,
    ShepardResult,
)


@pytest.fixture
def mock_edges_file(tmp_path: Path) -> Path:
    """Fixture with synthetic citation edges for multiple treatment scenarios."""
    f = tmp_path / "mock_edges.jsonl"
    edges = [
        # Precedent 1: Reliably followed
        {
            "citing_doc_id": "101",
            "citing_title": "Case Alpha vs. State",
            "citing_court": "Delhi High Court",
            "citing_year": 2023,
            "cited_citation": "AIR 1978 SC 597",
            "treatment": "relied_upon",
            "rhetorical_role": "analysis",
            "context_sentence": "We follow the landmark principles laid down in Maneka Gandhi.",
        },
        # Precedent 2: Distinguished
        {
            "citing_doc_id": "102",
            "citing_title": "Case Beta vs. UOI",
            "citing_court": "Delhi High Court",
            "citing_year": 2023,
            "cited_citation": "2010 SCC OnLine Del 45",
            "treatment": "distinguished",
            "rhetorical_role": "analysis",
            "context_sentence": "The judgment in 2010 SCC OnLine Del 45 is distinguishable on facts.",
        },
        {
            "citing_doc_id": "103",
            "citing_title": "Case Gamma vs. CBI",
            "citing_court": "Delhi High Court",
            "citing_year": 2022,
            "cited_citation": "2010 SCC OnLine Del 45",
            "treatment": "distinguished",
            "rhetorical_role": "analysis",
            "context_sentence": "Facts in that decision were materially different.",
        },
        # Precedent 3: Overruled
        {
            "citing_doc_id": "104",
            "citing_title": "Case Delta vs. Union",
            "citing_court": "Supreme Court of India",
            "citing_year": 2023,
            "cited_citation": "ILR 1995 Del 12",
            "treatment": "overruled",
            "rhetorical_role": "analysis",
            "context_sentence": "The view taken in ILR 1995 Del 12 no longer holds the field and is overruled.",
        },
    ]
    with open(f, "w", encoding="utf-8") as out:
        for e in edges:
            out.write(json.dumps(e) + "\n")
    return f


def test_shepardize_good_law(mock_edges_file: Path):
    """Verify precedent with relied_upon treatment is classified as good_law."""
    shepardizer = LegalShepardizer(edges_path=mock_edges_file)
    result = shepardizer.check("AIR 1978 SC 597")

    assert result.status == LawStatus.GOOD_LAW
    assert result.is_good_law is True
    assert result.total_citations == 1
    assert result.treatment_summary.get("relied_upon") == 1
    assert result.warning_message is None


def test_shepardize_distinguished(mock_edges_file: Path):
    """Verify precedent with multiple distinguished treatments is flagged with caution."""
    shepardizer = LegalShepardizer(edges_path=mock_edges_file)
    result = shepardizer.check("2010 SCC OnLine Del 45")

    assert result.status == LawStatus.DISTINGUISHED
    assert result.is_good_law is True  # Still technically law, but distinguished
    assert len(result.distinguishing_cases) == 2
    assert "DISTINGUISHED" in (result.warning_message or "")


def test_shepardize_overruled(mock_edges_file: Path):
    """Verify overruled precedent is marked not good law with critical alert."""
    shepardizer = LegalShepardizer(edges_path=mock_edges_file)
    result = shepardizer.check("ILR 1995 Del 12")

    assert result.status == LawStatus.OVERRULED
    assert result.is_good_law is False
    assert len(result.overruling_cases) == 1
    assert "OVERRULED" in (result.warning_message or "")


def test_shepardize_unknown(mock_edges_file: Path):
    """Verify unmentioned citation returns UNKNOWN without crashing."""
    shepardizer = LegalShepardizer(edges_path=mock_edges_file)
    result = shepardizer.check("Uncited Case Citation 2099")

    assert result.status == LawStatus.UNKNOWN
    assert result.total_citations == 0
    assert result.is_good_law is True


def test_shepardize_with_falkordb_mock():
    """Verify graph_store integration path when FalkorGraphStore returns records."""
    mock_store = MagicMock()
    mock_store.get_precedent_treatment.return_value = [
        {
            "citing_doc_id": "2001",
            "citing_title": "Govt vs. Corp",
            "citing_court": "Delhi High Court",
            "citing_year": 2023,
            "treatment": "overruled",
            "rhetorical_role": "analysis",
            "context": "We overrule this proposition.",
        }
    ]

    shepardizer = LegalShepardizer(graph_store=mock_store)
    result = shepardizer.check("2000 SCC 100")

    mock_store.get_precedent_treatment.assert_called_once_with("2000 SCC 100")
    assert result.status == LawStatus.OVERRULED
    assert result.is_good_law is False
