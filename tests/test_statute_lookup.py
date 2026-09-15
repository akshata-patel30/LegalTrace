"""Unit tests for Phase 4 Statute-to-Ratio Reverse Lookup Engine."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest

from ingestion_source.rag.statute_lookup import (
    StatuteRatioLookup,
    StatuteRatioResponse,
    StatuteRatioResult,
    normalize_act_name,
    normalize_section,
)


def test_normalize_act_name():
    """Verify standard act names and abbreviations normalize to canonical full title."""
    assert normalize_act_name("crpc") == "Code of Criminal Procedure, 1973"
    assert normalize_act_name("CrPC") == "Code of Criminal Procedure, 1973"
    assert normalize_act_name("Criminal Procedure Code") == "Code of Criminal Procedure, 1973"
    assert normalize_act_name("ipc") == "Indian Penal Code, 1860"
    assert normalize_act_name("Indian Penal Code") == "Indian Penal Code, 1860"
    assert normalize_act_name("ndps") == "Narcotic Drugs and Psychotropic Substances Act, 1985"
    assert normalize_act_name("NDPS Act") == "Narcotic Drugs and Psychotropic Substances Act, 1985"
    assert normalize_act_name("arbitration") == "Arbitration and Conciliation Act, 1996"
    assert normalize_act_name("Arbitration Act") == "Arbitration and Conciliation Act, 1996"
    assert normalize_act_name("constitution") == "Constitution of India"
    assert normalize_act_name("Customs Act") == "Customs Act, 1962"
    assert normalize_act_name("Specific Relief Act") == "Specific Relief Act, 1963"
    assert normalize_act_name("Unknown Statute 2024") == "Unknown Statute 2024"


def test_normalize_section():
    """Verify section strings with various prefixes are stripped to numeric clause."""
    assert normalize_section("Section 482") == "482"
    assert normalize_section("Sec. 438") == "438"
    assert normalize_section("s. 34") == "34"
    assert normalize_section("302") == "302"
    assert normalize_section("Section 52A") == "52A"
    assert normalize_section("Clause 9") == "9"


@pytest.mark.asyncio
async def test_statute_lookup_filesystem_real_corpus():
    """Verify reverse lookup on local raw-store finds judgments mentioning statute provisions."""
    engine = StatuteRatioLookup(raw_store_dir=Path("raw-store"))
    resp = await engine.lookup(act_name="NDPS", section="37", top_k=5)

    assert isinstance(resp, StatuteRatioResponse)
    assert resp.canonical_act == "Narcotic Drugs and Psychotropic Substances Act, 1985"
    assert resp.section == "37"
    assert resp.total_results >= 1
    assert len(resp.results) >= 1
    assert len(resp.key_principles) >= 1

    first = resp.results[0]
    assert isinstance(first, StatuteRatioResult)
    assert first.doc_id in ["104774187", "106735942", "19367453"]
    assert first.rhetorical_role in ["analysis", "ratio", "operative"]
    assert first.relevance_score > 0.0
    assert len(first.ratio_text) > 20


@pytest.mark.asyncio
async def test_statute_lookup_empty_query():
    """Verify lookup returns empty results and default principles for unmentioned sections."""
    engine = StatuteRatioLookup(raw_store_dir=Path("raw-store"))
    resp = await engine.lookup(act_name="Customs Act", section="99999", top_k=5)

    assert resp.total_results == 0
    assert len(resp.results) == 0
    assert len(resp.key_principles) >= 1
    assert "Section 99999" in resp.key_principles[0]


@pytest.mark.asyncio
async def test_statute_lookup_synthetic_dir(tmp_path: Path):
    """Verify lookup operates cleanly against a synthetic mock raw-store hierarchy."""
    doc_dir = tmp_path / "delhi" / "2023" / "mock_doc_999"
    doc_dir.mkdir(parents=True)

    enriched_data = {
        "doc_id": "mock_doc_999",
        "case_title": "Arbitration Petitioner v. Respondent",
        "statutes": [
            {
                "act_name": "Arbitration and Conciliation Act, 1996",
                "act_code": "Arbitration Act",
                "sections": ["34", "9"],
                "raw_mention": "Section 34 of the Arbitration Act, 1996",
            }
        ],
    }
    (doc_dir / "enriched.json").write_text(json.dumps(enriched_data), encoding="utf-8")

    meta_data = {
        "doc_id": "mock_doc_999",
        "title": "Arbitration Petitioner v. Respondent",
        "court": "Delhi High Court",
        "year": 2023,
    }
    (doc_dir / "metadata.json").write_text(json.dumps(meta_data), encoding="utf-8")

    headnote_data = {
        "doc_id": "mock_doc_999",
        "ratio_decidendi": "Under Section 34 of the Arbitration Act, the court does not sit in appeal over the arbitral award.",
    }
    (doc_dir / "headnote.json").write_text(json.dumps(headnote_data), encoding="utf-8")

    classified_data = {
        "doc_id": "mock_doc_999",
        "classified_paragraphs": [
            {
                "predicted_role": "analysis",
                "clean_text": "The scope of interference under Section 34 is narrow and limited to patent illegality or conflict with public policy.",
            }
        ],
    }
    (doc_dir / "classified.json").write_text(json.dumps(classified_data), encoding="utf-8")

    engine = StatuteRatioLookup(raw_store_dir=tmp_path)
    resp = await engine.lookup(act_name="Arbitration", section="34", top_k=5)

    assert resp.total_results >= 1
    assert resp.canonical_act == "Arbitration and Conciliation Act, 1996"
    assert resp.results[0].doc_id == "mock_doc_999"
    assert "Section 34" in resp.results[0].ratio_text


@pytest.mark.asyncio
async def test_statute_lookup_postgres_mocked():
    """Verify PostgreSQL lookup logic with mocked SQLAlchemy session."""
    engine = StatuteRatioLookup()

    mock_mention = MagicMock()
    mock_mention.doc_id = "doc_pg_1"
    mock_mention.act_name = "Code of Criminal Procedure, 1973"
    mock_mention.sections = ["482"]
    mock_mention.raw_mention = "Section 482 CrPC"

    mock_judgment = MagicMock()
    mock_judgment.doc_id = "doc_pg_1"
    mock_judgment.title = "Kailash v. State"
    mock_judgment.court = "Delhi High Court"
    mock_judgment.year = 2023
    mock_judgment.disposition = "quashed"

    mock_chunk = MagicMock()
    mock_chunk.doc_id = "doc_pg_1"
    mock_chunk.rhetorical_role = "analysis"
    mock_chunk.text = "Inherent power under Section 482 CrPC to prevent abuse of the process of court."

    mock_session = AsyncMock()
    # First query result: mentions & judgments
    mock_query_res1 = MagicMock()
    mock_query_res1.all.return_value = [(mock_mention, mock_judgment)]

    # Second query result: chunks
    mock_query_res2 = MagicMock()
    mock_query_res2.scalars.return_value.all.return_value = [mock_chunk]

    mock_session.execute.side_effect = [mock_query_res1, mock_query_res2]

    results = await engine._lookup_via_postgres(
        canonical_act="Code of Criminal Procedure, 1973",
        section="482",
        session=mock_session,
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].doc_id == "doc_pg_1"
    assert results[0].case_title == "Kailash v. State"
    assert results[0].section == "482"


@pytest.mark.asyncio
async def test_statute_lookup_qdrant_mocked():
    """Verify Qdrant vector store lookup logic with mocked Qdrant client."""
    engine = StatuteRatioLookup()

    mock_qdrant = MagicMock()
    mock_qdrant.COLLECTION_NAME = "legal_chunks"

    mock_point = MagicMock()
    mock_point.score = 0.89
    mock_point.payload = {
        "doc_id": "doc_qd_1",
        "case_title": "Deepak v. Union of India",
        "court": "Delhi High Court",
        "year": 2023,
        "rhetorical_role": "analysis",
        "text": "The statutory conditions under Section 37 must be satisfied before bail can be granted.",
    }
    mock_res = MagicMock()
    mock_res.points = [mock_point]
    mock_qdrant.client.query_points.return_value = mock_res

    results = await engine._lookup_via_qdrant(
        canonical_act="Narcotic Drugs and Psychotropic Substances Act, 1985",
        section="37",
        qdrant_client=mock_qdrant,
        top_k=5,
    )

    assert len(results) == 1
    assert results[0].doc_id == "doc_qd_1"
    assert results[0].case_title == "Deepak v. Union of India"
    assert results[0].relevance_score == 0.89
