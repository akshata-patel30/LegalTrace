"""Unit tests for the Phase 2 database migration pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from ingestion_source.db.migrate import (
    migrate_falkordb,
    migrate_postgres,
    migrate_qdrant,
)
from ingestion_source.db.session import close_db


def test_migrate_postgres_synthetic(tmp_path: Path):
    """Test PostgreSQL migration using temporary JSONL files and in-memory SQLite."""
    import asyncio

    # Create sample cleaned judgment
    cleaned_file = tmp_path / "sample_cleaned.jsonl"
    cleaned_data = {
        "doc_id": "test_001",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "title": "A vs. B on 12 Jan 2023",
        "header": {
            "decision_date": "2023-01-12",
            "docket_numbers": ["W.P.(C) 123/2023"],
            "bench": ["Justice X"],
            "petitioner": "A",
            "respondent": "B",
        },
        "paragraphs": [{"para_id": 1, "text": "Judgment text.", "word_count": 2}],
        "disposition": "allowed",
        "clean_full_text": "Full clean text of judgment.",
    }
    cleaned_file.write_text(json.dumps(cleaned_data) + "\n", encoding="utf-8")

    # Create sample enriched judgment
    enriched_file = tmp_path / "sample_enriched.jsonl"
    enriched_data = {
        "doc_id": "test_001",
        "legal_domain": "civil",
        "case_type": "Writ Petition",
        "bench_type": "single_bench",
        "canonical_judges": ["Justice X"],
        "statutes": [
            {
                "act_name": "Constitution of India",
                "act_code": "Constitution",
                "sections": ["Article 226"],
                "raw_mention": "Article 226 of the Constitution",
            }
        ],
        "citations": [{"citation_text": "AIR 1978 SC 597"}],
    }
    enriched_file.write_text(json.dumps(enriched_data) + "\n", encoding="utf-8")

    # Create sample headnote
    headnotes_file = tmp_path / "sample_headnotes.jsonl"
    headnote_data = {
        "doc_id": "test_001",
        "catchwords": ["CONSTITUTIONAL LAW", "WRIT"],
        "catchwords_line": "CONSTITUTIONAL LAW — WRIT",
        "brief_facts": "Brief facts here.",
        "issues": ["Issue 1"],
        "ratio_decidendi": "Ratio here.",
        "key_precedents": ["AIR 1978 SC 597"],
        "operative_holding": "Petition allowed.",
        "formatted_headnote": "Formatted text...",
        "word_count": 25,
    }
    headnotes_file.write_text(json.dumps(headnote_data) + "\n", encoding="utf-8")

    # Create sample chunk
    chunks_file = tmp_path / "sample_chunks.jsonl"
    chunk_data = {
        "chunk_id": "chunk_test_001_0",
        "doc_id": "test_001",
        "chunk_index": 0,
        "rhetorical_role": "analysis",
        "text": "Analysis chunk body",
        "text_with_context": "[Context] Analysis chunk body",
        "char_count": 19,
        "word_count": 3,
        "statute_mentions": ["Article 226"],
        "citations": [],
    }
    chunks_file.write_text(json.dumps(chunk_data) + "\n", encoding="utf-8")

    test_db_url = "sqlite+aiosqlite:///:memory:"

    # Reset any stray thread-local running loop from prior synchronous test fixtures
    try:
        import asyncio.events
        asyncio.events._set_running_loop(None)
    except Exception:
        pass

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        stats = loop.run_until_complete(
            migrate_postgres(
                db_url=test_db_url,
                cleaned_jsonl_path=cleaned_file,
                enriched_jsonl_path=enriched_file,
                headnotes_jsonl_path=headnotes_file,
                chunks_jsonl_path=chunks_file,
            )
        )

        assert stats["judgments"] == 1
        assert stats["headnotes"] == 1
        assert stats["statutes"] == 1
        assert stats["chunks"] == 1

        loop.run_until_complete(close_db())
    finally:
        loop.close()
        asyncio.set_event_loop(None)


def test_migrate_qdrant_mocked(tmp_path: Path):
    """Test Qdrant migration with mocked QdrantClient."""
    import numpy as np

    # Create fake index dir
    index_dir = tmp_path / "search_index"
    index_dir.mkdir()

    vecs = np.random.randn(5, 384).astype(np.float32)
    np.save(index_dir / "dense_vectors.npy", vecs)

    chunks = [
        {"chunk_id": f"chunk_{i}", "doc_id": "d1", "court": "Delhi", "year": 2023, "text": f"t{i}"}
        for i in range(5)
    ]
    (index_dir / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")

    with patch("ingestion_source.db.vector_store.QdrantClient") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.collection_exists.return_value = True
        mock_count = MagicMock()
        mock_count.count = 5
        mock_instance.count.return_value = mock_count
        mock_client_cls.return_value = mock_instance

        res = migrate_qdrant(
            qdrant_url="http://mock-qdrant:6333",
            index_dir=index_dir,
            collection_name="test_chunks",
        )

        assert res["upserted_points"] == 5
        assert res["total_points_in_qdrant"] == 5


def test_migrate_falkordb_mocked(tmp_path: Path):
    """Test FalkorDB migration with mocked FalkorDB client."""
    edges_file = tmp_path / "edges.jsonl"
    edges_file.write_text(
        json.dumps({
            "citing_doc_id": "doc_1",
            "citing_title": "Case 1",
            "citing_court": "Delhi High Court",
            "citing_year": 2023,
            "cited_citation": "AIR 1978 SC 597",
            "cited_reporter": "AIR",
            "cited_year": 1978,
            "cited_title": "Maneka Gandhi",
            "treatment": "relied_upon",
            "rhetorical_role": "analysis",
        }) + "\n"
    )

    nodes_file = tmp_path / "nodes.jsonl"
    nodes_file.write_text(
        json.dumps({
            "precedent_id": "prec_air_1978_sc_597",
            "canonical_citation": "AIR 1978 SC 597",
            "reporter": "AIR",
            "citation_year": 1978,
            "case_title": "Maneka Gandhi",
            "in_degree": 5,
        }) + "\n"
    )

    with patch("ingestion_source.db.graph_store.FalkorDB") as mock_falkor_cls:
        mock_db = MagicMock()
        mock_graph = MagicMock()
        mock_db.select_graph.return_value = mock_graph
        mock_falkor_cls.return_value = mock_db

        # Mock stats
        j_res = MagicMock(result_set=[[1]])
        p_res = MagicMock(result_set=[[1]])
        e_res = MagicMock(result_set=[[1]])
        mock_graph.query.side_effect = [
            MagicMock(), # index 1
            MagicMock(), # index 2
            MagicMock(), # index 3
            MagicMock(), # merge precedent
            MagicMock(), # merge judgment
            MagicMock(), # merge precedent fallback
            MagicMock(), # create edge
            j_res, p_res, e_res, # stats
            MagicMock(result_set=[]), # top authorities
        ]

        res = migrate_falkordb(
            host="mock-falkordb",
            port=6379,
            edges_path=edges_file,
            nodes_path=nodes_file,
        )

        assert res["graph_name"] == "citation_network"
        assert res["ingestion_stats"]["edges_created"] == 1
