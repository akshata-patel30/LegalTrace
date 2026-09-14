"""Unit tests for Step 9: Vector Embeddings & Hybrid Search Engine."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from ingestion_source.cli import app
from ingestion_source.models import LegalChunk, RhetoricalRole
from ingestion_source.search import (
    BM25Searcher,
    DenseSemanticEmbedder,
    HybridLegalSearchEngine,
    tokenize,
)

runner = CliRunner()


def test_tokenize():
    text = "Under Section 439 of the CrPC, regular bail is sought in FIR No. 12/2023."
    tokens = tokenize(text)
    assert "section" in tokens
    assert "439" in tokens
    assert "crpc" in tokens
    assert "bail" in tokens
    assert "the" not in tokens
    assert "is" not in tokens


def test_dense_semantic_embedder():
    embedder = DenseSemanticEmbedder(dim=384)
    vec1 = embedder.embed_text("application for regular bail under Section 439 CrPC")
    vec2 = embedder.embed_text("bail application for grant of regular bail")
    vec3 = embedder.embed_text("commercial intellectual property trademark infringement suit")

    assert vec1.shape == (384,)
    # Verify L2 normalization
    import numpy as np
    assert np.isclose(np.linalg.norm(vec1), 1.0, atol=1e-4)

    # Cosine similarity
    sim_related = float(np.dot(vec1, vec2))
    sim_unrelated = float(np.dot(vec1, vec3))

    assert sim_related > sim_unrelated


def test_bm25_searcher():
    bm25 = BM25Searcher()
    docs = [
        tokenize("The applicant was granted bail under Section 439 CrPC."),
        tokenize("Section 37 of the NDPS Act bars bail for commercial quantity."),
        tokenize("Arbitration petition under Section 34 of the Arbitration Act."),
    ]
    bm25.fit(docs)

    scores_ndps = bm25.score_query(tokenize("NDPS Section 37 commercial quantity"))
    assert scores_ndps[1] > scores_ndps[0]
    assert scores_ndps[1] > scores_ndps[2]

    scores_arb = bm25.score_query(tokenize("Section 34 Arbitration"))
    assert scores_arb[2] > scores_arb[0]
    assert scores_arb[2] > scores_arb[1]


def test_hybrid_legal_search_engine_and_filters():
    chunks = [
        LegalChunk(
            chunk_id="chunk_001",
            doc_id="101",
            chunk_index=1,
            court="Delhi High Court",
            court_slug="delhi",
            year=2023,
            case_title="Accused vs State",
            rhetorical_role=RhetoricalRole.OPERATIVE,
            text="Application allowed. The applicant is granted regular bail on terms.",
            text_with_context="[Court: Delhi High Court | Role: OPERATIVE]\nApplication allowed. The applicant is granted regular bail on terms.",
            char_count=65,
            word_count=10,
        ),
        LegalChunk(
            chunk_id="chunk_002",
            doc_id="101",
            chunk_index=2,
            court="Delhi High Court",
            court_slug="delhi",
            year=2023,
            case_title="Accused vs State",
            rhetorical_role=RhetoricalRole.FACTS,
            text="The complainant filed an FIR alleging recovery of commercial narcotics.",
            text_with_context="[Court: Delhi High Court | Role: FACTS]\nThe complainant filed an FIR alleging recovery of commercial narcotics.",
            char_count=70,
            word_count=10,
        ),
        LegalChunk(
            chunk_id="chunk_003",
            doc_id="102",
            chunk_index=1,
            court="Supreme Court",
            court_slug="supremecourt",
            year=2022,
            case_title="Company vs Contractor",
            rhetorical_role=RhetoricalRole.ANALYSIS,
            text="Section 34 of the Arbitration and Conciliation Act provides limited grounds.",
            text_with_context="[Court: Supreme Court | Role: ANALYSIS]\nSection 34 of the Arbitration and Conciliation Act provides limited grounds.",
            char_count=75,
            word_count=10,
        ),
    ]

    engine = HybridLegalSearchEngine(dim=384)
    engine.index_chunks(chunks)

    # 1. Hybrid Search
    resp = engine.search("bail application allowed", mode="hybrid", top_k=2)
    assert len(resp.results) > 0
    assert resp.results[0].chunk_id == "chunk_001"

    # 2. Filtering by role
    resp_facts = engine.search("recovery", mode="hybrid", top_k=2, filter_role="facts")
    assert len(resp_facts.results) == 1
    assert resp_facts.results[0].chunk_id == "chunk_002"
    assert resp_facts.results[0].rhetorical_role == RhetoricalRole.FACTS

    # 3. Dense search
    resp_dense = engine.search("arbitration challenge", mode="dense", top_k=1)
    assert len(resp_dense.results) == 1
    assert resp_dense.results[0].chunk_id == "chunk_003"


def test_search_index_save_and_load(tmp_path: Path):
    chunks = [
        LegalChunk(
            chunk_id="c1",
            doc_id="1",
            chunk_index=1,
            court="Delhi High Court",
            court_slug="delhi",
            year=2023,
            case_title="Case One",
            rhetorical_role=RhetoricalRole.ANALYSIS,
            text="This is a test legal judgment chunk.",
            text_with_context="[Court: Delhi High Court | Role: ANALYSIS]\nThis is a test legal judgment chunk.",
            char_count=35,
            word_count=7,
        )
    ]

    engine = HybridLegalSearchEngine(dim=384)
    engine.index_chunks(chunks)

    index_dir = tmp_path / "test_index"
    engine.save_index(index_dir)

    assert (index_dir / "chunks.json").exists()
    assert (index_dir / "dense_vectors.npy").exists()
    assert (index_dir / "bm25_index.json").exists()

    # Load into fresh engine
    loaded_engine = HybridLegalSearchEngine(dim=384)
    loaded_engine.load_index(index_dir)

    assert len(loaded_engine.chunks) == 1
    resp = loaded_engine.search("test judgment", mode="hybrid", top_k=1)
    assert len(resp.results) == 1
    assert resp.results[0].chunk_id == "c1"
