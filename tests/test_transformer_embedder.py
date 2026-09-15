"""Unit tests for Phase 4 TransformerSemanticEmbedder and hybrid search integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from ingestion_source.models import LegalChunk, RhetoricalRole
from ingestion_source.search import (
    DenseSemanticEmbedder,
    HybridLegalSearchEngine,
    TransformerSemanticEmbedder,
)


def test_transformer_embedder_initialization():
    """Verify embedder initializes with correct parameters."""
    embedder = TransformerSemanticEmbedder(model_name="all-MiniLM-L6-v2", dim=384)
    assert embedder.model_name == "all-MiniLM-L6-v2"
    assert embedder.dim == 384
    assert embedder._model is None
    assert isinstance(embedder._fallback_embedder, DenseSemanticEmbedder)


def test_transformer_embedder_fallback_when_model_load_fails():
    """Verify embedder gracefully falls back to DenseSemanticEmbedder if model fails to load."""
    embedder = TransformerSemanticEmbedder(model_name="non-existent-model-xyz", dim=384)
    with patch.object(embedder, "_load_model", return_value=None):
        vec = embedder.embed_text("Section 482 CrPC quashing of FIR")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (384,)
        assert vec.dtype == np.float32
        norm = np.linalg.norm(vec)
        assert np.isclose(norm, 1.0, atol=1e-4)

        batch_vecs = embedder.embed_batch(["text one", "text two"])
        assert batch_vecs.shape == (2, 384)
        assert batch_vecs.dtype == np.float32


def test_transformer_embedder_mocked_model():
    """Verify embedder uses sentence_transformers model when available."""
    embedder = TransformerSemanticEmbedder(model_name="all-MiniLM-L6-v2", dim=384)
    mock_model = MagicMock()
    # Mock single text encoding
    dummy_vec = np.random.randn(384).astype(np.float32)
    dummy_vec /= np.linalg.norm(dummy_vec)
    mock_model.encode.return_value = dummy_vec

    with patch.object(embedder, "_load_model", return_value=mock_model):
        res = embedder.embed_text("Bail application under Section 439 CrPC")
        assert res.shape == (384,)
        mock_model.encode.assert_called_once()

    # Mock batch encoding
    dummy_batch = np.random.randn(3, 384).astype(np.float32)
    dummy_batch /= np.linalg.norm(dummy_batch, axis=1, keepdims=True)
    mock_model.encode.return_value = dummy_batch

    with patch.object(embedder, "_load_model", return_value=mock_model):
        batch_res = embedder.embed_batch(["doc 1", "doc 2", "doc 3"])
        assert batch_res.shape == (3, 384)


def test_hybrid_search_engine_with_transformer_embedder():
    """Verify HybridLegalSearchEngine supports embedder_type='transformer'."""
    engine = HybridLegalSearchEngine(dim=384, embedder_type="transformer")
    assert isinstance(engine.embedder, TransformerSemanticEmbedder)

    # Index sample chunk
    chunk = LegalChunk(
        chunk_id="chunk_test_001",
        doc_id="doc_test_100",
        chunk_index=0,
        rhetorical_role=RhetoricalRole.ANALYSIS,
        text="The inherent power of the High Court under Section 482 CrPC is wide but must be exercised sparingly.",
        text_with_context="Delhi High Court. Justice A. Analysis: The inherent power of the High Court under Section 482 CrPC is wide.",
        char_count=120,
        word_count=22,
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        case_title="State v. Ramesh",
        statute_mentions=["Section 482 CrPC"],
        citations=[],
    )

    # Use mocked model to avoid downloading weights in unit test
    mock_model = MagicMock()
    mock_vec = np.random.randn(1, 384).astype(np.float32)
    mock_vec /= np.linalg.norm(mock_vec)
    mock_model.encode.return_value = mock_vec

    with patch.object(engine.embedder, "_load_model", return_value=mock_model):
        engine.index_chunks([chunk])
        assert len(engine.chunks) == 1
        assert engine.dense_matrix is not None
        assert engine.dense_matrix.shape == (1, 384)

        # Search query
        mock_model.encode.return_value = mock_vec[0]
        resp = engine.search(query="Section 482 inherent powers", mode="hybrid", top_k=1)
        assert resp.total_indexed == 1
        assert len(resp.results) == 1
        assert resp.results[0].doc_id == "doc_test_100"
