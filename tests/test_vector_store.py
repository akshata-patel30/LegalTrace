"""Unit tests for Qdrant vector store integration and hybrid search logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from ingestion_source.db.vector_store import QdrantVectorStore


def test_chunk_id_to_point_id_deterministic():
    """Verify deterministic and positive integer point ID generation."""
    pid1 = QdrantVectorStore.chunk_id_to_point_id("chunk_12345_0")
    pid2 = QdrantVectorStore.chunk_id_to_point_id("chunk_12345_0")
    pid3 = QdrantVectorStore.chunk_id_to_point_id("chunk_12345_1")

    assert isinstance(pid1, int)
    assert pid1 > 0
    assert pid1 == pid2  # Deterministic
    assert pid1 != pid3  # Unique per chunk_id
    assert pid1 < 2**63 - 1  # Fits in int64


def test_ensure_collection_creation():
    """Verify collection is created with correct dense + sparse specs when absent."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = False

    store = QdrantVectorStore(client=mock_client, collection_name="test_chunks")
    store.ensure_collection()

    mock_client.collection_exists.assert_called_once_with("test_chunks")
    mock_client.create_collection.assert_called_once()
    args, kwargs = mock_client.create_collection.call_args
    assert kwargs["collection_name"] == "test_chunks"
    assert "dense" in kwargs["vectors_config"]
    assert "sparse" in kwargs["sparse_vectors_config"]


def test_upsert_chunks_batching():
    """Verify batch upsert constructs PointStruct list and executes client.upsert."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True

    store = QdrantVectorStore(client=mock_client, collection_name="test_chunks", dim=4)

    chunks = [
        {
            "chunk_id": f"chunk_{i}",
            "doc_id": "doc_1",
            "court": "Delhi High Court",
            "year": 2023,
            "case_title": "Test Title",
            "rhetorical_role": "analysis",
            "text": f"Sample legal chunk text {i}",
            "text_with_context": f"[Context] Sample text {i}",
            "statute_mentions": ["Section 37 NDPS"],
            "citations": [],
        }
        for i in range(150)
    ]
    vectors = np.random.randn(150, 4).astype(np.float32)

    upserted = store.upsert_chunks(chunks=chunks, dense_vectors=vectors, batch_size=100)
    assert upserted == 150
    # With 150 items and batch_size 100, should call upsert twice (100 + 50)
    assert mock_client.upsert.call_count == 2


def test_search_hybrid_rrf_query():
    """Verify hybrid search invokes query_points with prefetch and RRF fusion."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_point = MagicMock()
    mock_point.id = 12345
    mock_point.score = 0.95
    mock_point.payload = {
        "chunk_id": "chunk_1",
        "doc_id": "doc_1",
        "court": "Delhi High Court",
        "year": 2023,
        "case_title": "Case 1",
        "rhetorical_role": "analysis",
        "text": "Ratio text",
        "text_with_context": "Context ratio text",
        "statute_mentions": ["Section 37"],
        "citations": [],
    }
    mock_response.points = [mock_point]
    mock_client.query_points.return_value = mock_response

    store = QdrantVectorStore(client=mock_client, collection_name="test_chunks")

    dense_q = [0.1, 0.2, 0.3, 0.4]
    sparse_idx = [10, 25]
    sparse_val = [1.5, 0.8]

    results = store.search_hybrid(
        dense_query=dense_q,
        sparse_indices=sparse_idx,
        sparse_values=sparse_val,
        filter_role="analysis",
        top_k=5,
    )

    mock_client.query_points.assert_called_once()
    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_1"
    assert results[0]["rhetorical_role"] == "analysis"
    assert results[0]["score"] == 0.95


def test_search_dense_only_fallback():
    """Verify fallback to standard query_points when sparse indices are None."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    store = QdrantVectorStore(client=mock_client, collection_name="test_chunks")
    results = store.search_hybrid(
        dense_query=[0.1, 0.2, 0.3],
        sparse_indices=None,
        sparse_values=None,
        top_k=10,
    )
    mock_client.query_points.assert_called_once()
    assert len(results) == 0


def test_count_and_delete_collection():
    """Verify count and delete collection operations."""
    mock_client = MagicMock()
    mock_client.collection_exists.return_value = True
    mock_count_res = MagicMock()
    mock_count_res.count = 9167
    mock_client.count.return_value = mock_count_res

    store = QdrantVectorStore(client=mock_client, collection_name="test_chunks")
    assert store.count() == 9167

    deleted = store.delete_collection()
    assert deleted is True
    mock_client.delete_collection.assert_called_once_with("test_chunks")
