"""Qdrant vector store integration for hybrid dense and sparse legal search.

Manages collection lifecycle, payload indexing, batch embedding upserts,
and hybrid multi-vector retrieval with Reciprocal Rank Fusion (RRF).
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Optional, Sequence

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION = "legal_chunks"
DENSE_DIM = 384
BATCH_SIZE = 100


class QdrantVectorStore:
    """Manages legal chunk vectors and hybrid search in Qdrant."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = DEFAULT_COLLECTION,
        dim: int = DENSE_DIM,
        client: Optional[QdrantClient] = None,
        timeout: float = 30.0,
    ) -> None:
        """Initialize connection to Qdrant server or wrap an existing client.

        Args:
            url: Qdrant HTTP service endpoint.
            collection_name: Target vector collection name.
            dim: Dimension of dense semantic vectors (default 384).
            client: Optional pre-configured QdrantClient instance.
            timeout: Network request timeout in seconds.
        """
        self.url = url
        self.collection_name = collection_name
        self.dim = dim
        self.client = client or QdrantClient(url=url, timeout=timeout, check_compatibility=False)

    @staticmethod
    def chunk_id_to_point_id(chunk_id: str) -> int:
        """Deterministically map string chunk_id to positive int64 point ID."""
        digest = hashlib.md5(chunk_id.encode("utf-8")).hexdigest()
        # 15 hex digits fit safely into positive signed 64-bit int (< 2^63 - 1)
        return int(digest[:15], 16)

    def ensure_collection(self) -> None:
        """Ensure collection exists with dense and sparse vector configurations and payload indexes."""
        if not self.client.collection_exists(self.collection_name):
            logger.info("Creating Qdrant collection '%s' (dense dim=%d, cosine)...", self.collection_name, self.dim)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(size=self.dim, distance=Distance.COSINE),
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(modifier=Modifier.IDF),
                },
            )
            self._create_payload_indexes()
        else:
            logger.debug("Qdrant collection '%s' already exists.", self.collection_name)

    def _create_payload_indexes(self) -> None:
        """Create payload indexes for fast filtered vector retrieval."""
        keyword_fields = ["doc_id", "court", "court_slug", "rhetorical_role", "legal_domain"]
        for field in keyword_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as e:
                logger.warning("Could not create keyword index on %s: %s", field, e)

        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="year",
                field_schema=models.PayloadSchemaType.INTEGER,
            )
        except Exception as e:
            logger.warning("Could not create integer index on year: %s", e)

    def upsert_chunks(
        self,
        chunks: Sequence[dict[str, Any]],
        dense_vectors: np.ndarray,
        sparse_vectors: Optional[Sequence[Optional[dict[str, Any]]]] = None,
        batch_size: int = BATCH_SIZE,
    ) -> int:
        """Batch upsert legal chunks with dense and optional sparse vectors.

        Args:
            chunks: List of chunk metadata dictionaries.
            dense_vectors: 2D numpy array of shape (N, dim).
            sparse_vectors: Optional list of dicts with 'indices' and 'values' keys.
            batch_size: Points per network upsert call.

        Returns:
            Total points successfully upserted.
        """
        self.ensure_collection()
        total = len(chunks)
        if total == 0:
            return 0

        points: list[PointStruct] = []
        for i, chunk in enumerate(chunks):
            cid = str(chunk.get("chunk_id", f"chunk_{i}"))
            pid = self.chunk_id_to_point_id(cid)

            vec_dict: dict[str, Any] = {
                "dense": dense_vectors[i].tolist() if hasattr(dense_vectors[i], "tolist") else list(dense_vectors[i])
            }

            if sparse_vectors and i < len(sparse_vectors) and sparse_vectors[i] is not None:
                sp = sparse_vectors[i]
                vec_dict["sparse"] = SparseVector(
                    indices=sp["indices"],
                    values=sp["values"],
                )

            payload = {
                "chunk_id": cid,
                "doc_id": str(chunk.get("doc_id", "")),
                "court": str(chunk.get("court", "")),
                "court_slug": str(chunk.get("court_slug", "")),
                "year": int(chunk.get("year", 2023)),
                "case_title": str(chunk.get("case_title", "")),
                "rhetorical_role": str(chunk.get("rhetorical_role", "analysis")),
                "text": str(chunk.get("text", "")),
                "text_with_context": str(chunk.get("text_with_context", chunk.get("text", ""))),
                "statute_mentions": list(chunk.get("statute_mentions", [])),
                "citations": list(chunk.get("citations", [])),
            }

            points.append(PointStruct(id=pid, vector=vec_dict, payload=payload))

        upserted_count = 0
        for start_idx in range(0, total, batch_size):
            batch = points[start_idx : start_idx + batch_size]
            self.client.upsert(collection_name=self.collection_name, points=batch)
            upserted_count += len(batch)
            logger.debug("Upserted points %d..%d / %d into Qdrant", start_idx + 1, start_idx + len(batch), total)

        logger.info("Successfully upserted %d points into Qdrant '%s'", upserted_count, self.collection_name)
        return upserted_count

    def search_hybrid(
        self,
        dense_query: np.ndarray | list[float],
        sparse_indices: Optional[list[int]] = None,
        sparse_values: Optional[list[float]] = None,
        filter_role: Optional[str] = None,
        filter_court: Optional[str] = None,
        filter_year: Optional[int] = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Perform hybrid retrieval using dense vectors + sparse lexical with RRF fusion.

        Falls back to dense-only vector search if sparse vectors are omitted.
        """
        must_filters = []
        if filter_role:
            must_filters.append(FieldCondition(key="rhetorical_role", match=MatchValue(value=filter_role.lower())))
        if filter_court:
            must_filters.append(FieldCondition(key="court", match=MatchText(text=filter_court)))
        if filter_year:
            must_filters.append(FieldCondition(key="year", match=MatchValue(value=filter_year)))

        query_filter = Filter(must=must_filters) if must_filters else None
        q_dense = dense_query.tolist() if hasattr(dense_query, "tolist") else list(dense_query)

        # If sparse vector is provided, execute prefetch + RRF fusion
        if sparse_indices is not None and sparse_values is not None and len(sparse_indices) > 0:
            prefetch_queries = [
                models.Prefetch(
                    query=q_dense,
                    using="dense",
                    limit=top_k * 3,
                    filter=query_filter,
                ),
                models.Prefetch(
                    query=models.SparseVector(indices=sparse_indices, values=sparse_values),
                    using="sparse",
                    limit=top_k * 3,
                    filter=query_filter,
                ),
            ]
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch_queries,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
            )
            scored_points = response.points
        else:
            # Dense vector query fallback
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=q_dense,
                using="dense",
                query_filter=query_filter,
                limit=top_k,
            )
            scored_points = response.points

        results = []
        for idx, p in enumerate(scored_points):
            payload = p.payload or {}
            results.append({
                "rank": idx + 1,
                "score": float(p.score),
                "chunk_id": payload.get("chunk_id", str(p.id)),
                "doc_id": payload.get("doc_id", ""),
                "court": payload.get("court", ""),
                "year": payload.get("year", 0),
                "case_title": payload.get("case_title", ""),
                "rhetorical_role": payload.get("rhetorical_role", "analysis"),
                "text": payload.get("text", ""),
                "text_with_context": payload.get("text_with_context", ""),
                "statute_mentions": payload.get("statute_mentions", []),
                "citations": payload.get("citations", []),
            })
        return results

    def count(self) -> int:
        """Return total number of points in the collection."""
        if not self.client.collection_exists(self.collection_name):
            return 0
        res = self.client.count(collection_name=self.collection_name, exact=True)
        return res.count

    def collection_info(self) -> dict[str, Any]:
        """Return collection status and telemetry metadata."""
        if not self.client.collection_exists(self.collection_name):
            return {"exists": False, "status": "not_found", "points_count": 0}
        info = self.client.get_collection(collection_name=self.collection_name)
        return {
            "exists": True,
            "status": str(info.status),
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "segments_count": info.segments_count,
        }

    def delete_collection(self) -> bool:
        """Delete the collection and all its vectors."""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
            logger.info("Deleted collection '%s'", self.collection_name)
            return True
        return False
