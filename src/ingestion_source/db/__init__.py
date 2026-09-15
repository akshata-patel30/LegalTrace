"""LegalTrace Database Infrastructure Package.

Provides async PostgreSQL (SQLAlchemy 2.0 + asyncpg), Qdrant vector search,
and FalkorDB graph database integrations for the LegalTrace legal intelligence platform.
"""

from __future__ import annotations

from ingestion_source.db.session import (
    Base,
    close_db,
    create_tables,
    get_db,
    get_engine,
    get_session_factory,
    init_db,
)
from ingestion_source.db.models import (
    BatchIngestionJob,
    Chunk,
    Headnote,
    Judgment,
    RAGSession,
    SearchSession,
    StatuteMention,
)
from ingestion_source.db.vector_store import QdrantVectorStore
from ingestion_source.db.graph_store import FalkorGraphStore

__all__ = [
    "Base",
    "init_db",
    "close_db",
    "create_tables",
    "get_db",
    "get_engine",
    "get_session_factory",
    "Judgment",
    "Chunk",
    "Headnote",
    "StatuteMention",
    "SearchSession",
    "RAGSession",
    "BatchIngestionJob",
    "QdrantVectorStore",
    "FalkorGraphStore",
]
