"""Automatic downstream pipeline processor for newly scraped documents.

Orchestrates:
1. Legal Document Cleaning & Normalization (cleaned.json)
2. Metadata & Entity Extraction (enriched.json)
3. Rhetorical Role Labeling
4. Semantic Legal Chunking (chunks.json)
5. Automated Headnote & Ratio Decidendi Generation (headnote.json)
6. Precedent Citation Graph Edge Extraction
7. Live Hybrid Search Index Update (search_index/ & in-memory engine)
8. Optional streaming to PostgreSQL, Qdrant, and FalkorDB
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Sequence

from ingestion_source.chunker import LegalChunk, LegalSemanticChunker
from ingestion_source.cleaner import LegalDocumentCleaner
from ingestion_source.config import Settings, get_settings
from ingestion_source.extractor import LegalMetadataExtractor
from ingestion_source.graph import CitationGraphBuilder
from ingestion_source.models import DocRef
from ingestion_source.structure import LegalStructureClassifier
from ingestion_source.summarizer import LegalHeadnoteGenerator

logger = logging.getLogger(__name__)


def process_and_index_scraped_documents(
    doc_refs: Sequence[DocRef],
    settings: Optional[Settings] = None,
    search_engine: Optional[Any] = None,
    index_dir: Optional[Path] = None,
) -> dict[str, Any]:
    """Execute end-to-end downstream processing on scraped document folders and update search index."""
    s = settings or get_settings()
    store_dir = Path(s.INTAKE_OUTPUT_PATH)
    idx_dir = index_dir or Path("search_index")

    cleaner = LegalDocumentCleaner()
    extractor = LegalMetadataExtractor()
    classifier = LegalStructureClassifier()
    chunker = LegalSemanticChunker()
    summarizer = LegalHeadnoteGenerator()
    graph_builder = CitationGraphBuilder()

    processed_doc_ids: list[str] = []
    all_chunks: list[LegalChunk] = []
    total_edges = 0
    headnotes_count = 0

    for ref in doc_refs:
        # Locate document directory
        doc_dir: Optional[Path] = None
        if ref.court_slug and ref.year:
            candidate = store_dir / ref.court_slug / str(ref.year) / str(ref.doc_id)
            if candidate.is_dir():
                doc_dir = candidate

        if doc_dir is None:
            matches = list(store_dir.glob(f"**/{ref.doc_id}"))
            if matches and matches[0].is_dir():
                doc_dir = matches[0]

        if not doc_dir or not (doc_dir / "full_text.txt").exists():
            logger.debug("Skipping doc %s: full_text.txt not found in %s", ref.doc_id, doc_dir)
            continue

        try:
            # 1. Clean & Segment
            cleaned = cleaner.clean_document_dir(doc_dir)

            # 2. Extract Entities (Statutes, Citations, Bench)
            enriched = extractor.enrich_document_dir(doc_dir)

            # 3. Classify Rhetorical Roles
            classifier.structure_document_dir(doc_dir)

            # 4. Semantic Chunking
            chunks = chunker.chunk_document_dir(doc_dir)
            all_chunks.extend(chunks)

            # 5. Automated Headnote
            headnote = summarizer.generate_for_document_dir(doc_dir)
            if headnote:
                headnotes_count += 1

            # 6. Citation Graph Edges
            edges = graph_builder.extract_document_edges(cleaned, enriched=enriched)
            total_edges += len(edges)
            if edges:
                import json
                edge_dicts = [e.model_dump(mode="json") for e in edges]
                (doc_dir / "citation_edges.json").write_text(
                    json.dumps(edge_dicts, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                global_edges = Path("citation_edges.jsonl")
                if global_edges.exists():
                    with open(global_edges, "a", encoding="utf-8") as f:
                        for ed in edge_dicts:
                            f.write(json.dumps(ed, ensure_ascii=False) + "\n")

            processed_doc_ids.append(ref.doc_id)
            logger.info(
                "Processed doc %s: %d paragraphs, %d chunks, %d statutes, %d citations",
                ref.doc_id,
                len(cleaned.paragraphs),
                len(chunks),
                len(enriched.statutes),
                len(edges),
            )

        except Exception as e:
            logger.warning("Error running downstream pipeline on doc %s: %s", ref.doc_id, e)

    # 7. Live Hybrid Search Engine Update
    indexed_chunks = 0
    if all_chunks and search_engine is not None:
        try:
            indexed_chunks = search_engine.add_chunks(all_chunks)
            search_engine.save_index(idx_dir)
            logger.info("Successfully added %d chunks to search index at %s", indexed_chunks, idx_dir)
        except Exception as e:
            logger.warning("Failed updating search index with new chunks: %s", e)

    # 8. Optional Async Stream to Phase 2 Databases (PostgreSQL / Qdrant / FalkorDB)
    _try_stream_to_databases(
        processed_doc_ids=processed_doc_ids,
        store_dir=store_dir,
        settings=s,
    )

    return {
        "processed_docs_count": len(processed_doc_ids),
        "processed_doc_ids": processed_doc_ids,
        "indexed_chunks": indexed_chunks,
        "headnotes_count": headnotes_count,
        "edges_count": total_edges,
    }


def _try_stream_to_databases(
    processed_doc_ids: list[str],
    store_dir: Path,
    settings: Settings,
) -> None:
    """Best-effort direct streaming to PostgreSQL, Qdrant, and FalkorDB if reachable."""
    if not processed_doc_ids:
        return

    try:
        from ingestion_source.db.graph_store import FalkorGraphStore
        from ingestion_source.db.vector_store import QdrantVectorStore

        # Qdrant upsert
        try:
            qdrant = QdrantVectorStore(
                url=settings.QDRANT_URL,
                collection_name=settings.QDRANT_COLLECTION,
                dim=384,
            )
            from ingestion_source.search import DenseSemanticEmbedder
            embedder = DenseSemanticEmbedder(dim=384)

            for doc_id in processed_doc_ids:
                matches = list(store_dir.glob(f"**/{doc_id}/chunks.json"))
                if matches and matches[0].exists():
                    import json
                    raw_chunks = json.loads(matches[0].read_text(encoding="utf-8"))
                    texts = [c.get("text_with_context") or c.get("text", "") for c in raw_chunks]
                    vectors = embedder.embed_batch(texts)
                    qdrant.upsert_chunks(raw_chunks, dense_vectors=vectors)
        except Exception as e:
            logger.debug("Qdrant streaming skipped (service offline or unreachable): %s", e)

        # FalkorDB upsert
        try:
            graph = FalkorGraphStore(
                host=settings.FALKORDB_HOST,
                port=settings.FALKORDB_PORT,
                graph_name=settings.FALKORDB_GRAPH,
            )
            for doc_id in processed_doc_ids:
                meta_matches = list(store_dir.glob(f"**/{doc_id}/cleaned.json"))
                if meta_matches and meta_matches[0].exists():
                    import json
                    cleaned_data = json.loads(meta_matches[0].read_text(encoding="utf-8"))
                    doc_edges = []
                    edges_match = list(store_dir.glob(f"**/{doc_id}/citation_edges.json"))
                    if edges_match and edges_match[0].exists():
                        try:
                            doc_edges = json.loads(edges_match[0].read_text(encoding="utf-8"))
                        except Exception:
                            doc_edges = []
                    graph.ingest_edges_and_nodes(
                        edges=doc_edges,
                        judgment_info={
                            "doc_id": doc_id,
                            "title": cleaned_data.get("case_title", f"Doc {doc_id}"),
                            "court": cleaned_data.get("court", "Delhi High Court"),
                            "year": cleaned_data.get("year", 2023),
                        },
                    )
        except Exception as e:
            logger.debug("FalkorDB streaming skipped: %s", e)

    except Exception as e:
        logger.debug("Database streaming skipped: %s", e)
