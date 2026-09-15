"""Database migration engine for LegalTrace Phase 2.

Populates enterprise databases from existing Pipeline A artifacts:
1. PostgreSQL: Relational Judgments, Chunks, Headnotes, and Statute Mentions
2. Qdrant: 9,167+ semantic chunks with 384-dimensional dense vectors
3. FalkorDB: Precedent citation graph with 2,854 directed edges and treatment labels
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ingestion_source.config import Settings, get_settings
from ingestion_source.db.graph_store import FalkorGraphStore
from ingestion_source.db.models import Chunk, Headnote, Judgment, StatuteMention
from ingestion_source.db.session import close_db, create_tables, get_session_factory, init_db
from ingestion_source.db.vector_store import QdrantVectorStore

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. PostgreSQL Migration
# ─────────────────────────────────────────────────────────────────────────────

async def migrate_postgres(
    db_url: str,
    cleaned_jsonl_path: Path | str = "all_200_cleaned_judgments.jsonl",
    enriched_jsonl_path: Path | str = "all_200_enriched_judgments.jsonl",
    headnotes_jsonl_path: Path | str = "all_200_headnotes.jsonl",
    chunks_jsonl_path: Path | str = "all_200_chunks.jsonl",
    raw_store_dir: Path | str = "raw-store",
    batch_size: int = 500,
) -> dict[str, int]:
    """Populate PostgreSQL with Judgments, Chunks, Headnotes, and Statutes."""
    logger.info("Initializing PostgreSQL schema and engine...")
    init_db(db_url)
    await create_tables()

    session_factory = get_session_factory()
    stats = {
        "judgments": 0,
        "headnotes": 0,
        "statutes": 0,
        "chunks": 0,
    }

    # Step A: Load Enriched Metadata lookup (doc_id -> dict)
    enriched_map: dict[str, dict[str, Any]] = {}
    en_path = Path(enriched_jsonl_path)
    if en_path.exists():
        with open(en_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    enriched_map[str(item.get("doc_id", ""))] = item
    else:
        logger.warning("Enriched judgments file %s not found. Scanning raw-store...", en_path)
        for ef in Path(raw_store_dir).glob("*/*/*/enriched.json"):
            try:
                item = json.loads(ef.read_text(encoding="utf-8"))
                enriched_map[str(item.get("doc_id", ""))] = item
            except Exception:
                pass

    # Step B: Load Headnotes lookup (doc_id -> dict)
    headnotes_map: dict[str, dict[str, Any]] = {}
    hn_path = Path(headnotes_jsonl_path)
    if hn_path.exists():
        with open(hn_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    headnotes_map[str(item.get("doc_id", ""))] = item
    else:
        logger.warning("Headnotes file %s not found. Scanning raw-store...", hn_path)
        for hf in Path(raw_store_dir).glob("*/*/*/headnote.json"):
            try:
                item = json.loads(hf.read_text(encoding="utf-8"))
                headnotes_map[str(item.get("doc_id", ""))] = item
            except Exception:
                pass

    # Step C: Ingest Judgments, Headnotes, and Statutes
    cl_path = Path(cleaned_jsonl_path)
    judgments_to_insert: list[dict[str, Any]] = []
    if cl_path.exists():
        with open(cl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    judgments_to_insert.append(json.loads(line))
    else:
        for cf in Path(raw_store_dir).glob("*/*/*/cleaned.json"):
            try:
                judgments_to_insert.append(json.loads(cf.read_text(encoding="utf-8")))
            except Exception:
                pass

    logger.info("Found %d judgments to insert into PostgreSQL", len(judgments_to_insert))

    async with session_factory() as session:
        for j_data in judgments_to_insert:
            doc_id = str(j_data.get("doc_id", ""))
            if not doc_id:
                continue

            en = enriched_map.get(doc_id, {})
            hn = headnotes_map.get(doc_id, {})
            header = j_data.get("header") or {}

            # Check if judgment already exists
            res = await session.execute(select(Judgment).where(Judgment.doc_id == doc_id))
            existing_j = res.scalar_one_or_none()

            dockets = header.get("docket_numbers") or []
            bench_list = header.get("bench") or []
            canonical_judges = en.get("canonical_judges") or []
            statutes_list = en.get("statutes") or []

            paras = j_data.get("paragraphs") or []
            total_words = sum(p.get("word_count", 0) for p in paras)

            if existing_j is None:
                judgment_obj = Judgment(
                    doc_id=doc_id,
                    title=str(j_data.get("title") or "Unknown Title"),
                    court=str(j_data.get("court") or "Delhi High Court"),
                    court_slug=str(j_data.get("court_slug") or "delhi"),
                    year=int(j_data.get("year") or 2023),
                    decision_date=header.get("decision_date"),
                    disposition=j_data.get("disposition"),
                    legal_domain=en.get("legal_domain") or hn.get("legal_domain"),
                    case_type=en.get("case_type") or hn.get("case_type"),
                    bench_type=en.get("bench_type"),
                    docket_numbers=dockets,
                    bench=bench_list,
                    canonical_judges=canonical_judges,
                    petitioner=header.get("petitioner"),
                    respondent=header.get("respondent"),
                    content_hash=j_data.get("content_hash") or doc_id,
                    source_url=f"https://indiankanoon.org/doc/{doc_id}/",
                    total_paragraphs=len(paras),
                    total_statutes=len(statutes_list),
                    total_citations=len(en.get("citations") or []),
                    word_count=total_words,
                    clean_full_text=j_data.get("clean_full_text", "")[:50000],  # cap at 50KB for fast indexing
                )
                session.add(judgment_obj)
                stats["judgments"] += 1

                # Headnote
                if hn:
                    hn_obj = Headnote(
                        doc_id=doc_id,
                        catchwords=hn.get("catchwords", []),
                        catchwords_line=hn.get("catchwords_line"),
                        brief_facts=hn.get("brief_facts"),
                        issues=hn.get("issues", []),
                        ratio_decidendi=hn.get("ratio_decidendi"),
                        key_precedents=hn.get("key_precedents", []),
                        operative_holding=hn.get("operative_holding"),
                        formatted_headnote=hn.get("formatted_headnote"),
                        word_count=int(hn.get("word_count", 0)),
                    )
                    session.add(hn_obj)
                    stats["headnotes"] += 1

                # Statute Mentions
                for st in statutes_list:
                    act_name = str(st.get("act_name", "")).strip()
                    if act_name:
                        st_obj = StatuteMention(
                            doc_id=doc_id,
                            act_name=act_name,
                            act_code=st.get("act_code"),
                            sections=st.get("sections", []),
                            raw_mention=st.get("raw_mention"),
                        )
                        session.add(st_obj)
                        stats["statutes"] += 1

        await session.commit()
        logger.info("Committed %d judgments, %d headnotes, %d statutes", stats["judgments"], stats["headnotes"], stats["statutes"])

        # Step D: Batch Ingest Chunks
        ch_path = Path(chunks_jsonl_path)
        if ch_path.exists():
            chunk_batch: list[Chunk] = []
            logger.info("Ingesting chunks from %s into PostgreSQL...", ch_path)
            with open(ch_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    ch_dict = json.loads(line)
                    cid = str(ch_dict.get("chunk_id", ""))
                    did = str(ch_dict.get("doc_id", ""))
                    if not cid or not did:
                        continue

                    chunk_batch.append(
                        Chunk(
                            chunk_id=cid,
                            doc_id=did,
                            chunk_index=int(ch_dict.get("chunk_index", 0)),
                            rhetorical_role=str(ch_dict.get("rhetorical_role", "analysis")),
                            text=str(ch_dict.get("text", "")),
                            text_with_context=ch_dict.get("text_with_context"),
                            char_count=int(ch_dict.get("char_count", 0)),
                            word_count=int(ch_dict.get("word_count", 0)),
                            statute_mentions=ch_dict.get("statute_mentions", []),
                            citations=ch_dict.get("citations", []),
                        )
                    )

                    if len(chunk_batch) >= batch_size:
                        session.add_all(chunk_batch)
                        await session.commit()
                        stats["chunks"] += len(chunk_batch)
                        chunk_batch.clear()

                if chunk_batch:
                    session.add_all(chunk_batch)
                    await session.commit()
                    stats["chunks"] += len(chunk_batch)
                    chunk_batch.clear()

            logger.info("Committed total %d chunks into PostgreSQL", stats["chunks"])

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 2. Qdrant Migration
# ─────────────────────────────────────────────────────────────────────────────

def migrate_qdrant(
    qdrant_url: str = "http://localhost:6333",
    index_dir: Path | str = "search_index",
    chunks_jsonl_path: Path | str = "all_200_chunks.jsonl",
    collection_name: str = "legal_chunks",
    batch_size: int = 100,
) -> dict[str, Any]:
    """Upsert 9,167+ semantic chunk embeddings into Qdrant."""
    idx_dir = Path(index_dir)
    vec_path = idx_dir / "dense_vectors.npy"
    meta_path = idx_dir / "chunks.json"

    if not vec_path.exists():
        raise FileNotFoundError(f"Dense vectors numpy file not found at {vec_path}. Run 'ingest index' first.")

    logger.info("Loading dense vector matrix from %s...", vec_path)
    vectors = np.load(vec_path)
    logger.info("Loaded vector matrix with shape: %s", vectors.shape)

    chunks_data: list[dict[str, Any]] = []
    if meta_path.exists():
        logger.info("Loading chunk metadata from %s...", meta_path)
        chunks_data = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        logger.info("Loading chunk metadata from %s...", chunks_jsonl_path)
        with open(chunks_jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks_data.append(json.loads(line))

    if len(chunks_data) != len(vectors):
        logger.warning("Length mismatch: %d chunks vs %d vectors. Using min length.", len(chunks_data), len(vectors))
        min_len = min(len(chunks_data), len(vectors))
        chunks_data = chunks_data[:min_len]
        vectors = vectors[:min_len]

    store = QdrantVectorStore(
        url=qdrant_url,
        collection_name=collection_name,
        dim=vectors.shape[1],
    )
    store.ensure_collection()

    upserted = store.upsert_chunks(
        chunks=chunks_data,
        dense_vectors=vectors,
        batch_size=batch_size,
    )

    count = store.count()
    info = store.collection_info()

    return {
        "collection_name": collection_name,
        "upserted_points": upserted,
        "total_points_in_qdrant": count,
        "status": info.get("status", "ok"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. FalkorDB Migration
# ─────────────────────────────────────────────────────────────────────────────

def migrate_falkordb(
    host: str = "localhost",
    port: int = 6379,
    edges_path: Path | str = "citation_edges.jsonl",
    nodes_path: Path | str = "precedent_nodes.jsonl",
    graph_name: str = "citation_network",
) -> dict[str, Any]:
    """Populate FalkorDB citation network from citation_edges and precedent_nodes."""
    store = FalkorGraphStore(
        host=host,
        port=port,
        graph_name=graph_name,
    )
    stats = store.ingest_from_jsonl(
        edges_path=edges_path,
        nodes_path=nodes_path,
    )
    graph_stats = store.get_graph_stats()
    top_authorities = store.get_top_authorities(limit=5)

    return {
        "graph_name": graph_name,
        "ingestion_stats": stats,
        "graph_cardinality": graph_stats,
        "top_5_authorities": top_authorities,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Master Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

async def migrate_all(settings: Optional[Settings] = None) -> dict[str, Any]:
    """Execute complete database migration across PostgreSQL, Qdrant, and FalkorDB."""
    cfg = settings or get_settings()
    results: dict[str, Any] = {}

    # 1. PostgreSQL
    try:
        logger.info("--- Starting PostgreSQL Migration ---")
        pg_res = await migrate_postgres(db_url=cfg.DATABASE_URL)
        results["postgres"] = {"status": "success", "details": pg_res}
    except Exception as e:
        logger.error("PostgreSQL migration failed: %s", e)
        results["postgres"] = {"status": "error", "error": str(e)}

    # 2. Qdrant
    try:
        logger.info("--- Starting Qdrant Migration ---")
        qd_res = migrate_qdrant(qdrant_url=cfg.QDRANT_URL, collection_name=cfg.QDRANT_COLLECTION)
        results["qdrant"] = {"status": "success", "details": qd_res}
    except Exception as e:
        logger.error("Qdrant migration failed: %s", e)
        results["qdrant"] = {"status": "error", "error": str(e)}

    # 3. FalkorDB
    try:
        logger.info("--- Starting FalkorDB Migration ---")
        fk_res = migrate_falkordb(host=cfg.FALKORDB_HOST, port=cfg.FALKORDB_PORT, graph_name=cfg.FALKORDB_GRAPH)
        results["falkordb"] = {"status": "success", "details": fk_res}
    except Exception as e:
        logger.error("FalkorDB migration failed: %s", e)
        results["falkordb"] = {"status": "error", "error": str(e)}

    await close_db()
    return results
