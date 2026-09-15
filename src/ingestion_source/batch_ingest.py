"""Batch ingestion orchestrator for multi-court, multi-year corpus expansion.

Coordinates the full pipeline from discovery, fetch, cleaning, entity extraction,
rhetorical role classification, semantic chunking, headnote generation, and direct
streaming into PostgreSQL, Qdrant, and FalkorDB with persistent checkpointing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Optional, Sequence

import numpy as np
from sqlalchemy import delete, select

from ingestion_source.chunker import LegalChunk, LegalSemanticChunker
from ingestion_source.cleaner import CleanedJudgment, LegalDocumentCleaner
from ingestion_source.config import Settings, get_settings
from ingestion_source.db.models import BatchIngestionJob, Chunk, Headnote, Judgment, StatuteMention
from ingestion_source.db.session import get_session_factory, init_db
from ingestion_source.db.vector_store import QdrantVectorStore
from ingestion_source.db.graph_store import FalkorGraphStore
from ingestion_source.discovery import DiscoveryEngine, normalize_court_slug
from ingestion_source.extractor import EnrichedLegalMetadata, LegalMetadataExtractor
from ingestion_source.graph import CitationGraphBuilder
from ingestion_source.job import JobOrchestrator
from ingestion_source.models import DocRef
from ingestion_source.search import DenseSemanticEmbedder
from ingestion_source.structure import LegalStructureClassifier
from ingestion_source.summarizer import LegalHeadnote, LegalHeadnoteGenerator

logger = logging.getLogger(__name__)


@dataclass
class BatchReport:
    """Consolidated report across all processed court-year batches."""

    total_batches: int = 0
    completed_batches: int = 0
    failed_batches: int = 0
    skipped_batches: int = 0
    total_docs_discovered: int = 0
    total_docs_fetched: int = 0
    total_docs_indexed: int = 0
    total_chunks_indexed: int = 0
    total_edges_indexed: int = 0
    batch_details: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert report to serializable dictionary."""
        return {
            "total_batches": self.total_batches,
            "completed_batches": self.completed_batches,
            "failed_batches": self.failed_batches,
            "skipped_batches": self.skipped_batches,
            "total_docs_discovered": self.total_docs_discovered,
            "total_docs_fetched": self.total_docs_fetched,
            "total_docs_indexed": self.total_docs_indexed,
            "total_chunks_indexed": self.total_chunks_indexed,
            "total_edges_indexed": self.total_edges_indexed,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "batch_details": self.batch_details,
        }


class BatchIngestionOrchestrator:
    """Orchestrates multi-court, multi-year ingestion with direct-to-database streaming."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        qdrant_client: Optional[QdrantVectorStore] = None,
        graph_store: Optional[FalkorGraphStore] = None,
        embedder: Optional[DenseSemanticEmbedder] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.store_dir = Path(self.settings.INTAKE_OUTPUT_PATH)

        # Database clients
        init_db(self.settings.DATABASE_URL)
        self.session_factory = get_session_factory()
        self.qdrant = qdrant_client or QdrantVectorStore(
            url=self.settings.QDRANT_URL,
            collection_name=self.settings.QDRANT_COLLECTION,
            dim=384,
        )
        self.graph = graph_store or FalkorGraphStore(
            host=self.settings.FALKORDB_HOST,
            port=self.settings.FALKORDB_PORT,
            graph_name=self.settings.FALKORDB_GRAPH,
        )
        self.embedder = embedder or DenseSemanticEmbedder(dim=384)

        # Pipeline components
        self.cleaner = LegalDocumentCleaner()
        self.extractor = LegalMetadataExtractor()
        self.classifier = LegalStructureClassifier()
        self.chunker = LegalSemanticChunker()
        self.summarizer = LegalHeadnoteGenerator()
        self.graph_builder = CitationGraphBuilder(store_dir=self.store_dir)
        self.job_orchestrator = JobOrchestrator(settings=self.settings)

    async def run_batch(
        self,
        courts: Optional[Sequence[str]] = None,
        years: Optional[Sequence[int]] = None,
        sample_limit: Optional[int] = None,
        batch_size: int = 50,
        workers: int = 2,
        force: bool = False,
        mode: str = "api",
    ) -> BatchReport:
        """Run batch ingestion across configured courts and years.

        Args:
            courts: Target court names. Defaults to settings.TARGET_COURTS.
            years: Target decision years. Defaults to settings.TARGET_YEARS.
            sample_limit: Document limit per court-year batch. Defaults to settings.SAMPLE_LIMIT.
            batch_size: Batch size for database upserts.
            workers: Concurrent worker threads for fetching.
            force: Re-ingest batches already marked completed.
            mode: Fetch mode ('api', 'hybrid', 'scrape_only').

        Returns:
            BatchReport detailing outcomes across all court-year combinations.
        """
        start_time = time.perf_counter()
        target_courts = list(courts or self.settings.TARGET_COURTS)
        target_years = list(years or self.settings.TARGET_YEARS)
        limit = sample_limit if sample_limit is not None else self.settings.SAMPLE_LIMIT

        self.qdrant.ensure_collection()
        self.graph.ensure_indexes()

        report = BatchReport(total_batches=len(target_courts) * len(target_years))

        logger.info(
            "Starting multi-court batch expansion: %d courts x %d years (limit=%d per batch)",
            len(target_courts),
            len(target_years),
            limit,
        )

        for court in target_courts:
            for year in target_years:
                court_slug = normalize_court_slug(court)
                batch_id = f"batch_{court_slug}_{year}"

                # Check PostgreSQL job state for checkpoint resume
                should_skip = await self._should_skip_batch(batch_id, force=force)
                if should_skip:
                    logger.info("Batch %s already completed and force=False; skipping.", batch_id)
                    report.skipped_batches += 1
                    report.batch_details.append({
                        "batch_id": batch_id,
                        "court": court,
                        "year": year,
                        "status": "skipped",
                        "message": "Already completed",
                    })
                    continue

                # Initialize / resume job record in PostgreSQL
                await self._set_job_status(batch_id, court, court_slug, year, status="running")

                try:
                    res = await self._ingest_court_year(
                        court=court,
                        court_slug=court_slug,
                        year=year,
                        batch_id=batch_id,
                        limit=limit,
                        batch_size=batch_size,
                        workers=workers,
                        mode=mode,
                    )
                    report.completed_batches += 1
                    report.total_docs_discovered += res.get("docs_discovered", 0)
                    report.total_docs_fetched += res.get("docs_fetched", 0)
                    report.total_docs_indexed += res.get("docs_indexed", 0)
                    report.total_chunks_indexed += res.get("chunks_created", 0)
                    report.total_edges_indexed += res.get("edges_created", 0)
                    report.batch_details.append(res)

                    await self._set_job_status(
                        batch_id,
                        court,
                        court_slug,
                        year,
                        status="completed",
                        docs_discovered=res.get("docs_discovered", 0),
                        docs_fetched=res.get("docs_fetched", 0),
                        docs_indexed=res.get("docs_indexed", 0),
                        chunks_created=res.get("chunks_created", 0),
                        edges_created=res.get("edges_created", 0),
                    )

                except Exception as err:
                    logger.error("Batch %s failed: %s", batch_id, err, exc_info=True)
                    report.failed_batches += 1
                    report.batch_details.append({
                        "batch_id": batch_id,
                        "court": court,
                        "year": year,
                        "status": "failed",
                        "error": str(err),
                    })
                    await self._set_job_status(
                        batch_id,
                        court,
                        court_slug,
                        year,
                        status="failed",
                        error_message=str(err),
                    )

        report.elapsed_seconds = time.perf_counter() - start_time
        logger.info(
            "Batch expansion complete in %.1fs: %d completed, %d failed, %d skipped",
            report.elapsed_seconds,
            report.completed_batches,
            report.failed_batches,
            report.skipped_batches,
        )
        return report

    async def _should_skip_batch(self, batch_id: str, force: bool) -> bool:
        """Check if batch is already successfully completed in PostgreSQL."""
        if force:
            return False
        async with self.session_factory() as session:
            stmt = select(BatchIngestionJob).where(BatchIngestionJob.batch_id == batch_id)
            job = (await session.execute(stmt)).scalar_one_or_none()
            return job is not None and job.status == "completed"

    async def _set_job_status(
        self,
        batch_id: str,
        court: str,
        court_slug: str,
        year: int,
        status: str,
        docs_discovered: int = 0,
        docs_fetched: int = 0,
        docs_indexed: int = 0,
        chunks_created: int = 0,
        edges_created: int = 0,
        error_message: Optional[str] = None,
    ) -> None:
        """Upsert batch job progress in PostgreSQL."""
        async with self.session_factory() as session:
            stmt = select(BatchIngestionJob).where(BatchIngestionJob.batch_id == batch_id)
            job = (await session.execute(stmt)).scalar_one_or_none()
            now = datetime.now(timezone.utc)

            if not job:
                job = BatchIngestionJob(
                    batch_id=batch_id,
                    court=court,
                    court_slug=court_slug,
                    year=year,
                    status=status,
                    docs_discovered=docs_discovered,
                    docs_fetched=docs_fetched,
                    docs_indexed=docs_indexed,
                    chunks_created=chunks_created,
                    edges_created=edges_created,
                    error_message=error_message,
                    started_at=now,
                )
                session.add(job)
            else:
                job.status = status
                if docs_discovered:
                    job.docs_discovered = docs_discovered
                if docs_fetched:
                    job.docs_fetched = docs_fetched
                if docs_indexed:
                    job.docs_indexed = docs_indexed
                if chunks_created:
                    job.chunks_created = chunks_created
                if edges_created:
                    job.edges_created = edges_created
                if error_message:
                    job.error_message = error_message
                if status == "completed":
                    job.completed_at = now

            await session.commit()

    async def _ingest_court_year(
        self,
        court: str,
        court_slug: str,
        year: int,
        batch_id: str,
        limit: int,
        batch_size: int,
        workers: int,
        mode: str,
    ) -> dict[str, Any]:
        """Process a single court-year batch from discovery to multi-database indexing."""
        logger.info("Processing batch %s (%s, Year %d, Limit %d)", batch_id, court, year, limit)

        # 1. Discover Documents
        discovery_engine = DiscoveryEngine(settings=self.settings)
        discovered_refs: list[DocRef] = discovery_engine.discover(court=court, year=year, limit=limit)
        docs_discovered = len(discovered_refs)
        logger.info("Batch %s: Discovered %d candidate documents", batch_id, docs_discovered)

        # 2. Fetch and Store Raw Documents via JobOrchestrator
        docs_fetched = 0
        if discovered_refs:
            fetch_job_id = f"fetch_{court_slug}_{year}_{int(time.time())}"
            self.job_orchestrator.create_job(job_id=fetch_job_id, mode=mode, doc_refs=discovered_refs)
            fetch_report = self.job_orchestrator.execute_job(job_id=fetch_job_id, limit=limit, workers=workers)
            docs_fetched = fetch_report.stored_count

        # 3. Locate Document Folders for this court & year in raw-store
        court_year_dir = self.store_dir / court_slug / str(year)
        if not court_year_dir.exists():
            court_year_dir.mkdir(parents=True, exist_ok=True)

        doc_dirs = [d for d in court_year_dir.iterdir() if d.is_dir()]
        logger.info("Batch %s: Found %d document folders to process", batch_id, len(doc_dirs))

        # 4. Process and Direct-Stream Each Document to Databases
        docs_indexed = 0
        total_chunks = 0
        total_edges = 0

        for doc_dir in doc_dirs:
            try:
                stats = await self._process_and_stream_document(doc_dir=doc_dir, court=court, year=year)
                docs_indexed += 1
                total_chunks += stats["chunks"]
                total_edges += stats["edges"]
            except Exception as e:
                logger.warning("Error processing document %s in batch %s: %s", doc_dir.name, batch_id, e)

        return {
            "batch_id": batch_id,
            "court": court,
            "year": year,
            "status": "completed",
            "docs_discovered": docs_discovered,
            "docs_fetched": docs_fetched,
            "docs_indexed": docs_indexed,
            "chunks_created": total_chunks,
            "edges_created": total_edges,
        }

    async def _process_and_stream_document(
        self,
        doc_dir: Path,
        court: str,
        year: int,
    ) -> dict[str, int]:
        """Run Steps 3-8 on a single judgment folder and directly stream to PostgreSQL, Qdrant, and FalkorDB."""
        # Step 3: Clean & Segment
        cleaned = self.cleaner.clean_document_dir(doc_dir)

        # Step 4: Extract Entities (Statutes, Citations, Bench)
        enriched = self.extractor.enrich_document_dir(doc_dir)

        # Step 5: Label Rhetorical Roles
        self.classifier.structure_document_dir(doc_dir)

        # Step 6: Chunk Semantically
        chunks = self.chunker.chunk_document_dir(doc_dir)

        # Step 7: Citation Graph Edges
        edges = self.graph_builder.extract_document_edges(cleaned, enriched=enriched)

        # Step 8: Headnote Syllabus
        headnote = self.summarizer.generate_for_document_dir(doc_dir)

        # ─── 1. PostgreSQL Direct Upsert ───
        async with self.session_factory() as session:
            # Upsert Judgment
            court_name = cleaned.court or court
            court_slug_val = cleaned.court_slug or normalize_court_slug(court_name)
            j_date = cleaned.header.decision_date or ""
            j_date = j_date[:120] if j_date else None
            disp = cleaned.disposition.value if cleaned.disposition else None
            disp = disp[:60] if disp else None
            case_type = cleaned.header.case_type or enriched.case_type or None
            case_type = case_type[:120] if case_type else None

            judgment_obj = Judgment(
                doc_id=cleaned.doc_id,
                title=cleaned.case_title,
                court=court_name,
                court_slug=court_slug_val,
                year=cleaned.year or year,
                decision_date=j_date,
                disposition=disp,
                legal_domain=enriched.legal_domain.value if enriched.legal_domain else "general",
                case_type=case_type,
                bench_type=enriched.bench_type.value if enriched.bench_type else "single_bench",
                docket_numbers=cleaned.header.docket_numbers or [],
                bench=cleaned.header.bench or [],
                canonical_judges=enriched.canonical_judges or [],
                petitioner=cleaned.header.petitioner,
                respondent=cleaned.header.respondent,
                content_hash=cleaned.content_hash,
                source_url=f"https://indiankanoon.org/doc/{cleaned.doc_id}/",
                total_paragraphs=len(cleaned.paragraphs),
                total_statutes=len(enriched.statutes),
                total_citations=len(enriched.citations),
                word_count=cleaned.metrics.cleaned_char_count // 5,
                clean_full_text=cleaned.clean_full_text[:50000] if cleaned.clean_full_text else None,
            )
            await session.merge(judgment_obj)

            # Upsert Headnote
            headnote_obj = Headnote(
                doc_id=cleaned.doc_id,
                catchwords=headnote.catchwords or [],
                catchwords_line=headnote.catchwords_line,
                brief_facts=headnote.brief_facts,
                issues=headnote.issues or [],
                ratio_decidendi=headnote.ratio_decidendi,
                key_precedents=headnote.key_precedents or [],
                operative_holding=headnote.operative_holding,
                formatted_headnote=headnote.formatted_headnote,
                word_count=headnote.word_count,
            )
            await session.merge(headnote_obj)

            # Refresh Statute Mentions
            await session.execute(delete(StatuteMention).where(StatuteMention.doc_id == cleaned.doc_id))
            for st in enriched.statutes:
                session.add(
                    StatuteMention(
                        doc_id=cleaned.doc_id,
                        act_name=st.act_name,
                        act_code=st.act_code,
                        sections=st.sections,
                        raw_mention=st.raw_mention[:200] if st.raw_mention else None,
                    )
                )

            # Refresh Chunks
            await session.execute(delete(Chunk).where(Chunk.doc_id == cleaned.doc_id))
            for ch in chunks:
                session.add(
                    Chunk(
                        chunk_id=ch.chunk_id,
                        doc_id=ch.doc_id,
                        chunk_index=ch.chunk_index,
                        rhetorical_role=ch.rhetorical_role.value,
                        text=ch.text,
                        text_with_context=ch.text_with_context,
                        char_count=ch.char_count,
                        word_count=ch.word_count,
                        statute_mentions=ch.statute_mentions,
                        citations=ch.citations,
                    )
                )

            await session.commit()

        # ─── 2. Qdrant Direct Upsert ───
        if chunks:
            chunk_dicts = [ch.model_dump() for ch in chunks]
            texts_to_embed = [ch.text_with_context or ch.text for ch in chunks]
            dense_vectors = self.embedder.embed_batch(texts_to_embed)
            self.qdrant.upsert_chunks(chunks=chunk_dicts, dense_vectors=dense_vectors)

        # ─── 3. FalkorDB Direct Graph Upsert ───
        edge_stats = self.graph.ingest_edges_and_nodes(
            edges=edges,
            judgment_info={
                "doc_id": cleaned.doc_id,
                "title": cleaned.case_title,
                "court": cleaned.court or court,
                "year": cleaned.year or year,
                "judge": (cleaned.header.bench[0] if cleaned.header.bench else None) or (enriched.canonical_judges[0] if enriched.canonical_judges else ""),
            },
        )

        return {
            "chunks": len(chunks),
            "edges": edge_stats.get("edges_created", len(edges)),
        }

    async def get_all_batch_status(self) -> list[dict[str, Any]]:
        """Retrieve progress and telemetry for all batch jobs from PostgreSQL."""
        async with self.session_factory() as session:
            stmt = select(BatchIngestionJob).order_by(BatchIngestionJob.created_at.desc())
            results = (await session.execute(stmt)).scalars().all()
            return [job.to_dict() for job in results]
