"""Unit and integration tests for Phase 3: Multi-Court Batch Ingestion Orchestrator."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from typer.testing import CliRunner

from ingestion_source.batch_ingest import BatchIngestionOrchestrator, BatchReport
from ingestion_source.cli import app
from ingestion_source.config import Settings
from ingestion_source.db.models import BatchIngestionJob
from ingestion_source.discovery import normalize_court_slug

runner = CliRunner()


def test_batch_report_dataclass():
    """Verify BatchReport metrics accumulation and serialization."""
    report = BatchReport(
        total_batches=10,
        completed_batches=8,
        failed_batches=1,
        skipped_batches=1,
        total_docs_discovered=250,
        total_docs_fetched=240,
        total_docs_indexed=235,
        total_chunks_indexed=3500,
        total_edges_indexed=1200,
        elapsed_seconds=45.678,
    )
    d = report.to_dict()
    assert d["total_batches"] == 10
    assert d["completed_batches"] == 8
    assert d["failed_batches"] == 1
    assert d["skipped_batches"] == 1
    assert d["total_chunks_indexed"] == 3500
    assert d["total_edges_indexed"] == 1200
    assert d["elapsed_seconds"] == 45.68


def test_multi_court_configuration():
    """Verify Phase 3 configuration for nationwide multi-court expansion."""
    settings = Settings()
    # 5 Target Courts
    expected_courts = [
        "Delhi High Court",
        "Supreme Court of India",
        "Bombay High Court",
        "Madras High Court",
        "Karnataka High Court",
    ]
    for court in expected_courts:
        assert court in settings.TARGET_COURTS

    # Target Years 2020-2024
    for y in [2020, 2021, 2022, 2023, 2024]:
        assert y in settings.TARGET_YEARS

    # Target limit
    assert settings.SAMPLE_LIMIT >= 25000

    # Court Slug mapping
    assert normalize_court_slug("Delhi High Court") == "delhi"
    assert normalize_court_slug("Supreme Court of India") == "supremecourt"
    assert normalize_court_slug("Bombay High Court") == "bombay"
    assert normalize_court_slug("Madras High Court") == "chennai"
    assert normalize_court_slug("Karnataka High Court") == "karnataka"


def test_batch_ingestion_job_model():
    """Verify PostgreSQL BatchIngestionJob ORM model attributes and serialization."""
    now = datetime.now(timezone.utc)
    job = BatchIngestionJob(
        id=1,
        batch_id="batch_bombay_2023",
        court="Bombay High Court",
        court_slug="bombay",
        year=2023,
        status="completed",
        docs_discovered=100,
        docs_fetched=98,
        docs_indexed=95,
        chunks_created=1420,
        edges_created=530,
        started_at=now,
        completed_at=now,
    )
    d = job.to_dict()
    assert d["batch_id"] == "batch_bombay_2023"
    assert d["court"] == "Bombay High Court"
    assert d["court_slug"] == "bombay"
    assert d["year"] == 2023
    assert d["status"] == "completed"
    assert d["docs_indexed"] == 95
    assert d["chunks_created"] == 1420
    assert d["edges_created"] == 530
    assert "BatchIngestionJob" in repr(job)


@pytest.mark.asyncio
async def test_batch_orchestrator_should_skip_batch():
    """Verify skipping batches that are already marked completed."""
    orchestrator = BatchIngestionOrchestrator()

    # If force=True, never skip
    assert await orchestrator._should_skip_batch("batch_test_2023", force=True) is False


@pytest.mark.asyncio
async def test_batch_orchestrator_run_batch_mocked():
    """Verify run_batch execution loop and progress accumulation."""
    orchestrator = BatchIngestionOrchestrator()

    mock_ingest = AsyncMock(return_value={
        "batch_id": "batch_mock_2023",
        "court": "Test Court",
        "year": 2023,
        "status": "completed",
        "docs_discovered": 10,
        "docs_fetched": 10,
        "docs_indexed": 10,
        "chunks_created": 150,
        "edges_created": 45,
    })

    with patch.object(orchestrator, "_should_skip_batch", AsyncMock(return_value=False)), \
         patch.object(orchestrator, "_set_job_status", AsyncMock()), \
         patch.object(orchestrator, "_ingest_court_year", mock_ingest), \
         patch.object(orchestrator.qdrant, "ensure_collection"), \
         patch.object(orchestrator.graph, "ensure_indexes"):

        report = await orchestrator.run_batch(
            courts=["Delhi High Court", "Bombay High Court"],
            years=[2022, 2023],
            sample_limit=10,
            force=True,
        )

        assert report.total_batches == 4
        assert report.completed_batches == 4
        assert report.failed_batches == 0
        assert report.total_docs_discovered == 40
        assert report.total_docs_indexed == 40
        assert report.total_chunks_indexed == 600
        assert report.total_edges_indexed == 180
        assert mock_ingest.call_count == 4


@pytest.mark.asyncio
async def test_batch_orchestrator_failure_handling():
    """Verify failed batch error capture and status update."""
    orchestrator = BatchIngestionOrchestrator()

    mock_fail = AsyncMock(side_effect=RuntimeError("Network timeout during fetch"))
    mock_status = AsyncMock()

    with patch.object(orchestrator, "_should_skip_batch", AsyncMock(return_value=False)), \
         patch.object(orchestrator, "_set_job_status", mock_status), \
         patch.object(orchestrator, "_ingest_court_year", mock_fail), \
         patch.object(orchestrator.qdrant, "ensure_collection"), \
         patch.object(orchestrator.graph, "ensure_indexes"):

        report = await orchestrator.run_batch(
            courts=["Madras High Court"],
            years=[2024],
            sample_limit=5,
            force=True,
        )

        assert report.total_batches == 1
        assert report.completed_batches == 0
        assert report.failed_batches == 1
        assert "Network timeout" in report.batch_details[0]["error"]
        assert mock_status.called


@pytest.mark.asyncio
async def test_batch_orchestrator_process_and_stream():
    """Verify document processing Steps 3-8 and database streaming."""
    orchestrator = BatchIngestionOrchestrator()

    mock_cleaned = MagicMock(
        doc_id="777001",
        case_title="State vs. Test Accused",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        header=MagicMock(decision_date="01.01.2023", docket_numbers=["BAIL 1/2023"], petitioner="State", respondent="Accused", bench=["Justice Sharma"], case_type="Bail"),
        disposition=MagicMock(value="allowed"),
        paragraphs=[MagicMock(text="Para 1", para_id=1)],
        clean_full_text="Cleaned text content",
        content_hash="hash_777001",
        metrics=MagicMock(cleaned_char_count=500),
    )
    mock_enriched = MagicMock(
        case_type="Bail",
        legal_domain=MagicMock(value="criminal"),
        bench_type=MagicMock(value="single_bench"),
        canonical_judges=["Justice Sharma"],
        statutes=[MagicMock(act_name="NDPS Act", act_code="NDPS", sections=["37"], raw_mention="Section 37 NDPS")],
        citations=[MagicMock(citation_text="(2016) 3 SCC 379")],
    )
    mock_chunk = MagicMock(
        chunk_id="chk_777001_0",
        doc_id="777001",
        chunk_index=0,
        rhetorical_role=MagicMock(value="analysis"),
        text="Sample chunk text",
        text_with_context="[Context] Sample chunk text",
        char_count=100,
        word_count=20,
        statute_mentions=["NDPS: Section 37"],
        citations=["(2016) 3 SCC 379"],
        model_dump=MagicMock(return_value={"chunk_id": "chk_777001_0", "doc_id": "777001"}),
    )
    mock_headnote = MagicMock(
        catchwords=["CRIMINAL", "BAIL"],
        catchwords_line="CRIMINAL - BAIL",
        brief_facts="Brief facts...",
        issues=["Whether bail should be granted?"],
        ratio_decidendi="Bail granted under conditions.",
        key_precedents=["(2016) 3 SCC 379"],
        operative_holding="Bail is allowed.",
        formatted_headnote="FORMATTED HEADNOTE",
        word_count=50,
    )
    mock_edge = MagicMock(
        citing_doc_id="777001",
        cited_citation="(2016) 3 SCC 379",
        treatment="relied_upon",
        rhetorical_role="analysis",
        model_dump=MagicMock(return_value={"citing_doc_id": "777001", "cited_citation": "(2016) 3 SCC 379"}),
    )

    with patch.object(orchestrator.cleaner, "clean_document_dir", return_value=mock_cleaned), \
         patch.object(orchestrator.extractor, "enrich_document_dir", return_value=mock_enriched), \
         patch.object(orchestrator.classifier, "structure_document_dir"), \
         patch.object(orchestrator.chunker, "chunk_document_dir", return_value=[mock_chunk]), \
         patch.object(orchestrator.graph_builder, "extract_document_edges", return_value=[mock_edge]), \
         patch.object(orchestrator.summarizer, "generate_for_document_dir", return_value=mock_headnote), \
         patch.object(orchestrator.embedder, "embed_batch", return_value=MagicMock()), \
         patch.object(orchestrator.qdrant, "upsert_chunks"), \
         patch.object(orchestrator.graph, "ingest_edges_and_nodes", return_value={"edges_created": 1}):

        # Create mock session context
        mock_session = AsyncMock()
        mock_session.add = MagicMock()
        orchestrator.session_factory = MagicMock(return_value=mock_session)
        mock_session.__aenter__.return_value = mock_session

        stats = await orchestrator._process_and_stream_document(
            doc_dir=Path("mock_doc_dir"),
            court="Delhi High Court",
            year=2023,
        )

        assert stats["chunks"] == 1
        assert stats["edges"] == 1
        assert orchestrator.qdrant.upsert_chunks.called
        assert orchestrator.graph.ingest_edges_and_nodes.called


def test_cli_batch_status():
    """Verify batch-status CLI command execution."""
    res = runner.invoke(app, ["batch-status"])
    assert res.exit_code == 0
