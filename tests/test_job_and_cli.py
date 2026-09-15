"""Unit tests for JobOrchestrator and Typer CLI."""

import json
from pathlib import Path
import pytest
import respx
import httpx
from typer.testing import CliRunner

from ingestion_source.cli import app
from ingestion_source.config import Settings
from ingestion_source.job import JobOrchestrator
from ingestion_source.models import DocRef

runner = CliRunner()


@respx.mock
def test_job_orchestrator_end_to_end(
    test_settings: Settings,
    sample_doc_ref: DocRef,
    mock_api_doc_json: dict,
):
    respx.post(f"https://api.indiankanoon.org/doc/{sample_doc_ref.doc_id}/").mock(
        return_value=httpx.Response(200, json=mock_api_doc_json)
    )

    orchestrator = JobOrchestrator(settings=test_settings)
    job_id = orchestrator.create_job([sample_doc_ref], mode="api")

    report = orchestrator.execute_job(job_id, workers=1)

    assert report.status == "completed"
    assert report.stored_count == 1
    assert report.failed_count == 0
    assert report.skipped_count == 0

    # Running again with same doc should skip it (deduped)
    job_id_2 = orchestrator.create_job([sample_doc_ref], mode="api")
    report_2 = orchestrator.execute_job(job_id_2, workers=1)

    assert report_2.stored_count == 0
    assert report_2.skipped_count == 1


@respx.mock
def test_job_orchestrator_multithreaded_parallel(
    test_settings: Settings,
    mock_api_doc_json: dict,
):
    """Verify multithreaded parallel execution across multiple workers."""
    refs = []
    for i in range(1, 5):
        doc_id = f"200{i}"
        ref = DocRef(
            doc_id=doc_id,
            court="Delhi High Court",
            court_slug="delhi",
            year=2023,
            source_url=f"https://indiankanoon.org/doc/{doc_id}/",
            title=f"Doc {doc_id}",
        )
        refs.append(ref)
        doc_data = dict(mock_api_doc_json)
        doc_data["tid"] = doc_id
        doc_data["doc"] = f"<div><p>Judgment content for {doc_id}</p></div>"
        respx.post(f"https://api.indiankanoon.org/doc/{doc_id}/").mock(
            return_value=httpx.Response(200, json=doc_data)
        )

    orchestrator = JobOrchestrator(settings=test_settings)
    job_id = orchestrator.create_job(refs, mode="api")

    # Run with 3 parallel workers
    report = orchestrator.execute_job(job_id, workers=3)

    assert report.status == "completed"
    assert report.stored_count == 4
    assert report.failed_count == 0
    assert report.skipped_count == 0


@respx.mock
def test_job_orchestrator_multithreaded_in_flight_dedupe(
    test_settings: Settings,
    mock_api_doc_json: dict,
):
    """Verify that duplicate doc_ids in parallel batches are claimed and skipped."""
    doc_id = "3001"
    ref = DocRef(
        doc_id=doc_id,
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        source_url=f"https://indiankanoon.org/doc/{doc_id}/",
        title="Duplicate Ref",
    )
    respx.post(f"https://api.indiankanoon.org/doc/{doc_id}/").mock(
        return_value=httpx.Response(200, json=mock_api_doc_json)
    )

    # Put same doc_id twice in queue
    orchestrator = JobOrchestrator(settings=test_settings)
    job_id = orchestrator.create_job([ref, ref], mode="api")

    report = orchestrator.execute_job(job_id, workers=2)
    assert report.stored_count == 1


@respx.mock
def test_job_orchestrator_resume(
    test_settings: Settings,
    sample_doc_ref: DocRef,
    mock_api_doc_json: dict,
):
    ref2 = DocRef(
        doc_id="1940429",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        source_url="https://indiankanoon.org/doc/1940429/",
        title="Doc 2",
    )
    mock_doc_2 = dict(mock_api_doc_json)
    mock_doc_2["doc"] = "<div><p>Completely different judgment text for doc 2.</p></div>"
    respx.post("https://api.indiankanoon.org/doc/1940428/").mock(
        return_value=httpx.Response(200, json=mock_api_doc_json)
    )
    respx.post("https://api.indiankanoon.org/doc/1940429/").mock(
        return_value=httpx.Response(200, json=mock_doc_2)
    )

    orchestrator = JobOrchestrator(settings=test_settings)
    job_id = orchestrator.create_job([sample_doc_ref, ref2], mode="api")

    # Execute only first document (limit=1)
    report_partial = orchestrator.execute_job(job_id, limit=1, workers=1)
    assert report_partial.stored_count == 1

    # Resume the job to fetch the remaining item
    report_resumed = orchestrator.execute_job(job_id, workers=2)
    assert report_resumed.stored_count == 2
    assert report_resumed.status == "completed"


@respx.mock
def test_job_orchestrator_failure_handling(
    test_settings: Settings,
    sample_doc_ref: DocRef,
):
    respx.post(f"https://api.indiankanoon.org/doc/{sample_doc_ref.doc_id}/").mock(
        return_value=httpx.Response(404, text="Not Found")
    )

    orchestrator = JobOrchestrator(settings=test_settings)
    job_id = orchestrator.create_job([sample_doc_ref], mode="api")

    report = orchestrator.execute_job(job_id, workers=1)

    assert report.failed_count == 1
    assert report.stored_count == 0
    assert "not_found" in report.error_summary


@respx.mock
def test_cli_discover_and_fetch(
    tmp_path: Path,
    mock_api_search_json: dict,
    mock_api_doc_json: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("INTAKE_OUTPUT_PATH", str(tmp_path / "raw-store"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "raw-store" / "job_state" / "test.sqlite"))
    monkeypatch.setenv("INDIANKANOON_API_TOKEN", "test_tok")

    respx.post("https://api.indiankanoon.org/search/").mock(
        return_value=httpx.Response(200, json=mock_api_search_json)
    )
    respx.post("https://api.indiankanoon.org/doc/1940428/").mock(
        return_value=httpx.Response(200, json=mock_api_doc_json)
    )
    respx.post("https://api.indiankanoon.org/doc/1940429/").mock(
        return_value=httpx.Response(200, json=mock_api_doc_json)
    )

    disc_file = tmp_path / "discovered.jsonl"

    # 1. Discover CLI command
    res_disc = runner.invoke(app, [
        "discover",
        "--court", "Delhi High Court",
        "--year", "2023",
        "--limit", "2",
        "--output", str(disc_file),
    ])
    assert res_disc.exit_code == 0
    assert disc_file.exists()

    # 2. Fetch CLI command with workers
    res_fetch = runner.invoke(app, [
        "fetch",
        "--input", str(disc_file),
        "--mode", "api",
        "--limit", "2",
        "--workers", "2",
    ])
    assert res_fetch.exit_code == 0


@respx.mock
def test_cli_run_and_report(
    tmp_path: Path,
    mock_api_search_json: dict,
    mock_api_doc_json: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("INTAKE_OUTPUT_PATH", str(tmp_path / "raw-store"))
    monkeypatch.setenv("DB_PATH", str(tmp_path / "raw-store" / "job_state" / "test.sqlite"))
    monkeypatch.setenv("INDIANKANOON_API_TOKEN", "test_tok")

    respx.post("https://api.indiankanoon.org/search/").mock(
        return_value=httpx.Response(200, json=mock_api_search_json)
    )
    respx.post("https://api.indiankanoon.org/doc/1940428/").mock(
        return_value=httpx.Response(200, json=mock_api_doc_json)
    )

    # CLI run command with workers
    res_run = runner.invoke(app, [
        "run",
        "--court", "Delhi High Court",
        "--year", "2023",
        "--mode", "api",
        "--limit", "1",
        "--workers", "2",
    ])
    assert res_run.exit_code == 0


def test_cli_check_robots(mock_robots_txt: str):
    with respx.mock:
        respx.get("https://indiankanoon.org/robots.txt").mock(
            return_value=httpx.Response(200, text=mock_robots_txt)
        )
        res = runner.invoke(app, ["check-robots", "--url", "https://indiankanoon.org/doc/12345/"])
        assert res.exit_code == 0
        assert "ALLOWED" in res.stdout


def test_cli_restructure(tmp_path: Path, mock_judgment_html: str):
    doc_dir = tmp_path / "delhi" / "2023" / "9999"
    doc_dir.mkdir(parents=True, exist_ok=True)
    (doc_dir / "raw.html").write_text(mock_judgment_html, encoding="utf-8")

    res = runner.invoke(app, ["restructure", "--store-dir", str(tmp_path)])
    assert res.exit_code == 0
    assert "Successfully upgraded 1 documents" in res.stdout
    assert (doc_dir / "judgment.json").exists()

    # CLI export command
    export_out = tmp_path / "all_test.jsonl"
    res_exp = runner.invoke(app, ["export", "--output", str(export_out), "--store-dir", str(tmp_path)])
    assert res_exp.exit_code == 0
    assert "Successfully consolidated 1 judgments" in res_exp.stdout
    assert export_out.exists()

    # CLI clean command
    (doc_dir / "full_text.txt").write_text("Indian Kanoon - http://indiankanoon.org/doc/9999/\n1. Application is allowed.", encoding="utf-8")
    clean_out = tmp_path / "all_cleaned.jsonl"
    res_clean = runner.invoke(app, ["clean", "--store-dir", str(tmp_path), "--output", str(clean_out)])
    assert res_clean.exit_code == 0
    assert "Cleaning complete" in res_clean.stdout
    assert clean_out.exists()
    assert (doc_dir / "cleaned_text.txt").exists()

    # Test single doc clean
    res_single = runner.invoke(app, ["clean", "--store-dir", str(tmp_path), "--doc-id", "9999"])
    assert res_single.exit_code == 0
    assert "Successfully cleaned doc_id=9999" in res_single.stdout


def test_cli_statute_lookup():
    """Verify CLI statute-lookup command executes and displays findings."""
    res = runner.invoke(app, [
        "statute-lookup",
        "--act", "NDPS",
        "--section", "37",
        "--limit", "3",
    ])
    assert res.exit_code == 0
    assert "Statute-to-Ratio Reverse Lookup" in res.stdout
    assert "Narcotic Drugs and Psychotropic Substances Act, 1985" in res.stdout


def test_cli_index_embedder_option(tmp_path: Path):
    """Verify CLI index command accepts --embedder dense."""
    chunks_file = tmp_path / "chunks.jsonl"
    chunk_data = {
        "chunk_id": "test_chunk_1",
        "doc_id": "doc_1",
        "chunk_index": 0,
        "rhetorical_role": "analysis",
        "text": "The high court has wide discretionary powers under Section 482.",
        "text_with_context": "Delhi High Court. Analysis: Section 482 CrPC.",
        "char_count": 60,
        "word_count": 10,
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "case_title": "Test Case",
        "statute_mentions": [],
        "citations": [],
    }
    chunks_file.write_text(json.dumps(chunk_data) + "\n", encoding="utf-8")

    index_dir = tmp_path / "search_index"
    res = runner.invoke(app, [
        "index",
        "--input", str(chunks_file),
        "--index-dir", str(index_dir),
        "--embedder", "dense",
    ])
    assert res.exit_code == 0
    assert "Index ready for high-speed hybrid search" in res.stdout
    assert (index_dir / "dense_vectors.npy").exists()



