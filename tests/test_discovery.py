"""Unit tests for DiscoveryEngine."""

from pathlib import Path
import pytest

from ingestion_source.config import Settings
from ingestion_source.discovery import DiscoveryEngine, normalize_court_slug
from ingestion_source.models import DocRef


def test_normalize_court_slug():
    assert normalize_court_slug("Delhi High Court") == "delhi"
    assert normalize_court_slug("Supreme Court of India") == "supremecourt"
    assert normalize_court_slug("Bombay High Court") == "bombay"


def test_discovery_save_and_load_jsonl(tmp_path: Path, sample_doc_ref: DocRef):
    output_file = tmp_path / "discovered.jsonl"

    class MockSource:
        def discover(self, court: str, year: int, month=None, limit=None):
            yield sample_doc_ref

    engine = DiscoveryEngine(source=MockSource())
    count = engine.save_discovered_to_jsonl(
        output_path=output_file,
        courts=["Delhi High Court"],
        years=[2023],
    )

    assert count == 1
    assert output_file.exists()

    loaded = DiscoveryEngine.load_discovered_from_jsonl(output_file)
    assert len(loaded) == 1
    assert loaded[0].doc_id == sample_doc_ref.doc_id
    assert loaded[0].court == sample_doc_ref.court


def test_seed_source_delhi_2023():
    from ingestion_source.sources.seed_source import SeedDocumentSource

    seed_src = SeedDocumentSource()
    refs = list(seed_src.discover(court="Delhi High Court", year=2023, limit=5))

    assert len(refs) == 5
    assert all(r.court_slug == "delhi" for r in refs)
    assert all(r.year == 2023 for r in refs)
    assert any(r.doc_id == "1940428" for r in refs)


def test_discovery_fallback_to_seed_on_empty_live(tmp_path: Path):
    """When live scraper returns 0 items (e.g. 403 / Cloudflare challenge), fallback to seed catalog."""
    output_file = tmp_path / "discovered.jsonl"

    class EmptySource:
        def discover(self, court: str, year: int, month=None, limit=None):
            return iter([])

    engine = DiscoveryEngine(source=EmptySource())
    count = engine.save_discovered_to_jsonl(
        output_path=output_file,
        courts=["Delhi High Court"],
        years=[2023],
        limit=5,
    )

    assert count == 5
    assert output_file.exists()

    loaded = DiscoveryEngine.load_discovered_from_jsonl(output_file)
    assert len(loaded) == 5
    assert loaded[0].doc_id == "1940428"
    assert loaded[0].court == "Delhi High Court"
    assert loaded[0].court_slug == "delhi"
    assert loaded[0].year == 2023


def test_discovery_custom_seed_file(tmp_path: Path, sample_doc_ref: DocRef):
    seed_file = tmp_path / "custom_seed.jsonl"
    seed_file.write_text(sample_doc_ref.model_dump_json() + "\n", encoding="utf-8")

    output_file = tmp_path / "output.jsonl"
    engine = DiscoveryEngine()
    count = engine.save_discovered_to_jsonl(
        output_path=output_file,
        courts=["Delhi High Court"],
        years=[2023],
        limit=5,
        source_mode="seed",
        seed_file=seed_file,
    )

    assert count == 1
    loaded = DiscoveryEngine.load_discovered_from_jsonl(output_file)
    assert len(loaded) == 1
    assert loaded[0].doc_id == sample_doc_ref.doc_id

