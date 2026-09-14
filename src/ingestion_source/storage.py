"""Storage writer and manifest ledger manager for Pipeline A intake."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import threading
from typing import Optional

from ingestion_source.models import (
    CitationMetrics,
    DocRef,
    FetchResult,
    IngestionStatus,
    ManifestEntry,
    RawJudgment,
    SourceType,
    StructuredJudgment,
)

logger = logging.getLogger(__name__)


class StorageWriter:
    """Manages writing normalized files and appending to manifest.jsonl."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.manifest_path = self.base_dir / "manifest.jsonl"
        self._lock = threading.Lock()
        self._init_storage()

    def _init_storage(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_judgment(
        self,
        judgment: RawJudgment,
        fetch_result: Optional[FetchResult] = None,
    ) -> RawJudgment:
        """Write raw judgment artifacts to filesystem and append entry to manifest."""
        court_slug = judgment.court_slug or "general"
        doc_year = judgment.date.year if judgment.date else (fetch_result.ref.year if fetch_result else 0)
        year_str = str(doc_year) if doc_year else "unknown_year"
        doc_dir = self.base_dir / court_slug / year_str / str(judgment.doc_id)
        doc_dir.mkdir(parents=True, exist_ok=True)

        # 1. Write full_text.txt
        full_text_file = doc_dir / "full_text.txt"
        full_text_file.write_text(judgment.full_text or "", encoding="utf-8")
        judgment.full_text_path = str(full_text_file.relative_to(self.base_dir).as_posix())

        # 2. Write raw content if available
        raw_path: Optional[str] = None
        if fetch_result and fetch_result.raw_content:
            if judgment.source == SourceType.API and fetch_result.raw_json:
                raw_json_file = doc_dir / "raw.json"
                raw_json_file.write_text(
                    json.dumps(fetch_result.raw_json, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                judgment.json_raw_path = str(raw_json_file.relative_to(self.base_dir).as_posix())
                raw_path = judgment.json_raw_path
            else:
                raw_html_file = doc_dir / "raw.html"
                raw_html_file.write_text(fetch_result.raw_content, encoding="utf-8")
                judgment.html_raw_path = str(raw_html_file.relative_to(self.base_dir).as_posix())
                raw_path = judgment.html_raw_path

        # 3. Ensure structured judgment exists
        if not judgment.structured:
            date_str = judgment.date.isoformat() if judgment.date else None
            judgment.structured = StructuredJudgment(
                doc_id=judgment.doc_id,
                title=judgment.title,
                court=judgment.court,
                court_slug=court_slug,
                date=date_str,
                year=doc_year or None,
                neutral_citation=judgment.neutral_citation,
                case_numbers=judgment.case_numbers,
                bench=judgment.bench,
                author=judgment.author,
                citations=judgment.citations,
                citation_metrics=CitationMetrics(
                    cites_count=judgment.cites_count,
                    cited_by_count=judgment.cited_by_count,
                ),
                full_text=judgment.full_text or "",
                source=judgment.source,
                source_url=judgment.source_url,
                content_hash=judgment.content_hash,
                fetched_at=judgment.fetched_at.isoformat(),
            )

        # 4. Write judgment.json (the canonical well-structured output JSON)
        judgment_json_file = doc_dir / "judgment.json"
        judgment_json_file.write_text(
            judgment.structured.model_dump_json(indent=2),
            encoding="utf-8",
        )
        judgment.structured_path = str(judgment_json_file.relative_to(self.base_dir).as_posix())

        # 5. Update status to STORED
        judgment.ingestion_status = IngestionStatus.STORED

        # 6. Write metadata.json
        metadata_file = doc_dir / "metadata.json"
        metadata_file.write_text(
            judgment.model_dump_json(indent=2),
            encoding="utf-8",
        )

        # 7. Append to manifest.jsonl
        manifest_entry = ManifestEntry(
            doc_id=judgment.doc_id,
            source=judgment.source,
            court=judgment.court,
            court_slug=court_slug,
            year=doc_year,
            title=judgment.title,
            date=judgment.date.isoformat() if judgment.date else None,
            content_hash=judgment.content_hash,
            full_text_path=str(full_text_file.relative_to(self.base_dir).as_posix()),
            metadata_path=str(metadata_file.relative_to(self.base_dir).as_posix()),
            judgment_json_path=judgment.structured_path,
            raw_content_path=raw_path,
            ingestion_status=IngestionStatus.STORED,
            fetched_at=judgment.fetched_at.isoformat(),
        )

        with self._lock:
            with open(self.manifest_path, "a", encoding="utf-8") as f:
                f.write(manifest_entry.model_dump_json() + "\n")

        logger.info("Stored document %s in %s", judgment.doc_id, doc_dir)
        return judgment

    def restructure_stored_documents(self) -> int:
        """Scan existing stored documents and generate well-structured judgment.json for all."""
        from ingestion_source.normalizer import JudgmentNormalizer

        count = 0
        if not self.base_dir.exists():
            return 0

        for doc_dir in [d for d in self.base_dir.glob("*/*/*") if d.is_dir()]:
            doc_id = doc_dir.name
            raw_html_path = doc_dir / "raw.html"
            raw_json_path = doc_dir / "raw.json"
            metadata_path = doc_dir / "metadata.json"

            court_slug = doc_dir.parent.parent.name
            year_str = doc_dir.parent.name
            year = int(year_str) if year_str.isdigit() else 2023

            if raw_html_path.exists():
                raw_html = raw_html_path.read_text(encoding="utf-8")
                ref = DocRef(
                    doc_id=doc_id,
                    court="Unknown Court",
                    court_slug=court_slug,
                    year=year,
                    source_url=f"https://indiankanoon.org/doc/{doc_id}/",
                )
                fetch_res = FetchResult(
                    ref=ref,
                    success=True,
                    source=SourceType.SCRAPE,
                    raw_content=raw_html,
                    status_code=200,
                )
                normalized = JudgmentNormalizer.normalize(fetch_res)
                # Write judgment.json
                if normalized.structured:
                    judgment_json = doc_dir / "judgment.json"
                    judgment_json.write_text(
                        normalized.structured.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                    normalized.structured_path = str(judgment_json.relative_to(self.base_dir).as_posix())
                    # Also update full_text.txt with clean text
                    (doc_dir / "full_text.txt").write_text(normalized.full_text or "", encoding="utf-8")
                    # Update metadata.json
                    normalized.ingestion_status = IngestionStatus.STORED
                    metadata_path.write_text(normalized.model_dump_json(indent=2), encoding="utf-8")
                    count += 1
            elif raw_json_path.exists():
                raw_json = json.loads(raw_json_path.read_text(encoding="utf-8"))
                ref = DocRef(
                    doc_id=doc_id,
                    court=raw_json.get("docsource", "Unknown Court"),
                    court_slug=court_slug,
                    year=year,
                    source_url=f"https://indiankanoon.org/doc/{doc_id}/",
                )
                fetch_res = FetchResult(
                    ref=ref,
                    success=True,
                    source=SourceType.API,
                    raw_json=raw_json,
                    raw_content=raw_json.get("doc", ""),
                    status_code=200,
                )
                normalized = JudgmentNormalizer.normalize(fetch_res)
                if normalized.structured:
                    judgment_json = doc_dir / "judgment.json"
                    judgment_json.write_text(
                        normalized.structured.model_dump_json(indent=2),
                        encoding="utf-8",
                    )
                    normalized.structured_path = str(judgment_json.relative_to(self.base_dir).as_posix())
                    (doc_dir / "full_text.txt").write_text(normalized.full_text or "", encoding="utf-8")
                    normalized.ingestion_status = IngestionStatus.STORED
                    metadata_path.write_text(normalized.model_dump_json(indent=2), encoding="utf-8")
                    count += 1

        return count

    def export_consolidated_jsonl(self, output_file: Path) -> int:
        """Consolidate all canonical judgment.json files into a single JSONL file."""
        count = 0
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as out:
            for j_path in sorted(self.base_dir.glob("*/*/*/judgment.json")):
                try:
                    data = j_path.read_text(encoding="utf-8")
                    compact = json.dumps(json.loads(data), ensure_ascii=False)
                    out.write(compact + "\n")
                    count += 1
                except Exception as e:
                    logger.warning("Error reading %s for export: %s", j_path, e)
        return count
