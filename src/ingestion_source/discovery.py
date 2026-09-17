"""Discovery engine for enumerating Indian Kanoon judgments across courts and years."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional, Sequence

from ingestion_source.config import Settings, get_settings
from ingestion_source.models import DocRef
from ingestion_source.sources.api_client import IndianKanoonAPIClient
from ingestion_source.sources.base import DocumentSource
from ingestion_source.sources.html_fallback import IndianKanoonHTMLFallback
from ingestion_source.sources.seed_source import SeedDocumentSource

logger = logging.getLogger(__name__)

COURT_SLUG_MAP = {
    "delhi high court": "delhi",
    "delhi": "delhi",
    "dhc": "delhi",
    "supreme court of india": "supremecourt",
    "supreme court": "supremecourt",
    "sci": "supremecourt",
    "sc": "supremecourt",
    "bombay high court": "bombay",
    "bombay": "bombay",
    "calcutta high court": "kolkata",
    "kolkata": "kolkata",
    "madras high court": "chennai",
    "madras": "chennai",
    "chennai": "chennai",
    "karnataka high court": "karnataka",
    "karnataka": "karnataka",
    "allahabad high court": "allahabad",
    "allahabad": "allahabad",
    "kerala high court": "kerala",
    "kerala": "kerala",
    "punjab and haryana high court": "punjab",
    "punjab": "punjab",
}


def normalize_court_slug(court: str) -> str:
    """Normalize human-readable court name into standard slug."""
    normalized = court.lower().strip()
    return COURT_SLUG_MAP.get(normalized, normalized.replace(" ", "_"))


class DiscoveryEngine:
    """Orchestrates candidate document enumeration across configured courts and years."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        source: Optional[DocumentSource] = None,
        seed_source: Optional[SeedDocumentSource] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._source = source
        self._seed_source = seed_source or SeedDocumentSource(settings=self.settings)

    def _get_default_source(self) -> DocumentSource:
        if self._source is not None:
            return self._source

        if self.settings.MODE == "scrape_only" or (
            self.settings.ENABLE_HTML_FALLBACK and not self.settings.INDIANKANOON_API_TOKEN
        ):
            return IndianKanoonHTMLFallback(settings=self.settings)
        return IndianKanoonAPIClient(settings=self.settings)

    @staticmethod
    def _invoke_source_discover(
        src: DocumentSource,
        court: str,
        year: int,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> Iterator[DocRef]:
        """Safely invoke source.discover with parameter adaptation for backward compatibility."""
        import inspect

        kwargs = {"court": court, "year": year, "limit": limit, "query": query}
        sig = inspect.signature(src.discover)
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        filtered = {k: v for k, v in kwargs.items() if has_varkw or k in sig.parameters}
        return src.discover(**filtered)

    def discover_refs(
        self,
        courts: Optional[Sequence[str]] = None,
        years: Optional[Sequence[int]] = None,
        limit_per_target: Optional[int] = None,
        total_limit: Optional[int] = None,
        source_mode: str = "auto",
        seed_file: Optional[Path] = None,
        query: Optional[str] = None,
    ) -> Iterator[DocRef]:
        """Enumerate DocRefs across courts, years, or search query with ethical access-control fallbacks."""
        target_courts = list(courts) if courts else self.settings.TARGET_COURTS
        target_years = list(years) if years else self.settings.TARGET_YEARS
        source = self._get_default_source()
        seed_source = SeedDocumentSource(settings=self.settings, custom_seed_file=seed_file) if seed_file else self._seed_source

        yielded_total = 0
        for court in target_courts:
            slug = normalize_court_slug(court)
            for year in target_years:
                logger.info("Discovering judgments for court='%s' (%s), year=%d, query='%s'", court, slug, year, query or "")
                target_yielded = 0

                # Determine discovery strategy
                if source_mode == "seed" or seed_file is not None:
                    generator = self._invoke_source_discover(seed_source, court=court, year=year, limit=limit_per_target, query=query)
                elif source_mode == "api":
                    generator = self._invoke_source_discover(source, court=court, year=year, limit=limit_per_target, query=query)
                elif source_mode == "scrape":
                    generator = self._invoke_source_discover(source, court=court, year=year, limit=limit_per_target, query=query)
                else:  # auto
                    # Attempt live discovery first
                    live_refs: list[DocRef] = []
                    try:
                        for r in self._invoke_source_discover(source, court=court, year=year, limit=limit_per_target, query=query):
                            live_refs.append(r)
                    except Exception as e:
                        logger.warning("Live discovery encountered error on court='%s', year=%d: %s", court, year, e)

                    if live_refs:
                        generator = iter(live_refs)
                    else:
                        logger.info(
                            "Live discovery for court='%s' (%s) year=%d yielded 0 documents (upstream access control / Cloudflare challenge). "
                            "Engaging verified seed catalog fallback.",
                            court,
                            slug,
                            year,
                        )
                        generator = self._invoke_source_discover(seed_source, court=court, year=year, limit=limit_per_target, query=query)

                for ref in generator:
                    if not ref.court_slug:
                        ref.court_slug = slug
                    yield ref

                    target_yielded += 1
                    yielded_total += 1
                    if total_limit and yielded_total >= total_limit:
                        return
                    if limit_per_target and target_yielded >= limit_per_target:
                        break

    def save_discovered_to_jsonl(
        self,
        output_path: Path,
        courts: Optional[Sequence[str]] = None,
        years: Optional[Sequence[int]] = None,
        limit: Optional[int] = None,
        source_mode: str = "auto",
        seed_file: Optional[Path] = None,
        query: Optional[str] = None,
    ) -> int:
        """Discover and save doc references to a JSONL file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for ref in self.discover_refs(
                courts=courts,
                years=years,
                total_limit=limit,
                source_mode=source_mode,
                seed_file=seed_file,
                query=query,
            ):
                f.write(ref.model_dump_json() + "\n")
                count += 1
        logger.info("Discovered %d documents saved to %s", count, output_path)
        return count

    @staticmethod
    def load_discovered_from_jsonl(file_path: Path) -> list[DocRef]:
        """Load DocRefs from a previously discovered JSONL file."""
        refs: list[DocRef] = []
        if not file_path.exists():
            raise FileNotFoundError(f"Discovered file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    refs.append(DocRef(**data))
                except Exception as e:
                    logger.warning("Error parsing line %d in %s: %s", line_no, file_path, e)
        return refs
