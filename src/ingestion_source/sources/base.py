"""Document source interface contract for LegalTrace acquisition."""

from __future__ import annotations

from typing import Iterator, Optional, Protocol, runtime_checkable

from ingestion_source.models import DocRef, FetchResult, SourceType


@runtime_checkable
class DocumentSource(Protocol):
    """Common protocol implemented by both APIClient and HTMLFallback scraper."""

    def discover(
        self,
        court: str,
        year: int,
        month: Optional[int] = None,
        limit: Optional[int] = None,
        query: Optional[str] = None,
    ) -> Iterator[DocRef]:
        """Discover document references from the source for a given court, time window, or search query."""
        ...

    def fetch(self, ref: DocRef) -> FetchResult:
        """Retrieve full document content and metadata for a given DocRef."""
        ...

    def get_source_type(self) -> SourceType:
        """Return the provenance type (api or scrape)."""
        ...

    def close(self) -> None:
        """Release underlying HTTP client connections or resources."""
        ...
