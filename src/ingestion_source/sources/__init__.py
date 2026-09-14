"""Document acquisition sources for Indian Kanoon."""

from ingestion_source.sources.base import DocumentSource
from ingestion_source.sources.api_client import IndianKanoonAPIClient
from ingestion_source.sources.html_fallback import IndianKanoonHTMLFallback

__all__ = ["DocumentSource", "IndianKanoonAPIClient", "IndianKanoonHTMLFallback"]
