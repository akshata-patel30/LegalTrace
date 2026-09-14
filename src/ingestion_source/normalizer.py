"""Normalizer for transforming API JSON or scraped HTML into canonical RawJudgment."""

from __future__ import annotations

from datetime import date, datetime
import hashlib
import logging
import re
from typing import Any, Optional

from bs4 import BeautifulSoup

from ingestion_source.models import (
    CitationMetrics,
    CitedDocument,
    DocRef,
    FetchResult,
    IngestionStatus,
    JudgmentParagraph,
    Parties,
    RawJudgment,
    SourceType,
    StructuredJudgment,
)

logger = logging.getLogger(__name__)

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}


def compute_sha256_hash(text: str) -> str:
    """Compute deterministic SHA-256 hash of normalized text."""
    normalized = re.sub(r"\s+", " ", text).strip().encode("utf-8")
    return hashlib.sha256(normalized).hexdigest()


def parse_indian_date(date_str: Optional[str]) -> Optional[date]:
    """Parse various Indian legal date string formats into standard date object."""
    if not date_str or not date_str.strip():
        return None

    cleaned = date_str.strip().lower()
    # Strip ordinal suffixes: 1st, 2nd, 3rd, 4th, etc.
    cleaned = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", cleaned)
    cleaned = re.sub(r"[,\.]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Pattern: 15 January 2023 or 15 Jan 2023
    m1 = re.search(r"(\d{1,2})\s+([a-z]+)\s+(\d{4})", cleaned)
    if m1:
        day = int(m1.group(1))
        month_str = m1.group(2)
        year = int(m1.group(3))
        month = MONTH_MAP.get(month_str)
        if month and 1 <= day <= 31:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    # Pattern: 2023-01-15 or 2023/01/15
    m2 = re.search(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", cleaned)
    if m2:
        year = int(m2.group(1))
        month = int(m2.group(2))
        day = int(m2.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            pass

    # Pattern: 15-01-2023 or 15/01/2023
    m3 = re.search(r"(\d{1,2})[-/](\d{1,2})[-/](\d{4})", cleaned)
    if m3:
        day = int(m3.group(1))
        month = int(m3.group(2))
        year = int(m3.group(3))
        try:
            return date(year, month, day)
        except ValueError:
            pass

    return None


def parse_parties(title: Optional[str]) -> Optional[Parties]:
    """Extract petitioner/appellant and respondent/defendant from case title."""
    if not title:
        return None
    cleaned = title.strip()
    m = re.match(
        r"^(.*?)\s+(?:vs\.?|v\.|versus)\s+(.*?)(?:\s+on\s+\d{1,2}\s+[A-Za-z]+,?\s+\d{4}|$)",
        cleaned,
        re.IGNORECASE,
    )
    if m:
        pet = m.group(1).strip()
        resp = m.group(2).strip()
        pet = re.sub(r"[,\s]+$", "", pet)
        resp = re.sub(r"[,\s]+$", "", resp)
        return Parties(petitioner=pet or None, respondent=resp or None)
    return None


def extract_neutral_citation(text: str) -> Optional[str]:
    """Extract neutral citation number if present in text."""
    if not text:
        return None
    m = re.search(r"Neutral Citation(?:\s+Number)?\s*:\s*([0-9A-Z/\-_]+)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m2 = re.search(r"\b(20\d{2}/[A-Z]{2,6}/\d+)\b", text)
    if m2:
        return m2.group(1).strip()
    return None


def extract_case_numbers(text: str) -> list[str]:
    """Extract Indian court case numbers (e.g. CS(COMM) 413/2021, W.P.(C) 1234/2022)."""
    if not text:
        return []
    pat = r"\b(?:CS\s*(?:\([A-Z\s]+\))?|I\.?A\.?|W\.?P\.?\s*(?:\([A-Z\s]+\))?|CRL\.?[A-Z\.\(\)\s]+|ARB\.?P\.?|RFA|FAO)\s*[\d/]+(?:\s*-\s*\d+)?\b"
    matches = re.findall(pat, text, re.IGNORECASE)
    cleaned: list[str] = []
    seen = set()
    for m in matches:
        c = re.sub(r"\s+", " ", m).strip()
        if c and c.lower() not in seen and len(c) > 3:
            seen.add(c.lower())
            cleaned.append(c)
    return cleaned[:10]


def extract_bench_from_text(text: str) -> list[str]:
    """Extract presiding judge names from document text or bench header."""
    judges: list[str] = []
    patterns = [
        r"Bench:\s*([^\n\r]+)",
        r"Author:\s*([^\n\r]+)",
        r"Hon'ble\s+(?:Mr\.|Ms\.|Mrs\.)?\s*Justice\s+([^\n\r,\.]+)",
        r"CORAM:\s*([^\n\r]+)",
    ]
    seen = set()
    for pat in patterns:
        for match in re.finditer(pat, text, re.IGNORECASE):
            raw_val = match.group(1).strip()
            parts = re.split(r",|\band\b|;", raw_val)
            for part in parts:
                clean_name = re.sub(r"\s+", " ", part).strip()
                # Normalize name key for deduplication
                name_key = re.sub(r"[^a-zA-Z]", "", clean_name.lower())
                if clean_name and len(clean_name) > 3 and not clean_name.lower().startswith("judgment"):
                    if name_key not in seen:
                        seen.add(name_key)
                        judges.append(clean_name)

    return judges[:10]


def extract_paragraphs_and_links(
    container: Any,
    doc_id: str,
) -> tuple[list[JudgmentParagraph], list[CitedDocument], str]:
    """Extract structured paragraphs, cited legal links, and clean full text."""
    if not container:
        return [], [], ""

    # 1. Collect cited document links before decomposing tags
    cited_docs: list[CitedDocument] = []
    seen_links: set[tuple[str, str]] = set()
    for a in container.find_all("a", href=re.compile(r"/doc/(\d+)/")):
        m = re.search(r"/doc/(\d+)/", a["href"])
        if m:
            target_id = m.group(1)
            target_text = re.sub(r"\s+", " ", a.get_text()).strip()
            if target_text and len(target_text) > 1 and target_id != doc_id:
                key = (target_id, target_text)
                if key not in seen_links:
                    seen_links.add(key)
                    cited_docs.append(
                        CitedDocument(
                            doc_id=target_id,
                            text=target_text,
                            url=f"https://indiankanoon.org/doc/{target_id}/",
                        )
                    )

    # 2. Decompose watermarks and navigation/header tags inside container
    for tag in container(["script", "style", "nav", "form", "button", "iframe"]):
        tag.decompose()
    for tag in container.find_all("span", class_="hidden_text"):
        tag.decompose()
    for tag in container.find_all(["div", "span", "h2", "h3"], class_=[
        "covers", "citetop", "docsource_main", "doc_title",
        "doc_author", "doc_bench", "doc_citations", "expanded_headline"
    ]):
        tag.decompose()

    # 3. Extract paragraphs
    paragraphs: list[JudgmentParagraph] = []
    raw_paras = container.find_all(["p", "pre", "blockquote"])
    idx = 1
    for el in raw_paras:
        if el.find_parent(["p", "blockquote"]):
            continue
        text = re.sub(r"\s+", " ", el.get_text()).strip()
        if not text or len(text) < 2:
            continue
        if text.startswith("[Cites") or text.startswith("Equivalent citations:"):
            continue

        p_id = el.get("id")
        p_struct = el.get("data-structure")
        paragraphs.append(
            JudgmentParagraph(
                index=idx,
                id=p_id if isinstance(p_id, str) else None,
                tag=el.name,
                structure_tag=p_struct if isinstance(p_struct, str) else None,
                text=text,
            )
        )
        idx += 1

    # 4. Construct clean full text
    if paragraphs:
        clean_text = "\n\n".join(p.text for p in paragraphs)
    else:
        clean_text = re.sub(r"\n{3,}", "\n\n", container.get_text(separator="\n")).strip()

    return paragraphs, cited_docs, clean_text


class JudgmentNormalizer:
    """Normalizes raw responses into canonical RawJudgment records."""

    @classmethod
    def normalize(cls, result: FetchResult) -> RawJudgment:
        """Transform a FetchResult (API or scrape) into a canonical RawJudgment."""
        if not result.success or not result.raw_content:
            return RawJudgment(
                doc_id=result.ref.doc_id,
                source=result.source,
                source_url=result.ref.source_url,
                fetched_at=result.fetched_at,
                content_hash="",
                court=result.ref.court,
                court_slug=result.ref.court_slug,
                title=result.ref.title or f"Document {result.ref.doc_id}",
                ingestion_status=IngestionStatus.FAILED,
                error=result.error_message or "Fetch failed with empty content",
            )

        if result.source == SourceType.API and result.raw_json:
            return cls._normalize_api(result)
        else:
            return cls._normalize_html(result)

    @classmethod
    def _normalize_api(cls, result: FetchResult) -> RawJudgment:
        """Normalize JSON payload from Indian Kanoon official API."""
        data = result.raw_json or {}
        raw_doc_text = data.get("doc", "") or result.raw_content or ""
        title = data.get("title") or result.ref.title or f"Document {result.ref.doc_id}"
        court = data.get("docsource") or result.ref.court

        # Date parsing
        raw_date = data.get("publishdate")
        judgment_date = parse_indian_date(raw_date)

        # Bench extraction
        bench_raw = data.get("bench")
        bench: list[str] = []
        if isinstance(bench_raw, list):
            bench = [str(b).strip() for b in bench_raw if str(b).strip()]
        elif isinstance(bench_raw, str):
            bench = extract_bench_from_text(f"Bench: {bench_raw}")

        num_cites = data.get("numcites", 0) or 0
        num_cited_by = data.get("numcitedby", 0) or 0
        citation_metrics = CitationMetrics(
            cites_count=num_cites,
            cited_by_count=num_cited_by,
            cites_url=f"https://indiankanoon.org/search/?formInput=cites:{result.ref.doc_id}" if num_cites > 0 else None,
            cited_by_url=f"https://indiankanoon.org/search/?formInput=citedby:{result.ref.doc_id}" if num_cited_by > 0 else None,
        )

        parties = parse_parties(title)

        # Parse doc HTML
        if "<" in raw_doc_text and ">" in raw_doc_text:
            soup = BeautifulSoup(raw_doc_text, "html.parser")
            paragraphs, cited_docs, cleaned_text = extract_paragraphs_and_links(
                soup, result.ref.doc_id
            )
        else:
            cleaned_text = raw_doc_text.strip()
            paragraphs = []
            cited_docs = []
            if cleaned_text:
                for idx, line in enumerate([l.strip() for l in cleaned_text.splitlines() if l.strip()], start=1):
                    paragraphs.append(JudgmentParagraph(index=idx, text=line))

        if not bench:
            bench = extract_bench_from_text(cleaned_text[:2000])

        neutral_citation = extract_neutral_citation(cleaned_text)
        case_numbers = extract_case_numbers(cleaned_text)
        content_hash = compute_sha256_hash(cleaned_text)
        date_str = judgment_date.isoformat() if judgment_date else None
        year = judgment_date.year if judgment_date else result.ref.year

        structured = StructuredJudgment(
            doc_id=result.ref.doc_id,
            title=title,
            court=court,
            court_slug=result.ref.court_slug,
            date=date_str,
            year=year,
            neutral_citation=neutral_citation,
            case_numbers=case_numbers,
            parties=parties,
            bench=bench,
            author=None,
            citations=[],
            citation_metrics=citation_metrics,
            cited_documents=cited_docs,
            paragraphs=paragraphs,
            full_text=cleaned_text,
            source=SourceType.API,
            source_url=result.ref.source_url,
            content_hash=content_hash,
            fetched_at=result.fetched_at.isoformat(),
        )

        return RawJudgment(
            doc_id=result.ref.doc_id,
            source=SourceType.API,
            source_url=result.ref.source_url,
            fetched_at=result.fetched_at,
            content_hash=content_hash,
            court=court,
            court_slug=result.ref.court_slug,
            title=title,
            date=judgment_date,
            bench=bench,
            neutral_citation=neutral_citation,
            case_numbers=case_numbers,
            cites_count=num_cites,
            cited_by_count=num_cited_by,
            doctype=result.ref.doctype or "judgment",
            citations_raw=f"Citations count: {num_cites}, Cited by count: {num_cited_by}" if num_cites or num_cited_by else None,
            full_text=cleaned_text,
            structured=structured,
            ingestion_status=IngestionStatus.PENDING,
        )

    @classmethod
    def _normalize_html(cls, result: FetchResult) -> RawJudgment:
        """Normalize scraped HTML markup from Indian Kanoon."""
        html_str = result.raw_content or ""
        soup = BeautifulSoup(html_str, "html.parser")

        # 1. Extract Title
        title = result.ref.title
        title_el = soup.find(class_="doc_title") or soup.find("h2", class_="doc_title") or soup.find("h1")
        if title_el:
            extracted_title = title_el.get_text(strip=True)
            if extracted_title:
                title = extracted_title

        # 2. Extract Court & Date from header / docsource
        court = result.ref.court
        docsource_el = soup.find(class_="docsource_main") or soup.find(class_="docsource")
        if docsource_el:
            court_text = docsource_el.get_text(strip=True)
            if court_text:
                court = court_text

        # Look for date in header or entire top block
        header_text = soup.get_text()[:1500]
        judgment_date = parse_indian_date(header_text)

        # 3. Extract Author
        author = None
        author_el = soup.find(class_="doc_author")
        if author_el:
            a_link = author_el.find("a")
            if a_link:
                author = a_link.get_text(strip=True)
            else:
                author_txt = author_el.get_text(strip=True)
                author = re.sub(r"^Author\s*:\s*", "", author_txt, flags=re.IGNORECASE).strip() or None

        # 4. Extract Bench
        bench: list[str] = []
        bench_el = soup.find(class_="doc_bench")
        if bench_el:
            links = bench_el.find_all("a")
            if links:
                for a in links:
                    name = a.get_text(strip=True)
                    if name and name not in bench:
                        bench.append(name)
            else:
                raw_b = bench_el.get_text(strip=True)
                raw_b = re.sub(r"^Bench\s*:\s*", "", raw_b, flags=re.IGNORECASE)
                for part in re.split(r",|\band\b|;", raw_b):
                    name = part.strip()
                    if name and name not in bench:
                        bench.append(name)

        # 5. Extract Citations
        citations: list[str] = []
        cits_el = soup.find(class_="doc_citations")
        if cits_el:
            c_txt = cits_el.get_text(strip=True)
            c_txt = re.sub(r"^Equivalent citations\s*:\s*", "", c_txt, flags=re.IGNORECASE)
            citations = [c.strip() for c in c_txt.split(";") if c.strip()]

        # 6. Citetop metrics
        cites_count = 0
        cited_by_count = 0
        citetop_el = soup.find(class_="citetop")
        if citetop_el:
            txt = citetop_el.get_text()
            m_cites = re.search(r"Cites\s*(\d+)", txt, re.IGNORECASE)
            m_citedby = re.search(r"Cited by\s*(\d+)", txt, re.IGNORECASE)
            if m_cites:
                cites_count = int(m_cites.group(1))
            if m_citedby:
                cited_by_count = int(m_citedby.group(1))

        citation_metrics = CitationMetrics(
            cites_count=cites_count,
            cited_by_count=cited_by_count,
            cites_url=f"https://indiankanoon.org/search/?formInput=cites:{result.ref.doc_id}" if cites_count > 0 else None,
            cited_by_url=f"https://indiankanoon.org/search/?formInput=citedby:{result.ref.doc_id}" if cited_by_count > 0 else None,
        )

        # 7. Parties from title
        parties = parse_parties(title)

        # 8. Judgment content container
        content_container = (
            soup.find(id="judgments")
            or soup.find(class_="judgments")
            or soup.find("div", class_="doc_content")
            or soup.body
            or soup
        )

        # Extract neutral citation and case numbers from container text
        full_raw_text = content_container.get_text() if content_container else ""
        neutral_citation = extract_neutral_citation(full_raw_text)
        case_numbers = extract_case_numbers(full_raw_text)

        # 9. Extract paragraphs, cited docs, and clean text
        paragraphs, cited_docs, cleaned_text = extract_paragraphs_and_links(
            content_container, result.ref.doc_id
        )

        # Fallback for bench if DOM bench element was absent
        if not bench:
            bench = extract_bench_from_text(cleaned_text[:2500])

        content_hash = compute_sha256_hash(cleaned_text)
        date_str = judgment_date.isoformat() if judgment_date else None
        year = judgment_date.year if judgment_date else result.ref.year

        structured = StructuredJudgment(
            doc_id=result.ref.doc_id,
            title=title or f"Document {result.ref.doc_id}",
            court=court,
            court_slug=result.ref.court_slug,
            date=date_str,
            year=year,
            neutral_citation=neutral_citation,
            case_numbers=case_numbers,
            parties=parties,
            bench=bench,
            author=author,
            citations=citations,
            citation_metrics=citation_metrics,
            cited_documents=cited_docs,
            paragraphs=paragraphs,
            full_text=cleaned_text,
            source=SourceType.SCRAPE,
            source_url=result.ref.source_url,
            content_hash=content_hash,
            fetched_at=result.fetched_at.isoformat(),
        )

        return RawJudgment(
            doc_id=result.ref.doc_id,
            source=SourceType.SCRAPE,
            source_url=result.ref.source_url,
            fetched_at=result.fetched_at,
            content_hash=content_hash,
            court=court,
            court_slug=result.ref.court_slug,
            title=title or f"Document {result.ref.doc_id}",
            date=judgment_date,
            bench=bench,
            author=author,
            neutral_citation=neutral_citation,
            case_numbers=case_numbers,
            citations=citations,
            cites_count=cites_count,
            cited_by_count=cited_by_count,
            doctype=result.ref.doctype or "judgment",
            citations_raw=f"Citations count: {cites_count}, Cited by count: {cited_by_count}" if cites_count or cited_by_count else None,
            full_text=cleaned_text,
            structured=structured,
            ingestion_status=IngestionStatus.PENDING,
        )

    @staticmethod
    def _clean_soup_text(container: Any) -> str:
        """Remove unwanted tags and normalize whitespace in BeautifulSoup container."""
        if not container:
            return ""

        for tag in container(["script", "style", "nav", "form", "button", "iframe"]):
            tag.decompose()

        text = container.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines()]
        cleaned_lines = []
        consecutive_blank = 0
        for line in lines:
            if not line:
                consecutive_blank += 1
                if consecutive_blank <= 2:
                    cleaned_lines.append("")
            else:
                consecutive_blank = 0
                cleaned_lines.append(line)

        return "\n".join(cleaned_lines).strip()
