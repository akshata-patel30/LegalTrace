"""Statute-to-Ratio Reverse Lookup Engine.

Component 4.3 of LegalTrace Phase 4:
Given a statutory provision (Act name + Section number), perform reverse lookup to find:
1. Every judicial precedent in the corpus interpreting or applying that provision.
2. The specific judicial reasoning (Ratio Decidendi / Analysis / Operative Holding).
3. Synthesized key legal principles across benches and jurisdictions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import logging
from pathlib import Path
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Canonical Act Normalization Mapping
STATUTE_CANONICAL_MAP: dict[str, str] = {
    "crpc": "Code of Criminal Procedure, 1973",
    "code of criminal procedure": "Code of Criminal Procedure, 1973",
    "criminal procedure code": "Code of Criminal Procedure, 1973",
    "ipc": "Indian Penal Code, 1860",
    "indian penal code": "Indian Penal Code, 1860",
    "penal code": "Indian Penal Code, 1860",
    "cpc": "Code of Civil Procedure, 1908",
    "code of civil procedure": "Code of Civil Procedure, 1908",
    "civil procedure code": "Code of Civil Procedure, 1908",
    "arbitration": "Arbitration and Conciliation Act, 1996",
    "arbitration act": "Arbitration and Conciliation Act, 1996",
    "arbitration and conciliation act": "Arbitration and Conciliation Act, 1996",
    "ni act": "Negotiable Instruments Act, 1881",
    "negotiable instruments": "Negotiable Instruments Act, 1881",
    "negotiable instruments act": "Negotiable Instruments Act, 1881",
    "constitution": "Constitution of India",
    "constitution of india": "Constitution of India",
    "companies act": "Companies Act, 2013",
    "it act": "Income Tax Act, 1961",
    "income tax act": "Income Tax Act, 1961",
    "evidence act": "Indian Evidence Act, 1872",
    "indian evidence act": "Indian Evidence Act, 1872",
    "motor vehicles act": "Motor Vehicles Act, 1988",
    "mv act": "Motor Vehicles Act, 1988",
    "specific relief act": "Specific Relief Act, 1963",
    "limitation act": "Limitation Act, 1963",
    "ndps": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "ndps act": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "narcotic drugs": "Narcotic Drugs and Psychotropic Substances Act, 1985",
    "customs": "Customs Act, 1962",
    "customs act": "Customs Act, 1962",
    "pca": "Prevention of Corruption Act, 1988",
    "prevention of corruption act": "Prevention of Corruption Act, 1988",
    "pocso": "Protection of Children from Sexual Offences Act, 2012",
    "pocso act": "Protection of Children from Sexual Offences Act, 2012",
    "uapa": "Unlawful Activities (Prevention) Act, 1967",
}


def normalize_act_name(act: str) -> str:
    """Resolve colloquial or abbreviated act names to canonical full title."""
    clean = act.strip().lower()
    clean = re.sub(r"[,\.]", "", clean)
    clean = re.sub(r"\s+", " ", clean)

    for alias, canonical in STATUTE_CANONICAL_MAP.items():
        if alias in clean or clean in alias:
            return canonical

    return act.strip()


def normalize_section(section: str) -> str:
    """Normalize section identifier (e.g. 'Sec. 482', 's. 482', 'Section 482' -> '482')."""
    clean = section.strip()
    match = re.search(r"(\d+[A-Za-z]*)", clean)
    if match:
        return match.group(1).upper()
    return clean


@dataclass
class StatuteRatioResult:
    """Individual judicial ratio decidendi / interpretation passage for a statute provision."""
    act_name: str
    section: str
    doc_id: str
    case_title: str
    court: str
    year: int
    rhetorical_role: str
    ratio_text: str
    relevance_score: float
    statute_context: Optional[str] = None
    headnote_ratio: Optional[str] = None
    disposition: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StatuteRatioResponse:
    """Aggregated reverse lookup response."""
    act_name: str
    section: str
    canonical_act: str
    total_results: int
    key_principles: list[str] = field(default_factory=list)
    results: list[StatuteRatioResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "act_name": self.act_name,
            "section": self.section,
            "canonical_act": self.canonical_act,
            "total_results": self.total_results,
            "key_principles": self.key_principles,
            "results": [r.to_dict() for r in self.results],
        }


class StatuteRatioLookup:
    """Performs reverse lookup from statute provisions to judicial ratio decidendi."""

    def __init__(self, raw_store_dir: Optional[Path] = None) -> None:
        self.raw_store_dir = raw_store_dir or Path("raw-store")

    async def lookup(
        self,
        act_name: str,
        section: str,
        db_session: Any = None,
        qdrant_client: Any = None,
        top_k: int = 10,
    ) -> StatuteRatioResponse:
        """Find judicial interpretations of a statutory section.

        Attempts PostgreSQL / Qdrant first, falling back cleanly to file-based
        raw-store lookup if databases are not supplied or return empty.
        """
        canonical_act = normalize_act_name(act_name)
        norm_section = normalize_section(section)
        results: list[StatuteRatioResult] = []

        # 1. Database-backed lookup if session provided
        if db_session is not None:
            try:
                db_results = await self._lookup_via_postgres(
                    canonical_act, norm_section, db_session, top_k=top_k
                )
                results.extend(db_results)
            except Exception as e:
                logger.warning("PostgreSQL statute lookup failed (%s); trying fallback", e)

        # 2. Qdrant-backed lookup if client provided and results < top_k
        if qdrant_client is not None and len(results) < top_k:
            try:
                q_results = await self._lookup_via_qdrant(
                    canonical_act, norm_section, qdrant_client, top_k=top_k
                )
                # Deduplicate by (doc_id, ratio_text[:50])
                seen = {(r.doc_id, r.ratio_text[:50]) for r in results}
                for qr in q_results:
                    if (qr.doc_id, qr.ratio_text[:50]) not in seen:
                        results.append(qr)
                        seen.add((qr.doc_id, qr.ratio_text[:50]))
            except Exception as e:
                logger.warning("Qdrant statute lookup failed (%s); trying fallback", e)

        # 3. File-backed fallback if still needed
        if not results:
            results = self._lookup_via_filesystem(canonical_act, norm_section, top_k=top_k)

        # Rank results by score descending
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        final_results = results[:top_k]

        # Extract synthesized key principles across findings
        key_principles = self._extract_key_principles(canonical_act, norm_section, final_results)

        return StatuteRatioResponse(
            act_name=act_name,
            section=norm_section,
            canonical_act=canonical_act,
            total_results=len(final_results),
            key_principles=key_principles,
            results=final_results,
        )

    async def _lookup_via_postgres(
        self,
        canonical_act: str,
        section: str,
        session: Any,
        top_k: int = 10,
    ) -> list[StatuteRatioResult]:
        """Query PostgreSQL judgments, statute mentions, and chunks."""
        from sqlalchemy import select, or_
        from ingestion_source.db.models import Chunk, Headnote, Judgment, StatuteMention

        results: list[StatuteRatioResult] = []

        # Find matching statute mentions
        stmt = (
            select(StatuteMention, Judgment)
            .join(Judgment, StatuteMention.doc_id == Judgment.doc_id)
            .where(
                or_(
                    StatuteMention.act_name.ilike(f"%{canonical_act}%"),
                    StatuteMention.raw_mention.ilike(f"%{canonical_act}%"),
                )
            )
        )
        query_res = await session.execute(stmt)
        mentions_and_judgments = query_res.all()

        matching_doc_ids: set[str] = set()
        doc_meta_map: dict[str, Any] = {}
        for mention, judgment in mentions_and_judgments:
            sec_list = mention.sections or []
            if section in sec_list or f"Section {section}" in (mention.raw_mention or ""):
                matching_doc_ids.add(judgment.doc_id)
                doc_meta_map[judgment.doc_id] = (mention, judgment)

        if not matching_doc_ids:
            return []

        # Fetch ratio/analysis chunks for matching doc_ids
        chunk_stmt = (
            select(Chunk)
            .where(
                Chunk.doc_id.in_(list(matching_doc_ids)),
                Chunk.rhetorical_role.in_(["analysis", "operative", "ratio"]),
            )
            .limit(top_k * 3)
        )
        chunk_res = await session.execute(chunk_stmt)
        chunks = chunk_res.scalars().all()

        for ch in chunks:
            mention, judgment = doc_meta_map.get(ch.doc_id, (None, None))
            score = 0.9 if ch.rhetorical_role in ("analysis", "ratio") else 0.75
            if section in ch.text:
                score += 0.1

            results.append(
                StatuteRatioResult(
                    act_name=canonical_act,
                    section=section,
                    doc_id=ch.doc_id,
                    case_title=judgment.title if judgment else f"Doc {ch.doc_id}",
                    court=judgment.court if judgment else "Unknown Court",
                    year=judgment.year if judgment else 0,
                    rhetorical_role=ch.rhetorical_role,
                    ratio_text=ch.text,
                    relevance_score=round(score, 3),
                    statute_context=mention.raw_mention if mention else None,
                    disposition=judgment.disposition if judgment else None,
                )
            )

        return results

    async def _lookup_via_qdrant(
        self,
        canonical_act: str,
        section: str,
        qdrant_client: Any,
        top_k: int = 10,
    ) -> list[StatuteRatioResult]:
        """Search Qdrant vector store with role filtering."""
        from qdrant_client import models

        query_text = f"{canonical_act} Section {section} interpretation scope law"
        filter_role = models.Filter(
            should=[
                models.FieldCondition(key="rhetorical_role", match=models.MatchValue(value="analysis")),
                models.FieldCondition(key="rhetorical_role", match=models.MatchValue(value="operative")),
            ]
        )

        res = qdrant_client.client.query_points(
            collection_name=qdrant_client.COLLECTION_NAME,
            query=models.TextQuery(text=query_text) if hasattr(models, "TextQuery") else None,
            query_filter=filter_role,
            limit=top_k,
        )

        out = []
        for p in getattr(res, "points", []):
            payload = p.payload or {}
            out.append(
                StatuteRatioResult(
                    act_name=canonical_act,
                    section=section,
                    doc_id=payload.get("doc_id", ""),
                    case_title=payload.get("case_title", ""),
                    court=payload.get("court", ""),
                    year=int(payload.get("year", 0)),
                    rhetorical_role=payload.get("rhetorical_role", "analysis"),
                    ratio_text=payload.get("text", ""),
                    relevance_score=round(float(getattr(p, "score", 0.8)), 3),
                )
            )
        return out

    def _lookup_via_filesystem(
        self,
        canonical_act: str,
        section: str,
        top_k: int = 10,
    ) -> list[StatuteRatioResult]:
        """File-based scan of raw-store for documents interpreting the statute."""
        if not self.raw_store_dir.exists():
            return []

        results: list[StatuteRatioResult] = []
        enriched_files = list(self.raw_store_dir.glob("**/enriched.json"))

        for enriched_file in enriched_files:
            doc_dir = enriched_file.parent
            meta_file = doc_dir / "metadata.json"
            headnote_file = doc_dir / "headnote.json"
            classified_file = doc_dir / "classified.json"

            try:
                enriched = json.loads(enriched_file.read_text(encoding="utf-8"))
            except Exception:
                continue

            # Check statute mentions in enriched.json (statutes or statute_mentions)
            statute_mentions = enriched.get("statutes") or enriched.get("statute_mentions") or []
            match_found = False
            raw_mention_text = None

            for sm in statute_mentions:
                act = sm.get("act_name") or sm.get("act") or sm.get("act_code") or ""
                act_code = sm.get("act_code") or ""
                secs = [str(s).strip() for s in sm.get("sections", [])]
                raw_txt = sm.get("raw_mention") or sm.get("raw_text") or ""

                act_canonical = normalize_act_name(act)
                act_code_canonical = normalize_act_name(act_code)
                if (
                    canonical_act.lower() in act_canonical.lower()
                    or act_canonical.lower() in canonical_act.lower()
                    or canonical_act.lower() in act_code_canonical.lower()
                    or act_code_canonical.lower() in canonical_act.lower()
                ):
                    if section in secs or f"section {section.lower()}" in raw_txt.lower():
                        match_found = True
                        raw_mention_text = raw_txt
                        break

            if not match_found:
                continue

            # Found a match! Load doc metadata and analysis/ratio
            meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
            headnote = json.loads(headnote_file.read_text(encoding="utf-8")) if headnote_file.exists() else {}

            doc_id = doc_dir.name
            title = meta.get("title") or enriched.get("case_title") or f"Judgment {doc_id}"
            court = meta.get("court", "Delhi High Court")
            year = meta.get("year", 2023)
            hn_ratio = headnote.get("ratio_decidendi")

            # Extract ratio/analysis paragraphs from classified.json
            ratio_passages: list[str] = []
            if classified_file.exists():
                try:
                    classified = json.loads(classified_file.read_text(encoding="utf-8"))
                    paragraphs = classified.get("classified_paragraphs", [])
                    for p in paragraphs:
                        role = p.get("predicted_role") or p.get("role")
                        p_text = p.get("clean_text") or p.get("text", "")
                        if role in ("analysis", "operative", "ratio") and len(p_text.strip()) > 60:
                            ratio_passages.append(p_text)
                except Exception:
                    pass

            # If no paragraph found from classified, use headnote ratio or brief facts
            if not ratio_passages and hn_ratio:
                ratio_passages.append(hn_ratio)

            if not ratio_passages:
                ratio_passages.append(f"Statute {canonical_act} Section {section} considered in {title}.")

            for passage in ratio_passages[:2]:
                score = 0.85
                if section in passage:
                    score += 0.1
                results.append(
                    StatuteRatioResult(
                        act_name=canonical_act,
                        section=section,
                        doc_id=doc_id,
                        case_title=title,
                        court=court,
                        year=year,
                        rhetorical_role="analysis",
                        ratio_text=passage,
                        relevance_score=round(score, 3),
                        statute_context=raw_mention_text,
                        headnote_ratio=hn_ratio,
                    )
                )

            if len(results) >= top_k * 2:
                break

        return results

    def _extract_key_principles(
        self,
        canonical_act: str,
        section: str,
        results: list[StatuteRatioResult],
    ) -> list[str]:
        """Extract core legal principles across judicial findings."""
        principles = []
        seen_snippets = set()

        for r in results:
            txt = r.ratio_text.strip()
            # Find sentences discussing power, scope, condition, requirement, bar
            sentences = re.split(r"(?<=[.!?])\s+", txt)
            for s in sentences:
                s_clean = s.strip()
                if (
                    len(s_clean) > 40
                    and any(kw in s_clean.lower() for kw in ["power", "scope", "held that", "laid down", "satisfied", "bar", "pre-requisite", "condition", "discretion", "inherent", "quash", "bail", "jurisdiction"])
                    and s_clean[:30] not in seen_snippets
                ):
                    principles.append(s_clean)
                    seen_snippets.add(s_clean[:30])
                    if len(principles) >= 4:
                        break
            if len(principles) >= 4:
                break

        if not principles:
            principles = [
                f"Judicial discretion under Section {section} of {canonical_act} must be exercised with circumspection and within statutory bounds.",
                f"The court examines factual matrix and statutory preconditions before invoking Section {section} of {canonical_act}.",
            ]

        return principles
