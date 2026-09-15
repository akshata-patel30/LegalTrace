"""Legal Shepardizer & Good-Law Checker for LegalTrace.

Cross-references the precedent citation graph (via FalkorDB or local citation edge logs)
to assess whether a cited legal precedent remains good law, has been distinguished,
or has faced judicial disapproval / overruling in subsequent decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from pathlib import Path
from typing import Any, Optional

from ingestion_source.config import Settings, get_settings
from ingestion_source.db.graph_store import FalkorGraphStore, normalize_precedent_key

logger = logging.getLogger(__name__)


class LawStatus(str, Enum):
    """Categorical legal validity status of a precedent."""
    GOOD_LAW = "good_law"            # Consistently followed or affirmed
    CAUTION = "caution"              # Mixed treatments or critical commentary
    DISTINGUISHED = "distinguished"  # Distinguished on facts by one or more courts
    OVERRULED = "overruled"          # Overruled or set aside by superior/larger bench
    QUESTIONED = "questioned"        # Doubts expressed regarding its ratio
    UNKNOWN = "unknown"              # Insufficient citations in corpus


@dataclass
class TreatmentRecord:
    """A specific judicial treatment instance by a citing court."""
    citing_doc_id: str
    citing_title: str
    citing_court: str
    citing_year: int
    treatment: str
    rhetorical_role: str
    context: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "citing_doc_id": self.citing_doc_id,
            "citing_title": self.citing_title,
            "citing_court": self.citing_court,
            "citing_year": self.citing_year,
            "treatment": self.treatment,
            "rhetorical_role": self.rhetorical_role,
            "context": self.context,
        }


@dataclass
class ShepardResult:
    """Shepardization report for a judicial precedent."""
    precedent_citation: str
    canonical_id: str
    status: LawStatus
    is_good_law: bool
    total_citations: int
    treatment_summary: dict[str, int]
    warning_message: Optional[str]
    overruling_cases: list[TreatmentRecord] = field(default_factory=list)
    distinguishing_cases: list[TreatmentRecord] = field(default_factory=list)
    relying_cases: list[TreatmentRecord] = field(default_factory=list)
    recent_treatments: list[TreatmentRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "precedent_citation": self.precedent_citation,
            "canonical_id": self.canonical_id,
            "status": self.status.value,
            "is_good_law": self.is_good_law,
            "total_citations": self.total_citations,
            "treatment_summary": self.treatment_summary,
            "warning_message": self.warning_message,
            "overruling_cases": [c.to_dict() for c in self.overruling_cases],
            "distinguishing_cases": [c.to_dict() for c in self.distinguishing_cases],
            "relying_cases": [c.to_dict() for c in self.relying_cases],
            "recent_treatments": [c.to_dict() for c in self.recent_treatments],
        }


class LegalShepardizer:
    """Audits judicial treatment history to verify the continued validity of precedents."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        graph_store: Optional[FalkorGraphStore] = None,
        edges_path: Path | str = "citation_edges.jsonl",
    ) -> None:
        self.settings = settings or get_settings()
        self.graph_store = graph_store
        self.edges_path = Path(edges_path)
        self._local_edges_cache: Optional[list[dict[str, Any]]] = None

    def check(self, citation: str) -> ShepardResult:
        """Evaluate good-law status for a given citation string."""
        clean_cit = citation.strip()
        canonical_id = normalize_precedent_key(clean_cit)

        # 1. Attempt graph query if FalkorGraphStore is initialized
        if self.graph_store is not None:
            try:
                records = self.graph_store.get_precedent_treatment(clean_cit)
                if records:
                    return self._process_treatment_records(clean_cit, canonical_id, records)
            except Exception as e:
                logger.warning("FalkorDB Shepardization query failed, using local edges fallback: %s", e)

        # 2. Fallback to local citation_edges.jsonl file
        return self._check_from_local_edges(clean_cit, canonical_id)

    def _load_local_edges(self) -> list[dict[str, Any]]:
        """Load citation edges from JSONL into memory cache."""
        if self._local_edges_cache is not None:
            return self._local_edges_cache

        edges = []
        if self.edges_path.exists():
            with open(self.edges_path, "r", encoding="utf-8") as f:
                for line in f:
                    line_str = line.strip()
                    if line_str:
                        edges.append(json.loads(line_str))
        self._local_edges_cache = edges
        return edges

    def _check_from_local_edges(self, citation: str, canonical_id: str) -> ShepardResult:
        """Scan local citation edges dataset for matching precedent treatments."""
        edges = self._load_local_edges()
        matches: list[dict[str, Any]] = []

        cit_lower = citation.lower()
        key_tokens = [t for t in normalize_precedent_key(citation).replace("prec_", "").split("_") if len(t) > 1]

        for e in edges:
            cited_cit = str(e.get("cited_citation") or "").lower()
            # Match either substring or shared key tokens
            if cit_lower in cited_cit or (key_tokens and all(k in cited_cit for k in key_tokens[:2])):
                matches.append({
                    "citing_doc_id": str(e.get("citing_doc_id", "")),
                    "citing_title": str(e.get("citing_title", "")),
                    "citing_court": str(e.get("citing_court", "Delhi High Court")),
                    "citing_year": int(e.get("citing_year") or 2023),
                    "treatment": str(e.get("treatment", "cited")).lower(),
                    "rhetorical_role": str(e.get("rhetorical_role", "precedents")),
                    "context": str(e.get("context_sentence", ""))[:500],
                })

        return self._process_treatment_records(citation, canonical_id, matches)

    def _process_treatment_records(
        self,
        citation: str,
        canonical_id: str,
        raw_records: list[dict[str, Any]],
    ) -> ShepardResult:
        """Compute treatment distribution and classify law status."""
        total = len(raw_records)
        if total == 0:
            return ShepardResult(
                precedent_citation=citation,
                canonical_id=canonical_id,
                status=LawStatus.UNKNOWN,
                is_good_law=True,  # Default neutral when not mentioned
                total_citations=0,
                treatment_summary={},
                warning_message="No treatment history found in the current corpus for this citation.",
            )

        treatment_summary: dict[str, int] = {}
        overruling: list[TreatmentRecord] = []
        distinguishing: list[TreatmentRecord] = []
        relying: list[TreatmentRecord] = []
        all_treatments: list[TreatmentRecord] = []

        for r in raw_records:
            t_str = str(r.get("treatment", "cited")).lower()
            treatment_summary[t_str] = treatment_summary.get(t_str, 0) + 1

            record = TreatmentRecord(
                citing_doc_id=r.get("citing_doc_id", ""),
                citing_title=r.get("citing_title", ""),
                citing_court=r.get("citing_court", ""),
                citing_year=int(r.get("citing_year", 0)),
                treatment=t_str,
                rhetorical_role=r.get("rhetorical_role", "precedents"),
                context=r.get("context", ""),
            )
            all_treatments.append(record)

            if "overruled" in t_str or "set_aside" in t_str or "disapproved" in t_str:
                overruling.append(record)
            elif "distinguished" in t_str or "differed" in t_str:
                distinguishing.append(record)
            elif "relied_upon" in t_str or "followed" in t_str or "affirmed" in t_str:
                relying.append(record)

        # Classification logic based on treatment severity
        if len(overruling) > 0:
            status = LawStatus.OVERRULED
            is_good = False
            warning = (
                f"CRITICAL WARNING: This authority was treated as OVERRULED or disapproved in "
                f"{len(overruling)} decision(s) in the corpus (e.g. {overruling[0].citing_title})."
            )
        elif len(distinguishing) > 1:
            status = LawStatus.DISTINGUISHED
            is_good = True
            warning = (
                f"CAUTION: This authority was DISTINGUISHED on facts in {len(distinguishing)} decision(s). "
                f"Review factual boundaries before relying upon it."
            )
        elif len(distinguishing) == 1:
            status = LawStatus.CAUTION
            is_good = True
            warning = (
                f"NOTE: This authority was distinguished once in {distinguishing[0].citing_title} "
                f"({distinguishing[0].citing_year}). Remains generally good law."
            )
        elif len(relying) > 0:
            status = LawStatus.GOOD_LAW
            is_good = True
            warning = None
        else:
            status = LawStatus.GOOD_LAW
            is_good = True
            warning = None

        return ShepardResult(
            precedent_citation=citation,
            canonical_id=canonical_id,
            status=status,
            is_good_law=is_good,
            total_citations=total,
            treatment_summary=treatment_summary,
            warning_message=warning,
            overruling_cases=overruling,
            distinguishing_cases=distinguishing,
            relying_cases=relying,
            recent_treatments=sorted(all_treatments, key=lambda x: x.citing_year, reverse=True)[:10],
        )
