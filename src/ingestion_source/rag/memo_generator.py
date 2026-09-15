"""Automated Legal Memorandum and Brief Generator for LegalTrace.

Generates formal court-ready briefs, bail applications, and structured legal research memos
using retrieved corpus evidence, headnotes, and Shepardization audit trails.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Optional

from ingestion_source.config import Settings, get_settings
from ingestion_source.rag.planner import LegalQueryPlanner
from ingestion_source.rag.shepardizer import LegalShepardizer
from ingestion_source.rag.synthesizer import CitedPassage, GroundedAnswer, LegalRAGSynthesizer
from ingestion_source.search import HybridLegalSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class LegalDocument:
    """A generated court or chamber document."""
    doc_type: str
    title: str
    content: str
    generated_at: str
    citations: list[dict[str, Any]]
    shepard_warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_type": self.doc_type,
            "title": self.title,
            "content": self.content,
            "generated_at": self.generated_at,
            "citations": self.citations,
            "shepard_warnings": self.shepard_warnings,
        }


class LegalMemoGenerator:
    """Generates publication briefs, legal research memos, and court pleadings."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        synthesizer: Optional[LegalRAGSynthesizer] = None,
        shepardizer: Optional[LegalShepardizer] = None,
        raw_store_dir: Path | str = "raw-store",
    ) -> None:
        self.settings = settings or get_settings()
        self.shepardizer = shepardizer or LegalShepardizer(self.settings)
        self.synthesizer = synthesizer or LegalRAGSynthesizer(self.settings, shepardizer=self.shepardizer)
        self.raw_store_dir = Path(raw_store_dir)

    def generate_case_brief(
        self,
        doc_id: str,
        court: str = "delhi",
        year: int = 2023,
    ) -> LegalDocument:
        """Generate an appellate chamber case brief from an ingested judgment."""
        doc_dir = self.raw_store_dir / court / str(year) / str(doc_id)

        cleaned_path = doc_dir / "cleaned.json"
        headnote_path = doc_dir / "headnote.json"
        enriched_path = doc_dir / "enriched.json"

        if not cleaned_path.exists():
            raise FileNotFoundError(f"Judgment document directory not found: {doc_dir}")

        cleaned = json.loads(cleaned_path.read_text(encoding="utf-8"))
        headnote = json.loads(headnote_path.read_text(encoding="utf-8")) if headnote_path.exists() else {}
        enriched = json.loads(enriched_path.read_text(encoding="utf-8")) if enriched_path.exists() else {}

        header = cleaned.get("header") or {}
        case_title = cleaned.get("title", f"Judgment {doc_id}")
        court_name = cleaned.get("court", "High Court of Delhi")
        disposition = cleaned.get("disposition", "Disposed").upper()

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        bench_str = ", ".join(header.get("bench") or enriched.get("canonical_judges") or ["Single Bench"])
        dockets_str = ", ".join(header.get("docket_numbers") or ["N/A"])
        petitioner = header.get("petitioner") or "Petitioner"
        respondent = header.get("respondent") or "Respondent"

        catchwords = headnote.get("catchwords_line") or " — ".join(headnote.get("catchwords") or ["LEGAL ANALYSIS"])
        facts = headnote.get("brief_facts") or "Factual matrix recorded in judgment body."
        issues = "\n".join(f"  {idx+1}. {iss}" for idx, iss in enumerate(headnote.get("issues") or ["Question of law and factual evaluation."]))
        ratio = headnote.get("ratio_decidendi") or "Legal reasoning and statutory principles articulated by court."
        holding = headnote.get("operative_holding") or f"Petition was {disposition}."

        statutes_lines = []
        for st in enriched.get("statutes", []):
            act = st.get("act_name", "")
            secs = ", ".join(st.get("sections", []))
            statutes_lines.append(f"  - {act}" + (f" ({secs})" if secs else ""))
        statutes_str = "\n".join(statutes_lines) if statutes_lines else "  - No explicit statutory references indexed."

        citations_lines = []
        citations_data = []
        warnings = []
        for cit in enriched.get("citations", []):
            cit_txt = cit.get("citation_text", "")
            if cit_txt:
                sh = self.shepardizer.check(cit_txt)
                status_tag = f"[{sh.status.value.upper()}]"
                citations_lines.append(f"  - {cit_txt} {status_tag}")
                citations_data.append({"citation": cit_txt, "status": sh.status.value})
                if sh.warning_message and not sh.is_good_law:
                    warnings.append(f"{cit_txt}: {sh.warning_message}")

        citations_str = "\n".join(citations_lines) if citations_lines else "  - None indexed."

        content = f"""================================================================================
APPELLATE CHAMBER CASE BRIEF
================================================================================
COURT:       {court_name}
DATE:        {header.get('decision_date') or str(year)}
BENCH:       {bench_str}
CAUSE:       {case_title}
DOCKET(S):   {dockets_str}
PARTIES:     {petitioner} (Petitioner) vs. {respondent} (Respondent)
DISPOSITION: {disposition}
--------------------------------------------------------------------------------

1. SUBJECT TAXONOMY & CATCHWORDS
{catchwords}

2. FACTUAL MATRIX
{facts}

3. QUESTIONS PRESENTED / ISSUES
{issues}

4. STATUTORY FRAMEWORK CONSIDERED
{statutes_str}

5. RATIO DECIDENDI (BINDING LEGAL PRINCIPLES)
{ratio}

6. JUDICIAL PRECEDENTS & AUTHORITIES
{citations_str}

7. OPERATIVE ORDER & HOLDING
{holding}

================================================================================
Generated by LegalTrace Automated Intelligence System on {now_str}
================================================================================
"""

        return LegalDocument(
            doc_type="case_brief",
            title=f"Case Brief: {case_title}",
            content=content,
            generated_at=now_str,
            citations=citations_data,
            shepard_warnings=warnings,
        )

    def generate_legal_memorandum(
        self,
        query: str,
        engine: HybridLegalSearchEngine,
    ) -> LegalDocument:
        """Generate an exhaustive formal legal research memorandum answering a legal query."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # 1. Execute grounded RAG synthesis
        rag_res = self.synthesizer.query(user_query=query, engine=engine, max_evidence_chunks=8)

        # 2. Format formal memorandum
        citations_summary = []
        for c in rag_res.citations:
            citations_summary.append({
                "index": c.citation_index,
                "case_title": c.case_title,
                "court": c.court,
                "year": c.year,
                "role": c.rhetorical_role,
            })

        memo_text = f"""================================================================================
LEGAL RESEARCH MEMORANDUM
================================================================================
TO:          Senior Advocates / Litigation Chamber
FROM:        Legal Intelligence Research Group (LegalTrace)
DATE:        {now_str}
SUBJECT:     Legal Analysis and Authorities on: {query}
--------------------------------------------------------------------------------

I. QUESTION PRESENTED / ISSUE
Whether, under the applicable Indian legal framework and precedents of the High
Courts and Supreme Court of India:
"{query}"

II. EXECUTIVE SUMMARY & BRIEF ANSWER
Confidence Score: {rag_res.confidence * 100:.1f}% | Evidence Passages Examined: {rag_res.total_chunks_examined}

{rag_res.answer_text}

III. EVIDENCE AND TABLE OF AUTHORITIES
"""

        for c in rag_res.citations:
            memo_text += f"\n[{c.citation_index}] {c.case_title}\n"
            memo_text += f"     Court: {c.court} | Year: {c.year} | Chunk Role: {c.rhetorical_role.upper()}\n"
            memo_text += f"     Relevant Passage: \"{c.text[:220].strip()}...\"\n"

        if rag_res.shepard_warnings:
            memo_text += "\nIV. JUDICIAL TREATMENT ALERTS (SHEPARD'S AUDIT)\n"
            for w in rag_res.shepard_warnings:
                memo_text += f"  - [WARNING] {w}\n"
        else:
            memo_text += "\nIV. JUDICIAL TREATMENT ALERTS (SHEPARD'S AUDIT)\n"
            memo_text += "  - All cited precedents appear to remain good law without adverse judicial disapproval in the indexed corpus.\n"

        memo_text += f"""
V. CONCLUSION & PRACTICAL RECOMMENDATIONS
Counsel should frame pleadings consistent with the statutory prerequisites and
judicial ratios documented above. In case of factual distinctions, argue on
the boundaries articulated in the cited authorities.

================================================================================
END OF MEMORANDUM — Generated by LegalTrace Platform
================================================================================
"""

        return LegalDocument(
            doc_type="legal_memorandum",
            title=f"Legal Research Memorandum: {query[:60]}...",
            content=memo_text,
            generated_at=now_str,
            citations=citations_summary,
            shepard_warnings=rag_res.shepard_warnings,
        )

    def generate_bail_application(
        self,
        applicant_name: str,
        fir_no: str,
        police_station: str,
        sections_invoked: list[str],
        custody_date: str,
        facts_summary: str,
        engine: HybridLegalSearchEngine,
    ) -> LegalDocument:
        """Generate a formal court draft for a Regular Bail Application under Section 439 CrPC."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Query relevant bail precedents
        bail_query = f"regular bail under Section 439 CrPC for offences {', '.join(sections_invoked)}"
        rag_res = self.synthesizer.query(user_query=bail_query, engine=engine, max_evidence_chunks=5)

        sections_str = ", ".join(sections_invoked)

        draft = f"""IN THE HIGH COURT OF DELHI AT NEW DELHI
(CRIMINAL MISCELLANEOUS JURISDICTION)
BAIL APPLN. NO.        OF {datetime.now().year}

IN THE MATTER OF:
{applicant_name.upper()}
S/o ________________________
R/o ________________________
(Currently in Judicial Custody since {custody_date})
                                                        ...APPLICANT / PETITIONER
VERSUS

STATE (GOVT. OF NCT OF DELHI)
Through: Standing Counsel (Criminal)
                                                        ...RESPONDENT

APPLICATION UNDER SECTION 439 OF THE CODE OF CRIMINAL PROCEDURE, 1973
FOR GRANT OF REGULAR BAIL TO THE APPLICANT IN FIR NO. {fir_no},
REGISTERED AT POLICE STATION {police_station.upper()}, UNDER SECTIONS {sections_str}.

MOST RESPECTFULLY SHOWETH:

1. That the Applicant is a law-abiding citizen of India, having deep roots in society,
   and has been falsely implicated in FIR No. {fir_no} registered at Police Station
   {police_station} under Sections {sections_str}.

2. FACTUAL MATRIX & ARREST:
   {facts_summary}
   The Applicant was taken into custody on {custody_date} and has been incarcerated
   for a substantial period without trial conclusion.

3. GROUNDS FOR GRANT OF REGULAR BAIL:
   A. That the investigation qua the Applicant is substantially complete and no recovery
      remains to be effected from the Applicant's personal possession.
   B. That the continued incarceration of the Applicant will amount to pre-trial punishment,
      violating the sacred personal liberty guaranteed under Article 21 of the Constitution.
   C. That the Applicant undertakes to strictly abide by all terms and conditions that may
      be imposed by this Hon'ble Court and will not tamper with evidence or influence witnesses.

4. BINDING PRECEDENTS & AUTHORITIES IN SUPPORT:
"""
        citations_data = []
        for c in rag_res.citations[:4]:
            draft += f"   - In '{c.case_title}' ({c.court}, {c.year}), it was held:\n"
            draft += f"     \"{c.text[:220].strip()}...\" [{c.citation_index}]\n\n"
            citations_data.append({"case_title": c.case_title, "court": c.court, "year": c.year})

        draft += f"""5. PRAYER:
   In light of the facts and grounds mentioned hereinabove, it is most respectfully
   prayed that this Hon'ble Court may be graciously pleased to:
   (a) Grant regular bail to the Applicant, {applicant_name.upper()}, in FIR No. {fir_no},
       registered at Police Station {police_station} under Sections {sections_str}, on such
       reasonable terms and conditions as this Hon'ble Court may deem fit;
   (b) Pass any other or further order(s) as this Hon'ble Court may deem fit and proper
       in the interest of justice.

AND FOR THIS ACT OF KINDNESS, THE APPLICANT SHALL EVER PRAY.

FILED BY:
ADVOCATE FOR THE APPLICANT
NEW DELHI
DATED: {now_str}
"""

        return LegalDocument(
            doc_type="bail_application",
            title=f"Bail Application: {applicant_name} (FIR {fir_no})",
            content=draft,
            generated_at=now_str,
            citations=citations_data,
            shepard_warnings=rag_res.shepard_warnings,
        )
