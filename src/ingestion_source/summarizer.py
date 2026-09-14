"""Legal Headnote and Executive Summary Generator for LegalTrace.

Step 8 of Pipeline A:
- Automated synthesis of publication-grade legal headnotes (syllabi).
- Law-reporter style catchwords taxonomy (Domain - Case Type - Statutes - Issues - Disposition).
- Extractive factual matrix synthesis from FACTS rhetorical paragraphs.
- Legal issues identification and party submission extraction.
- Ratio decidendi formulation prioritizing judicial finding sentences.
- Precedent treatment synthesis (followed, relied upon).
- Operative holding and final relief summary.
- Generates per-document headnote.json, headnote.txt, consolidated JSONL, and Parquet.
"""

from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import re
from typing import Optional

from ingestion_source.models import (
    CitationEdge,
    CitationTreatment,
    CleanedJudgment,
    CleanedParagraph,
    DispositionType,
    EnrichedLegalMetadata,
    HeadnoteReport,
    LegalHeadnote,
    RhetoricalRole,
)

logger = logging.getLogger(__name__)

RATIO_CUE_PATTERNS = [
    r"(?i)\b(?:this\s+court\s+(?:is\s+of\s+the\s+view|holds|finds)|it\s+is\s+well\s+settled|we\s+are\s+of\s+the\s+considered\s+opinion)\b",
    r"(?i)\b(?:the\s+position\s+of\s+law\s+is\s+clear|in\s+the\s+considered\s+view\s+of\s+this\s+court|the\s+law\s+on\s+this\s+point\s+is\s+well\s+settled)\b",
    r"(?i)\b(?:it\s+cannot\s+be\s+disputed\s+that|a\s+perusal\s+of\s+the\s+statutory\s+scheme|the\s+ratio\s+that\s+emerges)\b",
]

ISSUE_CUE_PATTERNS = [
    r"(?i)\b(?:whether\s+(?:the\s+)?(?:petitioner|applicant|plaintiff|appellant|court)|point\s+for\s+(?:our\s+)?determination\s+is|question\s+(?:that\s+)?arises\s+for\s+consideration)\b",
    r"(?i)\b(?:the\s+sole\s+issue\s+before\s+this\s+court|the\s+core\s+controversy\s+is|the\s+question\s+to\s+be\s+decided)\b",
]


class LegalHeadnoteGenerator:
    """Extracts, synthesizes, and formats legal headnotes from structured judgments."""

    def generate_headnote(
        self,
        judgment: CleanedJudgment,
        enriched: Optional[EnrichedLegalMetadata] = None,
        edges: Optional[list[CitationEdge]] = None,
    ) -> LegalHeadnote:
        """Construct a structured LegalHeadnote from judgment and enrichment artifacts."""
        doc_id = judgment.doc_id
        court = judgment.court
        court_slug = judgment.court_slug
        year = judgment.year
        case_title = judgment.title

        # Metadata resolution
        bench = enriched.canonical_judges if (enriched and enriched.canonical_judges) else judgment.header.bench
        docket = judgment.header.docket_numbers[0] if judgment.header.docket_numbers else None
        decision_date = judgment.header.decision_date
        legal_domain = enriched.legal_domain.value if enriched else None
        case_type = enriched.case_type if enriched else None
        disposition = judgment.disposition

        # 1. Catchwords construction
        catchwords, catchwords_line = self._build_catchwords(
            legal_domain=legal_domain,
            case_type=case_type,
            enriched=enriched,
            disposition=disposition,
        )

        # 2. Brief Facts extraction
        brief_facts = self._extract_brief_facts(judgment.paragraphs)

        # 3. Issues extraction
        issues = self._extract_issues(judgment.paragraphs)

        # 4. Party Submissions
        pet_args, resp_args = self._extract_party_arguments(judgment.paragraphs)

        # 5. Ratio Decidendi extraction
        ratio = self._extract_ratio_decidendi(judgment.paragraphs)

        # 6. Key Precedents
        key_precedents = self._extract_key_precedents(edges, judgment.paragraphs)

        # 7. Operative Holding
        holding = self._extract_operative_holding(judgment.paragraphs, disposition)

        # 8. Render Formatted Publication Headnote
        formatted = self._render_formatted_headnote(
            court=court,
            year=year,
            bench=bench,
            decision_date=decision_date,
            case_title=case_title,
            docket=docket,
            catchwords_line=catchwords_line,
            brief_facts=brief_facts,
            issues=issues,
            pet_args=pet_args,
            resp_args=resp_args,
            ratio=ratio,
            key_precedents=key_precedents,
            holding=holding,
            disposition=disposition,
        )

        word_count = len(formatted.split())

        return LegalHeadnote(
            doc_id=doc_id,
            court=court,
            court_slug=court_slug,
            year=year,
            case_title=case_title,
            bench=bench,
            docket_number=docket,
            decision_date=decision_date,
            legal_domain=legal_domain,
            case_type=case_type,
            disposition=disposition,
            catchwords=catchwords,
            catchwords_line=catchwords_line,
            brief_facts=brief_facts,
            issues=issues,
            petitioner_arguments=pet_args,
            respondent_arguments=resp_args,
            ratio_decidendi=ratio,
            key_precedents=key_precedents,
            operative_holding=holding,
            formatted_headnote=formatted,
            word_count=word_count,
        )

    def _build_catchwords(
        self,
        legal_domain: Optional[str],
        case_type: Optional[str],
        enriched: Optional[EnrichedLegalMetadata],
        disposition: DispositionType,
    ) -> tuple[list[str], str]:
        """Generate structured legal catchwords in standard law-reporter style."""
        parts: list[str] = []

        # Domain & Case Type
        if legal_domain:
            parts.append(legal_domain.upper())
        if case_type:
            parts.append(case_type)

        # Statutes & Sections
        if enriched and enriched.statutes:
            for st in enriched.statutes[:3]:
                if st.sections:
                    sec_str = ", ".join(st.sections[:4])
                    parts.append(f"{st.act_code}, Ss. {sec_str}")
                else:
                    parts.append(st.act_code)

        # Disposition
        disp_label = disposition.value.replace("_", " ").upper()
        parts.append(f"Outcome: {disp_label}")

        line = " — ".join(parts)
        return parts, line

    def _extract_brief_facts(self, paragraphs: list[CleanedParagraph]) -> str:
        """Synthesize concise factual narrative from FACTS role paragraphs."""
        fact_paras = [p for p in paragraphs if p.rhetorical_role == RhetoricalRole.FACTS]
        if fact_paras:
            # Take first 2-3 most descriptive paragraphs up to ~300 words
            selected: list[str] = []
            curr_words = 0
            for p in fact_paras:
                w_cnt = p.word_count or len(p.text.split())
                if curr_words + w_cnt <= 350:
                    selected.append(p.text.strip())
                    curr_words += w_cnt
                elif not selected:
                    # If first paragraph is huge, take first 250 words
                    words = p.text.split()[:250]
                    selected.append(" ".join(words) + "...")
                    break
                else:
                    break
            return "\n\n".join(selected)

        # Fallback to early paragraphs
        if paragraphs:
            return paragraphs[0].text[:400].strip() + ("..." if len(paragraphs[0].text) > 400 else "")
        return ""

    def _extract_issues(self, paragraphs: list[CleanedParagraph]) -> list[str]:
        """Extract framed questions of law from ISSUE role paragraphs or cue sentences."""
        issues: list[str] = []
        issue_paras = [p for p in paragraphs if p.rhetorical_role == RhetoricalRole.ISSUE]

        for p in issue_paras:
            sentences = re.split(r"(?<=[.!?])\s+", p.text)
            for s in sentences:
                s_clean = s.strip()
                if any(re.search(pat, s_clean) for pat in ISSUE_CUE_PATTERNS):
                    if len(s_clean) > 20 and s_clean not in issues:
                        issues.append(s_clean)

        # If no explicit issue paragraphs, scan all paragraphs for strong question patterns
        if not issues:
            for p in paragraphs[:15]:
                for s in re.split(r"(?<=[.!?])\s+", p.text):
                    s_clean = s.strip()
                    if re.search(r"(?i)\bwhether\s+(?:the\s+)?(?:petitioner|applicant|respondent|court)\b", s_clean):
                        if 25 < len(s_clean) < 300 and s_clean not in issues:
                            issues.append(s_clean)
                            if len(issues) >= 3:
                                break

        return issues[:3]

    def _extract_party_arguments(
        self,
        paragraphs: list[CleanedParagraph],
    ) -> tuple[list[str], list[str]]:
        """Extract key contentions submitted by petitioner and respondent."""
        pet_args: list[str] = []
        resp_args: list[str] = []

        pet_paras = [p for p in paragraphs if p.rhetorical_role == RhetoricalRole.ARGUMENTS_PETITIONER]
        resp_paras = [p for p in paragraphs if p.rhetorical_role == RhetoricalRole.ARGUMENTS_RESPONDENT]

        for p in pet_paras[:2]:
            text = p.text.strip()
            # Extract first 1-2 sentences
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if sentences:
                pet_args.append(" ".join(sentences[:2]))

        for p in resp_paras[:2]:
            text = p.text.strip()
            sentences = re.split(r"(?<=[.!?])\s+", text)
            if sentences:
                resp_args.append(" ".join(sentences[:2]))

        return pet_args, resp_args

    def _extract_ratio_decidendi(self, paragraphs: list[CleanedParagraph]) -> str:
        """Extract legal principles and findings from ANALYSIS paragraphs."""
        analysis_paras = [p for p in paragraphs if p.rhetorical_role == RhetoricalRole.ANALYSIS]
        ratio_sentences: list[str] = []

        for p in analysis_paras:
            sentences = re.split(r"(?<=[.!?])\s+", p.text)
            for s in sentences:
                s_clean = s.strip()
                if any(re.search(pat, s_clean) for pat in RATIO_CUE_PATTERNS):
                    if len(s_clean) > 30 and s_clean not in ratio_sentences:
                        ratio_sentences.append(s_clean)
                        if len(ratio_sentences) >= 3:
                            break
            if len(ratio_sentences) >= 3:
                break

        if ratio_sentences:
            return " ".join(ratio_sentences)

        # Fallback: take first substantive paragraph from analysis
        if analysis_paras:
            return analysis_paras[0].text[:400].strip() + ("..." if len(analysis_paras[0].text) > 400 else "")

        return ""

    def _extract_key_precedents(
        self,
        edges: Optional[list[CitationEdge]],
        paragraphs: list[CleanedParagraph],
    ) -> list[str]:
        """Collect authorities that were followed, relied upon, or prominent."""
        precedents: list[str] = []
        if edges:
            for e in edges:
                if e.treatment in (CitationTreatment.RELIED_UPON, CitationTreatment.FOLLOWED):
                    label = f"{e.cited_title} ({e.cited_citation})" if e.cited_title else e.cited_citation
                    treat_str = e.treatment.value.replace("_", " ").title()
                    entry = f"{label} [{treat_str}]"
                    if entry not in precedents:
                        precedents.append(entry)
                        if len(precedents) >= 4:
                            break

        # If none with explicit treatment, take first 2 edges from PRECEDENTS role
        if not precedents and edges:
            for e in edges:
                if e.rhetorical_role == RhetoricalRole.PRECEDENTS:
                    label = f"{e.cited_title} ({e.cited_citation})" if e.cited_title else e.cited_citation
                    if label not in precedents:
                        precedents.append(label)
                        if len(precedents) >= 3:
                            break

        return precedents

    def _extract_operative_holding(
        self,
        paragraphs: list[CleanedParagraph],
        disposition: DispositionType,
    ) -> str:
        """Extract final dispositive direction from OPERATIVE paragraphs."""
        op_paras = [p for p in paragraphs if p.rhetorical_role == RhetoricalRole.OPERATIVE]
        if op_paras:
            # Take the final 1-2 operative paragraphs
            last_paras = op_paras[-2:]
            return "\n\n".join(p.text.strip() for p in last_paras)

        disp_upper = disposition.value.replace("_", " ").upper()
        return f"The petition is accordingly {disp_upper}."

    def _render_formatted_headnote(
        self,
        court: str,
        year: int,
        bench: list[str],
        decision_date: Optional[str],
        case_title: str,
        docket: Optional[str],
        catchwords_line: str,
        brief_facts: str,
        issues: list[str],
        pet_args: list[str],
        resp_args: list[str],
        ratio: str,
        key_precedents: list[str],
        holding: str,
        disposition: DispositionType,
    ) -> str:
        """Render publication-grade law report headnote text."""
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append(f"[{year}] {court.upper()}")
        lines.append(f"Bench: {', '.join(bench) if bench else 'Single Bench'}")
        if decision_date:
            lines.append(f"Date of Decision: {decision_date}")
        lines.append(f"Case: {case_title}" + (f" | Docket: {docket}" if docket else ""))
        lines.append("-" * 80)

        if catchwords_line:
            lines.append(f"CATCHWORDS:\n{catchwords_line}\n")

        if brief_facts:
            lines.append(f"BRIEF FACTS:\n{brief_facts}\n")

        if issues:
            lines.append("ISSUES / QUESTIONS OF LAW:")
            for iss in issues:
                lines.append(f" • {iss}")
            lines.append("")

        if pet_args or resp_args:
            lines.append("SUBMISSIONS OF PARTIES:")
            if pet_args:
                lines.append(f" • Petitioner: {' '.join(pet_args)}")
            if resp_args:
                lines.append(f" • Respondent: {' '.join(resp_args)}")
            lines.append("")

        if ratio:
            lines.append(f"RATIO DECIDENDI:\n{ratio}\n")

        if key_precedents:
            lines.append("KEY PRECEDENTS CONSIDERED:")
            for p in key_precedents:
                lines.append(f" • {p}")
            lines.append("")

        disp_str = disposition.value.replace("_", " ").upper()
        lines.append(f"DISPOSITION & OPERATIVE HOLDING:\n[{disp_str}] {holding}")
        lines.append("=" * 80)

        return "\n".join(lines)

    def generate_for_document_dir(self, doc_dir: Path) -> LegalHeadnote:
        """Generate and write headnote.json and headnote.txt for a single document folder."""
        cleaned_file = doc_dir / "cleaned.json"
        if not cleaned_file.exists():
            raise FileNotFoundError(f"Missing cleaned.json in {doc_dir}")

        cleaned_data = json.loads(cleaned_file.read_text(encoding="utf-8"))
        judgment = CleanedJudgment(**cleaned_data)

        enriched: Optional[EnrichedLegalMetadata] = None
        enriched_file = doc_dir / "enriched.json"
        if enriched_file.exists():
            enriched_data = json.loads(enriched_file.read_text(encoding="utf-8"))
            enriched = EnrichedLegalMetadata(**enriched_data)

        # Edges if available
        edges: list[CitationEdge] = []
        # Fallback inline edge extraction if needed
        from ingestion_source.graph import CitationGraphBuilder
        edges = CitationGraphBuilder().extract_document_edges(judgment, enriched=enriched)

        headnote = self.generate_headnote(judgment=judgment, enriched=enriched, edges=edges)

        # Write headnote.json
        (doc_dir / "headnote.json").write_text(
            json.dumps(headnote.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Write headnote.txt
        (doc_dir / "headnote.txt").write_text(headnote.formatted_headnote, encoding="utf-8")

        return headnote

    def generate_for_store(
        self,
        store_dir: Path,
        output_jsonl: Optional[Path] = None,
    ) -> list[LegalHeadnote]:
        """Generate headnotes across all stored documents and produce consolidated JSONL."""
        cleaned_files = list(store_dir.glob("*/*/*/cleaned.json"))
        logger.info("Found %d judgments to generate legal headnotes in %s", len(cleaned_files), store_dir)

        all_headnotes: list[LegalHeadnote] = []
        for i, f in enumerate(cleaned_files, 1):
            try:
                hn = self.generate_for_document_dir(f.parent)
                all_headnotes.append(hn)
                if i % 25 == 0 or i == len(cleaned_files):
                    logger.info("Generated %d/%d headnotes...", i, len(cleaned_files))
            except Exception as e:
                logger.error("Error generating headnote for %s: %s", f.parent, e)

        if output_jsonl and all_headnotes:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(output_jsonl, "w", encoding="utf-8") as out:
                for h in all_headnotes:
                    out.write(json.dumps(h.model_dump(mode="json"), ensure_ascii=False) + "\n")
            logger.info("Saved %d headnotes to %s", len(all_headnotes), output_jsonl)

        return all_headnotes

    def generate_report(self, headnotes: list[LegalHeadnote]) -> HeadnoteReport:
        """Calculate quality metrics across generated headnotes."""
        total = len(headnotes)
        avg_words = (sum(h.word_count for h in headnotes) / total) if total > 0 else 0.0
        with_facts = sum(1 for h in headnotes if h.brief_facts)
        with_issues = sum(1 for h in headnotes if h.issues)
        with_ratio = sum(1 for h in headnotes if h.ratio_decidendi)
        with_holding = sum(1 for h in headnotes if h.operative_holding)

        return HeadnoteReport(
            total_documents=total,
            headnotes_generated=total,
            average_headnote_words=round(avg_words, 1),
            documents_with_facts=with_facts,
            documents_with_issues=with_issues,
            documents_with_ratio=with_ratio,
            documents_with_holding=with_holding,
        )
