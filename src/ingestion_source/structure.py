"""Legal Rhetorical Role Labeler (RRL) and Structure Detection Engine for LegalTrace.

Step 5 of Pipeline A:
- Classifies each paragraph in a court judgment into legal rhetorical roles:
  - PREAMBLE: Cause details, procedural introduction, bench and appearances.
  - FACTS: Case background, prosecution narrative, FIR details, commercial history.
  - ISSUE: Points of determination, framed legal questions, core controversies.
  - ARGUMENTS_PETITIONER: Pleas, submissions, contentions of the petitioner/appellant.
  - ARGUMENTS_RESPONDENT: Arguments, objections, and counter-pleas of the respondent/State.
  - STATUTE_REFERENCE: Quotation and dissection of statutory sections, rules, and clauses.
  - PRECEDENTS: Citations, analysis, and quotations of binding judicial authorities.
  - ANALYSIS: The Court's independent reasoning, legal assessment, findings, and ratio decidendi.
  - OPERATIVE: Final operative directions, grant/refusal of relief, bail terms, disposals.
- Implements sequential state-machine smoothing across section headings.
"""

from __future__ import annotations

from collections import Counter
import json
import logging
from pathlib import Path
import re
from typing import Optional

from ingestion_source.models import (
    CleanedJudgment,
    CleanedParagraph,
    ParagraphType,
    RhetoricalRole,
)

logger = logging.getLogger(__name__)

# Rhetorical Role Heading Patterns
HEADING_ROLE_MAP: list[tuple[str, RhetoricalRole]] = [
    (r"(?i)\b(?:FACTS|FACTUAL\s+MATRIX|BRIEF\s+FACTS|BACKGROUND|PROSECUTION\s+CASE)\b", RhetoricalRole.FACTS),
    (r"(?i)\b(?:ISSUES?|POINTS?\s+FOR\s+CONSIDERATION|QUESTIONS?\s+OF\s+LAW)\b", RhetoricalRole.ISSUE),
    (r"(?i)\b(?:SUBMISSIONS?|CONTENTIONS?|ARGUMENTS?)\b.*?\b(?:PETITIONER|APPELLANT|APPLICANT|PLAINTIFF)\b", RhetoricalRole.ARGUMENTS_PETITIONER),
    (r"(?i)\b(?:SUBMISSIONS?|CONTENTIONS?|ARGUMENTS?)\b.*?\b(?:RESPONDENT|DEFENDANT|STATE|PROSECUTION)\b", RhetoricalRole.ARGUMENTS_RESPONDENT),
    (r"(?i)\b(?:ANALYSIS|FINDINGS?|REASONING|CONSIDERATION|DISCUSSION|COURT'?S?\s+VIEW)\b", RhetoricalRole.ANALYSIS),
    (r"(?i)\b(?:ORDER|CONCLUSION|OPERATIVE\s+ORDER|DECISION)\b", RhetoricalRole.OPERATIVE),
]

# Paragraph-Level Linguistic Cue Patterns
CUE_PATTERNS: list[tuple[str, RhetoricalRole, float]] = [
    # OPERATIVE
    (r"(?i)\b(?:bail\s+is\s+granted|application\s+is\s+(?:allowed|dismissed|disposed)|petition\s+is\s+(?:allowed|dismissed|disposed)|appeal\s+is\s+(?:allowed|dismissed)|following\s+conditions:)\b", RhetoricalRole.OPERATIVE, 0.95),
    (r"(?i)\b(?:accordingly\s+disposed\s+of|petition\s+stands\s+disposed|impugned\s+order\s+is\s+set\s+aside|finding\s+no\s+merit|devoid\s+of\s+merit)\b", RhetoricalRole.OPERATIVE, 0.90),
    (r"(?i)\b(?:ordered\s+accordingly|registry\s+is\s+directed|be\s+released\s+on\s+bail|furnish\s+a\s+personal\s+bond)\b", RhetoricalRole.OPERATIVE, 0.85),

    # ARGUMENTS_PETITIONER
    (r"(?i)\b(?:learned\s+(?:senior\s+)?counsel\s+(?:appearing\s+)?(?:for|on\s+behalf\s+of)\s+the\s+(?:petitioner|appellant|applicant)|petitioner\s+submits?\s+that|applicant\s+contends?\s+that)\b", RhetoricalRole.ARGUMENTS_PETITIONER, 0.90),
    (r"(?i)\b(?:it\s+is\s+(?:vehemently\s+)?urged\s+by\s+the\s+learned\s+counsel\s+for\s+the\s+(?:petitioner|appellant)|counsel\s+for\s+the\s+petitioner\s+argued)\b", RhetoricalRole.ARGUMENTS_PETITIONER, 0.90),

    # ARGUMENTS_RESPONDENT
    (r"(?i)\b(?:learned\s+(?:APP|SPP|Standing\s+Counsel|Public\s+Prosecutor|counsel)\s+(?:for|on\s+behalf\s+of)\s+(?:the\s+)?(?:State|respondent|complainant)|State\s+opposes?\s+the\s+bail)\b", RhetoricalRole.ARGUMENTS_RESPONDENT, 0.90),
    (r"(?i)\b(?:learned\s+counsel\s+for\s+the\s+respondent\s+(?:submits?|argues?|contends?)|on\s+behalf\s+of\s+the\s+respondent,\s+it\s+is\s+urged)\b", RhetoricalRole.ARGUMENTS_RESPONDENT, 0.90),

    # ISSUE
    (r"(?i)\b(?:the\s+question\s+(?:that\s+)?arises?\s+for\s+consideration|the\s+issue\s+before\s+this\s+court|the\s+point\s+for\s+determination\s+is|whether\s+(?:the\s+)?(?:petitioner|applicant)\s+is\s+entitled)\b", RhetoricalRole.ISSUE, 0.85),

    # PRECEDENTS
    (r"(?i)\b(?:Hon'ble\s+Supreme\s+Court\s+in\s+(?:the\s+case\s+of)?|reliance\s+is\s+placed\s+on\s+the\s+(?:judgment|decision)\s+of|a\s+coordinate\s+bench\s+of\s+this\s+court\s+observed|in\s+the\s+case\s+of\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+v(?:s)?\.?)\b", RhetoricalRole.PRECEDENTS, 0.85),

    # FACTS
    (r"(?i)\b(?:briefly\s+stated,?\s+the\s+facts|the\s+facts\s+of\s+the\s+case\s+are|the\s+case\s+of\s+the\s+prosecution\s+is\s+that|an\s+FIR\s+bearing\s+No\.|as\s+per\s+the\s+complaint\s+dated|the\s+applicant\s+was\s+intercepted|recovery\s+was\s+effected)\b", RhetoricalRole.FACTS, 0.85),

    # ANALYSIS
    (r"(?i)\b(?:this\s+court\s+has\s+considered\s+the\s+(?:rival\s+)?submissions|we\s+have\s+perused\s+the\s+record|a\s+careful\s+reading\s+of|in\s+the\s+considered\s+opinion\s+of\s+this\s+court|it\s+is\s+well\s+settled\s+that|having\s+heard\s+the\s+learned\s+counsel|the\s+law\s+on\s+this\s+point\s+is\s+clear)\b", RhetoricalRole.ANALYSIS, 0.80),
]


class LegalStructureClassifier:
    """Classifies court judgment paragraphs into standardized legal rhetorical roles."""

    def classify_judgment(self, judgment: CleanedJudgment) -> CleanedJudgment:
        """Assign rhetorical roles to all paragraphs in CleanedJudgment."""
        paragraphs = judgment.paragraphs
        if not paragraphs:
            return judgment

        total_paras = len(paragraphs)
        active_section_role: Optional[RhetoricalRole] = None

        for idx, para in enumerate(paragraphs):
            text = para.text.strip()

            # 1. Check if paragraph is a section heading
            heading_role = self._check_heading_role(text, para.para_type)
            if heading_role:
                active_section_role = heading_role
                para.rhetorical_role = heading_role
                continue

            # 2. Check for statutory quote blocks
            if para.para_type == ParagraphType.STATUTORY_QUOTE or re.match(r'(?i)^["“]?\s*(?:Section|Article|Rule|Clause)\s+\d+', text):
                para.rhetorical_role = RhetoricalRole.STATUTE_REFERENCE
                continue

            # 3. Check for case quote blocks
            if para.para_type == ParagraphType.CASE_QUOTE:
                para.rhetorical_role = RhetoricalRole.PRECEDENTS
                continue

            # 4. Tail paragraphs check for operative orders (last 4 paragraphs)
            if idx >= max(0, total_paras - 4):
                if re.search(r"(?i)\b(?:application\s+is\s+(?:allowed|dismissed|disposed)|bail\s+is\s+granted|petition\s+is\s+(?:allowed|dismissed|disposed)|accordingly\s+disposed|ordered\s+accordingly|interim\s+order)\b", text):
                    para.rhetorical_role = RhetoricalRole.OPERATIVE
                    continue

            # 5. First paragraph check for preamble / intro
            if idx == 0 and para.label in ("1", None):
                if re.search(r"(?i)\b(?:seeks\s+regular\s+bail|filed\s+under\s+Section|present\s+petition\s+under|calls\s+in\s+question)\b", text):
                    para.rhetorical_role = RhetoricalRole.PREAMBLE
                    continue

            # 6. Linguistic cue matching
            matched_role = self._match_linguistic_cues(text)
            if matched_role:
                para.rhetorical_role = matched_role
                # If we encounter an explicit argument or analysis cue, update active section role
                if matched_role in (RhetoricalRole.ARGUMENTS_PETITIONER, RhetoricalRole.ARGUMENTS_RESPONDENT, RhetoricalRole.ANALYSIS):
                    active_section_role = matched_role
                continue

            # 7. Fallback to active section context or default
            if active_section_role:
                para.rhetorical_role = active_section_role
            else:
                # Early paragraphs lean toward FACTS, later paragraphs lean toward ANALYSIS
                if idx < max(3, total_paras // 4):
                    para.rhetorical_role = RhetoricalRole.FACTS
                else:
                    para.rhetorical_role = RhetoricalRole.ANALYSIS

        return judgment

    def _check_heading_role(self, text: str, para_type: ParagraphType) -> Optional[RhetoricalRole]:
        """Detect whether text represents a recognized legal section heading."""
        if para_type == ParagraphType.HEADING or len(text) < 80:
            for pat, role in HEADING_ROLE_MAP:
                if re.search(pat, text):
                    return role
        return None

    def _match_linguistic_cues(self, text: str) -> Optional[RhetoricalRole]:
        """Match paragraph text against weighted legal cue phrases."""
        for pattern, role, _ in CUE_PATTERNS:
            if re.search(pattern, text):
                return role
        return None

    def structure_document_dir(self, doc_dir: Path) -> CleanedJudgment:
        """Classify and update paragraphs in a document's cleaned.json."""
        cleaned_file = doc_dir / "cleaned.json"
        if not cleaned_file.exists():
            raise FileNotFoundError(f"Missing cleaned.json in {doc_dir}")

        data = json.loads(cleaned_file.read_text(encoding="utf-8"))
        judgment = CleanedJudgment(**data)

        classified_judgment = self.classify_judgment(judgment)

        # Write updated cleaned.json
        cleaned_file.write_text(
            json.dumps(classified_judgment.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return classified_judgment

    def structure_store(
        self,
        store_dir: Path,
        output_jsonl: Optional[Path] = None,
    ) -> list[CleanedJudgment]:
        """Batch process all cleaned judgments in raw-store."""
        cleaned_files = list(store_dir.glob("*/*/*/cleaned.json"))
        logger.info("Found %d judgments for rhetorical role labeling in %s", len(cleaned_files), store_dir)

        updated_list: list[CleanedJudgment] = []
        for i, f in enumerate(cleaned_files, 1):
            try:
                res = self.structure_document_dir(f.parent)
                updated_list.append(res)
                if i % 25 == 0 or i == len(cleaned_files):
                    logger.info("Structured %d/%d judgments...", i, len(cleaned_files))
            except Exception as e:
                logger.error("Error structuring %s: %s", f.parent, e)

        if output_jsonl and updated_list:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(output_jsonl, "w", encoding="utf-8") as out:
                for j in updated_list:
                    out.write(json.dumps(j.model_dump(mode="json"), ensure_ascii=False) + "\n")
            logger.info("Saved %d structured judgments to %s", len(updated_list), output_jsonl)

        return updated_list
