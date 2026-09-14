"""Robust Legal Document Cleaning Pipeline for Indian Kanoon Court Judgments.

Step 3 of Pipeline A:
- Unicode & typographical normalization (NFKC, quotes, hyphens, non-breaking spaces).
- Indian legal noise & artifact removal (listing dockets, steno marks, upload notices, Kanoon links).
- Structural segmentation (Header, Preamble, Paragraphs, Operative Order, Signatures).
- Fine-grained paragraph classification (Heading, Body, Quote, Operative).
- Deterministic disposition classification (Allowed, Dismissed, Disposed, Referred, etc.).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Optional
import unicodedata

from ingestion_source.models import (
    CleanedHeader,
    CleanedJudgment,
    CleanedParagraph,
    CleaningMetrics,
    DispositionType,
    ParagraphType,
)

logger = logging.getLogger(__name__)


# Standard Indian Legal Noise Patterns
NOISE_REGEXES = [
    # Indian Kanoon links & boilerplate
    (r"(?i)Indian\s+Kanoon\s*-\s*http[s]?://indiankanoon\.org/doc/\d+/?", "kanoon_url"),
    (r"(?i)Take\s+printout\s+of\s+this\s+judgment", "printout_boilerplate"),
    (r"(?i)Click\s+here\s+to\s+check\s+corrigendum[^\n]*", "corrigendum_note"),
    (r"(?i)Top\s+\d+\s+results", "search_noise"),
    # Listing dockets at top of Delhi High Court / tribunals
    (r"^\s*\$~(?:\s*\(Original\))?\s*\d*\s*\*?\s*", "listing_docket_prefix"),
    (r"^\s*%\s*(?=(?:Reserved|Pronounced|Judgment|Date))", "listing_percent_prefix"),
    # Digital signature stamps & certificates
    (
        r"(?i)Signature\s+Not\s+Verified\s+Digitally\s+Signed\s+By:[^\n]*(?:\n\s*(?:Signing\s+Date|Date):[^\n]*)?",
        "digital_signature_block",
    ),
    (
        r"(?i)Digitally\s+Signed\s+By:[^\n]*(?:\n\s*(?:Signing\s+Date|Date):[^\n]*)?",
        "digital_signature_stamp",
    ),
    # Pagination & header line tags
    (r"^\s*-\s*\d+\s*-\s*$", "page_number_hyphen"),
    (r"^\s*Page\s+\d+\s+(?:of\s+\d+)?\s*$", "page_number_text"),
    # Upload notices
    (
        r"(?i)(?:The\s+)?(?:order|judgment)\s+be\s+uploaded\s+on\s+the\s+website\s+(?:of\s+this\s+Court,?\s*)?forthwith\.?",
        "upload_notice",
    ),
]

# Words that should remain hyphenated when found across lines
HYPHENATED_LEGAL_PREFIXES = {
    "co", "pre", "post", "sub", "non", "cross", "inter", "intra", "anti",
    "re", "quasi", "ex", "pro", "self", "suo",
}


class LegalDocumentCleaner:
    """Production-grade legal judgment text cleaner for Indian court jurisprudence."""

    def __init__(self) -> None:
        self.noise_compiled = [(re.compile(pat, re.MULTILINE), name) for pat, name in NOISE_REGEXES]

    def clean_text(
        self,
        raw_text: str,
        doc_id: str = "unknown",
        court: str = "Unknown Court",
        court_slug: str = "court",
        year: int = 2023,
        title: str = "Unknown Title",
    ) -> CleanedJudgment:
        """Execute full cleaning and structural parsing on raw judgment text."""
        raw_char_count = len(raw_text)
        detected_noise: list[str] = []

        # 1. Unicode & Character Normalization
        text = self._normalize_unicode_and_characters(raw_text)

        # 2. Hyphenation & Line-wrap Repair
        text = self._repair_hyphenation(text)

        # 3. Strip Known Indian Kanoon / Registry Noise
        text, detected_noise = self._strip_noise(text)

        # 4. Extract Header & Separate from Judgment Body
        header, body_text = self._extract_header_and_body(text, court)

        # 5. Segment and Classify Paragraphs
        paragraphs = self._segment_paragraphs(body_text)

        # 6. Extract Operative Order & Disposition
        operative_text, disposition = self._extract_disposition_and_operative(paragraphs, body_text)

        # 7. Build Continuous Clean Text
        clean_full_text = self._reconstruct_clean_text(header, paragraphs)
        cleaned_char_count = len(clean_full_text)
        noise_removed = max(0, raw_char_count - cleaned_char_count)
        reduction_pct = (noise_removed / raw_char_count * 100.0) if raw_char_count > 0 else 0.0

        metrics = CleaningMetrics(
            raw_char_count=raw_char_count,
            cleaned_char_count=cleaned_char_count,
            noise_removed_chars=noise_removed,
            reduction_percentage=round(reduction_pct, 2),
            paragraphs_count=len(paragraphs),
            disposition=disposition,
            noise_patterns_detected=detected_noise,
        )

        return CleanedJudgment(
            doc_id=doc_id,
            court=court,
            court_slug=court_slug,
            year=year,
            title=title,
            header=header,
            paragraphs=paragraphs,
            operative_order=operative_text,
            disposition=disposition,
            clean_full_text=clean_full_text,
            metrics=metrics,
            cleaned_at=datetime.now(timezone.utc),
        )

    def _normalize_unicode_and_characters(self, text: str) -> str:
        """Normalize Unicode forms, whitespace, quotes, and dashes."""
        # NFKC normalizes compatibility characters
        text = unicodedata.normalize("NFKC", text)

        # Replace non-breaking spaces and exotic space characters with standard ASCII space
        text = re.sub(r"[\u00a0\u2000-\u200b\u202f\u205f\ufeff]", " ", text)

        # Remove zero-width spaces, joiners, non-joiners, soft-hyphens
        text = re.sub(r"[\u200b\u200c\u200d\u00ad]", "", text)

        # Normalize quotes to standard ASCII
        text = re.sub(r"[“”„‟«»]", '"', text)
        text = re.sub(r"[‘’‛′‵]", "'", text)

        # Normalize dashes (em-dash, en-dash, figure dash) to standard hyphen
        text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)

        # Remove trailing carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        return text

    def _repair_hyphenation(self, text: str) -> str:
        """Rejoin words split across line breaks while preserving compound prefixes."""
        def _rejoin(match: re.Match) -> str:
            prefix = match.group(1)
            suffix = match.group(2)
            if prefix.lower() in HYPHENATED_LEGAL_PREFIXES:
                return f"{prefix}-{suffix}"
            return f"{prefix}{suffix}"

        # Match: word- \n word
        pattern = r"(\b[a-zA-Z]{2,})-\s*\n\s*([a-zA-Z]{2,}\b)"
        return re.sub(pattern, _rejoin, text)

    def _strip_noise(self, text: str) -> tuple[str, list[str]]:
        """Apply noise-stripping regular expressions."""
        detected = []
        for regex, name in self.noise_compiled:
            if regex.search(text):
                detected.append(name)
                text = regex.sub("", text)

        # Strip stenographer initials at the end of the document (e.g., 'MAY 31, 2023/sn/bsr' or 'JUNE 01, 2023 RN/Anuj/DJ')
        text = re.sub(r"(?i)(\b(?:JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s+\d{1,2},\s+\d{4})\s*(?:/|\s+)[a-zA-Z0-9/_\-\s]{2,20}$", r"\1", text.strip())

        # Clean multiple consecutive horizontal spaces
        text = re.sub(r"[ \t]+", " ", text)
        # Clean excessive newlines (max 2 consecutive newlines)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip(), detected

    def _extract_header_and_body(self, text: str, default_court: str) -> tuple[CleanedHeader, str]:
        """Separate formal case header from judicial text."""
        header = CleanedHeader(court=default_court)

        # Check for Neutral Citation
        nc_match = re.search(r"(?i)(?:NEUTRAL\s+CITATION\s*(?:NO\.?|NUMBER)?:?\s*)([\d]{4}:[A-Z]+:[\d]+|[\d]{4}/[A-Z]+/[\d]+)", text[:2000])
        if nc_match:
            header.neutral_citation = nc_match.group(1).strip()

        # Check for Decision Date
        dec_match = re.search(r"(?i)(?:Date\s+of\s+decision|Judgment\s+pronounced\s+on|Pronounced\s+on|Order\s+Date)\s*:\s*([^\n\+]+)", text[:2500])
        if dec_match:
            header.decision_date = dec_match.group(1).strip().rstrip(".").strip()

        # Check for Reserved Date
        res_match = re.search(r"(?i)(?:Judgment\s+reserved\s+on|Reserved\s+on)\s*:\s*([^\n\+]+)", text[:2500])
        if res_match:
            header.reserved_date = res_match.group(1).strip().rstrip(".").strip()

        # Check for Case Numbers
        case_nums = re.findall(r"(?i)\b(?:BAIL\s+APPLN\.?|W\.?P\.?\(C\)|CRL\.?A\.?|CRL\.?M\.?C\.?|CS\(COMM\)|O\.?M\.?P\.?(?:\s*\(COMM\))?|CIVIL\s+APPEAL|CRIMINAL\s+APPEAL)\s*(?:NO\.?)?\s*[\d/]+", text[:2500])
        header.docket_numbers = list(dict.fromkeys(cn.strip() for cn in case_nums))

        # Check for Parties
        parties_match = re.search(
            r"(?i)(?:^|\n)\s*([^\n]+?)\s*(?:\.{2,}|\-+)\s*(?:Petitioner|Appellant|Plaintiff|Applicant)[^\n]*?\s+(?:versus|vs\.?|v\.)\s+([^\n]+?)\s*(?:\.{2,}|\-+)\s*(?:Respondent|Defendant)",
            text[:3500],
        )
        if parties_match:
            pet = re.sub(r"^[+\d\.\s]+", "", parties_match.group(1)).strip()
            # Strip preceding court name, date of decision, or docket if in same line
            pet = re.sub(r"(?i)^.*?\bIN THE [A-Z\s]+COURT[A-Z\s]*\b", "", pet)
            pet = re.sub(r"(?i)^.*?\b(?:Date of decision|Pronounced on|Order Date)\s*:[^+\n]+", "", pet)
            pet = re.sub(r"(?i)^.*?\b[A-Z\.\(\)]+\s*(?:NO\.?)?\s*[\d/]+(?:\s*\+\s*)?", "", pet)
            header.petitioner = pet.strip().lstrip("+").strip()

            resp = re.sub(r"^[+\d\.\s]+", "", parties_match.group(2)).strip()
            header.respondent = resp.strip()

        # Check for Advocates / Counsel
        pet_counsel = re.search(r"(?i)\bPetitioner\b[^\n]*?\bThrough\s*:\s*([^\n]+?)(?=\s+versus|\s+vs\.|\n)", text[:3500])
        if pet_counsel:
            header.petitioner_counsel = [c.strip() for c in re.split(r",\s*|\s+and\s+", pet_counsel.group(1)) if c.strip()]

        resp_counsel = re.search(r"(?i)\bRespondent\b[^\n]*?\bThrough\s*:\s*([^\n]+?)(?=\s+CORAM|\s+BEFORE|\s+JUDGMENT|\s+ORDER|\n)", text[:3500])
        if resp_counsel:
            header.respondent_counsel = [c.strip() for c in re.split(r",\s*|\s+and\s+", resp_counsel.group(1)) if c.strip()]

        # Check for Bench / Coram
        coram_match = re.search(r"(?i)\b(?:CORAM|BEFORE)\s*:\s*([^\n]+)", text[:3500])
        if coram_match:
            raw_coram = re.split(r"(?i)\b(?:JUDGMENT|ORDER)\b", coram_match.group(1))[0].strip()
            judges = re.findall(r"(?i)(?:HON'BLE\s+)?(?:MR\.|MS\.|MRS\.|DR\.)?\s*JUSTICE\s+[A-Z\s\.]+", raw_coram)
            if judges:
                header.bench = list(dict.fromkeys(re.sub(r"\s+", " ", j).strip() for j in judges))
            elif raw_coram:
                header.bench = [re.sub(r"\s+", " ", raw_coram).strip()]

        # Search for transition marker to judgment body
        split_match = re.search(
            r"(?i)\n\s*(?:(?:CORAM|BEFORE)\s*:[^\n]*\n+)?(?:JUDGMENT|ORDER|O\s*R\s*D\s*E\s*R)\s*\n+",
            text[:4000],
        )

        if split_match:
            split_idx = split_match.end()
            header.header_raw = text[:split_idx].strip()
            body = text[split_idx:].strip()
        else:
            # Fallback to first numbered paragraph if explicit header transition missing
            first_p = re.search(r"\n\s*(?:1\.|\[1\]|\(1\))\s+", text[:4000])
            if first_p:
                split_idx = first_p.start()
                header.header_raw = text[:split_idx].strip()
                body = text[split_idx:].strip()
            else:
                header.header_raw = ""
                body = text

        return header, body

    def _segment_paragraphs(self, body_text: str) -> list[CleanedParagraph]:
        """Segment judgment into numbered paragraphs, sub-sections, and headings."""
        raw_blocks = [b.strip() for b in body_text.split("\n\n") if b.strip()]
        paragraphs: list[CleanedParagraph] = []
        para_idx = 1

        for block in raw_blocks:
            # Check for standard paragraph numbering: 1., 2.1, A., (i), etc.
            match = re.match(r"^(\d+(?:\.\d+)*\.?|[A-Z]\.|\([a-z\divx]+\))\s+(.*)$", block, re.DOTALL)
            if match:
                label = match.group(1).rstrip(".")
                content = match.group(2).strip()
            else:
                label = None
                content = block

            # Classify paragraph type
            para_type = self._classify_paragraph(content, label)
            is_quote = (para_type in (ParagraphType.STATUTORY_QUOTE, ParagraphType.CASE_QUOTE))

            paragraphs.append(
                CleanedParagraph(
                    para_id=para_idx,
                    label=label,
                    text=content,
                    para_type=para_type,
                    is_quote=is_quote,
                    char_count=len(content),
                    word_count=len(content.split()),
                )
            )
            para_idx += 1

        return paragraphs

    def _classify_paragraph(self, text: str, label: Optional[str]) -> ParagraphType:
        """Classify paragraph content into legal semantic role."""
        cleaned = text.strip()

        # Headings: Short lines without terminating punctuation
        if len(cleaned) < 80 and not cleaned.endswith((".", ":", ";", ",", "?")):
            upper_ratio = sum(1 for c in cleaned if c.isupper()) / max(1, len(cleaned))
            if upper_ratio > 0.4 or cleaned.isupper() or cleaned.title() == cleaned:
                return ParagraphType.HEADING

        # Statutory Quotes: Sections, Clauses, Definitions
        if re.search(r'(?i)^["“]?\s*(?:Section\s+\d+|Article\s+\d+|Rule\s+\d+|Clause\s+[\d\.]+)', cleaned):
            return ParagraphType.STATUTORY_QUOTE

        # Block quotations starting and ending with quotes
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith('“') and cleaned.endswith('”')):
            return ParagraphType.CASE_QUOTE

        # Operative Order Directions
        if re.search(r"(?i)\b(?:bail is granted|application is (?:allowed|dismissed|disposed)|petition is (?:allowed|dismissed|disposed)|appeal is (?:allowed|dismissed)|following conditions:)\b", cleaned):
            return ParagraphType.OPERATIVE

        return ParagraphType.BODY

    def _extract_disposition_and_operative(
        self, paragraphs: list[CleanedParagraph], body_text: str
    ) -> tuple[Optional[str], DispositionType]:
        """Extract operative order passage and classify judicial disposition."""
        # Inspect tail paragraphs (last 5)
        tail_paras = paragraphs[-5:] if len(paragraphs) >= 5 else paragraphs
        operative_blocks = [p.text for p in tail_paras if p.para_type == ParagraphType.OPERATIVE]

        if operative_blocks:
            operative_text = "\n\n".join(operative_blocks)
        elif tail_paras:
            operative_text = tail_paras[-1].text
        else:
            operative_text = None

        # Determine disposition classification
        tail_text = " ".join([p.text.lower() for p in tail_paras])
        if not tail_text:
            tail_text = body_text[-2000:].lower()

        if "partly allowed" in tail_text or "partially allowed" in tail_text:
            disposition = DispositionType.PARTLY_ALLOWED
        elif re.search(r"\b(?:bail is granted|application is allowed|petition is allowed|appeal is allowed|quashed and set aside)\b", tail_text):
            disposition = DispositionType.ALLOWED
        elif re.search(r"\b(?:dismissed|rejected|devoid of (?:any )?merit|no ground to interfere)\b", tail_text):
            disposition = DispositionType.DISMISSED
        elif re.search(r"\b(?:disposed of|accordingly disposed)\b", tail_text):
            disposition = DispositionType.DISPOSED_OF
        elif re.search(r"\b(?:referred to|larger bench|place before hon'ble the chief justice)\b", tail_text):
            disposition = DispositionType.REFERRED
        elif re.search(r"\b(?:remanded|remitted back)\b", tail_text):
            disposition = DispositionType.REMANDED
        elif re.search(r"\b(?:set aside)\b", tail_text):
            disposition = DispositionType.SET_ASIDE
        elif re.search(r"\b(?:interim order|stay is granted)\b", tail_text):
            disposition = DispositionType.INTERIM_ORDER
        else:
            disposition = DispositionType.OTHER

        return operative_text, disposition

    def _reconstruct_clean_text(self, header: CleanedHeader, paragraphs: list[CleanedParagraph]) -> str:
        """Rebuild contiguous, noise-free text for downstream embeddings & indexing."""
        parts = []

        # Add structured header summary
        if header.court:
            parts.append(header.court.upper())
        if header.neutral_citation:
            parts.append(f"Neutral Citation: {header.neutral_citation}")
        if header.docket_numbers:
            parts.append(f"Docket: {', '.join(header.docket_numbers)}")
        if header.petitioner and header.respondent:
            parts.append(f"{header.petitioner} vs. {header.respondent}")
        if header.decision_date:
            parts.append(f"Date: {header.decision_date}")

        parts.append("\n--- JUDGMENT ---\n")

        for p in paragraphs:
            if p.label:
                parts.append(f"{p.label} {p.text}")
            else:
                parts.append(p.text)

        return "\n\n".join(parts)

    def clean_document_dir(self, doc_dir: Path) -> CleanedJudgment:
        """Clean an existing judgment directory in raw-store."""
        full_text_file = doc_dir / "full_text.txt"
        metadata_file = doc_dir / "metadata.json"

        if not full_text_file.exists():
            raise FileNotFoundError(f"Missing full_text.txt in {doc_dir}")

        raw_text = full_text_file.read_text(encoding="utf-8")
        meta = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}

        cleaned = self.clean_text(
            raw_text=raw_text,
            doc_id=meta.get("doc_id", doc_dir.name),
            court=meta.get("court", "Unknown Court"),
            court_slug=meta.get("court_slug", doc_dir.parent.parent.name),
            year=meta.get("year", 2023),
            title=meta.get("title", f"Document {doc_dir.name}"),
        )

        # Write clean files to disk
        (doc_dir / "cleaned_text.txt").write_text(cleaned.clean_full_text, encoding="utf-8")
        (doc_dir / "cleaned.json").write_text(
            json.dumps(cleaned.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return cleaned

    def clean_store(
        self,
        store_dir: Path,
        output_jsonl: Optional[Path] = None,
    ) -> list[CleanedJudgment]:
        """Batch process all judgments under raw-store."""
        judgment_dirs = [p.parent for p in store_dir.glob("*/*/*/full_text.txt")]
        logger.info("Found %d judgments to clean in %s", len(judgment_dirs), store_dir)

        cleaned_list: list[CleanedJudgment] = []
        for i, d in enumerate(judgment_dirs, 1):
            try:
                res = self.clean_document_dir(d)
                cleaned_list.append(res)
                if i % 25 == 0 or i == len(judgment_dirs):
                    logger.info("Cleaned %d/%d judgments...", i, len(judgment_dirs))
            except Exception as e:
                logger.error("Error cleaning document %s: %s", d, e)

        if output_jsonl and cleaned_list:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(output_jsonl, "w", encoding="utf-8") as f:
                for c in cleaned_list:
                    f.write(json.dumps(c.model_dump(mode="json"), ensure_ascii=False) + "\n")
            logger.info("Consolidated %d cleaned judgments to %s", len(cleaned_list), output_jsonl)

        return cleaned_list
