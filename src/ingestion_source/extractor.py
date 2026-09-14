"""Legal Metadata, Statutory Provisions, and Citation Extraction Engine for LegalTrace.

Step 4 of Pipeline A:
- Extracts canonical case types and maps them to judicial domains (Criminal, Civil, Constitutional, Commercial, Arbitration, etc.).
- Identifies and normalizes Judges and classifies Bench types (Single, Division, Full, Constitution).
- Extracts Acts, Codes, and statutory sections referenced in judgments with multi-section normalization.
- Discovers reported and neutral legal citations (SCC, AIR, DLT, SCC OnLine, Neutral Citations).
- Categorizes legal representation (Senior Advocates, Standing Counsels, Public Prosecutors).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Optional

from ingestion_source.models import (
    BenchType,
    CleanedJudgment,
    EnrichedLegalMetadata,
    ExtractedCitation,
    LegalDomain,
    StatutoryMention,
)

logger = logging.getLogger(__name__)

# Case Type Prefix Mapping
CASE_TYPE_PATTERNS: list[tuple[str, str, str, LegalDomain]] = [
    (r"\bBAIL\s+APPLN\.?\b", "Bail Application", "BAIL_APPLICATION", LegalDomain.CRIMINAL),
    (r"\bCRL\.?M\.?C\.?\b", "Criminal Miscellaneous Main", "CRL_MC", LegalDomain.CRIMINAL),
    (r"\bCRL\.?A\.?\b", "Criminal Appeal", "CRL_A", LegalDomain.CRIMINAL),
    (r"\bCRL\.?REV\.?P\.?\b", "Criminal Revision Petition", "CRL_REV_P", LegalDomain.CRIMINAL),
    (r"\bW\.?P\.?\s*\(CRL\)", "Writ Petition (Criminal)", "WP_CRL", LegalDomain.CONSTITUTIONAL),
    (r"\bW\.?P\.?\s*\(C\)", "Writ Petition (Civil)", "WP_C", LegalDomain.CONSTITUTIONAL),
    (r"\bCS\s*\(COMM\)", "Commercial Civil Suit", "CS_COMM", LegalDomain.COMMERCIAL),
    (r"\bCS\s*\(OS\)", "Civil Suit (Original Side)", "CS_OS", LegalDomain.CIVIL),
    (r"\bO\.?M\.?P\.?\s*\(COMM\)", "Commercial Arbitration Petition", "OMP_COMM", LegalDomain.ARBITRATION),
    (r"\bO\.?M\.?P\.?\b", "Arbitration Petition", "OMP", LegalDomain.ARBITRATION),
    (r"\bARB\.?P\.?\b", "Arbitration Petition", "ARB_P", LegalDomain.ARBITRATION),
    (r"\bFAO\s*\(OS\)", "First Appeal from Order (OS)", "FAO_OS", LegalDomain.CIVIL),
    (r"\bFAO\b", "First Appeal from Order", "FAO", LegalDomain.CIVIL),
    (r"\bRFA\s*\(OS\)", "Regular First Appeal (OS)", "RFA_OS", LegalDomain.CIVIL),
    (r"\bRFA\b", "Regular First Appeal", "RFA", LegalDomain.CIVIL),
    (r"\bMAT\.?APP\.?\b", "Matrimonial Appeal", "MAT_APP", LegalDomain.FAMILY),
    (r"\bCONT\.?CAS\s*\(C\)", "Contempt Case (Civil)", "CONT_CAS_C", LegalDomain.CIVIL),
    (r"\bLPA\b", "Letters Patent Appeal", "LPA", LegalDomain.CIVIL),
    (r"\bCO\.?PET\.?\b", "Company Petition", "CO_PET", LegalDomain.COMMERCIAL),
    (r"\bITA\b", "Income Tax Appeal", "ITA", LegalDomain.TAXATION),
]

# Statutory Codes & Acts Registry
KNOWN_ACTS: list[tuple[str, str, str]] = [
    # (regex pattern, Canonical Act Name, Short Code)
    (r"(?i)\b(?:Cr\.?P\.?C\.?|Code\s+of\s+Criminal\s+Procedure(?:,?\s*1973)?)\b", "Code of Criminal Procedure, 1973", "CrPC"),
    (r"(?i)\b(?:I\.?P\.?C\.?|Indian\s+Penal\s+Code(?:,?\s*1860)?)\b", "Indian Penal Code, 1860", "IPC"),
    (r"(?i)\b(?:C\.?P\.?C\.?|Code\s+of\s+Civil\s+Procedure(?:,?\s*1908)?)\b", "Code of Civil Procedure, 1908", "CPC"),
    (r"(?i)\b(?:Constitution(?:\s+of\s+India)?)\b", "Constitution of India", "Constitution"),
    (r"(?i)\b(?:N\.?D\.?P\.?S\.?(?:\s+Act)?|Narcotic\s+Drugs\s+and\s+Psychotropic\s+Substances\s+Act(?:,?\s*1985)?)\b", "Narcotic Drugs and Psychotropic Substances Act, 1985", "NDPS Act"),
    (r"(?i)\b(?:N\.?I\.?\s*Act|Negotiable\s+Instruments\s+Act(?:,?\s*1881)?)\b", "Negotiable Instruments Act, 1881", "NI Act"),
    (r"(?i)\b(?:Arbitration(?:\s+and\s+Conciliation)?\s+Act(?:,?\s*1996)?)\b", "Arbitration and Conciliation Act, 1996", "Arbitration Act"),
    (r"(?i)\b(?:Customs\s+Act(?:,?\s*1962)?)\b", "Customs Act, 1962", "Customs Act"),
    (r"(?i)\b(?:P\.?M\.?L\.?A\.?|Prevention\s+of\s+Money\s+Laundering\s+Act(?:,?\s*2002)?)\b", "Prevention of Money Laundering Act, 2002", "PMLA"),
    (r"(?i)\b(?:P\.?C\.?\s*Act|Prevention\s+of\s+Corruption\s+Act(?:,?\s*1988)?)\b", "Prevention of Corruption Act, 1988", "PC Act"),
    (r"(?i)\b(?:POCSO(?:\s+Act)?|Protection\s+of\s+Children\s+from\s+Sexual\s+Offences\s+Act(?:,?\s*2012)?)\b", "Protection of Children from Sexual Offences Act, 2012", "POCSO Act"),
    (r"(?i)\b(?:Commercial\s+Courts\s+Act(?:,?\s*2015)?)\b", "Commercial Courts Act, 2015", "Commercial Courts Act"),
    (r"(?i)\b(?:Specific\s+Relief\s+Act(?:,?\s*1963)?)\b", "Specific Relief Act, 1963", "Specific Relief Act"),
    (r"(?i)\b(?:Hindu\s+Marriage\s+Act(?:,?\s*1955)?)\b", "Hindu Marriage Act, 1955", "Hindu Marriage Act"),
    (r"(?i)\b(?:Companies\s+Act(?:,?\s*(?:2013|1956))?)\b", "Companies Act", "Companies Act"),
    (r"(?i)\b(?:Indian\s+Evidence\s+Act(?:,?\s*1872)?)\b", "Indian Evidence Act, 1872", "Evidence Act"),
    (r"(?i)\b(?:Information\s+Technology\s+Act(?:,?\s*2000)?|I\.?T\.?\s*Act)\b", "Information Technology Act, 2000", "IT Act"),
    (r"(?i)\b(?:Motor\s+Vehicles\s+Act(?:,?\s*1988)?|M\.?V\.?\s*Act)\b", "Motor Vehicles Act, 1988", "MV Act"),
]

# Citation Patterns
CITATION_REGEXES: list[tuple[str, str]] = [
    (r"\b(\d{4}:[A-Z]+:\d+)\b", "Neutral Citation"),
    (r"(?:^|\s|\()((?:19\d\d|20\d\d)\)?\s*\d+\s*SCC\s*\d+)\b", "Supreme Court Cases (SCC)"),
    (r"\b(AIR\s*(?:19\d\d|20\d\d)\s*(?:SC|Del|Delhi|Bom|Cal)\s*\d+)\b", "All India Reporter (AIR)"),
    (r"\b((?:19\d\d|20\d\d)\s*\(\d+\)\s*DLT\s*\d+)\b", "Delhi Law Times (DLT)"),
    (r"\b((?:19\d\d|20\d\d)\s*SCC\s*OnLine\s*(?:DHC|SC|Del)\s*\d+)\b", "SCC OnLine"),
    (r"\b((?:19\d\d|20\d\d)\s*\(\d+\)\s*SCALE\s*\d+)\b", "SCALE"),
    (r"\b((?:19\d\d|20\d\d)\s*(?:\(\d+\))?\s*Crimes\s*\d+)\b", "Crimes"),
]


class LegalMetadataExtractor:
    """Extracts structured legal entities, statutory provisions, citations, and counsel."""

    def extract(self, judgment: CleanedJudgment) -> EnrichedLegalMetadata:
        """Extract legal domain metadata from a CleanedJudgment."""
        doc_id = judgment.doc_id
        full_text = judgment.clean_full_text
        header = judgment.header

        # 1. Case Type & Legal Domain
        case_type, type_code, domain = self._classify_case_type_and_domain(header.docket_numbers, full_text, judgment.title)

        # 2. Canonical Judges & Bench Classification
        canonical_judges, bench_type = self._extract_judges_and_bench(header.bench, full_text)

        # 3. Statutory Provisions
        statutes = self._extract_statutes(full_text)

        # 4. Citations
        citations = self._extract_citations(full_text, header.neutral_citation)

        # 5. Counsel Classification
        senior_counsels, standing_counsels = self._classify_counsel(header.petitioner_counsel + header.respondent_counsel)

        return EnrichedLegalMetadata(
            doc_id=doc_id,
            case_type=case_type,
            case_type_code=type_code,
            legal_domain=domain,
            bench_type=bench_type,
            canonical_judges=canonical_judges,
            statutes=statutes,
            citations=citations,
            petitioner=header.petitioner,
            respondent=header.respondent,
            senior_advocates=senior_counsels,
            standing_counsels=standing_counsels,
            total_statutes_count=len(statutes),
            total_citations_count=len(citations),
        )

    def _classify_case_type_and_domain(
        self, dockets: list[str], full_text: str, title: str
    ) -> tuple[str, str, LegalDomain]:
        """Classify case type and jurisdiction domain from dockets or text."""
        docket_str = " ".join(dockets).upper()

        for pattern, type_name, type_code, domain in CASE_TYPE_PATTERNS:
            if re.search(pattern, docket_str, re.IGNORECASE):
                return type_name, type_code, domain

        # Fallback to inspecting text preamble
        preamble = (full_text[:3000] + " " + title).upper()
        if "BAIL APPL" in preamble or "REGULAR BAIL" in preamble or "ANTICIPATORY BAIL" in preamble:
            return "Bail Application", "BAIL_APPLICATION", LegalDomain.CRIMINAL
        elif "SECTION 482" in preamble:
            return "Criminal Miscellaneous Main", "CRL_MC", LegalDomain.CRIMINAL
        elif "WRIT PETITION" in preamble:
            if "CRIMINAL" in preamble:
                return "Writ Petition (Criminal)", "WP_CRL", LegalDomain.CONSTITUTIONAL
            return "Writ Petition (Civil)", "WP_C", LegalDomain.CONSTITUTIONAL
        elif "ARBITRATION" in preamble:
            return "Arbitration Petition", "ARB_P", LegalDomain.ARBITRATION
        elif "COMMERCIAL SUIT" in preamble:
            return "Commercial Civil Suit", "CS_COMM", LegalDomain.COMMERCIAL

        return "Civil Miscellaneous / General", "GENERAL", LegalDomain.CIVIL

    def _extract_judges_and_bench(
        self, bench_strings: list[str], full_text: str
    ) -> tuple[list[str], BenchType]:
        """Normalize judge names and classify bench composition."""
        judges: list[str] = []

        raw_candidates = list(bench_strings)
        if not raw_candidates:
            # Try finding CORAM in first 4000 chars
            coram_m = re.search(r"(?i)\b(?:CORAM|BEFORE)\s*:\s*([^\n]+)", full_text[:4000])
            if coram_m:
                raw_candidates.append(coram_m.group(1))

        for candidate in raw_candidates:
            # Clean Honorifics & Extract Names
            # Examples: 'HON'BLE MR. JUSTICE AMIT SHARMA', 'AMIT SHARMA, J.'
            extracted = re.findall(
                r"(?i)(?:HON'BLE\s+)?(?:MR\.|MS\.|MRS\.|DR\.)?\s*JUSTICE\s+([A-Z\s\.]+?)(?:,|;|AND|\n|$)",
                candidate,
            )
            if extracted:
                for name in extracted:
                    clean_name = self._normalize_judge_name(name)
                    if clean_name and clean_name not in judges:
                        judges.append(clean_name)
            else:
                j_match = re.search(r"([A-Z\s\.]+?),\s*J\b", candidate)
                if j_match:
                    clean_name = self._normalize_judge_name(j_match.group(1))
                    if clean_name and clean_name not in judges:
                        judges.append(clean_name)

        # Classify Bench Type
        judge_count = len(judges)
        if judge_count >= 5:
            bench_type = BenchType.CONSTITUTION_BENCH
        elif judge_count >= 3:
            bench_type = BenchType.FULL_BENCH
        elif judge_count == 2:
            bench_type = BenchType.DIVISION_BENCH
        elif judge_count == 1:
            bench_type = BenchType.SINGLE_BENCH
        else:
            bench_type = BenchType.SINGLE_BENCH  # Default for High Court judgments

        return judges, bench_type

    def _normalize_judge_name(self, raw_name: str) -> str:
        """Format judge name with canonical 'Justice <Name>'."""
        name = re.sub(r"(?i)\b(?:HON'BLE|MR\.|MS\.|MRS\.|DR\.|JUSTICE|THE|CHIEF)\b", "", raw_name)
        name = re.sub(r"[,\.]", " ", name)
        name = re.sub(r"\s+", " ", name).strip()
        if len(name) >= 3:
            return f"Justice {name.title()}"
        return ""

    def _extract_statutes(self, text: str) -> list[StatutoryMention]:
        """Extract all referenced Acts and sections."""
        statute_map: dict[str, StatutoryMention] = {}

        # 1. Extract Acts and Sections via regex patterns
        # Pattern: Section(s) X, Y and Z of Act
        sec_pattern = re.compile(
            r"(?i)\b(?:Sections?|Sec\.?|u/s)\s+([\d\w\s,/\(\)\-]+?)\s+(?:(?:of|in)\s+(?:the\s+)?([A-Za-z\s]{2,40}?(?:Act|Code|Rules)(?:,?\s*\d{4})?|Cr\.?P\.?C\.?|I\.?P\.?C\.?|C\.?P\.?C\.?|NDPS(?:\s+Act)?|NI\s*Act|PMLA|POCSO)|(Cr\.?P\.?C\.?|I\.?P\.?C\.?|C\.?P\.?C\.?|NDPS(?:\s+Act)?|NI\s*Act|PMLA|POCSO))",
            re.MULTILINE,
        )

        for match in sec_pattern.finditer(text):
            sec_raw = match.group(1).strip()
            act_raw = (match.group(2) or match.group(3) or "").strip()

            # Resolve canonical Act
            canonical_act, short_code = self._resolve_act(act_raw)
            if not canonical_act:
                continue

            # Parse section numbers
            sections = self._parse_section_numbers(sec_raw)
            if not sections:
                continue

            key = f"{short_code}"
            if key in statute_map:
                for s in sections:
                    if s not in statute_map[key].sections:
                        statute_map[key].sections.append(s)
            else:
                statute_map[key] = StatutoryMention(
                    act_name=canonical_act,
                    act_code=short_code,
                    sections=sections,
                    raw_mention=match.group(0)[:120],
                )

        # 2. Extract Constitutional Articles
        art_pattern = re.compile(r"(?i)\b(?:Articles?|Art\.?)\s+([\d\w\s,/\(\)\-\+and]+?)\s+(?:of\s+(?:the\s+)?)?Constitution(?:\s+of\s+India)?", re.MULTILINE)
        for match in art_pattern.finditer(text):
            arts = self._parse_section_numbers(match.group(1).strip())
            if arts:
                key = "Constitution"
                if key in statute_map:
                    for a in arts:
                        if a not in statute_map[key].sections:
                            statute_map[key].sections.append(a)
                else:
                    statute_map[key] = StatutoryMention(
                        act_name="Constitution of India",
                        act_code="Constitution",
                        sections=arts,
                        raw_mention=match.group(0)[:120],
                    )

        # Sort sections numerically where possible
        for mention in statute_map.values():
            mention.sections.sort(key=lambda s: (int(re.sub(r"\D", "", s)) if re.sub(r"\D", "", s) else 9999, s))

        return list(statute_map.values())

    def _resolve_act(self, raw_act: str) -> tuple[Optional[str], Optional[str]]:
        """Map raw act text to canonical name and code."""
        for pattern, canonical_name, code in KNOWN_ACTS:
            if re.search(pattern, raw_act, re.IGNORECASE):
                return canonical_name, code
        return None, None

    def _parse_section_numbers(self, raw_secs: str) -> list[str]:
        """Split and parse section tokens like '21, 23 and 29' or '439/482'."""
        cleaned = re.sub(r"(?i)\b(?:read\s+with|and|with)\b", ",", raw_secs)
        cleaned = cleaned.replace("/", ",").replace("&", ",")
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]

        valid_sections: list[str] = []
        for part in parts:
            # Check for alphanumeric section numbers like '439', '52A', '120B', '304-B', '498A'
            match = re.match(r"^(\d+[A-Za-z]?(?:-[A-Za-z])?|\([a-z\divx]+\))$", part)
            if match:
                valid_sections.append(match.group(1))
            else:
                # Extract first standalone numeric/alphanumeric word
                single = re.search(r"\b\d+[A-Za-z]?\b", part)
                if single:
                    valid_sections.append(single.group(0))

        return list(dict.fromkeys(valid_sections))

    def _extract_citations(self, text: str, neutral_citation: Optional[str]) -> list[ExtractedCitation]:
        """Extract reported legal citations and neutral citations."""
        citations: list[ExtractedCitation] = []
        seen: set[str] = set()

        if neutral_citation:
            citations.append(
                ExtractedCitation(
                    citation_text=neutral_citation,
                    reporter="Neutral Citation",
                    year=int(neutral_citation.split(":")[0]) if ":" in neutral_citation else None,
                )
            )
            seen.add(neutral_citation)

        for pat, reporter_name in CITATION_REGEXES:
            for m in re.finditer(pat, text):
                cit_str = m.group(0).strip()
                if cit_str not in seen:
                    year_val = None
                    year_match = re.search(r"\b(19\d\d|20\d\d)\b", cit_str)
                    if year_match:
                        year_val = int(year_match.group(1))

                    citations.append(
                        ExtractedCitation(
                            citation_text=cit_str,
                            reporter=reporter_name,
                            year=year_val,
                        )
                    )
                    seen.add(cit_str)

        return citations

    def _classify_counsel(self, counsel_list: list[str]) -> tuple[list[str], list[str]]:
        """Categorize counsels into Senior Advocates and Standing Counsels / Prosecutors."""
        senior_counsels: list[str] = []
        standing_counsels: list[str] = []

        for c in counsel_list:
            cleaned = re.sub(r"\s+", " ", c).strip()
            if re.search(r"(?i)\b(?:Senior\s+Advocate|Sr\.?\s*Adv(?:ocate)?)\b", cleaned):
                senior_counsels.append(cleaned)
            elif re.search(r"(?i)\b(?:Standing\s+Counsel|Special\s+Public\s+Prosecutor|APP|SPP|Public\s+Prosecutor)\b", cleaned):
                standing_counsels.append(cleaned)

        return senior_counsels, standing_counsels

    def enrich_document_dir(self, doc_dir: Path) -> EnrichedLegalMetadata:
        """Enrich a single judgment directory."""
        cleaned_json_path = doc_dir / "cleaned.json"
        if not cleaned_json_path.exists():
            raise FileNotFoundError(f"Missing cleaned.json in {doc_dir}. Run 'ingest clean' first.")

        data = json.loads(cleaned_json_path.read_text(encoding="utf-8"))
        cleaned_judgment = CleanedJudgment(**data)

        enriched = self.extract(cleaned_judgment)

        # Write enriched.json to doc_dir
        (doc_dir / "enriched.json").write_text(
            json.dumps(enriched.model_dump(mode="json"), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return enriched

    def enrich_store(
        self,
        store_dir: Path,
        output_jsonl: Optional[Path] = None,
    ) -> list[EnrichedLegalMetadata]:
        """Batch enrich all cleaned judgments in store."""
        cleaned_files = list(store_dir.glob("*/*/*/cleaned.json"))
        logger.info("Found %d cleaned judgments to enrich in %s", len(cleaned_files), store_dir)

        enriched_list: list[EnrichedLegalMetadata] = []
        for i, f in enumerate(cleaned_files, 1):
            try:
                res = self.enrich_document_dir(f.parent)
                enriched_list.append(res)
                if i % 25 == 0 or i == len(cleaned_files):
                    logger.info("Enriched %d/%d judgments...", i, len(cleaned_files))
            except Exception as e:
                logger.error("Error enriching %s: %s", f.parent, e)

        if output_jsonl and enriched_list:
            output_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(output_jsonl, "w", encoding="utf-8") as out:
                for e in enriched_list:
                    out.write(json.dumps(e.model_dump(mode="json"), ensure_ascii=False) + "\n")
            logger.info("Saved %d enriched judgments to %s", len(enriched_list), output_jsonl)

        return enriched_list
