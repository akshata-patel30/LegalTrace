"""Unit tests for LegalDocumentCleaner (Step 3: Data Cleaning & Normalization)."""

import json
from pathlib import Path
import pytest

from ingestion_source.cleaner import LegalDocumentCleaner
from ingestion_source.models import CleanedJudgment, DispositionType, ParagraphType


@pytest.fixture
def cleaner() -> LegalDocumentCleaner:
    return LegalDocumentCleaner()


def test_unicode_and_quote_normalization(cleaner: LegalDocumentCleaner) -> None:
    # Test exotic quotes: curly, reverse, guillemets
    raw = "‟Special” and “Double” quotes, ‘Single’ and ‛other’ quotes, «French» quotes."
    res = cleaner._normalize_unicode_and_characters(raw)
    assert '“' not in res
    assert '”' not in res
    assert '‟' not in res
    assert '‘' not in res
    assert '’' not in res
    assert '"Special" and "Double" quotes' in res
    assert "'Single' and 'other' quotes" in res

    # Test non-breaking and zero-width spaces
    raw_spaces = "Word\u00a0one\u202fWord\u200btwo\u200cWord\xadthree"
    res_spaces = cleaner._normalize_unicode_and_characters(raw_spaces)
    assert res_spaces == "Word one Word twoWordthree"

    # Test dash normalization
    raw_dashes = "Section 439\u2014Bail\u2013Application\u22122023"
    res_dashes = cleaner._normalize_unicode_and_characters(raw_dashes)
    assert res_dashes == "Section 439-Bail-Application-2023"


def test_hyphenation_repair(cleaner: LegalDocumentCleaner) -> None:
    # Standard split word
    text = "The juris-\n diction of this Court under Article 226 is wide."
    repaired = cleaner._repair_hyphenation(text)
    assert "jurisdiction" in repaired
    assert "juris-" not in repaired

    # Compound legal prefix preservation
    compound = "The co-\n accused filed for bail alleging non-\n compliance of rules."
    repaired_compound = cleaner._repair_hyphenation(compound)
    assert "co-accused" in repaired_compound
    assert "non-compliance" in repaired_compound


def test_noise_removal(cleaner: LegalDocumentCleaner) -> None:
    sample_text = (
        "Indian Kanoon - http://indiankanoon.org/doc/12345678/\n"
        "Take printout of this judgment\n"
        "Click here to check corrigendum\n"
        "$~5 * IN THE HIGH COURT OF DELHI AT NEW DELHI\n"
        "% Reserved on: 12.05.2023\n"
        "1. This is an application for regular bail.\n"
        "- 1 -\n"
        "2. The applicant is in custody.\n"
        "Page 2 of 5\n"
        "The order be uploaded on the website of this Court forthwith.\n"
        "Signature Not Verified Digitally Signed By:JUSTICE ABC\n"
        "Signing Date:15.05.2023 18:30:00\n"
        "MAY 31, 2023/sn/bsr"
    )

    cleaned, detected = cleaner._strip_noise(sample_text)
    assert "Indian Kanoon" not in cleaned
    assert "http://indiankanoon.org" not in cleaned
    assert "Take printout" not in cleaned
    assert "Click here to check corrigendum" not in cleaned
    assert "$~5 *" not in cleaned
    assert "% Reserved on:" not in cleaned
    assert "- 1 -" not in cleaned
    assert "Page 2 of 5" not in cleaned
    assert "Signature Not Verified" not in cleaned
    assert "Digitally Signed By" not in cleaned
    assert "The order be uploaded on the website" not in cleaned
    assert "/sn/bsr" not in cleaned
    assert "MAY 31, 2023" in cleaned

    assert "kanoon_url" in detected
    assert "printout_boilerplate" in detected
    assert "digital_signature_block" in detected


def test_header_and_body_extraction(cleaner: LegalDocumentCleaner) -> None:
    doc_text = (
        "IN THE HIGH COURT OF DELHI AT NEW DELHI\n"
        "NEUTRAL CITATION NO: 2023:DHC:3842\n"
        "BAIL APPLN. 1234/2023\n"
        "RAMESH KUMAR ..... Petitioner versus STATE (NCT OF DELHI) ..... Respondent\n"
        "Date of decision: 24.05.2023\n\n"
        "JUDGMENT\n\n"
        "1. This application under Section 439 Cr.P.C. is filed by the petitioner.\n\n"
        "2. The prosecution case is that contraband was recovered."
    )

    header, body = cleaner._extract_header_and_body(doc_text, "Delhi High Court")
    assert header.neutral_citation == "2023:DHC:3842"
    assert header.decision_date == "24.05.2023"
    assert "BAIL APPLN. 1234/2023" in header.docket_numbers
    assert header.petitioner == "RAMESH KUMAR"
    assert header.respondent == "STATE (NCT OF DELHI)"
    assert "1. This application under Section 439" in body


def test_paragraph_segmentation_and_classification(cleaner: LegalDocumentCleaner) -> None:
    body = (
        "FACTS AND CONTENTIONS\n\n"
        "1. The petitioner is seeking bail in FIR No. 12/2023.\n\n"
        '"Section 439. Special powers of High Court or Court of Session regarding bail."\n\n'
        '"The court must satisfy itself that reasonable grounds exist for granting bail."\n\n'
        "5. The application is allowed and bail is granted to the applicant on furnishing personal bond."
    )

    paragraphs = cleaner._segment_paragraphs(body)
    assert len(paragraphs) == 5

    # Paragraph 1: Heading
    assert paragraphs[0].para_type == ParagraphType.HEADING
    assert paragraphs[0].label is None

    # Paragraph 2: Numbered body
    assert paragraphs[1].label == "1"
    assert paragraphs[1].para_type == ParagraphType.BODY
    assert "petitioner is seeking bail" in paragraphs[1].text

    # Paragraph 3: Statutory Quote
    assert paragraphs[2].para_type == ParagraphType.STATUTORY_QUOTE
    assert paragraphs[2].is_quote is True

    # Paragraph 4: Case / Block Quote
    assert paragraphs[3].para_type == ParagraphType.CASE_QUOTE
    assert paragraphs[3].is_quote is True

    # Paragraph 5: Operative
    assert paragraphs[4].label == "5"
    assert paragraphs[4].para_type == ParagraphType.OPERATIVE


def test_disposition_classification_allowed(cleaner: LegalDocumentCleaner) -> None:
    body = (
        "1. Having considered all facts, the applicant has made out a case.\n\n"
        "2. The application is allowed. Bail is granted on the following conditions:\n"
        "(i) Furnish PR bond of Rs. 25,000."
    )
    paras = cleaner._segment_paragraphs(body)
    op_text, disp = cleaner._extract_disposition_and_operative(paras, body)
    assert disp == DispositionType.ALLOWED
    assert op_text is not None
    assert "application is allowed" in op_text


def test_disposition_classification_dismissed(cleaner: LegalDocumentCleaner) -> None:
    body = (
        "1. The allegations are serious in nature.\n\n"
        "2. Finding no merit in the present petition, the same is hereby dismissed."
    )
    paras = cleaner._segment_paragraphs(body)
    _, disp = cleaner._extract_disposition_and_operative(paras, body)
    assert disp == DispositionType.DISMISSED


def test_disposition_classification_referred(cleaner: LegalDocumentCleaner) -> None:
    body = (
        "1. In view of conflicting decisions of coordinate benches, this issue requires authoritative determination.\n\n"
        "2. The registry is directed to place before Hon'ble the Chief Justice for constitution of a larger bench."
    )
    paras = cleaner._segment_paragraphs(body)
    _, disp = cleaner._extract_disposition_and_operative(paras, body)
    assert disp == DispositionType.REFERRED


def test_clean_document_dir_and_clean_store(tmp_path: Path, cleaner: LegalDocumentCleaner) -> None:
    # Setup mock raw-store hierarchy: court/year/doc_id/full_text.txt
    court_dir = tmp_path / "delhi_high_court" / "2023" / "123456"
    court_dir.mkdir(parents=True)

    raw_text = (
        "Indian Kanoon - http://indiankanoon.org/doc/123456/\n"
        "IN THE HIGH COURT OF DELHI AT NEW DELHI\n"
        "NEUTRAL CITATION NO: 2023:DHC:1111\n"
        "CRL.M.C. 100/2023\n"
        "A ..... Petitioner versus B ..... Respondent\n\n"
        "JUDGMENT\n\n"
        "1. The present petition under Section 482 Cr.P.C. seeks quashing.\n\n"
        "2. In view of the settle-\n ment between parties, petition is allowed."
    )
    (court_dir / "full_text.txt").write_text(raw_text, encoding="utf-8")
    meta = {
        "doc_id": "123456",
        "court": "Delhi High Court",
        "court_slug": "delhi_high_court",
        "year": 2023,
        "title": "A vs B",
    }
    (court_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

    # Clean single doc dir
    cleaned = cleaner.clean_document_dir(court_dir)
    assert isinstance(cleaned, CleanedJudgment)
    assert cleaned.doc_id == "123456"
    assert cleaned.header.neutral_citation == "2023:DHC:1111"
    assert cleaned.disposition == DispositionType.ALLOWED
    assert (court_dir / "cleaned_text.txt").exists()
    assert (court_dir / "cleaned.json").exists()
    assert "Indian Kanoon" not in (court_dir / "cleaned_text.txt").read_text(encoding="utf-8")
    assert "settlement" in (court_dir / "cleaned_text.txt").read_text(encoding="utf-8")

    # Batch clean store
    output_jsonl = tmp_path / "cleaned_dataset.jsonl"
    cleaned_batch = cleaner.clean_store(tmp_path, output_jsonl=output_jsonl)
    assert len(cleaned_batch) == 1
    assert output_jsonl.exists()
    lines = [json.loads(line) for line in output_jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["doc_id"] == "123456"
    assert lines[0]["disposition"] == "allowed"
