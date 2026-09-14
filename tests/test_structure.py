"""Unit tests for LegalStructureClassifier (Step 5: Rhetorical Role Labeling)."""

import json
from pathlib import Path
import pytest

from ingestion_source.models import (
    CleanedHeader,
    CleanedJudgment,
    CleanedParagraph,
    CleaningMetrics,
    DispositionType,
    ParagraphType,
    RhetoricalRole,
)
from ingestion_source.structure import LegalStructureClassifier


@pytest.fixture
def classifier() -> LegalStructureClassifier:
    return LegalStructureClassifier()


def test_structure_headings_and_cues(classifier: LegalStructureClassifier) -> None:
    paras = [
        CleanedParagraph(para_id=1, label="1", text="The present petition under Section 482 Cr.P.C. seeks quashing of FIR.", para_type=ParagraphType.BODY),
        CleanedParagraph(para_id=2, label=None, text="FACTS OF THE CASE", para_type=ParagraphType.HEADING),
        CleanedParagraph(para_id=3, label="2", text="Briefly stated, the facts are that the complainant lodged an FIR.", para_type=ParagraphType.BODY),
        CleanedParagraph(para_id=4, label="3", text="The applicant was arrested on 12.05.2023.", para_type=ParagraphType.BODY),
        CleanedParagraph(para_id=5, label=None, text="SUBMISSIONS ON BEHALF OF THE PETITIONER", para_type=ParagraphType.HEADING),
        CleanedParagraph(para_id=6, label="4", text="Learned counsel for the petitioner submitted that no offence is made out.", para_type=ParagraphType.BODY),
        CleanedParagraph(para_id=7, label="5", text="It is further urged that the parties have settled the dispute.", para_type=ParagraphType.BODY),
        CleanedParagraph(para_id=8, label=None, text="FINDINGS AND ANALYSIS", para_type=ParagraphType.HEADING),
        CleanedParagraph(para_id=9, label="6", text="This Court has considered the rival submissions and perused the record.", para_type=ParagraphType.BODY),
        CleanedParagraph(para_id=10, label="7", text="The application is allowed and the petition stands disposed of.", para_type=ParagraphType.OPERATIVE),
    ]

    judgment = CleanedJudgment(
        doc_id="999",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        title="A vs B",
        header=CleanedHeader(court="Delhi High Court"),
        paragraphs=paras,
        clean_full_text="Test",
        metrics=CleaningMetrics(raw_char_count=100, cleaned_char_count=90, noise_removed_chars=10, reduction_percentage=10.0, paragraphs_count=10, disposition=DispositionType.ALLOWED),
    )

    classified = classifier.classify_judgment(judgment)
    roles = [p.rhetorical_role for p in classified.paragraphs]

    # Check mappings
    assert roles[0] == RhetoricalRole.PREAMBLE
    assert roles[1] == RhetoricalRole.FACTS  # Heading
    assert roles[2] == RhetoricalRole.FACTS  # Body under Facts
    assert roles[3] == RhetoricalRole.FACTS  # Inherited Facts
    assert roles[4] == RhetoricalRole.ARGUMENTS_PETITIONER  # Heading
    assert roles[5] == RhetoricalRole.ARGUMENTS_PETITIONER  # Cue
    assert roles[6] == RhetoricalRole.ARGUMENTS_PETITIONER  # Inherited
    assert roles[7] == RhetoricalRole.ANALYSIS  # Heading
    assert roles[8] == RhetoricalRole.ANALYSIS  # Cue
    assert roles[9] == RhetoricalRole.OPERATIVE  # Tail Operative


def test_structure_statutory_quote_and_precedents(classifier: LegalStructureClassifier) -> None:
    paras = [
        CleanedParagraph(para_id=1, label=None, text='"Section 482. Saving of inherent powers of High Court."', para_type=ParagraphType.STATUTORY_QUOTE, is_quote=True),
        CleanedParagraph(para_id=2, label=None, text='"In Gian Singh v. State of Punjab, the Supreme Court held that..."', para_type=ParagraphType.CASE_QUOTE, is_quote=True),
    ]

    judgment = CleanedJudgment(
        doc_id="888",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        title="X vs Y",
        header=CleanedHeader(court="Delhi High Court"),
        paragraphs=paras,
        clean_full_text="Test",
        metrics=CleaningMetrics(raw_char_count=100, cleaned_char_count=90, noise_removed_chars=10, reduction_percentage=10.0, paragraphs_count=2, disposition=DispositionType.OTHER),
    )

    classified = classifier.classify_judgment(judgment)
    assert classified.paragraphs[0].rhetorical_role == RhetoricalRole.STATUTE_REFERENCE
    assert classified.paragraphs[1].rhetorical_role == RhetoricalRole.PRECEDENTS


def test_structure_document_dir(tmp_path: Path, classifier: LegalStructureClassifier) -> None:
    doc_dir = tmp_path / "777"
    doc_dir.mkdir()

    paras = [
        {"para_id": 1, "label": "1", "text": "The petition is allowed.", "para_type": "operative", "is_quote": False, "char_count": 24, "word_count": 4}
    ]
    data = {
        "doc_id": "777",
        "court": "Delhi High Court",
        "court_slug": "delhi",
        "year": 2023,
        "title": "A vs B",
        "header": {"court": "Delhi High Court", "header_raw": ""},
        "paragraphs": paras,
        "clean_full_text": "The petition is allowed.",
        "metrics": {"raw_char_count": 30, "cleaned_char_count": 24, "noise_removed_chars": 6, "reduction_percentage": 20.0, "paragraphs_count": 1, "disposition": "allowed", "noise_patterns_detected": []},
    }
    (doc_dir / "cleaned.json").write_text(json.dumps(data), encoding="utf-8")

    res = classifier.structure_document_dir(doc_dir)
    assert len(res.paragraphs) == 1
    assert res.paragraphs[0].rhetorical_role == RhetoricalRole.OPERATIVE

    # Verify written back to cleaned.json
    saved = json.loads((doc_dir / "cleaned.json").read_text(encoding="utf-8"))
    assert saved["paragraphs"][0]["rhetorical_role"] == "operative"
