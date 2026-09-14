"""Unit tests for Step 8: Automated Headnote & Executive Summary Generation."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from ingestion_source.cli import app
from ingestion_source.models import (
    CitationEdge,
    CitationTreatment,
    CleanedHeader,
    CleanedJudgment,
    CleanedParagraph,
    CleaningMetrics,
    DispositionType,
    EnrichedLegalMetadata,
    ExtractedCitation,
    LegalDomain,
    ParagraphType,
    RhetoricalRole,
    StatutoryMention,
    BenchType,
)
from ingestion_source.summarizer import LegalHeadnoteGenerator

runner = CliRunner()


@pytest.fixture
def sample_structured_judgment() -> CleanedJudgment:
    paras = [
        CleanedParagraph(
            para_id=1,
            text="The present bail application seeks regular bail in FIR No. 12/2023 under Sections 20/23 NDPS Act.",
            para_type=ParagraphType.PREAMBLE,
            rhetorical_role=RhetoricalRole.PREAMBLE,
        ),
        CleanedParagraph(
            para_id=2,
            text="The prosecution case is that customs officials intercepted the applicant at IGI Airport with 5 kg contraband.",
            para_type=ParagraphType.FACTS,
            rhetorical_role=RhetoricalRole.FACTS,
        ),
        CleanedParagraph(
            para_id=3,
            text="The core question that arises for consideration is whether Section 37 bar applies when sampling was non-compliant.",
            para_type=ParagraphType.BODY,
            rhetorical_role=RhetoricalRole.ISSUE,
        ),
        CleanedParagraph(
            para_id=4,
            text="Learned senior counsel for the petitioner submitted that delay in trial entitles the applicant to bail.",
            para_type=ParagraphType.ARGUMENTS,
            rhetorical_role=RhetoricalRole.ARGUMENTS_PETITIONER,
        ),
        CleanedParagraph(
            para_id=5,
            text="Learned APP for the State vehemently opposed the bail contending commercial quantity recovery.",
            para_type=ParagraphType.ARGUMENTS,
            rhetorical_role=RhetoricalRole.ARGUMENTS_RESPONDENT,
        ),
        CleanedParagraph(
            para_id=6,
            text="In the considered opinion of this court, prolonged incarceration without trial violates Article 21 rights.",
            para_type=ParagraphType.ANALYSIS,
            rhetorical_role=RhetoricalRole.ANALYSIS,
        ),
        CleanedParagraph(
            para_id=7,
            text="The application is accordingly allowed. The applicant is admitted to bail on furnishing personal bond.",
            para_type=ParagraphType.OPERATIVE,
            rhetorical_role=RhetoricalRole.OPERATIVE,
        ),
    ]

    header = CleanedHeader(
        court="Delhi High Court",
        docket_numbers=["BAIL APPLN. 101/2023"],
        decision_date="15th April, 2023",
        bench=["Justice Amit Sharma"],
    )

    metrics = CleaningMetrics(
        raw_char_count=1200,
        cleaned_char_count=1100,
        noise_removed_chars=100,
        reduction_percentage=8.3,
        paragraphs_count=len(paras),
        disposition=DispositionType.ALLOWED,
    )

    return CleanedJudgment(
        doc_id="777001",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        title="Applicant vs Customs on 15 April, 2023",
        header=header,
        paragraphs=paras,
        clean_full_text="\n\n".join(p.text for p in paras),
        disposition=DispositionType.ALLOWED,
        metrics=metrics,
    )


@pytest.fixture
def sample_enriched_metadata() -> EnrichedLegalMetadata:
    return EnrichedLegalMetadata(
        doc_id="777001",
        case_type="Bail Application",
        case_type_code="BAIL APPLN.",
        legal_domain=LegalDomain.CRIMINAL,
        bench_type=BenchType.SINGLE_BENCH,
        canonical_judges=["Justice Amit Sharma"],
        statutes=[
            StatutoryMention(
                act_name="Narcotic Drugs and Psychotropic Substances Act, 1985",
                act_code="NDPS Act",
                sections=["20", "23", "37"],
                raw_mention="Sections 20, 23 NDPS Act",
            )
        ],
        citations=[
            ExtractedCitation(citation_text="(2016) 3 SCC 379", reporter="SCC", year=2016)
        ],
    )


def test_catchwords_construction(
    sample_structured_judgment: CleanedJudgment,
    sample_enriched_metadata: EnrichedLegalMetadata,
):
    generator = LegalHeadnoteGenerator()
    headnote = generator.generate_headnote(
        sample_structured_judgment,
        enriched=sample_enriched_metadata,
    )

    assert "CRIMINAL" in headnote.catchwords_line
    assert "Bail Application" in headnote.catchwords_line
    assert "NDPS Act, Ss. 20, 23, 37" in headnote.catchwords_line
    assert "Outcome: ALLOWED" in headnote.catchwords_line


def test_brief_facts_and_issues_extraction(
    sample_structured_judgment: CleanedJudgment,
    sample_enriched_metadata: EnrichedLegalMetadata,
):
    generator = LegalHeadnoteGenerator()
    headnote = generator.generate_headnote(
        sample_structured_judgment,
        enriched=sample_enriched_metadata,
    )

    assert "customs officials intercepted the applicant" in headnote.brief_facts
    assert len(headnote.issues) >= 1
    assert "Section 37 bar applies" in headnote.issues[0]


def test_party_arguments_and_ratio_extraction(
    sample_structured_judgment: CleanedJudgment,
    sample_enriched_metadata: EnrichedLegalMetadata,
):
    generator = LegalHeadnoteGenerator()
    headnote = generator.generate_headnote(
        sample_structured_judgment,
        enriched=sample_enriched_metadata,
    )

    assert len(headnote.petitioner_arguments) >= 1
    assert "delay in trial" in headnote.petitioner_arguments[0]

    assert len(headnote.respondent_arguments) >= 1
    assert "vehemently opposed" in headnote.respondent_arguments[0]

    assert "prolonged incarceration without trial violates Article 21" in headnote.ratio_decidendi


def test_operative_holding_and_formatting(
    sample_structured_judgment: CleanedJudgment,
    sample_enriched_metadata: EnrichedLegalMetadata,
):
    generator = LegalHeadnoteGenerator()
    headnote = generator.generate_headnote(
        sample_structured_judgment,
        enriched=sample_enriched_metadata,
    )

    assert "application is accordingly allowed" in headnote.operative_holding
    assert "[2023] DELHI HIGH COURT" in headnote.formatted_headnote
    assert "CATCHWORDS:" in headnote.formatted_headnote
    assert "BRIEF FACTS:" in headnote.formatted_headnote
    assert "RATIO DECIDENDI:" in headnote.formatted_headnote
    assert "DISPOSITION & OPERATIVE HOLDING:" in headnote.formatted_headnote
    assert headnote.word_count > 50


def test_headnote_document_dir_and_cli(
    tmp_path: Path,
    sample_structured_judgment: CleanedJudgment,
    sample_enriched_metadata: EnrichedLegalMetadata,
):
    doc_dir = tmp_path / "delhi" / "2023" / "777001"
    doc_dir.mkdir(parents=True, exist_ok=True)

    cleaned_file = doc_dir / "cleaned.json"
    cleaned_file.write_text(json.dumps(sample_structured_judgment.model_dump(mode="json")), encoding="utf-8")

    enriched_file = doc_dir / "enriched.json"
    enriched_file.write_text(json.dumps(sample_enriched_metadata.model_dump(mode="json")), encoding="utf-8")

    generator = LegalHeadnoteGenerator()
    hn = generator.generate_for_document_dir(doc_dir)

    assert (doc_dir / "headnote.json").exists()
    assert (doc_dir / "headnote.txt").exists()
    assert hn.doc_id == "777001"

    # Test report
    report = generator.generate_report([hn])
    assert report.total_documents == 1
    assert report.documents_with_facts == 1
    assert report.documents_with_holding == 1

    # Test CLI invocation
    output_jsonl = tmp_path / "test_headnotes.jsonl"
    res = runner.invoke(app, [
        "headnote",
        "--store-dir", str(tmp_path),
        "--output", str(output_jsonl),
    ])
    assert res.exit_code == 0
    assert output_jsonl.exists()
    lines = output_jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
