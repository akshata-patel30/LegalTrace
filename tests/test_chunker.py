"""Unit tests for Step 6: Legal-Aware Semantic Chunking."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from ingestion_source.chunker import LegalSemanticChunker
from ingestion_source.cli import app
from ingestion_source.models import (
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

runner = CliRunner()


@pytest.fixture
def sample_structured_judgment() -> CleanedJudgment:
    paras = [
        CleanedParagraph(
            para_id=1,
            text="This is the cause title and procedural history of the bail petition.",
            para_type=ParagraphType.PREAMBLE,
            rhetorical_role=RhetoricalRole.PREAMBLE,
        ),
        CleanedParagraph(
            para_id=2,
            text="The complainant registered an FIR alleging fraud under Section 420 IPC on 12.01.2023.",
            para_type=ParagraphType.FACTS,
            rhetorical_role=RhetoricalRole.FACTS,
        ),
        CleanedParagraph(
            para_id=3,
            text="Further investigation revealed that bank accounts were opened in New Delhi.",
            para_type=ParagraphType.FACTS,
            rhetorical_role=RhetoricalRole.FACTS,
        ),
        CleanedParagraph(
            para_id=4,
            text="Learned senior counsel for the petitioner argued that no custodial interrogation is required.",
            para_type=ParagraphType.ARGUMENTS,
            rhetorical_role=RhetoricalRole.ARGUMENTS_PETITIONER,
        ),
        CleanedParagraph(
            para_id=5,
            text="On the other hand, learned APP for the State vehemently opposed the bail plea.",
            para_type=ParagraphType.ARGUMENTS,
            rhetorical_role=RhetoricalRole.ARGUMENTS_RESPONDENT,
        ),
        CleanedParagraph(
            para_id=6,
            text="In the case of Arnesh Kumar v. State of Bihar, the Supreme Court laid down arrest guidelines.",
            para_type=ParagraphType.CASE_QUOTE,
            is_quote=True,
            rhetorical_role=RhetoricalRole.PRECEDENTS,
        ),
        CleanedParagraph(
            para_id=7,
            text="In view of the above discussion, the applicant is directed to be released on regular bail.",
            para_type=ParagraphType.OPERATIVE,
            rhetorical_role=RhetoricalRole.OPERATIVE,
        ),
    ]

    header = CleanedHeader(
        court="Delhi High Court",
        docket_numbers=["BAIL APPLN. 999/2023"],
        bench=["Justice Amit Sharma"],
    )

    metrics = CleaningMetrics(
        raw_char_count=1200,
        cleaned_char_count=1000,
        noise_removed_chars=200,
        reduction_percentage=16.6,
        paragraphs_count=len(paras),
        disposition=DispositionType.ALLOWED,
    )

    return CleanedJudgment(
        doc_id="999001",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        title="Test Accused vs State on 15 March, 2023",
        header=header,
        paragraphs=paras,
        clean_full_text="\n\n".join(p.text for p in paras),
        metrics=metrics,
    )


@pytest.fixture
def sample_enriched_metadata() -> EnrichedLegalMetadata:
    return EnrichedLegalMetadata(
        doc_id="999001",
        case_type="Bail Application",
        case_type_code="BAIL APPLN.",
        legal_domain=LegalDomain.CRIMINAL,
        bench_type=BenchType.SINGLE_BENCH,
        canonical_judges=["Justice Amit Sharma"],
        statutes=[
            StatutoryMention(
                act_name="Indian Penal Code",
                act_code="IPC",
                sections=["420"],
                raw_mention="Section 420 IPC",
            ),
        ],
        citations=[
            ExtractedCitation(
                citation_text="2014 8 SCC 273",
                reporter="SCC",
                year=2014,
            ),
        ],
    )


def test_legal_chunker_role_boundary_isolation(
    sample_structured_judgment: CleanedJudgment,
    sample_enriched_metadata: EnrichedLegalMetadata,
):
    chunker = LegalSemanticChunker(target_words=500, max_words=1000)
    chunks = chunker.chunk_judgment(sample_structured_judgment, enriched=sample_enriched_metadata)

    assert len(chunks) == 6

    # 1. Preamble
    assert chunks[0].rhetorical_role == RhetoricalRole.PREAMBLE
    assert chunks[0].para_ids == [1]

    # 2. Facts
    assert chunks[1].rhetorical_role == RhetoricalRole.FACTS
    assert chunks[1].para_ids == [2, 3]

    # 3. Petitioner arguments
    assert chunks[2].rhetorical_role == RhetoricalRole.ARGUMENTS_PETITIONER
    assert chunks[2].para_ids == [4]

    # 4. Respondent arguments
    assert chunks[3].rhetorical_role == RhetoricalRole.ARGUMENTS_RESPONDENT
    assert chunks[3].para_ids == [5]

    # 5. Precedents
    assert chunks[4].rhetorical_role == RhetoricalRole.PRECEDENTS
    assert chunks[4].para_ids == [6]
    assert chunks[4].is_quote_block is True

    # 6. Operative
    assert chunks[5].rhetorical_role == RhetoricalRole.OPERATIVE
    assert chunks[5].para_ids == [7]


def test_legal_chunker_context_header_injection(
    sample_structured_judgment: CleanedJudgment,
    sample_enriched_metadata: EnrichedLegalMetadata,
):
    chunker = LegalSemanticChunker()
    chunks = chunker.chunk_judgment(sample_structured_judgment, enriched=sample_enriched_metadata)

    chunk_facts = chunks[1]
    assert "[Court: Delhi High Court" in chunk_facts.text_with_context
    assert "Docket: BAIL APPLN. 999/2023" in chunk_facts.text_with_context
    assert "Year: 2023" in chunk_facts.text_with_context
    assert "Judge: Justice Amit Sharma" in chunk_facts.text_with_context
    assert "Role: FACTS" in chunk_facts.text_with_context
    assert "Title: Test Accused vs State" in chunk_facts.text_with_context


def test_legal_chunker_oversized_paragraph_split():
    long_sentence = "This is an important legal finding of fact in the case. "
    huge_paragraph_text = long_sentence * 80  # ~800 words

    judgment = CleanedJudgment(
        doc_id="999002",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        title="Oversized Case",
        header=CleanedHeader(court="Delhi High Court"),
        paragraphs=[
            CleanedParagraph(
                para_id=1,
                text=huge_paragraph_text,
                rhetorical_role=RhetoricalRole.FACTS,
            )
        ],
        clean_full_text=huge_paragraph_text,
        metrics=CleaningMetrics(
            raw_char_count=len(huge_paragraph_text),
            cleaned_char_count=len(huge_paragraph_text),
            noise_removed_chars=0,
            reduction_percentage=0.0,
            paragraphs_count=1,
            disposition=DispositionType.OTHER,
        ),
    )

    chunker = LegalSemanticChunker(target_words=200, max_words=300)
    chunks = chunker.chunk_judgment(judgment)

    assert len(chunks) > 1
    for c in chunks:
        assert c.rhetorical_role == RhetoricalRole.FACTS
        assert c.para_ids == [1]
        assert c.word_count <= 350


def test_legal_chunker_statute_and_citation_linking(
    sample_structured_judgment: CleanedJudgment,
    sample_enriched_metadata: EnrichedLegalMetadata,
):
    chunker = LegalSemanticChunker()
    chunks = chunker.chunk_judgment(sample_structured_judgment, enriched=sample_enriched_metadata)

    facts_chunk = chunks[1]
    assert "IPC: Section 420" in facts_chunk.statute_mentions

    precedent_chunk = chunks[4]
    assert precedent_chunk.rhetorical_role == RhetoricalRole.PRECEDENTS


def test_legal_chunker_document_dir_and_cli(
    tmp_path: Path,
    sample_structured_judgment: CleanedJudgment,
    sample_enriched_metadata: EnrichedLegalMetadata,
):
    doc_dir = tmp_path / "delhi" / "2023" / "999001"
    doc_dir.mkdir(parents=True, exist_ok=True)

    cleaned_file = doc_dir / "cleaned.json"
    cleaned_file.write_text(
        json.dumps(sample_structured_judgment.model_dump(mode="json")),
        encoding="utf-8",
    )

    enriched_file = doc_dir / "enriched.json"
    enriched_file.write_text(
        json.dumps(sample_enriched_metadata.model_dump(mode="json")),
        encoding="utf-8",
    )

    chunker = LegalSemanticChunker()
    chunks = chunker.chunk_document_dir(doc_dir)

    assert len(chunks) == 6
    assert (doc_dir / "chunks.json").exists()

    # Test report generation
    report = chunker.generate_report(chunks, total_docs=1)
    assert report.total_chunks == 6
    assert report.total_documents == 1
    assert "facts" in report.role_distribution

    # Test CLI invocation
    output_jsonl = tmp_path / "test_chunks.jsonl"
    res = runner.invoke(app, [
        "chunk",
        "--store-dir", str(tmp_path),
        "--output", str(output_jsonl),
    ])
    assert res.exit_code == 0
    assert output_jsonl.exists()
    lines = output_jsonl.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 6
