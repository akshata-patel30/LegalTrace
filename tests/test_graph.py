"""Unit tests for Step 7: Precedent Citation Graph & Cross-Document Linker."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from ingestion_source.cli import app
from ingestion_source.graph import (
    CitationGraphBuilder,
    normalize_citation_key,
)
from ingestion_source.models import (
    CitationTreatment,
    CleanedHeader,
    CleanedJudgment,
    CleanedParagraph,
    CleaningMetrics,
    DispositionType,
    EnrichedLegalMetadata,
    ExtractedCitation,
    ParagraphType,
    RhetoricalRole,
)

runner = CliRunner()


def test_citation_key_normalization():
    assert normalize_citation_key("(2014) 8 SCC 273") == "2014_8_scc_273"
    assert normalize_citation_key("2023:DHC:1234") == "2023_dhc_1234"
    assert normalize_citation_key("AIR 1980 SC 485") == "air_1980_sc_485"


def test_treatment_classification():
    builder = CitationGraphBuilder()

    s1 = "In the present case, reliance is placed on the judgment in (2014) 8 SCC 273."
    assert builder._classify_treatment(s1) == CitationTreatment.RELIED_UPON

    s2 = "However, that decision is clearly distinguishable on facts and does not apply here."
    assert builder._classify_treatment(s2) == CitationTreatment.DISTINGUISHED

    s3 = "Following the ratio laid down by the Hon'ble Supreme Court, bail is granted."
    assert builder._classify_treatment(s3) == CitationTreatment.FOLLOWED

    s4 = "The earlier view stands overruled and is no longer good law."
    assert builder._classify_treatment(s4) == CitationTreatment.OVERRULED

    s5 = "Reference was made to the decision in State of Haryana v. Bhajan Lal."
    assert builder._classify_treatment(s5) == CitationTreatment.REFERRED

    s6 = "The court perused the record and noted the date."
    assert builder._classify_treatment(s6) == CitationTreatment.CITED


def test_precedent_title_detection():
    builder = CitationGraphBuilder()
    context = "Reliance was placed on Arnesh Kumar v. State of Bihar, (2014) 8 SCC 273 by the counsel."
    title = builder._detect_precedent_title(context)
    assert title is not None
    assert "Arnesh Kumar" in title
    assert "State of Bihar" in title


def test_extract_document_edges():
    paras = [
        CleanedParagraph(
            para_id=1,
            text="The learned counsel placed reliance on the decision in (2014) 8 SCC 273.",
            para_type=ParagraphType.ARGUMENTS,
            rhetorical_role=RhetoricalRole.ARGUMENTS_PETITIONER,
        ),
        CleanedParagraph(
            para_id=2,
            text="In the case of Satender Kumar Antil v. CBI, 2022 SCC OnLine SC 929, the apex court held that bail is the rule.",
            para_type=ParagraphType.CASE_QUOTE,
            rhetorical_role=RhetoricalRole.PRECEDENTS,
        ),
    ]

    judgment = CleanedJudgment(
        doc_id="888001",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        title="Applicant vs State on 10 May, 2023",
        header=CleanedHeader(court="Delhi High Court", bench=["Justice Amit Sharma"]),
        paragraphs=paras,
        clean_full_text="\n\n".join(p.text for p in paras),
        metrics=CleaningMetrics(
            raw_char_count=500,
            cleaned_char_count=500,
            noise_removed_chars=0,
            reduction_percentage=0.0,
            paragraphs_count=2,
            disposition=DispositionType.ALLOWED,
        ),
    )

    enriched = EnrichedLegalMetadata(
        doc_id="888001",
        case_type="Bail Application",
        case_type_code="BAIL APPLN.",
        legal_domain="criminal",  # type: ignore
        bench_type="single_bench",  # type: ignore
        canonical_judges=["Justice Amit Sharma"],
        citations=[
            ExtractedCitation(citation_text="(2014) 8 SCC 273", reporter="SCC", year=2014),
            ExtractedCitation(citation_text="2022 SCC OnLine SC 929", reporter="SCC OnLine", year=2022),
        ],
    )

    builder = CitationGraphBuilder()
    edges = builder.extract_document_edges(judgment, enriched=enriched)

    assert len(edges) == 2

    e1 = edges[0]
    assert e1.cited_citation == "(2014) 8 SCC 273"
    assert e1.rhetorical_role == RhetoricalRole.ARGUMENTS_PETITIONER
    assert e1.treatment == CitationTreatment.RELIED_UPON
    assert e1.citing_judge == "Justice Amit Sharma"

    e2 = edges[1]
    assert e2.cited_citation == "2022 SCC OnLine SC 929"
    assert e2.rhetorical_role == RhetoricalRole.PRECEDENTS
    assert e2.cited_title is not None
    assert "Satender Kumar Antil" in e2.cited_title


def test_cross_document_resolution_and_build_graph(tmp_path: Path):
    doc1_dir = tmp_path / "delhi" / "2023" / "888001"
    doc2_dir = tmp_path / "delhi" / "2023" / "888002"
    doc1_dir.mkdir(parents=True, exist_ok=True)
    doc2_dir.mkdir(parents=True, exist_ok=True)

    # Document 2 (has neutral citation 2023:DHC:9999)
    j2 = CleanedJudgment(
        doc_id="888002",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        title="Prior Case on 01 Jan, 2023",
        header=CleanedHeader(court="Delhi High Court", neutral_citation="2023:DHC:9999"),
        paragraphs=[CleanedParagraph(para_id=1, text="Judgment content.")],
        clean_full_text="Judgment content.",
        metrics=CleaningMetrics(
            raw_char_count=100, cleaned_char_count=100, noise_removed_chars=0,
            reduction_percentage=0.0, paragraphs_count=1, disposition=DispositionType.OTHER
        ),
    )
    (doc2_dir / "cleaned.json").write_text(json.dumps(j2.model_dump(mode="json")), encoding="utf-8")

    # Document 1 (cites Document 2's neutral citation!)
    j1 = CleanedJudgment(
        doc_id="888001",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        title="Subsequent Case on 15 Feb, 2023",
        header=CleanedHeader(court="Delhi High Court"),
        paragraphs=[
            CleanedParagraph(
                para_id=1,
                text="The coordinate bench in 2023:DHC:9999 took the view that notice is required.",
                rhetorical_role=RhetoricalRole.ANALYSIS,
            )
        ],
        clean_full_text="The coordinate bench in 2023:DHC:9999 took the view that notice is required.",
        metrics=CleaningMetrics(
            raw_char_count=100, cleaned_char_count=100, noise_removed_chars=0,
            reduction_percentage=0.0, paragraphs_count=1, disposition=DispositionType.OTHER
        ),
    )
    (doc1_dir / "cleaned.json").write_text(json.dumps(j1.model_dump(mode="json")), encoding="utf-8")

    en1 = EnrichedLegalMetadata(
        doc_id="888001",
        case_type="Writ",
        case_type_code="WP",
        legal_domain="civil",  # type: ignore
        bench_type="single_bench",  # type: ignore
        citations=[ExtractedCitation(citation_text="2023:DHC:9999", reporter="Neutral Citation", year=2023)],
    )
    (doc1_dir / "enriched.json").write_text(json.dumps(en1.model_dump(mode="json")), encoding="utf-8")

    # Build graph
    builder = CitationGraphBuilder(store_dir=tmp_path)
    edges_out = tmp_path / "edges.jsonl"
    nodes_out = tmp_path / "nodes.jsonl"
    d3_out = tmp_path / "d3.json"
    gml_out = tmp_path / "graph.graphml"

    edges, nodes, summary = builder.build_graph_from_store(
        store_dir=tmp_path,
        output_edges_jsonl=edges_out,
        output_nodes_jsonl=nodes_out,
        output_graph_json=d3_out,
        output_graphml=gml_out,
    )

    assert len(edges) == 1
    edge = edges[0]
    assert edge.cited_citation == "2023:DHC:9999"
    assert edge.is_internal_corpus_match is True
    assert edge.target_doc_id == "888002"

    assert len(nodes) == 1
    node = nodes[0]
    assert node.in_degree == 1
    assert node.is_internal is True
    assert node.doc_id == "888002"

    assert edges_out.exists()
    assert nodes_out.exists()
    assert d3_out.exists()
    assert gml_out.exists()

    # Test CLI invocation
    res = runner.invoke(app, ["graph", "--store-dir", str(tmp_path)])
    assert res.exit_code == 0
    assert "Precedent Citation Network Summary" in res.stdout
