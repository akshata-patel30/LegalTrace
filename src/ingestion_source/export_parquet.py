"""Export canonical ingested judgments into LegalTrace Parquet format."""

from collections import Counter
import json
from pathlib import Path
import sys
import pyarrow as pa
import pyarrow.parquet as pq

def export_to_legaltrace_parquet(jsonl_path: Path, output_parquet: Path, start_id: int = 9000000) -> int:
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL {jsonl_path} does not exist.")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rows = []
    for idx, r in enumerate(records):
        doc_id = str(r.get("doc_id", ""))
        year = int(r.get("year") or 2023)
        court = str(r.get("court") or "")
        court_slug = str(r.get("court_slug") or "court")
        title = str(r.get("title") or "")

        is_cleaned = "clean_full_text" in r
        if is_cleaned:
            header = r.get("header") or {}
            petitioner = header.get("petitioner")
            respondent = header.get("respondent")
            bench = header.get("bench") or []
            bench_list = [str(b) for b in bench] if isinstance(bench, list) else []
            author = bench_list[0] if bench_list else None
            dockets = header.get("docket_numbers") or []
            docket = ", ".join(dockets) if dockets else None
            date_str = header.get("decision_date")
            neutral_cit = header.get("neutral_citation")
            full_text = str(r.get("clean_full_text") or "")
            headnote = header.get("header_raw")
            disposition = str(r.get("disposition") or "")
            content_hash = doc_id
            fetched_at = str(r.get("cleaned_at") or "")
            source_url = f"https://indiankanoon.org/doc/{doc_id}/"

            # Check for Step 4 enriched metadata
            doc_dir = Path("raw-store") / court_slug / str(year) / doc_id
            enriched_file = doc_dir / "enriched.json"
            enriched_data = json.loads(enriched_file.read_text(encoding="utf-8")) if enriched_file.exists() else {}

            law_report_cit = None
            if enriched_data.get("citations"):
                rep_cits = [c["citation_text"] for c in enriched_data["citations"] if c.get("reporter") != "Neutral Citation"]
                if rep_cits:
                    law_report_cit = rep_cits[0]

            if enriched_data.get("canonical_judges"):
                bench_list = enriched_data["canonical_judges"]
                author = bench_list[0] if bench_list else author

            parser_info = {
                "parser": "legal_document_cleaner_and_enricher",
                "cleaning_metrics": r.get("metrics") or {},
                "paragraphs_count": len(r.get("paragraphs") or []),
                "case_type": enriched_data.get("case_type"),
                "legal_domain": enriched_data.get("legal_domain"),
                "bench_type": enriched_data.get("bench_type"),
                "statutes": enriched_data.get("statutes", []),
                "total_statutes_count": enriched_data.get("total_statutes_count", 0),
                "total_citations_count": enriched_data.get("total_citations_count", 0),
                "rhetorical_roles": dict(Counter(p.get("rhetorical_role") for p in r.get("paragraphs") or [] if p.get("rhetorical_role"))),
            }
            source_filename = "cleaned.json"
            source_rel_path = f"year={year}/court={court_slug}/{doc_id}/cleaned.json"
            quality_completeness = {
                "petitioner": bool(petitioner),
                "respondent": bool(respondent),
                "date": bool(date_str),
                "neutral_citation": bool(neutral_cit),
                "disposition": bool(disposition),
                "case_type": bool(enriched_data.get("case_type")),
                "statutes": bool(enriched_data.get("statutes")),
            }
        else:
            parties = r.get("parties") or {}
            petitioner = parties.get("petitioner") if isinstance(parties, dict) else None
            respondent = parties.get("respondent") if isinstance(parties, dict) else None
            bench = r.get("bench") or []
            bench_list = [str(b) for b in bench] if isinstance(bench, list) else []
            author = str(r.get("author") or "") if r.get("author") else (bench_list[0] if bench_list else None)
            case_numbers = r.get("case_numbers") or []
            docket = ", ".join(case_numbers) if isinstance(case_numbers, list) and case_numbers else None
            date_str = str(r.get("date")) if r.get("date") else None
            neutral_cit = r.get("neutral_citation")
            full_text = str(r.get("full_text") or "")
            headnote = None
            disposition = None
            content_hash = str(r.get("content_hash") or "")
            fetched_at = str(r.get("fetched_at") or "")
            source_url = str(r.get("source_url") or "")
            citation_metrics = r.get("citation_metrics") or {}
            paragraphs = r.get("paragraphs") or []
            parser_info = {
                "parser": "scrapling_dynamic_normalizer",
                "paragraphs_count": len(paragraphs),
                "cites_count": citation_metrics.get("cites_count", 0),
                "cited_by_count": citation_metrics.get("cited_by_count", 0),
            }
            source_filename = "judgment.json"
            source_rel_path = f"year={year}/court={court_slug}/{doc_id}/judgment.json"
            quality_completeness = {
                "petitioner": bool(petitioner),
                "respondent": bool(respondent),
                "bench": bool(bench_list),
                "date": bool(date_str),
                "neutral_citation": bool(neutral_cit),
            }

        row = {
            "id": start_id + idx,
            "case_metadata_id": f"case_meta:ik:{year}:{court_slug}:{doc_id}",
            "dataset_source": "indiankanoon",
            "parser_record_id": content_hash,
            "ingestion_split": "all",
            "case_title": title,
            "party_petitioner": petitioner,
            "party_respondent": respondent,
            "party_caption": f"{petitioner} versus {respondent}" if petitioner and respondent else title,
            "docket_number": docket,
            "cnr_number": None,
            "neutral_citation": neutral_cit,
            "law_report_citation": law_report_cit if is_cleaned else None,
            "court_name": court,
            "court_code": court_slug,
            "bench_name": court_slug,
            "presiding_judge": author,
            "coram_members": bench_list,
            "coram_members_text": "; ".join(bench_list) if bench_list else "",
            "decision_date": date_str,
            "registration_date": None,
            "citation_year": year,
            "decision_year": year,
            "disposition_text": disposition,
            "source_relative_path": source_rel_path,
            "source_path_year": str(year),
            "source_path_court_code": court_slug,
            "source_path_bench": court_slug,
            "source_filename": source_filename,
            "source_json_s3_url": source_url,
            "source_pdf_s3_url": None,
            "language_codes": ["en"],
            "indexable_text": full_text,
            "headnote_text": headnote,
            "normalized_record_json": json.dumps(r, default=str),
            "parser_json": json.dumps(parser_info),
            "quality_json": json.dumps({"completeness": quality_completeness}),
            "created_at": fetched_at,
            "updated_at": fetched_at,
        }
        rows.append(row)

    # Convert to PyArrow Table
    schema = pa.schema([
        ("id", pa.int64()),
        ("case_metadata_id", pa.string()),
        ("dataset_source", pa.string()),
        ("parser_record_id", pa.string()),
        ("ingestion_split", pa.string()),
        ("case_title", pa.string()),
        ("party_petitioner", pa.string()),
        ("party_respondent", pa.string()),
        ("party_caption", pa.string()),
        ("docket_number", pa.string()),
        ("cnr_number", pa.string()),
        ("neutral_citation", pa.string()),
        ("law_report_citation", pa.string()),
        ("court_name", pa.string()),
        ("court_code", pa.string()),
        ("bench_name", pa.string()),
        ("presiding_judge", pa.string()),
        ("coram_members", pa.list_(pa.string())),
        ("coram_members_text", pa.string()),
        ("decision_date", pa.string()),
        ("registration_date", pa.string()),
        ("citation_year", pa.int32()),
        ("decision_year", pa.int32()),
        ("disposition_text", pa.string()),
        ("source_relative_path", pa.string()),
        ("source_path_year", pa.string()),
        ("source_path_court_code", pa.string()),
        ("source_path_bench", pa.string()),
        ("source_filename", pa.string()),
        ("source_json_s3_url", pa.string()),
        ("source_pdf_s3_url", pa.string()),
        ("language_codes", pa.list_(pa.string())),
        ("indexable_text", pa.string()),
        ("headnote_text", pa.string()),
        ("normalized_record_json", pa.string()),
        ("parser_json", pa.string()),
        ("quality_json", pa.string()),
        ("created_at", pa.string()),
        ("updated_at", pa.string()),
    ])

    table = pa.Table.from_pylist(rows, schema=schema)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_parquet, compression="snappy")
    print(f"[OK] Successfully exported {len(rows)} judgments to {output_parquet}")
    return len(rows)


def export_chunks_to_parquet(jsonl_path: Path, output_parquet: Path) -> int:
    """Export semantic legal chunks to high-performance Parquet format."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL {jsonl_path} does not exist.")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rows = []
    for r in records:
        rows.append({
            "chunk_id": str(r.get("chunk_id", "")),
            "doc_id": str(r.get("doc_id", "")),
            "chunk_index": int(r.get("chunk_index", 0)),
            "court": str(r.get("court", "")),
            "court_slug": str(r.get("court_slug", "")),
            "year": int(r.get("year", 2023)),
            "case_title": str(r.get("case_title", "")),
            "docket_number": r.get("docket_number"),
            "presiding_judge": r.get("presiding_judge"),
            "legal_domain": r.get("legal_domain"),
            "case_type": r.get("case_type"),
            "rhetorical_role": str(r.get("rhetorical_role", "")),
            "para_ids": [int(p) for p in r.get("para_ids", [])],
            "text": str(r.get("text", "")),
            "text_with_context": str(r.get("text_with_context", "")),
            "char_count": int(r.get("char_count", 0)),
            "word_count": int(r.get("word_count", 0)),
            "statute_mentions": [str(s) for s in r.get("statute_mentions", [])],
            "citations": [str(c) for c in r.get("citations", [])],
            "is_quote_block": bool(r.get("is_quote_block", False)),
        })

    schema = pa.schema([
        ("chunk_id", pa.string()),
        ("doc_id", pa.string()),
        ("chunk_index", pa.int32()),
        ("court", pa.string()),
        ("court_slug", pa.string()),
        ("year", pa.int32()),
        ("case_title", pa.string()),
        ("docket_number", pa.string()),
        ("presiding_judge", pa.string()),
        ("legal_domain", pa.string()),
        ("case_type", pa.string()),
        ("rhetorical_role", pa.string()),
        ("para_ids", pa.list_(pa.int32())),
        ("text", pa.string()),
        ("text_with_context", pa.string()),
        ("char_count", pa.int32()),
        ("word_count", pa.int32()),
        ("statute_mentions", pa.list_(pa.string())),
        ("citations", pa.list_(pa.string())),
        ("is_quote_block", pa.bool_()),
    ])

    table = pa.Table.from_pylist(rows, schema=schema)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_parquet, compression="snappy")
    print(f"[OK] Successfully exported {len(rows)} legal chunks to {output_parquet}")
    return len(rows)


def export_citation_edges_to_parquet(jsonl_path: Path, output_parquet: Path) -> int:
    """Export directed precedent citation edges to Parquet."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL {jsonl_path} does not exist.")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rows = []
    for r in records:
        rows.append({
            "edge_id": str(r.get("edge_id", "")),
            "citing_doc_id": str(r.get("citing_doc_id", "")),
            "citing_court": str(r.get("citing_court", "")),
            "citing_year": int(r.get("citing_year", 2023)),
            "citing_title": str(r.get("citing_title", "")),
            "citing_judge": r.get("citing_judge"),
            "citing_para_id": r.get("citing_para_id"),
            "rhetorical_role": str(r.get("rhetorical_role", "")),
            "treatment": str(r.get("treatment", "cited")),
            "cited_citation": str(r.get("cited_citation", "")),
            "cited_reporter": str(r.get("cited_reporter", "")),
            "cited_year": r.get("cited_year"),
            "cited_title": r.get("cited_title"),
            "context_sentence": str(r.get("context_sentence", "")),
            "is_internal_corpus_match": bool(r.get("is_internal_corpus_match", False)),
            "target_doc_id": r.get("target_doc_id"),
        })

    schema = pa.schema([
        ("edge_id", pa.string()),
        ("citing_doc_id", pa.string()),
        ("citing_court", pa.string()),
        ("citing_year", pa.int32()),
        ("citing_title", pa.string()),
        ("citing_judge", pa.string()),
        ("citing_para_id", pa.int32()),
        ("rhetorical_role", pa.string()),
        ("treatment", pa.string()),
        ("cited_citation", pa.string()),
        ("cited_reporter", pa.string()),
        ("cited_year", pa.int32()),
        ("cited_title", pa.string()),
        ("context_sentence", pa.string()),
        ("is_internal_corpus_match", pa.bool_()),
        ("target_doc_id", pa.string()),
    ])

    table = pa.Table.from_pylist(rows, schema=schema)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_parquet, compression="snappy")
    print(f"[OK] Successfully exported {len(rows)} citation edges to {output_parquet}")
    return len(rows)


def export_precedent_nodes_to_parquet(jsonl_path: Path, output_parquet: Path) -> int:
    """Export precedent nodes with network metrics to Parquet."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL {jsonl_path} does not exist.")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rows = []
    for r in records:
        rows.append({
            "precedent_id": str(r.get("precedent_id", "")),
            "canonical_citation": str(r.get("canonical_citation", "")),
            "reporter": str(r.get("reporter", "")),
            "citation_year": r.get("citation_year"),
            "case_title": r.get("case_title"),
            "in_degree": int(r.get("in_degree", 0)),
            "citing_doc_ids": [str(d) for d in r.get("citing_doc_ids", [])],
            "treatments_json": json.dumps(r.get("treatments", {})),
            "roles_json": json.dumps(r.get("roles", {})),
            "is_internal": bool(r.get("is_internal", False)),
            "doc_id": r.get("doc_id"),
        })

    schema = pa.schema([
        ("precedent_id", pa.string()),
        ("canonical_citation", pa.string()),
        ("reporter", pa.string()),
        ("citation_year", pa.int32()),
        ("case_title", pa.string()),
        ("in_degree", pa.int32()),
        ("citing_doc_ids", pa.list_(pa.string())),
        ("treatments_json", pa.string()),
        ("roles_json", pa.string()),
        ("is_internal", pa.bool_()),
        ("doc_id", pa.string()),
    ])

    table = pa.Table.from_pylist(rows, schema=schema)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_parquet, compression="snappy")
    print(f"[OK] Successfully exported {len(rows)} precedent nodes to {output_parquet}")
    return len(rows)


def export_headnotes_to_parquet(jsonl_path: Path, output_parquet: Path) -> int:
    """Export structured legal headnotes to Parquet format."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Input JSONL {jsonl_path} does not exist.")

    with open(jsonl_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    rows = []
    for r in records:
        rows.append({
            "doc_id": str(r.get("doc_id", "")),
            "court": str(r.get("court", "")),
            "court_slug": str(r.get("court_slug", "")),
            "year": int(r.get("year", 2023)),
            "case_title": str(r.get("case_title", "")),
            "bench": [str(b) for b in r.get("bench", [])],
            "docket_number": r.get("docket_number"),
            "decision_date": r.get("decision_date"),
            "legal_domain": r.get("legal_domain"),
            "case_type": r.get("case_type"),
            "disposition": str(r.get("disposition", "")),
            "catchwords": [str(c) for c in r.get("catchwords", [])],
            "catchwords_line": str(r.get("catchwords_line", "")),
            "brief_facts": str(r.get("brief_facts", "")),
            "issues": [str(i) for i in r.get("issues", [])],
            "petitioner_arguments": [str(a) for a in r.get("petitioner_arguments", [])],
            "respondent_arguments": [str(a) for a in r.get("respondent_arguments", [])],
            "ratio_decidendi": str(r.get("ratio_decidendi", "")),
            "key_precedents": [str(p) for p in r.get("key_precedents", [])],
            "operative_holding": str(r.get("operative_holding", "")),
            "formatted_headnote": str(r.get("formatted_headnote", "")),
            "word_count": int(r.get("word_count", 0)),
        })

    schema = pa.schema([
        ("doc_id", pa.string()),
        ("court", pa.string()),
        ("court_slug", pa.string()),
        ("year", pa.int32()),
        ("case_title", pa.string()),
        ("bench", pa.list_(pa.string())),
        ("docket_number", pa.string()),
        ("decision_date", pa.string()),
        ("legal_domain", pa.string()),
        ("case_type", pa.string()),
        ("disposition", pa.string()),
        ("catchwords", pa.list_(pa.string())),
        ("catchwords_line", pa.string()),
        ("brief_facts", pa.string()),
        ("issues", pa.list_(pa.string())),
        ("petitioner_arguments", pa.list_(pa.string())),
        ("respondent_arguments", pa.list_(pa.string())),
        ("ratio_decidendi", pa.string()),
        ("key_precedents", pa.list_(pa.string())),
        ("operative_holding", pa.string()),
        ("formatted_headnote", pa.string()),
        ("word_count", pa.int32()),
    ])

    table = pa.Table.from_pylist(rows, schema=schema)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_parquet, compression="snappy")
    print(f"[OK] Successfully exported {len(rows)} headnotes to {output_parquet}")
    return len(rows)


def export_evaluation_to_parquet(eval_json_path: Path, output_parquet: Path) -> int:
    """Exports per-query benchmark comparison rows to Parquet."""
    if not eval_json_path.exists():
        raise FileNotFoundError(f"Input JSON {eval_json_path} does not exist.")

    with open(eval_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    query_rows = data.get("per_query_comparison", [])
    if not query_rows:
        print("[WARN] No per-query comparison rows found in evaluation json.")
        return 0

    schema = pa.schema([
        ("query_id", pa.string()),
        ("query_text", pa.string()),
        ("category", pa.string()),
        ("mode", pa.string()),
        ("first_relevant_rank", pa.int32()),
        ("mrr", pa.float64()),
        ("precision_at_1", pa.float64()),
        ("precision_at_3", pa.float64()),
        ("precision_at_5", pa.float64()),
        ("precision_at_10", pa.float64()),
        ("recall_at_1", pa.float64()),
        ("recall_at_3", pa.float64()),
        ("recall_at_5", pa.float64()),
        ("recall_at_10", pa.float64()),
        ("hit_at_1", pa.float64()),
        ("hit_at_3", pa.float64()),
        ("hit_at_5", pa.float64()),
        ("hit_at_10", pa.float64()),
        ("ndcg_at_5", pa.float64()),
        ("ndcg_at_10", pa.float64()),
        ("latency_ms", pa.float64()),
    ])

    table = pa.Table.from_pylist(query_rows, schema=schema)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_parquet, compression="snappy")
    print(f"[OK] Successfully exported {len(query_rows)} evaluation query comparisons to {output_parquet}")
    return len(query_rows)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cleaned"
    if mode == "chunks":
        input_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("all_200_chunks.jsonl")
        output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("D:/LegalTrace/data/indiankanoon_2023_chunks.parquet")
        export_chunks_to_parquet(input_file, output_file)
    elif mode == "edges":
        input_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("citation_edges.jsonl")
        output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("D:/LegalTrace/data/indiankanoon_2023_citation_edges.parquet")
        export_citation_edges_to_parquet(input_file, output_file)
    elif mode == "nodes":
        input_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("precedent_nodes.jsonl")
        output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("D:/LegalTrace/data/indiankanoon_2023_precedent_nodes.parquet")
        export_precedent_nodes_to_parquet(input_file, output_file)
    elif mode == "headnotes":
        input_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("all_200_headnotes.jsonl")
        output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("D:/LegalTrace/data/indiankanoon_2023_headnotes.parquet")
        export_headnotes_to_parquet(input_file, output_file)
    elif mode == "evaluation":
        input_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("evaluation_report.json")
        output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("D:/LegalTrace/data/indiankanoon_2023_evaluation_report.parquet")
        export_evaluation_to_parquet(input_file, output_file)
    else:
        input_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("all_200_cleaned_judgments.jsonl")
        output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("D:/LegalTrace/data/indiankanoon_2023_cleaned.parquet")
        export_to_legaltrace_parquet(input_file, output_file)




