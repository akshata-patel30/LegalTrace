"""FastAPI REST API and Interactive Search Explorer for LegalTrace.

Step 11 of Pipeline A:
- Production REST API endpoints for hybrid legal search, headnote retrieval, and citation lookup.
- Embedded Single-Page Application (SPA) explorer at root (/) with live search, filters, headnotes modal, and precedent rankings.
- Endpoints:
  * GET  /health            - Engine status and index telemetry
  * POST /api/search        - Hybrid, Dense, or BM25 semantic chunk retrieval
  * GET  /api/judgments/{id}- Clean judgment metadata and header
  * GET  /api/headnotes/{id}- Structured legal headnote (catchwords, facts, ratio, holding)
  * GET  /api/citations/{id}- Citations cited by this judgment
  * GET  /api/precedents/top- Landmark precedents ranked by in-degree authority
  * GET  /api/benchmark     - Benchmark evaluation report
  * GET  /                  - Responsive Web Explorer Dashboard
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from ingestion_source.models import (
    LegalHeadnote,
    RhetoricalRole,
    SearchResponse,
    SearchResultItem,
)
from ingestion_source.rag import (
    GroundedAnswer,
    LegalMemoGenerator,
    LegalQueryPlanner,
    LegalRAGSynthesizer,
    LegalShepardizer,
    QueryPlan,
    ShepardResult,
    StatuteRatioLookup,
    StatuteRatioResponse,
    StatuteRatioResult,
)
from ingestion_source.search import HybridLegalSearchEngine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LegalTrace Indian Kanoon Retrieval & Search API",
    description="High-performance legal research API for Delhi High Court judgments with hybrid vector search, rhetorical role filtering, automated headnotes, citation graph lookup, and grounded RAG reasoning.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine and data stores
INDEX_DIR = Path("search_index")
RAW_STORE_DIR = Path("raw-store")
EVAL_REPORT_PATH = Path("evaluation_report.json")
PRECEDENT_NODES_PATH = Path("precedent_nodes.jsonl")
CITATION_EDGES_PATH = Path("citation_edges.jsonl")

_engine: Optional[HybridLegalSearchEngine] = None
_precedents_cache: list[dict[str, Any]] = []


def get_engine() -> HybridLegalSearchEngine:
    global _engine
    if _engine is None:
        if not INDEX_DIR.exists():
            raise HTTPException(status_code=503, detail="Search index not initialized. Run 'ingest index' first.")
        _engine = HybridLegalSearchEngine(dim=384)
        _engine.load_index(INDEX_DIR)
    return _engine


def get_top_precedents(limit: int = 15) -> list[dict[str, Any]]:
    global _precedents_cache
    if not _precedents_cache and PRECEDENT_NODES_PATH.exists():
        nodes = []
        with open(PRECEDENT_NODES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    nodes.append(json.loads(line))
        nodes.sort(key=lambda x: x.get("in_degree", 0), reverse=True)
        _precedents_cache = nodes
    return _precedents_cache[:limit]


_planner: Optional[LegalQueryPlanner] = None
_shepardizer: Optional[LegalShepardizer] = None
_synthesizer: Optional[LegalRAGSynthesizer] = None
_memo_generator: Optional[LegalMemoGenerator] = None


def get_planner() -> LegalQueryPlanner:
    global _planner
    if _planner is None:
        _planner = LegalQueryPlanner()
    return _planner


def get_shepardizer() -> LegalShepardizer:
    global _shepardizer
    if _shepardizer is None:
        _shepardizer = LegalShepardizer()
    return _shepardizer


def get_synthesizer() -> LegalRAGSynthesizer:
    global _synthesizer
    if _synthesizer is None:
        _synthesizer = LegalRAGSynthesizer(
            planner=get_planner(),
            shepardizer=get_shepardizer(),
            search_engine=get_engine(),
        )
    return _synthesizer


def get_memo_generator() -> LegalMemoGenerator:
    global _memo_generator
    if _memo_generator is None:
        _memo_generator = LegalMemoGenerator(
            synthesizer=get_synthesizer(),
            shepardizer=get_shepardizer(),
        )
    return _memo_generator


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str = Field(..., description="Legal query or natural language factual statement")
    mode: str = Field("hybrid", description="Retrieval mode: hybrid | dense | bm25")
    top_k: int = Field(10, ge=1, le=100, description="Number of top chunks to return")
    filter_role: Optional[str] = Field(None, description="Optional rhetorical role filter (e.g. analysis, facts, operative)")
    filter_court: Optional[str] = Field(None, description="Optional court filter (e.g. Delhi)")
    filter_year: Optional[int] = Field(None, description="Optional judgment year (e.g. 2023)")


class RAGPlanRequest(BaseModel):
    query: str = Field(..., description="Legal query to decompose into sub-tasks")


class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Legal question to plan, retrieve, and synthesize")
    max_evidence_chunks: int = Field(8, ge=1, le=20, description="Max evidence chunks to evaluate")


class RAGMemoRequest(BaseModel):
    query: Optional[str] = Field(None, description="Legal question or subject for memo")
    doc_type: str = Field("memo", description="Document type: memo | brief | bail")
    doc_id: Optional[str] = Field(None, description="Document ID for case brief")
    court: Optional[str] = Field("delhi", description="Court slug")
    year: Optional[int] = Field(2023, description="Decision year")
    applicant_name: Optional[str] = Field(None, description="Applicant name for bail application")
    fir_no: Optional[str] = Field(None, description="FIR number for bail application")
    police_station: Optional[str] = Field(None, description="Police station for bail application")
    sections_invoked: list[str] = Field(default_factory=list, description="Sections invoked")
    custody_date: Optional[str] = Field(None, description="Custody date string")
    facts_summary: Optional[str] = Field(None, description="Factual matrix summary")


class HealthResponse(BaseModel):
    status: str
    total_indexed_chunks: int
    dense_dimension: int
    bm25_vocabulary_size: int
    average_token_length: float
    index_directory: str


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Returns search engine status, loaded chunk count, and index statistics."""
    engine = get_engine()
    return HealthResponse(
        status="ok",
        total_indexed_chunks=len(engine.chunks),
        dense_dimension=engine.dim,
        bm25_vocabulary_size=len(engine.bm25.idf),
        average_token_length=round(engine.bm25.avgdl, 1),
        index_directory=str(INDEX_DIR),
    )


@app.post("/api/search", response_model=SearchResponse)
def search_api(req: SearchRequest) -> SearchResponse:
    """Execute real-time legal search across 9,167 chunks combining Dense and BM25 ranking."""
    engine = get_engine()
    resp = engine.search(
        query=req.query,
        mode=req.mode,
        top_k=req.top_k,
        filter_role=req.filter_role,
        filter_court=req.filter_court,
        filter_year=req.filter_year,
    )
    return resp


@app.get("/api/judgments/{doc_id}")
def get_judgment(doc_id: str) -> dict[str, Any]:
    """Retrieve judgment metadata and clean header details for a given document."""
    target_dirs = list(RAW_STORE_DIR.glob(f"**/{doc_id}"))
    if not target_dirs or not target_dirs[0].is_dir():
        raise HTTPException(status_code=404, detail=f"Judgment {doc_id} not found.")

    doc_dir = target_dirs[0]
    meta_file = doc_dir / "metadata.json"
    cleaned_file = doc_dir / "cleaned.json"
    enriched_file = doc_dir / "enriched.json"

    meta = json.loads(meta_file.read_text(encoding="utf-8")) if meta_file.exists() else {}
    cleaned = json.loads(cleaned_file.read_text(encoding="utf-8")) if cleaned_file.exists() else {}
    enriched = json.loads(enriched_file.read_text(encoding="utf-8")) if enriched_file.exists() else {}

    return {
        "doc_id": doc_id,
        "title": meta.get("title") or cleaned.get("case_title") or "Unknown Title",
        "court": meta.get("court") or "Delhi High Court",
        "year": meta.get("year") or 2023,
        "docket_numbers": cleaned.get("header", {}).get("docket_numbers") or [],
        "decision_date": cleaned.get("header", {}).get("decision_date"),
        "disposition": cleaned.get("disposition"),
        "bench": cleaned.get("header", {}).get("bench") or enriched.get("canonical_judges") or [],
        "legal_domain": enriched.get("legal_domain"),
        "total_paragraphs": len(cleaned.get("paragraphs") or []),
        "statute_count": enriched.get("total_statutes_count", 0),
        "citation_count": enriched.get("total_citations_count", 0),
    }


@app.get("/api/headnotes/{doc_id}", response_model=LegalHeadnote)
def get_headnote(doc_id: str) -> LegalHeadnote:
    """Retrieve structured legal headnote and syllabus for a given document."""
    target_dirs = list(RAW_STORE_DIR.glob(f"**/{doc_id}"))
    if not target_dirs or not target_dirs[0].is_dir():
        raise HTTPException(status_code=404, detail=f"Judgment {doc_id} not found.")

    doc_dir = target_dirs[0]
    hn_file = doc_dir / "headnote.json"
    if not hn_file.exists():
        from ingestion_source.summarizer import LegalHeadnoteGenerator
        gen = LegalHeadnoteGenerator()
        hn = gen.generate_for_document_dir(doc_dir)
        return hn

    return LegalHeadnote(**json.loads(hn_file.read_text(encoding="utf-8")))


@app.get("/api/citations/{doc_id}")
def get_judgment_citations(doc_id: str) -> dict[str, Any]:
    """Retrieve precedent citations and judicial treatments for a given document."""
    edges = []
    if CITATION_EDGES_PATH.exists():
        with open(CITATION_EDGES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if item.get("citing_doc_id") == doc_id:
                        edges.append(item)

    return {
        "doc_id": doc_id,
        "total_citations": len(edges),
        "citations": edges,
    }


@app.get("/api/precedents/top")
def list_top_precedents(limit: int = Query(15, ge=1, le=100)) -> list[dict[str, Any]]:
    """Retrieve top landmark legal authorities across the Delhi High Court network."""
    return get_top_precedents(limit=limit)


@app.get("/api/benchmark")
def get_benchmark_report() -> dict[str, Any]:
    """Retrieve the Step 10 empirical retrieval benchmark evaluation report."""
    if not EVAL_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="Benchmark report not found. Run 'ingest benchmark' first.")
    return json.loads(EVAL_REPORT_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Phase 1: Online RAG & Agentic Reasoning Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/rag/plan")
def plan_rag_query(req: RAGPlanRequest) -> dict[str, Any]:
    """Decompose complex legal query into sub-tasks and target rhetorical roles."""
    planner = get_planner()
    plan = planner.plan(req.query)
    return plan.to_dict()


@app.post("/api/rag/query")
def query_rag_api(req: RAGQueryRequest) -> dict[str, Any]:
    """Execute multi-hop legal retrieval, Shepardization audit, and citation-grounded synthesis."""
    synthesizer = get_synthesizer()
    engine = get_engine()
    answer = synthesizer.query(
        user_query=req.query,
        engine=engine,
        max_evidence_chunks=req.max_evidence_chunks,
    )
    return answer.to_dict()


@app.post("/api/rag/memo")
def generate_memo_api(req: RAGMemoRequest) -> dict[str, Any]:
    """Generate structured court brief, legal research memorandum, or bail application."""
    generator = get_memo_generator()
    engine = get_engine()
    doc_type = req.doc_type.lower().strip()

    if doc_type in ("brief", "case_brief"):
        if not req.doc_id:
            raise HTTPException(status_code=400, detail="doc_id is required for case brief generation.")
        doc = generator.generate_case_brief(
            doc_id=req.doc_id,
            court=req.court or "delhi",
            year=req.year or 2023,
        )
    elif doc_type in ("bail", "bail_application"):
        doc = generator.generate_bail_application(
            applicant_name=req.applicant_name or "Applicant",
            fir_no=req.fir_no or "000/2023",
            police_station=req.police_station or "Special Cell",
            sections_invoked=req.sections_invoked or ["Section 439 CrPC"],
            custody_date=req.custody_date or "01.01.2023",
            facts_summary=req.facts_summary or req.query or "The Applicant has been falsely implicated.",
            engine=engine,
        )
    else:  # default to memo
        if not req.query:
            raise HTTPException(status_code=400, detail="query is required for legal memorandum generation.")
        doc = generator.generate_legal_memorandum(
            query=req.query,
            engine=engine,
        )

    return doc.to_dict()


@app.get("/api/shepardize/{citation:path}")
def shepardize_precedent_api(citation: str) -> dict[str, Any]:
    """Audit precedent treatment history across the corpus and determine whether it remains good law."""
    shepardizer = get_shepardizer()
    res = shepardizer.check(citation)
    return res.to_dict()


@app.get("/api/graph/neighbors/{node_id:path}")
def get_graph_neighbors(node_id: str) -> dict[str, Any]:
    """Retrieve incoming and outgoing citation graph neighbors for a document or precedent citation."""
    incoming: list[dict[str, Any]] = []
    outgoing: list[dict[str, Any]] = []

    clean_id = node_id.strip()

    if CITATION_EDGES_PATH.exists():
        with open(CITATION_EDGES_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                citing = str(item.get("citing_doc_id", ""))
                prec_cit = str(item.get("precedent_citation", ""))
                canon_id = str(item.get("canonical_precedent_id", ""))

                if citing == clean_id:
                    outgoing.append(item)
                elif clean_id in (prec_cit, canon_id) or clean_id.lower() in prec_cit.lower():
                    incoming.append(item)

    return {
        "node_id": clean_id,
        "total_incoming": len(incoming),
        "total_outgoing": len(outgoing),
        "incoming_citations": incoming[:50],
        "outgoing_citations": outgoing[:50],
    }


_graph_store: Any = None


def get_graph_store():
    global _graph_store
    if _graph_store is None:
        from ingestion_source.db.graph_store import FalkorGraphStore
        _graph_store = FalkorGraphStore()
    return _graph_store


@app.get("/api/graph/binding/{doc_id}")
def get_binding_authorities(doc_id: str) -> dict[str, Any]:
    """Retrieve binding Supreme Court authorities for a judgment under Article 141 of the Constitution of India."""
    store = get_graph_store()
    try:
        authorities = store.get_binding_authority_chain(doc_id)
    except Exception as e:
        logger.warning("FalkorDB query failed in get_binding_authorities: %s", e)
        authorities = []

    return {
        "doc_id": doc_id,
        "total_binding_authorities": len(authorities),
        "article_141_declaration": "Law declared by the Supreme Court shall be binding on all courts within the territory of India.",
        "authorities": authorities,
    }


@app.get("/api/graph/hierarchy")
def get_hierarchy_influence(court: Optional[str] = None) -> dict[str, Any]:
    """Analyze breakdown of Supreme Court binding citations vs coordinate High Court citations."""
    store = get_graph_store()
    try:
        data = store.get_court_hierarchy_influence(court_slug=court)
    except Exception as e:
        logger.warning("FalkorDB query failed in get_hierarchy_influence: %s", e)
        data = {"court_scope": court or "All Courts", "total_citations": 0, "hierarchy_breakdown": {}}
    return data


@app.get("/api/batch/status")
async def get_batch_status_api() -> list[dict[str, Any]]:
    """Retrieve progress and telemetry for all multi-court batch ingestion jobs in PostgreSQL."""
    from ingestion_source.batch_ingest import BatchIngestionOrchestrator
    orchestrator = BatchIngestionOrchestrator()
    return await orchestrator.get_all_batch_status()


class StatuteRatioRequest(BaseModel):
    act: str = Field(..., description="Statute or Act name/alias (e.g. 'CrPC', 'IPC', 'Arbitration Act')")
    section: str = Field(..., description="Section number (e.g. '482', '438', '34')")
    limit: int = Field(10, ge=1, le=50, description="Maximum ratio passages to return")


class TargetedScrapeRequest(BaseModel):
    query: str = Field(..., description="Targeted legal query or keywords (e.g. 'Section 37 NDPS commercial quantity bail')")
    court: str = Field("Delhi High Court", description="Target court name (e.g. 'Delhi High Court')")
    year: Optional[int] = Field(2023, description="Decision year (or None for all years)")
    limit: int = Field(10, ge=1, le=50, description="Max candidate judgments to return")
    source_mode: str = Field("auto", description="Discovery source mode: auto | seed | api | scrape")
    auto_fetch: bool = Field(False, description="Whether to automatically download and store full text into raw-store")



@app.get("/api/graph/visualization/{doc_id}")
def get_graph_visualization(
    doc_id: str,
    depth: int = Query(2, ge=1, le=5, description="Graph traversal depth"),
    limit: int = Query(50, ge=5, le=200, description="Max nodes/links to return"),
) -> dict[str, Any]:
    """Return D3/Cytoscape-compatible node-link JSON for a judgment's citation neighborhood."""
    store = get_graph_store()
    nodes_map: dict[str, dict[str, Any]] = {}
    links: list[dict[str, Any]] = []
    treatment_counts: dict[str, int] = {}

    # Attempt FalkorDB graph traversal
    try:
        query = (
            "MATCH (j:Judgment {doc_id: $doc_id})-[r:CITES]->(p:Precedent) "
            "RETURN j.doc_id AS citing_id, j.title AS citing_title, j.court AS citing_court, "
            "       j.year AS citing_year, p.precedent_id AS cited_id, "
            "       p.canonical_citation AS cited_citation, p.case_title AS cited_title, "
            "       p.year AS cited_year, p.in_degree AS cited_in_degree, "
            "       r.treatment AS treatment, r.role AS role, r.context AS context "
            "LIMIT $limit"
        )
        res = store.graph.query(query, {"doc_id": doc_id, "limit": limit})
        for row in getattr(res, "result_set", []):
            citing_id = str(row[0])
            citing_title = row[1] or f"Judgment {citing_id}"
            citing_court = row[2] or "Delhi High Court"
            citing_year = row[3] or 0
            cited_id = str(row[4])
            cited_citation = row[5] or cited_id
            cited_title = row[6] or cited_citation
            cited_year = row[7] or 0
            cited_in_degree = row[8] or 1
            treatment = row[9] or "cited"
            role = row[10] or "precedents"
            context = row[11] or ""

            if citing_id not in nodes_map:
                nodes_map[citing_id] = {
                    "id": citing_id,
                    "label": f"Doc {citing_id}",
                    "title": citing_title,
                    "type": "judgment",
                    "court": citing_court,
                    "year": citing_year,
                    "group": 1,
                }

            if cited_id not in nodes_map:
                nodes_map[cited_id] = {
                    "id": cited_id,
                    "label": cited_citation,
                    "title": cited_title,
                    "type": "precedent",
                    "court": "Precedent Authority",
                    "year": cited_year,
                    "in_degree": cited_in_degree,
                    "group": 2,
                }

            links.append({
                "source": citing_id,
                "target": cited_id,
                "treatment": treatment,
                "role": role,
                "context": context[:250] if context else "",
            })
            treatment_counts[treatment] = treatment_counts.get(treatment, 0) + 1
    except Exception as e:
        logger.warning("FalkorDB query failed in get_graph_visualization: %s", e)

    # Fallback to local files if nodes_map is empty
    if not nodes_map:
        if CITATION_EDGES_PATH.exists():
            with open(CITATION_EDGES_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    edge = json.loads(line)
                    if edge.get("citing_doc_id") == doc_id:
                        citing_id = doc_id
                        cited_cit = edge.get("cited_citation", "")
                        clean = re.sub(r"[^\w\s]", " ", cited_cit)
                        norm_key = "_".join(p.lower() for p in clean.split() if p)
                        cited_id = f"prec_{norm_key}"
                        treatment = edge.get("treatment", "cited")
                        role = edge.get("rhetorical_role", "precedents")

                        if citing_id not in nodes_map:
                            nodes_map[citing_id] = {
                                "id": citing_id,
                                "label": f"Doc {citing_id}",
                                "title": edge.get("citing_title") or f"Judgment {citing_id}",
                                "type": "judgment",
                                "court": edge.get("citing_court", "Delhi High Court"),
                                "year": edge.get("citing_year", 2023),
                                "group": 1,
                            }
                        if cited_id not in nodes_map:
                            nodes_map[cited_id] = {
                                "id": cited_id,
                                "label": cited_cit,
                                "title": edge.get("case_title") or cited_cit,
                                "type": "precedent",
                                "court": "Precedent Authority",
                                "year": edge.get("citation_year", 0),
                                "in_degree": 1,
                                "group": 2,
                            }
                        links.append({
                            "source": citing_id,
                            "target": cited_id,
                            "treatment": treatment,
                            "role": role,
                            "context": edge.get("context_sentence", "")[:250],
                        })
                        treatment_counts[treatment] = treatment_counts.get(treatment, 0) + 1
                        if len(links) >= limit:
                            break

    # If still root node alone or empty, ensure root node exists
    if doc_id not in nodes_map:
        nodes_map[doc_id] = {
            "id": doc_id,
            "label": f"Doc {doc_id}",
            "title": f"Judgment {doc_id}",
            "type": "judgment",
            "court": "Delhi High Court",
            "year": 2023,
            "group": 1,
        }

    return {
        "root_id": doc_id,
        "depth": depth,
        "nodes": list(nodes_map.values()),
        "links": links,
        "stats": {
            "total_nodes": len(nodes_map),
            "total_links": len(links),
            "treatment_breakdown": treatment_counts,
        },
    }


@app.get("/api/statutes/lookup")
async def statute_ratio_lookup_get(
    act: str = Query(..., description="Act name or acronym, e.g. 'CrPC', 'IPC', 'Arbitration'"),
    section: str = Query(..., description="Section number, e.g. '482', '34'"),
    limit: int = Query(10, ge=1, le=50, description="Max results"),
) -> dict[str, Any]:
    """Reverse lookup of judicial interpretations and ratio decidendi for a statute provision."""
    lookup_engine = StatuteRatioLookup(raw_store_dir=RAW_STORE_DIR)
    resp = await lookup_engine.lookup(act_name=act, section=section, top_k=limit)
    return resp.to_dict()


@app.post("/api/statutes/ratio-lookup")
async def statute_ratio_lookup_post(req: StatuteRatioRequest) -> dict[str, Any]:
    """POST endpoint for statute-to-ratio reverse lookup with structured body."""
    lookup_engine = StatuteRatioLookup(raw_store_dir=RAW_STORE_DIR)
    resp = await lookup_engine.lookup(act_name=req.act, section=req.section, top_k=req.limit)
    return resp.to_dict()


@app.post("/api/scrape/targeted")
def targeted_scrape_api(req: TargetedScrapeRequest) -> dict[str, Any]:
    """Execute targeted keyword/statute judgment discovery with optional full-text download."""
    from ingestion_source.config import get_settings
    from ingestion_source.discovery import DiscoveryEngine
    from ingestion_source.job import JobOrchestrator

    settings = get_settings()
    engine = DiscoveryEngine(settings=settings)

    courts = [req.court] if req.court else ["Delhi High Court"]
    years = [req.year] if (req.year and req.year > 0) else None

    refs = list(engine.discover_refs(
        courts=courts,
        years=years,
        limit_per_target=req.limit,
        total_limit=req.limit,
        source_mode=req.source_mode,
        query=req.query,
    ))

    results_data = [
        {
            "doc_id": r.doc_id,
            "title": r.title,
            "court": r.court,
            "year": r.year,
            "source_url": r.source_url,
            "headline": r.headline,
            "month": r.month,
        }
        for r in refs
    ]

    fetched_count = 0
    job_id = None
    indexed_chunks_count = 0
    if req.auto_fetch and refs:
        orchestrator = JobOrchestrator(settings=settings)
        job_id = orchestrator.create_job(refs, mode=settings.MODE)
        report = orchestrator.execute_job(job_id=job_id, limit=len(refs), workers=1)
        fetched_count = getattr(report, "stored_count", 0)

        # ─── Downstream Automatic Pipeline B: Clean, Enrich, Chunk, Headnote & Live Search Indexing ───
        try:
            from ingestion_source.pipeline_processor import process_and_index_scraped_documents
            engine_inst = None
            try:
                engine_inst = get_engine()
            except Exception as e:
                logger.warning("Search engine could not be acquired for auto-indexing: %s", e)

            proc_stats = process_and_index_scraped_documents(
                doc_refs=refs,
                settings=settings,
                search_engine=engine_inst,
                index_dir=INDEX_DIR,
            )
            indexed_chunks_count = proc_stats.get("indexed_chunks", 0)
        except Exception as e:
            logger.warning("Downstream pipeline processing failed: %s", e)

    return {
        "status": "success",
        "query": req.query,
        "court": req.court,
        "year": req.year if (req.year and req.year > 0) else None,
        "total_discovered": len(refs),
        "auto_fetch": req.auto_fetch,
        "fetched_count": fetched_count,
        "indexed_chunks_count": indexed_chunks_count,
        "job_id": job_id,
        "results": results_data,
    }


class DownloadAndIndexRequest(BaseModel):
    doc_id: str
    court: str = Field("Delhi High Court", description="Court name")
    year: Optional[int] = Field(2023, description="Decision year")
    title: Optional[str] = Field(None, description="Case title")
    source_url: Optional[str] = Field(None, description="Source URL")


@app.post("/api/scrape/download-and-index")
def download_and_index_single_judgment(req: DownloadAndIndexRequest) -> dict[str, Any]:
    """Download a single discovered judgment, run downstream processing, and index into search/RAG."""
    from ingestion_source.config import get_settings
    from ingestion_source.job import JobOrchestrator
    from ingestion_source.models import DocRef
    from ingestion_source.pipeline_processor import process_and_index_scraped_documents

    settings = get_settings()
    ref = DocRef(
        doc_id=req.doc_id,
        court=req.court,
        year=req.year or 2023,
        title=req.title or f"Judgment {req.doc_id}",
        source_url=req.source_url or f"https://indiankanoon.org/doc/{req.doc_id}/",
    )
    orchestrator = JobOrchestrator(settings=settings)
    job_id = orchestrator.create_job([ref], mode=settings.MODE)
    report = orchestrator.execute_job(job_id=job_id, limit=1, workers=1)
    stored_count = getattr(report, "stored_count", 0)

    engine_inst = None
    try:
        engine_inst = get_engine()
    except Exception as e:
        logger.warning("Search engine could not be acquired for auto-indexing: %s", e)

    proc_stats = process_and_index_scraped_documents(
        doc_refs=[ref],
        settings=settings,
        search_engine=engine_inst,
        index_dir=INDEX_DIR,
    )

    return {
        "status": "success",
        "doc_id": req.doc_id,
        "title": ref.title,
        "stored": stored_count > 0,
        "indexed_chunks": proc_stats.get("indexed_chunks", 0),
        "headnotes_count": proc_stats.get("headnotes_count", 0),
    }


@app.get("/api/scrape/indexed-doc-ids")
def get_indexed_doc_ids() -> list[str]:
    """Return list of all doc_ids currently present in the search index."""
    try:
        eng = get_engine()
        return list({str(c.get("doc_id")) for c in eng.chunks if c.get("doc_id")})
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Interactive Web Explorer Dashboard (HTML5 + Tailwind CSS SPA)
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_dashboard() -> str:
    """Serves the interactive single-page legal search and headnote exploration dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>LegalTrace — Delhi High Court Research Explorer</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.13.3/dist/cdn.min.js"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    [x-cloak] { display: none !important; }
    .role-badge { font-family: ui-monospace, monospace; font-size: 0.72rem; }
  </style>
</head>
<body class="bg-slate-900 text-slate-100 min-h-screen font-sans" x-data="explorerApp()">

  <!-- Top Navigation Header -->
  <header class="border-b border-slate-800 bg-slate-950/80 backdrop-blur sticky top-0 z-40">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
      <div class="flex items-center space-x-3">
        <div class="h-9 w-9 rounded-lg bg-indigo-600 flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/30">
          <i class="fa-solid fa-scale-balanced text-lg"></i>
        </div>
        <div>
          <span class="font-bold text-lg text-white tracking-tight">LegalTrace</span>
          <span class="text-xs ml-2 px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 font-mono">DHC 2023</span>
        </div>
      </div>
      
      <!-- Tab Navigation -->
      <nav class="flex space-x-1 bg-slate-800/60 p-1 rounded-lg border border-slate-700/50">
        <button @click="tab = 'search'" :class="tab === 'search' ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400 hover:text-white'" class="px-3 py-1.5 rounded-md text-xs transition">
          <i class="fa-solid fa-magnifying-glass mr-1.5"></i>Search
        </button>
        <button @click="tab = 'rag'" :class="tab === 'rag' ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400 hover:text-white'" class="px-3 py-1.5 rounded-md text-xs transition">
          <i class="fa-solid fa-brain mr-1.5"></i>RAG Assistant
        </button>
        <button @click="tab = 'shepardize'" :class="tab === 'shepardize' ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400 hover:text-white'" class="px-3.5 py-1.5 rounded-md text-xs transition">
          <i class="fa-solid fa-shield-halved mr-1.5"></i>Shepardize
        </button>
        <button @click="tab = 'memo'" :class="tab === 'memo' ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400 hover:text-white'" class="px-3.5 py-1.5 rounded-md text-xs transition">
          <i class="fa-solid fa-file-contract mr-1.5"></i>Memo & Drafts
        </button>
        <button @click="tab = 'precedents'; loadPrecedents()" :class="tab === 'precedents' ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400 hover:text-white'" class="px-3 py-1.5 rounded-md text-xs transition">
          <i class="fa-solid fa-network-wired mr-1.5"></i>Citation Graph
        </button>
        <button @click="tab = 'statute'" :class="tab === 'statute' ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400 hover:text-white'" class="px-3 py-1.5 rounded-md text-xs transition">
          <i class="fa-solid fa-book-bookmark mr-1.5"></i>Statute Ratio
        </button>
        <button @click="tab = 'benchmark'; loadBenchmark()" :class="tab === 'benchmark' ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400 hover:text-white'" class="px-3 py-1.5 rounded-md text-xs transition">
          <i class="fa-solid fa-chart-line mr-1.5"></i>Benchmark QA
        </button>
        <button @click="tab = 'scrape'" :class="tab === 'scrape' ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400 hover:text-white'" class="px-3 py-1.5 rounded-md text-xs transition">
          <i class="fa-solid fa-cloud-arrow-down mr-1.5"></i>Keyword Scraper
        </button>
      </nav>

      <!-- Status Pill -->
      <div class="flex items-center text-xs text-slate-400 space-x-2">
        <span class="inline-block w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
        <span x-text="statusText">Index Active</span>
      </div>
    </div>
  </header>

  <main class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">

    <!-- ================================================================= -->
    <!-- TAB 1: SEARCH & RETRIEVAL EXPLORER -->
    <!-- ================================================================= -->
    <div x-show="tab === 'search'" x-transition>
      
      <!-- Query Control Panel -->
      <div class="bg-slate-800/80 rounded-2xl p-6 border border-slate-700 shadow-xl mb-8">
        <form @submit.prevent="executeSearch">
          <div class="relative">
            <i class="fa-solid fa-magnifying-glass absolute left-4 top-4 text-slate-400 text-lg"></i>
            <input 
              type="text" 
              x-model="query" 
              placeholder="Search legal issues, statutes, or precedent doctrine (e.g. 'regular bail commercial quantity Section 37 NDPS Act')..."
              class="w-full pl-12 pr-28 py-3.5 bg-slate-900 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-sm shadow-inner"
            />
            <button 
              type="submit" 
              :disabled="loading"
              class="absolute right-2 top-2 px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow-md transition"
            >
              <span x-show="!loading"><i class="fa-solid fa-bolt mr-1"></i>Search</span>
              <span x-show="loading"><i class="fa-solid fa-spinner fa-spin mr-1"></i>Searching...</span>
            </button>
          </div>

          <!-- Filters Row -->
          <div class="mt-4 flex flex-wrap items-center gap-4 text-xs text-slate-300">
            <!-- Mode Selector -->
            <div class="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700">
              <span class="text-slate-400 font-medium">Mode:</span>
              <label class="cursor-pointer flex items-center space-x-1">
                <input type="radio" name="mode" value="hybrid" x-model="mode" class="text-indigo-600 focus:ring-0">
                <span :class="mode === 'hybrid' ? 'text-indigo-400 font-bold' : ''">Hybrid RRF</span>
              </label>
              <label class="cursor-pointer flex items-center space-x-1">
                <input type="radio" name="mode" value="dense" x-model="mode" class="text-indigo-600 focus:ring-0">
                <span :class="mode === 'dense' ? 'text-indigo-400 font-bold' : ''">Dense (384-d)</span>
              </label>
              <label class="cursor-pointer flex items-center space-x-1">
                <input type="radio" name="mode" value="bm25" x-model="mode" class="text-indigo-600 focus:ring-0">
                <span :class="mode === 'bm25' ? 'text-indigo-400 font-bold' : ''">BM25 Okapi</span>
              </label>
            </div>

            <!-- Role Filter -->
            <div class="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700">
              <span class="text-slate-400 font-medium">Rhetorical Role:</span>
              <select x-model="roleFilter" class="bg-transparent text-slate-200 focus:outline-none cursor-pointer">
                <option value="" class="bg-slate-800">All Roles (Any)</option>
                <option value="analysis" class="bg-slate-800">ANALYSIS (Ratio)</option>
                <option value="facts" class="bg-slate-800">FACTS (Narrative)</option>
                <option value="operative" class="bg-slate-800">OPERATIVE (Holding)</option>
                <option value="arguments_petitioner" class="bg-slate-800">PETITIONER SUBMISSIONS</option>
                <option value="arguments_respondent" class="bg-slate-800">RESPONDENT SUBMISSIONS</option>
                <option value="statute_reference" class="bg-slate-800">STATUTORY TEXT</option>
                <option value="precedents" class="bg-slate-800">PRECEDENT CITATIONS</option>
              </select>
            </div>

            <!-- Top K -->
            <div class="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700">
              <span class="text-slate-400 font-medium">Top Results:</span>
              <select x-model.number="topK" class="bg-transparent text-slate-200 focus:outline-none cursor-pointer">
                <option value="5" class="bg-slate-800">Top 5</option>
                <option value="10" class="bg-slate-800">Top 10</option>
                <option value="20" class="bg-slate-800">Top 20</option>
              </select>
            </div>

            <!-- Pre-set Benchmark Queries Quick Pills -->
            <div class="flex items-center space-x-1 ml-auto">
              <span class="text-slate-500 mr-1">Sample queries:</span>
              <button type="button" @click="query = 'regular bail commercial quantity Section 37 NDPS Act'; executeSearch()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 transition">NDPS Bail</button>
              <button type="button" @click="query = 'Section 34 Arbitration Act patent illegality public policy'; executeSearch()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 transition">Arbitration</button>
              <button type="button" @click="query = 'Maneka Gandhi right to travel abroad personal liberty Article 21'; executeSearch()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 transition">Maneka Gandhi</button>
            </div>
          </div>
        </form>
      </div>

      <!-- Telemetry Bar -->
      <div x-show="searchResults.length > 0" class="flex items-center justify-between px-2 mb-4 text-xs text-slate-400 font-mono">
        <div>
          Found <span class="font-bold text-white" x-text="searchResults.length"></span> semantic chunks 
          (Retrieved in <span class="font-bold text-emerald-400" x-text="searchLatency + ' ms'"></span>)
        </div>
        <div>Indexed Chunks: <span class="text-indigo-400 font-bold" x-text="totalIndexed"></span></div>
      </div>

      <!-- Results Stream -->
      <div class="space-y-4">
        <template x-for="item in searchResults" :key="item.chunk_id">
          <div class="bg-slate-800/60 hover:bg-slate-800 rounded-xl p-5 border border-slate-700/80 hover:border-indigo-500/50 transition shadow-lg">
            <div class="flex items-start justify-between gap-4">
              <div class="space-y-1 flex-1">
                <div class="flex items-center space-x-2">
                  <span class="px-2 py-0.5 rounded font-mono text-xs font-bold" :class="getRoleBadgeClass(item.rhetorical_role)" x-text="item.rhetorical_role.toUpperCase()"></span>
                  <span class="text-xs text-slate-400 font-mono" x-text="'Rank #' + item.rank + ' | Score: ' + item.score.toFixed(4)"></span>
                </div>
                <h3 class="text-sm font-semibold text-white tracking-tight" x-text="item.case_title"></h3>
                <div class="text-xs text-slate-400 font-mono">
                  <span>Delhi High Court</span> • <span x-text="'Year ' + item.year"></span> • <span>Doc ID: </span><span class="text-indigo-400 font-bold" x-text="item.doc_id"></span>
                </div>
              </div>

              <!-- Action Buttons -->
              <div class="flex space-x-2">
                <button 
                  @click="openHeadnote(item.doc_id)"
                  class="px-3 py-1.5 bg-slate-700 hover:bg-indigo-600 text-white rounded-lg text-xs font-medium transition flex items-center space-x-1"
                >
                  <i class="fa-solid fa-file-lines mr-1"></i>Headnote
                </button>
                <button 
                  @click="openCitations(item.doc_id)"
                  class="px-3 py-1.5 bg-slate-700 hover:bg-indigo-600 text-white rounded-lg text-xs font-medium transition flex items-center space-x-1"
                >
                  <i class="fa-solid fa-quote-right mr-1"></i>Citations
                </button>
              </div>
            </div>

            <!-- Snippet Body -->
            <p class="mt-3 text-xs text-slate-300 leading-relaxed bg-slate-950/40 p-3 rounded-lg border border-slate-800 font-serif" x-text="item.text_snippet"></p>

            <!-- Metadata tags -->
            <div class="mt-3 flex flex-wrap gap-2 text-xs">
              <template x-for="st in (item.statute_mentions || [])" :key="st">
                <span class="px-2 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-800/60 font-mono text-[11px]" x-text="st"></span>
              </template>
              <template x-for="cit in (item.citations || [])" :key="cit">
                <span class="px-2 py-0.5 rounded bg-purple-950/80 text-purple-300 border border-purple-800/60 font-mono text-[11px]" x-text="cit"></span>
              </template>
            </div>
          </div>
        </template>
      </div>

      <!-- Empty State -->
      <div x-show="!loading && searchResults.length === 0" class="text-center py-16 text-slate-500">
        <i class="fa-solid fa-gavel text-4xl mb-3 text-slate-600"></i>
        <p class="text-sm">Enter a legal search query above to inspect hybrid vector retrieval across Delhi High Court judgments.</p>
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB: LEGAL RAG ASSISTANT (PIPELINE B) -->
    <!-- ================================================================= -->
    <div x-show="tab === 'rag'" x-cloak x-transition>
      <div class="bg-slate-800/80 rounded-2xl p-6 border border-slate-700 shadow-xl mb-8">
        <h2 class="text-base font-bold text-white mb-2 flex items-center">
          <i class="fa-solid fa-brain text-indigo-400 mr-2"></i>
          Online Legal RAG & Agentic Reasoning (Groq LLM + Hybrid Corpus)
        </h2>
        <p class="text-xs text-slate-400 mb-6">Decomposes complex legal queries into sub-tasks, performs multi-hop semantic retrieval, checks precedent good-law status, and synthesizes answers with strict inline [N] citations.</p>

        <form @submit.prevent="executeRAG">
          <div class="relative">
            <i class="fa-solid fa-scale-balanced absolute left-4 top-4 text-slate-400 text-lg"></i>
            <input 
              type="text" 
              x-model="ragQuery" 
              placeholder="Ask a legal research question (e.g. 'What are the mandatory conditions for regular bail involving commercial quantity under Section 37 NDPS Act?')..."
              class="w-full pl-12 pr-32 py-3.5 bg-slate-900 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-sm shadow-inner"
            />
            <button 
              type="submit" 
              :disabled="ragLoading"
              class="absolute right-2 top-2 px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow-md transition"
            >
              <span x-show="!ragLoading"><i class="fa-solid fa-wand-magic-sparkles mr-1"></i>Ask RAG</span>
              <span x-show="ragLoading"><i class="fa-solid fa-spinner fa-spin mr-1"></i>Reasoning...</span>
            </button>
          </div>

          <div class="mt-4 flex flex-wrap items-center gap-4 text-xs text-slate-300">
            <div class="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700">
              <span class="text-slate-400 font-medium">Max Evidence Chunks:</span>
              <select x-model.number="ragMaxChunks" class="bg-transparent text-slate-200 focus:outline-none cursor-pointer">
                <option value="4" class="bg-slate-800">4 chunks</option>
                <option value="8" class="bg-slate-800" selected>8 chunks</option>
                <option value="12" class="bg-slate-800">12 chunks</option>
              </select>
            </div>

            <div class="flex items-center space-x-1 ml-auto">
              <span class="text-slate-500 mr-1">Sample queries:</span>
              <button type="button" @click="ragQuery = 'What are the mandatory conditions for regular bail involving commercial quantity under Section 37 NDPS Act?'; executeRAG()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 transition">NDPS Bail</button>
              <button type="button" @click="ragQuery = 'Can an arbitral award be set aside for patent illegality under Section 34 of the Arbitration Act?'; executeRAG()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 transition">Arbitration</button>
              <button type="button" @click="ragQuery = 'Under what circumstances can an FIR be quashed under Section 482 CrPC upon matrimonial settlement?'; executeRAG()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 transition">Quash FIR</button>
            </div>
          </div>
        </form>
      </div>

      <!-- RAG Result Section -->
      <div x-show="ragResult" class="space-y-6">
        <!-- Telemetry & Confidence Banner -->
        <div class="bg-slate-800/80 rounded-xl p-4 border border-slate-700 flex flex-wrap items-center justify-between gap-4 font-mono text-xs">
          <div class="flex items-center space-x-3">
            <span class="px-2.5 py-1 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 font-bold" x-text="'Model: ' + ragResult?.model_used"></span>
            <span class="text-slate-400">Latency: <span class="text-emerald-400 font-bold" x-text="ragResult?.synthesis_latency_ms?.toFixed(1) + ' ms'"></span></span>
            <span class="text-slate-400">Evidence Passages: <span class="text-white font-bold" x-text="ragResult?.total_chunks_examined"></span></span>
          </div>
          <div class="flex items-center space-x-2">
            <span class="text-slate-400">Confidence Score:</span>
            <span class="px-2 py-0.5 rounded text-xs font-bold" :class="ragResult?.confidence >= 0.7 ? 'bg-emerald-950 text-emerald-300 border border-emerald-800' : 'bg-amber-950 text-amber-300 border border-amber-800'" x-text="(ragResult?.confidence * 100)?.toFixed(1) + '%'"></span>
          </div>
        </div>

        <!-- Shepardization Warnings (if any) -->
        <div x-show="ragResult?.shepard_warnings?.length > 0" class="bg-rose-950/40 border border-rose-800/80 rounded-xl p-4">
          <div class="flex items-center space-x-2 text-rose-300 font-bold text-xs mb-2">
            <i class="fa-solid fa-triangle-exclamation"></i>
            <span>Shepard's Treatment Warnings Detected in Retrieved Precedents</span>
          </div>
          <ul class="list-disc list-inside space-y-1 text-xs text-rose-200/90 font-mono">
            <template x-for="w in ragResult?.shepard_warnings" :key="w">
              <li x-text="w"></li>
            </template>
          </ul>
        </div>

        <!-- Query Plan Accordion -->
        <div class="bg-slate-800/50 rounded-xl border border-slate-700/80 p-4">
          <button @click="showPlan = !showPlan" class="w-full flex items-center justify-between text-xs font-semibold text-slate-300 hover:text-white">
            <span class="flex items-center"><i class="fa-solid fa-list-check text-indigo-400 mr-2"></i>Multi-Hop Query Decomposition Plan (<span x-text="ragResult?.query_plan?.sub_queries?.length || 0"></span> sub-tasks)</span>
            <i class="fa-solid" :class="showPlan ? 'fa-chevron-up' : 'fa-chevron-down'"></i>
          </button>
          <div x-show="showPlan" class="mt-3 space-y-2 border-t border-slate-700/60 pt-3">
            <div class="text-xs text-slate-400 font-mono">Domain: <span class="text-indigo-300 font-bold" x-text="ragResult?.query_plan?.detected_domain?.toUpperCase()"></span> | Strategy: <span class="text-emerald-300 font-bold" x-text="ragResult?.query_plan?.synthesis_strategy"></span></div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
              <template x-for="(sq, idx) in ragResult?.query_plan?.sub_queries" :key="idx">
                <div class="bg-slate-900/80 p-3 rounded-lg border border-slate-800 text-xs font-mono">
                  <div class="flex items-center justify-between text-[11px] mb-1">
                    <span class="text-indigo-400 font-bold" x-text="'Sub-query #' + (idx + 1) + ' [' + sq.query_type.toUpperCase() + ']'"></span>
                    <span class="text-slate-500" x-text="'Top ' + sq.top_k"></span>
                  </div>
                  <div class="text-slate-200 text-xs font-sans mb-1" x-text="sq.query_text"></div>
                  <div class="text-[10px] text-slate-400">Target Roles: <span class="text-amber-300" x-text="(sq.target_roles || []).join(', ') || 'Any'"></span></div>
                </div>
              </template>
            </div>
          </div>
        </div>

        <!-- Synthesized Answer Card -->
        <div class="bg-slate-800/90 rounded-2xl p-6 border border-slate-700 shadow-xl">
          <div class="flex items-center justify-between mb-4 pb-3 border-b border-slate-700">
            <h3 class="text-sm font-bold text-white flex items-center">
              <i class="fa-solid fa-gavel text-indigo-400 mr-2"></i>
              Synthesized Legal Opinion & Grounded Ratio
            </h3>
            <span class="text-xs font-mono text-slate-400">Strictly Grounded in Delhi High Court Corpus</span>
          </div>
          <div class="prose prose-invert max-w-none text-slate-200 text-sm leading-relaxed font-serif whitespace-pre-wrap" x-text="ragResult?.answer_text"></div>
        </div>

        <!-- Evidence Citations Table -->
        <div class="bg-slate-800/80 rounded-2xl p-6 border border-slate-700 shadow-xl">
          <h3 class="text-sm font-bold text-white mb-4 flex items-center">
            <i class="fa-solid fa-book-bookmark text-indigo-400 mr-2"></i>
            Table of Authorities & Ground Truth Passages (<span x-text="ragResult?.citations?.length || 0"></span> Passages)
          </h3>
          <div class="space-y-3">
            <template x-for="cit in ragResult?.citations" :key="cit.chunk_id">
              <div class="bg-slate-900 p-4 rounded-xl border border-slate-800 text-xs">
                <div class="flex items-center justify-between mb-2">
                  <div class="flex items-center space-x-2 font-mono">
                    <span class="px-2 py-0.5 rounded bg-indigo-600 text-white font-bold text-xs" x-text="'[' + cit.citation_index + ']'"></span>
                    <span class="font-bold text-white font-sans text-xs" x-text="cit.case_title"></span>
                    <span class="text-slate-400" x-text="'(' + cit.court + ', ' + cit.year + ')'"></span>
                  </div>
                  <div class="flex items-center space-x-2">
                    <span class="px-2 py-0.5 rounded font-mono text-[11px] font-bold" :class="getRoleBadgeClass(cit.rhetorical_role)" x-text="cit.rhetorical_role.toUpperCase()"></span>
                    <span class="text-slate-400 font-mono text-[11px]" x-text="'Score: ' + cit.relevance_score.toFixed(3)"></span>
                  </div>
                </div>
                <p class="text-slate-300 font-serif leading-relaxed bg-slate-950/60 p-3 rounded border border-slate-800/60 text-xs" x-text="cit.text"></p>
                <div class="mt-2 flex items-center justify-between text-[11px] text-slate-400 font-mono">
                  <span>Doc ID: <span class="text-indigo-400" x-text="cit.doc_id"></span> | Chunk ID: <span class="text-slate-500" x-text="cit.chunk_id"></span></span>
                  <button @click="openHeadnote(cit.doc_id)" class="text-indigo-400 hover:text-indigo-300">View Headnote &rarr;</button>
                </div>
              </div>
            </template>
          </div>
        </div>
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB: SHEPARDIZE PRECEDENT CHECKER -->
    <!-- ================================================================= -->
    <div x-show="tab === 'shepardize'" x-cloak x-transition>
      <div class="bg-slate-800/80 rounded-2xl p-6 border border-slate-700 shadow-xl mb-8">
        <h2 class="text-base font-bold text-white mb-2 flex items-center">
          <i class="fa-solid fa-shield-halved text-indigo-400 mr-2"></i>
          Precedent Shepardization & Good-Law Checker
        </h2>
        <p class="text-xs text-slate-400 mb-6">Audits the judicial history of any legal precedent across subsequent High Court judgments to determine whether it remains binding good law, or has been distinguished, cautioned, or overruled.</p>

        <form @submit.prevent="executeShepardize">
          <div class="relative">
            <i class="fa-solid fa-quote-left absolute left-4 top-4 text-slate-400 text-lg"></i>
            <input 
              type="text" 
              x-model="shepCitation" 
              placeholder="Enter standard Indian legal citation (e.g. 'AIR 1978 SC 597' or '(2014) 8 SCC 273')..."
              class="w-full pl-12 pr-32 py-3.5 bg-slate-900 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-sm shadow-inner font-mono"
            />
            <button 
              type="submit" 
              :disabled="shepLoading"
              class="absolute right-2 top-2 px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow-md transition"
            >
              <span x-show="!shepLoading"><i class="fa-solid fa-magnifying-glass mr-1"></i>Check Law</span>
              <span x-show="shepLoading"><i class="fa-solid fa-spinner fa-spin mr-1"></i>Auditing...</span>
            </button>
          </div>

          <div class="mt-4 flex flex-wrap items-center gap-2 text-xs text-slate-400">
            <span>Quick sample citations:</span>
            <button type="button" @click="shepCitation = 'AIR 1978 SC 597'; executeShepardize()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 font-mono transition">AIR 1978 SC 597 (Maneka Gandhi)</button>
            <button type="button" @click="shepCitation = '(2014) 8 SCC 273'; executeShepardize()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 font-mono transition">(2014) 8 SCC 273 (Arnesh Kumar)</button>
            <button type="button" @click="shepCitation = 'AIR 1973 SC 1461'; executeShepardize()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 font-mono transition">AIR 1973 SC 1461 (Kesavananda)</button>
            <button type="button" @click="shepCitation = '2023:DHC:2692'; executeShepardize()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 font-mono transition">2023:DHC:2692</button>
          </div>
        </form>
      </div>

      <!-- Shepardization Result -->
      <div x-show="shepResult" class="space-y-6">
        <div class="bg-slate-800/90 rounded-2xl p-6 border border-slate-700 shadow-xl">
          <div class="flex flex-wrap items-start justify-between gap-4 pb-6 border-b border-slate-700">
            <div>
              <div class="text-xs font-mono text-slate-400 mb-1">Precedent Authority</div>
              <h3 class="text-lg font-bold text-white font-mono" x-text="shepResult?.precedent_citation"></h3>
              <div class="text-xs text-slate-400 font-mono mt-1">Canonical Key: <span class="text-indigo-400" x-text="shepResult?.canonical_id"></span></div>
            </div>

            <!-- Status Badge -->
            <div class="text-right">
              <div class="text-xs font-mono text-slate-400 mb-1">Judicial Standing</div>
              <span class="px-4 py-2 rounded-xl text-sm font-bold font-mono tracking-wide inline-block shadow-lg" :class="getShepBadgeClass(shepResult?.status)">
                <i class="fa-solid mr-1.5" :class="getShepIcon(shepResult?.status)"></i>
                <span x-text="shepResult?.status?.toUpperCase()?.replace('_', ' ')"></span>
              </span>
            </div>
          </div>

          <!-- Treatment Statistics Bar -->
          <div class="grid grid-cols-2 sm:grid-cols-4 gap-4 mt-6">
            <div class="bg-slate-900 p-4 rounded-xl border border-slate-800">
              <div class="text-[11px] uppercase font-mono text-slate-400">Total Citations</div>
              <div class="text-2xl font-bold text-white font-mono mt-1" x-text="shepResult?.total_citations || 0"></div>
            </div>
            <div class="bg-slate-900 p-4 rounded-xl border border-slate-800">
              <div class="text-[11px] uppercase font-mono text-slate-400">Relied Upon / Followed</div>
              <div class="text-2xl font-bold text-emerald-400 font-mono mt-1" x-text="shepResult?.treatment_summary?.relied_upon || 0"></div>
            </div>
            <div class="bg-slate-900 p-4 rounded-xl border border-slate-800">
              <div class="text-[11px] uppercase font-mono text-slate-400">Distinguished on Facts</div>
              <div class="text-2xl font-bold text-amber-400 font-mono mt-1" x-text="shepResult?.treatment_summary?.distinguished || 0"></div>
            </div>
            <div class="bg-slate-900 p-4 rounded-xl border border-slate-800">
              <div class="text-[11px] uppercase font-mono text-slate-400">Overruled / Disapproved</div>
              <div class="text-2xl font-bold text-rose-400 font-mono mt-1" x-text="shepResult?.treatment_summary?.overruled || 0"></div>
            </div>
          </div>

          <!-- Warning Message (if any) -->
          <div x-show="shepResult?.warning_message" class="mt-6 p-4 rounded-xl border" :class="shepResult?.is_good_law ? 'bg-amber-950/40 border-amber-800/80 text-amber-200' : 'bg-rose-950/40 border-rose-800/80 text-rose-200'">
            <div class="font-bold text-xs flex items-center mb-1">
              <i class="fa-solid fa-triangle-exclamation mr-1.5"></i>Judicial Caution Notice
            </div>
            <p class="text-xs font-mono" x-text="shepResult?.warning_message"></p>
          </div>
        </div>

        <!-- Citing Cases Stream -->
        <div class="bg-slate-800/80 rounded-2xl p-6 border border-slate-700 shadow-xl">
          <h3 class="text-sm font-bold text-white mb-4 flex items-center">
            <i class="fa-solid fa-landmark text-indigo-400 mr-2"></i>
            Subsequent Citing Judgments in Corpus (<span x-text="shepResult?.recent_treatments?.length || 0"></span> instances)
          </h3>
          <div class="space-y-3">
            <template x-for="t in shepResult?.recent_treatments" :key="t.citing_doc_id + t.treatment">
              <div class="bg-slate-900 p-4 rounded-xl border border-slate-800 text-xs">
                <div class="flex items-center justify-between mb-2">
                  <div>
                    <h4 class="font-bold text-white text-xs" x-text="t.citing_title"></h4>
                    <div class="text-slate-400 font-mono text-[11px]">
                      <span x-text="t.citing_court"></span> • <span x-text="'Year ' + t.citing_year"></span> • <span>Doc ID: </span><span class="text-indigo-400 font-bold" x-text="t.citing_doc_id"></span>
                    </div>
                  </div>
                  <span class="px-2.5 py-1 rounded font-mono text-[11px] font-bold" :class="getTreatmentBadgeClass(t.treatment)" x-text="t.treatment.toUpperCase().replace('_', ' ')"></span>
                </div>
                <p class="text-slate-300 font-serif leading-relaxed bg-slate-950/60 p-3 rounded border border-slate-800/60 text-xs" x-text="t.context"></p>
              </div>
            </template>
          </div>
        </div>
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB: LEGAL MEMORANDUM & COURT DRAFT GENERATOR -->
    <!-- ================================================================= -->
    <div x-show="tab === 'memo'" x-cloak x-transition>
      <div class="bg-slate-800/80 rounded-2xl p-6 border border-slate-700 shadow-xl mb-8">
        <h2 class="text-base font-bold text-white mb-2 flex items-center">
          <i class="fa-solid fa-file-contract text-indigo-400 mr-2"></i>
          Automated Legal Document & Court Pleading Generator
        </h2>
        <p class="text-xs text-slate-400 mb-6">Generate formal appellate chamber briefs, exhaustive legal research memorandums, or Section 439 CrPC regular bail applications grounded in corpus precedents.</p>

        <!-- Document Type Selector -->
        <div class="grid grid-cols-1 sm:grid-cols-3 gap-3 mb-6">
          <button 
            type="button" 
            @click="memoDocType = 'memo'" 
            :class="memoDocType === 'memo' ? 'bg-indigo-600 border-indigo-500 text-white' : 'bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800'"
            class="p-3.5 rounded-xl border text-left transition"
          >
            <div class="font-bold text-xs flex items-center"><i class="fa-solid fa-book-open mr-2"></i>Legal Memorandum</div>
            <div class="text-[11px] text-slate-400 mt-1">Exhaustive research memo on an issue with citations</div>
          </button>
          <button 
            type="button" 
            @click="memoDocType = 'brief'" 
            :class="memoDocType === 'brief' ? 'bg-indigo-600 border-indigo-500 text-white' : 'bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800'"
            class="p-3.5 rounded-xl border text-left transition"
          >
            <div class="font-bold text-xs flex items-center"><i class="fa-solid fa-briefcase mr-2"></i>Appellate Case Brief</div>
            <div class="text-[11px] text-slate-400 mt-1">Structured briefing from an ingested judgment ID</div>
          </button>
          <button 
            type="button" 
            @click="memoDocType = 'bail'" 
            :class="memoDocType === 'bail' ? 'bg-indigo-600 border-indigo-500 text-white' : 'bg-slate-900 border-slate-700 text-slate-300 hover:bg-slate-800'"
            class="p-3.5 rounded-xl border text-left transition"
          >
            <div class="font-bold text-xs flex items-center"><i class="fa-solid fa-scale-unbalanced mr-2"></i>Bail Application (CrPC 439)</div>
            <div class="text-[11px] text-slate-400 mt-1">High Court court-ready bail pleading draft</div>
          </button>
        </div>

        <form @submit.prevent="executeGenerateMemo">
          <!-- Form for Legal Memorandum -->
          <div x-show="memoDocType === 'memo'" class="space-y-4">
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Subject / Question for Legal Memorandum</label>
              <input type="text" x-model="memoQuery" placeholder="e.g. Section 34 Arbitration Act grounds of patent illegality" class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white text-xs placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500" />
            </div>
          </div>

          <!-- Form for Case Brief -->
          <div x-show="memoDocType === 'brief'" class="space-y-4">
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Judgment Document ID</label>
              <input type="text" x-model="memoDocId" placeholder="e.g. 104774187" class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white text-xs placeholder-slate-500 font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500" />
            </div>
          </div>

          <!-- Form for Bail Application -->
          <div x-show="memoDocType === 'bail'" class="space-y-4">
            <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Applicant / Accused Name</label>
                <input type="text" x-model="memoApplicant" placeholder="e.g. Ramesh Kumar" class="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-xs" />
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">FIR Number</label>
                <input type="text" x-model="memoFirNo" placeholder="e.g. 124/2023" class="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-xs" />
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Police Station</label>
                <input type="text" x-model="memoStation" placeholder="e.g. Special Cell, Lodhi Colony" class="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-xs" />
              </div>
              <div>
                <label class="block text-xs font-medium text-slate-300 mb-1">Custody Date</label>
                <input type="text" x-model="memoCustody" placeholder="e.g. 15.02.2023" class="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-xs" />
              </div>
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Sections Invoked (comma-separated)</label>
              <input type="text" x-model="memoSections" placeholder="e.g. Section 20, 29 NDPS Act" class="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-xs" />
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Factual Matrix Summary</label>
              <textarea x-model="memoFacts" rows="3" placeholder="Brief facts of the arrest and alleged recovery..." class="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white text-xs"></textarea>
            </div>
          </div>

          <div class="mt-6 flex justify-end">
            <button 
              type="submit" 
              :disabled="memoLoading"
              class="px-6 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-xl text-xs font-bold shadow-lg transition flex items-center"
            >
              <span x-show="!memoLoading"><i class="fa-solid fa-file-arrow-down mr-1.5"></i>Generate Document</span>
              <span x-show="memoLoading"><i class="fa-solid fa-spinner fa-spin mr-1.5"></i>Drafting Legal Document...</span>
            </button>
          </div>
        </form>
      </div>

      <!-- Document Preview Output Card -->
      <div x-show="memoResult" class="bg-slate-800/90 rounded-2xl p-6 border border-slate-700 shadow-xl space-y-4">
        <div class="flex flex-wrap items-center justify-between gap-4 pb-4 border-b border-slate-700">
          <div>
            <h3 class="text-base font-bold text-white font-sans" x-text="memoResult?.title"></h3>
            <div class="text-xs text-slate-400 font-mono mt-1" x-text="'Generated: ' + memoResult?.generated_at"></div>
          </div>
          <div class="flex space-x-2">
            <button @click="copyMemoText" class="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded-lg text-xs font-medium transition flex items-center">
              <i class="fa-solid fa-copy mr-1.5"></i>
              <span x-text="copySuccess ? 'Copied!' : 'Copy Text'"></span>
            </button>
            <button @click="downloadMemoText" class="px-4 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-lg text-xs font-medium transition flex items-center">
              <i class="fa-solid fa-download mr-1.5"></i>Download .txt
            </button>
          </div>
        </div>

        <pre class="bg-slate-950 p-6 rounded-xl border border-slate-800 text-slate-200 font-mono text-xs leading-relaxed whitespace-pre-wrap max-h-[60vh] overflow-y-auto" x-text="memoResult?.content"></pre>
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB: CITATION GRAPH & LANDMARK PRECEDENTS -->
    <!-- ================================================================= -->
    <div x-show="tab === 'precedents'" x-cloak x-transition>
      <!-- Interactive Neighborhood Visualizer Card -->
      <div class="bg-slate-800/90 rounded-2xl p-6 border border-slate-700 shadow-xl mb-8">
        <div class="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h3 class="text-base font-bold text-white flex items-center">
              <i class="fa-solid fa-circle-nodes text-indigo-400 mr-2"></i>
              Interactive Citation Graph Neighborhood Visualizer
            </h3>
            <p class="text-xs text-slate-400">Explore multi-hop citation paths and treatment relationships for any judgment in the corpus.</p>
          </div>
          <div class="flex items-center space-x-2">
            <input 
              type="text" 
              x-model="graphDocId" 
              placeholder="Enter Document ID (e.g. 104774187)..." 
              class="px-3 py-1.5 bg-slate-900 border border-slate-700 rounded-lg text-white font-mono text-xs w-48"
            />
            <button 
              @click="loadGraphVisualization()" 
              :disabled="graphLoading"
              class="px-4 py-1.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow transition flex items-center"
            >
              <span x-show="!graphLoading"><i class="fa-solid fa-play mr-1"></i>Visualize</span>
              <span x-show="graphLoading"><i class="fa-solid fa-spinner fa-spin mr-1"></i>Loading...</span>
            </button>
          </div>
        </div>

        <!-- Graph Metrics Bar -->
        <div x-show="graphData" class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-4">
          <div class="bg-slate-900 p-3 rounded-lg border border-slate-800">
            <div class="text-[10px] uppercase font-mono text-slate-400">Root Judgment</div>
            <div class="text-sm font-bold text-white font-mono mt-0.5" x-text="graphData?.root_id"></div>
          </div>
          <div class="bg-slate-900 p-3 rounded-lg border border-slate-800">
            <div class="text-[10px] uppercase font-mono text-slate-400">Total Graph Nodes</div>
            <div class="text-sm font-bold text-indigo-400 font-mono mt-0.5" x-text="graphData?.stats?.total_nodes || 0"></div>
          </div>
          <div class="bg-slate-900 p-3 rounded-lg border border-slate-800">
            <div class="text-[10px] uppercase font-mono text-slate-400">Citation Edges</div>
            <div class="text-sm font-bold text-emerald-400 font-mono mt-0.5" x-text="graphData?.stats?.total_links || 0"></div>
          </div>
          <div class="bg-slate-900 p-3 rounded-lg border border-slate-800">
            <div class="text-[10px] uppercase font-mono text-slate-400">Relied Upon Authorities</div>
            <div class="text-sm font-bold text-cyan-400 font-mono mt-0.5" x-text="graphData?.stats?.treatment_breakdown?.relied_upon || 0"></div>
          </div>
        </div>

        <!-- SVG Force Graph Canvas -->
        <div x-show="graphData" class="bg-slate-950 rounded-xl border border-slate-800 p-4 relative overflow-hidden">
          <svg class="w-full h-72 bg-slate-950" viewBox="0 0 800 300">
            <!-- Draw Links -->
            <template x-for="(link, lIdx) in (graphData?.links || [])" :key="'link-'+lIdx">
              <line 
                :x1="400" 
                :y1="150" 
                :x2="getGraphNodeX(lIdx, graphData.links.length)" 
                :y2="getGraphNodeY(lIdx, graphData.links.length)" 
                :stroke="getTreatmentStroke(link.treatment)" 
                stroke-width="1.8" 
                stroke-dasharray="3 2"
              />
            </template>

            <!-- Center Root Node -->
            <g transform="translate(400, 150)" class="cursor-pointer" @click="selectedGraphNode = graphData.nodes[0]">
              <circle r="26" fill="#4f46e5" stroke="#818cf8" stroke-width="3" />
              <text text-anchor="middle" dy="4" fill="#ffffff" font-size="10" font-family="monospace" font-weight="bold">ROOT</text>
            </g>

            <!-- Precedent Target Nodes -->
            <template x-for="(link, lIdx) in (graphData?.links || [])" :key="'node-'+lIdx">
              <g 
                :transform="'translate(' + getGraphNodeX(lIdx, graphData.links.length) + ',' + getGraphNodeY(lIdx, graphData.links.length) + ')'" 
                class="cursor-pointer transition hover:opacity-80"
                @click="selectedGraphNode = getNodeById(link.target)"
              >
                <circle r="18" :fill="getTreatmentColor(link.treatment)" stroke="#ffffff" stroke-width="1.5" />
                <text text-anchor="middle" dy="3" fill="#ffffff" font-size="8" font-family="monospace" font-weight="bold" x-text="'P' + (lIdx + 1)"></text>
                <text text-anchor="middle" dy="28" fill="#94a3b8" font-size="9" font-family="monospace" x-text="(link.target || '').substring(0, 14)"></text>
              </g>
            </template>
          </svg>

          <!-- Selected Node Details Overlay -->
          <div x-show="selectedGraphNode" class="mt-3 p-3 bg-slate-900/90 rounded-lg border border-slate-700 text-xs font-mono flex items-center justify-between">
            <div>
              <span class="text-slate-400">Selected Node: </span>
              <span class="text-white font-bold" x-text="selectedGraphNode?.title || selectedGraphNode?.label"></span>
              <span class="ml-2 px-2 py-0.5 rounded text-[10px]" :class="selectedGraphNode?.type === 'judgment' ? 'bg-indigo-950 text-indigo-300 border border-indigo-800' : 'bg-emerald-950 text-emerald-300 border border-emerald-800'" x-text="selectedGraphNode?.type?.toUpperCase()"></span>
            </div>
            <button @click="selectedGraphNode = null" class="text-slate-400 hover:text-white"><i class="fa-solid fa-xmark"></i></button>
          </div>
        </div>
      </div>

      <div class="bg-slate-800/80 rounded-2xl p-6 border border-slate-700 shadow-xl">
        <h2 class="text-base font-bold text-white mb-2 flex items-center">
          <i class="fa-solid fa-network-wired text-indigo-400 mr-2"></i>
          Precedent Citation Network & Landmark Authority Rankings
        </h2>
        <p class="text-xs text-slate-400 mb-6">Top cited legal authorities ranked by in-degree citation count across the 203 Delhi High Court judgments (2,854 directed edges).</p>

        <div class="overflow-x-auto">
          <table class="w-full text-left text-xs text-slate-300">
            <thead class="bg-slate-900/80 uppercase font-mono text-[11px] text-slate-400 border-b border-slate-700">
              <tr>
                <th class="py-3 px-4">Rank</th>
                <th class="py-3 px-4">Precedent Citation</th>
                <th class="py-3 px-4">Law Reporter</th>
                <th class="py-3 px-4">Case Title / Authority</th>
                <th class="py-3 px-4 text-right">In-Degree Citations</th>
                <th class="py-3 px-4">Primary Treatment</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-slate-800 font-mono">
              <template x-for="(prec, idx) in precedents" :key="prec.precedent_id">
                <tr class="hover:bg-slate-800/50 transition">
                  <td class="py-3 px-4 font-bold text-indigo-400" x-text="'#' + (idx + 1)"></td>
                  <td class="py-3 px-4 font-bold text-white" x-text="prec.canonical_citation"></td>
                  <td class="py-3 px-4 text-slate-400" x-text="prec.reporter"></td>
                  <td class="py-3 px-4 text-slate-200 font-sans" x-text="prec.case_title || 'Supreme Court Landmark'"></td>
                  <td class="py-3 px-4 text-right font-bold text-emerald-400" x-text="prec.in_degree"></td>
                  <td class="py-3 px-4">
                    <span class="px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 text-[10px]">RELIED UPON</span>
                  </td>
                </tr>
              </template>
            </tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB: STATUTE-TO-RATIO REVERSE LOOKUP -->
    <!-- ================================================================= -->
    <div x-show="tab === 'statute'" x-cloak x-transition>
      <div class="bg-slate-800/80 rounded-2xl p-6 border border-slate-700 shadow-xl mb-8">
        <h2 class="text-base font-bold text-white mb-2 flex items-center">
          <i class="fa-solid fa-book-bookmark text-indigo-400 mr-2"></i>
          Statute-to-Ratio Reverse Lookup Engine
        </h2>
        <p class="text-xs text-slate-400 mb-6">Given any Indian statutory provision (Act name + Section number), retrieves all judicial decisions, ratio decidendi passages, and synthesized legal doctrines interpreting that section.</p>

        <form @submit.prevent="executeStatuteLookup">
          <div class="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Act / Statute Name or Acronym</label>
              <input 
                type="text" 
                x-model="statuteAct" 
                placeholder="e.g. 'CrPC', 'IPC', 'Arbitration Act'..."
                class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm font-mono"
              />
            </div>
            <div>
              <label class="block text-xs font-medium text-slate-300 mb-1">Section Number</label>
              <input 
                type="text" 
                x-model="statuteSection" 
                placeholder="e.g. '482', '438', '34'..."
                class="w-full px-4 py-2.5 bg-slate-900 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 text-sm font-mono"
              />
            </div>
            <div class="flex items-end">
              <button 
                type="submit" 
                :disabled="statuteLoading"
                class="w-full py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-xl text-xs font-bold shadow-lg transition flex items-center justify-center"
              >
                <span x-show="!statuteLoading"><i class="fa-solid fa-magnifying-glass mr-1.5"></i>Search Ratio Decidendi</span>
                <span x-show="statuteLoading"><i class="fa-solid fa-spinner fa-spin mr-1.5"></i>Searching...</span>
              </button>
            </div>
          </div>

          <div class="flex flex-wrap items-center gap-2 text-xs text-slate-400">
            <span>Quick samples:</span>
            <button type="button" @click="statuteAct = 'CrPC'; statuteSection = '482'; executeStatuteLookup()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 font-mono transition">Section 482 CrPC (Quashing)</button>
            <button type="button" @click="statuteAct = 'Arbitration'; statuteSection = '34'; executeStatuteLookup()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 font-mono transition">Section 34 Arbitration (Setting Aside)</button>
            <button type="button" @click="statuteAct = 'IPC'; statuteSection = '302'; executeStatuteLookup()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 font-mono transition">Section 302 IPC (Murder)</button>
            <button type="button" @click="statuteAct = 'Constitution'; statuteSection = '226'; executeStatuteLookup()" class="px-2 py-1 bg-slate-700/60 hover:bg-slate-700 rounded text-slate-300 font-mono transition">Article 226 (Writ Jurisdiction)</button>
          </div>
        </form>
      </div>

      <!-- Results Stream -->
      <div x-show="statuteResult" class="space-y-6">
        <!-- Key Principles Summary -->
        <div class="bg-slate-800/90 rounded-2xl p-6 border border-slate-700 shadow-xl">
          <div class="flex items-center justify-between pb-4 border-b border-slate-700 mb-4">
            <div>
              <span class="text-xs font-mono text-indigo-400 uppercase tracking-wide">Synthesized Judicial Principles</span>
              <h3 class="text-base font-bold text-white mt-0.5">
                <span x-text="statuteResult?.canonical_act"></span> — Section <span x-text="statuteResult?.section"></span>
              </h3>
            </div>
            <div class="text-right font-mono text-xs text-slate-400">
              Found <span class="text-emerald-400 font-bold" x-text="statuteResult?.total_results || 0"></span> interpretations
            </div>
          </div>

          <div class="space-y-2">
            <template x-for="(principle, pIdx) in (statuteResult?.key_principles || [])" :key="'p-'+pIdx">
              <div class="flex items-start space-x-3 p-3 bg-slate-900/80 rounded-xl border border-slate-800 text-xs text-slate-200 font-sans">
                <i class="fa-solid fa-gavel text-indigo-400 mt-0.5"></i>
                <span x-text="principle"></span>
              </div>
            </template>
          </div>
        </div>

        <!-- Matching Judicial Passages -->
        <div class="bg-slate-800/80 rounded-2xl p-6 border border-slate-700 shadow-xl space-y-4">
          <h3 class="text-sm font-bold text-white mb-2 flex items-center">
            <i class="fa-solid fa-scale-balanced text-indigo-400 mr-2"></i>
            Judicial Decisions Interpreting Provision
          </h3>

          <div class="space-y-4">
            <template x-for="(item, idx) in (statuteResult?.results || [])" :key="item.doc_id + '-' + idx">
              <div class="p-4 bg-slate-900 rounded-xl border border-slate-800 hover:border-slate-700 transition space-y-2">
                <div class="flex flex-wrap items-center justify-between gap-2">
                  <div class="font-bold text-xs text-white" x-text="item.case_title"></div>
                  <div class="flex items-center space-x-2">
                    <span class="px-2 py-0.5 rounded text-[10px] font-mono uppercase" :class="getRoleBadgeClass(item.rhetorical_role)" x-text="item.rhetorical_role"></span>
                    <span class="text-[10px] font-mono text-slate-400" x-text="item.court + ' (' + item.year + ')'"></span>
                  </div>
                </div>

                <p class="text-xs text-slate-300 font-serif leading-relaxed" x-text="item.ratio_text"></p>

                <div x-show="item.headnote_ratio" class="mt-2 p-2.5 bg-slate-950 rounded-lg border border-slate-800 text-[11px] text-slate-400 font-mono">
                  <span class="text-indigo-400 font-bold">Headnote Ratio: </span>
                  <span x-text="item.headnote_ratio"></span>
                </div>
              </div>
            </template>
          </div>
        </div>
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB 3: BENCHMARK QA TESTBED -->
    <!-- ================================================================= -->
    <div x-show="tab === 'benchmark'" x-cloak x-transition>
      <div class="bg-slate-800/80 rounded-2xl p-6 border border-slate-700 shadow-xl" x-data="{ reportData: null }">
        <h2 class="text-base font-bold text-white mb-2 flex items-center">
          <i class="fa-solid fa-chart-simple text-indigo-400 mr-2"></i>
          Step 10 Retrieval Benchmark Empirical Results
        </h2>
        <p class="text-xs text-slate-400 mb-6">Empirical comparison across 20 golden legal evaluation queries (Statutory, Precedent, Factual, Procedural) at ranking depth K=10.</p>

        <div class="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div class="bg-slate-900 p-4 rounded-xl border border-slate-700/80">
            <div class="text-xs text-slate-400 uppercase font-mono">Hybrid Hit Rate @ 5</div>
            <div class="text-2xl font-bold text-emerald-400 mt-1 font-mono">70.0%</div>
            <div class="text-[11px] text-emerald-500 mt-1">+7.7% lift over BM25 (65%)</div>
          </div>
          <div class="bg-slate-900 p-4 rounded-xl border border-slate-700/80">
            <div class="text-xs text-slate-400 uppercase font-mono">Hybrid Hit Rate @ 10</div>
            <div class="text-2xl font-bold text-emerald-400 mt-1 font-mono">85.0%</div>
            <div class="text-[11px] text-slate-400 mt-1">17 / 20 queries found</div>
          </div>
          <div class="bg-slate-900 p-4 rounded-xl border border-slate-700/80">
            <div class="text-xs text-slate-400 uppercase font-mono">Hybrid MRR</div>
            <div class="text-2xl font-bold text-indigo-400 mt-1 font-mono">0.555</div>
            <div class="text-[11px] text-slate-400 mt-1">First rank average ~1.8</div>
          </div>
          <div class="bg-slate-900 p-4 rounded-xl border border-slate-700/80">
            <div class="text-xs text-slate-400 uppercase font-mono">Average Query Latency</div>
            <div class="text-2xl font-bold text-cyan-400 mt-1 font-mono">11.69 ms</div>
            <div class="text-[11px] text-slate-400 mt-1">Sub-15ms real-time CPU</div>
          </div>
        </div>

        <div class="bg-slate-900 p-4 rounded-xl border border-slate-800 font-mono text-xs text-slate-300">
          <div class="font-bold text-slate-200 mb-2">Benchmark Parquet Deliverable:</div>
          <div class="text-indigo-400">D:/LegalTrace/data/indiankanoon_2023_evaluation_report.parquet (60 rows, 21 columns)</div>
        </div>
      </div>
    </div>

    <!-- ================================================================= -->
    <!-- TAB 8: TARGETED KEYWORD JUDGMENT SCRAPER -->
    <!-- ================================================================= -->
    <div x-show="tab === 'scrape'" x-cloak x-transition>
      <div class="bg-slate-800/80 rounded-2xl p-6 border border-slate-700 shadow-xl mb-8">
        <div class="flex items-center justify-between mb-4">
          <div>
            <h2 class="text-base font-bold text-white flex items-center">
              <i class="fa-solid fa-bullseye text-indigo-400 mr-2"></i>
              Targeted Judgment Scraper
            </h2>
            <p class="text-xs text-slate-400 mt-1">
              Scrape judgments filtered by specific keywords, sections, and legal doctrines directly from Indian Kanoon without crawling full court calendar indices.
            </p>
          </div>
          <div class="hidden sm:flex items-center space-x-2 text-xs text-slate-400 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700">
            <i class="fa-solid fa-bolt text-amber-400"></i>
            <span>Indian Kanoon Solr/Lucene Engine</span>
          </div>
        </div>

        <!-- Query Form -->
        <form @submit.prevent="executeTargetedScrape" class="space-y-4">
          <div class="relative">
            <i class="fa-solid fa-magnifying-glass absolute left-4 top-4 text-slate-400 text-lg"></i>
            <input 
              type="text" 
              x-model="scrapeQuery" 
              placeholder="Enter targeted legal keywords, sections, or case terms (e.g. 'Section 37 NDPS commercial quantity bail')..."
              class="w-full pl-12 pr-32 py-3.5 bg-slate-900 border border-slate-700 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent text-sm shadow-inner"
              required
            />
            <button 
              type="submit" 
              :disabled="scrapeLoading"
              class="absolute right-2 top-2 px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow-md transition"
            >
              <span x-show="!scrapeLoading"><i class="fa-solid fa-satellite-dish mr-1.5"></i>Scrape Now</span>
              <span x-show="scrapeLoading"><i class="fa-solid fa-spinner fa-spin mr-1.5"></i>Scraping...</span>
            </button>
          </div>

          <!-- Parameter Selectors -->
          <div class="flex flex-wrap items-center gap-3 text-xs text-slate-300">
            <!-- Court Selector -->
            <div class="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700">
              <span class="text-slate-400 font-medium">Court:</span>
              <select x-model="scrapeCourt" class="bg-transparent text-slate-200 focus:outline-none cursor-pointer">
                <option value="Delhi High Court" class="bg-slate-800">Delhi High Court</option>
                <option value="Supreme Court of India" class="bg-slate-800">Supreme Court of India</option>
                <option value="Bombay High Court" class="bg-slate-800">Bombay High Court</option>
                <option value="Calcutta High Court" class="bg-slate-800">Calcutta High Court</option>
                <option value="Karnataka High Court" class="bg-slate-800">Karnataka High Court</option>
              </select>
            </div>

            <!-- Year Selector -->
            <div class="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700">
              <span class="text-slate-400 font-medium">Year:</span>
              <select x-model.number="scrapeYear" class="bg-transparent text-slate-200 focus:outline-none cursor-pointer">
                <option value="2024" class="bg-slate-800">2024</option>
                <option value="2023" class="bg-slate-800">2023</option>
                <option value="2022" class="bg-slate-800">2022</option>
                <option value="2021" class="bg-slate-800">2021</option>
                <option value="2020" class="bg-slate-800">2020</option>
                <option value="0" class="bg-slate-800">All Years (Unrestricted)</option>
              </select>
            </div>

            <!-- Limit Selector -->
            <div class="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700">
              <span class="text-slate-400 font-medium">Max Limit:</span>
              <select x-model.number="scrapeLimit" class="bg-transparent text-slate-200 focus:outline-none cursor-pointer">
                <option value="2">2 judgments (Fastest)</option>
                <option value="3" selected>3 judgments (Recommended)</option>
                <option value="5">5 judgments</option>
                <option value="10">10 judgments</option>
              </select>
            </div>

            <!-- Auto-Fetch Toggle -->
            <label class="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-700 cursor-pointer" title="Automatically download full text, extract rhetorical roles, and index into RAG vector engine">
              <input type="checkbox" x-model="scrapeAutoFetch" class="rounded text-indigo-600 focus:ring-0">
              <span :class="scrapeAutoFetch ? 'text-emerald-400 font-bold' : 'text-slate-400'"><i class="fa-solid fa-bolt mr-1 text-amber-400"></i>Auto-Download & Index to RAG</span>
            </label>
          </div>

          <!-- Quick Presets -->
          <div class="flex items-center space-x-2 text-xs">
            <span class="text-slate-500 font-medium">Quick Presets:</span>
            <button type="button" @click="scrapeQuery = 'Section 37 NDPS commercial quantity bail'" class="px-2 py-1 bg-slate-900 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition">NDPS S.37 Bail</button>
            <button type="button" @click="scrapeQuery = 'Section 34 Arbitration patent illegality'" class="px-2 py-1 bg-slate-900 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition">Arbitration S.34</button>
            <button type="button" @click="scrapeQuery = 'Section 482 CrPC quashing FIR matrimonial dispute'" class="px-2 py-1 bg-slate-900 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition">S.482 Quashing</button>
            <button type="button" @click="scrapeQuery = 'anticipatory bail PMLA Section 45'" class="px-2 py-1 bg-slate-900 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition">PMLA S.45 Bail</button>
            <button type="button" @click="scrapeQuery = 'Article 15(3) Domestic Violence Act protection'" class="px-2 py-1 bg-slate-900 hover:bg-slate-700 text-slate-300 rounded border border-slate-700 transition">Domestic Violence Art.15(3)</button>
          </div>
        </form>
      </div>

      <!-- Scrape Status & Telemetry Banner -->
      <div x-show="scrapeData" class="mb-6 bg-slate-800/60 rounded-xl p-4 border border-slate-700 flex flex-wrap items-center justify-between gap-4">
        <div class="flex items-center space-x-3">
          <div class="h-8 w-8 rounded-lg bg-indigo-900/60 border border-indigo-700 flex items-center justify-center text-indigo-400 font-bold">
            <i class="fa-solid fa-list-check"></i>
          </div>
          <div>
            <div class="text-xs font-bold text-white">
              Targeted Discovery Completed: <span class="text-indigo-400" x-text="scrapeData?.total_discovered || 0"></span> judgments found
            </div>
            <div class="text-[11px] text-slate-400 font-mono">
              Query: "<span x-text="scrapeData?.query"></span>" | Court: <span x-text="scrapeData?.court"></span> (<span x-text="scrapeData?.year || 'All'"></span>)
            </div>
          </div>
        </div>

        <div class="flex items-center space-x-4 text-xs font-mono">
          <div x-show="scrapeData?.auto_fetch" class="px-2.5 py-1 rounded bg-emerald-950 text-emerald-300 border border-emerald-800">
            <i class="fa-solid fa-download mr-1"></i>
            Stored to Raw-Store: <span class="font-bold" x-text="scrapeData?.fetched_count || 0"></span>
          </div>
          <div x-show="scrapeData?.indexed_chunks_count" class="px-2.5 py-1 rounded bg-indigo-950 text-indigo-300 border border-indigo-800">
            <i class="fa-solid fa-bolt mr-1"></i>
            Indexed to Search: <span class="font-bold" x-text="scrapeData?.indexed_chunks_count || 0"></span> chunks
          </div>
          <div class="text-slate-400">
            Job ID: <span class="text-slate-200" x-text="scrapeData?.job_id || 'discovery_only'"></span>
          </div>
        </div>
      </div>

      <!-- Loading State -->
      <div x-show="scrapeLoading" class="py-16 text-center">
        <i class="fa-solid fa-circle-notch fa-spin text-4xl text-indigo-500 mb-4"></i>
        <p class="text-sm font-medium text-slate-300">Connecting to Indian Kanoon & Processing...</p>
        <p class="text-xs text-slate-500 mt-1" x-show="scrapeAutoFetch">Bypassing Cloudflare, downloading full judgment text, extracting rhetorical roles, and updating RAG vector index...</p>
        <p class="text-xs text-slate-500 mt-1" x-show="!scrapeAutoFetch">Filtering court decisions by keyword & Lucene parameters.</p>
      </div>

      <!-- Results Grid -->
      <div x-show="!scrapeLoading && scrapeData && scrapeData.results && scrapeData.results.length > 0" class="space-y-4">
        <template x-for="item in (scrapeData?.results || [])" :key="item.doc_id">
          <div class="p-5 bg-slate-800/80 rounded-2xl border border-slate-700 hover:border-slate-600 transition shadow-lg space-y-3">
            <div class="flex flex-wrap items-start justify-between gap-2">
              <div class="space-y-1">
                <div class="flex items-center space-x-2">
                  <span class="text-xs font-mono px-2 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 font-bold" x-text="'Doc #' + item.doc_id"></span>
                  <span class="text-xs text-slate-400 font-mono" x-text="item.court + (item.year ? ' • ' + item.year : '')"></span>
                </div>
                <h3 class="text-sm font-bold text-white hover:text-indigo-300 transition" x-text="item.title"></h3>
              </div>
              <div class="flex items-center space-x-2">
                <!-- If not yet downloaded & indexed in RAG -->
                <template x-if="!isDocIndexed(item.doc_id)">
                  <button 
                    type="button" 
                    @click="downloadAndIndexDoc(item)" 
                    :disabled="downloadingDocId === item.doc_id"
                    class="px-3 py-1 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-lg text-xs font-semibold shadow transition flex items-center space-x-1.5"
                  >
                    <span x-show="downloadingDocId !== item.doc_id"><i class="fa-solid fa-download mr-1"></i>Download & Index to RAG</span>
                    <span x-show="downloadingDocId === item.doc_id"><i class="fa-solid fa-spinner fa-spin mr-1"></i>Indexing...</span>
                  </button>
                </template>

                <!-- If already downloaded & indexed in RAG -->
                <template x-if="isDocIndexed(item.doc_id)">
                  <div class="flex items-center space-x-2">
                    <span class="px-2.5 py-1 bg-emerald-950 text-emerald-300 border border-emerald-800 rounded-lg text-xs font-mono font-bold flex items-center">
                      <i class="fa-solid fa-circle-check text-emerald-400 mr-1.5"></i>Ready in RAG
                    </span>
                    <button 
                      type="button" 
                      @click="askRAGAboutDoc(item)"
                      class="px-3 py-1 bg-indigo-900/70 hover:bg-indigo-800 text-indigo-200 rounded-lg text-xs font-semibold border border-indigo-700 transition flex items-center space-x-1"
                    >
                      <i class="fa-solid fa-wand-magic-sparkles mr-1 text-indigo-400"></i>Ask in RAG &rarr;
                    </button>
                  </div>
                </template>

                <a :href="item.source_url" target="_blank" class="px-3 py-1 bg-slate-900 hover:bg-slate-700 text-slate-300 rounded-lg text-xs font-medium border border-slate-700 transition flex items-center space-x-1.5">
                  <i class="fa-solid fa-arrow-up-right-from-square text-[10px]"></i>
                  <span>Indian Kanoon</span>
                </a>
              </div>
            </div>

            <!-- Snippet / Headline if available -->
            <div x-show="item.headline" class="p-3 bg-slate-950 rounded-xl border border-slate-800/80 text-xs text-slate-300 font-serif leading-relaxed">
              <div class="text-[10px] font-mono text-indigo-400 uppercase tracking-wider mb-1 flex items-center">
                <i class="fa-solid fa-highlighter mr-1"></i> SERP Keyword Context Snippet
              </div>
              <p x-html="item.headline"></p>
            </div>
      </div>

      <!-- Empty State -->
      <div x-show="!scrapeLoading && scrapeData && (!scrapeData.results || scrapeData.results.length === 0)" class="py-16 text-center bg-slate-800/40 rounded-2xl border border-slate-700/60">
        <i class="fa-solid fa-folder-open text-4xl text-slate-600 mb-3"></i>
        <p class="text-sm text-slate-400 font-medium">No judgments found matching this keyword and court combination.</p>
        <p class="text-xs text-slate-500 mt-1">Try broadening your search query or selecting "All Years".</p>
      </div>
    </div>

  </main>

  <!-- ================================================================= -->
  <!-- MODAL: LEGAL HEADNOTE / SYLLABUS VIEWER -->
  <!-- ================================================================= -->
  <div x-show="activeHeadnote" x-cloak class="fixed inset-0 z-50 overflow-y-auto bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
    <div @click.away="activeHeadnote = null" class="bg-slate-900 border border-slate-700 rounded-2xl max-w-4xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden">
      <!-- Header -->
      <div class="px-6 py-4 border-b border-slate-800 flex items-center justify-between bg-slate-950">
        <div>
          <span class="text-xs font-mono font-bold text-indigo-400 uppercase tracking-wide">Automated Legal Headnote</span>
          <h2 class="text-sm font-bold text-white" x-text="activeHeadnote?.case_title"></h2>
        </div>
        <button @click="activeHeadnote = null" class="text-slate-400 hover:text-white text-lg">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>

      <!-- Body -->
      <div class="p-6 overflow-y-auto space-y-6 text-xs text-slate-200 font-serif leading-relaxed">
        <!-- Catchwords -->
        <div>
          <span class="font-sans font-bold text-slate-400 uppercase tracking-wide block mb-1">Catchwords Taxonomy</span>
          <div class="bg-indigo-950/40 p-3 rounded-lg border border-indigo-900/60 font-mono text-[11px] text-indigo-300" x-text="activeHeadnote?.catchwords_line"></div>
        </div>

        <!-- Facts -->
        <div>
          <span class="font-sans font-bold text-slate-400 uppercase tracking-wide block mb-1">Brief Facts of the Case</span>
          <p class="bg-slate-950 p-4 rounded-lg border border-slate-800 text-slate-300" x-text="activeHeadnote?.brief_facts || 'No factual matrix extracted.'"></p>
        </div>

        <!-- Ratio Decidendi -->
        <div x-show="activeHeadnote?.ratio_decidendi">
          <span class="font-sans font-bold text-slate-400 uppercase tracking-wide block mb-1">Ratio Decidendi & Legal Findings</span>
          <p class="bg-slate-950 p-4 rounded-lg border border-slate-800 text-amber-200/90 font-serif" x-text="activeHeadnote?.ratio_decidendi"></p>
        </div>

        <!-- Operative Holding -->
        <div>
          <span class="font-sans font-bold text-slate-400 uppercase tracking-wide block mb-1">Operative Holding & Final Relief</span>
          <p class="bg-slate-950 p-4 rounded-lg border border-slate-800 text-emerald-300 font-mono" x-text="activeHeadnote?.operative_holding"></p>
        </div>

        <!-- Full Formatted Syllabus -->
        <div>
          <span class="font-sans font-bold text-slate-400 uppercase tracking-wide block mb-1">Law-Reporter Formatted Text</span>
          <pre class="bg-slate-950 p-4 rounded-lg border border-slate-800 text-[11px] font-mono text-slate-300 whitespace-pre-wrap" x-text="activeHeadnote?.formatted_headnote"></pre>
        </div>
      </div>

      <!-- Footer -->
      <div class="px-6 py-3 border-t border-slate-800 bg-slate-950 flex justify-end">
        <button @click="activeHeadnote = null" class="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs">Close</button>
      </div>
    </div>
  </div>

  <!-- JavaScript Application Controller -->
  <script>
    function explorerApp() {
      return {
        tab: 'search',
        query: 'Section 37 NDPS Act commercial quantity bail conditions',
        mode: 'hybrid',
        roleFilter: '',
        topK: 10,
        loading: false,
        searchResults: [],
        searchLatency: 0,
        totalIndexed: 9167,
        statusText: 'Index Active',
        precedents: [],
        activeHeadnote: null,

        // RAG State
        ragQuery: 'What are the mandatory conditions for regular bail involving commercial quantity under Section 37 NDPS Act?',
        ragMaxChunks: 8,
        ragLoading: false,
        ragResult: null,
        showPlan: false,

        // Shepardize State
        shepCitation: 'AIR 1978 SC 597',
        shepLoading: false,
        shepResult: null,

        // Memo Generator State
        memoDocType: 'memo',
        memoQuery: 'Section 34 Arbitration Act grounds of patent illegality and public policy',
        memoDocId: '104774187',
        memoApplicant: 'Ramesh Kumar',
        memoFirNo: '142/2023',
        memoStation: 'Special Cell, New Delhi',
        memoSections: 'Section 20(b)(ii)(C), Section 29 NDPS Act',
        memoCustody: '10.02.2023',
        memoFacts: 'Applicant was arrested on alleged recovery of contraband from co-accused. No direct recovery was effected from the personal possession of the applicant.',
        memoLoading: false,
        memoResult: null,
        copySuccess: false,

        // Targeted Scraper State
        scrapeQuery: 'Article 15(3) Domestic Violence Act protection',
        scrapeCourt: 'Delhi High Court',
        scrapeYear: 2023,
        scrapeLimit: 3,
        scrapeAutoFetch: true,
        scrapeLoading: false,
        scrapeData: null,
        downloadingDocId: null,
        indexedDocIds: [],

        async init() {
          this.executeSearch();
          this.loadIndexedDocIds();
        },

        async loadIndexedDocIds() {
          try {
            const res = await fetch('/api/scrape/indexed-doc-ids');
            if (res.ok) {
              const ids = await res.json();
              this.indexedDocIds = (ids || []).map(x => String(x));
            }
          } catch (e) {
            console.error('Failed loading indexed doc ids:', e);
          }
        },

        async executeSearch() {
          if (!this.query.trim()) return;
          this.loading = true;
          try {
            const res = await fetch('/api/search', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                query: this.query,
                mode: this.mode,
                top_k: this.topK,
                filter_role: this.roleFilter || null
              })
            });
            if (!res.ok) throw new Error('Search failed');
            const data = await res.json();
            this.searchResults = data.results || [];
            this.searchLatency = data.latency_ms;
            this.totalIndexed = data.total_indexed;
          } catch (err) {
            console.error(err);
          } finally {
            this.loading = false;
          }
        },

        async executeRAG() {
          if (!this.ragQuery.trim()) return;
          this.ragLoading = true;
          try {
            const res = await fetch('/api/rag/query', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                query: this.ragQuery,
                max_evidence_chunks: this.ragMaxChunks
              })
            });
            if (!res.ok) throw new Error('RAG query failed');
            this.ragResult = await res.json();
          } catch (err) {
            console.error(err);
            alert('Failed to execute RAG reasoning: ' + err.message);
          } finally {
            this.ragLoading = false;
          }
        },

        async executeShepardize() {
          if (!this.shepCitation.trim()) return;
          this.shepLoading = true;
          try {
            const citEnc = encodeURIComponent(this.shepCitation.trim());
            const res = await fetch('/api/shepardize/' + citEnc);
            if (!res.ok) throw new Error('Shepardize check failed');
            this.shepResult = await res.json();
          } catch (err) {
            console.error(err);
            alert('Failed to audit precedent: ' + err.message);
          } finally {
            this.shepLoading = false;
          }
        },

        async executeGenerateMemo() {
          this.memoLoading = true;
          try {
            const sections = (this.memoSections || '').split(',').map(s => s.trim()).filter(Boolean);
            const payload = {
              doc_type: this.memoDocType,
              query: this.memoQuery,
              doc_id: this.memoDocId,
              court: 'delhi',
              year: 2023,
              applicant_name: this.memoApplicant,
              fir_no: this.memoFirNo,
              police_station: this.memoStation,
              sections_invoked: sections,
              custody_date: this.memoCustody,
              facts_summary: this.memoFacts
            };

            const res = await fetch('/api/rag/memo', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(payload)
            });
            if (!res.ok) {
              const errData = await res.json();
              throw new Error(errData.detail || 'Memo generation failed');
            }
            this.memoResult = await res.json();
          } catch (err) {
            console.error(err);
            alert('Error drafting legal document: ' + err.message);
          } finally {
            this.memoLoading = false;
          }
        },

        copyMemoText() {
          if (!this.memoResult || !this.memoResult.content) return;
          navigator.clipboard.writeText(this.memoResult.content).then(() => {
            this.copySuccess = true;
            setTimeout(() => { this.copySuccess = false; }, 2000);
          });
        },

        downloadMemoText() {
          if (!this.memoResult || !this.memoResult.content) return;
          const blob = new Blob([this.memoResult.content], { type: 'text/plain;charset=utf-8' });
          const url = URL.createObjectURL(blob);
          const link = document.createElement('a');
          link.href = url;
          link.download = (this.memoResult.title || 'LegalTrace_Document').replace(/[^a-zA-Z0-9_-]/g, '_') + '.txt';
          document.body.appendChild(link);
          link.click();
          document.body.removeChild(link);
          URL.revokeObjectURL(url);
        },

        getShepBadgeClass(status) {
          const map = {
            'good_law': 'bg-emerald-950 text-emerald-300 border border-emerald-800',
            'distinguished': 'bg-amber-950 text-amber-300 border border-amber-800',
            'caution': 'bg-yellow-950 text-yellow-300 border border-yellow-800',
            'overruled': 'bg-rose-950 text-rose-300 border border-rose-800',
            'questioned': 'bg-purple-950 text-purple-300 border border-purple-800',
            'unknown': 'bg-slate-800 text-slate-300 border border-slate-700'
          };
          return map[status] || 'bg-slate-800 text-slate-300';
        },

        getShepIcon(status) {
          const map = {
            'good_law': 'fa-circle-check text-emerald-400',
            'distinguished': 'fa-triangle-exclamation text-amber-400',
            'caution': 'fa-circle-exclamation text-yellow-400',
            'overruled': 'fa-circle-xmark text-rose-400',
            'questioned': 'fa-circle-question text-purple-400',
            'unknown': 'fa-circle-question text-slate-400'
          };
          return map[status] || 'fa-circle-info';
        },

        getTreatmentBadgeClass(treatment) {
          const map = {
            'relied_upon': 'bg-emerald-950 text-emerald-300 border border-emerald-800',
            'followed': 'bg-emerald-950 text-emerald-300 border border-emerald-800',
            'distinguished': 'bg-amber-950 text-amber-300 border border-amber-800',
            'overruled': 'bg-rose-950 text-rose-300 border border-rose-800',
            'caution': 'bg-yellow-950 text-yellow-300 border border-yellow-800',
            'cited': 'bg-indigo-950 text-indigo-300 border border-indigo-800'
          };
          return map[treatment] || 'bg-slate-800 text-slate-300';
        },

        // Graph Visualizer State
        graphDocId: '104774187',
        graphLoading: false,
        graphData: null,
        selectedGraphNode: null,

        // Statute Ratio Lookup State
        statuteAct: 'CrPC',
        statuteSection: '482',
        statuteLoading: false,
        statuteResult: null,

        async loadGraphVisualization() {
          if (!this.graphDocId.trim()) return;
          this.graphLoading = true;
          try {
            const res = await fetch('/api/graph/visualization/' + encodeURIComponent(this.graphDocId.trim()) + '?depth=2');
            if (!res.ok) throw new Error('Failed to load citation graph');
            this.graphData = await res.json();
          } catch (err) {
            console.error(err);
            alert('Error loading citation graph: ' + err.message);
          } finally {
            this.graphLoading = false;
          }
        },

        async executeStatuteLookup() {
          if (!this.statuteAct.trim() || !this.statuteSection.trim()) return;
          this.statuteLoading = true;
          try {
            const act = encodeURIComponent(this.statuteAct.trim());
            const sec = encodeURIComponent(this.statuteSection.trim());
            const res = await fetch('/api/statutes/lookup?act=' + act + '&section=' + sec + '&limit=10');
            if (!res.ok) throw new Error('Statute lookup failed');
            this.statuteResult = await res.json();
          } catch (err) {
            console.error(err);
            alert('Error retrieving statute ratio: ' + err.message);
          } finally {
            this.statuteLoading = false;
          }
        },

        async executeTargetedScrape() {
          if (!this.scrapeQuery.trim()) return;
          this.scrapeLoading = true;
          try {
            const res = await fetch('/api/scrape/targeted', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                query: this.scrapeQuery.trim(),
                court: this.scrapeCourt,
                year: this.scrapeYear ? parseInt(this.scrapeYear) : null,
                limit: parseInt(this.scrapeLimit) || 3,
                auto_fetch: !!this.scrapeAutoFetch
              })
            });
            if (!res.ok) {
              const err = await res.json();
              throw new Error(err.detail || 'Scraping failed');
            }
            this.scrapeData = await res.json();
            if (this.scrapeData && this.scrapeData.indexed_chunks_count) {
              this.totalIndexed = (this.totalIndexed || 0) + this.scrapeData.indexed_chunks_count;
            }
            if (this.scrapeData && this.scrapeData.auto_fetch && this.scrapeData.results) {
              for (const r of this.scrapeData.results) {
                const sid = String(r.doc_id);
                if (!this.indexedDocIds.includes(sid)) {
                  this.indexedDocIds.push(sid);
                }
              }
            }
          } catch (err) {
            console.error(err);
            alert('Error executing targeted scrape: ' + err.message);
          } finally {
            this.scrapeLoading = false;
          }
        },

        async downloadAndIndexDoc(item) {
          this.downloadingDocId = item.doc_id;
          try {
            const res = await fetch('/api/scrape/download-and-index', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                doc_id: String(item.doc_id),
                court: item.court || 'Delhi High Court',
                year: item.year ? parseInt(item.year) : 2023,
                title: item.title,
                source_url: item.source_url
              })
            });
            if (!res.ok) {
              const err = await res.json();
              throw new Error(err.detail || 'Download failed');
            }
            const data = await res.json();
            const sid = String(item.doc_id);
            if (!this.indexedDocIds.includes(sid)) {
              this.indexedDocIds.push(sid);
            }
            if (data.indexed_chunks) {
              this.totalIndexed = (this.totalIndexed || 0) + data.indexed_chunks;
            }
            alert('Success! Judgment #' + item.doc_id + ' has been downloaded, cleaned, chunked (' + data.indexed_chunks + ' chunks), and indexed into RAG. You can now query it in the Legal RAG tab!');
          } catch (err) {
            console.error(err);
            alert('Failed to download & index judgment: ' + err.message);
          } finally {
            this.downloadingDocId = null;
          }
        },

        askRAGAboutDoc(item) {
          this.tab = 'rag';
          this.ragQuery = 'What was held in ' + (item.title || 'Doc #' + item.doc_id) + '? Explain the key legal principles, ratio decidendi, and final outcome.';
          this.executeRAG();
        },

        isDocIndexed(docId) {
          return this.indexedDocIds.includes(String(docId));
        },

        getGraphNodeX(idx, total) {
          if (total <= 0) return 400;
          const angle = (idx / total) * 2 * Math.PI;
          const radius = 110;
          return Math.round(400 + radius * Math.cos(angle));
        },

        getGraphNodeY(idx, total) {
          if (total <= 0) return 150;
          const angle = (idx / total) * 2 * Math.PI;
          const radius = 100;
          return Math.round(150 + radius * Math.sin(angle));
        },

        getTreatmentStroke(treatment) {
          const map = {
            'relied_upon': '#10b981',
            'followed': '#10b981',
            'distinguished': '#f59e0b',
            'overruled': '#f43f5e',
            'caution': '#eab308',
            'cited': '#6366f1'
          };
          return map[treatment] || '#64748b';
        },

        getTreatmentColor(treatment) {
          const map = {
            'relied_upon': '#059669',
            'followed': '#059669',
            'distinguished': '#d97706',
            'overruled': '#e11d48',
            'caution': '#ca8a04',
            'cited': '#4f46e5'
          };
          return map[treatment] || '#475569';
        },

        getNodeById(id) {
          if (!this.graphData || !this.graphData.nodes) return null;
          return this.graphData.nodes.find(n => n.id === id) || { id: id, label: id, type: 'precedent' };
        },

        async loadPrecedents() {
          if (this.precedents.length === 0) {
            try {
              const res = await fetch('/api/precedents/top?limit=25');
              this.precedents = await res.json();
            } catch (err) {
              console.error(err);
            }
          }
          if (!this.graphData) {
            this.loadGraphVisualization();
          }
        },

        async openHeadnote(docId) {
          try {
            const res = await fetch('/api/headnotes/' + docId);
            if (!res.ok) throw new Error('Headnote not found');
            this.activeHeadnote = await res.json();
          } catch (err) {
            alert('Headnote not available for doc ' + docId);
          }
        },

        async openCitations(docId) {
          try {
            const res = await fetch('/api/citations/' + docId);
            const data = await res.json();
            alert('Document ' + docId + ' cites ' + data.total_citations + ' precedent authorities across reporters.');
          } catch (err) {
            console.error(err);
          }
        },

        getRoleBadgeClass(role) {
          const map = {
            'analysis': 'bg-blue-950 text-blue-300 border border-blue-800',
            'facts': 'bg-amber-950 text-amber-300 border border-amber-800',
            'operative': 'bg-emerald-950 text-emerald-300 border border-emerald-800',
            'arguments_petitioner': 'bg-purple-950 text-purple-300 border border-purple-800',
            'arguments_respondent': 'bg-rose-950 text-rose-300 border border-rose-800',
            'precedents': 'bg-indigo-950 text-indigo-300 border border-indigo-800',
            'statute_reference': 'bg-cyan-950 text-cyan-300 border border-cyan-800'
          };
          return map[role] || 'bg-slate-800 text-slate-300';
        }
      }
    }
  </script>
</body>
</html>
"""
