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
from ingestion_source.search import HybridLegalSearchEngine

logger = logging.getLogger(__name__)

app = FastAPI(
    title="LegalTrace Indian Kanoon Retrieval & Search API",
    description="High-performance legal research API for Delhi High Court judgments with hybrid vector search, rhetorical role filtering, automated headnotes, and citation graph lookup.",
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
        <button @click="tab = 'search'" :class="tab === 'search' ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400 hover:text-white'" class="px-3.5 py-1.5 rounded-md text-xs transition">
          <i class="fa-solid fa-magnifying-glass mr-1.5"></i>Search & Retrieval
        </button>
        <button @click="tab = 'precedents'; loadPrecedents()" :class="tab === 'precedents' ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400 hover:text-white'" class="px-3.5 py-1.5 rounded-md text-xs transition">
          <i class="fa-solid fa-network-wired mr-1.5"></i>Citation Graph
        </button>
        <button @click="tab = 'benchmark'; loadBenchmark()" :class="tab === 'benchmark' ? 'bg-indigo-600 text-white font-medium' : 'text-slate-400 hover:text-white'" class="px-3.5 py-1.5 rounded-md text-xs transition">
          <i class="fa-solid fa-chart-line mr-1.5"></i>Benchmark QA
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
    <!-- TAB 2: CITATION GRAPH & LANDMARK PRECEDENTS -->
    <!-- ================================================================= -->
    <div x-show="tab === 'precedents'" x-cloak x-transition>
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

        async init() {
          this.executeSearch();
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

        async loadPrecedents() {
          if (this.precedents.length > 0) return;
          try {
            const res = await fetch('/api/precedents/top?limit=25');
            this.precedents = await res.json();
          } catch (err) {
            console.error(err);
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
