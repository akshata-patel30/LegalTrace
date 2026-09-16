# LegalTrace — Indian Legal Ingestion, Knowledge Graph & Agentic RAG Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-v1.12.6-red.svg)](https://qdrant.tech/)
[![FalkorDB](https://img.shields.io/badge/FalkorDB-latest-orange.svg)](https://www.falkordb.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16--alpine-336791.svg)](https://www.postgresql.org/)
[![Tests](https://img.shields.io/badge/tests-180%20passed-brightgreen.svg)](tests/)
[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-lightgrey.svg)](#license)

**LegalTrace** is a production-grade, zero-hallucination legal document acquisition, reasoning, and retrieval platform engineered specifically for the Indian judicial system. It seamlessly bridges raw unstructured court judgments from Indian Kanoon with an enterprise 3-database storage infrastructure (PostgreSQL 16, Qdrant Vector DB, FalkorDB Graph DB), a 9-role legal rhetorical segmenter, a multi-hop Agentic Legal RAG reasoning engine, an Article 141 binding precedent hierarchy resolver, and a reverse Statute-to-Ratio lookup engine.

---

## 1. System Overview & Core Capabilities

```
                       ┌──────────────────────────────────────────────┐
                       │   Indian Kanoon / Multi-Court Web Portals     │
                       └──────────────────────┬───────────────────────┘
                                              │ Polite Scraper / API
                                              ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ Pipeline A: 11-Step Document Ingestion & Rhetorical Transformation                     │
 │                                                                                        │
 │  [1. Intake]  ──>  [2. Validate]  ──>  [3. Clean]  ──>  [4. Extract]  ──>  [5. RRL]    │
 │   Discovery         SHA-256 Check       Unicode/Noise    Entities/Bench    9 Roles     │
 │                                                                               │        │
 │  [10. Benchmark] <── [9. Hybrid Search] <── [8. Headnotes] <── [7. Graph] <── [6. Chunk] │
 │   20 Golden Qs        BM25 + Dense 384-d    Syllabus Ratio     Precedents     ~360 Wds │
 └────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      │
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ Multi-Engine Production Storage (Phase 1 & Phase 3)                                    │
 │                                                                                        │
 │  ┌───────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐  │
 │  │     PostgreSQL 16     │  │    Qdrant Vector DB    │  │     FalkorDB Graph DB     │  │
 │  │  Relational Metadata, │  │   Dense (384-d) +      │  │  Citation Network,        │  │
 │  │  Judgments, Chunks,   │  │   Sparse (BM25) RRF    │  │  Article 141 Traversal,   │  │
 │  │  Headnotes, Telemetry │  │   Hybrid Collection    │  │  Good-Law Shepardization  │  │
 │  └───────────────────────┘  └────────────────────────┘  └───────────────────────────┘  │
 └────────────────────────────────────┬───────────────────────────────────────────────────┘
                                      │
                                      ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │ Pipeline B: Agentic Legal RAG & Retrieval Engine (Phases 2 & 4)                        │
 │                                                                                        │
 │  ┌───────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐  │
 │  │   LegalQueryPlanner   │  │    LegalShepardizer    │  │    LegalRAGSynthesizer    │  │
 │  │  Multi-query Subtask  │  │  Judicial Treatment    │  │  Grounded Pinpoint        │  │
 │  │  Decomposition (Groq) │  │  Good-Law Audit Trail  │  │  Citation Synthesis       │  │
 │  └───────────────────────┘  └────────────────────────┘  └───────────────────────────┘  │
 │  ┌───────────────────────┐  ┌────────────────────────┐  ┌───────────────────────────┐  │
 │  │  StatuteRatioLookup   │  │   LegalMemoGenerator   │  │   FastAPI + Tailwind SPA  │  │
 │  │  Act/Sec -> Judicial  │  │  Automated Court-Ready │  │  Interactive SVG Network  │  │
 │  │  Ratio Decidendi      │  │  Briefs & Applications │  │  Search, RAG, Visualizer  │  │
 │  └───────────────────────┘  └────────────────────────┘  └───────────────────────────┘  │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

### Key Engineering Features
- **Polite Resilient Ingestion Engine**:
  - Dynamically obeys `robots.txt` (`Crawl-delay: 5–8s` with jitter).
  - Multi-level deduplication: in-flight worker claims, SQLite pre-fetch status tracking, and post-fetch SHA-256 content hashes.
  - Turnstile challenge detection, structural DOM drift detection, and exponential circuit-breaker backoff.
- **Rhetorical Role Labeling (RRL)**:
  - Classifies legal text into 9 canonical legal roles: `facts`, `issues`, `petitioner_arguments`, `respondent_arguments`, `statute_reference`, `precedents`, `analysis`, `operative`, `procedural`.
  - Boundary-isolated semantic chunker (target ~360 words) injecting rich case headers: `[Doc {doc_id} | {court} {year} | Bench: {bench} | Role: {role}]`.
- **Tri-Store Database Architecture**:
  - **PostgreSQL 16**: Relational master storing structured judgments, chunks, headnotes, statutory mentions, search telemetry, and batch jobs.
  - **Qdrant v1.12.6**: Dense vectors (`all-MiniLM-L6-v2` / `DenseSemanticEmbedder`) + Sparse BM25 indexed payload collection (`legal_chunks`) with Reciprocal Rank Fusion (RRF).
  - **FalkorDB**: Low-latency Graph database storing directed citation networks (`:Judgment`-`[:CITES]`->`:Precedent`) with judicial treatment tracking (`RELIED_UPON`, `FOLLOWED`, `DISTINGUISHED`, `OVERRULED`).
- **Article 141 Binding Precedent Hierarchy**:
  - Automatically identifies whether an authority originates from the Supreme Court of India (`SCC`, `SCR`, `AIR SC`, `INSC`) or coordinate/persuasive High Courts (Delhi, Bombay, Madras, Karnataka).
- **Zero-Hallucination Agentic RAG**:
  - **LegalQueryPlanner**: Decomposes natural legal queries into statutory, precedent, factual, and procedural sub-tasks.
  - **LegalShepardizer**: Verifies that cited cases have not been overruled or distinguished before presentation.
  - **LegalRAGSynthesizer**: Synthesizes structured legal answers with strictly verified inline pinpoint citations (`[1]`, `[2]`).
  - **Statute-to-Ratio Reverse Lookup Engine**: Resolves canonical Act and Section queries (e.g. `CrPC Section 482`, `NDPS Section 37`, `Arbitration Act Section 34`) directly to judicial interpretations and formulation of legal principles.
- **Court Pleading & Chamber Memo Generator**:
  - Generates formal Case Briefs, Legal Opinions, Bail Applications, and Quashing Petitions formatted to high court standards.

---

## 2. Quickstart & Installation

### Prerequisites
- Python 3.11 or 3.12
- Docker & Docker Compose (for PostgreSQL, Qdrant, FalkorDB)
- Optional: Groq Cloud API key for LLaMA-3.1 RAG synthesis

### 1. Clone & Set Up Environment
```bash
cd D:/Timepasss/ingestion-source

# Create virtual environment
python -m venv .venv

# Activate environment
# On Windows:
.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

# Install dependencies in editable mode with development tools
pip install -e ".[dev]"
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Key configuration variables in `.env`:
```ini
# Storage and Ingestion
INTAKE_OUTPUT_PATH=./raw-store
DB_PATH=./raw-store/job_state/legaltrace_ingest.sqlite
MODE=scrape_only
SCRAPER_ENGINE=scrapling
MAX_WORKERS=1
SCRAPE_MIN_DELAY_SECONDS=5.0
SCRAPE_MAX_DELAY_SECONDS=8.0
ADAPTIVE_BACKOFF_SECONDS=15.0

# Relational Core (PostgreSQL 16 via asyncpg)
DATABASE_URL=postgresql+asyncpg://legaltrace:legaltrace_dev_2024@localhost:5431/legaltrace

# Vector Search Engine (Qdrant v1.12.6)
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=legal_chunks

# Graph Database (FalkorDB)
FALKORDB_HOST=localhost
FALKORDB_PORT=6379
FALKORDB_GRAPH=citation_network

# LLM / RAG Engine (Optional Groq API Key)
RAG_LLM_PROVIDER=groq
RAG_LLM_MODEL=llama-3.1-70b-versatile
GROQ_API_KEY=your_groq_api_key_here
```

---

## 3. Database Infrastructure (Docker Compose)

LegalTrace uses Docker Compose to orchestrate its three persistent database engines. Persistent volumes ensure all ingested judgments, vectors, and graph edges survive container restarts.

```bash
# Start all 3 databases in background
docker compose up -d

# Check status and health of containers
docker compose ps
```

| Service | Container Name | Host Port | Internal Port | Persistent Volume | Healthcheck |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PostgreSQL 16** | `legaltrace-postgres` | `5431` | `5432` | `legaltrace-pg-data` | `pg_isready -U legaltrace -d legaltrace` |
| **Qdrant v1.12.6** | `legaltrace-qdrant` | `6333` (HTTP)<br>`6334` (gRPC) | `6333`<br>`6334` | `legaltrace-qdrant-data` | `bash -c "exec 3<>/dev/tcp/localhost/6333"` |
| **FalkorDB** | `legaltrace-falkor-data` | `6379` (Redis)<br>`3000` (Web UI) | `6379`<br>`3000` | `legaltrace-falkor-data` | `redis-cli ping` |

### Apply Database Migrations & Schemas
Initialize tables, vector collections, payload indices, and graph constraints:
```bash
ingest db-migrate all
```
Or run individually:
```bash
ingest db-migrate postgres
ingest db-migrate qdrant
ingest db-migrate falkordb
```

---

## 4. Complete CLI Command Reference (All 27 Commands)

The CLI tool is registered as `ingest`. Run `ingest --help` at any time to inspect available commands.

### Group 1: Acquisition & Ingestion
```bash
# 1. Verify robots.txt compliance for a target URL
ingest check-robots --url "https://indiankanoon.org/doc/1940428/"

# 2. Discover judgments (without downloading body content)
ingest discover --court "Delhi High Court" --year 2023 --limit 50 --output discovered.jsonl

# 3. Fetch discovered judgments in parallel with rate-limiting and deduplication
ingest fetch --input-file discovered.jsonl --mode scrape_only --workers 3 --limit 20

# 4. One-step discovery + acquisition pipeline
ingest run --court "Delhi High Court" --year 2023 --mode scrape_only --limit 10 --workers 2

# 5. Resume an interrupted acquisition job
ingest resume --job-id <JOB_ID> --workers 2

# 6. Display telemetry and error breakdown report for an acquisition job
ingest report --job-id <JOB_ID>

# 7. Migrate legacy flat raw-store files into hierarchical {court}/{year}/{doc_id}/ structure
ingest restructure --store-dir ./raw-store
```

### Group 2: Document Processing Pipeline (Steps 2–8 & Export)
```bash
# 8. Validate cryptographic SHA-256 checksums and minimum content thresholds
ingest validate --store-dir ./raw-store

# 9. Clean raw judgments (Unicode repair, noise stripping, paragraph segmentation, disposition)
ingest clean --store-dir ./raw-store

# 10. Enrich metadata (extract bench, judge canonicalization, acts/sections, citations)
ingest enrich --store-dir ./raw-store

# 11. Label rhetorical roles (9 canonical roles: facts, analysis, operative, etc.)
ingest structure --store-dir ./raw-store

# 12. Segment judgments into role-isolated semantic chunks with context headers
ingest chunk --store-dir ./raw-store --output chunks.jsonl --min-words 100 --target-words 360 --max-words 600

# 13. Build directed citation network and classify judicial treatment
ingest graph --store-dir ./raw-store --output-edges citation_edges.jsonl --output-nodes precedent_nodes.jsonl

# 14. Synthesize law-reporter headnotes (catchwords, facts, ratio, holding)
ingest headnote --store-dir ./raw-store

# 15. Export raw store into compressed Apache Parquet tables
ingest export --output legaltrace_corpus.parquet --store-dir ./raw-store
```

### Group 3: Search, Indexing & Golden Benchmarks
```bash
# 16. Build hybrid index (Dense embeddings + BM25 inverted index)
# Embedder options: 'transformer' (SentenceTransformers all-MiniLM-L6-v2) or 'dense' (multi-resolution hash)
ingest index --input-chunks chunks.jsonl --index-dir ./search_index --embedder-type transformer

# 17. Execute hybrid, dense, or lexical queries with metadata filters
ingest search "bail conditions under section 37 ndps act" --index-dir ./search_index --mode hybrid --top-k 5 --role analysis

# 18. Run empirical evaluation against the 20-query Golden Legal QA Suite
ingest benchmark --index-dir ./search_index --output benchmark_report.json --top-k 5
```

### Group 4: Multi-Store Database Management
```bash
# 19. Initialize and verify PostgreSQL, Qdrant, and FalkorDB schemas
ingest db-migrate all
ingest db-migrate postgres
ingest db-migrate qdrant
ingest db-migrate falkordb
```

### Group 5: Agentic Legal RAG & Court Drafting
```bash
# 20. Decompose a complex legal question into structured retrieval sub-queries
ingest rag-plan "Can anticipatory bail be granted when chargesheet is already filed under section 438 CrPC?"

# 21. Execute end-to-end grounded RAG with Shepardization and inline pinpoint citations
ingest rag-query "What are the mandatory conditions for granting bail under Section 37 NDPS Act?" --max-chunks 10 --index-dir ./search_index

# 22. Audit judicial treatment history and verify if a precedent is good law
ingest shepardize "2008 16 SCC 417"

# 23. Automatically generate formal court pleadings and chamber memorandums
# Doc types: 'case_brief', 'memo', 'bail_application', 'quashing_petition'
ingest memo "bail under section 37 ndps act commercial quantity" --doc-type bail_application --output bail_petition.md
```

### Group 6: Corpus Expansion & Batch Operations
```bash
# 24. Batch ingest multi-court, multi-year decisions directly into PostgreSQL, Qdrant, and FalkorDB
ingest batch-ingest --courts "Delhi High Court" --courts "Supreme Court of India" --years 2023 --years 2024 --limit 500 --workers 3

# 25. Check live status and database metrics across all court-year ingestion batches
ingest batch-status
```

### Group 7: Statute Reverse Ratio Lookup
```bash
# 26. Reverse lookup judicial interpretations and ratio decidendi for any Act and Section
ingest statute-lookup --act "CrPC" --section "482" --limit 5
ingest statute-lookup --act "NDPS Act" --section "37" --limit 5
ingest statute-lookup --act "Arbitration Act" --section "34" --limit 5
```

### Group 8: Production Web Service
```bash
# 27. Start FastAPI server and embedded Tailwind SPA dashboard
ingest serve --host 0.0.0.0 --port 8000 --reload
```

---

## 5. REST API Reference

The FastAPI service is accessible at `http://localhost:8000`. Interactive OpenAPI documentation is available at `http://localhost:8000/docs` and Redoc at `http://localhost:8000/redoc`.

| Method | Endpoint | Description | Key Request / Query Parameters |
| :--- | :--- | :--- | :--- |
| `GET` | `/health` | System health, service status, and loaded indices | None |
| `POST` | `/api/search` | Sub-15ms Hybrid Semantic-Lexical Search | `query`, `mode` (`hybrid` \| `dense` \| `bm25`), `top_k`, `filter_role`, `filter_court`, `filter_year` |
| `GET` | `/api/judgments/{doc_id}` | Retrieve normalized judgment JSON | `doc_id` |
| `GET` | `/api/headnotes/{doc_id}` | Retrieve legal headnote & syllabus | `doc_id` |
| `GET` | `/api/citations/{doc_id}` | List outbound precedent citations & treatments | `doc_id` |
| `GET` | `/api/precedents/top` | Top cited authorities in corpus | `limit` (default: 20) |
| `GET` | `/api/benchmark` | Golden QA evaluation metrics (MRR, NDCG, Precision) | None |
| `POST` | `/api/rag/plan` | Legal query decomposition into sub-tasks | `query` |
| `POST` | `/api/rag/query` | Grounded legal Q&A with inline citations | `query`, `max_chunks` |
| `POST` | `/api/rag/memo` | Generate Case Brief, Memo, or Bail Petition | `query_or_doc_id`, `doc_type` |
| `GET` | `/api/shepardize/{citation}` | Judicial treatment audit & good-law check | `citation` (URL encoded) |
| `GET` | `/api/graph/neighbors/{node_id}` | Graph adjacency list for node | `node_id` |
| `GET` | `/api/graph/binding/{doc_id}` | Supreme Court Article 141 binding precedent chain | `doc_id` |
| `GET` | `/api/graph/hierarchy` | Court hierarchy citation breakdown | `court_slug` |
| `GET` | `/api/batch/status` | Ingestion batch status from PostgreSQL & SQLite | None |
| `GET` | `/api/graph/visualization/{doc_id}` | Graph visualization payload (nodes, links, categories) | `doc_id`, `max_nodes` |
| `GET` | `/api/statutes/lookup` | Reverse lookup statute ratio (GET) | `act`, `section`, `limit` |
| `POST` | `/api/statutes/ratio-lookup` | Reverse lookup statute ratio (POST) | JSON: `{"act_name": "...", "section": "...", "limit": 10}` |
| `GET` | `/` | Web SPA Dashboard interface | None |

---

## 6. Interactive Web Dashboard

LegalTrace includes a responsive single-page application built with modern vanilla JavaScript and Tailwind CSS embedded directly in `src/ingestion_source/api.py`.

Navigate to **`http://localhost:8000/`** to explore:
1. **Hybrid Search Tab**: Real-time search with interactive mode switching (Hybrid, Dense, BM25), rhetorical role filters, court selector, and latency meters.
2. **Agentic RAG Assistant Tab**: Multi-hop query planner visualizer, inline verified evidence passage drawer, confidence score gauge, and Shepardization warning alerts.
3. **Legal Memo Studio Tab**: Instant draft generation for Chamber Briefs, Legal Opinions, Bail Petitions, and Section 482 Quashing Applications with download options.
4. **Citation Graph Visualizer Tab**: Interactive SVG force-directed network graph rendering judgments, Supreme Court precedents, and citation links with pan/zoom and node inspection.
5. **Statute Ratio Lookup Tab**: Direct reverse lookup displaying judicially formulated *ratio decidendi*, extracted statutory principles, and citing case links.
6. **Headnote Reader Tab**: Structured law-reporter syllabus viewer featuring catchwords lines, brief facts, framed issues, ratio decidendi, and operative holdings.
7. **System Metrics & Benchmark Tab**: Golden QA evaluation suite results displaying Precision@K, Recall@K, Hit Rate@K, MRR, NDCG@K, and live database connection health.

---

## 7. Verification & Testing

The test suite comprises **180 automated unit and integration tests** validating every component of the system:
- HTML scraper, challenge detection, and robots.txt manager
- Content hash deduplication and atomic worker claiming
- Paragraph cleaning, unicode repairs, and disposition extraction
- Entity extractors (statutes, citations, judges, counsel)
- Rhetorical role labeling and chunk context header injection
- Okapi BM25 and dense/transformer embedding computations
- PostgreSQL SQLAlchemy models, Qdrant vector store, and FalkorDB graph store
- Query planner decomposition and prompt construction
- Shepardizer treatment classification and Article 141 hierarchy resolution
- Grounded RAG synthesizer and court memo generation
- Batch ingestion orchestrator and statute-to-ratio reverse lookup

### Run Tests
```bash
# Run all 180 tests
pytest tests/ -v

# Run with short traceback and summary
pytest tests/ -q

# Run benchmark evaluation CLI
ingest benchmark --index-dir ./search_index --top-k 5
```

---

## 8. Repository Layout

```
ingestion-source/
├── docker-compose.yml              # PostgreSQL 16, Qdrant v1.12.6, FalkorDB services
├── Dockerfile                      # Container definition for ingestion service
├── pyproject.toml                  # PEP 621 packaging and dependency metadata
├── .env.example                    # Template environment variables
├── README.md                       # High-level overview and operations manual
├── TECHNICAL_DOCUMENTATION.md      # Deep technical architecture and data dictionaries
├── docker/
│   └── postgres/
│       └── init.sql                # PostgreSQL init script (pg_trgm, btree_gin)
├── src/
│   └── ingestion_source/
│       ├── __init__.py
│       ├── api.py                  # FastAPI REST service & embedded Tailwind SPA
│       ├── batch_ingest.py         # Multi-court batch ingestion orchestrator
│       ├── chunker.py              # Semantic legal chunker (target ~360 words)
│       ├── cleaner.py              # Unicode normalizer, noise cleaner, segmenter
│       ├── cli.py                  # 27 Typer CLI commands
│       ├── config.py               # Pydantic v2 settings
│       ├── dedupe.py               # SQLite deduplication store & atomic claims
│       ├── discovery.py            # Indian Kanoon discovery engine
│       ├── evaluation.py           # Evaluation metrics (Precision, Recall, MRR, NDCG)
│       ├── export_parquet.py       # Apache Parquet exporter
│       ├── extractor.py            # Legal entity & citation extractor
│       ├── graph.py                # Precedent citation graph builder
│       ├── job.py                  # Acquisition job orchestrator & thread pool
│       ├── models.py               # Pydantic data schemas (RawJudgment, LegalChunk)
│       ├── normalizer.py           # Date parsers & bench extractors
│       ├── rate_limiter.py         # TokenBucketLimiter & RobotsTxtManager
│       ├── search.py               # HybridLegalSearchEngine, BM25, TransformerEmbedder
│       ├── storage.py              # File storage writer & JSONL manifest
│       ├── structure.py            # Rhetorical Role Labeler (9 legal roles)
│       ├── summarizer.py           # Automated legal headnote & syllabus generator
│       ├── validator.py            # SHA-256 integrity & QA validator
│       ├── db/
│       │   ├── __init__.py
│       │   ├── graph_store.py      # FalkorDB graph store & openCypher queries
│       │   ├── migrate.py          # Database migration runner (db-migrate)
│       │   ├── models.py           # SQLAlchemy 2.0 ORM models
│       │   ├── session.py          # Async SQLAlchemy engine & session factory
│       │   └── vector_store.py     # Qdrant client & hybrid collection management
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── memo_generator.py   # Court pleading & chamber brief generator
│       │   ├── planner.py          # Legal query decomposition planner
│       │   ├── shepardizer.py      # Precedent treatment audit & good-law checker
│       │   ├── statute_lookup.py   # Reverse Statute-to-Ratio lookup engine
│       │   └── synthesizer.py      # Grounded pinpoint citation synthesizer (Groq)
│       ├── seeds/
│       │   ├── __init__.py
│       │   └── catalog.py          # Seed manifest catalog
│       └── sources/
│           ├── __init__.py
│           ├── api_client.py       # Official Indian Kanoon API client
│           ├── base.py             # Abstract base class for intake sources
│           ├── html_fallback.py    # Polite HTML scraper & Scrapling fallback
│           └── seed_source.py      # Local seed dataset loader
└── tests/                          # 180 unit and integration tests
```

---

## 9. License

Proprietary and Confidential. Developed for the LegalTrace Platform.
