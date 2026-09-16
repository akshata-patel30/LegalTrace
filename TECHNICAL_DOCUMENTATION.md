# LegalTrace — Technical Architecture & System Documentation

> **System Name**: LegalTrace — Indian Legal Ingestion, Knowledge Graph & Agentic RAG Platform  
> **Architecture Subsystems**: Pipeline A (Intake, Transformation & Retrieval) & Pipeline B (Agentic RAG, Knowledge Graph & Legal Reasoning)  
> **Jurisdictions**: Supreme Court of India, Delhi High Court, Bombay High Court, Madras High Court, Karnataka High Court (Years: 2020–2024, Target: 25,000 Judgments)  
> **Runtime Environment**: Python 3.11+ / 3.12 (Isolated `.venv`), Windows AMD64 / Linux x86_64  
> **Storage Infrastructure**: PostgreSQL 16 (Relational Core), Qdrant v1.12.6 (Vector Engine), FalkorDB (Graph Engine)  
> **Current Version**: 2.0.0  
> **Test Suite**: 180 Automated Unit & Integration Tests  

---

## 1. Executive Summary & System Philosophy

**LegalTrace** is an enterprise-grade legal engineering platform built to solve the fundamental challenges of legal search and reasoning under the Indian legal system:
1. **Zero Hallucination Guarantee**: Grounded, verifiable citations where every legal assertion is tied via bracketed pinpoint markers (`[1]`, `[2]`) to exact paragraph-level evidence in the corpus.
2. **Strict Precedent Validity (Good-Law Audit)**: Automated Shepardization that traverses directed citation networks to ensure no overruled, distinguished, or cautioned decision is relied upon as binding authority.
3. **Constitutional Precedent Hierarchy (Article 141)**: Strict distinction between declarations of law by the Supreme Court of India (binding on all courts nationwide under Article 141 of the Constitution of India) versus coordinate High Court precedents (binding within jurisdiction, persuasive outside).
4. **Structural Rhetorical Role Isolation**: Parsing unstructured judicial opinions into 9 discrete rhetorical roles (e.g. `facts`, `petitioner_arguments`, `statute_reference`, `analysis`, `operative`) to eliminate cross-contamination during semantic chunking.
5. **Reverse Statute-to-Ratio Traversal**: Rapid reverse resolution from statutory provisions (e.g. `CrPC Section 482`, `NDPS Section 37`, `Arbitration Act Section 34`) directly to judicially articulated *ratio decidendi* and formulating legal principles.

---

## 2. End-to-End System Architecture

```mermaid
flowchart TD
    subgraph S1_Ingestion["Pipeline A: Document Acquisition & Ingestion"]
        API[Official Indian Kanoon API<br/>api.indiankanoon.org] --> Intake[Acquisition Orchestrator]
        HTML[Stealth Scrapling Fallback<br/>Dynamic robots.txt & Rate Limiter] --> Intake
        Intake --> RawStore[Raw Store<br/>raw-store/{court}/{year}/{doc_id}/]
        Intake --> Dedupe[(SQLite Dedupe Store)]
        RawStore --> Manifest[manifest.jsonl]
    end

    subgraph S2_Normalization["Pipeline A: Normalization, Cleaning & Structuring"]
        RawStore --> Validator[SHA-256 Validator & Content QA]
        Validator --> Cleaner[Unicode Repair, Noise Filter & Paragraph Segmenter]
        Cleaner --> Extractor[Metadata & Legal Entity Extractor]
        Extractor --> RRL[Rhetorical Role Labeler: 9 Legal Roles]
        RRL --> Chunker[Boundary-Aware Semantic Chunker<br/>Target ~360 words]
        Chunker --> Summarizer[Legal Headnote & Syllabus Generator]
        Extractor --> GraphLinker[Citation Linker & Treatment Classifier]
    end

    subgraph S3_Storage["Persistent Tri-Store Database Infrastructure"]
        Cleaner & Extractor & Summarizer --> PG[(PostgreSQL 16<br/>Judgments, Chunks, Headnotes, Telemetry)]
        Chunker --> Qdrant[(Qdrant Vector DB v1.12.6<br/>Dense 384-d + Sparse BM25)]
        GraphLinker --> Falkor[(FalkorDB Graph DB<br/>Citation Network & Art. 141 Hierarchy)]
    end

    subgraph S4_RAG["Pipeline B: Agentic Legal RAG & Retrieval Engine"]
        UserQuery[User Legal Query] --> Planner[LegalQueryPlanner<br/>Multi-Query Decomposition]
        Planner --> HybridSearch[Hybrid Search Engine<br/>RRF: Dense + BM25]
        HybridSearch <--> Qdrant
        HybridSearch <--> PG
        HybridSearch --> RetrievedPassages[Candidate Passages]
        RetrievedPassages --> Shepardizer[LegalShepardizer<br/>Precedent Audit & Good-Law Checker]
        Shepardizer <--> Falkor
        Shepardizer --> FilteredPassages[Audited Evidence Passages]
        FilteredPassages --> Synthesizer[LegalRAGSynthesizer<br/>Groq LLaMA-3.1-70B / Fallback]
        Synthesizer --> GroundedAnswer[Cited Legal Opinion]
        GroundedAnswer --> MemoGen[LegalMemoGenerator<br/>Court Briefs, Memos, Bail Pleadings]
        UserQuery --> StatuteRev[StatuteRatioLookup<br/>Act/Section -> Ratio Decidendi]
        StatuteRev <--> PG
        StatuteRev <--> Qdrant
    end

    subgraph S5_Delivery["Delivery & Interface Layer"]
        GroundedAnswer --> REST[FastAPI Service<br/>18 REST Endpoints]
        MemoGen --> REST
        StatuteRev --> REST
        REST --> WebSPA[Embedded Tailwind SPA<br/>SVG Force Graph & Multi-Tab Dashboard]
    end
```

---

## 3. Data Dictionaries & Complete Schemas

### 3.1 Pipeline A Data Models (`models.py`)

#### `DocRef`
Discovery reference identifying a candidate judgment before acquisition.
```python
class DocRef(BaseModel):
    doc_id: str                      # Unique document identifier (Indian Kanoon TID)
    title: str                       # Full case cause title
    court: str                       # Source court name
    court_slug: str                  # Normalized lowercase court slug (e.g. 'delhi')
    year: int                        # Decision year
    url: str                         # Source URL on Indian Kanoon
    headline: Optional[str] = None   # Keyword context snippet
    doc_source: str = "scrape"       # 'scrape', 'api', or 'seed'
    discovered_at: datetime          # Discovery timestamp (UTC)
```

#### `RawJudgment`
Normalized intake document written to `raw-store/{court}/{year}/{doc_id}/raw.json`.
```python
class RawJudgment(BaseModel):
    doc_id: str                      # Document TID
    title: str                       # Clean title
    court: str                       # Full court name
    court_slug: str                  # Normalized court slug
    year: int                        # Decision year
    date: Optional[str] = None       # Canonical decision date (YYYY-MM-DD)
    bench: list[str] = []            # Presiding judges as parsed from judicial header
    citation: Optional[str] = None   # Neutral citation (e.g. '2023:DHC:1234')
    docket_numbers: list[str] = []   # Petition / Appeal docket numbers
    petitioner: Optional[str] = None # Extracted petitioner party
    respondent: Optional[str] = None # Extracted respondent party
    full_text: str                   # Raw text body of judgment
    html_content: Optional[str] = None # Source HTML if acquired via scraper
    source_url: str                  # Source URL
    acquired_at: datetime            # Acquisition timestamp (UTC)
    content_hash: str                # SHA-256 digest of clean normalized body
```

#### `CleanedJudgment` (`cleaner.py`)
Result of text cleaning, Unicode repair, and paragraph segmentation.
```python
class ParagraphType(str, Enum):
    HEADER = "header"                # Judicial cause title, bench, dockets
    BODY = "body"                    # Core substantive legal text
    FOOTNOTE = "footnote"            # Footnote content
    FOOTER = "footer"                # Signatures, dates, bench seals

class DispositionType(str, Enum):
    ALLOWED = "allowed"              # Petition / appeal allowed
    DISMISSED = "dismissed"          # Petition / appeal dismissed
    DISPOSED = "disposed"            # Disposed of with directions
    PARTLY_ALLOWED = "partly_allowed"# Partly allowed / partly dismissed
    REFERRED = "referred"            # Referred to larger bench

class CleanedParagraph(BaseModel):
    para_id: int                     # 0-indexed sequence in judgment
    text: str                        # Clean text with normalized quotes and repaired hyphens
    para_type: ParagraphType         # HEADER, BODY, FOOTNOTE, FOOTER
    word_count: int                  # Paragraph word count
    char_count: int                  # Paragraph character count

class CleanedJudgment(BaseModel):
    doc_id: str
    title: str
    court: str
    year: int
    header: CleanedHeader            # Extracted header metadata
    paragraphs: list[CleanedParagraph] # Filtered substantive body paragraphs
    disposition: DispositionType     # Final judicial outcome
    word_count: int                  # Total word count across substantive paragraphs
    paragraph_count: int             # Substantive paragraph count
```

#### `EnrichedLegalMetadata` (`extractor.py`)
Extracted entities, statutory references, and judicial citations.
```python
class EnrichedLegalMetadata(BaseModel):
    doc_id: str
    case_type: Optional[str]         # Case classification (W.P.(C), CRL.A., etc.)
    legal_domain: str                # criminal, civil, constitutional, arbitration, tax
    bench_type: str                  # single, division, full, constitution
    presiding_judges: list[JudgeInfo] # Extracted judges with honorifics removed
    canonical_judges: list[str]      # Normalized uppercase canonical judge names
    statutes: list[StatuteMentionItem] # Acts and sections referenced
    precedents: list[PrecedentCitationItem] # External precedents cited (SCC, AIR, etc.)
    counsel_records: list[CounselRecord] # Appearing advocates and senior counsel
```

#### `RhetoricalRole` & `StructuredJudgment` (`structure.py`)
Rhetorical role labels for every substantive paragraph.
```python
class RhetoricalRole(str, Enum):
    FACTS = "facts"                  # Factual narrative, prosecution case
    ISSUES = "issues"                # Questions of law framed for adjudication
    PETITIONER = "petitioner_arguments" # Arguments raised by petitioner / appellant
    RESPONDENT = "respondent_arguments" # Arguments raised by respondent / State
    STATUTE = "statute_reference"    # Statutory provisions quoted or analyzed
    PRECEDENTS = "precedents"        # Discussion of prior case law
    ANALYSIS = "analysis"            # Judicial reasoning and findings of the court
    OPERATIVE = "operative"          # Final order, holding, and directions
    PROCEDURAL = "procedural"        # Court procedural history and appearances
```

#### `LegalChunk` (`chunker.py`)
Role-boundary semantic chunk for dense and sparse indexing.
```python
class LegalChunk(BaseModel):
    chunk_id: str                    # chunk_{doc_id}_{idx}
    doc_id: str                      # Parent document ID
    chunk_index: int                 # 0-indexed sequence within document
    rhetorical_role: RhetoricalRole  # Dominant rhetorical legal role
    text: str                        # Clean chunk text
    text_with_context: str           # Prepended with: [Doc {doc_id} | {court} {year} | Bench: {bench} | Role: {role}]
    char_count: int
    word_count: int
    statute_mentions: list[str]      # Statutes referenced in this chunk
    citations: list[str]             # Precedents cited in this chunk
    para_range: tuple[int, int]      # (start_para_id, end_para_id)
```

#### `LegalHeadnote` (`summarizer.py`)
Law-reporter publication syllabus.
```python
class LegalHeadnote(BaseModel):
    doc_id: str
    catchwords: list[str]            # Topical keywords (e.g. ['CRIMINAL LAW', 'BAIL', 'NDPS ACT'])
    catchwords_line: str             # Uppercase taxonomy header line
    brief_facts: str                 # 2-3 paragraph factual matrix
    issues: list[str]                # Framed questions of law
    ratio_decidendi: str             # Binding legal principle formulated
    key_precedents: list[str]        # Precedents considered
    operative_holding: str           # Final operative disposition
    formatted_headnote: str          # Complete law-reporter formatted syllabus
    word_count: int
```

---

### 3.2 Relational PostgreSQL Database Schema (`db/models.py`)

Implemented via SQLAlchemy 2.0 ORM:

| Table Name | Primary Key | Foreign Keys | Key Indexes | Description |
| :--- | :--- | :--- | :--- | :--- |
| `judgments` | `doc_id` (VARCHAR(32)) | None | `ix_judgments_court`, `ix_judgments_year`, `ix_judgments_disposition`, `ix_judgments_domain`, `uq_content_hash` | Core judgment master record with full metadata |
| `chunks` | `chunk_id` (VARCHAR(64)) | `doc_id` -> `judgments.doc_id` (CASCADE) | `ix_chunks_doc_id`, `ix_chunks_role`, `ix_chunks_role_doc` (composite) | 9,167+ semantic chunks with context headers |
| `headnotes` | `doc_id` (VARCHAR(32)) | `doc_id` -> `judgments.doc_id` (CASCADE) | PK index on `doc_id` | Automated law-reporter syllabi and catchwords |
| `statute_mentions` | `id` (INTEGER, AUTO) | `doc_id` -> `judgments.doc_id` (CASCADE) | `ix_statute_doc_id`, `ix_statute_act_name` | Canonical statutory provisions and sections |
| `search_sessions` | `id` (INTEGER, AUTO) | None | `ix_search_created_at` | Search telemetry, queries, filters, and latency |
| `rag_sessions` | `id` (INTEGER, AUTO) | None | `ix_rag_created_at` | RAG query plans, answers, confidence, citations |
| `batch_ingestion_jobs` | `id` (INTEGER, AUTO) | None | `ix_batch_batch_id` (UQ), `ix_batch_court`, `ix_batch_year`, `ix_batch_status` | Multi-court batch ingestion checkpoint tracker |

---

### 3.3 Qdrant Vector Collection Schema (`db/vector_store.py`)

- **Collection Name**: `legal_chunks`
- **Dense Vector Configuration**:
  - Vector Name: `"dense"`
  - Dimension: `384`
  - Distance Metric: `Cosine`
  - Payload Storage: On-Disk (`QDRANT__STORAGE__ON_DISK_PAYLOAD: true`)
- **Sparse Vector Configuration**:
  - Vector Name: `"sparse"`
  - Modifier: `Modifier.IDF` (BM25 sparse tokens)
- **Deterministic Point ID Derivation**:
  $$\text{point\_id} = \text{MD5}(\text{chunk\_id})[:15]_{16} \pmod{2^{63}-1}$$
- **Indexed Payload Fields**:
  - `doc_id` (`KEYWORD`)
  - `court` (`KEYWORD`)
  - `court_slug` (`KEYWORD`)
  - `rhetorical_role` (`KEYWORD`)
  - `legal_domain` (`KEYWORD`)
  - `year` (`INTEGER`)

#### Payload Structure:
```json
{
  "chunk_id": "chunk_104774187_0005",
  "doc_id": "104774187",
  "court": "Delhi High Court",
  "court_slug": "delhi",
  "year": 2023,
  "rhetorical_role": "analysis",
  "text": "The rigors of Section 37 NDPS Act require the Court to be satisfied...",
  "text_with_context": "[Doc 104774187 | Delhi High Court 2023 | Role: analysis] ...",
  "word_count": 348,
  "char_count": 2150,
  "statute_mentions": ["NDPS Act Section 37", "CrPC Section 439"],
  "citations": ["2008 16 SCC 417"]
}
```

---

### 3.4 FalkorDB Graph Schema (`db/graph_store.py`)

- **Graph Identifier**: `citation_network`
- **Node Labels**:
  - `:Judgment`:
    - `doc_id` (String, Indexed)
    - `title` (String)
    - `court` (String)
    - `year` (Integer)
    - `judge` (String)
  - `:Precedent`:
    - `precedent_id` (String, Indexed) — Format: `prec_{normalized_citation}`
    - `canonical_citation` (String, Indexed)
    - `reporter` (String) — E.g. `SCC`, `SCR`, `AIR SC`, `DHC`
    - `year` (Integer)
    - `case_title` (String)
    - `in_degree` (Integer)
- **Relationship Type**:
  - `[:CITES]`:
    - `treatment`: String (`RELIED_UPON`, `FOLLOWED`, `DISTINGUISHED`, `OVERRULED`, `CONSIDERED`, `REFERRED_TO`)
    - `role`: String (Rhetorical role where citation occurs, e.g. `analysis`)
    - `context`: String (Surrounding sentence providing judicial context)

---

## 4. Algorithms, Mathematical Formulations & Component Logic

### 4.1 Okapi BM25 Lexical Scoring (`search.py`)

The lexical search component utilizes the standard Okapi BM25 algorithm with inverse document frequency (IDF) term smoothing:

$$\text{IDF}(t) = \ln\left(1 + \frac{N - \text{DF}(t) + 0.5}{\text{DF}(t) + 0.5}\right)$$

For document $D$ and query $Q = \{t_1, t_2, \dots, t_m\}$:

$$\text{Score}_{\text{BM25}}(D, Q) = \sum_{t \in Q} \text{IDF}(t) \cdot \frac{f(t, D) \cdot (k_1 + 1)}{f(t, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

- $k_1 = 1.5$ (Term frequency saturation parameter)
- $b = 0.75$ (Document length normalization parameter)
- $|D| = \text{doc\_len}$ (Token count of document)
- $\text{avgdl} = \frac{1}{N} \sum_{i=1}^N |D_i|$ (Average token count across corpus)

---

### 4.2 Reciprocal Rank Fusion (RRF) (`search.py`)

Combines ranked lists from dense vector search ($\text{Rank}_{\text{dense}}$) and sparse BM25 search ($\text{Rank}_{\text{sparse}}$):

$$\text{RRF\_Score}(d) = \frac{w_{\text{dense}}}{k + \text{Rank}_{\text{dense}}(d)} + \frac{w_{\text{sparse}}}{k + \text{Rank}_{\text{sparse}}(d)}$$

- $k = 20.0$ (RRF smoothing constant)
- $w_{\text{dense}} = 0.4$ (Dense semantic vector weight)
- $w_{\text{sparse}} = 0.6$ (Lexical BM25 weight, prioritizing exact statutory clauses)
- Evaluated over top 250 dense candidates and top 250 non-zero BM25 candidates.

---

### 4.3 Semantic Embeddings

#### Multi-Resolution Hash Embedder (`DenseSemanticEmbedder`)
Generates reproducible 384-dimensional unit vectors without requiring neural weights:
1. **Word-level projection**: MD5 hash $h_1 = \text{MD5}(w)[:8]_{16}$. Dimension index $i_1 = h_1 \pmod{384}$. Weight $+1.5$ or $-1.5$ determined by bit shift sign.
2. **Subword prefix projection**: 4-character prefix $w[:4]$ hashed via SHA-256 $h_2 \pmod{384}$, weight $\pm 1.0$.
3. **Subword suffix projection**: 4-character suffix $w[-4:]$ hashed via SHA-256 $h_3 \pmod{384}$, weight $\pm 1.0$.
4. **Unit $L_2$ Normalization**:
   $$\mathbf{v} \leftarrow \frac{\mathbf{v}}{\|\mathbf{v}\|_2 + 10^{-9}}$$

#### Production Transformer Embedder (`TransformerSemanticEmbedder`)
- **Underlying Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- **Inference**: Batch size 32, normalized embeddings enabled (`normalize_embeddings=True`).
- **Resilience**: Dynamic device selection (CUDA GPU if available, else CPU). If the `sentence-transformers` package is missing or the model cannot be instantiated, it transparently falls back to `DenseSemanticEmbedder`.

---

### 4.4 Rate Limiting & Circuit Breakers (`rate_limiter.py`)

- **Token Bucket Model**:
  - Capacity: $B = 5$ tokens.
  - Refill Rate: $r = 0.2$ tokens/second.
  - Inter-request Delay: Uniform random jitter $t \in [5.0\text{s}, 8.0\text{s}]$.
- **Adaptive Circuit Breaker**:
  - Triggers immediately upon HTTP 429 (Too Many Requests), HTTP 403 (Cloudflare Challenge), or structural DOM drift.
  - Pauses all concurrent worker threads for an adaptive cooldown:
    $$\Delta t = 15.0 \times 2^{\text{consecutive\_throttles}}$$

---

### 4.5 Article 141 Binding Precedent Traversal (`db/graph_store.py`)

Article 141 of the Constitution of India mandates:
> *"The law declared by the Supreme Court shall be binding on all courts within the territory of India."*

To enforce this constitutional rule, FalkorDB executes openCypher pattern matching:

```cypher
MATCH (j:Judgment {doc_id: $doc_id})-[r:CITES]->(p:Precedent)
WHERE (
    p.reporter = 'Supreme Court Cases (SCC)'
    OR p.reporter = 'SCR'
    OR p.reporter = 'AIR SC'
    OR p.reporter CONTAINS 'Supreme Court'
    OR toLower(p.canonical_citation) CONTAINS 'scc online sc'
    OR toLower(p.canonical_citation) CONTAINS 'insc'
    OR (toLower(p.canonical_citation) CONTAINS 'scc' 
        AND NOT toLower(p.canonical_citation) CONTAINS 'del' 
        AND NOT toLower(p.canonical_citation) CONTAINS 'bom' 
        AND NOT toLower(p.canonical_citation) CONTAINS 'mad' 
        AND NOT toLower(p.canonical_citation) CONTAINS 'kar')
)
RETURN p.canonical_citation AS canonical_citation,
       p.case_title AS case_title,
       p.reporter AS reporter,
       p.year AS year,
       r.treatment AS treatment,
       r.role AS rhetorical_role,
       r.context AS context,
       p.in_degree AS in_degree
ORDER BY p.year DESC, p.in_degree DESC
```

#### Court Hierarchy Influence Classifier
Categorizes the citation portfolio of a court into jurisdictional tiers:
- **Supreme Court of India**: Binding on all courts nationwide (Article 141).
- **Delhi High Court**: Coordinate / binding on subordinate trial courts in NCT of Delhi.
- **Bombay / Madras / Karnataka High Courts**: Persuasive authority.
- **Other Authorities / Tribunals**: Collateral persuasive value.

---

### 4.6 Statute-to-Ratio Reverse Lookup Engine (`rag/statute_lookup.py`)

Provides bidirectional resolution between statutory provisions and judicial doctrine:

1. **Canonical Act Normalization**:
   Resolves abbreviations and aliases (`crpc`, `ipc`, `cpc`, `ndps`, `arbitration`, `ni act`, `pca`, `pocso`, `uapa`) to full statutory titles:
   - `crpc` $\rightarrow$ `Code of Criminal Procedure, 1973`
   - `ndps` $\rightarrow$ `Narcotic Drugs and Psychotropic Substances Act, 1985`
   - `arbitration act` $\rightarrow$ `Arbitration and Conciliation Act, 1996`
2. **Section Identifier Normalization**:
   Regex `r"(\d+[A-Za-z]*)"` strips prefixes (e.g. `'Section 482'`, `'Sec. 482'`, `'s. 482'` $\rightarrow$ `'482'`).
3. **Multi-Tier Retrieval Pipeline**:
   - **Tier 1 (PostgreSQL)**: Queries `statute_mentions` joined with `chunks` (filtered to roles `analysis`, `operative`, `statute_reference`) and `headnotes` (`ratio_decidendi`).
   - **Tier 2 (Qdrant)**: Filters `legal_chunks` by payload containing target statute mentions and retrieves top semantic passages.
   - **Tier 3 (Filesystem Fallback)**: Scans `raw-store/{court}/{year}/{doc_id}/enriched.json` and `headnote.json` directly.
4. **Cross-Bench Principle Synthesis**:
   Aggregates and formats distilled legal principles formulated by presiding benches across the retrieved authorities.

---

## 5. Agentic Legal RAG & Reasoning Architecture

```
 User Legal Query
       │
       ▼
 ┌──────────────────────────────────────────────────────────┐
 │ LegalQueryPlanner                                        │
 │ • Detects Legal Domain (Criminal, Arbitration, etc.)     │
 │ • Decomposes into 3–5 Targeted SubQueries:               │
 │   - Statutory Provisions (Roles: statute_reference)      │
 │   - Binding Precedents (Roles: precedents, analysis)     │
 │   - Operative Reliefs (Roles: operative, facts)          │
 └────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Hybrid Search & Candidate Passage Retrieval              │
 │ • Executes Dense + BM25 RRF per sub-query                │
 │ • Caps total retrieved passages to max_total_chunks (15) │
 └────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ LegalShepardizer (Good-Law Audit)                        │
 │ • Audits all cited precedents in candidate passages      │
 │ • Queries FalkorDB for negative treatments:              │
 │   - OVERRULED -> Status: OVERRULED, Warning Emitted      │
 │   - DISTINGUISHED -> Status: DISTINGUISHED, Flagged      │
 │   - RELIED_UPON / FOLLOWED -> Status: GOOD_LAW           │
 └────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ LegalRAGSynthesizer                                      │
 │ • System Prompt: Senior Appellate Research Counsel       │
 │ • Strict Grounding: Every statement requires [1], [2]    │
 │ • Groq API (LLaMA-3.1-70B-Versatile) or Deterministic    │
 │   Extractive Fallback                                    │
 └────────────────────────────┬─────────────────────────────┘
                              │
                              ▼
 ┌──────────────────────────────────────────────────────────┐
 │ Output: GroundedAnswer                                   │
 │ • answer_text with inline verified citations             │
 │ • citations list (doc_id, chunk_id, role, text, score)   │
 │ • shepard_warnings (overruled or cautioned cases)        │
 │ • confidence score & latency telemetry                   │
 └──────────────────────────────────────────────────────────┘
```

---

## 6. Infrastructure & Docker Compose Deployment

The infrastructure stack is defined in `docker-compose.yml`:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    container_name: legaltrace-postgres
    restart: unless-stopped
    ports:
      - "${POSTGRES_PORT:-5431}:5432"
    environment:
      POSTGRES_DB: legaltrace
      POSTGRES_USER: legaltrace
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-legaltrace_dev_2024}
    volumes:
      - pg_data:/var/lib/postgresql/data
      - ./docker/postgres/init.sql:/docker-entrypoint-initdb.d/01-init.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U legaltrace -d legaltrace"]
      interval: 5s
      timeout: 3s
      retries: 10

  qdrant:
    image: qdrant/qdrant:v1.12.6
    container_name: legaltrace-qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__STORAGE__ON_DISK_PAYLOAD: "true"
    healthcheck:
      test: ["CMD", "bash", "-c", "exec 3<>/dev/tcp/localhost/6333"]
      interval: 5s
      timeout: 3s
      retries: 10

  falkordb:
    image: falkordb/falkordb:latest
    container_name: legaltrace-falkordb
    restart: unless-stopped
    ports:
      - "6379:6379"
      - "3000:3000"
    volumes:
      - falkor_data:/var/lib/falkordb/data
    environment:
      REDIS_ARGS: "--appendonly yes --save 60 1 --maxmemory 512mb --maxmemory-policy noeviction"
      FALKORDB_ARGS: "THREAD_COUNT 4 CACHE_SIZE 50"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  pg_data:
    name: legaltrace-pg-data
  qdrant_data:
    name: legaltrace-qdrant-data
  falkor_data:
    name: legaltrace-falkor-data
```

---

## 7. Complete CLI Command Reference (All 27 Commands)

| # | Command | Callback Signature & Parameters | Description |
| :--- | :--- | :--- | :--- |
| 1 | `discover` | `court: str = "Delhi High Court"`, `year: int = 2023`, `month: Optional[int]`, `limit: Optional[int]`, `output: Path = "discovered.jsonl"`, `source: str = "auto"`, `seed_file: Optional[Path]`, `cookies: Optional[str]` | Discover judgment candidates without downloading content |
| 2 | `fetch` | `input_file: Path`, `mode: str = "scrape_only"`, `limit: Optional[int]`, `workers: int = 1`, `delay: Optional[float]`, `cookies: Optional[str]`, `engine: Optional[str]` | Download and store raw judgments in parallel |
| 3 | `run` | `court: str = "Delhi High Court"`, `year: int = 2023`, `mode: str = "scrape_only"`, `limit: int = 20`, `workers: int = 1`, `delay: Optional[float]`, `cookies: Optional[str]` | Single-command discovery + acquisition pipeline |
| 4 | `resume` | `job_id: str`, `limit: Optional[int]`, `workers: int = 1` | Resume interrupted acquisition job from SQLite checkpoint |
| 5 | `report` | `job_id: str` | Render Rich summary table of metrics, errors, and stored documents |
| 6 | `check-robots` | `url: str` | Check whether URL path is allowed under Indian Kanoon `robots.txt` |
| 7 | `restructure` | `store_dir: Path = "./raw-store"` | Migrate flat files to hierarchical `{court_slug}/{year}/{doc_id}/` |
| 8 | `validate` | `store_dir: Path = "./raw-store"`, `doc_id: Optional[str]` | Verify SHA-256 checksums and minimum content word thresholds |
| 9 | `clean` | `store_dir: Path = "./raw-store"`, `output: Optional[Path]`, `doc_id: Optional[str]` | Repair Unicode, strip noise, segment paragraphs, extract disposition |
| 10 | `enrich` | `store_dir: Path = "./raw-store"`, `output: Optional[Path]`, `doc_id: Optional[str]` | Extract bench, judges, acts, sections, precedents, counsel |
| 11 | `structure` | `store_dir: Path = "./raw-store"`, `output: Optional[Path]`, `doc_id: Optional[str]` | Label paragraphs with 9 canonical legal rhetorical roles |
| 12 | `chunk` | `store_dir: Path = "./raw-store"`, `output: Optional[Path]`, `doc_id: Optional[str]`, `min_words: int = 100`, `target_words: int = 360`, `max_words: int = 600`, `overlap_paras: int = 1` | Role-isolated semantic chunker with contextual header injection |
| 13 | `graph` | `store_dir: Path = "./raw-store"`, `output_edges: Optional[Path]`, `output_nodes: Optional[Path]`, `output_graph_json: Optional[Path]`, `output_graphml: Optional[Path]` | Build directed citation network and classify judicial treatments |
| 14 | `headnote` | `store_dir: Path = "./raw-store"`, `output: Optional[Path]`, `doc_id: Optional[str]` | Generate law-reporter syllabi, catchwords, facts, ratio, holding |
| 15 | `export` | `output: Path = "legaltrace_export.parquet"`, `store_dir: Path = "./raw-store"` | Export raw store and chunks into compressed Apache Parquet tables |
| 16 | `index` | `input_chunks: Path = "chunks.jsonl"`, `index_dir: Path = "search_index"`, `embedder_type: str = "dense"` (`transformer` or `dense`) | Build dense vector embeddings and BM25 inverted lexical index |
| 17 | `search` | `query: str`, `index_dir: Path = "search_index"`, `mode: str = "hybrid"`, `top_k: int = 5`, `role: Optional[str]`, `court: Optional[str]`, `year: Optional[int]` | Execute hybrid, dense, or lexical queries with metadata filters |
| 18 | `benchmark` | `index_dir: Path = "search_index"`, `output: Path = "benchmark_report.json"`, `top_k: int = 5` | Run empirical evaluation against the 20-query Golden Legal QA Suite |
| 19 | `serve` | `host: str = "0.0.0.0"`, `port: int = 8000`, `reload: bool = False` | Launch production FastAPI server and embedded Tailwind SPA dashboard |
| 20 | `db-migrate` | `target: str` (`all`, `postgres`, `qdrant`, `falkordb`), `db_url: Optional[str]`, `qdrant_url: Optional[str]`, `falkor_host: Optional[str]`, `falkor_port: Optional[int]` | Initialize tables, vector collections, payload indices, and graph indexes |
| 21 | `rag-plan` | `query: str` | Decompose legal question into structured retrieval sub-tasks |
| 22 | `rag-query` | `query: str`, `max_chunks: int = 10`, `index_dir: Path = "search_index"` | Run end-to-end grounded RAG with Shepardization and inline citations |
| 23 | `shepardize` | `citation: str` | Audit judicial treatment history and verify whether precedent is good law |
| 24 | `memo` | `query_or_doc_id: str`, `doc_type: str = "memo"` (`case_brief`, `memo`, `bail_application`, `quashing_petition`), `output: Optional[Path]`, `index_dir: Path = "search_index"` | Automatically draft court-ready briefs, memos, and pleadings |
| 25 | `batch-ingest` | `courts: Optional[list[str]]`, `years: Optional[list[int]]`, `limit: Optional[int]`, `workers: int = 3`, `batch_size: int = 200`, `force: bool = False`, `mode: str = "scrape_only"` | Orchestrate nationwide multi-court batch ingestion directly into databases |
| 26 | `batch-status` | None | Query live status and progress of all batch ingestion jobs in database |
| 27 | `statute-lookup`| `act: str`, `section: str`, `limit: int = 10`, `store_dir: Path = "raw-store"` | Reverse lookup judicial ratio decidendi and precedents for Act & Section |

---

## 8. REST API Reference (All 18 Endpoints)

| Method | Route | Description | Request Payload / Query Parameters | Response Schema |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | System health and index telemetry | None | `{"status": "ok", "total_indexed_chunks": int, "dense_dimension": int, "bm25_vocabulary_size": int, "database_connections": dict}` |
| `POST` | `/api/search` | Sub-15ms Hybrid Semantic-Lexical Search | `{"query": str, "mode": "hybrid"|"dense"|"bm25", "top_k": int, "filter_role": str, "filter_court": str, "filter_year": int}` | `SearchResponse` (`query`, `mode`, `total_indexed`, `latency_ms`, `results`: list of `SearchResultItem`) |
| `GET` | `/api/judgments/{doc_id}` | Retrieve normalized judgment JSON | Path: `doc_id` | Full normalized judgment dictionary |
| `GET` | `/api/headnotes/{doc_id}` | Retrieve legal headnote & syllabus | Path: `doc_id` | `LegalHeadnote` schema |
| `GET` | `/api/citations/{doc_id}` | Outbound precedent citations & treatments | Path: `doc_id` | List of `{"cited_citation", "reporter", "cited_title", "treatment", "rhetorical_role", "context"}` |
| `GET` | `/api/precedents/top` | Top cited authorities across corpus | Query: `limit: int = 20` | List of `{"precedent_id", "canonical_citation", "reporter", "year", "case_title", "in_degree"}` |
| `GET` | `/api/benchmark` | Golden QA benchmark evaluation metrics | None | `{"metrics": {"precision_at_k", "recall_at_k", "hit_rate_at_k", "mrr", "ndcg_at_k"}, "query_count": int, "evaluation_timestamp": str}` |
| `POST` | `/api/rag/plan` | Decompose legal question into sub-tasks | `{"query": str}` | `QueryPlan` schema (`original_query`, `detected_domain`, `sub_queries`: list of `SubQuery`) |
| `POST` | `/api/rag/query` | Grounded legal Q&A with inline citations | `{"query": str, "max_chunks": int = 10}` | `GroundedAnswer` (`query`, `answer_text`, `citations`: list of `CitedPassage`, `confidence`, `shepard_warnings`) |
| `POST` | `/api/rag/memo` | Generate Case Brief, Memo, or Bail Draft | `{"query_or_doc_id": str, "doc_type": str, ...}` | `LegalDocument` (`doc_type`, `title`, `content`, `generated_at`, `citations`, `shepard_warnings`) |
| `GET` | `/api/shepardize/{citation}` | Precedent treatment audit & good-law check | Path: `citation` (URL-encoded) | `ShepardResult` (`precedent_citation`, `canonical_id`, `status`, `is_good_law`, `treatment_summary`, `overruling_cases`) |
| `GET` | `/api/graph/neighbors/{node_id}` | Node adjacency list in citation network | Path: `node_id` | `{"node_id": str, "total_outgoing": int, "outgoing_citations": list}` |
| `GET` | `/api/graph/binding/{doc_id}` | Supreme Court Article 141 binding chain | Path: `doc_id` | `{"doc_id": str, "total_binding_authorities": int, "article_141_declaration": str, "authorities": list}` |
| `GET` | `/api/graph/hierarchy` | Court hierarchy citation breakdown | Query: `court_slug: Optional[str]` | `{"court_scope": str, "total_citations": int, "hierarchy_breakdown": dict}` |
| `GET` | `/api/batch/status` | Active and historical batch ingestion jobs | None | List of `BatchIngestionJob.to_dict()` |
| `GET` | `/api/graph/visualization/{doc_id}` | SVG / D3 force-directed network payload | Path: `doc_id`, Query: `depth: int = 2`, `limit: int = 25` | `{"root_id": str, "nodes": [{"id", "label", "type", "court", "year"}], "links": [{"source", "target", "treatment"}], "stats": dict}` |
| `GET` | `/api/statutes/lookup` | Reverse lookup statute ratio (GET) | Query: `act: str`, `section: str`, `limit: int = 10` | `StatuteRatioResponse` (`act_name`, `section`, `canonical_act`, `total_results`, `key_principles`, `results`) |
| `POST` | `/api/statutes/ratio-lookup` | Reverse lookup statute ratio (POST) | `{"act": str, "section": str, "limit": int = 10}` | `StatuteRatioResponse` |
| `GET` | `/` | Web SPA Dashboard interface | None | Rendered HTML / CSS / JavaScript SPA |

---

## 9. Golden QA Evaluation Benchmark (`evaluation.py`)

LegalTrace includes an automated benchmark evaluating retrieval accuracy across 20 curated Golden Legal Queries:

$$\text{Precision@}K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$

$$\text{Recall@}K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$

$$\text{Hit Rate@}K = \mathbb{I}(|\text{Retrieved}_K \cap \text{Relevant}| > 0)$$

$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

$$\text{DCG@}K = \sum_{i=1}^K \frac{2^{\text{rel}_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$

Benchmark evaluation command:
```bash
ingest benchmark --index-dir ./search_index --output benchmark_report.json --top-k 5
```

---

## 10. Concurrency, Fault Tolerance & Resumption

1. **Atomic In-Flight Claiming (`dedupe.py`)**:
   SQLite transaction with `BEGIN IMMEDIATE` claims document IDs before worker threads issue network requests, preventing race conditions.
2. **Resumable Batch Checkpointing (`batch_ingest.py`)**:
   Every court-year batch creates a persistent `batch_ingestion_jobs` record. If interrupted, `ingest resume --job-id <ID>` or `ingest batch-ingest` resumes without duplicate work.
3. **Signal Trapping (`job.py`)**:
   Registers POSIX and Windows `SIGINT` / `SIGTERM` handlers to cleanly shut down thread pools, flush open SQLite transactions, and record partial progress.
