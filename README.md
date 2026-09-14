# LegalTrace — Indian Kanoon Data Acquisition & Ingestion Module (`ingestion-source`)

The `ingestion-source` service is the legal document acquisition subsystem for **LegalTrace**. It acquires, validates, normalizes, deduplicates, and stores raw legal judgments from Indian Kanoon to feed **Pipeline A (Step 1: Document Intake)**.

Because official Indian Kanoon API approval can take time or be unavailable, the service features a **high-performance, polite, multithreaded scraper** as its primary default engine, alongside the official API client.

---

## 1. Key Features

- **Primary Polite Multithreaded Scraper (`scrape_only` mode)**:
  - Traverses `/browse/` hierarchy by court, year, and month with pagination.
  - Multithreaded parallel downloads (`--workers 3` by default, configurable).
  - Runtime `robots.txt` parsing and strict path compliance.
  - DOM structural-drift detection to reject corrupted or incomplete pages.
  - Cloudflare Turnstile challenge detection with automatic backoff.
  - Optional custom session/clearance cookie support (`--cookies` / `INDIANKANOON_COOKIES`).
- **Coordinated Rate Limiting & Adaptive Backoff**:
  - Global `TokenBucketLimiter` coordinated across all concurrent worker threads.
  - Circuit breaker: automatically pauses all threads for an adaptive cool-down (default 15s) upon receiving HTTP 429 / challenge.
- **Deep Multi-Level Deduplication**:
  - **In-flight atomic claim**: Prevents parallel workers from fetching the same `doc_id` simultaneously.
  - **Pre-fetch SQLite dedupe**: Skips any document already recorded as `STORED`.
  - **Post-fetch Content Hash dedupe**: Computes SHA-256 over normalized plain text; duplicate texts across different petition IDs are recorded without duplicate file writes.
- **Unified Canonical Schema**: Standardized `RawJudgment` model with Indian court date parsing and bench extraction.
- **Pipeline A Hand-off**: Writes structured directories under `raw-store/{court_slug}/{year}/{doc_id}/` and appends atomically to `manifest.jsonl`.
- **Resumable Execution**: Fully tracked in SQLite, handles `Ctrl+C` gracefully, resumes seamlessly with `ingest resume`.

---

## 2. Quickstart

### Prerequisites
- Python 3.11+ (or [uv](https://github.com/astral-sh/uv))

### Setup Virtual Environment
```bash
cd ingestion-source
uv venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

uv pip install -e ".[dev]"
```

### Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Default `.env` settings:
```ini
MODE=scrape_only
ENABLE_HTML_FALLBACK=true
MAX_WORKERS=3
SCRAPE_MIN_DELAY_SECONDS=2.0
SCRAPE_MAX_DELAY_SECONDS=4.0
ADAPTIVE_BACKOFF_SECONDS=15.0
INTAKE_OUTPUT_PATH=./raw-store
```

---

## 3. CLI Usage

### 1. Check `robots.txt` Compliance
```bash
ingest check-robots --url "https://indiankanoon.org/doc/1940428/"
```

### 2. Discover Judgments (No content download)
```bash
ingest discover --court "Delhi High Court" --year 2023 --limit 50 --output discovered.jsonl
```

### 3. Fetch with Multithreading & Deduplication
```bash
# Fetch with 3 parallel workers in scrape_only mode
ingest fetch --input discovered.jsonl --mode scrape_only --workers 3 --limit 20
```

### 4. One-Step Automated Run
```bash
ingest run --court "Delhi High Court" --year 2023 --mode scrape_only --workers 3 --limit 10
```

### 5. Resume an Interrupted Run
```bash
ingest resume --job-id <JOB_ID> --workers 3
```

### 6. Display Metrics Report
```bash
ingest report --job-id <JOB_ID>
```

---

## 4. Output Storage & Hand-off Layout

```
raw-store/
├── manifest.jsonl                     # Append-only queue ledger for Pipeline A
├── job_state/
│   └── legaltrace_ingest.sqlite       # SQLite dedupe store & job progress state
└── delhi/
    └── 2023/
        └── 1940428/
            ├── full_text.txt          # Normalized plain text
            ├── metadata.json          # RawJudgment schema
            └── raw.html               # Scraped HTML payload (or raw.json for API)
```

---

## 5. Running Tests

Run the complete 37-test suite offline (no external network calls):
```bash
pytest tests/ -v
```
