# Interface Contract: Acquisition Module → Pipeline A (Document Intake)

This document formalizes the handoff contract between the **Acquisition Layer (`ingestion-source/`)** and **Pipeline A (Step 1: Document Intake)** in the LegalTrace architecture.

---

## 1. Storage Organization

The acquisition module stores all ingested judgments under the configured `INTAKE_OUTPUT_PATH` (default: `./raw-store`).

```
raw-store/
├── manifest.jsonl                     # Append-only ledger of intake-ready judgments
├── job_state/
│   └── legaltrace_ingest.sqlite       # SQLite dedupe store & job state database
└── {court_slug}/
    └── {year}/
        └── {doc_id}/
            ├── full_text.txt          # Normalized plain-text judgment content
            ├── metadata.json          # Canonical RawJudgment metadata document
            ├── raw.json               # Original API response (if source == 'api')
            └── raw.html               # Original HTML document (if source == 'scrape')
```

---

## 2. Manifest Schema (`manifest.jsonl`)

Each line in `manifest.jsonl` is a standalone, single-line JSON object emitted immediately after a document and its metadata have been successfully written and verified on disk.

```json
{
  "doc_id": "1940428",
  "source": "api",
  "court": "Delhi High Court",
  "court_slug": "delhi",
  "year": 2023,
  "title": "Union Of India vs M/S Bright Power Projects",
  "date": "2023-01-16",
  "content_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "full_text_path": "raw-store/delhi/2023/1940428/full_text.txt",
  "metadata_path": "raw-store/delhi/2023/1940428/metadata.json",
  "raw_content_path": "raw-store/delhi/2023/1940428/raw.json",
  "ingestion_status": "stored",
  "fetched_at": "2026-09-02T10:00:00Z"
}
```

---

## 3. Metadata Document Schema (`metadata.json`)

The `metadata.json` file inside each judgment folder contains the complete `RawJudgment` model:

| Field | Type | Description |
|---|---|---|
| `doc_id` | `string` | Unique Indian Kanoon document identifier |
| `source` | `"api"` \| `"scrape"` | Acquisition channel provenance |
| `source_url` | `string` | Full source URL where document was retrieved |
| `fetched_at` | `ISO 8601 string` | UTC timestamp of fetch |
| `content_hash` | `string` | SHA-256 hash of normalized full_text |
| `court` | `string` | Official name of the court (e.g., "Delhi High Court") |
| `court_slug` | `string` | URL-safe slug for the court |
| `title` | `string` | Case title |
| `date` | `string (YYYY-MM-DD)` \| `null` | Judgment/order decision date |
| `bench` | `list[string]` | List of presiding judge names |
| `doctype` | `string` | Document type (`judgment`, `order`, `act`, etc.) |
| `citations_raw` | `string` \| `null` | Raw citation string/block extracted from document header |
| `full_text_path` | `string` | Relative path to normalized plain-text file |
| `html_raw_path` | `string` \| `null` | Relative path to audit HTML copy |
| `json_raw_path` | `string` \| `null` | Relative path to raw API response payload |
| `ingestion_status` | `"stored"` \| `"failed"` \| `"pending"` | Pipeline ingestion status |
| `error` | `string` \| `null` | Error diagnostic if status is failed |

---

## 4. Pipeline A Intake Consumption Protocol

1. **Intake Trigger**: Pipeline A consumers tail or poll `manifest.jsonl` using a line-offset pointer.
2. **Readiness Guarantee**: A record is only appended to `manifest.jsonl` **after** `full_text.txt` and `metadata.json` are completely flushed and closed on disk.
3. **Idempotency**: Downstream pipeline steps can verify the `content_hash` in `metadata.json` against the SHA-256 digest of `full_text.txt` to guarantee data integrity before proceeding with OCR, chunking, and indexing.
