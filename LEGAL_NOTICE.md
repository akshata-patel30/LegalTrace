# Legal & Compliance Notice — LegalTrace Acquisition Module

## 1. Purpose & Scope
This acquisition module (`ingestion-source`) is developed exclusively for non-commercial, academic research as part of the **LegalTrace** project. Its sole function is to acquire a bounded sample corpus of Indian court judgments and legal materials to study automated legal document processing, citation extraction, and retrieval systems.

## 2. API-First Policy
- The official Indian Kanoon API (`api.indiankanoon.org`) is the **primary, preferred, and authorized** acquisition channel.
- All high-volume operations must use valid API authentication tokens and stay strictly within pre-configured budget thresholds (`MAX_API_CALLS_PER_RUN`).

## 3. Fallback Web Scraping & Politeness Posture
Direct HTML fetching is only utilized as a secondary fallback for bounded sample queries where API coverage is unavailable, subject to the following non-negotiable operating rules:
1. **Dynamic `robots.txt` Compliance**: The service parses `https://indiankanoon.org/robots.txt` at runtime and strictly respects all `Disallow` directives and `Crawl-delay` parameters. Any request to a disallowed path is rejected immediately.
2. **Transparent User-Agent**: Every HTTP request transmits a descriptive `User-Agent` identifying the research project, contact email, and academic purpose (`LegalTraceAcademicBot/0.1`). Impersonation of standard desktop browsers is strictly prohibited.
3. **Aggressive Rate-Limiting**: Scraper requests are throttled with a mandatory minimum delay of 3–5 seconds with random jitter, operating on a single thread.
4. **Exponential Backoff**: HTTP `429 Too Many Requests` or `503 Service Unavailable` responses immediately trigger exponential backoff.
5. **No Bypass of Protective Controls**: The service will never attempt to bypass paywalls, CAPTCHAs, authentication gates, or IP blocks.
6. **Bounded Sample Volume**: Crawls are restricted to small, predetermined subsets (hundreds to low thousands of documents) for demo and evaluation sets, rather than full-site mirrors.

## 4. Data Retention, Attribution & Redistribution
- **No Redistribution**: Acquired raw documents and indexes are retained solely for internal research processing within the LegalTrace pipeline and will not be mirrored, published, or redistributed as a standalone public dataset.
- **Attribution & Provenance**: Every ingested judgment record permanently records its original Indian Kanoon document ID (`doc_id`), canonical `source_url`, retrieval timestamp, and SHA-256 content hash, enabling full auditability and direct citations back to the original source.
