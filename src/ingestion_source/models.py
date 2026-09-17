"""Data models and schemas for the LegalTrace acquisition module."""

from __future__ import annotations

from datetime import date as dt_date, datetime as dt_datetime, timezone
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator


def get_utc_now() -> dt_datetime:
    """Helper returning timezone-aware current UTC datetime."""
    return dt_datetime.now(timezone.utc)


class SourceType(str, Enum):
    """Provenance source of the acquired judgment."""
    API = "api"
    SCRAPE = "scrape"


class IngestionStatus(str, Enum):
    """Lifecycle status of a document in the acquisition stage."""
    PENDING = "pending"
    STORED = "stored"
    FAILED = "failed"
    SKIPPED = "skipped"


class FetchFailureType(str, Enum):
    """Typed classification of document fetch failures."""
    RATE_LIMITED = "rate_limited"
    NOT_FOUND = "not_found"
    FORBIDDEN = "forbidden"
    TRANSIENT = "transient"
    DISALLOWED_BY_ROBOTS = "disallowed_by_robots"
    BUDGET_EXCEEDED = "budget_exceeded"
    STRUCTURAL_DRIFT = "structural_drift"
    CHALLENGE_DETECTED = "challenge_detected"
    UNKNOWN = "unknown"


class DocRef(BaseModel):
    """Reference to a discovered document on Indian Kanoon."""
    doc_id: str
    court: str
    court_slug: str = ""
    year: int
    month: Optional[int] = None
    title: Optional[str] = None
    source_url: str
    doctype: str = "judgment"
    headline: Optional[str] = None

    @field_validator("court_slug", mode="before")
    @classmethod
    def set_court_slug(cls, v: str, values: Any) -> str:
        if v:
            return v
        return ""


class FetchResult(BaseModel):
    """Result of fetching a raw document from an acquisition source."""
    ref: DocRef
    success: bool
    source: SourceType
    raw_content: Optional[str] = None
    raw_json: Optional[dict[str, Any]] = None
    status_code: Optional[int] = None
    failure_type: Optional[FetchFailureType] = None
    error_message: Optional[str] = None
    fetched_at: dt_datetime = Field(default_factory=get_utc_now)
    latency_ms: Optional[float] = None


class CitationMetrics(BaseModel):
    """Citation counters and navigation references."""
    cites_count: int = 0
    cited_by_count: int = 0
    cites_url: Optional[str] = None
    cited_by_url: Optional[str] = None


class CitedDocument(BaseModel):
    """Reference to an external act, section, or cited judgment."""
    doc_id: str
    text: str
    url: str


class JudgmentParagraph(BaseModel):
    """Structured paragraph or section of a judgment."""
    index: int
    id: Optional[str] = None
    tag: str = "p"
    structure_tag: Optional[str] = None
    text: str


class Parties(BaseModel):
    """Litigating parties extracted from document title or header."""
    petitioner: Optional[str] = None
    respondent: Optional[str] = None


class StructuredJudgment(BaseModel):
    """Clean, high-fidelity canonical legal judgment document."""
    doc_id: str
    title: str
    court: str
    court_slug: str
    date: Optional[str] = None
    year: Optional[int] = None
    neutral_citation: Optional[str] = None
    case_numbers: list[str] = Field(default_factory=list)
    parties: Optional[Parties] = None
    bench: list[str] = Field(default_factory=list)
    author: Optional[str] = None
    citations: list[str] = Field(default_factory=list)
    citation_metrics: CitationMetrics = Field(default_factory=CitationMetrics)
    cited_documents: list[CitedDocument] = Field(default_factory=list)
    paragraphs: list[JudgmentParagraph] = Field(default_factory=list)
    full_text: str
    source: SourceType
    source_url: str
    content_hash: str
    fetched_at: str


class RawJudgment(BaseModel):
    """Canonical schema for raw legal judgment data ready for Pipeline A."""
    doc_id: str
    source: SourceType
    source_url: str
    fetched_at: dt_datetime = Field(default_factory=get_utc_now)
    content_hash: str
    court: str
    court_slug: str = ""
    title: str
    date: Optional[dt_date] = None
    bench: list[str] = Field(default_factory=list)
    author: Optional[str] = None
    neutral_citation: Optional[str] = None
    case_numbers: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    cites_count: int = 0
    cited_by_count: int = 0
    doctype: str = "judgment"
    citations_raw: Optional[str] = None
    full_text: Optional[str] = None
    full_text_path: Optional[str] = None
    html_raw_path: Optional[str] = None
    json_raw_path: Optional[str] = None
    structured_path: Optional[str] = None
    structured: Optional[StructuredJudgment] = None
    ingestion_status: IngestionStatus = IngestionStatus.PENDING
    error: Optional[str] = None


class ManifestEntry(BaseModel):
    """Single-line entry in the append-only manifest.jsonl ledger."""
    doc_id: str
    source: SourceType
    court: str
    court_slug: str
    year: int
    title: str
    date: Optional[str] = None
    content_hash: str
    full_text_path: str
    metadata_path: str
    judgment_json_path: Optional[str] = None
    raw_content_path: Optional[str] = None
    ingestion_status: IngestionStatus = IngestionStatus.STORED
    fetched_at: str


class JobItemStatus(str, Enum):
    """Status of an individual task inside a job run."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobItem(BaseModel):
    """Work item tracked in the SQLite job queue."""
    job_id: str
    doc_id: str
    court: str
    court_slug: str
    year: int
    source_url: str
    status: JobItemStatus = JobItemStatus.PENDING
    failure_type: Optional[FetchFailureType] = None
    error_message: Optional[str] = None
    attempts: int = 0
    last_attempt_at: Optional[dt_datetime] = None


class JobState(BaseModel):
    """Summary state of an acquisition job."""
    job_id: str
    started_at: dt_datetime = Field(default_factory=get_utc_now)
    updated_at: dt_datetime = Field(default_factory=get_utc_now)
    mode: str = "api"
    total_discovered: int = 0
    fetched_count: int = 0
    stored_count: int = 0
    skipped_count: int = 0
    failed_count: int = 0
    api_calls_count: int = 0
    status: str = "running"


class JobReport(BaseModel):
    """Formatted report emitted by `ingest report`."""
    job_id: str
    status: str
    mode: str
    started_at: dt_datetime
    updated_at: dt_datetime
    duration_seconds: float
    total_discovered: int
    stored_count: int
    skipped_count: int
    failed_count: int
    api_calls_count: int
    error_summary: dict[str, int] = Field(default_factory=dict)


class ParagraphType(str, Enum):
    HEADING = "heading"
    PREAMBLE = "preamble"
    FACTS = "facts"
    ARGUMENTS = "arguments"
    ANALYSIS = "analysis"
    STATUTORY_QUOTE = "statutory_quote"
    CASE_QUOTE = "case_quote"
    OPERATIVE = "operative"
    SIGNATURE = "signature"
    BODY = "body"


class DispositionType(str, Enum):
    ALLOWED = "allowed"
    DISMISSED = "dismissed"
    DISPOSED_OF = "disposed_of"
    PARTLY_ALLOWED = "partly_allowed"
    REFERRED = "referred"
    REMANDED = "remanded"
    SET_ASIDE = "set_aside"
    INTERIM_ORDER = "interim_order"
    OTHER = "other"


class RhetoricalRole(str, Enum):
    PREAMBLE = "preamble"
    FACTS = "facts"
    ISSUE = "issue"
    ARGUMENTS_PETITIONER = "arguments_petitioner"
    ARGUMENTS_RESPONDENT = "arguments_respondent"
    STATUTE_REFERENCE = "statute_reference"
    PRECEDENTS = "precedents"
    ANALYSIS = "analysis"
    OPERATIVE = "operative"


class CleanedParagraph(BaseModel):
    para_id: int
    label: Optional[str] = None
    text: str
    para_type: ParagraphType = ParagraphType.BODY
    is_quote: bool = False
    rhetorical_role: RhetoricalRole = RhetoricalRole.ANALYSIS
    char_count: int = 0
    word_count: int = 0


class CleanedHeader(BaseModel):
    court: Optional[str] = None
    docket_numbers: list[str] = Field(default_factory=list)
    neutral_citation: Optional[str] = None
    decision_date: Optional[str] = None
    reserved_date: Optional[str] = None
    petitioner: Optional[str] = None
    respondent: Optional[str] = None
    petitioner_counsel: list[str] = Field(default_factory=list)
    respondent_counsel: list[str] = Field(default_factory=list)
    bench: list[str] = Field(default_factory=list)
    header_raw: str = ""


class CleaningMetrics(BaseModel):
    raw_char_count: int
    cleaned_char_count: int
    noise_removed_chars: int
    reduction_percentage: float
    paragraphs_count: int
    disposition: DispositionType
    noise_patterns_detected: list[str] = Field(default_factory=list)


class CleanedJudgment(BaseModel):
    doc_id: str
    court: str
    court_slug: str
    year: int
    title: str
    header: CleanedHeader
    paragraphs: list[CleanedParagraph] = Field(default_factory=list)
    operative_order: Optional[str] = None
    disposition: DispositionType = DispositionType.OTHER
    clean_full_text: str
    metrics: CleaningMetrics
    cleaned_at: dt_datetime = Field(default_factory=get_utc_now)


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ValidationIssue(BaseModel):
    doc_id: str
    severity: ValidationSeverity
    field: str
    message: str


class DocumentValidationResult(BaseModel):
    doc_id: str
    is_valid: bool
    quality_score: float  # 0 to 100
    char_count: int
    content_hash_matched: bool
    issues: list[ValidationIssue] = Field(default_factory=list)


class ValidationReport(BaseModel):
    total_documents: int
    valid_documents: int
    invalid_documents: int
    total_errors: int
    total_warnings: int
    average_quality_score: float
    documents_with_warnings: int
    issues_breakdown: dict[str, int] = Field(default_factory=dict)
    results: list[DocumentValidationResult] = Field(default_factory=list)


class LegalDomain(str, Enum):
    CRIMINAL = "criminal"
    CIVIL = "civil"
    CONSTITUTIONAL = "constitutional"
    COMMERCIAL = "commercial"
    ARBITRATION = "arbitration"
    TAXATION = "taxation"
    LABOUR = "labour"
    FAMILY = "family"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    OTHER = "other"


class BenchType(str, Enum):
    SINGLE_BENCH = "single_bench"
    DIVISION_BENCH = "division_bench"
    FULL_BENCH = "full_bench"
    CONSTITUTION_BENCH = "constitution_bench"
    UNKNOWN = "unknown"


class StatutoryMention(BaseModel):
    act_name: str
    act_code: str
    sections: list[str] = Field(default_factory=list)
    raw_mention: str


class ExtractedCitation(BaseModel):
    citation_text: str
    reporter: str
    year: Optional[int] = None


class EnrichedLegalMetadata(BaseModel):
    doc_id: str
    case_type: str
    case_type_code: str
    legal_domain: LegalDomain
    bench_type: BenchType
    canonical_judges: list[str] = Field(default_factory=list)
    statutes: list[StatutoryMention] = Field(default_factory=list)
    citations: list[ExtractedCitation] = Field(default_factory=list)
    petitioner: Optional[str] = None
    respondent: Optional[str] = None
    senior_advocates: list[str] = Field(default_factory=list)
    standing_counsels: list[str] = Field(default_factory=list)
    total_statutes_count: int = 0
    total_citations_count: int = 0


class LegalChunk(BaseModel):
    """Semantic legal chunk optimized for vector embedding and RAG retrieval."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    court: str
    court_slug: str
    year: int
    case_title: str
    docket_number: Optional[str] = None
    presiding_judge: Optional[str] = None
    legal_domain: Optional[str] = None
    case_type: Optional[str] = None
    rhetorical_role: RhetoricalRole
    para_ids: list[int] = Field(default_factory=list)
    text: str
    text_with_context: str
    char_count: int
    word_count: int
    statute_mentions: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    is_quote_block: bool = False


class ChunkingReport(BaseModel):
    """Corpus-wide chunking aggregation and distribution report."""
    total_documents: int
    total_chunks: int
    average_chunks_per_doc: float
    average_words_per_chunk: float
    average_chars_per_chunk: float
    role_distribution: dict[str, int] = Field(default_factory=dict)


class CitationTreatment(str, Enum):
    """Substantive judicial treatment accorded to a cited precedent."""
    RELIED_UPON = "relied_upon"
    FOLLOWED = "followed"
    DISTINGUISHED = "distinguished"
    OVERRULED = "overruled"
    REFERRED = "referred"
    CITED = "cited"


class CitationEdge(BaseModel):
    """Directed precedent citation edge from citing case to cited authority."""
    edge_id: str
    citing_doc_id: str
    citing_court: str
    citing_year: int
    citing_title: str
    citing_judge: Optional[str] = None
    citing_para_id: Optional[int] = None
    rhetorical_role: RhetoricalRole = RhetoricalRole.PRECEDENTS
    treatment: CitationTreatment = CitationTreatment.CITED
    cited_citation: str
    cited_reporter: str
    cited_year: Optional[int] = None
    cited_title: Optional[str] = None
    context_sentence: str
    is_internal_corpus_match: bool = False
    target_doc_id: Optional[str] = None


class PrecedentNode(BaseModel):
    """Precedent entity node aggregating citation metrics across the network."""
    precedent_id: str
    canonical_citation: str
    reporter: str
    citation_year: Optional[int] = None
    case_title: Optional[str] = None
    in_degree: int = 0
    citing_doc_ids: list[str] = Field(default_factory=list)
    treatments: dict[str, int] = Field(default_factory=dict)
    roles: dict[str, int] = Field(default_factory=dict)
    is_internal: bool = False
    doc_id: Optional[str] = None


class CitationGraphSummary(BaseModel):
    """Summary topology and authority metrics of the precedent network."""
    total_citing_documents: int
    total_edges: int
    total_precedent_nodes: int
    internal_edges_count: int
    external_edges_count: int
    top_precedents: list[dict[str, Any]] = Field(default_factory=list)
    treatment_breakdown: dict[str, int] = Field(default_factory=dict)
    role_breakdown: dict[str, int] = Field(default_factory=dict)


class LegalHeadnote(BaseModel):
    """Structured publication-grade legal headnote (syllabi) and executive summary."""
    doc_id: str
    court: str
    court_slug: str
    year: int
    case_title: str
    bench: list[str] = Field(default_factory=list)
    docket_number: Optional[str] = None
    decision_date: Optional[str] = None
    legal_domain: Optional[str] = None
    case_type: Optional[str] = None
    disposition: DispositionType = DispositionType.OTHER
    catchwords: list[str] = Field(default_factory=list)
    catchwords_line: str = ""
    brief_facts: str = ""
    issues: list[str] = Field(default_factory=list)
    petitioner_arguments: list[str] = Field(default_factory=list)
    respondent_arguments: list[str] = Field(default_factory=list)
    ratio_decidendi: str = ""
    key_precedents: list[str] = Field(default_factory=list)
    operative_holding: str = ""
    formatted_headnote: str = ""
    word_count: int = 0
    generated_at: dt_datetime = Field(default_factory=get_utc_now)


class HeadnoteReport(BaseModel):
    """Batch headnote generation summary metrics."""
    total_documents: int
    headnotes_generated: int
    average_headnote_words: float
    documents_with_facts: int
    documents_with_issues: int
    documents_with_ratio: int
    documents_with_holding: int


class SearchResultItem(BaseModel):
    """Individual ranked result from vector/lexical/hybrid search."""
    rank: int
    score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    chunk_id: str
    doc_id: str
    court: str
    year: int
    case_title: str
    rhetorical_role: RhetoricalRole
    text_snippet: str
    text_with_context: str
    statute_mentions: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Container for search query results and timing telemetry."""
    query: str
    mode: str
    total_indexed: int
    latency_ms: float
    results: list[SearchResultItem] = Field(default_factory=list)


class QueryCategory(str, Enum):
    STATUTORY = "statutory"
    PRECEDENT = "precedent"
    FACTUAL = "factual"
    PROCEDURAL = "procedural"


class BenchmarkQuery(BaseModel):
    """Individual curated evaluation query with ground-truth relevant documents and chunks."""
    query_id: str
    query_text: str
    category: QueryCategory
    ground_truth_doc_ids: list[str]
    ground_truth_chunk_ids: list[str] = Field(default_factory=list)
    target_roles: list[RhetoricalRole] = Field(default_factory=list)
    description: str = ""


class QueryEvaluationMetric(BaseModel):
    """Evaluation metrics for a single query and retrieval mode."""
    query_id: str
    mode: str
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    hit_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    latency_ms: float
    first_relevant_rank: Optional[int] = None


class ModeAggregateMetrics(BaseModel):
    """Aggregated retrieval effectiveness metrics for an entire mode."""
    mode: str
    mrr: float
    precision_at_1: float
    precision_at_3: float
    precision_at_5: float
    precision_at_10: float
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    hit_at_1: float
    hit_at_3: float
    hit_at_5: float
    hit_at_10: float
    ndcg_at_5: float
    ndcg_at_10: float
    average_latency_ms: float


class BenchmarkEvaluationReport(BaseModel):
    """Consolidated legal retrieval evaluation report comparing BM25, Dense, and Hybrid."""
    total_queries: int
    category_counts: dict[str, int]
    dense_metrics: ModeAggregateMetrics
    bm25_metrics: ModeAggregateMetrics
    hybrid_metrics: ModeAggregateMetrics
    evaluated_at: dt_datetime = Field(default_factory=get_utc_now)







