"""SQLAlchemy 2.0 ORM models for LegalTrace relational persistence in PostgreSQL.

Defines schemas for:
- Judgment: Core document entity, judicial header, and full metadata
- Chunk: 9,167+ semantic chunks with rhetorical roles and statutory references
- Headnote: Automated syllabi, catchwords, ratio decidendi, and operative holdings
- StatuteMention: Normalised statutory acts and section citations
- SearchSession: Query latency, telemetry, and retrieval audit trail
- RAGSession: Grounded legal reasoning Q&A logs with plan decomposition
"""

from __future__ import annotations

from typing import Any, Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ingestion_source.db.session import Base


class Judgment(Base):
    """Core legal judgment entity representing an ingested court decision."""

    __tablename__ = "judgments"

    doc_id: Mapped[str] = mapped_column(String(32), primary_key=True, doc="Indian Kanoon doc ID or canonical ID")
    title: Mapped[str] = mapped_column(Text, nullable=False, doc="Full case title or cause title")
    court: Mapped[str] = mapped_column(String(128), nullable=False, index=True, doc="Court name (e.g. Delhi High Court)")
    court_slug: Mapped[str] = mapped_column(String(64), nullable=False, doc="Normalized slug (e.g. delhi)")
    year: Mapped[int] = mapped_column(Integer, nullable=False, index=True, doc="Decision year")
    decision_date: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, doc="Decision date string YYYY-MM-DD")
    disposition: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True, doc="Disposal status (allowed, dismissed, etc.)")
    legal_domain: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True, doc="Legal domain (criminal, civil, arbitration, etc.)")
    case_type: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, doc="Case type description")
    bench_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, doc="Single bench, division bench, etc.")
    docket_numbers: Mapped[list[str]] = mapped_column(JSON, default=list, doc="List of docket/petition numbers")
    bench: Mapped[list[str]] = mapped_column(JSON, default=list, doc="Presiding judges as listed in header")
    canonical_judges: Mapped[list[str]] = mapped_column(JSON, default=list, doc="Normalized canonical judge names")
    petitioner: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Petitioner party name")
    respondent: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Respondent party name")
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), unique=True, nullable=True, doc="SHA-256 hash of original raw content")
    source_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Source URL on Indian Kanoon")
    total_paragraphs: Mapped[int] = mapped_column(Integer, default=0, doc="Cleaned paragraph count")
    total_statutes: Mapped[int] = mapped_column(Integer, default=0, doc="Distinct statute mentions")
    total_citations: Mapped[int] = mapped_column(Integer, default=0, doc="External precedents cited")
    word_count: Mapped[int] = mapped_column(Integer, default=0, doc="Full judgment word count")
    clean_full_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Normalized clean text for search fallback")
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    chunks: Mapped[list[Chunk]] = relationship(
        "Chunk",
        back_populates="judgment",
        cascade="all, delete-orphan",
        order_by="Chunk.chunk_index",
    )
    headnote: Mapped[Optional[Headnote]] = relationship(
        "Headnote",
        back_populates="judgment",
        uselist=False,
        cascade="all, delete-orphan",
    )
    statutes: Mapped[list[StatuteMention]] = relationship(
        "StatuteMention",
        back_populates="judgment",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Judgment(doc_id='{self.doc_id}', court='{self.court}', year={self.year}, title='{self.title[:40]}...')>"


class Chunk(Base):
    """Semantic legal chunk optimized for dense/sparse vector retrieval."""

    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True, doc="Format: chunk_{doc_id}_{idx}")
    doc_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("judgments.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, doc="0-based sequence index within document")
    rhetorical_role: Mapped[str] = mapped_column(String(32), nullable=False, index=True, doc="facts, analysis, operative, etc.")
    text: Mapped[str] = mapped_column(Text, nullable=False, doc="Clean chunk body text")
    text_with_context: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Enriched text prefixed with case metadata")
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    statute_mentions: Mapped[list[str]] = mapped_column(JSON, default=list, doc="Extracted acts and sections in this chunk")
    citations: Mapped[list[str]] = mapped_column(JSON, default=list, doc="Precedents referenced in this chunk")

    judgment: Mapped[Judgment] = relationship("Judgment", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_role_doc", "rhetorical_role", "doc_id"),
    )

    def __repr__(self) -> str:
        return f"<Chunk(chunk_id='{self.chunk_id}', role='{self.rhetorical_role}', words={self.word_count})>"


class Headnote(Base):
    """Automated legal headnote and syllabus for a judgment."""

    __tablename__ = "headnotes"

    doc_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("judgments.doc_id", ondelete="CASCADE"),
        primary_key=True,
    )
    catchwords: Mapped[list[str]] = mapped_column(JSON, default=list, doc="Topical legal catchword keywords")
    catchwords_line: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Formatted uppercase taxonomy line")
    brief_facts: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Factual matrix summary")
    issues: Mapped[list[str]] = mapped_column(JSON, default=list, doc="Framed legal questions")
    ratio_decidendi: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Binding legal principles formulated")
    key_precedents: Mapped[list[str]] = mapped_column(JSON, default=list, doc="Key precedents considered")
    operative_holding: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Final operative direction/order")
    formatted_headnote: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Law reporter publication format")
    word_count: Mapped[int] = mapped_column(Integer, default=0)

    judgment: Mapped[Judgment] = relationship("Judgment", back_populates="headnote")

    def __repr__(self) -> str:
        return f"<Headnote(doc_id='{self.doc_id}', catchwords_count={len(self.catchwords or [])})>"


class StatuteMention(Base):
    """Statutory provisions identified in a judgment."""

    __tablename__ = "statute_mentions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(
        String(32),
        ForeignKey("judgments.doc_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    act_name: Mapped[str] = mapped_column(String(256), nullable=False, index=True, doc="Canonical act name")
    act_code: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, doc="Short identifier (e.g. NDPS, CrPC)")
    sections: Mapped[list[str]] = mapped_column(JSON, default=list, doc="List of sections referenced")
    raw_mention: Mapped[Optional[str]] = mapped_column(Text, nullable=True, doc="Verbatim textual mention")

    judgment: Mapped[Judgment] = relationship("Judgment", back_populates="statutes")

    def __repr__(self) -> str:
        return f"<StatuteMention(doc_id='{self.doc_id}', act='{self.act_name}', sections={self.sections})>"


class SearchSession(Base):
    """Search audit logging and performance telemetry."""

    __tablename__ = "search_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(16), nullable=False, default="hybrid", doc="hybrid, dense, bm25")
    filters: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    result_count: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    top_result_doc_id: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<SearchSession(id={self.id}, query='{self.query[:30]}...', latency={self.latency_ms}ms)>"


class RAGSession(Base):
    """RAG session logging grounded synthesis with citations and query decomposition."""

    __tablename__ = "rag_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    citations_used: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    plan_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())

    def __repr__(self) -> str:
        return f"<RAGSession(id={self.id}, query='{self.query[:30]}...', conf={self.confidence})>"


class BatchIngestionJob(Base):
    """Tracks per court-year batch ingestion progress, checkpoints, and telemetry."""

    __tablename__ = "batch_ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    batch_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, doc="Unique batch identifier (e.g. batch_delhi_2023)")
    court: Mapped[str] = mapped_column(String(128), nullable=False, index=True, doc="Court name")
    court_slug: Mapped[str] = mapped_column(String(64), nullable=False, index=True, doc="Normalized court slug")
    year: Mapped[int] = mapped_column(Integer, nullable=False, index=True, doc="Target decision year")
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True, doc="pending, running, completed, failed")
    docs_discovered: Mapped[int] = mapped_column(Integer, default=0)
    docs_fetched: Mapped[int] = mapped_column(Integer, default=0)
    docs_indexed: Mapped[int] = mapped_column(Integer, default=0)
    chunks_created: Mapped[int] = mapped_column(Integer, default=0)
    edges_created: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    started_at: Mapped[Optional[Any]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[Any]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[Any] = mapped_column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    def to_dict(self) -> dict[str, Any]:
        """Convert batch job record to dictionary."""
        return {
            "id": self.id,
            "batch_id": self.batch_id,
            "court": self.court,
            "court_slug": self.court_slug,
            "year": self.year,
            "status": self.status,
            "docs_discovered": self.docs_discovered,
            "docs_fetched": self.docs_fetched,
            "docs_indexed": self.docs_indexed,
            "chunks_created": self.chunks_created,
            "edges_created": self.edges_created,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self) -> str:
        return f"<BatchIngestionJob(batch_id='{self.batch_id}', status='{self.status}', indexed={self.docs_indexed})>"
