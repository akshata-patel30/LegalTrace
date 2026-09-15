"""Unit tests for LegalTrace SQLAlchemy 2.0 ORM models and async session management."""

from __future__ import annotations

import pytest
from sqlalchemy import select

from ingestion_source.db.models import (
    Chunk,
    Headnote,
    Judgment,
    RAGSession,
    SearchSession,
    StatuteMention,
)
from ingestion_source.db.session import (
    Base,
    close_db,
    create_tables,
    drop_tables,
    get_session_factory,
    init_db,
)


@pytest.fixture
async def async_db():
    """Fixture providing an in-memory SQLite database for testing."""
    test_db_url = "sqlite+aiosqlite:///:memory:"
    init_db(test_db_url, echo=False)
    await create_tables()

    factory = get_session_factory()
    async with factory() as session:
        yield session

    await drop_tables()
    await close_db()


@pytest.mark.asyncio
async def test_judgment_crud_and_relationships(async_db):
    """Verify Judgment creation, foreign key cascade, and relationship loading."""
    # 1. Create Judgment
    j = Judgment(
        doc_id="12345",
        title="State of Delhi vs. A. Kumar",
        court="Delhi High Court",
        court_slug="delhi",
        year=2023,
        decision_date="2023-05-15",
        disposition="allowed",
        legal_domain="criminal",
        case_type="Bail Application",
        bench_type="single_bench",
        docket_numbers=["BAIL APPLN. 101/2023"],
        bench=["Hon'ble Mr. Justice S. Sharma"],
        canonical_judges=["S. Sharma"],
        petitioner="A. Kumar",
        respondent="State of Delhi",
        content_hash="hash123",
        source_url="https://indiankanoon.org/doc/12345/",
        total_paragraphs=10,
        total_statutes=1,
        total_citations=2,
        word_count=1200,
        clean_full_text="Clean judgment text here...",
    )
    async_db.add(j)
    await async_db.commit()

    # 2. Add Chunk
    c1 = Chunk(
        chunk_id="chunk_12345_0",
        doc_id="12345",
        chunk_index=0,
        rhetorical_role="facts",
        text="The applicant was arrested on...",
        text_with_context="[Context] The applicant was arrested on...",
        char_count=35,
        word_count=6,
        statute_mentions=["Section 439 CrPC"],
        citations=["2020 SCC OnLine Del 10"],
    )
    async_db.add(c1)

    # 3. Add Headnote
    hn = Headnote(
        doc_id="12345",
        catchwords=["CRIMINAL", "BAIL", "SECTION 439"],
        catchwords_line="CRIMINAL — BAIL — SECTION 439",
        brief_facts="Applicant sought regular bail.",
        issues=["Whether bail should be granted."],
        ratio_decidendi="Bail is the rule and jail is the exception.",
        key_precedents=["2020 SCC OnLine Del 10"],
        operative_holding="Bail application is allowed.",
        formatted_headnote="Formatted Headnote...",
        word_count=50,
    )
    async_db.add(hn)

    # 4. Add Statute Mention
    st = StatuteMention(
        doc_id="12345",
        act_name="Code of Criminal Procedure, 1973",
        act_code="CrPC",
        sections=["Section 439"],
        raw_mention="Section 439 of Cr.P.C.",
    )
    async_db.add(st)
    await async_db.commit()

    # 5. Query and Assert
    stmt = select(Judgment).where(Judgment.doc_id == "12345")
    res = await async_db.execute(stmt)
    retrieved_j = res.scalar_one()

    assert retrieved_j.title == "State of Delhi vs. A. Kumar"
    assert retrieved_j.court == "Delhi High Court"
    assert retrieved_j.year == 2023

    # Verify relationships
    chunks_stmt = select(Chunk).where(Chunk.doc_id == "12345")
    c_res = await async_db.execute(chunks_stmt)
    retrieved_chunks = c_res.scalars().all()
    assert len(retrieved_chunks) == 1
    assert retrieved_chunks[0].rhetorical_role == "facts"

    hn_stmt = select(Headnote).where(Headnote.doc_id == "12345")
    hn_res = await async_db.execute(hn_stmt)
    retrieved_hn = hn_res.scalar_one()
    assert retrieved_hn.operative_holding == "Bail application is allowed."
    assert "SECTION 439" in retrieved_hn.catchwords

    st_stmt = select(StatuteMention).where(StatuteMention.doc_id == "12345")
    st_res = await async_db.execute(st_stmt)
    retrieved_st = st_res.scalars().all()
    assert len(retrieved_st) == 1
    assert retrieved_st[0].act_code == "CrPC"


@pytest.mark.asyncio
async def test_search_and_rag_session_logging(async_db):
    """Verify SearchSession and RAGSession audit trail tables."""
    s_sess = SearchSession(
        query="regular bail commercial quantity NDPS",
        mode="hybrid",
        filters={"rhetorical_role": "analysis", "year": 2023},
        result_count=10,
        latency_ms=12.5,
        top_result_doc_id="12345",
    )
    async_db.add(s_sess)

    rag_sess = RAGSession(
        query="What are the conditions for bail under Section 37 NDPS?",
        answer="Under Section 37 NDPS Act, the court must be satisfied that...",
        citations_used=[{"doc_id": "12345", "citation": "2023 DHC 101"}],
        confidence=0.92,
        plan_json={"sub_queries": ["Section 37 NDPS conditions"]},
        latency_ms=350.0,
    )
    async_db.add(rag_sess)
    await async_db.commit()

    # Query Search Session
    stmt = select(SearchSession).where(SearchSession.mode == "hybrid")
    res = await async_db.execute(stmt)
    s = res.scalar_one()
    assert s.query == "regular bail commercial quantity NDPS"
    assert s.latency_ms == 12.5

    # Query RAG Session
    stmt_rag = select(RAGSession).where(RAGSession.confidence > 0.9)
    res_rag = await async_db.execute(stmt_rag)
    rag = res_rag.scalar_one()
    assert "Section 37 NDPS" in rag.answer
    assert rag.confidence == 0.92
