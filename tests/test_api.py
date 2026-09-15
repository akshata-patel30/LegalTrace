"""Unit tests for Step 11: Production REST API and Interactive UI."""

from __future__ import annotations

from pathlib import Path
import pytest
from starlette.testclient import TestClient

from ingestion_source.api import app

client = TestClient(app)


def test_dashboard_html():
    """Verify that root / serves the interactive SPA explorer."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "LegalTrace" in response.text
    assert "Delhi High Court" in response.text


def test_health_check():
    """Verify GET /health returns engine status and index dimensions."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total_indexed_chunks"] >= 9000
    assert data["dense_dimension"] == 384
    assert data["bm25_vocabulary_size"] > 30000


def test_search_api_hybrid():
    """Verify POST /api/search with hybrid retrieval."""
    payload = {
        "query": "regular bail commercial quantity Section 37 NDPS Act",
        "mode": "hybrid",
        "top_k": 5,
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == payload["query"]
    assert data["mode"] == "hybrid"
    assert len(data["results"]) == 5
    assert data["results"][0]["rank"] == 1
    assert data["results"][0]["score"] > 0.0
    assert data["latency_ms"] > 0.0


def test_search_api_role_filter():
    """Verify POST /api/search with rhetorical role filter."""
    payload = {
        "query": "arbitration challenge patent illegality Section 34",
        "mode": "hybrid",
        "top_k": 5,
        "filter_role": "analysis",
    }
    response = client.post("/api/search", json=payload)
    assert response.status_code == 200
    data = response.json()
    for item in data["results"]:
        assert item["rhetorical_role"] == "analysis"


def test_get_judgment_metadata():
    """Verify GET /api/judgments/{doc_id} retrieves case details."""
    response = client.get("/api/judgments/104774187")
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "104774187"
    assert "Quentin Decon" in data["title"]
    assert data["court"] == "Delhi High Court"
    assert data["year"] == 2023


def test_get_judgment_not_found():
    """Verify GET /api/judgments/99999999 returns 404."""
    response = client.get("/api/judgments/99999999")
    assert response.status_code == 404


def test_get_headnote():
    """Verify GET /api/headnotes/{doc_id} returns structured legal syllabus."""
    response = client.get("/api/headnotes/104774187")
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "104774187"
    assert "CRIMINAL" in data["catchwords_line"]
    assert len(data["brief_facts"]) > 50
    assert len(data["operative_holding"]) > 10


def test_get_citations():
    """Verify GET /api/citations/{doc_id} returns citations cited by case."""
    response = client.get("/api/citations/104774187")
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == "104774187"
    assert "total_citations" in data
    assert isinstance(data["citations"], list)


def test_get_top_precedents():
    """Verify GET /api/precedents/top returns landmark authorities."""
    response = client.get("/api/precedents/top?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert len(data) <= 10
    if data:
        assert "canonical_citation" in data[0]
        assert "in_degree" in data[0]
        assert data[0]["in_degree"] >= data[-1]["in_degree"]


def test_get_benchmark_report():
    """Verify GET /api/benchmark returns Step 10 evaluation report."""
    response = client.get("/api/benchmark")
    assert response.status_code == 200
    data = response.json()
    assert "hybrid_metrics" in data
    assert "bm25_metrics" in data
    assert "dense_metrics" in data


# ---------------------------------------------------------------------------
# Phase 1: Online RAG and Shepardization Endpoint Tests
# ---------------------------------------------------------------------------

def test_rag_plan_endpoint():
    """Verify POST /api/rag/plan decomposes complex queries into sub-tasks."""
    payload = {"query": "regular bail conditions commercial quantity Section 37 NDPS Act"}
    response = client.post("/api/rag/plan", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["original_query"] == payload["query"]
    assert "sub_queries" in data
    assert len(data["sub_queries"]) >= 1
    assert "detected_domain" in data


def test_rag_query_endpoint():
    """Verify POST /api/rag/query retrieves evidence and synthesizes grounded answer."""
    payload = {
        "query": "What are the principles for grant of regular bail under Section 37 NDPS Act?",
        "max_evidence_chunks": 4,
    }
    response = client.post("/api/rag/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer_text" in data
    assert len(data["answer_text"]) > 20
    assert "citations" in data
    assert len(data["citations"]) > 0
    assert "confidence" in data
    assert data["confidence"] > 0.0
    assert "model_used" in data


def test_rag_memo_endpoint_memo():
    """Verify POST /api/rag/memo generates formal legal research memorandum."""
    payload = {
        "query": "Section 34 patent illegality challenge Arbitration Act",
        "doc_type": "memo",
    }
    response = client.post("/api/rag/memo", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["doc_type"] == "legal_memorandum"
    assert "LEGAL RESEARCH MEMORANDUM" in data["content"]
    assert "citations" in data


def test_rag_memo_endpoint_brief():
    """Verify POST /api/rag/memo generates case brief for document ID."""
    payload = {
        "doc_type": "brief",
        "doc_id": "104774187",
        "court": "delhi",
        "year": 2023,
    }
    response = client.post("/api/rag/memo", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["doc_type"] == "case_brief"
    assert "APPELLATE CHAMBER CASE BRIEF" in data["content"]
    assert "Quentin Decon" in data["title"]


def test_rag_memo_endpoint_brief_missing_id():
    """Verify POST /api/rag/memo returns 400 when doc_id is missing for case brief."""
    payload = {
        "doc_type": "brief",
    }
    response = client.post("/api/rag/memo", json=payload)
    assert response.status_code == 400


def test_rag_memo_endpoint_bail():
    """Verify POST /api/rag/memo generates Section 439 CrPC court bail draft."""
    payload = {
        "doc_type": "bail",
        "applicant_name": "Ramesh Kumar",
        "fir_no": "101/2023",
        "police_station": "Special Cell",
        "sections_invoked": ["Section 37 NDPS Act", "Section 439 CrPC"],
        "custody_date": "01.01.2023",
        "facts_summary": "The applicant was arrested on alleged recovery of commercial contraband without independent witnesses.",
    }
    response = client.post("/api/rag/memo", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["doc_type"] == "bail_application"
    assert "SECTION 439" in data["content"]
    assert "RAMESH KUMAR" in data["content"]


def test_shepardize_endpoint():
    """Verify GET /api/shepardize/{citation} returns good-law audit report."""
    response = client.get("/api/shepardize/2023:DHC:9999")
    assert response.status_code == 200
    data = response.json()
    assert "precedent_citation" in data
    assert "status" in data
    assert "is_good_law" in data
    assert "treatment_summary" in data


def test_graph_neighbors_endpoint():
    """Verify GET /api/graph/neighbors/{node_id} returns graph adjacency."""
    response = client.get("/api/graph/neighbors/888001")
    assert response.status_code == 200
    data = response.json()
    assert data["node_id"] == "888001"
    assert "total_outgoing" in data
    assert data["total_outgoing"] >= 1
    assert "outgoing_citations" in data

