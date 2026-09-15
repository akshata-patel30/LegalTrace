"""Unit tests for LegalQueryPlanner query decomposition engine."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch
import pytest

from ingestion_source.config import Settings
from ingestion_source.rag.planner import (
    LegalQueryPlanner,
    QueryPlan,
    SubQuery,
    SubQueryType,
)


def test_planner_bail_pattern():
    """Verify rule-based decomposition for regular and anticipatory bail queries."""
    planner = LegalQueryPlanner()
    plan = planner.plan("What are the grounds for regular bail in commercial quantity NDPS case?")

    assert plan.detected_domain == "criminal"
    assert len(plan.sub_queries) >= 2
    assert any(sq.query_type == SubQueryType.STATUTORY for sq in plan.sub_queries)
    assert any(sq.query_type == SubQueryType.PRECEDENT for sq in plan.sub_queries)

    # Verify rhetorical role filters
    statutory_sq = next(sq for sq in plan.sub_queries if sq.query_type == SubQueryType.STATUTORY)
    assert "statute_reference" in statutory_sq.target_roles or "analysis" in statutory_sq.target_roles


def test_planner_quashing_pattern():
    """Verify decomposition for Section 482 CrPC quashing petitions."""
    planner = LegalQueryPlanner()
    plan = planner.plan("Can High Court quash FIR under Section 482 based on compromise?")

    assert plan.detected_domain == "criminal"
    assert len(plan.sub_queries) >= 2
    assert any("482" in sq.query_text for sq in plan.sub_queries)


def test_planner_arbitration_pattern():
    """Verify decomposition for Section 34 arbitration challenge."""
    planner = LegalQueryPlanner()
    plan = planner.plan("Grounds for setting aside arbitral award under Section 34 patent illegality")

    assert plan.detected_domain == "arbitration"
    assert len(plan.sub_queries) >= 2
    assert any(sq.query_type == SubQueryType.STATUTORY for sq in plan.sub_queries)


def test_planner_heuristic_fallback():
    """Verify general novel query falls back to entity heuristics without crashing."""
    planner = LegalQueryPlanner(settings=Settings(GROQ_API_KEY=None))
    plan = planner.plan("Environmental clearances required for riverfront infrastructure projects")

    assert plan.detected_domain == "general"
    assert len(plan.sub_queries) == 3
    assert plan.sub_queries[0].query_type == SubQueryType.STATUTORY
    assert plan.sub_queries[1].query_type == SubQueryType.PRECEDENT
    assert plan.sub_queries[2].query_type == SubQueryType.FACTUAL


def test_planner_empty_query():
    """Verify empty query returns empty plan."""
    planner = LegalQueryPlanner()
    plan = planner.plan("   ")
    assert plan.original_query == ""
    assert len(plan.sub_queries) == 0


def test_planner_groq_mocked():
    """Verify Groq LLM-assisted query decomposition with mocked client."""
    mock_settings = Settings(GROQ_API_KEY="mock_key_xyz", RAG_LLM_MODEL="llama-3.1-70b-versatile")
    planner = LegalQueryPlanner(settings=mock_settings)

    mock_llm_json = {
        "detected_domain": "taxation",
        "sub_queries": [
            {
                "query_text": "Section 148 notice limitation income escaping assessment",
                "query_type": "statutory",
                "target_roles": ["statute_reference", "analysis"],
                "required": True,
                "top_k": 5,
            },
            {
                "query_text": "High Court reassessment notice reassessment proceedings jurisdictional defect",
                "query_type": "precedent",
                "target_roles": ["precedents", "analysis"],
                "required": True,
                "top_k": 5,
            },
        ],
    }

    with patch("groq.Groq") as mock_groq_cls:
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps(mock_llm_json)
        mock_resp = MagicMock(choices=[mock_choice])
        mock_client.chat.completions.create.return_value = mock_resp
        mock_groq_cls.return_value = mock_client

        plan = planner.plan("Jurisdiction to issue notice under Section 148 Income Tax Act")

        assert plan.detected_domain == "taxation"
        assert len(plan.sub_queries) == 2
        assert plan.sub_queries[0].query_type == SubQueryType.STATUTORY
        assert plan.sub_queries[1].query_type == SubQueryType.PRECEDENT
