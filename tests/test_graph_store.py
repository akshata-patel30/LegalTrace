"""Unit tests for FalkorDB graph store integration and openCypher queries."""

from __future__ import annotations

from unittest.mock import MagicMock
import pytest

from ingestion_source.db.graph_store import FalkorGraphStore, normalize_precedent_key


def test_normalize_precedent_key():
    """Verify consistent slugification of legal citations."""
    key1 = normalize_precedent_key("2023 SCC OnLine Del 1234")
    assert key1 == "prec_2023_scc_online_del_1234"

    key2 = normalize_precedent_key("AIR 1978 SC 597 (Maneka Gandhi)")
    assert "maneka" in key2
    assert key2.startswith("prec_")

    # Idempotent structure
    key3 = normalize_precedent_key("  (2021) 5  SCC  1  ")
    assert key3 == "prec_2021_5_scc_1"


def test_ensure_indexes():
    """Verify graph index queries are executed."""
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph

    store = FalkorGraphStore(client=mock_db, graph_name="test_graph")
    store.ensure_indexes()

    assert mock_graph.query.call_count == 3


def test_get_precedent_treatment():
    """Verify Shepardization treatment lookup openCypher query and result parsing."""
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph

    mock_result = MagicMock()
    mock_result.result_set = [
        ["1001", "Union of India vs. X", "Delhi High Court", 2023, "relied_upon", "analysis", "The principles in Maneka Gandhi were applied..."],
        ["1002", "State vs. Y", "Delhi High Court", 2022, "distinguished", "analysis", "Facts differ significantly..."],
    ]
    mock_graph.query.return_value = mock_result

    store = FalkorGraphStore(client=mock_db, graph_name="test_graph")
    treatments = store.get_precedent_treatment("AIR 1978 SC 597")

    assert len(treatments) == 2
    assert treatments[0]["citing_doc_id"] == "1001"
    assert treatments[0]["treatment"] == "relied_upon"
    assert treatments[1]["treatment"] == "distinguished"


def test_get_top_authorities():
    """Verify landmark authority ranking openCypher query."""
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph

    mock_result = MagicMock()
    mock_result.result_set = [
        ["prec_1978_sc_597", "AIR 1978 SC 597", "Maneka Gandhi vs. Union of India", "AIR", 42],
        ["prec_1973_sc_1461", "AIR 1973 SC 1461", "Kesavananda Bharati vs. State of Kerala", "AIR", 38],
    ]
    mock_graph.query.return_value = mock_result

    store = FalkorGraphStore(client=mock_db, graph_name="test_graph")
    authorities = store.get_top_authorities(limit=10)

    assert len(authorities) == 2
    assert authorities[0]["rank"] == 1
    assert authorities[0]["canonical_citation"] == "AIR 1978 SC 597"
    assert authorities[0]["in_degree"] == 42


def test_find_common_authorities():
    """Verify multi-judgment co-citation query."""
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph

    mock_result = MagicMock()
    mock_result.result_set = [
        ["AIR 1978 SC 597", "Maneka Gandhi", 3, ["doc_1", "doc_2", "doc_3"]],
    ]
    mock_graph.query.return_value = mock_result

    store = FalkorGraphStore(client=mock_db, graph_name="test_graph")
    common = store.find_common_authorities(doc_ids=["doc_1", "doc_2", "doc_3"])

    assert len(common) == 1
    assert common[0]["common_count"] == 3
    assert common[0]["citing_docs"] == ["doc_1", "doc_2", "doc_3"]


def test_get_graph_stats():
    """Verify graph statistics retrieval."""
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph

    j_res = MagicMock(result_set=[[203]])
    p_res = MagicMock(result_set=[[2360]])
    e_res = MagicMock(result_set=[[2854]])
    mock_graph.query.side_effect = [j_res, p_res, e_res]

    store = FalkorGraphStore(client=mock_db, graph_name="test_graph")
    stats = store.get_graph_stats()

    assert stats["total_judgments"] == 203
    assert stats["total_precedents"] == 2360
    assert stats["total_edges"] == 2854


def test_get_binding_authority_chain():
    """Verify Article 141 Supreme Court binding authority chain retrieval."""
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph

    mock_result = MagicMock()
    mock_result.result_set = [
        ["(2016) 3 SCC 379", "Union of India v. Mohanlal", "Supreme Court Cases (SCC)", 2016, "relied_upon", "analysis", "Context sentence...", 12],
        ["AIR 1978 SC 597", "Maneka Gandhi vs. Union of India", "AIR SC", 1978, "cited", "precedents", "Context sentence...", 42],
    ]
    mock_graph.query.return_value = mock_result

    store = FalkorGraphStore(client=mock_db, graph_name="test_graph")
    authorities = store.get_binding_authority_chain(doc_id="104774187")

    assert len(authorities) == 2
    assert authorities[0]["canonical_citation"] == "(2016) 3 SCC 379"
    assert authorities[0]["is_binding"] is True
    assert "Article 141" in authorities[0]["binding_basis"]
    assert authorities[1]["canonical_citation"] == "AIR 1978 SC 597"


def test_get_court_hierarchy_influence():
    """Verify hierarchical citation influence breakdown across jurisdiction tiers."""
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph

    mock_result = MagicMock()
    mock_result.result_set = [
        ["Supreme Court (Binding Art. 141)", 2473],
        ["Delhi High Court (Coordinate/Binding)", 72],
        ["Other High Courts / Authorities", 302],
    ]
    mock_graph.query.return_value = mock_result

    store = FalkorGraphStore(client=mock_db, graph_name="test_graph")
    influence = store.get_court_hierarchy_influence(court_slug="delhi")

    assert influence["court_scope"] == "delhi"
    assert influence["total_citations"] == 2847
    assert influence["hierarchy_breakdown"]["Supreme Court (Binding Art. 141)"] == 2473
    assert influence["hierarchy_breakdown"]["Delhi High Court (Coordinate/Binding)"] == 72


def test_ingest_edges_and_nodes_streaming():
    """Verify direct on-the-fly streaming of edges and precedent nodes into FalkorDB."""
    mock_db = MagicMock()
    mock_graph = MagicMock()
    mock_db.select_graph.return_value = mock_graph

    store = FalkorGraphStore(client=mock_db, graph_name="test_graph")

    edges = [
        {
            "citing_doc_id": "9991",
            "cited_citation": "(2020) 2 SCC 1",
            "cited_reporter": "SCC",
            "cited_year": 2020,
            "cited_title": "Vidya Drolia vs. Durga Trading",
            "treatment": "relied_upon",
            "rhetorical_role": "analysis",
            "citing_para_id": 5,
            "context_sentence": "The fourfold test in Vidya Drolia applies...",
        }
    ]
    judgment_info = {
        "doc_id": "9991",
        "title": "Arbitration Appeal Judgment",
        "court": "Delhi High Court",
        "year": 2023,
        "judge": "Justice Prathiba M. Singh",
    }

    res = store.ingest_edges_and_nodes(edges=edges, judgment_info=judgment_info)

    assert res["judgments_ingested"] == 1
    assert res["precedents_ingested"] == 1
    assert res["edges_created"] == 1
    assert mock_graph.query.call_count >= 3
    assert mock_db.connection.save.called
