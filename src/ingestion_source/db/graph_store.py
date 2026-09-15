"""FalkorDB graph store integration for precedent citation networks.

Manages openCypher queries, node/edge schema, Shepardization treatment history,
and landmark authority network rankings across High Court and Supreme Court decisions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import re
from typing import Any, Optional

from falkordb import FalkorDB

logger = logging.getLogger(__name__)

DEFAULT_GRAPH = "citation_network"


def normalize_precedent_key(citation_str: str) -> str:
    """Normalize raw citation string into a consistent canonical precedent ID."""
    clean = re.sub(r"[^\w\s]", " ", citation_str)
    parts = clean.split()
    return "prec_" + "_".join(p.lower() for p in parts if p)


class FalkorGraphStore:
    """Manages the FalkorDB citation graph with openCypher queries."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        graph_name: str = DEFAULT_GRAPH,
        client: Optional[FalkorDB] = None,
    ) -> None:
        """Initialize connection to FalkorDB.

        Args:
            host: FalkorDB / Redis host.
            port: FalkorDB port (default 6379).
            graph_name: Graph key identifier.
            client: Optional pre-configured FalkorDB instance.
        """
        self.host = host
        self.port = port
        self.graph_name = graph_name
        self.db = client or FalkorDB(host=host, port=port)
        self.graph = self.db.select_graph(graph_name)

    def ensure_indexes(self) -> None:
        """Create graph indexes for fast entity lookup."""
        index_queries = [
            "CREATE INDEX FOR (j:Judgment) ON (j.doc_id)",
            "CREATE INDEX FOR (p:Precedent) ON (p.precedent_id)",
            "CREATE INDEX FOR (p:Precedent) ON (p.canonical_citation)",
        ]
        for q in index_queries:
            try:
                self.graph.query(q)
            except Exception as e:
                logger.debug("Index query ignored (may already exist): %s (%s)", q, e)

    def ingest_from_jsonl(
        self,
        edges_path: Path | str,
        nodes_path: Path | str,
        batch_size: int = 200,
    ) -> dict[str, int]:
        """Bulk load citation edges and precedent nodes from Pipeline A JSONL files.

        Args:
            edges_path: Path to citation_edges.jsonl.
            nodes_path: Path to precedent_nodes.jsonl.
            batch_size: Cypher batch operation window.

        Returns:
            Dictionary with counts of judgments, precedents, and edges processed.
        """
        edges_p = Path(edges_path)
        nodes_p = Path(nodes_path)

        if not edges_p.exists():
            raise FileNotFoundError(f"Citation edges file not found: {edges_p}")
        if not nodes_p.exists():
            raise FileNotFoundError(f"Precedent nodes file not found: {nodes_p}")

        self.ensure_indexes()

        judgments_created = 0
        precedents_created = 0
        edges_created = 0

        # Step 1: Ingest Precedent Nodes
        logger.info("Ingesting precedent nodes into FalkorDB from %s...", nodes_p)
        with open(nodes_p, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                node = json.loads(line_str)
                pid = str(node.get("precedent_id") or normalize_precedent_key(node.get("canonical_citation", "")))
                cit = str(node.get("canonical_citation", ""))
                reporter = str(node.get("reporter", "Unknown"))
                year = int(node.get("citation_year") or 0)
                title = str(node.get("case_title") or "")
                in_degree = int(node.get("in_degree", 0))

                self.graph.query(
                    """
                    MERGE (p:Precedent {precedent_id: $pid})
                    SET p.canonical_citation = $cit,
                        p.reporter = $reporter,
                        p.year = $year,
                        p.case_title = $title,
                        p.in_degree = $in_degree
                    """,
                    {
                        "pid": pid,
                        "cit": cit,
                        "reporter": reporter,
                        "year": year,
                        "title": title,
                        "in_degree": in_degree,
                    },
                )
                precedents_created += 1

        # Step 2: Ingest Citing Judgments and Edges from citation_edges.jsonl
        logger.info("Ingesting citation edges and citing judgments from %s...", edges_p)
        seen_judgments: set[str] = set()

        with open(edges_p, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                edge = json.loads(line_str)

                doc_id = str(edge.get("citing_doc_id", ""))
                citing_title = str(edge.get("citing_title", ""))
                court = str(edge.get("citing_court", "Delhi High Court"))
                citing_year = int(edge.get("citing_year") or 2023)
                judge = str(edge.get("citing_judge") or "")

                # Merge Citing Judgment node if not seen in this pass
                if doc_id and doc_id not in seen_judgments:
                    seen_judgments.add(doc_id)
                    self.graph.query(
                        """
                        MERGE (j:Judgment {doc_id: $doc_id})
                        SET j.title = $title,
                            j.court = $court,
                            j.year = $year,
                            j.judge = $judge
                        """,
                        {
                            "doc_id": doc_id,
                            "title": citing_title,
                            "court": court,
                            "year": citing_year,
                            "judge": judge,
                        },
                    )
                    judgments_created += 1

                # Derive target precedent ID
                raw_cit = str(edge.get("cited_citation", ""))
                prec_id = normalize_precedent_key(raw_cit)

                # Ensure precedent node exists as fallback if not in precedent_nodes.jsonl
                self.graph.query(
                    """
                    MERGE (p:Precedent {precedent_id: $pid})
                    ON CREATE SET p.canonical_citation = $cit,
                                  p.reporter = $reporter,
                                  p.year = $year,
                                  p.case_title = $title,
                                  p.in_degree = 1
                    """,
                    {
                        "pid": prec_id,
                        "cit": raw_cit,
                        "reporter": str(edge.get("cited_reporter") or "Unknown"),
                        "year": int(edge.get("cited_year") or 0),
                        "title": str(edge.get("cited_title") or ""),
                    },
                )

                treatment = str(edge.get("treatment") or "cited")
                role = str(edge.get("rhetorical_role") or "precedents")
                para_id = edge.get("citing_para_id")
                context = str(edge.get("context_sentence") or "")[:500]
                edge_id = str(edge.get("edge_id") or f"e_{doc_id}_{prec_id}")

                self.graph.query(
                    """
                    MATCH (j:Judgment {doc_id: $doc_id}), (p:Precedent {precedent_id: $prec_id})
                    CREATE (j)-[:CITES {
                        edge_id: $edge_id,
                        treatment: $treatment,
                        role: $role,
                        para_id: $para_id,
                        context: $context
                    }]->(p)
                    """,
                    {
                        "doc_id": doc_id,
                        "prec_id": prec_id,
                        "edge_id": edge_id,
                        "treatment": treatment,
                        "role": role,
                        "para_id": para_id,
                        "context": context,
                    },
                )
                edges_created += 1

        stats = {
            "judgments_ingested": judgments_created,
            "precedents_ingested": precedents_created,
            "edges_created": edges_created,
        }
        self.save()
        logger.info("FalkorDB ingestion complete and snapshot saved: %s", stats)
        return stats

    def save(self) -> bool:
        """Persist graph and Redis state synchronously to disk (RDB / AOF)."""
        try:
            self.db.connection.save()
            logger.info("Successfully persisted FalkorDB snapshot to disk via Redis SAVE.")
            return True
        except Exception as err:
            logger.warning("FalkorDB SAVE returned warning/error: %s", err)
            return False

    # ── Analytical and Shepardization Queries ──

    def get_precedent_treatment(self, citation: str) -> list[dict[str, Any]]:
        """Shepardize a legal citation to review how courts have treated it (good law audit)."""
        clean_cit = citation.strip()
        result = self.graph.query(
            """
            MATCH (j:Judgment)-[r:CITES]->(p:Precedent)
            WHERE toLower(p.canonical_citation) CONTAINS toLower($cit)
            RETURN j.doc_id AS citing_doc_id,
                   j.title AS citing_title,
                   j.court AS citing_court,
                   j.year AS citing_year,
                   r.treatment AS treatment,
                   r.role AS rhetorical_role,
                   r.context AS context
            ORDER BY j.year DESC
            """,
            {"cit": clean_cit},
        )
        records = []
        for row in result.result_set:
            records.append({
                "citing_doc_id": row[0],
                "citing_title": row[1],
                "citing_court": row[2],
                "citing_year": row[3],
                "treatment": row[4],
                "rhetorical_role": row[5],
                "context": row[6],
            })
        return records

    def get_top_authorities(self, limit: int = 25) -> list[dict[str, Any]]:
        """Rank legal precedents across the corpus by incoming citation frequency."""
        result = self.graph.query(
            """
            MATCH (j:Judgment)-[r:CITES]->(p:Precedent)
            RETURN p.precedent_id AS precedent_id,
                   p.canonical_citation AS canonical_citation,
                   p.case_title AS case_title,
                   p.reporter AS reporter,
                   count(j) AS in_degree
            ORDER BY in_degree DESC
            LIMIT $limit
            """,
            {"limit": limit},
        )
        authorities = []
        for idx, row in enumerate(result.result_set):
            authorities.append({
                "rank": idx + 1,
                "precedent_id": row[0],
                "canonical_citation": row[1],
                "case_title": row[2] or "Landmark Authority",
                "reporter": row[3],
                "in_degree": row[4],
            })
        return authorities

    def get_judgment_citations(self, doc_id: str) -> list[dict[str, Any]]:
        """Retrieve all outbound precedents cited by a given judgment."""
        result = self.graph.query(
            """
            MATCH (j:Judgment {doc_id: $doc_id})-[r:CITES]->(p:Precedent)
            RETURN p.canonical_citation AS cited_citation,
                   p.reporter AS reporter,
                   p.case_title AS cited_title,
                   r.treatment AS treatment,
                   r.role AS rhetorical_role,
                   r.context AS context
            """,
            {"doc_id": str(doc_id)},
        )
        citations = []
        for row in result.result_set:
            citations.append({
                "cited_citation": row[0],
                "reporter": row[1],
                "cited_title": row[2],
                "treatment": row[3],
                "rhetorical_role": row[4],
                "context": row[5],
            })
        return citations

    def find_common_authorities(self, doc_ids: list[str]) -> list[dict[str, Any]]:
        """Discover shared precedent authorities cited by multiple input judgments."""
        result = self.graph.query(
            """
            MATCH (j:Judgment)-[r:CITES]->(p:Precedent)
            WHERE j.doc_id IN $ids
            WITH p, count(DISTINCT j) AS common_count, collect(j.doc_id) AS citing_docs
            WHERE common_count >= 2
            RETURN p.canonical_citation AS citation,
                   p.case_title AS title,
                   common_count,
                   citing_docs
            ORDER BY common_count DESC
            """,
            {"ids": [str(d) for d in doc_ids]},
        )
        common = []
        for row in result.result_set:
            common.append({
                "canonical_citation": row[0],
                "case_title": row[1],
                "common_count": row[2],
                "citing_docs": row[3],
            })
        return common

    def get_graph_stats(self) -> dict[str, int]:
        """Return node and relationship cardinality statistics."""
        j_res = self.graph.query("MATCH (j:Judgment) RETURN count(j)")
        p_res = self.graph.query("MATCH (p:Precedent) RETURN count(p)")
        e_res = self.graph.query("MATCH ()-[r:CITES]->() RETURN count(r)")

        return {
            "total_judgments": j_res.result_set[0][0] if j_res.result_set else 0,
            "total_precedents": p_res.result_set[0][0] if p_res.result_set else 0,
            "total_edges": e_res.result_set[0][0] if e_res.result_set else 0,
        }

    def clear_graph(self) -> None:
        """Delete all nodes and edges from the graph."""
        self.graph.query("MATCH (n) DETACH DELETE n")
        logger.warning("Cleared all nodes and relationships in FalkorDB graph '%s'", self.graph_name)
