"""Precedent Citation Graph & Cross-Document Linker for LegalTrace.

Step 7 of Pipeline A:
- Constructs directed citation networks (Citing Judgment -> Cited Precedent).
- Normalizes law report citations (SCC, AIR, DLT, SCR, Neutral Citations).
- Performs cross-document resolution against the ingested corpus (internal matches).
- Classifies citation treatment (relied_upon, followed, distinguished, overruled, referred, cited).
- Associates citations with paragraph rhetorical roles (ANALYSIS, ARGUMENTS, PRECEDENTS).
- Extracts context sentences and landmark case titles.
- Calculates network metrics (in-degree, authority ranking, citation centrality).
- Exports to JSON, JSONL, GraphML, and Parquet.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import logging
from pathlib import Path
import re
from typing import Any, Optional
import xml.etree.ElementTree as ET

from ingestion_source.models import (
    CitationEdge,
    CitationGraphSummary,
    CitationTreatment,
    CleanedJudgment,
    EnrichedLegalMetadata,
    ExtractedCitation,
    PrecedentNode,
    RhetoricalRole,
)

logger = logging.getLogger(__name__)

# Treatment classification rules based on judicial language cues
TREATMENT_RULES: list[tuple[str, CitationTreatment]] = [
    (r"(?i)\b(?:relied\s+upon|relying\s+on|reliance\s+(?:is|was|has\s+been)?\s*placed\s+on|guided\s+by|placed\s+reliance|support\s+from)\b", CitationTreatment.RELIED_UPON),
    (r"(?i)\b(?:following\s+the\s+(?:dictum|decision|ratio|principles?)|followed\s+by|bound\s+by\s+the\s+decision|in\s+terms\s+of\s+the\s+judgment|reiterated\s+by)\b", CitationTreatment.FOLLOWED),
    (r"(?i)\b(?:distinguished|distinguishable|does\s+not\s+apply|not\s+applicable|inapplicable|differs\s+from|distinction\s+was\s+drawn)\b", CitationTreatment.DISTINGUISHED),
    (r"(?i)\b(?:overruled|no\s+longer\s+good\s+law|per\s+incuriam|set\s+aside\s+by|expressly\s+negated)\b", CitationTreatment.OVERRULED),
    (r"(?i)\b(?:referred\s+to|reference\s+(?:is|was|has\s+been)?\s*made\s+to|adverted\s+to|noticed\s+in|cited\s+with\s+approval)\b", CitationTreatment.REFERRED),
]

CASE_TITLE_PATTERN = re.compile(
    r"\b([A-Z][a-zA-Z0-9\'\.\-]+(?:\s+[A-Z][a-zA-Z0-9\'\.\-]+)*\s+(?:v\.|vs\.?|versus)\s+[A-Z][a-zA-Z0-9\'\.\-]+(?:\s+[A-Z][a-zA-Z0-9\'\.\-]+)*)\b",
    re.IGNORECASE,
)


def normalize_citation_key(citation_str: str) -> str:
    """Normalize raw citation string into a consistent canonical precedent ID."""
    clean = re.sub(r"[^\w\s]", " ", citation_str)
    parts = clean.split()
    return "_".join(p.lower() for p in parts if p)


class CitationGraphBuilder:
    """Extracts citation edges, resolves internal cross-links, and builds precedent networks."""

    def __init__(self, store_dir: Optional[Path] = None) -> None:
        self.store_dir = store_dir
        self.corpus_index: dict[str, str] = {}  # key -> doc_id
        if store_dir and store_dir.exists():
            self._index_corpus(store_dir)

    def _index_corpus(self, store_dir: Path) -> None:
        """Index local corpus documents by doc_id, neutral citation, and docket numbers."""
        cleaned_files = list(store_dir.glob("*/*/*/cleaned.json"))
        logger.info("Indexing %d local judgments for cross-document resolution...", len(cleaned_files))

        for f in cleaned_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                doc_id = str(data.get("doc_id", ""))
                self.corpus_index[f"doc:{doc_id}"] = doc_id

                header = data.get("header") or {}
                neutral_cit = header.get("neutral_citation")
                if neutral_cit:
                    norm_nc = re.sub(r"[^\w]", "", neutral_cit.lower())
                    self.corpus_index[f"nc:{norm_nc}"] = doc_id

                for docket in header.get("docket_numbers", []):
                    norm_doc = re.sub(r"[^\w]", "", docket.lower())
                    self.corpus_index[f"docket:{norm_doc}"] = doc_id

            except Exception as e:
                logger.debug("Failed indexing %s: %s", f, e)

    def extract_document_edges(
        self,
        judgment: CleanedJudgment,
        enriched: Optional[EnrichedLegalMetadata] = None,
    ) -> list[CitationEdge]:
        """Extract all outgoing citation edges from a single judgment."""
        doc_id = judgment.doc_id
        court = judgment.court
        year = judgment.year
        title = judgment.title

        judge = None
        if enriched and enriched.canonical_judges:
            judge = ", ".join(enriched.canonical_judges)
        elif judgment.header.bench:
            judge = ", ".join(judgment.header.bench)

        # Collect citations from enriched.json or detect from paragraphs
        raw_citations: list[tuple[str, str, Optional[int]]] = []
        if enriched and enriched.citations:
            for cit in enriched.citations:
                raw_citations.append((cit.citation_text, cit.reporter, cit.year))

        # If no enriched citations, scan paragraph text
        if not raw_citations:
            for p in judgment.paragraphs:
                # Basic scanner fallback
                matches = re.findall(r"\b(\d{4}\s+(?:\d+\s+)?(?:SCC|SCR|AIR|DLT)\s+\d+)\b", p.text)
                for m in matches:
                    raw_citations.append((m, "Reporter", None))

        edges: list[CitationEdge] = []
        edge_count = 0

        for cit_text, reporter, cit_year in raw_citations:
            norm_cit = cit_text.strip()
            # Find the paragraph where this citation appears
            matching_para = None
            for p in judgment.paragraphs:
                if norm_cit.lower() in p.text.lower():
                    matching_para = p
                    break

            para_id = matching_para.para_id if matching_para else None
            role = matching_para.rhetorical_role if matching_para else RhetoricalRole.PRECEDENTS
            para_text = matching_para.text if matching_para else norm_cit

            # Extract context sentence
            context_sentence = self._extract_context_sentence(para_text, norm_cit)

            # Classify treatment
            treatment = self._classify_treatment(context_sentence)

            # Detect case title of precedent
            precedent_title = self._detect_precedent_title(context_sentence, para_text, norm_cit)

            # Check internal corpus match
            is_internal, target_id = self._resolve_internal_match(norm_cit)

            edge_count += 1
            norm_key = normalize_citation_key(norm_cit)
            edge_id = f"edge_{doc_id}_{norm_key}_{edge_count:02d}"

            edges.append(
                CitationEdge(
                    edge_id=edge_id,
                    citing_doc_id=doc_id,
                    citing_court=court,
                    citing_year=year,
                    citing_title=title,
                    citing_judge=judge,
                    citing_para_id=para_id,
                    rhetorical_role=role,
                    treatment=treatment,
                    cited_citation=norm_cit,
                    cited_reporter=reporter,
                    cited_year=cit_year,
                    cited_title=precedent_title,
                    context_sentence=context_sentence,
                    is_internal_corpus_match=is_internal,
                    target_doc_id=target_id,
                )
            )

        return edges

    def _extract_context_sentence(self, para_text: str, citation_text: str) -> str:
        """Locate the exact sentence containing the citation, protecting legal abbreviations."""
        subbed = re.sub(r"\b(v|vs|no|ors|anr|sec|art|cl|dr|mr|mrs|hon'ble|sh)\.", r"\1_DOT_", para_text, flags=re.IGNORECASE)
        sentences = re.split(r"(?<=[.!?])\s+", subbed)
        cit_lower = citation_text.lower()
        for s in sentences:
            if cit_lower in s.lower():
                return s.replace("_DOT_", ".").strip()
        return para_text[:300].strip()

    def _classify_treatment(self, context_sentence: str) -> CitationTreatment:
        """Classify judicial treatment based on textual cues."""
        for pattern, treatment in TREATMENT_RULES:
            if re.search(pattern, context_sentence):
                return treatment
        return CitationTreatment.CITED

    def _detect_precedent_title(
        self,
        context_sentence: str,
        para_text: Optional[str] = None,
        citation_text: Optional[str] = None,
    ) -> Optional[str]:
        """Extract named parties/title of cited case if present in context or surrounding paragraph."""
        m = CASE_TITLE_PATTERN.search(context_sentence)
        if m:
            title = m.group(1).strip()
            if len(title) > 8 and not title.lower().startswith("vs"):
                return title

        if para_text and citation_text:
            pos = para_text.lower().find(citation_text.lower())
            if pos != -1:
                window_start = max(0, pos - 250)
                window_end = min(len(para_text), pos + len(citation_text) + 250)
                window = para_text[window_start:window_end]
                m_win = CASE_TITLE_PATTERN.search(window)
                if m_win:
                    title = m_win.group(1).strip()
                    if len(title) > 8 and not title.lower().startswith("vs"):
                        return title
        return None

    def _resolve_internal_match(self, citation_text: str) -> tuple[bool, Optional[str]]:
        """Check if citation maps to another judgment in the local repository."""
        # 1. Check direct doc_id match
        doc_m = re.search(r"\bdoc(?:_id)?[:\s]+(\d+)\b", citation_text, re.IGNORECASE)
        if doc_m:
            target = doc_m.group(1)
            if f"doc:{target}" in self.corpus_index:
                return True, target

        # 2. Check neutral citation match
        norm_nc = re.sub(r"[^\w]", "", citation_text.lower())
        if f"nc:{norm_nc}" in self.corpus_index:
            return True, self.corpus_index[f"nc:{norm_nc}"]

        # 3. Check docket match
        if f"docket:{norm_nc}" in self.corpus_index:
            return True, self.corpus_index[f"docket:{norm_nc}"]

        return False, None

    def build_graph_from_store(
        self,
        store_dir: Path,
        output_edges_jsonl: Optional[Path] = None,
        output_nodes_jsonl: Optional[Path] = None,
        output_graph_json: Optional[Path] = None,
        output_graphml: Optional[Path] = None,
    ) -> tuple[list[CitationEdge], list[PrecedentNode], CitationGraphSummary]:
        """Process all documents in store_dir, construct edges, aggregate nodes, and export graph."""
        cleaned_files = list(store_dir.glob("*/*/*/cleaned.json"))
        logger.info("Found %d judgments to build citation network in %s", len(cleaned_files), store_dir)

        all_edges: list[CitationEdge] = []
        citing_doc_ids: set[str] = set()

        for f in cleaned_files:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                judgment = CleanedJudgment(**data)

                enriched: Optional[EnrichedLegalMetadata] = None
                enriched_file = f.parent / "enriched.json"
                if enriched_file.exists():
                    enriched_data = json.loads(enriched_file.read_text(encoding="utf-8"))
                    enriched = EnrichedLegalMetadata(**enriched_data)

                edges = self.extract_document_edges(judgment, enriched=enriched)
                if edges:
                    all_edges.extend(edges)
                    citing_doc_ids.add(judgment.doc_id)

            except Exception as e:
                logger.error("Error extracting citations for %s: %s", f.parent, e)

        # Aggregate edges into PrecedentNodes
        precedent_nodes = self._aggregate_nodes(all_edges)

        # Generate summary
        summary = self._generate_summary(all_edges, precedent_nodes, len(citing_doc_ids))

        # Write outputs
        if output_edges_jsonl and all_edges:
            output_edges_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(output_edges_jsonl, "w", encoding="utf-8") as out:
                for edge in all_edges:
                    out.write(json.dumps(edge.model_dump(mode="json"), ensure_ascii=False) + "\n")
            logger.info("Saved %d citation edges to %s", len(all_edges), output_edges_jsonl)

        if output_nodes_jsonl and precedent_nodes:
            output_nodes_jsonl.parent.mkdir(parents=True, exist_ok=True)
            with open(output_nodes_jsonl, "w", encoding="utf-8") as out:
                for node in precedent_nodes:
                    out.write(json.dumps(node.model_dump(mode="json"), ensure_ascii=False) + "\n")
            logger.info("Saved %d precedent nodes to %s", len(precedent_nodes), output_nodes_jsonl)

        if output_graph_json:
            self.export_d3_graph(all_edges, precedent_nodes, output_graph_json)

        if output_graphml:
            self.export_graphml(all_edges, precedent_nodes, output_graphml)

        return all_edges, precedent_nodes, summary

    def _aggregate_nodes(self, edges: list[CitationEdge]) -> list[PrecedentNode]:
        """Group citation edges by canonical precedent ID and compute authority metrics."""
        grouped: dict[str, list[CitationEdge]] = defaultdict(list)
        for e in edges:
            norm_key = normalize_citation_key(e.cited_citation)
            grouped[norm_key].append(e)

        nodes: list[PrecedentNode] = []
        for norm_key, edge_list in grouped.items():
            first = edge_list[0]
            citing_docs = sorted(list(set(e.citing_doc_id for e in edge_list)))
            in_degree = len(citing_docs)

            treatments = Counter(e.treatment.value for e in edge_list)
            roles = Counter(e.rhetorical_role.value for e in edge_list)

            # Pick best known title if available
            best_title = None
            for e in edge_list:
                if e.cited_title:
                    best_title = e.cited_title
                    break

            is_internal = any(e.is_internal_corpus_match for e in edge_list)
            target_doc_id = next((e.target_doc_id for e in edge_list if e.target_doc_id), None)

            nodes.append(
                PrecedentNode(
                    precedent_id=f"prec_{norm_key}",
                    canonical_citation=first.cited_citation,
                    reporter=first.cited_reporter,
                    citation_year=first.cited_year,
                    case_title=best_title,
                    in_degree=in_degree,
                    citing_doc_ids=citing_docs,
                    treatments=dict(treatments),
                    roles=dict(roles),
                    is_internal=is_internal,
                    doc_id=target_doc_id,
                )
            )

        # Sort descending by authority / in-degree
        nodes.sort(key=lambda n: n.in_degree, reverse=True)
        return nodes

    def _generate_summary(
        self,
        edges: list[CitationEdge],
        nodes: list[PrecedentNode],
        citing_doc_count: int,
    ) -> CitationGraphSummary:
        """Compute network metrics and distribution breakdown."""
        internal_cnt = sum(1 for e in edges if e.is_internal_corpus_match)
        external_cnt = len(edges) - internal_cnt

        treatments = Counter(e.treatment.value for e in edges)
        roles = Counter(e.rhetorical_role.value for e in edges)

        top_list: list[dict[str, Any]] = []
        for n in nodes[:15]:
            top_list.append({
                "citation": n.canonical_citation,
                "reporter": n.reporter,
                "year": n.citation_year,
                "title": n.case_title or "Unspecified",
                "in_degree": n.in_degree,
                "is_internal": n.is_internal,
            })

        return CitationGraphSummary(
            total_citing_documents=citing_doc_count,
            total_edges=len(edges),
            total_precedent_nodes=len(nodes),
            internal_edges_count=internal_cnt,
            external_edges_count=external_cnt,
            top_precedents=top_list,
            treatment_breakdown=dict(treatments.most_common()),
            role_breakdown=dict(roles.most_common()),
        )

    def export_d3_graph(
        self,
        edges: list[CitationEdge],
        nodes: list[PrecedentNode],
        output_file: Path,
    ) -> None:
        """Export graph to D3.js / Cytoscape node-link JSON format."""
        d3_nodes: list[dict[str, Any]] = []
        # Add citing documents as nodes
        seen_citing: set[str] = set()
        for e in edges:
            if e.citing_doc_id not in seen_citing:
                seen_citing.add(e.citing_doc_id)
                d3_nodes.append({
                    "id": f"doc_{e.citing_doc_id}",
                    "label": e.citing_title[:50],
                    "node_type": "judgment",
                    "court": e.citing_court,
                    "year": e.citing_year,
                })

        # Add precedent nodes
        for n in nodes:
            d3_nodes.append({
                "id": n.precedent_id,
                "label": n.case_title or n.canonical_citation,
                "citation": n.canonical_citation,
                "reporter": n.reporter,
                "node_type": "precedent",
                "in_degree": n.in_degree,
                "is_internal": n.is_internal,
            })

        # Add links
        d3_links: list[dict[str, Any]] = []
        for e in edges:
            norm_key = normalize_citation_key(e.cited_citation)
            d3_links.append({
                "source": f"doc_{e.citing_doc_id}",
                "target": f"prec_{norm_key}",
                "treatment": e.treatment.value,
                "role": e.rhetorical_role.value,
            })

        output_file.parent.mkdir(parents=True, exist_ok=True)
        graph_data = {"nodes": d3_nodes, "links": d3_links}
        output_file.write_text(json.dumps(graph_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Exported D3 graph with %d nodes and %d links to %s", len(d3_nodes), len(d3_links), output_file)

    def export_graphml(
        self,
        edges: list[CitationEdge],
        nodes: list[PrecedentNode],
        output_file: Path,
    ) -> None:
        """Export directed citation graph to standard GraphML XML format."""
        root = ET.Element("graphml", xmlns="http://graphml.graphdrawing.org/xmlns")
        
        # Define keys
        k_type = ET.SubElement(root, "key", id="d0", attr_name="node_type", attr_type="string")
        k_type.set("for", "node")
        k_label = ET.SubElement(root, "key", id="d1", attr_name="label", attr_type="string")
        k_label.set("for", "node")
        k_indegree = ET.SubElement(root, "key", id="d2", attr_name="in_degree", attr_type="int")
        k_indegree.set("for", "node")
        k_treatment = ET.SubElement(root, "key", id="d3", attr_name="treatment", attr_type="string")
        k_treatment.set("for", "edge")
        k_role = ET.SubElement(root, "key", id="d4", attr_name="role", attr_type="string")
        k_role.set("for", "edge")

        graph = ET.SubElement(root, "graph", id="G", edgedefault="directed")

        # Citing judgment nodes
        seen_citing: set[str] = set()
        for e in edges:
            if e.citing_doc_id not in seen_citing:
                seen_citing.add(e.citing_doc_id)
                n_el = ET.SubElement(graph, "node", id=f"doc_{e.citing_doc_id}")
                d_type = ET.SubElement(n_el, "data", key="d0")
                d_type.text = "judgment"
                d_label = ET.SubElement(n_el, "data", key="d1")
                d_label.text = e.citing_title[:60]

        # Precedent nodes
        for n in nodes:
            n_el = ET.SubElement(graph, "node", id=n.precedent_id)
            d_type = ET.SubElement(n_el, "data", key="d0")
            d_type.text = "precedent"
            d_label = ET.SubElement(n_el, "data", key="d1")
            d_label.text = n.case_title or n.canonical_citation
            d_indeg = ET.SubElement(n_el, "data", key="d2")
            d_indeg.text = str(n.in_degree)

        # Edges
        for idx, e in enumerate(edges):
            norm_key = normalize_citation_key(e.cited_citation)
            edge_el = ET.SubElement(
                graph,
                "edge",
                id=f"e{idx}",
                source=f"doc_{e.citing_doc_id}",
                target=f"prec_{norm_key}",
            )
            d_treat = ET.SubElement(edge_el, "data", key="d3")
            d_treat.text = e.treatment.value
            d_role = ET.SubElement(edge_el, "data", key="d4")
            d_role.text = e.rhetorical_role.value

        tree = ET.ElementTree(root)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        tree.write(str(output_file), encoding="utf-8", xml_declaration=True)
        logger.info("Exported GraphML file to %s", output_file)
