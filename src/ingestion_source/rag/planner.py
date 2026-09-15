"""Legal Query Planner for LegalTrace Pipeline B.

Decomposes complex natural language legal questions into structured retrieval sub-tasks,
leveraging legal knowledge patterns, statutory cues, and rhetorical role target filters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import re
from typing import Any, Optional

from ingestion_source.config import Settings, get_settings

logger = logging.getLogger(__name__)


class SubQueryType(str, Enum):
    """Categorisation of retrieval sub-tasks."""
    STATUTORY = "statutory"        # Relevant sections, acts, legislative provisions
    PRECEDENT = "precedent"        # Judicial ratio, landmark authorities, binding case law
    FACTUAL = "factual"            # Factually similar narratives and factual matrixes
    PROCEDURAL = "procedural"      # Court jurisdiction, limitation, maintainability
    DOCTRINAL = "doctrinal"        # Foundational legal doctrines and jurisprudence


@dataclass
class SubQuery:
    """A single focused retrieval task derived from the overarching query."""
    query_text: str
    query_type: SubQueryType
    target_roles: list[str] = field(default_factory=list)  # rhetorical roles to filter
    required: bool = True                                   # Critical for answer synthesis
    depends_on: list[int] = field(default_factory=list)     # Dependency sub-query indices
    top_k: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_text": self.query_text,
            "query_type": self.query_type.value,
            "target_roles": self.target_roles,
            "required": self.required,
            "depends_on": self.depends_on,
            "top_k": self.top_k,
        }


@dataclass
class QueryPlan:
    """Structured plan orchestrating multi-hop legal retrieval."""
    original_query: str
    sub_queries: list[SubQuery]
    detected_domain: str = "general"
    synthesis_strategy: str = "grounded_merge"
    max_total_chunks: int = 15

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "detected_domain": self.detected_domain,
            "synthesis_strategy": self.synthesis_strategy,
            "max_total_chunks": self.max_total_chunks,
            "sub_queries": [sq.to_dict() for sq in self.sub_queries],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Pattern-Based Indian Legal Query Decomposition Rules
# ─────────────────────────────────────────────────────────────────────────────

LEGAL_DECOMPOSITION_PATTERNS: list[dict[str, Any]] = [
    {
        "pattern": r"(?i)\b(?:bail|regular\s+bail|anticipatory\s+bail|section\s+437|section\s+438|section\s+439|section\s+37\s+ndps)\b",
        "domain": "criminal",
        "sub_queries": [
            SubQuery(
                query_text="statutory conditions and embargo for grant of bail",
                query_type=SubQueryType.STATUTORY,
                target_roles=["statute_reference", "analysis"],
                required=True,
                top_k=5,
            ),
            SubQuery(
                query_text="precedent principles for bail custody duration parity and charge sheet",
                query_type=SubQueryType.PRECEDENT,
                target_roles=["precedents", "analysis"],
                required=True,
                top_k=5,
            ),
            SubQuery(
                query_text="operative orders granting or rejecting bail with conditions",
                query_type=SubQueryType.FACTUAL,
                target_roles=["operative", "facts"],
                required=False,
                top_k=3,
            ),
        ],
    },
    {
        "pattern": r"(?i)\b(?:quash|quashing|section\s+482|inherent\s+powers|fir\s+quash)\b",
        "domain": "criminal",
        "sub_queries": [
            SubQuery(
                query_text="Section 482 CrPC inherent powers parameters and guidelines for quashing FIR",
                query_type=SubQueryType.STATUTORY,
                target_roles=["statute_reference", "analysis"],
                required=True,
                top_k=5,
            ),
            SubQuery(
                query_text="compromise settlement matrimonial dispute commercial dispute quashing criminal proceedings",
                query_type=SubQueryType.PRECEDENT,
                target_roles=["precedents", "analysis"],
                required=True,
                top_k=5,
            ),
            SubQuery(
                query_text="operative order FIR quashed or dismissed",
                query_type=SubQueryType.FACTUAL,
                target_roles=["operative"],
                required=False,
                top_k=3,
            ),
        ],
    },
    {
        "pattern": r"(?i)\b(?:arbitration|section\s+34|section\s+11|patent\s+illegality|public\s+policy|arbitral\s+award)\b",
        "domain": "arbitration",
        "sub_queries": [
            SubQuery(
                query_text="Section 34 Arbitration and Conciliation Act grounds to set aside award patent illegality public policy",
                query_type=SubQueryType.STATUTORY,
                target_roles=["statute_reference", "analysis"],
                required=True,
                top_k=5,
            ),
            SubQuery(
                query_text="scope of judicial interference with arbitral award plausible view of arbitrator",
                query_type=SubQueryType.PRECEDENT,
                target_roles=["precedents", "analysis"],
                required=True,
                top_k=5,
            ),
            SubQuery(
                query_text="operative holding award set aside or upheld",
                query_type=SubQueryType.FACTUAL,
                target_roles=["operative"],
                required=False,
                top_k=3,
            ),
        ],
    },
    {
        "pattern": r"(?i)\b(?:section\s+138|negotiable\s+instruments|cheque\s+bounce|statutory\s+notice)\b",
        "domain": "commercial",
        "sub_queries": [
            SubQuery(
                query_text="Section 138 Negotiable Instruments Act statutory notice demand legally enforceable debt",
                query_type=SubQueryType.STATUTORY,
                target_roles=["statute_reference", "analysis"],
                required=True,
                top_k=5,
            ),
            SubQuery(
                query_text="presumption under Section 139 NI Act rebuttal by accused standard of proof",
                query_type=SubQueryType.PRECEDENT,
                target_roles=["precedents", "analysis"],
                required=True,
                top_k=5,
            ),
        ],
    },
    {
        "pattern": r"(?i)\b(?:article\s+21|article\s+14|article\s+226|fundamental\s+right|personal\s+liberty|writ\s+petition)\b",
        "domain": "constitutional",
        "sub_queries": [
            SubQuery(
                query_text="constitutional provisions fundamental rights Article 14 19 21 and writ jurisdiction Article 226",
                query_type=SubQueryType.STATUTORY,
                target_roles=["statute_reference", "analysis"],
                required=True,
                top_k=5,
            ),
            SubQuery(
                query_text="precedent doctrine proportionality natural justice arbitrary administrative action",
                query_type=SubQueryType.PRECEDENT,
                target_roles=["precedents", "analysis"],
                required=True,
                top_k=5,
            ),
        ],
    },
    {
        "pattern": r"(?i)\b(?:injunction|order\s+39|temporary\s+injunction|prima\s+facie\s+case|balance\s+of\s+convenience|irreparable\s+injury)\b",
        "domain": "civil",
        "sub_queries": [
            SubQuery(
                query_text="Order 39 Rules 1 and 2 CPC temporary injunction statutory criteria",
                query_type=SubQueryType.STATUTORY,
                target_roles=["statute_reference", "analysis"],
                required=True,
                top_k=5,
            ),
            SubQuery(
                query_text="tripartite test prima facie case balance of convenience irreparable injury injunction",
                query_type=SubQueryType.PRECEDENT,
                target_roles=["precedents", "analysis"],
                required=True,
                top_k=5,
            ),
        ],
    },
]


class LegalQueryPlanner:
    """Decomposes complex legal queries into targeted retrieval sub-tasks."""

    def __init__(self, settings: Optional[Settings] = None) -> None:
        self.settings = settings or get_settings()

    def plan(self, query: str) -> QueryPlan:
        """Create a retrieval execution plan for a legal question."""
        clean_q = query.strip()
        if not clean_q:
            return QueryPlan(original_query="", sub_queries=[])

        # Step 1: Check pattern-based decomposition rules
        for rule in LEGAL_DECOMPOSITION_PATTERNS:
            if re.search(rule["pattern"], clean_q):
                logger.info("Matched legal pattern decomposition for domain '%s'", rule["domain"])
                sub_queries = []
                for sq in rule["sub_queries"]:
                    # Blend original query terms with sub-query focus
                    blended_text = f"{clean_q} {sq.query_text}".strip()
                    sub_queries.append(
                        SubQuery(
                            query_text=blended_text,
                            query_type=sq.query_type,
                            target_roles=list(sq.target_roles),
                            required=sq.required,
                            depends_on=list(sq.depends_on),
                            top_k=sq.top_k,
                        )
                    )
                return QueryPlan(
                    original_query=clean_q,
                    sub_queries=sub_queries,
                    detected_domain=rule["domain"],
                    synthesis_strategy="grounded_merge",
                )

        # Step 2: Try LLM-assisted decomposition if Groq API key is available
        if self.settings.GROQ_API_KEY:
            try:
                llm_plan = self._plan_with_groq(clean_q)
                if llm_plan and len(llm_plan.sub_queries) > 0:
                    return llm_plan
            except Exception as e:
                logger.warning("Groq-assisted query planning failed, falling back to heuristics: %s", e)

        # Step 3: Heuristic legal entity decomposition fallback
        return self._heuristic_decomposition(clean_q)

    def _plan_with_groq(self, query: str) -> Optional[QueryPlan]:
        """Decompose query using Groq API and structured JSON extraction."""
        from groq import Groq

        client = Groq(api_key=self.settings.GROQ_API_KEY)
        system_prompt = (
            "You are a Senior Legal Research Specialist in Indian law. "
            "Decompose the given legal query into 2 to 4 focused retrieval sub-queries. "
            "Output strictly valid JSON with this schema:\n"
            "{\n"
            '  "detected_domain": "criminal|civil|commercial|arbitration|constitutional|general",\n'
            '  "sub_queries": [\n'
            '    {\n'
            '      "query_text": "search terms",\n'
            '      "query_type": "statutory|precedent|factual|procedural|doctrinal",\n'
            '      "target_roles": ["analysis", "statute_reference", "precedents", "facts", "operative"],\n'
            '      "required": true,\n'
            '      "top_k": 5\n'
            "    }\n"
            "  ]\n"
            "}"
        )

        response = client.chat.completions.create(
            model=self.settings.RAG_LLM_MODEL or "llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Decompose this legal question:\n{query}"},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=600,
        )

        raw_json = response.choices[0].message.content
        if not raw_json:
            return None

        data = json.loads(raw_json)
        subs = []
        for item in data.get("sub_queries", []):
            qtype_str = item.get("query_type", "precedent").lower()
            try:
                qtype = SubQueryType(qtype_str)
            except ValueError:
                qtype = SubQueryType.PRECEDENT

            subs.append(
                SubQuery(
                    query_text=str(item.get("query_text", query)),
                    query_type=qtype,
                    target_roles=list(item.get("target_roles", ["analysis"])),
                    required=bool(item.get("required", True)),
                    top_k=int(item.get("top_k", 5)),
                )
            )

        return QueryPlan(
            original_query=query,
            sub_queries=subs,
            detected_domain=str(data.get("detected_domain", "general")),
            synthesis_strategy="grounded_merge",
        )

    def _heuristic_decomposition(self, query: str) -> QueryPlan:
        """Construct robust multi-perspective legal sub-queries via entity extraction heuristics."""
        sub_queries: list[SubQuery] = []

        # 1. Statutory perspective
        has_statute = bool(re.search(r"(?i)\b(?:section|sec|article|art|order|rule|act|code)\b", query))
        sub_queries.append(
            SubQuery(
                query_text=f"{query} statutory provisions legislative mandate",
                query_type=SubQueryType.STATUTORY,
                target_roles=["statute_reference", "analysis"],
                required=has_statute,
                top_k=5,
            )
        )

        # 2. Judicial precedent & ratio decidendi perspective
        sub_queries.append(
            SubQuery(
                query_text=f"{query} binding precedent ratio decidendi High Court Supreme Court",
                query_type=SubQueryType.PRECEDENT,
                target_roles=["analysis", "precedents"],
                required=True,
                top_k=5,
            )
        )

        # 3. Factual & operative outcome perspective
        sub_queries.append(
            SubQuery(
                query_text=f"{query} factual matrix operative relief holding",
                query_type=SubQueryType.FACTUAL,
                target_roles=["facts", "operative"],
                required=False,
                top_k=3,
            )
        )

        return QueryPlan(
            original_query=query,
            sub_queries=sub_queries,
            detected_domain="general",
            synthesis_strategy="grounded_merge",
        )
