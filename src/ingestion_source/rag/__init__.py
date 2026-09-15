"""LegalTrace Pipeline B — Online Legal RAG & Agentic Reasoning Package.

Provides query decomposition, grounded synthesis with Groq LLM,
Shepardization precedent auditing, and court document drafting.
"""

from __future__ import annotations

from ingestion_source.rag.planner import (
    LegalQueryPlanner,
    QueryPlan,
    SubQuery,
    SubQueryType,
)
from ingestion_source.rag.shepardizer import (
    LawStatus,
    LegalShepardizer,
    ShepardResult,
    TreatmentRecord,
)
from ingestion_source.rag.synthesizer import (
    CitedPassage,
    GroundedAnswer,
    LegalRAGSynthesizer,
)
from ingestion_source.rag.memo_generator import (
    LegalDocument,
    LegalMemoGenerator,
)
from ingestion_source.rag.statute_lookup import (
    StatuteRatioLookup,
    StatuteRatioResponse,
    StatuteRatioResult,
    normalize_act_name,
    normalize_section,
)

__all__ = [
    "SubQueryType",
    "SubQuery",
    "QueryPlan",
    "LegalQueryPlanner",
    "LawStatus",
    "TreatmentRecord",
    "ShepardResult",
    "LegalShepardizer",
    "CitedPassage",
    "GroundedAnswer",
    "LegalRAGSynthesizer",
    "LegalDocument",
    "LegalMemoGenerator",
    "StatuteRatioLookup",
    "StatuteRatioResponse",
    "StatuteRatioResult",
    "normalize_act_name",
    "normalize_section",
]
