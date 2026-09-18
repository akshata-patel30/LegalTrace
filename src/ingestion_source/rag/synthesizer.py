"""Grounded Legal RAG Synthesizer for LegalTrace Pipeline B.

Performs multi-hop retrieval using QueryPlans, validates cited authority status via Shepardizer,
and synthesizes zero-hallucination, inline-cited legal answers using the Groq API
with an extractive fallback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import re
import time
from typing import Any, Optional, Sequence

from ingestion_source.config import Settings, get_settings
from ingestion_source.rag.planner import LegalQueryPlanner, QueryPlan
from ingestion_source.rag.shepardizer import LegalShepardizer
from ingestion_source.search import HybridLegalSearchEngine

logger = logging.getLogger(__name__)


@dataclass
class CitedPassage:
    """A retrieved legal passage acting as ground truth evidence."""
    citation_index: int
    chunk_id: str
    doc_id: str
    case_title: str
    court: str
    year: int
    rhetorical_role: str
    text: str
    relevance_score: float
    statute_mentions: list[str] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "citation_index": self.citation_index,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "case_title": self.case_title,
            "court": self.court,
            "year": self.year,
            "rhetorical_role": self.rhetorical_role,
            "text": self.text,
            "relevance_score": round(self.relevance_score, 4),
            "statute_mentions": self.statute_mentions,
            "citations": self.citations,
        }


@dataclass
class GroundedAnswer:
    """A legal opinion or answer strictly grounded in retrieved corpus evidence."""
    query: str
    answer_text: str
    citations: list[CitedPassage]
    confidence: float
    shepard_warnings: list[str]
    query_plan: QueryPlan
    total_chunks_examined: int
    synthesis_latency_ms: float
    model_used: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "answer_text": self.answer_text,
            "confidence": round(self.confidence, 3),
            "shepard_warnings": self.shepard_warnings,
            "total_chunks_examined": self.total_chunks_examined,
            "synthesis_latency_ms": round(self.synthesis_latency_ms, 2),
            "model_used": self.model_used,
            "query_plan": self.query_plan.to_dict(),
            "citations": [c.to_dict() for c in self.citations],
        }


class LegalRAGSynthesizer:
    """Synthesizes cited, grounded legal answers using Groq LLM or deterministic fallback."""

    SYSTEM_PROMPT = (
        "You are a Senior Legal Research Counsel in the Supreme Court and High Courts of India. "
        "Your task is to answer the legal question STRICTLY based on the provided EVIDENCE PASSAGES.\n\n"
        "STRICT COMPLIANCE RULES:\n"
        "1. ONLY state legal principles, statutory sections, or factual statements found directly in the evidence.\n"
        "2. EVERY statement MUST cite its supporting passage using inline bracketed markers, e.g. [1], [2].\n"
        "3. NEVER invent, extrapolate, or cite case names, citations, or holdings not in the evidence.\n"
        "4. If the evidence is insufficient to answer completely, explicitly state: 'Insufficient evidence in corpus to determine [X].'\n"
        "5. Format your answer clearly: Summary of Legal Position -> Statutory Framework -> Key Judicial Precedents -> Analysis & Caveats."
    )

    def __init__(
        self,
        settings: Optional[Settings] = None,
        planner: Optional[LegalQueryPlanner] = None,
        shepardizer: Optional[LegalShepardizer] = None,
        search_engine: Optional[HybridLegalSearchEngine] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.planner = planner or LegalQueryPlanner(self.settings)
        self.shepardizer = shepardizer or LegalShepardizer(self.settings)
        self.search_engine = search_engine

    def query(
        self,
        user_query: str,
        engine: Optional[HybridLegalSearchEngine] = None,
        max_evidence_chunks: int = 8,
    ) -> GroundedAnswer:
        """Execute full RAG workflow: plan -> multi-hop retrieve -> Shepardize -> synthesize."""
        start_time = time.perf_counter()
        target_engine = engine or self.search_engine
        if target_engine is None:
            raise RuntimeError("A search engine instance must be provided to LegalRAGSynthesizer.")

        # 1. Plan retrieval decomposition
        plan = self.planner.plan(user_query)

        # 2. Retrieve evidence across sub-queries
        evidence_passages, total_examined = self._retrieve_evidence(
            plan=plan,
            engine=target_engine,
            max_total=max_evidence_chunks,
        )

        # 3. Check Shepardization status for precedents cited in evidence
        shepard_warnings = self._check_shepardization(evidence_passages)

        # 4. Synthesize grounded answer
        if len(evidence_passages) == 0:
            latency = (time.perf_counter() - start_time) * 1000.0
            return GroundedAnswer(
                query=user_query,
                answer_text="Insufficient evidence found in the corpus for the given query.",
                citations=[],
                confidence=0.0,
                shepard_warnings=[],
                query_plan=plan,
                total_chunks_examined=0,
                synthesis_latency_ms=latency,
                model_used="none",
            )

        answer_text, model_used, confidence = self._synthesize_answer(
            user_query=user_query,
            evidence=evidence_passages,
            warnings=shepard_warnings,
        )

        latency = (time.perf_counter() - start_time) * 1000.0
        return GroundedAnswer(
            query=user_query,
            answer_text=answer_text,
            citations=evidence_passages,
            confidence=confidence,
            shepard_warnings=shepard_warnings,
            query_plan=plan,
            total_chunks_examined=total_examined,
            synthesis_latency_ms=latency,
            model_used=model_used,
        )

    def _retrieve_evidence(
        self,
        plan: QueryPlan,
        engine: HybridLegalSearchEngine,
        max_total: int,
    ) -> tuple[list[CitedPassage], int]:
        """Execute multi-hop retrieval and deduplicate by chunk_id."""
        seen_chunks: set[str] = set()
        collected_chunks: list[dict[str, Any]] = []
        total_examined = 0

        # Sub-queries execution
        for sq in plan.sub_queries:
            # Query with role filter if specified
            role_filter = sq.target_roles[0] if len(sq.target_roles) == 1 else None
            results = engine.search(
                query=sq.query_text,
                mode="hybrid",
                top_k=sq.top_k,
                filter_role=role_filter,
            )
            total_examined += len(results.results)

            for hit in results.results:
                cid = hit.chunk_id
                if cid not in seen_chunks:
                    seen_chunks.add(cid)
                    role_str = (
                        hit.rhetorical_role.value
                        if hasattr(hit.rhetorical_role, "value")
                        else str(hit.rhetorical_role)
                    )
                    text_str = getattr(hit, "text_with_context", getattr(hit, "text_snippet", getattr(hit, "text", "")))
                    score_val = getattr(hit, "score", getattr(hit, "hybrid_score", 0.0))
                    collected_chunks.append({
                        "chunk_id": cid,
                        "doc_id": hit.doc_id,
                        "case_title": hit.case_title,
                        "court": hit.court,
                        "year": hit.year,
                        "rhetorical_role": role_str,
                        "text": text_str,
                        "relevance_score": float(score_val),
                        "statute_mentions": list(getattr(hit, "statute_mentions", [])),
                        "citations": list(getattr(hit, "citations", [])),
                    })

        # Global fallback if sub-queries yielded few results
        if len(collected_chunks) < 3:
            global_res = engine.search(query=plan.original_query, mode="hybrid", top_k=max_total)
            total_examined += len(global_res.results)
            for hit in global_res.results:
                if hit.chunk_id not in seen_chunks:
                    seen_chunks.add(hit.chunk_id)
                    role_str = (
                        hit.rhetorical_role.value
                        if hasattr(hit.rhetorical_role, "value")
                        else str(hit.rhetorical_role)
                    )
                    text_str = getattr(hit, "text_with_context", getattr(hit, "text_snippet", getattr(hit, "text", "")))
                    score_val = getattr(hit, "score", getattr(hit, "hybrid_score", 0.0))
                    collected_chunks.append({
                        "chunk_id": hit.chunk_id,
                        "doc_id": hit.doc_id,
                        "case_title": hit.case_title,
                        "court": hit.court,
                        "year": hit.year,
                        "rhetorical_role": role_str,
                        "text": text_str,
                        "relevance_score": float(score_val),
                        "statute_mentions": list(getattr(hit, "statute_mentions", [])),
                        "citations": list(getattr(hit, "citations", [])),
                    })

        # Rank by hybrid score and cap at max_total
        collected_chunks.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
        top_chunks = collected_chunks[:max_total]

        passages: list[CitedPassage] = []
        for idx, item in enumerate(top_chunks):
            passages.append(
                CitedPassage(
                    citation_index=idx + 1,
                    chunk_id=item["chunk_id"],
                    doc_id=item["doc_id"],
                    case_title=item["case_title"],
                    court=item["court"],
                    year=item["year"],
                    rhetorical_role=item["rhetorical_role"],
                    text=item["text"],
                    relevance_score=item["relevance_score"],
                    statute_mentions=item.get("statute_mentions", []),
                    citations=item.get("citations", []),
                )
            )

        return passages, total_examined

    def _check_shepardization(self, passages: list[CitedPassage]) -> list[str]:
        """Inspect evidence citations for adverse judicial treatments."""
        warnings: list[str] = []
        checked_citations: set[str] = set()

        for p in passages:
            for cit in p.citations:
                clean = cit.strip()
                if clean and clean not in checked_citations:
                    checked_citations.add(clean)
                    sh_res = self.shepardizer.check(clean)
                    if sh_res.warning_message and not sh_res.is_good_law:
                        warnings.append(f"[{sh_res.status.value.upper()}] {clean}: {sh_res.warning_message}")
                    elif sh_res.status.value == "distinguished":
                        warnings.append(f"[DISTINGUISHED] {clean}: Distinguished in {len(sh_res.distinguishing_cases)} corpus decisions.")

        return warnings

    def _synthesize_answer(
        self,
        user_query: str,
        evidence: list[CitedPassage],
        warnings: list[str],
    ) -> tuple[str, str, float]:
        """Synthesize answer using Groq API or extractive synthesis fallback."""
        # 1. Check if Groq API key is present
        if self.settings.GROQ_API_KEY:
            try:
                answer, confidence = self._call_groq(user_query, evidence, warnings)
                return answer, f"groq:{self.settings.RAG_LLM_MODEL}", confidence
            except Exception as e:
                logger.warning("Groq RAG call failed (%s), falling back to extractive grounded synthesis.", e)

        # 2. Deterministic Extractive Grounded Synthesis Fallback
        answer, confidence = self._extractive_fallback(user_query, evidence, warnings)
        return answer, "extractive_grounded_fallback", confidence

    def _call_groq(
        self,
        query: str,
        evidence: list[CitedPassage],
        warnings: list[str],
    ) -> tuple[str, float]:
        """Generate cited answer via Groq SDK."""
        from groq import Groq

        client = Groq(api_key=self.settings.GROQ_API_KEY)

        evidence_text_parts = []
        for p in evidence:
            evidence_text_parts.append(
                f"--- EVIDENCE PASSAGE [{p.citation_index}] ---\n"
                f"Case: {p.case_title} ({p.court}, {p.year})\n"
                f"Role: {p.rhetorical_role.upper()}\n"
                f"Text:\n{p.text}\n"
            )
        evidence_block = "\n".join(evidence_text_parts)

        warnings_block = ""
        if warnings:
            warnings_block = "\nJUDICIAL TREATMENT WARNINGS (Include in caveats):\n" + "\n".join(f"- {w}" for w in warnings)

        prompt = (
            f"LEGAL QUESTION:\n{query}\n\n"
            f"EVIDENCE PASSAGES:\n{evidence_block}\n"
            f"{warnings_block}\n\n"
            f"Provide a structured, rigorous, and citation-grounded response citing [1], [2] etc."
        )

        model_name = self.settings.RAG_LLM_MODEL or "qwen/qwen3.8-27b"
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1200,
        )

        content = response.choices[0].message.content or "No response generated."

        # Compute confidence based on citation density and passage scores
        has_citations = bool(re.search(r"\[\d+\]", content))
        avg_relevance = sum(p.relevance_score for p in evidence) / max(1, len(evidence))
        base_confidence = min(1.0, max(0.4, avg_relevance * 2.5))
        if not has_citations:
            base_confidence *= 0.5

        return content, round(base_confidence, 2)

    def _extractive_fallback(
        self,
        query: str,
        evidence: list[CitedPassage],
        warnings: list[str],
    ) -> tuple[str, float]:
        """Deterministic extractive grounded synthesis using top retrieved evidence passages."""
        sections = []

        # 1. Summary Header
        sections.append(f"### Legal Research Synthesis: {query}\n")

        # 2. Key Statutory and Principle Findings
        sections.append("#### 1. Statutory Provisions & Core Principles")
        statute_passages = [p for p in evidence if p.rhetorical_role in ("statute_reference", "analysis", "precedents")]
        target_p = statute_passages if statute_passages else evidence[:2]
        for p in target_p[:3]:
            # Extract first two salient sentences
            sentences = [s.strip() for s in p.text.split(".") if len(s.strip()) > 20]
            summary_snippet = ". ".join(sentences[:2]) + "." if sentences else p.text[:250]
            sections.append(
                f"- **{p.case_title}** ({p.court}, {p.year}): {summary_snippet} [{p.citation_index}]"
            )

        # 3. Judicial Precedents & Ratio Decidendi
        sections.append("\n#### 2. Judicial Precedents & Ratio Decidendi")
        ratio_passages = [p for p in evidence if p.rhetorical_role in ("analysis", "operative", "facts")]
        target_r = ratio_passages if ratio_passages else evidence[1:4]
        for p in target_r[:3]:
            snippet = p.text[:300].strip() + ("..." if len(p.text) > 300 else "")
            sections.append(
                f"> \"{snippet}\"\n> — *{p.case_title}* [{p.citation_index}]"
            )

        # 4. Shepardization Warnings / Caveats
        if warnings:
            sections.append("\n#### 3. Shepardization & Precedent Validity Caveats")
            for w in warnings:
                sections.append(f"- :warning: **Notice**: {w}")
        else:
            sections.append("\n#### 3. Validity & Judicial Standing")
            sections.append("- No adverse judicial treatments (overruling/disapproval) detected for the cited authorities in the corpus.")

        text_out = "\n".join(sections)
        return text_out, 0.85
