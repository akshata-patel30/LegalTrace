"""Legal Retrieval Evaluation Benchmark and Testbed for LegalTrace.

Step 10 of Pipeline A:
- Curated multi-faceted legal testbed (Statutory, Precedent, Factual, Procedural queries).
- Ground-truth evaluation against real Delhi High Court corpus judgments.
- Empirical comparison of BM25 (Sparse), Dense Vector Embeddings, and Hybrid RRF.
- Information Retrieval Metrics:
  * Precision@K (K = 1, 3, 5, 10)
  * Recall@K (K = 1, 3, 5, 10)
  * Mean Reciprocal Rank (MRR)
  * Hit Rate@K (Success Rate)
  * Normalized Discounted Cumulative Gain (NDCG@K)
  * Latency (ms) telemetry
- Generates JSON report, Rich comparison tables, and Parquet export.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import time
from typing import Any, Optional

from ingestion_source.models import (
    BenchmarkEvaluationReport,
    BenchmarkQuery,
    ModeAggregateMetrics,
    QueryCategory,
    QueryEvaluationMetric,
    RhetoricalRole,
)
from ingestion_source.search import HybridLegalSearchEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metric Math Primitives
# ---------------------------------------------------------------------------

def compute_precision_at_k(retrieved_doc_ids: list[str], ground_truth_doc_ids: set[str], k: int) -> float:
    """Compute Precision@K: fraction of top-K retrieved docs that are relevant."""
    if k <= 0 or not retrieved_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    relevant_count = sum(1 for doc_id in top_k if doc_id in ground_truth_doc_ids)
    return float(relevant_count) / float(k)


def compute_recall_at_k(retrieved_doc_ids: list[str], ground_truth_doc_ids: set[str], k: int) -> float:
    """Compute Recall@K: fraction of all relevant ground-truth docs found in top-K."""
    if not ground_truth_doc_ids or not retrieved_doc_ids or k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    found_count = sum(1 for doc_id in set(top_k) if doc_id in ground_truth_doc_ids)
    return float(found_count) / float(len(ground_truth_doc_ids))


def compute_hit_at_k(retrieved_doc_ids: list[str], ground_truth_doc_ids: set[str], k: int) -> float:
    """Compute Hit@K: 1.0 if at least one relevant doc is in top-K, else 0.0."""
    if k <= 0 or not retrieved_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    return 1.0 if any(doc_id in ground_truth_doc_ids for doc_id in top_k) else 0.0


def compute_mrr(retrieved_doc_ids: list[str], ground_truth_doc_ids: set[str]) -> tuple[float, Optional[int]]:
    """Compute Reciprocal Rank (1/rank) for the first relevant doc. Returns (rr, first_rank)."""
    if not ground_truth_doc_ids or not retrieved_doc_ids:
        return 0.0, None
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in ground_truth_doc_ids:
            return 1.0 / float(rank), rank
    return 0.0, None


def compute_ndcg_at_k(retrieved_doc_ids: list[str], ground_truth_doc_ids: set[str], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain at rank K (NDCG@K)."""
    if k <= 0 or not retrieved_doc_ids or not ground_truth_doc_ids:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    dcg = 0.0
    for r, doc_id in enumerate(top_k, start=1):
        rel = 1.0 if doc_id in ground_truth_doc_ids else 0.0
        dcg += rel / math.log2(r + 1)

    # Ideal DCG
    idcg = 0.0
    num_ideal = min(k, len(ground_truth_doc_ids))
    for r in range(1, num_ideal + 1):
        idcg += 1.0 / math.log2(r + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg


# ---------------------------------------------------------------------------
# Default Curated Legal Benchmark Suite
# ---------------------------------------------------------------------------

def build_default_benchmark() -> list[BenchmarkQuery]:
    """Returns the standardized 20-query LegalTrace golden benchmark suite.

    Every query is grounded in empirical documents verified present in the Delhi High Court 2023 corpus.
    """
    return [
        # --- Statutory Category ---
        BenchmarkQuery(
            query_id="STAT-01",
            query_text="Section 37 NDPS Act commercial quantity bail parameters satisfaction of court",
            category=QueryCategory.STATUTORY,
            ground_truth_doc_ids=["104774187", "106735942", "19367453", "193050765", "41789071"],
            target_roles=[RhetoricalRole.ANALYSIS, RhetoricalRole.STATUTE_REFERENCE],
            description="Judicial satisfaction under Section 37 NDPS Act for granting bail in commercial quantity",
        ),
        BenchmarkQuery(
            query_id="STAT-02",
            query_text="Section 34 Arbitration Act setting aside arbitral award patent illegality public policy",
            category=QueryCategory.STATUTORY,
            ground_truth_doc_ids=["131353411", "11933721", "58622039", "159871827", "11563168"],
            target_roles=[RhetoricalRole.ANALYSIS],
            description="Grounds for setting aside arbitral awards on patent illegality and public policy",
        ),
        BenchmarkQuery(
            query_id="STAT-03",
            query_text="Section 482 CrPC quashing FIR compromise matrimonial settlement amicable resolution",
            category=QueryCategory.STATUTORY,
            ground_truth_doc_ids=["37623680", "181835314", "165428571"],
            target_roles=[RhetoricalRole.OPERATIVE, RhetoricalRole.ANALYSIS],
            description="Quashing of matrimonial criminal proceedings under Section 482 CrPC on compromise",
        ),
        BenchmarkQuery(
            query_id="STAT-04",
            query_text="Section 138 Negotiable Instruments Act cheque dishonour statutory presumption Section 139",
            category=QueryCategory.STATUTORY,
            ground_truth_doc_ids=["131849650", "75370339", "178346061", "122166354"],
            target_roles=[RhetoricalRole.ANALYSIS, RhetoricalRole.STATUTE_REFERENCE],
            description="Statutory presumption under Section 139 NI Act in cheque bounce complaints",
        ),
        BenchmarkQuery(
            query_id="STAT-05",
            query_text="Section 11 Arbitration Act appointment of arbitrator existence of valid agreement",
            category=QueryCategory.STATUTORY,
            ground_truth_doc_ids=["64125057", "29943308", "23050754"],
            target_roles=[RhetoricalRole.ANALYSIS, RhetoricalRole.OPERATIVE],
            description="Jurisdiction of High Court under Section 11(6) Arbitration Act to appoint arbitrator",
        ),

        # --- Precedent Category ---
        BenchmarkQuery(
            query_id="PREC-01",
            query_text="Maneka Gandhi right to travel abroad personal liberty Article 21 procedure established by law",
            category=QueryCategory.PRECEDENT,
            ground_truth_doc_ids=["180972370", "22199176", "38950634", "93515504", "35682636"],
            target_roles=[RhetoricalRole.PRECEDENTS, RhetoricalRole.ANALYSIS],
            description="Maneka Gandhi v. UOI landmark precedent on personal liberty and right to travel",
        ),
        BenchmarkQuery(
            query_id="PREC-02",
            query_text="Satender Kumar Antil compliance Section 41A CrPC arrest guidelines undertrial bail",
            category=QueryCategory.PRECEDENT,
            ground_truth_doc_ids=["30437693", "35682636", "106735942", "2922069", "93515504"],
            target_roles=[RhetoricalRole.PRECEDENTS, RhetoricalRole.ANALYSIS],
            description="Satender Kumar Antil Supreme Court guidelines regarding arrest and undertrial bail",
        ),
        BenchmarkQuery(
            query_id="PREC-03",
            query_text="Gurbaksh Singh Sibbia anticipatory bail parameters Constitution Bench judicial discretion",
            category=QueryCategory.PRECEDENT,
            ground_truth_doc_ids=["48845313", "180972370", "73887298", "63394054", "2252807"],
            target_roles=[RhetoricalRole.PRECEDENTS, RhetoricalRole.ANALYSIS],
            description="Gurbaksh Singh Sibbia landmark principles governing grant of anticipatory bail",
        ),
        BenchmarkQuery(
            query_id="PREC-04",
            query_text="Vidya Drolia non-arbitrability of disputes four-fold test landlord tenant eviction",
            category=QueryCategory.PRECEDENT,
            ground_truth_doc_ids=["64125057", "29943308", "175316112", "197054318", "119751489"],
            target_roles=[RhetoricalRole.PRECEDENTS, RhetoricalRole.ANALYSIS],
            description="Vidya Drolia four-fold test for determining non-arbitrability of disputes",
        ),
        BenchmarkQuery(
            query_id="PREC-05",
            query_text="Associate Builders scope of public policy Section 34 patent illegality fundamental policy",
            category=QueryCategory.PRECEDENT,
            ground_truth_doc_ids=["131353411", "11933721", "64125057", "11563168"],
            target_roles=[RhetoricalRole.PRECEDENTS, RhetoricalRole.ANALYSIS],
            description="Associate Builders v. DDA public policy and patent illegality standard",
        ),

        # --- Factual Category ---
        BenchmarkQuery(
            query_id="FACT-01",
            query_text="Customs search and interception at IGI airport South African passengers narcotics contraband",
            category=QueryCategory.FACTUAL,
            ground_truth_doc_ids=["104774187"],
            target_roles=[RhetoricalRole.FACTS],
            description="Customs interception of foreign national at IGI airport carrying narcotic substance",
        ),
        BenchmarkQuery(
            query_id="FACT-02",
            query_text="Medical ground interim bail serious medical condition sickness petitioner hospital treatment",
            category=QueryCategory.FACTUAL,
            ground_truth_doc_ids=["35682636"],
            target_roles=[RhetoricalRole.FACTS, RhetoricalRole.ANALYSIS],
            description="Interim bail application filed on medical grounds of acute illness",
        ),
        BenchmarkQuery(
            query_id="FACT-03",
            query_text="Suspension of sentence during pendency of criminal appeal Section 389 CrPC custody period",
            category=QueryCategory.FACTUAL,
            ground_truth_doc_ids=["109930655"],
            target_roles=[RhetoricalRole.OPERATIVE, RhetoricalRole.ANALYSIS],
            description="Application for suspension of sentence during the pendency of criminal appeal",
        ),
        BenchmarkQuery(
            query_id="FACT-04",
            query_text="Trade mark infringement passing off label mark deceptive similarity injunction cosmetic",
            category=QueryCategory.FACTUAL,
            ground_truth_doc_ids=["55361901", "86425479", "142217829", "175316112"],
            target_roles=[RhetoricalRole.FACTS, RhetoricalRole.ARGUMENTS_PETITIONER],
            description="Commercial suit alleging trademark infringement and passing off of brand labels",
        ),
        BenchmarkQuery(
            query_id="FACT-05",
            query_text="Compassionate appointment scheme dependency death in harness family pension livelihood",
            category=QueryCategory.FACTUAL,
            ground_truth_doc_ids=["188047913", "88622769"],
            target_roles=[RhetoricalRole.FACTS, RhetoricalRole.ANALYSIS],
            description="Claim for compassionate appointment on demise of employee during service",
        ),

        # --- Procedural Category ---
        BenchmarkQuery(
            query_id="PROC-01",
            query_text="Maintainability of writ petition under Article 226 when alternative statutory remedy available",
            category=QueryCategory.PROCEDURAL,
            ground_truth_doc_ids=["149474401", "88622769", "167070425"],
            target_roles=[RhetoricalRole.ANALYSIS, RhetoricalRole.OPERATIVE],
            description="High Court jurisdiction under Article 226 in presence of alternate statutory remedy",
        ),
        BenchmarkQuery(
            query_id="PROC-02",
            query_text="Condonation of delay under Section 5 Limitation Act sufficient cause explanation for delay",
            category=QueryCategory.PROCEDURAL,
            ground_truth_doc_ids=["26098995", "40313824", "47253871"],
            target_roles=[RhetoricalRole.ANALYSIS, RhetoricalRole.OPERATIVE],
            description="Sufficient cause standards for condoning delay under Section 5 Limitation Act",
        ),
        BenchmarkQuery(
            query_id="PROC-03",
            query_text="Commercial Courts Act Section 12A pre-institution mediation mandatory requirement non-urgent",
            category=QueryCategory.PROCEDURAL,
            ground_truth_doc_ids=["146393726"],
            target_roles=[RhetoricalRole.ANALYSIS, RhetoricalRole.STATUTE_REFERENCE],
            description="Mandatory compliance with pre-institution mediation under Section 12A Commercial Courts Act",
        ),
        BenchmarkQuery(
            query_id="PROC-04",
            query_text="Order VII Rule 11 CPC rejection of plaint disclose cause of action barred by law",
            category=QueryCategory.PROCEDURAL,
            ground_truth_doc_ids=["124122702", "140008397"],
            target_roles=[RhetoricalRole.ANALYSIS, RhetoricalRole.OPERATIVE],
            description="Parameters for rejection of plaint under Order VII Rule 11 Code of Civil Procedure",
        ),
        BenchmarkQuery(
            query_id="PROC-05",
            query_text="Passport renewal re-issuance criminal proceedings pending no objection certificate trial court",
            category=QueryCategory.PROCEDURAL,
            ground_truth_doc_ids=["180972370", "38950634", "93515504"],
            target_roles=[RhetoricalRole.ANALYSIS, RhetoricalRole.OPERATIVE],
            description="Procedure for renewal or re-issuance of passport while criminal trial is pending",
        ),
    ]


# ---------------------------------------------------------------------------
# Legal Benchmark Evaluator
# ---------------------------------------------------------------------------

class LegalBenchmarkEvaluator:
    """Evaluates search retrieval effectiveness across BM25, Dense, and Hybrid modes."""

    def __init__(self, engine: Optional[HybridLegalSearchEngine] = None, index_dir: Optional[Path] = None):
        if engine is not None:
            self.engine = engine
        elif index_dir is not None and index_dir.exists():
            self.engine = HybridLegalSearchEngine(dim=384)
            self.engine.load_index(index_dir)
        else:
            self.engine = None

    def evaluate_query(
        self,
        query: BenchmarkQuery,
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> QueryEvaluationMetric:
        """Runs search for a single query and computes all standard IR metrics."""
        if self.engine is None:
            raise ValueError("HybridLegalSearchEngine not initialized.")

        gt_docs = set(query.ground_truth_doc_ids)

        t0 = time.perf_counter()
        resp = self.engine.search(
            query=query.query_text,
            mode=mode,
            top_k=top_k,
        )
        latency_ms = (time.perf_counter() - t0) * 1000.0

        retrieved_docs = [r.doc_id for r in resp.results]

        k_thresholds = [1, 3, 5, 10]
        precision_at_k = {k: compute_precision_at_k(retrieved_docs, gt_docs, k) for k in k_thresholds}
        recall_at_k = {k: compute_recall_at_k(retrieved_docs, gt_docs, k) for k in k_thresholds}
        hit_at_k = {k: compute_hit_at_k(retrieved_docs, gt_docs, k) for k in k_thresholds}
        ndcg_at_k = {k: compute_ndcg_at_k(retrieved_docs, gt_docs, k) for k in k_thresholds}
        mrr, first_rank = compute_mrr(retrieved_docs, gt_docs)

        return QueryEvaluationMetric(
            query_id=query.query_id,
            mode=mode,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            hit_at_k=hit_at_k,
            mrr=mrr,
            ndcg_at_k=ndcg_at_k,
            latency_ms=latency_ms,
            first_relevant_rank=first_rank,
        )

    def evaluate_mode(
        self,
        benchmark: list[BenchmarkQuery],
        mode: str = "hybrid",
        top_k: int = 10,
    ) -> tuple[ModeAggregateMetrics, list[QueryEvaluationMetric]]:
        """Evaluates an entire benchmark suite under a specific retrieval mode."""
        per_query_metrics: list[QueryEvaluationMetric] = []
        for q in benchmark:
            metric = self.evaluate_query(q, mode=mode, top_k=top_k)
            per_query_metrics.append(metric)

        n = len(per_query_metrics)
        if n == 0:
            return ModeAggregateMetrics(
                mode=mode,
                mrr=0.0,
                precision_at_1=0.0,
                precision_at_3=0.0,
                precision_at_5=0.0,
                precision_at_10=0.0,
                recall_at_1=0.0,
                recall_at_3=0.0,
                recall_at_5=0.0,
                recall_at_10=0.0,
                hit_at_1=0.0,
                hit_at_3=0.0,
                hit_at_5=0.0,
                hit_at_10=0.0,
                ndcg_at_5=0.0,
                ndcg_at_10=0.0,
                average_latency_ms=0.0,
            ), []

        avg_mrr = sum(m.mrr for m in per_query_metrics) / n
        avg_p1 = sum(m.precision_at_k.get(1, 0.0) for m in per_query_metrics) / n
        avg_p3 = sum(m.precision_at_k.get(3, 0.0) for m in per_query_metrics) / n
        avg_p5 = sum(m.precision_at_k.get(5, 0.0) for m in per_query_metrics) / n
        avg_p10 = sum(m.precision_at_k.get(10, 0.0) for m in per_query_metrics) / n

        avg_r1 = sum(m.recall_at_k.get(1, 0.0) for m in per_query_metrics) / n
        avg_r3 = sum(m.recall_at_k.get(3, 0.0) for m in per_query_metrics) / n
        avg_r5 = sum(m.recall_at_k.get(5, 0.0) for m in per_query_metrics) / n
        avg_r10 = sum(m.recall_at_k.get(10, 0.0) for m in per_query_metrics) / n

        avg_hit1 = sum(m.hit_at_k.get(1, 0.0) for m in per_query_metrics) / n
        avg_hit3 = sum(m.hit_at_k.get(3, 0.0) for m in per_query_metrics) / n
        avg_hit5 = sum(m.hit_at_k.get(5, 0.0) for m in per_query_metrics) / n
        avg_hit10 = sum(m.hit_at_k.get(10, 0.0) for m in per_query_metrics) / n

        avg_ndcg5 = sum(m.ndcg_at_k.get(5, 0.0) for m in per_query_metrics) / n
        avg_ndcg10 = sum(m.ndcg_at_k.get(10, 0.0) for m in per_query_metrics) / n
        avg_lat = sum(m.latency_ms for m in per_query_metrics) / n

        agg = ModeAggregateMetrics(
            mode=mode,
            mrr=avg_mrr,
            precision_at_1=avg_p1,
            precision_at_3=avg_p3,
            precision_at_5=avg_p5,
            precision_at_10=avg_p10,
            recall_at_1=avg_r1,
            recall_at_3=avg_r3,
            recall_at_5=avg_r5,
            recall_at_10=avg_r10,
            hit_at_1=avg_hit1,
            hit_at_3=avg_hit3,
            hit_at_5=avg_hit5,
            hit_at_10=avg_hit10,
            ndcg_at_5=avg_ndcg5,
            ndcg_at_10=avg_ndcg10,
            average_latency_ms=avg_lat,
        )
        return agg, per_query_metrics

    def run_benchmark(
        self,
        benchmark: Optional[list[BenchmarkQuery]] = None,
        top_k: int = 10,
    ) -> BenchmarkEvaluationReport:
        """Executes full comparative benchmark across BM25, Dense, and Hybrid."""
        if benchmark is None:
            benchmark = build_default_benchmark()

        cat_counts = Counter(q.category.value for q in benchmark)

        dense_agg, _ = self.evaluate_mode(benchmark, mode="dense", top_k=top_k)
        bm25_agg, _ = self.evaluate_mode(benchmark, mode="bm25", top_k=top_k)
        hybrid_agg, _ = self.evaluate_mode(benchmark, mode="hybrid", top_k=top_k)

        return BenchmarkEvaluationReport(
            total_queries=len(benchmark),
            category_counts=dict(cat_counts),
            dense_metrics=dense_agg,
            bm25_metrics=bm25_agg,
            hybrid_metrics=hybrid_agg,
        )


from collections import Counter
