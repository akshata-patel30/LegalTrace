"""Unit tests for Step 10: Legal Retrieval Evaluation Benchmark & Testbed."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from typer.testing import CliRunner

from ingestion_source.cli import app
from ingestion_source.evaluation import (
    LegalBenchmarkEvaluator,
    build_default_benchmark,
    compute_hit_at_k,
    compute_mrr,
    compute_ndcg_at_k,
    compute_precision_at_k,
    compute_recall_at_k,
)
from ingestion_source.models import BenchmarkQuery, QueryCategory, RhetoricalRole
from ingestion_source.search import HybridLegalSearchEngine

runner = CliRunner()


def test_metric_precision_at_k():
    gt = {"doc1", "doc2", "doc3"}
    retrieved = ["doc1", "doc99", "doc2", "doc88", "doc77"]

    # Top 1: doc1 is relevant -> 1/1 = 1.0
    assert compute_precision_at_k(retrieved, gt, k=1) == 1.0
    # Top 2: doc1, doc99 -> 1/2 = 0.5
    assert compute_precision_at_k(retrieved, gt, k=2) == 0.5
    # Top 3: doc1, doc99, doc2 -> 2/3 = 0.6666...
    assert pytest.approx(compute_precision_at_k(retrieved, gt, k=3), 0.001) == 0.667
    # Edge cases
    assert compute_precision_at_k([], gt, k=5) == 0.0
    assert compute_precision_at_k(retrieved, set(), k=5) == 0.0
    assert compute_precision_at_k(retrieved, gt, k=0) == 0.0


def test_metric_recall_at_k():
    gt = {"doc1", "doc2", "doc3", "doc4"}
    retrieved = ["doc1", "doc99", "doc2", "doc88", "doc77"]

    # Top 1: 1 out of 4 -> 0.25
    assert compute_recall_at_k(retrieved, gt, k=1) == 0.25
    # Top 3: 2 out of 4 -> 0.50
    assert compute_recall_at_k(retrieved, gt, k=3) == 0.50
    # Top 5: 2 out of 4 -> 0.50
    assert compute_recall_at_k(retrieved, gt, k=5) == 0.50
    # Edge cases
    assert compute_recall_at_k([], gt, k=5) == 0.0
    assert compute_recall_at_k(retrieved, set(), k=5) == 0.0


def test_metric_hit_at_k():
    gt = {"doc10"}
    retrieved = ["doc1", "doc2", "doc10", "doc4"]

    assert compute_hit_at_k(retrieved, gt, k=1) == 0.0
    assert compute_hit_at_k(retrieved, gt, k=2) == 0.0
    assert compute_hit_at_k(retrieved, gt, k=3) == 1.0
    assert compute_hit_at_k(retrieved, gt, k=4) == 1.0


def test_metric_mrr():
    gt = {"target_doc"}

    # First rank
    rr1, rank1 = compute_mrr(["target_doc", "other"], gt)
    assert rr1 == 1.0
    assert rank1 == 1

    # Second rank
    rr2, rank2 = compute_mrr(["other", "target_doc"], gt)
    assert rr2 == 0.5
    assert rank2 == 2

    # Fourth rank
    rr4, rank4 = compute_mrr(["a", "b", "c", "target_doc"], gt)
    assert rr4 == 0.25
    assert rank4 == 4

    # Not found
    rr0, rank0 = compute_mrr(["a", "b", "c"], gt)
    assert rr0 == 0.0
    assert rank0 is None


def test_metric_ndcg_at_k():
    gt = {"doc1", "doc2"}

    # Perfect ranking: [doc1, doc2, doc3]
    perfect = compute_ndcg_at_k(["doc1", "doc2", "doc3"], gt, k=3)
    assert pytest.approx(perfect, 0.001) == 1.0

    # Suboptimal ranking: [doc9, doc1, doc2]
    subopt = compute_ndcg_at_k(["doc9", "doc1", "doc2"], gt, k=3)
    assert 0.0 < subopt < 1.0

    # Zero ranking: [doc9, doc8, doc7]
    zero = compute_ndcg_at_k(["doc9", "doc8", "doc7"], gt, k=3)
    assert zero == 0.0


def test_default_benchmark_suite_integrity():
    benchmark = build_default_benchmark()
    assert len(benchmark) == 20

    from collections import Counter
    cat_counts = Counter(q.category.value for q in benchmark)
    assert cat_counts["statutory"] == 5
    assert cat_counts["precedent"] == 5
    assert cat_counts["factual"] == 5
    assert cat_counts["procedural"] == 5

    for q in benchmark:
        assert len(q.ground_truth_doc_ids) >= 1
        assert len(q.query_text) > 15
        assert q.query_id.startswith(("STAT", "PREC", "FACT", "PROC"))


def test_benchmark_evaluator_with_index(tmp_path: Path):
    index_dir = Path("search_index")
    if not index_dir.exists():
        pytest.skip("search_index not present")

    evaluator = LegalBenchmarkEvaluator(index_dir=index_dir)
    # Evaluate a mini benchmark of 2 queries
    benchmark = build_default_benchmark()[:2]
    report = evaluator.run_benchmark(benchmark=benchmark, top_k=5)

    assert report.total_queries == 2
    assert report.bm25_metrics.mode == "bm25"
    assert report.dense_metrics.mode == "dense"
    assert report.hybrid_metrics.mode == "hybrid"
    assert report.hybrid_metrics.mrr >= 0.0
    assert report.hybrid_metrics.average_latency_ms > 0.0


def test_benchmark_cli_command(tmp_path: Path):
    index_dir = Path("search_index")
    if not index_dir.exists():
        pytest.skip("search_index not present")

    out_json = tmp_path / "test_eval_report.json"
    result = runner.invoke(app, [
        "benchmark",
        "--index-dir", str(index_dir),
        "--output", str(out_json),
        "--top-k", "5",
    ])

    assert result.exit_code == 0
    assert out_json.exists()
    with open(out_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "hybrid_metrics" in data
    assert "bm25_metrics" in data
    assert "per_query_comparison" in data
    assert len(data["per_query_comparison"]) == 60  # 20 queries * 3 modes
