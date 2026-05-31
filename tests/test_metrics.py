"""Synthetic-data tests for retrieval metrics + bias-aware scoring."""
from __future__ import annotations

import math

import pytest

from src.experiment.metrics import (
    compute_per_user_metrics,
    hit_rate_at_k,
    mrr_first_hit,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    unpopularity_score,
)


def test_precision_basic():
    rec = ["a", "b", "c", "d"]
    rel = ["b", "d"]
    assert precision_at_k(rec, rel, 4) == pytest.approx(0.5)
    assert precision_at_k(rec, rel, 2) == pytest.approx(0.5)
    assert precision_at_k(rec, rel, 1) == pytest.approx(0.0)


def test_recall_basic():
    rec = ["a", "b", "c", "d"]
    rel = ["b", "d", "e"]
    assert recall_at_k(rec, rel, 4) == pytest.approx(2 / 3)
    assert recall_at_k(rec, rel, 4, total_relevant=10) == pytest.approx(0.2)


def test_hit_rate():
    assert hit_rate_at_k(["a", "b"], ["b"], 2) == 1.0
    assert hit_rate_at_k(["a", "b"], ["c"], 2) == 0.0
    assert hit_rate_at_k(["a", "b"], ["b"], 1) == 0.0  # b at rank 2, k=1


def test_mrr_first_hit():
    assert mrr_first_hit(["x", "y", "z"], ["z"]) == pytest.approx(1 / 3)
    assert mrr_first_hit(["a", "b"], ["a"]) == pytest.approx(1.0)
    assert mrr_first_hit(["a", "b"], ["c"]) == 0.0


def test_ndcg_perfect_order_equals_one():
    rec = ["a", "b", "c"]
    rel = ["a", "b", "c"]
    assert ndcg_at_k(rec, rel, 3) == pytest.approx(1.0)


def test_ndcg_reversed_order_lower():
    rec = ["c", "b", "a"]
    rel = ["a"]
    # Only one relevant. DCG = 1/log2(4) = 0.5, IDCG = 1/log2(2) = 1.
    assert ndcg_at_k(rec, rel, 3) == pytest.approx(0.5)


def test_ndcg_no_relevant_zero():
    assert ndcg_at_k(["a", "b"], [], 5) == 0.0


def test_unpopularity_score_picks_unpop_hits():
    rec = ["A", "B", "C"]
    rel = ["A", "C"]

    pop_map = {
        "A": 1.0,   # max popularity -> unpop = 0
        "B": 0.0,
        "C": 0.0,   # min popularity -> unpop = 10
    }

    def lookup(title, col):
        return pop_map.get(title, 0.5)

    out = unpopularity_score(rec, rel, lookup, "x", rank_weight=0.0, list_size=3)
    # A hit has unpop 0, C hit has unpop 10. With rank_weight=0, no rank bonus.
    assert out["hits"] == 2
    assert out["total_score"] == pytest.approx(10.0)
    assert out["average_score"] == pytest.approx(5.0)


def test_unpopularity_score_rank_bonus():
    rec = ["X"]
    rel = ["X"]

    def lookup(title, col):
        return 0.0  # fully unpopular

    # rank_weight=1, list_size=1 -> rank_score=(1-0)/1=1, multiplier=1+1=2
    out = unpopularity_score(rec, rel, lookup, "x", rank_weight=1.0, list_size=1)
    # base unpop = (1-0)*10 = 10, multiplied by 2 -> 20
    assert out["total_score"] == pytest.approx(20.0)


def test_compute_per_user_metrics_full_pipeline():
    orig = ["A", "B", "C", "D"]
    rer = ["C", "A", "B", "D"]
    relevant = ["A", "C"]

    def lookup(title, col):
        return 0.5

    out = compute_per_user_metrics(
        recommendations_original=orig,
        recommendations_reranked=rer,
        relevant_pool=relevant,
        test_movies_count=2,
        k=4,
        popularity_lookup=lookup,
        pop_col="x",
        rank_weight=0.2,
    )

    assert out["k"] == 4
    assert out["original"]["hits"] == 2
    assert out["reranked"]["hits"] == 2
    # Reranked puts A and C earlier, so MRR_rer >= MRR_orig.
    assert out["reranked"]["mrr"] >= out["original"]["mrr"]
    # NDCG_rer should also be >= because reranked promotes hits to higher ranks.
    assert out["reranked"]["ndcg_at_k"] >= out["original"]["ndcg_at_k"]
