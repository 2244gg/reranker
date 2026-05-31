"""Final report for v2 600-user main run.

Renders the orig vs v2 head-to-head plus a Pareto-frontier scatter
across all available rerank variants in this directory tree.

Run after the main experiment finishes:
    python final_report_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def _safe_load(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


SCENARIOS = ["neutral", "gender", "age_group", "gender+age_group"]

# (label, orig_key, rer_key)
PER_USER_METRICS = [
    ("Precision@20",      "mean_orig_precision",          "mean_rer_precision"),
    ("Recall@20",         "mean_orig_recall",             "mean_rer_recall"),
    ("HitRate@20",        "mean_orig_hit_rate",           "mean_rer_hit_rate"),
    ("MRR",               "mean_orig_mrr",                "mean_rer_mrr"),
    ("NDCG@20",           "mean_orig_ndcg_at_k",          "mean_rer_ndcg_at_k"),
    ("Hits",              "mean_orig_hits",               "mean_rer_hits"),
    ("APLT (slice)",      "mean_orig_aplt",               "mean_rer_aplt"),
    ("APLT (neutral)",    "mean_orig_aplt_neutral",       "mean_rer_aplt_neutral"),
    ("UnpopAvg",          "mean_orig_unpopularity_average","mean_rer_unpopularity_average"),
    ("DiffAvg",           "mean_orig_difficulty_average", "mean_rer_difficulty_average"),
]
CORPUS_METRICS = [
    ("UniqueRecs",  "orig_unique_recommendations", "rer_unique_recommendations"),
    ("Coverage",    "orig_coverage",               "rer_coverage"),
    ("Gini",        "orig_gini",                   "rer_gini"),
]


def render_block(label: str, agg_path: Path) -> None:
    data = _safe_load(agg_path)
    scns = data.get("scenarios", {})
    if not scns:
        print(f"  [skip] {label}: no data at {agg_path}")
        return
    print()
    print("=" * 84)
    print(f"  {label}")
    print(f"  source: {agg_path}")
    print("=" * 84)
    for sc in SCENARIOS:
        s = scns.get(sc, {})
        if not s:
            continue
        print(f"\n--- scenario={sc}  (n_users={int(s.get('n_users',0))}) ---")
        print(f"  {'metric':<22} {'orig':>10} {'rer':>10} {'delta':>10} {'%chg':>9}")
        print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*10} {'-'*9}")
        for lab, ok, rk in PER_USER_METRICS:
            ov, rv = s.get(ok), s.get(rk)
            if ov is None or rv is None:
                continue
            d = rv - ov
            pct = (d / ov * 100.0) if ov else float("inf")
            print(f"  {lab:<22} {ov:>10.4f} {rv:>10.4f} {d:>+10.4f} {pct:>+8.1f}%")
        # corpus
        any_corpus = False
        for lab, ok, rk in CORPUS_METRICS:
            ov, rv = s.get(ok), s.get(rk)
            if ov is None or rv is None:
                continue
            if not any_corpus:
                print("  -- corpus --")
                any_corpus = True
            d = rv - ov
            pct = (d / ov * 100.0) if ov else float("inf")
            print(f"  {lab:<22} {ov:>10.4f} {rv:>10.4f} {d:>+10.4f} {pct:>+8.1f}%")


def pareto_scatter() -> None:
    """Cross-config (APLT, NDCG) scatter to visualize Pareto frontier."""
    candidates: List[Tuple[str, Path]] = [
        ("orig (no rerank)",            Path("results_v2/gpt-4.1-mini/lastfm1k/metrics/aggregate.json")),
        ("alpha=0.7 raw",               Path("results/gpt-4.1-mini/lastfm1k/metrics/aggregate.json")),
        ("alpha=0.5 raw",               Path("results_a05/gpt-4.1-mini/lastfm1k/metrics/aggregate.json")),
        ("v2 (rank-B + protect5)",      Path("results_v2/gpt-4.1-mini/lastfm1k/metrics/aggregate.json")),
    ]
    print()
    print("=" * 84)
    print("  Pareto scatter: APLT (long-tail exposure) vs NDCG (accuracy)")
    print("  Higher-right = better. Configs in the upper-right quadrant of the cloud")
    print("  define the empirical Pareto frontier; any config below+left is dominated.")
    print("=" * 84)
    print()
    print(f"  {'config':<28} {'scenario':<22} {'APLT(slice)':>12} {'NDCG@20':>10}")
    print(f"  {'-'*28} {'-'*22} {'-'*12} {'-'*10}")
    for label, path in candidates:
        data = _safe_load(path)
        scns = data.get("scenarios", {})
        if not scns:
            continue
        for sc in SCENARIOS:
            s = scns.get(sc, {})
            if not s:
                continue
            # For "orig (no rerank)" use mean_orig_* (no rerank applied).
            # For rerank configs use mean_rer_*.
            if label.startswith("orig"):
                aplt = s.get("mean_orig_aplt") or s.get("mean_orig_aplt_neutral")
                ndcg = s.get("mean_orig_ndcg_at_k")
            else:
                aplt = s.get("mean_rer_aplt") or s.get("mean_rer_aplt_neutral")
                ndcg = s.get("mean_rer_ndcg_at_k")
            if aplt is None or ndcg is None:
                continue
            print(f"  {label:<28} {sc:<22} {aplt:>12.4f} {ndcg:>10.4f}")


def main() -> None:
    main_path = Path("results_v2/gpt-4.1-mini/lastfm1k/metrics/aggregate.json")
    print("=" * 84)
    print("  V2 RERANKER — FINAL REPORT")
    print("  Algorithm: rank-normalized B + protect_top=5, alpha_base=0.7 (dynamic)")
    print("  Dataset:   LastFM-1K (track granularity, 942 users with min_int=20)")
    print("  Metrics:   Precision/Recall/HitRate/MRR/NDCG (accuracy)")
    print("             APLT (long-tail exposure), Gini (distribution equality)")
    print("=" * 84)
    render_block("Main run (n=600)", main_path)
    pareto_scatter()


if __name__ == "__main__":
    main()
