#!/usr/bin/env python3
"""
Top-k truncation and metrics for reranked_JSON: evaluates both the original list
(`recommendations`) and the reranked list (`reranked_recommendations`) at the same k.

Candidates are the same 20 titles (rerank is a reordering), so `true_preferred_movies`
(test hits within the top-20 pool) is the relevance set for both lists.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Tuple

SCENARIOS_FOR_SUMMARY = ("neutral", "age", "gender", "cross")
TOP_K_DEFAULT = 15


def mrr_first_hit(rec_top: List[str], preferred_pool: List[str]) -> float:
    pref = set(preferred_pool)
    for i, title in enumerate(rec_top):
        if title in pref:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(rec_top: List[str], preferred_pool: List[str], k: int) -> float:
    relevant = set(preferred_pool)
    if not relevant:
        return 0.0

    dcg = 0.0
    for i, title in enumerate(rec_top[:k], start=1):
        if title in relevant:
            dcg += 1.0 / math.log2(i + 1)

    ideal_hits = min(k, len(relevant))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


def compute_metrics(
    rec_top: List[str],
    preferred_pool: List[str],
    k: int,
    tmc: int,
) -> Tuple[List[str], Dict[str, Any]]:
    top_set = set(rec_top)
    matched = [m for m in preferred_pool if m in top_set]
    hits = len(matched)
    denom_p = min(k, len(rec_top)) if rec_top else 0
    precision = (hits / denom_p) if denom_p else 0.0
    recall = (hits / tmc) if tmc else 0.0
    ev = {
        "precision": precision,
        "recall": recall,
        "hits": hits,
        "test_movies_count": tmc,
        "k": k,
        "ndcg_at_k": ndcg_at_k(rec_top, preferred_pool, k),
    }
    return matched, ev


def transform_scenario_block(block: Mapping[str, Any], k: int) -> Dict[str, Any]:
    rec_full = list(block.get("recommendations") or [])
    rer_full = list(block.get("reranked_recommendations") or rec_full)

    orig_top = rec_full[:k]
    rer_top = rer_full[:k]

    eval_old = block.get("evaluation")
    tmc = 0
    if isinstance(eval_old, dict):
        tmc = int(eval_old.get("test_movies_count") or 0)

    preferred_pool = list(block.get("true_preferred_movies") or [])

    pref_orig, ev_orig = compute_metrics(orig_top, preferred_pool, k, tmc)
    pref_rer, ev_rer = compute_metrics(rer_top, preferred_pool, k, tmc)

    out: Dict[str, Any] = dict(block)
    out["recommendations"] = orig_top
    out["reranked_recommendations"] = rer_top
    out["true_preferred_movies"] = pref_orig
    out["true_preferred_movies_reranked_at_k"] = pref_rer

    if eval_old is not None:
        out["evaluation_at_20"] = eval_old
        out.pop("evaluation", None)

    out["evaluation_at_15_original"] = ev_orig
    out["evaluation_at_15_reranked"] = ev_rer

    return out


def process_user_record(user_data: Mapping[str, Any], k: int) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    out = dict(user_data)
    results = user_data.get("results") or {}
    if not isinstance(results, dict):
        return out, {}

    new_results: Dict[str, Any] = {}
    stats: Dict[str, Dict[str, Any]] = {}

    for scenario, block in results.items():
        if not isinstance(block, dict):
            new_results[scenario] = block
            continue
        if "recommendations" not in block:
            new_results[scenario] = block
            continue

        new_block = transform_scenario_block(block, k)
        new_results[scenario] = new_block

        if scenario not in SCENARIOS_FOR_SUMMARY:
            continue

        preferred_pool = list(block.get("true_preferred_movies") or [])
        orig_top = list(new_block.get("recommendations") or [])
        rer_top = list(new_block.get("reranked_recommendations") or [])

        ev_o = new_block.get("evaluation_at_15_original") or {}
        ev_r = new_block.get("evaluation_at_15_reranked") or {}
        ho = int(ev_o.get("hits") or 0)
        hr = int(ev_r.get("hits") or 0)
        tmc = int(ev_o.get("test_movies_count") or 0)

        stats[scenario] = {
            "original": {
                "precision": float(ev_o.get("precision") or 0.0),
                "recall": float(ev_o.get("recall") or 0.0),
                "hits": float(ho),
                "test_movies_count": float(tmc),
                "hit_rate_unit": 1.0 if ho > 0 else 0.0,
                "mrr": mrr_first_hit(orig_top, preferred_pool),
                "ndcg_at_k": float(ev_o.get("ndcg_at_k") or 0.0),
            },
            "reranked": {
                "precision": float(ev_r.get("precision") or 0.0),
                "recall": float(ev_r.get("recall") or 0.0),
                "hits": float(hr),
                "test_movies_count": float(tmc),
                "hit_rate_unit": 1.0 if hr > 0 else 0.0,
                "mrr": mrr_first_hit(rer_top, preferred_pool),
                "ndcg_at_k": float(ev_r.get("ndcg_at_k") or 0.0),
            },
        }

    out["results"] = new_results
    return out, stats


def merge_aggregate(
    agg_orig: MutableMapping[str, Any],
    agg_rer: MutableMapping[str, Any],
    stats: Mapping[str, Mapping[str, Any]],
) -> None:
    for scen, pair in stats.items():
        for suffix, agg in (("original", agg_orig), ("reranked", agg_rer)):
            s = pair.get(suffix) or {}
            if scen not in agg:
                agg[scen] = {
                    "n_users": 0,
                    "sum_precision": 0.0,
                    "sum_recall": 0.0,
                    "sum_hits": 0.0,
                    "sum_test_movies_count": 0.0,
                    "sum_hit_rate": 0.0,
                    "sum_mrr": 0.0,
                    "sum_ndcg": 0.0,
                }
            a = agg[scen]
            a["n_users"] += 1
            a["sum_precision"] += float(s.get("precision") or 0.0)
            a["sum_recall"] += float(s.get("recall") or 0.0)
            a["sum_hits"] += float(s.get("hits") or 0.0)
            a["sum_test_movies_count"] += float(s.get("test_movies_count") or 0.0)
            a["sum_hit_rate"] += float(s.get("hit_rate_unit") or 0.0)
            a["sum_mrr"] += float(s.get("mrr") or 0.0)
            a["sum_ndcg"] += float(s.get("ndcg_at_k") or 0.0)


def finalize_track(agg: Mapping[str, Any], k: int) -> Dict[str, Any]:
    scenarios_out: Dict[str, Any] = {}
    for scen, a in agg.items():
        n = max(int(a["n_users"]), 1)
        sum_tmc = a["sum_test_movies_count"]
        scenarios_out[scen] = {
            "n_users": int(a["n_users"]),
            "macro_avg_precision": a["sum_precision"] / n,
            "macro_avg_recall": a["sum_recall"] / n,
            "micro_precision": (a["sum_hits"] / (n * k)) if n else 0.0,
            "micro_recall": (a["sum_hits"] / sum_tmc) if sum_tmc > 0 else 0.0,
            "hit_rate_at_k": a["sum_hit_rate"] / n,
            "mrr_at_k": a["sum_mrr"] / n,
            "ndcg_at_k": a["sum_ndcg"] / n,
            "total_hits": int(a["sum_hits"]),
            "total_test_movies_count": int(sum_tmc),
        }
    return scenarios_out


def build_comparison(orig: Mapping[str, Any], rer: Mapping[str, Any], k: int) -> Dict[str, Any]:
    comp: Dict[str, Any] = {}
    keys = sorted(set(orig.keys()) | set(rer.keys()))
    metrics = (
        "macro_avg_precision",
        "macro_avg_recall",
        "micro_precision",
        "micro_recall",
        "hit_rate_at_k",
        "mrr_at_k",
        "ndcg_at_k",
    )
    for scen in keys:
        bo = orig.get(scen) or {}
        br = rer.get(scen) or {}
        row: Dict[str, Any] = {"original": bo, "reranked_list": br}
        for m in metrics:
            vo = bo.get(m)
            vr = br.get(m)
            if isinstance(vo, (int, float)) and isinstance(vr, (int, float)):
                row[f"delta_reranked_minus_original_{m}"] = float(vr) - float(vo)
        comp[scen] = row
    return {"k": k, "by_scenario": comp}


def cmd_process(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    k = int(args.k)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("recommendation_results_*.json"))
    if not json_files:
        raise SystemExit(f"No recommendation_results_*.json in {input_dir}")

    agg_orig: Dict[str, Any] = {}
    agg_rer: Dict[str, Any] = {}

    for path in json_files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            continue

        out_data: Dict[str, Any] = {}
        for uid, user_rec in data.items():
            if not isinstance(user_rec, dict):
                out_data[uid] = user_rec
                continue
            new_rec, stats = process_user_record(user_rec, k)
            out_data[uid] = new_rec
            merge_aggregate(agg_orig, agg_rer, stats)

        out_path = output_dir / path.name
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

    mo = finalize_track(agg_orig, k)
    mr = finalize_track(agg_rer, k)

    summary = {
        "k": k,
        "input_dir": str(input_dir),
        "notes": {
            "original_list": "recommendations (truncated to k)",
            "reranked_list": "reranked_recommendations (truncated to k)",
            "relevance": "true_preferred_movies = test hits within the shared top-20 candidate pool",
            "macro_avg": "mean of per-user precision/recall",
            "micro_precision": "sum(hits) / (n_users * k)",
            "micro_recall": "sum(hits) / sum(test_movies_count)",
            "mrr_at_k": "mean reciprocal rank of first preferred-pool hit in truncated list",
            "ndcg_at_k": "mean NDCG@k with binary relevance over true_preferred_movies",
        },
        "metrics_original_list": mo,
        "metrics_reranked_list": mr,
        "comparison_reranked_minus_original": build_comparison(mo, mr, k)["by_scenario"],
    }

    summary_path = output_dir / "summary_top15.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    comparison_root = Path(args.comparison_output).resolve()
    comp_full = {
        "k": k,
        "summary_source": str(summary_path),
        "interpretation": (
            "Original vs reranked metrics both use the same test_movies_count denominator. "
            "Hits count preferred movies from the shared top-20 pool that appear in each truncated list. "
            "NDCG@k uses binary relevance and log-discounted rank positions."
        ),
        **build_comparison(mo, mr, k),
    }
    comparison_root.parent.mkdir(parents=True, exist_ok=True)
    with comparison_root.open("w", encoding="utf-8") as f:
        json.dump(comp_full, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(json_files)} shard files under {output_dir}")
    print(f"Summary: {summary_path}")
    print(f"Comparison: {comparison_root}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Top-k metrics for reranked_JSON: original vs reranked_recommendations.",
    )
    p.add_argument(
        "--input-dir",
        default="reranked_JSON",
        help="Directory with reranked shard JSON files (default: reranked_JSON).",
    )
    p.add_argument(
        "--output-dir",
        default="recommendations_top15_reranked_analysis",
        help="Where to write truncated shards and summary_top15.json.",
    )
    p.add_argument("--k", type=int, default=TOP_K_DEFAULT)
    p.add_argument(
        "--comparison-output",
        default="comparison_top15_original_vs_reranked_list.json",
        help="Path for comparison JSON (relative to cwd if not absolute).",
    )
    p.set_defaults(func=cmd_process)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not Path(args.comparison_output).is_absolute():
        args.comparison_output = str((Path.cwd() / args.comparison_output).resolve())
    args.func(args)


if __name__ == "__main__":
    main()
