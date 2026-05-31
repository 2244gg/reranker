"""Re-run rerank + evaluation on already-generated raw LLM recommendations.

Reads ``results/<model>/<dataset>/raw/recommendation_results_full.json``
(produced by a prior pilot run) and recomputes:
  * dynamic alpha (per-user, per-scenario)
  * rerank top-k under the current formula
  * full metric suite (precision, recall, NDCG, MRR, hit rate,
    unpopularity, difficulty)
  * aggregate.json + per_user.parquet / .csv
  * meta.json (notes which raw JSON was reused)

Does NOT touch the raw JSON and makes ZERO API calls. Use this to
iterate on rerank parameters without re-paying the LLM bill.

Usage:
    python scripts/rerank_only.py --results-dir results/gpt-4.1-mini/lastfm1k
    python scripts/rerank_only.py --results-dir results/... \\
        --alpha-base 0.5 --alpha-gain 0.3
    python scripts/rerank_only.py --results-dir results/... \\
        --top-k 15 --rank-weight 0.5
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment.metrics import compute_per_user_metrics, normalize_title  # noqa: E402
from src.experiment.rerank import (  # noqa: E402
    compute_dynamic_alpha,
    rerank_one_list,
)
from src.experiment.runner import PopularityTable  # noqa: E402
from src.popularity.item_scoring import ItemScoreTable  # noqa: E402


def _load_raw(results_dir: Path) -> Dict:
    raw_path = results_dir / "raw" / "recommendation_results_full.json"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw LLM outputs not found at {raw_path}. "
            f"Run a pilot first (e.g. scripts/run_pilot_*.py)."
        )
    return json.loads(raw_path.read_text(encoding="utf-8"))


def _resolve_popularity_csv(results_dir: Path, dataset_name: str) -> Path:
    candidates = [
        Path("data/processed") / dataset_name / "items_with_popularity.csv",
        Path("movies_with_popularity.csv"),  # legacy ML-1M
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"No popularity CSV found for dataset {dataset_name!r}. "
        "Run scripts/build_popularity.py first."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-dir",
        required=True,
        help="Path like results/<model>/<dataset>/ containing raw/ + meta.json",
    )
    p.add_argument(
        "--popularity-csv",
        default=None,
        help="Override popularity CSV path. Default: data/processed/<dataset>/items_with_popularity.csv",
    )
    p.add_argument("--top-k", type=int, default=None, help="Override final list length M")
    p.add_argument(
        "--alpha-base",
        type=float,
        default=None,
        help="Override rerank alpha_base (default 0.7)",
    )
    p.add_argument("--alpha-min", type=float, default=None)
    p.add_argument("--alpha-max", type=float, default=None)
    p.add_argument(
        "--alpha-gain",
        type=float,
        default=None,
        help="Override alpha_gain. Sign matters: + = personalization, - = fairness convention.",
    )
    p.add_argument("--decay-rate", type=float, default=None)
    p.add_argument("--base", type=float, default=None, help="Override P_norm base floor")
    p.add_argument("--rank-weight", type=float, default=None)
    p.add_argument(
        "--disable-dynamic-alpha",
        action="store_true",
        help="Use a fixed alpha_base instead of dynamic per-user alpha",
    )
    p.add_argument(
        "--score-scheme",
        default="linear_x10",
        choices=["linear_x10", "log", "bucket"],
        help="Difficulty score scheme (default: linear_x10).",
    )
    p.add_argument(
        "--output-suffix",
        default="",
        help="Append a suffix to the metrics dir, e.g. '_alpha050_gain030', "
        "to store multiple rerank variants without overwriting each other.",
    )
    args = p.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results dir not found: {results_dir}")

    raw = _load_raw(results_dir)
    meta_path = results_dir / "meta.json"
    if not meta_path.exists():
        raise SystemExit(f"meta.json not found in {results_dir}")
    prior_meta = json.loads(meta_path.read_text(encoding="utf-8"))

    dataset = prior_meta.get("dataset", results_dir.name)
    top_k = args.top_k or prior_meta.get("top_k", 20)
    candidate_n = prior_meta.get("candidate_pool_size", top_k)
    rank_weight = args.rank_weight if args.rank_weight is not None else 0.2

    rerank_prior = prior_meta.get("rerank") or {}
    alpha_base = args.alpha_base if args.alpha_base is not None else rerank_prior.get("alpha_base", 0.7)
    alpha_min = args.alpha_min if args.alpha_min is not None else rerank_prior.get("alpha_min", 0.5)
    alpha_max = args.alpha_max if args.alpha_max is not None else rerank_prior.get("alpha_max", 0.9)
    alpha_gain = args.alpha_gain if args.alpha_gain is not None else rerank_prior.get("alpha_gain", 0.2)
    base = args.base if args.base is not None else rerank_prior.get("base", 0.2)
    decay_rate = args.decay_rate if args.decay_rate is not None else rerank_prior.get("decay_rate", 0.9)
    disable_dynamic = args.disable_dynamic_alpha or rerank_prior.get("disable_dynamic_alpha", False)

    pop_csv = Path(args.popularity_csv) if args.popularity_csv else _resolve_popularity_csv(
        results_dir, dataset
    )
    pop_df = pd.read_csv(pop_csv)
    pop_table = PopularityTable(pop_df)
    score_table = ItemScoreTable.from_popularity(pop_df, scheme=args.score_scheme)

    print(f"[rerank_only] results_dir = {results_dir}")
    print(f"[rerank_only] dataset     = {dataset}")
    print(f"[rerank_only] top_k       = {top_k}")
    print(f"[rerank_only] cand_pool   = {candidate_n}")
    print(f"[rerank_only] alpha_base  = {alpha_base}, gain = {alpha_gain}, range = [{alpha_min}, {alpha_max}]")
    print(f"[rerank_only] decay/base  = {decay_rate} / {base}")
    print(f"[rerank_only] disable_dyn = {disable_dynamic}")
    print(f"[rerank_only] score_scheme= {args.score_scheme}")
    print(f"[rerank_only] popularity  = {pop_csv}")
    print()

    # ----------------------------------------------------------------
    # Pass 1: collect scene_mean from already-saved history_popularity_score
    # ----------------------------------------------------------------
    scene_score_sum: Dict[str, float] = defaultdict(float)
    scene_user_count: Dict[str, int] = defaultdict(int)
    for user_id, payload in raw.items():
        for label, scn in (payload.get("results") or {}).items():
            score = scn.get("history_popularity_score")
            if score is None:
                continue
            scene_score_sum[label] += float(score)
            scene_user_count[label] += 1
    scene_means: Dict[str, float] = {
        label: (scene_score_sum[label] / scene_user_count[label])
        if scene_user_count[label] else 0.0
        for label in scene_score_sum
    }

    # ----------------------------------------------------------------
    # Pass 2: rerank + metrics for each user x scenario
    # ----------------------------------------------------------------
    rer_payload: Dict[str, dict] = {}
    per_user_rows: List[dict] = []

    for user_id, user_block in raw.items():
        out_block = {"user_info": user_block.get("user_info", {}), "results": {}}
        for label, scn in (user_block.get("results") or {}).items():
            recs = list(scn.get("recommendations") or [])
            if not recs:
                out_block["results"][label] = scn
                continue
            scenario_dims = tuple(scn.get("scenario") or ())
            pop_col = scn.get("popularity_column_used") or "neutral_All"
            user_score = float(scn.get("history_popularity_score") or 0.0)
            scene_mean = scene_means.get(label, 0.0)

            if not scenario_dims:  # neutral: bypass
                reranked = list(recs)[:top_k]
                dynamic_alpha = None
                rerank_scores: List[dict] = []
            else:
                if disable_dynamic:
                    dynamic_alpha = alpha_base
                else:
                    dynamic_alpha = compute_dynamic_alpha(
                        user_score=user_score,
                        scene_mean_score=scene_mean,
                        alpha_base=alpha_base,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max,
                        alpha_gain=alpha_gain,
                    )
                rerank_out = rerank_one_list(
                    recs,
                    pop_table.lookup_with_found,
                    pop_col,
                    alpha=dynamic_alpha,
                    base=base,
                    decay_rate=decay_rate,
                    top_n=top_k,
                )
                reranked = rerank_out["reranked_recommendations"]
                rerank_scores = rerank_out["rerank_scores"]

            relevant_pool = scn.get("true_preferred_items") or scn.get("true_preferred_movies") or []
            metrics = compute_per_user_metrics(
                recommendations_original=recs,
                recommendations_reranked=reranked,
                relevant_pool=relevant_pool,
                test_movies_count=len(relevant_pool) or 1,
                k=top_k,
                popularity_lookup=lambda t, c, _pop=pop_table: _pop.lookup(t, c),
                pop_col=pop_col,
                rank_weight=rank_weight,
                score_table=score_table,
            )

            out_block["results"][label] = {
                **scn,
                "reranked_recommendations": reranked,
                "rerank_scores": rerank_scores,
                "dynamic_alpha": dynamic_alpha,
                "scene_population_mean_score": scene_mean,
                "metrics": metrics,
            }

            row = {"user_id": user_id, "scenario": label, "alpha": dynamic_alpha or alpha_base}
            for side in ("original", "reranked"):
                m = metrics[side]
                for key in ("precision", "recall", "hit_rate", "mrr", "ndcg_at_k", "hits"):
                    row[f"{'orig' if side == 'original' else 'rer'}_{key}"] = float(m[key])
                row[f"{'orig' if side == 'original' else 'rer'}_unpopularity_total"] = float(
                    m["unpopularity"]["total_score"]
                )
                row[f"{'orig' if side == 'original' else 'rer'}_unpopularity_average"] = float(
                    m["unpopularity"]["average_score"]
                )
                if "difficulty" in m:
                    row[f"{'orig' if side == 'original' else 'rer'}_difficulty_total"] = float(
                        m["difficulty"]["total_score"]
                    )
                    row[f"{'orig' if side == 'original' else 'rer'}_difficulty_average"] = float(
                        m["difficulty"]["average_score"]
                    )
            per_user_rows.append(row)

        rer_payload[user_id] = out_block

    # ----------------------------------------------------------------
    # Persist (suffix-aware so multiple variants can coexist)
    # ----------------------------------------------------------------
    suffix = args.output_suffix
    rer_dir = results_dir / ("reranked" + suffix)
    metrics_dir = results_dir / ("metrics" + suffix)
    rer_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rer_path = rer_dir / "recommendation_results_full.json"
    rer_path.write_text(json.dumps(rer_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame(per_user_rows)
    parquet_path = metrics_dir / "per_user.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        parquet_path = metrics_dir / "per_user.csv"
        df.to_csv(parquet_path, index=False)

    aggregate: Dict[str, dict] = {"scenarios": {}}
    if not df.empty:
        for scenario, sub in df.groupby("scenario"):
            bucket = {"n_users": int(len(sub))}
            for col in sub.columns:
                if col in ("user_id", "scenario"):
                    continue
                bucket[f"mean_{col}"] = float(sub[col].mean())
            aggregate["scenarios"][str(scenario)] = bucket
        aggregate["n_users_total"] = int(df["user_id"].nunique())

    (metrics_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    new_meta = {
        **prior_meta,
        "rerank_run": {
            "regenerated_at": datetime.now(timezone.utc).isoformat(),
            "alpha_base": alpha_base,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "alpha_gain": alpha_gain,
            "base": base,
            "decay_rate": decay_rate,
            "disable_dynamic_alpha": disable_dynamic,
            "top_k": top_k,
            "rank_weight": rank_weight,
            "score_scheme": args.score_scheme,
            "output_suffix": suffix,
            "popularity_csv": str(pop_csv),
        },
    }
    (results_dir / f"meta{suffix}.json").write_text(
        json.dumps(new_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[rerank_only] wrote {rer_path}")
    print(f"[rerank_only] wrote {parquet_path}")
    print(f"[rerank_only] wrote {metrics_dir / 'aggregate.json'}")
    print(f"[rerank_only] wrote {results_dir / f'meta{suffix}.json'}")
    print()
    print("Per-scenario summary:")
    if not df.empty:
        cols = [
            "orig_hits", "rer_hits",
            "orig_ndcg_at_k", "rer_ndcg_at_k",
            "orig_mrr", "rer_mrr",
            "orig_difficulty_total", "rer_difficulty_total",
        ]
        cols = [c for c in cols if c in df.columns]
        g = df.groupby("scenario")[cols].mean().round(4)
        print(g.to_string())


if __name__ == "__main__":
    main()
