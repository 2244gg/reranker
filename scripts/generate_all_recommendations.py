"""Generate raw LLM recommendations for the FULL user population of a dataset.

Usage:
    python scripts/generate_all_recommendations.py --dataset lastfm1k
    python scripts/generate_all_recommendations.py --dataset lastfm1k --max-users 200 --dry-run
    python scripts/generate_all_recommendations.py --dataset ml1m --model gpt-4.1-mini

Notes
-----
* This is a one-time-ish bulk LLM call. Cost on gpt-4.1-mini for LFM-1K is
  about US$3 and takes several hours sequentially. Output is cached in the
  SQLite LLM cache, so resuming after a crash costs nothing extra.

* The result lands in ``results/<model>/<dataset>/raw/recommendation_results_full.json``
  which is what ``scripts/rerank_only.py`` consumes. After this run, you can
  iterate on rerank parameters / score schemes / etc. without paying again.

* Per-user progress markers under ``results/<model>/<dataset>/.progress/<experiment_id>/``
  let you resume a partial run.

* Default candidate_pool_size is 30 (M+N where M=top_k=20, N=10).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.experiment.config import ExperimentConfig, LLMConfig, RerankConfig  # noqa: E402
from src.experiment.runner import run  # noqa: E402


_DATASET_DEFAULTS = {
    "lastfm1k": {
        "system_prompt": "You are a professional music recommendation system.",
        "history_limit": 10,
        "max_tokens": 1500,
        "popularity_csv": None,  # auto-build
    },
    "ml1m": {
        "system_prompt": "You are a professional movie recommendation system.",
        "history_limit": 10,
        "max_tokens": 1500,
        "popularity_csv": "movies_with_popularity.csv",
    },
    "bookcrossing": {
        "system_prompt": "You are a professional book recommendation system.",
        "history_limit": 10,
        "max_tokens": 1500,
        "popularity_csv": None,
    },
}


def _build_config(args: argparse.Namespace, n_users_target: int) -> ExperimentConfig:
    ds = args.dataset
    defaults = _DATASET_DEFAULTS.get(ds, _DATASET_DEFAULTS["lastfm1k"])
    return ExperimentConfig(
        experiment_id=args.experiment_id or f"full_{args.model.replace('/', '_')}_{ds}",
        dataset=ds,
        sample_size=n_users_target,
        sample_seed=args.sample_seed,
        test_split=args.test_split,
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        history_limit=defaults["history_limit"],
        rank_weight=args.rank_weight,
        popularity_csv=args.popularity_csv or defaults["popularity_csv"],
        output_root=args.output_root,
        cache_dir=args.cache_dir,
        scenarios=None,  # auto from adapter schema
        llm=LLMConfig(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens or defaults["max_tokens"],
            system_prompt=defaults["system_prompt"],
        ),
        rerank=RerankConfig(),
        extra={"implicit": args.implicit} if ds == "bookcrossing" else {},
    )


def _eligible_users(dataset: str, sample_seed: int, test_split: float) -> int:
    """Return the count of users that pass the adapter's filters (post-split)."""
    from src.datasets.factory import build_dataset

    print(f"[generate_all] loading dataset '{dataset}' to count eligible users...")
    t0 = time.time()
    adapter = build_dataset(dataset)
    adapter.split(seed=sample_seed, test_size=test_split)
    n = len(adapter.load())
    print(f"[generate_all]   loaded in {time.time() - t0:.1f}s  ->  {n} users")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        required=True,
        choices=list(_DATASET_DEFAULTS.keys()),
        help="Which dataset's full population to generate for.",
    )
    p.add_argument("--model", default="gpt-4.1-mini")
    p.add_argument("--provider", default="openai")
    p.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Cap the population (handy for cost-controlled trial runs). "
        "Default = all eligible users.",
    )
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--candidate-pool-size", type=int, default=30)
    p.add_argument("--sample-seed", type=int, default=42)
    p.add_argument("--test-split", type=float, default=0.7)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--rank-weight", type=float, default=0.2)
    p.add_argument("--output-root", default="results")
    p.add_argument("--cache-dir", default="data/cache")
    p.add_argument("--popularity-csv", default=None)
    p.add_argument(
        "--experiment-id",
        default=None,
        help="Override experiment_id. Defaults to full_<model>_<dataset>.",
    )
    p.add_argument(
        "--implicit",
        action="store_true",
        help="BookCrossing only: treat any interaction as positive feedback.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the projected token / cost / time budget and exit.",
    )
    p.add_argument("--no-confirm", action="store_true")
    args = p.parse_args()

    eligible = _eligible_users(args.dataset, args.sample_seed, args.test_split)
    target = min(args.max_users, eligible) if args.max_users else eligible

    cfg = _build_config(args, target)
    n_scenarios = 4  # 2 dims => 2^2 = 4 (neutral + 2 single + 1 cross)
    if args.dataset == "ml1m":
        # ML-1M default: 2 dims (gender + age_group)
        n_scenarios = 4
    total_calls = target * n_scenarios

    # Token + cost projection (gpt-4.1-mini-ish; vary by model).
    pool = cfg.candidate_pool_size or cfg.top_k
    est_in_per_call = 600 + max(0, (pool - 20)) * 20
    est_out_per_call = 300 + max(0, (pool - 20)) * 50
    est_in = total_calls * est_in_per_call
    est_out = total_calls * est_out_per_call
    cost_in_per_M = 0.40 if "gpt-4.1-mini" in cfg.llm.model else 0.50
    cost_out_per_M = 1.60 if "gpt-4.1-mini" in cfg.llm.model else 1.50
    est_cost = est_in / 1e6 * cost_in_per_M + est_out / 1e6 * cost_out_per_M
    sec_per_call = 5.0
    est_hours = total_calls * sec_per_call / 3600

    print()
    print("=" * 64)
    print("FULL-POPULATION RAW RECOMMENDATION GENERATION")
    print("=" * 64)
    print(f"  experiment_id    : {cfg.experiment_id}")
    print(f"  dataset          : {cfg.dataset}")
    print(f"  model            : {cfg.llm.model}  (provider={cfg.llm.provider})")
    print(f"  eligible users   : {eligible}")
    print(f"  target users     : {target}{' (capped via --max-users)' if args.max_users else ' (all)'}")
    print(f"  scenarios        : {n_scenarios}")
    print(f"  candidate pool   : {pool} (M+N) -> truncate to top_k={cfg.top_k}")
    print(f"  total LLM calls  : {total_calls}  (cached calls cost $0)")
    print(f"  est. tokens      : ~{est_in:,} in / ~{est_out:,} out")
    print(f"  est. cost        : ${est_cost:.2f}  (assumes 0% cache hit; less if resuming)")
    print(f"  est. wall time   : ~{est_hours:.1f} hours sequential @ {sec_per_call}s/call")
    print(f"  output           : {args.output_root}/{cfg.llm.model.replace('/', '_')}/{cfg.dataset}/raw/")
    print("=" * 64)

    if args.dry_run:
        print("\n[dry-run] Nothing executed.")
        return

    # API key check.
    import os

    if cfg.llm.provider in ("openai", "siliconflow", "together"):
        env_key = "OPENAI_API_KEY" if cfg.llm.provider == "openai" else "SILICONFLOW_API_KEY"
    elif cfg.llm.provider == "anthropic":
        env_key = "ANTHROPIC_API_KEY"
    else:
        env_key = "OPENAI_API_KEY"

    if not os.getenv(env_key):
        print(f"\nERROR: {env_key} is not set. Configure .env first.")
        sys.exit(2)
    key_val = os.getenv(env_key, "")
    if "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" in key_val or key_val.startswith("sk-xxxxxx"):
        print(f"\nERROR: {env_key} still contains the .env.example placeholder.")
        sys.exit(2)

    if not args.no_confirm:
        print(f"\nThis will make up to {total_calls} LLM calls and may take hours.")
        try:
            answer = input("Type 'yes' to start, anything else to abort: ")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)
        if answer.strip().lower() != "yes":
            print("Aborted.")
            sys.exit(0)

    print(f"\nStarting at {time.strftime('%Y-%m-%d %H:%M:%S')}...")
    t0 = time.time()
    meta = run(cfg)
    elapsed = time.time() - t0
    hours = elapsed / 3600

    print()
    print("=" * 64)
    print("FULL-POPULATION GENERATION COMPLETE")
    print("=" * 64)
    print(f"  finished_at      : {meta.get('finished_at')}")
    print(f"  api calls        : {meta.get('total_api_calls')}")
    print(f"  prompt tokens    : {meta.get('total_prompt_tokens'):,}")
    print(f"  completion tokens: {meta.get('total_completion_tokens'):,}")
    print(f"  wall time        : {hours:.2f} hours")
    print(f"  raw output       : results/{cfg.llm.model.replace('/', '_')}/{cfg.dataset}/raw/")
    print()
    print("Now you can iterate on rerank parameters cheaply, e.g.:")
    print(
        f"  python scripts/rerank_only.py --results-dir "
        f"results/{cfg.llm.model.replace('/', '_')}/{cfg.dataset}"
    )


if __name__ == "__main__":
    main()
