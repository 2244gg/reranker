"""Single-file minimal pilot for ML-1M + GPT-4.1-mini.

Designed for:
  * Quick smoke test of the entire pipeline on real data with a small budget.
  * Sane defaults so you can do `python scripts/run_pilot_ml1m.py` and
    see real LLM recommendations within a few minutes.

Defaults: 10 users x 8 scenarios = 80 LLM calls. Estimated cost on
``gpt-4.1-mini`` ~ $0.05.

Usage:
    python scripts/run_pilot_ml1m.py
    python scripts/run_pilot_ml1m.py --sample-size 50
    python scripts/run_pilot_ml1m.py --dry-run            # show config and
                                                          # prompt sample,
                                                          # no API calls
    python scripts/run_pilot_ml1m.py --model gpt-4o-mini  # try other model
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make project root importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.experiment.config import (  # noqa: E402
    ExperimentConfig,
    LLMConfig,
    RerankConfig,
)
from src.experiment.prompt_builder import PromptBuilder  # noqa: E402
from src.experiment.runner import run  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    experiment_id="pilot_gpt41mini_ml1m",
    dataset="ml1m",   # 2 demographic dims by default (gender + age_group)
    sample_size=10,
    sample_seed=42,
    test_split=0.7,
    top_k=20,
    history_limit=10,
    rank_weight=0.2,
    popularity_csv=str(PROJECT_ROOT / "movies_with_popularity.csv"),
    output_root="results",
    cache_dir="data/cache",
    model="gpt-4.1-mini",
    provider="openai",
    temperature=0.7,
    max_tokens=1000,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=args.experiment_id,
        dataset=DEFAULTS["dataset"],
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        test_split=DEFAULTS["test_split"],
        top_k=args.top_k,
        history_limit=DEFAULTS["history_limit"],
        rank_weight=DEFAULTS["rank_weight"],
        popularity_csv=DEFAULTS["popularity_csv"],
        output_root=DEFAULTS["output_root"],
        cache_dir=DEFAULTS["cache_dir"],
        scenarios=None,  # auto = all subsets up to 3-way cross
        llm=LLMConfig(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt="You are a professional movie recommendation system.",
        ),
        rerank=RerankConfig(),
    )


def _print_banner(cfg: ExperimentConfig, n_scenarios: int) -> None:
    total_calls = cfg.sample_size * n_scenarios
    # gpt-4.1-mini approx pricing (May 2025): $0.40/1M input, $1.60/1M output.
    # Pilot budget assumption: ~600 input + ~300 output tokens per call.
    est_in = total_calls * 600
    est_out = total_calls * 300
    est_cost = est_in / 1e6 * 0.40 + est_out / 1e6 * 1.60

    print("=" * 60)
    print("ML-1M PILOT")
    print("=" * 60)
    print(f"  experiment_id : {cfg.experiment_id}")
    print(f"  dataset       : {cfg.dataset}")
    print(f"  model         : {cfg.llm.model}  (provider={cfg.llm.provider})")
    print(f"  sample size   : {cfg.sample_size} users")
    print(f"  scenarios     : {n_scenarios}")
    print(f"  total calls   : {total_calls}")
    print(f"  est tokens    : ~{est_in:,} in / ~{est_out:,} out")
    print(f"  est cost      : ${est_cost:.3f} (gpt-4.1-mini, may vary)")
    print(f"  cache         : {cfg.cache_dir}/llm_cache.sqlite (cached calls cost $0)")
    print("=" * 60)


def _show_prompt_sample(cfg: ExperimentConfig) -> None:
    """Print one rendered prompt to stdout so the user sees what gets sent."""
    from src.datasets.factory import build_dataset

    adapter = build_dataset(cfg.dataset)
    adapter.split(seed=cfg.sample_seed, test_size=cfg.test_split)
    sample = adapter.stratified_sample(1, seed=cfg.sample_seed)
    if not sample:
        print("[no users to sample]")
        return
    user = sample[0]
    pb = PromptBuilder(domain=adapter.domain(), history_limit=cfg.history_limit)
    print("\n--- Sample user ---")
    print(f"  user_id       : {user.user_id}")
    print(f"  demographics  : {user.demographics}")
    print(f"  history (top 5):")
    for it in user.history[:5]:
        print(f"    - {it.title} (rating={it.rating})")

    print("\n--- Sample prompt: cross scenario (gender + age + occupation) ---")
    rendered = pb.render(
        user, scenario=("gender", "age_group", "occupation_group"), movie_count=cfg.top_k
    )
    print(rendered.text)
    print(f"\n  prompt sha256 = {rendered.sha256[:16]}...")


def _print_summary(meta: dict) -> None:
    print("\n" + "=" * 60)
    print("RUN COMPLETE")
    print("=" * 60)
    print(f"  experiment_id : {meta.get('experiment_id')}")
    print(f"  finished      : {meta.get('finished_at')}")
    print(f"  api calls     : {meta.get('total_api_calls')}")
    print(
        "  tokens used   : "
        f"{meta.get('total_prompt_tokens'):,} prompt + "
        f"{meta.get('total_completion_tokens'):,} completion"
    )
    print(f"  output root   : results/{meta.get('model').replace('/', '_')}/{meta.get('dataset')}")
    print()
    print("Next steps:")
    print(f"  1. Inspect raw LLM outputs:")
    print(f"     results/{meta.get('model').replace('/', '_')}/{meta.get('dataset')}/raw/")
    print(f"  2. Inspect reranked + metrics:")
    print(f"     results/{meta.get('model').replace('/', '_')}/{meta.get('dataset')}/reranked/")
    print(f"     results/{meta.get('model').replace('/', '_')}/{meta.get('dataset')}/metrics/")
    print(f"  3. View aggregate.json:")
    aggregate_path = (
        Path("results")
        / meta.get("model").replace("/", "_")
        / meta.get("dataset")
        / "metrics"
        / "aggregate.json"
    )
    if aggregate_path.exists():
        agg = json.loads(aggregate_path.read_text(encoding="utf-8"))
        scenarios = agg.get("scenarios", {})
        print(f"\n  Per-scenario means (n_users={agg.get('n_users_total', '?')}):")
        for scn, vals in scenarios.items():
            ndcg_o = vals.get("mean_orig_ndcg_at_k", 0.0)
            ndcg_r = vals.get("mean_rer_ndcg_at_k", 0.0)
            unp_o = vals.get("mean_orig_unpopularity_average", 0.0)
            unp_r = vals.get("mean_rer_unpopularity_average", 0.0)
            print(
                f"    {scn:>40}  "
                f"NDCG: {ndcg_o:.4f} -> {ndcg_r:.4f} ({ndcg_r - ndcg_o:+.4f})  "
                f"Unpop: {unp_o:.3f} -> {unp_r:.3f} ({unp_r - unp_o:+.3f})"
            )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-size", type=int, default=DEFAULTS["sample_size"])
    p.add_argument("--sample-seed", type=int, default=DEFAULTS["sample_seed"])
    p.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--provider", default=DEFAULTS["provider"])
    p.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    p.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    p.add_argument(
        "--experiment-id",
        default=DEFAULTS["experiment_id"],
        help="Tag for this run (used in usage_log.jsonl).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show banner and a sample prompt without making any API calls.",
    )
    p.add_argument(
        "--no-confirm",
        action="store_true",
        help="Skip the 'press enter to continue' confirmation.",
    )
    args = p.parse_args()

    cfg = _build_config(args)

    # Quick scenario count: ML-1M has 3 dims -> 1 + 3 + 3 + 1 = 8 scenarios.
    # Compute exactly using the adapter to stay in sync with future schema changes.
    from src.datasets.factory import build_dataset

    adapter = build_dataset(cfg.dataset)
    n_scenarios = len(adapter.scenario_combinations())

    _print_banner(cfg, n_scenarios)

    if args.dry_run:
        _show_prompt_sample(cfg)
        print("\n[dry-run] No API calls were made.")
        return

    # Make sure the API key is set BEFORE building the adapter.
    import os

    if cfg.llm.provider in ("openai", "siliconflow", "together"):
        env_key = "OPENAI_API_KEY" if cfg.llm.provider == "openai" else "SILICONFLOW_API_KEY"
    elif cfg.llm.provider == "anthropic":
        env_key = "ANTHROPIC_API_KEY"
    else:
        env_key = "OPENAI_API_KEY"

    if not os.getenv(env_key):
        print(f"\nERROR: {env_key} is not set.")
        print("Copy .env.example to .env and fill in your key, then retry.")
        sys.exit(2)

    # Detect unfilled placeholder from .env.example.
    key_val = os.getenv(env_key, "")
    if "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" in key_val or key_val.startswith("sk-xxxxxx"):
        print(f"\nERROR: {env_key} still contains the .env.example placeholder.")
        print("Please edit .env and replace the placeholder with your real API key.")
        sys.exit(2)

    if not args.no_confirm:
        try:
            input("\nPress ENTER to start, Ctrl+C to abort... ")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    print(f"\nRunning... (cached prompts cost $0)")
    t0 = time.time()
    meta = run(cfg)
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s")
    _print_summary(meta)


if __name__ == "__main__":
    main()
