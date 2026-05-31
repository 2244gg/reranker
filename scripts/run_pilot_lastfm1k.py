"""Single-file minimal pilot for LastFM-1K + GPT-4.1-mini.

Defaults: 10 users x 4 scenarios (neutral + gender + age + gender_age) =
40 LLM calls. Estimated cost on gpt-4.1-mini ~ $0.03.

Demographics: gender + age_group only (2 dims).

Usage:
    python scripts/run_pilot_lastfm1k.py
    python scripts/run_pilot_lastfm1k.py --dry-run
    python scripts/run_pilot_lastfm1k.py --sample-size 50
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

from src.experiment.config import (  # noqa: E402
    ExperimentConfig,
    LLMConfig,
    RerankConfig,
)
from src.experiment.prompt_builder import PromptBuilder  # noqa: E402
from src.experiment.runner import run  # noqa: E402


DEFAULTS = dict(
    experiment_id="pilot_gpt41mini_lastfm1k",
    dataset="lastfm1k",
    sample_size=10,
    sample_seed=42,
    test_split=0.7,
    top_k=20,                  # final list length M
    candidate_pool_size=30,    # LLM + rerank candidate pool size (M + N)
    history_limit=10,
    rank_weight=0.2,
    output_root="results",
    cache_dir="data/cache",
    model="gpt-4.1-mini",
    provider="openai",
    temperature=0.7,
    max_tokens=1500,           # bigger pool needs more tokens
)


def _build_config(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=args.experiment_id,
        dataset=DEFAULTS["dataset"],
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        test_split=DEFAULTS["test_split"],
        top_k=args.top_k,
        candidate_pool_size=args.candidate_pool_size,
        history_limit=DEFAULTS["history_limit"],
        rank_weight=DEFAULTS["rank_weight"],
        popularity_csv=None,
        output_root=DEFAULTS["output_root"],
        cache_dir=DEFAULTS["cache_dir"],
        scenarios=None,
        llm=LLMConfig(
            provider=args.provider,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            system_prompt="You are a professional music recommendation system.",
        ),
        rerank=RerankConfig(),
    )


def _print_banner(cfg: ExperimentConfig, n_scenarios: int) -> None:
    candidate_n = cfg.candidate_pool_size or cfg.top_k
    total_calls = cfg.sample_size * n_scenarios
    est_in = total_calls * (600 + 200 * (candidate_n - 20) // 10)
    est_out = total_calls * (300 + 100 * (candidate_n - 20) // 10)
    est_cost = est_in / 1e6 * 0.40 + est_out / 1e6 * 1.60

    print("=" * 60)
    print("LASTFM-1K PILOT")
    print("=" * 60)
    print(f"  experiment_id   : {cfg.experiment_id}")
    print(f"  dataset         : {cfg.dataset}  (gender + age_group, 2 dims)")
    print(f"  model           : {cfg.llm.model}  (provider={cfg.llm.provider})")
    print(f"  sample size     : {cfg.sample_size} users")
    print(f"  scenarios       : {n_scenarios}")
    print(f"  candidate pool  : {candidate_n} (M+N) -> truncate to top_k={cfg.top_k}")
    print(f"  total calls     : {total_calls}")
    print(f"  est tokens      : ~{est_in:,} in / ~{est_out:,} out")
    print(f"  est cost        : ${est_cost:.3f} (gpt-4.1-mini, may vary)")
    print(f"  cache           : {cfg.cache_dir}/llm_cache.sqlite")
    print("=" * 60)


def _show_prompt_sample(cfg: ExperimentConfig) -> None:
    from src.datasets.factory import build_dataset

    candidate_n = cfg.candidate_pool_size or cfg.top_k
    adapter = build_dataset(cfg.dataset)
    print("\nLoading LastFM-1K (this can take 1-3 min on first load: 19M event rows)...")
    t0 = time.time()
    adapter.split(seed=cfg.sample_seed, test_size=cfg.test_split)
    sample = adapter.stratified_sample(1, seed=cfg.sample_seed)
    print(f"  loaded in {time.time() - t0:.1f}s")
    if not sample:
        print("[no users to sample]")
        return
    user = sample[0]
    pb = PromptBuilder(domain=adapter.domain(), history_limit=cfg.history_limit)
    print("\n--- Sample user ---")
    print(f"  user_id       : {user.user_id}")
    print(f"  demographics  : {user.demographics}")
    print(f"  top 5 artists by playcount:")
    for it in user.history[:5]:
        print(f"    - {it.title}  (playcount={int(it.rating or 0)})")

    print(f"\n--- Sample prompt: cross scenario (gender + age), asking for {candidate_n} items ---")
    rendered = pb.render(
        user, scenario=("gender", "age_group"), movie_count=candidate_n
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
    safe_model = meta.get("model").replace("/", "_")
    print(f"  output root   : results/{safe_model}/{meta.get('dataset')}")

    aggregate_path = (
        Path("results") / safe_model / meta.get("dataset") / "metrics" / "aggregate.json"
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
                f"    {scn:>20}  "
                f"NDCG: {ndcg_o:.4f} -> {ndcg_r:.4f} ({ndcg_r - ndcg_o:+.4f})  "
                f"Unpop: {unp_o:.3f} -> {unp_r:.3f} ({unp_r - unp_o:+.3f})"
            )
    print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-size", type=int, default=DEFAULTS["sample_size"])
    p.add_argument("--sample-seed", type=int, default=DEFAULTS["sample_seed"])
    p.add_argument("--top-k", type=int, default=DEFAULTS["top_k"])
    p.add_argument(
        "--candidate-pool-size",
        type=int,
        default=DEFAULTS["candidate_pool_size"],
        help="Number of items the LLM is asked to produce; rerank picks "
        "top_k from this pool. Must be >= top_k.",
    )
    p.add_argument("--model", default=DEFAULTS["model"])
    p.add_argument("--provider", default=DEFAULTS["provider"])
    p.add_argument("--temperature", type=float, default=DEFAULTS["temperature"])
    p.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    p.add_argument(
        "--experiment-id",
        default=DEFAULTS["experiment_id"],
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--no-confirm", action="store_true")
    args = p.parse_args()

    cfg = _build_config(args)

    from src.datasets.factory import build_dataset

    adapter = build_dataset(cfg.dataset)
    n_scenarios = len(adapter.scenario_combinations())

    _print_banner(cfg, n_scenarios)

    if args.dry_run:
        _show_prompt_sample(cfg)
        print("\n[dry-run] No API calls were made.")
        return

    import os

    if cfg.llm.provider in ("openai", "siliconflow", "together"):
        env_key = (
            "OPENAI_API_KEY" if cfg.llm.provider == "openai" else "SILICONFLOW_API_KEY"
        )
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
        try:
            input("\nPress ENTER to start, Ctrl+C to abort... ")
        except KeyboardInterrupt:
            print("\nAborted.")
            sys.exit(0)

    print("\nRunning... (cached prompts cost $0)")
    t0 = time.time()
    meta = run(cfg)
    elapsed = time.time() - t0
    print(f"Elapsed: {elapsed:.1f}s")
    _print_summary(meta)


if __name__ == "__main__":
    main()
