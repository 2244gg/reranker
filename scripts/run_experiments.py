"""CLI to run one or more experiments from YAML configs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from src.experiment.config import ExperimentConfig, load_all_configs  # noqa: E402
from src.experiment.runner import run  # noqa: E402


def _matches_filter(cfg: ExperimentConfig, filt: str) -> bool:
    """Match comma-separated key=value pairs against the config."""
    if not filt:
        return True
    for piece in filt.split(","):
        if "=" not in piece:
            continue
        k, v = piece.split("=", 1)
        k, v = k.strip(), v.strip()
        if k == "model" and v.lower() not in cfg.llm.model.lower():
            return False
        if k == "dataset" and v.lower() != cfg.dataset.lower():
            return False
        if k == "experiment_id" and v != cfg.experiment_id:
            return False
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-dir", default="configs", help="Directory or single YAML")
    p.add_argument("--filter", default="", help="key=value[,key=value] (model/dataset/experiment_id)")
    p.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Override sample_size for all selected configs (handy for pilots)",
    )
    p.add_argument("--dry-run", action="store_true", help="List selected configs and exit")
    args = p.parse_args()

    cfgs = load_all_configs(args.config_dir)
    selected = [c for c in cfgs if _matches_filter(c, args.filter)]
    if not selected:
        print("No configs matched the filter.")
        sys.exit(2)

    print(f"Selected {len(selected)} config(s):")
    for c in selected:
        print(f"  - {c.experiment_id}  model={c.llm.model}  dataset={c.dataset}  n={c.sample_size}")

    if args.dry_run:
        return

    for c in selected:
        if args.sample_size is not None:
            c.sample_size = args.sample_size
        print(f"\n=== Running {c.experiment_id} ===")
        meta = run(c)
        print(
            f"Done {c.experiment_id}: "
            f"users={meta.get('sample_size')} "
            f"calls={meta.get('total_api_calls')} "
            f"tokens={meta.get('total_prompt_tokens')}+{meta.get('total_completion_tokens')}"
        )


if __name__ == "__main__":
    main()
