"""CLI: aggregate per-user metrics and run significance tests."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.stats.runner import run_all  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing <model>/<dataset>/metrics/per_user.parquet",
    )
    p.add_argument(
        "--output",
        default="reports/significance_tests.json",
        help="Where to write the significance JSON",
    )
    p.add_argument("--n-boot", type=int, default=10000)
    args = p.parse_args()

    out = run_all(args.results_root, args.output, n_boot=args.n_boot)
    print(f"Wrote {len(out['results'])} cells to {args.output}")


if __name__ == "__main__":
    main()
