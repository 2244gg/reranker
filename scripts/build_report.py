"""Build all aggregate reports (cross-table, by-scenario, Pareto, heatmap)."""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.reporting.by_scenario import build_by_scenario_reports  # noqa: E402
from src.reporting.cross_table import build_cross_table  # noqa: E402
from src.reporting.heatmap import build_intersectional_heatmap  # noqa: E402
from src.reporting.pareto import build_pareto  # noqa: E402


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def _prepend_footer(path: Path, source: Path) -> None:
    if not path.exists():
        return
    if path.suffix not in (".md", ".csv"):
        return
    header = (
        f"# Generated at: {dt.datetime.now(dt.timezone.utc).isoformat()}\n"
        f"# Source: {source}\n"
        f"# Commit: {_git_commit()}\n"
        f"# (file follows)\n"
    )
    if path.suffix == ".csv":
        comment_prefix = "# "  # most CSV readers skip lines starting with #.
    else:
        comment_prefix = ""

    original = path.read_text(encoding="utf-8")
    if original.lstrip().startswith("# Generated at"):
        return  # already has footer
    path.write_text(comment_prefix + header.replace("\n", "\n" + comment_prefix).rstrip(comment_prefix) + original, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--significance",
        default="reports/significance_tests.json",
    )
    p.add_argument("--results-root", default="results")
    p.add_argument("--output-dir", default="reports")
    p.add_argument("--cross-table", action="store_true")
    p.add_argument("--by-scenario", action="store_true")
    p.add_argument("--pareto", action="store_true")
    p.add_argument("--heatmap", action="store_true")
    p.add_argument("--all", action="store_true", help="Run all reports")
    p.add_argument(
        "--heatmap-datasets",
        nargs="*",
        default=["ml1m", "lfm"],
        help="Datasets to render intersectional heatmaps for (must have >=3 dims).",
    )
    args = p.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sig = Path(args.significance)

    do_all = args.all or not (
        args.cross_table or args.by_scenario or args.pareto or args.heatmap
    )

    if do_all or args.cross_table:
        path = build_cross_table(sig, out_dir / "cross_model_dataset_summary.csv")
        _prepend_footer(path, sig)
        print(f"  + {path}")
    if do_all or args.by_scenario:
        for path in build_by_scenario_reports(sig, out_dir):
            _prepend_footer(path, sig)
            print(f"  + {path}")
    if do_all or args.pareto:
        path = build_pareto(sig, out_dir / "pareto_frontier.png")
        print(f"  + {path}")
    if do_all or args.heatmap:
        for ds in args.heatmap_datasets:
            path = build_intersectional_heatmap(
                args.results_root, ds, out_dir / f"intersectional_heatmap_{ds}.png"
            )
            print(f"  + {path}")


if __name__ == "__main__":
    main()
