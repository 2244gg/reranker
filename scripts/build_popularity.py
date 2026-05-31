"""Build group-conditional popularity CSV for a given dataset.

Usage:
    python scripts/build_popularity.py --dataset ml1m
    python scripts/build_popularity.py --dataset bookcrossing --output data/processed/bookcrossing/items_with_popularity.csv

Track-level LFM-1K (song-aware recommendation):
    python scripts/build_popularity.py --dataset lastfm1k --granularity track \
        --output data/processed/lastfm1k_track/items_with_popularity.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.factory import build_dataset  # noqa: E402
from src.popularity.slicer import PopularitySlicer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ml1m", "ml1m-legacy", "bookcrossing", "lfm", "lastfm1k", "tenrec"],
    )
    parser.add_argument("--output", default=None, help="Output CSV path")
    parser.add_argument(
        "--sparse-threshold",
        type=int,
        default=30,
        help="Slices with fewer users than this will be blended with neutral.",
    )
    parser.add_argument(
        "--max-cross-dims",
        type=int,
        default=None,
        help="Limit cross slice dimensionality (default: all).",
    )
    parser.add_argument(
        "--no-positive-only",
        action="store_true",
        help="Count all interactions, not only positive feedback.",
    )
    parser.add_argument(
        "--granularity",
        choices=["artist", "track"],
        default=None,
        help="(lastfm1k only) item granularity. Default = adapter default = artist.",
    )
    parser.add_argument(
        "--include-country",
        action="store_true",
        help="(lastfm1k / lfm) include country as a third demographic dim.",
    )
    parser.add_argument(
        "--min-track-listeners",
        type=int,
        default=1,
        help="(lastfm1k --granularity track) drop tracks listened by fewer "
             "than this many distinct users to combat extreme sparsity.",
    )
    args = parser.parse_args()

    # Auto-pick output suffix when running track-level LFM-1K so we don't
    # silently overwrite the legacy artist-level CSV.
    default_subdir = args.dataset
    if args.dataset == "lastfm1k" and args.granularity == "track":
        default_subdir = "lastfm1k_track"
    output = args.output or f"data/processed/{default_subdir}/items_with_popularity.csv"
    output_path = Path(output)

    adapter_kwargs = {}
    if args.dataset == "lastfm1k":
        if args.granularity:
            adapter_kwargs["granularity"] = args.granularity
        if args.include_country:
            adapter_kwargs["include_country"] = True
        if args.min_track_listeners > 1:
            adapter_kwargs["min_track_listeners"] = args.min_track_listeners

    print(f"[build_popularity] dataset={args.dataset} kwargs={adapter_kwargs}")
    adapter = build_dataset(args.dataset, **adapter_kwargs)
    slicer = PopularitySlicer(
        adapter,
        max_cross_dims=args.max_cross_dims,
        sparse_threshold=args.sparse_threshold,
        positive_only=not args.no_positive_only,
    )
    out = slicer.to_csv(output_path)
    print(f"[build_popularity] wrote {out}")


if __name__ == "__main__":
    main()
