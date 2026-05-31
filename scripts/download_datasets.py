"""Helper script to obtain external datasets used in this project.

ML-1M is already shipped under ``ml-1m/``. The other three datasets have
licenses that require manual download from official sources. This script
prints exact instructions and verifies expected paths.
"""
from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"


_INSTRUCTIONS = {
    "ml1m": (
        "ML-1M is already present at ml-1m/ml-1m/.\n"
        "Official source: https://files.grouplens.org/datasets/movielens/ml-1m.zip\n"
        "Citation: F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens "
        "Datasets: History and Context. ACM TiiS 5, 4 (2015)."
    ),
    "bookcrossing": (
        "BookCrossing requires manual download (free for research):\n"
        "  1. Visit: http://www2.informatik.uni-freiburg.de/~cziegler/BX/\n"
        "  2. Download `BX-CSV-Dump.zip`.\n"
        "  3. Extract `BX-Users.csv`, `BX-Books.csv`, `BX-Book-Ratings.csv`\n"
        f"     into {RAW_DIR / 'bookcrossing'}\n"
        "Citation: Ziegler, McNee, Konstan, Lausen. 2005. Improving "
        "Recommendation Lists Through Topic Diversification. WWW'05."
    ),
    "lfm": (
        "LFM-2b requires manual download (research-only EULA):\n"
        "  1. Apply at: http://www.cp.jku.at/datasets/LFM-2b/\n"
        "  2. After approval, download the user demographics + listening events.\n"
        "  3. Place `users.tsv`, `user_artist_playcount.tsv`, `artists.tsv`\n"
        f"     into {RAW_DIR / 'lfm-2b'}\n"
        "Citation: Schedl, Brost, Liem, Knees, Pampalk, Anelli, Di Noia, Lex. 2022. "
        "LFM-2b: A Dataset of Enriched Music Listening Events for Recommender Systems "
        "Research and Fairness Analysis. CHIIR'22."
    ),
    "tenrec": (
        "Tenrec requires manual download:\n"
        "  1. Visit: https://github.com/yuangh-x/2022-NIPS-Tenrec\n"
        "  2. Follow instructions for downloading QK-video or QB-video splits.\n"
        "  3. Place `user_features.csv`, `video_impressions.csv`, "
        "`items.csv` (if available)\n"
        f"     into {RAW_DIR / 'tenrec'}\n"
        "Citation: Yuan et al. 2022. Tenrec: A Large-scale Multipurpose "
        "Benchmark Dataset for Recommender Systems. NeurIPS Datasets & Benchmarks."
    ),
}


def _verify(dataset: str) -> bool:
    path = RAW_DIR / dataset
    if dataset == "ml1m":
        candidate = PROJECT_ROOT / "ml-1m" / "ml-1m" / "users.dat"
        return candidate.exists()
    if dataset == "bookcrossing":
        return (path / "BX-Users.csv").exists()
    if dataset == "lfm":
        return (path / "users.tsv").exists()
    if dataset == "tenrec":
        return (path / "user_features.csv").exists()
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show download / verification instructions for datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=list(_INSTRUCTIONS.keys()) + ["all"],
        default="all",
    )
    args = parser.parse_args()

    targets = list(_INSTRUCTIONS.keys()) if args.dataset == "all" else [args.dataset]
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    rc = 0
    for ds in targets:
        print(f"\n=== {ds} ===")
        print(_INSTRUCTIONS[ds])
        ok = _verify(ds)
        if ok:
            print(f"\nStatus: OK (files detected for '{ds}').")
        else:
            print(f"\nStatus: MISSING. Please follow the instructions above.")
            rc = 2
    print("\nSee data/README.md for licenses and citations.")
    sys.exit(rc)


if __name__ == "__main__":
    main()
