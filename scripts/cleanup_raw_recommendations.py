"""Post-hoc cleanup of an existing raw recommendation JSON.

This script does NOT call the LLM. It re-processes an already-saved raw JSON
to fix three issues that surface in real LLM outputs:

  1. Duplicate items (same artist / movie / book recommended multiple times
     in the same list — common when the LLM's coverage thins out near the
     end of a long requested list).

  2. History leakage: the LLM frequently echoes back items already in the
     user's history, despite an explicit "Avoid items already in the
     history above" instruction in the prompt.

  3. Stale ``true_preferred_movies`` field name — overwrites with the
     domain-neutral ``true_preferred_items`` (alias kept for back-compat),
     and recomputes the matches against the user's test set after the list
     has been deduplicated and history-filtered.

The cleanup loads the dataset adapter to access each user's history and
test set, then re-derives the cleaned recommendations and hit list.

Usage:
    python scripts/cleanup_raw_recommendations.py \\
        --raw-json results/gpt-4.1-mini/lastfm1k/raw/recommendation_results_full.json \\
        --dataset lastfm1k

By default writes ``recommendation_results_full_cleaned.json`` next to the
input. Use ``--in-place`` to overwrite the original.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiment.metrics import title_key, titles_match  # noqa: E402


def _dedup_and_filter(recs, history_keys):
    """Drop duplicates (by title_key) and items appearing in history_keys."""
    out = []
    seen = set()
    for r in recs:
        k = title_key(r)
        if not k or k in seen or k in history_keys:
            continue
        seen.add(k)
        out.append(r)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-json", required=True, help="Path to the raw recommendation_results_full.json")
    p.add_argument("--dataset", required=True, help="Dataset short name (lastfm1k, ml1m, bookcrossing, ...)")
    p.add_argument(
        "--history-limit",
        type=int,
        default=10,
        help="How many top-rated history items to consider as 'already seen'. Default 10.",
    )
    p.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Must match the seed used at generation time (default 42).",
    )
    p.add_argument(
        "--test-split",
        type=float,
        default=0.7,
        help="Must match the test_split used at generation time (default 0.7).",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input JSON. Default writes a *_cleaned.json sibling.",
    )
    p.add_argument("--output", default=None, help="Explicit output path.")
    args = p.parse_args()

    raw_path = Path(args.raw_json)
    if not raw_path.exists():
        raise SystemExit(f"Raw JSON not found: {raw_path}")

    if args.output:
        out_path = Path(args.output)
    elif args.in_place:
        out_path = raw_path
    else:
        out_path = raw_path.with_name(raw_path.stem + "_cleaned.json")

    print(f"[cleanup] input  : {raw_path}")
    print(f"[cleanup] output : {out_path}")
    print(f"[cleanup] loading dataset '{args.dataset}' (this can take 30-60s for LFM)...")

    from src.datasets.factory import build_dataset

    extra = {}
    raw = json.loads(raw_path.read_text(encoding="utf-8"))

    # Some datasets (BookCrossing) need the implicit flag to match generation.
    # Pull it from the input file's first user's recommendations metadata if
    # available; otherwise fall back to defaults.
    if args.dataset == "bookcrossing":
        # Heuristic: re-derive the eligible user set with implicit=True so users
        # not present in implicit-mode won't break. If the dataset was generated
        # with implicit=False, we can still find users — they're a subset.
        extra["implicit"] = True

    adapter = build_dataset(args.dataset, **extra)
    adapter.split(seed=args.sample_seed, test_size=args.test_split)
    user_index = {u.user_id: u for u in adapter.load()}

    n_users = len(raw)
    n_scn = 0
    n_dups_removed = 0
    n_hist_removed = 0
    n_users_missing = 0

    for uid, payload in raw.items():
        adapter_user = user_index.get(uid)
        if adapter_user is None:
            n_users_missing += 1
            continue

        history_keys = {
            title_key(it.title)
            for it in adapter_user.history[: args.history_limit]
            if title_key(it.title)
        }
        test_titles = [it.title for it in adapter_user.test_set]

        for label, scn in (payload.get("results") or {}).items():
            n_scn += 1
            recs = scn.get("recommendations") or []
            before = len(recs)
            cleaned = _dedup_and_filter(recs, history_keys)
            removed = before - len(cleaned)
            seen_count = sum(
                1 for r in recs if title_key(r) in history_keys
            )
            n_hist_removed += seen_count
            n_dups_removed += (removed - seen_count)
            scn["recommendations"] = cleaned
            # Recompute true_preferred against the cleaned list.
            true_pref = [t for t in cleaned if any(titles_match(t, tt) for tt in test_titles)]
            scn["true_preferred_items"] = true_pref
            scn["true_preferred_movies"] = true_pref  # back-compat alias

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    print(f"  users in raw JSON         : {n_users}")
    print(f"  users not found in adapter: {n_users_missing}")
    print(f"  scenarios processed       : {n_scn}")
    print(f"  duplicate items removed   : {n_dups_removed}")
    print(f"  history-leak items removed: {n_hist_removed}")
    print(f"  cleaned file              : {out_path}")
    print()
    print("Next: re-run rerank/metrics on the cleaned file:")
    if args.in_place:
        print(f"  python scripts/rerank_only.py --results-dir {raw_path.parents[1]}")
    else:
        # rerank_only currently expects raw/recommendation_results_full.json,
        # so for non-in-place we suggest swapping it in.
        print("  Either run with --in-place, or move the *_cleaned.json over")
        print("  raw/recommendation_results_full.json before invoking rerank_only.")


if __name__ == "__main__":
    main()
