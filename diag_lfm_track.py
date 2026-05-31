"""Quick diagnostic for adapter loading and sampling under track mode."""
import sys
from collections import Counter

from src.datasets.factory import build_dataset


def main():
    print("[diag] building adapter...", flush=True)
    a = build_dataset("lastfm1k", granularity="track", min_track_listeners=3)
    print("[diag] adapter:", a, flush=True)
    print("[diag] domain:", a.domain(), "schema:", a.demographic_schema(), flush=True)
    print("[diag] loading users (3-5 min)...", flush=True)
    users = a.load()
    print(f"[diag] users passing min_interactions={a.min_interactions}: {len(users)}", flush=True)
    if not users:
        print("[diag] !!! ZERO users — min_interactions too strict or positive predicate broken")
        return
    print("[diag] splitting...", flush=True)
    a.split(seed=42, test_size=0.7)
    users = a.load()
    print(f"[diag] users with test_set: {sum(1 for u in users if u.test_set)} / {len(users)}", flush=True)
    print("[diag] sampling 30...", flush=True)
    s = a.stratified_sample(30, seed=42)
    print(f"[diag] sampled: {len(s)}", flush=True)
    if s:
        demo = Counter(tuple(sorted(u.demographics.items())) for u in s)
        for k, v in demo.most_common():
            print(f"  {k}: {v}", flush=True)
        print("[diag] first user history len:", len(s[0].history), "test_set len:", len(s[0].test_set), flush=True)
        print("[diag] first user history sample title:", s[0].history[0].title if s[0].history else None, flush=True)


if __name__ == "__main__":
    sys.exit(main())
