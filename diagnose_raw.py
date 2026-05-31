"""Quick diagnostic of the raw LLM output JSON."""
import json
import sys
from pathlib import Path

PATH = sys.argv[1] if len(sys.argv) > 1 else "results/gpt-4.1-mini/lastfm1k/raw/recommendation_results_full.json"

d = json.load(open(PATH, encoding="utf-8"))
print(f"Total users: {len(d)}")

total_scn = 0
empty_pref = 0
dup_count = 0
total_dups = 0
hist_intersect = 0
short_recs = 0
recs_lens = []

per_user_test_size = []

for uid, payload in d.items():
    test_titles = []
    for label, scn in payload.get("results", {}).items():
        total_scn += 1
        recs = scn.get("recommendations") or []
        recs_lens.append(len(recs))
        pref = scn.get("true_preferred_movies") or []
        if not pref:
            empty_pref += 1
        if len(set(recs)) < len(recs):
            dup_count += 1
            total_dups += len(recs) - len(set(recs))
        if len(recs) < 25:
            short_recs += 1

print(f"\n--- Summary ---")
print(f"  scenarios total          : {total_scn}")
print(f"  scenarios w/ empty pref  : {empty_pref}  ({empty_pref/total_scn*100:.0f}%)")
print(f"  scenarios w/ duplicates  : {dup_count}  ({dup_count/total_scn*100:.0f}%)")
print(f"  total duplicate items    : {total_dups}")
print(f"  scenarios w/ <25 recs    : {short_recs}")
print(f"  rec len: min={min(recs_lens)} max={max(recs_lens)} avg={sum(recs_lens)/len(recs_lens):.1f}")

# Inspect one user
print("\n--- Sample user user_000282 neutral ---")
u = d.get("user_000282", {})
neu = u.get("results", {}).get("neutral", {})
recs = neu.get("recommendations") or []
print(f"  {len(recs)} recs, {len(set(recs))} unique")
print(f"  true_preferred_movies: {neu.get('true_preferred_movies') or 'EMPTY'}")
print(f"  pop_col: {neu.get('popularity_column_used')}")
print(f"  history_pop_score: {neu.get('history_popularity_score')}")
