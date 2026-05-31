"""Investigate empty true_preferred for user_000282."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.datasets.factory import build_dataset
from src.experiment.metrics import title_key, titles_match

# Load raw to see what LLM recommended
d = json.load(open("results/gpt-4.1-mini/lastfm1k/raw/recommendation_results_full.json", encoding="utf-8"))
u = d["user_000282"]
print("user_000282 demographics:", u["user_info"])

print("\n--- LLM recs for neutral (30 items, dedup):")
recs = u["results"]["neutral"]["recommendations"]
seen = set()
deduped = [r for r in recs if not (r in seen or seen.add(r))]
for i, r in enumerate(deduped, 1):
    print(f"  {i}. {r}")
print(f"  {len(recs)} total, {len(deduped)} unique")

# Load test set from adapter
print("\n--- Loading user_000282's actual test set...")
adapter = build_dataset("lastfm1k")
adapter.split(seed=42, test_size=0.7)
all_users = adapter.load()
target = next((x for x in all_users if x.user_id == "user_000282"), None)
if target is None:
    print("user_000282 not found - maybe didn't pass min_interactions filter?")
else:
    print(f"  test_set size: {len(target.test_set)}")
    print(f"  history (top 10 by playcount):")
    for it in target.history[:10]:
        print(f"    - {it.title}  (rating={it.rating})")
    print(f"\n  test_set top 20 (sorted by playcount):")
    test_sorted = sorted(target.test_set, key=lambda x: x.rating or 0, reverse=True)
    for it in test_sorted[:20]:
        print(f"    - {it.title}  (rating={it.rating})")

    # Check overlap
    print("\n--- Overlap check (LLM recs vs test set):")
    test_keys = {title_key(it.title) for it in target.test_set}
    matches = []
    for r in deduped:
        if any(titles_match(r, it.title) for it in target.test_set):
            matches.append(r)
    print(f"  matches: {matches}")
