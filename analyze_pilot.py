"""Comprehensive analysis of the 100-user LFM-1K rerank pilot."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path("results/gpt-4.1-mini/lastfm1k")
df = pd.read_parquet(ROOT / "metrics" / "per_user.parquet")

# Per-scenario summary with deltas + dispersion
g = df.groupby("scenario").agg({
    "orig_hits": ["mean", "std", "min", "max"],
    "rer_hits": ["mean", "std", "min", "max"],
    "orig_ndcg_at_k": ["mean", "std"],
    "rer_ndcg_at_k": ["mean", "std"],
    "orig_mrr": "mean",
    "rer_mrr": "mean",
    "orig_difficulty_total": "mean",
    "rer_difficulty_total": "mean",
    "orig_hit_rate": "mean",
    "rer_hit_rate": "mean",
    "orig_precision": "mean",
    "rer_precision": "mean",
    "orig_recall": "mean",
    "rer_recall": "mean",
})

# Flatten columns
g.columns = [f"{a}_{b}" if b else a for a, b in g.columns]
g = g.round(4)

print("=" * 100)
print(f"100-user LFM-1K rerank analysis (n_users in summary = {df['user_id'].nunique()})")
print("=" * 100)

print("\n--- HIT COUNTS (orig_hits and rer_hits, per user) ---")
print(g[["orig_hits_mean", "orig_hits_std", "rer_hits_mean", "rer_hits_std",
        "orig_hits_min", "orig_hits_max"]].to_string())

print("\n--- TRADITIONAL RETRIEVAL METRICS ---")
cols = ["orig_precision_mean", "rer_precision_mean", "orig_recall_mean", "rer_recall_mean",
        "orig_ndcg_at_k_mean", "rer_ndcg_at_k_mean", "orig_mrr_mean", "rer_mrr_mean",
        "orig_hit_rate_mean", "rer_hit_rate_mean"]
print(g[cols].T.to_string())

print("\n--- BIAS-AWARE METRIC (difficulty) ---")
print(g[["orig_difficulty_total_mean", "rer_difficulty_total_mean"]].to_string())

# Compute deltas
print("\n--- DELTAS (rer - orig) ---")
deltas = pd.DataFrame({
    "Δ_hits": (g["rer_hits_mean"] - g["orig_hits_mean"]).round(3),
    "Δ_NDCG": (g["rer_ndcg_at_k_mean"] - g["orig_ndcg_at_k_mean"]).round(4),
    "Δ_MRR": (g["rer_mrr_mean"] - g["orig_mrr_mean"]).round(4),
    "Δ_difficulty": (g["rer_difficulty_total_mean"] - g["orig_difficulty_total_mean"]).round(3),
})
print(deltas.to_string())

# Significance tests on the LFM-1K data
from scipy import stats as sp
print("\n--- WILCOXON SIGNED-RANK TESTS (rer vs orig per user) ---")
for scn in df["scenario"].unique():
    sub = df[df["scenario"] == scn]
    if scn == "neutral":
        continue
    print(f"\n[{scn}]  n = {len(sub)}")
    for metric in ["ndcg_at_k", "mrr", "hits", "difficulty_total"]:
        ocol, rcol = f"orig_{metric}", f"rer_{metric}"
        if ocol not in sub.columns or rcol not in sub.columns:
            continue
        diff = sub[rcol] - sub[ocol]
        if not diff.any():
            print(f"  {metric:20s}: all-zero diffs (skip)")
            continue
        try:
            res = sp.wilcoxon(sub[rcol], sub[ocol], alternative="two-sided")
            sig = "***" if res.pvalue < 0.001 else "**" if res.pvalue < 0.01 else "*" if res.pvalue < 0.05 else "ns"
            print(f"  {metric:20s}: median Δ = {diff.median():+.4f}  p = {res.pvalue:.4f}  [{sig}]")
        except Exception as e:
            print(f"  {metric:20s}: error {e}")

# Distribution: how often does rerank help vs hurt?
print("\n--- DELTA DISTRIBUTION (rer better / equal / worse counts) ---")
for scn in [s for s in df["scenario"].unique() if s != "neutral"]:
    sub = df[df["scenario"] == scn]
    print(f"\n[{scn}]")
    for metric in ["ndcg_at_k", "mrr", "hits", "difficulty_total"]:
        diff = sub[f"rer_{metric}"] - sub[f"orig_{metric}"]
        better = (diff > 0).sum()
        worse = (diff < 0).sum()
        equal = (diff == 0).sum()
        print(f"  {metric:20s}: rer > orig: {better}  | == : {equal}  | rer < orig: {worse}")

# Scope check on alpha distribution
print("\n--- DYNAMIC ALPHA DISTRIBUTION ---")
if "alpha" in df.columns:
    a = df["alpha"]
    print(f"  alpha  mean={a.mean():.3f}  std={a.std():.3f}  min={a.min():.3f}  max={a.max():.3f}")
    print(f"  alpha histogram bins:")
    for b in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        count = ((a >= b - 0.05) & (a < b + 0.05)).sum()
        print(f"    ~{b:.1f}: {count}")
