"""Detailed statistical analysis of the v2 600-user main experiment.

Outputs paired-comparison stats, by-demographic breakdowns, and CSV
summaries to feed back into the report markdown.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean, stdev

import pandas as pd

RESULTS = Path("results_v2/gpt-4.1-mini/lastfm1k")


def paired_t(orig: pd.Series, rer: pd.Series):
    """Simple paired t-statistic + 95% CI without scipy.

    Returns (mean_diff, t, p_two_sided_approx, ci_lo, ci_hi).
    """
    diff = (rer - orig).dropna()
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    md = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / math.sqrt(n)
    t = md / se if se > 0 else float("nan")
    # Approximate two-sided p using the normal-approximation tail
    # (n=600 makes the t-distribution practically normal).
    if math.isnan(t):
        p = float("nan")
    else:
        from math import erf, sqrt
        z = abs(t)
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(z / sqrt(2.0))))
    ci_half = 1.96 * se
    return md, t, p, md - ci_half, md + ci_half


def main() -> None:
    df = pd.read_parquet(RESULTS / "metrics" / "per_user.parquet")
    raw = json.loads((RESULTS / "raw" / "recommendation_results_full.json").read_text(encoding="utf-8"))
    print(f"per_user.parquet shape: {df.shape}")
    print(f"columns: {list(df.columns)}")
    print(f"scenarios: {df['scenario'].unique().tolist()}")

    # Attach demographics for sub-group analysis.
    demo = {uid: blk["user_info"] for uid, blk in raw.items()}
    df["gender"]    = df["user_id"].map(lambda u: demo.get(u, {}).get("gender", "unknown"))
    df["age_group"] = df["user_id"].map(lambda u: demo.get(u, {}).get("age_group", "unknown"))

    metric_pairs = [
        ("Precision@20",   "orig_precision",            "rer_precision"),
        ("Recall@20",      "orig_recall",               "rer_recall"),
        ("HitRate@20",     "orig_hit_rate",             "rer_hit_rate"),
        ("MRR",            "orig_mrr",                  "rer_mrr"),
        ("NDCG@20",        "orig_ndcg_at_k",            "rer_ndcg_at_k"),
        ("Hits",           "orig_hits",                 "rer_hits"),
        ("APLT (slice)",   "orig_aplt",                 "rer_aplt"),
        ("APLT (neutral)", "orig_aplt_neutral",         "rer_aplt_neutral"),
        ("UnpopAvg",       "orig_unpopularity_average", "rer_unpopularity_average"),
        ("DiffAvg",        "orig_difficulty_average",   "rer_difficulty_average"),
    ]

    print("\n\n================ Paired comparisons by scenario ================")
    rows = []
    for sc in ["neutral", "gender", "age_group", "gender+age_group"]:
        sub = df[df["scenario"] == sc]
        print(f"\n--- scenario = {sc}  (n_users = {len(sub)}) ---")
        print(f"  {'metric':<18} {'orig':>9} {'rer':>9} {'delta':>9} {'95% CI':>22} {'t':>8} {'p':>10}")
        print(f"  {'-'*18} {'-'*9} {'-'*9} {'-'*9} {'-'*22} {'-'*8} {'-'*10}")
        for label, ok, rk in metric_pairs:
            if ok not in sub.columns or rk not in sub.columns:
                continue
            md, t, p, lo, hi = paired_t(sub[ok], sub[rk])
            ov = float(sub[ok].mean()) if ok in sub else float("nan")
            rv = float(sub[rk].mean()) if rk in sub else float("nan")
            ci = f"[{lo:>+.4f}, {hi:>+.4f}]"
            p_s = f"{p:.2e}" if not math.isnan(p) else "nan"
            print(f"  {label:<18} {ov:>9.4f} {rv:>9.4f} {md:>+9.4f} {ci:>22} {t:>+8.2f} {p_s:>10}")
            rows.append({"scenario": sc, "metric": label, "orig": ov, "rer": rv,
                         "delta": md, "ci_lo": lo, "ci_hi": hi, "t": t, "p": p,
                         "n": len(sub)})

    # Save flat table for the markdown report.
    out = pd.DataFrame(rows)
    out_path = RESULTS / "metrics" / "paired_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")

    # ------------- Demographic sub-group breakdown -------------
    print("\n\n================ APLT lift by demographic sub-group ================\n")
    for sc in ["gender", "age_group", "gender+age_group"]:
        sub = df[df["scenario"] == sc]
        if sc == "gender":
            grouper = "gender"
        elif sc == "age_group":
            grouper = "age_group"
        else:
            sub = sub.copy()
            sub["gxa"] = sub["gender"] + " / " + sub["age_group"]
            grouper = "gxa"
        print(f"\n--- scenario={sc}, group by {grouper} ---")
        print(f"  {grouper:<28} {'n':>5} {'orig APLT':>10} {'rer APLT':>10} {'delta':>10}")
        for g, gsub in sub.groupby(grouper):
            ov = gsub["orig_aplt"].mean()
            rv = gsub["rer_aplt"].mean()
            print(f"  {g:<28} {len(gsub):>5} {ov:>10.4f} {rv:>10.4f} {rv-ov:>+10.4f}")

    # ------------- Per-user variability -------------
    print("\n\n================ Per-user variability of APLT lift ================\n")
    for sc in ["gender", "age_group", "gender+age_group"]:
        sub = df[df["scenario"] == sc]
        diff = (sub["rer_aplt"] - sub["orig_aplt"]).dropna()
        positive = (diff > 0).sum()
        zero = (diff == 0).sum()
        negative = (diff < 0).sum()
        print(f"\n{sc}  (n={len(sub)})")
        print(f"  users with APLT improved : {positive:>4} ({positive*100/len(sub):.1f}%)")
        print(f"  users with APLT unchanged: {zero:>4} ({zero*100/len(sub):.1f}%)")
        print(f"  users with APLT regressed: {negative:>4} ({negative*100/len(sub):.1f}%)")
        print(f"  APLT delta:  mean={diff.mean():+.4f}  std={diff.std():.4f}  "
              f"min={diff.min():+.4f}  max={diff.max():+.4f}")


if __name__ == "__main__":
    main()
