#!/usr/bin/env python3
import argparse
import csv
import glob
import json
import os
import re
from typing import Any, Dict, List, Tuple

SCENARIOS = ("neutral", "gender", "age", "cross")


def normalize_title(title: str) -> str:
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", title.strip())
    if re.search(r",\s*The$", t, re.IGNORECASE):
        t = "The " + re.sub(r",\s*The$", "", t, flags=re.IGNORECASE)
    elif re.search(r",\s*A$", t, re.IGNORECASE):
        t = "A " + re.sub(r",\s*A$", "", t, flags=re.IGNORECASE)
    elif re.search(r",\s*An$", t, re.IGNORECASE):
        t = "An " + re.sub(r",\s*An$", "", t, flags=re.IGNORECASE)
    return t


def load_movie_popularity(csv_path: str) -> Dict[str, Dict[str, float]]:
    pop: Dict[str, Dict[str, float]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = normalize_title(str(row["Title"]))
            pop[title] = {
                "neutral_all": float(row["neutral_All"]),
                "gender_M": float(row["gender_M"]),
                "gender_F": float(row["gender_F"]),
                "age_Youth": float(row["age_Youth"]),
                "age_Middle": float(row["age_Middle"]),
                "age_Senior": float(row["age_Senior"]),
                "cross_M_Youth": float(row["cross_M_Youth"]),
                "cross_F_Youth": float(row["cross_F_Youth"]),
                "cross_M_Middle": float(row["cross_M_Middle"]),
                "cross_F_Middle": float(row["cross_F_Middle"]),
                "cross_M_Senior": float(row["cross_M_Senior"]),
                "cross_F_Senior": float(row["cross_F_Senior"]),
            }
    return pop


def get_popularity_column(gender: str, age_group: str, rec_type: str) -> str:
    if rec_type == "neutral":
        return "neutral_all"
    if rec_type == "gender":
        return f"gender_{gender[0].upper()}"
    age_map = {"young": "Youth", "middle-aged": "Middle", "elderly": "Senior"}
    if rec_type == "age":
        return f"age_{age_map[age_group]}"
    if rec_type == "cross":
        return f"cross_{gender[0].upper()}_{age_map[age_group]}"
    return "neutral_all"


def calculate_movie_score(popularity: float, rank0: int, rank_weight: float) -> float:
    popularity = max(0.0, min(1.0, float(popularity)))
    unpopularity_score = (1 - popularity) * 10
    rank_score = (20 - rank0) / 20
    total_score = unpopularity_score * (1 + rank_weight * rank_score)
    return round(total_score, 4)


def score_one_list(
    recommendations: List[str],
    preferred: List[str],
    pop_col: str,
    pop_map: Dict[str, Dict[str, float]],
    top_k: int,
    rank_weight: float,
) -> Dict[str, Any]:
    rec_top = recommendations[:top_k]
    total = 0.0
    items = []
    for movie in preferred:
        if movie not in rec_top:
            continue
        rank0 = rec_top.index(movie)
        title = normalize_title(movie)
        popularity = pop_map.get(title, {}).get(pop_col, 0.5)
        s = calculate_movie_score(popularity, rank0, rank_weight)
        total += s
        items.append({
            "movie": movie,
            "rank": rank0 + 1,
            "popularity": round(float(popularity), 4),
            "score": s,
        })
    avg = round(total / len(items), 4) if items else 0.0
    return {
        "total_score": round(total, 4),
        "average_score": avg,
        "num_preferred_movies": len(items),
        "movie_scores": items,
    }


def process_all(input_dir: str, csv_path: str, output_dir: str, top_k: int, rank_weight: float) -> Dict[str, Any]:
    pop_map = load_movie_popularity(csv_path)
    files = sorted(glob.glob(os.path.join(input_dir, "recommendation_results_*.json")))
    if not files:
        raise SystemExit(f"No files in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    user_results: Dict[str, Any] = {}
    summary = {
        s: {
            "count": 0,
            "orig_sum_total": 0.0,
            "rer_sum_total": 0.0,
            "delta_sum_total": 0.0,
            "orig_sum_avg": 0.0,
            "rer_sum_avg": 0.0,
            "delta_sum_avg": 0.0,
            "orig_total_scores": [],
            "rer_total_scores": [],
            "delta_total_scores": [],
            "orig_avg_scores": [],
            "rer_avg_scores": [],
            "delta_avg_scores": [],
        }
        for s in SCENARIOS
    }

    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)

        for uid, payload in data.items():
            info = payload.get("user_info", {})
            gender = info.get("gender", "male")
            age_group = info.get("age_group", "young")
            results = payload.get("results", {})
            per_user = {}
            for s in SCENARIOS:
                block = results.get(s, {})
                rec_orig = list(block.get("recommendations") or [])
                rec_rer = list(block.get("reranked_recommendations") or rec_orig)
                preferred = list(block.get("true_preferred_movies") or [])
                pop_col = get_popularity_column(gender, age_group, s)

                orig_score = score_one_list(rec_orig, preferred, pop_col, pop_map, top_k, rank_weight)
                rer_score = score_one_list(rec_rer, preferred, pop_col, pop_map, top_k, rank_weight)
                delta_total = round(rer_score["total_score"] - orig_score["total_score"], 4)
                delta_avg = round(rer_score["average_score"] - orig_score["average_score"], 4)

                per_user[s] = {
                    "original_top15": orig_score,
                    "reranked_top15": rer_score,
                    "delta": {
                        "total_score": delta_total,
                        "average_score": delta_avg,
                    },
                }

                summary[s]["count"] += 1
                summary[s]["orig_sum_total"] += orig_score["total_score"]
                summary[s]["rer_sum_total"] += rer_score["total_score"]
                summary[s]["delta_sum_total"] += delta_total
                summary[s]["orig_sum_avg"] += orig_score["average_score"]
                summary[s]["rer_sum_avg"] += rer_score["average_score"]
                summary[s]["delta_sum_avg"] += delta_avg
                summary[s]["orig_total_scores"].append(orig_score["total_score"])
                summary[s]["rer_total_scores"].append(rer_score["total_score"])
                summary[s]["delta_total_scores"].append(delta_total)
                summary[s]["orig_avg_scores"].append(orig_score["average_score"])
                summary[s]["rer_avg_scores"].append(rer_score["average_score"])
                summary[s]["delta_avg_scores"].append(delta_avg)

            user_results[uid] = {"user_info": info, "scores": per_user}

    summary_out = {}
    for s in SCENARIOS:
        c = max(1, summary[s]["count"])
        summary_out[s] = {
            "count": summary[s]["count"],
            "original_total_score_sum": round(summary[s]["orig_sum_total"], 6),
            "reranked_total_score_sum": round(summary[s]["rer_sum_total"], 6),
            "delta_total_score_sum": round(summary[s]["delta_sum_total"], 6),
            "original_total_score_per_user": round(summary[s]["orig_sum_total"] / c, 6),
            "reranked_total_score_per_user": round(summary[s]["rer_sum_total"] / c, 6),
            "delta_total_score_per_user": round(summary[s]["delta_sum_total"] / c, 6),
            "original_total_max": round(max(summary[s]["orig_total_scores"]) if summary[s]["orig_total_scores"] else 0.0, 4),
            "original_total_min": round(min(summary[s]["orig_total_scores"]) if summary[s]["orig_total_scores"] else 0.0, 4),
            "reranked_total_max": round(max(summary[s]["rer_total_scores"]) if summary[s]["rer_total_scores"] else 0.0, 4),
            "reranked_total_min": round(min(summary[s]["rer_total_scores"]) if summary[s]["rer_total_scores"] else 0.0, 4),
            "original_average_score": round(summary[s]["orig_sum_avg"] / c, 6),
            "reranked_average_score": round(summary[s]["rer_sum_avg"] / c, 6),
            "delta_average_score": round(summary[s]["delta_sum_avg"] / c, 6),
            "original_max": round(max(summary[s]["orig_avg_scores"]) if summary[s]["orig_avg_scores"] else 0.0, 4),
            "original_min": round(min(summary[s]["orig_avg_scores"]) if summary[s]["orig_avg_scores"] else 0.0, 4),
            "reranked_max": round(max(summary[s]["rer_avg_scores"]) if summary[s]["rer_avg_scores"] else 0.0, 4),
            "reranked_min": round(min(summary[s]["rer_avg_scores"]) if summary[s]["rer_avg_scores"] else 0.0, 4),
        }

    full = {
        "top_k": top_k,
        "rank_weight": rank_weight,
        "input_dir": input_dir,
        "summary": summary_out,
        "user_results": user_results,
    }

    with open(os.path.join(output_dir, "top15_dual_scores.json"), "w", encoding="utf-8") as f:
        json.dump(full, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "top15_dual_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "scenario", "count",
            "original_total_sum", "reranked_total_sum", "delta_total_sum",
            "original_total_per_user", "reranked_total_per_user", "delta_total_per_user",
            "original_avg", "reranked_avg", "delta_avg",
            "original_max", "original_min", "reranked_max", "reranked_min"
        ])
        for s in SCENARIOS:
            x = summary_out[s]
            w.writerow([
                s, x["count"],
                x["original_total_score_sum"], x["reranked_total_score_sum"], x["delta_total_score_sum"],
                x["original_total_score_per_user"], x["reranked_total_score_per_user"], x["delta_total_score_per_user"],
                x["original_average_score"], x["reranked_average_score"], x["delta_average_score"],
                x["original_max"], x["original_min"], x["reranked_max"], x["reranked_min"]
            ])

    lines = [
        "# Top-15 双列表评分总结", "",
        f"- 输入目录：`{input_dir}`",
        f"- 评分规则来源：`recommend_scoring.py`（同公式）",
        f"- 排名权重 rank_weight：`{rank_weight}`",
        f"- Top-K：`{top_k}`", "",
        "## 各场景结果（推荐列表总分 & average_score）", "",
    ]
    for s in SCENARIOS:
        x = summary_out[s]
        trend_total = "增强" if x["delta_total_score_per_user"] > 0 else ("降低" if x["delta_total_score_per_user"] < 0 else "不变")
        trend_avg = "增强" if x["delta_average_score"] > 0 else ("降低" if x["delta_average_score"] < 0 else "不变")
        lines += [
            f"### {s}",
            f"- 用户数：`{x['count']}`",
            f"- 总分（全体求和）original：`{x['original_total_score_sum']:.6f}`",
            f"- 总分（全体求和）reranked：`{x['reranked_total_score_sum']:.6f}`",
            f"- 总分变化（reranked-original）：`{x['delta_total_score_sum']:+.6f}`",
            f"- 人均总分 original：`{x['original_total_score_per_user']:.6f}`",
            f"- 人均总分 reranked：`{x['reranked_total_score_per_user']:.6f}`",
            f"- 人均总分变化：`{x['delta_total_score_per_user']:+.6f}`",
            f"- original：`{x['original_average_score']:.6f}`",
            f"- reranked：`{x['reranked_average_score']:.6f}`",
            f"- delta：`{x['delta_average_score']:+.6f}`",
            f"- 结论（总分）：**{trend_total}**",
            f"- 结论（average_score）：**{trend_avg}**", "",
        ]

    with open(os.path.join(output_dir, "top15_dual_readable_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return full


def main() -> None:
    p = argparse.ArgumentParser(description="Score top15 original and reranked lists using recommend_scoring rule.")
    p.add_argument("--input-dir", default="reranked_JSON")
    p.add_argument("--popularity-csv", default="movies_with_popularity.csv")
    p.add_argument("--output-dir", default="top15_scoring_dual_lists")
    p.add_argument("--top-k", type=int, default=15)
    p.add_argument("--rank-weight", type=float, default=0.2)
    args = p.parse_args()

    process_all(args.input_dir, args.popularity_csv, args.output_dir, args.top_k, args.rank_weight)
    print(f"Done. Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
