#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys as _sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Ensure project root is importable so the wrapper functions below can
# delegate to ``src.experiment.rerank`` (single source of truth for formulas).
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_PROJECT_ROOT))


SCENARIOS = ("neutral", "age", "gender", "cross")
RERANK_SCENARIOS = ("age", "gender", "cross")

GENDER_TO_SUFFIX = {
    "male": "M",
    "female": "F",
}

AGE_TO_SUFFIX = {
    "young": "Youth",
    "middle-aged": "Middle",
    "elderly": "Senior",
}


def normalize_title(title: str) -> str:
    return " ".join(title.strip().lower().split())


def strip_trailing_year(title: str) -> str:
    t = title.strip()
    if len(t) >= 7 and t.endswith(")") and t[-6] == "(" and t[-5:-1].isdigit():
        return t[:-7].strip()
    return t


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank LLM recommendation lists with popularity-aware debiasing score."
    )
    parser.add_argument("--input-dir", default="JSON", help="Directory with original JSON files.")
    parser.add_argument(
        "--popularity-csv",
        default="movies_with_popularity.csv",
        help="CSV file containing precomputed normalized popularity columns.",
    )
    parser.add_argument(
        "--output-dir",
        default="reranked_JSON",
        help="Directory to write enhanced JSON files.",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Final recommendation list length N.")
    parser.add_argument("--alpha", type=float, default=0.7, help="Weight for preference probability.")
    parser.add_argument(
        "--disable-dynamic-alpha",
        action="store_true",
        help="Use fixed alpha for all reranked scenarios.",
    )
    parser.add_argument("--alpha-min", type=float, default=0.5, help="Lower bound for dynamic alpha.")
    parser.add_argument("--alpha-max", type=float, default=0.9, help="Upper bound for dynamic alpha.")
    parser.add_argument(
        "--alpha-gain",
        type=float,
        default=0.2,
        help="Linear gain used in dynamic alpha mapping.",
    )
    parser.add_argument("--base", type=float, default=0.2, help="Base value in rank normalization.")
    parser.add_argument("--decay-rate", type=float, default=0.9, help="Non-linear decay rate in rank normalization.")
    return parser.parse_args()


def load_popularity_map(csv_path: Path) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    exact_map: Dict[str, Dict[str, float]] = {}
    normalized_map: Dict[str, Dict[str, float]] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            title = (row.get("Title") or "").strip()
            if not title:
                continue
            parsed_row: Dict[str, float] = {}
            for key, value in row.items():
                if key in {"MovieID", "Title", "Genres"}:
                    continue
                try:
                    parsed_row[key] = float(value) if value not in (None, "") else 0.0
                except ValueError:
                    parsed_row[key] = 0.0

            exact_map[title] = parsed_row
            normalized_map[normalize_title(title)] = parsed_row
            normalized_map[normalize_title(strip_trailing_year(title))] = parsed_row

    return exact_map, normalized_map


def resolve_popularity_column(scenario: str, user_info: Dict[str, str]) -> str:
    gender = user_info.get("gender", "").strip().lower()
    age_group = user_info.get("age_group", "").strip().lower()

    if scenario == "neutral":
        return "neutral_All"
    if scenario == "gender":
        return f"gender_{GENDER_TO_SUFFIX.get(gender, 'M')}"
    if scenario == "age":
        return f"age_{AGE_TO_SUFFIX.get(age_group, 'Youth')}"
    if scenario == "cross":
        gender_suffix = GENDER_TO_SUFFIX.get(gender, "M")
        age_suffix = AGE_TO_SUFFIX.get(age_group, "Youth")
        return f"cross_{gender_suffix}_{age_suffix}"
    raise ValueError(f"Unsupported scenario: {scenario}")


def preference_probability(rank_n: int, candidate_n: int, base: float, decay_rate: float) -> float:
    # Implementation lives in src.experiment.rerank; re-exported via top-of-file import.
    from src.experiment.rerank import preference_probability as _impl
    return _impl(rank_n, candidate_n, base, decay_rate)


def lookup_popularity(
    title: str,
    exact_map: Dict[str, Dict[str, float]],
    normalized_map: Dict[str, Dict[str, float]],
    column: str,
) -> Tuple[float, bool]:
    row = exact_map.get(title)
    if row is None:
        row = normalized_map.get(normalize_title(title))
    if row is None:
        row = normalized_map.get(normalize_title(strip_trailing_year(title)))
    if row is None:
        return 0.0, False
    return float(row.get(column, 0.0)), True


def safe_clip(value: float, low: float, high: float) -> float:
    from src.experiment.rerank import safe_clip as _impl
    return _impl(value, low, high)


def get_history_titles(results: Dict[str, Any], scenario: str, fallback_limit: int = 10) -> List[str]:
    scenario_block = results.get(scenario, {})
    preferred = scenario_block.get("true_preferred_movies", [])
    if isinstance(preferred, list) and preferred:
        return [x for x in preferred if isinstance(x, str)]

    neutral_preferred = results.get("neutral", {}).get("true_preferred_movies", [])
    if isinstance(neutral_preferred, list) and neutral_preferred:
        return [x for x in neutral_preferred if isinstance(x, str)]

    neutral_recs = results.get("neutral", {}).get("recommendations", [])
    if isinstance(neutral_recs, list) and neutral_recs:
        return [x for x in neutral_recs[:fallback_limit] if isinstance(x, str)]
    return []


def compute_history_score(
    history_titles: List[str],
    pop_col: str,
    exact_map: Dict[str, Dict[str, float]],
    normalized_map: Dict[str, Dict[str, float]],
) -> Tuple[float, List[str]]:
    score_sum = 0.0
    missing_titles: List[str] = []
    for title in history_titles:
        b_norm, found = lookup_popularity(title, exact_map, normalized_map, pop_col)
        score_sum += b_norm
        if not found:
            missing_titles.append(title)
    return score_sum, missing_titles


def compute_dynamic_alpha(
    user_score: float,
    scene_mean_score: float,
    alpha_base: float,
    alpha_min: float,
    alpha_max: float,
    alpha_gain: float,
) -> float:
    from src.experiment.rerank import compute_dynamic_alpha as _impl
    return _impl(
        user_score,
        scene_mean_score,
        alpha_base,
        alpha_min,
        alpha_max,
        alpha_gain,
    )


def collect_scene_history_scores(
    data: Dict[str, Any],
    exact_map: Dict[str, Dict[str, float]],
    normalized_map: Dict[str, Dict[str, float]],
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, float], Dict[str, Dict[str, List[str]]]]:
    user_scene_scores: Dict[str, Dict[str, float]] = {}
    user_scene_missing: Dict[str, Dict[str, List[str]]] = {}
    score_sum = {scene: 0.0 for scene in RERANK_SCENARIOS}
    score_count = {scene: 0 for scene in RERANK_SCENARIOS}

    for user_id, user_payload in data.items():
        user_info = user_payload.get("user_info", {})
        results = user_payload.get("results", {})
        scene_scores: Dict[str, float] = {}
        scene_missing: Dict[str, List[str]] = {}
        for scenario in RERANK_SCENARIOS:
            if scenario not in results:
                continue
            pop_col = resolve_popularity_column(scenario, user_info)
            history_titles = get_history_titles(results, scenario)
            hist_score, missing_titles = compute_history_score(
                history_titles=history_titles,
                pop_col=pop_col,
                exact_map=exact_map,
                normalized_map=normalized_map,
            )
            scene_scores[scenario] = hist_score
            scene_missing[scenario] = missing_titles
            score_sum[scenario] += hist_score
            score_count[scenario] += 1
        user_scene_scores[user_id] = scene_scores
        user_scene_missing[user_id] = scene_missing

    scene_means: Dict[str, float] = {}
    for scenario in RERANK_SCENARIOS:
        if score_count[scenario] == 0:
            scene_means[scenario] = 0.0
        else:
            scene_means[scenario] = score_sum[scenario] / score_count[scenario]

    return user_scene_scores, scene_means, user_scene_missing


def rerank_one_list(
    recommendations: List[str],
    user_info: Dict[str, str],
    scenario: str,
    exact_map: Dict[str, Dict[str, float]],
    normalized_map: Dict[str, Dict[str, float]],
    alpha: float,
    base: float,
    decay_rate: float,
    top_n: int,
) -> Dict[str, Any]:
    pop_col = resolve_popularity_column(scenario, user_info)
    candidate_n = len(recommendations)
    keep_n = min(top_n, candidate_n)
    scored_items: List[Dict[str, Any]] = []
    missing_titles: List[str] = []

    for idx, title in enumerate(recommendations, start=1):
        p_norm = preference_probability(idx, candidate_n, base, decay_rate)
        b_norm, found = lookup_popularity(title, exact_map, normalized_map, pop_col)
        if not found:
            missing_titles.append(title)
        # Popularity-debiasing formula: reward niche items (low B_i_norm).
        # See src/experiment/rerank.py for the full derivation.
        score = alpha * p_norm + (1 - alpha) * (1 - b_norm)
        scored_items.append(
            {
                "title": title,
                "rank_original": idx,
                "P_i_norm": p_norm,
                "B_i_norm": b_norm,
                "S_i": score,
            }
        )

    scored_items.sort(key=lambda x: x["S_i"], reverse=True)
    reranked_titles = [item["title"] for item in scored_items[:keep_n]]

    return {
        "reranked_recommendations": reranked_titles,
        "rerank_scores": scored_items,
        "missing_titles": missing_titles,
        "popularity_column_used": pop_col,
    }


def process_file(
    file_path: Path,
    output_path: Path,
    exact_map: Dict[str, Dict[str, float]],
    normalized_map: Dict[str, Dict[str, float]],
    top_n: int,
    alpha: float,
    base: float,
    decay_rate: float,
    disable_dynamic_alpha: bool,
    alpha_min: float,
    alpha_max: float,
    alpha_gain: float,
) -> Tuple[int, int]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    total_lists = 0
    total_missing_titles = 0
    user_scene_scores, scene_mean_scores, user_scene_missing = collect_scene_history_scores(
        data=data,
        exact_map=exact_map,
        normalized_map=normalized_map,
    )

    for user_id, user_payload in data.items():
        user_info = user_payload.get("user_info", {})
        results = user_payload.get("results", {})
        for scenario in SCENARIOS:
            if scenario not in results:
                continue
            recs = results[scenario].get("recommendations", [])
            if not isinstance(recs, list) or len(recs) == 0:
                continue
            if scenario == "neutral":
                keep_n = min(top_n, len(recs))
                results[scenario]["rerank_params"] = {
                    "top_n": top_n,
                    "mode": "bypass_neutral",
                    "reason": "neutral scenario is intentionally not reranked",
                }
                results[scenario]["reranked_recommendations"] = recs[:keep_n]
                results[scenario]["rerank_scores"] = []
                results[scenario]["missing_titles"] = []
                results[scenario]["history_popularity_score"] = None
                results[scenario]["scene_population_mean_score"] = None
                results[scenario]["dynamic_alpha"] = None
                results[scenario]["alpha_mapping_meta"] = {
                    "mapping": "bypass",
                }
                results[scenario]["popularity_column_used"] = "neutral_All"
                total_lists += 1
                continue

            user_score = user_scene_scores.get(user_id, {}).get(scenario, 0.0)
            scene_mean_score = scene_mean_scores.get(scenario, 0.0)
            if disable_dynamic_alpha:
                dynamic_alpha = alpha
                mapping_mode = "fixed_alpha"
            else:
                dynamic_alpha = compute_dynamic_alpha(
                    user_score=user_score,
                    scene_mean_score=scene_mean_score,
                    alpha_base=alpha,
                    alpha_min=alpha_min,
                    alpha_max=alpha_max,
                    alpha_gain=alpha_gain,
                )
                mapping_mode = "linear_mean_centered"

            rerank_result = rerank_one_list(
                recommendations=recs,
                user_info=user_info,
                scenario=scenario,
                exact_map=exact_map,
                normalized_map=normalized_map,
                alpha=dynamic_alpha,
                base=base,
                decay_rate=decay_rate,
                top_n=top_n,
            )
            results[scenario]["rerank_params"] = {
                "top_n": top_n,
                "alpha_base": alpha,
                "base": base,
                "decay_rate": decay_rate,
                "formula": "S_i = alpha * P_i_norm + (1-alpha) * (1-B_i_norm)",
            }
            results[scenario]["reranked_recommendations"] = rerank_result["reranked_recommendations"]
            results[scenario]["rerank_scores"] = rerank_result["rerank_scores"]
            # Merge missing titles from ranking list and history score list.
            history_missing = user_scene_missing.get(user_id, {}).get(scenario, [])
            merged_missing = list(dict.fromkeys(rerank_result["missing_titles"] + history_missing))
            results[scenario]["missing_titles"] = merged_missing
            results[scenario]["popularity_column_used"] = rerank_result["popularity_column_used"]
            results[scenario]["history_popularity_score"] = user_score
            results[scenario]["scene_population_mean_score"] = scene_mean_score
            results[scenario]["dynamic_alpha"] = dynamic_alpha
            results[scenario]["alpha_mapping_meta"] = {
                "mode": mapping_mode,
                "alpha_min": alpha_min,
                "alpha_max": alpha_max,
                "alpha_gain": alpha_gain,
                "alpha_base": alpha,
            }

            total_lists += 1
            total_missing_titles += len(merged_missing)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return total_lists, total_missing_titles


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    csv_path = Path(args.popularity_csv)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Popularity CSV not found: {csv_path}")
    if args.top_n <= 0:
        raise ValueError("--top-n must be positive")
    if not (0.0 <= args.alpha <= 1.0):
        raise ValueError("--alpha must be in [0, 1]")
    if not (0.0 <= args.alpha_min <= 1.0):
        raise ValueError("--alpha-min must be in [0, 1]")
    if not (0.0 <= args.alpha_max <= 1.0):
        raise ValueError("--alpha-max must be in [0, 1]")
    if args.alpha_min > args.alpha_max:
        raise ValueError("--alpha-min must be <= --alpha-max")
    if not (0.0 <= args.base <= 1.0):
        raise ValueError("--base must be in [0, 1]")
    if not (0.0 < args.decay_rate <= 1.0):
        raise ValueError("--decay-rate must be in (0, 1]")
    if args.alpha_gain < 0.0:
        raise ValueError("--alpha-gain must be non-negative")

    exact_map, normalized_map = load_popularity_map(csv_path)
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {input_dir}")

    grand_lists = 0
    grand_missing = 0
    for input_file in json_files:
        output_file = output_dir / input_file.name
        list_count, missing_count = process_file(
            file_path=input_file,
            output_path=output_file,
            exact_map=exact_map,
            normalized_map=normalized_map,
            top_n=args.top_n,
            alpha=args.alpha,
            base=args.base,
            decay_rate=args.decay_rate,
            disable_dynamic_alpha=args.disable_dynamic_alpha,
            alpha_min=args.alpha_min,
            alpha_max=args.alpha_max,
            alpha_gain=args.alpha_gain,
        )
        grand_lists += list_count
        grand_missing += missing_count
        print(f"[OK] {input_file.name}: reranked_lists={list_count}, missing_titles={missing_count}")

    print(
        f"Done. files={len(json_files)}, reranked_lists={grand_lists}, "
        f"missing_titles={grand_missing}, output_dir={output_dir}"
    )


if __name__ == "__main__":
    main()
