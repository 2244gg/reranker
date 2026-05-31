"""Rerank engine extracted from ``rerank_recommendations.py``.

This module is dataset-agnostic: ``popularity_columns_for(scenario, demographics)``
maps any (scenario, user demographics) pair to the appropriate column in a
popularity table built by ``src.popularity.slicer``.

Backwards compatibility:
- The legacy ML-1M scenarios ("neutral", "gender", "age", "cross") are mapped
  to the corresponding column suffixes from ``movies_with_popularity.csv``.
- New experiments may pass a tuple of demographic dimension names as the
  scenario (e.g. ``("gender", "age_group")``); this module then composes the
  cross column dynamically.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Column resolution
# ---------------------------------------------------------------------------


# Same suffix conventions as src.popularity.slicer for backward compatibility.
_VALUE_SUFFIX = {
    "gender": {
        "male": "M",
        "female": "F",
        "M": "M",
        "F": "F",
        "unknown": "U",
    },
    "age_group": {
        "young": "Youth",
        "middle-aged": "Middle",
        "elderly": "Senior",
        "young-adult": "YoungAdult",
        "Youth": "Youth",
        "Middle": "Middle",
        "Senior": "Senior",
        "unknown": "Unknown",
    },
    "occupation_group": {
        "student": "Student",
        "professional": "Pro",
        "service": "Service",
        "other": "Other",
    },
}


def _suffix(dim: str, value: str) -> str:
    table = _VALUE_SUFFIX.get(dim, {})
    if value in table:
        return table[value]
    return str(value).replace(" ", "").replace("-", "_") or "Unknown"


def popularity_columns_for(
    scenario: Any,
    demographics: Mapping[str, str],
) -> str:
    """Return the popularity-table column name for a (scenario, user) pair.

    ``scenario`` may be:
      * ``"neutral"``                 -> ``neutral_All``
      * ``"gender"``                  -> ``gender_<M|F>``
      * ``"age"``                     -> ``age_<Youth|Middle|Senior>``
      * ``"occupation"``              -> ``occupation_<Student|Pro|...>``
      * ``"cross"``                   -> legacy ML-1M cross with gender+age
      * a tuple of dim names, e.g. ``("gender", "age_group")`` -> dynamic cross
    """
    if scenario in (None, (), "neutral"):
        return "neutral_All"

    if isinstance(scenario, tuple):
        if len(scenario) == 1:
            dim = scenario[0]
            val = demographics.get(dim, "unknown")
            prefix = "age" if dim == "age_group" else dim.split("_")[0]
            return f"{prefix}_{_suffix(dim, val)}"
        # 2D / 3D cross
        suffixes = [_suffix(d, demographics.get(d, "unknown")) for d in scenario]
        return "cross_" + "_".join(suffixes)

    if scenario == "gender":
        return f"gender_{_suffix('gender', demographics.get('gender', 'unknown'))}"
    if scenario == "age":
        return f"age_{_suffix('age_group', demographics.get('age_group', 'unknown'))}"
    if scenario == "occupation":
        return (
            "occupation_"
            f"{_suffix('occupation_group', demographics.get('occupation_group', 'unknown'))}"
        )
    if scenario == "cross":
        # Legacy ML-1M: gender x age_group only
        g = _suffix("gender", demographics.get("gender", "unknown"))
        a = _suffix("age_group", demographics.get("age_group", "unknown"))
        return f"cross_{g}_{a}"

    raise ValueError(f"Unsupported scenario spec: {scenario!r}")


# ---------------------------------------------------------------------------
# Core formulas
# ---------------------------------------------------------------------------


def preference_probability(
    rank_n: int,
    candidate_n: int,
    base: float,
    decay_rate: float,
) -> float:
    """Geometric-decay rank score P_i_norm in [base, 1]."""
    if candidate_n <= 1:
        return 1.0
    denom = 1 - (decay_rate ** candidate_n)
    if math.isclose(denom, 0.0):
        progress = (candidate_n - rank_n) / (candidate_n - 1)
        return base + (1 - base) * progress
    numer = (decay_rate ** rank_n) - (decay_rate ** candidate_n)
    return base + (1 - base) * (numer / denom)


def safe_clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def compute_dynamic_alpha(
    user_score: float,
    scene_mean_score: float,
    alpha_base: float,
    alpha_min: float,
    alpha_max: float,
    alpha_gain: float,
) -> float:
    """User-adaptive alpha.

    The rerank score is ``S = alpha * P + (1 - alpha) * (1 - B)``: a larger
    ``alpha`` trusts the LLM ordering more (less long-tail intervention).

    This implementation follows the *personalization* philosophy: the
    rerank intensity should *match* a user's existing taste profile rather
    than correct it.

      * Mainstream user (``user_score > scene_mean``)
        -> ``centered_ratio > 0``
        -> ``alpha`` increases -> less long-tail bonus -> keeps LLM order
        -> mainstream-loving user keeps getting mainstream picks.

      * Niche user (``user_score < scene_mean``)
        -> ``centered_ratio < 0``
        -> ``alpha`` decreases -> more long-tail bonus
        -> niche-loving user gets more obscure picks promoted.

    The sign of the gain term is therefore ``+ alpha_gain * centered_ratio``.
    The opposite sign would correspond to the fairness-paper convention of
    *correcting* mainstream users toward the long tail.
    """
    eps = 1e-12
    centered_ratio = (user_score - scene_mean_score) / max(scene_mean_score, eps)
    alpha_value = alpha_base + alpha_gain * centered_ratio
    return safe_clip(alpha_value, alpha_min, alpha_max)


def rerank_one_list(
    recommendations: Sequence[str],
    popularity_lookup,
    column: str,
    *,
    alpha: float,
    base: float = 0.2,
    decay_rate: float = 0.9,
    top_n: int = 20,
    protect_top: int = 0,
) -> Dict[str, Any]:
    """Rerank a single list of recommendations toward unpopular items.

    Formula (popularity-debiasing):

        S_i = alpha * P_i_norm + (1 - alpha) * (1 - B_i_norm)

    where:
      * P_i_norm in [base, 1]: rank-decay preference score (large for top
        ranks). Encodes the LLM's original ordering signal.
      * B_i_norm in [0, 1]: group-conditional popularity (large for items
        the user's demographic slice frequently consumes).
      * (1 - B_i_norm): unpopularity bonus (large for niche items).

    Larger alpha -> trust LLM's original ordering more.
    Smaller alpha -> apply more aggressive long-tail promotion.

    ``protect_top``:
        Number of leading positions whose original LLM order is preserved
        verbatim. The remaining ``len(recommendations) - protect_top``
        candidates are reranked by S among themselves and concatenated
        after the protected prefix. This trades some long-tail promotion
        for stable Precision@K1 and MRR.

    ``popularity_lookup(title, column)`` MUST return ``(b_norm, found)`` where
    ``b_norm in [0, 1]`` and ``found`` indicates whether the title was matched.
    """
    candidate_n = len(recommendations)
    keep_n = min(top_n, candidate_n)
    protect_n = max(0, min(int(protect_top), candidate_n))

    scored: List[Dict[str, Any]] = []
    missing: List[str] = []
    for idx, title in enumerate(recommendations, start=1):
        p_norm = preference_probability(idx, candidate_n, base, decay_rate)
        b_norm, found = popularity_lookup(title, column)
        if not found:
            missing.append(title)
        score = alpha * p_norm + (1 - alpha) * (1 - b_norm)
        scored.append(
            {
                "title": title,
                "rank_original": idx,
                "P_i_norm": p_norm,
                "B_i_norm": b_norm,
                "S_i": score,
            }
        )

    if protect_n == 0:
        ordered = sorted(scored, key=lambda x: x["S_i"], reverse=True)
    else:
        head = scored[:protect_n]  # keep LLM order verbatim
        tail = sorted(scored[protect_n:], key=lambda x: x["S_i"], reverse=True)
        ordered = head + tail

    reranked_titles = [it["title"] for it in ordered[:keep_n]]
    return {
        "reranked_recommendations": reranked_titles,
        "rerank_scores": ordered,
        "missing_titles": missing,
        "popularity_column_used": column,
        "protect_top": protect_n,
    }
