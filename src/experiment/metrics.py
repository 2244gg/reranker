"""Per-user evaluation metrics for original vs reranked recommendation lists.

All metrics are dataset-agnostic and operate on title strings. Title matching
goes through ``normalize_title`` (strict, kept for back-compat with shipped
``movies_with_popularity.csv``) and ``title_key`` (aggressive, used for hit
detection across LLM outputs which often vary in punctuation, parenthetical
metadata, HTML entities, leading articles, etc).
"""
from __future__ import annotations

import html
import math
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence


# ---------------------------------------------------------------------------
# Title normalization
# ---------------------------------------------------------------------------


def normalize_title(title: str) -> str:
    """Light normalization compatible with the legacy ``movies_with_popularity.csv``.

    Used for popularity table lookups so legacy keys keep working.
    """
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", title.strip())
    if re.search(r",\s*The$", t, re.IGNORECASE):
        t = "The " + re.sub(r",\s*The$", "", t, flags=re.IGNORECASE)
    elif re.search(r",\s*A$", t, re.IGNORECASE):
        t = "A " + re.sub(r",\s*A$", "", t, flags=re.IGNORECASE)
    elif re.search(r",\s*An$", t, re.IGNORECASE):
        t = "An " + re.sub(r",\s*An$", "", t, flags=re.IGNORECASE)
    return t


# Pattern for trailing parenthetical metadata, e.g.
#   "(Hardcover)", "(Penguin Classics)", "(Vol. 1)", "(1995)", "[Paperback]"
_TRAILING_PAREN_RE = re.compile(r"\s*[\(\[][^\(\[\]\)]{1,80}[\)\]]\s*$")
# Pattern for trailing " by Author Name" decoration
_TRAILING_BY_AUTHOR_RE = re.compile(r"\s+by\s+[A-Z][\w\.\-' ]{1,60}\s*$")
# Pattern to drop subtitle after a colon when title is long
_SUBTITLE_RE = re.compile(r":\s+.+$")
# Punctuation we drop entirely for fuzzy matching
_PUNCT_RE = re.compile(r"[^\w\s]")


def title_key(title: str) -> str:
    """Aggressive canonicalization for fuzzy title matching.

    Steps applied in order:
      1. HTML-unescape (`&amp;` -> `&`, `&quot;` -> `"`)
      2. Apply ``normalize_title`` (legacy article + year handling)
      3. Strip trailing parenthetical metadata (up to 3 times)
      4. Strip trailing " by Author"
      5. Drop subtitle after ":" when remaining base is long enough
      6. Lowercase, drop punctuation, collapse whitespace
    """
    if not title:
        return ""
    t = html.unescape(title)
    t = normalize_title(t)
    # Repeatedly strip trailing parens (up to 3 levels of nesting/decoration).
    for _ in range(3):
        new = _TRAILING_PAREN_RE.sub("", t)
        if new == t:
            break
        t = new
    t = _TRAILING_BY_AUTHOR_RE.sub("", t)
    # If we have a colon-subtitle and the base title is long enough (>3 chars),
    # drop the subtitle. Keeps "Lord of the Rings: The Fellowship of the Ring"
    # but removes only the ":..." part. We default to NOT dropping unless the
    # base title alone is already substantial.
    base_before_colon = t.split(":", 1)[0].strip()
    if len(base_before_colon) >= 6 and ":" in t:
        t = base_before_colon
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def titles_match(a: str, b: str, min_substring: int = 8) -> bool:
    """Return True if two titles are likely the same work.

    Two titles match when their canonicalized keys are equal, or when one is
    a substring of the other and the shorter side is at least
    ``min_substring`` characters long.
    """
    ka = title_key(a)
    kb = title_key(b)
    if not ka or not kb:
        return False
    if ka == kb:
        return True
    short, long = (ka, kb) if len(ka) <= len(kb) else (kb, ka)
    if len(short) >= min_substring and short in long:
        return True
    return False


def first_match_in(rec_titles: Sequence[str], relevant: Sequence[str]) -> Optional[int]:
    """Return the 0-based rank of the first ``rec_titles`` entry that matches
    any title in ``relevant`` (under ``titles_match``), or ``None``.
    """
    rel_keys = [title_key(r) for r in relevant]
    rel_keys = [k for k in rel_keys if k]
    if not rel_keys:
        return None
    for i, t in enumerate(rec_titles):
        kt = title_key(t)
        if not kt:
            continue
        for rk in rel_keys:
            if kt == rk:
                return i
            short, long = (kt, rk) if len(kt) <= len(rk) else (rk, kt)
            if len(short) >= 8 and short in long:
                return i
    return None


def count_matches(rec_titles: Sequence[str], relevant: Sequence[str]) -> int:
    """Count how many ``relevant`` titles appear in ``rec_titles`` under fuzzy match.

    Each relevant title is counted at most once.
    """
    if not relevant or not rec_titles:
        return 0
    rec_keys = [title_key(t) for t in rec_titles]
    rec_keys_set = set(k for k in rec_keys if k)
    if not rec_keys_set:
        return 0
    hits = 0
    for r in relevant:
        kr = title_key(r)
        if not kr:
            continue
        if kr in rec_keys_set:
            hits += 1
            continue
        # Fuzzy substring containment.
        matched = False
        for kt in rec_keys_set:
            short, long = (kt, kr) if len(kt) <= len(kr) else (kr, kt)
            if len(short) >= 8 and short in long:
                hits += 1
                matched = True
                break
        if matched:
            continue
    return hits


# ---------------------------------------------------------------------------
# Retrieval metrics (use fuzzy `titles_match` under the hood)
# ---------------------------------------------------------------------------


def precision_at_k(rec_top: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not rec_top or k <= 0:
        return 0.0
    top = list(rec_top)[:k]
    hits = count_matches(top, relevant)
    denom = min(k, len(rec_top))
    return hits / denom if denom else 0.0


def recall_at_k(rec_top: Sequence[str], relevant: Sequence[str], k: int, total_relevant: Optional[int] = None) -> float:
    top = list(rec_top)[:k]
    hits = count_matches(top, relevant)
    denom = total_relevant if total_relevant is not None else len(set(relevant))
    return hits / denom if denom else 0.0


def hit_rate_at_k(rec_top: Sequence[str], relevant: Sequence[str], k: int) -> float:
    top = list(rec_top)[:k]
    return 1.0 if count_matches(top, relevant) > 0 else 0.0


# ---------------------------------------------------------------------------
# Long-tail / diversity metrics (popularity-debias literature standards)
# ---------------------------------------------------------------------------


def aplt_at_k(
    rec_top: Sequence[str],
    popularity_lookup,
    pop_col: str,
    k: int,
    threshold: float = 0.2,
) -> float:
    """Average Percentage of Long-Tail items among the top-k.

    A "long-tail" item is any item whose group-conditional popularity
    ``B_norm`` in ``pop_col`` is *strictly less* than ``threshold``. Returns
    the fraction of top-k items meeting the criterion (0..1). Higher = more
    long-tail exposure.

    This is the standard popularity-debias diagnostic from Abdollahpouri
    et al. (2017): "Controlling Popularity Bias in Learning-to-Rank
    Recommendation".
    """
    top = list(rec_top)[:k]
    if not top:
        return 0.0
    long_tail = 0
    for t in top:
        result = popularity_lookup(t, pop_col)
        # Support both lookup conventions in the codebase: ``lookup`` returns
        # a bare float; ``lookup_with_found`` returns ``(float, bool)``.
        if isinstance(result, tuple):
            b = float(result[0])
        else:
            b = float(result)
        if b < threshold:
            long_tail += 1
    return long_tail / float(len(top))


def gini_index(values: Sequence[float]) -> float:
    """Gini coefficient of a 1-D distribution (0 = uniform, 1 = single-winner).

    Used at corpus level to quantify how concentrated a recommendation
    distribution is across items. Lower Gini after rerank = the algorithm
    spreads exposure more evenly.
    """
    arr = sorted(float(v) for v in values if v > 0)
    n = len(arr)
    if n == 0:
        return 0.0
    s = sum(arr)
    if s <= 0:
        return 0.0
    cum = 0.0
    for i, v in enumerate(arr, start=1):
        cum += i * v
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def mrr_first_hit(rec_top: Sequence[str], relevant: Sequence[str]) -> float:
    idx = first_match_in(rec_top, relevant)
    return 1.0 / (idx + 1) if idx is not None else 0.0


def ndcg_at_k(rec_top: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel_keys = [title_key(r) for r in relevant]
    rel_keys = [k for k in rel_keys if k]
    if not rel_keys:
        return 0.0

    dcg = 0.0
    for i, t in enumerate(rec_top[:k], start=1):
        kt = title_key(t)
        if not kt:
            continue
        is_hit = False
        for rk in rel_keys:
            if kt == rk:
                is_hit = True
                break
            short, long = (kt, rk) if len(kt) <= len(rk) else (rk, kt)
            if len(short) >= 8 and short in long:
                is_hit = True
                break
        if is_hit:
            dcg += 1.0 / math.log2(i + 1)
    ideal_hits = min(k, len(rel_keys))
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return (dcg / idcg) if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Bias-targeted unpopularity score (mirrors score_top15_dual_lists.py)
# ---------------------------------------------------------------------------


def unpopularity_score(
    rec_top: Sequence[str],
    relevant: Sequence[str],
    popularity_lookup,
    pop_col: str,
    *,
    rank_weight: float = 0.2,
    list_size: int = 20,
) -> Dict[str, Any]:
    """Compute the bias-aware score from ``score_top15_dual_lists.py``.

    For each ``relevant`` movie hit in ``rec_top``:

        movie_score = (1 - pop) * 10 * (1 + rank_weight * (list_size - rank0) / list_size)

    Returns ``{total, average, hits}``.
    """
    rel_keys = [title_key(r) for r in relevant]
    rel_keys = [k for k in rel_keys if k]
    total = 0.0
    items = []
    for r0, title in enumerate(rec_top):
        kt = title_key(title)
        if not kt:
            continue
        is_hit = False
        for rk in rel_keys:
            if kt == rk:
                is_hit = True
                break
            short, long = (kt, rk) if len(kt) <= len(rk) else (rk, kt)
            if len(short) >= 8 and short in long:
                is_hit = True
                break
        if not is_hit:
            continue
        # Pass the raw title; the popularity lookup is responsible for trying
        # exact / title_key / normalize_title in order. This keeps hit
        # detection (title_key based) consistent with B-value retrieval.
        pop_raw = popularity_lookup(title, pop_col)
        pop = float(pop_raw if pop_raw else 0.5)
        pop = max(0.0, min(1.0, pop))
        unpop = (1.0 - pop) * 10.0
        rank_score = (list_size - r0) / list_size if list_size else 0.0
        s = unpop * (1.0 + rank_weight * rank_score)
        s = round(s, 4)
        total += s
        items.append({"title": title, "rank": r0 + 1, "popularity": round(pop, 4), "score": s})
    avg = round(total / len(items), 4) if items else 0.0
    return {
        "total_score": round(total, 4),
        "average_score": avg,
        "hits": len(items),
        "items": items,
    }


# ---------------------------------------------------------------------------
# High-level helper
# ---------------------------------------------------------------------------


def compute_per_user_metrics(
    *,
    recommendations_original: Sequence[str],
    recommendations_reranked: Sequence[str],
    relevant_pool: Sequence[str],
    test_movies_count: int,
    k: int,
    popularity_lookup,
    pop_col: str,
    rank_weight: float = 0.2,
    score_table: Optional[Any] = None,
) -> Dict[str, Any]:
    """One-stop computation for a (user, scenario) pair.

    Both lists are truncated to ``k``. ``relevant_pool`` is the set of test
    items that fall inside the shared candidate pool (typically top-20).

    When ``score_table`` (a ``src.popularity.item_scoring.ItemScoreTable``) is
    provided, an additional ``difficulty_weighted_hit_score`` is reported per
    side (original / reranked).
    """
    orig_top = list(recommendations_original)[:k]
    rer_top = list(recommendations_reranked)[:k]

    out: Dict[str, Any] = {"k": k}

    for label, top in (("original", orig_top), ("reranked", rer_top)):
        hits = sum(1 for t in top if t in set(relevant_pool))
        side: Dict[str, Any] = {
            "precision": precision_at_k(top, relevant_pool, k),
            "recall": (hits / test_movies_count) if test_movies_count else 0.0,
            "hit_rate": hit_rate_at_k(top, relevant_pool, k),
            "mrr": mrr_first_hit(top, relevant_pool),
            "ndcg_at_k": ndcg_at_k(top, relevant_pool, k),
            "hits": hits,
            "test_movies_count": test_movies_count,
            "unpopularity": unpopularity_score(
                top,
                relevant_pool,
                popularity_lookup,
                pop_col,
                rank_weight=rank_weight,
                list_size=k,
            ),
            # APLT in the user's slice column (long-tail threshold 0.2).
            # A second key uses the neutral column when available, capturing
            # *global* long-tail exposure regardless of demographics.
            "aplt": aplt_at_k(top, popularity_lookup, pop_col, k, threshold=0.2),
            "aplt_neutral": aplt_at_k(top, popularity_lookup, "neutral_All", k, threshold=0.2),
        }
        if score_table is not None:
            from src.popularity.item_scoring import difficulty_weighted_hit_score

            side["difficulty"] = difficulty_weighted_hit_score(
                top,
                relevant_pool,
                score_table,
                pop_col,
                rank_weight=rank_weight,
                list_size=k,
            )
        out[label] = side
    return out
