"""Main experiment runner.

Orchestrates the full pipeline for one ``ExperimentConfig``:

  1. Load dataset adapter, sample users, split history/test.
  2. Build (or load) per-(dim) popularity table.
  3. For each user x scenario:
     a. Render prompt -> call LLM adapter (cached) -> parse recommendations.
     b. Build candidate pool (top-20). Mark which test items are inside.
     c. Compute history_popularity_score and scene mean (collected first pass).
     d. Compute dynamic alpha and rerank.
     e. Compute per-user metrics for original vs reranked at top_k.
  4. Persist:
     - results/<model>/<dataset>/raw/<batch>.json
     - results/<model>/<dataset>/reranked/<batch>.json
     - results/<model>/<dataset>/metrics/per_user.parquet
     - results/<model>/<dataset>/metrics/aggregate.json
     - results/<model>/<dataset>/meta.json
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from src.datasets.base import BaseDatasetAdapter, UserRecord
from src.datasets.factory import build_dataset
from src.experiment.config import ExperimentConfig
from src.experiment.metrics import compute_per_user_metrics, normalize_title, titles_match
from src.experiment.prompt_builder import PromptBuilder
from src.experiment.rerank import (
    compute_dynamic_alpha,
    popularity_columns_for,
    rerank_one_list,
)
from src.llm_adapters.base import BaseLLMAdapter
from src.llm_adapters.cache import LLMCache
from src.llm_adapters.factory import build_adapter
from src.llm_adapters.usage import UsageLogger
from src.popularity.slicer import PopularitySlicer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_LINE_RE = re.compile(r"^\s*(\d+)[\.\)]\s*(.+?)\s*$")


def parse_recommendations(
    text: str,
    *,
    max_items: int = 20,
    history_titles: Optional[Sequence[str]] = None,
    strip_trailing_metadata: bool = True,
) -> List[str]:
    """Parse a numbered list LLM response into clean, deduplicated title strings.

    Behavior:
      * Numbered-list pattern ``^\\s*\\d+[\\.\\)]`` is required per line.
      * Trailing decorations (e.g. ``" - Genre"`` or ``" - Author"``) are
        stripped when ``strip_trailing_metadata`` is True (default). For
        domains where the dash is a meaningful field separator (e.g.
        ``music_track`` where each line is ``"Track - Artist"``), pass
        ``strip_trailing_metadata=False`` to keep the full string.
      * Duplicate titles (after canonicalization via ``title_key``) are skipped.
      * Items in ``history_titles`` (case-insensitive, canonicalized) are
        skipped — LLMs frequently echo back history despite the
        ``Avoid items already in history`` prompt instruction.
    """
    from src.experiment.metrics import title_key

    history_keys = set()
    if history_titles:
        for h in history_titles:
            k = title_key(h)
            if k:
                history_keys.add(k)

    items: List[str] = []
    seen_keys: set = set()
    for line in text.splitlines():
        m = _LINE_RE.match(line)
        if not m:
            continue
        title = m.group(2).strip()
        if strip_trailing_metadata:
            # Strip trailing " - Genre" / " - Author" / em-dash variants.
            title = re.split(r"\s+-\s+|\s+\u2014\s+", title)[0].strip()
        title = title.strip().strip('"').strip()
        if not title:
            continue
        key = title_key(title)
        if not key:
            continue
        if key in seen_keys:
            continue
        if key in history_keys:
            continue
        seen_keys.add(key)
        items.append(title)
        if len(items) >= max_items:
            break
    return items


def _scenario_label(scenario: Sequence[str]) -> str:
    if not scenario:
        return "neutral"
    return "+".join(scenario)


def _scenario_to_pop_column_name(scenario: Sequence[str], demographics: Dict[str, str]) -> str:
    return popularity_columns_for(tuple(scenario) if scenario else "neutral", demographics)


# ---------------------------------------------------------------------------
# Popularity lookup
# ---------------------------------------------------------------------------


class PopularityTable:
    """Wraps the popularity DataFrame with a normalized-title lookup.

    Lookup uses the aggressive ``title_key`` normalizer (lowercase, strip
    punctuation/HTML/parentheticals, collapse whitespace) so that LLM-emitted
    titles with cosmetic decoration ("Remastered", "(Live)", "feat.",
    em-dash vs hyphen, ...) still resolve to the underlying catalog row.

    A legacy ``normalize_title`` index is kept as a secondary fallback for
    back-compat with the original ML-1M pipeline.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        from src.experiment.metrics import title_key

        title_col = "title" if "title" in df.columns else "Title"
        id_col = "item_id" if "item_id" in df.columns else "MovieID"
        self._df = df
        self._exact: Dict[str, Dict[str, float]] = {}
        self._norm: Dict[str, Dict[str, float]] = {}    # legacy: normalize_title
        self._key:  Dict[str, Dict[str, float]] = {}    # primary: title_key
        non_meta = {c for c in df.columns if c not in {title_col, id_col, "Genres"}}
        for _, row in df.iterrows():
            title = str(row[title_col])
            payload = {c: float(row[c]) for c in non_meta if c in row}
            self._exact[title] = payload
            self._norm[normalize_title(title)] = payload
            k = title_key(title)
            if k:
                self._key[k] = payload

    def lookup(self, title: str, column: str) -> float:
        from src.experiment.metrics import title_key

        row = (
            self._exact.get(title)
            or self._key.get(title_key(title))
            or self._norm.get(normalize_title(title))
        )
        if row is None:
            return 0.0
        return float(row.get(column, 0.0))

    def lookup_with_found(self, title: str, column: str) -> Tuple[float, bool]:
        from src.experiment.metrics import title_key

        row = (
            self._exact.get(title)
            or self._key.get(title_key(title))
            or self._norm.get(normalize_title(title))
        )
        if row is None:
            return 0.0, False
        return float(row.get(column, 0.0)), True


def _apply_b_transform(
    df: "pd.DataFrame",
    transform: str,
    saturate_at: float = 0.3,
) -> "pd.DataFrame":
    """Reshape the popularity table's slice cells before rerank lookup.

    See ``RerankConfig.b_transform`` docstring for semantics.
    """
    if transform in (None, "", "raw"):
        return df
    df2 = df.copy()
    meta_cols = {"item_id", "title", "MovieID", "Title", "Genres", "_score_scheme"}
    slice_cols = [c for c in df2.columns if c not in meta_cols]

    if transform == "rank":
        # Per column: replace each cell by its percentile rank (0..1, where
        # 1 = most popular item in the slice). Items with B==0 stay at 0.
        for c in slice_cols:
            col = df2[c].astype(float)
            mask_pos = col > 0
            ranked = col[mask_pos].rank(method="average", pct=True)
            new = pd.Series(0.0, index=col.index)
            new[mask_pos] = ranked
            df2[c] = new
        return df2

    if transform == "saturate":
        thr = max(1e-6, float(saturate_at))
        for c in slice_cols:
            col = df2[c].astype(float)
            df2[c] = (col / thr).clip(upper=1.0)
        return df2

    raise ValueError(f"unknown b_transform: {transform!r}")


def _load_or_build_popularity(
    cfg: ExperimentConfig,
    adapter: BaseDatasetAdapter,
) -> PopularityTable:
    if cfg.popularity_csv and Path(cfg.popularity_csv).exists():
        df = pd.read_csv(cfg.popularity_csv)
    else:
        slicer = PopularitySlicer(adapter)
        df = slicer.build()
        out_path = Path("data/processed") / cfg.dataset / "items_with_popularity.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
    df = _apply_b_transform(
        df,
        getattr(cfg.rerank, "b_transform", "raw"),
        saturate_at=getattr(cfg.rerank, "b_saturate_at", 0.3),
    )
    return PopularityTable(df)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(cfg: ExperimentConfig) -> Dict[str, Any]:
    started_at = datetime.now(timezone.utc).isoformat()

    # 1. Dataset
    adapter = build_dataset(cfg.dataset, **(cfg.extra or {}))
    adapter.split(seed=cfg.sample_seed, test_size=cfg.test_split)
    sampled_users = adapter.stratified_sample(cfg.sample_size, seed=cfg.sample_seed)

    # 2. Scenarios
    if cfg.scenarios is not None:
        scenario_list: List[Tuple[str, ...]] = [tuple(s) for s in cfg.scenarios]
    else:
        scenario_list = list(adapter.scenario_combinations())

    # 3. Popularity table
    pop_table = _load_or_build_popularity(cfg, adapter)
    # Pool size for LLM + rerank candidate set (M+N).
    # The final evaluated list is truncated to top_k afterwards (M).
    candidate_n = cfg.candidate_pool_size or cfg.top_k
    if candidate_n < cfg.top_k:
        raise ValueError(
            f"candidate_pool_size={candidate_n} must be >= top_k={cfg.top_k}"
        )

    # 3b. Difficulty score table (precomputed once, indexed in memory).
    from src.popularity.item_scoring import ItemScoreTable

    score_table = ItemScoreTable.from_popularity(pop_table._df)

    # 4. LLM adapter (cache + usage logger)
    cache_path = Path(cfg.cache_dir) / "llm_cache.sqlite"
    usage_path = Path(cfg.cache_dir) / "usage_log.jsonl"
    cache = LLMCache(cache_path)
    usage = UsageLogger(usage_path)
    llm: BaseLLMAdapter = build_adapter(
        provider=cfg.llm.provider,
        model=cfg.llm.model,
        base_url=cfg.llm.base_url,
        max_retries=cfg.llm.max_retries,
        timeout=cfg.llm.timeout,
        cache=cache,
        usage_logger=usage,
    )

    # 5. Prompt builder
    pb = PromptBuilder(domain=adapter.domain(), history_limit=cfg.history_limit)
    prompt_template_hash = pb.template_hash()
    # In song-level music recommendation each line is "Track - Artist"; the
    # default trailing-metadata stripper would chop off the artist, so we
    # disable it for that domain only.
    keep_full_title = adapter.domain() == "music_track"

    # 6. Output paths and progress markers
    safe_model = cfg.llm.model.replace("/", "_")
    out_root = Path(cfg.output_root) / safe_model / cfg.dataset
    raw_dir = out_root / "raw"
    rer_dir = out_root / "reranked"
    metrics_dir = out_root / "metrics"
    # Scope progress markers by experiment_id so different configs (e.g.
    # implicit vs explicit BookCrossing) don't false-positive each other.
    progress_dir = out_root / ".progress" / cfg.experiment_id
    for d in (raw_dir, rer_dir, metrics_dir, progress_dir):
        d.mkdir(parents=True, exist_ok=True)

    raw_payload: Dict[str, Any] = {}
    rer_payload: Dict[str, Any] = {}
    per_user_rows: List[Dict[str, Any]] = []

    # Resume support: when ``progress_dir`` already has ``<user>.done`` markers
    # from a prior run, reload the previously persisted JSON payloads so the
    # final write does not clobber good results with an empty dict. Without
    # this restore step the marker-based skip in pass 1 would short-circuit
    # the loop for every user but ``raw_payload`` / ``rer_payload`` would
    # stay ``{}`` and overwrite the on-disk JSON with empty objects.
    raw_persist_path = raw_dir / "recommendation_results_full.json"
    rer_persist_path = rer_dir / "recommendation_results_full.json"
    if any(progress_dir.glob("*.done")):
        try:
            if raw_persist_path.exists():
                raw_payload.update(json.loads(raw_persist_path.read_text(encoding="utf-8")) or {})
            if rer_persist_path.exists():
                rer_payload.update(json.loads(rer_persist_path.read_text(encoding="utf-8")) or {})
        except (json.JSONDecodeError, OSError):
            # Corrupt or missing JSON — fall back to a fresh run by clearing
            # progress markers so pass 1 actually executes.
            for m in progress_dir.glob("*.done"):
                m.unlink(missing_ok=True)
            raw_payload.clear()
            rer_payload.clear()

    # ---------- Pass 1: LLM calls + history popularity scores ----------
    user_scene_history_score: Dict[str, Dict[str, float]] = defaultdict(dict)
    scene_history_score_sum: Dict[str, float] = defaultdict(float)
    scene_user_count: Dict[str, int] = defaultdict(int)

    for u_idx, user in enumerate(sampled_users):
        marker = progress_dir / f"{user.user_id}.done"
        if marker.exists():
            # Reuse cached output: load JSON if present, else just count progress.
            continue

        user_block_raw: Dict[str, Any] = {
            "user_info": {**user.demographics, "user_id": user.user_id},
            "results": {},
        }
        for scenario in scenario_list:
            label = _scenario_label(scenario)
            prompt = pb.render(user, scenario, movie_count=candidate_n)
            response = llm.generate(
                prompt.text,
                temperature=cfg.llm.temperature,
                top_p=cfg.llm.top_p,
                max_tokens=cfg.llm.max_tokens,
                system=cfg.llm.system_prompt,
                experiment_id=cfg.experiment_id,
            )
            recs = parse_recommendations(
                response.text,
                max_items=candidate_n,
                history_titles=[it.title for it in user.history[: cfg.history_limit]],
                strip_trailing_metadata=not keep_full_title,
            )
            test_titles = [item.title for item in user.test_set]
            true_pref = [
                t for t in recs
                if any(titles_match(t, tt) for tt in test_titles)
            ]

            # History popularity score (to derive scene mean later).
            pop_col = _scenario_to_pop_column_name(scenario, user.demographics)
            hist_score = sum(
                pop_table.lookup(item.title, pop_col)
                for item in user.history[: cfg.history_limit]
            )
            user_scene_history_score[user.user_id][label] = hist_score
            scene_history_score_sum[label] += hist_score
            scene_user_count[label] += 1

            user_block_raw["results"][label] = {
                "scenario": list(scenario),
                "recommendations": recs,
                "true_preferred_items": true_pref,
                # Back-compat alias for legacy code expecting the old name.
                "true_preferred_movies": true_pref,
                "popularity_column_used": pop_col,
                "history_popularity_score": hist_score,
                "prompt_sha256": prompt.sha256,
                "llm_response_meta": {
                    "model": response.model,
                    "cached": response.cached,
                    "finish_reason": response.finish_reason,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "latency_ms": response.latency_ms,
                },
            }
        raw_payload[user.user_id] = user_block_raw

    # Scene mean scores.
    scene_means: Dict[str, float] = {
        label: (scene_history_score_sum[label] / scene_user_count[label])
        if scene_user_count[label] else 0.0
        for label in scene_history_score_sum
    }

    # ---------- Pass 2: dynamic-alpha rerank + metrics ----------
    for user_id, user_block in raw_payload.items():
        rer_block: Dict[str, Any] = {
            "user_info": user_block["user_info"],
            "results": {},
        }
        for label, scn in user_block["results"].items():
            recs = scn["recommendations"]
            scenario_tuple = tuple(scn["scenario"])
            pop_col = scn["popularity_column_used"]
            user_score = scn["history_popularity_score"]
            scene_mean = scene_means.get(label, 0.0)

            if not scenario_tuple:  # neutral scenario: bypass
                reranked = list(recs)[: cfg.top_k]
                dynamic_alpha = None
                rerank_scores: List[Dict[str, Any]] = []
            else:
                if cfg.rerank.disable_dynamic_alpha:
                    dynamic_alpha = cfg.rerank.alpha_base
                else:
                    dynamic_alpha = compute_dynamic_alpha(
                        user_score=user_score,
                        scene_mean_score=scene_mean,
                        alpha_base=cfg.rerank.alpha_base,
                        alpha_min=cfg.rerank.alpha_min,
                        alpha_max=cfg.rerank.alpha_max,
                        alpha_gain=cfg.rerank.alpha_gain,
                    )
                rerank_out = rerank_one_list(
                    recs,
                    pop_table.lookup_with_found,
                    pop_col,
                    alpha=dynamic_alpha,
                    base=cfg.rerank.base,
                    decay_rate=cfg.rerank.decay_rate,
                    top_n=cfg.top_k,
                    protect_top=getattr(cfg.rerank, "protect_top", 0),
                )
                reranked = rerank_out["reranked_recommendations"]
                rerank_scores = rerank_out["rerank_scores"]

            relevant_pool = scn.get("true_preferred_items") or scn.get("true_preferred_movies") or []
            metrics = compute_per_user_metrics(
                recommendations_original=recs,
                recommendations_reranked=reranked,
                relevant_pool=relevant_pool,
                test_movies_count=len(relevant_pool) or 1,
                k=cfg.top_k,
                popularity_lookup=lambda t, c, _pop=pop_table: _pop.lookup(t, c),
                pop_col=pop_col,
                rank_weight=cfg.rank_weight,
                score_table=score_table,
            )

            rer_block["results"][label] = {
                **scn,
                "reranked_recommendations": reranked,
                "rerank_scores": rerank_scores,
                "dynamic_alpha": dynamic_alpha,
                "scene_population_mean_score": scene_mean,
                "metrics": metrics,
            }

            # Flatten one parquet row.
            row = {
                "user_id": user_id,
                "scenario": label,
            }
            for side in ("original", "reranked"):
                m = metrics[side]
                for key in ("precision", "recall", "hit_rate", "mrr", "ndcg_at_k", "hits", "aplt", "aplt_neutral"):
                    if key in m:
                        row[f"{'orig' if side == 'original' else 'rer'}_{key}"] = float(m[key])
                row[f"{'orig' if side == 'original' else 'rer'}_unpopularity_total"] = float(
                    m["unpopularity"]["total_score"]
                )
                row[f"{'orig' if side == 'original' else 'rer'}_unpopularity_average"] = float(
                    m["unpopularity"]["average_score"]
                )
                if "difficulty" in m:
                    row[f"{'orig' if side == 'original' else 'rer'}_difficulty_total"] = float(
                        m["difficulty"]["total_score"]
                    )
                    row[f"{'orig' if side == 'original' else 'rer'}_difficulty_average"] = float(
                        m["difficulty"]["average_score"]
                    )
            per_user_rows.append(row)

        rer_payload[user_id] = rer_block

        # Mark per-user progress.
        marker = progress_dir / f"{user_id}.done"
        marker.touch()

    # ---------- Persist ----------
    raw_path = raw_dir / "recommendation_results_full.json"
    rer_path = rer_dir / "recommendation_results_full.json"
    raw_path.write_text(json.dumps(raw_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    rer_path.write_text(json.dumps(rer_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame(per_user_rows)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = metrics_dir / "per_user.parquet"
    try:
        df.to_parquet(parquet_path, index=False)
    except Exception:
        # pyarrow optional; fall back to CSV.
        parquet_path = metrics_dir / "per_user.csv"
        df.to_csv(parquet_path, index=False)

    aggregate = _aggregate_metrics(
        df,
        raw_payload=raw_payload,
        rer_payload=rer_payload,
        catalog_size=len(getattr(pop_table, "_df", [])) or None,
    )
    (metrics_dir / "aggregate.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    finished_at = datetime.now(timezone.utc).isoformat()
    usage_summary = usage.total_tokens(experiment_id=cfg.experiment_id)
    meta = {
        "experiment_id": cfg.experiment_id,
        "model": cfg.llm.model,
        "dataset": cfg.dataset,
        "sample_size": cfg.sample_size,
        "sample_seed": cfg.sample_seed,
        "test_split": cfg.test_split,
        "top_k": cfg.top_k,
        "candidate_pool_size": candidate_n,
        "scenarios": [list(s) for s in scenario_list],
        "rerank": cfg.rerank.__dict__,
        "started_at": started_at,
        "finished_at": finished_at,
        "total_api_calls": usage_summary["calls"],
        "total_prompt_tokens": usage_summary["prompt_tokens"],
        "total_completion_tokens": usage_summary["completion_tokens"],
        "prompt_template_hash": prompt_template_hash,
    }
    (out_root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cache.close()
    return meta


def _aggregate_metrics(
    df: pd.DataFrame,
    raw_payload: Optional[Dict[str, Any]] = None,
    rer_payload: Optional[Dict[str, Any]] = None,
    catalog_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Reduce per-user metrics to per-scenario means, plus corpus-level
    coverage and Gini computed over the union of users' recommendation lists.

    ``catalog_size`` is the size of the candidate item catalog (e.g. number
    of rows in the popularity table). When provided, Coverage = unique
    recommended items / catalog_size; otherwise the corpus-level coverage is
    skipped and only Gini is reported.
    """
    if df.empty:
        return {"n_users": 0, "scenarios": {}}
    from src.experiment.metrics import gini_index, title_key

    agg: Dict[str, Any] = {"scenarios": {}}
    for scenario, sub in df.groupby("scenario"):
        bucket: Dict[str, Any] = {"n_users": int(len(sub))}
        for col in sub.columns:
            if col in ("user_id", "scenario"):
                continue
            bucket[f"mean_{col}"] = float(sub[col].mean())

        # Corpus-level diversity: union of titles across all users for this
        # scenario, both before and after rerank. Lower Gini / higher
        # Coverage after rerank = the algorithm spreads exposure further.
        if raw_payload is not None and rer_payload is not None:
            for src_label, payload in (("orig", raw_payload), ("rer", rer_payload)):
                title_freq: Dict[str, int] = defaultdict(int)
                for uid, blk in payload.items():
                    res = blk.get("results", {}).get(scenario) or {}
                    if src_label == "orig":
                        rec_list = res.get("recommendations") or []
                    else:
                        rec_list = (
                            res.get("reranked_recommendations")
                            or res.get("recommendations")
                            or []
                        )
                    for t in rec_list:
                        k = title_key(t)
                        if k:
                            title_freq[k] += 1
                unique = len(title_freq)
                bucket[f"{src_label}_unique_recommendations"] = unique
                if catalog_size:
                    bucket[f"{src_label}_coverage"] = float(unique) / float(catalog_size)
                bucket[f"{src_label}_gini"] = float(gini_index(list(title_freq.values())))

        agg["scenarios"][str(scenario)] = bucket
    agg["n_users_total"] = int(df["user_id"].nunique())
    return agg
