"""Experiment configuration loader (YAML -> ExperimentConfig)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class RerankConfig:
    alpha_base: float = 0.7
    alpha_min: float = 0.5
    alpha_max: float = 0.9
    alpha_gain: float = 0.2
    base: float = 0.2
    decay_rate: float = 0.9
    disable_dynamic_alpha: bool = False
    # B-value transform applied to the popularity table at load time.
    #   "raw"      : keep count/column_max as-is (legacy behaviour)
    #   "rank"     : replace each cell with its rank-percentile in the column
    #                (1.0 = most popular, 0.0 = rarest). Distributes (1-B)
    #                uniformly so the long-tail bonus actually discriminates
    #                between mid and tail items.
    #   "saturate" : B' = min(B / b_saturate_at, 1). Penalizes only items
    #                above the threshold; mid/tail items get B'=B/threshold,
    #                preserving LLM order in the bulk.
    b_transform: str = "raw"
    b_saturate_at: float = 0.3
    # Protect the first ``protect_top`` slots of the LLM ranking from
    # reranking. The remaining (candidate_n - protect_top) candidates are
    # reranked among themselves and merged after the protected prefix.
    # 0 = no protection (legacy behaviour).
    protect_top: int = 0


@dataclass
class LLMConfig:
    provider: Optional[str] = None  # auto-inferred when None
    model: str = ""
    base_url: Optional[str] = None
    temperature: float = 0.7
    top_p: float = 1.0
    max_tokens: int = 1000
    system_prompt: Optional[str] = "You are a professional recommendation system."
    max_retries: int = 6
    timeout: float = 60.0


@dataclass
class ExperimentConfig:
    experiment_id: str
    dataset: str
    sample_size: int
    sample_seed: int
    test_split: float
    top_k: int                     # final list length (M)
    history_limit: int
    llm: LLMConfig
    rerank: RerankConfig
    scenarios: Optional[List[List[str]]] = None
    popularity_csv: Optional[str] = None
    output_root: str = "results"
    cache_dir: str = "data/cache"
    rank_weight: float = 0.2
    candidate_pool_size: Optional[int] = None  # M+N; defaults to top_k when None
    extra: Dict[str, Any] = field(default_factory=dict)


_REQUIRED_TOP_LEVEL = (
    "experiment_id",
    "dataset",
    "sample_size",
    "sample_seed",
    "top_k",
    "llm",
)


class ConfigValidationError(ValueError):
    pass


def _require(d: Dict[str, Any], key: str, where: str) -> Any:
    if key not in d:
        raise ConfigValidationError(f"Missing required key '{key}' in {where}")
    return d[key]


def load_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ConfigValidationError(f"{path}: top-level YAML must be a mapping")

    for k in _REQUIRED_TOP_LEVEL:
        if k not in raw:
            raise ConfigValidationError(f"{path}: missing required key '{k}'")

    llm_raw = raw.get("llm") or {}
    if not isinstance(llm_raw, dict):
        raise ConfigValidationError(f"{path}: 'llm' must be a mapping")
    if not llm_raw.get("model"):
        raise ConfigValidationError(f"{path}: 'llm.model' is required")
    llm = LLMConfig(
        provider=llm_raw.get("provider"),
        model=llm_raw["model"],
        base_url=llm_raw.get("base_url"),
        temperature=float(llm_raw.get("temperature", 0.7)),
        top_p=float(llm_raw.get("top_p", 1.0)),
        max_tokens=int(llm_raw.get("max_tokens", 1000)),
        system_prompt=llm_raw.get("system_prompt", "You are a professional recommendation system."),
        max_retries=int(llm_raw.get("max_retries", 6)),
        timeout=float(llm_raw.get("timeout", 60.0)),
    )

    rerank_raw = raw.get("rerank") or {}
    rerank = RerankConfig(
        alpha_base=float(rerank_raw.get("alpha_base", 0.7)),
        alpha_min=float(rerank_raw.get("alpha_min", 0.5)),
        alpha_max=float(rerank_raw.get("alpha_max", 0.9)),
        alpha_gain=float(rerank_raw.get("alpha_gain", 0.2)),
        base=float(rerank_raw.get("base", 0.2)),
        decay_rate=float(rerank_raw.get("decay_rate", 0.9)),
        disable_dynamic_alpha=bool(rerank_raw.get("disable_dynamic_alpha", False)),
        b_transform=str(rerank_raw.get("b_transform", "raw")).lower(),
        b_saturate_at=float(rerank_raw.get("b_saturate_at", 0.3)),
        protect_top=int(rerank_raw.get("protect_top", 0)),
    )

    scenarios = raw.get("scenarios")
    if scenarios is not None:
        if not isinstance(scenarios, list):
            raise ConfigValidationError(f"{path}: 'scenarios' must be a list of lists")
        # Normalize to list-of-lists for stable JSON serialization.
        scenarios = [list(s) for s in scenarios]

    return ExperimentConfig(
        experiment_id=str(raw["experiment_id"]),
        dataset=str(raw["dataset"]),
        sample_size=int(raw["sample_size"]),
        sample_seed=int(raw["sample_seed"]),
        test_split=float(raw.get("test_split", 0.7)),
        top_k=int(raw["top_k"]),
        history_limit=int(raw.get("history_limit", 10)),
        llm=llm,
        rerank=rerank,
        scenarios=scenarios,
        popularity_csv=raw.get("popularity_csv"),
        output_root=str(raw.get("output_root", "results")),
        cache_dir=str(raw.get("cache_dir", "data/cache")),
        rank_weight=float(raw.get("rank_weight", 0.2)),
        candidate_pool_size=(
            int(raw["candidate_pool_size"])
            if raw.get("candidate_pool_size") is not None
            else None
        ),
        extra={k: v for k, v in raw.items() if k not in {
            "experiment_id", "dataset", "sample_size", "sample_seed",
            "top_k", "history_limit", "llm", "rerank", "scenarios",
            "test_split", "popularity_csv", "output_root", "cache_dir",
            "rank_weight", "candidate_pool_size",
        }},
    )


def load_all_configs(config_dir: str | Path) -> List[ExperimentConfig]:
    p = Path(config_dir)
    if p.is_file():
        return [load_config(p)]
    cfgs: List[ExperimentConfig] = []
    for f in sorted(p.glob("*.yaml")):
        cfgs.append(load_config(f))
    return cfgs
