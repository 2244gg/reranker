"""Aggregate per-user metric parquets across the experiment matrix and run
significance tests on each (model, dataset, scenario, metric) cell.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

from .correction import holm_bonferroni, label_significance
from .effect_size import cohens_d, label_effect_size, odds_ratio
from .tests import bootstrap_ci, mcnemar_test, paired_t_test, paired_wilcoxon


CONTINUOUS_METRICS = (
    "precision",
    "recall",
    "mrr",
    "ndcg_at_k",
    "unpopularity_total",
    "unpopularity_average",
)
BINARY_METRICS = ("hit_rate",)
ALL_METRICS = CONTINUOUS_METRICS + BINARY_METRICS


@dataclass
class CellKey:
    model: str
    dataset: str
    scenario: str

    def as_str(self) -> str:
        return f"{self.model}|{self.dataset}|{self.scenario}"


def _flatten_metric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure parquet has columns: scenario, orig_<metric>, rer_<metric>.

    Per-user-per-scenario rows are expected. Unpopularity sub-fields are
    flattened from the dict structure.
    """
    if df.empty:
        return df
    if "scenario" not in df.columns:
        raise ValueError("Per-user metrics parquet missing 'scenario' column.")
    return df


def run_cell_tests(
    df: pd.DataFrame,
    *,
    n_boot: int = 10000,
    boot_seed: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Run all metric tests for one (model, dataset) DataFrame.

    Expected columns per row: ``scenario`` plus ``orig_<metric>`` /
    ``rer_<metric>`` for every metric in ``ALL_METRICS``.

    Returns ``{f"{scenario}|{metric}": {...test results...}}``.
    """
    out: Dict[str, Dict[str, Any]] = {}
    raw_p_values: List[float] = []
    keys_in_order: List[tuple] = []

    for scenario, sub in df.groupby("scenario"):
        for metric in ALL_METRICS:
            orig_col = f"orig_{metric}"
            rer_col = f"rer_{metric}"
            if orig_col not in sub or rer_col not in sub:
                continue

            orig = sub[orig_col].astype(float).to_numpy()
            rer = sub[rer_col].astype(float).to_numpy()
            mean_o = float(np.mean(orig)) if orig.size else 0.0
            mean_r = float(np.mean(rer)) if rer.size else 0.0
            delta = mean_r - mean_o

            entry: Dict[str, Any] = {
                "n": int(orig.size),
                "mean_orig": mean_o,
                "mean_rer": mean_r,
                "delta": delta,
            }

            if metric in BINARY_METRICS:
                bin_o = (orig > 0.5).astype(int)
                bin_r = (rer > 0.5).astype(int)
                m = mcnemar_test(bin_o, bin_r)
                or_value = odds_ratio(bin_o, bin_r)
                entry.update(
                    {
                        "test": "mcnemar",
                        "statistic": m.statistic,
                        "p_value": m.p_value,
                        "odds_ratio": or_value,
                        "extras": m.extras,
                    }
                )
                raw_p_values.append(m.p_value)
            else:
                w = paired_wilcoxon(orig, rer)
                t = paired_t_test(orig, rer)
                d_eff = cohens_d(orig, rer)
                mean_delta, ci_low, ci_high = bootstrap_ci(
                    rer - orig, n_boot=n_boot, seed=boot_seed
                )
                entry.update(
                    {
                        "test": "wilcoxon",
                        "wilcoxon_statistic": w.statistic,
                        "wilcoxon_p": w.p_value,
                        "t_statistic": t.statistic,
                        "t_p": t.p_value,
                        "p_value": w.p_value,
                        "cohens_d": d_eff,
                        "effect_size_label": label_effect_size(d_eff),
                        "ci_95": [ci_low, ci_high],
                        "boot_mean": mean_delta,
                    }
                )
                raw_p_values.append(w.p_value)

            keys_in_order.append((scenario, metric))
            out[f"{scenario}|{metric}"] = entry

    # Holm correction within this cell.
    adj = holm_bonferroni(raw_p_values)
    for (scenario, metric), p_holm in zip(keys_in_order, adj):
        key = f"{scenario}|{metric}"
        out[key]["p_value_holm"] = p_holm
        out[key]["significance_label"] = label_significance(p_holm)
        out[key]["significant_after_correction"] = p_holm < 0.05

    return out


def run_all(
    results_root: str | Path,
    output_path: str | Path,
    *,
    n_boot: int = 10000,
) -> Dict[str, Any]:
    """Walk ``results/<model>/<dataset>/metrics/per_user.parquet`` and
    produce ``reports/significance_tests.json``.
    """
    root = Path(results_root)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    aggregated: Dict[str, Dict[str, Any]] = {}
    for model_dir in sorted(d for d in root.glob("*") if d.is_dir()):
        for dataset_dir in sorted(d for d in model_dir.glob("*") if d.is_dir()):
            parquet = dataset_dir / "metrics" / "per_user.parquet"
            if not parquet.exists():
                continue
            df = pd.read_parquet(parquet)
            df = _flatten_metric_columns(df)
            cell = run_cell_tests(df, n_boot=n_boot, boot_seed=42)
            for scn_metric, payload in cell.items():
                scenario, metric = scn_metric.split("|", 1)
                key = f"{model_dir.name}|{dataset_dir.name}|{scenario}|{metric}"
                aggregated[key] = payload

    out_payload = {
        "version": 1,
        "n_boot": n_boot,
        "results": aggregated,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, ensure_ascii=False, indent=2)
    return out_payload
