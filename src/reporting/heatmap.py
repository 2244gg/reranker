"""Intersectional 3-way heatmaps for datasets with >=3 demographic dimensions."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def build_intersectional_heatmap(
    results_root: str | Path,
    dataset: str,
    output_path: str | Path,
    *,
    metric: str = "rer_unpopularity_average",
    baseline_metric: str = "orig_unpopularity_average",
) -> Path:
    """Heatmap of mean delta across the 3-way Cartesian product of demographics.

    Walks ``results/<model>/<dataset>/metrics/per_user.parquet`` for the given
    dataset and aggregates over all models. Requires the dataset to have a
    3-way cross scenario (e.g. ``gender+age_group+occupation_group``).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(results_root)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    frames: List[pd.DataFrame] = []
    for model_dir in sorted(p for p in root.glob("*") if p.is_dir()):
        parquet = model_dir / dataset / "metrics" / "per_user.parquet"
        if not parquet.exists():
            parquet = parquet.with_suffix(".csv")
            if not parquet.exists():
                continue
            df = pd.read_csv(parquet)
        else:
            df = pd.read_parquet(parquet)
        df["model"] = model_dir.name
        frames.append(df)

    if not frames:
        # Empty figure to satisfy contract.
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No data for dataset={dataset}", ha="center", va="center")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out

    full = pd.concat(frames, ignore_index=True)
    # Three-way scenarios contain '+' twice in label.
    full = full[full["scenario"].str.count("\\+") == 2]

    if full.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, f"No 3-way scenarios in {dataset}", ha="center", va="center")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        return out

    full["delta"] = full[metric] - full[baseline_metric]
    pivot = (
        full.groupby("scenario")["delta"]
        .mean()
        .reset_index()
        .sort_values("scenario")
    )

    # Render as a flat heatmap (1 column).
    matrix = pivot["delta"].to_numpy().reshape(-1, 1)
    fig, ax = plt.subplots(figsize=(4, max(4, len(pivot) * 0.3)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-abs(matrix).max(), vmax=abs(matrix).max())
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot["scenario"].tolist(), fontsize=7)
    ax.set_xticks([])
    ax.set_title(f"{dataset}: 3-way intersectional delta\n({metric} - {baseline_metric})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
