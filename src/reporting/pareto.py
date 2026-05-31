"""Pareto frontier scatter plot."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def build_pareto(
    significance_path: str | Path,
    output_path: str | Path,
    *,
    x_metric: str = "ndcg_at_k",
    y_metric: str = "unpopularity_average",
) -> Path:
    """Scatter (delta_<x_metric>, delta_<y_metric>) per (model, dataset, scenario)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sig_path = Path(significance_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(sig_path.read_text(encoding="utf-8"))
    results: Dict[str, Dict] = payload["results"]

    # Collect (model, dataset, scenario) -> (delta_x, delta_y).
    points: Dict[tuple, Dict[str, float]] = {}
    for key, entry in results.items():
        try:
            model, dataset, scenario, metric = key.split("|", 3)
        except ValueError:
            continue
        if metric not in (x_metric, y_metric):
            continue
        bucket = points.setdefault((model, dataset, scenario), {})
        bucket[metric] = float(entry.get("delta", 0.0))

    # Color by model, marker by dataset.
    models: List[str] = sorted({k[0] for k in points})
    datasets: List[str] = sorted({k[1] for k in points})
    cmap = plt.get_cmap("tab10")
    color_for = {m: cmap(i % 10) for i, m in enumerate(models)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*"]
    marker_for = {d: markers[i % len(markers)] for i, d in enumerate(datasets)}

    fig, ax = plt.subplots(figsize=(7, 5))
    for (model, dataset, scenario), vals in points.items():
        x = vals.get(x_metric)
        y = vals.get(y_metric)
        if x is None or y is None:
            continue
        ax.scatter(
            x, y,
            color=color_for[model],
            marker=marker_for[dataset],
            s=70,
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )

    ax.axhline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.5)
    ax.set_xlabel(f"delta {x_metric}")
    ax.set_ylabel(f"delta {y_metric}")
    ax.set_title("Pareto frontier: retrieval quality vs bias mitigation")

    # Build legend: 2 sub-legends merged manually.
    from matplotlib.lines import Line2D

    color_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=color_for[m],
               markeredgecolor="black", label=m, markersize=8)
        for m in models
    ]
    marker_handles = [
        Line2D([0], [0], marker=marker_for[d], color="grey",
               markerfacecolor="lightgrey", markeredgecolor="black",
               label=d, markersize=8, linestyle="")
        for d in datasets
    ]
    legend1 = ax.legend(handles=color_handles, title="Model", loc="upper left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=marker_handles, title="Dataset", loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
