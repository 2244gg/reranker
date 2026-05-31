"""Generate cross-model x dataset summary CSV."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.stats.correction import label_significance


def build_cross_table(
    significance_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Build a wide CSV: rows = (scenario, metric), cols = (model, dataset).

    Cells contain ``delta +/- ci_half [sig]``.
    """
    sig_path = Path(significance_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(sig_path.read_text(encoding="utf-8"))
    results: Dict[str, Dict[str, Any]] = payload["results"]

    rows: Dict[Tuple[str, str], Dict[Tuple[str, str], str]] = {}
    col_keys = set()

    for key, entry in results.items():
        try:
            model, dataset, scenario, metric = key.split("|", 3)
        except ValueError:
            continue
        delta = entry.get("delta", 0.0)
        ci = entry.get("ci_95") or [None, None]
        if ci and ci[0] is not None and ci[1] is not None:
            half = (ci[1] - ci[0]) / 2.0
            ci_str = f"+/-{half:.4f}"
        else:
            ci_str = ""
        sig = entry.get("significance_label") or label_significance(
            entry.get("p_value_holm", entry.get("p_value", 1.0))
        )
        cell = f"{delta:+.4f} {ci_str} [{sig}]".strip()
        rows.setdefault((scenario, metric), {})[(model, dataset)] = cell
        col_keys.add((model, dataset))

    sorted_cols = sorted(col_keys)
    sorted_rows = sorted(rows.keys())

    header = ["scenario", "metric"] + [f"{m}|{d}" for m, d in sorted_cols]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for (scn, met) in sorted_rows:
            row = [scn, met]
            for col in sorted_cols:
                row.append(rows[(scn, met)].get(col, ""))
            w.writerow(row)
    return out_path
