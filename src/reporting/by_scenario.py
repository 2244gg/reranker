"""Per-metric scenario breakdown Markdown reports."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Set

from src.stats.correction import label_significance


def build_by_scenario_reports(
    significance_path: str | Path,
    output_dir: str | Path,
) -> List[Path]:
    sig_path = Path(significance_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(sig_path.read_text(encoding="utf-8"))
    results: Dict[str, Dict[str, Any]] = payload["results"]

    by_metric: Dict[str, Dict] = {}
    for key, entry in results.items():
        try:
            model, dataset, scenario, metric = key.split("|", 3)
        except ValueError:
            continue
        by_metric.setdefault(metric, {})
        by_metric[metric].setdefault(scenario, {})
        by_metric[metric][scenario][(dataset, model)] = entry

    written: List[Path] = []
    for metric, scn_table in by_metric.items():
        lines = [f"# Metric: `{metric}`", ""]
        all_pairs: Set = set()
        for scn, mp in scn_table.items():
            all_pairs.update(mp.keys())
        sorted_pairs = sorted(all_pairs)
        header_cells = ["scenario"] + [f"{d}|{m}" for d, m in sorted_pairs]
        lines.append("| " + " | ".join(header_cells) + " |")
        lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")
        for scn in sorted(scn_table.keys()):
            row_cells = [scn]
            for pair in sorted_pairs:
                e = scn_table[scn].get(pair)
                if e is None:
                    row_cells.append("")
                    continue
                delta = e.get("delta", 0.0)
                p = e.get("p_value_holm", e.get("p_value", 1.0))
                sig = e.get("significance_label") or label_significance(p)
                row_cells.append(f"{delta:+.4f} [{sig}]")
            lines.append("| " + " | ".join(row_cells) + " |")
        out_path = out_dir / f"by_scenario_{metric}.md"
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(out_path)
    return written
