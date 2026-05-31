# LLM Recommender Bias Audit & Group-Conditional Debiasing

A research framework for auditing **demographic bias** in LLM-based
recommendation pipelines and applying **group-conditional, user-adaptive
post-hoc reranking** to mitigate it.

The framework is designed to validate findings across **multiple LLMs** and
**multiple domains** with statistically rigorous comparisons.

---

## Research questions

1. Does exposing demographic attributes (gender / age / occupation / country)
   in an LLM prompt push recommendations toward group stereotypes?
2. How do single-attribute and **intersectional** combinations interact?
3. Can a lightweight, training-free reranker (group-conditional popularity
   debiasing + user-adaptive alpha) reduce bias **without sacrificing
   retrieval quality**?
4. Are the effects **consistent across LLMs and domains**, with statistical
   significance after multiple-comparison correction?

## Experiment matrix

|                  | ML-1M (movie) | BookCrossing (book) | LFM-2b (music) | Tenrec (video) |
|------------------|---------------|---------------------|----------------|----------------|
| GPT-4.1-mini     | YES           | YES                 | YES            | YES            |
| Claude-3.5-Sonnet| YES           | YES                 | YES            | YES            |
| Qwen2.5-72B      | YES           | YES                 | YES            | YES            |

Each cell: 1500 stratified-sampled users x scenarios derived from the
dataset's demographic schema. ML-1M and LFM-2b additionally enable 3-way
intersectional analysis.

| Dataset       | Demographics                              | Dims | Scenarios |
|---------------|-------------------------------------------|------|-----------|
| ML-1M         | gender, age_group, occupation_group       | 3    | 8         |
| BookCrossing  | age_group, country                        | 2    | 4         |
| LFM-2b        | gender, age_group, country                | 3    | 8         |
| Tenrec        | gender, age_group                         | 2    | 4         |

## Repository layout

```
src/                          Library code
  llm_adapters/               Unified OpenAI / Anthropic / SiliconFlow interface
  datasets/                   Per-dataset adapters -> UserRecord
  popularity/                 Group-conditional popularity slicer
  experiment/                 Prompt builder, rerank engine, metrics, runner
  stats/                      Significance tests, effect size, multi-comparison
  reporting/                  Cross tables, Pareto plots, heatmaps
configs/                      One YAML per (model, dataset) experiment
prompts/                      Domain-specific Jinja2 templates + glossary
scripts/                      CLIs for downloading data, building popularity,
                              running experiments, stats, and reports
tests/                        Unit / regression tests
data/                         Raw + processed data (NOT committed)
results/                      Per-experiment outputs (NOT committed)
reports/                      Cross-experiment summaries + figures
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Configure API keys
copy .env.example .env
# then edit .env to fill in OPENAI_API_KEY / ANTHROPIC_API_KEY / SILICONFLOW_API_KEY
```

Each provider can use either the official endpoint or an OpenAI-compatible
proxy (e.g. ``laozhang.ai`` for OpenAI / Anthropic). See ``.env.example`` for
all environment variables.

## Datasets

ML-1M is shipped under ``ml-1m/``. Other datasets must be obtained from their
respective official sources due to license restrictions:

```bash
python scripts/download_datasets.py --dataset all
```

This prints exact instructions and verifies which files exist.

See ``data/README.md`` for full citation requirements.

## Reproducing the pipeline

```bash
# 1. Build group-conditional popularity tables
python scripts/build_popularity.py --dataset ml1m
python scripts/build_popularity.py --dataset bookcrossing
python scripts/build_popularity.py --dataset lfm
python scripts/build_popularity.py --dataset tenrec

# 2. Run experiments (full matrix)
python scripts/run_experiments.py --config-dir configs/

# 2a. Or filter a subset
python scripts/run_experiments.py --filter "model=gpt-4.1-mini,dataset=ml1m"

# 2b. Or run a small pilot
python scripts/run_experiments.py --filter "model=gpt-4.1-mini,dataset=ml1m" --sample-size 200

# 3. Statistical significance tests
python scripts/run_stats.py --results-root results --output reports/significance_tests.json

# 4. Reports (cross-table, by-metric, Pareto, heatmaps)
python scripts/build_report.py --all
```

Tests:

```bash
python -m pytest tests/ -v
```

Tests for non-shipped datasets (BookCrossing / LFM / Tenrec) auto-skip if the
raw files are absent.

## Reranking formula (recap)

For each candidate item ``i`` in a length-N LLM list:

```
  S_i = alpha * P_i_norm  +  (1 - alpha) * (1 - B_i_norm)
```

- ``P_i_norm`` is a geometric-decay rank score (rank 1 -> ~1.0, rank N -> base).
- ``B_i_norm`` is the **slice-conditioned** item popularity.
- ``alpha`` is **user-adaptive**: heavy-mainstream users get aggressive
  debiasing, niche users keep the LLM's order. Bounded by
  ``[alpha_min, alpha_max]``.

The neutral scenario bypasses reranking, providing the unconditioned baseline.

## Statistics

Per (model, dataset, scenario, metric) cell:

- Continuous metrics (NDCG, MRR, Precision, Recall, unpopularity) -> paired
  Wilcoxon signed-rank + paired t-test + Cohen's d + 10000-sample bootstrap
  95% CI.
- Binary metrics (HitRate@k) -> McNemar test + odds ratio.
- All p-values are corrected via **Holm-Bonferroni** within each
  (model, dataset) cell.

Output: ``reports/significance_tests.json`` with the standard schema.

## License

Code: see ``LICENSE`` (TBD by maintainer).
Datasets retain their original licenses; see ``data/README.md``.
