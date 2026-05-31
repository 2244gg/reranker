# End-to-end reproducibility

This document lists the exact command sequence to reproduce the published
results of this project, plus the random seeds, prompt hashes, and data
versions involved.

## 0. Environment

- Python 3.10+
- A POSIX or Windows shell with ``curl`` available
- Approximately 1.5 GB free disk for processed data + caches
- Internet access for LLM APIs

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

Set provider keys in ``.env``:

```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.laozhang.ai/v1   # or official
ANTHROPIC_API_KEY=sk-ant-...
SILICONFLOW_API_KEY=sk-...
```

## 1. Download datasets

```bash
python scripts/download_datasets.py --dataset all
```

ML-1M is in-repo. BookCrossing, LFM-2b, Tenrec require manual download per
the printed instructions. Place the files under:

- ``data/raw/bookcrossing/{BX-Users.csv, BX-Books.csv, BX-Book-Ratings.csv}``
- ``data/raw/lfm-2b/{users.tsv, user_artist_playcount.tsv, artists.tsv}``
- ``data/raw/tenrec/{user_features.csv, video_impressions.csv, items.csv}``

## 2. Build group-conditional popularity tables

```bash
python scripts/build_popularity.py --dataset ml1m
python scripts/build_popularity.py --dataset bookcrossing
python scripts/build_popularity.py --dataset lfm
python scripts/build_popularity.py --dataset tenrec
```

For ML-1M, the legacy 2-D popularity table is in
``movies_with_popularity.csv`` (already shipped). The new 3-D table is at
``data/processed/ml1m/items_with_popularity.csv``.

A regression test confirms the 2-D legacy slicer reproduces shipped values
within Pearson correlation >= 0.95:

```bash
python -m pytest tests/test_popularity_regression.py -v
```

## 3. Run experiments

The experiment matrix is encoded in ``configs/{model}_{dataset}.yaml``.

### 3a. Smoke pilot (200 users / cell)

```bash
python scripts/run_experiments.py --config-dir configs/ --sample-size 200
```

### 3b. Full matrix (1500 users / cell)

```bash
python scripts/run_experiments.py --config-dir configs/
```

### 3c. Run a subset

```bash
python scripts/run_experiments.py --filter "model=gpt-4.1-mini,dataset=ml1m"
```

Resume on interrupt: progress is tracked per user under
``results/<model>/<dataset>/.progress/<user_id>.done``. Re-running the same
command picks up where it left off; LLM responses are cached in
``data/cache/llm_cache.sqlite`` keyed by (model, sha256(prompt+kwargs)).

## 4. Statistical significance

```bash
python scripts/run_stats.py \
    --results-root results \
    --output reports/significance_tests.json \
    --n-boot 10000
```

This runs paired Wilcoxon, paired t-test, McNemar (for HitRate), Cohen's d,
and bootstrap 95% CI for every (model, dataset, scenario, metric) cell, then
applies Holm-Bonferroni correction within each (model, dataset) cell.

## 5. Aggregate reports

```bash
python scripts/build_report.py --all
```

Outputs (under ``reports/``):

- ``cross_model_dataset_summary.csv`` -- wide table for the paper
- ``by_scenario_<metric>.md`` -- one Markdown table per metric
- ``pareto_frontier.png`` -- delta NDCG vs delta unpopularity scatter
- ``intersectional_heatmap_ml1m.png`` and ``intersectional_heatmap_lfm.png``
  -- 3-way demographic intersectional bias maps

Each Markdown / CSV report begins with a comment header containing the
generation timestamp, source path, and git commit short hash.

## 6. Run all unit tests

```bash
python -m pytest tests/ -v
```

Tests touching missing external datasets are auto-skipped. The required
"green" tests are:

- ``tests/test_llm_cache.py``
- ``tests/test_llm_adapters_smoke.py``
- ``tests/test_datasets_schema.py`` (ML-1M parts)
- ``tests/test_popularity_regression.py`` (uses shipped CSV)
- ``tests/test_rerank_regression.py`` (uses shipped JSON / reranked_JSON)
- ``tests/test_prompt_builder.py``
- ``tests/test_metrics.py``
- ``tests/test_stats.py``

## Reproducibility metadata

Every experiment run writes ``results/<model>/<dataset>/meta.json`` with:

- ``experiment_id``, ``sample_seed``, ``test_split``, ``top_k``
- ``model``, ``rerank`` parameters
- ``started_at`` / ``finished_at`` (UTC ISO-8601)
- ``total_api_calls``, ``total_prompt_tokens``, ``total_completion_tokens``
- ``prompt_template_hash`` (sha256 of the active Jinja2 template + glossary)

Combined with the YAML config and ``movies_with_popularity.csv``, this is
sufficient for an independent third party to re-execute and compare.

## Random seeds policy

- ``sample_seed`` controls user sampling AND the train/test split (passed to
  both ``adapter.split`` and ``adapter.stratified_sample``). Default: 42.
- LLM calls run at ``temperature=0.7``; while individual outputs are not
  deterministic, the cache layer makes the *experimental* run reproducible
  once a result has been obtained at least once.
- Bootstrap confidence intervals use ``seed=42`` by default (configurable
  via ``--boot-seed`` if added to the runner CLI).
