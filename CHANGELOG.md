# Changelog

## [Unreleased]

### Added

- **Multi-LLM adapter layer** with unified interface: ``OpenAIAdapter``,
  ``AnthropicAdapter``, ``TogetherAdapter`` (SiliconFlow). Includes SQLite
  response cache, JSONL token usage logger, exponential-backoff retries.
- **Dataset adapter layer** with unified ``UserRecord`` schema:
  - ``Ml1mAdapter`` (3 dims: gender, age_group, occupation_group)
  - ``Ml1mLegacyAdapter`` (2 dims, regression-compatible)
  - ``BookCrossingAdapter`` (age_group, country)
  - ``LfmAdapter`` (gender, age_group, country)
  - ``TenrecAdapter`` (gender, age_group)
- **Group-conditional popularity slicer** (``src/popularity/slicer.py``)
  with sparse-slice protection (slice n < 30 blends with neutral) and
  3-way cross enumeration when demographic dimensions >= 3.
- **Prompt builder** with Jinja2 templates per domain
  (``movie_en``, ``book_en``, ``music_en``, ``video_zh``) plus a shared
  glossary for demographic value translation.
- **Experiment runner** (``src/experiment/runner.py``) orchestrating
  per-(user, scenario) prompt -> LLM -> parse -> rerank -> metrics ->
  parquet/JSON persistence with per-user progress markers for resume.
- **Statistics module** (``src/stats``):
  - Paired Wilcoxon signed-rank, paired t-test, McNemar
  - Cohen's d (paired d_z) and odds ratio
  - 10000-sample percentile bootstrap CI
  - Holm-Bonferroni multiple-comparison correction
- **Aggregate reporting** (``src/reporting``):
  - cross-model x dataset summary CSV
  - per-metric Markdown breakdowns
  - Pareto frontier scatter (delta NDCG vs delta unpopularity)
  - 3-way intersectional heatmap for ML-1M and LFM-2b
- **CLI scripts**: ``download_datasets.py``, ``build_popularity.py``,
  ``run_experiments.py``, ``run_stats.py``, ``build_report.py``.
- **12 YAML configs** for the 3-LLM x 4-dataset matrix.
- Comprehensive test suite covering adapters, cache, datasets, popularity
  regression, rerank regression, prompt builder, metrics, and statistics.
- Documentation: ``README.md``, ``docs/reproducibility.md``,
  ``docs/dataset_licenses.md``, ``data/README.md``.

### Changed

- **BREAKING (semantic): rerank formula direction flipped** to actually
  perform popularity debiasing (long-tail promotion). The score is now
  ``S_i = alpha * P_i_norm + (1 - alpha) * (1 - B_i_norm)``. The legacy
  formula (``... - (1 - alpha) * (1 - B_i_norm)``) effectively *promoted*
  popular items because the negative sign flipped the unpopularity-bonus
  into a penalty. Updated in ``src/experiment/rerank.py`` and
  ``rerank_recommendations.py`` (legacy CLI). Existing
  ``reranked_JSON/`` artifacts are no longer reproducible; regression
  tests for them are skipped with explanatory comments.
- **BREAKING**: ``recommander.py`` no longer hard-codes the OpenAI API key.
  The key now must come from environment variable ``OPENAI_API_KEY``
  (with optional ``OPENAI_BASE_URL``). Use ``.env`` (loaded via
  ``python-dotenv``) for local development.
- ``rerank_recommendations.py``'s ``preference_probability``, ``safe_clip``,
  and ``compute_dynamic_alpha`` are now thin wrappers that delegate to
  ``src.experiment.rerank``. Public CLI behavior is unchanged.

### Notes

- Raw data, results, reports, and caches are git-ignored. See
  ``data/README.md`` and ``docs/dataset_licenses.md`` for download +
  citation requirements.
