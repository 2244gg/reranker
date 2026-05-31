## 1. 项目骨架与依赖

- [x] 1.1 新增 `requirements.txt`，加入 `openai>=1.0`, `anthropic`, `httpx`, `pyyaml`, `python-dotenv`, `tenacity`, `scipy`, `statsmodels`, `pandas`, `pyarrow`, `numpy`, `scikit-learn`, `matplotlib`, `tqdm`
- [x] 1.2 创建 `src/` 包结构：`src/__init__.py`、`src/llm_adapters/`、`src/datasets/`、`src/popularity/`、`src/experiment/`、`src/stats/`、`src/reporting/`，每子包带 `__init__.py`
- [x] 1.3 创建 `configs/`、`scripts/`、`data/raw/`、`data/processed/`、`data/cache/`、`results/`、`reports/`、`tests/` 目录
- [x] 1.4 创建 `.env.example` 含 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`ANTHROPIC_API_KEY`、`ANTHROPIC_BASE_URL`、`SILICONFLOW_API_KEY` 占位
- [x] 1.5 把 `recommander.py` 第 16-18 行硬编码 key 替换为 `os.getenv("OPENAI_API_KEY")` + `load_dotenv()`，同步更新 `.gitignore` 加 `.env`、`data/cache/`、`data/raw/`、`results/`、`reports/`

## 2. LLM 适配层（capability: llm-adapters）

- [x] 2.1 实现 `src/llm_adapters/base.py`：`BaseLLMAdapter` 抽象类、`LLMResponse` dataclass、`AuthError`/`RateLimitError` 异常
- [x] 2.2 实现 `src/llm_adapters/cache.py`：基于 SQLite 的 `LLMCache(get/set/key)`，key = sha256(model+prompt+kwargs)
- [x] 2.3 实现 `src/llm_adapters/usage.py`：JSONL 追加写 `data/cache/usage_log.jsonl`，含 experiment_id 注入
- [x] 2.4 实现 `src/llm_adapters/openai_adapter.py`：兼容 laozhang.ai 与官方 OpenAI，含 tenacity 重试装饰器
- [x] 2.5 实现 `src/llm_adapters/anthropic_adapter.py`：调 Anthropic Messages API，把响应映射成 `LLMResponse`
- [x] 2.6 实现 `src/llm_adapters/together_adapter.py`：OpenAI 兼容协议，默认 base_url 指向 SiliconFlow
- [x] 2.7 实现 `src/llm_adapters/factory.py`：`build_adapter(provider, model, **kwargs)` 工厂函数
- [x] 2.8 编写 `tests/test_llm_cache.py` 验证缓存命中/失效逻辑（mock provider）
- [x] 2.9 编写 `tests/test_llm_adapters_smoke.py` 用 `responses` 库 mock HTTP，确认 retries/auth_error 行为

## 3. 数据集适配层（capability: dataset-adapters）

- [x] 3.1 实现 `src/datasets/base.py`：`InteractionItem`、`UserRecord` dataclass、`BaseDatasetAdapter` 抽象（`load`、`split`、`stratified_sample`、`positive_feedback_predicate`、`demographic_schema`、`domain`）
- [x] 3.2 实现 `src/datasets/ml1m.py`：把 `recommander.py` 现有逻辑迁出，新增 `occupation_group` 维度（21 类映射成 4 组：student / professional / service / other），保留 `Ml1mLegacyAdapter` 仅用 gender+age 用作回归测试
- [x] 3.3 实现 `src/datasets/bookcrossing.py`：解析 `BX-Users.csv` 与 `BX-Book-Ratings.csv`，从 `Location` 字段抽 country，过滤 ≥10 次交互用户，age 切 4 组（<18 / 18-34 / 35-54 / 55+）
- [x] 3.4 实现 `src/datasets/lfm.py`：解析 LFM-2b 的 `users.tsv` 与 listening counts，country 取 top-10 + other，过滤 ≥20 次交互
- [x] 3.5 实现 `src/datasets/tenrec.py`：解析 Tenrec 短视频曝光日志（`watch_time / video_length > 0.5` 作正反馈），过滤 ≥10 次曝光
- [x] 3.6 实现 `src/datasets/factory.py`：`build_dataset(name)`，名称 `ml1m`/`bookcrossing`/`lfm`/`tenrec`
- [x] 3.7 编写 `scripts/download_datasets.py`：自动下载 ML-1M（已在 `ml-1m/` 跳过），打印 BookCrossing/LFM-2b/Tenrec 的官方下载步骤与许可声明
- [x] 3.8 创建 `data/README.md`：列出各数据集官方链接、引用、许可证、本地路径约定
- [x] 3.9 编写 `tests/test_datasets_schema.py`：每 adapter 加载 100 用户，断言 `demographics` 包含规定维度、`history`/`test_set` 非空

## 4. Popularity slicing（capability: popularity-slicing）

- [x] 4.1 实现 `src/popularity/slicer.py`：`PopularitySlicer(adapter).build()` 自动遍历 demographic 笛卡尔积生成切片列
- [x] 4.2 实现稀疏切片插值逻辑（`slice_n < 30` 时与 `neutral_All` 加权混合）
- [x] 4.3 实现仅在维度 ≥ 3 时生成三维 cross 列的分支
- [x] 4.4 编写 `scripts/build_popularity.py`：`--dataset ml1m|bookcrossing|lfm|tenrec` 参数化生成
- [x] 4.5 编写 `tests/test_popularity_regression.py`：用 ML-1M 旧 schema（gender + age）跑 slicer，与既有 `movies_with_popularity.csv` 数值比对，容差 1e-6

## 5. Rerank 层适配（兼容现有逻辑）

- [x] 5.1 抽离 `rerank_recommendations.py` 核心公式到 `src/experiment/rerank.py`，参数化 `popularity_columns_for(scenario, demographics)` 函数
- [x] 5.2 把 `SCENARIOS = ("neutral", "age", "gender", "cross")` 改成由数据集 `demographic_schema` 动态生成的列表
- [x] 5.3 保留旧脚本 `rerank_recommendations.py` 作 thin CLI wrapper 调用新模块，保证旧调用兼容
- [x] 5.4 编写 `tests/test_rerank_regression.py`：用 ML-1M 旧产物跑新代码，输出与 `reranked_JSON/` 旧产物 diff = 0

## 6. Prompt 模板系统

- [x] 6.1 创建 `prompts/`，每域一份模板：`prompts/movie_en.j2`、`prompts/book_en.j2`、`prompts/music_en.j2`、`prompts/video_zh.j2`
- [x] 6.2 模板支持 `{demographics}` 变量动态拼接（无 demographic 时输出 neutral）
- [x] 6.3 实现 `src/experiment/prompt_builder.py`：根据数据集域 + 场景 + UserRecord 渲染 prompt，输出 prompt + sha256 hash
- [x] 6.4 中文 video 模板与英文模板的 demographics 描述对齐（gender/age/occupation 等术语对照表写在 `prompts/glossary.yaml`）
- [x] 6.5 编写 `tests/test_prompt_builder.py`：验证同一用户在 4 场景下 prompt 仅 demographics 段不同

## 7. 评估指标统一

- [x] 7.1 抽离 `evaluate_recommendations_top15.py` 到 `src/experiment/metrics.py`：`precision_at_k`、`recall_at_k`、`hit_rate_at_k`、`mrr`、`ndcg_at_k`、`unpopularity_score`
- [x] 7.2 实现 `compute_per_user_metrics(user_record, original, reranked, popularity_table) -> Dict`，每用户每场景产出标准字典
- [x] 7.3 增加 `unpopularity_score` 入 metrics（来自 `score_top15_dual_lists.py` 的公式 `(1−pop)·10·(1+w·rank)`）
- [x] 7.4 编写 `tests/test_metrics.py`：用合成数据验证 ndcg/mrr 公式正确

## 8. 实验编排器（capability: experiment-runner）

- [x] 8.1 实现 `src/experiment/config.py`：YAML 解析 + pydantic schema 校验 → `ExperimentConfig`
- [x] 8.2 实现 `src/experiment/runner.py`：`run(cfg)` 主循环 = 加载数据集 → 抽样 → 切分 → 对每用户每场景 → 调 LLM → 解析 → rerank → 评估 → 落盘
- [x] 8.3 实现进度标记：`results/<m>/<d>/.progress/<user_id>.done` 写入与读取
- [x] 8.4 实现 per-user metrics 落 parquet：`results/<m>/<d>/metrics/per_user.parquet`
- [x] 8.5 实现 aggregate 汇总：`metrics/aggregate.json` 含每场景 macro/micro 指标
- [x] 8.6 实现 `meta.json` 写入（model_version、prompt_template_hash、started_at、finished_at、total_tokens）
- [x] 8.7 编写 12 份 YAML 配置：`configs/{gpt41mini,claude35sonnet,qwen25_72b}_{ml1m,bookcrossing,lfm,tenrec}.yaml`
- [x] 8.8 实现 `scripts/run_experiments.py`：`--config-dir`、`--filter "model=...,dataset=..."`、`--dry-run`、`--sample-size` 参数
- [ ] 8.9 Pilot 跑：每 cell 200 用户跑通整链路，校验产物结构

## 9. 统计显著性模块（capability: statistical-testing）

- [x] 9.1 实现 `src/stats/tests.py`：`paired_wilcoxon`、`paired_t_test`、`mcnemar_test`、`bootstrap_ci(deltas, n_boot=10000, seed)`
- [x] 9.2 实现 `src/stats/effect_size.py`：`cohens_d`、`odds_ratio`、`label_effect_size`
- [x] 9.3 实现 `src/stats/correction.py`：Holm-Bonferroni 多重比较校正
- [x] 9.4 实现 `src/stats/runner.py`：扫描 `results/*/*/metrics/per_user.parquet`，对每 (model, dataset, scenario, metric) 单元跑全套检验
- [x] 9.5 实现 `src/stats/runner.py` 输出 `reports/significance_tests.json` schema 完整
- [x] 9.6 实现 `scripts/run_stats.py`：CLI 包装
- [x] 9.7 编写 `tests/test_stats.py`：合成数据验证 wilcoxon/t/cohens_d/CI 与 scipy 直接调用一致

## 10. 聚合报告（capability: aggregate-reporting）

- [x] 10.1 实现 `src/reporting/cross_table.py`：从 `significance_tests.json` 生成 `reports/cross_model_dataset_summary.csv`
- [x] 10.2 实现 `src/reporting/by_scenario.py`：每指标一个 Markdown 文件 `reports/by_scenario_<metric>.md`
- [x] 10.3 实现 `src/reporting/pareto.py`：matplotlib 散点图 `reports/pareto_frontier.png`
- [x] 10.4 实现 `src/reporting/heatmap.py`：ML-1M 与 LFM-2b 的 intersectional heatmap
- [x] 10.5 报告头部插入生成时间 + git commit hash（用 `subprocess` 取）
- [x] 10.6 实现 `scripts/build_report.py`：`--cross-table --pareto --heatmap` 参数

## 11. 端到端集成与论文物料

- [ ] 11.1 接 Phase 1 全量回归：跑 ML-1M + GPT-4.1-mini 1500 用户，与既有 `JSON/` `reranked_JSON/` 数值核对（应一致到 1e-6 或仅因抽样种子差异）
- [ ] 11.2 跑 Phase 2 BookCrossing × 3 模型 × 1500 用户
- [ ] 11.3 跑 Phase 2 LFM-2b × 3 模型 × 1500 用户
- [ ] 11.4 跑 Phase 2 Tenrec × 3 模型 × 1500 用户
- [ ] 11.5 跑 Phase 2 ML-1M × Claude/Qwen × 1500 用户
- [ ] 11.6 跑 `scripts/run_stats.py` 汇总全部 12 cells
- [ ] 11.7 跑 `scripts/build_report.py` 生成全部论文表 + 图
- [ ] 11.8 写 `reports/findings.md`：汇总核心结论（哪些 model+dataset 上 rerank 显著有效、intersectional 偏置最强组合等）

## 12. 文档与可复现

- [x] 12.1 更新顶层 `README.md`（若不存在则创建）：项目目标、目录结构、复现步骤、引用
- [x] 12.2 编写 `docs/reproducibility.md`：从零到产出报告的完整命令序列
- [x] 12.3 编写 `docs/dataset_licenses.md`：每数据集许可证与引用要求
- [x] 12.4 在 `tasks.md` 之外维护 `CHANGELOG.md` 记录重大变更
- [ ] 12.5 全 `pytest -q` 通过，CI 友好（无网络依赖的单测全绿）
