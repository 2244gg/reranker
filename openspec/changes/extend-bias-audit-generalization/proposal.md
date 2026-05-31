## Why

当前研究只在 1 个 LLM（GPT-4.1-mini）+ 1 个数据集（ML-1M）上验证 demographic bias 审计与去偏 rerank 框架，泛化性不足以支撑论文。审稿人会质疑：结论是否仅是该 LLM 的特殊行为？是否仅是电影域、英文语料、ML-1M 用户结构的偶然结果？同时缺统计显著性检验，delta 数值无法证明真实效应。本提案补足上述三大缺口，使框架具备跨模型、跨域、跨语种、可统计验证的强证据。

## What Changes

- 新增 **3-LLM 适配层**：GPT-4.1-mini（保留）、Claude 3.5 Sonnet、Qwen2.5-72B-Instruct，统一接口屏蔽 provider 差异
- 新增 **3 个外部数据集** 加入实验矩阵（每个均含 ≥ 2 demographic 维度，可做 intersectional 分析）：
  - **BookCrossing**：图书域，age + location（country-level）
  - **LFM-2b（LastFM 子集）**：音乐域，gender + age + country
  - **Tenrec**：中文短视频域，gender + age
- 现有 ML-1M 扩展用 **occupation** 作为第三维度（验证 3-way intersectional bias）
- 新增 **统一数据加载器**：抽象 `BaseDataset` 接口，把每个数据集映射成 `{user_demographics, history, test_set}` 标准格式
- 新增 **流行度切片预计算工具**：自动按数据集的 demographic 组合生成 `*_with_popularity.csv`
- 新增 **统计显著性模块**：paired Wilcoxon、paired t-test、bootstrap 95% CI、Cohen's d、Holm 多重检验校正
- 新增 **实验编排器**：执行 `Models × Datasets × Scenarios × Seeds` 矩阵，分层抽样（每数据集 1500-2000 用户），统一 reproducibility seed
- 新增 **聚合报告**：跨模型/跨域 delta 表、Pareto frontier 图、显著性标注
- **BREAKING**：`recommander.py` 中硬编码的 OpenAI client 与 ML-1M 路径剥离到配置文件，改用新的 LLM 适配层与数据集适配层（旧脚本作为兼容入口保留）

## Capabilities

### New Capabilities

- `llm-adapters`：统一多 LLM 调用接口（OpenAI 兼容、Anthropic、Together/SiliconFlow），含限速、重试、缓存、token 计费记录
- `dataset-adapters`：跨数据集统一加载与预处理，输出标准 `UserRecord` 与 `PopularityTable`，支持 ML-1M、BookCrossing、LFM-2b、Tenrec
- `popularity-slicing`：根据数据集 demographic schema 自动生成 group-conditional popularity CSV
- `experiment-runner`：执行多模型 × 多数据集 × 多场景的批量推荐 + rerank + 评估，支持断点续跑、分层抽样、随机种子管理
- `statistical-testing`：对 original vs reranked 各指标做配对显著性检验、效应量、置信区间、多重比较校正
- `aggregate-reporting`：跨实验单元汇总成论文可用的表/图（CSV + Markdown + 可选图表脚本）

### Modified Capabilities

<!-- 当前 openspec/specs/ 目录为空，无既有 capabilities，本次全为新增 -->

## Impact

- **新增代码模块**（建议位置）：
  - `src/llm_adapters/`：`base.py`、`openai_adapter.py`、`anthropic_adapter.py`、`together_adapter.py`
  - `src/datasets/`：`base.py`、`ml1m.py`、`bookcrossing.py`、`lfm.py`、`tenrec.py`
  - `src/popularity/`：`slicer.py`
  - `src/experiment/`：`runner.py`、`config.py`
  - `src/stats/`：`tests.py`、`effect_size.py`、`reporting.py`
  - `configs/`：每个 (model, dataset) 实验的 YAML
  - `scripts/`：`run_experiments.py`、`build_popularity.py`、`download_datasets.py`、`run_stats.py`
- **既有代码**：
  - `recommander.py`：拆分逻辑到适配层，保留 ML-1M 入口为兼容 wrapper
  - `rerank_recommendations.py`：参数化数据集 schema，支持任意 demographic 列组合
  - `evaluate_recommendations_top15.py`：增加跨数据集汇总与显著性结果接入
- **依赖新增**：`anthropic`、`together`（或 `httpx` 自实现）、`scipy`（统计）、`statsmodels`（可选，多重检验）、`pyyaml`（配置）
- **数据存储**：新增 `data/raw/{dataset}/`、`data/processed/{dataset}/`、`results/{model}/{dataset}/`、`reports/`
- **API key 管理**：从硬编码改为环境变量 + `.env`（添加 `.env.example`）
- **算力/成本**：3 模型 × 4 数据集 × 4 场景 × 1500 用户 ≈ 72K 次 LLM 调用，闭源预估 $50-150，开源走 SiliconFlow 免费额度或 Together API
- **可复现性**：所有 seed、抽样索引、prompt 模板、模型版本号入版本控制；产物 JSON 含完整 metadata
