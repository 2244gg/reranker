## Context

现状：项目已实现 LLM 推荐 + 流行度感知 rerank 流水线，但只跑了 GPT-4.1-mini × ML-1M 一组。审稿人会质疑泛化性与统计可信度。本次变更要在不破坏既有产物（`JSON/`、`reranked_JSON/`）前提下，把流水线扩展为多 LLM × 多数据集矩阵实验，并加入显著性检验。

**约束**：
- 既有脚本（`recommander.py`、`rerank_recommendations.py`、`score_top15_dual_lists.py`、`evaluate_recommendations_top15.py`）的 JSON 数据契约不能破坏，新增字段以扩展形式存在
- API key 必须从硬编码迁出到环境变量
- 所有新代码可在单台开发机（Windows + Python 3.10+）跑通；闭源走 API，开源大模型走第三方 API（不强制本地 GPU）
- 算力预算：3 模型 × 4 数据集 × 1500 用户 × 4 场景 ≈ 72K 次调用，需缓存与断点续跑

**Stakeholders**：研究者本人（论文一作）、审稿人（公平性 + 推荐方向）、复现者。

## Goals / Non-Goals

**Goals**：

- G1：建立 **LLM 适配层**，统一 OpenAI/Anthropic/Together(or SiliconFlow) 三类 API 的同步调用、限速、重试、缓存、token 计费记录
- G2：建立 **数据集适配层**，把 ML-1M、BookCrossing、LFM-2b、Tenrec 映射成统一 `UserRecord` 结构，每数据集至少暴露 2 个 demographic 维度供 intersectional 实验
- G3：自动化 **流行度切片表生成**，按各数据集 demographic schema 算 group-conditional popularity
- G4：实现 **实验编排器**，跑 `M × D × Scn × Seeds` 矩阵，分层抽样、断点续跑、产物结构化
- G5：实现 **统计显著性模块**，给每条 (model, dataset, scenario, metric) 出 paired 检验 + 效应量 + 95% CI + 多重比较校正
- G6：产出 **聚合报告**：跨模型/跨域 delta 表 + Pareto 散点 + 显著性标注，论文可直接引用

**Non-Goals**：

- 不重训 / 不微调任何 LLM
- 不引入新去偏算法（rerank 公式与 dynamic α 复用现有实现，仅参数化）
- 不做 GUI 或交互前端
- 不追求所有数据集的全量用户，统一分层抽样到 1500-2000 用户
- 不在本次提案里做绘图美化（Pareto 图先用 matplotlib 默认样式）

## Decisions

### D1：LLM 适配层接口与提供方选择

**决定**：定义 `BaseLLMAdapter.generate(prompt, **kwargs) -> LLMResponse` 抽象，3 个子类：

| Adapter | Provider | 模型 |
|---|---|---|
| `OpenAIAdapter` | laozhang.ai 代理（OpenAI 兼容） | `gpt-4.1-mini` |
| `AnthropicAdapter` | laozhang.ai 代理（Anthropic 兼容）或官方 | `claude-3-5-sonnet-20241022` |
| `TogetherAdapter` | SiliconFlow（国内访问稳定）或 Together AI | `Qwen/Qwen2.5-72B-Instruct` |

`LLMResponse` 含 `text, prompt_tokens, completion_tokens, model, latency_ms, retries, finish_reason`。

**Alternative 1**：直接用 langchain 抽象。**否决** —— 依赖重，调试复杂，token 计量需要二次封装。
**Alternative 2**：为每个模型写独立脚本。**否决** —— 实验编排无法统一，矩阵跑无法横向对齐。

**关键设计**：
- 限速：`tenacity` 退避重试，最大 6 次
- 缓存：基于 `(model, prompt_sha256, temperature, top_p)` key 的本地 SQLite 缓存（`data/cache/llm_cache.sqlite`），命中直接返回，断点续跑零成本
- temperature 全部固定 0.7（保留与原实验一致），seed 走 prompt-level 多样性（5 个 random_state）

### D2：数据集统一 schema

**决定**：`UserRecord` dataclass：

```python
@dataclass
class UserRecord:
    user_id: str
    demographics: Dict[str, str]   # e.g. {"gender": "F", "age_group": "young", "country": "DE"}
    history: List[InteractionItem] # 训练集，rating>阈值 / 隐式正反馈
    test_set: List[InteractionItem]
    
@dataclass
class InteractionItem:
    item_id: str
    title: str
    metadata: Dict[str, Any]       # genres, artist, country, etc.
    rating: Optional[float]
    timestamp: Optional[int]
```

各数据集 demographic 维度与正反馈定义：

| Dataset | demographics 维度 | 正反馈定义 | 抽样规模 |
|---|---|---|---|
| ML-1M | gender, age_group, occupation_group | rating > 3 | 1500 用户分层抽样 |
| BookCrossing | age_group, country | rating > 5（1-10 量表）或隐式（任意交互） | 1500 用户，过滤 ≥10 次交互 |
| LFM-2b | gender, age_group, country | playcount > 中位数 | 1500 用户，过滤 ≥20 次交互 |
| Tenrec | gender, age_group | watch_time > 50%（隐式正反馈） | 1500 用户，过滤 ≥10 次曝光 |

**occupation_group**（ML-1M）：21 类合并成 4 组（student / professional / service / other）以避免过细切片导致流行度估计稀疏。

**country**（BookCrossing/LFM）：保留 top-10 国家 + `other`，避免长尾切片噪声。

**Alternative**：完整保留 21 occupation。**否决** —— 切片用户数过少，popularity 估计方差大。

### D3：流行度切片预计算

**决定**：`PopularitySlicer` 接收数据集与 demographic schema，自动生成：

- `neutral_All`
- `gender_<value>` 每 demographic 单维（marginal）
- `age_<value>`
- ...
- `cross_<dim1>_<dim2>` 二维交叉
- 三维交叉仅 ML-1M（occupation 维度）做扩展实验

每列做 `min-max` 归一化到 [0,1]，**slice 内独立归一化**（与现有 `movies_with_popularity.csv` 一致）。

输出 `data/processed/<dataset>/items_with_popularity.csv`，schema 含 `item_id, title, <slice columns>`。

### D4：实验编排器与产物结构

**决定**：

```
results/
  <model_id>/
    <dataset_id>/
      raw/recommendation_results_<batch>.json     # 原始 LLM 输出
      reranked/recommendation_results_<batch>.json
      metrics/per_user.parquet                    # 每用户每场景指标
      metrics/aggregate.json                      # 汇总
      meta.json                                   # 模型版本、采样 seed、运行时长
reports/
  cross_model_dataset_summary.csv
  significance_tests.json
  pareto_frontier.png
```

`ExperimentConfig` YAML：

```yaml
experiment_id: gpt41mini_ml1m_v1
model: gpt-4.1-mini
dataset: ml1m
sample_size: 1500
sample_seed: 42
scenarios: [neutral, age, gender, occupation, age_gender, age_gender_occupation]
top_k: 20
rerank:
  alpha_base: 0.7
  alpha_min: 0.5
  alpha_max: 0.9
  alpha_gain: 0.2
  base: 0.2
  decay_rate: 0.9
```

Runner 调度：`for cfg in load_all_configs(): runner.run(cfg)`，每步落盘，崩溃可重入（基于 `done` flag 文件）。

### D5：统计显著性方案

**决定**：核心检验栈：

| 指标类型 | 主检验 | 副检验 | 效应量 |
|---|---|---|---|
| 连续指标（NDCG、MRR、Precision、Recall） | Wilcoxon signed-rank | paired t-test | Cohen's d |
| 二值指标（HitRate@k） | McNemar | — | odds ratio |
| 自定义连续分（unpopularity score） | Wilcoxon signed-rank | — | Cohen's d |

- **配对**：每用户每场景算 (original_metric, reranked_metric) 一对
- **方向**：默认双侧；论文表格里同时报单侧（`alternative='greater'` for rerank improvement）
- **多重检验**：每个 (dataset, model) 单元内对所有 (scenario × metric) 做 **Holm-Bonferroni** 校正
- **CI**：bootstrap 10000 次，95% percentile CI for `mean(reranked - original)`
- **显著性符号**：`***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` 不显著

输出 `reports/significance_tests.json`，结构：

```json
{
  "(model, dataset, scenario, metric)": {
    "n": 1500,
    "mean_orig": 0.342,
    "mean_rer": 0.351,
    "delta": 0.009,
    "ci_95": [0.005, 0.013],
    "wilcoxon_p": 0.0012,
    "wilcoxon_p_holm": 0.0084,
    "cohens_d": 0.21,
    "significant": true
  }
}
```

### D6：聚合报告与论文物料

**决定**：脚本 `scripts/build_report.py` 产出：

1. `reports/cross_model_dataset_summary.csv`：宽表，行 = (scenario, metric)，列 = (model × dataset)，单元 = `delta ± CI [sig_label]`
2. `reports/by_scenario_<metric>.md`：人读 Markdown 子表
3. `reports/pareto_frontier.png`：x = NDCG@15 提升, y = unpopularity-score 提升，颜色按模型，形状按数据集
4. `reports/intersectional_heatmap_<dataset>.png`：3-way intersectional 偏置热力图（仅 ML-1M occupation 扩展）

## Risks / Trade-offs

- **R1：LLM 跨语言能力差异** → Tenrec 是中文，Qwen 友好但 Claude/GPT 表现可能不如英文 → **Mitigation**：prompt 提供中英双版本，每模型用其原生语言版本，结果中分别标注 prompt 语种
- **R2：BookCrossing 评分稀疏** → 1-10 量表很多用户只评 1-2 本 → **Mitigation**：要求 ≥10 次交互，正反馈用 rating>5 ∨ 隐式（出现即正）双轨，分别跑实验
- **R3：第三方代理（laozhang.ai）稳定性** → API 可能限流或挂掉 → **Mitigation**：缓存机制保证已成功调用不重跑；Anthropic 备选官方 API；Qwen 备选 SiliconFlow + Together
- **R4：72K 次调用成本超预期** → **Mitigation**：先 200 用户 pilot 校准 token 用量，再决定是否缩到 1000 用户；缓存所有响应防止 prompt 改写后重跑
- **R5：多重检验过度校正** → 28 项 × 3 模型 × 4 数据集 = 336 项 Holm 校正会很严苛 → **Mitigation**：分层校正（每 model+dataset 单元内独立校正），论文里明确说明
- **R6：occupation 切片用户数太少** → 21 类原始 occupation 单切片可能 < 50 用户，估计不稳 → **Mitigation**：合并到 4 组；显著性检验时检查 effective sample size
- **R7：数据集下载与许可证** → BookCrossing/LFM-2b/Tenrec 各有不同许可 → **Mitigation**：写 `data/README.md` 列出官方下载入口与引用要求，**不**入仓原始数据
- **R8：现有产物兼容** → 原 `JSON/recommendation_results_*.json` 字段被新流水线读到时报错 → **Mitigation**：新代码读取走严格 schema 校验，旧产物迁移脚本一次性转换或并存

## Migration Plan

1. **Phase 0（兼容修复）**：把 `recommander.py` 的 API key 移到 `.env`，加 `.env.example`，旧脚本仍可独立跑
2. **Phase 1（基础设施）**：实现 `llm_adapters/` + `datasets/` + `popularity/`，单跑 ML-1M + GPT-4.1-mini 复现现有结果（差值 < 1e-6）作为回归测试
3. **Phase 2（数据集铺开）**：依次接入 BookCrossing、LFM-2b、Tenrec，每个 200 用户 pilot
4. **Phase 3（模型铺开）**：接入 Claude 3.5 Sonnet 与 Qwen2.5-72B，先 200 用户 pilot 验证
5. **Phase 4（全量矩阵）**：1500 用户跑全部 12 cells（3 模型 × 4 数据集），断点续跑
6. **Phase 5（统计与报告）**：跑 `scripts/run_stats.py` 与 `build_report.py`，产出论文物料

回滚：所有新代码独立目录，删除 `src/`、`configs/`、`scripts/`、`data/processed/`、`results/`、`reports/` 即恢复原状。`.env` 与 `requirements.txt` 改动可 `git revert`。

## Open Questions

- **OQ1**：Tenrec 的中文 prompt 是否对所有 3 个模型都用中文，还是 GPT/Claude 用中英对照？暂定**全部用中文**避免引入语种偏置变量，但需 pilot 验证 Claude 中文输出质量
- **OQ2**：LFM-2b country 维度切片粒度（top-10 vs 大区）→ 暂定 top-10 + `other`，pilot 后视稀疏度调整
- **OQ3**：BookCrossing 的隐式 vs 显式正反馈两轨是否都做，还是择一？默认两轨都做，论文报隐式作为主结果
- **OQ4**：是否给 occupation 维度做单独 ablation 论文章节？暂定**做**，作为 ML-1M 独有的 3-way intersectional 案例研究
- **OQ5**：是否引入 BERTScore / 标题相似度做"推荐刻板印象"二级评估？暂列扩展项，本次不实现
