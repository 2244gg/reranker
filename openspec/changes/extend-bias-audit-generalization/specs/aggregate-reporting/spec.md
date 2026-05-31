## ADDED Requirements

### Requirement: 跨模型跨数据集汇总表

系统 SHALL 输出 `reports/cross_model_dataset_summary.csv`，行为 `(scenario, metric)`，列为 `(model, dataset)` 组合，单元格内容形如 `delta ± half_ci_width [sig_label]`。

#### Scenario: 论文宽表
- **WHEN** 12 个实验单元跑完
- **THEN** CSV 行覆盖所有 (scenario, metric)，列覆盖 12 单元，可直接复制进 LaTeX

### Requirement: 按场景分指标 Markdown 子表

系统 SHALL 输出 `reports/by_scenario_<metric>.md` 一组文件，每文件聚焦一个指标，子表行=场景、列=数据集（嵌套模型）。

#### Scenario: 单指标专题表
- **WHEN** `metric=ndcg_at_k`
- **THEN** 生成 `reports/by_scenario_ndcg_at_k.md`，含 4 数据集 × 3 模型 的二级表头

### Requirement: Pareto 散点图

系统 SHALL 生成 `reports/pareto_frontier.png`：x 轴 = `delta_ndcg_at_k`，y 轴 = `delta_unpopularity_score`，每点为一个 (model, dataset, scenario) 单元，颜色按模型区分，形状按数据集区分。

#### Scenario: 输出可读 Pareto 图
- **WHEN** 跑完 stats 后调 `python scripts/build_report.py --pareto`
- **THEN** 生成 PNG 图，含图例与坐标轴标签

### Requirement: Intersectional heatmap

对 demographic 维度 ≥ 3 的数据集（ML-1M、LFM-2b），系统 SHALL 输出 `reports/intersectional_heatmap_<dataset>.png`，矩阵呈现交叉切片下 `delta_unpopularity_score` 的强弱，标注显著性。

#### Scenario: ML-1M 三维 heatmap
- **WHEN** 跑完 ML-1M occupation 扩展实验
- **THEN** 输出 heatmap 含 (gender × age × occupation) 三维切片单元格

### Requirement: 报告含可复现脚注

每个生成的报告文件 MUST 在文件头部记录生成时间、源 `summary` 路径、commit hash（若仓库为 git）。

#### Scenario: 脚注完整
- **WHEN** 打开任意报告
- **THEN** 头部 4 行内含 `generated_at`, `source`, `commit`
