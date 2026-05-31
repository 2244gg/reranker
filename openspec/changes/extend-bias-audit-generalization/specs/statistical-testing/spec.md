## ADDED Requirements

### Requirement: 配对显著性检验

系统 SHALL 对每个 `(model, dataset, scenario, metric)` 单元计算 paired Wilcoxon signed-rank test 与 paired t-test，输入是每用户的 `(original_metric, reranked_metric)` 对。

#### Scenario: NDCG 配对检验
- **WHEN** 输入 1500 用户的 NDCG@15 配对值
- **THEN** 输出包含 `wilcoxon_statistic`, `wilcoxon_p`, `t_statistic`, `t_p`, `n_pairs`

#### Scenario: 全零差值处理
- **WHEN** 所有用户 delta 为 0
- **THEN** 返回 `wilcoxon_p = 1.0` 不报错

### Requirement: 二值指标 McNemar 检验

对 HitRate@k 等二值指标，系统 SHALL 用 McNemar test 而非 Wilcoxon。

#### Scenario: HitRate 用 McNemar
- **WHEN** 调 `run_significance(metric="hit_rate", values_orig, values_rer)`
- **THEN** 内部走 McNemar，输出 `mcnemar_p` 与 `b, c` 不一致计数

### Requirement: Bootstrap 置信区间

系统 SHALL 对每个 `delta = reranked - original` 序列做 10000 次 bootstrap，输出 `mean_delta, ci_low, ci_high`（95% percentile）。

#### Scenario: CI 包含 0 不显著
- **WHEN** delta 序列 mean=0.001 std=0.05
- **THEN** 95% CI 通常跨 0，significance 标记为 `ns`

#### Scenario: 复现 bootstrap
- **WHEN** 同 seed 两次跑 bootstrap
- **THEN** 输出 CI 完全一致

### Requirement: 效应量计算

系统 SHALL 计算 Cohen's d（连续指标）与 odds ratio（二值指标）作为效应量。

#### Scenario: Cohen's d 输出
- **WHEN** 给定 NDCG 配对差值序列
- **THEN** 输出 `cohens_d` 浮点值，并附等级标签 `negligible/small/medium/large`

### Requirement: 多重比较校正

系统 SHALL 在每 `(model, dataset)` 单元内对所有 `(scenario × metric)` 检验 p 值做 Holm-Bonferroni 校正，输出 `wilcoxon_p_holm` 字段。

#### Scenario: 28 项检验校正
- **WHEN** 单元内有 4 场景 × 7 指标 = 28 项
- **THEN** 28 项 p 值按 Holm 算法升序校正后输出

### Requirement: 显著性符号映射

系统 SHALL 按校正后 p 值给出标签：`***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` 否则。

#### Scenario: 标签映射
- **WHEN** 校正后 p=0.008
- **THEN** 标签为 `**`

### Requirement: 结果产物 schema

输出 SHALL 写入 `reports/significance_tests.json`，键为 `(model, dataset, scenario, metric)` 元组（序列化成字符串），值含 `n, mean_orig, mean_rer, delta, ci_95, wilcoxon_p, wilcoxon_p_holm, t_p, cohens_d, significance_label, significant_after_correction`。

#### Scenario: JSON 结构稳定
- **WHEN** 加载该 JSON
- **THEN** 每条记录字段集合一致，方便下游脚本消费
