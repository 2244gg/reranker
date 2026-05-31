## ADDED Requirements

### Requirement: 自动生成 group-conditional popularity

系统 SHALL 提供 `PopularitySlicer` 类，输入数据集 adapter 与训练集交互记录，输出包含 `item_id, title, neutral_All, <single-dim slices>, <pairwise cross slices>` 的 CSV。每列 SHALL 在切片内独立做 `min-max` 归一化到 [0, 1]。

#### Scenario: ML-1M 生成 popularity 表
- **WHEN** 调 `PopularitySlicer(Ml1mAdapter()).build()`
- **THEN** 输出 `data/processed/ml1m/items_with_popularity.csv` 含列：`neutral_All`, `gender_M/F`, `age_<group>`, `occupation_<group>`, `cross_gender_age_*`, `cross_gender_occupation_*`, `cross_age_occupation_*`

#### Scenario: BookCrossing 生成 popularity 表（仅 2 维）
- **WHEN** 调 `PopularitySlicer(BookCrossingAdapter()).build()`
- **THEN** 输出 CSV 含 `neutral_All`, `age_<group>`, `country_<top10>_or_other`, `cross_age_country_*`，不含三维 cross

### Requirement: 流行度估计稀疏性保护

当某切片用户数 < 30 时，该切片下的 popularity 估计 SHALL 与 `neutral_All` 做线性插值（拉向全局均值），插值权重为 `min(1, slice_n / 30)`，避免估计方差过大。

#### Scenario: 小切片回退到全局
- **WHEN** 某 (gender=F, occupation=other) 切片只有 5 个用户
- **THEN** 该切片每电影的 popularity = `(5/30) * raw_slice_pop + (25/30) * neutral_All`

### Requirement: 三维 cross slice 仅在维度 ≥ 3 数据集启用

当数据集 demographic 维度 < 3 时，slicer MUST NOT 生成三维 cross 列。当 ≥ 3 时 SHALL 生成全部 C(d,3) 组合的三维列。

#### Scenario: BookCrossing 不生成三维 cross
- **WHEN** 调 BookCrossing slicer
- **THEN** 输出 CSV 不含 `cross_*_*_*` 三维列

#### Scenario: ML-1M 生成三维 cross
- **WHEN** 调 ML-1M slicer
- **THEN** 输出 CSV 含 `cross_M_young_student` 等三维列

### Requirement: 复现现有 ML-1M popularity 表

slicer 在 ML-1M + 旧 demographic 配置（仅 gender + age_group）下生成的列 MUST 与既有 `movies_with_popularity.csv` 数值一致（容差 1e-6）。

#### Scenario: 回归一致性测试
- **WHEN** 运行 `pytest tests/test_popularity_regression.py`
- **THEN** 新 slicer 输出与旧 CSV 在 `neutral_All`、`gender_M/F`、`age_*`、`cross_M_*`、`cross_F_*` 所有列数值匹配
