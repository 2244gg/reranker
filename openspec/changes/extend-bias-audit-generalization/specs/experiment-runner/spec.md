## ADDED Requirements

### Requirement: YAML 实验配置

系统 SHALL 通过 YAML 文件描述每个实验单元，必填字段：`experiment_id, model, dataset, sample_size, sample_seed, scenarios, top_k, rerank` 子段。配置 MUST 通过 schema 校验。

#### Scenario: 加载合法配置
- **WHEN** 加载 `configs/gpt41mini_ml1m.yaml`
- **THEN** 返回 `ExperimentConfig` 对象，所有字段完整

#### Scenario: 缺字段报错
- **WHEN** 加载缺 `sample_seed` 的配置
- **THEN** 抛 `ConfigValidationError` 带具体字段路径

### Requirement: 矩阵执行能力

Runner SHALL 接受配置目录路径或单文件，按顺序或并行执行所有配置；MUST 支持 `--filter` 按 model/dataset 子集筛选。

#### Scenario: 跑全矩阵
- **WHEN** `python scripts/run_experiments.py --config-dir configs/`
- **THEN** 依次执行所有 YAML 实验，每个落盘到 `results/<model>/<dataset>/`

#### Scenario: 子集筛选
- **WHEN** `python scripts/run_experiments.py --filter "model=qwen-2.5-72b"`
- **THEN** 仅跑 model 为 Qwen 的配置

### Requirement: 断点续跑

Runner MUST 在每个用户处理完成后写入 `results/<model>/<dataset>/.progress/<user_id>.done` 标记文件，重启时跳过已完成用户。

#### Scenario: 中断后恢复
- **WHEN** 运行到 800 用户被中断，重启同配置
- **THEN** Runner 跳过前 800 用户，从 801 继续

#### Scenario: 缓存与进度协同
- **WHEN** 缓存命中且进度标记存在
- **THEN** 跳过该用户，不调 LLM、不重写 JSON

### Requirement: 4 类基础场景 + 高维交叉场景

Runner MUST 为每用户生成至少 4 类基础 prompt 场景：`neutral`、各 demographic 单维、二维 cross、若数据集维度 ≥ 3 还包含三维 cross。

#### Scenario: ML-1M 生成 7 个场景
- **WHEN** 运行 ML-1M 实验
- **THEN** 每用户产出 `neutral`, `gender`, `age`, `occupation`, `gender_age`, `gender_occupation`, `age_occupation`, `gender_age_occupation` 共 8 个场景的推荐

#### Scenario: BookCrossing 生成 4 个场景
- **WHEN** 运行 BookCrossing 实验
- **THEN** 每用户产出 `neutral`, `age`, `country`, `age_country` 共 4 个场景

### Requirement: 产物结构一致

每个实验单元 MUST 输出固定目录结构：`results/<model_id>/<dataset_id>/raw/`（LLM 原始）、`reranked/`、`metrics/per_user.parquet`、`metrics/aggregate.json`、`meta.json`。`meta.json` MUST 含 `experiment_id, model, model_version, dataset, sample_size, sample_seed, started_at, finished_at, total_api_calls, total_tokens, prompt_template_hash`。

#### Scenario: 实验完成后产物完整
- **WHEN** 一个实验单元执行完毕
- **THEN** 上述所有路径文件存在且非空

### Requirement: Reproducibility

所有 prompt 模板、采样种子、模型版本号、rerank 超参 MUST 入 `meta.json`，仓库 MUST 把 prompt 模板纳入版本控制并以 hash 写入产物。

#### Scenario: 复现声明
- **WHEN** 第三方拿到 `meta.json` 与 `configs/`
- **THEN** 可在相同环境完全复现产物（缓存命中除外）
