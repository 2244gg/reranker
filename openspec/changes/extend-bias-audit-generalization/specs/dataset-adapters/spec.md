## ADDED Requirements

### Requirement: 统一 UserRecord 数据结构

系统 SHALL 定义 `UserRecord` 与 `InteractionItem` dataclass，所有数据集 adapter 加载后输出此统一结构。`UserRecord` MUST 包含 `user_id, demographics, history, test_set` 四字段。`demographics` MUST 是 `Dict[str, str]`，至少包含 2 个不同维度的身份属性。

#### Scenario: ML-1M 加载后 demographics 含 3 维
- **WHEN** `Ml1mAdapter().load_user("100")`
- **THEN** 返回 `UserRecord.demographics == {"gender": "M", "age_group": "young", "occupation_group": "student"}`

#### Scenario: BookCrossing 加载后 demographics 含 2 维
- **WHEN** `BookCrossingAdapter().load_user("user-001")`
- **THEN** 返回 `UserRecord.demographics` 含 `age_group` 与 `country` 两键

#### Scenario: LFM-2b 加载后 demographics 含 3 维
- **WHEN** `LfmAdapter().load_user("user-001")`
- **THEN** 返回 `UserRecord.demographics` 含 `gender`, `age_group`, `country`

#### Scenario: Tenrec 加载后 demographics 含 2 维
- **WHEN** `TenrecAdapter().load_user("user-001")`
- **THEN** 返回 `UserRecord.demographics` 含 `gender`, `age_group`

### Requirement: 数据集 demographic schema 暴露

每个 `BaseDatasetAdapter` 子类 MUST 实现 `demographic_schema() -> List[str]` 方法返回该数据集的 demographic 维度名列表，以及 `domain() -> str` 返回域类型字符串（`movie`/`book`/`music`/`video`）。

#### Scenario: 编排器自动构建场景列表
- **WHEN** 编排器调 `adapter.demographic_schema()`
- **THEN** 返回如 `["gender", "age_group", "occupation_group"]`，编排器据此自动生成 `gender`、`age_group`、`occupation_group`、`gender_age`、`gender_occupation`、`age_occupation`、`gender_age_occupation` 等场景

### Requirement: 训练/测试切分确定性

每个 adapter MUST 提供 `split(seed: int, test_size: float)` 方法，相同 seed 必须产出相同切分，按用户内部切分以保证每用户在两侧都有数据。

#### Scenario: 同 seed 复现同切分
- **WHEN** 两次调 `adapter.split(seed=42, test_size=0.7)`
- **THEN** 两次返回的训练/测试 user-item 对完全一致

### Requirement: 分层抽样

每个 adapter MUST 提供 `stratified_sample(n_users: int, seed: int)` 方法，按 demographic 维度的笛卡尔积分层抽样，每个切片至少分配 `max(1, n_users / num_slices)` 用户。当某切片用户数不足时，记录警告但不报错。

#### Scenario: 分层抽样切片均衡
- **WHEN** 用 ML-1M 调 `stratified_sample(n_users=1500, seed=42)`
- **THEN** 抽出的用户在 (gender × age_group × occupation_group) 各切片数差异 <= 1（受切片大小限制）

### Requirement: 正反馈定义可配置

每个 adapter MUST 暴露 `positive_feedback_predicate(item) -> bool`，定义见下：

| 数据集 | 默认正反馈 |
|---|---|
| ML-1M | rating > 3 |
| BookCrossing | rating > 5（显式）OR 任意交互（隐式，可切换） |
| LFM-2b | playcount > user 中位数 |
| Tenrec | watch_time_ratio > 0.5 |

#### Scenario: ML-1M 正反馈
- **WHEN** 用 `Ml1mAdapter` 加载 user 评分 4 的电影
- **THEN** 该项进入 `history`/`test_set` 的正反馈集

#### Scenario: BookCrossing 隐式正反馈
- **WHEN** 用 `BookCrossingAdapter(implicit=True)` 加载 rating=0 的交互
- **THEN** 仍纳入正反馈集

### Requirement: 数据集元信息与下载入口

仓库 MUST 提供 `data/README.md` 列出每个数据集的官方下载链接、许可证、引用要求；下载脚本 `scripts/download_datasets.py` SHOULD 自动获取（在许可允许时）或打印手动下载步骤。原始数据 MUST NOT 入仓。

#### Scenario: 新人复现实验
- **WHEN** 新开发者克隆仓库执行 `python scripts/download_datasets.py --dataset bookcrossing`
- **THEN** 数据被下载到 `data/raw/bookcrossing/`，或打印官方下载步骤
