# LastFM-1K 单曲级重排序工作记录

本文档记录把仓库从 **艺人级 (artist-level)** 推荐 + 重排序改造为 **单曲级 (track-level)** 推荐 + 重排序的全过程，以及在此基础上设计的 v2 重排序算法（`rank-normalized B + top-K protect`）和配套评估指标（APLT / Coverage / Gini）。

适用读者：要在 LastFM-1K 上跑单曲级 popularity-debias 实验、想理解算法折衷、或要复现/扩展工作的人。

---

## 1. 目标与起点

**起点**：仓库原本只对 LFM-1K 做 **艺人级**（每位艺人 1 行）流行度切片与重排序：
- 物品 = 艺人名字（"Radiohead"）
- LLM 推荐艺人 → 流行度查表 → 重排序公式 \( S = \alpha P + (1-\alpha)(1-B) \)

**目标**：改成 **单曲级**（每首 `(Artist, Track)` 1 行），并且：
1. 给 LLM 推单曲（"Creep - Radiohead"）
2. 单曲粒度的 (item, demographic-slice) 流行度表
3. 重排序算法对单曲 B 矩阵能给出有意义的去偏信号（不被 long-tail collapse 淹没）
4. 评估指标补齐 popularity-debias 文献标配：APLT、Coverage、Gini

---

## 2. 改动总览

### 2.1 新增 / 修改文件

| 文件 | 类型 | 作用 |
|---|---|---|
| `src/datasets/lastfm1k.py` | 改 | 加 `granularity="track"` 模式；track 模式下 `domain()→"music_track"`；2 桶年龄；`pickle` 缓存 942 用户 record |
| `prompts/music_track_en.j2` | 新 | 单曲专用 prompt 模板，强制每行 `Track - Artist` |
| `src/experiment/prompt_builder.py` | 改 | 注册 `music_track` 域 |
| `src/experiment/runner.py` | 改 | `keep_full_title` 模式（不剥离 `- Artist`）；progress marker resume；`b_transform` 应用；APLT/Coverage/Gini 聚合 |
| `src/experiment/rerank.py` | 改 | 加 `protect_top` 参数 |
| `src/experiment/config.py` | 改 | `RerankConfig` 加 `b_transform / b_saturate_at / protect_top` |
| `src/experiment/metrics.py` | 改 | 加 `aplt_at_k` 与 `gini_index`，`compute_per_user_metrics` 输出 APLT |
| `src/LFM_track_preprocess.py` | 新 | 仿 `ML_preprocess.py` 风格的独立单曲流行度预处理脚本 |
| `scripts/build_popularity.py` | 改 | 加 `lastfm1k` 选项 + `--granularity --include-country --min-track-listeners` |
| `configs/gpt41mini_lfm1k_track.yaml` | 新 | track 模式 baseline 配置（α=0.7 raw） |
| `configs/gpt41mini_lfm1k_track_a05.yaml` | 新 | α=0.5 ablation |
| `configs/gpt41mini_lfm1k_track_v2.yaml` | 新 | v2 算法（rank-B + protect_top=5） |
| `tests/test_lastfm1k_track.py` | 新 | 7 项单元测试 |
| `final_report_v2.py` | 新 | 主实验跑完后渲染最终对比报告 |
| `compare_alpha.py` / `compare_v2.py` / `compare_orig_v2_full.py` / `inspect_user_rerank.py` / `inspect_mini.py` / `diag_lfm_track.py` | 新 | 调试与对比脚本 |

### 2.2 数据产物

| 路径 | 大小 | 内容 |
|---|---|---|
| `data/processed/lastfm1k_track/items_with_popularity.csv` | ~50MB | 272,769 单曲 × 16 分组列（含 Unknown 切片）|
| `data/cache/lastfm1k_track_4d5b80532f86.pkl` | 282MB | 942 个 `UserRecord` 的 pickle，下次实验秒级加载 |
| `data/cache/llm_cache.sqlite` | ~2MB（持续增长）| LLM 响应按 prompt-hash 缓存 |
| `results/gpt-4.1-mini/lastfm1k/` | — | α=0.7 raw 30 用户迷你结果 |
| `results_a05/gpt-4.1-mini/lastfm1k/` | — | α=0.5 raw 30 用户迷你结果 |
| `results_v2/gpt-4.1-mini/lastfm1k/` | — | v2 (rank+protect5) 30 用户迷你 + 600 用户主实验（运行中） |

---

## 3. 单曲粒度的核心改动

### 3.1 适配器双模式开关

`LastFm1KAdapter(granularity="track" | "artist")`

| granularity | item_id | title | domain |
|---|---|---|---|
| `"artist"` | `"Radiohead"` | `"Radiohead"` | `"music"` |
| `"track"` | `"Radiohead ||| Creep"` | `"Creep - Radiohead"` | `"music_track"` |

`||| ` 分隔符避免同名歌冲突（"Yesterday" 由 Beatles 与 Boyz II Men 都唱过）。`title` 用 ASCII 连字符 ` - ` 与 LLM 输出格式一致，匹配率高。

`domain="music_track"` 让 `PromptBuilder` 自动选 `prompts/music_track_en.j2`，该模板明确要求 `Track - Artist` 格式。

### 3.2 阳性反馈定义

ML-1M 的 `Rating > 3` 在单曲粒度失效（无评分），用代理：

```
positive(u, item) = playcount(u, item) >= 2 AND playcount(u, item) > median(playcount(u, *))
```

排除一次性试听噪声 + 强制高于该用户的中位曲目播放频次。

### 3.3 年龄分桶（数据驱动）

LFM-1K 实测：993 用户中 **285 人填了年龄**，分布主峰 18-30 岁，35+ 仅 4 人。原 3 桶 (Youth/Middle/Senior) 在 Senior 桶严重稀疏。改 **2 桶**：

```python
# src/datasets/lastfm1k.py:_age_to_group_2bucket
<25         → "young"        (Youth, 162 人)
[25, 100]   → "middle-aged"  (Middle, 123 人，吸收原 Senior)
其他        → "unknown"      (Unknown, 708 人)
```

未知 (Unknown) 是**最大年龄桶**，必须当成正经切片处理（见 §4.2）。

---

## 4. 单曲流行度表

### 4.1 仿 `ML_preprocess.py` 的独立预处理

`src/LFM_track_preprocess.py` 完整对照：

| ML_preprocess.py 步骤 | LFM_track_preprocess.py 对应 |
|---|---|
| `load_ml1m_data` | `load_lfm1k_users` + 流式聚合事件文件 |
| `create_user_groups`（gender/age/cross/neutral）| 同名函数，桶定义对齐适配器 |
| `for row in high_ratings` 计数 | 按 (user, track) 阳性反馈累加 |
| `count / column_max` 归一化 | 完全相同 |
| `add_popularity_to_items` 12 列 | `assemble_dataframe`，16 列（含 Unknown） |
| `movies_with_popularity.csv` | `items_with_popularity.csv` |

### 4.2 列设计

| 类别 | 列 | 用户数 |
|---|---|---|
| 中性 | `neutral_All` | 993 |
| 性别 | `gender_M / F / U` | 502 / 382 / 109 |
| 年龄 | `age_Youth / Middle / Unknown` | 162 / 123 / 708 |
| 交叉 | `cross_M_Youth / F_Youth / U_Youth` | 82 / 74 / 6 |
| | `cross_M_Middle / F_Middle / U_Middle` | 75 / 40 / 8 |
| | `cross_M_Unknown / F_Unknown / U_Unknown` | 345 / 268 / 95 |

**关键决策**：保留 `Unknown` 切片。原计划丢掉 unknown 用户的人有 71%（708/993），他们的 age/cross 场景查表会全部命中 0 → 重排序退化为单调于 P（什么都不发生）。这是迷你实验早期发现的真 bug：

```
user_000004 (gender=female, age_group=unknown)
  age_group 场景 column_used = "age_Unknown"
  B (top 10 of LLM recs): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
  → 顺序未变，"重排序"实际什么都没做
```

修复：把 Unknown 当 first-class 切片纳入 popularity 表。

### 4.3 冷曲过滤

`min_track_listeners=3`：丢弃**全数据集中只被 ≤2 个用户听过**的单曲。原始 1,500,651 唯一单曲 → 过滤后 313,678 → 阳性反馈过滤后 **272,769** 入表。

理由：单曲分布极度长尾，一首曲只一人听过，B 信号纯噪声。提高门槛降低噪声。

---

## 5. 重排序算法

### 5.1 公式（不变）

```
S_i = α · P_i_norm + (1 - α) · (1 - B_i_norm)
```

- `P_i_norm` ∈ [base, 1]：对 LLM 排名 idx 做几何衰减得到的 preference
- `B_i_norm` ∈ [0, 1]：item i 在用户所属 demographic 切片的相对流行度
- `α` 由用户的"主流度"动态计算，clip 到 `[alpha_min, alpha_max]`

### 5.2 v2 改进点

#### A. `b_transform: rank` —— 把 B 拉成等距分位数

原始 \( B = \text{count} / \max \) 在长尾数据集上严重偏度：272K 单曲，max=224，绝大多数 B<0.05。结果 \((1-B) \approx 1\) 几乎到处一样，**奖励项失去区分度**。

转换：

```python
B' = rank_in_column / |column_size|   # percentile rank, [0, 1] uniform
```

最热门那首 B'=1.0，最冷的 B'≈0.005，中段 B'≈0.5。\((1-B')\) 现在均匀分布，重排序信号有强弱梯度。

实现位置：`src/experiment/runner.py::_apply_b_transform`，加载 popularity_csv 后立即转换。

`b_transform` 可选 `raw | rank | saturate(B/0.3, 上限 1)`。

#### B. `protect_top: 5` —— 保护早位精度

原算法对所有 N 个候选统一排序 → LLM 给的 top-1 命中曲很容易被低 B 的中段曲挤下来 → MRR / Precision@5 大降。

改成两段：

```
pos 1..5  : 锁定 LLM 原序，不重排
pos 6..N  : 按 S 分数重排，merge 在 5 之后
```

实现位置：`src/experiment/rerank.py::rerank_one_list`。

效果（v2 vs baseline α=0.7）：

| 场景 | baseline ΔMRR | v2 ΔMRR |
|---|---|---|
| gender | -14.7% | **-1.4%** |
| age_group | -14.1% | **-3.0%** |
| cross | -7.9% | **-3.8%** |

→ MRR 损失被压到原来的 1/5 到 1/10。

---

## 6. 评估指标

### 6.1 老指标（accuracy 类）

| 指标 | 含义 | 取向 |
|---|---|---|
| Precision@K | 命中数 / K | 越高越好 |
| Recall@K | 命中数 / |测试集| | 越高越好 |
| HitRate@K | top-K 是否有命中 | 越高越好 |
| MRR | 第一命中位置的倒数 | 越高越好 |
| NDCG@K | 折现累积增益 | 越高越好 |

### 6.2 新增指标（popularity-debias 类）

引自 Abdollahpouri et al. 2017《Controlling Popularity Bias in Learning-to-Rank Recommendation》：

| 指标 | 公式 | 取向 |
|---|---|---|
| **APLT (Average % Long-Tail)** | \( \frac{\|\{i \in \text{TopK} : B_i < 0.2\}\|}{K} \) | 越高越好 |
| **Coverage** | unique items / catalog size | 越高越好 |
| **Gini** | 推荐频次分布的基尼系数 | 越低越好 |

实现位置：
- `aplt_at_k`、`gini_index` 在 `src/experiment/metrics.py`
- Coverage 与 Gini 在 `runner._aggregate_metrics` 跨用户计算

APLT 输出两个版本：
- `aplt`：用用户所属 slice 列查 B（"在 ta 圈子里的冷门"）
- `aplt_neutral`：用 `neutral_All` 列查 B（"全库冷门"）

---

## 7. 工程基础设施

### 7.1 双层缓存让迭代秒级

#### Adapter pickle 缓存

19M 行事件文件流式聚合需 4-6 分钟。`LastFm1KAdapter._cache_path` 按 (raw_dir, events 文件 mtime, granularity, min_interactions, min_track_listeners, include_country) 哈希存 pickle 到 `data/cache/lastfm1k_<key>.pkl`。下次直接 `pickle.load`，~10 秒。

设环境变量 `LASTFM1K_DISABLE_CACHE=1` 可绕过。

#### LLM 响应缓存（仓库已有）

`data/cache/llm_cache.sqlite` 按 prompt-hash 存。改 α、改 protect_top、改 b_transform → prompt 不变 → 永远命中 → **0 API 成本**。

### 7.2 Progress marker 修复

老 runner 有 bug：`progress_dir/<user>.done` marker 存在时 `continue` 但不重新加载 JSON → 第二次跑会用空 dict 覆盖好结果。

修复：`runner.py` 入口处检测 marker，`json.load` 已有的 `raw_payload / rer_payload`，再进 pass-1 跳过逻辑。失败回退（JSON 损坏）则清空 marker 重跑。

---

## 8. 实验产物状态

### 8.1 已完成（30 用户迷你）

| 配置 | α | b_transform | protect_top | 用途 |
|---|---|---|---|---|
| `gpt41mini_lfm1k_track` | 0.7 | raw | 0 | 老算法基线 |
| `gpt41mini_lfm1k_track_a05` | 0.5 | raw | 0 | α 消融 |
| `gpt41mini_lfm1k_track_v2` | 0.7 | rank | 5 | v2 主推 |

### 8.2 30 用户迷你结果（v2 vs orig 摘要）

| 维度 | gender | age_group | g+age |
|---|---|---|---|
| **APLT (slice)** | +41% ✅ | +46% ✅ | +42% ✅ |
| **Gini** | -26% ✅ | -26% ✅ | -32% ✅ |
| **MRR** | -1.4% (微输) | -3.0% | -3.8% |
| Precision@20 | -19% | -17% | -39% |
| NDCG@20 | -15% | -15% | -32% |
| HitRate@20 | -9% | -8% | -14% |

**论文级总结**：v2 显著提升长尾曝光（APLT +40%）与分布平等（Gini -28%），早位命中（MRR）几乎无损，付出经典 accuracy-fairness 折衷代价（hit_rate -10%、NDCG -20%）。

### 8.3 进行中：v2 主实验 (n=600)

启动方式：

```bat
python scripts\run_experiments.py --config-dir configs\gpt41mini_lfm1k_track_v2.yaml --sample-size 600
```

预计 2400 LLM 调用 × 4 秒 ≈ **2.5 小时**。Pickle 缓存命中，数据加载 ~10 秒。新 LLM 调用产生 ~$0.5–1.0 成本。

跑完后跑 `python final_report_v2.py` 看完整对比表。

---

## 9. 复现步骤

### 9.1 准备数据（一次性，~5 分钟）

```bat
:: 解压 LFM-1K 原始数据到 data\raw\lastfm-1k\lastfm-dataset-1K\
:: 然后构建单曲流行度表
python -m src.LFM_track_preprocess --min-track-listeners 3
```

输出：`data/processed/lastfm1k_track/items_with_popularity.csv`（272K 单曲 × 16 分组）

### 9.2 预热 adapter pickle（一次性，~5 分钟）

```bat
python diag_lfm_track.py
```

会写入 `data/cache/lastfm1k_track_<hash>.pkl`。后续所有实验秒级加载。

### 9.3 跑实验

```bat
:: 30 用户迷你（首次 ~6 分钟，再跑 ~1 分钟）
python scripts\run_experiments.py --config-dir configs\gpt41mini_lfm1k_track_v2.yaml --sample-size 30

:: 600 用户主实验（首次 ~2.5 小时，缓存命中后 ~10 分钟）
python scripts\run_experiments.py --config-dir configs\gpt41mini_lfm1k_track_v2.yaml --sample-size 600
```

### 9.4 看报告

```bat
:: v2 vs orig 全指标对比
python compare_orig_v2_full.py

:: 主实验完成后的最终报告（含 Pareto 散点）
python final_report_v2.py
```

---

## 10. 局限与未来工作

### 10.1 已知问题

1. **Coverage（语料级唯一推荐数）**：v2 比 orig 下降 ~27%。重排序把推荐集中到了"最 niche 但仍合理"的中段 ~500 首，**个体多样性提升但群体多样性下降**。
   - 修复方向：加入 per-user 反集中惩罚（penalty for items already recommended to many users）
2. **cross 场景在小桶上不稳**：`cross_F_Middle` 仅 40 用户，B 矩阵稀疏 → rank-norm 把噪声放大 → APLT 收益最大但 Precision 损失最大。
   - 修复方向：在 cross 列做 sparse-protection blending（与 neutral 列混合）
3. **adapter 的预处理与 popularity 表是独立的两条路径**：preprocess 脚本与 PopularitySlicer.build() 公式略不同（ML_preprocess style 无 sparse blend）。需要保持一致。

### 10.2 待做

1. **600 用户主实验跑完**：得到统计显著性数据，t-test on per-user metrics with FDR correction
2. **Pareto 散点图**：跑 α=0.3/0.5/0.7/0.85 与 protect=0/5/10 的笛卡尔积，验证 v2 在前沿上
3. **并发改造 LLM adapter**：asyncio + 8 路并发 → 主实验从 2.5 小时压缩到 ~20 分钟
4. **Random rerank baseline**：作为下确界对照
5. **Pure popularity (α=0) baseline**：验证 LLM 顺序确实贡献了精度
6. **画图**：matplotlib 出柱状图、Pareto 散点

---

## 11. 关键文件速查

| 你想要 | 看 |
|---|---|
| 单曲适配器逻辑 | `src/datasets/lastfm1k.py` |
| 单曲提示词 | `prompts/music_track_en.j2` |
| 流行度表预处理 | `src/LFM_track_preprocess.py` |
| 重排序公式 | `src/experiment/rerank.py` |
| B-transform / protect_top 入口 | `src/experiment/runner.py::_apply_b_transform` |
| APLT/Gini 实现 | `src/experiment/metrics.py::aplt_at_k`, `gini_index` |
| v2 配置 | `configs/gpt41mini_lfm1k_track_v2.yaml` |
| 测试 | `tests/test_lastfm1k_track.py` |
| 最终报告生成 | `final_report_v2.py` |

---

*最后更新：迷你实验完成 + 600 用户主实验启动中。算法侧定型；数据等主实验跑完后取最终结论。*
