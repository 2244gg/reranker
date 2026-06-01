# LastFM-1K 单曲重排序研究——全部计算公式与变量定义

本文档罗列研究流水线**每一处使用的数学公式**，按"数据预处理 → 流行度建模 → 重排序算法 → 评估指标 → 统计推断"的顺序组织。每个公式都给出含义、变量解释、代码位置、用途。

---

## 目录

1. [数据预处理](#1-数据预处理)
2. [流行度建模](#2-流行度建模)
3. [B 值变换（v2 创新）](#3-b-值变换v2-创新)
4. [重排序核心公式](#4-重排序核心公式)
5. [评估指标——准确性类](#5-评估指标准确性类)
6. [评估指标——长尾/公平性类](#6-评估指标长尾公平性类)
7. [评估指标——加权命中分类](#7-评估指标加权命中分类)
8. [统计推断](#8-统计推断)
9. [辅助：标题归一化](#9-辅助标题归一化)

---

## 1. 数据预处理

### 1.1 用户-曲目播放次数聚合

**公式**：

$$
c_{u,i} = \sum_{e \in \text{events}} \mathbb{1}[e.\text{user} = u \land e.\text{item} = i]
$$

**变量**：
- $c_{u,i}$：用户 $u$ 听单曲 $i$ 的总播放次数
- events：监听事件流（19.15M 条）
- $\mathbb{1}[\cdot]$：指示函数，括号内成立时为 1，否则 0

**作用**：把流式事件压缩到 (用户 × 单曲) 的二维播放计数矩阵。

**代码位置**：`src/datasets/lastfm1k.py::_stream_aggregate_plays`

---

### 1.2 用户中位播放次数

**公式**：

$$
m_u = \text{median}\{c_{u,i} : i \in \mathcal{I}_u\}
$$

**变量**：
- $m_u$：用户 $u$ 听过所有曲的播放次数中位数
- $\mathcal{I}_u$：用户 $u$ 听过的单曲集合（$c_{u,i} > 0$）

**作用**：作为该用户"喜欢 vs 路过试听"的阈值——播放次数在中位以上视为真喜欢。

**代码位置**：`src/datasets/lastfm1k.py::_load_users` 中 `user_median_pc = float(median(counter.values()))`

---

### 1.3 阳性反馈定义（track 模式）

**公式**：

$$
\text{positive}(u, i) = (c_{u,i} \geq 2) \,\land\, (c_{u,i} > m_u)
$$

**变量**：
- $\text{positive}(u, i)$：布尔，用户 $u$ 是否真"喜欢"单曲 $i$

**作用**：把"听过"细化为"喜欢"。`≥ 2` 排除一次性试听，`> m_u` 要求高于该用户的中位 listen 强度。两个条件之合取去掉所有"无意听到一次"的噪声。

**代码位置**：`src/datasets/lastfm1k.py::positive_feedback_predicate`

---

### 1.4 训练/测试集划分

**公式**：

$$
\mathcal{T}_u = \text{Sample}_{\text{rng}}(\mathcal{P}_u, \,\lfloor 0.7 \cdot |\mathcal{P}_u| \rfloor)
$$

$$
\mathcal{H}_u = \mathcal{P}_u \setminus \mathcal{T}_u
$$

**变量**：
- $\mathcal{P}_u = \{i : \text{positive}(u, i)\}$：用户 $u$ 的阳性反馈集合
- $\mathcal{T}_u$：测试集（占 70%）
- $\mathcal{H}_u$：作为给 LLM 的历史输入（占 30%）
- rng：固定随机种子 42 保证复现

**作用**：留出 70% 真实喜爱用作"金标准"评估，剩 30% 给 LLM 做上下文。

**代码位置**：`src/datasets/base.py::split`

---

## 2. 流行度建模

### 2.1 分组原始计数

**公式**：

$$
\text{count}(i, g) = \sum_{u \in g} \mathbb{1}[\text{positive}(u, i)]
$$

**变量**：
- $g$：人口切片（如 `gender_M`、`age_Youth`、`cross_F_Middle`）
- $\text{count}(i, g)$：分组 $g$ 内有多少用户阳性反馈听过 $i$

**作用**：在每个人口切片内度量物品的"被喜爱次数"。

**代码位置**：`src/LFM_track_preprocess.py::calculate_popularity`

---

### 2.2 分组流行度归一化（B_raw）

**公式**：

$$
B_{\text{raw}}(i, g) = \dfrac{\text{count}(i, g)}{\max\limits_{j} \text{count}(j, g)}
$$

**变量**：
- $B_{\text{raw}}(i, g) \in [0, 1]$：item $i$ 在切片 $g$ 内的相对流行度
- $\max_j \text{count}(j, g)$：切片 $g$ 中最热门那首曲的被喜爱次数

**作用**：让每个切片列内最热门 = 1.0，最冷门 ≈ 0，从绝对计数转成相对热度，跨切片可比。

**代码位置**：`src/LFM_track_preprocess.py::normalize_popularity`

---

### 2.3 稀疏切片保护混合（PopularitySlicer 内置，预处理脚本未启用）

**公式**：

$$
B_{\text{blended}}(i, g) = w \cdot B_{\text{raw}}(i, g) + (1 - w) \cdot B_{\text{raw}}(i, \text{neutral})
$$

$$
w = \min\!\left(1,\; \dfrac{n_g}{n_{\text{thr}}}\right)
$$

**变量**：
- $n_g$：切片 $g$ 的用户数
- $n_{\text{thr}}$：保护阈值（默认 30）
- $w$：混合权重

**作用**：当切片用户数少（如 cross_U_Youth 才 6 人）时，把该列向 neutral 列收缩，缓解小样本噪声。我们的 `LFM_track_preprocess.py` 跳过此步（按 ML_preprocess.py 风格不含），改用别的稳健化（参见 §3.1 rank 转换）。

**代码位置**：`src/popularity/slicer.py::PopularitySlicer.build`

---

## 3. B 值变换（v2 创新）

### 3.1 排名分位数转换（rank-normalized）

**公式**：

$$
B'(i, g) = \begin{cases}
\dfrac{\text{rank}_g(i)}{|\{j : B_{\text{raw}}(j, g) > 0\}|} & \text{if } B_{\text{raw}}(i, g) > 0 \\
0 & \text{otherwise}
\end{cases}
$$

其中 $\text{rank}_g(i)$ 是 $i$ 在切片 $g$ 列里**非零项**中的从小到大排名（method=`average`）。

**变量**：
- $B'(i, g) \in [0, 1]$：分位数（最热门 = 1.0，最冷门 ≈ 1/n_pos）
- $|\{j : B_{\text{raw}}(j, g) > 0\}|$：切片 $g$ 内有非零计数的物品数

**作用**：
- 解决长尾 collapse 问题：原 $B_{\text{raw}}$ 极度偏度（max=224，median≈0.005），导致 $(1-B)$ 几乎处处 ≈ 1，奖励项失去区分度。
- 转换后 $B'$ 在非零集合内**均匀分布**，$(1-B')$ 在 [0, 1] 也均匀，重排序信号有真实梯度。

**代码位置**：`src/experiment/runner.py::_apply_b_transform` 中 transform=`rank` 分支

---

### 3.2 饱和阈值变换（备选）

**公式**：

$$
B''(i, g) = \min\!\left(\dfrac{B_{\text{raw}}(i, g)}{B_{\text{thr}}},\; 1\right)
$$

**变量**：
- $B_{\text{thr}}$：阈值（默认 0.3）

**作用**：只对真主流（$B > 0.3$）的物品给满分惩罚，中冷段保留 LLM 原序。**v2 实验未启用**，作为消融备选。

**代码位置**：同上，transform=`saturate` 分支

---

## 4. 重排序核心公式

### 4.1 排名衰减偏好分数（Preference）

**公式**：

$$
P_i^{\text{norm}} = \begin{cases}
\text{base} + (1 - \text{base}) \cdot \dfrac{r^{\text{rank}_i} - r^{N}}{1 - r^N} & \text{if } 1 - r^N \not\approx 0 \\
\text{base} + (1 - \text{base}) \cdot \dfrac{N - \text{rank}_i}{N - 1} & \text{otherwise(数值兜底)}
\end{cases}
$$

**变量**：
- $\text{rank}_i$：物品 $i$ 在 LLM 推荐列表中的位次（1-indexed，1 = 第一位）
- $N$：候选池大小（v2 配置 = 30）
- $\text{base}$：最低分（v2 = 0.2，第 N 位获得 0.2 分）
- $r$：衰减率（v2 = 0.9）
- $P_i^{\text{norm}} \in [\text{base}, 1]$：归一化偏好分数

**作用**：把 LLM 的整数排名转成 [0.2, 1] 平滑分数，把"LLM 排第几"作为推荐分量编码到重排序公式里。指数衰减让前几位差距大、后几位差距小，符合 LLM 推荐的可信度梯度。

**代码位置**：`src/experiment/rerank.py::preference_probability`

---

### 4.2 用户主流度分数

**公式**：

$$
s_u = \sum_{i \in \mathcal{H}_u^{\text{top-}L}} B'(i, g_u)
$$

**变量**：
- $\mathcal{H}_u^{\text{top-}L}$：用户 $u$ 历史中按播放次数排序的 top-L 曲（L = `history_limit` = 25）
- $g_u$：用户 $u$ 在当前场景下所属的切片
- $s_u$：用户主流度——其历史 top 25 曲在所属人口切片下的 B' 之和

**作用**：度量该用户"听的曲是否在他/她的人口圈子里也算热门"，用于动态 $\alpha$ 调节。

**代码位置**：`src/experiment/runner.py` pass-1 中 `hist_score` 计算

---

### 4.3 场景平均主流度

**公式**：

$$
\bar{s}_g = \dfrac{1}{|\mathcal{U}_g|} \sum_{u \in \mathcal{U}_g} s_u
$$

**变量**：
- $\mathcal{U}_g$：当前场景下采样到的用户集合（典型 600）
- $\bar{s}_g$：所有这些用户的 $s_u$ 均值

**作用**：作为基准点，与 $s_u$ 比较以判定该用户是否比群体更主流。

**代码位置**：`src/experiment/runner.py` 第二遍循环之前从 pass-1 累计算出

---

### 4.4 动态 alpha（user-adaptive）

**公式**：

$$
\alpha_u = \text{clip}\!\left(\alpha_{\text{base}} + \alpha_{\text{gain}} \cdot \dfrac{s_u - \bar{s}_g}{\max(\bar{s}_g, \varepsilon)},\;\; \alpha_{\min},\; \alpha_{\max}\right)
$$

**变量**：
- $\alpha_{\text{base}} = 0.7$：基础 $\alpha$
- $\alpha_{\text{gain}} = 0.2$：调整幅度
- $\alpha_{\min} = 0.5$，$\alpha_{\max} = 0.9$：clip 区间
- $\varepsilon = 10^{-12}$：除零保护
- $\text{clip}(x, a, b) = \max(a, \min(b, x))$

**作用**：
- 主流用户（$s_u > \bar{s}_g$）→ 中心化比率为正 → $\alpha \uparrow$ → 更信 LLM → 少去偏
- niche 用户（$s_u < \bar{s}_g$）→ 中心化比率为负 → $\alpha \downarrow$ → 更激进给 (1-B) 奖励
- 这是**个人化哲学**：算法**符合**用户口味而非**纠正**用户。

**代码位置**：`src/experiment/rerank.py::compute_dynamic_alpha`

---

### 4.5 重排序总分

**公式**：

$$
S_i = \alpha_u \cdot P_i^{\text{norm}} + (1 - \alpha_u) \cdot (1 - B'(i, g_u))
$$

**变量**：
- $S_i$：物品 $i$ 的重排序总分
- 第一项 $\alpha_u P_i^{\text{norm}}$：偏好分（信 LLM）
- 第二项 $(1 - \alpha_u)(1 - B'(i, g_u))$：长尾奖励（信群体冷门度）

**作用**：以 $\alpha_u$ 加权两者得到候选项的最终分。重排序后 top-K 由 $S_i$ 降序选出。

**代码位置**：`src/experiment/rerank.py::rerank_one_list`

---

### 4.6 Top-K 保护排序（v2 创新）

**公式**（伪代码）：

```
ordered_head = LLM_rec[:protect_top]                         # 保留前 K' 名 LLM 排序
ordered_tail = sort(LLM_rec[protect_top:], key=S_i, reverse=True)
final_top_K  = (ordered_head + ordered_tail)[:K]
```

**变量**：
- protect_top $= K' = 5$：保护位数
- LLM_rec：LLM 给出的 N 个候选（按 LLM 排序）
- final_top_K：最终 top-K 推荐列表（K=20）

**作用**：锁定 LLM 给出的精准头部命中（前 5 位），剩余 25 位再做 $S_i$ 重排，merge 后 cut 到 K=20。这是 v2 关键改进，让 MRR 损失从 -10% 压到 -2%。

**代码位置**：`src/experiment/rerank.py::rerank_one_list` 末尾 protect_top 分支

---

## 5. 评估指标——准确性类

### 5.1 Hits（命中数）

**公式**：

$$
\text{Hits}_u^{(K)} = |\text{Rec}_u^{(K)} \cap \mathcal{T}_u|
$$

**变量**：
- $\text{Rec}_u^{(K)}$：用户 $u$ 的 top-K 推荐列表
- $\mathcal{T}_u$：用户 $u$ 的测试集
- 实际比较使用 `title_key` 模糊匹配（参见 §9）

**代码位置**：`src/experiment/metrics.py::count_matches`

---

### 5.2 Precision@K

**公式**：

$$
\text{Precision@K}_u = \dfrac{\text{Hits}_u^{(K)}}{\min(K, |\text{Rec}_u|)}
$$

**作用**：top-K 里有多少比例是命中。越高越好。

**代码位置**：`src/experiment/metrics.py::precision_at_k`

---

### 5.3 Recall@K

**公式**：

$$
\text{Recall@K}_u = \dfrac{\text{Hits}_u^{(K)}}{|\mathcal{T}_u|}
$$

**作用**：测试集里有多少比例被推到了 top-K。越高越好。

**代码位置**：`src/experiment/metrics.py::recall_at_k`

---

### 5.4 HitRate@K

**公式**：

$$
\text{HitRate@K}_u = \mathbb{1}[\text{Hits}_u^{(K)} > 0]
$$

**作用**：top-K 是否至少有一个命中（二值）。最低限度的有效推荐。

**代码位置**：`src/experiment/metrics.py::hit_rate_at_k`

---

### 5.5 MRR（Mean Reciprocal Rank）

**公式**：

$$
\text{MRR}_u = \dfrac{1}{r^{*}_u} \quad \text{若有命中,否则 0}
$$

$$
r^{*}_u = \min\{r : \text{Rec}_u[r] \in \mathcal{T}_u\}
$$

**变量**：
- $r^{*}_u$：第一个命中的 1-indexed 位置

**作用**：第一命中越靠前分数越高（位置 1 → 1.0，位置 2 → 0.5，位置 5 → 0.2）。重视早位精度。

**代码位置**：`src/experiment/metrics.py::mrr_first_hit`

---

### 5.6 NDCG@K（Normalized Discounted Cumulative Gain）

**公式**：

$$
\text{DCG@K}_u = \sum_{i=1}^{K} \dfrac{\mathbb{1}[\text{Rec}_u[i] \in \mathcal{T}_u]}{\log_2(i + 1)}
$$

$$
\text{IDCG@K}_u = \sum_{i=1}^{\min(K, |\mathcal{T}_u|)} \dfrac{1}{\log_2(i + 1)}
$$

$$
\text{NDCG@K}_u = \dfrac{\text{DCG@K}_u}{\text{IDCG@K}_u}
$$

**作用**：考虑命中位置的折扣累积增益，标准化到 [0, 1]。位置越靠前贡献越大，是综合性最强的排序指标。

**代码位置**：`src/experiment/metrics.py::ndcg_at_k`

---

## 6. 评估指标——长尾/公平性类

### 6.1 APLT（Average Percentage of Long-Tail items）

**公式**：

$$
\text{APLT}_u = \dfrac{\big|\{i \in \text{Rec}_u^{(K)} : B'(i, g_u) < \theta\}\big|}{K}
$$

**变量**：
- $\theta = 0.2$：长尾阈值
- $g_u$：用户 $u$ 在当前场景下的切片
- 我们同时报 `aplt`（用 $g_u$）与 `aplt_neutral`（用 `neutral_All` 列）

**作用**：top-K 里有多大比例是冷门曲。这是 Abdollahpouri et al. 2017 提出的去偏研究标配指标，不依赖测试集匹配，纯粹度量曝光。

**代码位置**：`src/experiment/metrics.py::aplt_at_k`

---

### 6.2 Coverage（语料级）

**公式**：

$$
\text{Coverage} = \dfrac{\big|\bigcup_{u} \text{Rec}_u^{(K)}\big|}{|\mathcal{C}|}
$$

**变量**：
- $\bigcup_u$：跨全部用户取并集
- $|\mathcal{C}|$：物品库总数（流行度表行数 = 272,769）
- 比较时用 `title_key` 归一化

**作用**：所有用户加起来推荐到了库里多少比例的物品。越高代表多样性越好。

**代码位置**：`src/experiment/runner.py::_aggregate_metrics`

---

### 6.3 Gini 系数

**公式**：

$$
G = \dfrac{2 \sum_{i=1}^{n} i \cdot v_{(i)}}{n \cdot \sum_{i=1}^{n} v_{(i)}} - \dfrac{n + 1}{n}
$$

**变量**：
- $v_{(1)} \le v_{(2)} \le \dots \le v_{(n)}$：跨用户聚合得到的物品被推荐次数（仅非零，从小到大排序）
- $G \in [0, 1)$：0 = 完全平等，越接近 1 越被少数物品垄断

**作用**：度量推荐曝光在物品集上的不平等度。低 Gini = 算法把曝光分得更均匀。

**代码位置**：`src/experiment/metrics.py::gini_index`

---

## 7. 评估指标——加权命中分类

### 7.1 难度分函数（三种 scheme）

**公式**：

$$
\text{score}(B) = \begin{cases}
(1 - B) \cdot 10 & \text{linear\_x10}（默认） \\
-\log_2(B + \varepsilon) & \text{log},\; \varepsilon = 10^{-3} \\
\begin{cases}1 & B \geq 0.8 \\ 2 & 0.5 \leq B < 0.8 \\ 3 & 0.2 \leq B < 0.5 \\ 5 & B < 0.2\end{cases} & \text{bucket}
\end{cases}
$$

**变量**：
- 输入 $B \in [0, 1]$（在 v2 实验中是 $B'$ 排名分位数）
- 输出范围：linear_x10 ∈ [0, 10]，log ∈ [0, 9.97]，bucket ∈ {1, 2, 3, 5}

**作用**：把 B 转成"难度"。我们 v2 实验用 linear_x10。log 在长尾上分辨率更高，bucket 给离散档位。

**代码位置**：`src/popularity/item_scoring.py::difficulty_score`

---

### 7.2 UnpopularityScore（命中加权总分，UnpopTotal / UnpopAvg）

**公式**（每命中的加权分）：

$$
w_i = \text{score}_{\text{linear\_x10}}(B^{*}(i, g_u)) \cdot \left(1 + \rho \cdot \dfrac{K - r_i^{(0)}}{K}\right)
$$

$$
B^{*}(i, g_u) = \begin{cases}
0.5 & \text{if popularity\_lookup}(i, g_u) = 0 \;(\text{fallback}) \\
\text{popularity\_lookup}(i, g_u) & \text{otherwise}
\end{cases}
$$

$$
\text{UnpopTotal}_u = \sum_{i \in \text{Rec}_u^{(K)} \cap \mathcal{T}_u} w_i
$$

$$
\text{UnpopAvg}_u = \dfrac{\text{UnpopTotal}_u}{\text{Hits}_u^{(K)}}
$$

**变量**：
- $r_i^{(0)} \in \{0, \dots, K-1\}$：物品 $i$ 的 0-indexed 排位
- $\rho = 0.2$：rank_weight，对靠前命中加成的权重
- $B^{*}$：含 0.5 fallback 的 lookup 值（参见你之前问过的 footgun）

**作用**：累计每命中曲的加权"冷门度"。这是仓库原作者设计的"按命中物品总分"算法。但因 fallback 把"切片内零正反馈"也一并替换为 0.5，**会高估真冷门度**。

**代码位置**：`src/experiment/metrics.py::unpopularity_score`

---

### 7.3 DifficultyScore（命中加权总分严格版，DiffTotal / DiffAvg）

**公式**（每命中的加权分）：

$$
w_i^{\text{diff}} = \text{score}_{\text{linear\_x10}}(B^{**}(i, g_u)) \cdot \left(1 + \rho \cdot \dfrac{K - r_i^{(0)}}{K}\right)
$$

$$
B^{**}(i, g_u) = \begin{cases}
0 & \text{if title 不在 popularity 表} \\
B'(i, g_u) & \text{否则（含 cell 值为 0 的情况）}
\end{cases}
$$

$$
\text{DiffTotal}_u = \sum_{i \in \text{Rec}_u^{(K)} \cap \mathcal{T}_u} w_i^{\text{diff}}
$$

$$
\text{DiffAvg}_u = \dfrac{\text{DiffTotal}_u}{\text{Hits}_u^{(K)}}
$$

**变量**：
- $B^{**}$：**没有 0.5 fallback**——表内但 cell = 0 时按真冷门 $B = 0$ 处理 → score = 10
- 物理意义：「该切片的所有人都没正反馈听过 = 极度难推」

**作用**：严格版加权命中总分，不被 fallback 工件污染。论文里推荐用此作为主指标。

**代码位置**：`src/popularity/item_scoring.py::difficulty_weighted_hit_score`

---

### 7.4 UnpopAvg vs DiffAvg 数值差异（同公式 + 不同 fallback）

| 命中曲在切片的状态 | UnpopAvg 给分 | DiffAvg 给分 |
|---|---|---|
| 切片内 $B' > 0$（有正反馈） | $(1 - B') \cdot 10$ | $(1 - B') \cdot 10$ |
| 切片内 $B' = 0$（cell 是 0）| $(1 - 0.5) \cdot 10 = 5$ ← fallback | $(1 - 0) \cdot 10 = 10$ |
| 表里完全找不到 | $5$（fallback）| $0$ |

**作用**：UnpopAvg / DiffAvg 是同一估计目标的两种估计器，它们的差距反映 fallback 假设的影响范围。

---

## 8. 统计推断

### 8.1 配对差分

**公式**：

$$
d_u = \text{metric}_u^{\text{rer}} - \text{metric}_u^{\text{orig}}
$$

**变量**：
- 每个用户都有一对 (orig, rer) 同指标，两者用同一个 LLM 候选池所以可配对
- $d_u$：该用户该指标的逐个差值

**作用**：消除用户间方差，专注算法效应。

---

### 8.2 配对均值与标准差

**公式**：

$$
\bar{d} = \dfrac{1}{n} \sum_{u=1}^{n} d_u, \quad
s_d = \sqrt{\dfrac{1}{n-1} \sum_{u=1}^{n} (d_u - \bar{d})^2}
$$

$$
\text{SE}(\bar{d}) = \dfrac{s_d}{\sqrt{n}}
$$

**作用**：算法效应量与其样本不确定性。

---

### 8.3 配对 t 统计量

**公式**：

$$
t = \dfrac{\bar{d}}{\text{SE}(\bar{d})} = \dfrac{\bar{d}}{s_d / \sqrt{n}}
$$

**作用**：检验"算法在此指标上有效"假设的显著性（$H_0: \bar{d} = 0$）。

---

### 8.4 双尾 p 值（用正态近似，n=600 充足大）

**公式**：

$$
p \approx 2 \cdot \big(1 - \Phi(|t|)\big)
$$

其中 $\Phi$ 是标准正态 CDF。

**作用**：量化效应观察到的偶然概率。

---

### 8.5 95% 置信区间（正态近似）

**公式**：

$$
\bar{d} \pm 1.96 \cdot \text{SE}(\bar{d})
$$

**作用**：给均值差一个区间估计，比单点估计更稳健。

**代码位置**：`detailed_stats_v2.py::paired_t`

---

### 8.6 效应量（Cohen's d）

**公式**：

$$
d_{\text{Cohen}} = \dfrac{\bar{d}}{s_d}
$$

**变量**：标准化差异，单位为标准差。
- $|d| < 0.2$：可忽略
- $0.2 \leq |d| < 0.5$：小
- $0.5 \leq |d| < 0.8$：中
- $|d| \geq 0.8$：大

**作用**：摆脱样本量影响纯报效应量。我们 APLT 增益的 $d \approx 2.3$ 远超"大"阈值。

---

## 9. 辅助：标题归一化

### 9.1 normalize_title（轻量）

**算法**：
1. 去尾随年份 `(YYYY)`
2. 重排定冠词："X, The" → "The X"

**作用**：兼容 ML-1M `movies_with_popularity.csv` 的祖传键格式。

**代码位置**：`src/experiment/metrics.py::normalize_title`

---

### 9.2 title_key（激进）

**算法**：
1. HTML 反转义（`&amp;` → `&`）
2. 调用 `normalize_title`
3. 反复去括号 `[…]` `(…)`（最多 3 层）
4. 去尾随 ` by Author Name`
5. 冒号副标题截断（`Lord of Rings: Fellowship` → `Lord of Rings`）
6. 转小写、去标点（`[^\w\s]`）、空白合并

**作用**：兼容 LLM 装饰输出（"(Remastered)"、"feat."、em-dash 等）。在我们的代码里**统一用于命中检测和 popularity 表查询**。

**代码位置**：`src/experiment/metrics.py::title_key`

---

### 9.3 模糊匹配（titles_match）

**公式**（布尔）：

$$
\text{match}(a, b) = \mathbb{1}[\text{key}(a) = \text{key}(b)] \lor \mathbb{1}[\min(|s|,|l|) \geq 8 \land s \subset l]
$$

其中 $s, l$ 是两个 key 中的较短 / 较长方。

**作用**：除了精确相等，允许较短 key 是较长 key 的子串（≥8 字符）也算匹配，覆盖 "Yesterday - The Beatles" 与 "Yesterday - The Beatles (2009 Remaster)" 这种装饰差异。

**代码位置**：`src/experiment/metrics.py::titles_match`

---

## 10. 公式间的依赖关系图

```
原始事件 → c_{u,i} (1.1) → m_u (1.2) → positive(u,i) (1.3) → 划分 H_u, T_u (1.4)
                                          ↓
                                    count(i, g) (2.1)
                                          ↓
                                    B_raw(i, g) (2.2)
                                          ↓
                                    B'(i, g) (3.1, rank转换)
                                          ↓
              ┌───────────────────────────┼───────────────────────────┐
              ▼                           ▼                           ▼
        s_u (4.2)                    P_i_norm (4.1)             score(B') (7.1)
              ↓                           ↓                           ↓
         α_u (4.4)              ↘                           ↗     w_i (7.2/7.3)
              ↓                  ↘                         ↗            ↓
              └→→→→→→→→→→→→→→→→→ S_i (4.5) ←←←←←←←←←←←←←←      UnpopTotal/DiffTotal
                                  ↓                                     ↓
                          protect_top 排序 (4.6)                 配对 t 检验 (§8)
                                  ↓
                            top-K 列表
                                  ↓
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        Hits/MRR (5)       APLT/Coverage/Gini (6)   配对 t 检验 (§8)
```

---

## 11. 配置参数总表

| 参数 | 值 | 控制公式 | 作用 |
|---|---|---|---|
| `min_interactions` | 20 | (1.4) 用户筛选 | 排除监听太少的用户 |
| `min_track_listeners` | 3 | (2.1) 物品筛选 | 排除冷曲噪声 |
| `test_split` | 0.7 | (1.4) | 70% 留作测试 |
| `top_k` (K) | 20 | (4.6, 5.x) | 最终输出列表长度 |
| `candidate_pool_size` (N) | 30 | (4.1) | LLM 候选池大小 |
| `history_limit` (L) | 25 | (4.2) | 历史 top-L 用于 $s_u$ |
| `base` | 0.2 | (4.1) | P 最低分 |
| `decay_rate` (r) | 0.9 | (4.1) | 排名衰减 |
| `alpha_base` | 0.7 | (4.4) | α 中心 |
| `alpha_min` | 0.5 | (4.4) | α 下限 |
| `alpha_max` | 0.9 | (4.4) | α 上限 |
| `alpha_gain` | 0.2 | (4.4) | α 调节幅度 |
| `b_transform` | "rank" | (3.1) | B 转换方式 |
| `protect_top` (K') | 5 | (4.6) | 前 K' 位锁定 |
| `rank_weight` (ρ) | 0.2 | (7.2, 7.3) | 命中位置加成 |
| `aplt threshold` (θ) | 0.2 | (6.1) | 长尾分界 |

---

## 12. 公式按"研究 contribution"分类

| 公式编号 | 出处 / 性质 |
|---|---|
| 1.x、2.x | **数据规范定义**——领域标准做法 |
| 3.1 (rank 转换) | **本研究 v2 创新**——解决长尾 collapse |
| 4.1 (P)、4.2、4.3、4.5 (S) | **仓库原作者设计**——动态 $\alpha$ 重排序 |
| 4.4 (动态 $\alpha$) | **仓库原作者设计**，本研究保留并 justify "个人化方向" |
| 4.6 (protect_top) | **本研究 v2 创新**——保护早位精度 |
| 5.x | 信息检索领域标准（CIKM、SIGIR 沿用数十年） |
| 6.1 APLT | Abdollahpouri et al. 2017（去偏文献标配） |
| 6.2 Coverage | Steck 2018（推荐多样性标准） |
| 6.3 Gini | 经济学借用（Atkinson 1970 引入推荐评估） |
| 7.x | **仓库原作者设计**——扩展自 popularity-debias 文献 |
| 8.x | 经典统计学 |
| 9.x | **仓库原作者设计 + 本研究 title_key 一致化** |

---

**报告人**：[填写]  
**日期**：2026 年 5 月-6 月  
**配合阅读**：`docs/lastfm1k_v2_main_results.md`（实验结果）、`docs/lastfm_track_rerank.md`（代码与流水线技术细节）
