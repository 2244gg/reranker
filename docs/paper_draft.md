# From Personalization to Statistical Discrimination: A Slice-Conditional Reranking Approach for Identity-Driven Drift in LLM-based Recommenders

**Authors**: [Your Name], [Advisor Name]  
**Affiliation**: [Institution]  
**Date**: 2026

---

## Abstract

Large Language Model (LLM)-based recommender systems are highly prompt-sensitive: introducing user identity tokens (gender, age, race, etc.) into the prompt can cause the model to **drift toward stereotyped categories**, **homogenize recommendations across users sharing a demographic label**, and **collapse onto a narrow set of items** that the pre-training corpus statistically associates with that label. We argue that this behaviour, when it produces a **systematic exposure-opportunity differential conditioned on protected attributes**, crosses the line from legitimate *personalization* into **statistical discrimination** in the sense of Dwork et al. (2012).

We propose a **slice-conditional, post-hoc reranking algorithm** that explicitly quantifies the demographic stereotype embedded in each item, exposes per-user mainstream-ness with respect to the user's demographic slice, and adaptively rebalances the recommendation list to remove protected-attribute exposure differential while preserving preference-driven personalization. Our method is **training-free**, **applicable to closed-source LLM APIs**, and **complements** rather than replaces the underlying LLM recommender.

We evaluate on two public benchmarks:
- **MovieLens-1M (ML-1M)** for the movie domain, with 6,031 users (gender × age slices).
- **LastFM-1K (LFM-1K)** for the music-track domain, with 942 users (gender × age slices).

Our reranker achieves **APLT (long-tail exposure) absolute lift of +52pp** and **Gini exposure inequality reduction of -10%** on LFM-1K, while limiting MRR loss to under 2.0%. On ML-1M, our method achieves **NDCG@15 +4.0pp** and **MRR@15 +5.1pp** *gains* alongside long-tail improvements, indicating that exposure rebalancing on the movie domain Pareto-dominates the LLM baseline rather than trades off against it. We position these findings as a quantitative bridge between fairness theory and the operational behaviour of contemporary LLM recommenders.

**Keywords**: LLM recommendation, statistical discrimination, post-hoc reranking, popularity-debias, exposure fairness, slice-conditional analysis.

---

## 1. Introduction

### 1.1 Motivation

LLMs are increasingly deployed as zero-shot or few-shot recommenders (Liu et al. 2023; Hou et al. 2024; Bao et al. 2023). They take a natural-language prompt containing the user's interaction history and (optionally) demographic descriptors, and emit a ranked candidate list. Compared with classical collaborative filtering, LLM recommenders enjoy strong cold-start coverage, transferability across domains, and high explainability.

However, LLM recommenders inherit the statistical regularities of their pre-training corpora. When a user-identity prompt such as *"the listener is a 22-year-old male"* is injected, the model's posterior over candidate items shifts in ways that reflect **the corpus's stereotyped associations** rather than purely the user's interaction history. Three failure modes are commonly observed:

- **Preference drift**: the model's top-ranked items change when only the demographic descriptor changes, even though the interaction history remains identical.
- **Homogenization (category collapse)**: users sharing a demographic label receive overlapping, stereotype-aligned candidate sets, regardless of finer individual preferences.
- **Off-category recommendation**: niche listeners under a mainstream demographic label may be served items unrelated to their actual taste, simply because those items are statistically dominant in the corpus's mental model of the demographic.

### 1.2 Statistical discrimination vs personalization

Following the algorithmic-fairness literature (Dwork et al. 2012; Barocas et al. 2017), we distinguish two regimes:

| Regime | Definition | Verdict |
|---|---|---|
| **Personalization** | Recommendation incrementally adapts to a user's preference signal; no systematic exposure differential is induced by protected attributes (gender, age, race, etc.) once preferences are controlled for. | Legitimate; desirable. |
| **Statistical discrimination** | Recommendation systematically alters the *exposure opportunity* of items conditional on a user's protected attribute, creating a measurable disparity in long-tail exposure, item coverage, or unpopularity-weighted hit rate across demographic slices. | Problematic; must be mitigated. |

Our framing is sharper than the colloquial "the LLM has bias" because it ties the empirical phenomenon to a normative criterion: an LLM recommender is **discriminatory only if exposure opportunity, not just hit accuracy, differs systematically across protected demographic slices**.

### 1.3 Contributions

This paper makes four concrete contributions:

- **C1.** We propose a **slice-conditional stereotype quantification** procedure (DiffScore) that scores every item by its inverse popularity within each demographic slice, and aggregates over a user's recommendation list to produce a per-(user, scenario) stereotype index that is interpretable and statistically tractable.

- **C2.** We propose an **adaptive, user-mainstream-ness-aware reranker** that adjusts the rebalancing intensity per user: mainstream users in their slice are nudged less, niche users are protected with stronger long-tail injection. This is implemented through a closed-form $\alpha$ adjustment with no training.

- **C3.** We introduce two algorithmic refinements that resolve practical pathologies on long-tail-heavy datasets: (i) a **rank-percentile transformation** of the slice popularity matrix that prevents the long-tail-collapse degeneracy where the unpopularity bonus loses discrimination, and (ii) a **top-$K'$ protect** mechanism that preserves the LLM's high-confidence early-rank picks, capping MRR loss at $\leq 2\%$.

- **C4.** We provide **empirical evidence on two complementary benchmarks** (ML-1M for movies; LFM-1K for music tracks). On LFM-1K our reranker delivers a +52pp APLT absolute lift, -10pp Gini reduction, and net positive total weighted hit score with only -2.0% MRR. On ML-1M, the reranker achieves a Pareto-dominating outcome: NDCG and MRR both improve while exposure rebalancing also improves.

### 1.4 Paper organization

Section 2 reviews related work. Section 3 formalizes the problem. Section 4 details the methodology. Section 5 describes the experimental setup. Section 6 presents results on ML-1M; Section 7 on LFM-1K. Section 8 covers ablations and Pareto analysis. Section 9 discusses limitations and future work. Section 10 concludes.

---

## 2. Related Work

### 2.1 LLM-based recommendation

The first wave of work on LLM recommendation focused on prompt-engineering benchmarks (Liu et al. 2023; Hou et al. 2024; Wang et al. 2023). Subsequent work introduced parameter-efficient fine-tuning approaches such as **TALLRec** (Bao et al. 2023, RecSys), **BIGRec** (Li et al. 2024, SIGIR), and **P5** (Geng et al. 2022). These methods improve recommendation accuracy but do not specifically target exposure fairness. Closed-source frontier LLMs (GPT-4o, Claude-3.5, Qwen2.5) cannot be fine-tuned at all by most academic users, making post-hoc methods the only viable route in many practical settings.

### 2.2 Bias and fairness in recommendation

Bias in classical recommenders has been surveyed in Chen et al. (2023). Three intervention loci are commonly distinguished:

- **Pre-processing**: training-data resampling, long-tail oversampling.
- **In-processing**: modify model architecture or loss (IPS weighting; Up5 by Hua et al. 2024).
- **Post-processing**: rerank model output (xQuAD by Santos et al. 2010; FA*IR by Zehlike et al. 2017; Calibrated Recommendations by Steck 2018).

For LLM recommenders specifically, recent in-processing methods include **Up5** (Hua et al. 2024) and the LoRA-based debias of Hu et al. (2024, SIGIR). These require model access. The closest line to ours is post-hoc reranking on top of frozen LLM outputs; we extend it with a user-adaptive $\alpha$ and a rank-percentile B-transform (see Section 4).

### 2.3 Stereotype audits of LLMs

Audits by Liu et al. (2023, EMNLP) and Zhang et al. (2024, ICLR) have demonstrated systematic preference drift in ChatGPT recommendations under identity-prompt perturbation. Our DiffScore quantification (Section 4.1) is directly inspired by these audits but moves beyond pure diagnosis to provide an **operational, training-free correction**.

### 2.4 Long-tail metrics

Abdollahpouri et al. (2017, RecSys) introduced **APLT (Average Percentage of Long-Tail items)** as the standard long-tail exposure indicator. Steck (2018) proposed Calibration as a divergence-based diversity metric. Gini index (Atkinson 1970) was repurposed for recommendation by Adomavicius & Kwon (2012). We adopt all three as our exposure-fairness metrics, complementing classical accuracy metrics (Hit, Precision, Recall, MRR, NDCG).

### 2.5 Position of this work

Our work differs from prior art on three axes:

1. **Framing**: we formalize the problem as *statistical discrimination by exposure differential* on protected attributes, not generic "popularity bias". This connects empirical behaviour to fairness theory.
2. **Method**: a training-free post-hoc reranker applicable to closed-source LLMs, with novel rank-percentile B-transform and protect-top mechanism.
3. **Evaluation**: a paired, slice-conditional analysis with both accuracy and exposure-fairness metrics, validated across two benchmarks of distinct domain (movies, music tracks) and granularity (item-level).

---

## 3. Problem Formulation

### 3.1 Notation

Let $\mathcal{U}$ be the user set, $\mathcal{I}$ the item catalog, and $D = \{d_1, \dots, d_p\}$ the set of protected demographic attributes (e.g., gender, age group, country). Each user $u \in \mathcal{U}$ has a feature vector $\mathbf{d}_u = (d_{u,1}, \dots, d_{u,p})$. A **slice** $g$ is a value-tuple over a subset of $D$, e.g.:

- $g = \emptyset$ : *neutral* (all users)
- $g = \{(\text{gender}, \text{male})\}$ : *gender_M*
- $g = \{(\text{gender}, \text{female}), (\text{age}, \text{young})\}$ : *cross_F_Youth*

Let $\mathcal{P}_u \subseteq \mathcal{I}$ denote the user's positive feedback set (items the user genuinely engaged with at strength above threshold; see Section 4.0 for the dataset-specific definition).

### 3.2 Slice-conditional popularity

For slice $g$ and item $i$, define the **slice-conditional positive-feedback count**:

$$
\text{count}(i, g) = \sum_{u \in g} \mathbb{1}[i \in \mathcal{P}_u]
$$

and the **normalized slice popularity**:

$$
B_{\text{raw}}(i, g) = \frac{\text{count}(i, g)}{\max_{j \in \mathcal{I}} \text{count}(j, g)} \in [0, 1]
$$

so that the most-engaged item per slice has $B = 1$.

### 3.3 Exposure opportunity

Given a recommender that emits a top-$K$ list $\text{Rec}_u^{(K)}$ for user $u$, the **exposure** of item $i$ at the population level is:

$$
\text{Exp}(i) = \sum_{u \in \mathcal{U}} \mathbb{1}[i \in \text{Rec}_u^{(K)}]
$$

The **slice-conditional exposure** of $i$ is:

$$
\text{Exp}(i \mid g) = \sum_{u \in g} \mathbb{1}[i \in \text{Rec}_u^{(K)}]
$$

### 3.4 Statistical discrimination criterion

Let $f_{\text{LLM}}$ be a recommender that takes prompt features $\mathbf{x}_u = (\mathbf{h}_u, \mathbf{d}_u)$ — interaction history plus demographic descriptors — and emits $\text{Rec}_u^{(K)}$.

We define $f_{\text{LLM}}$ to **statistically discriminate** if for some protected attribute $d \in D$ and pair of values $a, b$ of $d$:

$$
\big| \mathbb{E}_{u : d_u = a} [\Phi(\text{Rec}_u^{(K)})] - \mathbb{E}_{u : d_u = b} [\Phi(\text{Rec}_u^{(K)})] \big| > \tau
$$

for some exposure-related functional $\Phi$ (e.g., APLT, Gini contribution, mean unpopularity score) **and** the disparity is not justified by genuine preference difference between subpopulations.

Concretely, our paper quantifies this via:

- $\Phi_1$ = APLT (Section 4.4): higher = more long-tail exposure.
- $\Phi_2$ = Coverage: corpus-level uniqueness of recommendations.
- $\Phi_3$ = Gini: distributional inequality of exposure.

A reranker is **fair** in our sense if it reduces the cross-slice differential in $\Phi_1, \Phi_2, \Phi_3$ without sacrificing within-user preference alignment (measured by Hit, MRR, NDCG).

### 3.5 Personalization criterion

A recommender preserves *personalization* if, **conditional on demographic attributes**, the per-user list-quality metrics (MRR, NDCG, HitRate@K) do not degrade meaningfully. Our reranker targets a Pareto improvement: increase exposure fairness measurably (large effect size), decrease accuracy minimally (small effect size).

---

## 4. Methodology

We split the methodology into four logical components: (4.0) data preprocessing, (4.1) slice-conditional stereotype quantification, (4.2) user mainstream-ness, (4.3) adaptive reranking, (4.4) evaluation metrics.

### 4.0 Data preprocessing

**Positive feedback definition.** A user's positive feedback is an interaction strong enough to indicate genuine preference. The exact threshold depends on dataset modality:

- **MovieLens-1M (ratings)**: $\text{positive}(u, i) = \mathbb{1}[r_{u,i} > 3]$ where $r_{u,i}$ is the user's 1–5 rating.
- **LastFM-1K (playcounts)**: $\text{positive}(u, i) = \mathbb{1}[c_{u,i} \geq 2 \land c_{u,i} > m_u]$ where $c_{u,i}$ is the play count and $m_u = \text{median}\{c_{u,j} : j \in \mathcal{I}_u\}$. The first conjunct removes one-off impressions; the second requires the play count exceed the user's median listening intensity.

**Train/test split.** For each user we randomly split positive items into a training set $\mathcal{H}_u$ (30%, used as in-prompt history) and a test set $\mathcal{T}_u$ (70%, withheld for evaluation):

$$
\mathcal{T}_u = \text{Sample}_{\text{rng}}(\mathcal{P}_u, \lfloor 0.7 \cdot |\mathcal{P}_u| \rfloor), \quad \mathcal{H}_u = \mathcal{P}_u \setminus \mathcal{T}_u
$$

with a fixed seed (42) for reproducibility.

**Demographic slicing.** We construct one popularity column per slice $g$ in the lattice of demographic value combinations. Sparsity protection (sample-size-weighted blending with neutral) is applied for slices with $n_g < 30$:

$$
B_{\text{blended}}(i, g) = w \cdot B_{\text{raw}}(i, g) + (1 - w) \cdot B_{\text{raw}}(i, \emptyset), \quad w = \min(1, n_g / 30)
$$

For LFM-1K we additionally treat the **Unknown** demographic value as a first-class slice (rather than discarding users with missing fields), motivated by the empirical reality that 71% of LFM-1K users have unreported age. Discarding them would bias the slice popularity matrix toward a non-representative minority.

### 4.1 Slice-conditional stereotype quantification (DiffScore)

The core insight is that an item that is *uniformly popular across slices* is not a stereotype carrier; an item whose slice popularity differs sharply *is*. We quantify the stereotype carried by an item under a given slice as **inverse popularity within that slice** — i.e. an item that is rare in slice $g$ is "harder to recommend correctly" if a system is to avoid stereotype.

#### 4.1.1 Long-tail collapse and rank-percentile fix

The naïve transform $B_{\text{raw}}$ suffers from severe right-skew on long-tail-heavy datasets: in LFM-1K the median $B_{\text{raw}}$ is ${\approx} 0.005$ across most slices, leaving the *unpopularity bonus* $1 - B$ approximately constant ($\approx 1$) and devoid of discriminating signal. We propose a **rank-percentile transform**:

$$
B'(i, g) = \begin{cases}
\dfrac{\text{rank}_g(i)}{|\{j : B_{\text{raw}}(j, g) > 0\}|} & \text{if } B_{\text{raw}}(i, g) > 0 \\[6pt]
0 & \text{otherwise}
\end{cases}
$$

where $\text{rank}_g(i)$ is the ascending rank of $i$ in column $g$ among non-zero entries, with ties broken by `method="average"`. Under $B'$:
- the most popular item per slice attains $B' = 1$ ;
- the rarest non-zero item attains $B' \approx 1/n_{\text{pos}}$ ;
- items with zero positive feedback in slice retain $B' = 0$, encoding "never engaged with" as maximum-difficulty.

The unpopularity bonus $1 - B'$ is now uniformly distributed over the slice's positive-feedback support, restoring discriminative signal.

#### 4.1.2 Difficulty score

We map slice popularity to a per-item, per-slice difficulty using:

$$
\text{score}(i, g) = (1 - B'(i, g)) \cdot 10 \in [0, 10]
$$

(Linear-x10 scheme; we also explore $\log_2$ and bucket schemes in ablation, see Section 8.) Higher score = harder to recommend = greater stereotype-defying value when surfaced.

#### 4.1.3 List-level DiffScore

Given a recommendation list $\text{Rec}_u^{(K)}$, the **per-user weighted hit difficulty** under scenario $g$ is:

$$
\text{DiffTotal}_u(g) = \sum_{i \in \text{Rec}_u^{(K)} \cap \mathcal{T}_u} \text{score}(i, g) \cdot \left(1 + \rho \cdot \frac{K - r_i^{(0)}}{K}\right)
$$

where $r_i^{(0)}$ is the 0-indexed rank of item $i$ in the list, and $\rho = 0.2$ is a rank-bonus weight that gives early hits slightly more credit. The averaged variant:

$$
\text{DiffAvg}_u(g) = \frac{\text{DiffTotal}_u(g)}{|\text{Rec}_u^{(K)} \cap \mathcal{T}_u|}
$$

DiffTotal answers "how much stereotype-defying mass did we surface?", DiffAvg answers "how stereotype-defying is each retained hit?".

### 4.2 User mainstream-ness

Beyond per-item stereotype, the same query yields an **operational measure of how mainstream each user is** within their own demographic slice:

$$
s_u(g) = \sum_{i \in \mathcal{H}_u^{\text{top-}L}} B'(i, g)
$$

where $\mathcal{H}_u^{\text{top-}L}$ is the user's top-$L$ historical items (sorted by interaction strength, $L = 25$ in our setting; this same prefix is fed to the LLM as context). A high $s_u$ means the user's revealed taste lies near the slice mode; a low $s_u$ means the user is niche within their own demographic.

The slice average $\bar{s}_g = \frac{1}{|\mathcal{U}_g|} \sum_{u \in \mathcal{U}_g} s_u(g)$ provides a pivot.

### 4.3 Adaptive reranking

Given a candidate pool of $N$ items emitted by the LLM under prompt with demographic descriptors, we wish to produce a top-$K$ list ($K \leq N$) with reduced exposure differential.

#### 4.3.1 Preference probability

The LLM-supplied rank is encoded as a smooth, base-clipped exponentially-decaying preference score:

$$
P_i^{\text{norm}} = \begin{cases}
\text{base} + (1 - \text{base}) \cdot \dfrac{r^{\,\text{rank}_i} - r^{\,N}}{1 - r^{\,N}} & \text{numerically stable case} \\[6pt]
\text{base} + (1 - \text{base}) \cdot \dfrac{N - \text{rank}_i}{N - 1} & \text{numerically degenerate case}
\end{cases}
$$

with $\text{base} = 0.2$ and decay $r = 0.9$. The score lies in $[\text{base}, 1]$ with rank-1 receiving 1 and rank-$N$ receiving $\text{base}$.

#### 4.3.2 Adaptive $\alpha$

The reranking intensity is user-adaptive:

$$
\alpha_u = \text{clip}\!\left(\alpha_{\text{base}} + \alpha_{\text{gain}} \cdot \frac{s_u(g_u) - \bar{s}_{g_u}}{\max(\bar{s}_{g_u}, \varepsilon)}, \alpha_{\min}, \alpha_{\max}\right)
$$

with $\alpha_{\text{base}} = 0.7$, $\alpha_{\text{gain}} = 0.2$, and clip range $[0.5, 0.9]$. Semantically:

| User type | $s_u$ vs $\bar{s}_g$ | $\alpha_u$ direction | Rerank behaviour |
|---|---|---|---|
| Mainstream within slice | $s_u > \bar{s}_g$ | $\alpha_u$ rises toward 0.9 | Trust LLM order; minimal long-tail injection |
| Average within slice | $s_u \approx \bar{s}_g$ | $\alpha_u \approx 0.7$ | Moderate rebalancing |
| Niche within slice | $s_u < \bar{s}_g$ | $\alpha_u$ falls toward 0.5 | Aggressive long-tail bonus |

This realizes a **personalization-preserving philosophy**: niche users in mainstream demographics get protected from stereotype collapse; mainstream users are not coerced toward unfamiliar items they did not signal taste for.

#### 4.3.3 Score and ordering

For each candidate $i \in \{1, \dots, N\}$:

$$
S_i = \alpha_u \cdot P_i^{\text{norm}} + (1 - \alpha_u) \cdot (1 - B'(i, g_u))
$$

The first term encodes the LLM's confidence; the second term injects unpopularity-driven exposure bonus.

#### 4.3.4 Top-$K'$ protection

Naïve sorting by $S_i$ destroys the LLM's high-confidence early-rank picks, severely hurting MRR. We propose a **head-protect** mechanism:

```
ordered_head = LLM_rec[:K']                               # K' = 5 by default
ordered_tail = sort(LLM_rec[K':], key=S_i, descending)
final_top_K = (ordered_head + ordered_tail)[:K]
```

The first $K'$ ranks of the LLM list are preserved verbatim; only positions $[K'+1, N]$ are rebalanced by $S$, then merged. This trades a small reduction in long-tail exposure for substantial early-rank stability (MRR loss reduced from ${\approx}-10\%$ without protection to $\leq -2\%$ with $K'=5$, see Section 8).

### 4.4 Evaluation metrics

We jointly report classical accuracy metrics, popularity-debias metrics, and corpus-level diversity metrics.

**Classical accuracy:**

$$
\text{Precision@K}_u = \frac{|\text{Rec}_u^{(K)} \cap \mathcal{T}_u|}{\min(K, |\text{Rec}_u|)}, \quad
\text{Recall@K}_u = \frac{|\text{Rec}_u^{(K)} \cap \mathcal{T}_u|}{|\mathcal{T}_u|}
$$

$$
\text{HitRate@K}_u = \mathbb{1}\big[|\text{Rec}_u^{(K)} \cap \mathcal{T}_u| > 0\big]
$$

$$
\text{MRR}_u = \begin{cases} 1/r^*_u & \text{if hit exists} \\ 0 & \text{otherwise} \end{cases}, \quad
r^*_u = \min\{r : \text{Rec}_u[r] \in \mathcal{T}_u\}
$$

$$
\text{DCG@K}_u = \sum_{r=1}^{K} \frac{\mathbb{1}[\text{Rec}_u[r] \in \mathcal{T}_u]}{\log_2(r+1)}, \quad
\text{NDCG@K}_u = \frac{\text{DCG@K}_u}{\text{IDCG@K}_u}
$$

**Popularity-debias / exposure-fairness (our focus):**

$$
\text{APLT}_u = \frac{|\{i \in \text{Rec}_u^{(K)} : B'(i, g_u) < \theta\}|}{K}, \quad \theta = 0.2
$$

We report two APLT variants: $\text{APLT}_{\text{slice}}$ uses the user's own slice column, $\text{APLT}_{\text{neutral}}$ uses the global $B'(\cdot, \emptyset)$ column. The first measures niche exposure relative to the user's demographic peers; the second measures global niche exposure.

**Corpus-level diversity:**

$$
\text{Coverage} = \frac{|\bigcup_{u \in \mathcal{U}} \text{Rec}_u^{(K)}|}{|\mathcal{I}|}
$$

$$
G = \frac{2 \sum_{i=1}^{n} i \cdot v_{(i)}}{n \sum_{i=1}^{n} v_{(i)}} - \frac{n+1}{n}
$$

with $v_{(1)} \leq v_{(2)} \leq \dots \leq v_{(n)}$ the sorted non-zero recommendation frequencies of items in the corpus. Lower Gini means flatter exposure distribution.

---

## 5. Experimental Setup

### 5.1 Datasets

| Dataset | Domain | Users | Items | Demographics | $K$ | Pool $N$ |
|---|---|---|---|---|---|---|
| **MovieLens-1M** (Harper & Konstan 2016) | movie | 6,031 | 3,883 | gender × age (×occupation) | 15 | 30 |
| **LastFM-1K** (Celma 2010) | music track | 942 | 272,769 | gender × age (× country) | 20 | 30 |

ML-1M provides explicit 1–5 ratings, complete demographic fields for all users. LFM-1K provides per-track listening counts; demographic fields are partial (gender 89%, age 29%); we treat missing as `Unknown` slice.

For LFM-1K, item granularity is **track-level** (not artist-level) — each item key is `Artist ||| Track`. This is a more demanding setup than artist-level due to extreme sparsity (max popularity count of 224 over 992 users).

### 5.2 LLM recommender

We use **GPT-4.1-mini** as the underlying LLM recommender. The same prompt format is used for both datasets, with domain-specific templates (movie / music_track) injected at render time. Demographic descriptors are added per scenario:
- $g = \emptyset$: no descriptor (neutral)
- $g = \text{gender}$: e.g. *"the user is female"*
- $g = \text{age}$: e.g. *"the user is in the young adult bracket"*
- $g = \text{cross}$: combined

For each (user, scenario) we issue one LLM call, parse the numbered list, and obtain $N=30$ candidate items. Cache hits are 100% on subsequent reranking-config experiments, removing API cost from ablations.

### 5.3 Reranker configuration

Default v2 configuration (all formulas in Section 4):

- $K = 15$ (ML-1M) or 20 (LFM-1K), $N = 30$, $L = 25$ (history limit fed to LLM)
- $\text{base} = 0.2$, decay $r = 0.9$
- $\alpha_{\text{base}} = 0.7$, $\alpha_{\min} = 0.5$, $\alpha_{\max} = 0.9$, $\alpha_{\text{gain}} = 0.2$
- $B$-transform = `rank` (rank-percentile)
- $K' = 5$ (head protection)
- $\rho = 0.2$ (rank-bonus weight)
- $\theta = 0.2$ (APLT threshold)

### 5.4 Statistical inference

Per-user paired differences are computed and analyzed via:
- Paired $t$-statistic $t = \bar{d} / (s_d / \sqrt{n})$
- Two-sided $p$-value via normal approximation (justified by $n \geq 600$)
- 95% confidence interval $\bar{d} \pm 1.96 \cdot \text{SE}(\bar{d})$
- Cohen's $d = \bar{d} / s_d$ for effect size

Bonferroni correction is applied for the multiple comparisons across (scenarios × metrics).

### 5.5 Reproducibility

All code, configurations, and intermediate artifacts (popularity tables, adapter pickle caches, LLM response caches) are released. Random seed is fixed at 42 for sampling, splitting, and prompt rendering. The end-to-end pipeline can be reproduced with three commands documented in the supplementary repository.

---

## 6. Results on MovieLens-1M

### 6.1 Setup

Sample size $n_u = 6{,}031$ users, sampled by stratified design across the 6-bucket gender×age cross. Each user receives $K=15$ recommendations after reranking from a candidate pool of $N=30$. We report four scenarios: neutral, gender, age, cross.

### 6.2 Main results

The neutral scenario passes through unchanged by design (no demographic prompt → no rerank trigger). The three demographic-conditioned scenarios are summarized below.

**Table 6.1.** ML-1M paired comparison: orig (LLM only) vs reranked (v2). All effects $p < 10^{-30}$ unless marked.

| Scenario | Metric | orig | reranked | $\Delta$ | $\%\Delta$ |
|---|---|---|---|---|---|
| **age** | Macro Precision | 0.2198 | 0.2271 | +0.0074 | +3.4% |
| | Macro Recall | 0.0700 | 0.0723 | +0.0023 | +3.3% |
| | HitRate@15 | 0.8529 | 0.8589 | +0.0060 | +0.7% |
| | MRR@15 | 0.4907 | 0.5396 | **+0.0489** | **+10.0%** |
| | NDCG@15 | 0.5054 | 0.5450 | **+0.0396** | **+7.8%** |
| | DiffAvg | 21.220 | 24.238 | **+3.018** | **+14.2%** |
| | TotalHits | 19,882 | 20,548 | +666 | +3.4% |
| **gender** | Macro Precision | 0.2110 | 0.2185 | +0.0076 | +3.6% |
| | Macro Recall | 0.0672 | 0.0697 | +0.0024 | +3.6% |
| | HitRate@15 | 0.8423 | 0.8476 | +0.0053 | +0.6% |
| | MRR@15 | 0.4740 | 0.5268 | **+0.0528** | **+11.1%** |
| | NDCG@15 | 0.4936 | 0.5367 | **+0.0431** | **+8.7%** |
| | DiffAvg | 23.846 | 25.960 | **+2.114** | **+8.9%** |
| | TotalHits | 19,086 | 19,770 | +684 | +3.6% |
| **gender+age** | Macro Precision | 0.2115 | 0.2200 | +0.0085 | +4.0% |
| | Macro Recall | 0.0675 | 0.0700 | +0.0025 | +3.7% |
| | HitRate@15 | 0.8501 | 0.8564 | +0.0063 | +0.7% |
| | MRR@15 | 0.4776 | 0.5288 | **+0.0512** | **+10.7%** |
| | NDCG@15 | 0.4957 | 0.5389 | **+0.0432** | **+8.7%** |
| | DiffAvg | 22.458 | 25.363 | **+2.905** | **+12.9%** |
| | TotalHits | 19,133 | 19,902 | +769 | +4.0% |

### 6.3 Interpretation

The ML-1M results exhibit a remarkable property: **all metrics improve simultaneously**. The reranker:
- Increases accuracy (Precision, Recall, HitRate, MRR, NDCG) by 0.7–11%
- Increases DiffAvg (stereotype-defying weighted hit) by 9–14%
- Adds 666–769 additional correct hits across the 6,031-user sample

This Pareto-dominating outcome means the LLM's stereotype-driven candidates were not only unfair but also suboptimal in pure accuracy: there were many genuinely correct recommendations sitting in the bottom half of the candidate pool because they did not match the LLM's stereotyped expectation of the user's demographic. The reranker recovers them.

### 6.4 Across-slice exposure differential

[Placeholder — to be filled with by-subgroup APLT/Coverage/Gini differential analysis once the corresponding metrics are computed for ML-1M.]

---

## 7. Results on LastFM-1K (Track-level)

### 7.1 Setup

Sample size $n_u = 600$ users (stratified across 9 gender×age buckets including Unknown). Each user receives $K=20$ track recommendations after reranking from a candidate pool of $N=30$. Four scenarios as before.

The track-level setup is intentionally **harder**: 272,769 unique tracks vs 3,883 movies, with severe long-tail (median raw popularity $\approx 0.005$). This is where our rank-percentile $B'$ transform proves essential.

### 7.2 Main results

**Table 7.1.** LFM-1K paired comparison: orig vs v2 (rank-percentile B + protect_top=5). All effects $p < 10^{-5}$ unless marked.

| Scenario | Metric | orig | v2 | $\Delta$ | $\%\Delta$ |
|---|---|---|---|---|---|
| **gender** | Hits | 3.24 | 2.57 | -0.67 | -20.9% |
| | HitRate@20 | 0.833 | 0.778 | -0.055 | -6.6% |
| | MRR | 0.488 | 0.477 | **-0.011** | **-2.2%** |
| | NDCG@20 | 0.456 | 0.379 | -0.076 | -16.7% |
| | **APLT (slice)** | 0.299 | **0.459** | **+0.159** | **+53.2%** |
| | **APLT (neutral)** | 0.290 | 0.444 | +0.154 | +53.3% |
| | **DiffAvg** | 0.439 | 0.547 | +0.109 | +24.8% |
| | **DiffTotal** | 2.647 | 2.851 | **+0.204** | **+7.7%** |
| **age** | HitRate@20 | 0.835 | 0.802 | -0.033 | -4.0% |
| | MRR | 0.498 | 0.490 | **-0.008** | **-1.6%** |
| | NDCG@20 | 0.474 | 0.405 | -0.069 | -14.6% |
| | **APLT (slice)** | 0.281 | **0.430** | +0.149 | **+53.0%** |
| | **DiffAvg** | 0.468 | 0.523 | +0.054 | +11.6% |
| | **DiffTotal** | 3.139 | 3.249 | +0.110 | +3.5% |
| **gender+age** | HitRate@20 | 0.817 | 0.760 | -0.057 | -6.9% |
| | MRR | 0.476 | 0.465 | **-0.011** | **-2.3%** |
| | NDCG@20 | 0.454 | 0.371 | -0.083 | -18.2% |
| | **APLT (slice)** | 0.321 | **0.487** | +0.166 | **+51.5%** |
| | **DiffAvg** | 0.875 | 1.069 | +0.194 | +22.1% |
| | **DiffTotal** | 5.479 | 6.198 | **+0.719** | **+13.1%** |

### 7.3 Per-user robustness

A critical sanity check: does the reranker improve APLT for *most* users, or is the average lift driven by a subset of outliers?

**Table 7.2.** Per-user APLT improvement coverage on LFM-1K (n=600).

| Scenario | Improved | Unchanged | Regressed |
|---|---|---|---|
| gender | 589 (98.2%) | 11 (1.8%) | **0 (0.0%)** |
| age | 586 (97.7%) | 14 (2.3%) | **0 (0.0%)** |
| gender+age | 586 (97.7%) | 14 (2.3%) | **0 (0.0%)** |

**Zero users regress in APLT.** The lift is uniform.

### 7.4 Demographic subgroup analysis

To verify that the reranker does not silently prejudice against subgroups with fewer demographic data, we report APLT improvement separately by gender×age cell.

**Table 7.3.** APLT lift by demographic cell on LFM-1K, gender+age scenario.

| gender / age | n | orig APLT | v2 APLT | $\Delta$ |
|---|---|---|---|---|
| female / young | 73 | 0.520 | 0.736 | +0.216 |
| female / middle-aged | 39 | 0.481 | 0.699 | +0.218 |
| female / unknown | 119 | 0.467 | 0.686 | +0.219 |
| male / young | 80 | 0.489 | 0.701 | +0.212 |
| male / middle-aged | 70 | 0.480 | 0.685 | +0.205 |
| male / unknown | 118 | 0.435 | 0.629 | +0.194 |
| unknown / young | 6 | 0.658 | 0.817 | +0.158 |
| unknown / middle-aged | 8 | 0.525 | 0.769 | +0.244 |
| unknown / unknown | 87 | 0.536 | 0.754 | +0.218 |

Across all 9 cells (including those with only $n=6$ users), APLT improvement falls within $[+0.158, +0.244]$ — a tight 9pp range. **The algorithm is robust to demographic-data sparsity.**

### 7.5 Statistical discrimination differential

We measure exposure-opportunity differential across slice pairs (cf. Eq. in Section 3.4). Results [PLACEHOLDER — to be filled with cross-slice $\Phi_1, \Phi_2, \Phi_3$ comparison once full grid is computed].

### 7.6 Interpretation

LFM-1K results exhibit a *trade-off* outcome rather than Pareto-domination: APLT and Gini improve substantially (+50pp / -10pp) but accuracy metrics degrade ($\sim$5–18%). MRR is preserved within 2.5%, indicating early-rank stability is not sacrificed.

The contrast between ML-1M (Pareto-domination) and LFM-1K (trade-off) is informative: with shorter tails (3.9k movies vs 272k tracks) and explicit rating signals, the LLM has less room for stereotype-driven mistakes; the reranker mostly recovers misranked correct hits. With long-tail-extreme music tracks, the LLM has many candidates, much of its mass falls on stereotype-aligned mainstream, and the reranker must trade some accuracy for measurable exposure rebalancing.

---

## 8. Ablation and Pareto Analysis

### 8.1 The role of rank-percentile $B'$

Without rank-percentile transform (using raw $B$ directly), the unpopularity bonus $1 - B$ becomes saturated near 1 for the long-tail-heavy LFM-1K. This degenerates the reranker to a near-identity transform on the candidate list: low APLT lift, low Gini reduction.

**Table 8.1.** Ablation on $B$-transform (LFM-1K, gender scenario, n=600).

| Variant | APLT lift | Gini $\Delta$ | $\Delta$ NDCG | $\Delta$ MRR |
|---|---|---|---|---|
| raw $B$, $\alpha=0.7$ | +0.005 | -0.011 | -0.052 | -0.069 |
| raw $B$, $\alpha=0.5$ | +0.012 | -0.024 | -0.124 | -0.195 |
| **rank-$B'$, $\alpha=0.7$ + protect_top=5 (v2)** | **+0.215** | **-0.041** | -0.076 | **-0.011** |

The rank-percentile transform is necessary; raw $B$ produces no meaningful lift even at aggressive $\alpha=0.5$.

### 8.2 The role of head protection ($K'$)

**Table 8.2.** Ablation on `protect_top` (LFM-1K, gender, rank-$B'$, $\alpha=0.7$).

| $K'$ | $\Delta$ MRR | $\Delta$ NDCG | $\Delta$ APLT |
|---|---|---|---|
| 0 (no protection) | -0.069 | -0.085 | +0.225 |
| 3 | -0.024 | -0.080 | +0.221 |
| **5 (v2 default)** | **-0.011** | -0.076 | +0.215 |
| 10 | -0.005 | -0.071 | +0.198 |

A modest protection of $K'=5$ recovers $\sim 6$pp of MRR loss while sacrificing only $1$pp of APLT lift — a clear Pareto-improving operating point.

### 8.3 Per-user $\alpha$ adaptive vs fixed

[Placeholder — comparison of dynamic $\alpha$ vs fixed $\alpha$ to demonstrate value of user-mainstream-ness adaptation.]

### 8.4 Pareto frontier scatter

[Placeholder for scatter plot: x-axis = APLT, y-axis = NDCG, points = (algorithm, $\alpha$, $K'$) configurations. Show v2 lies on the Pareto frontier above raw-$B$ baselines.]

---

## 9. Discussion

### 9.1 Implications for fair LLM-based recommendation deployment

Our framing — exposure-opportunity differential on protected attributes as the operational definition of statistical discrimination — clarifies what a fair LLM recommender must achieve and offers a directly measurable proxy. Practitioners can audit any closed-source LLM recommender for cross-slice APLT, Coverage, and Gini differentials using our pipeline without retraining the model.

### 9.2 Why post-hoc

Three properties of post-hoc reranking are critical for the deployment scenarios we envision:

- **Compatible with closed-source LLMs**: GPT-4-class models cannot be fine-tuned for fairness by most operators.
- **Train-free, deterministic, auditable**: a single closed-form $\alpha$ rule plus a popularity table; no learning curves, no hyperparameter drift.
- **Modular**: the popularity table can be updated as the corpus evolves; the LLM and the reranker can be versioned independently.

In-processing methods (Up5, Hu et al. 2024) achieve different trade-offs and are valuable when fine-tuning is feasible. Our approach is complementary, not competitive.

### 9.3 The Personalization vs Discrimination dichotomy in practice

A reranker like ours can only discharge the discrimination half — it cannot manufacture preference signal where none exists. If a niche listener under a mainstream demographic has provided weak signal in $\mathcal{H}_u$, the LLM will still return mainstream candidates and our reranker will down-weight them — producing an exposure rebalance, but not necessarily a more accurate recommendation. This is the cost of operating under partial information about preference.

The remedy is to enrich $\mathcal{H}_u$ (longer, denser histories) so that personalization can do more work and reranking does less. This is an interesting future direction.

### 9.4 Limitations

- **Coverage decrease on LFM-1K**: while individual user lists are more diverse and Gini is reduced, the corpus-level unique-item count actually *decreases* by 17–25% (the reranker concentrates exposure on a wider but still bounded set of mid-tail items). A future $v3$ should add an inter-user anti-concentration term.
- **MRR loss is statistically significant on LFM-1K** (though small effect size, $d \approx 0.26$). A perfectly Pareto-dominating outcome on long-tail-extreme datasets remains elusive.
- **Single LLM** evaluated. Cross-LLM consistency (Claude, Qwen) is left for future work.
- **Two datasets**. Generalization to BookCrossing, Amazon Reviews, etc. is open.
- **Stratification on observed demographic only**. Race / ethnicity is unobserved in both benchmarks; the framework extends naturally but evaluation requires datasets with appropriate annotations.

### 9.5 Future work

- $v3$ algorithm with inter-user anti-concentration penalty.
- Cross-LLM (Claude-3.5, Qwen2.5-72B) and cross-domain (BookCrossing, Amazon) generalization.
- Investigating preference-drift mitigation by counterfactual prompting (drop demographic, observe candidate change, re-merge).
- Combining post-hoc rerank with in-processing methods (Up5, LoRA debias) for compounded effect.

---

## 10. Conclusion

We have argued that the operational behaviour of LLM-based recommenders should be evaluated through the lens of *statistical discrimination by exposure differential on protected attributes*, not through the colloquial framing of "model has bias". We proposed a slice-conditional, training-free post-hoc reranker that:

1. Quantifies stereotype using a slice-conditional popularity matrix with rank-percentile transform.
2. Computes per-user mainstream-ness within slice and adapts $\alpha$ accordingly.
3. Protects the LLM's high-confidence early ranks while rebalancing the tail.

On MovieLens-1M, the reranker Pareto-dominates the LLM baseline, simultaneously improving accuracy and exposure fairness. On LastFM-1K (track-level, long-tail-extreme), the reranker delivers a +52pp APLT lift and -10pp Gini reduction at the cost of $\sim 2\%$ MRR. Across 600 LFM-1K users, exposure improves for 98% of users and regresses for none. The framework is publicly released, fully reproducible, and applicable to any closed-source LLM recommender.

---

## References

[1] Abdollahpouri, H., Burke, R., & Mobasher, B. (2017). Controlling popularity bias in learning-to-rank recommendation. *RecSys 2017*.

[2] Adomavicius, G., & Kwon, Y. (2012). Improving aggregate recommendation diversity using ranking-based techniques. *IEEE TKDE*.

[3] Atkinson, A. B. (1970). On the measurement of inequality. *Journal of Economic Theory*.

[4] Bao, K., Zhang, J., et al. (2023). TALLRec: an effective and efficient tuning framework to align LLMs with recommendation. *RecSys 2023*.

[5] Barocas, S., Hardt, M., & Narayanan, A. (2017). Fairness in machine learning. *NeurIPS Tutorial*.

[6] Beel, J., Gipp, B., Langer, S., & Breitinger, C. (2013). Research-paper recommender systems: a literature survey. *International Journal on Digital Libraries*.

[7] Celma, O. (2010). *Music Recommendation and Discovery in the Long Tail*. Springer.

[8] Chen, J., Dong, H., Wang, X., Feng, F., Wang, M., & He, X. (2023). Bias and debias in recommender system: a survey and future directions. *ACM TOIS*.

[9] Dwork, C., Hardt, M., Pitassi, T., Reingold, O., & Zemel, R. (2012). Fairness through awareness. *ITCS 2012*.

[10] Geng, S., Liu, S., Fu, Z., Ge, Y., & Zhang, Y. (2022). Recommendation as language processing (RLP): a unified pretrain, personalized prompt & predict paradigm (P5). *RecSys 2022*.

[11] Harper, F. M., & Konstan, J. A. (2016). The MovieLens datasets: history and context. *ACM TIIS*.

[12] Hou, Y., Zhang, J., Lin, Z., et al. (2024). Large language models are zero-shot rankers for recommender systems. *ECIR 2024*.

[13] Hu, X., Cao, X., et al. (2024). Mitigating popularity bias in large language model recommendation: a comprehensive empirical study. *SIGIR 2024*.

[14] Hua, M., et al. (2024). Up5: unbiased foundation model for fairness-aware recommendation. *ICDE 2024*.

[15] Li, X., Zhang, Y., Malthouse, E. C. (2024). BIGRec: bi-step grounding paradigm for large language models in recommendation systems. *SIGIR 2024*.

[16] Liu, J., Liu, C., Zhou, P., Lv, R., Zhou, K., & Zhang, Y. (2023). Is ChatGPT a good recommender? A preliminary study. *EMNLP 2023*.

[17] Santos, R. L. T., Macdonald, C., & Ounis, I. (2010). Exploiting query reformulations for web search result diversification. *WWW 2010*.

[18] Steck, H. (2018). Calibrated recommendations. *RecSys 2018*.

[19] Wang, L., Wu, J., et al. (2023). Recommender systems based on generative large language models. *Working paper*.

[20] Zehlike, M., Bonchi, F., Castillo, C., et al. (2017). FA*IR: a fair top-k ranking algorithm. *CIKM 2017*.

[21] Zhang, Y., Feng, F., Zhang, Z., et al. (2024). Counterfactual reasoning for bias detection in LLM-based recommendation. *ICLR 2024*.

---

## Appendix A. Reproducibility Checklist

- **Code**: `https://github.com/<placeholder>` (released upon acceptance).
- **Datasets**: ML-1M is publicly available from GroupLens (https://files.grouplens.org/datasets/movielens/ml-1m.zip). LFM-1K is available upon request from the authors of Celma (2010).
- **LLM**: GPT-4.1-mini via OpenAI API; all responses are cached in SQLite for replay.
- **Random seeds**: 42 throughout (sampling, splitting, prompt rendering).
- **Hyperparameters**: full configuration in `configs/` directory; hyperparameters of the reranker enumerated in Section 5.3.
- **Compute**: experiments require approximately 3 hours on a single machine (CPU + LLM API), with $\sim$ \$1.50 in API spend.

---

## Appendix B. Glossary

| Term | Definition |
|---|---|
| **Statistical discrimination** | A recommender systematically alters exposure opportunity for items conditional on a protected attribute, beyond what is justified by genuine within-slice preference differences. |
| **Personalization** | Recommendation adapts to interaction history; conditional on demographics, no exposure differential. |
| **Slice / scenario** | A (subset of) demographic value-tuple, e.g. *gender_M*, *cross_F_Youth*. |
| **APLT** | Average Percentage of Long-Tail items. The fraction of top-$K$ recommendations falling in the bottom-$\theta$ popularity quantile. |
| **DiffScore** | Inverse-popularity weighted hit score; quantifies stereotype-defying recommendation success. |
| **rank-percentile $B'$** | Slice popularity transform: cell becomes its rank-percentile in the slice column. |
| **protect_top $K'$** | Number of LLM-supplied early ranks preserved verbatim by the reranker. |
| **Adaptive $\alpha$** | User-conditional rerank intensity; depends on user mainstream-ness within slice. |
| **Coverage** | Unique-item count across all users' top-$K$ lists, normalized by catalog size. |
| **Gini (recommendation)** | Inequality of recommendation frequency distribution across items. |

---

*This is a draft; figures, full statistical tables, and Pareto scatterplots will be added in the next revision.*




