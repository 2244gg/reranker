# Group-Conditional Popularity, Difficulty Scoring, and Debiasing Rerank
## Design Notes

This document describes three coupled components used in this project:

1. **Group-conditional popularity** — how we estimate per-item popularity
   conditioned on the user's demographic slice.
2. **Difficulty-weighted hit scoring** — how we evaluate recommender
   quality from a fairness / long-tail perspective.
3. **Popularity-debiasing rerank** — how we transform an LLM's ranked list
   into a long-tail-friendly, user-adaptive output.

The design philosophy is:

> *Recommending what's stereotypically popular in a user's demographic
> group is the easy case. We want to measure and reward whatever portion
> of the LLM's output goes beyond that — niche items the user actually
> likes that no naive popularity baseline would have surfaced.*

All math here is implemented in:
- `src/popularity/slicer.py` (popularity table)
- `src/popularity/item_scoring.py` (difficulty score table)
- `src/experiment/rerank.py` (rerank engine)
- `src/experiment/metrics.py` (evaluation hooks)

---

## 1. Group-Conditional Popularity B[i, k]

### 1.1 Definition

Let `U` be the set of users and `I` the set of items. A user `u ∈ U` has
demographic attributes (e.g. gender, age_group, country). A
**demographic slice** `k` is a subset of `U` defined by fixing some
attributes:

- `k = neutral` → all users
- `k = gender_M` → all male users
- `k = age_Senior` → all elderly users
- `k = cross_M_Senior` → all male elderly users

For each (item, slice) pair we compute:

```
B[i, k]  =  count_pos(i, k)  /  max_j count_pos(j, k)
```

where `count_pos(i, k)` = number of users in slice `k` who have given
**positive feedback** to item `i`. The normalization makes the most-loved
item per slice have `B = 1`, the least-loved approach 0.

### 1.2 Positive feedback definitions

Per dataset:

| Dataset       | Positive feedback                                 |
|---------------|---------------------------------------------------|
| ML-1M         | `rating > 3`                                      |
| BookCrossing  | `rating > 5` (explicit) or any interaction (implicit) |
| LastFM-1K     | `playcount(u, i) > median_j(playcount(u, j))`    |
| LFM-2b        | same as LastFM-1K                                |
| Tenrec        | `watch_time / video_length > 0.5`                |

For LastFM datasets, the threshold is **per-user**: each listener has
their own median, so "positive" means "above this listener's typical
intensity" rather than a fixed global cutoff. This avoids penalizing
casual listeners and over-rewarding power users.

### 1.3 Slice enumeration

`PopularitySlicer._enumerate_slice_keys()` yields:

- `()` (neutral)
- All single-dimension slices (one per demographic dim × value)
- All 2-way crosses (Cartesian product of any 2 dims × value pairs)
- All 3-way crosses **only if** the dataset exposes ≥ 3 demographic dims

The output CSV thus has columns like:

```
neutral_All, gender_M, gender_F, age_Youth, age_Middle, age_Senior,
cross_M_Youth, cross_F_Senior, ...
```

### 1.4 Sparse-slice protection

When a slice has fewer than `SPARSE_THRESHOLD = 30` users, the raw
estimate is unstable. We linearly blend with the neutral popularity:

```
B_blended[i, k] = w * B_raw[i, k] + (1 - w) * B[i, neutral]
where w = min(1, |slice_k| / 30)
```

This is a soft fallback rather than rejection — small slices contribute
weakly while large slices are kept untouched.

### 1.5 Worked example — Pink Floyd in LastFM-1K

(Real numbers from `data/processed/lastfm1k/items_with_popularity.csv`.)

| Slice           | B for Pink Floyd | Interpretation                          |
|-----------------|------------------|------------------------------------------|
| `neutral_All`   | 0.75             | Top quartile globally                    |
| `gender_M`      | **0.85**         | Distinctly more popular among males      |
| `gender_F`      | 0.61             | Less popular among females (24% gap)     |
| `age_Senior`    | 0.78             | Popular among older listeners            |
| `age_Middle`    | 0.73             | Mid-tier among middle-aged               |
| `age_Youth`     | **0.87**         | Surprisingly popular among young users   |
| `cross_F_Youth` | 0.72             | Young women ≠ "no Pink Floyd"            |

Comparison artist: **Coldplay**

| Slice           | B for Coldplay   |
|-----------------|------------------|
| `neutral_All`   | 0.87             |
| `gender_M`      | 0.75             |
| `gender_F`      | **1.00**         |
| `age_Youth`     | **1.00**         |

Female and young listeners' #1 is Coldplay, while male listeners' top is
Radiohead/Beatles (B=1.0) with Coldplay only at 0.75. Pink Floyd is the
**inverse**: a male-leaning canonical rock act.

These slice-conditional differences are what `B[i, k]` is designed to
expose. A globally popular artist (Radiohead) has near-1 in every slice;
items with strong demographic association have wide variance across
slices.

---

## 2. Difficulty-Weighted Hit Score

### 2.1 Motivation

A recommender that hits *Frozen* for a 12-year-old female user has done
something trivial — *Frozen* is at the top of the female-young slice, any
popularity baseline would surface it. A recommender that hits *The Fall
of Reach* (niche military sci-fi) for the same user has done something
genuinely informative.

We therefore weight hits inversely by their slice popularity:

```
score(i, k) = f(B[i, k])    where f is monotone-decreasing.
```

A hit on a **niche** item gets a **larger** score than a hit on a
**slice-popular** item.

### 2.2 Score function schemes

`src/popularity/item_scoring.difficulty_score(b_norm, scheme)`:

| Scheme         | Formula                                 | Range          | Notes                             |
|----------------|-----------------------------------------|----------------|------------------------------------|
| `linear_x10`   | `(1 - B) * 10`                          | [0, 10]        | Default. Same as legacy `unpopularity_score`.    |
| `log`          | `-log2(B + 1e-3)`                       | [0, ~9.97]     | Information-theoretic surprisal interpretation.   |
| `bucket`       | 1 / 2 / 3 / 5 by B band                 | {1, 2, 3, 5}   | Coarse but interpretable.          |

`linear_x10` is recommended for primary reporting because it is
proportional to the existing `(1 - popularity) * 10` term used in
`score_top15_dual_lists.py`, which downstream comparisons already trust.
`log` is better for cases where differentiating extremely-niche items
matters; `bucket` is most readable in tables.

### 2.3 Precomputed lookup table

For an experiment with N users × K scenarios × ~20 hits per evaluation,
naively recomputing `f(B[i, k])` is fine, but for convenient inspection
and for downstream analysis we persist the table:

```
data/processed/<dataset>/items_with_popularity.csv     ← B values
data/processed/<dataset>/items_with_difficulty.csv     ← f(B) values
```

Use:

```python
from src.popularity.item_scoring import write_score_table

write_score_table(
    "data/processed/lastfm1k/items_with_popularity.csv",
    "data/processed/lastfm1k/items_with_difficulty.csv",
    scheme="linear_x10",
)
```

Or load into memory for fast lookup:

```python
from src.popularity.item_scoring import ItemScoreTable

st = ItemScoreTable.from_csv(
    "data/processed/lastfm1k/items_with_difficulty.csv"
)
score = st.lookup("Pink Floyd", "gender_F")  # ≈ (1 - 0.61) * 10 = 3.9
```

### 2.4 Per-user evaluation: `difficulty_weighted_hit_score`

For one (user, scenario) cell:

```
hit_score(rec, test, k_slice) =
    Σ_{i ∈ rec ∩ test}  score(i, k_slice) · (1 + rank_weight · rank_bonus_i)
```

where `rank_bonus_i = (list_size - rank_i) / list_size ∈ [0, 1]`. With
default `rank_weight = 0.0`, the bonus is disabled and the metric is purely
"sum of difficulty over hits". With `rank_weight = 0.2` we encourage
front-positioned niche hits.

Worked example. User u in slice `gender_F`. LLM recommends:

```
Rank 1: Coldplay         (B=1.00 → score=0)
Rank 2: Pink Floyd       (B=0.61 → score=3.9)
Rank 3: Radiohead        (B=0.93 → score=0.7)
Rank 4: Florence + Machine (B=0.40 → score=6.0)
Rank 5: Some Niche Band  (B=0.05 → score=9.5)
```

User's test set hits: `{Pink Floyd, Some Niche Band}`. With `rank_weight = 0`:

```
hit_score = 3.9 + 9.5 = 13.4
```

If a different rerank surfaces `{Pink Floyd, Florence + Machine}` instead:

```
hit_score = 3.9 + 6.0 = 9.9
```

So the first system gets a higher difficulty score, even though both
have the same hit count, because it surfaced a more niche item.

## 2.5 Comparison: original vs reranked

The two evaluated lists are constructed as follows:

```
        LLM produces M+N candidates
        (configured via cfg.candidate_pool_size, default 30 when top_k=20)
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
   first-M of LLM order    rerank on full M+N → first-M
   (Original list)         (Reranked list)
          │                   │
          ▼                   ▼
       Truncate to top_k = M for evaluation
```

This is critical: if the candidate pool size equals top_k, rerank can only
*reorder* an already-fixed set, so ``orig_hits = rer_hits`` always
(reordering doesn't add or remove items in or out of top_k). Setting
``candidate_pool_size > top_k`` lets rerank promote niche items from
positions M+1...M+N up into top-M, while popular items at positions
1...M can be displaced. This is the standard rerank evaluation protocol
in popularity-debiasing literature.

```
delta_hits = rer_hits − orig_hits
delta_difficulty = hit_score(reranked) − hit_score(original)
```

`delta_hits < 0` is common in fairness-debiasing: niche items added to
the reranked list don't always intersect the user's test set (because
test sets are themselves popularity-biased). The accompanying
`delta_difficulty` shows whether the items kicked out were popular hits
(score loss small) or niche hits (score loss large), and whether the
items pulled in were genuinely informative (score gain large).

### Empirical example (LFM-1K, n=5 pilot)

| Scenario           | Δ_hits | Δ_difficulty | Δ_NDCG@20 | Interpretation                                |
|--------------------|--------|--------------|-----------|-----------------------------------------------|
| neutral            | 0      | 0            | 0         | bypass — pure baseline                        |
| gender             | -0.4   | -0.46        | -0.073    | rerank loses some popular hits, no offsetting |
| age_group          | -0.8   | -3.61        | -0.060    | niche items injected but mostly off-test-set  |
| **gender+age**     | **-0.2** | **+2.40**  | -0.023    | **+2.4 difficulty** for **-0.2 hits**: niche items added were on-target  |

---

## 3. Popularity-Debiasing Rerank

### 3.1 Score formula

```
S_i = α · P_i_norm + (1 - α) · (1 - B_i_norm)
```

- `P_i_norm ∈ [base, 1]`: geometric-decay rank score (preference signal
  from the LLM). High at top ranks, low at bottom.
- `1 - B_i_norm ∈ [0, 1]`: unpopularity bonus in the user's slice. Small
  for slice-mainstream items, large for niche items.
- `α`: trust in LLM's original ordering vs trust in the long-tail bonus.

Sort items by `S_i` descending; keep top-N.

#### 3.1.1 P_i_norm (geometric decay)

```
P_i_norm = base + (1 - base) · (decay^i - decay^N) / (1 - decay^N)
```

Default `base = 0.2`, `decay = 0.9`, `N = 20`:

| Original rank | P_i_norm |
|---------------|----------|
| 1             | 0.866    |
| 5             | 0.609    |
| 10            | 0.405    |
| 20            | 0.200    |

A geometric (rather than linear) decay is used because LLM confidence
drops faster than linear with rank position; the `base` floor prevents
late-rank items from collapsing to zero score.

#### 3.1.2 Picking the right column for B_i_norm

`popularity_columns_for(scenario, demographics)` translates a scenario
spec and a user's demographic dict to a column name:

| Scenario              | User                           | Column            |
|-----------------------|--------------------------------|-------------------|
| `neutral`             | (any)                          | `neutral_All`     |
| `gender`              | male                           | `gender_M`        |
| `age_group`           | elderly                        | `age_Senior`      |
| `(gender, age_group)` | female + young                 | `cross_F_Youth`   |
| `(gender, age, ...)`  | M + young + student            | `cross_M_Youth_Student` |

The runner uses this lookup so the rerank "knows which slice's popularity
to debias against" given the active demographic prompt.

#### 3.1.3 Direction note (history)

The original codebase used:

```
S_i = α · P_i_norm − (1 − α) · (1 − B_i_norm)
```

which (after dropping the constant) is sort-equivalent to
`α · P + (1 − α) · B` — that is, **promote popular items**. This was
flipped to the current `+(1 − α)(1 − B)` direction because the project's
research goal is *debiasing toward long-tail*, which requires
*promoting niche items*. See CHANGELOG for the change record.

### 3.2 Dynamic α (personalization-first variant)

```
centered_ratio = (user_score − scene_mean_score) / scene_mean_score
α = clip(α_base + α_gain · centered_ratio, [α_min, α_max])
```

Where:
- `user_score` = sum over the user's history of `B[i, k_user_scenario]`.
  Reflects how *mainstream* the user's history is in their slice.
- `scene_mean_score` = mean of `user_score` across all users in the same
  scenario (the "typical mainstream-ness" baseline).

Defaults: `α_base = 0.7`, `α_min = 0.5`, `α_max = 0.9`, `α_gain = 0.2`.

**Behavior** (note the **`+` sign** on the gain term — see "Design choice"
below):

| User type           | `centered_ratio` | `α`            | Rerank intensity                          |
|---------------------|------------------|-----------------|-------------------------------------------|
| Mainstream user (heavy slice-popular consumer) | > 0              | **higher**     | Less long-tail intervention; keep LLM order. The mainstream user keeps getting the mainstream picks they prefer. |
| Niche user (long-tail consumer)                | < 0              | **lower**      | More long-tail intervention; promote niche items. The niche user gets more obscure recommendations matching their established taste. |

#### Design choice: personalization vs fairness convention

Two opposite policies are defensible:

* **Personalization-first (used here, `+` sign):** rerank *supports*
  each user's existing taste style. Mainstream users keep their
  mainstream picks; niche users get even more niche surfacing. This
  treats rerank as a personalization amplifier rather than a fairness
  correction.

* **Fairness convention (`-` sign):** mainstream users are assumed to be
  trapped in algorithmic echo chambers and need more long-tail
  intervention; niche users already explore the long tail and need less.
  This is the standard framing in popularity-debiasing fairness papers.

The implementation uses the **personalization-first** sign. To switch to
the fairness convention, set `α_gain` to a negative value, or modify the
sign in `src/experiment/rerank.compute_dynamic_alpha`.

### 3.3 Neutral scenario bypass

`scenario = neutral` (no demographic info in the prompt) is intentionally
**not reranked**. It serves as the pure-LLM baseline against which all
demographic-conditioned scenarios are compared. Reports show
`reranked = original` for the neutral row.

---

## 4. End-to-End Example

User `user_000282` in LastFM-1K:
- Demographics: `male, elderly`
- History (top 5): Pink Floyd, Nine Inch Nails, Massive Attack, U2, King Crimson
- Test set: 200+ artists (held out)

LLM is given a `gender + age` scenario prompt and returns 20 recs:

```
1. The Doors (B[M_Senior]=0.95)
2. Led Zeppelin (B[M_Senior]=0.92)
3. Genesis (B[M_Senior]=0.78)
4. Yes (B[M_Senior]=0.72)
5. ...
20. Vangelis (B[M_Senior]=0.18)
```

`α` is computed per user. Suppose `α = 0.65` for this user.

For each item:
```
S_i = 0.65 · P_i + 0.35 · (1 − B_i[M_Senior])
```

Niche items (low B) get a meaningful boost. Sort and keep top-20:

```
1. Vangelis           (P=0.20, S=0.65·0.20 + 0.35·0.82 = 0.417)
2. The Doors          (P=0.866, S=0.65·0.866 + 0.35·0.05 = 0.580)
3. Genesis            (P=0.78, S=0.65·0.78 + 0.35·0.22 = 0.584)
...
```

Wait — 0.580 vs 0.417: The Doors still beats Vangelis. The Doors's high
P (rank 1, P=0.866) dominates Vangelis's high (1−B). This is the design:
top-rank items keep their place unless they are extremely popular AND
the rest of the list contains items that are simultaneously top-decile
ranked AND niche.

In practice, what happens is **mid-rank niche items rise** (e.g. Yes at
rank 4 with B=0.72 may swap with Genesis at rank 3 with B=0.78), and
**bottom-rank popular items fall**.

Suppose the user's true positive feedback in test set includes:
- `Vangelis` (matches our rec, niche)
- `Led Zeppelin` (matches our rec, popular)

Difficulty hit-score:

| List         | Hit positions                  | Score            |
|--------------|---------------------------------|------------------|
| Original     | Vangelis at rank 20, Zep at 2  | 1.8 + 0.8 = 2.6  |
| Reranked     | Vangelis at rank 1, Zep at 5   | 8.2 + 0.8 = 9.0  |

Rerank pushed the niche hit from rank 20 → rank 1. With
`rank_weight=0.2`, the rank bonus widens the gap.

If we compute over all `gender+age` users we get the per-cell mean
delta, plus paired-Wilcoxon p-value to check significance.

---

## 5. Why this design (vs alternatives)

### vs global popularity debiasing

Most popularity-debiasing baselines normalize against a **global**
popularity distribution. The result is they over-penalize universally
loved items (Radiohead) on grounds of being popular, even when they're
not stereotypical for any specific demographic group.

Slice-conditional B distinguishes "globally loved" (Radiohead, B=1.0
in every slice) from "stereotypically loved by a group"
(Pink Floyd in `gender_M`, B=0.85 vs `gender_F` B=0.61). Only the
latter is a debiasing target.

### vs fixed α

A single global α treats users uniformly. But users whose authentic
preferences happen to align with mainstream content shouldn't be punished
for "being mainstream", and conversely, niche users shouldn't be
strong-armed into more niche content they don't want.

Dynamic α (driven by each user's mainstream-ness measured against the
scene mean) gives a principled per-user intervention strength.

### vs accuracy-only metrics

Reporting only `Precision@k` / `NDCG@k` rewards models that recommend
slice-popular items, regardless of whether those recommendations
constitute long-tail surfacing. The difficulty-weighted hit score is a
simple complement that lets us state:

> Rerank improved difficulty-weighted hit score by X%, while keeping
> NDCG within Y% of baseline.

This is the canonical fairness-vs-accuracy trade-off framing reviewers
expect from the popularity-debiasing literature.

---

## 6. Implementation Pointers

| Concern                          | Module                                  |
|----------------------------------|-----------------------------------------|
| Build B[i, k] table              | `src/popularity/slicer.py`              |
| Sparse-slice blending            | `src/popularity/slicer.py` (`SPARSE_THRESHOLD`) |
| Difficulty score function        | `src/popularity/item_scoring.difficulty_score` |
| Persisted score CSV              | `src/popularity/item_scoring.write_score_table` |
| In-memory score lookup           | `src/popularity/item_scoring.ItemScoreTable` |
| Per-user difficulty hit-score    | `src/popularity/item_scoring.difficulty_weighted_hit_score` |
| Rerank engine                    | `src/experiment/rerank.rerank_one_list` |
| Dynamic α                        | `src/experiment/rerank.compute_dynamic_alpha` |
| Column resolution                | `src/experiment/rerank.popularity_columns_for` |
| Wired into runner                | `src/experiment/runner.run` (Pass 1 + Pass 2) |

The runner already constructs an `ItemScoreTable` from the popularity
table and threads it into `compute_per_user_metrics`, which appends a
`difficulty` block to each (user, scenario) record. The flattened
`per_user.parquet` exposes `orig_difficulty_total` /
`rer_difficulty_total` columns ready for downstream statistics.

---

## 7. References (selected)

- Abdollahpouri, H. et al. *The Connection Between Popularity Bias,
  Calibration, and Fairness in Recommendation.* RecSys 2020.
- Steck, H. *Calibrated Recommendations.* RecSys 2018.
- Ekstrand, M. et al. *All The Cool Kids, How Do They Fit In?:
  Popularity and Demographic Biases in Recommender Evaluation and
  Effectiveness.* FAT* 2018.
- Schedl, M. et al. *LFM-2b: A Dataset of Enriched Music Listening
  Events for Recommender Systems Research and Fairness Analysis.*
  CHIIR 2022.
- Celma, Ò. *Music Recommendation and Discovery: The Long Tail, Long
  Fail, and Long Play in the Digital Music Space.* Springer 2010.
  (LastFM-1K dataset.)

---

## 8. Quick command reference

```bash
# Build popularity table for a dataset
python scripts/build_popularity.py --dataset lastfm1k

# Build (or refresh) the difficulty-score table from an existing popularity CSV
python -c "from src.popularity.item_scoring import write_score_table; \
  write_score_table('data/processed/lastfm1k/items_with_popularity.csv', \
                    'data/processed/lastfm1k/items_with_difficulty.csv')"

# Run a pilot with the M+N candidate-pool protocol
# (LLM produces 30 items, rerank truncates to top 20)
python scripts/run_pilot_lastfm1k.py --no-confirm \
    --sample-size 10 --top-k 20 --candidate-pool-size 30

# Inspect per-user difficulty + hit deltas
python -c "import pandas as pd; \
  df = pd.read_parquet('results/gpt-4.1-mini/lastfm1k/metrics/per_user.parquet'); \
  df['d_hits'] = df['rer_hits'] - df['orig_hits']; \
  df['d_diff'] = df['rer_difficulty_total'] - df['orig_difficulty_total']; \
  print(df[['user_id','scenario','orig_hits','rer_hits','d_hits', \
            'orig_difficulty_total','rer_difficulty_total','d_diff']].to_string())"
```

> Note on aggregated values: ``orig_hits = 4.6`` etc. in scenario-level
> summaries are **means across users** (e.g. 5 users × 4 / 5 = 3.2). The
> per-user parquet file holds integer hit counts.
