# Top-15 Rerank 指标总结（可读版）

数据来源：`reranked_JSON`（共 6031 用户，`k=15`）

## 总体结论

- 在 `age`、`gender`、`cross` 场景下，重排序后整体增强。
- 在 `neutral` 场景下，指标保持不变（bypass）。

## 指标说明

- `original`：原始推荐列表 `recommendations` 的指标值
- `reranked`：重排序列表 `reranked_recommendations` 的指标值
- `delta`：`reranked - original`

## 场景：age

- 用户数：`6031`
- Macro Precision：original=`0.219776`，reranked=`0.227138`，delta=`+0.007362`
- Macro Recall：original=`0.069962`，reranked=`0.072270`，delta=`+0.002309`
- Micro Precision：original=`0.219776`，reranked=`0.227138`，delta=`+0.007362`
- Micro Recall：original=`0.049194`，reranked=`0.050842`，delta=`+0.001648`
- HitRate@15：original=`0.852927`，reranked=`0.858896`，delta=`+0.005969`
- MRR@15：original=`0.490703`，reranked=`0.539631`，delta=`+0.048928`
- NDCG@15：original=`0.505424`，reranked=`0.545028`，delta=`+0.039604`
- Total Hits：original=`19882`，reranked=`20548`
- Total Test Movies Count：`404151`

结论：**增强**。

## 场景：gender

- 用户数：`6031`
- Macro Precision：original=`0.210977`，reranked=`0.218538`，delta=`+0.007561`
- Macro Recall：original=`0.067240`，reranked=`0.069662`，delta=`+0.002422`
- Micro Precision：original=`0.210977`，reranked=`0.218538`，delta=`+0.007561`
- Micro Recall：original=`0.047225`，reranked=`0.048917`，delta=`+0.001692`
- HitRate@15：original=`0.842315`，reranked=`0.847621`，delta=`+0.005306`
- MRR@15：original=`0.474000`，reranked=`0.526836`，delta=`+0.052836`
- NDCG@15：original=`0.493594`，reranked=`0.536663`，delta=`+0.043069`
- Total Hits：original=`19086`，reranked=`19770`
- Total Test Movies Count：`404151`

结论：**增强**。

## 场景：cross

- 用户数：`6031`
- Macro Precision：original=`0.211496`，reranked=`0.219997`，delta=`+0.008501`
- Macro Recall：original=`0.067501`，reranked=`0.070013`，delta=`+0.002512`
- Micro Precision：original=`0.211496`，reranked=`0.219997`，delta=`+0.008501`
- Micro Recall：original=`0.047341`，reranked=`0.049244`，delta=`+0.001903`
- HitRate@15：original=`0.850108`，reranked=`0.856409`，delta=`+0.006301`
- MRR@15：original=`0.477566`，reranked=`0.528800`，delta=`+0.051234`
- NDCG@15：original=`0.495734`，reranked=`0.538938`，delta=`+0.043204`
- Total Hits：original=`19133`，reranked=`19902`
- Total Test Movies Count：`404151`

结论：**增强**。

## 场景：neutral

- 用户数：`6031`
- Macro Precision：original=`0.216857`，reranked=`0.216857`，delta=`+0.000000`
- Macro Recall：original=`0.069983`，reranked=`0.069983`，delta=`+0.000000`
- Micro Precision：original=`0.216857`，reranked=`0.216857`，delta=`+0.000000`
- Micro Recall：original=`0.048541`，reranked=`0.048541`，delta=`+0.000000`
- HitRate@15：original=`0.857238`，reranked=`0.857238`，delta=`+0.000000`
- MRR@15：original=`0.476400`，reranked=`0.476400`，delta=`+0.000000`
- NDCG@15：original=`0.498029`，reranked=`0.498029`，delta=`+0.000000`
- Total Hits：original=`19618`，reranked=`19618`
- Total Test Movies Count：`404151`

结论：**不变**。

## 文件对应关系

- 汇总数据：`recommendations_top15_reranked_analysis/summary_top15.json`
- 对比数据：`comparison_top15_original_vs_reranked_list.json`
