# Top-15 双列表评分总结

- 输入目录：`reranked_JSON`
- 评分规则来源：`recommend_scoring.py`（同公式）
- Top-K：`15`

## 各场景结果（推荐列表总分 & average_score）

### neutral
- 用户数：`6031`
- 总分（全体求和）original：`120086.185400`
- 总分（全体求和）reranked：`120086.185400`
- 总分变化（reranked-original）：`+0.000000`
- 人均总分 original：`19.911488`
- 人均总分 reranked：`19.911488`
- 人均总分变化：`+0.000000`
- original：`5.185637`
- reranked：`5.185637`
- delta：`+0.000000`
- 结论（总分）：**不变**
- 结论（average_score）：**不变**

### gender
- 用户数：`6031`
- 总分（全体求和）original：`112393.765000`
- 总分（全体求和）reranked：`116290.521600`
- 总分变化（reranked-original）：`+3896.756600`
- 人均总分 original：`18.636008`
- 人均总分 reranked：`19.282129`
- 人均总分变化：`+0.646121`
- original：`4.921147`
- reranked：`4.935052`
- delta：`+0.013906`
- 结论（总分）：**增强**
- 结论（average_score）：**增强**

### age
- 用户数：`6031`
- 总分（全体求和）original：`116237.453600`
- 总分（全体求和）reranked：`120111.812000`
- 总分变化（reranked-original）：`+3874.358400`
- 人均总分 original：`19.273330`
- 人均总分 reranked：`19.915737`
- 人均总分变化：`+0.642407`
- original：`4.933660`
- reranked：`4.948151`
- delta：`+0.014490`
- 结论（总分）：**增强**
- 结论（average_score）：**增强**

### cross
- 用户数：`6031`
- 总分（全体求和）original：`111503.188500`
- 总分（全体求和）reranked：`115864.144300`
- 总分变化（reranked-original）：`+4360.955800`
- 人均总分 original：`18.488342`
- 人均总分 reranked：`19.211432`
- 人均总分变化：`+0.723090`
- original：`4.895657`
- reranked：`4.917118`
- delta：`+0.021461`
- 结论（总分）：**增强**
- 结论（average_score）：**增强**

