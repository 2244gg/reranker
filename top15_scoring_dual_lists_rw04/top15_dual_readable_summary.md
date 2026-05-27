# Top-15 双列表评分总结

- 输入目录：`reranked_JSON`
- 评分规则来源：`recommend_scoring.py`（同公式）
- 排名权重 rank_weight：`0.4`
- Top-K：`15`

## 各场景结果（推荐列表总分 & average_score）

### neutral
- 用户数：`6031`
- 总分（全体求和）original：`134565.988600`
- 总分（全体求和）reranked：`134565.988600`
- 总分变化（reranked-original）：`+0.000000`
- 人均总分 original：`22.312384`
- 人均总分 reranked：`22.312384`
- 人均总分变化：`+0.000000`
- original：`5.817794`
- reranked：`5.817794`
- delta：`+0.000000`
- 结论（总分）：**不变**
- 结论（average_score）：**不变**

### gender
- 用户数：`6031`
- 总分（全体求和）original：`125974.483900`
- 总分（全体求和）reranked：`130426.199100`
- 总分变化（reranked-original）：`+4451.715200`
- 人均总分 original：`20.887827`
- 人均总分 reranked：`21.625966`
- 人均总分变化：`+0.738139`
- original：`5.522808`
- reranked：`5.544219`
- delta：`+0.021411`
- 结论（总分）：**增强**
- 结论（average_score）：**增强**

### age
- 用户数：`6031`
- 总分（全体求和）original：`130237.588500`
- 总分（全体求和）reranked：`134641.661100`
- 总分变化（reranked-original）：`+4404.072600`
- 人均总分 original：`21.594692`
- 人均总分 reranked：`22.324931`
- 人均总分变化：`+0.730239`
- original：`5.535206`
- reranked：`5.555931`
- delta：`+0.020725`
- 结论（总分）：**增强**
- 结论（average_score）：**增强**

### cross
- 用户数：`6031`
- 总分（全体求和）original：`124941.974800`
- 总分（全体求和）reranked：`129892.333000`
- 总分变化（reranked-original）：`+4950.358200`
- 人均总分 original：`20.716627`
- 人均总分 reranked：`21.537445`
- 人均总分变化：`+0.820819`
- original：`5.491667`
- reranked：`5.520811`
- delta：`+0.029144`
- 结论（总分）：**增强**
- 结论（average_score）：**增强**

