# 检索评测集

本目录包含 120 个问题，分布为 attr 25、reverse 20、intersect 20、compare 20、
path 20、safety/unanswerable 15。dev/test 按固定规则划分，不在运行时重新抽样。

题目字段包括答案要点、页码与 Chunk、必需事实、禁止无依据的表述以及 `should_abstain`。
现有标注记录为 2026-09-05 完成人工仲裁：113 题可答、7 题拒答；118 题填写了
`gold_pages` 和 `gold_chunk_ids`，两道没有来源证据的拒答题保留空数组。
全部题目的 `annotation_status` 为 `human_adjudicated`。

反向和交集问题另有 `gold_result_entities`，保存仲裁后的答案集合，数量可超过 5。
金标依据教材原文维护，不可从待测系统的当前输出反推。

```bash
python scripts/eval_retrieval.py --validate-only
python scripts/eval_retrieval.py --predictions path/to/predictions.jsonl
```

当前评分覆盖路由、实体、谓词、拒答和页码，预测字段见
`graphragexpr/evaluation/retrieval_eval.py`。排序文本块的 `Recall@5` 尚未实现。
