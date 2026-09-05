# 检索评测集

本目录冻结 120 个检索问题，分布为 attr 25、reverse 20、intersect 20、
compare 20、path 20、safety/unanswerable 15。dev/test 按固定规则划分，不在运行时重切。

每题保留设计文档要求的 answer points、页码/Chunk、required facts、
forbidden unsupported claims 与 should_abstain 字段。2026-09-05 已逐题回看教材
`Chunk.text` 并完成人工仲裁：113 题正常作答，7 题必须拒答；118 题已填写
`gold_pages` 和 `gold_chunk_ids`，两道题设本身无来源证据的拒答题保留空数组。
全部题目的 `annotation_status` 均为 `human_adjudicated`。

反向和交集查询另有 `gold_result_entities`，保存本次仲裁冻结的答案集合；答案数量
可以超过 5。金标必须来自教材原文与人工仲裁，不可从待测系统的当前输出反推，
否则会把被评测系统本身当成答案。

当前评分脚本尚未实现排序 Chunk 的 `Recall@5`，也不会把“前 5 个输出 token”当作
检索结果。完成排序预测格式与对应指标前，不可声称 Recall@5 已达到发布门槛。

校验数据集：

```bash
python scripts/eval_retrieval.py --validate-only
```

提供预测 JSONL 后运行：

```bash
python scripts/eval_retrieval.py --predictions path/to/predictions.jsonl
```
