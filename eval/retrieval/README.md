# 检索评测集

本目录冻结 120 个检索问题，分布为 attr 25、reverse 20、intersect 20、
compare 20、path 20、safety/unanswerable 15。dev/test 按固定规则划分，不在运行时重切。

每题保留设计文档要求的 answer points、页码/Chunk、required facts、
forbidden unsupported claims 与 should_abstain 字段。当前仓库没有独立双人标注的
页码和 Chunk 金标准，因此 gold_pages 与 gold_chunk_ids 有意留空，并将
annotation_status 标为 question_frozen_evidence_pending_human_arbitration。

这些空字段只能由标注者查阅教材后双人标注并人工仲裁补齐，不可从当前图谱输出
反向生成，否则会把待评测系统本身当成金标准。补齐前可评测路由、实体和谓词召回，
不可据此声称 Recall@5、页码正确率或完整事实召回达到发布门槛。

校验数据集：

```bash
python scripts/eval_retrieval.py --validate-only
```

提供预测 JSONL 后运行：

```bash
python scripts/eval_retrieval.py --predictions path/to/predictions.jsonl
```

