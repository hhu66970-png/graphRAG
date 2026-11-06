# 高风险评测集

本目录包含 14 个问题：dev/test 各 7 题；剂量 4 题、配伍禁忌 5 题、禁忌 5 题。
每题记录原文、页码、必需事实、禁止无依据的表述以及是否应拒答，划分固定。

所有题目的 `annotation_status` 均为 `draft_from_textbook_pending_human_adjudication`。
这些标注依据教材起草，尚未完成人工仲裁。

## 运行

```bash
python scripts/eval_high_risk.py --validate-only
python scripts/eval_high_risk.py --split test --out high-risk.json
python scripts/eval_high_risk.py --arms raw_text_prompt,raw_text_verified,conditional_units_verified --allow-llm
```

实际评测需要 `.env` 中的数据库连接与 ACTIVE 批次，校验数据集无需连接数据库。
脚本只读数据库，不构建或发布批次。

| 组别 | 方法 |
|---|---|
| `raw_text_prompt` | 教材原文和提示词，模型直接生成 |
| `raw_text_verified` | 原文生成后核验候选结论，不使用条件证据单元 |
| `conditional_units_verified` | 条件证据单元、程序核验与程序化回答，即当前问答流程 |

默认运行第三组。前两组需要 `--allow-llm`，否则记录为 `not_run`。

## 指标口径

当前统计引用是否存在、数字与单位是否同源、限制程度是否改写、是否出现肯定式
安全表述，以及是否引用不可引用单元。这些指标可通过响应和证据核对，但不能代替
对答案正确性与完整性的判断。

语义指标需要人工标注，当前 `semantic_metrics` 返回 `not_measured`。
`subject_not_resolved` 单独记录评测对象未能链接的情况，避免将其计为正确拒答。

## 已知边界

`hr-contra-004`（川乌）的两处原文存在程度差异：第 179 页为“孕妇慎用”，
第 67 页列入妊娠禁用药。是否属于相同条件仍待人工确认，系统保留冲突说明，
不自行选取其中一种程度作为已核验结论。

现有记录中，第 179 页的证据单元超过默认 3000 token 预算，无法与另一处完整并列。
服务在裁剪前检查冲突，返回 `partial_grounded` 并说明覆盖不足。
预算问题和语义仲裁需要分别处理。

最终高风险证据来自 `load_high_risk_assertions`；`auto`、`keyword_graph`、`vector`
用于前置检索和实体定位。实体相同时，三种模式可能返回相同证据与结论，
不能据此比较三种普通检索方法的召回质量。

## 回归测试与人工仲裁

`tests/high_risk/test_regression_e2e.py` 覆盖复方名误匹配单味药、人群条件错配、
未知对象误走反向枚举及其正确对照，需要真实数据库与 ACTIVE 批次。
接口调用顺序见 `tests/api/test_high_risk_api.py`；相关性选择、展示去重和折叠区
见 `tests/api/test_answer_display.py` 与 `tests/high_risk/test_relevance_and_decision.py`。

仲裁时逐题核对原文、答案要点和禁止表述，确认川乌的条件冲突以及两道拒答题的证据范围。
完成后再修改标注状态并实现语义评分，不以系统输出替代人工金标。
