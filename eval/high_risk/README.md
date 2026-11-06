# 高风险评测集

本目录包含 14 个问题：dev/test 各 7 题；剂量 4 题、配伍禁忌 5 题、禁忌 5 题。
每题记录原文、页码、必需事实、禁止无依据的表述以及是否应拒答，划分固定。

所有题目的 `annotation_status` 均为 `draft_from_textbook_pending_human_adjudication`。
这些标注依据教材起草，尚未完成人工仲裁。

## 运行

```bash
python scripts/eval_high_risk.py --validate-only
python scripts/eval_high_risk.py --export-adjudication high-risk-review.pending.json
python scripts/eval_high_risk.py --split test --out high-risk.json

# 显式实验臂：必须配置自己的兼容接口。不会自动生成或写入密钥。
python scripts/eval_high_risk.py --split test --scope question --arms raw_text_prompt,raw_text_verified,conditional_units_prompt,conditional_units_claims_verified --allow-llm --evidence-char-budget 6000 --max-output-tokens 512 --out high-risk-experiment.json --predictions-out high-risk-predictions.jsonl
```

实际评测需要 `.env` 中的数据库连接与 ACTIVE 批次；校验和仲裁模板导出均不连接数据库或模型。
脚本只读数据库，不构建或发布批次。

| 组别 | 方法 |
|---|---|
| `raw_text_prompt` | 原文及引用 ID，模型生成候选结论，不启用核验 |
| `raw_text_verified` | **相同原文、提示和候选**，检查引用存在并可定位；不声称语义蕴含或条件正确 |
| `conditional_units_prompt` | 条件证据单元，模型生成候选结论，关闭核验；只作实验，不进入生产 API |
| `conditional_units_claims_verified` | **相同条件单元、提示和候选**，启用现有 `verify_claims`；仅测试核验组件 |
| `conditional_units_verified` | 保留原行为的生产参考 C：条件单元、完整高风险服务和程序化回答，`llm=None` |

默认仍只运行生产参考 C；现场 API、前端、PPT 示例和原 14 题均不因新增实验臂改变。
四个实验臂需要 `--allow-llm`，否则记录为 `not_run`，不计为已回答。
允许模型时必须明确配置 `LLM_ENDPOINT`、`LLM_MODEL` 和 `LLM_TOKEN`，兼容现有 DeepSeek
配置；缺配置报错，不静默选取另一个服务。评测适配器固定温度 0 和输出 `max_tokens` 上限。

## 公平比较的范围

评测版本为 `high-risk-eval-2`。旧版 B 把引用固定为空并对空证据核验，结果天然拒绝；
本版改为真实的引用定位对照，**新旧 B 的数字不可直接比较**。

- A/B 和“条件单元无核验/有核验”各自复用一次生成的同一批候选，而非重新抽样两次。
  这样可分别观察引用门控和完整结论核验对同一批候选的影响。
- 四个实验臂共享冻结的断言来源，完整保留或跳过一条来源，保证原文和条件表示都满足
  字符预算。字符数不是模型 token 数；两种表示实际输入长度不同，不声称 token 完全等长。
- `--scope question`（实验臂默认）只从问句链接对象，不读取 `expected_subject` 等答案标签。
  `--scope component` 显式使用金标主语，仅适合隔离回答组件，存在金标主语信息。
  两者都直接读取高风险断言，**都不是检索与 API 的完整端到端评测**；未解析对象单独记为失败，
  不算正确拒答。反向枚举等完整生产行为应另外通过 API 端到端测试检验。
- 生产参考 C 保留旧评测的金标主语接线、自己的筛选预算及程序化生成方式。它和实验臂的
  差别混合了证据筛选、生成和核验，不能把二者的差值都归功于条件结构或核验。
- `controlled_pairs` 逐题校验来源、提示和候选摘要是否相同；只有对应检查成立，才可描述为
  该生成样本上的开关消融。跨表示的优劣仍需人工语义评价，并报告输入长度和调用成本。

## 留存证据与仲裁流程

结果 JSON 总是保存逐题 `predictions`；`--predictions-out` 可另导出 JSONL。记录包含最终可见
结论、输入证据、候选字段、引用、拒因代码、来源/提示/候选摘要。模型 `notes`、推理字段、
原始异常文本与密钥不导出。生产参考未公开的中间候选明确标为不可获取，不伪造记录。
运行元数据保存 Git SHA、分支、工作区是否有修改、评测代码文件摘要、题集摘要、
批次 ID、模型名、温度、输出上限和证据预算；不保存 token、端点认证信息或整份环境配置。

`--export-adjudication` 只导出待审工作表，不修改原题集：

1. 两名审核者独立核对教材版本、原文和页码，在不看系统答案的情况下写答案要点、禁止表述
   和应否拒答；既有字段放在 `seed_proposal_not_gold`，不能自动当作医学金标。
2. 每题附一个空的 `paired_variant_draft`，由人工只改变一个明确条件并填写新问句、
   变化字段及来源。新题的答案、拒答标签和与原题关系均待审核，不自动复制专业结论。
3. 仲裁分歧并保留审核人、证据及最终决定。同一对题留在同一 split；冻结 test 后不用于
   改提示词或调阈值。空模板不是可直接评分的数据集。
4. 当前仍不自动计算语义分数；14 题保持 draft，`semantic_scoring_allowed=false`。
   即使将来人工完成标注，仍须另外实现并审阅语义评分程序，不能只改状态来制造分数。

## 指标口径

当前描述性统计引用是否存在、数字与单位是否同源、限制程度是否改写、是否出现肯定式
安全表述，以及是否引用不可引用单元。这些指标可通过响应和证据核对，但不能代替
对答案正确性与完整性的判断。

语义指标需要人工标注，当前 `semantic_metrics` 返回 `not_measured`。
`subject_not_resolved` 单独记录评测对象未能链接的情况，避免将其计为正确拒答。
同时报告回答、拒答、未运行和失败数；不能以大量拒答后错误计数下降冒充正确率提高。
不同表示的机械指标可观察范围也不同，`delta` 仅是描述计数，不是语义质量或因果收益。

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
