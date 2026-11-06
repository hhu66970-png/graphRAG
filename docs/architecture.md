# 架构说明

## 构建流程

```text
扫描 PDF → OCR Markdown → Chunk 与向量入库
                         ↓
药物条目识别 → 受控本体抽取 → 实体消歧 → 证据定位 → 关系校验
                         ↓
                Assertion 证据层 → Serving 查询层
```

`domain/ontology.yaml` 定义 14 类实体、13 个谓词及各自的 domain/range。
实体以 `canonical_id` 标识，名称用于展示与召回；不同类型的同名实体保留独立身份。
未知类型与谓词记录为抽取失败，不自动归入“概念”。

实体消歧（I1）按类型生成候选。权威别名可精确确认，包含关系与向量相似度仅用于
产生候选，避免将麻黄、麻黄根等相关但不同的药物合并。

关系校验（I2）先检查类型、自环、OCR 噪声与证据位置。非法关系直接拒绝；
高分的非高风险关系直接接受；灰区与高风险谓词进入模型复核。
`rule_score` 是规则分，`llm_verdict` 是复核结论，`final_score` 用于排序和决策，
均不表示事实正确概率。

跨块补全（I3）以药物条目 `entry_id` 为主要范围，补全“本品”等省略主语。
新增候选仍需经过证据定位与关系校验。

## 存储与发布

Assertion 保存事实、原文位置、极性和版本，Serving 边是其可重建的查询投影：

```text
(Assertion)-[:SUBJECT]->(Entity)
           -[:OBJECT]->(Entity)
           -[:EVIDENCED_BY]->(Chunk)

(Entity)-[:HAS_EFFICACY {fact_key, evidence_ids, page_start, ...}]->(Entity)
```

只有 `ACTIVE + ACCEPTED + POSITIVE` 的断言进入 Serving 图。发布前检查证据覆盖、
未知谓词比例和文本块失败率。批次从 `BUILDING` 转为 `ACTIVE` 或 `FAILED`，
新批次失败时已有的活动批次继续服务。

旧图保留作对照，使用中文标签和自由文本关系。新图通过稳定 ID、`run_id` 和
`fact_key` 区分。`build_kg_dyn.py` 的清库选项会同时影响两类数据。

## 检索与回答

Flask 托管页面与静态资源，浏览器通过 `/api/*` 请求数据，不直连 Neo4j。
自动路由包含属性、反查、交集、比较、路径和文本回退六类意图。
路径检索限领域白名单关系，最多三跳；比较查询对齐药物属性并汇集章节证据。

普通问题将证据包送入模型生成回答。高风险问题先定位实体，再从断言层读取
剂量、禁忌和配伍禁忌，组装包含对象、用法、人群、数值与原文位置的条件单元。
程序核验候选结论后生成回答，并用最终展示单元重建 `sources` 和 `subgraph`。

不同检索模式可能定位到同一组实体，此时高风险证据和结论也可能相同。
检索或核验失败时，仅展示标为未经核验的原文，或返回拒答说明。

前端通过 `answer_blocks` 展示结论、条件、原文依据与核验说明；与提问条件不一致的
有据内容放在折叠区。完整核验、部分核验、仅来源和拒答分别展示状态。

## 维护入口

- 阈值、预算与消融开关：`graphragexpr/config/settings.py`
- 实体与关系定义：`graphragexpr/domain/ontology.yaml`
- 高风险证据类型：`graphragexpr/domain/high_risk.py`
- 页面结构、样式与交互：`frontend/index.html`、`frontend/assets/app.css`、`frontend/assets/app.js`
- 回答展示回归：`tests/api/test_answer_display.py`

修改提示词或本体定义时更新缓存版本；修改展示时运行 Node 支持的前端回归。
教材引用、页码、标注状态和模型提示词是业务数据，不按文案风格统一改写。
