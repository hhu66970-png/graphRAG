# 《中药学》Evidence GraphRAG 知识图谱问答系统

面向《中药学（新世纪第五版）》的证据优先 GraphRAG。基线是老师的
[tjudb/2026graphrag](https://gitee.com/tjudb/2026graphrag)，主流程仍是
「文本分块 → LLM 抽实体关系 → Neo4j → Flask 问答可视化」，语料改为中药学教材，
建图改为**双层证据图谱**。

```text
PDF/OCR → 结构化 Chunk → 受控本体抽取 → Assertion 证据层
       → Neo4j Serving 投影 → 意图路由检索 → LLM 回答与前端可视化
```

每条正式事实必须能回到教材原句与页码；实体身份用 `canonical_id` 而不是药名；
构建走 `BUILDING → ACTIVE` 原子发布，失败不影响正在服务的版本。

教材是 502 页扫描 PDF（无文字层），仓库带可断点续跑的本地中文 OCR。
原始 PDF、OCR 全文、密钥和数据库数据默认不进 Git。

---

## 目录

1. [设计原则](#设计原则)
2. [建图改进与实测](#建图改进与实测)
3. [项目结构](#1-项目结构)
4. [安装](#2-安装)
5. [PDF 转语料](#3-pdf-转语料)
6. [构建图谱](#4-构建证据知识图谱)
7. [启动问答](#5-启动问答界面)
8. [测试](#6-测试)
9. [抽取评测](#7-抽取质量评测)
10. [检索评测](#8-检索评测集)
11. [离线数据包](#9-把已建好的图谱分享给别人无需云服务器)
12. [旧管线对照](#10-旧管线对照基线)
13. [已知局限与注意事项](#已知局限与注意事项)

文档、Schema 与代码不一致时的优先级：

```text
ontology.yaml 与稳定数据契约  >  API/测试契约  >  当前实现  >  过时文字说明
```

---

## 设计原则

- **证据优先**：`ACCEPTED` 断言必须有 Chunk、页码和可定位的逐字 `evidence_quote`；
  定位不到只能 `REVIEW`/`REJECTED`。禁止把模型常识写入正式事实图。
- **双层图**：Assertion 是事实源（证据、版本、极性、冲突）；Serving 边是可重建投影。
  只投影 `ACTIVE + ACCEPTED + POSITIVE`。否定事实（如「不宜与藜芦同用」）不画成正向边。
- **稳定身份**：`canonical_id = 类型 + hash(规范名)`。禁止用 `name` 做唯一身份，
  也不对实体名建全局唯一约束。
- **受控本体**：类型与谓词只来自 `graphragexpr/domain/ontology.yaml`（14 类实体、
  13 个稳定谓词，带 domain/range 与高风险标记）。未知类型**记录并拒绝**，
  不得静默降成「概念」。
- **按意图检索**：规则路由，不调 LLM。路径最多 3 跳且只走领域白名单边。
- **分数语义分离**：`rule_score`（规则）、`llm_verdict`（未复核为 `NOT_RUN`）、
  `final_score`（排序用，**不是**事实正确概率）。
- **关系校验 I2**：明确非法直接拒、不调 LLM；高分非高风险直接接受；灰区与全部
  高风险谓词（剂量、禁忌、配伍禁忌）才二次校验。
- **消歧 I1**：按类型分桶，禁止跨类型合并；除权威别名精确匹配外，包含式/向量
  只出候选（避免麻黄并进麻黄根）。
- **跨块 I3**：同药物条目内把「本品…」补回锚点药，新候选必须再过 I2。
- **技术边界**：Neo4j **5.26 LTS**；问答主链用原生 `neo4j` Driver，不依赖
  `neo4j-graphrag`；`hash` 嵌入只许测试，不能当演示结论。

关系校验流程：

```text
先算 rule_score
  ├─ 非法（domain/range、自环、OCR 噪声）→ REJECTED，不调 LLM
  ├─ 规则够高且非高风险 → ACCEPTED，llm_verdict = NOT_RUN
  └─ 灰区 + 全部高风险谓词 → LLM verifier → final_score
```

检索意图：

| 意图 | 做什么 |
|---|---|
| `attr` | 单味药属性卡片 |
| `reverse` | 按功效/主治反向查药 |
| `intersect` | 多约束硬交集（部分匹配单独标记） |
| `compare` | 多药事实槽对齐 + 章节摘要 |
| `path` | 白名单关系最多 3 跳 |
| `fallback` | 总论等走文本块检索 |

Flask 托管 `frontend/index.html` 与 `/api/*`。问答默认 `auto` 路由；
`keyword_graph` / `vector` / `hybrid` 等保留作课程对照。前端不直连 Neo4j。
`/api/search` 仍在，但当前页面不再调用。

---

## 建图改进与实测

老师基线是「开放本体 + 按名称建实体 + 自由文本关系 + 全库删除重建」。
在 870 个文本块上的问题（`python scripts/kg_v2_demo.py --audit` 可复现）：

| 问题 | 实测 | 根因 |
|---|---:|---|
| 同名跨类型实体 | 506 个名字 | 「呕吐」同时是概念/功效/禁忌/主治 |
| 笛卡尔积冗余边 | 8,623 / 38,085（22.6%） | `MATCH (n {name})` 无类型 |
| 关系类型失控 | 182 种 | 关系是自由文本 |
| 关系无来源 | 0 条带页码 | 写入不带 provenance |
| 未知类型被吞 | 982 个「概念」 | 不在白名单就降级 |

| 指标 | 旧图 | 新图 |
|---|---:|---:|
| 同名跨类型错连 | 506 | **0** |
| 冗余重复边 | 8,623（22.6%） | **0** |
| 关系类型数 | 182 | **12**（白名单内） |
| 带页码的边 | 0 / 38,085 | **16,681 / 16,681** |
| 「概念」兜底节点 | 982 | 88 |
| ACCEPTED 证据覆盖率 | 无断言层 | **100%** |

例：「人参的性味、归经和功效」旧链路会串到黄芪；新图跨页聚合且每条带页码。

与老师仓库的对应：红楼梦开放抽 → 中药学受控本体；`name` 身份 → `canonical_id`；
自由关系 → 13 谓词；无来源 → 逐字证据；`DETACH DELETE` → `IngestRun` 发布。

---

## 1. 项目结构

```text
backend/api.py                          Flask：问答、子图、健康检查
frontend/index.html                     问答前端（D3 子图，不直连数据库）

graphragexpr/
├── config/settings.py                  阈值、预算、消融开关
├── domain/ontology.yaml                受控本体（唯一类型/谓词事实源）
├── ingest/structure.py                 药物条目边界（entry_id）
├── extract/                            抽取、证据、剂量、I2/I3；旧 build_kg_dyn 作对照
├── linking/entity_resolver.py          I1 消歧
├── storage/                            Neo4j、Assertion、投影、原子发布
├── retrieval/                          意图路由与证据包
├── highrisk/                           剂量/禁忌程序核验
└── evaluation/                         抽取/检索/高风险评测逻辑

scripts/                                OCR、建图、评测、导入导出
eval/retrieval/questions.jsonl          120 题检索金标
eval/high_risk/questions.jsonl          高风险评测题
tests/                                  单元与契约测试
origdata/                               本机 OCR 语料（不提交）
```

Windows 下 `.\xxx.ps1` 是对应 Python 脚本的包装；macOS/Linux 直接跑
`python scripts/xxx.py`。

---

## 2. 安装

推荐 Python 3.12（用到 `zip(strict=True)` 等 3.10+ 语法）。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Anaconda：

```powershell
conda create -n zhongyao_graphrag python=3.12 -y
conda activate zhongyao_graphrag
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

编辑 `.env`，至少填 Neo4j 与 LLM。默认本地 `BAAI/bge-small-zh-v1.5`（512 维），
首次会下载模型。`EMBED_CACHE_DIR` 必须是**绝对路径**（代码不展开 `~`）。

```powershell
docker compose up -d neo4j
```

---

## 3. PDF 转语料

```powershell
python scripts\preprocess_pdf.py --pdf "D:\路径\中药学.pdf" --output origdata\zhongyaoxue.md
```

可加 `--start-page` / `--end-page` 试跑。OCR 缓存在 `origdata/.ocr-pages/`，
中断后重跑会跳过已完成页。

---

## 4. 构建证据知识图谱

```powershell
.\build_kg_v2.ps1
```

等价于 `python scripts\build_kg_v2.py --workers 6 --activate`：建索引 →
`IngestRun(BUILDING)` → 条目识别 → 并发抽取（磁盘缓存可续跑）→ I1/证据/I2 →
I3 跨块补全 → 写断言并投影 → 质量门通过后原子切 `ACTIVE`。

抽取缓存 `.cache/kg_v2/`。改 Prompt 或 Schema 须升 `settings.py` 版本号。

```powershell
python scripts\verify_kg_v2.py --herb 麻黄
```

```text
事实源  (Assertion)-[:SUBJECT]->(Entity)
                   -[:OBJECT]->(Entity)
                   -[:EVIDENCED_BY]->(Chunk)
投影    (Entity)-[:HAS_EFFICACY {fact_key, page_start, evidence_ids, ...}]->(Entity)
```

---

## 5. 启动问答界面

```powershell
.\start_backend.ps1
```

打开 <http://localhost:5000>。默认自动路由。可试：

- 人参的性味、归经和功效是什么？
- 哪些药同时具有清热解毒和凉血功效？
- 比较人参和黄芪的功效异同。
- 影响中药贮藏变异的外界因素有哪些？（总论，走 fallback）
- 什么是中药？

对照模式：中文混合向量+图谱、纯向量、向量+图遍历、混合+图遍历
（原生 Driver，读库中的旧图实体；不必再跑已删除的独立可视化脚本）。

---

## 6. 测试

```powershell
.\run_tests.ps1
```

默认无外部服务的单元测试。集成测试：`python -m pytest -m integration`。

`clean_database` 在目标库非空时会**跳过**而不是清空；破坏性测试须设
`ALLOW_DESTRUCTIVE_TESTS=true`。测试实体名带 `ITEST药` 前缀。最好用单独 Neo4j 实例。

---

## 7. 抽取质量评测

```powershell
.\eval_kg.ps1
```

教材 `【药性】`/`【功效】` 等可规则解析作参照（375 味药）：

| 字段 | Recall@field | Prec@field↓ |
|---|---:|---:|
| 性味 | 90.1% | 88.0% |
| 归经 | 78.6% | 90.9% |
| 功效 | 86.1% | 50.0% |
| **合计** | **85.2%** | **69.0%** |

口径：这是结构化字段参照，**不是**双人金标；**主治/配伍/禁忌无参照**；
Precision 是**下界**（字段外的合理事实不算错）。可说「字段召回 85.2%」，
**不能**说「准确率 85%」。`final_score` 也不是正确概率。

`final_accept_threshold = 0.80` 由标定曲线冻结。

---

## 8. 检索评测集

`eval/retrieval/questions.jsonl`：120 题（attr 25 / reverse 20 / intersect 20 /
compare 20 / path 20 / safety 15）。人工仲裁：113 题可答、7 题应拒答。
说明见 [`eval/retrieval/README.md`](eval/retrieval/README.md)。

```powershell
python scripts\eval_retrieval.py --validate-only
```

可评路由、实体、谓词、拒答、页码；**尚未**实现排序 Chunk 的 `Recall@5`，
不得声称达到 90% 门槛。

高风险题集见 [`eval/high_risk/README.md`](eval/high_risk/README.md)
（部分仍待人工仲裁）。

---

## 9. 把已建好的图谱分享给别人（无需云服务器）

```powershell
python scripts\export_kg_bundle.py --active-only
```

输出 `dist/zhongyaoxue-kg-bundle.zip`（已 gitignore，用 Release 或网盘分发）。

对方：

```powershell
Copy-Item .env.example .env
docker compose up -d neo4j
python scripts\import_kg_bundle.py --clear
.\start_backend.ps1
```

不必重新 OCR/抽全书。导入会重建向量与全文索引。建议再补约束和 `cjk` 分析器：

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); from graphragexpr.storage.neo4j_store import Neo4jStore; from graphragexpr.storage.indexes import ensure_schema; s=Neo4jStore(); print(ensure_schema(s, embedding_dim=512, rebuild_fulltext=True)); s.close()"
```

包内约有 Chunk 870（含向量）、Entity、Assertion、Serving 边及对照用旧图实体。
**不含** `LLM_TOKEN` 和 BGE 模型。`Chunk.text` 是教材 OCR，公开分发前确认授权。

---

## 10. 旧管线（对照基线）

`graphragexpr/extract/build_kg_dyn.py` 与 `build_kg_dyn.ps1` **刻意保留**：
课程需要新旧对照；前端向量/混合模式仍读旧图实体。新旧靠 `run_id`/`fact_key`
共存。只跑新管线时不必执行旧建图——导入数据包即可。

> 旧管线仍可能 `DETACH DELETE` 全库（`.env` 的 `CLEAR_DATABASE`）。
> 在已有 v2 数据的库上跑会把新图一起清掉。

---

## 已知局限与注意事项

- **总论题**：图里是类型化中药事实，「什么是中药」「贮藏因素」等散文不在图中，
  自动路由走文本 fallback。属性题上新图更好且带页码。
- **抽取噪声**：对比句可能误归属（如人参槽混入他人性味）。REVIEW 不进 Serving。
- **条目覆盖**：参照集里仍有少数药没有属性边；I3 只覆盖同条目。
- **生成**：没有 Claim–Evidence 程序化校验，模型仍可能编页码；前端只展示后端给的溯源。
- **链接阈值** `0.94/0.84` 未在完整金标上标定。
- 评测数字必须带口径；hash 嵌入不能当正式检索质量。
- 上传 Git 前确认 `.env`、zip、PDF、`.cache/`、`origdata/` 未被提交。

| 环节 | 状态 |
|---|---|
| OCR / 条目识别 | 已实现（条目覆盖约 86.7%） |
| 建图与本体 | 已完成 |
| 检索主链与 120 题金标 | 已完成 |
| 生成侧引用校验 | 未做 |
| Chunk Recall@5 | 未做 |
