# 《中药学》Evidence GraphRAG 知识图谱问答系统

本项目以老师的 [tjudb/2026graphrag](https://gitee.com/tjudb/2026graphrag) 为基线，保留其“文本分块 → LLM 抽取实体关系 → Neo4j 图谱与向量索引 → Flask API → 图谱可视化问答”的主流程，语料与抽取模式改为《中药学（新世纪第五版）》，并在建图环节重构为**证据优先的双层图谱**：

> 每条正式事实都必须能回到教材原句与页码；实体身份用稳定 `canonical_id` 而非名称；
> 构建走 `BUILDING → ACTIVE` 原子发布，失败不影响正在服务的版本。

> 原教材是 502 页扫描 PDF，没有文字层，仓库因此包含可断点续跑的本地中文 OCR 流程。
> 原始 PDF、OCR 全文、密钥和数据库数据均不会被 Git 跟踪。

**文档入口**

- [`docs/`](docs/) —— 技术文档：[团队开发规范](docs/团队开发规范.md)（动代码前必读）、
  [总体系统设计](docs/总体系统设计与可行性评审.md)、
  [KG 构建环节改进技术文档](docs/KG构建环节改进技术文档.md)
- [`待办与注意事项/`](待办与注意事项/) —— [注意事项](待办与注意事项/注意事项.md)（会踩的坑）、
  [待办清单](待办与注意事项/待办清单.md)（按 Owner 划分）、
  [已知局限](待办与注意事项/已知局限.md)（系统边界）

---

## 建图环节改进概览

老师基线的建图方式是「开放本体 + 按名称建实体 + 自由文本关系 + 全库删除重建」。
在全书 870 个文本块上实测，这套方式产生了如下问题（`scripts/kg_v2_demo.py --audit`
可随时复现）：

| 问题 | 实测 | 根因 |
|---|---:|---|
| 同名跨类型实体 | 506 个名字 | 「呕吐」同时是概念/功效/禁忌/主治，各建一个节点 |
| 笛卡尔积冗余边 | 8,623 / 38,085（22.6%） | 写关系时 `MATCH (n {name})` 无类型，一条事实被复制成多条边 |
| 关系类型失控 | 182 种（82 种只出现 ≤2 次） | 关系类型是自由文本，「主编」「习称」都成了正式关系 |
| 关系无来源 | 0 条带页码 | 图谱推出的结论标不出教材出处 |
| 未知类型被吞 | 982 个「概念」节点 | 不在白名单的类型一律降级为「概念」，无处查证 |

### 改了什么

**身份**：实体身份改用稳定 `canonical_id = 类型 + hash(规范名)`，类型编进身份，
因此「呕吐(主治)」与「呕吐(功效)」天然是两个实体，笛卡尔积在结构上无法发生。

**本体**：`domain/ontology.yaml` 定义 14 类实体与 13 个稳定谓词，每个谓词带
domain/range 与高风险标记。未知类型**记录并拒绝**，不再静默降级。

**证据**：每条正式事实都要有逐字原句、原文字符跨度与页码。证据定位走四级匹配
（原文精确 → 规范化精确 → 极小编辑距离），能容忍 OCR 的空格、换行与全半角差异；
**定位不到的只能进 REVIEW，不得入正式事实图**。

**双层图**：`Assertion` 证据层是事实源，类型化直接边是可重建的服务投影。
审计、版本、多来源冲突由断言层承担，检索侧仍然查得快。

**分数语义**：`rule_score`（确定性规则分）、`llm_verdict`（未复核时为 `NOT_RUN`）、
`final_score`（决策支持值）三者分离，不再混用一个 `support_score`。

**发布**：新构建先 `BUILDING`，质量门槛（证据覆盖率、块级失败率）通过后**原子切换**
为 `ACTIVE`，失败则标记 `FAILED` 且旧版本继续服务。不再 `DETACH DELETE` 全库。

**质量三件套**：I1 分层实体消歧（严禁跨类型与包含式自动合并，
麻黄/麻黄根、黄连/胡黄连不会被并成一个）、I2 关系规则验证 + 灰区二次校验、
I3 跨块补全（把「本品味辛性凉…」这类省略主语的句子归回条目锚点药）。

### 效果

| 指标 | 旧图 | 新图 |
|---|---:|---:|
| 同名跨类型导致的错连 | 506 | **0** |
| 冗余重复边 | 8,623（22.6%） | **0** |
| 关系类型数 | 182 | **12**（本体白名单内） |
| 带页码 provenance 的边 | 0 / 38,085 | **16,681 / 16,681** |
| 「概念」兜底节点 | 982 | 88 |
| ACCEPTED 断言证据覆盖率 | 无断言层 | **100%** |

最直观的例子是「人参的性味、归经和功效」——旧检索链路答不出来（返回的是黄芪的
三元组），新图里这些属性跨第 50/68/378/379/380 页聚合在一起，每条带页码：

```
性味为   甘、微苦，微温                              第378页
归经为   归脾、肺、心、肾经                          第378页
具有功效 大补元气、复脉固脱、补脾益肺、生津养血、安神益智  第378页
证据     第378页「本品甘温补虚，能大补元气，复脉固脱，为拯危救脱之要药。」
```

完整的问题诊断、逐条改动与实测数据见
[KG 构建环节改进技术文档](docs/KG构建环节改进技术文档.md)。

---

## 1. 项目结构

```text
backend/api.py                          Flask API（五种检索模式的编排）
frontend/index.html                     中药学知识图谱前端

graphragexpr/
├── config/settings.py                  全部阈值、预算、版本与消融开关
├── domain/
│   ├── ontology.yaml                   受控本体：14 类实体 + 13 个稳定谓词
│   ├── ontology.py                     本体加载与 domain/range 校验
│   ├── models.py                       ChunkRecord / EntityRecord / AssertionRecord
│   └── ids.py                          稳定 ID（canonical_id / chunk_id / fact_key）
├── ingest/structure.py                 从教材版式识别药物条目边界（entry_id）
├── extract/
│   ├── evidence.py                     四级证据定位，返回原文字符跨度
│   ├── dosage.py                       剂量值对象解析（数值/单位/给药方式/条件）
│   ├── relation_validator.py           I2 规则验证 + 灰区二次校验
│   ├── cross_chunk.py                  I3 跨块补全的分组策略
│   ├── extractor.py                    抽取管线整合
│   ├── llm_client.py                   统一 LLM 客户端（重试/JSON 校验/失败清单）
│   ├── corpus.py                       页码感知的 Markdown 分块（新旧共用）
│   ├── build_kg_dyn.py                 ← 旧版建图，保留作对照基线（见 §9）
│   └── rag_graph.py / rag_vector.py    ← 旧版检索脚本，同上
├── linking/entity_resolver.py          I1 分层实体消歧
├── storage/
│   ├── assertions.py                   Assertion 事实源写入
│   ├── serving_projection.py           类型化 Serving 边投影
│   ├── serving_read.py                 属性卡片 / 反向检索 / 证据回溯
│   ├── runs.py                         IngestRun 原子发布与回滚
│   └── indexes.py                      约束与索引
└── evaluation/                         参照集构建与抽取质量评测

scripts/
├── preprocess_pdf.py                   扫描 PDF OCR → Markdown
├── build_kg_v2.py                      构建证据图谱（主入口）
├── verify_kg_v2.py                     新旧对照 + 属性卡片演示
├── eval_extraction.py                  抽取质量评测
├── calibrate_thresholds.py             阈值标定曲线
└── export_kg_bundle.py / import_kg_bundle.py   离线数据包导出/导入

origdata/                               本地语料目录（默认不提交）
tests/kg/                               建图环节测试（160 项）
```

---

## 2. 安装

推荐 Python 3.12（代码使用 `zip(strict=True)` 等 3.10+ 语法，3.9 无法运行）。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

如果使用 Anaconda：

```powershell
conda create -n zhongyao_graphrag python=3.12 -y
conda activate zhongyao_graphrag
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS / Linux 用 `python3 -m venv .venv && source .venv/bin/activate`，
其余命令把 `.\xxx.ps1` 换成对应的 `python scripts/xxx.py` 即可。

编辑 `.env`，至少填写 Neo4j 与 LLM 配置。默认使用本地中文模型
`BAAI/bge-small-zh-v1.5` 生成 512 维向量，首次运行会下载模型。
如需改用兼容 OpenAI Embeddings API 的服务，设 `EMBED_PROVIDER=external`
并填写 `EMBED_ENDPOINT`、`EMBED_MODEL`、`EMBED_TOKEN`。

> `EMBED_CACHE_DIR` 请填**绝对路径**。代码不做 `~` 展开，写 `~/...` 会在仓库里
> 创建一个名为 `~` 的字面量目录。

Neo4j 固定为 **5.26 LTS**（2026 系列已将 `db.index.vector.queryNodes` 标记为
deprecated）。没有本机 Neo4j 时：

```powershell
docker compose up -d neo4j
```

---

## 3. PDF 转语料

先做小范围试跑：

```powershell
python scripts\preprocess_pdf.py `
  --pdf "D:\你的路径\中药学.pdf" `
  --output origdata\zhongyaoxue.md `
  --start-page 28 --end-page 50
```

确认效果后处理全书：

```powershell
python scripts\preprocess_pdf.py --pdf "D:\你的路径\中药学.pdf" --output origdata\zhongyaoxue.md
```

OCR 每页结果缓存在 `origdata/.ocr-pages/`，中断后执行同一命令会跳过已完成页面。

---

## 4. 构建证据知识图谱

```powershell
.\build_kg_v2.ps1
```

等价于 `python scripts\build_kg_v2.py --workers 6 --activate`。管线依次执行：

1. 建立约束与索引（5 个唯一约束 + 向量/全文索引，全文用 `cjk` 分析器）；
2. 新建 `IngestRun(status=BUILDING)`；
3. 从教材版式识别药物条目边界，为每块赋 `entry_id`；
4. 并发抽取，**结果按 `内容哈希 + prompt_version + schema_version` 磁盘缓存**，可断点续跑；
5. I1 实体消歧 → 证据定位 → I2 规则验证（灰区与全部高风险谓词走二次校验）；
6. I3 跨块补全：把「本品味辛性凉…」这类省略主语的句子归回条目锚点药；
7. 写入 Entity 与 Assertion，重建类型化 Serving 投影；
8. 质量门槛通过后**原子切换**为 `ACTIVE`，否则标记 `FAILED` 并保留旧版本继续服务。

抽取缓存在 `.cache/kg_v2/`，重跑不会重复付费。改动 Prompt 或 Schema 必须升
`settings.py` 里的版本号，缓存会随之失效。

查看构建结果与新旧对照：

```powershell
python scripts\verify_kg_v2.py --herb 麻黄
```

### 双层图数据模型

```text
事实源层  (Assertion)-[:SUBJECT]->(Entity)
                     -[:OBJECT]->(Entity)
                     -[:EVIDENCED_BY]->(Chunk)
                     -[:IN_RUN]->(IngestRun)

投影层    (Entity)-[:HAS_EFFICACY {fact_key, evidence_ids,
                                   page_start, page_end, active_run_id}]->(Entity)
```

Serving 边只投影 `ACTIVE + ACCEPTED + POSITIVE` 的断言，可随时重建；
审计、版本与多来源由 Assertion 层承担。否定事实（如「不宜与藜芦同用」）
不会被投影成肯定边。

---

## 5. 启动问答界面

```powershell
.\start_backend.ps1
```

浏览器打开 <http://localhost:5000>。前端下拉提供五种检索方式：

| 模式 | 说明 |
|---|---|
| **证据知识图谱 v2**（默认） | 实体链接 → 类型化属性卡片 → 带页码作答，读 Serving 投影 |
| 中文混合向量+图谱 | 中文 BM25 与 BGE 向量融合，块内展开图谱关系 |
| 纯向量 / 向量+图遍历 / 混合+图遍历 | 保留作课程对比实验 |

可尝试的问题：

- “人参的性味、归经和功效是什么？”
- “影响中药贮藏变异的外界因素有哪些？”
- “什么是中药？中药材、中药饮片和中成药有什么区别？”

---

## 6. 测试

```powershell
.\run_tests.ps1
```

默认只运行无需外部服务的单元测试；集成测试用 `python -m pytest -m integration`。

> **集成测试与生产共用数据库时的注意事项**：`clean_database` fixture 在目标库
> 非空时会**跳过**而不是清空，需要在非空库上跑破坏性测试须显式设置
> `ALLOW_DESTRUCTIVE_TESTS=true`。测试实体名一律带 `ITEST药` 前缀，避免与真实
> 实体撞上同一个 `canonical_id`。更稳妥的做法是给测试单独起一个 Neo4j 实例。

---

## 7. 抽取质量评测

教材条目自带结构化字段（`【药性】`、`【功效】`、`【用法用量】`），可**纯规则解析、
不调用 LLM**，作为抽取质量的参照基准：

```powershell
.\eval_kg.ps1
```

等价于依次运行 `scripts\eval_extraction.py` 与 `scripts\calibrate_thresholds.py`。

在 375 味药的参照集上的实测结果：

| 字段 | 金标取值 | 图谱取值 | Recall@field | Prec@field↓ |
|---|---:|---:|---:|---:|
| 性味 | 1,099 | 1,125 | 90.1% | 88.0% |
| 归经 | 966 | 835 | 78.6% | 90.9% |
| 功效 | 1,211 | 2,086 | 86.1% | 50.0% |
| **合计** | **3,276** | **4,046** | **85.2%** | **69.0%** |

**引用这些数字时必须连同口径说明**：

- 这是「教材结构化字段参照集」，**不是**双人标注仲裁的金标准；
- 只覆盖 性味/归经/功效，**主治、配伍、禁忌没有任何参照**；
- Precision 标注 ↓ 表示**下界**——图谱合理包含字段之外的正确事实
  （【应用】段落、跨章节引用），它们不在参照集里但并不算错；
- 因此可以说「在教材结构化字段上字段召回率 85.2%」，
  **不能**简化成「准确率 85%」。

接受阈值 `final_accept_threshold = 0.80` 由标定曲线冻结：相比 0.60 只损失
0.4pp 召回，换来 1.8pp 精确率；再往上到 0.85 召回骤降 7pp。

---

## 8. 把已建好的图谱分享给别人（无需云服务器）

向量写在 Neo4j 的 `Chunk.embedding` 上，与图谱、索引在同一库中，打包成离线 zip 分发。

### 导出

```powershell
python scripts\export_kg_bundle.py --active-only
```

`--active-only` 排除已废弃 run 的 Assertion/Entity，只导出当前 `ACTIVE` 版本；
不加则导出全库。输出 `dist/zhongyaoxue-kg-bundle.zip`。

### 导入

对方只需：Python 环境、Neo4j、本仓库代码与该 zip。不必重新 OCR，也不必再调 LLM。

```powershell
Copy-Item .env.example .env   # 填写 Neo4j；问答还需要自己的 LLM_TOKEN
docker compose up -d neo4j
python scripts\import_kg_bundle.py --clear
.\start_backend.ps1
```

导入脚本会重建向量与全文索引。若要补齐 5 个唯一约束并把全文索引换成 `cjk`
分析器（推荐），再执行一次：

```powershell
python -c "from dotenv import load_dotenv; load_dotenv(); from graphragexpr.storage.neo4j_store import Neo4jStore; from graphragexpr.storage.indexes import ensure_schema; s=Neo4jStore(); print(ensure_schema(s, embedding_dim=512, rebuild_fulltext=True)); s.close()"
```

### 数据包内容（v2.0.0）

| 内容 | 数量 | 是否包含 |
|---|---:|---|
| Chunk（含 512 维向量） | 870 | 是 |
| Entity（规范实体） | 9,243 | 是 |
| Assertion（证据断言） | 22,211 | 是 |
| 类型化 Serving 边 | 16,681 | 是 |
| 旧图实体（对照检索模式用） | 11,107 | 是 |
| 向量 / 全文索引 | — | 导入脚本会重建 |
| 查询用 BGE 模型 | — | 否，首次运行自动下载 |
| `LLM_TOKEN` | — | **否，切勿分享密钥** |
| OCR 全文 `zhongyaoxue.md` | — | 否（Chunk 文本已在包内） |

分享方式：GitHub **Releases** 附件或网盘，**不要**把 zip 推进源码仓库
（`dist/*.zip` 已在 `.gitignore`）。

---

## 9. 旧管线（对照基线）

`graphragexpr/extract/build_kg_dyn.py` 与 `build_kg_dyn.ps1` 是改造前的建图方式，
**刻意保留**，用途有二：

1. 课程报告需要新旧对照，旧图与新图靠 `run_id` / `fact_key` 在同一库中并存；
2. 前端的纯向量、向量+图遍历、混合+图遍历三种模式仍读旧图实体。

两者互不干扰：旧管线的关系不带 `fact_key`，新管线的投影清理只作用于带 `fact_key`
的边。若只想跑新管线，不需要执行 `build_kg_dyn.ps1`——从数据包导入即可（§8）。

> ⚠ 旧管线仍会 `DETACH DELETE` 全库（受 `.env` 的 `CLEAR_DATABASE` 控制），
> 在已有 v2 数据的库上执行会连新图一起清掉。详见
> [注意事项 §1](待办与注意事项/注意事项.md)。

## 与老师仓库的对应关系

| 老师仓库 | 本项目 |
|---|---|
| `origdata/orig.txt` 红楼梦文本 | `origdata/zhongyaoxue.md` OCR 教材文本 |
| 人物、地点、物品、概念（开放本体） | `domain/ontology.yaml` 受控本体，未知类型记录并拒绝 |
| 实体身份用 `name` | 稳定 `canonical_id`，同名不同类型天然是不同实体 |
| 自由文本关系类型 | 13 个稳定谓词，带 domain/range 与高风险标记 |
| 关系无来源 | 每条断言带 `chunk_ids`、页码与逐字 `evidence_quote` |
| `DETACH DELETE` 全库重建 | `IngestRun` 原子发布与回滚 |
| `sample_data_dyn.py` 按回目取样 | `corpus.py` 按页码与标题取样 |
| Neo4j 向量/全文检索 | 保持一致，全文索引改用 `cjk` 分析器 |
| Flask + D3 前端 | 保持一致，新增 v2 检索模式 |

---

## 当前状态与下一步

| 环节 | Owner | 状态 |
|---|---|---|
| ingest / OCR / 结构解析 | A | 条目边界识别已实现（覆盖 86.7%），字段级解析待补 |
| **建图与本体** | **B** | **已完成**，含量化评测与阈值标定 |
| 检索 | C | 仅设计。**意图路由是当前最大的用户可见缺口** |
| 结构化注入与生成 | D | 仅设计，无引用校验 |
| 评测与集成 | E | 抽取侧评测已有；检索/生成侧未做 |

实测对比（全新环境、按金标事实评分）：单药属性题上新图 3/3 且全部带页码，
旧链路 0/3；但总论题上新图 1/2、旧纯向量 2/2——因为总论知识不构成实体关系，
不在图里。**这正是需要意图路由的原因**，详见
[已知局限 §1](待办与注意事项/已知局限.md)。

## 数据与合规

请仅在课程实践和你有权使用的范围内处理教材。上传 GitHub 前请确认：

- `.env` 已被 ignore，且其中没有真实 `LLM_TOKEN`；
- `dist/*.zip`、PDF、`.cache/`、`.venv/`、`origdata/` 不会被提交；
- **数据包内含教材 OCR 全文**。公开仓库的 Release 附件同样是公开可下载的，
  发布前请确认分发范围符合授权；必要时把仓库转为私有，或分发去掉
  `Chunk.text` 的精简版。
