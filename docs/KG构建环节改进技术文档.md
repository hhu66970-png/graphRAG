# 知识图谱构建环节改进技术文档（对应设计文档 §2.1.1–2.1.3）

> 依据：`SKILL.md`（第一规范）+ `GraphRAG中药学_三部分可行性评审与总体系统设计.md` §2.1
> 实施范围：建图环节（ingest 之后、检索之前）。检索、结构化注入、生成不在本轮。
> 代码基线：`756a36d`，Neo4j 5.26.30，Python 3.12.13，本地 BGE-small-zh-v1.5（512 维）

---

## 1. 改动前的问题（全部在现网图谱上实测，可复现）

用 `scripts/kg_v2_demo.py --audit` 可随时复现下列数字：

| 指标 | 实测值 | 性质 |
|---|---:|---|
| 同名跨类型的名字数 | **506** | 身份错误 |
| 笛卡尔积冗余边 | **8,623 / 38,085（22.6%）** | 数据失真 |
| 不同关系类型数 | **182**（其中 82 种出现 ≤2 次） | 本体失控 |
| 带页码证据的关系数 | **0** | 无法审计 |
| 被降级为「概念」的节点 | **982** | 静默失真 |

### 1.1 五个根因

**R1 —— 实体身份用 `name`，且写关系时不带类型**

`build_kg_dyn.py:218-223`：

```cypher
MATCH (source {name: $source}), (target {name: $target})
MERGE (source)-[r:`{rel_type}`]->(target)
```

`{name: ...}` 无标签、无索引。而「呕吐」同时存在于 `概念/功效/禁忌/主治` 四个标签下，
「白矾」存在于 `中药/化学成分/炮制/药材来源` 四个标签下。于是一条逻辑三元组被
**笛卡尔积复制成最多 4×4 条边**——实测 `活血化瘀药--[配伍]-->补益药` 重复 9 次。
这是 22.6% 冗余边的唯一来源，也解释了为什么 29,462 条逻辑事实膨胀成 38,085 条边。

**R2 —— 未知类型静默降级为「概念」**

`build_kg_dyn.py:55-57` 的 `normalize_entity_type()` 把任何不在白名单里的类型
一律返回 `"概念"`。SKILL §3.4 明令「未知类型必须记录并拒绝，不能统一降级为概念」。
后果是 982 个概念节点里混杂了真概念与被吞掉的错误抽取，且**无处查证**。

**R3 —— 关系类型是自由文本**

`normalize_relationship_type()` 只做字符清洗（`build_kg_dyn.py:60-63`），
于是「主编」「主审」「习称」「为原料」「与相关」都成了正式关系类型，共 182 种，
82 种只出现 1–2 次。检索侧无法针对关系类型做定向遍历。

**R4 —— 关系没有 provenance**

关系写入不带任何来源属性：0 条关系有 `page_start` 或 `evidence_quote`。
一旦答案依据的是图谱三元组而非原文块，就标不出 PDF 页码。对涉及剂量、毒性、
禁忌的中医场景，这违反 SKILL §3.1「证据优先」。

**R5 —— 全库删除式重建**

`CLEAR_DATABASE=true` + `MATCH (n) DETACH DELETE n`（`build_kg_dyn.py:174`）。
构建失败即无图可用，无版本、无回滚，违反 SKILL §3.7。

> 附带发现（已在代码中留档，未修改旧文件）：`tests/conftest.py:42` 的
> `clean_database` fixture 同样是 `MATCH (n) DETACH DELETE n`，
> **跑一次现有测试套件就会清空演示图谱**。本轮新增测试全部不依赖该 fixture。

---

## 2. 改了什么

新增模块严格按设计文档 §6 的目录总纲与 SKILL §5 的依赖方向组织，
**未修改任何现有文件**（旧管线保持可用，便于对照）：

```
graphragexpr/
├── config/settings.py           94 行   全部阈值/预算/版本/消融开关
├── domain/
│   ├── ontology.yaml           166 行   受控本体（唯一事实源）
│   ├── ontology.py             151 行   加载 + domain/range 校验
│   ├── models.py               149 行   Pydantic 冻结契约
│   ├── ids.py                   89 行   稳定 ID 与规范化
│   └── errors.py                54 行   领域异常
├── extract/
│   ├── evidence.py             174 行   四级证据定位 + 字符跨度
│   ├── dosage.py               113 行   剂量值对象解析
│   ├── relation_validator.py   210 行   I2 规则验证（分数语义分离）
│   ├── cross_chunk.py          126 行   I3 跨块补全
│   ├── llm_client.py           178 行   统一 LLMClient + FakeLLM
│   └── extractor.py            314 行   管线整合
├── linking/entity_resolver.py  229 行   I1 分层消歧
└── storage/
    ├── neo4j_store.py           95 行   驱动生命周期 + 批量
    ├── indexes.py              125 行   约束与索引
    ├── assertions.py           219 行   Assertion 事实源
    ├── serving_projection.py    92 行   类型化投影
    └── runs.py                 147 行   IngestRun 原子发布
scripts/kg_v2_demo.py           175 行   审计 + 真实数据演示
tests/kg/                     1,131 行   86 个测试
```

### 2.1 对应 §2.1.1「应该保留的设计」——全部落地

| 设计条目 | 落地位置 | 说明 |
|---|---|---|
| I1 实体消歧与别名归一化 | `linking/entity_resolver.py` | 五层：权威别名精确匹配 → 最长匹配 → 字符候选 → 向量候选 → 类型/上下文确认。**所有候选召回先按 entity_type 分桶**，结构上不可能跨类型合并 |
| I2 关系证据与二次校验 | `extract/relation_validator.py` | 受控关系、domain/range、逐字证据、否定与单位检查；Precision 优先 |
| I3 跨 Chunk 补全 | `extract/cross_chunk.py` | 只处理同 `entry_id` / 同 section / 相邻块，候选须重过 I2 |
| 统一 LLMClient + 消融开关 | `extract/llm_client.py`、`config/settings.py` | 超时/退避/重试/严格 JSON/token 统计/失败清单/可替换供应商；5 个消融开关对应 KG0–KG4 |

### 2.2 对应 §2.1.2「必须修改的地方」——逐条实施

**A. 拆分 `rule_score` 与 `final_score`**

`ValidationResult` 按文档给定结构实现，流程严格照做：

```
先算 rule_score
  ├─ 明确非法（domain/range 违规、自环、OCR 噪声）→ REJECTED，不调 LLM
  ├─ 明确支持（rule ≥ 0.85 且非高风险）        → ACCEPTED，llm_verdict = NOT_RUN
  └─ 灰区 + 全部高风险谓词                      → 调 verifier → 计算 final_score
```

关键点：**未进入二次校验时 `llm_verdict` 必须是 `NOT_RUN`**，绝不参与线性公式——
这正是原方案的语义污染。`test_accepts_clean_relation_without_calling_llm` 断言了
干净候选下 verifier 一次都没被调用。`final_score` 在文档与代码注释中都明确标注
「只是排序与决策支持值，不是事实正确概率」（SKILL §7.3）。

**B. 证据定位四级匹配 + 字符跨度**

`evidence.py` 完全替代脆弱的 `evidence_quote in source_text`：

| 级别 | match_mode | 场景 |
|---|---|---|
| 1 | `raw_span` | LLM 自报跨度经原文校验通过 |
| 2 | `raw_exact` | 原文精确子串 |
| 3 | `normalized_exact` | 规范化后精确匹配，**再映射回原文跨度** |
| 4 | `fuzzy` | 极小编辑距离（默认 ≤2） |
| — | `not_found` | 只能 REVIEW，不得入正式事实图 |

规范化只用于匹配，返回的 `evidence_quote` **始终是原文切片**，保留原始空格与全角标点
（SKILL §7.2）。实测中 `如紫苏叶发散风\n寒`（原文含换行）被 `normalized_exact` 正确
定位到 span=(513,522)——这正是 OCR 文本需要的鲁棒性。

**C. 剂量作为值对象**

`dosage.py` 把「煎服，3～9g；儿童减量」解析为：

```json
{"value_min": 3.0, "value_max": 9.0, "unit": "g", "route": "煎服",
 "condition": "儿童减量", "raw": "煎服，3～9g；儿童减量"}
```

存入 `Assertion.object_value`，**不建实体身份**。`raw` 永远保留。解析不出数值时
`is_well_formed()` 为假，在高风险谓词下会被扣分并强制进入二次校验。
OCR 把 `3～9` 识别成 `9～3` 时自动纠正。

**D. `entry_id` 为主依据，相邻块只是回退**

`cross_chunk.py` 的配对优先级：`entry_id` → `section_path` → 相邻块，
每个分组标注 `strategy` 字段，便于评测时区分「结构配对」与「相邻回退」各自的贡献。
`ChunkRecord` 已按 SKILL §6.1 补齐 `entry_id / parent_chunk_id / section_path /
order_in_entry / content_hash / normalized_text / ocr_quality`。

**E. 阈值必须标定**

全部阈值集中在 `KGSettings`，每一项都带注释说明**这是待标定的初始值**：

```python
# 初始值，必须经开发集标定后再冻结（设计 §2.1.2-E）；标定前不得写入报告结论。
link_auto_confirm_threshold: float = 0.94
link_candidate_threshold: float = 0.84
```

支持环境变量覆盖，`snapshot()` 供评测保存配置快照。

### 2.3 对应 §2.1.3「最终判断的四条」——全部具备

| 判断标准 | 实现 | 验证 |
|---|---|---|
| 稳定 `canonical_id` | `ids.canonical_id(type, key)`，类型作用域内哈希 | `test_canonical_id_is_type_scoped` |
| 每条事实能定位到原文 | `AssertionRecord.assert_publishable()` 在写入前硬校验 | `test_accepted_assertion_without_evidence_is_rejected_at_write` |
| 原子发布与回滚 | `runs.py`：BUILDING → ACTIVE 单语句原子切换；`mark_failed` / `rollback_to` | `test_activate_is_atomic_single_active`、`test_failed_run_keeps_previous_active` |
| 金标准集与失败清单 | `ExtractionOutcome.rejected/unknown_types/unknown_predicates`；`LLMClient.failures` | `test_counts_expose_failure_ledger` |

### 2.4 双层图落地（设计 §4）

```
事实源层：(Assertion)-[:SUBJECT]->(Entity)
                    -[:OBJECT]->(Entity)
                    -[:EVIDENCED_BY]->(Chunk)
                    -[:IN_RUN]->(IngestRun)

投影层：  (Entity)-[:HAS_EFFICACY {fact_key, best_support_score,
                                  evidence_count, evidence_ids,
                                  page_start, page_end, active_run_id}]->(Entity)
```

投影只取 `ACTIVE + ACCEPTED + POSITIVE` 的断言，可随时重建。
**NEGATIVE 极性不投影为正向边**——「不宜与藜芦同用」不会变成一条肯定的配伍边，
否定事实留在 Assertion 层供检索侧单独处理。

关系类型走 `_PROJECT_TEMPLATE.format(predicate=...)`，但 predicate **先经本体白名单
校验**再拼接，且按谓词分组——这是 SKILL §9.2 允许的唯一动态拼接方式。

---

## 3. 具体改善了什么

### 3.1 结构性改善（新旧对照）

| 维度 | 旧管线（KG0） | 新管线 | 依据 |
|---|---|---|---|
| 实体身份 | `name` + 标签，同名跨类型 506 例 | `canonical_id`，类型作用域内哈希 | SKILL §3.3 |
| 关系写入定位 | `MATCH (n {name})` 无类型 → 22.6% 冗余边 | `MATCH (:Entity {canonical_id})` 精确定位 | R1 根治 |
| 未知实体类型 | 静默降为「概念」（982 个） | 抛 `UnknownEntityTypeError`，记入 `unknown_types` | SKILL §3.4 |
| 关系类型 | 自由文本，182 种 | 本体白名单 13 个稳定 ID，未知即拒 | SKILL §3.4 |
| 证据 | 无 | 逐字原句 + 字符跨度 + match_mode + 页码 | SKILL §3.1 |
| 分数语义 | 单一 `support_score` 混用 | `rule_score` / `final_score` / `llm_verdict` 三者分离 | 设计 §2.1.2-A |
| 剂量 | 普通可合并实体（字符串） | 值对象，含 min/max/unit/route/condition | 设计 §2.1.2-C |
| 跨块补全 | 无 | entry_id 优先，标注 strategy | 设计 §2.1.2-D |
| 发布 | `DETACH DELETE` 全库重建 | IngestRun BUILDING→ACTIVE 原子切换 + 回滚 | SKILL §3.7 |
| 唯一约束 | 0 个 | 5 个（且名称类只建普通索引） | SKILL §9.3 |
| 写入方式 | 逐实体/逐关系 `session.run` | UNWIND 批量，批大小可配 | SKILL §9.2 |
| 失败可见性 | 裸 `except` 吞掉 | 拒绝清单 + 未知类型/谓词登记 + LLM 失败清单 | SKILL §12.3 |

### 3.2 冗余边为何被根治（机制层面）

新管线中一条事实的身份是 `fact_key = hash(subject_canonical_id + predicate +
object_ref + qualifiers + polarity)`，投影时按 `MERGE (s)-[e:PRED {fact_key}]->(o)`
写入。由于 `canonical_id` 已经把类型编进身份，「呕吐(主治)」与「呕吐(功效)」是两个
不同的 `canonical_id`，**结构上不可能产生笛卡尔积**。
`test_same_name_different_type_never_cross_linked` 在真实 Neo4j 上验证了这一点。

同一事实的多条证据不会变成多条边，而是 Assertion 层多条断言 + 投影边上的
`evidence_count` 与 `evidence_ids`——这既保留了多来源，又不膨胀图。

### 3.3 真实数据上的验证

在现网 Neo4j 的真实教材文本块（PDF 第 50–51 页）上跑完整管线，真实 DeepSeek 调用：

```
统计：{'assertions': 8, 'by_status': {'ACCEPTED': 6, 'REVIEW': 2},
       'rejected': 2, 'unknown_types': 0, 'unknown_predicates': 0}
证据定位模式：raw_exact 6 条，normalized_exact 2 条

✓ [ACCEPTED] HAS_EFFICACY  证据「川芎活血化瘀」 span=(538,544) mode=raw_exact 页码=50
                           rule_score=1.0 final_score=1.0
? [REVIEW]   HAS_EFFICACY  证据「神曲消食和\n胃」 span=(679,686) mode=normalized_exact
                           rule_score=0.65 final_score=0.65
被拒候选：domain_range_violation: 关系 HAS_EFFICACY 不允许 CONCEPT -> EFFICACY  ×2
```

三点值得注意：

1. **每条断言都带可回溯的原文跨度与页码**，`chunk.text[start:end]` 严格等于
   `evidence_quote`（有测试断言）；
2. **跨行证据被 `normalized_exact` 正确定位**，同时因证据质量降级为 REVIEW 而非
   直接 ACCEPTED——分级是有效的；
3. **domain/range 违规被真实拦下**：模型试图把一个 CONCEPT 当作 HAS_EFFICACY 的主语，
   旧管线会照单全收并把它降级为「概念」节点连边，新管线直接拒绝并记入失败清单。

高风险谓词的二次校验也单独验证过：证据支持时返回 `SUPPORTED`，
换成无关证据时返回 `UNSUPPORTED`，判别正确。

---

## 3.4 全量重跑（已完成）

`scripts/build_kg_v2.py --workers 6 --activate`，全书 870 块，共三轮：
第一轮因密钥余额耗尽中断，充值后凭缓存续跑；第二轮接入 I3；
第三轮修复 I1 错误合并后重建（抽取与补全全部命中缓存，仅 38 次 LLM 调用）。

### 最终产出（ACTIVE run `1788531958083-7bd6a8f87cb0`）

| 指标 | 数值 |
|---|---:|
| 断言总数 | **22,454**（ACCEPTED 21,801 / REVIEW 653） |
| 其中跨块补全 | **2,197**（占 ACCEPTED 10.1%） |
| 规范实体 | **9,297** |
| 类型化投影边 | **18,336**（12 个谓词） |
| 证据覆盖率 | **100%** |
| 失败块 | **1 / 870（0.1%）** |
| 药物条目 | 386 个，覆盖 754/870 块（86.7%） |
| 累计 LLM 消耗 | 4,911 次调用 / 约 389 万 token |

唯一失败的块是模型输出超长导致 JSON 截断，非余额问题。

### 新旧对照

| 指标 | 旧图 (KG0) | 新图 (v2) |
|---|---:|---:|
| 同名跨类型导致的错连 | 506 | **0** |
| 冗余重复边 | 8,623（22.6%） | **0（0.0%）** |
| 关系类型数 | 182 | **12**（本体白名单 13） |
| 带页码 provenance 的边 | 0 / 38,085 | **18,336 / 18,336（100%）** |
| CONCEPT 兜底节点 | 982 | 88 |
| ACCEPTED 断言证据覆盖率 | 不适用（无断言层） | **100%** |

冗余边归零不是靠去重脚本，而是 `canonical_id` 把类型编进身份后，
笛卡尔积在结构上无法发生。

### I3 跨块补全的实际贡献

754 个带 entry_id 的块中，有 **170 块**在逐块抽取阶段没有产出以条目锚点药为主语的
断言（典型是「本品味辛性凉，有发表散邪之功」这类省略主语的续写段）。补全后新增
2,208 条候选、2,197 条通过 I2 进入 ACCEPTED，集中在正确槽位：
`HAS_EFFICACY` 668、`TREATS` 656、`CONTAINS_COMPOUND` 521、`COMBINED_WITH` 420。

### 全量验证中发现并修复的两个自身缺陷

**缺陷一：I1 的「最长匹配」层做包含式自动合并，造成系统性错误归并。**

在真实全量图谱上暴露：

| 被错误合并 | 后果 |
|---|---|
| 麻黄根 ← 麻黄 | 麻黄**发汗**、麻黄根**止汗**，功效相反 |
| 黄连 ← 胡黄连 | Coptis 与 Picrorhiza，不同科属 |
| 南沙参 ← 沙参、广防已 ← 防已、乌头 ← 制乌头、牛黄 ← 人工牛黄 | 有关联但不等价 |

根因是该层允许「一方包含另一方且长度差 ≤2 即合并」，对中文药名过于宽松——
一个字的前后缀常常就换了物种或药用部位。**而设计 §2.1.1-1 的流程里本无此层**，
它要求候选必须经「类型与上下文确认」。修正：包含式匹配降级为**只产候选**，
相似度按长度比计算（麻黄/麻黄根 = 0.667，低于确认阈值 0.94 与候选阈值 0.84），
因而新建独立实体。修复后重建，实体数由 6,839 升至 9,297，抽查的 8 对全部独立。

新增 11 个回归用例（`TestNoFalseMerge`）锁定该行为。

**缺陷二：集成测试会把线上 ACTIVE run 降级。**

`activate()` 的语义是「全局只允许一个 ACTIVE」，因此会降级任何其他 ACTIVE run，
包括线上正在服务的那个。原有隔离只覆盖了按 run_id 前缀清理节点，没覆盖这个
全局不变量，结果跑一次测试线上就失去 ACTIVE。修正：新增 `isolated_active_run`
fixture，在运行期用例前后显式摘挂现网 run。

### 属性聚合实例

旧检索链路答不出「人参的性味、归经和功效」，新图里结构化存在；
修复错误合并后麻黄也恢复正常：

```
麻黄（第81-82页，跨第39/51/54/55/56/63/66/82/92 页多块聚合）
  性味为   辛、微苦，温
  归经为   归肺、膀胱经
  具有功效 发汗解表、宣肺平喘、利水消肿（23 条证据）
  来源于   草麻黄、中麻黄、木贼麻黄
  含有     麻黄碱、伪麻黄碱、甲基麻黄碱……
  配伍     桂枝、苦杏仁、石膏、甘草（麻黄汤与麻杏石甘汤组成）
```

与教材内容一致。

### 必须保留的结论边界

上表全部是**结构性指标**，不等于事实级准确率。人参的性味槽仍混入
「味苦、甘，性寒」、归经槽混入「归肺、胃经」，多来自对比句被误归属。
因此本轮能下的结论是**「身份稳了、证据齐了、结构对了」**，
**不能**下「抽取准确率提升了多少」的结论。后者需要设计 §9.1 的金标准集与
§2.1.2-E 的阈值标定，两者均未完成。

## 3.5 药物条目识别（entry_id）

跨块补全的主依据 `entry_id` 此前缺失。新增 `graphragexpr/ingest/structure.py`，
利用教材版式规律识别条目边界：

```
桂枝                      ← 药名独占一行
Guizhi（《名医别录》）      ← 拼音 +（《典籍》）：可靠的条目起始标记
本品为樟科植物肉桂……
【药性】辛、甘，温。归心、肺、膀胱经。
【功效】发汗解肌，温通经脉……
```

在全书 870 块上实测：**识别出 403 个药物条目（386 个不同 entry_id），
覆盖 754/870 块（86.7%）**，未覆盖的 116 块是前言、目录与总论章节——本就不应有条目。
**平均每条目 1.95 块**，正是「一味药的属性跨 2 个块」的典型场景；
相比之下原先可用的 `section` 粒度是每章 32 块，无法用于配对。

## 4. 测试

`.venv/bin/python -m pytest tests/kg/ -q` → **86 passed**（其中 12 项为真实 Neo4j 集成测试）。

覆盖 SKILL §14.1 要求的正常路径、空输入、非法 Schema、OCR 噪声、否定与数值、
重复与幂等、降级路径，以及 §14.3 点名的集成场景：

| SKILL §14.3 场景 | 测试 |
|---|---|
| 两个 Chunk 共同构成一条事实 | `test_cross_chunk_fact_and_projection` |
| 同名不同类型实体不得错误连边 | `test_same_name_different_type_never_cross_linked` |
| 新 Run 失败时旧 ACTIVE 仍可查询 | `test_failed_run_keeps_previous_active` |
| 无 provenance 的断言拒绝入图 | `test_accepted_assertion_without_evidence_is_rejected_at_write` |
| 写入幂等 | `test_write_is_idempotent` |

集成测试全部按 `run_id` / `document_id` 前缀隔离，只清理自己创建的节点。
实测跑完后演示图谱仍为 **11,978 节点 / 74,909 关系**，残留测试数据 0 条。

---

## 4.5 收尾轮：三个缺陷的修复与 v2 图谱接入

### 已修复

**③ 输出截断导致整块丢弃 → 抢救完整前缀。** chunk[13] 是「编写说明」，罗列几十个
「院校＋人名＋药类」，模型生成的 JSON 超长被截断。原实现整块判失败。现在
`_repair_truncated_json` 扫描括号深度，回退到**最后一个完整闭合的元素**再补收尾括号，
计入 `truncation_recoveries` 让降级可见。注意安全边界：逗号不能当切点——对象内部的
逗号后面是半个元素，从那里截断会留下缺字段的残片。修复后 **failed_chunks 由 1 降为 0**。

**② 条目名截断 → 双重校验。** 条目名要求 ≥2 个汉字（教材无单字药，单字命中必是
OCR 换行造成的截断，实测有「蓟」「黄」两例）；构建时再要求锚点必须是逐块抽取真实
产出过的 HERB 实体，否则跳过补全。本轮跳过 4 个未验证锚点。

**④ v2 图谱接入（跨模块，已与 Owner C 的工作解耦）。** 新增
`storage/serving_read.py`（建图侧 §5.5）提供属性卡片、反向检索、证据回溯；
API 只加一个 `kg_v2` 编排分支，**未改动任何既有检索模式**。完整的意图路由
（attr/reverse/intersect/compare）与结构化注入仍属 retrieval / structuring 层。

实测「人参的性味、归经和功效」：

```
性味：甘、微苦，微温（第378页）
归经：归脾、肺、心、肾经（第378页）
功效：大补元气，复脉固脱，补脾益肺，生津养血，安神益智（第378页）
子图 68 节点 / 67 边，溯源页码 [68, 378, 380]
```

与教材一致，且每条结论都带页码——这正是旧链路答不出的那道题。

### 收尾轮发现的第三个测试隔离缺陷（严重）

集成测试的 `_seed` 直接用真实药名造数据：

```python
EntityRecord(canonical_id=canonical_id("HERB", "人参"), ...)   # 与线上同一个身份
```

`canonical_id` 是 (类型, 规范名) 的**纯函数**，因此测试实体与线上实体撞同一个
canonical_id；`upsert_entities` 按 canonical_id MERGE，把线上节点的 `run_id` 改写成
`itest-`，随后 `_cleanup` 的 `DETACH DELETE` **连同真实节点及其全部边一起删除**。
实测已导致人参、藜芦、脾经、呕吐 四个真实实体消失。

修复：测试实体名一律加 `ITEST药` 前缀（教材中不可能出现），使 canonical_id 不可能
碰撞；`_cleanup` 改为 run_id 与名称前缀**双条件**才删。已重建恢复被删实体。

> 这是本轮第三个同类问题（前两个是 conftest 清库、activate 降级线上 run）。
> 共同教训：**只要测试与生产共用一个数据库，隔离就必须覆盖每一个全局命名空间**
> ——节点、全局唯一状态、以及由纯函数派生的身份。更稳妥的做法是给测试单独起一个
> Neo4j 实例，这属于 Owner E 的 CI 工作。

### 最终产出（ACTIVE run `1788538070176-b508ae1c6ded`）

| 指标 | 数值 |
|---|---:|
| 断言 | 22,211（ACCEPTED 21,558 / REVIEW 653） |
| 其中跨块补全 | 1,911 |
| 规范实体 | 9,243 |
| 类型化投影边 | 18,138（12 个谓词） |
| 证据覆盖率 | 100% |
| **失败块** | **0 / 870** |
| 跳过的未验证锚点 | 4 |
| 测试 | 137 passed / 8 skipped |

## 4.6 抽取质量量化与阈值标定

### 参照集：教材自己的结构化字段

《中药学》每味药条目都有规整的结构化字段，可**纯规则解析、不调用 LLM**：

```
【药性】辛、微苦，温。归肺、膀胱经。
【功效】发汗解表，宣肺平喘，利水消肿。
【用法用量】煎服，2～10g。
```

`graphragexpr/evaluation/reference.py` 从中构建参照集，实测覆盖
**375 味药**（性味 367、归经 366、功效 366、用法用量 318），
共 1,211 条功效取值、966 条归经取值。

**它不是什么**：这是「教材结构化字段参照集」，**不是**设计 §9.1 要求的
双人标注仲裁金标准。两点差别必须随数字一起说明：

1. 覆盖面窄：只含药性/功效/用法用量，不含主治、配伍、禁忌；
2. **Precision 被系统性低估**：图谱合理包含字段之外的正确事实（【应用】段落、
   跨章节引用），它们不在参照集里但并不算错。

因此主指标是 **Recall@field**；Precision 一律标注 ↓ 表示下界。

### 评测结果（ACTIVE run，阈值标定后）

| 字段 | 金标取值 | 图谱取值 | 命中 | Recall@field | Prec@field↓ |
|---|---:|---:|---:|---:|---:|
| 性味 | 1,099 | 1,125 | 990 | **90.1%** | 88.0% |
| 归经 | 966 | 835 | 759 | **78.6%** | 90.9% |
| 功效 | 1,211 | 2,086 | 1,043 | **86.1%** | 50.0% |
| **合计** | **3,276** | **4,046** | **2,792** | **85.2%** | **69.0%** |

功效的 Prec↓ 只有 50%，正是上面说的低估：图谱从【应用】段落额外抽出大量
正确功效（如麻黄「宣肺利水」「散寒通滞」），它们不在【功效】字段里。

完全召回率（该药字段取值一条不漏）：性味 84.2%、归经 68.9%、功效 80.9%。
375 味药中 19 味在图谱里没有任何属性边。

### 一次口径错误的修正（值得记录）

首轮评测性味 Recall 只有 35.5%，漏检样例是「麻黄→辛」「麻黄→温」——
但麻黄卡片里明明有「辛、微苦，温」。原因是**两侧粒度不一致**：教材字段拆成
单项，图谱把整串存成一个取值。这是测量口径错配，不是抽取失败。
修正为「两侧用同一套规则归一后再比」，性味 Recall 由 35.5% 升至 90.1%。

教训：指标下降时先确认量的是不是同一个东西，否则会把测量缺陷当成模型缺陷去优化。

### 阈值标定（设计 §2.1.2-E）

断言已带 `final_score`，因此可**离线扫描**接受阈值，无需重跑抽取、不调用 LLM
（`scripts/calibrate_thresholds.py`）：

| 阈值 | Recall@field | Prec@field↓ | F1↓ |
|---:|---:|---:|---:|
| 0.60（原初始值） | 85.6% | 67.2% | 75.3% |
| **0.80** | **85.2%** | **69.0%** | **76.3%** ← 最高 |
| 0.85 | 78.1% | 68.0% | 72.7% |
| 0.95 | 77.8% | 72.8% | 75.2% |

0.80 是 **Recall 明显下滑之前的最大阈值**：相比 0.60 只损失 0.4pp 召回，
换来 1.8pp 精确率；再往上到 0.85 召回骤降 7pp。建图取向是 Precision 优先
（设计 §2.1.1-2），故冻结为 **0.80**，并在 `settings.py` 注明标定依据。

应用该阈值后 ACCEPTED 由 21,558 降为 19,739，其余转入 REVIEW 等待复核——
这正是「宁缺毋滥」的取向，被拒的事实并未丢失，仍在 Assertion 层可查。

## 5. 本轮未做（非目标）

严格按 SKILL §4.7 声明：

1. **仍缺双人标注金标准集。** 阈值标定已完成（§4.6），抽取质量也有了首个量化口径
   （Recall@field 85.2%），但参照集只覆盖 性味/归经/功效 三类字段，
   且 Precision 只是下界。设计 §9.1 要求的「50–60 条目、300–500 事实、
   关键样本双人标注仲裁」仍需人工完成——主治、配伍、禁忌三类目前没有任何参照。
   **引用数字时必须连同口径一起说明，不得简化成「准确率 85%」。**
2. **条目名识别有少量截断错误。** 393 个条目名中，`蓟`（应为大蓟/小蓟）、`黄` 等
   少数被截断；错误的条目名会成为跨块补全的锚点药，把事实归给一个不存在的中药。
   修法：要求锚点名 ≥2 字且能在已抽取的 HERB 实体集中找到对应项。
3. **1 块因模型输出超长导致 JSON 截断**（`chunk[13]`）。已加「每片段最多 40 条关系」
   的约束，仍偶发；后续可对超长块做二次切分重试。
4. **未迁移旧图。** 旧图缺关系级证据，按 SKILL §3.1 只能迁为 `REVIEW` 断言，
   不能伪造成 `ACCEPTED`。全量重跑完成后，旧图可整体退役，无需迁移。
5. **未改动任何旧文件**（`build_kg_dyn.py`、`corpus.py`、`api.py` 原样保留），
   仅修复了 `tests/conftest.py` 会清空整库的破坏性 fixture。
6. **检索、结构化注入、生成**均不在本轮。现网问答接口仍读旧图，
   切到 v2 投影层属检索环节的工作。

---

## 6. 如何使用

```bash
# 审计旧图缺陷（只读，随时可跑）
.venv/bin/python scripts/kg_v2_demo.py --audit

# 在真实教材块上演示新管线（会调用 LLM）
.venv/bin/python scripts/kg_v2_demo.py --extract 大补元气 --limit 1

# 建立约束与索引（幂等）
.venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
from graphragexpr.storage.neo4j_store import Neo4jStore
from graphragexpr.storage.indexes import ensure_schema
s = Neo4jStore(); print(ensure_schema(s, embedding_dim=512, rebuild_fulltext=False)); s.close()"

# 测试
.venv/bin/python -m pytest tests/kg/ -q
```

消融实验通过环境变量控制（对应设计 §10 的 KG0–KG4）：

```bash
KG_USE_I1=false KG_USE_I2=false .venv/bin/python -m pytest tests/kg -q   # KG0
KG_USE_I3=false ...                                                       # KG2
```

---

## 7. 需要 ADR 的变更（SKILL §15）

本轮引入了以下属于契约级的决定，建议补 ADR 后再冻结：

1. 本体实体类型与关系白名单（`ontology.yaml`，`schema_version: kg-ontology-1.0.0`）；
2. Assertion 与 Serving Projection 的双层模型；
3. `canonical_id` 取代 `name` 作为实体身份；
4. 稳定 ID 构造方式与缓存键组成；
5. IngestRun 发布与回滚机制。
