# 中药学 GraphRAG

基于《中药学（新世纪第五版）》的知识图谱问答系统。项目从扫描教材提取文本，
用受控本体构建带原文和页码的证据图谱，支持药物属性查询、功效反查、多药比较和关系路径查询。
剂量、禁忌与配伍禁忌问题通过独立的条件证据核验流程回答。

项目基于 [tjudb/2026graphrag](https://gitee.com/tjudb/2026graphrag) 的课程示例扩展，
后端使用 Flask 和 Neo4j，前端使用 D3 展示检索子图。

## 运行环境

- Python 3.11 或 3.12
- Neo4j 5.26（仓库提供 Docker Compose 配置）
- 兼容 OpenAI Chat Completions 的模型接口
- 默认嵌入模型为 `BAAI/bge-small-zh-v1.5`，512 维，首次使用需要下载

所有命令在仓库根目录执行。

```bash
python -m venv .venv
```

macOS / Linux：

```bash
source .venv/bin/activate
cp .env.example .env
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
Copy-Item .env.example .env
```

安装依赖并启动数据库：

```bash
python -m pip install -r requirements.txt
docker compose up -d neo4j
```

在 `.env` 中填写 `NEO4J_URL`、`NEO4J_USER`、`NEO4J_PASSWORD`、`LLM_ENDPOINT`、
`LLM_MODEL` 和 `LLM_TOKEN`。示例数据库密码与 Compose 的本地配置一致。
`EMBED_CACHE_DIR` 可留空；指定时使用绝对路径，不支持 `~` 展开。

只使用外部嵌入接口时，可以安装 `requirements/base.txt`，并配置
`EMBED_PROVIDER=external` 及 `EMBED_ENDPOINT`、`EMBED_MODEL`、`EMBED_TOKEN`。
`EMBED_DIM` 需与数据库索引一致。`hash` 模式用于测试，没有语义检索能力。

## 准备数据

### 导入已构建的数据包

将数据包放到 `dist/zhongyaoxue-kg-bundle.zip`，在空数据库中执行：

```bash
python scripts/import_kg_bundle.py
```

也可用 `--bundle /path/to/bundle.zip` 指定文件。需要替换整个数据库时才加 `--clear`，
该参数会删除目标库中的全部数据。

导入脚本会重建向量和全文索引。补齐约束并使用中文全文分析器：

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from graphragexpr.storage.neo4j_store import Neo4jStore; from graphragexpr.storage.indexes import ensure_schema; s=Neo4jStore(); print(ensure_schema(s, embedding_dim=512, rebuild_fulltext=True)); s.close()"
```

此处 `512` 对应默认 BGE 模型；其他模型需使用数据包的实际维度。

### 从教材构建

仓库不附带教材 PDF、OCR 全文或数据库。准备教材后，先完成 OCR：

```bash
python scripts/preprocess_pdf.py --pdf /path/to/中药学.pdf --output origdata/zhongyaoxue.md
```

可用 `--start-page`、`--end-page` 试跑。页面缓存位于 `origdata/.ocr-pages/`，
中断后重跑会跳过已有结果。

**新建图脚本读取 Neo4j 中已有的 Chunk，不直接读取 Markdown。**
首次从空库构建，需要先用基线管线写入文本块、向量和对照图：

```bash
python graphragexpr/extract/build_kg_dyn.py
python graphragexpr/extract/create_index.py
python scripts/build_kg_v2.py --workers 6 --activate
python scripts/verify_kg_v2.py --herb 麻黄
```

`.env.example` 中 `MAX_CHUNKS=30` 用于试跑，全书构建时改为 `0`。
保留 `CLEAR_DATABASE=false`；旧管线在该值为 `true` 时会清空数据库，包括已有的新图。
已有文本块或导入数据包后，可直接运行 `build_kg_v2.py`。

新管线按 `IngestRun` 批次写入，检查通过后发布为 `ACTIVE`。
抽取缓存位于 `.cache/kg_v2/`，更改提示词或本体时同步更新
`graphragexpr/config/settings.py` 中的版本号。

## 启动问答

```bash
python backend/api.py
```

打开 [本地问答页面](http://localhost:5000)。Windows 也可执行 `.\start_backend.ps1`，
加 `-Detached` 可在后台运行。

当前页面和 `/api/graphrag` 支持三种模式：

| 模式 | 用途 |
|---|---|
| `auto` | 根据意图读取证据图谱；总论问题回退到文本检索 |
| `keyword_graph` | 关键词定位后沿实体关系补充相关文本块 |
| `vector` | 按语义向量检索文本块 |

可试问“人参的性味、归经和功效是什么？”、“比较人参和黄芪的功效异同。”。
高风险问题在三种模式下均进入统一核验流程，最终证据来源标记为
`high_risk_assertion`。未配置 `LLM_TOKEN` 时，`auto` 的普通问题只返回结构化证据。

## 项目结构

```text
backend/api.py             HTTP 接口、检索与回答编排
frontend/index.html        页面结构
frontend/assets/           样式、交互脚本与图片
graphragexpr/
  config/                  阈值、检索预算和消融开关
  domain/                  本体、数据模型与稳定 ID
  ingest/                  药物条目识别
  extract/                 抽取、证据定位和关系校验；含旧管线
  linking/                 实体消歧与别名
  storage/                 断言读写、索引、Serving 投影与批次发布
  retrieval/               意图路由、图谱查询与证据包
  highrisk/                条件匹配、剂量与禁忌核验、回答组织
  evaluation/              抽取、检索与高风险评测
scripts/                   OCR、构建、评测、数据包导入导出
requirements/              基础运行依赖与测试依赖
eval/                      检索题集与高风险题集
tests/                     单元、接口、前端渲染与数据库集成测试
origdata/                  本机语料，内容不提交到 Git
```

数据模型与查询流程见 [架构说明](docs/architecture.md)。

## 测试与评测

离线测试只需基础依赖；前端渲染测试还需要 Node.js：

```bash
python -m pip install -r requirements/dev.txt
python -m pytest -m "not integration"
python scripts/eval_retrieval.py --validate-only
python scripts/eval_high_risk.py --validate-only
```

Windows 的 `.\run_tests.ps1` 运行同一组离线测试。
数据库集成测试使用 `python -m pytest -m integration`，需配置专用 Neo4j 实例；
旧管线测试还需完整的 `requirements.txt`。需要空库的测试默认跳过非空数据库，
设置 `ALLOW_DESTRUCTIVE_TESTS=true` 会允许测试清库。

评测口径与历史结果见 [评测说明](docs/evaluation.md)。
检索题集有 120 题；高风险题集有 14 题，仍待人工仲裁。
普通生成回答尚无逐条结论核验，排序文本块的 `Recall@5` 也尚未实现。
这些限制与已实现的高风险条件核验是两回事。

## 数据包导出

```bash
python scripts/export_kg_bundle.py --active-only
```

默认输出 `dist/zhongyaoxue-kg-bundle.zip`，包含文本块及向量、实体、断言与图关系，
不含模型权重和接口密钥。包内文本来自教材 OCR，分发时需确认教材使用权限。
`.env`、语料、缓存、数据库和导出包均已加入 Git 忽略规则。
