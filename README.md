# 《中药学》GraphRAG 知识图谱问答系统

本项目以老师的 [tjudb/2026graphrag](https://gitee.com/tjudb/2026graphrag) 为基线，保留其“文本分块 → LLM 抽取实体关系 → Neo4j 图谱与向量索引 → Flask API → 图谱可视化问答”的主流程，并将语料与抽取模式改为《中药学（新世纪第五版）》。

> 原教材是 502 页扫描 PDF，没有文字层。仓库因此增加了可断点续跑的本地中文 OCR 流程。原始 PDF、OCR 全文、密钥和数据库数据均不会被 Git 跟踪。

## 1. 项目结构

```text
backend/api.py                         Flask API 与 GraphRAG 问答
frontend/index.html                    中药学知识图谱前端
graphragexpr/extract/corpus.py         页码感知的 Markdown 分块
graphragexpr/extract/build_kg_dyn.py   中药领域实体/关系抽取与 Neo4j 入库
graphragexpr/extract/create_index.py   向量索引与全文索引
scripts/preprocess_pdf.py              扫描 PDF OCR → Markdown
origdata/                              本地语料目录（默认不提交）
tests/                                 单元测试与可选 Neo4j 集成测试
```

## 2. 安装

推荐 Python 3.12。首次安装：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

如果使用 Anaconda，也可以创建独立环境，避免与已有课程环境混装：

```powershell
conda create -n zhongyao_graphrag python=3.12 -y
conda activate zhongyao_graphrag
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

当前配置过的本机已经有 `zhongyao_graphrag` 环境。仓库内的 PowerShell 脚本会自动
定位该环境，不依赖 PATH 中排在最前面的 Python；Neo4j 安装在
`D:\neo4j\neo4j-community-5.26.30`，浏览器管理界面为 <http://localhost:7474>。

编辑 `.env`，至少填写 Neo4j 和 LLM 配置。默认使用本地中文模型
`BAAI/bge-small-zh-v1.5` 生成 512 维语义向量，首次运行会把模型下载到
`D:\ai-models\huggingface`。如需改用兼容 OpenAI Embeddings API 的服务，将
`EMBED_PROVIDER=external` 并填写 `EMBED_ENDPOINT`、`EMBED_MODEL`、`EMBED_TOKEN`。

如本机没有 Neo4j，可使用 Docker：

```powershell
docker compose up -d neo4j
```

## 3. PDF 转语料

先做小范围实践（PDF 第 28–50 页，约对应教材正文开头）：

```powershell
python scripts\preprocess_pdf.py `
  --pdf "D:\你的路径\中药学.pdf" `
  --output origdata\zhongyaoxue.md `
  --start-page 28 `
  --end-page 50
```

确认效果后处理全书：

```powershell
python scripts\preprocess_pdf.py --pdf "D:\你的路径\中药学.pdf" --output origdata\zhongyaoxue.md
```

OCR 会把每页结果缓存在 `origdata/.ocr-pages/`，中断后执行同一命令会跳过已完成页面。若要重做某页，删除对应的 `page-XXXX.md` 后再次运行。

## 4. 构建知识图谱

默认只处理前 30 个文本块，便于第一次控制时间和模型费用；可在 `.env` 中修改 `MAX_CHUNKS`，设为 `0` 表示处理全部。

```powershell
.\build_kg_dyn.ps1
```

脚本会：

1. 读取 `origdata/zhongyaoxue.md` 并按页码、章节、句号分块；
2. 抽取中药、功效、主治、性味、归经、用法用量、禁忌、配伍、药材来源、炮制等实体；
3. 写入 `Document`、`Chunk` 和领域实体节点；
4. 创建 `text_embeddings` 向量索引及 `text_fulltext` 全文索引。

默认会清空当前 Neo4j 数据库。若不希望清空，将 `.env` 中的 `CLEAR_DATABASE=false`。

如果知识图谱已经构建，只想更换向量模型或重新生成向量，不必再次调用 LLM：

```powershell
.\rebuild_embeddings.ps1
```

## 5. 启动问答界面

```powershell
.\start_backend.ps1
```

浏览器打开 <http://localhost:5000>。可尝试：

- “什么是中药？中药材、中药饮片和中成药有什么区别？”
- “影响中药贮藏变异的外界因素有哪些？”
- “人参的性味、归经、功效和注意事项是什么？”（需先 OCR 并构建到对应页）

界面默认使用“中文混合向量+图谱”模式：融合中文 BM25 与 BGE 语义向量排序，再仅展开
命中文本块内部的图谱关系。下拉框中的纯向量、向量图遍历和 Neo4j 混合检索模式保留用于
课程对比实验。

## 6. 测试

```powershell
.\run_tests.ps1
```

默认只运行无需外部服务的单元测试。运行 Neo4j 集成测试：

```powershell
python -m pytest -m integration
```

## 7. 把已建好的图谱 / 向量分享给别人（无需云服务器）

本项目的向量写在 Neo4j 的 `Chunk.embedding` 上，和图谱、索引在同一库里。推荐打包成离线 zip：

### 你这边导出

```powershell
.\export_kg_bundle.ps1
```

生成 `dist/zhongyaoxue-kg-bundle.zip`（当前约数 MB，含全部节点、关系与文本块向量）。

分享方式建议：

- GitHub **Releases** 附件（不要把大 zip 直接推进源码仓库）
- 网盘 / U 盘 / 局域网拷贝

### 别人本机导入

对方只需：Python 环境、Neo4j、本仓库代码、该 zip。不必重新 OCR，也不必再调用 LLM 抽全书。

```powershell
Copy-Item .env.example .env   # 填写 Neo4j；问答还需要 LLM_TOKEN
docker compose up -d neo4j    # 或使用已有 Neo4j
.\import_kg_bundle.ps1 -Clear
.\start_backend.ps1
```

说明：

| 内容 | 是否包含在 zip | 对方是否还要准备 |
|------|----------------|------------------|
| 实体 / 关系 / Chunk 文本 | 是 | 否 |
| Chunk 向量（检索用） | 是 | 否 |
| 向量 / 全文索引 | 导入脚本会重建 | 否 |
| 查询用 BGE 模型 | 否 | 首次运行会下载，或自备缓存目录 |
| DeepSeek `LLM_TOKEN` | 否（切勿分享密钥） | 需要自己的密钥才能生成回答 |
| OCR 全文 `zhongyaoxue.md` | 否（默认不进 Git） | 仅在需要从头重建时才要 |

更细的说明见 `dist/README.md`。

## 与老师仓库的对应关系

| 老师仓库 | 本项目 |
|---|---|
| `origdata/orig.txt` 红楼梦文本 | `origdata/zhongyaoxue.md` OCR 教材文本 |
| 人物、地点、物品、概念 | 中药、功效、主治、性味、归经等领域实体 |
| `sample_data_dyn.py` 按回目取样 | `corpus.py` 按页码与标题取样 |
| `build_kg_dyn.py` | 同名脚本，增加缓存、来源页码、合法标签和失败重试支持 |
| Neo4j 向量/全文检索 | 保持一致 |
| Flask + D3/vis 前端 | 保持一致，改为中药学界面与提问示例 |

## 数据与合规

请仅在课程实践和你有权使用的范围内处理教材。不要把原始 PDF、OCR 全文、`.env`、Neo4j 数据目录或模型密钥提交到公开仓库。上传 GitHub 前请确认：

- `.env` 已被 ignore，且其中没有真实 `LLM_TOKEN`
- `dist/*.zip`、PDF、`.cache/`、`.venv/` 不会被提交
- 教材 OCR 与图谱包仅在授权同学/课程范围内分发
