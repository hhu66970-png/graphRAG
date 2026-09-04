# dist/

本目录存放可离线分发的知识图谱数据包（含节点、关系与 `Chunk.embedding` 向量）。
`dist/*.zip` 已在 `.gitignore` 中，**不要直接提交进源码仓库**——数据走 Release，
代码走 git，两条路分开。

## 生成（本机已有 Neo4j 数据时）

```bash
python scripts/export_kg_bundle.py --active-only
```

`--active-only` 只导出当前 `ACTIVE` 的 IngestRun，排除已废弃 run 的 Assertion/Entity；
不加该参数则导出全库（体积明显更大且含历史版本）。
Windows 上也可用 `.\export_kg_bundle.ps1`（等价于不带参数的全库导出）。

输出 `dist/zhongyaoxue-kg-bundle.zip`，通过 GitHub **Releases** 附件或网盘分发。

## 他人本机导入

对方只需：Python 3.12 环境、Neo4j 5.26、本仓库代码与该 zip。
**不必重新 OCR，也不必再调用 LLM 抽全书。**

```bash
cp .env.example .env          # 填写 Neo4j；问答还需要自己的 LLM_TOKEN
docker compose up -d neo4j
python scripts/import_kg_bundle.py --clear
python backend/api.py         # 或 .\start_backend.ps1
```

导入脚本会重建 `text_embeddings` 向量索引与 `text_fulltext` 全文索引。
建议再补一步，建立 5 个唯一约束并把全文索引换成 `cjk` 分析器：

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); from graphragexpr.storage.neo4j_store import Neo4jStore; from graphragexpr.storage.indexes import ensure_schema; s=Neo4jStore(); print(ensure_schema(s, embedding_dim=512, rebuild_fulltext=True)); s.close()"
```

## 包内容（v2.0.0）

| 内容 | 数量 | 说明 |
|---|---:|---|
| Chunk（含 512 维向量） | 870 | 向量检索与证据溯源 |
| Entity（规范实体） | 9,243 | v2 类型化查询 |
| Assertion（证据断言） | 22,211 | 逐字原句 + 页码 |
| 类型化 Serving 边 | 16,681 | 属性卡片、反向检索 |
| 旧图实体 | 11,107 | 前端四种对照检索模式的数据来源 |
| IngestRun | 1（ACTIVE） | 历史 run 已排除 |

导入后可用 `python scripts/verify_kg_v2.py --herb 麻黄` 验证是否可用。

## 不在包内的东西

| 内容 | 对方需要自己准备 |
|---|---|
| 查询用 BGE 模型 | 首次运行自动下载，或配置 `EMBED_CACHE_DIR` 指向已有缓存 |
| `LLM_TOKEN` | **必须用自己的密钥**，切勿分享 |
| OCR 全文 `zhongyaoxue.md` | 仅在需要从头重建图谱时才需要（Chunk 文本已在包内） |
| 抽取缓存 `.cache/` | 仅影响重建成本，不影响使用 |

## 合规提醒

数据包内的 `Chunk.text` 是教材 OCR 全文。**公开仓库的 Release 附件同样是公开
可下载的**，发布前请确认分发范围符合授权；必要时把仓库转为私有，或分发一个
去掉 `Chunk.text` 的精简版。
