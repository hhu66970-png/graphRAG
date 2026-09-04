# dist/

本目录用于存放可离线分发的知识图谱压缩包（含节点、关系、`Chunk.embedding` 向量）。

## 生成（本机已有 Neo4j 数据时）

```powershell
.\export_kg_bundle.ps1
```

输出：`dist/zhongyaoxue-kg-bundle.zip`（约数 MB 到数十 MB，不建议直接提交到 Git 仓库，可用 GitHub Releases / 网盘分享）。

## 他人本机导入

1. 安装依赖并配置 `.env`（Neo4j + 可选 LLM）
2. 启动 Neo4j（`docker compose up -d neo4j` 或本机 Neo4j）
3. 将 zip 放到 `dist/` 后执行：

```powershell
.\import_kg_bundle.ps1 -Clear
.\start_backend.ps1
```

导入后会重建 `text_embeddings` 向量索引与 `text_fulltext` 全文索引。

问答时的查询向量仍需要本机 `BAAI/bge-small-zh-v1.5`（或改用外部 Embedding API）；图谱里已经带好的是**文档块向量**，不必再跑全书 LLM 抽取。
