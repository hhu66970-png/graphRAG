"""Neo4j 约束与索引。

Entity、Chunk、Assertion、Document 和 IngestRun 使用各自的稳定 ID 作为唯一键。
实体名称只建普通索引，允许不同类型的实体同名。
向量索引维度与运行时模型维度不一致时抛出异常。
"""

from __future__ import annotations

from .neo4j_store import Neo4jStore

PUBLICATION_LOCK_CONSTRAINT = (
    "CREATE CONSTRAINT publication_name IF NOT EXISTS "
    "FOR (p:GraphPublication) REQUIRE p.name IS UNIQUE"
)

CONSTRAINTS: tuple[str, ...] = (
    PUBLICATION_LOCK_CONSTRAINT,
    "CREATE CONSTRAINT entity_canonical_id IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE",
    "CREATE CONSTRAINT chunk_chunk_id IF NOT EXISTS "
    "FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT assertion_assertion_id IF NOT EXISTS "
    "FOR (a:Assertion) REQUIRE a.assertion_id IS UNIQUE",
    "CREATE CONSTRAINT document_document_id IF NOT EXISTS "
    "FOR (d:Document) REQUIRE d.document_id IS UNIQUE",
    "CREATE CONSTRAINT run_run_id IF NOT EXISTS "
    "FOR (r:IngestRun) REQUIRE r.run_id IS UNIQUE",
)

# 名称类：只建普通索引，用于召回与展示，不承担身份
INDEXES: tuple[str, ...] = (
    "CREATE INDEX entity_normalized_name IF NOT EXISTS "
    "FOR (e:Entity) ON (e.normalized_name)",
    "CREATE INDEX entity_type_name IF NOT EXISTS "
    "FOR (e:Entity) ON (e.entity_type, e.normalized_name)",
    "CREATE INDEX assertion_fact_key IF NOT EXISTS "
    "FOR (a:Assertion) ON (a.fact_key)",
    "CREATE INDEX assertion_run_status IF NOT EXISTS "
    "FOR (a:Assertion) ON (a.run_id, a.validation_status)",
    "CREATE INDEX chunk_entry IF NOT EXISTS FOR (c:Chunk) ON (c.entry_id)",
)

VECTOR_INDEX = """
CREATE VECTOR INDEX {name} IF NOT EXISTS
FOR (n:{label}) ON (n.{property})
OPTIONS {{indexConfig: {{
  `vector.dimensions`: {dimensions},
  `vector.similarity_function`: 'cosine'
}}}}
"""

FULLTEXT_DROP = "DROP INDEX {name} IF EXISTS"
FULLTEXT_INDEX = """
CREATE FULLTEXT INDEX {name}
FOR (n:{label}) ON EACH [n.{property}]
OPTIONS {{indexConfig: {{`fulltext.analyzer`: '{analyzer}'}}}}
"""


def available_analyzers(store: Neo4jStore) -> set[str]:
    """列出 Neo4j 可用的全文分析器（含 cjk）。"""
    rows = store.run("CALL db.index.fulltext.listAvailableAnalyzers() YIELD analyzer RETURN analyzer")
    return {row["analyzer"] for row in rows}


def ensure_schema(
    store: Neo4jStore,
    *,
    embedding_dim: int,
    fulltext_analyzer: str = "cjk",
    rebuild_fulltext: bool = True,
) -> dict:
    """Create all constraints and indexes. Idempotent.

    Args:
        store: 连接句柄。
        embedding_dim: 运行时向量模型维度；写入向量索引配置。
        fulltext_analyzer: 中文需要 cjk；目标实例不支持时回退到 standard 并告警。
        rebuild_fulltext: 分析器只能在创建时指定，`IF NOT EXISTS` 不会修改既有索引，
            因此需要先 DROP 才能真正切换分析器。

    Returns:
        {"constraints": n, "indexes": n, "fulltext_analyzer": 实际使用的分析器}

    Side effects: 修改 Neo4j schema。
    """
    for statement in CONSTRAINTS:
        store.run(statement)
    for statement in INDEXES:
        store.run(statement)

    analyzer = fulltext_analyzer
    if analyzer not in available_analyzers(store):
        analyzer = "standard"

    store.run(VECTOR_INDEX.format(
        name="text_embeddings", label="Chunk", property="embedding", dimensions=int(embedding_dim)
    ))
    if rebuild_fulltext:
        store.run(FULLTEXT_DROP.format(name="text_fulltext"))
        store.run(FULLTEXT_INDEX.format(
            name="text_fulltext", label="Chunk", property="text", analyzer=analyzer
        ))
    store.run("CALL db.awaitIndexes(600)")
    return {
        "constraints": len(CONSTRAINTS),
        "indexes": len(INDEXES),
        "fulltext_analyzer": analyzer,
    }


def check_vector_dimension(store: Neo4jStore, *, index_name: str, expected: int) -> None:
    """Fail loudly when the index dimension disagrees with the runtime model."""
    rows = store.run(
        "SHOW INDEXES YIELD name, options WHERE name = $name RETURN options",
        name=index_name,
    )
    if not rows:
        raise RuntimeError(f"向量索引 {index_name} 不存在")
    config = (rows[0].get("options") or {}).get("indexConfig", {})
    actual = config.get("vector.dimensions")
    if actual is not None and int(actual) != int(expected):
        raise RuntimeError(
            f"向量索引维度不一致：索引 {actual}，运行时模型 {expected}。"
            "请使索引维度与嵌入模型保持一致。"
        )
