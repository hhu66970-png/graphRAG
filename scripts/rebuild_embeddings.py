"""Re-embed existing Neo4j Chunk nodes without repeating LLM graph extraction."""

from __future__ import annotations

import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "graphragexpr"))
load_dotenv(ROOT / ".env")

from external_embedder import ExternalEmbedder  # noqa: E402


def main() -> None:
    dimension = int(os.getenv("EMBED_DIM", "512"))
    embedder = ExternalEmbedder(dimension=dimension)
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URL"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
    )
    database = os.getenv("NEO4J_DB") or None

    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            chunks = [
                dict(record)
                for record in session.run(
                    """
                    MATCH (chunk:Chunk)
                    RETURN elementId(chunk) AS id, chunk.text AS text, chunk.index AS index
                    ORDER BY chunk.index
                    """
                )
            ]
        if not chunks:
            raise RuntimeError("Neo4j 中没有 Chunk 节点，请先构建知识图谱")

        print(f"正在为 {len(chunks)} 个文本块生成真实语义向量...")
        vectors = embedder.embed_documents(chunk["text"] for chunk in chunks)
        rows = [
            {
                "id": chunk["id"],
                "embedding": vector,
                "model": embedder.model_name or embedder.provider,
                "dimension": dimension,
            }
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        with driver.session(database=database) as session:
            session.run("DROP INDEX text_embeddings IF EXISTS").consume()
            session.run(
                """
                UNWIND $rows AS row
                MATCH (chunk:Chunk) WHERE elementId(chunk) = row.id
                SET chunk.embedding = row.embedding,
                    chunk.embedding_model = row.model,
                    chunk.embedding_dimension = row.dimension
                """,
                rows=rows,
            ).consume()

        create_vector_index(
            driver,
            name="text_embeddings",
            label="Chunk",
            embedding_property="embedding",
            dimensions=dimension,
            similarity_fn="cosine",
        )
        with driver.session(database=database) as session:
            session.run(
                """
                CREATE FULLTEXT INDEX text_fulltext IF NOT EXISTS
                FOR (chunk:Chunk) ON EACH [chunk.text]
                """
            ).consume()
            session.run("CALL db.awaitIndexes(300)").consume()
            record = session.run(
                """
                SHOW INDEXES YIELD name, type, state
                WHERE name IN ['text_embeddings', 'text_fulltext']
                RETURN collect({name: name, type: type, state: state}) AS indexes
                """
            ).single()
        print(f"✓ 已更新 {len(rows)} 个文本块；索引状态：{record['indexes']}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
