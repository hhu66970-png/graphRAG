# create_index.py
import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index

load_dotenv()

def create_index():
    """Create vector and full-text indexes for Chunk nodes."""
    print("正在连接到 Neo4j 数据库...")

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URL"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
    )

    try:
        driver.verify_connectivity()
        print("✓ Neo4j 数据库连接成功")
    except Exception as e:
        print(f"✗ 无法连接到 Neo4j: {e}")
        driver.close()
        return

    print("正在创建向量索引...")

    try:
        create_vector_index(
            driver,
            name="text_embeddings",
            label="Chunk",
            embedding_property="embedding",
            dimensions=int(os.getenv("EMBED_DIM", "1536")),
            similarity_fn="cosine",
        )
        print("✓ 向量索引创建成功!")
    except Exception as e:
        if "already exists" in str(e).lower() or "equivalent" in str(e).lower():
            print("✓ 向量索引已存在")
        else:
            print(f"✗ 创建索引时出错: {e}")

    print("正在创建全文索引...")
    try:
        with driver.session(database=os.getenv("NEO4J_DB") or None) as session:
            session.run("""
                CREATE FULLTEXT INDEX text_fulltext IF NOT EXISTS
                FOR (n:Chunk) ON EACH [n.text]
            """).consume()
        print("✓ 全文索引已就绪: text_fulltext")
    finally:
        driver.close()

if __name__ == "__main__":
    create_index()
