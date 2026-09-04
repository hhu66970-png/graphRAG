# rag_vector.py
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorRetriever
from neo4j_graphrag.generation import GraphRAG
from external_embedder import ExternalEmbedder

load_dotenv()

def vector_rag_search(query_text="中药材、中药饮片和中成药有什么区别？"):
    """Perform basic vector-based RAG search"""
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
        return

    print("正在初始化向量检索器...")

    embedder = ExternalEmbedder(dimension=int(os.getenv("EMBED_DIM", "1536")))

    api_key = os.getenv("LLM_TOKEN")
    base_url = os.getenv("LLM_ENDPOINT", "https://api.deepseek.com")

    llm = OpenAILLM(
        model_name=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=base_url,
        model_params={"temperature": 0}
    )

    vector_retriever = VectorRetriever(
        driver,
        index_name="text_embeddings",
        embedder=embedder,
        return_properties=["text"],
    )

    rag = GraphRAG(retriever=vector_retriever, llm=llm)

    print(f"\n问题: {query_text}")
    print("正在检索并生成答案...\n")

    response = rag.search(
        query_text=query_text,
        retriever_config={"top_k": 5},
    )

    print("=" * 60)
    print("基于向量检索的 RAG 回答:")
    print("=" * 60)
    print(response.answer)
    print("=" * 60)

    driver.close()

if __name__ == "__main__":
    vector_rag_search()
