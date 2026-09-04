# rag_graph.py
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.generation import GraphRAG, RagTemplate
from external_embedder import ExternalEmbedder

load_dotenv()

def graph_rag_search(query_text="哪些药物通过抑制环氧合酶发挥作用？它们分别由谁研发或发明？"):
    """Perform GraphRAG search with graph traversal"""
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

    print("正在初始化 GraphRAG 检索器...")

    embedder = ExternalEmbedder(dimension=int(os.getenv("EMBED_DIM", "1536")))

    api_key = os.getenv("LLM_TOKEN")
    base_url = os.getenv("LLM_ENDPOINT", "https://api.deepseek.com")

    llm = OpenAILLM(
        model_name=os.getenv("LLM_MODEL", "deepseek-chat"),
        api_key=api_key,
        base_url=base_url,
        model_params={"temperature": 0}
    )

    # 检索命中的 chunk 为 node，向外扩展 1~2 跳收集实体关系
    # 不使用 APOC 函数，用纯 Cypher 实现
    retrieval_query = """
    // 1) 从命中的文本块出发，找到关联实体并向外扩展1-2跳
    WITH node AS chunk
    MATCH (chunk)<-[:FROM_CHUNK]-(entity)-[rel*1..2]-(neighbor)
    WITH chunk, rel
    UNWIND rel AS r
    // 2) 收集文本与三元组
    WITH collect(DISTINCT chunk.text) AS texts,
         collect(DISTINCT
           coalesce(startNode(r).name, '') + ' -[' + type(r) + ']-> ' +
           coalesce(endNode(r).name, '')
         ) AS triples
    // 使用 reduce 代替 apoc.text.join
    WITH reduce(s = '', text IN texts | s + CASE WHEN s = '' THEN text ELSE '\n---\n' + text END) AS joined_texts,
         reduce(s = '', triple IN triples | s + CASE WHEN s = '' THEN triple ELSE '\n' + triple END) AS joined_triples
    RETURN '=== 相关文本 ===\n' + joined_texts +
           '\n\n=== 相关知识 ===\n' + joined_triples AS info
    """

    graph_retriever = VectorCypherRetriever(
        driver,
        index_name="text_embeddings",
        embedder=embedder,
        retrieval_query=retrieval_query,
    )

    rag_template = RagTemplate(
        template="""你是一个严谨的中药学教材问答助手。请仅根据以下上下文回答问题，
上下文包括相关教材片段和知识图谱三元组。如果上下文不足以回答，请如实说明，不得补造药效、剂量或禁忌。

# 问题：
{query_text}

# 上下文：
{context}

# 回答：
""",
        expected_inputs=["query_text", "context"],
    )

    graph_rag = GraphRAG(retriever=graph_retriever, llm=llm, prompt_template=rag_template)

    print(f"\n问题: {query_text}")
    print("正在使用图遍历检索并生成答案...\n")

    response = graph_rag.search(
        query_text=query_text,
        retriever_config={"top_k": 5},
    )

    print("=" * 60)
    print("基于图遍历的 GraphRAG 回答:")
    print("=" * 60)
    print(response.answer)
    print("=" * 60)

    driver.close()

if __name__ == "__main__":
    graph_rag_search()
