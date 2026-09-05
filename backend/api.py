# backend/api.py
"""
Flask backend API for serving knowledge graph data to visualization
"""
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from neo4j import GraphDatabase
from openai import OpenAI

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "graphragexpr"))
from external_embedder import ExternalEmbedder

load_dotenv()

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)  # Enable CORS for frontend

# Neo4j connection
driver = GraphDatabase.driver(
    os.getenv("NEO4J_URL"),
    auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
)

# Retrieval / subgraph caps for the default keyword_graph mode
RETRIEVAL_CANDIDATE_K = 10
RETRIEVAL_TOP_K = 2
MAX_SUBGRAPH_NODES = 40
MAX_SUBGRAPH_EDGES = 60
MAX_GRAPH_TRIPLES = 40
_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')


@lru_cache(maxsize=1)
def _configured_embedder():
    """Load the local model once per backend process."""
    return ExternalEmbedder(dimension=int(os.getenv("EMBED_DIM", "512")))


def _escape_lucene(term: str) -> str:
    return _LUCENE_SPECIAL.sub(r"\\\1", term)


def _build_fulltext_query(question: str, exact_terms: list[str]) -> str:
    """Build a Lucene OR query from entity names and Chinese character bigrams."""
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        cleaned = term.strip()
        if len(cleaned) < 2 or cleaned in seen:
            return
        seen.add(cleaned)
        terms.append(_escape_lucene(cleaned))

    for name in exact_terms:
        add(name)

    compact = re.sub(r"\s+", "", question.lower())
    chinese = "".join(re.findall(r"[\u3400-\u9fff]+", compact))
    for index in range(len(chinese) - 1):
        add(chinese[index : index + 2])
        if len(terms) >= 24:
            break

    for token in re.findall(r"[a-z0-9]{2,}", compact):
        add(token)

    return " OR ".join(terms[:24]) if terms else _escape_lucene(compact[:40] or "中药")


def _chunk_record(record) -> dict:
    return {
        "id": record["id"],
        "index": record["index"],
        "text": record["text"],
        "page_start": record["page_start"],
        "page_end": record["page_end"],
        "section": record["section"],
    }


def _keyword_graph_search(
    question,
    top_k=RETRIEVAL_TOP_K,
    *,
    use_fulltext=True,
    use_vector=None,
    expand_graph=True,
):
    """Native fulltext/vector retrieval with optional same-source graph expansion."""
    database = os.getenv("NEO4J_DB") or None
    candidate_count = RETRIEVAL_CANDIDATE_K

    with driver.session(database=database) as session:
        exact_terms = [
            record["name"]
            for record in session.run(
                """
                MATCH (entity)
                WHERE (entity:中药 OR entity:方剂 OR entity:功效 OR entity:主治
                       OR entity:概念 OR entity:典籍 OR entity:性味 OR entity:归经)
                  AND entity.name IS NOT NULL
                  AND size(entity.name) >= 2
                  AND size(entity.name) <= 20
                  AND $question CONTAINS entity.name
                RETURN DISTINCT entity.name AS name
                ORDER BY size(entity.name) DESC
                LIMIT 20
                """,
                question=question,
            )
        ]

        lexical_candidates = []
        if use_fulltext:
            fulltext_query = _build_fulltext_query(question, exact_terms)
            try:
                lexical_candidates = [
                    {
                        **_chunk_record(record),
                        "score": float(record["lexical_score"] or 0.0),
                        "lexical_score": float(record["lexical_score"] or 0.0),
                        "matched_terms": [
                            term for term in exact_terms
                            if term and term in (record["text"] or "")
                        ],
                    }
                    for record in session.run(
                        """
                        CALL db.index.fulltext.queryNodes('text_fulltext', $fulltext_query)
                        YIELD node, score
                        WHERE node:Chunk AND node.text IS NOT NULL
                        RETURN elementId(node) AS id, node.index AS index,
                               node.text AS text, node.page_start AS page_start,
                               node.page_end AS page_end, node.section AS section,
                               score AS lexical_score
                        ORDER BY score DESC
                        LIMIT $candidate_count
                        """,
                        fulltext_query=fulltext_query,
                        candidate_count=candidate_count,
                    )
                ]
            except Exception as exc:
                print(f"Fulltext retrieval fallback ({exc})")
                # Targeted CONTAINS fallback; never scan every Chunk in Python.
                seed = exact_terms[:5] or [question[:8]]
                lexical_candidates = [
                    {
                        **_chunk_record(record),
                        "score": float(record["hit_count"] or 0.0),
                        "lexical_score": float(record["hit_count"] or 0.0),
                        "matched_terms": [
                            term for term in exact_terms
                            if term and term in (record["text"] or "")
                        ],
                    }
                    for record in session.run(
                        """
                        UNWIND $seeds AS seed
                        MATCH (chunk:Chunk)
                        WHERE chunk.text CONTAINS seed
                        WITH chunk, count(*) AS hit_count
                        RETURN elementId(chunk) AS id, chunk.index AS index,
                               chunk.text AS text, chunk.page_start AS page_start,
                               chunk.page_end AS page_end, chunk.section AS section,
                               hit_count
                        ORDER BY hit_count DESC, chunk.index
                        LIMIT $candidate_count
                        """,
                        seeds=seed,
                        candidate_count=candidate_count,
                    )
                ]

        vector_candidates = []
        vector_enabled = (
            (os.getenv("EMBED_PROVIDER") or "hash").lower() != "hash"
            if use_vector is None
            else use_vector
        )
        if vector_enabled:
            query_embedding = _configured_embedder().embed_query(question)
            vector_candidates = [
                {
                    "id": record["id"],
                    "vector_score": float(record["vector_score"] or 0.0),
                    **{
                        key: record[key]
                        for key in ("index", "text", "page_start", "page_end", "section")
                    },
                }
                for record in session.run(
                    """
                    CALL db.index.vector.queryNodes(
                        'text_embeddings', $candidate_count, $query_embedding
                    )
                    YIELD node, score
                    RETURN elementId(node) AS id, node.index AS index,
                           node.text AS text, node.page_start AS page_start,
                           node.page_end AS page_end, node.section AS section,
                           score AS vector_score
                    """,
                    candidate_count=candidate_count,
                    query_embedding=query_embedding,
                )
            ]

    chunks_by_id = {}
    for chunk in lexical_candidates:
        chunks_by_id[chunk["id"]] = dict(chunk)
    for chunk in vector_candidates:
        chunks_by_id.setdefault(chunk["id"], {
            "id": chunk["id"],
            "index": chunk.get("index"),
            "text": chunk.get("text"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "section": chunk.get("section"),
            "matched_terms": [],
        })

    fused = {}
    lexical_details = {chunk["id"]: chunk for chunk in lexical_candidates}
    for rank, chunk in enumerate(lexical_candidates, start=1):
        fused.setdefault(chunk["id"], 0.0)
        fused[chunk["id"]] += 1.25 / (60 + rank)
    for rank, result in enumerate(vector_candidates, start=1):
        fused.setdefault(result["id"], 0.0)
        fused[result["id"]] += 1.0 / (60 + rank)

    ranked_chunks = []
    for chunk_id, fusion_score in sorted(
        fused.items(), key=lambda item: item[1], reverse=True
    )[:top_k]:
        chunk = dict(chunks_by_id[chunk_id])
        lexical = lexical_details.get(chunk_id, {})
        vector = next(
            (item for item in vector_candidates if item["id"] == chunk_id),
            {},
        )
        chunk["score"] = fusion_score
        chunk["lexical_score"] = lexical.get("lexical_score", lexical.get("score"))
        chunk["vector_score"] = vector.get("vector_score")
        chunk["matched_terms"] = lexical.get("matched_terms", [])
        ranked_chunks.append(chunk)

    if ranked_chunks and ranked_chunks[0].get("index") is not None:
        lead_index = ranked_chunks[0]["index"]
        with driver.session(database=database) as session:
            companion_record = session.run(
                """
                MATCH (chunk:Chunk {index: $index})
                RETURN elementId(chunk) AS id, chunk.index AS index,
                       chunk.text AS text, chunk.page_start AS page_start,
                       chunk.page_end AS page_end, chunk.section AS section
                LIMIT 1
                """,
                index=lead_index + 1,
            ).single()
        if companion_record and all(
            chunk["id"] != companion_record["id"] for chunk in ranked_chunks
        ):
            companion = _chunk_record(companion_record)
            companion["score"] = ranked_chunks[0]["score"] * 0.5
            companion["lexical_score"] = None
            companion["vector_score"] = None
            companion["matched_terms"] = ["相邻文本块"]
            ranked_chunks = [ranked_chunks[0], companion, *ranked_chunks[1: max(top_k - 1, 0)]]

    chunk_ids = [chunk["id"] for chunk in ranked_chunks]

    nodes = {}
    edges = []
    triples = []
    if chunk_ids and expand_graph:
        with driver.session(database=database) as session:
            records = session.run(
                """
                UNWIND $chunk_ids AS chunk_id
                MATCH (chunk:Chunk) WHERE elementId(chunk) = chunk_id
                MATCH (entity)-[:FROM_CHUNK]->(chunk)
                WHERE NOT entity:Chunk AND NOT entity:Document
                OPTIONAL MATCH (entity)-[relationship]-(neighbor)
                WHERE NOT neighbor:Chunk AND NOT neighbor:Document
                  AND (neighbor)-[:FROM_CHUNK]->(chunk)
                RETURN DISTINCT
                       elementId(entity) AS source_id,
                       labels(entity)[0] AS source_label,
                       coalesce(entity.preferred_name, entity.name) AS source_name,
                       elementId(neighbor) AS target_id,
                       labels(neighbor)[0] AS target_label,
                       coalesce(neighbor.preferred_name, neighbor.name) AS target_name,
                       type(relationship) AS relationship_type
                LIMIT $record_limit
                """,
                chunk_ids=chunk_ids,
                record_limit=MAX_SUBGRAPH_EDGES * 2,
            )
            for record in records:
                if len(nodes) >= MAX_SUBGRAPH_NODES and len(edges) >= MAX_SUBGRAPH_EDGES:
                    break

                source_id = record["source_id"]
                if source_id not in nodes:
                    if len(nodes) >= MAX_SUBGRAPH_NODES:
                        continue
                    nodes[source_id] = {
                        "id": source_id,
                        "label": record["source_name"],
                        "type": record["source_label"],
                    }

                target_id = record["target_id"]
                relationship_type = record["relationship_type"]
                if target_id:
                    if target_id not in nodes:
                        if len(nodes) >= MAX_SUBGRAPH_NODES:
                            continue
                        nodes[target_id] = {
                            "id": target_id,
                            "label": record["target_name"],
                            "type": record["target_label"],
                        }
                    if relationship_type and len(edges) < MAX_SUBGRAPH_EDGES:
                        edge = {
                            "source": source_id,
                            "target": target_id,
                            "type": relationship_type,
                        }
                        if edge not in edges:
                            edges.append(edge)
                        triple = (
                            record["source_name"],
                            relationship_type,
                            record["target_name"],
                        )
                        if triple not in triples and len(triples) < MAX_GRAPH_TRIPLES:
                            triples.append(triple)

    return ranked_chunks, {"nodes": list(nodes.values()), "edges": edges}, triples


def _evidence_only_answer(packet):
    """Render a truthful answer when retrieval is available but no LLM is configured."""
    statements = []
    seen = set()
    for item in packet.items[:12]:
        subject = item.qualifiers.get("subject_name") or item.subject_id
        object_name = item.qualifiers.get("object_name") or item.object_id
        predicate = item.predicate or "教材证据"
        page = (
            item.page_start
            if item.page_start == item.page_end
            else f"{item.page_start}-{item.page_end}"
        )
        if subject and object_name:
            claim = f"{subject} -[{predicate}]-> {object_name}"
        elif subject:
            claim = f"{subject}：{predicate}"
        else:
            claim = predicate
        quote = " ".join(item.evidence_quote.split())
        statement = f"{claim}（PDF 第 {page} 页）：{quote}"
        if statement not in seen:
            seen.add(statement)
            statements.append(statement)
    return "未配置 LLM_TOKEN，以下为真实知识图谱检索到的结构化证据：" + "；".join(
        statements
    )


def _answer_with_auto(question):
    """Run rule-first v2 retrieval and pass only its EvidencePacket to generation."""
    from graphragexpr.domain.ontology import load_ontology
    from graphragexpr.retrieval import retrieve
    from graphragexpr.storage.neo4j_store import Neo4jStore

    store = Neo4jStore()
    try:
        result = retrieve(
            store,
            load_ontology(),
            question,
            fallback_retriever=_keyword_graph_search,
        )
    finally:
        store.close()

    packet = result.packet
    sources = [
        {
            "evidence_id": item.evidence_id,
            "chunk_id": item.chunk_id,
            "page_start": item.page_start,
            "page_end": item.page_end,
            "score": item.retrieval_score,
            "matched_terms": [item.predicate] if item.predicate else [],
            "evidence_quote": item.evidence_quote,
        }
        for item in packet.items
    ]
    if not packet.items:
        warning = packet.warnings[0] if packet.warnings else "没有检索到可定位证据"
        return (
            f"证据不足，无法基于当前知识图谱作答：{warning}。",
            result.subgraph,
            sources,
            packet,
        )

    llm_token = (os.getenv("LLM_TOKEN") or "").strip()
    if not llm_token:
        return _evidence_only_answer(packet), result.subgraph, sources, packet

    context_lines = []
    for index, item in enumerate(packet.items, start=1):
        page = (
            item.page_start
            if item.page_start == item.page_end
            else f"{item.page_start}-{item.page_end}"
        )
        relation = f"，关系 {item.predicate}" if item.predicate else ""
        path = f"，路径 {item.path}" if item.path else ""
        subject = item.qualifiers.get("subject_name") or item.subject_id or "未知主语"
        object_name = item.qualifiers.get("object_name") or item.object_id
        object_label = f"｜宾语={object_name}" if object_name else ""
        context_lines.append(
            f"[证据 {index}｜evidence_id={item.evidence_id}｜"
            f"PDF 第 {page} 页｜retrieval_score={item.retrieval_score:.3f}"
            f"｜主语={subject}{object_label}{relation}{path}]\n"
            f"{item.evidence_quote}"
        )

    retrieval_trace = json.dumps(packet.trace, ensure_ascii=False, indent=2)

    prompt = f"""请严格依据下面的《中药学》结构化检索证据回答问题。
不得使用证据外知识；关键结论必须引用对应 PDF 页码。
部分匹配不得表述为全部满足。
涉及剂量、毒性、禁忌、孕妇或儿童时，只能逐字依据证据，
并提醒以教材原文和专业人员意见为准。

# 检索意图
{packet.intent}

# 问题
{question}

# EvidencePacket
{chr(10).join(context_lines)}

# 检索 trace（交集题以 constraint_matrix.exact_match 为准）
{retrieval_trace}
"""
    client = OpenAI(api_key=llm_token, base_url=os.getenv("LLM_ENDPOINT") or None)
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": "你是严谨的中药学教材问答助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content or "模型没有返回答案"
    return answer, result.subgraph, sources, packet


def _answer_with_keyword_graph(
    question, *, use_fulltext=True, use_vector=None, expand_graph=True
):
    chunks, subgraph, triples = _keyword_graph_search(
        question,
        use_fulltext=use_fulltext,
        use_vector=use_vector,
        expand_graph=expand_graph,
    )
    if not chunks:
        raise RuntimeError("数据库中没有可检索的教材文本块")

    textbook_context = []
    for position, chunk in enumerate(chunks, start=1):
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end") or page_start
        page_label = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        textbook_context.append(
            f"[教材片段 {position}｜PDF 第 {page_label} 页]\n{chunk['text']}"
        )

    graph_context = "\n".join(
        f"- {source} --[{relationship}]--> {target}"
        for source, relationship, target in triples[:MAX_GRAPH_TRIPLES]
    ) or "- 未抽取到与命中文本块直接相关的三元组"

    prompt = f"""请严格依据下面的《中药学》教材片段回答问题。
优先使用教材原文中的定义和分类，不要因为知识图谱三元组不完整而忽略教材正文。
回答应直接、分点清楚，并在关键结论后标注对应的 PDF 页码。
如果问题涉及剂量、毒性或临床建议，提醒用户以教材原文和专业人员意见为准。

# 问题
{question}

# 命中的教材原文
{chr(10).join(textbook_context)}

# 命中文本块内部的知识图谱关系（仅作补充）
{graph_context}
"""

    client = OpenAI(
        api_key=os.getenv("LLM_TOKEN"),
        base_url=os.getenv("LLM_ENDPOINT") or None,
    )
    response = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": "你是严谨的中药学教材问答助手。"},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )
    answer = response.choices[0].message.content or "模型没有返回答案"
    sources = [
        {
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "score": round(chunk["score"], 3),
            "lexical_score": chunk.get("lexical_score"),
            "vector_score": chunk.get("vector_score"),
            "matched_terms": chunk["matched_terms"],
        }
        for chunk in chunks
    ]
    return answer, subgraph, sources


@app.route('/')
def index_page():
    """Serve the bundled visualization and question-answering UI."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        driver.verify_connectivity()
        return jsonify({"status": "ok", "message": "Connected to Neo4j"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/graph', methods=['GET'])
def get_graph():
    """Get knowledge graph entities.

    By default returns a capped preview. Pass limit=0 for the full graph
    (not recommended for the browser visualization).
    """
    raw_limit = request.args.get('limit', '300')
    try:
        limit = int(raw_limit)
    except ValueError:
        return jsonify({"error": "limit must be an integer"}), 400

    # limit <= 0 means unrestricted; positive values cap returned nodes.
    limit_clause = ""
    params = {}
    if limit > 0:
        limit_clause = "LIMIT $limit"
        params["limit"] = limit

    query = f"""
    MATCH (n)-[r]->(m)
    WHERE NOT n:Chunk AND NOT n:Document
      AND NOT m:Chunk AND NOT m:Document
    RETURN elementId(n) AS source_id, labels(n)[0] AS source_label,
           coalesce(n.name, 'unknown') AS source_name,
           type(r) AS rel_type,
           elementId(m) AS target_id, labels(m)[0] AS target_label,
           coalesce(m.name, 'unknown') AS target_name
    {limit_clause}
    """

    try:
        with driver.session() as session:
            result = session.run(query, **params)

            nodes = {}
            edges = []

            for record in result:
                # Add source node
                source_id = record["source_id"]
                if source_id not in nodes:
                    nodes[source_id] = {
                        "id": source_id,
                        "label": record["source_name"],
                        "type": record["source_label"]
                    }

                # Add target node
                target_id = record["target_id"]
                if target_id not in nodes:
                    nodes[target_id] = {
                        "id": target_id,
                        "label": record["target_name"],
                        "type": record["target_label"]
                    }

                # Add edge
                edges.append({
                    "source": source_id,
                    "target": target_id,
                    "type": record["rel_type"]
                })

            return jsonify({
                "nodes": list(nodes.values()),
                "edges": edges,
                "capped": limit > 0,
                "limit": limit if limit > 0 else None,
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/entities', methods=['GET'])
def get_entities():
    """Get all entities with their types"""
    entity_type = request.args.get('type', None)

    allowed_types = {
        "中药", "功效", "主治", "性味", "归经", "用法用量", "禁忌", "配伍",
        "药材来源", "炮制", "化学成分", "方剂", "典籍", "人物", "概念"
    }
    if entity_type and entity_type not in allowed_types:
        return jsonify({"error": "Unknown entity type"}), 400

    if entity_type:
        query = f"""
        MATCH (n:`{entity_type}`)
        WHERE NOT n:Chunk AND NOT n:Document
        RETURN elementId(n) as id, labels(n)[0] as type,
               coalesce(n.name, 'unknown') as name
        """
    else:
        query = """
        MATCH (n)
        WHERE NOT n:Chunk AND NOT n:Document
        RETURN elementId(n) as id, labels(n)[0] as type,
               coalesce(n.name, 'unknown') as name
        """

    try:
        with driver.session() as session:
            result = session.run(query)
            entities = [dict(record) for record in result]
            return jsonify({"entities": entities}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/entity/<entity_id>', methods=['GET'])
def get_entity_details(entity_id):
    """Get detailed information about a specific entity"""
    query = """
    MATCH (n)
    WHERE elementId(n) = $entity_id
    OPTIONAL MATCH (n)-[r]-(related)
    WHERE NOT related:Chunk AND NOT related:Document
    RETURN n,
           labels(n)[0] as type,
           collect(DISTINCT {
               node: related.name,
               relationship: type(r),
               direction: CASE
                   WHEN startNode(r) = n THEN 'outgoing'
                   ELSE 'incoming'
               END
           }) as connections
    """

    try:
        with driver.session() as session:
            result = session.run(query, entity_id=entity_id)
            record = result.single()

            if not record:
                return jsonify({"error": "Entity not found"}), 404

            node = record["n"]
            return jsonify({
                "id": entity_id,
                "type": record["type"],
                "properties": dict(node),
                "connections": record["connections"]
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get statistics about the knowledge graph"""
    try:
        with driver.session() as session:
            # Count nodes by type
            node_counts = session.run("""
                MATCH (n)
                WHERE NOT n:Chunk AND NOT n:Document
                RETURN labels(n)[0] as type, count(n) as count
            """)

            nodes_by_type = {record["type"]: record["count"] for record in node_counts}

            # Count relationships by type
            rel_counts = session.run("""
                MATCH ()-[r]->()
                RETURN type(r) as type, count(r) as count
            """)

            rels_by_type = {record["type"]: record["count"] for record in rel_counts}

            # Total counts
            total_nodes = session.run("""
                MATCH (n)
                WHERE NOT n:Chunk AND NOT n:Document
                RETURN count(n) as count
            """).single()["count"]

            total_rels = session.run("""
                MATCH ()-[r]->()
                RETURN count(r) as count
            """).single()["count"]

            return jsonify({
                "total_nodes": total_nodes,
                "total_relationships": total_rels,
                "nodes_by_type": nodes_by_type,
                "relationships_by_type": rels_by_type
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/search', methods=['GET'])
def search_entities():
    """Search entities by name"""
    query_text = request.args.get('q', '')

    if not query_text:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    query = """
    MATCH (n)
    WHERE NOT n:Chunk AND NOT n:Document
      AND toLower(coalesce(n.name, '')) CONTAINS toLower($search_text)
    RETURN elementId(n) as id, labels(n)[0] as type, n.name as name
    LIMIT 20
    """

    try:
        with driver.session() as session:
            result = session.run(query, search_text=query_text)
            entities = [dict(record) for record in result]
            return jsonify({"results": entities}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/graphrag', methods=['POST'])
def graphrag_query():
    """Perform GraphRAG query and return answer with subgraph used"""
    data = request.get_json()
    question = data.get('question', '')
    method = data.get('method', 'auto')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        if method in {'auto', 'kg_v2'}:
            answer, subgraph, sources, packet = _answer_with_auto(question)
            return jsonify({
                "answer": answer,
                "method": method,
                "intent": packet.intent,
                "subgraph": subgraph,
                "sources": sources,
                "trace": packet.trace,
                "warnings": packet.warnings,
                "abstained": not bool(packet.items),
                "claims": [],
            }), 200

        native_modes = {
            "keyword_graph": {"use_fulltext": True, "use_vector": None, "expand_graph": True},
            "vector": {"use_fulltext": False, "use_vector": True, "expand_graph": False},
            "vector_cypher": {"use_fulltext": False, "use_vector": True, "expand_graph": True},
            "hybrid": {"use_fulltext": True, "use_vector": True, "expand_graph": False},
            "hybrid_cypher": {"use_fulltext": True, "use_vector": True, "expand_graph": True},
        }
        if method in native_modes:
            answer, subgraph, sources = _answer_with_keyword_graph(
                question, **native_modes[method]
            )
            return jsonify({
                "answer": answer,
                "method": method,
                "subgraph": subgraph,
                "sources": sources,
            }), 200
        return jsonify({"error": f"Unknown method: {method}"}), 400

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    print("Starting GraphRAG Backend API...")
    print(f"Neo4j URI: {os.getenv('NEO4J_URL')}")
    print("API will be available at http://localhost:5000")
    print("\nAvailable endpoints:")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/graph           - Get complete knowledge graph")
    print("  GET  /api/entities        - Get all entities")
    print("  GET  /api/entity/<id>     - Get entity details")
    print("  GET  /api/stats           - Get graph statistics")
    print("  GET  /api/search?q=...    - Search entities")

    app.run(debug=True, host='0.0.0.0', port=5000)
