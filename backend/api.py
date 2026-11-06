# backend/api.py
"""
Flask backend API for serving knowledge graph data to visualization
"""
import json
import os
import re
import sys
import traceback
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from neo4j import GraphDatabase
from openai import OpenAI
from werkzeug.exceptions import RequestEntityTooLarge

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
sys.path.insert(0, str(PROJECT_ROOT / "graphragexpr"))
from external_embedder import ExternalEmbedder

load_dotenv(PROJECT_ROOT / ".env")


def _positive_env_int(name, default, *, maximum=None):
    value = int(os.getenv(name) or default)
    if value <= 0 or (maximum is not None and value > maximum):
        raise ValueError(f"Invalid {name}: {value}")
    return value


def _server_options():
    """Local defaults; external listening and debugging require explicit settings."""
    return {
        "host": (os.getenv("BACKEND_HOST") or "").strip() or "127.0.0.1",
        "port": _positive_env_int("BACKEND_PORT", 5000, maximum=65535),
        "debug": (os.getenv("BACKEND_DEBUG") or "false").strip().lower() in {"1", "true", "yes", "on"},
        # Keep the process identity stable for the launcher health check.
        "use_reloader": False,
    }


def _configure_http(application):
    application.config["MAX_CONTENT_LENGTH"] = _positive_env_int(
        "BACKEND_MAX_REQUEST_BYTES", 65536
    )
    origins = [origin.strip() for origin in (os.getenv("CORS_ORIGINS") or "").split(",") if origin.strip()]
    if origins:
        CORS(application, resources={r"/api/*": {"origins": origins}})

app = Flask(__name__, static_folder="../frontend", static_url_path="")
_configure_http(app)


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify({"error": "Request body is too large"}), 413

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
MAX_CONTEXT_CHUNKS = 8
_LUCENE_SPECIAL = re.compile(r'([+\-&|!(){}\[\]^"~*?:\\/])')
# 问句里出现的槽位词/范畴词不能当检索实体：否则「人参的性味、归经和功效」
# 会把「功效」送进全文检索，高频词把大黄等条目排到人参前面。
_SLOT_QUERY_TERMS = frozenset({
    "性味", "归经", "功效", "主治", "用法", "用量", "禁忌", "配伍",
    "中药", "药材", "药性", "炮制", "来源", "概念", "方剂", "典籍",
    "什么", "哪些", "如何", "怎么", "区别",
})


@lru_cache(maxsize=1)
def _configured_embedder():
    """Load the local model once per backend process."""
    return ExternalEmbedder(dimension=int(os.getenv("EMBED_DIM", "512")))


def _escape_lucene(term: str) -> str:
    """转义 Lucene 特殊字符，避免用户问题被当成检索语法。"""
    return _LUCENE_SPECIAL.sub(r"\\\1", term)


def _build_fulltext_query(question: str, exact_terms: list[str]) -> str:
    """Build a Lucene OR query from entity names and Chinese character bigrams."""
    terms: list[str] = []
    seen: set[str] = set()

    def add(term: str) -> None:
        """把一个检索词追加进 Lucene 查询字符串。"""
        cleaned = term.strip()
        if len(cleaned) < 2 or cleaned in seen:
            return
        seen.add(cleaned)
        terms.append(_escape_lucene(cleaned))

    anchors = [
        term.strip()
        for term in exact_terms
        if term and term.strip() not in _SLOT_QUERY_TERMS and len(term.strip()) >= 2
    ]
    # 问句里已经锚定到药名/功效名时，只搜这些词，避免「功效」「性味」带跑全库。
    if anchors:
        return " OR ".join(f'"{_escape_lucene(term)}"' for term in anchors[:8])

    compact = re.sub(r"\s+", "", question.lower())
    chinese = "".join(re.findall(r"[\u3400-\u9fff]+", compact))
    for index in range(len(chinese) - 1):
        add(chinese[index : index + 2])
        if len(terms) >= 24:
            break

    for token in re.findall(r"[a-z0-9]{2,}", compact):
        add(token)

    return " OR ".join(terms[:24]) if terms else _escape_lucene(compact[:40] or "中药")


def _is_slot_query_term(name: str) -> bool:
    """槽位词不能当作检索锚点。"""
    return "".join((name or "").split()) in _SLOT_QUERY_TERMS


def _question_anchor_names(session, question: str) -> list[str]:
    """从问句里抽出中药/方剂，以及长度足够的功效/主治名。"""
    rows = session.run(
        """
        MATCH (entity)
        WHERE coalesce(entity.preferred_name, entity.name) IS NOT NULL
          AND size(coalesce(entity.preferred_name, entity.name)) >= 2
          AND size(coalesce(entity.preferred_name, entity.name)) <= 20
          AND $question CONTAINS coalesce(entity.preferred_name, entity.name)
          AND (
            entity:中药 OR entity:方剂
            OR (entity:Entity AND entity.entity_type IN ['HERB', 'FORMULA'])
            OR (
              size(coalesce(entity.preferred_name, entity.name)) >= 3
              AND (
                entity:功效 OR entity:主治
                OR (entity:Entity AND entity.entity_type IN ['EFFICACY', 'INDICATION'])
              )
            )
          )
        RETURN DISTINCT coalesce(entity.preferred_name, entity.name) AS name
        ORDER BY size(name) DESC
        LIMIT 12
        """,
        question=question,
    )
    names = []
    seen = set()
    for record in rows:
        name = (record["name"] or "").strip()
        if not name or _is_slot_query_term(name) or name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _chunks_mention_anchors(chunks: list[dict], anchors: list[str]) -> list[dict]:
    """只保留原文提到问句锚点的文本块；有锚点时绝不回退到其他药的条目。"""
    if not anchors or not chunks:
        return chunks
    kept = [
        chunk
        for chunk in chunks
        if any(anchor in (chunk.get("text") or "") for anchor in anchors)
    ]
    return kept


def _matched_anchor_terms(text: str, anchors: list[str]) -> list[str]:
    """文本块里实际出现的问句锚点。"""
    return [anchor for anchor in anchors if anchor and anchor in (text or "")]


def _chunk_record(record) -> dict:
    """把 Neo4j 查询行转成前端/生成可用的文本块字典。"""
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

    exact_terms = []
    lexical_candidates = []
    with driver.session(database=database) as session:
        exact_terms = _question_anchor_names(session, question)
        if use_fulltext:
            fulltext_query = _build_fulltext_query(question, exact_terms)
            try:
                lexical_candidates = [
                    {
                        **_chunk_record(record),
                        "score": float(record["lexical_score"] or 0.0),
                        "lexical_score": float(record["lexical_score"] or 0.0),
                        "matched_terms": _matched_anchor_terms(record["text"] or "", exact_terms),
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
                seed = exact_terms[:5] or [
                    token
                    for token in re.findall(r"[\u3400-\u9fff]{2,4}", question)
                    if not _is_slot_query_term(token)
                ][:3] or [question[:8]]
                lexical_candidates = [
                    {
                        **_chunk_record(record),
                        "score": float(record["hit_count"] or 0.0),
                        "lexical_score": float(record["hit_count"] or 0.0),
                        "matched_terms": _matched_anchor_terms(record["text"] or "", exact_terms),
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
        if vector_enabled and not use_fulltext:
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

        lexical_candidates = _chunks_mention_anchors(lexical_candidates, exact_terms)
        vector_candidates = _chunks_mention_anchors(vector_candidates, exact_terms)

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

    if expand_graph and ranked_chunks and ranked_chunks[0].get("index") is not None:
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
            if exact_terms and not _chunks_mention_anchors([companion], exact_terms):
                companion = None
            if companion:
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
        from graphragexpr.retrieval.cross_chunk_graph import (
            expand_cross_chunk_neighborhood,
            merge_extra_chunks,
            read_active_run_id,
        )

        with driver.session(database=database) as session:
            node_list, edges, triples, extra_chunks = expand_cross_chunk_neighborhood(
                session,
                chunk_ids,
                run_id=read_active_run_id(session),
                node_limit=MAX_SUBGRAPH_NODES,
                edge_limit=MAX_SUBGRAPH_EDGES,
                triple_limit=MAX_GRAPH_TRIPLES,
                extra_chunk_limit=MAX_CONTEXT_CHUNKS,
                seed_names=exact_terms,
            )
        nodes = {node["id"]: node for node in node_list}
        ranked_chunks = merge_extra_chunks(
            ranked_chunks, extra_chunks, limit=MAX_CONTEXT_CHUNKS
        )
        ranked_chunks = _chunks_mention_anchors(ranked_chunks, exact_terms)

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


def _retrieve_auto(question):
    """auto 的**检索部分**：链接、路由、检索，不生成自然语言。

    拆出来的原因是高风险问题只允许执行检索部分：先跑一遍自由文本生成再
    丢掉，既浪费一次模型调用，也让高风险答案无谓地依赖普通 LLM 是否可用。

    Returns:
        `(packet, subgraph, sources)`。
    """
    from graphragexpr.domain.ontology import load_ontology
    from graphragexpr.retrieval import retrieve

    store = _neo4j_store()
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
    return packet, result.subgraph, _sources_from_packet(packet)


def _sources_from_packet(packet):
    """从证据包抽出带页码的来源列表，供前端溯源。"""
    return [
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


def _generate_auto(question, packet, subgraph, sources):
    """auto 的**生成部分**。非高风险问题才会走到这里。"""
    if not packet.items:
        warning = packet.warnings[0] if packet.warnings else "没有检索到可定位证据"
        return (
            f"证据不足，无法基于当前知识图谱作答：{warning}。",
            subgraph,
            sources,
            packet,
        )

    llm_token = (os.getenv("LLM_TOKEN") or "").strip()
    if not llm_token:
        return _evidence_only_answer(packet), subgraph, sources, packet

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
    warning_block = ""
    if packet.warnings:
        warning_block = (
            "# 检索警告（必须写入回答，不得忽略）\n"
            + "\n".join(packet.warnings)
            + "\n\n"
        )

    prompt = f"""请严格依据下面的《中药学》结构化检索证据回答问题。
不得使用证据外知识；关键结论必须引用对应 PDF 页码。
同一主体的属性可能来自不同页，请综合全部证据作答，不要只根据第一条。
部分匹配不得表述为全部满足。
问「均不能 / 都不能搭配」时，必须同时交代各药各自的禁忌，以及是否存在共同对象；
没有共同对象时要明确写出，不得把单味药的禁忌说成两药共有。
COMBINED_WITH（适宜配伍）不能当作配伍禁忌。
涉及剂量、毒性、禁忌、孕妇或儿童时，只能逐字依据证据，
并提醒以教材原文和专业人员意见为准。

{warning_block}# 检索意图
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
    return answer, subgraph, sources, packet


def _retrieve_keyword_graph(
    question, *, use_fulltext=True, use_vector=None, expand_graph=True
):
    """原生检索的**检索部分**。

    「一个文本块都没检索到」是**检索结果**而不是异常：这里不再抛错，
    否则高风险问题会因为所选 method 恰好没检索到文本块而变成 HTTP 500，
    而高风险层本来可以照常用结构化证据作答。空结果由生成部分决定怎么处理。
    """
    chunks, subgraph, triples = _keyword_graph_search(
        question,
        use_fulltext=use_fulltext,
        use_vector=use_vector,
        expand_graph=expand_graph,
    )
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
    return chunks, subgraph, triples, sources


def _generate_keyword_graph(question, chunks, triples, *, use_graph=True):
    """原生检索的**生成部分**。非高风险问题才会走到这里。"""
    if not chunks:
        return "未检索到与问题直接相关的教材原文，无法根据当前图检索结果作答。请改用「自动路由」，或换一种包含药名的问法。"

    textbook_context = []
    for position, chunk in enumerate(chunks, start=1):
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end") or page_start
        page_label = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        textbook_context.append(
            f"[教材片段 {position}｜PDF 第 {page_label} 页]\n{chunk['text']}"
        )

    if use_graph:
        graph_context = "\n".join(
            f"- {source} --[{relationship}]--> {target}"
            for source, relationship, target in triples[:MAX_GRAPH_TRIPLES]
        ) or "- 未抽取到与命中文本块直接相关的三元组"
        prompt = f"""请严格依据下面的《中药学》教材片段与知识图谱关系回答问题。
图谱关系可能来自其他页的文本块，请把多块证据拼成完整答案，不要只根据第一段作答。
优先使用教材原文中的定义和分类；关键结论后标注对应的 PDF 页码。
如果问题问的是某一味药（如人参），只能使用主语为该药的证据；
不得把其他药物（如大黄、黄芪）的性味、归经或功效说成该药的属性。
若片段与图谱都不含该药的对应条目，必须明确说「未检索到」，不得改用其他药凑答案。
如果问题涉及剂量、毒性或临床建议，提醒用户以教材原文和专业人员意见为准。

# 问题
{question}

# 命中的教材原文
{chr(10).join(textbook_context)}

# 命中文本块内部的知识图谱关系（仅作补充）
{graph_context}
"""
    else:
        prompt = f"""请严格依据下面的《中药学》向量检索片段回答问题。
只能使用下列原文，不得调用或补充知识图谱关系，也不得引入未出现在片段中的配伍、禁忌或跨页事实。
如果问题问的是某一味药，不得用片段里其他药物的属性代替作答；没有该药原文时必须说明未检索到。
关键结论后标注对应的 PDF 页码。
如果问题涉及剂量、毒性或临床建议，提醒用户以教材原文和专业人员意见为准。

# 问题
{question}

# 向量检索命中的教材原文
{chr(10).join(textbook_context)}
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
    return answer


@app.route('/')
def index_page():
    """Serve the bundled visualization and question-answering UI."""
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        driver.verify_connectivity()
        return jsonify({"status": "ok", "message": "Connected to Neo4j", "pid": os.getpid()}), 200
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


#: 是否让模型参与高风险候选项生成。默认关闭：高风险最终文字由程序模板渲染，
#: 结构化单元本身已经能给出有据可查的结论，多一次模型调用只增加延迟与失败面。
#: 需要做「原文+提示词 / +输出校验 / 条件单元+输出校验」对比时再打开。
HIGH_RISK_USE_LLM = (os.getenv("HIGH_RISK_USE_LLM") or "").strip().lower() in {
    "1", "true", "yes", "on",
}


def _high_risk_client():
    """Build the model client for candidate claims, or None when not configured."""
    if not HIGH_RISK_USE_LLM or not (os.getenv("LLM_TOKEN") or "").strip():
        return None
    try:
        from graphragexpr.config.settings import KGSettings
        from graphragexpr.extract.llm_client import OpenAICompatibleClient

        return OpenAICompatibleClient(KGSettings())
    except Exception:  # 缺密钥或依赖不可用时不影响主链路
        return None


# ---- 接口层的调用顺序策略 ------------------------------------------------
# 高风险问题的顺序约束：**先检索、再统一核验、最后程序化文字**。
# 普通自由文本生成一步都不跑——先跑再丢掉既浪费一次模型调用，又让高风险答案
# 的可用性无谓地取决于普通 LLM 是否可用（它抛异常就会变成 HTTP 500）。
# 另外三点同样必须靠接口层保证：
#   1. 核验层异常时，`answer` 必须被**替换**为原文展示或拒答文本，绝不能把未经
#      核验的自由文本照原样返回；
#   2. 顶层 `sources` / `subgraph` 必须由**最终保留或展示的高风险证据单元**重建，
#      与 `claims` / `evidence_units` 同源，否则前端展开的溯源可能来自另一次检索；
#   3. 最终证据的来源必须**如实**写在响应里（见 `_EVIDENCE_METHOD`）。

#: 核验链路不可用时**唯一**允许出现的 `answer_status`。前端
#: `HR_STATUS_LABEL` / `HR_STATUS_CLASS` 已认识该状态（渲染为红色徽标）；
#: 具体失败原因放在 `high_risk.error`（`retrieval_unavailable` /
#: `verification_unavailable` / `no_active_run`），避免新增前端不认识的枚举值。
_VERIFICATION_UNAVAILABLE = "verification_unavailable"

#: 高风险响应的最终证据**来自哪里**。取值只有一个：高风险断言读取
#: （`load_high_risk_assertions` 按已识别主语/宾语与风险谓词读出的条件证据单元，
#: 再经统一核验）。写进响应是为了不让人误以为「选了 vector 就是用向量检索到的
#: 文本块当证据」。
#:
#: 所选 method 在高风险链路上的作用只有**前置定位**：
#:   - 决定执行哪条检索、带什么参数（`retrieval.params` 如实记录）；
#:   - `auto` 由检索层的路由给出 `linked_entities`，图检索 / 向量检索由接口层
#:     用既有链接器在 ACTIVE 批次里解析——两者都只决定「核验层读哪些主语」；
#:   - 检索命中的原文只在**核验不可用时**被照录展示（逐条 `verified: false`），
#:     不是结论的证据。
#: 也就是说：换 method 会换检索结果与定位，但只要解析出的实体相同，最终证据单元
#: 与结论就相同。原生 method 只参与前置定位。
_EVIDENCE_METHOD = "high_risk_assertion"

#: auto 家族：走 `retrieve()` 管线（含意图路由，其 `route.is_high_risk` 与词表
#: 判定相互印证）。
_AUTO_METHODS = frozenset({"auto"})

#: 对外只保留两种原生检索：图检索跨实体补其他页，向量检索只走向量通道。
_NATIVE_METHODS = {
    "keyword_graph": {"use_fulltext": True, "use_vector": False, "expand_graph": True},
    "vector": {"use_fulltext": False, "use_vector": True, "expand_graph": False},
}

_ALL_METHODS = _AUTO_METHODS | frozenset(_NATIVE_METHODS)

#: 高风险证据卡 → 前端 `colorScheme` 的中文节点类型。前端对未知类型会回退
#: `default`，这里**不编造**实体类型，只用风险类别决定的展示标签；主语一律
#: 是「中药」，因为高风险主语只可能是药材。
_HR_NODE_TYPE = {
    "DOSAGE": "用法用量",
    "CONTRAINDICATION": "禁忌",
    "INCOMPATIBILITY": "配伍",
}

#: 核验不可用时原文展示的篇幅上限：每段截断到 400 字、最多 3 段。
#: 展示的是检索命中的教材原文，不是结论，也不会因为截断而变成结论。
_HR_EXCERPT_CHARS = 400
_HR_EXCERPT_MAX = 3


def _empty_high_risk_fields():
    """非高风险响应里的高风险字段形态（保持既有契约）。

    每次返回**新**字典：这些键会随各模式自己的字段一起 `jsonify`，
    共用可变默认值会让某次响应不小心改到另一次。
    `answer_status` 为 None、不含 `abstained`，以免覆盖各模式自己的弃权语义。
    """
    return {
        "answer_status": None,
        "claims": [],
        "evidence_units": [],
        "sub_questions": [],
        "rejections": [],
        "coverage_notes": [],
        "high_risk": {"is_high_risk": False},
    }


def _neo4j_store():
    """打开一个只读 Neo4j 句柄，**调用方负责 close**。

    检索部分与核验部分各开各的句柄：两段都可能失败，把生命周期放在各自函数里，
    异常路径才不会泄漏连接，也不会出现「检索关掉的句柄被核验继续用」。
    """
    from graphragexpr.storage.neo4j_store import Neo4jStore

    return Neo4jStore()


def _is_high_risk_question(question):
    """是否走高风险链路。词表判定是**叠加策略**，不改六意图枚举。"""
    from graphragexpr.highrisk import is_high_risk_question

    return is_high_risk_question(question)


def _format_page_label(page_start, page_end):
    """页码展示。页码缺失时写「页码不可用」，不编造页码。"""
    if page_start is None:
        return "页码不可用"
    if page_end is None or page_end == page_start:
        return f"第 {page_start} 页"
    return f"第 {page_start}-{page_end} 页"


def _excerpt(text):
    """把原文折成一段可展示的摘录，截断时**明写**已截断。"""
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= _HR_EXCERPT_CHARS:
        return collapsed
    return collapsed[:_HR_EXCERPT_CHARS] + "…（已截断，仅展示此处前 400 字）"


def _link_question_entities(question):
    """在 ACTIVE 批次里只读链接问题中的 HERB。

    与拆改前 `_high_risk_finalize(linked_entities=None)` 的行为一致：调用既有的
    `link_entities_in_text`，**没有**重做实体消歧。

    Returns:
        `(run_id, linked_entities)`；没有 ACTIVE 批次时 run_id 为 None。
    """
    from graphragexpr.storage import serving_read

    store = _neo4j_store()
    try:
        run_id = serving_read.active_run(store)
        if not run_id:
            return None, []
        return run_id, serving_read.link_entities_in_text(
            store, question, run_id=run_id, types=("HERB",)
        )
    finally:
        store.close()


def _high_risk_retrieval(question, method, *, auto_result=None):
    """只执行用户所选 method 的**检索部分**，不生成任何自由文本。

    `auto` 走 `retrieve()`（路由、链接、检索都在里面）；图检索 / 向量检索走
    原生检索，参数就是用户所选 method 的参数。检索结果随后进入统一的证据组织
    与核验：`linked_entities` 决定核验层读哪些主语/宾语，`raw_text` 只在核验层
    不可用时用于**原文展示**（明确标注未经核验）。

    Args:
        question: 用户问题。
        method: 用户所选检索方式。
        auto_result: 已经跑过的 `_retrieve_auto` 结果。auto 分支在检索之后才
            发现高风险时复用同一份结果，不做第二次检索。

    Returns:
        dict：`mode` / `method` / `linked_entities` / `run_id` / `intent` /
        `trace` / `items` / `raw_text` / `warnings`。

    Side effects: 只读数据库。
    """
    if method in _AUTO_METHODS:
        if auto_result is None:
            auto_result = _retrieve_auto(question)
        packet, _subgraph, _sources = auto_result
        trace = packet.trace or {}
        return {
            "mode": "auto",
            "method": method,
            "linked_entities": list(packet.linked_entities or []),
            "run_id": trace.get("run_id"),
            "intent": packet.intent,
            "trace": packet.trace,
            "items": len(packet.items),
            "raw_text": [
                {
                    "page_start": item.page_start,
                    "page_end": item.page_end,
                    "text": _excerpt(item.evidence_quote),
                }
                for item in packet.items[:_HR_EXCERPT_MAX]
            ],
            "warnings": list(packet.warnings or []),
        }

    params = dict(_NATIVE_METHODS[method])
    chunks, _subgraph, _triples, _sources = _retrieve_keyword_graph(
        question, **params
    )
    run_id, linked = _link_question_entities(question)
    return {
        "mode": "keyword_graph",
        "method": method,
        "params": params,
        "linked_entities": linked,
        "run_id": run_id,
        "intent": None,
        "trace": None,
        "items": len(chunks),
        "raw_text": [
            {
                "page_start": chunk.get("page_start"),
                "page_end": chunk.get("page_end"),
                "text": _excerpt(chunk.get("text")),
            }
            for chunk in chunks[:_HR_EXCERPT_MAX]
        ],
        "warnings": [],
    }


def _retrieval_summary(retrieval, *, run_id=None):
    """本次高风险回答实际执行的检索，用于核对所选 method 确实被执行。"""
    summary = {
        "method": retrieval.get("method"),
        "mode": retrieval.get("mode"),
        "run_id": run_id or retrieval.get("run_id"),
        "items": retrieval.get("items"),
        "linked_entities": [
            {
                "canonical_id": entity.get("canonical_id"),
                "name": entity.get("preferred_name") or entity.get("name"),
                "entity_type": entity.get("entity_type"),
            }
            for entity in (retrieval.get("linked_entities") or [])
        ],
    }
    if retrieval.get("params"):
        summary["params"] = dict(retrieval["params"])
    return summary


def _high_risk_answer(question, retrieval):
    """把所选 method 的检索结果交给**唯一**的高风险证据组织与核验入口。

    检索方式各不相同，核验入口只有一个：只要有一种方式能绕过核验，用户就可以
    靠换 method 拿到未经核验的高风险结论。

    Returns:
        `(answer, run_id)`；没有 ACTIVE 批次时 answer 为 None。

    Side effects: 只读数据库。
    """
    from graphragexpr.highrisk import HighRiskService
    from graphragexpr.storage import serving_read

    store = _neo4j_store()
    try:
        run_id = retrieval.get("run_id") or serving_read.active_run(store)
        if not run_id:
            return None, None
        service = HighRiskService(store, run_id=run_id, llm=_high_risk_client())
        return (
            service.answer(
                question, linked_entities=retrieval.get("linked_entities") or []
            ),
            run_id,
        )
    finally:
        store.close()


def _sources_from_units(units):
    """由**最终保留或展示的高风险证据单元**重建顶层 `sources`。

    与 `claims` / `evidence_units` 同源：三者只来自这一批单元，因此前端溯源面板
    展开的原文，一定就是结论所依据的那一段，而不是另一次检索命中的文本。
    """
    return [
        {
            "evidence_id": card.unit_id,
            "unit_id": card.unit_id,
            "assertion_id": card.assertion_id,
            "chunk_id": card.chunk_ids[0] if card.chunk_ids else None,
            "page_start": card.page_start,
            "page_end": card.page_end,
            # 高风险单元来自断言，不是向量/词法检索，没有可比的检索分数：
            # 这里给 None 而不是编一个（前端对 None 会省略分数展示）。
            "score": None,
            "matched_terms": [],
            "evidence_quote": card.evidence_quote,
            "risk_category": card.risk_category.value,
            "severity": card.severity.value,
        }
        for card in units
    ]


def _subgraph_from_units(units):
    """由**最终保留或展示的高风险证据单元**重建 `subgraph`（同源）。

    只画证据里真实出现过的关系：主语是药材、宾语是证据声明的对象，边就是该
    风险关系的谓词。没有对象名时只保留主语节点，不补画一条没有证据的边。
    """
    nodes = {}
    edges = {}
    for card in units:
        subject_id = card.subject_id or card.subject_name
        if not subject_id:
            continue
        nodes.setdefault(
            subject_id,
            {
                "id": subject_id,
                "label": card.subject_name or subject_id,
                "type": "中药",
            },
        )
        object_id = card.object_id or card.object_name
        if not object_id:
            continue
        nodes.setdefault(
            object_id,
            {
                "id": object_id,
                "label": card.object_name or object_id,
                "type": _HR_NODE_TYPE.get(card.risk_category.value, "概念"),
            },
        )
        edges.setdefault(
            (subject_id, object_id, card.predicate),
            {
                "source": subject_id,
                "target": object_id,
                "type": card.predicate,
            },
        )
    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


def _sources_from_raw(raw):
    """核验不可用时被**展示**的教材原文。逐条标 `verified=False`。"""
    return [
        {
            "evidence_id": f"raw-excerpt-{index}",
            "page_start": item.get("page_start"),
            "page_end": item.get("page_end"),
            "score": None,
            "matched_terms": [],
            "evidence_quote": item.get("text") or "",
            # 明确可机读：这些原文**没有**经过核验，只是照录。
            "verified": False,
        }
        for index, item in enumerate(raw, start=1)
    ]


def _high_risk_unavailable(question, exc, *, method, retrieval=None, warnings=None):
    """核验链路不可用时的**受控**回答：原文展示或拒答，绝不保留自由文本。

    拆改前这里会「保留原答案」，理由是用户不至于一无所获。但代价是把一段未经
    核验的自由文本连同「高风险」一起返回给用户，会绕过高风险核验。
    现在 `answer` 一律被**替换**为程序化文本：要么是明确标注「未经核验」的
    教材原文展示（来源为所选 method 的检索结果），要么是直接拒答。

    顶层 `sources` / `subgraph` 同样只描述被展示的原文：`evidence_units` 与
    `claims` 为空，因为这次**没有任何单元通过核验**。`evidence_method` 描述的是
    本链路的证据来源（高风险断言）；本次没有单元通过核验，`sources` 里只是照录
    的检索原文（`verified: false`），不是结论的证据。

    `abstained` 显式为 True：不再有未经核验的自由文本当答案，这个响应就是一次
    弃权，前端与下游不能把它读成「有答案」。

    Returns:
        `(payload, status_code)`，状态码恒为 200：高风险问题必须给出受控回答，
        不能因为核验层异常就变成 500 让前端只剩一句报错。
    """
    notes = list(warnings or [])
    raw = list((retrieval or {}).get("raw_text") or [])
    notes.append(f"高风险核验未能运行：{exc}")

    if raw:
        excerpts = "\n".join(
            f"[{_format_page_label(item.get('page_start'), item.get('page_end'))}] "
            f"{item.get('text') or ''}"
            for item in raw
        )
        answer_text = (
            "本次高风险核验未能完成，因此不能给出任何结论；"
            "也不会展示未经核验的模型生成内容。\n"
            f"原因：{exc}\n"
            f"以下为所选检索方式（{method}）命中的教材原文，仅为原文展示，"
            "**未经核验**，不得据此作出用药决定：\n"
            f"{excerpts}"
        )
    else:
        answer_text = (
            "本次高风险核验未能完成，所选检索方式"
            f"（{method}）也没有取到可展示的教材原文，因此不予回答。\n"
            f"原因：{exc}\n"
            "请不要依据本次回答作出用药决定；请查阅教材原文或咨询专业人员。"
        )

    payload = {
        "answer": answer_text,
        "method": method,
        "sources": _sources_from_raw(raw),
        "subgraph": {"nodes": [], "edges": []},
        "warnings": notes,
        "abstained": True,
        "evidence_method": _EVIDENCE_METHOD,
        "answer_status": _VERIFICATION_UNAVAILABLE,
        "claims": [],
        "evidence_units": [],
        "sub_questions": [],
        "rejections": [],
        "coverage_notes": notes,
        # 与正常高风险回应保持同一组字段形状，前端不必为异常路径另写一套分支。
        # `verification_note` 为空：本次没有任何结论通过核验，不显示核验说明。
        "decision": "not_applicable",
        "decision_text": "",
        "verification_note": "",
        "answer_blocks": [],
        "other_conditions": [],
        "high_risk": {
            "is_high_risk": True,
            "status": _VERIFICATION_UNAVAILABLE,
            "error": "verification_unavailable",
        },
    }
    if retrieval is not None:
        payload["retrieval"] = _retrieval_summary(retrieval)
        if retrieval.get("intent"):
            payload["intent"] = retrieval["intent"]
    return payload, 200


def _no_active_run_response(question, method, retrieval, warnings):
    """没有 ACTIVE 批次：高风险问题只能受控拒答（与拆改前一致）。"""
    message = "没有 ACTIVE 的 IngestRun，高风险核验无法进行"
    return {
        "answer": (
            "没有 ACTIVE 的数据批次，无法进行高风险核验；"
            "请不要依据本次回答作出用药决定。"
        ),
        "method": method,
        "sources": [],
        "subgraph": {"nodes": [], "edges": []},
        "warnings": [message] + list(warnings),
        "answer_status": "abstain",
        "abstained": True,
        "evidence_method": _EVIDENCE_METHOD,
        "claims": [],
        "evidence_units": [],
        "sub_questions": [],
        "rejections": [],
        "coverage_notes": [message],
        # 同 `_high_risk_unavailable`：字段形状一致，核验说明为空。
        "decision": "not_applicable",
        "decision_text": "",
        "verification_note": "",
        "answer_blocks": [],
        "other_conditions": [],
        "retrieval": _retrieval_summary(retrieval),
        "high_risk": {"is_high_risk": True, "error": "no_active_run"},
    }, 200


def _high_risk_response(question, method, *, auto_result=None):
    """高风险问题的完整链路：检索 → 统一证据组织与核验 → 程序化文字 + 同源溯源。

    Returns:
        `(payload, status_code)`。检索或核验任何一步异常都返回受控文本的 200，
        不返回未经核验的自由文本，也不把高风险问题降级成一句 HTTP 500。
    """
    from graphragexpr.highrisk import pack_answer_payload

    warnings = []
    try:
        retrieval = _high_risk_retrieval(question, method, auto_result=auto_result)
    except Exception as exc:
        # 检索部分就失败了：连原文都取不到，只能受控拒答。
        traceback.print_exc()
        return _high_risk_unavailable(question, exc, method=method, warnings=warnings)
    warnings.extend(retrieval.get("warnings") or [])

    try:
        answer, run_id = _high_risk_answer(question, retrieval)
    except Exception as exc:
        traceback.print_exc()
        return _high_risk_unavailable(
            question, exc, method=method, retrieval=retrieval, warnings=warnings
        )
    if answer is None:
        return _no_active_run_response(question, method, retrieval, warnings)

    notes = list(answer.coverage_notes) + warnings
    if answer.status.value != "grounded":
        notes.insert(0, f"高风险回答状态：{answer.status.value}")

    payload = pack_answer_payload(answer)
    payload.update(
        {
            "answer": answer.answer_text,
            "method": method,
            # 最终证据来自高风险断言读取，与所选检索方式无关；检索只做前置定位。
            "evidence_method": _EVIDENCE_METHOD,
            # 溯源与结论同源：都由**最终保留或展示的单元**重建。
            "sources": _sources_from_units(answer.evidence_units),
            "subgraph": _subgraph_from_units(answer.evidence_units),
            "warnings": notes,
            "retrieval": _retrieval_summary(retrieval, run_id=run_id),
        }
    )
    if retrieval.get("intent"):
        payload["intent"] = retrieval["intent"]
    if retrieval.get("trace"):
        payload["trace"] = retrieval["trace"]
    return payload, 200


@app.route('/api/graphrag', methods=['POST'])
def graphrag_query():
    """Perform GraphRAG query and return answer with subgraph used"""
    if not request.is_json:
        return jsonify({"error": "A JSON object is required"}), 400
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400
    question = data.get('question', '')
    method = data.get('method', 'auto')

    if not isinstance(question, str) or not question.strip():
        return jsonify({"error": "Question is required and must be a non-empty string"}), 400
    question = question.strip()
    # method 校验提前到最前面：未知 method 不再走完一次检索才报错。
    if not isinstance(method, str) or method not in _ALL_METHODS:
        return jsonify({"error": f"Unknown method: {method}"}), 400

    # **尽早**识别高风险：只执行所选 method 的检索部分，普通自由文本生成一步都
    # 不跑。拆改前高风险问题要先跑完普通检索 + 普通 LLM 生成，再被核验结果覆盖，
    # 于是高风险答案既无谓地依赖普通 LLM（它失败就是 HTTP 500），又白烧一次调用。
    if _is_high_risk_question(question):
        body, status = _high_risk_response(question, method)
        return jsonify(body), status

    try:
        if method in _AUTO_METHODS:
            packet, subgraph, sources = _retrieve_auto(question)
            route = (packet.trace or {}).get("route") or {}
            # 词表之外的路由判定（例如由意图触发的）同样必须走高风险链路：
            # 检索结果已经拿到，直接交给统一核验，不再跑生成。
            if route.get("is_high_risk"):
                body, status = _high_risk_response(
                    question, method, auto_result=(packet, subgraph, sources)
                )
                return jsonify(body), status
            answer, subgraph, sources, packet = _generate_auto(
                question, packet, subgraph, sources
            )
            return jsonify({
                "answer": answer,
                "method": method,
                "intent": packet.intent,
                "subgraph": subgraph,
                "sources": sources,
                "trace": packet.trace,
                "warnings": packet.warnings,
                "abstained": not bool(packet.items),
                **_empty_high_risk_fields(),
            }), 200

        params = _NATIVE_METHODS[method]
        chunks, subgraph, triples, sources = _retrieve_keyword_graph(
            question, **params
        )
        use_graph = bool(params.get("expand_graph"))
        if not use_graph:
            subgraph = {"nodes": [], "edges": []}
            triples = []
        answer = _generate_keyword_graph(
            question, chunks, triples, use_graph=use_graph
        )
        return jsonify({
            "answer": answer,
            "method": method,
            "subgraph": subgraph,
            "sources": sources,
            "warnings": [],
            **_empty_high_risk_fields(),
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _warmup_embedder():
    """Load the local embedding model in the background so vector modes are not cold."""
    provider = (os.getenv("EMBED_PROVIDER") or "hash").strip().lower()
    if provider == "hash":
        return
    try:
        _configured_embedder()
        print("[ok] 向量模型已预热，向量检索可直接使用")
    except Exception as exc:
        print(f"向量模型预热失败（不影响自动路由）：{exc}")


if __name__ == '__main__':
    server_options = _server_options()
    print("Starting GraphRAG Backend API...")
    print(f"Neo4j URI: {os.getenv('NEO4J_URL')}")
    print(f"API will be available at http://{server_options['host']}:{server_options['port']}")
    import threading
    threading.Thread(target=_warmup_embedder, daemon=True).start()
    print("\nAvailable endpoints:")
    print("  GET  /api/health          - Health check")
    print("  GET  /api/graph           - Get complete knowledge graph")
    print("  GET  /api/entities        - Get all entities")
    print("  GET  /api/entity/<id>     - Get entity details")
    print("  GET  /api/stats           - Get graph statistics")
    print("  GET  /api/search?q=...    - Search entities")

    app.run(**server_options)
