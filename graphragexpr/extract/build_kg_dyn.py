"""Build the Chinese materia medica knowledge graph from OCR Markdown."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "graphragexpr"))

from external_embedder import ExternalEmbedder  # noqa: E402
from extract.corpus import ChunkRecord, records_from_environment  # noqa: E402

load_dotenv(ROOT / ".env")

ENTITY_TYPES = {
    "中药",
    "功效",
    "主治",
    "性味",
    "归经",
    "用法用量",
    "禁忌",
    "配伍",
    "药材来源",
    "炮制",
    "化学成分",
    "方剂",
    "典籍",
    "人物",
    "概念",
}
TYPE_ALIASES = {
    "药物": "中药",
    "药材": "中药",
    "中药材": "中药",
    "作用": "功效",
    "功能": "功效",
    "适应证": "主治",
    "适应症": "主治",
    "注意事项": "禁忌",
    "来源": "药材来源",
}
RELATION_RE = re.compile(r"[^\w\u4e00-\u9fff]", re.UNICODE)


def normalize_entity_type(value: str) -> str:
    """旧管线：把实体类型规范到白名单，未知会降为「概念」。"""
    value = TYPE_ALIASES.get((value or "").strip(), (value or "").strip())
    return value if value in ENTITY_TYPES else "概念"


def normalize_relationship_type(value: str) -> str:
    """旧管线：清洗关系名以便写入 Cypher。"""
    cleaned = RELATION_RE.sub("_", (value or "相关").strip()).strip("_")
    cleaned = re.sub(r"_+", "_", cleaned)
    return cleaned[:40] or "相关"


def parse_json_response(text: str) -> dict:
    """从 LLM 回复中抽出 JSON 对象。"""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0]
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0]
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM 返回值不是 JSON 对象")
    data.setdefault("entities", [])
    data.setdefault("relationships", [])
    return data


def extraction_prompt(record: ChunkRecord) -> str:
    """拼出旧管线的抽取提示词。"""
    return f"""你是中药学知识图谱抽取专家。请严格根据给定教材片段抽取实体和关系，不补充片段外知识。

实体类型只允许：中药、功效、主治、性味、归经、用法用量、禁忌、配伍、药材来源、炮制、化学成分、方剂、典籍、人物、概念。
关系类型使用简短中文动词，例如：具有功效、主治、性味为、归经为、用量为、禁忌、配伍、来源于、炮制为、含有、记载于。
同一实体使用统一规范名称；不要把整句话当作实体；OCR 明显乱码不要抽取。

来源：PDF 第 {record.page_start}-{record.page_end} 页，章节：{record.section or '未识别'}
文本：
{record.text}

只返回以下 JSON，不要解释：
{{
  "entities": [{{"name": "实体名", "type": "允许的实体类型"}}],
  "relationships": [{{"source": "实体1", "target": "实体2", "type": "关系类型"}}]
}}"""


def _cache_path(record: ChunkRecord) -> Path:
    """返回该文本块抽取结果的本地缓存路径。"""
    digest = hashlib.sha256(record.text.encode("utf-8")).hexdigest()[:20]
    path = ROOT / ".cache" / "extractions" / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def extract_record(record: ChunkRecord, client: OpenAI, model: str) -> dict:
    """对单个文本块调用 LLM 抽取实体关系。"""
    cache_path = _cache_path(record)
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": extraction_prompt(record)}],
        temperature=0,
    )
    data = parse_json_response(response.choices[0].message.content or "")

    entities = []
    seen = set()
    for entity in data.get("entities", []):
        name = str(entity.get("name", "")).strip()
        if not name or len(name) > 80 or name in seen:
            continue
        seen.add(name)
        entities.append({"name": name, "type": normalize_entity_type(str(entity.get("type", "")))})

    names = {entity["name"] for entity in entities}
    relationships = []
    for relationship in data.get("relationships", []):
        source = str(relationship.get("source", "")).strip()
        target = str(relationship.get("target", "")).strip()
        if source in names and target in names and source != target:
            relationships.append(
                {
                    "source": source,
                    "target": target,
                    "type": normalize_relationship_type(str(relationship.get("type", ""))),
                }
            )

    normalized = {"entities": entities, "relationships": relationships}
    cache_path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
    return normalized


def _required_environment() -> tuple[str, str, str, str, str]:
    """检查旧建图所需的 Neo4j 与 LLM 环境变量。"""
    values = {
        "NEO4J_URL": os.getenv("NEO4J_URL", ""),
        "NEO4J_USER": os.getenv("NEO4J_USER", ""),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD", ""),
        "LLM_TOKEN": os.getenv("LLM_TOKEN", ""),
        "LLM_MODEL": os.getenv("LLM_MODEL", ""),
    }
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError(f".env 缺少配置：{', '.join(missing)}")
    return tuple(values[name] for name in values)  # type: ignore[return-value]


def _clear_database_enabled() -> bool:
    return (os.getenv("CLEAR_DATABASE") or "false").strip().lower() in {"1", "true", "yes", "on"}


def _ensure_legacy_write_allowed(session, *, clear: bool) -> None:
    if clear:
        return
    row = session.run(
        "RETURN EXISTS { MATCH (:IngestRun) } "
        "OR EXISTS { MATCH (:Assertion) } "
        "OR EXISTS { MATCH (c:Chunk) WHERE c.chunk_id IS NOT NULL } AS protected"
    ).single()
    if row is None or row["protected"]:
        raise RuntimeError(
            "目标库已有版本化图谱或稳定 Chunk，旧管线不能覆盖其原文、页码或向量。"
            "请使用 scripts/build_kg_v2.py 重建事实，或将新语料写入独立数据库。"
            "仅在确实需要删除目标库全部数据后重新初始化时，显式设置 CLEAR_DATABASE=true。"
        )


def write_graph(driver, records: list[ChunkRecord], extractions: list[dict]) -> None:
    """把旧管线抽取结果按名称写入 Neo4j。"""
    dimension = int(os.getenv("EMBED_DIM", "1536"))
    clear_database = _clear_database_enabled()
    database = os.getenv("NEO4J_DB") or None

    with driver.session(database=database) as session:
        _ensure_legacy_write_allowed(session, clear=clear_database)
        embedder = ExternalEmbedder(dimension=dimension)
        if clear_database:
            print("清空当前 Neo4j 数据库...")
            session.run("MATCH (n) DETACH DELETE n").consume()

        session.run(
            """
            MERGE (d:Document {id: $id})
            SET d.name = $name, d.source = $source
            """,
            id="zhongyaoxue-5e",
            name="中药学（新世纪第五版）",
            source=records[0].source if records else "zhongyaoxue.md",
        ).consume()

        for record, extraction in zip(records, extractions, strict=True):
            print(f"写入 Chunk {record.index + 1}/{len(records)}...", flush=True)
            embedding = embedder.embed_document(record.text)
            session.run(
                """
                MATCH (d:Document {id: 'zhongyaoxue-5e'})
                MERGE (c:Chunk {document_id: d.id, index: $index})
                SET c.text = $text, c.embedding = $embedding,
                    c.page_start = $page_start, c.page_end = $page_end,
                    c.section = $section, c.source = $source
                MERGE (c)-[:PART_OF]->(d)
                """,
                **record.to_dict(),
                embedding=embedding,
            ).consume()

            for entity in extraction["entities"]:
                label = normalize_entity_type(entity["type"])
                session.run(
                    f"""
                    MERGE (e:`{label}` {{name: $name}})
                    WITH e
                    MATCH (c:Chunk {{document_id: 'zhongyaoxue-5e', index: $index}})
                    MERGE (e)-[:FROM_CHUNK]->(c)
                    """,
                    name=entity["name"],
                    index=record.index,
                ).consume()

            for relationship in extraction["relationships"]:
                rel_type = normalize_relationship_type(relationship["type"])
                session.run(
                    f"""
                    MATCH (source {{name: $source}}), (target {{name: $target}})
                    MERGE (source)-[r:`{rel_type}`]->(target)
                    """,
                    source=relationship["source"],
                    target=relationship["target"],
                ).consume()


def build_knowledge_graph() -> None:
    """旧管线入口：分块抽取并全量写图。"""
    neo4j_url, neo4j_user, neo4j_password, llm_token, llm_model = _required_environment()
    records = records_from_environment()
    if not records:
        raise RuntimeError("语料没有生成任何文本块")
    print(f"读取到 {len(records)} 个文本块。")

    driver = GraphDatabase.driver(neo4j_url, auth=(neo4j_user, neo4j_password))
    try:
        driver.verify_connectivity()
        with driver.session(database=os.getenv("NEO4J_DB") or None) as session:
            _ensure_legacy_write_allowed(session, clear=_clear_database_enabled())

        client = OpenAI(api_key=llm_token, base_url=os.getenv("LLM_ENDPOINT") or None)
        extractions: list[dict] = []
        for position, record in enumerate(records, start=1):
            print(f"抽取实体关系 {position}/{len(records)}（PDF 页 {record.page_start}-{record.page_end}）...", flush=True)
            try:
                extractions.append(extract_record(record, client, llm_model))
            except Exception as exc:
                print(f"警告：Chunk {record.index} 抽取失败，将保留纯文本块：{exc}")
                extractions.append({"entities": [], "relationships": []})

        if not any(item["entities"] for item in extractions):
            raise RuntimeError("所有实体抽取均失败；数据库未被修改。请检查 LLM 配置和接口响应。")

        write_graph(driver, records, extractions)
        with driver.session(database=os.getenv("NEO4J_DB") or None) as session:
            nodes = session.run("MATCH (n) RETURN count(n) AS count").single()["count"]
            relationships = session.run("MATCH ()-[r]->() RETURN count(r) AS count").single()["count"]
        print(f"知识图谱构建完成：{nodes} 个节点，{relationships} 条关系。")
    finally:
        driver.close()


if __name__ == "__main__":
    build_knowledge_graph()
