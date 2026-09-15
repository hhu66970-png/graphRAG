"""Build the v2 evidence knowledge graph from the chunks already in Neo4j.

管线（设计 §2.1 + §4，SKILL §3.1/§3.2/§3.7）：

    ensure_schema
      -> start_run(BUILDING)
      -> 迁移既有 Chunk 节点（补稳定 chunk_id，保留 embedding，纯增量）
      -> 并发抽取（磁盘缓存，可断点续跑）
      -> I1 链接 / 证据定位 / I2 验证  -> AssertionRecord
      -> 写 Entity + Assertion
      -> 重建类型化 Serving 投影
      -> 质量门槛 -> activate 或 mark_failed

旧图的自由文本关系不带 fact_key，投影清理只作用于带 fact_key 的边，
因此新旧数据在同一库中并存、互不干扰，随时可回滚。

    python scripts/build_kg_v2.py --limit 20            # 小样试跑
    python scripts/build_kg_v2.py --workers 6 --activate # 全量并激活
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from graphragexpr.config.settings import KGSettings  # noqa: E402
from graphragexpr.domain.ids import chunk_id as make_chunk_id  # noqa: E402
from graphragexpr.domain.ids import normalize_text  # noqa: E402
from graphragexpr.domain.models import ChunkRecord  # noqa: E402
from graphragexpr.domain.ontology import load_ontology  # noqa: E402
from graphragexpr.extract.extractor import ChunkExtractor  # noqa: E402
from graphragexpr.extract.llm_client import OpenAICompatibleClient  # noqa: E402
from graphragexpr.domain.ids import canonical_id  # noqa: E402
from graphragexpr.ingest.structure import (  # noqa: E402
    assign_entry_ids, build_entry_index, entry_coverage,
)
from graphragexpr.linking.entity_resolver import AliasStore, EntityResolver  # noqa: E402
from graphragexpr.storage import assertions as assertion_store  # noqa: E402
from graphragexpr.storage import runs as run_store  # noqa: E402
from graphragexpr.storage import serving_projection  # noqa: E402
from graphragexpr.storage.indexes import ensure_schema  # noqa: E402
from graphragexpr.storage.neo4j_store import Neo4jStore  # noqa: E402

DOCUMENT_ID = "zhongyaoxue-5e"
CACHE_DIR = ROOT / ".cache" / "kg_v2"

LOAD_CHUNKS = """
MATCH (c:Chunk)
WHERE c.text IS NOT NULL
RETURN c.index AS index, c.text AS text,
       c.page_start AS page_start, c.page_end AS page_end,
       coalesce(c.section, '') AS section
ORDER BY c.index
"""

MIGRATE_DOCUMENT = """
MATCH (d:Document) WHERE d.id = $legacy_id OR d.document_id = $document_id
SET d.document_id = $document_id, d.name = coalesce(d.name, $name)
RETURN count(d) AS migrated
"""

MIGRATE_CHUNK = """
UNWIND $batch AS row
MATCH (c:Chunk {index: row.index})
SET c.chunk_id       = row.chunk_id,
    c.document_id    = row.document_id,
    c.normalized_text= row.normalized_text,
    c.content_hash   = row.content_hash,
    c.section_path   = row.section_path,
    c.entry_id       = row.entry_id,
    c.order_in_entry = row.order_in_entry
"""


class CachingLLMClient:
    """Disk-cached wrapper around the real client (设计 §5 缓存键含版本).

    缓存键 = sha256(purpose + prompt + schema 名 + prompt_version + schema_version)。
    Prompt/Schema 版本一变缓存自动失效，符合 SKILL §3.7。
    """

    def __init__(self, inner: OpenAICompatibleClient, settings: KGSettings) -> None:
        """初始化该对象，保存后续调用所需的配置或依赖。"""
        self._inner = inner
        self._settings = settings
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def usage(self):
        """累计 LLM token 用量。"""
        return self._inner.usage

    @property
    def failures(self):
        """返回 LLM 失败清单。"""
        return self._inner.failures

    def _key(self, purpose: str, user: str, schema: type[BaseModel]) -> Path:
        """构造缓存键。"""
        raw = "␟".join([purpose, user, schema.__name__, *self._settings.cache_key_parts()])
        return CACHE_DIR / f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()[:32]}.json"

    def complete_json(self, *, system: str, user: str, schema: type[BaseModel], purpose: str):
        """调用 LLM 并校验返回值为指定 JSON Schema。"""
        path = self._key(purpose, user, schema)
        if path.exists():
            with self._lock:
                self.hits += 1
            return schema.model_validate_json(path.read_text(encoding="utf-8"))
        result = self._inner.complete_json(system=system, user=user, schema=schema, purpose=purpose)
        path.write_text(result.model_dump_json(), encoding="utf-8")
        with self._lock:
            self.misses += 1
        return result


class LockedResolver:
    """Thread-safe facade over EntityResolver (AliasStore 是共享可变状态)."""

    def __init__(self, inner: EntityResolver) -> None:
        """初始化该对象，保存后续调用所需的配置或依赖。"""
        self._inner = inner
        self._lock = threading.Lock()

    def resolve(self, surface: str, entity_type: str, *, context: str = ""):
        """解析实体表面名为 canonical 实体。"""
        with self._lock:
            return self._inner.resolve(surface, entity_type, context=context)


def load_chunks(
    store: Neo4jStore, limit: int | None
) -> tuple[list[ChunkRecord], dict[str, int]]:
    """Load chunks and attach herb-entry structure (设计 §2.1.2-D)。

    entry_id 由 ingest.structure 从 OCR 版式识别（药名 + 拼音（《典籍》）），
    是跨块补全的主依据；识别不到条目的前言/总论块 entry_id 为 None。

    Returns:
        (记录列表, chunk_id -> Neo4j 原始 index)。原始 index 必须单独保留：
        assign_entry_ids 会把 order_in_entry 重置为**条目内**序号，
        不能再拿它去 MATCH 既有节点。
    """
    rows = store.run(LOAD_CHUNKS)
    if limit:
        rows = rows[:limit]
    records: list[ChunkRecord] = []
    index_by_chunk: dict[str, int] = {}
    for row in rows:
        text = row["text"]
        section_path = [row["section"]] if row["section"] else []
        normalized = normalize_text(text)
        cid = make_chunk_id(DOCUMENT_ID, section_path, normalized)
        records.append(
            ChunkRecord(
                chunk_id=cid,
                document_id=DOCUMENT_ID,
                entry_id=None,
                section_path=section_path,
                order_in_entry=row["index"],
                text=text,
                normalized_text=normalized,
                page_start=row["page_start"] or 0,
                page_end=row["page_end"] or row["page_start"] or 0,
                content_hash=cid,
            )
        )
        index_by_chunk[cid] = row["index"]
    return assign_entry_ids(records, document_id=DOCUMENT_ID), index_by_chunk


def migrate_chunks(
    store: Neo4jStore, chunks: list[ChunkRecord],
    index_by_chunk: dict[str, int], batch_size: int,
) -> int:
    """Attach stable IDs to the existing Chunk nodes. 纯增量 SET，保留 embedding。"""
    store.run(MIGRATE_DOCUMENT, legacy_id=DOCUMENT_ID, document_id=DOCUMENT_ID,
              name="中药学（新世纪第五版）")
    rows = [
        {
            "index": index_by_chunk[c.chunk_id],
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "normalized_text": c.normalized_text,
            "content_hash": c.content_hash,
            "section_path": list(c.section_path),
            "entry_id": c.entry_id,
            "order_in_entry": c.order_in_entry,
        }
        for c in chunks
    ]
    return store.execute_batches(MIGRATE_CHUNK, rows, batch_size=batch_size)


def main() -> int:
    """命令行入口：解析参数并执行本脚本的主流程。"""
    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Build the v2 evidence KG")
    parser.add_argument("--limit", type=int, default=None, help="只处理前 N 个文本块")
    parser.add_argument("--workers", type=int, default=6, help="并发抽取线程数")
    parser.add_argument("--activate", action="store_true", help="质量达标则切为 ACTIVE")
    args = parser.parse_args()

    settings = KGSettings()
    ontology = load_ontology()
    store = Neo4jStore()
    started = time.time()

    try:
        store.verify()
        print(f"[1/7] 建立约束与索引 ...", flush=True)
        schema_info = ensure_schema(store, embedding_dim=512, rebuild_fulltext=False)
        print(f"      {schema_info}")

        run = run_store.start_run(
            store, settings_snapshot=settings.snapshot(), notes="v2 evidence KG build"
        )
        print(f"[2/7] 新建 IngestRun {run.run_id} (BUILDING)")

        chunks, index_by_chunk = load_chunks(store, args.limit)
        coverage = entry_coverage(chunks)
        print(f"[3/7] 载入 {len(chunks)} 个文本块｜识别药物条目 "
              f"{coverage['distinct_entries']} 个，覆盖 {coverage['coverage']:.1%}")

        migrated = migrate_chunks(store, chunks, index_by_chunk, settings.write_batch_size)
        print(f"[4/7] 迁移既有 Chunk 节点（补稳定 ID）: {migrated} 条")

        base_llm = OpenAICompatibleClient(settings)
        llm = CachingLLMClient(base_llm, settings)
        store_alias = AliasStore()
        resolver = LockedResolver(EntityResolver(store_alias, settings))

        all_assertions = []
        rejected = unknown_types = unknown_predicates = 0
        failed_chunks: list[tuple[int, str]] = []
        done = 0
        progress_lock = threading.Lock()

        def work(chunk: ChunkRecord):
            """建图子进程/线程中处理一批文本块。"""
            extractor = ChunkExtractor(
                ontology=ontology, settings=settings, resolver=resolver,
                llm=llm, run_id=run.run_id,
            )
            return chunk, extractor.extract(chunk)

        print(f"[5/7] 抽取中（{args.workers} 并发，命中缓存则不调用 LLM）...", flush=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(work, c): c for c in chunks}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    _, outcome = future.result()
                    all_assertions.extend(outcome.assertions)
                    rejected += len(outcome.rejected)
                    unknown_types += len(outcome.unknown_types)
                    unknown_predicates += len(outcome.unknown_predicates)
                except Exception as exc:  # noqa: BLE001 - 记录失败但不中断整批
                    failed_chunks.append((chunk.order_in_entry or -1, str(exc)[:120]))
                with progress_lock:
                    done += 1
                    if done % 25 == 0 or done == len(chunks):
                        print(f"      {done}/{len(chunks)} 块 | 断言 {len(all_assertions)} "
                              f"| 缓存命中 {llm.hits} | LLM 调用 {llm.usage.calls} "
                              f"| 失败块 {len(failed_chunks)}", flush=True)

        # ---- I3 跨块补全（设计 §2.1.1-3）--------------------------------
        # 只补「该条目锚点药在该块尚无断言」的块：教材条目里大量句子省略主语
        # （「本品味辛性凉…」），逐块抽取会丢掉或错认，补全把它们归回锚点药。
        cross_added = 0
        if settings.use_cross_chunk:
            entry_names = build_entry_index(chunks, document_id=DOCUMENT_ID)
            by_entry: dict[str, list[ChunkRecord]] = {}
            for c in chunks:
                if c.entry_id:
                    by_entry.setdefault(c.entry_id, []).append(c)
            covered = {(a.chunk_ids[0], a.subject_id) for a in all_assertions if a.chunk_ids}

            # 锚点必须是逐块抽取真实产出过的 HERB 实体：条目名若来自 OCR 截断，
            # 图里不会有对应的中药，此时补全等于把事实归给一个不存在的药。
            known_herbs = {
                a.subject_id for a in all_assertions
                if a.subject_id.startswith("HERB:")
            }
            targets: list[tuple[ChunkRecord, str, list[str], tuple[int, int]]] = []
            skipped_anchors: list[str] = []
            for entry_id, members in by_entry.items():
                name = entry_names.get(entry_id)
                if not name or len(members) < 2:
                    continue
                if canonical_id("HERB", name) not in known_herbs:
                    skipped_anchors.append(name)
                    continue
                anchor_id = canonical_id("HERB", name)
                ids = [m.chunk_id for m in members]
                pages = (min(m.page_start for m in members), max(m.page_end for m in members))
                for member in members:
                    if (member.chunk_id, anchor_id) not in covered:
                        targets.append((member, name, ids, pages))

            print(f"[5b/7] I3 跨块补全：{len(by_entry)} 个条目，待补全 {len(targets)} 块"
                  f"（跳过 {len(skipped_anchors)} 个未验证锚点）...", flush=True)
            if skipped_anchors:
                print(f"      跳过的锚点: {skipped_anchors[:10]}")

            def complete(item):
                """完成一次 LLM JSON 补全。"""
                chunk, name, ids, pages = item
                extractor = ChunkExtractor(
                    ontology=ontology, settings=settings, resolver=resolver,
                    llm=llm, run_id=run.run_id,
                )
                return extractor.extract_entry_continuation(
                    chunk, anchor_name=name, group_chunk_ids=ids, page_range=pages
                )

            done_c = 0
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(complete, t): t for t in targets}
                for future in as_completed(futures):
                    try:
                        outcome = future.result()
                        all_assertions.extend(outcome.assertions)
                        cross_added += len(outcome.assertions)
                        rejected += len(outcome.rejected)
                    except Exception as exc:  # noqa: BLE001
                        failed_chunks.append((futures[future][0].order_in_entry or -1,
                                              f"complete: {str(exc)[:100]}"))
                    done_c += 1
                    if done_c % 50 == 0 or done_c == len(targets):
                        print(f"      补全 {done_c}/{len(targets)} | 新增断言 {cross_added}",
                              flush=True)

        entities = store_alias.all_records()
        print(f"[6/7] 写入 {len(entities)} 个规范实体 / {len(all_assertions)} 条断言 "
              f"（其中跨块补全 {cross_added} 条）...")
        assertion_store.upsert_entities(
            store, entities, run_id=run.run_id, batch_size=settings.write_batch_size
        )
        written = assertion_store.upsert_assertions(
            store, all_assertions, batch_size=settings.write_batch_size
        )

        projection = serving_projection.rebuild_projection(
            store, ontology, run_id=run.run_id, batch_size=settings.write_batch_size
        )
        report = assertion_store.quality_report(store, run.run_id)
        report.update(
            {
                "chunks": len(chunks),
                "entry_coverage": coverage,
                "cross_chunk_added": cross_added,
                "skipped_anchors": len(skipped_anchors) if settings.use_cross_chunk else 0,
                "entities": len(entities),
                "rejected_candidates": rejected,
                "unknown_types": unknown_types,
                "unknown_predicates": unknown_predicates,
                "failed_chunks": len(failed_chunks),
                "projection": projection,
                "llm_calls": llm.usage.calls,
                "llm_tokens": llm.usage.total_tokens,
                "cache_hits": llm.hits,
                "elapsed_s": round(time.time() - started, 1),
            }
        )

        print(f"[7/7] 质量报告：")
        print(json.dumps(report, ensure_ascii=False, indent=2))

        failed_rate = report["failed_chunks"] / max(len(chunks), 1)
        report["failed_chunk_rate"] = round(failed_rate, 4)
        gate_failures: list[str] = []
        if report["evidence_coverage"] < settings.gate_min_evidence_coverage:
            gate_failures.append(
                f"证据覆盖率 {report['evidence_coverage']:.3f} < "
                f"{settings.gate_min_evidence_coverage}"
            )
        if failed_rate > settings.gate_max_failed_chunk_rate:
            gate_failures.append(
                f"块级失败率 {failed_rate:.1%} > {settings.gate_max_failed_chunk_rate:.1%}"
                f"（{report['failed_chunks']}/{len(chunks)} 块未抽取成功）"
            )
        if gate_failures:
            run_store.mark_failed(store, run.run_id, "；".join(gate_failures))
            print("✗ 未达发布门槛，run 标记 FAILED；旧 ACTIVE 继续服务")
            for item in gate_failures:
                print(f"    - {item}")
            print("  已抽取的断言保留在库中（按 run_id 隔离），补足额度后重跑将命中缓存续跑。")
            return 1

        if args.activate:
            run_store.activate(store, run.run_id, quality_report=report)
            print(f"✓ run {run.run_id} 已切换为 ACTIVE")
        else:
            print(f"✓ 质量达标；未加 --activate，run 仍为 BUILDING（可稍后激活）")

        if failed_chunks:
            print(f"\n失败块清单（前 10 条）：")
            for index, error in failed_chunks[:10]:
                print(f"  chunk[{index}]: {error}")
        return 0
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
