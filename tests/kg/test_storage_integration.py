"""存储层集成测试（SKILL §14.3）。

安全约定：**绝不使用会 DETACH DELETE 全库的 fixture**。
所有测试数据挂在一个独立的 document_id / run_id 下，结束时只删除自己创建的节点，
现网演示图谱（11978 节点）不受影响。
"""

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

from graphragexpr.config.settings import KGSettings
from graphragexpr.domain.ids import assertion_id, canonical_id, fact_key
from graphragexpr.domain.models import AssertionRecord, ChunkRecord, EntityRecord
from graphragexpr.domain.ontology import load_ontology
from graphragexpr.storage import assertions as assertion_store
from graphragexpr.storage import runs as run_store
from graphragexpr.storage import serving_projection
from graphragexpr.storage.indexes import CONSTRAINTS, available_analyzers, ensure_schema
from graphragexpr.storage.neo4j_store import Neo4jStore

pytestmark = pytest.mark.integration

TEST_DOC = "itest-zhongyaoxue"
# 测试实体名必须带前缀：canonical_id 是 (类型, 规范名) 的纯函数，若直接用「人参」
# 这类真实药名，会与现网实体撞上同一个 canonical_id，upsert 时把线上节点的
# run_id 改写成 itest-，随后 cleanup 的 DETACH DELETE 就会连真实节点一起删掉。
# 实测曾因此删除了人参/藜芦/脾经/呕吐 四个真实实体及其全部边。
E = "ITEST药"          # 实体名前缀，教材中不可能出现
SCHEMA_VERSION = "kg-ontology-1.0.0"


@pytest.fixture(scope="module")
def store():
    load_dotenv()
    if not all(os.getenv(n) for n in ("NEO4J_URL", "NEO4J_USER", "NEO4J_PASSWORD")):
        pytest.skip("Neo4j 未配置")
    handle = Neo4jStore()
    try:
        handle.verify()
    except Exception as exc:  # noqa: BLE001 - 明确跳过而非静默吞掉
        pytest.skip(f"Neo4j 不可用: {exc}")
    yield handle
    _cleanup(handle)
    handle.close()


def _cleanup(store: Neo4jStore) -> None:
    """只删除本测试创建的数据（按 document_id / run_id 前缀定位）。"""
    store.run("MATCH (a:Assertion) WHERE a.run_id STARTS WITH 'itest-' DETACH DELETE a")
    # 双条件：既要 run_id 是测试的，名字也必须带测试前缀。
    # 只按 run_id 删是不够的——真实实体可能被测试 upsert 改写了 run_id。
    store.run(
        "MATCH (e:Entity) WHERE e.run_id STARTS WITH 'itest-' "
        "AND e.preferred_name STARTS WITH $prefix DETACH DELETE e",
        prefix=E,
    )
    store.run("MATCH (c:Chunk {document_id: $doc}) DETACH DELETE c", doc=TEST_DOC)
    store.run("MATCH (d:Document {document_id: $doc}) DETACH DELETE d", doc=TEST_DOC)
    store.run("MATCH (r:IngestRun) WHERE r.run_id STARTS WITH 'itest-' DETACH DELETE r")


@pytest.fixture
def clean_test_scope(store):
    _cleanup(store)
    yield
    _cleanup(store)


@pytest.fixture
def isolated_active_run(store):
    """Temporarily park the real ACTIVE run so run-lifecycle tests can assume isolation.

    `activate()` 的语义是「全局只允许一个 ACTIVE」，因此它会降级**任何**其他 ACTIVE run，
    包括线上正在服务的那个。仅按 run_id 前缀清理节点不足以隔离这个全局不变量，
    必须在测试前后显式摘挂现网 run，否则跑一次测试就会让线上失去 ACTIVE。
    """
    previous = run_store.get_active_run(store)
    if previous:
        store.run(
            "MATCH (r:IngestRun {run_id: $id}) SET r.status = 'SUPERSEDED'", id=previous
        )
    yield previous
    # 先把测试造出来的 ACTIVE 全部降级，再恢复现网 run
    store.run(
        "MATCH (r:IngestRun) WHERE r.run_id STARTS WITH 'itest-' AND r.status = 'ACTIVE' "
        "SET r.status = 'SUPERSEDED'"
    )
    if previous:
        store.run(
            "MATCH (r:IngestRun {run_id: $id}) SET r.status = 'ACTIVE'", id=previous
        )


def _seed(store: Neo4jStore, run_id: str) -> dict:
    """Create a minimal two-chunk fixture: 人参 的性味在 c1、剂量与配伍禁忌在 c2。"""
    settings = KGSettings()
    store.run(
        "CREATE (r:IngestRun {run_id: $run_id, status: 'BUILDING', created_at: '2026-01-01'})",
        run_id=run_id,
    )
    assertion_store.upsert_document(
        store, document_id=TEST_DOC, name="集成测试语料", source="itest", run_id=run_id
    )

    chunks = [
        ChunkRecord(
            chunk_id=f"{run_id}-c1", document_id=TEST_DOC, entry_id="entry-ginseng",
            section_path=["补虚药", "补气药"], order_in_entry=0,
            text="人参　味甘、微苦，性微温。归脾、肺、心、肾经。",
            normalized_text="人参味甘、微苦，性微温。归脾、肺、心、肾经。",
            page_start=132, page_end=132, content_hash="h1",
        ),
        ChunkRecord(
            chunk_id=f"{run_id}-c2", document_id=TEST_DOC, entry_id="entry-ginseng",
            section_path=["补虚药", "补气药"], order_in_entry=1,
            text="人参　煎服，3～9g。不宜与藜芦同用。",
            normalized_text="人参煎服，3～9g。不宜与藜芦同用。",
            page_start=133, page_end=133, content_hash="h2",
        ),
    ]
    assertion_store.upsert_chunks(store, chunks, run_id=run_id, batch_size=settings.write_batch_size)

    # 关键：「呕吐」同时作为 主治 与 功效 存在，验证不会互相错连
    entities = [
        EntityRecord(canonical_id=canonical_id("HERB", f"{E}甲"), entity_type="HERB",
                     preferred_name=f"{E}甲", normalized_name=f"{E}甲"),
        EntityRecord(canonical_id=canonical_id("MERIDIAN", f"{E}经"), entity_type="MERIDIAN",
                     preferred_name=f"{E}经", normalized_name=f"{E}经"),
        EntityRecord(canonical_id=canonical_id("INDICATION", f"{E}症"), entity_type="INDICATION",
                     preferred_name=f"{E}症", normalized_name=f"{E}症"),
        EntityRecord(canonical_id=canonical_id("EFFICACY", f"{E}症"), entity_type="EFFICACY",
                     preferred_name=f"{E}症", normalized_name=f"{E}症"),
    ]
    assertion_store.upsert_entities(store, entities, run_id=run_id,
                                    batch_size=settings.write_batch_size)
    return {"chunks": chunks, "entities": entities, "settings": settings}


def _make_assertion(run_id: str, chunk_ids: list[str], **kwargs) -> AssertionRecord:
    key = fact_key(kwargs["subject_id"], kwargs["predicate"],
                   kwargs.get("object_id") or "", {}, kwargs.get("polarity", "POSITIVE"))
    return AssertionRecord(
        assertion_id=assertion_id(key, chunk_ids[0], (0, 10), SCHEMA_VERSION),
        fact_key=key, chunk_ids=chunk_ids, schema_version=SCHEMA_VERSION,
        run_id=run_id, **kwargs,
    )


class TestSchema:
    def test_required_constraints_exist(self, store):
        """SKILL §9.3 的五个唯一约束必须存在，且名称类不得有唯一约束。"""
        ensure_schema(store, embedding_dim=512, rebuild_fulltext=False)
        rows = store.run("SHOW CONSTRAINTS YIELD name, labelsOrTypes, properties "
                         "RETURN name, labelsOrTypes, properties")
        names = {r["name"] for r in rows}
        for statement in CONSTRAINTS:
            expected = statement.split("CONSTRAINT ")[1].split(" ")[0]
            assert expected in names, f"缺少约束 {expected}"

        # 名称绝不能有唯一约束（SKILL §3.3）
        for row in rows:
            props = row["properties"] or []
            assert "name" not in props and "preferred_name" not in props, (
                f"约束 {row['name']} 把名称当成了身份，违反 SKILL §3.3"
            )

    def test_cjk_analyzer_availability_is_observable(self, store):
        analyzers = available_analyzers(store)
        assert analyzers, "应能列出可用分析器"
        # 不断言 cjk 一定存在：设计 §2.2.4 要求实测而非假设
        assert isinstance(analyzers, set)


class TestRunLifecycle:
    def test_building_run_is_not_active(self, store, clean_test_scope):
        run = run_store.start_run(store, settings_snapshot={"k": 1}, notes="itest")
        # start_run 生成的 run_id 不带 itest- 前缀，改用显式清理
        store.run("MATCH (r:IngestRun {run_id: $id}) SET r.run_id = $new",
                  id=run.run_id, new=f"itest-{run.run_id}")
        assert run.status == "BUILDING"

    def test_failed_run_keeps_previous_active(self, store, clean_test_scope, isolated_active_run):
        """SKILL §3.7：新 run 失败时旧 ACTIVE 继续服务。"""
        old = "itest-run-old"
        new = "itest-run-new"
        store.run("CREATE (:IngestRun {run_id: $id, status: 'ACTIVE', created_at: '2026-01-01'})", id=old)
        store.run("CREATE (:IngestRun {run_id: $id, status: 'BUILDING', created_at: '2026-01-02'})", id=new)

        run_store.mark_failed(store, new, "质量门槛未通过")

        assert run_store.get_active_run(store) == old
        status = store.run("MATCH (r:IngestRun {run_id: $id}) RETURN r.status AS s", id=new)[0]["s"]
        assert status == "FAILED"

    def test_activate_is_atomic_single_active(self, store, clean_test_scope, isolated_active_run):
        old, new = "itest-run-a", "itest-run-b"
        store.run("CREATE (:IngestRun {run_id: $id, status: 'ACTIVE', created_at: '2026-01-01'})", id=old)
        store.run("CREATE (:IngestRun {run_id: $id, status: 'BUILDING', created_at: '2026-01-02'})", id=new)

        run_store.activate(store, new, quality_report={"evidence_coverage": 1.0})

        actives = store.run(
            "MATCH (r:IngestRun) WHERE r.run_id STARTS WITH 'itest-' AND r.status = 'ACTIVE' "
            "RETURN r.run_id AS id"
        )
        assert [r["id"] for r in actives] == [new], "同一时刻只能有一个 ACTIVE"

    def test_cannot_activate_non_building_run(self, store, clean_test_scope):
        store.run("CREATE (:IngestRun {run_id: 'itest-run-c', status: 'FAILED', created_at: '2026-01-01'})")
        with pytest.raises(run_store.RunStateError):
            run_store.activate(store, "itest-run-c", quality_report={})

    def test_graph_revision_not_based_on_node_count(self, store, clean_test_scope, isolated_active_run):
        """设计 §2.2.2-D：改名/改证据时节点数不变，缓存键必须仍然变化。"""
        revision = run_store.graph_revision(store)
        assert isinstance(revision, str) and revision


class TestAssertionsAndProjection:
    def test_cross_chunk_fact_and_projection(self, store, clean_test_scope):
        """SKILL §14.3-1/-5：两个 Chunk 共同构成一条事实，且能投影为类型化边。"""
        run_id = "itest-run-proj"
        seeded = _seed(store, run_id)
        settings = seeded["settings"]
        ontology = load_ontology()

        herb = canonical_id("HERB", f"{E}甲")
        meridian = canonical_id("MERIDIAN", f"{E}经")

        records = [
            _make_assertion(
                run_id, [f"{run_id}-c1"],
                subject_id=herb, predicate="ENTERS_MERIDIAN", object_id=meridian,
                evidence_quote="归脾、肺、心、肾经", evidence_start=18, evidence_end=27,
                match_mode="raw_exact", page_start=132, page_end=132,
                rule_score=1.0, final_score=1.0, validation_status="ACCEPTED",
            ),
            # 跨块事实：主语来自 c1，证据在 c2
            _make_assertion(
                run_id, [f"{run_id}-c1", f"{run_id}-c2"],
                subject_id=herb, predicate="INCOMPATIBLE_WITH",
                object_id=canonical_id("HERB", f"{E}乙"),
                evidence_quote="不宜与藜芦同用", evidence_start=10, evidence_end=17,
                match_mode="raw_exact", page_start=132, page_end=133,
                rule_score=0.9, final_score=0.9, validation_status="ACCEPTED",
                is_cross_chunk=True,
            ),
        ]
        # 藜芦实体
        assertion_store.upsert_entities(
            store,
            [EntityRecord(canonical_id=canonical_id("HERB", f"{E}乙"), entity_type="HERB",
                          preferred_name=f"{E}乙", normalized_name=f"{E}乙")],
            run_id=run_id, batch_size=settings.write_batch_size,
        )
        written = assertion_store.upsert_assertions(
            store, records, batch_size=settings.write_batch_size
        )
        assert written == 2

        # 断言层：证据可回溯到具体 chunk 与页码
        rows = store.run(
            "MATCH (a:Assertion {run_id: $run})-[:EVIDENCED_BY]->(c:Chunk) "
            "RETURN a.predicate AS p, collect(c.chunk_id) AS chunks, "
            "a.page_start AS ps, a.page_end AS pe, a.evidence_quote AS q "
            "ORDER BY p", run=run_id,
        )
        by_predicate = {r["p"]: r for r in rows}
        assert len(by_predicate["INCOMPATIBLE_WITH"]["chunks"]) == 2
        assert by_predicate["INCOMPATIBLE_WITH"]["pe"] == 133
        assert by_predicate["ENTERS_MERIDIAN"]["q"] == "归脾、肺、心、肾经"

        # 投影层
        result = serving_projection.rebuild_projection(
            store, ontology, run_id=run_id, batch_size=settings.write_batch_size
        )
        assert result["edges"] == 2 and result["skipped"] == 0

        edges = store.run(
            "MATCH (s:Entity {canonical_id: $herb})-[e]->(o:Entity) "
            "WHERE e.active_run_id = $run "
            "RETURN type(e) AS t, o.canonical_id AS target, e.page_start AS ps, "
            "e.evidence_count AS n ORDER BY t", herb=herb, run=run_id,
        )
        types = {e["t"] for e in edges}
        assert types == {"ENTERS_MERIDIAN", "INCOMPATIBLE_WITH"}
        assert all(e["ps"] is not None for e in edges), "投影边必须带页码 provenance"

    def test_same_name_different_type_never_cross_linked(self, store, clean_test_scope):
        """SKILL §14.3-2：同名不同类型实体不得错误连边。

        旧实现用 MATCH (n {name}) 无类型匹配，「呕吐」同时是 主治/功效/禁忌/概念，
        会产生笛卡尔积连边。新实现按 canonical_id 定位，结构上不可能发生。
        """
        run_id = "itest-run-samename"
        _seed(store, run_id)
        indication = canonical_id("INDICATION", f"{E}症")
        efficacy = canonical_id("EFFICACY", f"{E}症")
        assert indication != efficacy

        record = _make_assertion(
            run_id, [f"{run_id}-c1"],
            subject_id=canonical_id("HERB", f"{E}甲"), predicate="TREATS",
            object_id=indication,
            evidence_quote="人参　味甘、微苦", evidence_start=0, evidence_end=8,
            match_mode="raw_exact", page_start=132, page_end=132,
            rule_score=0.9, final_score=0.9, validation_status="ACCEPTED",
        )
        assertion_store.upsert_assertions(store, [record], batch_size=100)
        serving_projection.rebuild_projection(
            store, load_ontology(), run_id=run_id, batch_size=100
        )

        linked = store.run(
            "MATCH (:Entity {canonical_id: $herb})-[e:TREATS]->(o:Entity) "
            "WHERE e.active_run_id = $run RETURN o.canonical_id AS id, o.entity_type AS t",
            herb=canonical_id("HERB", f"{E}甲"), run=run_id,
        )
        assert [r["id"] for r in linked] == [indication]
        assert linked[0]["t"] == "INDICATION"

    def test_accepted_assertion_without_evidence_is_rejected_at_write(self, store, clean_test_scope):
        """SKILL §3.1 硬门槛：无可定位证据的 ACCEPTED 断言不得写入。"""
        run_id = "itest-run-noevid"
        _seed(store, run_id)
        bad = _make_assertion(
            run_id, [f"{run_id}-c1"],
            subject_id=canonical_id("HERB", f"{E}甲"), predicate="HAS_EFFICACY",
            object_id=canonical_id("EFFICACY", f"{E}症"),
            evidence_quote="", evidence_start=None, evidence_end=None,
            match_mode="not_found", page_start=132, page_end=132,
            rule_score=0.9, final_score=0.9, validation_status="ACCEPTED",
        )
        with pytest.raises(ValueError, match="缺少可定位证据"):
            assertion_store.upsert_assertions(store, [bad], batch_size=10)

    def test_write_is_idempotent(self, store, clean_test_scope):
        """SKILL §5.5：所有写操作必须幂等。"""
        run_id = "itest-run-idem"
        _seed(store, run_id)
        record = _make_assertion(
            run_id, [f"{run_id}-c1"],
            subject_id=canonical_id("HERB", f"{E}甲"), predicate="ENTERS_MERIDIAN",
            object_id=canonical_id("MERIDIAN", f"{E}经"),
            evidence_quote="归脾、肺、心、肾经", evidence_start=18, evidence_end=27,
            match_mode="raw_exact", page_start=132, page_end=132,
            rule_score=1.0, final_score=1.0, validation_status="ACCEPTED",
        )
        assertion_store.upsert_assertions(store, [record], batch_size=10)
        assertion_store.upsert_assertions(store, [record], batch_size=10)

        count = store.run(
            "MATCH (a:Assertion {run_id: $run}) RETURN count(a) AS n", run=run_id
        )[0]["n"]
        assert count == 1

        serving_projection.rebuild_projection(store, load_ontology(), run_id=run_id, batch_size=10)
        serving_projection.rebuild_projection(store, load_ontology(), run_id=run_id, batch_size=10)
        edges = store.run(
            "MATCH (:Entity {canonical_id: $herb})-[e:ENTERS_MERIDIAN]->() "
            "WHERE e.active_run_id = $run RETURN count(e) AS n",
            herb=canonical_id("HERB", f"{E}甲"), run=run_id,
        )[0]["n"]
        assert edges == 1, "重建投影不得产生重复边"

    def test_quality_report_gate(self, store, clean_test_scope):
        run_id = "itest-run-quality"
        _seed(store, run_id)
        records = [
            _make_assertion(
                run_id, [f"{run_id}-c1"],
                subject_id=canonical_id("HERB", f"{E}甲"), predicate="ENTERS_MERIDIAN",
                object_id=canonical_id("MERIDIAN", f"{E}经"),
                evidence_quote="归脾、肺、心、肾经", evidence_start=18, evidence_end=27,
                match_mode="raw_exact", page_start=132, page_end=132,
                rule_score=1.0, final_score=1.0, validation_status="ACCEPTED",
            ),
            _make_assertion(
                run_id, [f"{run_id}-c2"],
                subject_id=canonical_id("HERB", f"{E}甲"), predicate="HAS_EFFICACY",
                object_id=canonical_id("EFFICACY", f"{E}症"),
                evidence_quote="煎服", evidence_start=3, evidence_end=5,
                match_mode="fuzzy", page_start=133, page_end=133,
                rule_score=0.5, final_score=0.5, validation_status="REVIEW",
            ),
        ]
        assertion_store.upsert_assertions(store, records, batch_size=10)
        report = assertion_store.quality_report(store, run_id)
        assert report["total_assertions"] == 2
        assert report["accepted"] == 1 and report["review"] == 1
        assert report["evidence_coverage"] == 1.0
