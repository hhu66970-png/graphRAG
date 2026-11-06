"""IngestRun lifecycle: BUILDING -> ACTIVE / FAILED.

- 新构建写入 `IngestRun(status=BUILDING)`；
- 质量检查通过后**原子切换**为 ACTIVE，同一时刻只允许一个 ACTIVE；
- 失败构建标记 FAILED，旧 ACTIVE 版本继续服务；
- **禁止默认 DETACH DELETE 全库**——旧数据靠 run_id 隔离，不靠删除。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from ..domain.errors import RunStateError
from ..domain.ids import new_run_id
from .neo4j_store import Neo4jStore

BUILDING = "BUILDING"
ACTIVE = "ACTIVE"
FAILED = "FAILED"
SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True)
class RunHandle:
    run_id: str
    status: str
    created_at: str


def start_run(store: Neo4jStore, *, settings_snapshot: dict, notes: str = "") -> RunHandle:
    """Create a BUILDING run carrying the full config snapshot."""
    run_id = new_run_id()
    created_at = datetime.now(timezone.utc).isoformat()
    store.run(
        """
        CREATE (r:IngestRun {
            run_id: $run_id, status: $status, created_at: $created_at,
            settings_json: $settings_json, notes: $notes
        })
        """,
        run_id=run_id,
        status=BUILDING,
        created_at=created_at,
        settings_json=json.dumps(settings_snapshot, ensure_ascii=False, sort_keys=True),
        notes=notes,
    )
    return RunHandle(run_id=run_id, status=BUILDING, created_at=created_at)


def get_active_run(store: Neo4jStore) -> str | None:
    """读取当前标记为 ACTIVE 的构建版本。"""
    rows = store.run(
        "MATCH (r:IngestRun {status: $status}) RETURN r.run_id AS run_id "
        "ORDER BY r.created_at DESC LIMIT 1",
        status=ACTIVE,
    )
    return rows[0]["run_id"] if rows else None


def graph_revision(store: Neo4jStore) -> str:
    """Cache-invalidation token.

    **不得用节点数**：改名、改别名、改证据时节点数可能不变。这里用
    active_run_id + 该 run 下断言数与最新写入时间的组合。
    """
    rows = store.run(
        """
        MATCH (r:IngestRun {status: 'ACTIVE'})
        OPTIONAL MATCH (a:Assertion {run_id: r.run_id})
        RETURN r.run_id AS run_id, count(a) AS assertions,
               max(a.created_at) AS last_write, r.created_at AS run_created
        """
    )
    if not rows or not rows[0]["run_id"]:
        return "no-active-run"
    row = rows[0]
    return f"{row['run_id']}:{row['assertions']}:{row['last_write'] or row['run_created']}"


def _lock_run(tx, run_id: str, allowed: set[str]):
    row = tx.run(
        "MATCH (r:IngestRun {run_id: $run_id}) "
        "SET r.write_lock = coalesce(r.write_lock, 0) + 1 RETURN r.status AS status",
        run_id=run_id,
    ).single()
    if row is None:
        raise RunStateError(f"IngestRun 不存在: {run_id}")
    if row["status"] not in allowed:
        raise RunStateError(f"批次状态 {row['status']} 不允许此操作")


def mark_failed(store: Neo4jStore, run_id: str, reason: str) -> None:
    """Fail an unfinished build; never demote a published version."""
    def fail(tx):
        _lock_run(tx, run_id, {BUILDING})
        tx.run(
            "MATCH (r:IngestRun {run_id: $run_id}) "
            "SET r.status = $status, r.failed_reason = $reason, r.finished_at = $now",
            run_id=run_id, status=FAILED, reason=reason,
            now=datetime.now(timezone.utc).isoformat(),
        ).consume()

    with store.session() as session:
        session.execute_write(fail)


def _switch_active(
    store: Neo4jStore, run_id: str, *, allowed: set[str], quality_report: dict | None
) -> None:
    from .indexes import PUBLICATION_LOCK_CONSTRAINT

    # Also supports existing databases created before this constraint was introduced.
    store.run(PUBLICATION_LOCK_CONSTRAINT)

    def switch(tx):
        tx.run(
            "MERGE (p:GraphPublication {name: 'serving'}) "
            "SET p.revision = coalesce(p.revision, 0) + 1"
        ).consume()
        _lock_run(tx, run_id, allowed)
        now = datetime.now(timezone.utc).isoformat()
        tx.run(
            "MATCH (old:IngestRun {status: 'ACTIVE'}) WHERE old.run_id <> $run_id "
            "SET old.status = $superseded, old.superseded_at = $now",
            run_id=run_id, superseded=SUPERSEDED, now=now,
        ).consume()
        tx.run(
            "MATCH (target:IngestRun {run_id: $run_id}) "
            "SET target.status = $active, target.activated_at = $now "
            "SET target.quality_json = coalesce($quality, target.quality_json)",
            run_id=run_id, active=ACTIVE, now=now,
            quality=(json.dumps(quality_report, ensure_ascii=False, sort_keys=True)
                     if quality_report is not None else None),
        ).consume()

    with store.session() as session:
        session.execute_write(switch)


def activate(store: Neo4jStore, run_id: str, *, quality_report: dict) -> None:
    """Publish a BUILDING run under a database-wide publication lock."""
    _switch_active(store, run_id, allowed={BUILDING}, quality_report=quality_report)


def rollback_to(store: Neo4jStore, run_id: str) -> None:
    """Restore a retained version, including when no run is currently ACTIVE."""
    _switch_active(store, run_id, allowed={SUPERSEDED, ACTIVE}, quality_report=None)
