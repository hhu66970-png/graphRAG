"""IngestRun lifecycle: BUILDING -> ACTIVE / FAILED (SKILL §3.7, 设计 §2.1.3).

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
    """RunHandle：本文件定义的数据结构或组件。"""
    run_id: str
    status: str
    created_at: str


def start_run(store: Neo4jStore, *, settings_snapshot: dict, notes: str = "") -> RunHandle:
    """Create a BUILDING run carrying the full config snapshot (SKILL §5.1)."""
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
    """Cache-invalidation token (设计 §2.2.2-D).

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


def mark_failed(store: Neo4jStore, run_id: str, reason: str) -> None:
    """Mark a run FAILED; the previous ACTIVE run keeps serving."""
    store.run(
        "MATCH (r:IngestRun {run_id: $run_id}) "
        "SET r.status = $status, r.failed_reason = $reason, r.finished_at = $now",
        run_id=run_id, status=FAILED, reason=reason,
        now=datetime.now(timezone.utc).isoformat(),
    )


def activate(store: Neo4jStore, run_id: str, *, quality_report: dict) -> None:
    """Atomically switch `run_id` to ACTIVE and demote the previous one.

    Args:
        store: 连接句柄。
        run_id: 必须处于 BUILDING 状态。
        quality_report: 发布门槛检查结果，随 run 一起留档（设计 §9.3）。

    Raises:
        RunStateError: run 不存在或不处于 BUILDING。

    Side effects: 单条语句内完成降级 + 升级，避免出现零个或两个 ACTIVE。
    """
    rows = store.run(
        "MATCH (r:IngestRun {run_id: $run_id}) RETURN r.status AS status", run_id=run_id
    )
    if not rows:
        raise RunStateError(f"IngestRun 不存在: {run_id}")
    if rows[0]["status"] != BUILDING:
        raise RunStateError(f"只能激活 BUILDING 状态的 run，当前为 {rows[0]['status']}")

    store.run(
        """
        MATCH (target:IngestRun {run_id: $run_id})
        WITH target
        OPTIONAL MATCH (old:IngestRun {status: 'ACTIVE'}) WHERE old.run_id <> $run_id
        SET old.status = $superseded, old.superseded_at = $now
        WITH target
        SET target.status = $active, target.activated_at = $now,
            target.quality_json = $quality
        """,
        run_id=run_id, active=ACTIVE, superseded=SUPERSEDED,
        now=datetime.now(timezone.utc).isoformat(),
        quality=json.dumps(quality_report, ensure_ascii=False, sort_keys=True),
    )


def rollback_to(store: Neo4jStore, run_id: str) -> None:
    """Re-activate a previously superseded run (回滚)."""
    rows = store.run(
        "MATCH (r:IngestRun {run_id: $run_id}) RETURN r.status AS status", run_id=run_id
    )
    if not rows:
        raise RunStateError(f"IngestRun 不存在: {run_id}")
    if rows[0]["status"] not in {SUPERSEDED, ACTIVE}:
        raise RunStateError(f"只能回滚到 SUPERSEDED/ACTIVE 的 run，当前为 {rows[0]['status']}")
    store.run(
        """
        MATCH (old:IngestRun {status: 'ACTIVE'}) WHERE old.run_id <> $run_id
        SET old.status = $superseded, old.superseded_at = $now
        WITH 1 AS _
        MATCH (target:IngestRun {run_id: $run_id})
        SET target.status = $active, target.activated_at = $now
        """,
        run_id=run_id, active=ACTIVE, superseded=SUPERSEDED,
        now=datetime.now(timezone.utc).isoformat(),
    )
