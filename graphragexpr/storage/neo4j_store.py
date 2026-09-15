"""Neo4j driver lifecycle and batched writes (SKILL §9.1 / §9.2).

规则：
  - 进程生命周期内复用一个 Driver；
  - 每次事务显式指定 database；
  - 不在模块导入时建立网络连接；
  - 批量写入统一走 UNWIND，批大小配置化。
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Sequence

from neo4j import Driver, GraphDatabase


class Neo4jStore:
    """Thin wrapper owning the driver and providing batched write helpers."""

    def __init__(
        self,
        url: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
    ) -> None:
        """初始化该对象，保存后续调用所需的配置或依赖。"""
        self._url = url or os.getenv("NEO4J_URL")
        self._user = user or os.getenv("NEO4J_USER")
        self._password = password or os.getenv("NEO4J_PASSWORD")
        self._database = database if database is not None else (os.getenv("NEO4J_DB") or None)
        if not all([self._url, self._user, self._password]):
            raise RuntimeError("缺少 NEO4J_URL / NEO4J_USER / NEO4J_PASSWORD")
        self._driver: Driver | None = None

    @property
    def driver(self) -> Driver:
        """Lazily create the driver; never connect at import time."""
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self._url, auth=(self._user, self._password)  # type: ignore[arg-type]
            )
        return self._driver

    @property
    def database(self) -> str | None:
        """返回当前会话使用的 Neo4j 数据库名。"""
        return self._database

    def verify(self) -> None:
        """调用驱动探测 Bolt 连通性。"""
        self.driver.verify_connectivity()

    @contextmanager
    def session(self) -> Iterator[Any]:
        """打开指定 database 的 Neo4j 会话。"""
        with self.driver.session(database=self._database) as session:
            yield session

    def run(self, cypher: str, **params: Any) -> list[dict]:
        """Run a single statement and materialise the result."""
        with self.session() as session:
            return [dict(record) for record in session.run(cypher, **params)]

    def execute_batches(
        self, cypher: str, rows: Sequence[dict], *, batch_size: int, param_name: str = "batch"
    ) -> int:
        """Apply `cypher` over `rows` in UNWIND batches. Returns rows written."""
        written = 0
        with self.session() as session:
            for start in range(0, len(rows), batch_size):
                chunk = list(rows[start : start + batch_size])
                session.run(cypher, **{param_name: chunk}).consume()
                written += len(chunk)
        return written

    def close(self) -> None:
        """关闭底层 Driver 连接池。"""
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def __enter__(self) -> "Neo4jStore":
        """进入 with 语句时返回自身。"""
        return self

    def __exit__(self, *exc: object) -> None:
        """退出 with 语句时关闭连接或释放资源。"""
        self.close()


def chunked(items: Iterable[dict], size: int) -> Iterator[list[dict]]:
    """按批大小切分写入行。"""
    batch: list[dict] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
