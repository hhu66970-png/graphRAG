"""接口层调用顺序测试的全部替身（**不连数据库、不联网、不加载模型**）。

设计原则
--------
1. **只替最外层。** 检索部分、Neo4j 句柄、普通 LLM 客户端被替换，
   但高风险核验（条件单元 → 候选结论 → 程序核验 → 状态 → 渲染）全部是
   生产代码。因此这些用例检查的是「接口层接线」，而不是核验算法本身。
2. **模型仓库探测要能立刻失败。** `_configured_embedder` 一旦被调用就直接抛错：
   真实嵌入器会去模型仓库联网，那属于**环境失败**，混进来会让
   「顺序接错了」看起来像「这台机器没网」。本文件的用例必须在无网、
   无 Neo4j、无本地模型的机器上同样全绿。
3. **记录调用，而不是靠读代码相信。** 普通 LLM、两种检索、实体链接各自留下
   调用记录，「高风险问题没有跑普通生成」「所选 method 的检索确实执行了」
   因此可以被断言。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphragexpr.highrisk import is_high_risk_question  # noqa: E402
from tests.high_risk.conftest import (  # noqa: E402
    FakeHighRiskStore,
    assertion_row,
    dosage_row,
)

#: 普通（非高风险）自由文本生成的替身文本。高风险响应里出现它即失败：
#: 那说明未经核验的模型文本被返回了。
FREE_TEXT = "【替身普通 LLM 的自由生成文本，高风险管理链路上绝不应出现】"

#: 含「用量」，命中高风险词表。
HIGH_RISK_QUESTION = "麻黄用量是多少？"
#: 含「孕妇」，命中高风险词表，且证据本身讲的就是孕妇。
POPULATION_QUESTION = "麻黄孕妇能用吗？"
#: 刻意不含任何高风险词：用于「非高风险请求保持原行为」的对照。
BENIGN_QUESTION = "麻黄的功效是什么？"

MAHUANG_ID = "H:麻黄"

#: 干麻黄：煎服 3～9g（与既有验收测试同一原文）。
DOSAGE_QUOTE = "麻黄煎服，3～9g。"
DOSAGE_PAGE = 82


def dosage_rows() -> list[dict]:
    """测试夹具：构造剂量断言行。"""
    return [dosage_row("a-dosage", "煎服，3～9g。", quote=DOSAGE_QUOTE)]


def contraindication_rows() -> list[dict]:
    """带对象的禁忌行：让最终证据里**既有主语也有宾语**，可断言关系边。"""
    return [
        assertion_row(
            assertion_id="a-contra",
            predicate="CONTRAINDICATED_IN",
            polarity="NEGATIVE",
            object_id="P:孕妇",
            object_name="孕妇",
            evidence_quote="孕妇慎用。",
            chunk_texts=["孕妇慎用。"],
        )
    ]


@dataclass
class FakeItem:
    """`EvidencePacket.items` 的最小形态（字段名与真实 EvidenceItem 一致）。"""

    evidence_id: str = "e-1"
    page_start: int = DOSAGE_PAGE
    page_end: int = DOSAGE_PAGE
    evidence_quote: str = DOSAGE_QUOTE
    predicate: str | None = "HAS_DOSAGE"
    path: str | None = None
    subject_id: str | None = MAHUANG_ID
    object_id: str | None = None
    retrieval_score: float = 1.0
    chunk_id: str | None = "c-1"
    qualifiers: dict = field(default_factory=dict)


@dataclass
class FakePacket:
    """`EvidencePacket` 的最小形态：auto 检索的返回物。"""

    intent: str = "dosage_query"
    linked_entities: list = field(default_factory=list)
    items: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    trace: dict = field(default_factory=dict)


def linked_herb(name: str = "麻黄", canonical_id: str = MAHUANG_ID) -> list[dict]:
    """`EvidencePacket.linked_entities` 的行（检索层用 `preferred_name`）。"""
    return [
        {
            "canonical_id": canonical_id,
            "entity_type": "HERB",
            "preferred_name": name,
        }
    ]


def linked_herb_rows(name: str = "麻黄", canonical_id: str = MAHUANG_ID) -> list[dict]:
    """`link_entities_in_text` 的行（链接器用 `name`）。"""
    return [{"canonical_id": canonical_id, "entity_type": "HERB", "name": name}]


class ApiFakeStore(FakeHighRiskStore):
    """在 `FakeHighRiskStore` 上补出接口层会用到的那几条只读查询。

    仍然不连数据库：ACTIVE 批次与实体链接都由预设行回答，使接口层的顺序约束
    可以在没有 Neo4j 的环境里被断言。`close()` 被计数，
    「谁开的句柄谁关」因此也是可测的。
    """

    def __init__(
        self,
        rows: list[dict] | None = None,
        *,
        active_run_id: str | None = "run-test",
        linked_rows: list[dict] | None = None,
    ) -> None:
        super().__init__(rows)
        self.active_run_id = active_run_id
        self.linked_rows = list(linked_rows or [])
        self.closed = 0

    def run(self, cypher: str, **params):
        """测试替身：执行一条 Cypher 并返回假结果。"""
        if "IngestRun" in cypher:
            self.calls.append({"cypher": cypher, **params})
            return [{"run_id": self.active_run_id}] if self.active_run_id else []
        # 先判「更长实体名」：它的条件与链接查询不重叠，但顺序固定下来更清楚。
        if "size(surface) > size($name)" in cypher:
            self.calls.append({"cypher": cypher, **params})
            return []
        if "size(surface) >= 2" in cypher:  # _LINK_IN_TEXT
            self.calls.append({"cypher": cypher, **params})
            return [dict(row) for row in self.linked_rows]
        return super().run(cypher, **params)

    def close(self) -> None:
        """记录关闭次数。"""
        self.closed += 1


class NormalLLM:
    """普通（非高风险）自由文本生成的替身，记录它有没有被调用。

    记录「构造了几次、生成了几次」比记录生成内容更重要：本文件要证明的正是
    高风险问题**根本不会**走到这里。`error` 用来模拟「普通 LLM 不可用」。
    """

    def __init__(self, *, text: str = FREE_TEXT, error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.constructed = 0
        self.prompts: list[str] = []

    @property
    def calls(self) -> int:
        """记录 Fake 客户端被调用的次数。"""
        return len(self.prompts)

    def as_client_factory(self):
        """返回可注入 API 的假 LLM 工厂。"""
        owner = self

        class _FakeOpenAI:
            def __init__(self, *args, **kwargs):
                owner.constructed += 1
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(create=self._create)
                )

            def _create(self, **kwargs):
                """记录请求并返回预设的模型响应。"""
                owner.prompts.append(
                    "\n".join(
                        str(message.get("content", ""))
                        for message in kwargs.get("messages", [])
                    )
                )
                if owner.error is not None:
                    raise owner.error
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(message=SimpleNamespace(content=owner.text))
                    ]
                )

        return _FakeOpenAI


@pytest.fixture
def api_module(monkeypatch):
    """后端模块；`_high_risk_client` 置空，使最终文字只由程序模板生成。"""
    import backend.api as api

    # Ordinary-answer tests exercise the fake LLM even without a local .env.
    monkeypatch.setenv("LLM_TOKEN", "test-token")
    monkeypatch.setattr(api, "_high_risk_client", lambda: None)
    return api


@pytest.fixture
def forbid_high_risk(api_module, monkeypatch):
    """把高风险入口变成断言失败：用于「非高风险请求不得进入高风险链路」。"""

    def _boom(*args, **kwargs):
        """测试用：故意抛出异常。"""
        raise AssertionError("非高风险问题不得进入高风险链路")

    monkeypatch.setattr(api_module, "_high_risk_response", _boom)
    return _boom


@pytest.fixture
def forbid_normal_llm(api_module, monkeypatch):
    """把普通生成入口变成断言失败：用于「高风险问题不得跑普通生成」。"""

    def _boom(*args, **kwargs):
        """测试用：故意抛出异常。"""
        raise AssertionError("高风险问题不得执行普通自由文本生成")

    monkeypatch.setattr(api_module, "OpenAI", _boom)
    monkeypatch.setattr(api_module, "_generate_auto", _boom)
    monkeypatch.setattr(api_module, "_generate_keyword_graph", _boom)
    return _boom


@pytest.fixture
def wire(api_module, monkeypatch):
    """把检索、存储、普通 LLM 全部换成替身并记录调用。

    Returns:
        一个可读的记录对象：`store` / `llm` / `auto_calls` / `native_calls` /
        `link_calls` / `embedder_calls`。
    """

    def _wire(
        *,
        rows=None,
        llm: NormalLLM | None = None,
        auto_items=None,
        auto_linked=None,
        auto_run_id: str | None = "run-test",
        auto_error: Exception | None = None,
        auto_route_high_risk: bool | None = None,
        native_chunks=None,
        linked_rows=None,
        active_run_id: str | None = "run-test",
    ):
        """把假依赖接到被测 Flask 应用。"""
        llm = llm or NormalLLM()
        store = ApiFakeStore(
            rows if rows is not None else dosage_rows(),
            active_run_id=active_run_id,
            linked_rows=linked_rows if linked_rows is not None else linked_herb_rows(),
        )
        auto_items = (
            list(auto_items) if auto_items is not None else [FakeItem()]
        )
        auto_linked = (
            list(auto_linked) if auto_linked is not None else linked_herb()
        )
        native_chunks = (
            list(native_chunks)
            if native_chunks is not None
            else [
                {
                    "page_start": DOSAGE_PAGE,
                    "page_end": DOSAGE_PAGE,
                    "text": DOSAGE_QUOTE,
                    "matched_terms": ["麻黄"],
                }
            ]
        )

        auto_calls: list[dict] = []
        native_calls: list[dict] = []
        link_calls: list[str] = []
        embedder_calls: list[str] = []

        def fake_retrieve_auto(question):
            """测试用：假的自动路由检索。"""
            auto_calls.append({"question": question})
            if auto_error is not None:
                raise auto_error
            # 替身要如实反映真实路由：`route.is_high_risk` 默认按词表判定给出，
            # 也可以显式指定，用于测「词表没命中、路由判定命中」这条兜底分支。
            route_high_risk = (
                is_high_risk_question(question)
                if auto_route_high_risk is None
                else auto_route_high_risk
            )
            packet = FakePacket(
                linked_entities=auto_linked,
                items=auto_items,
                warnings=[],
                trace={
                    "run_id": auto_run_id,
                    "route": {
                        "intent": "dosage_query",
                        "is_high_risk": route_high_risk,
                    },
                },
            )
            # sources 用生产代码自己的投影函数，替身只替检索、不替投影，
            # 「顶层 sources 来自检索结果」这一点在非高风险分支上仍是真实的。
            return packet, {"nodes": [], "edges": []}, api_module._sources_from_packet(
                packet
            )

        def fake_retrieve_native(question, **kwargs):
            """测试用：假的原生驱动检索。"""
            native_calls.append({"question": question, **kwargs})
            return (
                native_chunks,
                {"nodes": [], "edges": []},
                [],
                [],
            )

        def forbid_embedder():
            """测试用：禁止误触发本地嵌入模型加载。"""
            embedder_calls.append("called")
            raise AssertionError(
                "本用例不应触发嵌入器：模型仓库探测属于环境失败，"
                "不能混进接口层顺序的逻辑回归"
            )

        def recording_link(question):
            """测试用：记录实体链接调用。"""
            link_calls.append(question)
            return store.active_run_id, [
                {
                    "canonical_id": row["canonical_id"],
                    "name": row["name"],
                    "entity_type": row["entity_type"],
                }
                for row in store.linked_rows
            ]

        monkeypatch.setattr(api_module, "OpenAI", llm.as_client_factory())
        monkeypatch.setattr(api_module, "_neo4j_store", lambda: store)
        monkeypatch.setattr(api_module, "_retrieve_auto", fake_retrieve_auto)
        monkeypatch.setattr(api_module, "_retrieve_keyword_graph", fake_retrieve_native)
        monkeypatch.setattr(api_module, "_link_question_entities", recording_link)
        monkeypatch.setattr(api_module, "_configured_embedder", forbid_embedder)

        return SimpleNamespace(
            store=store,
            llm=llm,
            auto_calls=auto_calls,
            native_calls=native_calls,
            link_calls=link_calls,
            embedder_calls=embedder_calls,
        )

    return _wire


@pytest.fixture
def client(api_module):
    """Flask 测试客户端。不启动服务器、不连数据库。"""
    api_module.app.config.update(TESTING=True)
    return api_module.app.test_client()


def ask(client, question: str, method: str = "auto"):
    """一次请求，返回 `(status_code, json)`。"""
    response = client.post(
        "/api/graphrag", json={"question": question, "method": method}
    )
    return response.status_code, response.get_json()


__all__ = [
    "ApiFakeStore",
    "BENIGN_QUESTION",
    "DOSAGE_PAGE",
    "DOSAGE_QUOTE",
    "FREE_TEXT",
    "FakeItem",
    "FakePacket",
    "HIGH_RISK_QUESTION",
    "MAHUANG_ID",
    "NormalLLM",
    "POPULATION_QUESTION",
    "ask",
    "contraindication_rows",
    "dosage_rows",
    "linked_herb",
    "linked_herb_rows",
]
