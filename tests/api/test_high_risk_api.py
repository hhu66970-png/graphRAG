"""接口层高风险调用顺序的自动化回归。

为什么单独一组：这里检查的不是核验算法，而是 `backend/api.py` 的**接线顺序**。
拆改前高风险问题要先跑完普通检索 + 普通自由文本生成，再拿核验结果去覆盖它，
于是四类问题同时存在：高风险答案无谓地依赖普通 LLM；普通 LLM 一失败就是
HTTP 500；核验层异常时未经核验的自由文本仍可能被返回；顶层
`subgraph`/`sources` 来自检索而不是最终证据。这四条都不会被「核验算法」的
单元测试发现，所以必须有一组从请求到响应的用例盯住它们。

环境隔离：检索、Neo4j 句柄、普通 LLM 全部是替身，`_configured_embedder`
一旦被调用即失败。因此本文件在**没有数据库、没有网络、没有本地模型**的机器上
也必须全绿——「顺序接错了」不会被伪装成「这台机器没网」。需要真实库、
真实链路的对照在 `tests/high_risk/test_regression_e2e.py`（环境不具备时 skip）。

覆盖用户点名要求的五条：
 1. 普通 LLM 抛异常时，高风险问题仍返回 200 与受控答案／拒答；
 2. 高风险核验抛异常时，响应不得包含原自由生成答案；
 3. `sources` / `subgraph` / `claims` / `evidence_units` 来自同一批最终证据；
 4. 参数化覆盖七种 method，且所选 method 的检索确实被执行；
 5. 非高风险问题保持原调用链（含普通 LLM 失败仍是 500 这一既有行为）。
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from .conftest import (
    BENIGN_QUESTION,
    DOSAGE_PAGE,
    DOSAGE_QUOTE,
    FREE_TEXT,
    HIGH_RISK_QUESTION,
    MAHUANG_ID,
    POPULATION_QUESTION,
    NormalLLM,
    ask,
    contraindication_rows,
    dosage_rows,
)

#: method → （检索家族, 该 method 自己的检索参数）。
#: 参数必须与 `backend/api.py` 的 `_NATIVE_METHODS` 一致；不一致时本用例失败，
#: 这样「六种 method 名义不同、实际全忽略」不可能悄悄发生。
METHOD_TABLE = [
    ("auto", "auto", None),
    ("kg_v2", "auto", None),
    (
        "keyword_graph",
        "native",
        {"use_fulltext": True, "use_vector": None, "expand_graph": True},
    ),
    (
        "vector",
        "native",
        {"use_fulltext": False, "use_vector": True, "expand_graph": False},
    ),
    (
        "vector_cypher",
        "native",
        {"use_fulltext": False, "use_vector": True, "expand_graph": True},
    ),
    (
        "hybrid",
        "native",
        {"use_fulltext": True, "use_vector": True, "expand_graph": False},
    ),
    (
        "hybrid_cypher",
        "native",
        {"use_fulltext": True, "use_vector": True, "expand_graph": True},
    ),
]

METHOD_IDS = [row[0] for row in METHOD_TABLE]


# ------------------------------------------------------------------ 第 1 条
@pytest.mark.parametrize("method,family,params", METHOD_TABLE, ids=METHOD_IDS)
def test_high_risk_answer_does_not_depend_on_the_normal_llm(
    client, wire, method, family, params
):
    """普通 LLM **不可用**时，高风险问题仍必须 200，并给出受控的受核验答案。

    替身 LLM 一旦被构造就会抛异常：所以「返回 200」本身就证明普通生成没跑，
    `llm.constructed == 0` 把这一点写成断言。
    """
    wiring = wire(
        rows=dosage_rows(),
        llm=NormalLLM(error=RuntimeError("普通 LLM 不可用（替身）")),
    )

    code, data = ask(client, HIGH_RISK_QUESTION, method)

    assert code == 200, data
    assert data["answer_status"] == "grounded"
    assert DOSAGE_QUOTE in data["answer"] or "3～9g" in data["answer"]
    assert "煎服" in data["answer"]
    assert data["high_risk"]["is_high_risk"] is True
    assert data["abstained"] is False
    # 最终证据来自高风险断言，而不是所选检索方式命中的文本块
    assert data["evidence_method"] == "high_risk_assertion"
    # 普通生成一次都没跑，也没被构造
    assert wiring.llm.constructed == 0
    assert wiring.llm.calls == 0
    assert FREE_TEXT not in json.dumps(data, ensure_ascii=False)
    # 嵌入器（会去模型仓库联网）在接口层高风险链路上不应被触发
    assert wiring.embedder_calls == []


# ------------------------------------------------------------------ 第 4 条
@pytest.mark.parametrize("method,family,params", METHOD_TABLE, ids=METHOD_IDS)
def test_selected_method_runs_its_own_retrieval(client, wire, method, family, params):
    """用户选的 method 必须真的决定跑哪条检索、用什么参数。

    否则「六种 method」就只是名字不同，用户在不知情的情况下拿到的其实是
    另一种检索的证据。这里两种检索各自留调用记录，谁跑了、带什么参数都可断言。
    """
    wiring = wire(rows=dosage_rows())

    code, data = ask(client, HIGH_RISK_QUESTION, method)

    assert code == 200, data
    assert data["method"] == method
    assert data["retrieval"]["method"] == method

    if family == "auto":
        assert wiring.auto_calls == [{"question": HIGH_RISK_QUESTION}]
        assert wiring.native_calls == []
        assert data["retrieval"]["mode"] == "auto"
        # auto 用检索层自己的链接结果，不再自己链接一次
        assert wiring.link_calls == []
        assert MAHUANG_ID in {node["id"] for node in data["subgraph"]["nodes"]}
    else:
        assert wiring.native_calls == [{"question": HIGH_RISK_QUESTION, **params}]
        assert wiring.auto_calls == []
        assert data["retrieval"]["mode"] == "keyword_graph"
        assert data["retrieval"]["params"] == params
        # 原生检索只给文本块，主语由既有链接器在 ACTIVE 批次里解析
        assert wiring.link_calls == [HIGH_RISK_QUESTION]


# ------------------------------------------------------------------ 第 2 条
@pytest.mark.parametrize("method", ["auto", "keyword_graph"])
def test_verification_failure_never_returns_the_free_text(
    client, wire, api_module, monkeypatch, method
):
    """核验层抛异常时，`answer` 必须被**替换**，原始自由文本一个字都不能留。

    这里的普通 LLM 是「可用」的：如果接口层仍按旧顺序先跑生成，这段自由文本
    就会出现在响应里——正是要拦的「未经核验却看起来正常」。
    """
    wiring = wire(rows=dosage_rows(), llm=NormalLLM(text=FREE_TEXT))

    def _boom(*args, **kwargs):
        raise RuntimeError("核验层不可用（替身）")

    monkeypatch.setattr(api_module, "_high_risk_answer", _boom)

    code, data = ask(client, HIGH_RISK_QUESTION, method)
    body = json.dumps(data, ensure_ascii=False)

    assert code == 200, data
    assert data["answer_status"] == "verification_unavailable"
    assert FREE_TEXT not in body
    assert wiring.llm.constructed == 0
    assert wiring.llm.calls == 0
    # 没有结论就是一次弃权：不能让下游把它读成「有答案」
    assert data["abstained"] is True
    # 证据来源仍如实声明为高风险断言；本次没有单元通过核验，
    # sources 里只是照录的检索原文（下面逐条断言 verified=False）
    assert data["evidence_method"] == "high_risk_assertion"
    # 没有任何单元通过核验：结论与单元都必须为空
    assert data["claims"] == []
    assert data["evidence_units"] == []
    # 展示的是检索到的教材原文，且逐条标明未经核验
    assert data["sources"], "核验不可用时至少要给出原文展示"
    assert all(source["verified"] is False for source in data["sources"])
    assert all(source["page_start"] == DOSAGE_PAGE for source in data["sources"])
    assert "未经核验" in data["answer"]
    assert "核验未能完成" in data["answer"]
    # 没有通过核验的关系，就不画关系图
    assert data["subgraph"] == {"nodes": [], "edges": []}
    assert data["high_risk"]["error"] == "verification_unavailable"


# ------------------------------------------------------------------ 第 3 条
def test_provenance_and_claims_come_from_one_final_batch(client, wire):
    """`sources` / `subgraph` / `claims` / `evidence_units` 必须同源。

    同源的可操作定义：溯源里的每一条都对应一个最终证据单元（同一 `unit_id`、
    同一段原文、同一页），`claims` 引用的单元在其中，`subgraph` 的节点与边
    只由这些单元的主语／宾语／谓词构成。
    """
    wiring = wire(rows=contraindication_rows())

    code, data = ask(client, POPULATION_QUESTION, "auto")

    assert code == 200, data
    assert data["answer_status"] == "grounded", data["coverage_notes"]

    unit_ids = [unit["unit_id"] for unit in data["evidence_units"]]
    assert unit_ids, "grounded 的回答必须有最终证据单元"
    assert [source["unit_id"] for source in data["sources"]] == unit_ids
    assert {source["evidence_quote"] for source in data["sources"]} == {
        unit["evidence_quote"] for unit in data["evidence_units"]
    }
    assert {source["page_start"] for source in data["sources"]} == {
        unit["page_start"] for unit in data["evidence_units"]
    }

    cited = {unit_id for claim in data["claims"] for unit_id in claim["unit_ids"]}
    assert cited, "grounded 的回答必须有引用"
    assert cited <= set(unit_ids)

    nodes = {node["id"] for node in data["subgraph"]["nodes"]}
    assert nodes == {MAHUANG_ID, "P:孕妇"}
    assert {node["type"] for node in data["subgraph"]["nodes"]} == {"中药", "禁忌"}
    edges = {(edge["source"], edge["target"], edge["type"]) for edge in data["subgraph"]["edges"]}
    assert edges == {(MAHUANG_ID, "P:孕妇", "CONTRAINDICATED_IN")}
    assert edges <= {
        (unit["subject_id"], unit["object_id"], unit["predicate"])
        for unit in data["evidence_units"]
    }


def test_native_retrieval_only_positions_the_subject_not_the_evidence(client, wire):
    """原生 method 的检索只做**前置定位**，不决定最终高风险证据。

    同一个问题换两种原生 method，并让替身返回**完全不同**的页与原文：最终证据
    单元必须逐一相同（同一 `unit_id`、同一段原文、同一页），结论与子图也相同；
    变的只有检索参数与检索元数据。响应里 `evidence_method=high_risk_assertion`
    把这个关系写明，因此「选了 vector 就是用向量命中的文本块当证据」这种说法
    在接口层就站不住。

    反过来，如果将来有人把最终证据限制在检索命中的 chunk 范围内，这条用例会
    失败——那时必须显式改口径并同步文档，而不是让「method 决定证据」悄悄成立。
    """
    wire(
        rows=contraindication_rows(),
        native_chunks=[
            {
                "page_start": DOSAGE_PAGE,  # 与断言的页码相同的那一版
                "page_end": DOSAGE_PAGE,
                "text": "孕妇慎用。",
                "matched_terms": ["麻黄"],
            }
        ],
    )
    code_a, first = ask(client, POPULATION_QUESTION, "keyword_graph")

    wire(
        rows=contraindication_rows(),
        native_chunks=[
            {
                "page_start": 999,
                "page_end": 1000,
                "text": "另一条检索命中的、完全不同的教材片段。",
                "matched_terms": [],
            }
        ],
    )
    code_b, second = ask(client, POPULATION_QUESTION, "vector")

    assert code_a == 200 and code_b == 200
    # 两边都真的拿到了证据，否则下面的「逐项一致」会是空集之间的假通过
    assert first["answer_status"] == second["answer_status"] == "grounded"
    assert first["evidence_units"] and second["evidence_units"]
    assert first["retrieval"]["items"] == second["retrieval"]["items"] == 1
    # 检索参数确实不同（vector 打开了向量通道，keyword_graph 没有）
    assert first["retrieval"]["params"]["use_vector"] is None
    assert second["retrieval"]["params"]["use_vector"] is True

    # 证据、结论、子图逐项一致
    assert [unit["unit_id"] for unit in first["evidence_units"]] == [
        unit["unit_id"] for unit in second["evidence_units"]
    ]
    assert [unit["evidence_quote"] for unit in first["evidence_units"]] == [
        unit["evidence_quote"] for unit in second["evidence_units"]
    ]
    assert [unit["page_start"] for unit in first["evidence_units"]] == [
        unit["page_start"] for unit in second["evidence_units"]
    ]
    assert first["subgraph"] == second["subgraph"]
    assert [claim["statement"] for claim in first["claims"]] == [
        claim["statement"] for claim in second["claims"]
    ]
    assert first["evidence_method"] == second["evidence_method"] == (
        "high_risk_assertion"
    )
    # 最终证据的页码来自断言所在的教材页（82），而不是替身检索报出来的 999/1000
    assert {unit["page_start"] for unit in first["evidence_units"]} == {DOSAGE_PAGE}
    assert {unit["page_start"] for unit in second["evidence_units"]} == {DOSAGE_PAGE}


def test_abstained_high_risk_answer_shows_no_foreign_sources(client, wire):
    """弃权的回答不能挂着一批「看起来像证据」的检索来源。

    拆改前 `sources` 来自检索结果，而 `claims` / `evidence_units` 来自核验：
    两者不同源，于是「没有结论」的回答下面仍列着一堆教材原文卡片。
    """
    wiring = wire(rows=[])  # 检索有命中，但断言层读不到任何高风险证据

    code, data = ask(client, HIGH_RISK_QUESTION, "auto")

    assert code == 200, data
    assert data["answer_status"] == "abstain"
    assert data["abstained"] is True
    assert data["claims"] == []
    assert data["evidence_units"] == []
    assert data["sources"] == []
    assert data["subgraph"] == {"nodes": [], "edges": []}
    # 检索确实跑过（否则上面几条不成立）
    assert wiring.auto_calls == [{"question": HIGH_RISK_QUESTION}]
    assert data["retrieval"]["items"] == 1


def test_no_active_run_is_a_controlled_abstain(client, wire):
    """没有 ACTIVE 批次：受控弃权，不是 500，也不是自由文本。"""
    wire(rows=dosage_rows(), active_run_id=None, auto_run_id=None)

    code, data = ask(client, HIGH_RISK_QUESTION, "auto")

    assert code == 200, data
    assert data["answer_status"] == "abstain"
    assert data["abstained"] is True
    assert data["evidence_method"] == "high_risk_assertion"
    assert data["high_risk"]["error"] == "no_active_run"
    assert "ACTIVE" in data["answer"]
    assert data["claims"] == [] and data["sources"] == []


def test_retrieval_failure_is_a_controlled_refusal(client, wire):
    """检索部分就失败：连原文都没有，只能受控拒答（仍是 200，不是 500）。"""
    wire(auto_error=RuntimeError("图数据库不可达（替身）"))

    code, data = ask(client, HIGH_RISK_QUESTION, "auto")

    assert code == 200, data
    assert data["answer_status"] == "verification_unavailable"
    assert data["abstained"] is True
    assert data["evidence_method"] == "high_risk_assertion"
    assert data["sources"] == []
    assert data["claims"] == [] and data["evidence_units"] == []
    assert "不予回答" in data["answer"]
    assert "图数据库不可达" in data["answer"]
    assert FREE_TEXT not in json.dumps(data, ensure_ascii=False)


# ------------------------------------------------------------------ 第 5 条
@pytest.mark.parametrize("method", ["auto", "keyword_graph"])
def test_benign_question_keeps_the_original_chain(
    client, wire, api_module, forbid_high_risk, method
):
    """非高风险问题保持原调用链：检索 → 普通生成 → 各自的原有响应字段。

    `forbid_high_risk` 让任何进入高风险链路的尝试直接失败，因此这条用例同时
    证明高风险分支没有被误扩大。
    """
    assert api_module._is_high_risk_question(BENIGN_QUESTION) is False, (
        "对照问题必须不在高风险词表内，否则这条用例失去意义"
    )
    wiring = wire(rows=[])

    code, data = ask(client, BENIGN_QUESTION, method)

    assert code == 200, data
    assert wiring.llm.calls == 1
    assert data["answer"] == FREE_TEXT
    assert data["answer_status"] is None
    assert data["high_risk"] == {"is_high_risk": False}
    assert data["claims"] == [] and data["evidence_units"] == []
    assert data["method"] == method
    # 非高风险链路没有「高风险断言证据」可言，这个字段不得出现——
    # 否则客户端会把普通回答也当成断言核验过的证据
    assert "evidence_method" not in data
    if method == "auto":
        assert "abstained" in data
        assert data["sources"][0]["evidence_id"] == "e-1"
    else:
        assert wiring.native_calls == [
            {
                "question": BENIGN_QUESTION,
                "use_fulltext": True,
                "use_vector": None,
                "expand_graph": True,
            }
        ]


def test_benign_question_still_surfaces_llm_failure_as_500(
    client, wire, forbid_high_risk
):
    """非高风险问题的既有行为**不变**：普通 LLM 失败仍是 500。

    与「高风险问题必须 200 + 受控文本」并列，说明这次改动只动了高风险链路。
    """
    wire(rows=[], llm=NormalLLM(error=RuntimeError("普通 LLM 不可用（替身）")))

    code, data = ask(client, BENIGN_QUESTION, "auto")

    assert code == 500
    assert "普通 LLM 不可用" in data["error"]


def test_router_flagged_high_risk_also_takes_the_high_risk_path(
    client, wire, forbid_normal_llm
):
    """词表没命中、但检索层的路由判为高风险时，同样必须转高风险链路。

    这是兜底而不是主路径：词表判定比路由词表更宽，正常不会走到这里。留这条
    是为了「路由说高风险、接口层却照旧跑完普通生成」不会重新出现。
    检索结果在这里已经被拿到，**不得**再检索一次。
    """
    wiring = wire(rows=dosage_rows(), auto_route_high_risk=True)

    code, data = ask(client, BENIGN_QUESTION, "auto")

    assert code == 200, data
    assert data["high_risk"]["is_high_risk"] is True
    assert data["answer_status"] == "grounded"
    assert "3～9g" in data["answer"], "最终文字来自程序模板，不是自由生成"
    assert data["evidence_units"][0]["evidence_quote"] == DOSAGE_QUOTE
    # 已经拿到的检索结果被复用，没有第二次检索，也没有普通生成
    assert wiring.auto_calls == [{"question": BENIGN_QUESTION}]
    assert wiring.llm.constructed == 0


@pytest.mark.parametrize("method", ["nope", "kg_v3", ""])
def test_unknown_method_is_rejected_before_any_retrieval(
    client, wire, forbid_high_risk, method
):
    """未知 method 在任何检索之前就被拒绝，不做无用的数据库往返。"""
    wiring = wire(rows=dosage_rows())

    code, data = ask(client, HIGH_RISK_QUESTION, method)

    assert code == 400
    assert "Unknown method" in data["error"]
    assert wiring.auto_calls == []
    assert wiring.native_calls == []
    assert wiring.llm.constructed == 0


def test_missing_question_is_rejected(client, wire):
    wire(rows=dosage_rows())

    response = client.post("/api/graphrag", json={"method": "auto"})

    assert response.status_code == 400
    assert "Question is required" in response.get_json()["error"]


# ---------------------------------------------------- 前端徽标一致性
FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"
_MAP_BLOCK = re.compile(r"const (HR_STATUS_[A-Z]+)\s*=\s*\{(.*?)\};", re.S)
_MAP_ENTRY = re.compile(r"'([a-z_]+)'\s*:\s*'([^']*)'")


def _frontend_hr_maps() -> dict[str, dict[str, str]]:
    source = FRONTEND.read_text(encoding="utf-8")
    return {
        name: dict(_MAP_ENTRY.findall(body))
        for name, body in _MAP_BLOCK.findall(source)
    }


def test_emitted_statuses_have_a_red_badge_in_the_frontend():
    """接口层能发出的每个状态，前端都必须认识，且非 grounded 一律不是「已核验」。

    核验不可用时的回答尤其不能被渲染成绿色徽标——那会让用户以为它通过了核验。
    """
    maps = _frontend_hr_maps()
    classes = maps["HR_STATUS_CLASS"]
    labels = maps["HR_STATUS_LABEL"]

    for status in ("grounded", "abstain", "verification_unavailable"):
        assert status in classes, f"前端不认识状态 {status}"
        assert status in labels, f"前端没有 {status} 的文案"

    assert classes["grounded"] == "hr-ok"
    assert classes["verification_unavailable"] != "hr-ok"
    assert classes["abstain"] != "hr-ok"
