"""真实请求 → 最终响应的自动化回归（任务书要求的三反例 + 正确对照）。

与其它高风险测试的区别：这里**不注入任何替身**。从 Flask 的
`POST /api/graphrag` 出发，走真实链接、真实只读读取、真实核验与渲染，
断言用户最终看到的那段文字、对象、人群、状态、引用，以及前端据此渲染的
「已核验」徽标。

模型客户端被显式置为 None：本文件检查的是**可复现的核验链路**，不是模型
措辞；配置了模型时结论由模型候选，会引入不确定性。核验层的行为与是否配置
模型无关（同一入口、同一证据集合）。

需要可读的真实库与 ACTIVE 批次；环境不具备时 skip。**不会**用 skip 冒充通过，
也不写入任何数据。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphragexpr.domain.high_risk import AnswerStatus  # noqa: E402

pytestmark = pytest.mark.integration

FRONTEND = ROOT / "frontend" / "index.html"

#: 前端的状态 → 徽标类名映射。用户看到的「已核验」完全由它决定，
#: 因此必须与后端四档状态一起回归，而不是只测后端字段。
_CLASS_BLOCK = re.compile(r"const HR_STATUS_CLASS\s*=\s*\{(.*?)\};", re.S)
_CLASS_ENTRY = re.compile(r"'([a-z_]+)'\s*:\s*'([a-z\-]+)'")


def _frontend_status_classes() -> dict[str, str]:
    """前端高风险状态对应的样式类。"""
    source = FRONTEND.read_text(encoding="utf-8")
    block = _CLASS_BLOCK.search(source)
    assert block, "frontend/index.html 里找不到 HR_STATUS_CLASS，徽标回归无从进行"
    return dict(_CLASS_ENTRY.findall(block.group(1)))


def _badge_class(status: str | None) -> str:
    """复刻前端 `HR_STATUS_CLASS[status] || 'hr-bad'` 的取值。"""
    return _frontend_status_classes().get(status or "", "hr-bad")


def test_only_grounded_gets_the_verified_badge():
    """「已核验」徽标只允许出现在 grounded。任何一档降级的回答都不能显示已核验。"""
    mapping = _frontend_status_classes()

    assert mapping.get(AnswerStatus.GROUNDED.value) == "hr-ok"
    for status in AnswerStatus:
        if status is AnswerStatus.GROUNDED:
            continue
        assert _badge_class(status.value) != "hr-ok", (
            f"{status.value} 不该显示「已核验」"
        )
    # 未知/缺失状态一律按最坏档处理，不能默认为已核验
    assert _badge_class(None) != "hr-ok"
    assert _badge_class("something_new") != "hr-ok"


@pytest.fixture(scope="module")
def client():
    """Flask 测试客户端；库不可达或没有 ACTIVE 批次时 skip。"""
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    try:
        import backend.api as api
    except Exception as exc:  # 依赖缺失不应伪装成通过
        pytest.skip(f"后端 API 无法导入，跳过端到端回归：{exc}")

    from graphragexpr.storage import serving_read
    from graphragexpr.storage.neo4j_store import Neo4jStore

    store = Neo4jStore()
    try:
        active_run = serving_read.active_run(store)
    except Exception as exc:
        pytest.skip(f"读不到图数据库，跳过端到端回归：{exc}")
    finally:
        store.close()
    if not active_run:
        pytest.skip("没有 ACTIVE 的 IngestRun，跳过端到端回归")

    original = api._high_risk_client
    api._high_risk_client = lambda: None  # 确定性：不让模型改写最终文字
    try:
        yield api.app.test_client()
    finally:
        api._high_risk_client = original


def ask(client, question: str) -> dict:
    """一次真实请求，返回最终响应体。"""
    response = client.post(
        "/api/graphrag", json={"question": question, "method": "keyword_graph"}
    )
    assert response.status_code == 200, f"HTTP {response.status_code}"
    return response.get_json()


def claims_of(data: dict) -> list[dict]:
    """从回答中取出候选结论。"""
    return data.get("claims") or []


# --------------------------------------------------------------- 反例一
@pytest.mark.parametrize(
    "question, longer",
    [
        ("六味地黄丸孕妇能用吗？", "六味地黄丸"),
        ("安宫牛黄丸孕妇能用吗？", "安宫牛黄丸"),
        ("龙胆泻肝丸的用量是多少？", "龙胆泻肝丸"),
    ],
)
def test_patent_medicine_is_not_answered_as_a_single_herb(client, question, longer):
    """成药的名称被链接截成单味药时，不得把单味药的证据当作该问题的答案。

    第三例（龙胆泻肝丸）在**没有人群条件**的问题里走过一遍：它在本体中没有
    实体，「存在更长实体名」这条检查不触发，曾返回 grounded 并给出
    「龙胆：用量3～6g」——正是要拦的对象错配。
    """
    data = ask(client, question)
    text = data["answer"]

    assert data["answer_status"] == AnswerStatus.ABSTAIN.value
    assert not claims_of(data), "对象未确认时不得给出任何结论"
    assert longer in text, "最终文字必须点名问题里那个更长的对象"
    assert "无法确认为同一对象" in text
    assert _badge_class(data["answer_status"]) != "hr-ok"


# --------------------------------------------------------------- 反例二
@pytest.mark.parametrize(
    "question, population",
    [("附子哺乳期能用吗？", "哺乳期"), ("附子小儿能用吗？", "小儿")],
)
def test_pregnancy_evidence_does_not_answer_other_populations(client, question, population):
    """取到的证据只涉及孕妇时，不能用它回答哺乳期／小儿的问题。"""
    data = ask(client, question)
    text = data["answer"]

    assert data["answer_status"] != AnswerStatus.GROUNDED.value
    assert not claims_of(data), "人群不匹配时不得给出结论"
    assert population in text, "理由必须点名所问人群"
    assert "不能用来回答该人群的用法" in text
    # 证据确实存在（不是「没读到」），只是不适用于所问人群——两者必须区别开
    assert "所取证据未涉及问题指定的人群" in text
    assert _badge_class(data["answer_status"]) != "hr-ok"


# --------------------------------------------------------------- 反例三
def test_unknown_object_does_not_fall_back_to_enumeration(client):
    """没识别出指定药物时，不得改去反向枚举别的药物。"""
    data = ask(client, "布洛芬孕妇能吃吗？")
    text = data["answer"]

    assert data["answer_status"] == AnswerStatus.ABSTAIN.value
    assert not claims_of(data)
    assert "不按人群反向读取" in text
    assert _badge_class(data["answer_status"]) == "hr-bad"
    # 反向枚举会端上来的那几味药，一个都不许出现
    for other in ("徐长卿", "千金子", "厚朴", "孕妇慎用"):
        assert other not in text, f"不得把「{other}」当成布洛芬的答案"


# --------------------------------------------------------------- 对照组
@pytest.mark.parametrize(
    "question, subject, marker",
    [
        ("附子孕妇能用吗？", "附子", "慎用"),
        ("藜芦孕妇能用吗？", "藜芦", "禁用"),
    ],
)
def test_matching_population_is_still_answered(client, question, subject, marker):
    """正确对照：人群匹配时必须照常给出结论，不能因为加了核验就一律拒答。"""
    data = ask(client, question)
    claims = claims_of(data)

    assert data["answer_status"] == AnswerStatus.GROUNDED.value
    assert claims, "证据与问题对象、条件一致时应当给出结论"
    assert _badge_class(data["answer_status"]) == "hr-ok"
    assert subject in data["answer"]

    for claim in claims:
        assert claim["subject_name"] == subject, "结论的主语必须与问题对象一致"
        assert claim["pages"], "每条结论都必须带可回溯的引用页码"
    assert any("孕妇" in (claim.get("conditions") or "") for claim in claims)
    assert marker in data["answer"]


def test_incompatibility_and_dosage_controls_still_answer(client):
    """另外两组正确对照：配伍禁忌与用量。"""
    incompat = ask(client, "附子不宜与什么药同用？")
    assert incompat["answer_status"] == AnswerStatus.GROUNDED.value
    assert incompat["claims"]
    assert "半夏" in incompat["answer"]

    dosage = ask(client, "附子的用量是多少？")
    assert dosage["answer_status"] == AnswerStatus.GROUNDED.value
    assert dosage["claims"]
    assert "3～15g" in dosage["answer"], "剂量数值必须原样保留"
    assert "先煎" in dosage["answer"], "剂量的条件不得丢失"
    for claim in claims_of(dosage):
        assert claim["pages"]


def test_unresolved_severity_conflict_is_not_presented_as_verified(client):
    """两处来源程度不一致且未经人工裁定时，不得挑一档当已核验的结论。

    「川乌孕妇能用吗？」有两处来源：第 179 页「孕妇慎用」（慎用）、
    第 67 页把川乌列入妊娠**禁用**药。第 179 页那一条单条就超过 token 预算，
    会被整条裁掉——若不额外处理，回答就只剩一条「禁用」且标为已核验，
    等于由裁剪替用户做了一次未经核验的临床取舍。
    """
    data = ask(client, "川乌孕妇能用吗？")
    text = data["answer"]
    notes = " ".join(data.get("coverage_notes") or [])

    assert data["answer_status"] == AnswerStatus.PARTIAL_GROUNDED.value
    assert _badge_class(data["answer_status"]) == "hr-warn", "有未裁决的分歧时不得显示已核验"
    assert "两处来源并列保留" in notes and "未做取舍" in notes
    assert "因 token 预算" in notes and "无法并列展示" in notes, (
        "裁掉的那一处必须写明，否则「并列保留」名不副实"
    )
    assert "禁用" in text and "慎用" in notes


def test_legal_enumeration_lists_objects_and_separates_severity(client):
    """合法的反向枚举必须能答，且逐条区分「禁用」与「慎用」。

    这条问句问的是一个集合，按人群反查是正确的读法——不能因为反例三就把
    整类问法一起堵死。
    """
    data = ask(client, "孕妇禁用的中药有哪些？")
    claims = claims_of(data)

    assert data["answer_status"] == AnswerStatus.GROUNDED.value, "合法的反向枚举必须能答"
    assert claims
    assert _badge_class(data["answer_status"]) == "hr-ok"

    subjects = {claim["subject_name"] for claim in claims}
    assert "中药" not in subjects, "泛称不是对象，不能当成某一味药"
    assert subjects - {None}, "每条结论都要落到具体对象上"

    severities = {
        level
        for claim in claims
        for level in ("禁用", "慎用")
        if level in (claim.get("conditions") or "")
    }
    assert severities == {"禁用", "慎用"}, "不同限制程度必须逐条标出，不能只留一档"
    # 重的排在前：禁用 → 慎用
    order = [
        level
        for claim in claims
        for level in ("禁用", "慎用")
        if level in (claim.get("conditions") or "")
    ]
    assert order == sorted(order, key=lambda item: 0 if item == "禁用" else 1)
    for claim in claims:
        assert claim["pages"], "每条结论都必须带可回溯的引用页码"
