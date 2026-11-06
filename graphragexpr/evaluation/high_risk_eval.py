"""比较原文生成、原文生成后核验和条件证据单元三种高风险回答方法。

统计引用、数字单位同源、限制程度和安全表述等可从响应核对的指标。
题集尚待人工仲裁，语义正确性与完整性返回 not_measured。
评测只读数据库，不构建或发布批次。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..highrisk.parsing import parse_dosage_units
from ..highrisk.units import build_units_from_assertions
from ..highrisk.verify import _safety_phrase_in

DEFAULT_DATASET = Path(__file__).parents[2] / "eval" / "high_risk" / "questions.jsonl"

#: 已人工仲裁的标注状态。只有取该值的数据才允许被当作金标用于语义评分。
ADJUDICATED = "human_adjudicated"

ARMS = ("raw_text_prompt", "raw_text_verified", "conditional_units_verified")


@dataclass(frozen=True)
class HighRiskDataset:
    """冻结的高风险评测集。"""

    items: tuple[dict, ...]
    validation_errors: tuple[str, ...] = ()

    @property
    def summary(self) -> dict:
        """汇总一组高风险评测分数。"""
        by_split: dict[str, int] = {}
        by_category: dict[str, int] = {}
        for item in self.items:
            by_split[item["split"]] = by_split.get(item["split"], 0) + 1
            by_category[item["risk_category"]] = (
                by_category.get(item["risk_category"], 0) + 1
            )
        statuses = sorted({item["annotation_status"] for item in self.items})
        return {
            "total": len(self.items),
            "by_split": by_split,
            "by_category": by_category,
            "annotation_statuses": statuses,
            "semantic_scoring_allowed": all(
                item["annotation_status"] == ADJUDICATED for item in self.items
            ),
        }


def _read_jsonl(path: Path) -> list[dict]:
    """按行读取 JSONL 评测题或预测。"""
    items: list[dict] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{number} 不是合法 JSON：{exc}") from exc
    return items


_REQUIRED_FIELDS = (
    "id", "split", "risk_category", "question", "required_predicates",
    "forbidden_unsupported_claims", "should_abstain", "annotation_status",
)


def load_dataset(path: str | Path = DEFAULT_DATASET) -> HighRiskDataset:
    """Load and validate the frozen high-risk question set."""
    path = Path(path)
    items = _read_jsonl(path)
    errors: list[str] = []
    seen: set[str] = set()
    for item in items:
        ident = item.get("id", "<missing id>")
        for name in _REQUIRED_FIELDS:
            if name not in item:
                errors.append(f"{ident}: 缺字段 {name}")
        if ident in seen:
            errors.append(f"{ident}: id 重复")
        seen.add(ident)
        if item.get("split") not in {"dev", "test"}:
            errors.append(f"{ident}: split 必须是 dev 或 test")
        if item.get("risk_category") not in {"dosage", "contraindication", "incompatibility"}:
            errors.append(f"{ident}: risk_category 不在三类之内")
        if item.get("should_abstain") and item.get("gold_answer_points"):
            errors.append(f"{ident}: 拒答题不应带 gold_answer_points")
        if not item.get("should_abstain") and not item.get("gold_answer_points"):
            errors.append(f"{ident}: 非拒答题必须给 gold_answer_points")
    return HighRiskDataset(items=tuple(items), validation_errors=tuple(errors))


# --------------------------------------------------------------- 机械指标
_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _answer_text(payload: dict[str, Any]) -> str:
    """从模型/系统输出中取出用于评分的答案文本。"""
    return str(payload.get("answer_text") or payload.get("answer") or "")


def _cited_units(payload: dict[str, Any]) -> list[dict]:
    """抽出回答里引用的证据单元。"""
    return list(payload.get("evidence_units") or [])


def _unit_haystack(unit: dict) -> str:
    """一个证据单元里全部「答案可以合法引用到的字面」。

    必须含 `page_label`：答案文本按模板写「（第 231 页）」，页码是引用格式
    的一部分，不是凭空出现的数值。少了它，页码会被算成「数值不可回溯」，
    于是指标反映的是本函数漏了字段，而不是系统挪用了数值。
    """
    parts = [
        unit.get("dosage_raw"),
        unit.get("object_name"),
        unit.get("subject_name"),
        unit.get("evidence_quote"),
        unit.get("context"),
        unit.get("conditions"),
        unit.get("page_label"),
    ]
    return " ".join(str(part) for part in parts if part)


def mechanical_metrics(payloads: dict[str, dict]) -> dict[str, Any]:
    """Label-free metrics computed from the responses alone.

    Args:
        payloads: 问题 id → `pack_answer_payload` 的输出。

    Returns:
        可直接对比的计数与比率。**不含任何语义正确性判断**。
    """
    total = len(payloads)
    answered = abstained = 0
    claims_total = 0
    claims_without_citation = 0
    numbers_total = 0
    numbers_untraceable = 0
    safety_phrase_hits = 0
    severity_dropped = 0
    severity_expected = 0
    uncitable_cited = 0
    subject_not_resolved = 0
    statuses: dict[str, int] = {}

    for payload in payloads.values():
        status = str(payload.get("answer_status") or "unknown")
        statuses[status] = statuses.get(status, 0) + 1
        if payload.get("_subject_not_resolved"):
            subject_not_resolved += 1
        text = _answer_text(payload)
        units = _cited_units(payload)
        if payload.get("abstained") or status == "abstain":
            abstained += 1
        else:
            answered += 1

        if _safety_phrase_in(text):
            safety_phrase_hits += 1

        haystack = " ".join(_unit_haystack(unit) for unit in units)

        # 答案里出现的每一个数字都必须能在所选证据里找到同源。
        # 这是「数值可回溯」的机械检查，不需要知道正确答案是什么。
        for number in _NUMBER_RE.findall(text):
            numbers_total += 1
            if number not in haystack:
                numbers_untraceable += 1

        # 限制程度不得被改写：单元带程度词时，答案里必须出现同一档词。
        for unit in units:
            label = unit.get("severity_label")
            if label and label != "原文未标注程度":
                severity_expected += 1
                if label not in text:
                    severity_dropped += 1
            if not unit.get("is_citable", True):
                uncitable_cited += 1

        for claim in payload.get("claims") or []:
            claims_total += 1
            if not (claim.get("unit_ids") or []):
                claims_without_citation += 1

    return {
        "questions": total,
        "answered": answered,
        "abstained": abstained,
        "subject_not_resolved": subject_not_resolved,
        "status_distribution": statuses,
        "claims": claims_total,
        "claims_without_citation": claims_without_citation,
        "numbers_in_answer": numbers_total,
        "numbers_untraceable_to_evidence": numbers_untraceable,
        "safety_phrase_answers": safety_phrase_hits,
        "severity_terms_expected": severity_expected,
        "severity_terms_dropped": severity_dropped,
        "uncitable_units_cited": uncitable_cited,
    }


def semantic_metrics(dataset: HighRiskDataset) -> dict[str, Any]:
    """Semantic metrics — **deliberately not computed**.

    正确性、完整性、条件遗漏的语义判断、拒答是否恰当都需要人工标注。
    本数据集尚为 `draft_from_textbook_pending_human_adjudication`，
    因此这里明确返回 `not_measured`，而不是拿一个自动指标冒充语义质量，
    更不允许用待测系统自身的输出反推金标。
    """
    adjudicated = all(item["annotation_status"] == ADJUDICATED for item in dataset.items)
    return {
        "status": "not_measured",
        "reason": (
            "缺少人工仲裁的金标；语义指标不做自动打分"
            if not adjudicated
            else "数据集已人工仲裁，但语义评分尚未实现"
        ),
        "unmeasured": [
            "answer_correctness",
            "answer_completeness",
            "condition_omission_semantic",
            "abstention_appropriateness",
            "forbidden_claim_rate",
        ],
        "requires": "人工逐题仲裁 annotation_status=human_adjudicated 后另行实现",
    }


# --------------------------------------------------------------- 三组臂
@dataclass
class ArmRunner:
    """构造某一组对比臂的执行函数。

    A/B 两组需要调用模型；未配置模型时**不假装跑过**，而是如实返回
    `not_run`，由调用方在报告里写明「未运行」。
    """

    llm: Any = None
    store: Any = None
    run_id: str | None = None
    unit_token_budget: int | None = None

    def runner(self, arm: str) -> Callable[[dict], dict]:
        """构造评测运行器。"""
        if arm == "conditional_units_verified":
            return self._arm_c
        if arm in ("raw_text_prompt", "raw_text_verified"):
            return self._arm_a_or_b(verify=arm == "raw_text_verified")
        raise ValueError(f"未知对比臂：{arm}")

    # --- C：生产链路 -------------------------------------------------
    def _arm_c(self, item: dict) -> dict:
        """跑高风险评测的核验臂（程序化校验）。"""
        from ..highrisk.service import HighRiskService, pack_answer_payload
        from ..storage.high_risk_read import load_high_risk_assertions

        subject_ids, linked = self._resolve_subject(item)
        rows = load_high_risk_assertions(
            self.store, run_id=self.run_id, subject_ids=subject_ids
        )
        service = HighRiskService(
            self.store, run_id=self.run_id, llm=None,
            **(
                {"unit_token_budget": self.unit_token_budget}
                if self.unit_token_budget
                else {}
            ),
        )
        answer = service.answer(item["question"], linked_entities=linked)
        payload = pack_answer_payload(answer)
        payload["answer_text"] = answer.answer_text
        payload["_units_built"] = len(build_units_from_assertions(rows))
        if item.get("expected_subject") and not subject_ids:
            # 金标给了主语名，但链接器没能落到本体上。这条不能算「系统弃权」，
            # 它是「评测没有把题喂到该走的路径上」，与真实弃权分开计数。
            payload["_subject_not_resolved"] = item["expected_subject"]
        return payload

    # --- A/B：原文 + 提示词 -------------------------------------------
    def _arm_a_or_b(self, *, verify: bool) -> Callable[[dict], dict]:
        """跑高风险评测的对照臂（自由生成或仅提示）。"""
        def run(item: dict) -> dict:
            """执行该评测臂或该条预测的一次完整跑批。"""
            if self.llm is None:
                return {
                    "answer_status": "not_run",
                    "abstained": None,
                    "answer_text": "",
                    "claims": [],
                    "evidence_units": [],
                    "_note": "未配置模型（llm=None），本对比臂未运行",
                }
            return _run_raw_text_arm(
                self.llm, self.store, self.run_id, item, verify=verify
            )

        return run

    def _resolve_subject(self, item: dict) -> tuple[list[str] | None, list[dict]]:
        """把金标主语落到 ACTIVE 批次的本体 id 上，并给出对应的 linked_entities。

        Returns:
            `(subject_ids, linked_entities)`；数据集没给主语、或链接器落不到
            本体上时返回 `(None, [])`。**不编造 id**：编出来的 id 查不到任何
            证据，产出的弃权与「真的没有证据」在响应里长得一模一样，会把
            工具缺陷记成系统成绩。
        """
        name = item.get("expected_subject")
        if not name:
            return None, []
        canonical_id = self._canonical_id(name)
        if canonical_id is None:
            return None, []
        return [canonical_id], [
            {"canonical_id": canonical_id, "entity_type": "HERB", "preferred_name": name}
        ]

    def _canonical_id(self, name: str) -> str | None:
        """按真实链接器解析药材名，解析不到返回 None。

        `_LINK_IN_TEXT` 的返回行只有 `canonical_id` / `name` / `entity_type` /
        `len` / `type_rank`，**没有** `preferred_name`；按不存在的键做等值匹配
        会永远落空。该查询已按「HERB 优先、名称最长优先」排序，取首行即为
        文档承诺的最长匹配。
        """
        from ..storage import serving_read

        linked = serving_read.link_entities_in_text(
            self.store, name, run_id=self.run_id, types=("HERB",)
        )
        if not linked:
            return None
        canonical = linked[0].get("canonical_id")
        return str(canonical) if canonical else None


_RAW_TEXT_PROMPT = """你是中药学助手。请只依据下面的教材原文回答问题。

教材原文：
{context}

问题：{question}

要求：
1. 只使用原文中出现的信息，不要补充原文没有的条件、频次或人群。
2. 数值与单位必须与原文一致。
3. 原文没有给出时，直接说明没有检索到。
"""


def _run_raw_text_arm(
    llm: Any, store: Any, run_id: str | None, item: dict, *, verify: bool
) -> dict:
    """A/B 两组共用的实现：原文直接喂模型。

    `verify=False`（A 组）：模型原话直接作为答案。
    `verify=True` （B 组）：把模型结论包成 `CandidateClaim` 过一遍
    `verify_claims`，但**不构造条件证据单元**——这正是两组对比要暴露的差别。
    """
    from ..storage.high_risk_read import load_high_risk_assertions

    subject_ids = None
    if item.get("expected_subject"):
        subjects = [e for e in _link(store, run_id, item["expected_subject"])]
        subject_ids = [str(e["canonical_id"]) for e in subjects] or None

    rows = load_high_risk_assertions(store, run_id=run_id, subject_ids=subject_ids)
    context = "\n".join(
        sorted({str(row.get("evidence_quote") or "") for row in rows if row.get("evidence_quote")})
    )
    response = llm.complete(
        _RAW_TEXT_PROMPT.format(context=context, question=item["question"])
    )
    text = str(response or "").strip()
    payload: dict[str, Any] = {
        "answer_status": "raw",
        "abstained": False,
        "answer_text": text,
        "claims": [],
        "evidence_units": [],
    }
    if not verify:
        return payload

    # B 组：把模型文本当作候选结论核验，但没有条件证据单元可核对，
    # 因此核验能查到的只有「有没有引用」与「有没有安全性表述」。
    from ..highrisk.claims import CandidateClaim, ClaimKind
    from ..highrisk.verify import verify_claims

    claim = CandidateClaim(
        claim_id=f"raw:{item['id']}",
        kind=ClaimKind.DOSAGE if item["risk_category"] == "dosage" else ClaimKind.CONTRAINDICATION,
        statement=text,
        cited_unit_ids=[],
    )
    outcome = verify_claims([claim], [])
    payload["claims"] = [
        {
            "claim_id": claim.claim_id,
            "statement": text,
            "unit_ids": list(verdict.supporting_unit_ids),
        }
        for verdict in outcome.accepted
    ]
    payload["rejections"] = outcome.rejections
    payload["answer_status"] = "raw_verified" if outcome.accepted else "abstain"
    payload["abstained"] = not outcome.accepted
    if not outcome.accepted:
        payload["answer_text"] = ""
    return payload


def _link(store: Any, run_id: str | None, name: str) -> list[dict]:
    """评测时把问题实体链接到图谱实体。"""
    from ..storage import serving_read

    return serving_read.link_entities_in_text(
        store, name, run_id=run_id, types=("HERB",)
    )


def compare(
    dataset: HighRiskDataset,
    *,
    runners: dict[str, Callable[[dict], dict]],
    split: str | None = None,
) -> dict[str, Any]:
    """Run every requested arm over the dataset and compare mechanically.

    Args:
        dataset: 冻结题目集。
        runners: 对比臂 → 执行函数。
        split: 只跑该划分（dev/test），None 表示全部。

    Returns:
        {"arms": {臂: {...}}, "delta": {...} 或 None, "semantic": {...}}。
        `delta` 只在恰好两个臂都跑成时给出，避免把多个臂的差异混成一个数字。
    """
    items = [item for item in dataset.items if split is None or item["split"] == split]
    arms: dict[str, Any] = {}
    for arm, run in runners.items():
        payloads: dict[str, dict] = {}
        for item in items:
            payloads[item["id"]] = run(item)
        arms[arm] = {
            "questions": len(items),
            "mechanical": mechanical_metrics(payloads),
        }

    delta = None
    ran = [arm for arm, result in arms.items()
           if result["mechanical"]["questions"] and
           "not_run" not in result["mechanical"]["status_distribution"]]
    if len(ran) == 2:
        left, right = ran
        a = arms[left]["mechanical"]
        b = arms[right]["mechanical"]
        delta = {
            "from": left,
            "to": right,
            "abstained": b["abstained"] - a["abstained"],
            "numbers_untraceable_to_evidence": (
                b["numbers_untraceable_to_evidence"] - a["numbers_untraceable_to_evidence"]
            ),
            "safety_phrase_answers": (
                b["safety_phrase_answers"] - a["safety_phrase_answers"]
            ),
            "severity_terms_dropped": (
                b["severity_terms_dropped"] - a["severity_terms_dropped"]
            ),
        }

    return {
        "split": split or "all",
        "dataset": dataset.summary,
        "arms": arms,
        "delta": delta,
        "semantic": semantic_metrics(dataset),
    }
