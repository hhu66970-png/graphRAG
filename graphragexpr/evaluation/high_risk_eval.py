"""高风险组件评测：受控生成/核验消融与保持原行为的生产参考。

统计引用、数字单位同源、限制程度和安全表述等可从响应核对的指标。
题集尚待人工仲裁，语义正确性与完整性返回 not_measured。
评测只读数据库，不构建或发布批次。
"""

from __future__ import annotations

import hashlib
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

HARNESS_VERSION = "high-risk-eval-2"
ARMS = (
    "raw_text_prompt", "raw_text_verified", "conditional_units_prompt",
    "conditional_units_claims_verified", "conditional_units_verified",
)


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


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
            split, category = item.get("split", "missing"), item.get("risk_category", "missing")
            by_split[split] = by_split.get(split, 0) + 1
            by_category[category] = (
                by_category.get(category, 0) + 1
            )
        statuses = sorted({str(item.get("annotation_status", "missing")) for item in self.items})
        return {
            "total": len(self.items),
            "by_split": by_split,
            "by_category": by_category,
            "annotation_statuses": statuses,
            "semantic_scoring_allowed": bool(self.items) and not self.validation_errors and all(
                item.get("annotation_status") == ADJUDICATED for item in self.items
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
    if not items:
        errors.append("数据集不能为空")
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
        if not isinstance(item.get("should_abstain"), bool):
            errors.append(f"{ident}: should_abstain 必须是已审核的布尔提议，不能为 null")
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
        描述性计数。不同表示/生成流程的可观察范围可能不同，不能直接作为质量收益。
        **不含任何语义正确性判断**。
    """
    total = len(payloads)
    answered = abstained = not_run = failed = 0
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
        if status == "not_run":
            not_run += 1
            continue
        if status in {"error", "unresolved_subject"}:
            failed += 1
            continue
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
        "not_run": not_run,
        "failed": failed,
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
    adjudicated = bool(dataset.items) and not dataset.validation_errors and all(
        item.get("annotation_status") == ADJUDICATED for item in dataset.items)
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


# ------------------------------------------------------------- 对比臂
@dataclass
class ArmRunner:
    """构造某一组对比臂的执行函数。

    四个显式实验臂需要调用模型；未配置模型时**不假装跑过**，而是如实返回
    `not_run`，由调用方在报告里写明「未运行」。
    """

    llm: Any = None
    store: Any = None
    run_id: str | None = None
    unit_token_budget: int | None = None
    evidence_char_budget: int = 6000
    assertion_limit: int = 200
    scope: str = "question"
    _prepared: dict = field(default_factory=dict, init=False, repr=False)
    _generated: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.scope not in {"question", "component"}:
            raise ValueError("scope 必须是 question 或 component")
        if self.evidence_char_budget <= 0 or self.assertion_limit <= 0:
            raise ValueError("证据预算与断言上限必须为正数")

    def runner(self, arm: str) -> Callable[[dict], dict]:
        """构造评测运行器。"""
        if arm == "conditional_units_verified":
            return self._arm_c
        if arm in ARMS:
            return lambda item: self._controlled_arm(arm, item)
        raise ValueError(f"未知对比臂：{arm}")

    # --- C：生产链路 -------------------------------------------------
    def _arm_c(self, item: dict) -> dict:
        """跑高风险评测的核验臂（程序化校验）。"""
        from ..highrisk.service import (
            DEFAULT_ASSERTION_LIMIT, DEFAULT_UNIT_TOKEN_BUDGET,
            HighRiskService, pack_answer_payload,
        )
        from ..storage.high_risk_read import load_high_risk_assertions

        subject_ids, linked = self._resolve_subject(item, production_reference=True)
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
        payload["_audit"] = {
            "comparison_family": "production_reference",
            "scope": "legacy_component_gold_subject",
            "uses_gold_subject": True,
            "generator": "production_deterministic_rendering_no_llm",
            "production_unit_token_budget": self.unit_token_budget or DEFAULT_UNIT_TOKEN_BUDGET,
            "production_assertion_limit": DEFAULT_ASSERTION_LIMIT,
            "candidate_capture": "production_service_does_not_expose_all_candidates",
            "warning": "生产参考另行读取和筛选证据，不是控制生成器的纯核验消融",
        }
        if item.get("expected_subject") and not subject_ids:
            # 金标给了主语名，但链接器没能落到本体上。这条不能算「系统弃权」，
            # 它是「评测没有把题喂到该走的路径上」，与真实弃权分开计数。
            payload["_subject_not_resolved"] = item["expected_subject"]
        return payload

    def _prepare(self, item: dict) -> dict:
        """Freeze one read pool for all controlled arms; never read answer labels."""
        from ..storage.high_risk_read import load_high_risk_assertions
        from ..domain.high_risk import TASK_HIGH_RISK_PREDICATES

        key = _digest([item["id"], item["question"], self.scope])
        if key in self._prepared:
            return self._prepared[key]
        subject_ids, linked = self._resolve_subject(item)
        rows = []
        if subject_ids:
            rows = load_high_risk_assertions(
                self.store, run_id=self.run_id, subject_ids=subject_ids,
                predicates=TASK_HIGH_RISK_PREDICATES, limit=self.assertion_limit,
            )
        # Whole assertions only. The same source rows must fit BOTH representations;
        # this is a character bound, not a provider tokenizer or equal token counts.
        selected: list[dict] = []
        skipped: list[str] = []
        for row in sorted(rows, key=lambda value: str(value.get("assertion_id", ""))):
            candidate = selected + [row]
            raw = _raw_evidence(candidate)
            units = build_units_from_assertions(candidate)
            if max(len(_context(raw)), len(_context(_unit_evidence(units)))) > self.evidence_char_budget:
                skipped.append(str(row.get("assertion_id") or "unknown"))
            else:
                selected.append(row)
        raw = _raw_evidence(selected)
        units = build_units_from_assertions(selected)
        result = {
            "raw": raw, "units": units, "conditional": _unit_evidence(units),
            "source_ids": [entry["unit_id"] for entry in raw],
            "source_hash": _digest(raw), "skipped_source_ids": skipped,
            "linked_entities": linked,
        }
        self._prepared[key] = result
        return result

    def _controlled_arm(self, arm: str, item: dict) -> dict:
        from ..highrisk.claims import ClaimBatch
        from ..highrisk.verify import verify_claims

        if self.llm is None:
            return {"answer_status": "not_run", "abstained": None, "answer_text": "",
                    "claims": [], "evidence_units": [],
                    "_note": "未配置模型，本对比臂未运行"}
        prepared = self._prepare(item)
        raw_arm = arm.startswith("raw_text_")
        evidence = prepared["raw"] if raw_arm else prepared["conditional"]
        representation = "raw_quotes" if raw_arm else "conditional_units"
        audit = {
            "comparison_family": representation, "scope": self.scope,
            "uses_gold_subject": self.scope == "component",
            "generator": "shared_complete_json_adapter_candidates",
            "source_hash": prepared["source_hash"], "source_ids": prepared["source_ids"],
            "skipped_source_ids": prepared["skipped_source_ids"],
            "evidence": evidence, "evidence_char_budget": self.evidence_char_budget,
            "context_chars": len(_context(evidence)),
            "budget_unit": "serialized_unicode_characters_not_model_tokens",
        }
        if not prepared["linked_entities"]:
            return {"answer_status": "unresolved_subject", "abstained": None,
                    "answer_text": "", "claims": [], "evidence_units": [],
                    "_subject_not_resolved": True, "_audit": audit}
        if not evidence:
            return {"answer_status": "abstain", "abstained": True, "answer_text": "",
                    "claims": [], "evidence_units": [], "_audit": audit}
        prompt = _CANDIDATE_PROMPT.format(question=item["question"], context=_context(evidence))
        audit["prompt_hash"] = _digest(prompt)
        # Reuse exact candidates for each on/off pair, including when a stochastic
        # provider ignores temperature=0. No model notes/reasoning are exported.
        cache_key = _digest([representation, prompt])
        if cache_key not in self._generated:
            try:
                batch = self.llm.complete_json(
                    system="只输出可展示的候选结论JSON，不输出思考过程；不得使用证据外知识。",
                    user=prompt, schema=ClaimBatch, purpose="high_risk_eval_candidates",
                )
                self._generated[cache_key] = ClaimBatch.model_validate(batch)
            except Exception as exc:
                # Exception messages may contain HTTP credentials or raw responses.
                self._generated[cache_key] = {"error_type": type(exc).__name__}
        batch = self._generated[cache_key]
        if isinstance(batch, dict):
            audit.update(batch)
            return {"answer_status": "error", "abstained": None, "answer_text": "",
                    "claims": [], "evidence_units": [], "_audit": audit}
        candidates = [claim.model_dump(mode="json") for claim in batch.claims]
        audit["candidates"] = candidates
        audit["candidate_hash"] = _digest(candidates)
        rejections = []
        if arm == "raw_text_verified":
            accepted, rejections = _verify_raw_citations(batch.claims, evidence)
            audit["verification"] = "citation_existence_and_locator_only_not_semantic"
        elif arm == "conditional_units_claims_verified":
            outcome = verify_claims(batch.claims, prepared["units"])
            accepted = [claim for claim, verdict in zip(batch.claims, outcome.verdicts)
                        if verdict.passed]
            rejections = outcome.rejections
            audit["verification"] = "production_claim_verifier_only_not_full_answer_service"
        else:
            accepted = list(batch.claims)
            audit["verification"] = "disabled"
        claims = [dict(claim.model_dump(mode="json"), unit_ids=list(claim.cited_unit_ids))
                  for claim in accepted]
        cited = {ident for claim in accepted for ident in claim.cited_unit_ids}
        audit["rejections"] = rejections
        return {
            "answer_status": "evaluation_answer" if accepted else "abstain",
            "abstained": not bool(accepted),
            "answer_text": "\n".join(claim.statement for claim in accepted),
            "claims": claims,
            "evidence_units": [entry for entry in evidence if entry["unit_id"] in cited],
            "rejections": rejections, "_audit": audit,
        }

    def _resolve_subject(
        self, item: dict, *, production_reference: bool = False
    ) -> tuple[list[str] | None, list[dict]]:
        """组件范围显式使用金标主语；问句范围仅链接 question，不读答案标签。

        Returns:
            `(subject_ids, linked_entities)`；数据集没给主语、或链接器落不到
            本体上时返回 `(None, [])`。**不编造 id**：编出来的 id 查不到任何
            证据，产出的弃权与「真的没有证据」在响应里长得一模一样，会把
            工具缺陷记成系统成绩。
        """
        if self.scope == "question" and not production_reference:
            linked = _link(self.store, self.run_id, item["question"])
            return ([str(entity["canonical_id"]) for entity in linked
                     if entity.get("canonical_id")] or None, linked)
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


_CANDIDATE_PROMPT = """只依据以下证据回答问题。输出 ClaimBatch JSON：claims 数组内每条结论包含
claim_id、kind、statement、cited_unit_ids（引用下列 unit_id）；剂量、条件和极性等
asserted_* 字段仅填写证据明确支持的值。没有证据时 claims=[]。
不要输出思考过程、置信度或未被证据支持的补充。
证据：
{context}
问题：{question}
"""


def _context(evidence: list[dict]) -> str:
    return json.dumps(evidence, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _raw_evidence(rows: list[dict]) -> list[dict]:
    records = []
    for row in rows:
        quote = str(row.get("evidence_quote") or "")
        chunk_ids = list(row.get("chunk_ids") or [])
        pages = [p for p in (row.get("chunk_page_starts") or []) if p is not None]
        ident = str(row.get("assertion_id") or "raw:" + _digest([quote, chunk_ids])[:20])
        records.append({
            "unit_id": ident, "evidence_quote": quote, "chunk_ids": chunk_ids,
            "page_label": "、".join(str(p) for p in sorted(set(pages))),
            "is_citable": bool(quote.strip() and chunk_ids and row.get("match_mode") != "not_found"),
        })
    return records


def _unit_evidence(units) -> list[dict]:
    return [dict(unit.model_dump(mode="json"), is_citable=unit.is_citable(),
                 page_label=str(unit.page_start) if unit.page_available else "") for unit in units]


def _verify_raw_citations(claims, evidence: list[dict]):
    """Real citation-contract baseline; this does NOT establish semantic entailment."""
    by_id = {record["unit_id"]: record for record in evidence}
    accepted, rejected = [], []
    for claim in claims:
        failures = []
        if not claim.cited_unit_ids:
            failures.append("missing_citation")
        for ident in claim.cited_unit_ids:
            if ident not in by_id:
                failures.append("unknown_evidence")
            elif not by_id[ident]["is_citable"]:
                failures.append("evidence_not_locatable")
        if failures:
            rejected.append({"claim_id": claim.claim_id,
                             "hard_failures": sorted(set(failures))})
        else:
            accepted.append(claim)
    return accepted, rejected



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
    run_metadata: dict | None = None,
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
    if dataset.validation_errors:
        raise ValueError("数据集校验失败，不能执行评测")
    if split not in {None, "dev", "test"}:
        raise ValueError("未知 split")
    items = [item for item in dataset.items if split is None or item["split"] == split]
    arms: dict[str, Any] = {}
    predictions: dict[str, dict] = {}
    for arm, run in runners.items():
        payloads: dict[str, dict] = {}
        for item in items:
            payloads[item["id"]] = _prediction_record(run(item))
            payloads[item["id"]]["question"] = item["question"]
        predictions[arm] = payloads
        arms[arm] = {
            "questions": len(items),
            "mechanical": mechanical_metrics(payloads),
        }

    delta = None
    ran = [arm for arm, result in arms.items()
           if result["mechanical"]["questions"] and
           not result["mechanical"]["not_run"] and not result["mechanical"]["failed"]]
    if len(ran) == 2:
        left, right = ran
        a = arms[left]["mechanical"]
        b = arms[right]["mechanical"]
        delta = {
            "interpretation": "descriptive_counts_only_not_semantic_quality_or_causal_gain",
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
        "harness_version": HARNESS_VERSION,
        "metadata": {key: value for key, value in (run_metadata or {}).items() if key in {
            "evaluation_id", "harness_version", "created_at_utc", "code_sha", "branch",
            "evaluation_code_sha256",
            "working_tree_dirty", "dataset_file_sha256", "ingest_run_id", "arms",
            "scope_experimental_arms", "model", "temperature", "max_output_tokens",
            "evidence_char_budget", "assertion_limit", "generation_pair_policy",
            "representation_comparison", "production_reference",
        }},
        "dataset_sha256": _digest(list(dataset.items)),
        "split": split or "all",
        "dataset": dataset.summary,
        "arms": arms,
        "predictions": predictions,
        "controlled_pairs": _controlled_pairs(predictions),
        "delta": delta,
        "semantic": semantic_metrics(HighRiskDataset(tuple(items))),
        "limitations": [
            "question/component 都是高风险组件评测，不是 API 全链路检索比较",
            "component 使用 expected_subject；生产参考保留原来的金标主语接线",
            "字符预算不是模型 tokenizer 的 token 数，表示之间实际输入长度不同",
            "production_reference 使用不同生成与筛选流程，不是纯核验消融",
            "机械计数有不同可观察范围，不能直接代替语义正确率和完整性",
        ],
    }


def _prediction_record(payload: dict) -> dict:
    """Only export visible outputs and our audit contract, not arbitrary debug data."""
    allowed = {"answer", "answer_text", "answer_status", "abstained", "claims",
               "evidence_units", "rejections", "warnings", "_subject_not_resolved"}
    result = {key: payload[key] for key in allowed if key in payload}
    audit_fields = {
        "comparison_family", "scope", "uses_gold_subject", "generator", "candidate_capture",
        "production_unit_token_budget", "production_assertion_limit",
        "warning", "source_hash", "source_ids", "skipped_source_ids", "evidence",
        "evidence_char_budget", "context_chars", "budget_unit", "prompt_hash",
        "candidates", "candidate_hash", "verification", "rejections", "error_type",
    }
    if "_audit" in payload:
        result["_audit"] = {key: value for key, value in payload["_audit"].items()
                            if key in audit_fields}
    return _public_json(result)


def _public_json(value):
    """Do not serialize model notes/reasoning or generic credential/debug fields."""
    if isinstance(value, dict):
        forbidden = {"notes", "reasoning", "analysis", "thinking", "chain_of_thought",
                     "api_key", "token", "secret", "authorization", "llm_token", "password"}
        return {key: _public_json(item) for key, item in value.items()
                if key.lower() not in forbidden}
    if isinstance(value, list):
        return [_public_json(item) for item in value]
    return value


def _controlled_pairs(predictions: dict[str, dict]) -> list[dict]:
    results = []
    for left, right, effect in (
        ("raw_text_prompt", "raw_text_verified", "citation_locator_gate"),
        ("conditional_units_prompt", "conditional_units_claims_verified", "claim_verifier"),
    ):
        if left not in predictions or right not in predictions:
            continue
        checks = []
        for ident, payload in predictions[left].items():
            other = predictions[right].get(ident, {})
            a, b = payload.get("_audit", {}), other.get("_audit", {})
            checks.append(all(a.get(key) and a.get(key) == b.get(key)
                              for key in ("source_hash", "prompt_hash", "candidate_hash")))
        results.append({"from": left, "to": right, "effect": effect,
                        "same_sources_prompt_candidates": bool(checks) and all(checks),
                        "semantic_effect_measured": False})
    return results


def adjudication_template(dataset: HighRiskDataset) -> dict:
    """Export pending review work, never convert proposed labels into human gold."""
    if dataset.validation_errors:
        raise ValueError("数据集校验失败，不能导出仲裁模板")
    rows = []
    for item in dataset.items:
        rows.append({
            "id": item["id"], "split": item["split"], "question": item["question"],
            "annotation_status": "pending_human_review",
            "seed_proposal_not_gold": {key: item.get(key) for key in (
                "expected_subject", "required_predicates", "gold_answer_points", "gold_pages",
                "forbidden_unsupported_claims", "should_abstain", "annotation_status")},
            "reviews": [{"reviewer_id": "", "checked_source_pages": [],
                         "answer_points": [], "forbidden_claims": [],
                         "should_abstain": None, "notes": ""} for _ in range(2)],
            "adjudication": {"adjudicator_id": "", "decision": "pending", "notes": ""},
            "paired_variant_draft": {
                "pair_id": item["id"] + "-pair", "question": "", "changed_field": "",
                "changed_from": "", "changed_to": "", "source_pages": [],
                "expected_relation_to_seed": "pending_human_review",
                "gold_answer_points": [], "should_abstain": None,
                "split": item["split"], "annotation_status": "pending_human_review",
            },
        })
    return {"template_version": 1, "source_dataset_sha256": _digest(list(dataset.items)),
            "semantic_scoring_allowed": False, "items": rows,
            "instructions": [
                "两位审核者先独立核原文/页码；不看系统答案形成金标，再交第三方仲裁分歧",
                "成对题只改一个明确条件，人工填写新问句及变化，不能推导或复制医学结论",
                "同一 pair 保持相同 split，冻结 test 后不得用于调阈值或修改提示词",
                "保留教材版本、页码、审核人和仲裁记录；此模板不是可直接评分的数据集",
            ]}
