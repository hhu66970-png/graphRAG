"""Extraction quality metrics against the structured-field reference (设计 §9.2).

指标口径（必须连同数字一起引用，否则会被误读）：

- **Recall@field**：教材字段里明确写出的取值，图谱抽到了多少。这是主指标，口径干净。
- **Precision@field**：图谱给出的取值里，有多少落在教材字段内。**这是下界**——
  图谱合理地包含字段之外的事实（【应用】段落、其它章节交叉引用），
  它们不在字段里但并不算错，因此真实精确率高于此值。
- 两者都**不是**「事实正确概率」（SKILL §7.3）。

evaluation 层只读各层公开接口，不复制业务逻辑（SKILL §5）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.ids import normalize_text
from ..storage.neo4j_store import Neo4jStore
from .reference import HerbReference, parse_efficacies, parse_nature_flavor

# 参照字段 -> 图谱谓词
FIELD_PREDICATES: dict[str, str] = {
    "nature_flavor": "HAS_NATURE_FLAVOR",
    "meridians": "ENTERS_MERIDIAN",
    "efficacies": "HAS_EFFICACY",
}

_GRAPH_VALUES = """
MATCH (h:Entity {entity_type: 'HERB'})
WHERE h.run_id = $run AND (h.preferred_name = $name OR $name IN h.aliases)
MATCH (h)-[e]->(o:Entity)
WHERE e.fact_key IS NOT NULL AND e.active_run_id = $run AND type(e) = $predicate
RETURN collect(DISTINCT o.preferred_name) AS values
"""

_GRAPH_DOSAGE = """
MATCH (h:Entity {entity_type: 'HERB'})
WHERE h.run_id = $run AND (h.preferred_name = $name OR $name IN h.aliases)
MATCH (a:Assertion {run_id: $run, predicate: 'HAS_DOSAGE'})-[:SUBJECT]->(h)
RETURN collect(a.object_value_json) AS values
"""


@dataclass
class FieldScore:
    """FieldScore：本文件定义的数据结构或组件。"""
    predicate: str
    gold: int = 0
    predicted: int = 0
    matched: int = 0
    herbs_evaluated: int = 0
    herbs_fully_recalled: int = 0
    misses: list[tuple[str, str]] = field(default_factory=list)   # (herb, 漏掉的取值)
    extras: list[tuple[str, str]] = field(default_factory=list)   # (herb, 字段外取值)

    @property
    def recall(self) -> float:
        """计算字段级召回率（教材字段被抽到的比例）。"""
        return self.matched / self.gold if self.gold else 0.0

    @property
    def field_precision(self) -> float:
        """计算字段精确率下界（预测落在教材字段内的比例）。"""
        return self.matched / self.predicted if self.predicted else 0.0

    @property
    def f1(self) -> float:
        """由召回与字段精确率下界计算 F1。"""
        p, r = self.field_precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        """把评测结果转成可序列化字典。"""
        return {
            "predicate": self.predicate,
            "gold_values": self.gold,
            "predicted_values": self.predicted,
            "matched": self.matched,
            "recall_at_field": round(self.recall, 4),
            "precision_at_field_lower_bound": round(self.field_precision, 4),
            "f1_lower_bound": round(self.f1, 4),
            "herbs_evaluated": self.herbs_evaluated,
            "herbs_fully_recalled": self.herbs_fully_recalled,
        }


def canonicalize_predicted(field_name: str, values: set[str]) -> set[str]:
    """Bring graph values to the same granularity as the reference before matching.

    教材字段是拆开的（性味「辛」「微苦」「温」，归经「肺经」「膀胱经」），
    而图谱常把整句抽成一个取值（「辛、微苦，温」「归肺、膀胱经」）。
    不先归一就比，会把「粒度不同」误报成「没抽到」——首次评测时性味召回率
    只有 35.5%，实际信息都在图里，纯属口径错配。

    因此这里对图谱取值套用与 reference 完全相同的解析规则。
    """
    if field_name == "meridians":
        out: set[str] = set()
        for value in values:
            _, _, meridians = parse_nature_flavor(value)
            if meridians:
                out |= meridians
            else:  # 已经是「肺经」这种单项
                name = value.strip().rstrip("经")
                if name:
                    out.add(f"{name}经")
        return out
    if field_name == "nature_flavor":
        out = set()
        for value in values:
            flavours, _, _ = parse_nature_flavor(value)
            out |= flavours or {value.strip()}
        return out
    if field_name == "efficacies":
        out = set()
        for value in values:
            terms = parse_efficacies(value)
            out |= terms or {value.strip()}
        return out
    return values


def _match(
    gold: set[str], predicted: set[str], *, min_token_len: int = 2
) -> tuple[set[str], set[str], set[str]]:
    """Match on normalised text, with containment as a fallback.

    教材写「发汗解表」，图谱可能抽成「发汗」「解表」两条，或反过来抽成
    「发汗解表散寒」。严格相等会低估召回，因此允许一方包含另一方，
    但要求较短一方 ≥2 字，避免单字碰瓷。
    """
    norm_gold = {normalize_text(g): g for g in gold if g}
    norm_pred = {normalize_text(p): p for p in predicted if p}

    hit_gold: set[str] = set()
    hit_pred: set[str] = set()
    for gk, gv in norm_gold.items():
        for pk, pv in norm_pred.items():
            shortest = min(len(gk), len(pk))
            if gk == pk or (shortest >= min_token_len and (gk in pk or pk in gk)):
                hit_gold.add(gv)
                hit_pred.add(pv)
    missed = {v for k, v in norm_gold.items() if v not in hit_gold}
    extra = {v for k, v in norm_pred.items() if v not in hit_pred}
    return hit_gold, missed, extra


def evaluate(
    store: Neo4jStore,
    references: dict[str, HerbReference],
    *,
    run_id: str,
    max_examples: int = 20,
) -> dict:
    """Score the graph against the structured-field reference set.

    Args:
        store: 连接句柄。
        references: `reference.build_reference` 的输出。
        run_id: 被评测的 IngestRun。
        max_examples: 每个谓词保留的漏检/额外取值样例数，用于失败分析。

    Returns:
        {"per_predicate": {...}, "overall": {...}, "examples": {...}}

    Side effects: 只读。
    """
    scores = {f: FieldScore(predicate=p) for f, p in FIELD_PREDICATES.items()}
    herbs_not_in_graph: list[str] = []

    for ref in references.values():
        present = False
        for field_name, predicate in FIELD_PREDICATES.items():
            gold: set[str] = getattr(ref, field_name)
            if not gold:
                continue
            rows = store.run(
                _GRAPH_VALUES, name=ref.herb, run=run_id, predicate=predicate
            )
            predicted = set(rows[0]["values"]) if rows and rows[0]["values"] else set()
            if predicted:
                present = True

            score = scores[field_name]
            # 两侧用同一套规则归一后再比，避免把粒度差异误报成漏检
            predicted = canonicalize_predicted(field_name, predicted)
            # 性味成分天然是单字（辛/甘/苦/寒/温），此处放开到 1
            hit, missed, extra = _match(
                gold, predicted, min_token_len=1 if field_name == "nature_flavor" else 2
            )
            score.gold += len(gold)
            score.predicted += len(predicted)
            score.matched += len(hit)
            score.herbs_evaluated += 1
            if not missed:
                score.herbs_fully_recalled += 1
            score.misses.extend((ref.herb, m) for m in sorted(missed)[:3])
            score.extras.extend((ref.herb, e) for e in sorted(extra)[:3])
        if not present:
            herbs_not_in_graph.append(ref.herb)

    total_gold = sum(s.gold for s in scores.values())
    total_matched = sum(s.matched for s in scores.values())
    total_pred = sum(s.predicted for s in scores.values())

    return {
        "per_predicate": {f: s.as_dict() for f, s in scores.items()},
        "overall": {
            "reference_herbs": len(references),
            "herbs_absent_from_graph": len(herbs_not_in_graph),
            "gold_values": total_gold,
            "predicted_values": total_pred,
            "matched": total_matched,
            "recall_at_field": round(total_matched / total_gold, 4) if total_gold else 0.0,
            "precision_at_field_lower_bound": (
                round(total_matched / total_pred, 4) if total_pred else 0.0
            ),
        },
        "examples": {
            f: {
                "missed": s.misses[:max_examples],
                "extra_beyond_field": s.extras[:max_examples],
            }
            for f, s in scores.items()
        },
        "absent_herbs_sample": herbs_not_in_graph[:20],
    }


_ALL_VALUES_WITH_SCORE = """
MATCH (a:Assertion {run_id: $run})-[:SUBJECT]->(h:Entity {entity_type: 'HERB'})
MATCH (a)-[:OBJECT]->(o:Entity)
WHERE a.predicate IN $predicates AND a.polarity = 'POSITIVE'
RETURN h.preferred_name AS herb, a.predicate AS predicate,
       o.preferred_name AS value, a.final_score AS score,
       a.validation_status AS status
"""


def load_scored_values(store: Neo4jStore, *, run_id: str) -> list[dict]:
    """Load every positive assertion's (herb, predicate, value, score) for calibration."""
    return store.run(
        _ALL_VALUES_WITH_SCORE, run=run_id,
        predicates=list(FIELD_PREDICATES.values()),
    )


def sweep_thresholds(
    references: dict[str, HerbReference],
    scored: list[dict],
    thresholds: list[float],
) -> list[dict]:
    """Recompute Recall/Precision@field at each acceptance threshold (设计 §2.1.2-E).

    离线标定：断言已带 `final_score`，无需重跑抽取即可扫描阈值。
    这正是文档要求的「先画曲线再冻结阈值」，而不是把 0.60 直接写进报告。

    Args:
        references: 参照集。
        scored: `load_scored_values` 的输出。
        thresholds: 待扫描的 final_score 下限。

    Returns:
        每个阈值一行：保留取值数、命中数、Recall、Precision(下界)、F1(下界)。

    Side effects: 无。
    """
    predicate_to_field = {v: k for k, v in FIELD_PREDICATES.items()}
    by_herb: dict[tuple[str, str], list[tuple[str, float]]] = {}
    for row in scored:
        field_name = predicate_to_field.get(row["predicate"])
        if not field_name:
            continue
        key = (normalize_text(row["herb"]), field_name)
        by_herb.setdefault(key, []).append((row["value"], row["score"] or 0.0))

    results: list[dict] = []
    for threshold in thresholds:
        gold_total = matched_total = pred_total = 0
        for ref in references.values():
            herb_key = normalize_text(ref.herb)
            for field_name in FIELD_PREDICATES:
                gold: set[str] = getattr(ref, field_name)
                if not gold:
                    continue
                raw = {
                    value for value, score in by_herb.get((herb_key, field_name), [])
                    if score >= threshold
                }
                predicted = canonicalize_predicted(field_name, raw)
                hit, _, _ = _match(
                    gold, predicted,
                    min_token_len=1 if field_name == "nature_flavor" else 2,
                )
                gold_total += len(gold)
                pred_total += len(predicted)
                matched_total += len(hit)
        recall = matched_total / gold_total if gold_total else 0.0
        precision = matched_total / pred_total if pred_total else 0.0
        f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
        results.append({
            "threshold": threshold,
            "predicted_values": pred_total,
            "matched": matched_total,
            "recall_at_field": round(recall, 4),
            "precision_at_field_lower_bound": round(precision, 4),
            "f1_lower_bound": round(f1, 4),
        })
    return results
