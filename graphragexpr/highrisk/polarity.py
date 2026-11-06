"""旧极性的服务侧兼容解释。

问题背景：抽取提示词按「不宜／忌／禁／勿」等词填 `NEGATIVE` 极性，
但教材里最常见的那类句子恰恰是**肯定地陈述了一条禁忌**：

    麻黄「不宜与藜芦同用」   → 否定的对象是「同用」这个互动
    麻黄「孕妇慎用」         → 否定的对象是「用」这个动作

两种句式里 `NEGATIVE` 都**不**表示「这条禁忌不存在」。若服务侧把 NEGATIVE
直接读成「关系不成立」，就会得出「麻黄可以和藜芦同用」或「麻黄孕妇可用」——
这正是最危险的错误。因此否定作用域必须分场景解释：

- **配伍禁忌**：识别「与…同用／合用／配伍」句式被否定 → 关系成立；
  缺少该句式时不猜（例如把配伍谓词错标到人群对象上），判为无法确定。
- **使用禁忌**：程度词（禁用／忌用／慎用／不宜…）本身就是关系成立的依据，
  不需要「同用」句式。

本模块的职责边界：

- **只解释，不修改。** 返回解释结果、依据与歧义标记；原始谓词、原始极性和原文
  由调用方原样保留。历史 `fact_key` 不做任何静默改写。
- **不放宽。** 不把所有 NEGATIVE 断言开放为正向服务边，也不把所有禁忌断言
  一律改成 POSITIVE；「无禁忌」「不忌」这类否定关系本身的写法优先判为关系不成立。
- **判不出来就不判。** 关系含义、限制程度或对象无法确定时返回 `UNDETERMINED`，
  由上层降级为原文展示或拒答，而不是输出确定性结论。

纯函数，无副作用，可独立测试。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.high_risk import PolarityVerdict

# 「否定的是同用这个动作，而不是禁忌关系本身」。
# 命中这些线索时，配伍禁忌关系**成立**——这正是最容易解释反的一类。
_NEGATED_ACT_CUES: tuple[str, ...] = (
    "不宜与",
    "不可与",
    "不能与",
    "忌与",
    "勿与",
    "不得与",
    "严禁与",
    "切忌与",
    "不宜同用",
    "不可同用",
    "不能同用",
    "不宜合用",
    "不可合用",
    "不能合用",
    "不宜配伍",
    "不可配伍",
    "不宜同服",
    "不可同服",
    "不能同服",
    "不宜并用",
    "不可并用",
    "忌同用",
    "禁同用",
)

# 「否定的是对该人群／病证的使用」——使用禁忌场景的主句式。
# 配伍禁忌与使用禁忌的否定作用域不同，必须分开处理：
#     麻黄「不宜与藜芦同用」  → 被否定的是「同用」这个互动 → 配伍禁忌成立
#     麻黄「孕妇慎用」        → 被否定的是「用」这个动作 → 使用禁忌成立
# 后一句里根本没有「同用」句式，如果只按配伍线索判断就会一律得到
# 「无法判定」，从而在最需要给结论的孕妇／儿童场景上全部弃权。
# 「无禁忌」「不忌」这类**否定关系本身**的写法已在上面的线索表里
# 先行拦下，因此这里可以安全地把程度词读成使用禁忌成立。
_RESTRICTED_USE_CUES: tuple[str, ...] = (
    "禁用", "禁服", "严禁", "禁止", "切忌",
    "忌用", "忌服", "不宜", "忌", "勿", "不可", "不能", "不得",
    "慎用", "慎服", "谨慎",
    "不宜使用", "不宜服用", "禁止使用", "避免使用", "不适用",
)

# 「否定的是禁忌关系本身」：这类句子表示**不存在**该禁忌。
# 与上面的区别必须保留，否则「无配伍禁忌」会被读成一条肯定的配伍禁忌。
_RELATION_NEGATION_CUES: tuple[str, ...] = (
    "无禁忌",
    "没有禁忌",
    "无配伍禁忌",
    "没有配伍禁忌",
    "未见禁忌",
    "未发现禁忌",
    "未记载禁忌",
    "不存在禁忌",
    "无此禁忌",
    "并无禁忌",
    "不忌",
    "不予禁忌",
    "不属禁忌",
    "无使用禁忌",
    "无特殊禁忌",
)

# 通用否定线索：仅用于判断「极性标注是否与原文一致」，不单独决定关系成立与否。
_NEGATION_CUES: tuple[str, ...] = (
    "不宜", "不可", "不能", "忌", "禁", "勿", "不得", "严禁", "切忌",
)


@dataclass(frozen=True)
class PolarityInterpretation:
    """一次极性解释的结果。所有字段都只描述「怎么读」，不改动原数据。"""

    verdict: PolarityVerdict
    basis: str
    ambiguous: bool
    raw_polarity: str | None
    matched_cues: tuple[str, ...] = ()

    @property
    def relation_holds(self) -> bool | None:
        """关系是否成立。`None` 表示无法判定，等价于「不得输出确定结论」。"""
        if self.verdict is PolarityVerdict.RELATION_HOLDS:
            return True
        if self.verdict is PolarityVerdict.RELATION_DOES_NOT_HOLD:
            return False
        return None


def _first_match(text: str, cues: tuple[str, ...]) -> tuple[str, ...]:
    """返回模式的第一次匹配。"""
    return tuple(cue for cue in cues if cue in text)


def interpret_polarity(
    *,
    predicate: str,
    raw_polarity: str | None,
    evidence_quote: str,
    object_name: str | None = None,
) -> PolarityInterpretation:
    """解释一条高风险断言的极性含义。

    Args:
        predicate: 稳定谓词 ID（支持范围内为三类高风险谓词之一）。
        raw_polarity: 数据库中保存的原始极性，原样传入。
        evidence_quote: 该断言的逐字原文。
        object_name: 相关对象展示名，仅用于生成人类可读的依据说明。

    Returns:
        PolarityInterpretation。调用方必须同时保存 `raw_polarity` 与本结果，
        不能只留其一。

    Side effects: 无。本函数不修改任何持久化数据。
    """
    quote = evidence_quote or ""
    normalized_polarity = (raw_polarity or "").strip().upper() or None
    target = f"「{object_name}」" if object_name else "相关对象"

    relation_negations = _first_match(quote, _RELATION_NEGATION_CUES)
    if relation_negations:
        return PolarityInterpretation(
            verdict=PolarityVerdict.RELATION_DOES_NOT_HOLD,
            basis=(
                f"原文出现「{relation_negations[0]}」，否定的是禁忌关系本身，"
                f"表示不存在关于{target}的该禁忌"
            ),
            ambiguous=False,
            raw_polarity=normalized_polarity,
            matched_cues=relation_negations,
        )

    if predicate == "INCOMPATIBLE_WITH" and object_name:
        reverse_cue = f"反{object_name}"
        if reverse_cue in quote or "相反" in quote:
            cue = reverse_cue if reverse_cue in quote else "相反"
            return PolarityInterpretation(
                verdict=PolarityVerdict.RELATION_HOLDS,
                basis=(
                    f"原文以十八反／相反句式肯定关于{target}的配伍禁忌"
                    f"（线索：「{cue}」）"
                ),
                ambiguous=False,
                raw_polarity=normalized_polarity,
                matched_cues=(cue,),
            )

    act_negations = _first_match(quote, _NEGATED_ACT_CUES)
    if act_negations:
        polarity_note = (
            "原始极性为 NEGATIVE，但否定的是「同用／合用」这个动作，"
            if normalized_polarity == "NEGATIVE"
            else "原文否定的是「同用／合用」这个动作，"
        )
        return PolarityInterpretation(
            verdict=PolarityVerdict.RELATION_HOLDS,
            basis=(
                f"{polarity_note}该动作被否定恰恰**肯定**了关于{target}的禁忌关系成立"
                f"（线索：「{act_negations[0]}」）"
            ),
            ambiguous=False,
            raw_polarity=normalized_polarity,
            matched_cues=act_negations,
        )

    # 使用禁忌（对象是人群／病证）的场景：程度词直接限定「能不能用」，
    # 不存在「同用动作」可被否定，因此该禁忌关系成立。
    # 只对 CONTRAINDICATED_IN 适用；配伍禁忌缺少「与…同用」句式时仍判为无法确定。
    if predicate == "CONTRAINDICATED_IN":
        restricted = _first_match(quote, _RESTRICTED_USE_CUES)
        if restricted:
            polarity_note = (
                "原始极性为 NEGATIVE，"
                if normalized_polarity == "NEGATIVE"
                else ""
            )
            return PolarityInterpretation(
                verdict=PolarityVerdict.RELATION_HOLDS,
                basis=(
                    f"{polarity_note}原文「{restricted[0]}」限定的是对该对象的**使用**，"
                    f"被否定的是使用行为本身，因此关于{target}的使用禁忌成立"
                ),
                ambiguous=False,
                raw_polarity=normalized_polarity,
                matched_cues=restricted,
            )

    if normalized_polarity == "POSITIVE":
        return PolarityInterpretation(
            verdict=PolarityVerdict.RELATION_HOLDS,
            basis="原始极性为 POSITIVE，且原文未见否定禁忌关系的线索",
            ambiguous=False,
            raw_polarity=normalized_polarity,
        )

    if normalized_polarity == "NEGATIVE":
        generic = _first_match(quote, _NEGATION_CUES)
        return PolarityInterpretation(
            verdict=PolarityVerdict.UNDETERMINED,
            basis=(
                "原始极性为 NEGATIVE，但原文未出现可判定的否定作用域线索，"
                "无法确定否定的是「同用动作」还是「禁忌关系本身」"
                + (f"（通用否定词：「{generic[0]}」）" if generic else "")
            ),
            ambiguous=True,
            raw_polarity=normalized_polarity,
            matched_cues=generic,
        )

    return PolarityInterpretation(
        verdict=PolarityVerdict.UNDETERMINED,
        basis="缺少可用的极性标注，原文也未提供足以判定关系成立与否的线索",
        ambiguous=True,
        raw_polarity=normalized_polarity,
    )
