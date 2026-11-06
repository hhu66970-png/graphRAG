"""高风险条件的保守解析：多剂量单元、限制程度、条件槽位。

与 `extract/dosage.py` 中旧解析器的关系
----------------------------------------
旧解析器服务于**建图管线**，一次只产出一组结果，并且为了「让数据看起来完整」
做了两件在高风险问答里不能接受的事：

1. 倒置范围自动交换（`9～3` 自动改成 `3～9`）；
2. 数值缺单位时在**整段**里搜索任意单位来补。

本模块不修改旧解析器（避免影响既有建图行为与历史数据），而是为高风险链路提供
一个**保守**的替代实现：

- 倒置范围只标记 `is_inverted`，**绝不交换**；交换与否由核验与人工判断。
- 单位只允许与数值**紧邻**配对；找不到就留空并记 `parse_notes`，
  绝不到别处取一个单位拼上来。
- 按条件片段切分，为每个数值建立**局部窗口**，窗口内的用法／生制品／人群
  才允许绑定到该数值。窗口边界即上一处数值与下一处数值，从结构上杜绝
  「整段各取一个数字、单位、用法拼成一条事实」。

纯函数，无副作用；所有跨度都相对传入的 `text` 原串。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..domain.high_risk import (
    DosageBinding,
    FieldSpan,
    RestrictionSeverity,
)

_NUMBER = r"\d+(?:\.\d+)?"
_RANGE_SEPARATOR = r"[~～\-—－]|至|到"

# 计量单位。与旧解析器保持同一集合，仅补充中药饮片常见的计数单位。
_UNIT_ALTERNATION = "毫克|毫升|克|mg|ml|g|钱|两|片|枚|粒|支|丸"

_UNIT_CANONICAL: dict[str, str] = {
    "克": "g", "g": "g",
    "毫克": "mg", "mg": "mg",
    "毫升": "ml", "ml": "ml",
    "钱": "钱", "两": "两", "片": "片", "枚": "枚",
    "粒": "粒", "支": "支", "丸": "丸",
}

# 一处剂量出现：数值（可选范围）+ **紧邻**的可选单位。
# 单位后允许有空格，但中间不允许插入其他汉字，这样「3 天，每次 9g」里的 3
# 不会被配到后面的 g。
_DOSAGE_OCCURRENCE_RE = re.compile(
    rf"(?P<min>{_NUMBER})\s*"
    rf"(?:(?:{_RANGE_SEPARATOR})\s*(?P<max>{_NUMBER})\s*)?"
    rf"(?P<unit>{_UNIT_ALTERNATION})?",
    re.IGNORECASE,
)

# 条件片段分隔符。只用强分隔符切分：「煎服，3～9g」必须留在同一片段内，
# 否则用法与数值会被切开。
_CLAUSE_SPLIT_RE = re.compile(r"[；;。\n！？!?]")

# 用法／给药途径。顺序有意义：先匹配更具体的。
_ROUTE_CUES: tuple[str, ...] = (
    "煎汤外洗", "水煎服", "开水泡服", "入丸散", "研末服", "冲服",
    "先煎", "后下", "另煎", "包煎", "烊化", "泡服", "含服", "口服",
    "内服", "外用", "煎服", "研末", "吞服", "灌服", "敷患处", "调敷",
)

# 生制品／药物形态。中药剂量常随炮制而异（生品与制品剂量不可互相套用）。
# 只收多字线索：单字「生」「制」「炒」会命中「生姜」「炮制」「制剂」等无关词，
# 造成把别处的形态条件绑到本数值上。
_PREPARATION_CUES: tuple[str, ...] = (
    "生品", "制品", "炮制品", "熟品", "鲜品", "干品", "饮片",
    "生用", "炒用", "炙用", "蜜炙", "酒炙", "醋炙", "盐炙", "姜炙",
    "麸炒", "土炒", "米炒", "清炒", "砂炒", "煅用", "煨用", "炒炭",
    "酒炒", "醋炒", "盐炒", "姜汁炒", "蜜制", "酒制", "醋制", "盐制",
    "姜制", "胆汁制", "未炮制",
)

# 频次／时间条件。注意：这些只是**从原文抽取**，绝不自动补「每日」「每次」。
_FREQUENCY_CUES: tuple[str, ...] = (
    "一日三次", "一日二次", "每日三次", "每日二次", "一日一次", "每日一次",
    "分三次", "分二次", "分三次服", "分次服", "频服", "顿服", "每服",
    "每日", "每天", "一日", "日服", "空腹服", "饭后服", "饭前服",
    "睡前服", "温服", "冷服", "不拘时服",
)

# 适用人群。
_POPULATION_CUES: tuple[str, ...] = (
    "婴幼儿", "哺乳期", "妊娠期", "小儿", "儿童", "孕妇", "妊娠",
    "产妇", "年老", "老人", "体弱", "久病", "肝肾功能不全", "肾功能不全",
    "肝功能不全", "新生儿",
)

# 限制程度三档，**顺序即优先级**：禁用 > 忌用 > 慎用。
# 三档必须保持区分——「禁用」与「慎用」在临床含义上相差很远，
# 用同一个布尔值抹平后，核验就无法发现「慎用被写成了禁用」。
_SEVERITY_CUES: tuple[tuple[RestrictionSeverity, tuple[str, ...]], ...] = (
    (RestrictionSeverity.PROHIBITED, ("禁用", "禁服", "严禁", "禁止", "切忌")),
    (RestrictionSeverity.AVOID, ("忌用", "忌服", "不宜", "忌", "勿", "不可", "不能", "不得")),
    (RestrictionSeverity.CAUTION, ("慎用", "慎服", "谨慎")),
)


#: 全部条件线索词。渲染层用它判断「一段剂量条件窗口里是否还有线索词表
#: 未覆盖的实质内容」——只有确定窗口已被结构化槽位吸收干净，才不必把
#: 窗口原文一并带出。集中在此处定义，避免渲染层另抄一份词表后发生漂移。
ALL_CONDITION_CUES: tuple[str, ...] = (
    _ROUTE_CUES
    + _PREPARATION_CUES
    + _FREQUENCY_CUES
    + _POPULATION_CUES
    + tuple(cue for _, cues in _SEVERITY_CUES for cue in cues)
)


def classify_severity(text: str) -> tuple[RestrictionSeverity, str | None]:
    """Extract the restriction degree wording from a text span.

    Args:
        text: 候选原文片段（通常是证据句或条件窗口）。

    Returns:
        (程度枚举, 命中的原文词)。未命中时返回 `(UNSPECIFIED, None)`。
        禁用／忌用／慎用三档必须保持区分，不能用布尔值抹平。

    Side effects: 无。
    """
    if not text:
        return RestrictionSeverity.UNSPECIFIED, None
    for severity, cues in _SEVERITY_CUES:
        for cue in cues:
            if cue in text:
                return severity, cue
    return RestrictionSeverity.UNSPECIFIED, None


def _canonical_unit(raw_unit: str | None) -> str | None:
    """把剂量单位规范成统一写法（如 g）。"""
    if not raw_unit:
        return None
    return _UNIT_CANONICAL.get(raw_unit.strip().lower(), raw_unit.strip().lower())


def _nearest_cue(
    window: str, cues: tuple[str, ...], anchor_start: int, anchor_end: int
) -> tuple[str, int, int] | None:
    """Pick the cue occurrence closest to the anchor span inside `window`.

    距离相同时**必须优先取数值前面（同侧）的那个线索**，否则会把后一个剂量的
    条件绑到本数值上：「煎服 3～9g，研末服 1～2g」中，`煎服`（前，距离 1）与
    `研末服`（后，距离 1）等距，只按「线索更长优先」会选中后一个，于是
    3～9g 被写成研末服用量；「制品 3～6g，生品 1.5～3g」同理，两个分支会一起
    变成生品。窗口的前半段（上一处数值之后到本数值）本来就是**本数值**的条件，
    后半段才属于下一个数值，因此同侧优先是有依据的，而不是启发式偏好。

    这个错误核验抓不到：单元自身已经错了，渲染出来的结论与它自洽。原实现只在
    「生品 1.5～3g，制品 3～6g」这种正序写法下因为线索表恰好把「生品」排在前面
    而侥幸正确，把两个分支对调就会暴露——因此测试必须覆盖对调后的写法。
    """
    best: tuple[int, int, int, str, int, int] | None = None
    for cue in cues:
        position = window.find(cue)
        while position >= 0:
            end = position + len(cue)
            if end <= anchor_start:
                side, distance = 0, anchor_start - end      # 数值之前，属本数值
            elif position >= anchor_end:
                side, distance = 1, position - anchor_end   # 数值之后，属下一个数值
            else:  # 与锚点重叠
                side, distance = 0, 0
            # 依次比较：同侧优先 → 距离更近 → 线索更长（「生品」优先于「生」）
            key = (side, distance, -len(cue))
            if best is None or key < best[0]:
                best = (key, cue, position, end)
            position = window.find(cue, position + 1)
    if best is None:
        return None
    _, cue, start, end = best
    return cue, start, end


def _field_span(
    window: str, window_offset: int, cue: str, start: int, end: int
) -> FieldSpan:
    """在原文中定位某个字段的字符跨度。"""
    return FieldSpan(
        start=window_offset + start,
        end=window_offset + end,
        text=window[start:end] or cue,
    )


@dataclass(frozen=True)
class _Occurrence:
    """一处剂量数值在原文中的位置与紧邻单位。"""

    start: int
    end: int
    value_min: float
    value_max: float
    unit: str | None
    unit_raw: str | None
    notes: list[str] = field(default_factory=list)


def _is_page_reference(text: str, start: int, end: int) -> bool:
    """页码区间（如「第 66-67 页」）不是剂量，必须排除。

    页码与剂量在形态上完全一样（数字或数字范围），只看数字无法区分，
    因此需要看紧邻的「第」「页」字样。
    """
    before = text[max(0, start - 3) : start]
    after = text[end : end + 3]
    return "页" in after or "第" in before or "页" in before


def _find_occurrences(text: str) -> list[_Occurrence]:
    """查找关键词在文本中的出现位置。"""
    occurrences: list[_Occurrence] = []
    for match in _DOSAGE_OCCURRENCE_RE.finditer(text):
        raw_unit = match.group("unit")
        max_text = match.group("max")
        # 孤立数字（无范围、无紧邻单位）不当作剂量，避免把「3 天」「2 次」等
        # 计数误收为剂量。范围形式即使缺单位也保留，但会记 note。
        if max_text is None and raw_unit is None:
            continue
        if _is_page_reference(text, match.start(), match.end()):
            continue
        value_min = float(match.group("min"))
        value_max = float(max_text) if max_text is not None else value_min
        notes: list[str] = []
        if raw_unit is None:
            notes.append("unit_not_adjacent:数值紧邻处没有单位，未到片段他处补单位")
        occurrences.append(
            _Occurrence(
                start=match.start(),
                end=match.end(),
                value_min=value_min,
                value_max=value_max,
                unit=_canonical_unit(raw_unit),
                unit_raw=raw_unit,
                notes=notes,
            )
        )
    return occurrences


def _clause_ranges(text: str) -> list[tuple[int, int]]:
    """Split into strong-delimiter clauses, keeping offsets into `text`."""
    ranges: list[tuple[int, int]] = []
    cursor = 0
    for separator in _CLAUSE_SPLIT_RE.finditer(text):
        ranges.append((cursor, separator.start()))
        cursor = separator.end()
    ranges.append((cursor, len(text)))
    return [(start, end) for start, end in ranges if end > start]


def parse_dosage_units(text: str) -> list[DosageBinding]:
    """Parse every dosage occurrence in a text into its own bound unit.

    Args:
        text: 教材中的剂量表述，如
            "煎服，3～9g；挽救虚脱可用 15～30g。生品 1.5～3g，制品 3～6g。"

    Returns:
        DosageBinding 列表，**每处数值一项**，各自绑定其局部窗口内的用法、
        生制品、人群等条件。解析不出任何剂量时返回空列表。

    Guarantees:
        - 不交换倒置范围的上下界，只置 `is_inverted` 并记 note；
        - 不在数值窗口之外寻找单位；
        - 不补充原文没有的条件（不写「成人」「每日」「每次」）。

    Side effects: 无。
    """
    if not text or not text.strip():
        return []

    clause_ranges = _clause_ranges(text)
    clause_occurrences = [
        _find_occurrences(text[start:end]) for start, end in clause_ranges
    ]

    units: list[DosageBinding] = []
    for clause_index, (clause_start, clause_end) in enumerate(clause_ranges):
        clause = text[clause_start:clause_end]
        occurrences = clause_occurrences[clause_index]
        if not occurrences:
            continue

        # 后续**自身不含独立剂量**的小句仍在描述同一个剂量，是它的强制条件：
        # 「煎服，3～15g；先煎，久煎，口尝至无麻辣感为度」里的后半句若被丢弃，
        # 附子最关键的减毒要求就消失了，而这恰恰是不能丢的内容。
        # 因此把这串无剂量小句并入本小句最后一个数值的条件窗口；
        # 一旦遇到自带剂量的小句就停止，绝不跨剂量串条件。
        trailing_end = clause_end
        for following in range(clause_index + 1, len(clause_ranges)):
            if clause_occurrences[following]:
                break
            trailing_end = clause_ranges[following][1]
        trailing_text = text[clause_end:trailing_end].strip("；;。\n \t")

        for index, occurrence in enumerate(occurrences):
            # 局部窗口 = 上一处数值之后 到 下一处数值之前。
            # 这是本模块最重要的结构约束：条件只能来自数值自己的这一段，
            # 从根上防止把整段不同位置的数字/单位/用法拼成一条事实。
            # 线索归属哪个数值由 `_nearest_cue` 按距离判定
            # （「生品 1.5～3g，制品 3～6g」中两个程度词各归其主）。
            window_start = occurrences[index - 1].end if index > 0 else 0
            window_end = (
                occurrences[index + 1].start
                if index + 1 < len(occurrences)
                else len(clause)
            )
            window = clause[window_start:window_end]
            window_offset = clause_start + window_start
            anchor_start = occurrence.start - window_start
            anchor_end = occurrence.end - window_start

            is_last_in_clause = index + 1 == len(occurrences)

            notes = list(occurrence.notes)
            if trailing_text and is_last_in_clause:
                notes.append(
                    f"后续小句「{trailing_text}」自身不含剂量，已作为本数值的条件并入，"
                    "不是另一处独立剂量"
                )
            field_spans: dict[str, FieldSpan] = {}

            route_hit = _nearest_cue(window, _ROUTE_CUES, anchor_start, anchor_end)
            preparation_hit = _nearest_cue(
                window, _PREPARATION_CUES, anchor_start, anchor_end
            )
            frequency_hit = _nearest_cue(
                window, _FREQUENCY_CUES, anchor_start, anchor_end
            )
            population_hit = _nearest_cue(
                window, _POPULATION_CUES, anchor_start, anchor_end
            )

            route = preparation = frequency = population = None
            if route_hit:
                route = route_hit[0]
                field_spans["route"] = _field_span(
                    window, window_offset, *route_hit
                )
            if preparation_hit:
                preparation = preparation_hit[0]
                field_spans["preparation"] = _field_span(
                    window, window_offset, *preparation_hit
                )
            if frequency_hit:
                frequency = frequency_hit[0]
                field_spans["frequency"] = _field_span(
                    window, window_offset, *frequency_hit
                )
            if population_hit:
                population = population_hit[0]
                field_spans["population"] = _field_span(
                    window, window_offset, *population_hit
                )

            is_inverted = occurrence.value_min > occurrence.value_max
            if is_inverted:
                # 只标记，不修正：程序擅自交换上下界可能把 OCR 错误固化成事实。
                notes.append(
                    "inverted_range:原文上界小于下界，未做自动交换，需人工或核验判断"
                )

            field_spans["value"] = FieldSpan(
                start=clause_start + occurrence.start,
                end=clause_start + occurrence.end,
                text=clause[occurrence.start : occurrence.end],
            )

            units.append(
                DosageBinding(
                    raw=clause[occurrence.start : occurrence.end],
                    value_min=occurrence.value_min,
                    value_max=occurrence.value_max,
                    unit=occurrence.unit,
                    unit_raw=occurrence.unit_raw,
                    is_single_value=occurrence.value_min == occurrence.value_max,
                    is_inverted=is_inverted,
                    preparation=preparation,
                    route=route,
                    frequency=frequency,
                    population=population,
                    condition_text=(
                        text[
                            clause_start + window_start : (
                                trailing_end if is_last_in_clause else clause_start + window_end
                            )
                        ].strip()
                        or None
                    ),
                    field_spans=field_spans,
                    parse_notes=notes,
                )
            )
    return units


def extract_population(text: str) -> str | None:
    """First population cue in `text`, if any. 不补默认人群。"""
    hit = _nearest_cue(text or "", _POPULATION_CUES, 0, 0)
    return hit[0] if hit else None


def extract_preparation(text: str) -> str | None:
    """First preparation/form cue in `text`, if any. 不推断药物形态。"""
    hit = _nearest_cue(text or "", _PREPARATION_CUES, 0, 0)
    return hit[0] if hit else None


def extract_route(text: str) -> str | None:
    """First route cue in `text`, if any. 不推断给药途径。

    与 `extract_population` / `extract_preparation` 同一约定：只认原文或问题里
    **明说**的用法，没有就返回 None（调用方据此判定「未指定用法」，
    而不是当成某个默认用法）。
    """
    hit = _nearest_cue(text or "", _ROUTE_CUES, 0, 0)
    return hit[0] if hit else None
