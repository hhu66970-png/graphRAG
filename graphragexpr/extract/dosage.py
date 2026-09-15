"""Dosage parsed into a value object (设计 §2.1.2-C).

「剂量」不宜作为普通可合并实体：`9～30g`、`煎服`、`儿童减量`、`先煎` 往往带条件。
本模块把剂量文本解析为标准化数值 + 给药方式 + 条件，作为 Assertion.object_value
保存；即便前端仍要展示 DOSAGE 节点，数值与条件也必须落在这里而不是只存字符串。
"""

from __future__ import annotations

import re

from ..domain.models import DosageValue

# 数值范围：9～30g / 3-9 g / 1.5—3克 / 3～9
_RANGE_RE = re.compile(
    r"(?P<min>\d+(?:\.\d+)?)\s*[~～\-—－至到]\s*(?P<max>\d+(?:\.\d+)?)\s*(?P<unit>克|g|mg|毫克|ml|毫升|钱|两|片|枚)?",
    re.IGNORECASE,
)
# 单值：9g / 0.5克 / 每次 3 g
_SINGLE_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>克|g|mg|毫克|ml|毫升|钱|两|片|枚)",
    re.IGNORECASE,
)

_UNIT_CANONICAL = {
    "克": "g", "g": "g", "G": "g",
    "mg": "mg", "毫克": "mg",
    "ml": "ml", "毫升": "ml",
    "钱": "钱", "两": "两", "片": "片", "枚": "枚",
}

# 给药方式/煎法。顺序有意义：先匹配更具体的。
_ROUTES = (
    "水煎服", "煎汤外洗", "煎服", "先煎", "后下", "另煎", "烊化", "冲服",
    "入丸散", "研末服", "研末", "泡服", "开水泡服", "外用", "内服", "含服", "口服",
)

_CONDITION_PATTERNS = (
    r"(?:小儿|儿童|婴幼儿)[^，。；]{0,10}(?:减量|酌减|慎用|禁用)",
    r"(?:孕妇|妊娠)[^，。；]{0,10}(?:慎用|忌用|禁用|减量)",
    r"(?:年老|体弱|久病)[^，。；]{0,10}(?:减量|酌减)",
    r"(?:大剂量|重症|急症)[^，。；]{0,12}",
    r"(?:鲜品|生品|炮制品)[^，。；]{0,12}",
)


def _canonical_unit(raw_unit: str | None) -> str | None:
    """把剂量单位规范成统一写法（如 g）。"""
    if not raw_unit:
        return None
    return _UNIT_CANONICAL.get(raw_unit.strip(), raw_unit.strip().lower())


def parse_dosage(text: str) -> DosageValue:
    """Parse a dosage phrase into a structured value object.

    Args:
        text: 教材中的剂量表述，如 "煎服，9～30g；儿童减量"。

    Returns:
        DosageValue；解析不出数值时 value_min/value_max 为 None，但 raw 始终保留。
        调用方据此判断是否需要人工复核——高风险谓词下无数值应进 REVIEW。

    Side effects: 无。
    """
    raw = (text or "").strip()
    if not raw:
        return DosageValue(raw="")

    value_min: float | None = None
    value_max: float | None = None
    unit: str | None = None

    range_match = _RANGE_RE.search(raw)
    if range_match:
        value_min = float(range_match.group("min"))
        value_max = float(range_match.group("max"))
        unit = _canonical_unit(range_match.group("unit"))
        if value_min > value_max:          # OCR 可能把 30～9 写反
            value_min, value_max = value_max, value_min
    else:
        single_match = _SINGLE_RE.search(raw)
        if single_match:
            value_min = value_max = float(single_match.group("value"))
            unit = _canonical_unit(single_match.group("unit"))

    # 单位可能与数值分离，例如「9～30」后面才出现「g」
    if unit is None:
        unit_match = re.search(r"(克|g|mg|毫克|ml|毫升|钱|两)", raw, re.IGNORECASE)
        if unit_match:
            unit = _canonical_unit(unit_match.group(1))

    route = next((r for r in _ROUTES if r in raw), None)

    condition = None
    for pattern in _CONDITION_PATTERNS:
        found = re.search(pattern, raw)
        if found:
            condition = found.group(0)
            break

    return DosageValue(
        value_min=value_min,
        value_max=value_max,
        unit=unit,
        route=route,
        condition=condition,
        raw=raw,
    )


def is_well_formed(dosage: DosageValue) -> bool:
    """高风险剂量断言的最低结构要求：有数值且有单位。"""
    return dosage.value_min is not None and dosage.unit is not None
