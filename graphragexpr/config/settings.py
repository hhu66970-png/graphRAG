"""Central settings and feature flags (SKILL §5.1 / §12.4).

规则：
  - 业务文件不得散落魔法常量，所有阈值/预算/开关集中在此。
  - 所有默认值都注明来源与依据。
  - 评测运行必须保存配置快照：用 Settings.snapshot()。
  - 消融开关（SKILL §10 / 设计 §10 的 KG0–KG4）全部可关闭。
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class KGSettings:
    """建图管线的全部可调参数。字段顺序 = 管线顺序。"""

    # ---- 版本（进入缓存键；任何改动必须升版本，SKILL §3.7）----------------
    # 1.1.0: 抽取 Prompt 增加「每片段最多 40 条关系」上限，避免超长输出被截断
    prompt_version: str = "kg-extract-1.1.0"
    schema_version: str = "kg-ontology-1.0.0"
    embedding_version: str = "bge-small-zh-v1.5"

    # ---- LLM（SKILL §8）--------------------------------------------------
    llm_timeout_s: float = _env_float("KG_LLM_TIMEOUT_S", 90.0)
    llm_max_retries: int = _env_int("KG_LLM_MAX_RETRIES", 3)
    llm_backoff_base_s: float = _env_float("KG_LLM_BACKOFF_BASE_S", 1.5)
    llm_temperature: float = 0.0

    # ---- I1 实体消歧（设计 §2.1.1-1）--------------------------------------
    # 初始值，必须经开发集标定后再冻结（设计 §2.1.2-E）；标定前不得写入报告结论。
    link_auto_confirm_threshold: float = _env_float("KG_LINK_AUTO", 0.94)
    link_candidate_threshold: float = _env_float("KG_LINK_CAND", 0.84)
    link_max_candidates: int = _env_int("KG_LINK_MAX_CAND", 8)
    link_max_edit_distance: int = _env_int("KG_LINK_MAX_EDIT", 1)

    # ---- I2 关系验证（设计 §2.1.2-A）--------------------------------------
    rule_reject_below: float = _env_float("KG_RULE_REJECT", 0.35)
    rule_accept_above: float = _env_float("KG_RULE_ACCEPT", 0.85)
    # 已按设计 §2.1.2-E 用开发集标定曲线冻结（scripts/calibrate_thresholds.py）：
    # 在 375 味药的教材字段参照集上扫描 0.00–1.00，0.80 是 Recall 明显下滑前的
    # 最大阈值——Recall 85.6%→85.2%（-0.4pp）而 Prec↓ 67.2%→69.0%（+1.8pp）；
    # 再往上到 0.85 召回骤降至 78.1%。建图取向 Precision 优先，故取 0.80。
    final_accept_threshold: float = _env_float("KG_FINAL_ACCEPT", 0.80)

    # ---- 证据定位（设计 §2.1.2-B）-----------------------------------------
    evidence_max_edit_distance: int = _env_int("KG_EVID_MAX_EDIT", 2)
    evidence_min_quote_len: int = _env_int("KG_EVID_MIN_LEN", 2)

    # ---- I3 跨块补全（设计 §2.1.2-D）--------------------------------------
    cross_chunk_max_neighbors: int = _env_int("KG_XCHUNK_NEIGHBORS", 1)

    # ---- 存储（SKILL §9.2）------------------------------------------------
    write_batch_size: int = _env_int("KG_WRITE_BATCH", 500)

    # ---- 发布门槛（设计 §9.3）---------------------------------------------
    gate_min_evidence_coverage: float = _env_float("KG_GATE_EVID_COV", 1.0)
    gate_max_unknown_predicate_rate: float = _env_float("KG_GATE_UNKNOWN_PRED", 0.02)
    # 证据覆盖率只衡量「已写入的断言」，当抽取整块失败时它恒为 1.0，
    # 无法发现「语料只覆盖了两成」这种情况，必须单独设块级失败率门槛。
    gate_max_failed_chunk_rate: float = _env_float("KG_GATE_FAILED_CHUNKS", 0.05)

    # ---- 消融开关（设计 §10：KG0–KG4）-------------------------------------
    use_entity_resolution: bool = _env_bool("KG_USE_I1", True)   # KG1
    use_relation_validation: bool = _env_bool("KG_USE_I2", True)  # KG2
    use_cross_chunk: bool = _env_bool("KG_USE_I3", True)          # KG3
    use_assertion_layer: bool = _env_bool("KG_USE_ASSERTION", True)  # KG4
    use_llm_verifier: bool = _env_bool("KG_USE_LLM_VERIFY", True)

    def snapshot(self) -> dict:
        """评测/构建必须随结果保存的配置快照（SKILL §5.1、§14.4）。"""
        return asdict(self)

    def cache_key_parts(self) -> tuple[str, ...]:
        """影响抽取结果的版本项，进入缓存键（设计 §5）。"""
        return (self.prompt_version, self.schema_version, f"T{self.llm_temperature}")

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        return tuple(f.name for f in fields(cls))


DEFAULT_SETTINGS = KGSettings()


@dataclass(frozen=True)
class RetrievalSettings:
    """Budgets for intent-aware retrieval (design sections 7.2 and 8.1)."""

    entity_link_limit: int = _env_int("RETRIEVAL_ENTITY_LIMIT", 16)
    directed_result_limit: int = _env_int("RETRIEVAL_DIRECTED_LIMIT", 25)
    comparison_fact_limit: int = _env_int("RETRIEVAL_COMPARE_FACT_LIMIT", 120)
    comparison_chunk_limit: int = _env_int("RETRIEVAL_COMPARE_CHUNK_LIMIT", 80)
    comparison_per_entity: int = _env_int("RETRIEVAL_COMPARE_PER_ENTITY", 3)
    comparison_total: int = _env_int("RETRIEVAL_COMPARE_TOTAL", 8)
    path_max_hops: int = _env_int("RETRIEVAL_PATH_MAX_HOPS", 3)
    path_limit: int = _env_int("RETRIEVAL_PATH_LIMIT", 20)
    token_budget: int = _env_int("RETRIEVAL_TOKEN_BUDGET", 4000)

    def __post_init__(self) -> None:
        for setting in fields(self):
            if getattr(self, setting.name) <= 0:
                raise ValueError(f"{setting.name} must be positive")

    def snapshot(self) -> dict:
        return asdict(self)


DEFAULT_RETRIEVAL_SETTINGS = RetrievalSettings()
