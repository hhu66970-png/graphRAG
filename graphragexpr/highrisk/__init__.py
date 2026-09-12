"""高风险问答服务（任务书 §3：集中为可测试的独立服务／策略模块）。

分层职责：

    storage.high_risk_read   读取三类高风险证据（含值对象与否定极性）
        ↑
    highrisk.polarity        旧极性的服务侧兼容解释
    highrisk.parsing         保守的条件解析（多剂量单元、程度、条件槽位）
    highrisk.units           组装条件证据单元
        ↑
    highrisk.claims          候选结论结构
    highrisk.verify          程序核验（硬性失败不可被模型自评推翻）
    highrisk.service         编排：读取 → 单元 → 候选 → 核验 → 状态
        ↑
    backend.api              所有检索模式共用的最终核验入口

本包不修改既有六意图枚举，也不重写实体消歧、图谱模式、六路由或路径检索。
"""

from .claims import CandidateClaim, ClaimKind
from .models import HighRiskAnswer, SubQuestionOutcome
from .service import HighRiskService, is_high_risk_question, pack_answer_payload

__all__ = [
    "CandidateClaim",
    "ClaimKind",
    "HighRiskAnswer",
    "HighRiskService",
    "SubQuestionOutcome",
    "is_high_risk_question",
    "pack_answer_payload",
]
