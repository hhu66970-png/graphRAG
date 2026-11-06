"""Unified LLM client.

业务模块一律通过本模块调用大模型，不得自行拼 HTTP 请求。提供：超时、指数退避、
最大重试、严格 JSON 校验、token 统计、request_id、版本标记、失败清单、可替换供应商。
测试使用 FakeLLMClient，不依赖在线 API。
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from ..config.settings import KGSettings
from ..domain.errors import LLMContractError

logger = logging.getLogger(__name__)


@dataclass
class LLMUsage:
    """Token 与调用计数，用于成本核算。"""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        """返回本客户端累计消耗的 token 数。"""
        return self.prompt_tokens + self.completion_tokens


@dataclass
class LLMFailure:
    request_id: str
    purpose: str
    attempts: int
    last_error: str
    payload_digest: str


def _repair_truncated_json(payload: str) -> str | None:
    """Recover the well-formed prefix of a JSON object cut off by an output limit.

    只处理「机器生成的 JSON 被截断」这一种情况：扫描括号深度，回退到最后一个
    **完整闭合的元素**，再补上必要的收尾括号。若字符串在中途被截断，则连同该
    元素一并丢弃。任何无法这样修复的输入返回 None，由调用方按契约错误处理。

    Args:
        payload: 去掉围栏后的原始返回。

    Returns:
        可解析的 JSON 文本；无法修复时为 None。

    Side effects: 无。
    """
    if not payload.startswith("{"):
        return None

    stack: list[str] = []
    in_string = False
    escaped = False
    # 最后一个「元素边界」：容器深度回到某层且刚闭合一个元素
    last_safe: tuple[int, tuple[str, ...]] | None = None

    for index, char in enumerate(payload):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            stack.append("}" if char == "{" else "]")
        elif char in "}]":
            if not stack or stack[-1] != char:
                return None
            stack.pop()
            if stack:  # 刚闭合的是某个容器里的一个元素
                last_safe = (index + 1, tuple(stack))
        # 注意：不能把逗号当安全切点。对象内部的逗号后面跟的是半个元素，
        # 从那里截断会留下缺字段的残片，Schema 校验必然失败。

    if not stack:
        return None  # 本来就闭合，说明不是截断问题
    if last_safe is None:
        return None

    cut, depth = last_safe
    text = payload[:cut].rstrip().rstrip(",")
    return text + "".join(reversed(depth))


class LLMClient(Protocol):
    """Provider-agnostic contract used by every extraction module."""

    def complete_json(
        self, *, system: str, user: str, schema: type[BaseModel], purpose: str
    ) -> BaseModel:
        """调用 LLM 并校验返回值为指定 JSON Schema。"""
        ...


class _BaseClient:
    """Shared retry / validation logic."""

    def __init__(self, settings: KGSettings) -> None:
        self._settings = settings
        self.usage = LLMUsage()
        self.failures: list[LLMFailure] = []
        self.truncation_recoveries = 0

    def _validate(self, text: str, schema: type[BaseModel], request_id: str) -> BaseModel:
        """Strict JSON parsing. 禁止从自然语言里正则猜字段作为正式结果。

        唯一的例外是**输出被长度上限截断**：此时返回的仍是机器生成的合法 JSON 前缀，
        只是结尾不完整。这种情况下抢救已完整的元素，比整块丢弃更符合「不隐藏失败」
        的原则——但必须计入 truncation_recoveries，让降级可见。
        """
        payload = text.strip()
        if payload.startswith("```"):
            # 只剥离围栏，不做任何语义猜测
            payload = payload.split("```", 2)[1]
            if payload.lstrip().startswith("json"):
                payload = payload.lstrip()[4:]
            payload = payload.rsplit("```", 1)[0]
        payload = payload.strip()

        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            repaired = _repair_truncated_json(payload)
            if repaired is None:
                raise LLMContractError(f"[{request_id}] 返回不是合法 JSON: {exc}") from exc
            try:
                data = json.loads(repaired)
            except json.JSONDecodeError:
                raise LLMContractError(f"[{request_id}] 返回不是合法 JSON: {exc}") from exc
            self.truncation_recoveries += 1
            logger.warning(
                "llm_output_truncated_recovered",
                extra={"request_id": request_id, "kept_chars": len(repaired),
                       "raw_chars": len(payload)},
            )
        try:
            return schema.model_validate(data)
        except ValidationError as exc:
            raise LLMContractError(f"[{request_id}] JSON 不满足 Schema: {exc}") from exc

    def _run_with_retry(self, call, *, purpose: str, schema, digest: str):
        """带超时与指数退避地重试 LLM 调用。"""
        request_id = uuid.uuid4().hex[:12]
        last_error = ""
        for attempt in range(1, self._settings.llm_max_retries + 1):
            try:
                text = call()
                return self._validate(text, schema, request_id)
            except LLMContractError as exc:
                # 重试不得改变事实语义，只重试同一请求
                last_error = str(exc)
                logger.warning(
                    "llm_contract_error", extra={"request_id": request_id,
                                                 "purpose": purpose, "attempt": attempt}
                )
            except Exception as exc:  # 供应商/网络错误
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning(
                    "llm_transport_error", extra={"request_id": request_id,
                                                  "purpose": purpose, "attempt": attempt}
                )
            if attempt < self._settings.llm_max_retries:
                delay = self._settings.llm_backoff_base_s ** attempt
                time.sleep(delay * (0.5 + random.random() / 2))
        self.failures.append(
            LLMFailure(request_id=request_id, purpose=purpose,
                       attempts=self._settings.llm_max_retries,
                       last_error=last_error, payload_digest=digest)
        )
        raise LLMContractError(f"[{request_id}] {purpose} 超过最大重试次数: {last_error}")


class OpenAICompatibleClient(_BaseClient):
    """OpenAI-compatible chat completion provider (DeepSeek 等)."""

    def __init__(self, settings: KGSettings, *, client: Any | None = None) -> None:
        super().__init__(settings)
        if client is not None:
            self._client = client
        else:
            from openai import OpenAI

            token = os.getenv("LLM_TOKEN")
            if not token:
                raise RuntimeError("缺少 LLM_TOKEN；密钥只从环境变量读取")
            self._client = OpenAI(
                api_key=token,
                base_url=os.getenv("LLM_ENDPOINT") or None,
                timeout=settings.llm_timeout_s,
            )
        self._model = os.getenv("LLM_MODEL", "deepseek-chat")

    def complete_json(
        self, *, system: str, user: str, schema: type[BaseModel], purpose: str
    ) -> BaseModel:
        """调用 LLM 并校验返回值为指定 JSON Schema。"""
        def _call() -> str:
            """向兼容 OpenAI 的接口发送一次 chat completion。"""
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                temperature=self._settings.llm_temperature,
                response_format={"type": "json_object"},
            )
            usage = getattr(response, "usage", None)
            self.usage.calls += 1
            if usage is not None:
                self.usage.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                self.usage.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
            return response.choices[0].message.content or ""

        digest = f"{hash(user) & 0xFFFFFFFF:08x}"
        return self._run_with_retry(_call, purpose=purpose, schema=schema, digest=digest)


@dataclass
class FakeLLMClient(_BaseClient):
    """Deterministic client for tests 。按 purpose 返回预置响应。"""

    responses: dict[str, list[str]] = field(default_factory=dict)
    settings: KGSettings = field(default_factory=KGSettings)

    def __post_init__(self) -> None:
        _BaseClient.__init__(self, self.settings)
        self._cursor: dict[str, int] = {}
        self.seen: list[tuple[str, str]] = []

    def complete_json(
        self, *, system: str, user: str, schema: type[BaseModel], purpose: str
    ) -> BaseModel:
        """调用 LLM 并校验返回值为指定 JSON Schema。"""
        self.seen.append((purpose, user))
        queue = self.responses.get(purpose)
        if not queue:
            raise LLMContractError(f"FakeLLMClient 没有为 purpose={purpose!r} 预置响应")
        index = min(self._cursor.get(purpose, 0), len(queue) - 1)
        self._cursor[purpose] = index + 1
        self.usage.calls += 1
        return self._validate(queue[index], schema, request_id=f"fake-{purpose}-{index}")
