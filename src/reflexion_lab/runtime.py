from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from .mock_runtime import (
    FAILURE_MODE_BY_QID,
    actor_answer as mock_actor_answer,
    evaluator as mock_evaluator,
    reflector as mock_reflector,
)
from .prompts import (
    ACTOR_SYSTEM,
    EVALUATOR_SYSTEM,
    REFLECTOR_SYSTEM,
    build_actor_user_prompt,
    build_evaluator_user_prompt,
    build_reflector_user_prompt,
)
from .schemas import JudgeResult, QAExample, ReflectionEntry
from .utils import normalize_answer


@dataclass
class RuntimeResult:
    value: Any
    token_count: int
    latency_ms: int


class AgentRuntime(Protocol):
    def answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> RuntimeResult: ...

    def evaluate(self, example: QAExample, answer: str) -> RuntimeResult: ...

    def reflect(
        self,
        example: QAExample,
        attempt_id: int,
        answer: str,
        judge: JudgeResult,
        reflection_memory: list[str],
    ) -> RuntimeResult: ...


@dataclass
class MockRuntime:
    def answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> RuntimeResult:
        answer = mock_actor_answer(example, attempt_id, agent_type, reflection_memory)
        token_count = 240 + (attempt_id * 45) + (90 if agent_type == "reflexion" else 0)
        latency_ms = 120 + (attempt_id * 25) + (40 if agent_type == "reflexion" else 0)
        return RuntimeResult(value=answer, token_count=token_count, latency_ms=latency_ms)

    def evaluate(self, example: QAExample, answer: str) -> RuntimeResult:
        judge = mock_evaluator(example, answer)
        return RuntimeResult(value=judge, token_count=80, latency_ms=30)

    def reflect(
        self,
        example: QAExample,
        attempt_id: int,
        answer: str,
        judge: JudgeResult,
        reflection_memory: list[str],
    ) -> RuntimeResult:
        reflection = mock_reflector(example, attempt_id, judge)
        return RuntimeResult(value=reflection, token_count=120, latency_ms=90)


@dataclass
class OpenAICompatibleRuntime:
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.0
    timeout_s: int = 120

    def __post_init__(self) -> None:
        self.base_url = (
            self.base_url
            or os.getenv("REFLEXION_API_BASE")
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        self.api_key = (
            self.api_key
            or os.getenv("REFLEXION_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or "dummy"
        )

    def answer(
        self,
        example: QAExample,
        attempt_id: int,
        agent_type: str,
        reflection_memory: list[str],
    ) -> RuntimeResult:
        text, token_count, latency_ms = self._chat(
            [
                {"role": "system", "content": ACTOR_SYSTEM.strip()},
                {
                    "role": "user",
                    "content": build_actor_user_prompt(example, reflection_memory),
                },
            ]
        )
        return RuntimeResult(value=text.strip(), token_count=token_count, latency_ms=latency_ms)

    def evaluate(self, example: QAExample, answer: str) -> RuntimeResult:
        text, token_count, latency_ms = self._chat(
            [
                {"role": "system", "content": EVALUATOR_SYSTEM.strip()},
                {
                    "role": "user",
                    "content": build_evaluator_user_prompt(example, answer),
                },
            ]
        )
        try:
            judge = JudgeResult.model_validate(self._parse_json_object(text))
        except Exception:
            judge = JudgeResult(
                score=0,
                reason="Fallback evaluator used because the model response was not valid JSON.",
                missing_evidence=["Need to verify that the answer is fully grounded in the provided context."],
                spurious_claims=[] if not answer.strip() else [answer],
            )
        return RuntimeResult(value=judge, token_count=token_count, latency_ms=latency_ms)

    def reflect(
        self,
        example: QAExample,
        attempt_id: int,
        answer: str,
        judge: JudgeResult,
        reflection_memory: list[str],
    ) -> RuntimeResult:
        text, token_count, latency_ms = self._chat(
            [
                {"role": "system", "content": REFLECTOR_SYSTEM.strip()},
                {
                    "role": "user",
                    "content": build_reflector_user_prompt(
                        example, attempt_id, answer, judge, reflection_memory
                    ),
                },
            ]
        )
        try:
            reflection = ReflectionEntry.model_validate(self._parse_json_object(text))
        except Exception:
            reflection = ReflectionEntry(
                attempt_id=attempt_id,
                failure_reason=judge.reason,
                lesson="Verify each hop against the provided context before finalizing the answer.",
                next_strategy="Trace the entities step by step and rewrite the final answer using only grounded evidence.",
            )
        return RuntimeResult(
            value=reflection,
            token_count=token_count,
            latency_ms=latency_ms,
        )

    def _chat(self, messages: list[dict[str, str]]) -> tuple[str, int, int]:
        request = urllib.request.Request(
            self._chat_url(),
            data=json.dumps(
                {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature,
                }
            ).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        started_at = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM request failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc.reason}") from exc
        latency_ms = int((time.perf_counter() - started_at) * 1000)
        return (
            self._extract_text(payload).strip(),
            self._extract_token_count(payload),
            latency_ms,
        )

    def _chat_url(self) -> str:
        assert self.base_url is not None
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    @staticmethod
    def _extract_text(payload: dict[str, Any]) -> str:
        if "choices" in payload and payload["choices"]:
            content = payload["choices"][0].get("message", {}).get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict)
                )
        if "message" in payload:
            content = payload["message"].get("content", "")
            if isinstance(content, str):
                return content
        raise RuntimeError("Unsupported chat response payload: missing assistant content.")

    @staticmethod
    def _extract_token_count(payload: dict[str, Any]) -> int:
        usage = payload.get("usage")
        if isinstance(usage, dict):
            total = usage.get("total_tokens")
            if isinstance(total, int):
                return total
            prompt = usage.get("prompt_tokens")
            completion = usage.get("completion_tokens")
            if isinstance(prompt, int) and isinstance(completion, int):
                return prompt + completion
        prompt_eval = payload.get("prompt_eval_count")
        eval_count = payload.get("eval_count")
        if isinstance(prompt_eval, int) and isinstance(eval_count, int):
            return prompt_eval + eval_count
        total = payload.get("total_tokens")
        if isinstance(total, int):
            return total
        return 0

    @staticmethod
    def _parse_json_object(text: str) -> dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(cleaned[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        raise RuntimeError(f"Model did not return a valid JSON object: {text}")


def build_runtime(
    mode: str,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
) -> AgentRuntime:
    normalized_mode = mode.strip().lower()
    if normalized_mode == "mock":
        return MockRuntime()
    if normalized_mode == "real":
        if not model:
            raise ValueError("Real mode requires a model name.")
        return OpenAICompatibleRuntime(
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
        )
    raise ValueError(f"Unsupported mode: {mode}")


def infer_failure_mode(
    example: QAExample,
    is_correct: bool,
    final_answer: str,
    judge: JudgeResult | None,
) -> str:
    if is_correct:
        return "none"
    if example.qid in FAILURE_MODE_BY_QID:
        return FAILURE_MODE_BY_QID[example.qid]
    if judge and judge.missing_evidence and not judge.spurious_claims:
        return "incomplete_multi_hop"
    if judge and judge.spurious_claims:
        return "entity_drift"
    if normalize_answer(final_answer) == "":
        return "looping"
    return "wrong_final_answer"
