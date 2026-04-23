from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .runtime import AgentRuntime, MockRuntime, infer_failure_mode
from .schemas import AttemptTrace, QAExample, ReflectionEntry, RunRecord
from .utils import normalize_answer


@dataclass
class BaseAgent:
    agent_type: Literal["react", "reflexion"]
    max_attempts: int = 1
    runtime: AgentRuntime = field(default_factory=MockRuntime)

    def run(self, example: QAExample) -> RunRecord:
        reflection_memory: list[str] = []
        reflections: list[ReflectionEntry] = []
        traces: list[AttemptTrace] = []
        final_answer = ""
        final_score = 0
        final_judge = None

        for attempt_id in range(1, self.max_attempts + 1):
            answer_result = self.runtime.answer(
                example, attempt_id, self.agent_type, reflection_memory
            )
            judge_result = self.runtime.evaluate(example, answer_result.value)

            trace = AttemptTrace(
                attempt_id=attempt_id,
                answer=answer_result.value,
                score=judge_result.value.score,
                reason=judge_result.value.reason,
                token_estimate=answer_result.token_count + judge_result.token_count,
                latency_ms=answer_result.latency_ms + judge_result.latency_ms,
            )

            final_answer = answer_result.value
            final_score = judge_result.value.score
            final_judge = judge_result.value

            if (
                self.agent_type == "reflexion"
                and judge_result.value.score == 0
                and attempt_id < self.max_attempts
            ):
                reflection_result = self.runtime.reflect(
                    example,
                    attempt_id,
                    answer_result.value,
                    judge_result.value,
                    reflection_memory,
                )
                reflection = reflection_result.value
                reflections.append(reflection)
                reflection_memory.append(
                    f"Lesson: {reflection.lesson} Strategy: {reflection.next_strategy}"
                )
                trace.reflection = reflection
                trace.token_estimate += reflection_result.token_count
                trace.latency_ms += reflection_result.latency_ms

            traces.append(trace)
            if judge_result.value.score == 1:
                break

        total_tokens = sum(t.token_estimate for t in traces)
        total_latency = sum(t.latency_ms for t in traces)
        final_is_correct = (
            normalize_answer(final_answer) == normalize_answer(example.gold_answer)
        )
        failure_mode = infer_failure_mode(
            example,
            final_is_correct,
            final_answer,
            final_judge,
        )
        return RunRecord(
            qid=example.qid,
            question=example.question,
            gold_answer=example.gold_answer,
            agent_type=self.agent_type,
            predicted_answer=final_answer,
            is_correct=final_is_correct,
            attempts=len(traces),
            token_estimate=total_tokens,
            latency_ms=total_latency,
            failure_mode=failure_mode,
            reflections=reflections,
            traces=traces,
        )


class ReActAgent(BaseAgent):
    def __init__(self, runtime: AgentRuntime | None = None) -> None:
        super().__init__(agent_type="react", max_attempts=1, runtime=runtime or MockRuntime())


class ReflexionAgent(BaseAgent):
    def __init__(
        self,
        max_attempts: int = 3,
        runtime: AgentRuntime | None = None,
    ) -> None:
        super().__init__(
            agent_type="reflexion",
            max_attempts=max_attempts,
            runtime=runtime or MockRuntime(),
        )
