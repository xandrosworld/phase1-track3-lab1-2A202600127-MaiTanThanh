from __future__ import annotations

import json

from .schemas import JudgeResult, QAExample


ACTOR_SYSTEM = """
You answer multi-hop QA questions using only the provided context.
Rules:
- Use the context as the source of truth.
- Resolve all hops before answering.
- If reflection memory is provided, use it to avoid repeating the previous mistake.
- Return only the final answer text, with no explanation, no bullet points, and no JSON.
"""


EVALUATOR_SYSTEM = """
You are a strict self-evaluator for a multi-hop QA agent.
Judge whether the candidate answer is fully supported by the provided context and whether it completes every hop required by the question.
Return valid JSON with keys:
- score: integer 0 or 1
- reason: short explanation
- missing_evidence: list of strings
- spurious_claims: list of strings
Do not return markdown fences or extra text.
"""


REFLECTOR_SYSTEM = """
You analyze failed attempts for a Reflexion agent.
Return valid JSON with keys:
- attempt_id: integer
- failure_reason: short description of the mistake
- lesson: one reusable lesson learned from the failure
- next_strategy: one concrete strategy for the next attempt
Be specific to the current question and context. Do not return markdown fences or extra text.
"""


def format_context(example: QAExample) -> str:
    return "\n\n".join(
        f"Title: {chunk.title}\nText: {chunk.text}" for chunk in example.context
    )


def build_actor_user_prompt(example: QAExample, reflection_memory: list[str]) -> str:
    reflection_block = (
        "\n".join(f"- {item}" for item in reflection_memory)
        if reflection_memory
        else "- None"
    )
    return f"""Question:
{example.question}

Context:
{format_context(example)}

Reflection memory:
{reflection_block}

Answer with the shortest correct final answer only."""


def build_evaluator_user_prompt(example: QAExample, answer: str) -> str:
    return f"""Question:
{example.question}

Candidate answer:
{answer}

Context:
{format_context(example)}"""


def build_reflector_user_prompt(
    example: QAExample,
    attempt_id: int,
    answer: str,
    judge: JudgeResult,
    reflection_memory: list[str],
) -> str:
    prior_memory = (
        "\n".join(f"- {item}" for item in reflection_memory)
        if reflection_memory
        else "- None"
    )
    return f"""Question:
{example.question}

Context:
{format_context(example)}

Attempt id:
{attempt_id}

Candidate answer:
{answer}

Evaluator feedback:
{json.dumps(judge.model_dump(), ensure_ascii=True, indent=2)}

Existing reflection memory:
{prior_memory}"""
