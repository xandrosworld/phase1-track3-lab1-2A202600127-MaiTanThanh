from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from .schemas import ReportPayload, RunRecord


def summarize(records: list[RunRecord]) -> dict:
    grouped: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[record.agent_type].append(record)
    summary: dict[str, dict] = {}
    for agent_type, rows in grouped.items():
        summary[agent_type] = {
            "count": len(rows),
            "em": round(mean(1.0 if r.is_correct else 0.0 for r in rows), 4),
            "avg_attempts": round(mean(r.attempts for r in rows), 4),
            "avg_token_estimate": round(mean(r.token_estimate for r in rows), 2),
            "avg_latency_ms": round(mean(r.latency_ms for r in rows), 2),
        }
    if "react" in summary and "reflexion" in summary:
        summary["delta_reflexion_minus_react"] = {
            "em_abs": round(summary["reflexion"]["em"] - summary["react"]["em"], 4),
            "attempts_abs": round(
                summary["reflexion"]["avg_attempts"] - summary["react"]["avg_attempts"],
                4,
            ),
            "tokens_abs": round(
                summary["reflexion"]["avg_token_estimate"]
                - summary["react"]["avg_token_estimate"],
                2,
            ),
            "latency_abs": round(
                summary["reflexion"]["avg_latency_ms"] - summary["react"]["avg_latency_ms"],
                2,
            ),
        }
    return summary


def failure_breakdown(records: list[RunRecord]) -> dict:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for record in records:
        grouped[record.agent_type][record.failure_mode] += 1
    breakdown = {agent: dict(counter) for agent, counter in grouped.items()}
    overall = Counter(record.failure_mode for record in records)
    breakdown["overall"] = dict(overall)
    return breakdown


def build_discussion(summary: dict, failure_modes: dict) -> str:
    react = summary.get("react", {})
    reflexion = summary.get("reflexion", {})
    delta = summary.get("delta_reflexion_minus_react", {})
    react_failures = failure_modes.get("react", {})
    reflexion_failures = failure_modes.get("reflexion", {})
    return (
        "This benchmark compares a single-shot ReAct baseline against a multi-attempt "
        "Reflexion agent that stores short lessons between attempts. "
        f"ReAct reached EM={react.get('em', 0)} while Reflexion reached EM={reflexion.get('em', 0)}, "
        f"for an absolute gain of {delta.get('em_abs', 0)}. "
        "The extra accuracy comes with a cost in attempts, tokens, and latency because failed runs trigger "
        "an evaluator call and a reflector step before retrying. "
        f"Observed ReAct failure modes were {json.dumps(react_failures, ensure_ascii=True)}, "
        f"while Reflexion failure modes were {json.dumps(reflexion_failures, ensure_ascii=True)}. "
        "In practice, reflection memory is most useful when the first attempt stops after the first hop, "
        "confuses the final entity, or ignores evidence from the second supporting passage. "
        "Remaining errors usually indicate either weak evaluator feedback, low-quality context grounding, "
        "or a model that repeats the same answer even after receiving a correction strategy."
    )


def build_report(
    records: list[RunRecord],
    dataset_name: str,
    mode: str = "mock",
    author: dict | None = None,
    model_name: str | None = None,
) -> ReportPayload:
    summary = summarize(records)
    failure_modes = failure_breakdown(records)
    examples = [
        {
            "qid": r.qid,
            "agent_type": r.agent_type,
            "gold_answer": r.gold_answer,
            "predicted_answer": r.predicted_answer,
            "is_correct": r.is_correct,
            "attempts": r.attempts,
            "failure_mode": r.failure_mode,
            "reflection_count": len(r.reflections),
        }
        for r in records
    ]
    return ReportPayload(
        meta={
            "dataset": dataset_name,
            "mode": mode,
            "num_records": len(records),
            "agents": sorted({r.agent_type for r in records}),
            **({"model": model_name} if model_name else {}),
            **({"author": author} if author else {}),
        },
        summary=summary,
        failure_modes=failure_modes,
        examples=examples,
        extensions=_extensions_for_mode(mode),
        discussion=build_discussion(summary, failure_modes),
    )


def _extensions_for_mode(mode: str) -> list[str]:
    extensions = [
        "structured_evaluator",
        "reflection_memory",
        "benchmark_report_json",
    ]
    if mode == "mock":
        extensions.append("mock_mode_for_autograding")
    return extensions


def save_report(report: ReportPayload, out_dir: str | Path) -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "report.md"
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    s = report.summary
    react = s.get("react", {})
    reflexion = s.get("reflexion", {})
    delta = s.get("delta_reflexion_minus_react", {})
    ext_lines = "\n".join(f"- {item}" for item in report.extensions)
    author = report.meta.get("author", {})
    author_lines = ""
    if author:
        author_lines = (
            "\n## Student Information\n"
            f"- Name: {author.get('name', '')}\n"
            f"- Student ID: {author.get('student_id', '')}\n"
            f"- Email: {author.get('email', '')}\n"
        )
    model_line = (
        f"- Model: {report.meta['model']}\n"
        if report.meta.get("model")
        else ""
    )
    md = f"""# Lab 16 Benchmark Report

## Metadata
- Dataset: {report.meta['dataset']}
- Mode: {report.meta['mode']}
- Records: {report.meta['num_records']}
- Agents: {', '.join(report.meta['agents'])}
{model_line}{author_lines}

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | {react.get('em', 0)} | {reflexion.get('em', 0)} | {delta.get('em_abs', 0)} |
| Avg attempts | {react.get('avg_attempts', 0)} | {reflexion.get('avg_attempts', 0)} | {delta.get('attempts_abs', 0)} |
| Avg token estimate | {react.get('avg_token_estimate', 0)} | {reflexion.get('avg_token_estimate', 0)} | {delta.get('tokens_abs', 0)} |
| Avg latency (ms) | {react.get('avg_latency_ms', 0)} | {reflexion.get('avg_latency_ms', 0)} | {delta.get('latency_abs', 0)} |

## Failure modes
```json
{json.dumps(report.failure_modes, indent=2)}
```

## Extensions implemented
{ext_lines}

## Discussion
{report.discussion}
"""
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path
