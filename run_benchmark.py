from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich import print

from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.reporting import build_report, save_report
from src.reflexion_lab.runtime import build_runtime
from src.reflexion_lab.utils import load_dataset, save_jsonl

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dataset: str = "data/hotpot_mini.json",
    out_dir: str = "outputs/sample_run",
    reflexion_attempts: int = 3,
    mode: str = "mock",
    model: str = "",
    base_url: str = "",
    api_key: str = "",
    temperature: float = 0.0,
) -> None:
    load_dotenv()
    model = model or os.getenv("REFLEXION_MODEL", "")
    author = {
        "name": os.getenv("STUDENT_NAME", "").strip(),
        "student_id": os.getenv("STUDENT_ID", "").strip(),
        "email": os.getenv("STUDENT_EMAIL", "").strip(),
    }
    author = {key: value for key, value in author.items() if value}
    examples = load_dataset(dataset)
    react_runtime = build_runtime(
        mode=mode,
        model=model or None,
        base_url=base_url or None,
        api_key=api_key or None,
        temperature=temperature,
    )
    reflexion_runtime = build_runtime(
        mode=mode,
        model=model or None,
        base_url=base_url or None,
        api_key=api_key or None,
        temperature=temperature,
    )
    react = ReActAgent(runtime=react_runtime)
    reflexion = ReflexionAgent(
        max_attempts=reflexion_attempts,
        runtime=reflexion_runtime,
    )
    react_records = [react.run(example) for example in examples]
    reflexion_records = [reflexion.run(example) for example in examples]
    all_records = react_records + reflexion_records
    out_path = Path(out_dir)
    save_jsonl(out_path / "react_runs.jsonl", react_records)
    save_jsonl(out_path / "reflexion_runs.jsonl", reflexion_records)
    report = build_report(
        all_records,
        dataset_name=Path(dataset).name,
        mode=mode,
        author=author or None,
        model_name=model or None,
    )
    json_path, md_path = save_report(report, out_path)
    print(f"[green]Saved[/green] {json_path}")
    print(f"[green]Saved[/green] {md_path}")
    print(json.dumps(report.summary, indent=2))


if __name__ == "__main__":
    app()
