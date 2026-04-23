from src.reflexion_lab.agents import ReActAgent, ReflexionAgent
from src.reflexion_lab.prompts import build_evaluator_user_prompt
from src.reflexion_lab.reporting import build_report
from src.reflexion_lab.utils import load_dataset


def test_mock_agents_smoke() -> None:
    examples = load_dataset("data/hotpot_mini.json")
    react_records = [ReActAgent().run(example) for example in examples]
    reflexion_records = [ReflexionAgent(max_attempts=3).run(example) for example in examples]

    assert len(react_records) == len(examples)
    assert len(reflexion_records) == len(examples)
    assert any(record.reflections for record in reflexion_records)
    assert all(record.attempts == 1 for record in react_records)
    assert all(record.attempts >= 1 for record in reflexion_records)


def test_report_payload_contains_expected_extensions() -> None:
    examples = load_dataset("data/hotpot_mini.json")
    records = [ReActAgent().run(examples[0]), ReflexionAgent(max_attempts=3).run(examples[0])]
    report = build_report(records, dataset_name="hotpot_mini.json", mode="mock")

    assert "structured_evaluator" in report.extensions
    assert "reflection_memory" in report.extensions
    assert report.meta["mode"] == "mock"


def test_real_evaluator_prompt_does_not_leak_gold_answer() -> None:
    example = load_dataset("data/hotpot_mini.json")[0]
    prompt = build_evaluator_user_prompt(example, "Oxford University")

    assert "Gold answer:" not in prompt
    assert f"Question:\n{example.question}" in prompt
    assert "Candidate answer:\nOxford University" in prompt
