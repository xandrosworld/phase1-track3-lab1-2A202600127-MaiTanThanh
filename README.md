# Lab 16 Reflexion Agent Benchmark

Student:
- Name: Mai Tan Thanh
- Student ID: 2A202600127
- Email: 26ai.thanhmt@vinuni.edu.vn

This repository contains a completed benchmark scaffold for comparing a single-shot ReAct baseline against a Reflexion agent on HotpotQA.

## What Was Implemented

- Real LLM runtime using an OpenAI-compatible chat completion API
- Reflexion loop with evaluator, reflector, retry, and reflection memory
- Benchmark reporting to `report.json` and `report.md`
- Real benchmark run on `data/hotpot_100.json`
- Final benchmark outputs stored in `outputs/final_run/`

## Run

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the benchmark:

```bash
python run_benchmark.py --mode real --dataset data/hotpot_100.json --out-dir outputs/final_run
```

Run autograding:

```bash
python autograde.py --report-path outputs/final_run/report.json
```

## Final Outputs

- `outputs/final_run/report.json`
- `outputs/final_run/report.md`
- `outputs/final_run/react_runs.jsonl`
- `outputs/final_run/reflexion_runs.jsonl`

## Notes

- The final real-model benchmark used `gpt-4.1-nano-2025-04-14`.
- `.env` is intentionally excluded from version control.
