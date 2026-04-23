# Lab 16 Benchmark Report

## Metadata
- Dataset: hotpot_100.json
- Mode: real
- Records: 200
- Agents: react, reflexion
- Model: gpt-4.1-nano-2025-04-14

## Student Information
- Name: Mai Tấn Thành
- Student ID: 2A202600127
- Email: 26ai.thanhmt@vinuni.edu.vn

## Summary
| Metric | ReAct | Reflexion | Delta |
|---|---:|---:|---:|
| EM | 0.52 | 0.53 | 0.01 |
| Avg attempts | 1 | 1.7 | 0.7 |
| Avg token estimate | 3112.44 | 6721.57 | 3609.13 |
| Avg latency (ms) | 2791.81 | 5574.36 | 2782.55 |

## Failure modes
```json
{
  "react": {
    "none": 52,
    "incomplete_multi_hop": 23,
    "entity_drift": 16,
    "wrong_final_answer": 9
  },
  "reflexion": {
    "none": 53,
    "incomplete_multi_hop": 25,
    "wrong_final_answer": 13,
    "entity_drift": 9
  },
  "overall": {
    "none": 105,
    "incomplete_multi_hop": 48,
    "entity_drift": 25,
    "wrong_final_answer": 22
  }
}
```

## Extensions implemented
- structured_evaluator
- reflection_memory
- benchmark_report_json

## Discussion
This benchmark compares a single-shot ReAct baseline against a multi-attempt Reflexion agent that stores short lessons between attempts. ReAct reached EM=0.52 while Reflexion reached EM=0.53, for an absolute gain of 0.01. The extra accuracy comes with a cost in attempts, tokens, and latency because failed runs trigger an evaluator call and a reflector step before retrying. Observed ReAct failure modes were {"none": 52, "incomplete_multi_hop": 23, "entity_drift": 16, "wrong_final_answer": 9}, while Reflexion failure modes were {"none": 53, "incomplete_multi_hop": 25, "wrong_final_answer": 13, "entity_drift": 9}. In practice, reflection memory is most useful when the first attempt stops after the first hop, confuses the final entity, or ignores evidence from the second supporting passage. Remaining errors usually indicate either weak evaluator feedback, low-quality context grounding, or a model that repeats the same answer even after receiving a correction strategy.
