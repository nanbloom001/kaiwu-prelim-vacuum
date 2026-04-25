# Holdout Benchmark Analysis

- Run ID: `holdout-benchmark-20260425-052257`
- Checkpoint: `D:\TcKaiwuFinal\code\model.ckpt-resume.pkl`
- Maps: `[4, 7]`
- Episodes per map: `10`
- Combined status: `NO_EPISODES`

## Combined

- Episode count: `0`
- Avg clean score: `0.0`
- P10 / P50 / P90: `0.0` / `0.0` / `0.0`
- Completed / Battery fail / Collision fail: `0.0` / `0.0` / `0.0`

## Per Map

| Map | Status | Episodes | Avg Score | P10 | P90 | Battery Fail | Collision Fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | NO_EPISODES | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 7 | NO_EPISODES | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Failure Classification

- Analyzed failure count: `0`

| Category | Count | Example |
| --- | ---: | --- |
| Charger unknown / first charger not found | 0 | - |
| Known charger but return too late | 0 | - |
| Optimistic route budget / negative slack | 0 | - |
| NPC or path blocked | 0 | - |
| Repeated invalid move / stuck pattern | 0 | - |
| High-score battery death | 0 | - |
| Late or high-step battery death | 0 | - |
| Unknown | 0 | - |

## Next Step

- Status: `NEED_MORE_DATA`
- Optimization level: `infrastructure`
- Recommendation: Implement or connect a safe inference-only runtime path, then rerun the fixed 2x10 holdout baseline so real episodes and replay-backed failures exist for diagnosis.
- Evidence paths:
  - `D:\TcKaiwuFinal\train\context\HOLDOUT_BENCHMARK_BASELINE_20260425_0522.json`
  - `D:\TcKaiwuFinal\train\holdout_detail_logs\holdout-benchmark-20260425-052257\schema.json`
  - `D:\TcKaiwuFinal\train\holdout_detail_logs\holdout-benchmark-20260425-052257`

## Risks

- `info` `MODEL_MUTATION_GUARD`: Tracked model artifacts remained unchanged during benchmark setup.
- `warning` `REAL_EXECUTION_DEFERRED`: T2 intentionally does not execute real holdout episodes. T3 should extend this runner with a safe runtime path.
- `warning` `NO_EPISODES`: Runner output contains no executed episodes yet; decision inputs are contract-only.

## Missing Replay Warnings

- None
