# Holdout Benchmark Analysis

- Run ID: `holdout-benchmark-20260425-051907`
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

## Risks

- `info` `MODEL_MUTATION_GUARD`: Tracked model artifacts remained unchanged during benchmark setup.
- `warning` `REAL_EXECUTION_DEFERRED`: T2 intentionally does not execute real holdout episodes. T3 should extend this runner with a safe runtime path.
- `warning` `NO_EPISODES`: Runner output contains no executed episodes yet; decision inputs are contract-only.
