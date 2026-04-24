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

## Risks

- `info` `MODEL_MUTATION_GUARD`: Tracked model artifacts remained unchanged during benchmark setup.
- `warning` `REAL_EXECUTION_DEFERRED`: T2 intentionally does not execute real holdout episodes. T3 should extend this runner with a safe runtime path.
- `warning` `NO_EPISODES`: Runner output contains no executed episodes yet; decision inputs are contract-only.

## T3 Runtime Failure Record

- Baseline established: **NO**. The fixed 2x10 holdout benchmark did not run any real episodes.
- Command: `python train/run_holdout_benchmark.py --maps 4,7 --episodes-per-map 10 --checkpoint code/model.ckpt-resume.pkl --output train/context/HOLDOUT_BENCHMARK_BASELINE_20260425_0522.json`
- Exit code: `3`
- Stdout: `{"status": "NOT_IMPLEMENTED", "output": "D:\\TcKaiwuFinal\\train\\context\\HOLDOUT_BENCHMARK_BASELINE_20260425_0522.json"}`
- Stderr: empty
- Runner failure reason: `REAL_EXECUTION_UNSUPPORTED_IN_T2`
- Runner failure message: Non-dry-run execution is intentionally unsupported in T2 to avoid hidden training/runtime side effects. Use `--dry-run` in T2; T3 will add real execution.
- Model mutation detected: `false`; drifted paths: `[]`.
- Concrete next step: add safe runtime execution to the runner or connect an inference-only Docker path, then rerun this exact maps `[4,7]`, 10 episodes per map baseline without `--dry-run`.

## Docker Status at Failure

```text
NAMES                              STATUS
kaiwu-train-aisrv-1                Up 3 hours
kaiwu-train-aisrv-2                Up 3 hours
kaiwu-train-learner-1              Up 3 hours
kaiwu-train-fe-monitor-service-1   Up 5 hours
kaiwu-train-monitor-service-1      Up 5 hours
kaiwu-train-vector-1               Up 5 hours
kaiwu-train-gamecore-3             Up 5 hours
kaiwu-train-gamecore-6             Up 5 hours
kaiwu-train-gamecore-4             Up 5 hours
kaiwu-train-gamecore-8             Up 5 hours
kaiwu-train-gamecore-5             Up 5 hours
kaiwu-train-gamecore-1             Up 5 hours
kaiwu-train-gamecore-2             Up 5 hours
kaiwu-train-gamecore-7             Up 5 hours
kaiwu-train-backup_model-1         Up 5 hours
kaiwu-train-greptimedb-1           Up 5 hours (healthy)
kaiwu-train-pushgateway-1          Up 5 hours
```
