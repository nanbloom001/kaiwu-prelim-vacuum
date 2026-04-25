# Dual benchmark concurrency implementation - 2026-04-26

## Summary

Implemented the first pass of the 2x4 dual-concurrency benchmark plan:

- Outer concurrency: 2 AISRV containers with deterministic hostname/index assignments.
- Inner framework concurrency: 4 KaiwuEnv/gamecore processes per AISRV via `aisrv_connect_to_kaiwu_env_count`.
- Task concurrency: dynamic shared task queue under `/workspace/code/holdout_shards/dynamic`.

## Key behavior

- `KAIWU_BENCHMARK_PARALLEL_MODE=1` enters the dynamic holdout benchmark branch.
- The coordinator initializes one task per holdout episode.
- Logical workers claim pending task JSON files atomically and write completed task JSON files.
- The coordinator writes `/workspace/code/holdout_result.json` and `/workspace/code/.benchmark_done`.
- The outer runner validates that container `configure.toml` contains the requested
  `aisrv_connect_to_kaiwu_env_count`.
- After completion, the outer runner validates observed logical worker count against requested dynamic concurrency.

## Verification

- `python -m py_compile train/run_holdout_benchmark.py code/agent_ppo/eval/holdout_benchmark.py code/agent_ppo/workflow/train_workflow.py`
- `python train/run_holdout_benchmark.py --dry-run --output train/context/HOLDOUT_BENCHMARK_DYNAMIC_2X4_DRYRUN.json`

Dry-run recorded:

- `scheduler = dynamic`
- `planned_aisrv_num = 2`
- `envs_per_aisrv = 4`
- `planned_gamecore_num = 8`
- `planned_episode_count = 8`

## Real validation

Command:

```bash
python train/run_holdout_benchmark.py --output train/context/HOLDOUT_BENCHMARK_DYNAMIC_2X4_REAL.json
```

Result:

- runner status: `COMPLETED`
- progress reached done marker at about `190s`
- script wall time: about `218s`
- result `elapsed_seconds`: `140.5`
- `completed_task_count = 8`
- `observed_worker_count = 8`
- `logical_worker_count = 8`
- worker identities covered AISRV 1 process indexes 0-3 and AISRV 2 process indexes 0-3.

Each workflow process still sees `visible_env_handles = 1`, which is expected. The dual concurrency is exposed as
multiple framework helper processes per AISRV, not multiple env handles in one Python workflow invocation.
