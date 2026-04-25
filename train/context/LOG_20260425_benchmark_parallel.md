# 2026-04-25 Benchmark Parallel Execution Fix

## Root cause

The real holdout benchmark had sharded support, but normal execution still behaved like a serial run unless the caller explicitly passed sharding flags. When sharding was enabled, worker assignment also depended on files written after Docker containers started, so an AISRV could enter benchmark mode before its assignment file existed.

## Changes

- `train/run_holdout_benchmark.py`
  - Real execution and dry-run now default to sharded mode unless `--serial` is passed.
  - The default two-shard plan uses `map_partition`, assigning map 4 episodes to shard 0 and map 7 episodes to shard 1.
  - Indexed assignment files (`shard_<index>.json`) are written before `docker compose up`; hostname assignment files are still written after container hostnames are resolvable for backward compatibility.
  - Sharded aggregation reports wall-clock-style elapsed time via `elapsed_seconds`/`wall_elapsed_seconds`, while preserving `sum_shard_elapsed_seconds` for diagnostics.
  - Invalid shard counts that exceed planned episode count now fail closed.

- `code/agent_ppo/eval/holdout_benchmark.py`
  - Sharded AISRV workers first use `KAIWU_AISRV_INDEX` to find indexed assignments when available.
  - Workers fall back to hostname-keyed assignments and wait up to `KAIWU_BENCHMARK_ASSIGNMENT_WAIT_SECONDS` (default 180s), closing the startup race.
  - Extra AISRV workers with an index greater than shard count skip cleanly.

- `train/.docker-compose.benchmark.yaml`
  - Benchmark overlay defaults to `KAIWU_BENCHMARK_SHARDED=1` and `KAIWU_BENCHMARK_SHARD_COUNT=2`.

## Multi-GC per AISRV extension

- `code/agent_ppo/workflow/train_workflow.py` now passes full `envs` and `agents` into the benchmark path.
- `code/agent_ppo/eval/holdout_benchmark.py` supports `KAIWU_BENCHMARK_WORKERS_PER_AISRV`.
  - It only enables in-AISRV threading when enough independent `(env, agent)` pairs exist.
  - Each worker owns one env and one agent, avoiding shared `preprocessor`, `planner`, `last_action`, and `last_reward` state.
  - If the framework provides fewer env/agent pairs than requested, it records a downgrade reason and runs with the safe effective worker count.
- `train/run_holdout_benchmark.py` now sets benchmark runtime sizing explicitly:
  - `KAIWU_AISRV_NUM = shard_count`
  - `KAIWU_PARALLEL_ENV_PER_AISRV = workers_per_aisrv`
  - `KAIWU_GAMECORE_NUM = shard_count * workers_per_aisrv`
  - Attempted 2 AISRV x 5 workers = 10 gamecores, but runtime exposed only one env/agent pair per AISRV, so in-AISRV work downgraded to serial.
  - Default is now 2 AISRV x 4 workers = 8 gamecores with 4 episodes per map for faster benchmark turnaround.

## Verification

- `python -m py_compile train/run_holdout_benchmark.py code/agent_ppo/eval/holdout_benchmark.py train/analyze_holdout_benchmark.py` passed.
- `python train/run_holdout_benchmark.py --maps 4,7 --episodes-per-map 2 --dry-run --output train/context/HOLDOUT_BENCHMARK_PARALLEL_DRYRUN.json` passed and emitted `strategy="map_partition"`.
- `python train/run_holdout_benchmark.py --maps 4,7 --episodes-per-map 1 --serial --dry-run --output train/context/HOLDOUT_BENCHMARK_SERIAL_DRYRUN.json` passed and emitted no sharding block.
- `python train/run_holdout_benchmark.py --maps 4,7 --episodes-per-map 1 --shard-count 3 --dry-run --output train/context/HOLDOUT_BENCHMARK_INVALID_SHARDS.json` failed closed with `--shard-count cannot exceed the planned episode count.`
- `python train/analyze_holdout_benchmark.py --input train/context/HOLDOUT_BENCHMARK_PARALLEL_DRYRUN.json` passed and preserved analyzer compatibility for the sharded dry-run contract.
- `python -m py_compile train/run_holdout_benchmark.py code/agent_ppo/eval/holdout_benchmark.py code/agent_ppo/workflow/train_workflow.py train/analyze_holdout_benchmark.py` passed after the multi-GC worker extension.
- `python train/run_holdout_benchmark.py --maps 4,7 --episodes-per-map 2 --dry-run --output train/context/HOLDOUT_BENCHMARK_MULTI_GC_DRYRUN.json` passed and recorded `planned_aisrv_num=2`, `workers_per_aisrv=4`, `planned_gamecore_num=8`.
- `python train/run_holdout_benchmark.py --maps 4,7 --episodes-per-map 1 --workers-per-aisrv 0 --dry-run --output train/context/HOLDOUT_BENCHMARK_INVALID_WORKERS.json` failed closed with `--workers-per-aisrv must be positive.`
- Real 2 AISRV x 5 workers attempt was interrupted after roughly 18 minutes; manifests showed both shards downgraded to `workers_per_aisrv_effective=1` because `env_count=1` and `agent_count=1` inside each AISRV. Partial logs reached about map4 episode 5 and map7 episode 6 before interruption.

## Notes

- Docker full benchmark was not started in this pass.
- No PPO policy, reward, planner, model, or checkpoint files were changed for this fix.
