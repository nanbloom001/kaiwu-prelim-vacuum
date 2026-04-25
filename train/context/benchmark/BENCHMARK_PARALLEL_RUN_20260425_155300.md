# Parallel Benchmark Run 20260425-155300

## Run

- Branch: `linux-yjy`
- Session: `20260425-155300`
- Command: `bash train/run_target_benchmark_900.sh --runner parallel --max-wait 1800`
- Profile: `target_3c4r_1000_150_40`
- Result path: `train/eval_parallel_logs/20260425-155300/result.json`
- Summary result: `WR=70%`, `Avg CS=386`, `28/40`

## Fixed Difficulty

- Maps: official maps `1-10`
- Rounds per map: `4`
- Total episodes: `40`
- `max_step=1000`
- `battery_max=150`
- `charger_count=3`
- `robot_count=4`

## Parallel Evidence

- AISRV containers observed during run: `4`
- Gamecore containers observed during run: `40`
- Container env:
  - `KAIWU_BENCHMARK_PARALLEL_MODE=1`
  - `KAIWU_BENCHMARK_WORKER_COUNT=4`
  - `KAIWU_BENCHMARK_ENVS_PER_WORKER=10`
  - `KAIWU_GAMECORE_NUM=40`
  - `KAIWU_PARALLEL_ENV_PER_AISRV=10`
- Container TOML:
  - `aisrv_connect_to_kaiwu_env_count = 10`
- Runtime workers: `40`
- Completed tasks: `40`
- Remaining claimed tasks after completion: `0`

## Timing

- Script wall time: `1041s`
- Benchmark internal `elapsed_seconds`: `978.4s`
- Progress reached `40/40` at script loop time: `905s`

## Notes

- The run confirmed that `4x10` parallel expansion is active.
- Runtime still showed a heavy tail: the final episode was `target_round_3/map4`, claimed by worker `28`, and it kept heartbeating while progressing slowly.
- The long wall time is therefore not caused by the `4x10` configuration failing to propagate; it is dominated by slow long-tail episode execution under the current benchmark/runtime stack.
