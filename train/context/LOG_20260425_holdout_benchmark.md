# LOG 2026-04-25 - Holdout Benchmark Contract

## Scope

Implemented only T2 benchmark infrastructure for the fixed local holdout contract. No PPO behavior files, training config, checkpoint payloads, Docker runtime, or training workflows were modified.

## Files changed

- `train/run_holdout_benchmark.py`
- `train/analyze_holdout_benchmark.py`
- `train/context/CHANGELOG.md`
- `.sisyphus/notepads/closed-loop-training-benchmark/learnings.md`
- `.sisyphus/notepads/closed-loop-training-benchmark/issues.md`

## Runner contract

- Hard-locks benchmark maps to `[4, 7]` and rejects any other map id.
- Reads `code/agent_ppo/conf/train_env_conf.toml` and verifies training maps remain `[1, 2, 3, 5, 6, 8, 9, 10]`.
- Hard-codes benchmark config for auditability:
  - `robot_count = 4`
  - `charger_count = 3`
  - `max_step = 1000`
  - `battery_max = 150`
  - `map_random = false`
- Captures pre/post snapshots for:
  - `code/latest_model.pkl`
  - `code/model.ckpt-resume.pkl`
  - `code/model.ckpt-resume.meta.json`
- Writes dry-run output JSON plus `schema.json` under `train/holdout_detail_logs/<run_id>/`.
- Emits a planned per-step detail log schema with required fields:
  - `episode_id`, `map_id`, `step`, `action`, `planner_mode`, `planner_target`, `battery`, `charger_distance`, `return_slack`, `reward_components`, `fail_reason`, `death_replay_path`, `checkpoint_id`
- T2 intentionally refuses real execution with a clear `REAL_EXECUTION_UNSUPPORTED_IN_T2` failure instead of pretending to run episodes.

## Analyzer contract

- Accepts runner JSON via `--input` and prints JSON summary to stdout.
- Optional `--output-md` writes a Markdown-friendly summary.
- Always emits:
  - `run_id`
  - `checkpoint`
  - `maps`
  - `episodes_per_map`
  - `fixed_config`
  - `combined`
  - `per_map`
  - `risks`
  - `decision_inputs`
- Handles dry-run / empty episode sets safely by returning `status="NO_EPISODES"` for combined and each configured map.

## Verification commands and results

1. `python -m py_compile "train/run_holdout_benchmark.py" "train/analyze_holdout_benchmark.py"`
   - Result: passed.
2. `python "train/run_holdout_benchmark.py" --maps 4,7 --episodes-per-map 10 --dry-run --output "train/context/HOLDOUT_BENCHMARK_dryrun.json"`
   - Result: passed with exit code 0.
   - Output confirms only maps `[4,7]`, fixed config, checkpoint snapshot/hash/mtime, mutation guard before/after snapshots, and planned detail-log schema.
3. `python "train/run_holdout_benchmark.py" --maps 1,4 --episodes-per-map 1 --dry-run`
   - Result: rejected with exit code 2.
   - Error clearly reports invalid holdout map `1` plus overlap with training maps `[1, 2, 3, 5, 6, 8, 9, 10]`.
4. `python "train/analyze_holdout_benchmark.py" --input "train/context/HOLDOUT_BENCHMARK_dryrun.json"`
   - Result: passed.
   - Output includes `combined` and per-map (`4`, `7`) metrics with `NO_EPISODES` status.

## Diagnostics note

- `lsp_diagnostics` was attempted on both modified Python files, but the configured Python LSP (`basedpyright-langserver`) is not installed in this environment.
- Fallback verification used `py_compile` plus runner/analyzer smoke tests.

## Deferred to T3

- Real benchmark episode execution/runtime integration remains intentionally unsupported in T2.
- T3 should extend the runner with a safe standalone inference path while preserving the fixed output/schema contract added here.
