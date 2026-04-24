# LTSPPO Benchmark Adaptation - 2026-04-25

## Scope

This note records the read-only reference pass for `origin/linux-LTSPPO-charge-constraint` and defines the local `win_YJY` benchmark/logging contract for later implementation. No checkout, merge, cherry-pick, algorithm copy, or PPO source change was performed in T0.

## Reference branch evidence

- Remote branch exists: `refs/heads/linux-LTSPPO-charge-constraint` at `553066cf05d76d5df26890a2cf66b49c73374bf4`.
- Local remote-tracking ref was missing initially and was fetched read-only to `refs/remotes/origin/linux-LTSPPO-charge-constraint`.
- Current branch after fetch remained `win_YJY`.
- Relevant reference files inspected without checkout:
  - `code/agent_ppo/eval/benchmark.py`
  - `code/agent_ppo/eval/benchmark_parallel.py`
  - `code/agent_ppo/utils/constraint_utils.py`
  - `code/agent_ppo/utils/reward_metrics.py`
  - `train/.docker-compose.benchmark.yaml`
  - `train/compare_benchmarks.py`

## Reusable ideas from LTSPPO

1. **Inference-only benchmark entrypoint**: LTSPPO keeps evaluation separate from the training loop and documents that it should not send data to learner, pollute curriculum, or change episode counters.
2. **Config injection per episode**: benchmark code derives base `usr_conf`, then overrides `map`, `map_random=false`, `robot_count`, `charger_count`, `max_step`, and `battery_max` before `env.reset()`.
3. **Checkpoint identity in runtime**: benchmark resolves a requested checkpoint, loads it explicitly, and injects `checkpoint_id` / checkpoint step into runtime metadata for traceability.
4. **Structured artifacts**: each session writes `manifest.json`, per-episode JSONL step logs under `episodes/`, `result.json`, appended aggregate `eval_results.json`, and `ai_summary.json`.
5. **AI-oriented schema**: reference logs include step-level state, decision, reward breakdown, planner guidance, route/slack diagnostics, anomaly flags, and episode-level evidence windows.
6. **Charge/battery diagnostics**: reference metrics include charger slack, recoverability, late return, late contract, return stall, missed charge opportunity, charger contention, and reward attribution over battery-failure windows.
7. **Parallel scheduling pattern**: `benchmark_parallel.py` uses a runtime directory with `pending/claimed/completed` task files, worker heartbeats, stale-claim recovery, and final aggregation. This is useful later, but T2 should start single-process/dry-run first unless runtime constraints require parallelism.
8. **Comparison report**: `train/compare_benchmarks.py` compares snapshots by win rate, clean score, battery fail rate, collision fail rate, and per-round metrics.

## Local fixed benchmark contract for win_YJY

The local benchmark must not inherit LTSPPO's multi-round defaults. For this plan, the contract is fixed:

- `max_step = 1000`
- `battery_max = 150`
- `robot_count = 4`
- `charger_count = 3`
- `maps = [4, 7]`
- `episodes_per_map = 10`
- Total quick benchmark size: `2 * 10 = 20` episodes
- `map_random = false` during each episode because the runner explicitly selects map 4 or map 7.
- Training maps remain `[1, 2, 3, 5, 6, 8, 9, 10]`; benchmark maps must not enter training sampler.

The local runner should expose these as an auditable contract and reject conflicting CLI/env overrides unless a later task explicitly adds a separate confirmation mode. The dry-run command in T2 should print this contract without touching model files.

## Local adaptation plan

1. Implement later runner under `train/run_holdout_benchmark.py`, not under active PPO algorithm modules, so benchmark infrastructure can be reviewed independently.
2. Use active source under `code/`; treat `extracted_code/`, `code_WK/`, archives, and the LTSPPO branch as references only.
3. Preserve Docker Compose operating style from `win_YJY`; if container execution is needed, pass benchmark controls through environment variables without changing `train_env_conf.toml` or monitor URL behavior.
4. Keep `train/context/` lightweight. Store raw JSONL/result artifacts under a runtime/eval directory such as `train/eval_logs/` or `.sisyphus/evidence/` summaries, not in `train/context/`.
5. Detect model mutation by recording checkpoint path/hash/mtime before and after benchmark. If any eval path can call training save logic, fail the benchmark.
6. Reuse local `code/agent_ppo/utils/archive_analysis.py` ideas for aggregate fields: avg score, p10/p90, per-map score, completed rate, battery/collision fail rates, clean_per_step, charge count, remaining charge, invalid move rate, and checkpoint id.
7. Extend local analysis to consume existing `win_YJY` death replay paths (`death_replay.*.jsonl`) from `train_workflow.py` when available, instead of copying LTSPPO reward/network changes.

## Minimum AI-analysis log schema

Every benchmark episode summary must include at least:

- `episode_id`
- `map_id`
- `score` / `clean_score`
- `done_reason` and `fail_reason`
- `step` / `finished_steps`
- `action` trace or compact action histogram plus final action window
- `planner_mode`
- `planner_target`
- `battery` and `battery_max`
- `charger_distance`
- `charger_slack`
- `reward_components`
- `death_replay_path`
- `checkpoint_id`

Recommended additional fields from LTSPPO/reference/local context:

- `completed_rate`, `battery_fail_rate`, `collision_fail_rate`, `clean_per_step`, `score_p10`, `score_p50`, `score_p90`
- `charge_count`, `remaining_charge`, `nearest_charger_dist`, `min_margin_any_charger`
- `late_return_rate`, `late_contract_rate`, `return_stall_rate`, `return_progress_per_step`, `recoverability_violation_rate`
- `action_entropy`, `planner_suggested_action`, `action_vs_planner_match`, `target_switch_rate`, `anchor_switch_rate`
- `reward_attribution` for battery-failure / return-stall windows
- `evidence_windows` around first late return, missed charge opportunity, and final death/failure steps

## Report schema for T2/T3

The benchmark result should contain:

- `contract`: fixed maps/config/episodes and source checkpoint metadata.
- `combined`: total 20-episode summary over maps `[4,7]`.
- `per_map`: separate map 4 and map 7 metrics.
- `episodes`: compact episode summaries with links to step JSONL and death replay.
- `risks`: flags for model mutation, holdout leakage, high-score battery deaths, low p10, map imbalance, and excessive fail rates.
- `decision_inputs`: fields needed by the closed loop: combined avg, per-map avg, p10, completed rate, battery fail rate, collision fail rate, clean_per_step, and checkpoint id.

## Risks and unresolved issues

- The LTSPPO benchmark references richer fields (`mode_prob`, `target_prob`, route anchors, reward components) that may not all exist in current `win_YJY` action/preprocessor objects. T2 should degrade gracefully and emit `null`/default fields rather than modifying PPO algorithm files.
- LTSPPO stores some benchmark artifacts under `/workspace/code/eval_logs` and `/workspace/code/eval_results.json`; local adaptation should avoid mutating model/checkpoint locations and prefer `train/eval_logs` or explicit output paths.
- Current worktree was already dirty before T0. Future T1 must record baseline branch/commit/diff/checkpoint hashes before any behavior change.
