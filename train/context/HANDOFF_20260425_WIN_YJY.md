# Handoff - win_YJY training project

Updated: 2026-04-25
Branch: `win_YJY`
Main goal: produce a final model with average clean score >= 950 under the fixed competition training/evaluation condition.

## 1. Current State

### Repository and branch

- Worktree root: `D:\TcKaiwuFinal`
- Active branch at handoff time: `win_YJY`
- Remote branch exists: `origin/win_YJY`
- Important local dirty files:
  - `code/agent_ppo/algorithm/algorithm.py`
  - `code/agent_ppo/feature/preprocessor.py`
  - `code/agent_ppo/workflow/train_workflow.py`
  - `code/latest_model.pkl`
  - `code/model.ckpt-resume.pkl`
  - `code/model.ckpt-resume.meta.json`
- Do not blindly reset the worktree. The dirty code files contain recent algorithm/debug changes. The dirty model files are training artifacts.

### Fixed training condition required by user

The intended fixed condition is:

- `battery_max = 150`
- `max_step = 1000`
- `charger_count = 3`
- `robot_count = 4`
- `map_random = true`
- Training maps: `[1, 2, 3, 5, 6, 8, 9, 10]`
- Holdout / benchmark maps: `[4, 7]`

Current config file:

- `code/agent_ppo/conf/train_env_conf.toml`

Current config already uses:

```toml
[env_conf]
map = [1, 2, 3, 5, 6, 8, 9, 10]
map_random = true
robot_count = 4
charger_count = 3
max_step = 1000
battery_max = 150
```

### Training startup

This project is normally started with Docker Compose, not by directly running Python training scripts.

Useful commands:

```powershell
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

docker compose --profile distributed --env-file train/.env -f train/.docker-compose.yaml -p kaiwu-train up -d --force-recreate learner aisrv

docker exec kaiwu-train-learner-1 bash -lc "grep -E 'global step is|algorithm.learn|sample_production_and_consumption_ratio|data_fetch|real_train' /data/projects/robot_vacuum/log/learner.log | tail -80"

docker exec kaiwu-train-aisrv-1 bash -lc "grep -E 'GAMEOVER|DEATH_TRAJ|BEST|MAP_STATS|CONFIG_RISK' /workspace/log/aisrv/aisrv_kaiwu_rl_helper_pid*_log_2026-04-25-02.log | tail -160"
```

Monitor config pitfall:

- Keep `server_req_base_url = http://127.0.0.1:${MONITOR_TRPC_PORT}`.
- Do not change it to `0.0.0.0`; that caused the monitor frontend to report `Fail to fetch`.
- See `train/context/MONITOR_CONFIG_NOTE.md`.

## 2. Algorithm Currently in This Branch

### High-level architecture

The current branch is a hybrid of `win` infrastructure and `yjy` algorithm ideas:

- Rule-based global coverage planner in `code/agent_ppo/algorithm/algorithm.py`.
- PPO residual policy blended with planner probabilities.
- Feature/reward pipeline in `code/agent_ppo/feature/preprocessor.py`.
- Training workflow, fixed config sampler, resume snapshots, archive logging, monitor metrics, and death replay in `code/agent_ppo/workflow/train_workflow.py`.

### Important current algorithm behaviors

- Planner maintains a 128x128 global memory map from local 21x21 observations.
- It parses and merges charger/NPC data from both local `observation.frame_state` and global `extra_info.frame_state`.
- It uses A* path planning to coverage targets and chargers.
- It has `return_mode` for charging and holds that mode until battery is high enough on charger.
- Current `BASE_RETURN_MARGIN` has been raised from `22.0` to `28.0`.
- When charger is unknown, target reuse is disabled and frontier/edge expansion is prioritized to find the first charger.
- NPC avoidance has a hard mask plus soft penalty while charging, to avoid blocking the only return route.
- Preprocessor reward has dynamic return margin and stronger low/critical battery penalty.
- Death replay has been upgraded from short text-only summary to in-memory ring buffer plus JSONL logging on battery/collision failure.

### Recent verification already done

These checks passed after the latest death replay/debug edits:

```powershell
python -m py_compile code\agent_ppo\algorithm\algorithm.py code\agent_ppo\feature\preprocessor.py code\agent_ppo\workflow\train_workflow.py
```

Container-side compile also passed for:

```bash
python3 -m py_compile agent_ppo/algorithm/algorithm.py agent_ppo/feature/preprocessor.py agent_ppo/workflow/train_workflow.py
```

A temporary container script verified that nested `observation.extra_info.frame_state` is read correctly for global NPC/charger information.

## 3. Current Training Trend and Problem Diagnosis

### Latest observed trend

From local logs around 2026-04-25 02:03 to 02:57:

- Total parsed episodes: 49
- Overall win/completed rate: about 0.735
- Overall average clean score: about 754
- First 20 episodes: win rate 0.65, average score 690
- Episodes 21-40: win rate 0.80, average score 798
- Last 9 episodes: win rate 0.78, average score 800

This does not prove monotonic long-training degradation. However, battery failure remains the main structural problem.

### Failure pattern

Parsed failures:

- 13 failures total in the inspected window.
- 12 were battery failures.
- 1 was collision.
- Many battery failures happened after the robot had already reached 1-3 chargers.
- Therefore, the main problem is not only "cannot find the first charger"; it is also "leaves charger and expands too far / returns too late / route budget is too optimistic".

Examples from logs:

- Map 5, step 994, score 930, charger arrivals 3, battery fail.
- Map 9, multiple early deaths around step 150 with 0 arrivals.
- Map 3, battery fail with very negative charger slack.

### Important interpretation

Do not optimize only for short-term clean score. The current reward can make high-score battery deaths look acceptable to PPO, because a high clean score episode can still have high total reward even if it dies. The final target requires stable benchmark performance, not occasional high-score deaths.

Current primary objective:

1. Lower battery fail rate.
2. Eliminate high-score battery deaths, especially `score > 850` and `steps > 900`.
3. Preserve clean-per-step and average score.
4. Then push fixed-condition average score toward 950.

## 4. Immediate Direction for Next Agent

### First low-risk algorithm fixes to implement

These were discussed and reviewed with a sub-agent. They are recommended as the next code changes, but were not implemented at the time this handoff document was written.

1. Prevent failed episodes from saving best checkpoints.

   File: `code/agent_ppo/workflow/train_workflow.py`

   Current issue:

   - A battery/collision episode can still update `best_robust_score` and trigger `[BEST]` because the rolling score can improve despite the current failure.
   - This risks replacing the best model with a checkpoint associated with a failed episode.

   Required behavior:

   - Only `fail_reason == "completed"` can set `self.is_new_best = True`.
   - Failed episodes still train, log, and update monitor/archive statistics.
   - Failed episodes must not raise `self.best_robust_score`; otherwise they can block future valid completed episodes.

2. Add a small coverage return buffer.

   File: `code/agent_ppo/algorithm/algorithm.py`

   Current issue:

   - `_select_coverage_target()` gates candidates with:

     ```python
     battery <= dist + self._heuristic_charger_distance(pos) + reserve
     ```

   - `_heuristic_charger_distance()` is a Chebyshev lower bound, so it can underestimate true return cost on hard maps.

   Required behavior:

   - Add a constant near `BASE_RETURN_MARGIN`, e.g.:

     ```python
     COVERAGE_RETURN_BUFFER = 8.0
     ```

   - Use it only for coverage target gating when charger is known:

     ```python
     charger_need = self._heuristic_charger_distance(pos) + self.COVERAGE_RETURN_BUFFER
     ```

   - Do not change charge-mode A* behavior.
   - Do not globally raise `BASE_RETURN_MARGIN` yet.

### Changes not recommended as first step

Do not immediately do these unless the above two changes fail:

- Do not heavily truncate reward for high-score battery failures. It may teach the model to be too conservative and reduce CPS.
- Do not freeze or reduce PPO residual alpha based only on recent failure rate. Charge mode already caps alpha to a very low value.
- Do not keep increasing `BASE_RETURN_MARGIN` globally. Large negative slack failures are usually target-budget/path-estimation issues, not only a trigger threshold issue.

## 5. Benchmark Requirement: Hold Out Maps 4 and 7

The user explicitly wants two maps reserved for benchmark/generalization. The intended holdout maps are:

- Map 4
- Map 7

Current branch status:

- Training config excludes maps 4 and 7.
- A full benchmark/evaluation workflow for holdout maps is not yet migrated into `win_YJY`.

Need to migrate from `origin/linux`:

- `code/agent_ppo/utils/archive_analysis.py`
  - Ranks checkpoints using episode summaries and per-map/per-config metrics.
- `code/agent_ppo/utils/model_signer.py`
  - Builds signed model packages and patches eval config.
- Relevant evaluation/benchmark ideas from `origin/linux` workflow and tooling:
  - ability to run/evaluate a selected checkpoint,
  - report per-map average score,
  - separate training maps from benchmark/holdout maps,
  - keep benchmark metrics out of training sampler.

Recommended benchmark design for this branch:

1. Training sampler always uses maps `[1,2,3,5,6,8,9,10]`.
2. Add an explicit benchmark/eval script or workflow option that runs maps `[4,7]` only.
3. Benchmark must use the same fixed condition:
   - `battery_max=150`
   - `max_step=1000`
   - `charger_count=3`
   - `robot_count=4`
   - `map_random=true` or explicit repeated runs per map, but do not mix benchmark maps into training.
4. Benchmark output must include:
   - average clean score over holdout maps,
   - per-map average score,
   - completed rate,
   - battery fail rate,
   - collision fail rate,
   - average clean_per_step,
   - score p10/p50/p90,
   - number of episodes per map.
5. Use at least 20 episodes per holdout map for quick checks and 50+ episodes per map before declaring success.

Acceptance target requested by user:

- Final model average score >= 950 under fixed conditions.
- Interpret this as benchmark average over the fixed config, not cherry-picked training episodes.
- Stronger recommended acceptance:
  - training-map average >= 950,
  - holdout map 4/7 average >= 900 initially, then push toward >= 950,
  - battery fail rate <= 5%,
  - collision fail rate <= 1-2%,
  - no repeated `score > 850` battery failures.

## 6. Linux Branch Migration Notes

Remote branch:

- `origin/linux`

Observed useful files/features there:

- `code/agent_ppo/utils/model_signer.py`
- `code/agent_ppo/utils/archive_analysis.py`
- `code/agent_ppo/utils/container_routing.py`
- `code/agent_ppo/utils/zmq_patch.py`
- `train/run_datafetch_benchmark.py`
- `train/run_env_scaling_experiment.py`
- `train/run_replay_stability_experiments.py`
- `train/run_speed_experiments.py`
- `code/tests/test_container_routing.py`
- `code/tests/test_runtime_optimizations.py`
- `code/tests/test_zmq_patch.py`

Do not blindly merge `origin/linux` into `win_YJY`.

Reason:

- `origin/linux` has broad changes across model, features, workflow, Docker, signing, and many deleted/added files.
- `win_YJY` currently has a different yjy/planner hybrid algorithm and fixed-condition training requirement.
- Migrate individual utilities and concepts, not the whole branch.

Recommended migration order:

1. First migrate `archive_analysis.py` or create a local equivalent for checkpoint/benchmark ranking.
2. Then migrate or adapt model signing only if official upload/package flow is needed.
3. Only migrate benchmark runner logic after the evaluation entrypoint is clear.
4. Keep ZMQ runtime notes from `train/context/ZMQ_RUNTIME_OPTIMIZATION_GUIDE.md`; current branch already has ZMQ-related work, but benchmark migration should not disturb replay buffer unless necessary.

## 7. Performance and Runtime Notes

Current runtime observations:

- Learner is using GPU (`cuda:0`) after prior fixes.
- AISRV/gamecore remain CPU-heavy.
- `data_fetch` in learner logs is usually about 1-2 ms.
- `real_train` is usually around 18-30 ms.
- Current bottleneck is not learner data_fetch. It is more likely sample generation / environment CPU / policy logic.
- `sample_production_and_consumption_ratio` values around 2000-3000 were seen. Treat that metric carefully; do not assume it means GPU is bottlenecked.

ZMQ notes:

- See `train/context/ZMQ_RUNTIME_OPTIMIZATION_GUIDE.md`.
- Avoid moving mem_buffer/sample server logic to GPU. ZMQ + multiprocessing/fork can interact badly with CUDA.
- Single-GPU platform only; do not spend time on multi-GPU migration.

## 8. Logging and Diagnostics

Important logs:

```powershell
docker exec kaiwu-train-learner-1 bash -lc "tail -200 /data/projects/robot_vacuum/log/learner.log"

docker exec kaiwu-train-aisrv-1 bash -lc "grep -hE 'GAMEOVER|DEATH_TRAJ|RESIDUAL|BEST|MAP_STATS|CONFIG_RISK' /workspace/log/aisrv/aisrv_kaiwu_rl_helper_pid*_log_2026-04-25-02.log | tail -240"
```

Death replay:

- `train_workflow.py` now records a 30-step death trajectory ring buffer.
- On battery/collision failure it writes `death_replay.*.jsonl` through `ExperimentArchive.log_jsonl`.
- Short `[DEATH_TRAJ]` log stays in AISRV logs.
- Use death replay to distinguish:
  - first charger not found,
  - known charger but return triggered too late,
  - path blocked by NPC,
  - target budget too optimistic,
  - repeated invalid moves.

Metrics that matter now:

- `completed_rate`
- `battery_fail_rate`
- `collision_fail_rate`
- `avg_clean_per_step`
- `avg_cleaned_cells`
- `charger_arrived_count`
- `charger_first_arrival_step`
- count of `score > 850` battery failures
- count of `steps > 900` battery failures
- last/min `charger_slack` in death replay
- per-map averages, especially maps 1, 3, 6, 9 and holdout maps 4, 7

## 9. Recommended Work Plan for New Agent

### Phase A - Stabilize current branch

1. Do not reset dirty code.
2. Review dirty diff in:
   - `algorithm.py`
   - `preprocessor.py`
   - `train_workflow.py`
3. Implement the two low-risk fixes:
   - completed-only best checkpoint update,
   - `COVERAGE_RETURN_BUFFER = 8.0`.
4. Run `py_compile` locally and inside container.
5. Restart learner and AISRV.
6. Run 50-100 episodes.
7. Compare against current baseline:
   - battery fail should decrease,
   - high-score battery death should decrease,
   - average clean_per_step should not drop more than 1-2%.

### Phase B - Add benchmark/holdout evaluation

1. Migrate/adapt `origin/linux:code/agent_ppo/utils/archive_analysis.py`.
2. Add benchmark runner or workflow mode for maps `[4,7]`.
3. Ensure benchmark does not train or update model.
4. Emit a simple report with training maps and holdout maps separated.
5. Run at least 20 episodes per holdout map for quick smoke, 50+ per map for serious evaluation.

### Phase C - Push toward average score 950

Only after battery fail is controlled:

1. Tune coverage target scoring for CPS.
2. Revisit reward shaping only with small changes.
3. Consider per-map diagnosis for weak maps.
4. Use checkpoint ranking by robust score, not raw best single episode.
5. Never select final model from a failed episode.

## 10. Definition of Done

The project is ready for final model submission when:

- Fixed config is confirmed:
  - maps for training exclude 4 and 7,
  - holdout benchmark uses 4 and 7,
  - battery/max_step/charger/robot counts match user requirement.
- Final model average clean score is >= 950 under fixed condition.
- Completed rate is stable and battery fail is low.
- Benchmark report includes maps 4 and 7 separately.
- Selected checkpoint is from a completed episode or from a robust ranking procedure that excludes failed-episode checkpoint promotion.
- No monitor regression: frontend still works and no `Fail to fetch`.
- No packaging/signing regression if official platform upload is required.

