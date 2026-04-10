# Training Session Log: 2026-04-08 ~ 2026-04-09

## Overview
Robot Vacuum PPO training optimization using Tencent Kaiwu framework.
Session started with battery_max=400, agent not charging, scores ~120-180.
Session ended with battery_max=300, agent learned charging, max score 929.

---

## Code Changes

### 1. New File: `train/resume_best.py`
Checkpoint management utility. Usage:
- `python train/resume_best.py list` — List all checkpoints sorted by train_step
- `python train/resume_best.py best` — Show best checkpoint info
- `python train/resume_best.py prepare` — Extract best .pkl to code/ for resume
  - Priority: code/best_model.pkl (YOLO auto-save) > latest zip in backup_model/
- `python train/resume_best.py clean [--keep 3]` — Remove old checkpoints

### 2. Modified: `code/agent_ppo/workflow/train_workflow.py`

**Added `import torch`** at top.

**Added Resume from checkpoint** (workflow function, lines 49-62):
- Loads `code/model.ckpt-resume.pkl` at startup into agent model
- Uses hardcoded `/workspace/code/` path (host-mounted volume, visible in containers)
- try/except around both load and delete (handles multi-worker race condition)
- Auto-deletes file after loading (one-time use)

**Added YOLO-style best model saving** (EpisodeRunner class):
- `self.score_window`: rolling window of last 30 episode clean_scores
- `self.best_avg_score`: tracks best rolling average seen
- `self.last_clean_score`: stores clean_score from _handle_episode_end for use in run_episodes
- `self.is_new_best`: flag set when rolling avg exceeds best
- `_save_best_model()`: saves state_dict to `/workspace/code/best_model.pkl` (host-accessible)
- 30-episode rolling window, triggers save when window >= 20 and avg > best
- Periodic checkpoint: `agent.save_model()` every 100 episodes

**Removed**: old 30-minute time-based periodic save (`last_save_model_time`)

**Changed terminal rewards** in `_handle_episode_end()`:
- WIN: `cleaning_ratio` weight 6.0 -> 10.0
- FAIL: `cleaning_ratio` weight 3.0 -> 5.0
- Battery fail penalty: -8.0 (was -4.0)
- Collision fail penalty: -4.0 (unchanged)

### 3. Modified: `code/agent_ppo/feature/preprocessor.py`

**Changed `reset()` defaults**: battery/battery_max 200 -> 300

**Reward function changes** (reward_process):

Current state is v4 (written but NOT activated - containers stopped before restart):

| Parameter | Original (v1) | v3 (last active) | v4 (not yet active) |
|-----------|---------------|-------------------|---------------------|
| low_battery_pressure threshold | 0.50 | 0.60 | 0.40 |
| charger_progress coeff | 0.08 | 0.15 | 0.06 |
| slack_improve coeff | 0.05 | 0.10 | 0.04 |
| charge_event_reward | 1.5 | 0.5 | 0.15 |
| low_battery_penalty coeff | 0.03 | 0.05 | 0.02 |
| low_battery_penalty threshold | 0.15 | 0.20 | 0.15 |

Unchanged rewards:
- cleaning_reward = 0.12 * cleaned_this_step
- explore_reward = 0.001 * new_explored_cells
- npc_penalty = -0.06 * clip((3-npc_dist)/3)
- revisit_penalty = -0.01 * clip(visit_count-1, 0, 3)
- stuck_penalty = -0.01 * clip(stuck_steps/4)
- idle_penalty = -0.012 * clip(no_progress/30)
- step_penalty = -0.0015
- clip(reward, -1.5, 1.5)

### 4. Modified: `code/agent_ppo/conf/train_env_conf.toml`
- battery_max: 150 -> 300
- Comment updated to reflect Phase2

### 5. Modified: `train/collect_data.py`
- plan identifier updated to track current training phase

---

## Training Results

### Key Metrics Timeline (Phase2, bat=300, v3 rewards)

| Time | Episodes | Avg | Max | R30 | Wins | Charge Rate |
|------|----------|-----|-----|-----|------|-------------|
| 15min | 88 | 163 | 365 | 217 | 0 | 3.5% |
| 28min | 203 | 182 | 405 | 233 | 0 | ~5% |
| 40min | 379 | 231 | 850 | 359 | 2 | 7.2% |
| 46min | 429 | 243 | 850 | 359 | 2 | ~10% |
| 57min | 534 | 277 | 915 | 497 | 8 | 21% |
| 67min | 635 | 311 | 929 | 285 | 33 | ~40% |
| 80min | 754 | 330 | 929 | 285 | 64 | 88% (exploit) |

**Best score: 929** (ep 951, 1000 steps, 929/6590 dirt cleaned)

### Charging Exploit Pattern
v3 rewards (charge_event=0.5, charger_coeff=0.15/0.10) caused:
- Phase 1: Agent learns to charge (good) - scores rise 100->500
- Phase 2: Agent charges obsessively (bad) - charge_count exploded to 268
- Scores dropped from R30=497 to R30=225 as agent spent time charging instead of cleaning

v4 rewards (charge_event=0.15, charger_coeff=0.06/0.04) were written to fix this but NOT activated.

---

## Key Architecture Knowledge

### Container File Paths (critical for code changes)
- Host `code/` mounts to `/workspace/code/` in containers (read-write)
- At container startup, code is COPIED from `/workspace/code/` to `/data/projects/robot_vacuum/`
- Python imports resolve to `/data/projects/robot_vacuum/` (not `/workspace/code/`)
- **Code changes only take effect on container restart**
- `best_model.pkl` and `model.ckpt-resume.pkl` should use `/workspace/code/` path

### Distributed Training Architecture
- 2 aisrv containers, each with 4 worker processes (8 total)
- 1 learner container (shared model updates)
- 8 gamecore containers (game environments)
- All workers send samples to same learner -> model quality depends on worst worker
- Resume checkpoint: only one worker loads it (file deleted after first load), others start fresh

### Docker Commands
- Start: `cd D:/TcKaiwuFinal/train && docker-compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d`
- Stop: `cd D:/TcKaiwuFinal/train && docker-compose -p kaiwu-train -f .docker-compose.yaml --profile distributed down`
- Do NOT add `-v` flag (would delete shared_model volume with checkpoints)

---

## Known Issues & Open Problems

1. **Charging reward balance**: The core unsolved challenge. Too low = no charging. Too high = charging exploit. The sweet spot hasn't been found yet. v4 parameters are a guess.

2. **Multi-worker resume inconsistency**: Only one worker loads the resume checkpoint. Other workers start from random weights. Mixed-quality data goes to learner, degrading the resumed model. Possible fix: use learner to distribute weights instead of file-based resume.

3. **Monitor panel unavailable**: Port 11000 monitor requires Tencent Galileo service mesh. Standalone mode cannot use it.

4. **Current code state**: v4 reward params written to preprocessor.py but never activated. Next session needs to decide whether to use v4, adjust, or try different approach.

---

## Preserved Assets
- `code/best_model.pkl` — Best model (avg=367, max=929), 4.0MB
- `train/backup_model/` — All checkpoint zips (19 total, ~80MB)
- `train/TRAINING_DATA.json` — Latest training data snapshot
- `train/resume_best.py` — Checkpoint management tool

## Untracked Files (not committed)
- code/best_model.pkl
- train/resume_best.py
- train/.docker-compose.yaml (and .bak)
- train/SESSION_LOG_20260409.md (this file)
- dev/, license.dat, tencentarena-docs/

---

## Appendix A: System Architecture (from prior AI sessions)

### Observation Design (1560-dim feature vector)
1. **local_map** (3 x 21 x 21 = 1323): obstacle/cleaned/dirt one-hot channels
2. **global_memory** (3 x 8 x 8 = 192): explored ratio / dirt ratio / visit heat (pooled from 128x128)
3. **scalar_state** (37 dim): step, battery, progress, dirt density, revisit/stuck/idle, charger dist/dx/dz/slack, NPC dist/dx/dz/risk, explored/dirty ratio, last action
4. **legal_action** (8 dim)

### Model Architecture
- local_map -> CNN
- global_memory -> small CNN
- scalar_state + legal_action -> MLP
- Fused shared backbone -> actor/critic dual-head output

### Archive System (from prior sessions)
Prior AI built a dual-track archive under `train/archive/<run_id>/`:
- `ai/`: JSON/JSONL for automated analysis (run_manifest, episode_summary, train_window, key_events, checkpoint_index)
- `human/`: raw logs, configs, checkpoints, report.md
- Key files: `code/agent_ppo/utils/experiment_archive.py`, `code/agent_ppo/utils/archive_agent.py`
- Archive writes to `/workspace/archive` inside containers
- Env vars: KAIWU_ARCHIVE_DIR, KAIWU_ARCHIVE_SYNC_SECONDS, KAIWU_ARCHIVE_IDLE_SECONDS

### Monitor Metrics (from prior sessions)
- `reward`: learner batch avg reward (rising = good)
- `total_loss / value_loss / policy_loss / entropy_loss`: PPO training signals
- `avg_episode_steps`: how long agent survives
- `avg_charge_count`: per-episode charging frequency
- `avg_cleaned_cells`: primary task metric
- `battery_fail_rate / collision_fail_rate / completed_rate`: failure analysis
- Monitor panel: http://127.0.0.1:11000 (requires Galileo service mesh in standalone mode)

---

## Appendix B: Prior Training History (from temp md files)

### Early Experiments (2026-04-08 ~ 2026-04-09 early morning)
These were before the current session's work:

| Plan | bat_max | Key Change | Result |
|------|---------|------------|--------|
| Original P0 | 400 | 69-dim obs, MLP only | avg~191, max 489, no charging |
| Plan A v1 | 400 | charge_event=2.0, charger_coeff 6x | avg 106-157, charge 6-10%, charging exploit began |
| Plan A v1+ | 400 | charge_event=3.0, more aggressive | avg 157, max 350, charge rate 6.7% |

Key finding from early sessions: charge_event=1.5 with bat=150 caused severe charging exploit (300+ charges/episode).

### Best Checkpoint from Early Runs
- Best checkpoint ID: 6300 (from bat=400 era)
- Location: inside backup_model zip archives
- Can be extracted via `python train/resume_best.py prepare`

---

## Appendix C: Historical Training Data (from MONITOR_RECORD.md)

### Plan A Monitoring Records (2026-04-09 00:40 - 01:32)

| Time | Episodes | Avg Score | Max Score | Charge Rate | Notes |
|------|----------|-----------|-----------|-------------|-------|
| 00:51 | 146 | 106.4 | 260 | 9.6% | Charging appeared |
| 00:57 | 186 | 119.0 | 301 | 8.6% | New max 301 |
| 01:02 | 219 | 125.1 | 301 | 8.2% | Stable |
| 01:06 | 258 | 133.0 | 332 | 7.8% | New max 332 |
| 01:11 | 298 | 139.7 | 332 | 7.0% | Approaching 150 |
| 01:17 | 338 | 145.2 | 332 | 6.8% | Almost 150 |
| 01:21 | 379 | 150.4 | 350 | 6.6% | Broke 150 & 350! |
| 01:27 | 415 | 154.3 | 350 | 6.7% | Convergence slowing |
| 01:32 | 452 | 157.1 | 350 | 6.4% | Plateauing |

---

## Appendix D: Early Stop Criteria

Trigger improvement if ANY met:
1. std < 10 and avg < 250 (converged without breakthrough)
2. No max score improvement in 15 consecutive episodes (plateau)
3. avg < 150 with no upward trend (poor performance)
4. Near-20ep growth < 5 (convergence slowdown)

Success criteria (any one met = stop optimizing):
- 20 consecutive episodes WIN rate > 10% and avg_score > 400
- Max clean_score > 1000
- Avg clean_score > 500

---

## Appendix E: 18080 Dashboard NA Incident (2026-04-09 20:18 ~ 20:27 CST)

### Symptom
- Custom dashboard at `http://127.0.0.1:18080/` showed `n/a` for almost all summary cards.
- Recent episode table still had data from AISRV logs, which meant training was running but the metrics query path was failing.

### Environment Constraints Confirmed
- Host has a Clash proxy, so local HTTP checks must bypass proxy settings.
- Host port `4000` is occupied by NoMachine; GreptimeDB is intentionally exposed on host port `14000`.
- The dashboard process was already configured correctly with:
  - `--prom-base http://127.0.0.1:14000/v1/prometheus`

### Root Cause
- `pushgateway` contained live `kaiwu_*` metrics, so the training side was still publishing metrics.
- `vector` was repeatedly waiting on `http://greptimedb:4000` and not ingesting metrics into GreptimeDB.
- The healthy GreptimeDB container existed as `train-greptimedb-1`, but it was only attached to Docker network `train_default`.
- The training stack containers (`vector`, `monitor-service`, learner, aisrv, gamecore) were attached to `kaiwu-train_default`.
- Result: inside the `kaiwu-train` network, hostname `greptimedb` was not reachable, so metrics never flowed from `pushgateway -> vector -> greptimedb`.

### Runtime Fix Applied
- Connected `train-greptimedb-1` to Docker network `kaiwu-train_default` with alias `greptimedb`.
- Restarted `kaiwu-train-vector-1`.

### Verification After Fix
- `vector` changed from `Restarting` to `Up`.
- GreptimeDB query endpoint on host `14000` began returning live values for metrics such as:
  - `kaiwu_episode_cnt`
  - `kaiwu_clean_score`
  - `kaiwu_charge_count`
  - `kaiwu_remaining_charge`
  - `kaiwu_finished_steps`
- Dashboard cards and charts for the above metrics recovered.

### Remaining n/a Fields After Recovery
- `Train Global Step`
- `Prod/Cons Ratio`
- `Learner Global Step`
- `AISRV Loaded CKPT`

These were not caused by the ingestion outage. At time of inspection:
- `pushgateway` did not expose `kaiwu_train_global_step`
- `pushgateway` did not expose `kaiwu_sample_production_and_consumption_ratio`
- AISRV logs reported `train_global_step: 0`
- learner log did not contain rolling `global step is ...` lines
- AISRV logs did not show checkpoint load success records, and `load_model_succ_cnt` remained `0.0`

### Operational Takeaway
- Host `14000` is the correct GreptimeDB port for browser or host-side queries.
- Container-side `greptimedb:4000` remains correct, because it is an internal container-network address and does not conflict with NoMachine.
- If 18080 shows widespread `n/a` again, first verify:
  1. `pushgateway` has `kaiwu_*` metrics
  2. `vector` is `Up`
  3. `train-greptimedb-1` is attached to `kaiwu-train_default`
  4. host query `http://127.0.0.1:14000/v1/prometheus/...` returns non-empty results

### Follow-up Full-Chain Check (2026-04-09 20:31 ~ 20:34 CST)

#### What is healthy
- All main containers were `Up`: learner, 2 aisrv, 8 gamecore, pushgateway, vector, monitor-service.
- Metrics ingestion recovered:
  - `sum(kaiwu_episode_cnt{})` returned `5314`
  - `avg(kaiwu_clean_score{})` returned `105.625`
  - `avg(kaiwu_reward{})` returned `83.656...`
- AISRV logs still showed fresh episode starts and gameovers around `20:33`, so environment interaction was continuing.
- GPU was in use:
  - host `nvidia-smi` showed GPU 0 at about `19% ~ 21%` utilization and ~`3407 MiB` memory used
  - learner container reported `torch.cuda.is_available() == True`
  - learner container saw `device_name NVIDIA A10`

#### What is not healthy
- This did **not** look like normal learner-driven training.
- learner checkpoint directory `/data/ckpt/robot_vacuum_ppo/` still only contained:
  - `model.ckpt-0.pkl`
  - `id_list`
- No later learner checkpoints were created after startup.
- learner training log still only contained startup lines plus the initial `model.ckpt-0.pkl` save.
- AISRV kept logging:
  - `policy.send_train_data failed, please check`
- AISRV also kept logging:
  - `model_file_sync current_available_model_files is empty`
- learner-side auxiliary processes were unhealthy:
  - `model_file_save` repeatedly threw `FileNotFoundError`
  - learner `monitor_proxy` repeatedly logged `Broken pipe`

#### Interpretation
- Current state is best described as:
  - rollout / environment interaction is running
  - dashboard metrics ingestion is running
  - GPU is being used
  - **but the learner / sample send / model sync chain is not functioning normally**
- Therefore this stack should **not** be treated as “normal training started successfully”.

### Standalone Linux Re-evaluation (2026-04-09 22:03 ~ 22:08 CST)

#### Context Shift
- This repository was migrated directly from Windows 11 and originally depended on Kaiwu/Open Platform runtime assumptions.
- Current target is a standalone Linux Docker training stack, so framework code paths that assume platform-managed directories or Windows-style process behavior must be re-evaluated.

#### Findings Confirmed
- Forcing learner/aisrv to `multiprocessing.spawn` is not a viable standalone fix:
  - learner failed with `_pickle.PicklingError: Can't pickle <class 'common_python.logging.kaiwu_logger.KaiwuLogger'>`
- `monitor_manager.py` using `multiprocessing.Manager().Queue()` on Linux is part of the instability:
  - startup injection was updated to switch Linux to `multiprocessing.Queue(CONFIG.queue_size)`
- learner still stopped immediately after the first framework checkpoint save:
  - latest learner log ended at `learner save model /data/ckpt/robot_vacuum_ppo/model.ckpt-0.pkl successfully`
  - debug milestones inserted after that line never appeared

#### Root Cause Narrowed
- Manual reproduction inside `kaiwu-train-learner-1` showed `clear_user_ckpt_dir()` fails deterministically:
  - `OSError: [Errno 16] Device or resource busy: '/data/user_ckpt_dir'`
- Root cause is the standalone mount layout:
  - compose maps `${KAIWU_TRAIN_LOG}/framework_ckpt` to `/data/user_ckpt_dir`
  - framework function `clear_user_ckpt_dir()` called `shutil.rmtree(CONFIG.user_ckpt_dir)`
  - on Linux this attempts to remove the mount root itself, which is invalid and blocks learner startup progression

#### Runtime Remediation Applied In Repo
- `train/.docker-compose.yaml` startup patch now rewrites `clear_user_ckpt_dir()` so it only removes children under `/data/user_ckpt_dir` instead of deleting the mount root.
- Both learner and aisrv startup commands now begin with `set -euo pipefail` so patch injection failures stop container startup instead of silently continuing with a broken runtime.

#### Expected Verification Targets After Restart
- learner log should continue past:
  - `train debug milestone after_monitor_state_reset`
  - `train debug milestone after_strategy_before_run`
  - `train process start success`
- `/data/ckpt/robot_vacuum_ppo/` should gain checkpoints beyond `model.ckpt-0.pkl`
- AISRV should stop repeating:
  - `model_file_sync current_available_model_files is empty`
  - `policy.send_train_data failed, please check`

### Final Verification After Standalone Fixes (2026-04-09 22:16 ~ 22:25 CST)

#### What Was Fixed
- `train/.docker-compose.yaml`
  - Linux startup injection now rewrites `clear_user_ckpt_dir()` so it clears mounted directory contents instead of deleting the mount root.
  - Linux monitor queue uses native `multiprocessing.Queue(CONFIG.queue_size)` instead of `Manager().Queue()`.
  - learner skips `model_file_saver` when `push_to_cos = 0`.
  - off-policy model sync runs in-process for standalone Linux and pushes checkpoints directly to modelpool.
  - learner/aisrv startup commands use `set -euo pipefail` so failed runtime patching no longer hides behind apparently healthy containers.
- `code/agent_ppo/conf/monitor_builder.py`
  - user monitor panel names `Avg Invalid Move Rate` and `Avg Charge Efficiency` were renamed to legal labels:
    - `平均无效移动率`
    - `平均充电效率`
  - this removed the last learner init `ERROR` from monitor config validation.

#### Verified Healthy
- learner now starts successfully and passes the previous hard stop:
  - `train debug milestone after_monitor_state_reset`
  - `train debug milestone after_strategy_before_run`
  - `train process start success`
- learner is training normally:
  - `global step` progressed from startup to `89`, `229`, `507`, then after final restart to `91+`
  - periodic learner loss logs continue printing
- replay buffer is being consumed normally:
  - `sample_completed` reached `206860`, then `1216031`, and after final restart quickly recovered to `441532`
- checkpoints are now generated continuously:
  - learner produced `model.ckpt-0.pkl`, `model.ckpt-100.pkl`, `model.ckpt-200.pkl`, `model.ckpt-300.pkl`, `model.ckpt-400.pkl`, `model.ckpt-500.pkl`
  - after final restart, checkpoint generation restarted cleanly from `0/100/200`
- learner -> modelpool -> AISRV model sync recovered:
  - learner logged `train first model file push to modelpool success`
  - AISRV pulled model archives from modelpool and logged repeated `load model ... success`
  - `load_model_succ_cnt` rose from `0.0` to `84.0`, `172.0`, `262.0`, `443.0`
- AISRV sample send recovered:
  - `sample_receive_cnt` rose from `0` to `32430`, `65410`, `97841`, `131515`, `197024`, `228299`
- GPU is in use:
  - host `nvidia-smi` showed GPU 0 (`NVIDIA A10`) at roughly `22% ~ 36%` utilization with about `4050 MiB` memory used

#### Error Status
- In the latest post-fix learner and AISRV logs, no matches remained for:
  - `policy.send_train_data failed`
  - `Broken pipe`
  - `FileNotFoundError`
  - log level `ERROR`
- `model_file_sync current_available_model_files is empty` still appears as an early startup `INFO` before the first model pull completes, but in inspected tails it becomes non-blocking and is followed by successful model loads.

#### Conclusion
- The training stack is now operating as a valid standalone Linux Docker training chain.
- The original severe blocker was not the dashboard or GPU path; it was a runtime mismatch between framework assumptions and standalone mounted filesystem/process behavior after migration away from Kaiwu/Open Platform.
