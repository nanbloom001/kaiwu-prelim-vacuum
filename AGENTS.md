# PROJECT KNOWLEDGE BASE

**Branch:** `win_YJY` | **Framework:** Kaiwu DRL 13.0.1 | **Algorithm:** PPO | **Env:** Robot Vacuum (清扫大作战)

## OVERVIEW

Tencent Kaiwu competition project. Active code is `code/agent_ppo/`, training orchestration is `train/`, competition memory is `train/context/`. No CI/Makefile — Docker Compose + scripts only.

## STRUCTURE

```text
code/                        # → mounted at /workspace/code (authoritative source)
code/agent_ppo/              # PPO agent (active competition code)
code/agent_diy/              # Framework stub, not competition-ready
code/conf/                   # Framework TOML configs (overridden at runtime by compose)
code/session_best/           # Per-session best models + manifest.json
train/                       # Docker Compose, packaging, monitoring, logs
train/context/               # Plans, diagnoses, handoffs, changelog
train/.docker-compose.yaml   # Main compose file
train/.docker-compose.benchmark.yaml  # Benchmark overlay
tencentarena-docs/           # Local reference docs
extracted_code/              # Snapshot — do not edit
code_WK/                     # Historical copy — do not edit
dev/                         # Docker images (gitignored)
```

## COMMANDS

```bash
# ── Training (from train/ directory) ──
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d   # start
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed down    # stop
docker rm kaiwu-train-learner-1 kaiwu-train-aisrv-1  # required before restart (clears process_stop.done)
docker logs -f kaiwu-train-learner-1                  # learner logs
docker logs -f kaiwu-train-aisrv-1                    # aisrv logs

# ── First-time image load ──
docker load -i train/kaiwu-images-13.0.1.tar.zst

# ── Holdout Benchmark ──
docker compose -p kaiwu-benchmark \
  -f .docker-compose.yaml -f .docker-compose.benchmark.yaml \
  --profile distributed up -d
# Env overrides: KAIWU_BENCHMARK_MAPS=4,7 KAIWU_BENCHMARK_EPISODES_PER_MAP=10

# ── Package & sign submission (from repo root) ──
bash train/package_and_sign.sh <pkl_path> <step>
# Requires sidecar container kaiwu-train-backup_model-1 running
# Output: train/backup_model/<zip>.zip (signed, submittable)

# ── Monitoring ──
python train/local_monitor_dashboard.py --port 18080   # custom dashboard
python train/tb_writer.py && tensorboard --logdir train/tb_logs --port 18081 --bind_all
# Official panel: http://127.0.0.1:11000/p/v5/exp/monitor?domain_id=1&exp_id=1&task_uuid=1&task_id=0&platform=competition_stage

# ── Checkpoint management ──
python train/resume_best.py list
python train/resume_best.py best
python train/resume_best.py latest
```

## CODE MAP

| File | Role |
|------|------|
| `code/train_test.py` | Framework entry point, `algorithm_name = "ppo"` |
| `code/agent_ppo/agent.py` | `BaseAgent`: predict/exploit/learn/save_model/load_model, guided predict, resume load |
| `code/agent_ppo/conf/conf.py` | **Config**: 84D features, PPO hyperparams, residual planner alpha, snapshot intervals |
| `code/agent_ppo/algorithm/algorithm.py` | PPO loss + `CoveragePlanner` + charging/NPC logic |
| `code/agent_ppo/feature/preprocessor.py` | 84D feature vector, reward shaping, dynamic return margin |
| `code/agent_ppo/feature/definition.py` | ObsData/ActData dataclasses, GAE computation |
| `code/agent_ppo/feature/expert.py` | `ExpertPolicy`: A* charger nav, NPC safety filter, battery hysteresis state machine |
| `code/agent_ppo/model/model.py` | Actor-Critic MLP (256→128, LayerNorm, orthogonal init) |
| `code/agent_ppo/workflow/train_workflow.py` | Episode runner, curriculum, GAE, best/resume snapshots, death replay JSONL |
| `code/agent_ppo/eval/holdout_benchmark.py` | Inference-only holdout eval (maps [4,7]), runs inside aisrv container |
| `code/agent_ppo/utils/zmq_patch.py` | Runtime ZMQ patches applied by compose entrypoint |
| `train/package_and_sign.sh` | One-shot: package → copy to sidecar → wait for signed zip |
| `train/run_holdout_benchmark.py` | Benchmark runner with sharding support |

## KEY CONFIG (conf.py)

- **Features**: `local_view(49) + global_state(27) + legal_action(8) = 84D` — keep `Config.DIM_OF_OBSERVATION` aligned with `Preprocessor.feature_process()` output
- **Architecture**: MLP 256→128, actor 8D logits, critic 1D value
- **Residual planner**: warmup from α=0.10 → 0.18 over 240 episodes, max 0.45; charge cap 0.01; BC regularization decays from 1.10 → 0.28
- **Current safety margins**: `BASE_RETURN_MARGIN=24` (expert), `COVERAGE_RETURN_BUFFER=14` (planner)
- **Snapshots**: episode every 50, time every 10min, keep 8 episode + 6 time + 5 best
- **Stale comments**: `definition.py` and `model.py` say 77D but actual config is 84D

## DOCKER ARCHITECTURE

- **learner** (GPU): PPO training, profiles: `single`, `distributed`. Entrypoint patches ZMQ runtime files and TOML configs at startup.
- **aisrv** (CPU): Inference server, profile: `distributed`, scale: `${KAIWU_AISRV_NUM}` (default 2).
- **gamecore**: Game engine, scale: `${KAIWU_GAMECORE_NUM}` (default 8).
- **backup_model**: Sidecar signer (always runs). Required for submission packaging.
- **pushgateway → greptimedb → monitor-service → fe-monitor-service**: Metrics pipeline.
- **vector**: Log/metric collection → greptimedb.

## KEY ENV VARS (.env)

| Variable | Default | Purpose |
|----------|---------|---------|
| `KAIWU_ALGORITHM` | `ppo` | Algorithm selection |
| `KAIWU_TRAINING_MODE` | `distributed` | Training mode |
| `KAIWU_AISRV_NUM` | `2` | AISRV instance count |
| `KAIWU_GAMECORE_NUM` | `8` | Gamecore instance count |
| `KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE` | `zmq` | Buffer type (zmq or reverb) |
| `KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE` | `4096` | Training batch size |
| `KAIWU_EXPERIMENT_DUMP_MODEL_FREQ` | `5000` | Model dump frequency (steps) |
| `KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE` | `64` | Inference batch size |
| `KAIWU_MONITOR_PORT` | `11000` | FE monitor port |
| `MONITOR_TRPC_PORT` | `11001` | Monitor backend port |

## LOGGING CONVENTIONS

- After code edits or bug investigations, append `train/context/CHANGELOG.md`: `YYYY-MM-DD HH:MM | one-line summary`
- Create `train/context/LOG_YYYYMMDD_topic.md` only for multi-file changes, complex bugs, or architecture changes
- Small changes (single-line fix, param tweak) → CHANGELOG only

## COMPETITION STATE

- v5.4 solved collision deaths; **battery death is primary bottleneck**
- Current branch focus: earlier return margins, dynamic charger/NPC visibility via `extra_info.frame_state`, richer death replay JSONL
- Expert A* pathing is valid but non-emergency bias (3–8) was too weak; return triggers too late
- Resume checkpoint: `code/model.ckpt-resume.pkl` + `code/model.ckpt-resume.meta.json`

## ANTI-PATTERNS

- Do not change monitor `server_req_base_url` from `http://127.0.0.1:${MONITOR_TRPC_PORT}`
- Do not commit: `train/log/`, `train/archive/`, `train/backup_model/`, `code/resume_snapshots/`, `code/manual_checkpoints/`, `*.tmp` files
- Do not edit `extracted_code/`, `code_WK/`, or `train/archive/` as active source
- Do not assume GitHub Actions/Makefile exists; Docker Compose + scripts only
- Do not introduce multi-GPU/DDP; single GPU assumption
- Do not weaken legal-action masking; invalid moves cost steps and battery
- Do not assume visible charger list means safe return (check path distance, battery slack, NPC risk)
- Do not increase AISRV/gamecore count blindly; CPU-side bottlenecks noted in docs
- Preserve Kaiwu framework method names: `predict`, `exploit`, `learn`, `save_model`, `load_model`

## NOTES

- `rg` not available in this environment; use PowerShell `Get-ChildItem`/`Select-String` or built-in grep/glob tools
- Python LSP `basedpyright` configured but not installed; use AST-grep for code mapping
- Compose dynamically patches framework TOML configs at container startup; static `code/conf/configure_app.toml` is overridden
- ZMQ mode requires both `.env` config and runtime patching from `agent_ppo/utils/zmq_patch.py`
- Business logs: container `/data/projects/robot_vacuum/log/*.log`, synced to `train/log/`
- Docker image tar located at `train/kaiwu-images-13.0.1.tar.zst` (not `dev/images/`)
- `auto_monitor.sh` uses legacy `docker-compose` command and hardcoded Windows paths — not reliable for unattended use
