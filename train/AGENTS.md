# TRAIN KNOWLEDGE BASE

## OVERVIEW

Training orchestration, submission packaging, monitoring, and context handoff layer. No CI/Makefile — Docker Compose + scripts only.

## COMMANDS

```bash
# ── First-time image load ──
docker load -i kaiwu-images-13.0.1.tar.zst   # from train/ directory

# ── Distributed training ──
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed down
# Restart requires container removal to clear process_stop.done:
docker rm kaiwu-train-learner-1 kaiwu-train-aisrv-1

# ── Logs ──
docker logs -f kaiwu-train-learner-1
docker logs -f kaiwu-train-aisrv-1

# ── Holdout Benchmark (maps [4,7]) ──
docker compose -p kaiwu-benchmark \
  -f .docker-compose.yaml -f .docker-compose.benchmark.yaml \
  --profile distributed up -d
# Override: KAIWU_BENCHMARK_MAPS=4,7 KAIWU_BENCHMARK_EPISODES_PER_MAP=10

# ── Package & sign submission (from repo root) ──
bash train/package_and_sign.sh <pkl_path> <step>
# Requires sidecar: kaiwu-train-backup_model-1 must be running

# ── Monitoring ──
python train/local_monitor_dashboard.py --port 18080   # http://127.0.0.1:18080
python train/tb_writer.py && tensorboard --logdir train/tb_logs --port 18081 --bind_all
# Official: http://127.0.0.1:11000/p/v5/exp/monitor?domain_id=1&exp_id=1&task_uuid=1&task_id=0&platform=competition_stage

# ── Checkpoints ──
python train/resume_best.py list
python train/resume_best.py best
python train/resume_best.py latest
```

## KEY SCRIPTS

| Script | Purpose |
|--------|---------|
| `package_and_sign.sh` | One-shot: package → copy to sidecar → wait for signed zip |
| `package_model.py` | Manual packaging: zip + metadata + sidecar JSON |
| `resume_best.py` | Checkpoint inspect/prepare/list/clean |
| `run_holdout_benchmark.py` | Holdout runner with sharding support (maps [4,7]) |
| `analyze_holdout_benchmark.py` | Post-process holdout results into metrics/reports |
| `run_closed_loop_iteration.py` | Dry-run closed-loop train/benchmark/analyze/rollback |
| `local_monitor_dashboard.py` | Custom training dashboard (port 18080) |
| `tb_writer.py` | Training logs → TensorBoard events |
| `collect_data.py` | Training log GAMEOVER metric extraction |
| `auto_monitor.sh` | Semi-auto monitor/restart (uses legacy `docker-compose` command; unreliable) |

## RUNTIME CONVENTIONS

- `.env` is the primary parameter source for AISRV/gamecore counts, buffer type, batch size, dump frequency, and monitor ports
- Compose dynamically patches framework TOML configs at container startup via `post_init_patch.py`; static `code/conf/configure_app.toml` is overridden
- ZMQ mode requires both `.env` config (`KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE=zmq`) and runtime patching from `agent_ppo/utils/zmq_patch.py`
- Business logs: container `/data/projects/robot_vacuum/log/*.log`, synced to `train/log/`
- Docker image tar at `train/kaiwu-images-13.0.1.tar.zst` (not `dev/images/` — that dir is gitignored)

## SUBMISSION FLOW

1. `bash train/package_and_sign.sh <pkl_path> <step>`
2. Validates sidecar `kaiwu-train-backup_model-1` is running
3. Calls `package_model.py` → creates zip + json in `train/_package_tmp/`
4. Copies both into sidecar, polls for signed zip in `train/backup_model/`
5. Output filename: `<project>-ppo-<step>-<timestamp>-<version>.zip`

## ANTI-PATTERNS

- Do not change `server_req_base_url` from `http://127.0.0.1:${MONITOR_TRPC_PORT}`
- Do not treat `auto_monitor.sh` as reliable unattended automation (legacy `docker-compose` cmd, hardcoded Windows paths)
- Do not commit: `log/`, `archive/`, `backup_model/`, `_package_tmp/`, `_package_submit/`, `_package_test/`, `tb_logs/`, `*.tar.zst`, `*.zip`
- Do not increase AISRV/gamecore blindly; CPU-side bottlenecks documented
