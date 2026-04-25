# TRAIN KNOWLEDGE BASE

## OVERVIEW

Training orchestration, submission packaging, monitoring, and context handoff layer. This repo has no CI/Makefile; operations are Docker Compose + scripts.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Start/stop training | `.docker-compose.yaml`, `.env` | Use `--profile distributed`, project name `kaiwu-train` |
| Package/sign submission | `package_and_sign.sh`, `package_model.py`, `backup_model/` | Requires sidecar `kaiwu-train-backup_model-1` |
| Checkpoint management | `resume_best.py` | `list`, `best`, `latest`, `prepare`, `clean` |
| Local monitor | `local_monitor_dashboard.py`, `tb_writer.py` | ports 18080/18081; official panel at 11000 |
| Automation helper | `auto_monitor.sh` | semi-automatic monitor/restart; contains TODO plan switches |
| Competition memory | `context/` | plans, logs, handoffs, runtime caveats |

## COMMANDS

```bash
# first-time image load
docker load -i ../dev/images/kaiwu-images-13.0.1.tar.zst

# distributed training
cd train
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d

# stop
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed down

# package and sign
bash train/package_and_sign.sh code/model.ckpt-resume.pkl 9339

# local dashboard
python train/local_monitor_dashboard.py --port 18080
python train/tb_writer.py
tensorboard --logdir train/tb_logs --port 18081 --bind_all
```

## RUNTIME CONVENTIONS

- `.env` is the primary parameter source for AISRV/gamecore counts, ZMQ/reverb, batch size, dump frequency, and monitor ports.
- Compose dynamically patches official Kaiwu runtime/config files after startup; do not rely only on static `code/conf/configure_app.toml`.
- ZMQ mode depends on both env/config and runtime patching; see `context/ZMQ_RUNTIME_OPTIMIZATION_GUIDE.md`.
- Business logs are primarily in container `/data/projects/robot_vacuum/log/*.log`, locally synchronized under `train/log/`.

## SUBMISSION NOTES

- `package_and_sign.sh` calls `package_model.py`, copies zip/json into the sidecar, waits for signed zip to appear in `train/backup_model/`.
- If signing fails, first verify sidecar container is running: `kaiwu-train-backup_model-1`.
- Official competition page says latest submitted model is automatically run for ranking; do not assume source-only submission.

## ANTI-PATTERNS

- Do not change `server_req_base_url`; keep `http://127.0.0.1:${MONITOR_TRPC_PORT}`.
- Do not treat `auto_monitor.sh` as complete unattended optimization; its A/B/C/D code-change steps are TODO-like.
- Do not commit bulky runtime outputs from `log/`, `archive/`, `backup_model/`, `_package_tmp/`, or Docker images.
- Do not increase AISRV/gamecore blindly; current notes identify CPU-side bottlenecks.
