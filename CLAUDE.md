# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Reinforcement learning training system for Tencent Arena's "Robot Vacuum Cleaning" competition. Uses PPO algorithm on the KaiwuDRL framework, migrated from the managed Windows/Kaiwu platform to run independently on Linux via Docker Compose.

## Running Training

All commands run from `train/` directory:

```bash
# Start full distributed training stack
cd train
docker compose -f .docker-compose.yaml --profile distributed up -d

# Force recreate (after config changes or patch edits)
docker compose -f .docker-compose.yaml --profile distributed up -d --force-recreate

# Stop everything
docker compose -f .docker-compose.yaml --profile distributed down

# View learner logs (most important for debugging)
docker logs -f kaiwu-train-learner-1

# Check container status
docker compose -f .docker-compose.yaml ps
```

Key environment variables are in `train/.env`. Training parameter overrides (replay type, batch size, etc.) are controlled via `KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE`, `KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE`, and related `KAIWU_EXPERIMENT_*` vars.

## Running Experiments

```bash
cd train
python3 run_replay_stability_experiments.py
```

Results go to `train/context/data/REPLAY_STABILITY_RESULTS.json`. The experiment script controls docker lifecycle (up/down) and collects training metrics from container logs.

## Running Tests

```bash
cd code
python3 -m pytest tests/ -v
```

## Benchmark Evaluation

Runs fixed scenarios against a model checkpoint for A/B comparison. Independent from training — no learner data, no curriculum interference.

```bash
cd train

# Run benchmark (stops training, runs eval, stops stack)
bash run_benchmark.sh                              # default checkpoint
bash run_benchmark.sh path/to/checkpoint.pkl       # specific checkpoint
RESTART=1 bash run_benchmark.sh                    # auto-restart training after

# View results
python3 compare_benchmarks.py latest               # latest benchmark
python3 compare_benchmarks.py 0 1                  # compare two runs
```

Results: `train/eval_results.json`. Detailed logs: `train/eval_logs/{session_id}/`.
Full documentation: `train/context/benchmark/BENCHMARK_SYSTEM.md`.

## Architecture

### Distributed Training Topology

- **learner** (1 instance, GPU 0): PPO training loop, writes checkpoints to shared volume
- **aisrv** (2 instances, GPU 1-2): AI server for environment interaction, sends experience data via Reverb
- **gamecore** (8 instances): Game environment simulation (robot vacuum arena)
- **backup_model**: Model checkpoint persistence sidecar
- **pushgateway → vector → greptimedb → monitor-service → fe-monitor-service**: Metrics pipeline

### Code Mounting

`code/` is mounted into containers at `/workspace/code`. The KaiwuDRL framework lives at `/data/projects/robot_vacuum/` inside the container image and is **not** in this repo. Container logs go to `/workspace/log` (mapped to `train/log/ on host).

### Hot-Patching System

The docker-compose `command` blocks contain inline Python that patches framework files at container startup. These patches fix Linux-specific issues:

1. **`clear_user_ckpt_dir()`** in `model_file_common.py` — prevents `shutil.rmtree` on mount points
2. **`_get_shared_queue()`** in `monitor_manager.py` — uses native `multiprocessing.Queue` instead of `Manager.Queue` on Linux (avoids socket disconnects)
3. **`trainer.py`** — skips `model_file_saver` fork when COS is disabled; adds debug milestone logs
4. **`off_policy_strategy.py`** — makes model sync process-internal instead of forking; fixes cleanup to guard against None
5. **TOML key replacement** — `replace_toml_key()` rewrites specific config keys in learner.toml and configure_app.toml from env vars (delete-old-then-write-new to avoid duplicates)

Patches are idempotent — they check for a marker string before applying. If the marker already exists, the patch is skipped.

### Agent Code (`code/agent_ppo/`)

- `agent.py` — Main PPO agent: model loading (with caching), prediction, batch-tensor learning, AMP support
- `algorithm/algorithm.py` — PPO loss computation (policy + value + entropy), gradient clipping
- `conf/conf.py` — Architecture (69D obs, 8 actions) and PPO hyperparameters (gamma=0.99, clip=0.2)
- `workflow/train_workflow.py` — Episode execution, checkpoint saving, resume mechanism, perf monitoring

### Configuration Layers

1. `train/.env` → Docker Compose env vars → container environment
2. Hot-patch `replace_toml_key()` rewrites TOML files from env vars at startup
3. `code/agent_ppo/conf/conf.py` — Python-level defaults for agent architecture and runtime tuning

## Important Notes

### Container Log Locations

Training logs are inside the container at `/data/projects/robot_vacuum/log/learner.log`, NOT at `/workspace/log/learner/`. The host `train/log/learner/` directory is typically empty. Always check `docker logs kaiwu-train-learner-1` or `docker exec kaiwu-train-learner-1 cat /data/projects/robot_vacuum/log/learner.log`.

### Current Focus

The priority is **replay/reverb data pipeline stability**, not algorithm changes. The main bottleneck is `data_fetch` variance and high `sample_production_and_consumption_ratio`. Business-layer `real_train` time is already optimized (~25ms). Do not modify PPO algorithm logic until the replay pipeline is characterized and stable.

### Context Files

`train/context/` holds session records, diagnosis reports, and experiment results. Key file: `train/context/diagnosis/DIAGNOSIS_REMEDIATION_REPORT_20260409.md` — contains the full migration diagnosis and remediation history. Keep this directory lightweight and commit-friendly.

### Experiment Script Caveats

`train/run_replay_stability_experiments.py` controls docker compose lifecycle. Its `collect_rows()` function must read from the correct container log path. If results show `row_count=0`, the log parsing path is wrong, not necessarily that training failed.
