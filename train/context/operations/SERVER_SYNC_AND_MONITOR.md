# Linux Server Sync And Monitor

This project is intended to use Git for code and a small set of core model files,
while keeping bulky training artifacts out of the repository.

## What Git Should Sync

Git is suitable for:

- code changes under `code/`
- compose and env templates under `train/`
- monitoring and resume helper scripts
- key run context docs such as session notes
- a small set of core model handoff files:
  - `code/best_model.pkl`
  - `code/latest_model.pkl`
  - `code/model.ckpt-resume.pkl`
  - `code/model.ckpt-resume.meta.json`

These four files are intentionally not ignored. They allow a lightweight
"developer machine -> server" model handoff through Git when needed.

## What Git Should Not Sync

The following are intentionally ignored because they are large, fast-changing,
or purely local runtime artifacts:

- `train/log/`
- `train/archive/`
- `train/backup_model/`
- `train/TRAINING_DATA.json`
- exported image archives such as `train/*.tar.zst`
- `code/resume_snapshots/`
- `code/manual_checkpoints/`
- `license.dat`
- `dev/`

## Recommended Sync Strategy

Use a hybrid approach:

1. GitHub for code, config, docs, and the 4 core model handoff files.
2. `scp` or `rsync` for large one-off transfers:
   - docker image exports
   - `train/backup_model/`
   - `code/manual_checkpoints/`
   - `code/resume_snapshots/`
3. Keep `license.dat` off Git. Copy it to the server manually.

## Typical Local To Server Flow

From the Windows development machine:

1. Commit and push code changes.
2. If the server needs a fresh resume point, also commit:
   - `code/best_model.pkl`
   - `code/latest_model.pkl`
   - `code/model.ckpt-resume.pkl`
   - `code/model.ckpt-resume.meta.json`
3. On the server, run `git pull`.

For large artifacts, transfer them outside Git.

## Server Pull Flow

Example on Linux:

```bash
cd ~/kaiwuFinal
git pull --ff-only
```

If a resume file was synced through Git, the training workflow will be able to
pick up `code/model.ckpt-resume.pkl` from the shared code directory.

## Custom Monitor Instead Of Official Panel

The official Kaiwu local monitor page is not required and may fail even when
training is healthy. Use the custom dashboard script instead.

### Prerequisites

- `greptimedb` is running
- port `4000` is available inside the training host
- Python 3 is available on the host

### Start The Dashboard On The Server

Run from the repository root:

```bash
python3 train/local_monitor_dashboard.py \
  --host 127.0.0.1 \
  --port 18080 \
  --prom-base http://127.0.0.1:4000/v1/prometheus
```

If you need LAN access from inside a trusted network, bind to `0.0.0.0`
instead of `127.0.0.1`.

### Access From The Local Machine Through SSH Tunnel

From the local machine:

```bash
ssh -L 18080:127.0.0.1:18080 user@SERVER -p PORT
```

Then open:

```text
http://127.0.0.1:18080
```

### What The Dashboard Uses

The dashboard reads GreptimeDB's Prometheus-compatible API directly:

- `http://127.0.0.1:4000/v1/prometheus`

It does not rely on Tencent's official local frontend.

### Useful Metrics

The dashboard and underlying API expose metrics such as:

- `kaiwu_train_global_step`
- `kaiwu_episode_cnt`
- `kaiwu_clean_score`
- `kaiwu_charge_count`
- `kaiwu_remaining_charge`
- `kaiwu_finished_steps`
- `kaiwu_sample_production_and_consumption_ratio`

## Suggested Server AI Checklist

When another AI is operating on the server, it should follow this order:

1. `git pull --ff-only`
2. verify `license.dat` is present outside Git
3. verify docker images are available
4. start the training compose stack
5. verify `greptimedb` responds on port `4000`
6. launch `train/local_monitor_dashboard.py`
7. inspect learner and aisrv logs if the dashboard has no data

For a compact execution handoff, see:

- `train/context/operations/SERVER_AI_PROMPT.md`

## Notes On Large Checkpoints

Framework checkpoints under container paths such as `/data/user_ckpt_dir/...`
should not be synced through Git. If these need to be preserved or moved
between machines, use explicit file transfer or add a dedicated host mount.
