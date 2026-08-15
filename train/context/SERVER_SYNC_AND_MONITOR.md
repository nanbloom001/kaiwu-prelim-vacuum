# Linux Server Sync And Monitor

This project uses Git for code, configs and docs. Model checkpoints are **not**
stored in Git (they are large training artifacts); they are transferred between
machines with explicit file transfer (scp/rsync) or distributed through
GitHub Releases.

## What Git Should Sync

- code changes under `code/`
- compose and env templates under `train/`
- monitoring and resume helper scripts
- run context docs such as session notes

## What Git Should Not Sync

The following are intentionally ignored because they are large, fast-changing,
or purely local runtime artifacts:

- `code/*.pkl`, `code/*.meta.json` (model checkpoints)
- `train/log/`
- `train/archive/`
- `train/backup_model/`
- `train/TRAINING_DATA.json`
- exported image archives such as `train/*.tar.zst`
- `code/resume_snapshots/`
- `code/manual_checkpoints/`
- `license.dat` (out-of-band platform license, copied to servers manually)
- `.env`

## Recommended Sync Strategy

1. GitHub for code, config and docs.
2. `scp` / `rsync` for large one-off transfers:
   - docker image exports
   - `train/backup_model/`
   - `code/manual_checkpoints/`
   - `code/resume_snapshots/`
3. Model checkpoints for resume (`code/model.ckpt-resume.pkl` etc.) are copied
   to the server manually or published as release assets; never committed.

## Typical Local To Server Flow

1. Commit and push code changes.
2. Transfer any needed checkpoint to the server with `scp`/`rsync`.
3. On the server, run `git pull`.

## Server Pull Flow

Example on Linux:

```bash
cd ~/kaiwuFinal
git pull --ff-only
```

Place a resume checkpoint at `code/model.ckpt-resume.pkl` (outside Git) and the
training workflow will pick it up from the shared code directory.

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

- `train/context/SERVER_AI_PROMPT.md`

## Notes On Large Checkpoints

Framework checkpoints under container paths such as `/data/user_ckpt_dir/...`
should not be synced through Git. If these need to be preserved or moved
between machines, use explicit file transfer or add a dedicated host mount.
