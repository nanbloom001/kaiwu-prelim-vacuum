# Linux Server AI Prompt

Use this prompt when another AI is operating directly on the Linux training
server.

## Mission

Operate the Kaiwu training stack on Linux without relying on the official local
monitor page. Preserve compatibility with final Kaiwu platform submission.

## Working Rules

1. Do not default to de-Kaiwu refactors.
2. Prefer static inspection before starting or changing training.
3. Treat `license.dat` as an out-of-band local secret, not a Git artifact.
4. Use Git for code, config and context docs only. Model checkpoints
   (`code/*.pkl`) are **never** committed; transfer them with `scp`/`rsync`
   or fetch release assets.
5. Keep bulky runtime artifacts out of Git:
   - `train/log/`
   - `train/archive/`
   - `train/backup_model/`
   - `code/resume_snapshots/`
   - `code/manual_checkpoints/`

## First Checks On The Server

1. `git pull --ff-only`
2. verify docker and compose
3. verify GPU runtime
4. verify required Kaiwu images exist locally
5. verify `license.dat` is present at the path used by compose

## Start Training

From the repository root:

```bash
cd train
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d
```

## Custom Monitor

Do not depend on the official panel.

From the repository root:

```bash
python3 train/local_monitor_dashboard.py \
  --host 127.0.0.1 \
  --port 18080 \
  --prom-base http://127.0.0.1:4000/v1/prometheus
```

Then use SSH tunneling from the local machine:

```bash
ssh -L 18080:127.0.0.1:18080 user@SERVER -p PORT
```

Open:

```text
http://127.0.0.1:18080
```

## If Training State Must Be Handed Off

Transfer checkpoints out-of-band (`scp`/`rsync`, release assets, or a shared
storage mount). Never commit snapshot directories or model files.
