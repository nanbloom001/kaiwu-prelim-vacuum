# Baseline - win_YJY

Captured: 2026-04-25 (refreshed after verification drift)

## Repository state

- Branch: `win_YJY`
- HEAD: `a34e9aae96b40b6427aa76b34290167a88e091a8`
- Latest commit: `a34e9aa Adapt LTSPPO charge constraint benchmark design`
- Upstream: `origin/win_YJY`
- Branch status: `ahead 1`

## Dirty worktree at capture time

Pre-existing dirty tracked files:

- `code/agent_ppo/algorithm/algorithm.py`
- `code/agent_ppo/feature/preprocessor.py`
- `code/agent_ppo/workflow/train_workflow.py`
- `code/latest_model.pkl`
- `code/model.ckpt-resume.meta.json`
- `code/model.ckpt-resume.pkl`
- `train/context/CHANGELOG.md`

Pre-existing untracked paths observed before T1 docs were created:

- `.sisyphus/`
- `AGENTS.md`
- `code/agent_ppo/AGENTS.md`
- `train/AGENTS.md`
- `train/context/AGENTS.md`
- `train/context/HANDOFF_20260425_WIN_YJY.md`

Source diff summary from `git diff --stat`:

- `code/agent_ppo/algorithm/algorithm.py` | 61 lines
- `code/agent_ppo/feature/preprocessor.py` | 70 lines
- `code/agent_ppo/workflow/train_workflow.py` | 163 lines
- `code/latest_model.pkl` | binary change
- `code/model.ckpt-resume.meta.json` | 10 lines
- `code/model.ckpt-resume.pkl` | binary change
- `train/context/CHANGELOG.md` | 2 lines

## Fixed training / holdout contract

From `code/agent_ppo/conf/train_env_conf.toml` and handoff notes:

- Training maps: `[1, 2, 3, 5, 6, 8, 9, 10]`
- Holdout / benchmark maps: `[4, 7]`
- `battery_max = 150`
- `max_step = 1000`
- `charger_count = 3`
- `robot_count = 4`
- `map_random = true`

## Rollback anchor

This baseline is anchored to the current dirty state at HEAD `a34e9aa`.

- Do **not** `git reset` or clear the worktree blindly.
- Future benchmark / behavior commits should be reverted individually if rejected.
- Preserve model artifacts unless explicitly directed otherwise.

## Resume checkpoint metadata

`python train/resume_best.py latest` reports (current capture):

- path: `D:\TcKaiwuFinal\code\model.ckpt-resume.pkl`
- size: `0.4MB`
- trigger: `time`
- episode_cnt: `74`
- clean_score: `856.0`
- saved_at: `2026-04-25 04:55:14`
- pid: `322`
- snapshots: `25 files in D:\TcKaiwuFinal\code\resume_snapshots`

`code/model.ckpt-resume.meta.json` matches the resume tool output:

- `clean_score = 856.0`
- `episode_cnt = 74`
- `saved_at = 2026-04-25 04:55:14`
- `trigger = time`

## Model artifact inventory

| Path | Exists | Size (bytes) | SHA256 | MTime (UTC) |
|---|---:|---:|---|---|
| `code/latest_model.pkl` | yes | 367125 | `528521621D2B772266091C5C785A75CDBD83E6DA704B8359F6D65641F15422C7` | `2026-04-24T20:55:13.4304617Z` |
| `code/model.ckpt-resume.pkl` | yes | 367235 | `1A05766A3EAA2A472A94752899E7F00016FF033DA7ADF76C238FF602B2A6BDA5` | `2026-04-24T20:55:13.4520912Z` |
| `code/model.ckpt-resume.meta.json` | yes | 120 | `776EA45231BB731963151AAE9C2EB49FFEAFF7AE5D10BB6A459AA12C97510C0D` | `2026-04-24T20:55:13.4652236Z` |

## Baseline warning

The current dirty source and model files predate any benchmark or behavior change in T1. This baseline records them as-is so later work can detect unintended model/file mutation.

## Verification drift warning

Checkpoint/model artifacts changed during T1 verification. The current snapshot likely reflects an active/background training process or another external writer.

- T2/T3 must rely on hash/mtime mutation checks.
- Treat this document as the latest baseline snapshot as of the refreshed capture time above.
- Do not reset or overwrite model artifacts unless explicitly directed.
