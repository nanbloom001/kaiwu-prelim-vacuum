# Task 1 Baseline Evidence

Captured from the current win_YJY state before any benchmark or behavior change, then refreshed after verification drift.

## Command evidence

### git status --short --branch

```text
## win_YJY...origin/win_YJY [ahead 1]
 M code/agent_ppo/algorithm/algorithm.py
 M code/agent_ppo/feature/preprocessor.py
 M code/agent_ppo/workflow/train_workflow.py
 M code/latest_model.pkl
 M code/model.ckpt-resume.meta.json
 M code/model.ckpt-resume.pkl
 M train/context/CHANGELOG.md
?? .sisyphus/
?? AGENTS.md
?? code/agent_ppo/AGENTS.md
?? train/AGENTS.md
?? train/context/AGENTS.md
?? train/context/HANDOFF_20260425_WIN_YJY.md
```

### git log -1 --oneline

```text
a34e9aa Adapt LTSPPO charge constraint benchmark design
```

### git rev-parse HEAD

```text
a34e9aae96b40b6427aa76b34290167a88e091a8
```

### git rev-parse --abbrev-ref @{upstream}

```text
origin/win_YJY
```

### git diff --stat

```text
 code/agent_ppo/algorithm/algorithm.py     |  61 ++++++++++-
 code/agent_ppo/feature/preprocessor.py    |  70 +++++++++++--
 code/agent_ppo/workflow/train_workflow.py | 163 ++++++++++++++++++++++++------
 code/latest_model.pkl                     | Bin 4185615 -> 367125 bytes
 code/model.ckpt-resume.meta.json          |  10 +-
 code/model.ckpt-resume.pkl                | Bin 4185755 -> 367235 bytes
 train/context/CHANGELOG.md                |   2 +
```

### git diff --name-only

```text
code/agent_ppo/algorithm/algorithm.py
code/agent_ppo/feature/preprocessor.py
code/agent_ppo/workflow/train_workflow.py
code/latest_model.pkl
code/model.ckpt-resume.meta.json
code/model.ckpt-resume.pkl
train/context/CHANGELOG.md
```

### python train/resume_best.py latest

```text
Latest local resume checkpoint:
  path   : D:\TcKaiwuFinal\code\model.ckpt-resume.pkl
  size   : 0.4MB
  trigger   : time
  episode_cnt: 74
  clean_score: 856.0
  saved_at  : 2026-04-25 04:55:14
  pid       : 322
  snapshots : 25 files in D:\TcKaiwuFinal\code\resume_snapshots
```

## Model hashes

```json
[
  {
    "Path": "code/latest_model.pkl",
    "Exists": true,
    "Size": 367125,
    "SHA256": "528521621D2B772266091C5C785A75CDBD83E6DA704B8359F6D65641F15422C7",
    "MTime": "2026-04-24T20:55:13.4304617Z"
  },
  {
    "Path": "code/model.ckpt-resume.pkl",
    "Exists": true,
    "Size": 367235,
    "SHA256": "1A05766A3EAA2A472A94752899E7F00016FF033DA7ADF76C238FF602B2A6BDA5",
    "MTime": "2026-04-24T20:55:13.4520912Z"
  },
  {
    "Path": "code/model.ckpt-resume.meta.json",
    "Exists": true,
    "Size": 120,
    "SHA256": "776EA45231BB731963151AAE9C2EB49FFEAFF7AE5D10BB6A459AA12C97510C0D",
    "MTime": "2026-04-24T20:55:13.4652236Z"
  }
]
```

## Drift note

The checkpoint/model files changed during verification. This is likely due to an active/background writer outside T1. The hashes above are the refreshed baseline for this snapshot; T2/T3 should compare against these values and treat any further drift as mutation.
