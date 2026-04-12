# Resume Checkpoint Loading Fix — 2026-04-12 17:00

## Problem

After the expert charging overhaul (commit 9c1083b), training started from random weights despite the resume checkpoint `model.ckpt-resume.pkl` being present. Evidence: entropy=2.0 (max for 8-action space), avg_score=340.

## Root Cause

Distributed training architecture has two processes:

- **aisrv** (actor): runs episodes, code loaded from `/workspace/code/`
- **learner**: runs PPO training, code installed to `/data/projects/robot_vacuum/`

Agent `__init__` used `os.path.dirname(__file__)` to locate resume checkpoint:
```python
_resume_path = os.path.join(os.path.dirname(__file__), "..", "model.ckpt-resume.pkl")
```

On the **aisrv**, `__file__` = `/workspace/code/agent_ppo/agent.py` -> resume found at `/workspace/code/model.ckpt-resume.pkl` (exists).

On the **learner**, `__file__` = `/data/projects/robot_vacuum/agent_ppo/agent.py` -> resume path `/data/projects/robot_vacuum/model.ckpt-resume.pkl` (**does not exist**).

### Failure Chain

1. aisrv loads resume checkpoint (correct weights) via both `__init__` and `workflow()` function
2. learner creates Agent with random weights (resume not found, `os.path.isfile` returns False)
3. learner trains from scratch, saves `model.ckpt-0.pkl` with random weights
4. framework model sync (every 1 min) pushes learner's random checkpoint to aisrv
5. aisrv loads the synced random checkpoint, overwriting the resume weights
6. Result: both sides use random weights

### Evidence from Logs

```
aisrv 16:27:12 - [RESUME] Loaded checkpoint from /workspace/code/model.ckpt-resume.pkl  (SUCCESS)
aisrv 16:28:24 - load model /data/ckpt/.../model.ckpt-0.pkl successfully  (OVERWRITTEN with random)
learner          - entropy=2.0 at step 0  (random weights confirmed)
```

## Fix

Modified `code/agent_ppo/agent.py` to check multiple paths:

```python
_resume_candidates = [
    os.path.join(os.path.dirname(__file__), "..", "model.ckpt-resume.pkl"),  # aisrv path
    "/workspace/code/model.ckpt-resume.pkl",  # learner fallback
]
for _resume_path in _resume_candidates:
    if os.path.isfile(_resume_path):
        try:
            state_dict = torch.load(_resume_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            break
        except Exception as e:
            ...
```

Also added `print(..., file=sys.stderr)` for debugging since `self.logger` is None during `__init__`.

## Verification

Restarted training with fresh Docker volumes. Logs confirmed:

```
learner 17:04:27 - [RESUME] Loaded checkpoint from /workspace/code/model.ckpt-resume.pkl  (SUCCESS!)
learner          - save model model.ckpt-0.pkl  (now has pretrained weights)
aisrv   17:04:37 - [RESUME] Loaded checkpoint from /workspace/code/model.ckpt-resume.pkl
aisrv   17:05:40 - entropy_loss: 0.98, avg_score: 838  (pretrained, not random!)
```

| Metric | Before (random) | After (resume) |
|--------|----------------|----------------|
| entropy | 2.0 | 0.98 |
| avg_score | 340 | 838 |
| reward | 580 | 1636 |
| charge_count | 9.5 | 3.75 |

## Files Changed

- `code/agent_ppo/agent.py` — multi-path resume loading with stderr logging

## Commit

410274b on branch win
