# AGENT_PPO KNOWLEDGE BASE

## OVERVIEW

Active competition implementation for Kaiwu Robot Vacuum PPO. Current `win_YJY` changes are concentrated here.

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Agent entry | `agent.py` | `BaseAgent`, guided predict/exploit, model save/load, resume checkpoint |
| Hyperparameters | `conf/conf.py` | 84D feature layout, PPO params, residual planner alpha, resume intervals |
| Planner + PPO logic | `algorithm/algorithm.py` | PPO loss plus `CoveragePlanner` and current branch charging/NPC changes |
| Feature/reward | `feature/preprocessor.py` | 84D feature vector, charger/NPC merge, dynamic return margin, reward shaping |
| Model | `model/model.py` | MLP + residual block, actor logits + critic value |
| Training loop | `workflow/train_workflow.py` | episode runner, curriculum, GAE/sample yield, best/resume snapshots, death replay |
| Runtime utils | `utils/` | archive, analysis, ZMQ patch |

## CURRENT BRANCH FOCUS

- `algorithm/algorithm.py`: earlier return margin (`BASE_RETURN_MARGIN 22 -> 28`), merge local and `extra_info.frame_state` NPC/organ data, de-prioritize interior dirt before charger discovery, soften NPC risk during charge return.
- `feature/preprocessor.py`: dynamic return margin using charger scarcity, battery capacity, long horizon; merge global frame organs/NPCs; stronger low/critical battery penalties.
- `workflow/train_workflow.py`: richer `death_replay` JSONL snapshots, longer death trajectory buffer, sanitized action/model/planner fields.
- Model files changed too: `code/latest_model.pkl`, `code/model.ckpt-resume.pkl`, `code/model.ckpt-resume.meta.json`.

## COMPETITION LOGIC

- Active objective is robust cleaning under partial observation, battery limits, chargers, and NPC collision risk.
- Historical docs say v5.4 solved most collision deaths; battery deaths remain primary bottleneck.
- Current branch attempts to reduce battery deaths by earlier return pressure, better global charger/NPC visibility, and more forensic death logging.
- Feature layout in `conf.py`: `local_view(49) + global_state(27) + legal_action(8) = 84D`.

## CONVENTIONS

- Keep `Config.DIM_OF_OBSERVATION` aligned with `Preprocessor.feature_process()` output.
- Preserve Kaiwu method names: `predict`, `exploit`, `learn`, `save_model`, `load_model`.
- Keep AISRV inference CPU-friendly; learner is the GPU-heavy side.
- Prefer additive diagnostics over invasive framework rewrites.
- When changing reward/feature dimensions, update docs and resume compatibility notes.

## ANTI-PATTERNS

- Do not edit `extracted_code/agent_ppo` or `code_WK/code/agent_ppo` as active source.
- Do not switch to `agent_diy` for competition unless it is fully implemented first.
- Do not weaken legal-action masking; invalid moves cost steps and battery.
- Do not assume a visible charger list means safe return; check path distance, battery slack, NPC risk, and return mode.
- Do not leave multi-file PPO changes without a `train/context/CHANGELOG.md` entry.
