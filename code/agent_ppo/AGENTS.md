# AGENT_PPO KNOWLEDGE BASE

## CODE MAP

| File | Role |
|------|------|
| `agent.py` | `BaseAgent`: predict/exploit/learn/save_model/load_model, guided predict, resume load |
| `conf/conf.py` | **84D feature layout**, PPO hyperparams (γ=0.99, λ=0.95, clip=0.2), residual planner alpha, BC regularization, snapshot intervals |
| `algorithm/algorithm.py` | PPO loss + `CoveragePlanner` + charging/NPC logic + coverage-target return buffer |
| `feature/preprocessor.py` | 84D feature vector (local_view 49 + global_state 27 + legal_action 8), reward shaping, dynamic return margin, charger/NPC merge from `extra_info.frame_state` |
| `feature/definition.py` | ObsData/ActData dataclasses, GAE computation. **Stale comment says 77D; actual is 84D** |
| `feature/expert.py` | `ExpertPolicy`: A* charger nav, NPC safety filter, battery hysteresis state machine (return_mode persists until battery≥95% on charger) |
| `model/model.py` | Actor-Critic MLP 256→128 with LayerNorm + orthogonal init. **Stale comment says 77D input; actual is 84D** |
| `workflow/train_workflow.py` | Episode runner, curriculum, GAE, best/resume snapshots, death replay JSONL |
| `eval/holdout_benchmark.py` | Inference-only holdout eval for maps [4,7]. Runs in aisrv container via `KAIWU_BENCHMARK_MODE=1` |
| `utils/zmq_patch.py` | Runtime ZMQ patches applied by compose entrypoint |
| `utils/experiment_archive.py` | Checkpoint analysis, fail reason inference |

## KEY CONFIG VALUES (conf.py)

- **Features**: `local_view(49) + global_state(27) + legal_action(8) = 84D`
- **PPO**: γ=0.99, λ=0.95, clip=0.2, LR=3e-4, vf_coef=0.5
- **Entropy**: β starts 0.004 → decays to 0.0018
- **Residual planner**: α warmup 0.10→0.18 over 240 episodes, max 0.45, charge cap 0.01, fallback cap 0.006
- **BC regularization**: decays from 1.10 → 0.28 as α rises
- **Snapshots**: episode every 50, time every 10min; keep 8 episode + 6 time + 5 best
- **Resume snapshot naming**: `best-ep{NNNNNN}-score{NNNNN}.pkl`, `resume-episode-ep{NNNNNN}.pkl`, `resume-time-{YYYYMMDD-HHMMSS}.pkl`

## REWARD SHAPING (preprocessor.py)

- **Cleaning/explore/approach/fresh_path rewards** — positive for advancing coverage
- **Charge rewards** — efficiency, event, arrival, unarrived-charger progress
- **Starvation gating** — `is_starving` state disables cleaning/explore/approach rewards during return mode
- **Penalties** — revisit penalty, step penalty, low/critical battery penalties
- **Return pressure** — dynamic margin based on charger scarcity, battery capacity, horizon

## PLANNER SAFETY (algorithm.py + expert.py)

- **NPC hard collision zone**: Chebyshev distance ≤ 2 (5×5 block) — always blocked
- **NPC path risk radius**: ≤ 4 — planner increases safety margin for paths near NPCs
- **Expert return state machine**: `return_mode` persists once triggered until battery ≥ 95% AND on charger
- **Current safety margins**: `BASE_RETURN_MARGIN=24`, `COVERAGE_RETURN_BUFFER=14`
- **Low battery force-return**: ratio < 32% triggers return mode unconditionally

## GUIDED PREDICT POST-PROCESSING (agent.py)

- Legal-action mask applied first; invalid actions zeroed
- NaN/negative fallback: uniform distribution over legal actions
- Re-normalization after masking
- Final correction: `prob[-1] = max(0, 1 - sum(prob[:-1]))` to avoid floating-point drift

## CONVENTIONS

- Keep `Config.DIM_OF_OBSERVATION` aligned with `Preprocessor.feature_process()` output
- Preserve Kaiwu method names: `predict`, `exploit`, `learn`, `save_model`, `load_model`
- Keep AISRV inference CPU-friendly; learner is the GPU-heavy side
- Prefer additive diagnostics over invasive framework rewrites
- When changing reward/feature dimensions, update both conf.py AND stale comments in definition.py/model.py

## ANTI-PATTERNS

- Do not edit `extracted_code/agent_ppo` or `code_WK/code/agent_ppo` as active source
- Do not switch to `agent_diy` for competition unless fully implemented
- Do not weaken legal-action masking; invalid moves cost steps and battery
- Do not assume a visible charger list means safe return; check path distance, battery slack, NPC risk, and return mode
- Do not leave multi-file PPO changes without a `train/context/CHANGELOG.md` entry
