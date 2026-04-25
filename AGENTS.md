# PROJECT KNOWLEDGE BASE

**Generated:** 2026-04-25 03:45
**Commit:** eabf3d0
**Branch:** win_YJY

## OVERVIEW

Tencent Kaiwu Robot Vacuum / 清扫大作战 PPO competition project. Active work is Python RL code under `code/`, Docker Compose training/submission tooling under `train/`, and competition planning under `train/context/`.

## STRUCTURE

```text
TcKaiwuFinal/
├── code/                  # authoritative code mounted as /workspace/code
├── train/                 # Docker Compose, packaging/signing, monitoring, logs
├── train/context/         # high-value competition plans, diagnoses, handoffs
├── tencentarena-docs/     # local Tencent Arena / Kaiwu reference docs
├── extracted_code/        # extracted snapshot; do not treat as active source
├── code_WK/               # historical/work copy; do not treat as active source
└── dev/images/            # offline Kaiwu Docker images
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Current competition plan | `train/context/CHANGELOG.md`, `OPT_PLAN_v53_20260414.md`, `LOG_20260414_v54_battery_death_diagnosis.md` | Main decision trail: v5.2/v5.3/v5.4, battery deaths, entropy, reward credit |
| Active PPO code | `code/agent_ppo/` | Current branch modifies algorithm, preprocessor, workflow |
| Training launch | `train/.docker-compose.yaml`, `train/.env`, `README.md` | Use `--profile distributed`; no GitHub Actions/Makefile |
| Submission package | `train/package_and_sign.sh`, `train/package_model.py`, `train/backup_model/` | Requires sidecar container `kaiwu-train-backup_model-1` |
| Runtime constraints | `train/context/ZMQ_RUNTIME_OPTIMIZATION_GUIDE.md`, `MONITOR_CONFIG_NOTE.md` | ZMQ/runtime patch and monitor URL pitfalls |
| Environment model | `ENVIRONMENT_MODEL.md`, `tencentarena-docs/` | Robot vacuum rules, partial observation, battery/NPC/charger concepts |
| Checkpoint handoff | `code/latest_model.pkl`, `code/model.ckpt-resume.pkl`, `code/model.ckpt-resume.meta.json`, `code/session_best/` | Core model state; avoid touching snapshots blindly |

## CURRENT BRANCH STATE

- Branch: `win_YJY` tracking `origin/win_YJY`.
- Recent theme: merge YJY algorithm/runtime optimizations; resume checkpoints v5.4 around step ~4500.
- Uncommitted files observed during init:
  - `code/agent_ppo/algorithm/algorithm.py`
  - `code/agent_ppo/feature/preprocessor.py`
  - `code/agent_ppo/workflow/train_workflow.py`
  - `code/latest_model.pkl`
  - `code/model.ckpt-resume.pkl`
  - `code/model.ckpt-resume.meta.json`
- Current resume metadata: `clean_score=897.0`, `episode_cnt=37`, `saved_at=2026-04-25 03:25:26`, `trigger=best`.

## CODE MAP

| Area | Role |
|------|------|
| `code/train_test.py` | Minimal framework entry; default `algorithm_name = "ppo"` |
| `code/agent_ppo/agent.py` | Kaiwu `BaseAgent` implementation, model load/save, guided predict/exploit, resume load |
| `code/agent_ppo/conf/conf.py` | PPO hyperparameters, residual planner controls, resume snapshot strategy |
| `code/agent_ppo/algorithm/algorithm.py` | PPO loss plus current rule/planner-heavy coverage logic |
| `code/agent_ppo/feature/preprocessor.py` | 84D feature layout and reward shaping |
| `code/agent_ppo/workflow/train_workflow.py` | Episode runner, curriculum/config sampling, metrics, checkpoint/save logic |
| `code/agent_ppo/utils/` | Experiment archive, checkpoint reports, ZMQ runtime patch |
| `code/agent_diy/` | Framework placeholder; mostly stubbed, not competition-ready |

## COMPETITION DECISION TRAIL

- v5.2 best usable checkpoint was documented around `v52-step8500` with strong AvgCS but MinCS variance.
- v5.3 focus: diagnostics, death trajectory logging, config-risk tracking, reward credit repair.
- v5.4 finding: collision deaths largely reduced; battery death became main bottleneck.
- Battery death root cause in docs: Expert A* and actions are valid, but non-emergency bias `3-8` was too weak versus cleaning preference until emergency threshold.
- Current branch diff indicates a new direction: earlier/dynamic return margin, merging `extra_info.frame_state` NPC/organ data, richer death replay JSONL, and updated checkpoint models.

## COMMANDS

```bash
# start distributed training from repo root
cd train
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d

# stop distributed training
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed down

# monitor logs
docker logs -f kaiwu-train-learner-1
docker logs -f kaiwu-train-aisrv-1

# package and sidecar-sign a model from repo root
cd ..
bash train/package_and_sign.sh code/model.ckpt-resume.pkl 9339

# inspect checkpoints from repo root
python train/resume_best.py list
python train/resume_best.py latest
```

## PROJECT-SPECIFIC CONVENTIONS

- Treat `code/` as active source. `extracted_code/`, `code_WK/`, and `train/archive/` are references/snapshots unless explicitly instructed.
- Keep `train/context/` lightweight and commit-friendly: plans, handoffs, diagnoses, not logs/checkpoints/images.
- After code edits or bug investigation, append `train/context/CHANGELOG.md`; create detailed `LOG_YYYYMMDD_topic.md` only for multi-file or architectural work.
- Preserve Kaiwu framework interfaces and decorators; avoid de-Kaiwu refactors.
- Use `ppo` as active algorithm. `diy` is not implemented enough for competition use.

## ANTI-PATTERNS (THIS PROJECT)

- Do not change monitor `server_req_base_url` away from `http://127.0.0.1:${MONITOR_TRPC_PORT}`.
- Do not commit bulky runtime artifacts: `train/log/`, `train/archive/`, `train/backup_model/`, `code/resume_snapshots/`, `code/manual_checkpoints/`.
- Do not edit archive/snapshot copies as if they are active code.
- Do not assume GitHub Actions/Makefile automation exists; this repo is Docker Compose + scripts driven.
- Do not introduce multi-GPU/DDP logic; current platform assumption is single GPU.

## NOTES

- `rg` is not available in this environment; PowerShell `Get-ChildItem`/`Select-String` were used for direct searches.
- Python LSP server `basedpyright` is configured but not installed; AST-grep was used for code mapping.
