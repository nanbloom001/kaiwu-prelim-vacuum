# TRAIN CONTEXT KNOWLEDGE BASE

## OVERVIEW

High-value competition memory: plans, root-cause analyses, server handoffs, and runtime caveats. Keep lightweight and commit-friendly.

## WHERE TO LOOK

| Question | File | Notes |
|----------|------|-------|
| What changed over time? | `CHANGELOG.md` | Primary timeline; append one-line summaries after code edits/bug investigations |
| Current optimization path | `OPT_PLAN_v53_20260414.md` | Diagnosis, credit assignment, config-aware training plan |
| Battery death root cause | `LOG_20260414_v54_battery_death_diagnosis.md` | Bias too weak despite valid A* path/action |
| Full bottleneck map | `BOTTLENECK_ANALYSIS_FULL_20260415.md` | v4-v5.4 seven-bottleneck overview |
| v5 reasoning | `V5_OPTIMIZATION_ANALYSIS.md` | Logit-bias gradient conflict, entropy collapse, checkpoint selection |
| Runtime/ZMQ caveats | `ZMQ_RUNTIME_OPTIMIZATION_GUIDE.md` | Current branch `win_YJY`, ZMQ, single-GPU, monitor warnings |
| Monitor pitfall | `MONITOR_CONFIG_NOTE.md` | Do not rewrite `server_req_base_url` |
| Server handoff | `SERVER_AI_PROMPT.md`, `SERVER_SYNC_AND_MONITOR.md` | Linux server operating rules |

## CURRENT COMPETITION SUMMARY

- Project target: maximize robust cleaning under partial observation, battery limits, chargers, and NPC collision risk.
- v5.2 had strong average score but low MinCS due catastrophic scenarios.
- v5.3 emphasized diagnostics and credit assignment: death trajectories, reward components, config failure rates.
- v5.4 reduced collision deaths sharply; battery death became dominant.
- Current `win_YJY` branch aligns with battery-death mitigation: earlier return margins, global `extra_info` merge for organs/NPCs, and richer death replay logs.

## WRITING RULES

- Append small changes to `CHANGELOG.md` as: `YYYY-MM-DD HH:MM | one-line summary`.
- Create `LOG_YYYYMMDD_topic.md` only for multi-file changes, complex bug investigations, or architecture/runtime changes.
- Do not store logs, checkpoints, metric dumps, archives, Docker images, or backup zips here.
- Keep entries useful to another AI/server operator who must continue without reconstructing history.

## ANTI-PATTERNS

- Do not delete old context entries; append or add new dated docs.
- Do not let context docs contradict active source under `code/` without explicitly noting branch/state.
- Do not use this folder for raw experiment output; summarize findings and link paths instead.
