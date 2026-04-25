# TRAIN CONTEXT KNOWLEDGE BASE

## OVERVIEW

High-value competition memory: plans, root-cause analyses, server handoffs, and runtime caveats. Keep lightweight and commit-friendly.

## KEY DOCUMENTS

| Question | File |
|----------|------|
| What changed? | `CHANGELOG.md` — append-only timeline |
| Current optimization path | `OPT_PLAN_v53_20260414.md` |
| Battery death root cause | `LOG_20260414_v54_battery_death_diagnosis.md` |
| Full bottleneck map | `BOTTLENECK_ANALYSIS_FULL_20260415.md` |
| v5 reasoning | `V5_OPTIMIZATION_ANALYSIS.md` |
| Runtime/ZMQ caveats | `ZMQ_RUNTIME_OPTIMIZATION_GUIDE.md` |
| Monitor pitfall | `MONITOR_CONFIG_NOTE.md` |
| Baseline snapshot | `BASELINE_20260425_win_YJY.md` |
| Current handoff | `HANDOFF_20260425_WIN_YJY.md` |
| Server handoff | `SERVER_AI_PROMPT.md`, `SERVER_SYNC_AND_MONITOR.md` |
| Holdout benchmark logs | `LOG_20260425_holdout_benchmark.md`, `LOG_20260425_holdout_shard_aggregation.md` |
| Benchmark results | `HOLDOUT_BENCHMARK_*.json`, `HOLDOUT_ANALYSIS_*.md` |

## WRITING RULES

- Append `CHANGELOG.md`: `YYYY-MM-DD HH:MM | one-line summary`
- Create `LOG_YYYYMMDD_topic.md` only for multi-file changes, complex bugs, or architecture changes
- Small changes (single-line fix, param tweak) → CHANGELOG only
- Do not store logs, checkpoints, metric dumps, archives, or Docker images here
- Summarize findings and link external paths; keep entries useful for another AI/operator

## ANTI-PATTERNS

- Do not delete old context entries; append or add new dated docs
- Do not let context docs contradict active source under `code/` without noting branch/state
- Do not use this folder for raw experiment output; summarize and link paths instead
