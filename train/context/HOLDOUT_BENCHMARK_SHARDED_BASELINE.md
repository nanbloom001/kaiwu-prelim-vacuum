# Holdout Benchmark Analysis

- Run ID: `holdout-benchmark-20260425-163009`
- Checkpoint: `/workspace/code/model.ckpt-resume.pkl`
- Maps: `[4, 7]`
- Episodes per map: `10`
- Combined status: `OK`

## Combined

- Episode count: `20`
- Avg clean score: `652.4`
- P10 / P50 / P90: `209.5` / `787.0` / `941.4`
- Completed / Battery fail / Collision fail: `0.65` / `0.3` / `0.05`

## Per Map

| Map | Status | Episodes | Avg Score | P10 | P90 | Battery Fail | Collision Fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | OK | 10 | 605.8 | 214.0 | 872.6 | 0.4 | 0.0 |
| 7 | OK | 10 | 699.0 | 197.8 | 945.7 | 0.2 | 0.1 |

## Failure Classification

- Analyzed failure count: `7`

| Category | Count | Example |
| --- | ---: | --- |
| Charger unknown / first charger not found | 0 | - |
| Known charger but return too late | 0 | - |
| Optimistic route budget / negative slack | 0 | - |
| NPC or path blocked | 1 | map7_ep03 map=7 score=25.0 steps=28 replay=None |
| Repeated invalid move / stuck pattern | 6 | map4_ep02 map=4 score=454.0 steps=505 replay=None |
| High-score battery death | 0 | - |
| Late or high-step battery death | 0 | - |
| Unknown | 0 | - |

## Next Step

- Status: `ACTIONABLE`
- Optimization level: `targeted`
- Recommendation: Prioritize invalid-move/stuck diagnostics because the dominant failures show repeated ineffective actions before termination.
- Evidence paths:
  - `D:\TcKaiwuFinal\train\context\HOLDOUT_BENCHMARK_SHARDED_BASELINE.json`

## Risks

- None

## Missing Replay Warnings

- None

## Schema Quality

- Missing optional diagnostic fields: `decision_context`, `outcome_state`
- Present optional diagnostic fields: `step_log`, `final_window`, `evidence_windows`, `field_availability`, `reward_attribution_lite`, `anomaly_summary_lite`
- Episode-level missing signals: `preprocessor.current_mode`, `preprocessor.invalid_move_count`

## AI Diagnostic Quality

- Reliability level: `medium`
- Adjacent ai_summary.json detected: `False`
- Missing optional signals affecting AI diagnostics: `decision_context`, `outcome_state`
- Reliability notes:
  - `reliable` charger_unknown / return_too_late: Episode summaries include charger knowledge and charge-return timing.
  - `reliable` optimistic_route_budget: Route-budget calls can use episode-level charger slack.
  - `reliable` repeated_invalid_move / loop_suspect: Stuck-pattern calls can use action streak and revisit summaries.
  - `limited` npc_or_path_blocked: Nearest-NPC distance or step telemetry is missing, so blockage calls should be treated as weaker than charger/battery signals.
  - `reliable` high_score_battery_death / late_battery_death: Battery-death calls can use clean score, step count, and remaining charge from episode summaries.
