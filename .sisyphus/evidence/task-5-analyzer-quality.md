# Holdout Benchmark Analysis

- Run ID: `task-5-synthetic`
- Checkpoint: `synthetic.ckpt`
- Maps: `[4]`
- Episodes per map: `1`
- Combined status: `OK`

## Combined

- Episode count: `1`
- Avg clean score: `932.0`
- P10 / P50 / P90: `932.0` / `932.0` / `932.0`
- Completed / Battery fail / Collision fail: `0.0` / `1.0` / `0.0`

## Per Map

| Map | Status | Episodes | Avg Score | P10 | P90 | Battery Fail | Collision Fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | OK | 1 | 932.0 | 932.0 | 932.0 | 1.0 | 0.0 |

## Failure Classification

- Analyzed failure count: `1`

| Category | Count | Example |
| --- | ---: | --- |
| Charger unknown / first charger not found | 0 | - |
| Known charger but return too late | 0 | - |
| Optimistic route budget / negative slack | 0 | - |
| NPC or path blocked | 0 | - |
| Repeated invalid move / stuck pattern | 1 | map4_ep01 map=4 score=932.0 steps=988 replay=None |
| High-score battery death | 0 | - |
| Late or high-step battery death | 0 | - |
| Unknown | 0 | - |

## Next Step

- Status: `ACTIONABLE`
- Optimization level: `targeted`
- Recommendation: Prioritize invalid-move/stuck diagnostics because the dominant failures show repeated ineffective actions before termination.
- Evidence paths:
  - `C:\Users\lenovo\AppData\Local\Temp\tmp4fc5e2up\result.json`

## Risks

- None

## Missing Replay Warnings

- None

## Schema Quality

- Missing optional diagnostic fields: `decision_context`, `nearest_npc_dist`, `outcome_state`, `policy_info.nearest_npc_distance`
- Present optional diagnostic fields: `step_log`, `final_window`, `evidence_windows`, `field_availability`, `reward_attribution_lite`, `anomaly_summary_lite`
- Episode-level missing signals: `policy_info.nearest_npc_distance`

## AI Diagnostic Quality

- Reliability level: `medium`
- Adjacent ai_summary.json detected: `True`
- AI summary schema version: `1`
- Missing optional signals affecting AI diagnostics: `decision_context`, `nearest_npc_dist`, `outcome_state`, `policy_info.nearest_npc_distance`
- Reliability notes:
  - `reliable` charger_unknown / return_too_late: Episode summaries include charger knowledge and charge-return timing.
  - `reliable` optimistic_route_budget: Route-budget calls can use episode-level charger slack.
  - `reliable` repeated_invalid_move / loop_suspect: Stuck-pattern calls can use action streak and revisit summaries.
  - `limited` npc_or_path_blocked: Nearest-NPC distance or step telemetry is missing, so blockage calls should be treated as weaker than charger/battery signals.
  - `reliable` high_score_battery_death / late_battery_death: Battery-death calls can use clean score, step count, and remaining charge from episode summaries.
