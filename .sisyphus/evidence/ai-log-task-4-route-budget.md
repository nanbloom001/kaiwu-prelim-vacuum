# Holdout Benchmark Analysis

- Run ID: `task-4-route-budget`
- Checkpoint: `synthetic-route-budget.pkl`
- Maps: `[4]`
- Episodes per map: `1`
- Combined status: `OK`

## Combined

- Episode count: `1`
- Avg clean score: `812.0`
- P10 / P50 / P90: `812.0` / `812.0` / `812.0`
- Completed / Battery fail / Collision fail: `0.0` / `1.0` / `0.0`

## Per Map

| Map | Status | Episodes | Avg Score | P10 | P90 | Battery Fail | Collision Fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | OK | 1 | 812.0 | 812.0 | 812.0 | 1.0 | 0.0 |

## Failure Classification

- Analyzed failure count: `1`

| Category | Count | Example |
| --- | ---: | --- |
| Charger unknown / first charger not found | 0 | - |
| Known charger but return too late | 0 | - |
| Optimistic route budget / negative slack | 1 | map4_ep01 map=4 score=812.0 steps=742 replay=None |
| NPC or path blocked | 0 | - |
| Repeated invalid move / stuck pattern | 0 | - |
| High-score battery death | 0 | - |
| Late or high-step battery death | 0 | - |
| Unknown | 0 | - |

## Next Step

- Status: `ACTIONABLE`
- Optimization level: `targeted`
- Recommendation: Prioritize route-budget calibration because episode or replay evidence shows negative charger slack before death.
- Evidence paths:
  - `D:\TcKaiwuFinal\.sisyphus\evidence\task-4-route-budget-fixture.json`

## Risks

- None

## Missing Replay Warnings

- None

## Schema Quality

- Missing optional diagnostic fields: `anomaly_summary_lite`, `decision_context`, `field_availability`, `outcome_state`, `reward_attribution_lite`, `step_log`
- Present optional diagnostic fields: `final_window`, `evidence_windows`

## AI Diagnostic Quality

- Reliability level: `medium`
- Adjacent ai_summary.json detected: `False`
- Missing optional signals affecting AI diagnostics: `anomaly_summary_lite`, `decision_context`, `field_availability`, `outcome_state`, `reward_attribution_lite`, `step_log`
- Reliability notes:
  - `reliable` charger_unknown / return_too_late: Episode summaries include charger knowledge and charge-return timing.
  - `reliable` optimistic_route_budget: Route-budget calls can use episode-level charger slack.
  - `reliable` repeated_invalid_move / loop_suspect: Stuck-pattern calls can use action streak and revisit summaries.
  - `limited` npc_or_path_blocked: Nearest-NPC distance or step telemetry is missing, so blockage calls should be treated as weaker than charger/battery signals.
  - `reliable` high_score_battery_death / late_battery_death: Battery-death calls can use clean score, step count, and remaining charge from episode summaries.
