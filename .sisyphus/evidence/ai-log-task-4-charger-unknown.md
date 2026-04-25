# Holdout Benchmark Analysis

- Run ID: `task-4-charger-unknown`
- Checkpoint: `synthetic-charger-unknown.pkl`
- Maps: `[7]`
- Episodes per map: `1`
- Combined status: `OK`

## Combined

- Episode count: `1`
- Avg clean score: `388.0`
- P10 / P50 / P90: `388.0` / `388.0` / `388.0`
- Completed / Battery fail / Collision fail: `0.0` / `1.0` / `0.0`

## Per Map

| Map | Status | Episodes | Avg Score | P10 | P90 | Battery Fail | Collision Fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | OK | 1 | 388.0 | 388.0 | 388.0 | 1.0 | 0.0 |

## Failure Classification

- Analyzed failure count: `1`

| Category | Count | Example |
| --- | ---: | --- |
| Charger unknown / first charger not found | 1 | map7_ep01 map=7 score=388.0 steps=321 replay=None |
| Known charger but return too late | 0 | - |
| Optimistic route budget / negative slack | 0 | - |
| NPC or path blocked | 0 | - |
| Repeated invalid move / stuck pattern | 0 | - |
| High-score battery death | 0 | - |
| Late or high-step battery death | 0 | - |
| Unknown | 0 | - |

## Next Step

- Status: `ACTIONABLE`
- Optimization level: `targeted`
- Recommendation: Prioritize charger-discovery diagnostics on failed maps and verify the agent can enter charge-return mode before battery enters the terminal zone.
- Evidence paths:
  - `D:\TcKaiwuFinal\.sisyphus\evidence\task-4-charger-unknown-fixture.json`

## Risks

- None

## Missing Replay Warnings

- None

## Schema Quality

- Missing optional diagnostic fields: `anomaly_summary_lite`, `decision_context`, `field_availability`, `outcome_state`, `reward_attribution_lite`, `step_log`
- Present optional diagnostic fields: `final_window`, `evidence_windows`

## AI Diagnostic Quality

- Reliability level: `low`
- Adjacent ai_summary.json detected: `False`
- Missing optional signals affecting AI diagnostics: `anomaly_summary_lite`, `decision_context`, `field_availability`, `outcome_state`, `reward_attribution_lite`, `step_log`
- Reliability notes:
  - `reliable` charger_unknown / return_too_late: Episode summaries include charger knowledge and charge-return timing.
  - `unreliable` optimistic_route_budget: No charger-slack signal was captured, so route-budget conclusions would be speculative.
  - `reliable` repeated_invalid_move / loop_suspect: Stuck-pattern calls can use action streak and revisit summaries.
  - `limited` npc_or_path_blocked: Nearest-NPC distance or step telemetry is missing, so blockage calls should be treated as weaker than charger/battery signals.
  - `reliable` high_score_battery_death / late_battery_death: Battery-death calls can use clean score, step count, and remaining charge from episode summaries.
