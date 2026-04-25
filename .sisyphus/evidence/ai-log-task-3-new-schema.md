# Holdout Benchmark Analysis

- Run ID: `fixture-new-schema`
- Checkpoint: `code/model.ckpt-resume.pkl`
- Maps: `[1]`
- Episodes per map: `1`
- Combined status: `OK`

## Combined

- Episode count: `1`
- Avg clean score: `456.7`
- P10 / P50 / P90: `456.7` / `456.7` / `456.7`
- Completed / Battery fail / Collision fail: `0.0` / `0.0` / `0.0`

## Per Map

| Map | Status | Episodes | Avg Score | P10 | P90 | Battery Fail | Collision Fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | OK | 1 | 456.7 | 456.7 | 456.7 | 0.0 | 0.0 |

## Failure Classification

- Analyzed failure count: `1`

| Category | Count | Example |
| --- | ---: | --- |
| Charger unknown / first charger not found | 0 | - |
| Known charger but return too late | 0 | - |
| Optimistic route budget / negative slack | 0 | - |
| NPC or path blocked | 0 | - |
| Repeated invalid move / stuck pattern | 0 | - |
| High-score battery death | 0 | - |
| Late or high-step battery death | 0 | - |
| Unknown | 1 | ep-new-1 map=1 score=456.7 steps=123 replay=None |

## Next Step

- Status: `ACTIONABLE`
- Optimization level: `data`
- Recommendation: Collect additional replay-backed failures and richer death traces because the current evidence does not isolate a single dominant root cause.
- Evidence paths:
  - `D:\TcKaiwuFinal\.sisyphus\evidence\task-3-new-schema-fixture.json`

## Risks

- None

## Missing Replay Warnings

- None

## Schema Quality

- Missing optional diagnostic fields: `anomaly_summary_lite`, `final_window`, `reward_attribution_lite`, `step_log`
- Present optional diagnostic fields: `decision_context`, `outcome_state`, `evidence_windows`, `field_availability`

## AI Diagnostic Quality

- Reliability level: `low`
- Adjacent ai_summary.json detected: `False`
- Missing optional signals affecting AI diagnostics: `anomaly_summary_lite`, `final_window`, `reward_attribution_lite`, `step_log`
- Reliability notes:
  - `limited` charger_unknown / return_too_late: Charger knowledge or charge-return timing is partially missing, so early-vs-late return calls are weaker.
  - `unreliable` optimistic_route_budget: No charger-slack signal was captured, so route-budget conclusions would be speculative.
  - `limited` repeated_invalid_move / loop_suspect: Loop and invalid-move signals are too sparse for confident stuck-pattern classification.
  - `reliable` npc_or_path_blocked: NPC/path blocking calls can use step telemetry with nearest-NPC distance.
  - `reliable` high_score_battery_death / late_battery_death: Battery-death calls can use clean score, step count, and remaining charge from episode summaries.
