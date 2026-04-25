# Holdout Benchmark Analysis

- Run ID: `holdout-benchmark-20260425-094137`
- Checkpoint: `D:\TcKaiwuFinal\code\model.ckpt-resume.pkl`
- Maps: `[4, 7]`
- Episodes per map: `1`
- Combined status: `NO_EPISODES`

## Combined

- Episode count: `0`
- Avg clean score: `0.0`
- P10 / P50 / P90: `0.0` / `0.0` / `0.0`
- Completed / Battery fail / Collision fail: `0.0` / `0.0` / `0.0`

## Per Map

| Map | Status | Episodes | Avg Score | P10 | P90 | Battery Fail | Collision Fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | NO_EPISODES | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 7 | NO_EPISODES | 0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Failure Classification

- Analyzed failure count: `0`

| Category | Count | Example |
| --- | ---: | --- |
| Charger unknown / first charger not found | 0 | - |
| Known charger but return too late | 0 | - |
| Optimistic route budget / negative slack | 0 | - |
| NPC or path blocked | 0 | - |
| Repeated invalid move / stuck pattern | 0 | - |
| High-score battery death | 0 | - |
| Late or high-step battery death | 0 | - |
| Unknown | 0 | - |

## Next Step

- Status: `NEED_MORE_DATA`
- Optimization level: `infrastructure`
- Recommendation: Implement or connect a safe inference-only runtime path, then rerun the fixed 2x10 holdout baseline so real episodes and replay-backed failures exist for diagnosis.
- Evidence paths:
  - `D:\TcKaiwuFinal\train\context\HOLDOUT_BENCHMARK_20260425_0941.json`
  - `D:\TcKaiwuFinal\train\holdout_detail_logs\holdout-benchmark-20260425-094137\schema.json`
  - `D:\TcKaiwuFinal\train\holdout_detail_logs\holdout-benchmark-20260425-094137`

## Risks

- `info` `MODEL_MUTATION_GUARD`: Tracked model artifacts remained unchanged during benchmark setup.
- `warning` `REAL_EXECUTION_VIA_DOCKER`: Dry-run: no real episodes executed.
- `warning` `NO_EPISODES`: Runner output contains no executed episodes yet; decision inputs are contract-only.

## Missing Replay Warnings

- None

## Schema Quality

- Missing optional diagnostic fields: `anomaly_summary_lite`, `decision_context`, `evidence_windows`, `field_availability`, `final_window`, `outcome_state`, `reward_attribution_lite`, `step_log`
- Present optional diagnostic fields: none

## AI Diagnostic Quality

- Reliability level: `low`
- Adjacent ai_summary.json detected: `False`
- Missing optional signals affecting AI diagnostics: `anomaly_summary_lite`, `decision_context`, `evidence_windows`, `field_availability`, `final_window`, `outcome_state`, `reward_attribution_lite`, `step_log`
- Reliability notes:
  - `limited` charger_unknown / return_too_late: Charger knowledge or charge-return timing is partially missing, so early-vs-late return calls are weaker.
  - `unreliable` optimistic_route_budget: No charger-slack signal was captured, so route-budget conclusions would be speculative.
  - `limited` repeated_invalid_move / loop_suspect: Loop and invalid-move signals are too sparse for confident stuck-pattern classification.
  - `limited` npc_or_path_blocked: Nearest-NPC distance or step telemetry is missing, so blockage calls should be treated as weaker than charger/battery signals.
  - `unreliable` high_score_battery_death / late_battery_death: No episodes were available, so battery-death quality cannot be assessed.
