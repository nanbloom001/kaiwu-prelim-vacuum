# Holdout Benchmark Analysis

- Run ID: `task-8-synthetic-holdout`
- Checkpoint: `D:\TcKaiwuFinal\code\model.ckpt-resume.pkl`
- Maps: `[4, 7]`
- Episodes per map: `3`
- Combined status: `OK`

## Combined

- Episode count: `5`
- Avg clean score: `805.0`
- P10 / P50 / P90: `584.0` / `880.0` / `948.0`
- Completed / Battery fail / Collision fail: `0.2` / `0.6` / `0.2`

## Per Map

| Map | Status | Episodes | Avg Score | P10 | P90 | Battery Fail | Collision Fail |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | OK | 2 | 887.5 | 853.5 | 921.5 | 1.0 | 0.0 |
| 7 | OK | 3 | 750.0 | 504.0 | 944.0 | 0.3333 | 0.3333 |

## Failure Classification

- Analyzed failure count: `4`

| Category | Count | Example |
| --- | ---: | --- |
| Charger unknown / first charger not found | 0 | - |
| Known charger but return too late | 0 | - |
| Optimistic route budget / negative slack | 1 | task8-replay-route-budget map=4 score=845.0 steps=720 replay=D:\TcKaiwuFinal\.sisyphus\evidence\task-8-synthetic-death-replay.jsonl |
| NPC or path blocked | 1 | task8-collision-fail map=7 score=410.0 steps=430 replay=None |
| Repeated invalid move / stuck pattern | 0 | - |
| High-score battery death | 1 | task8-highscore-missing-replay map=4 score=930.0 steps=760 replay=D:\TcKaiwuFinal\.sisyphus\evidence\task-8-missing-replay.jsonl |
| Late or high-step battery death | 1 | task8-late-battery-death map=7 score=880.0 steps=980 replay=None |
| Unknown | 0 | - |

## Next Step

- Status: `ACTIONABLE`
- Optimization level: `high`
- Recommendation: Prioritize battery-death mitigation over coverage gains because strong-cleaning episodes are still dying before safely returning.
- Evidence paths:
  - `D:\TcKaiwuFinal\.sisyphus\evidence\task-8-synthetic-holdout.json`
  - `D:\TcKaiwuFinal\.sisyphus\evidence\task-8-synthetic-death-replay.jsonl`
  - `D:\TcKaiwuFinal\.sisyphus\evidence\task-8-missing-replay.jsonl`

## Risks

- `warning` `MISSING_OR_UNREADABLE_DEATH_REPLAY`: Episode death_replay_path is missing; JSON fields were used without replay enrichment. Path: D:\TcKaiwuFinal\.sisyphus\evidence\task-8-missing-replay.jsonl

## Missing Replay Warnings

- `D:\TcKaiwuFinal\.sisyphus\evidence\task-8-missing-replay.jsonl`: Episode death_replay_path is missing; JSON fields were used without replay enrichment.
