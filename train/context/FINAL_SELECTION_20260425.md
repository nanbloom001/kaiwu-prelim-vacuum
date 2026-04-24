Final Selection Gate: NOT_READY

Date: 2026-04-25
Gate status: NOT_READY | THRESHOLD_NOT_MET_OR_UNVERIFIED

Rationale:
- There is no real [4,7] holdout configuration with a combined average > 900. The required real holdout path is not available; T3 runtime indicates REAL_EXECUTION_UNSUPPORTED_IN_T2 and NO_EPISODES / NEED_MORE_DATA.
- Without real holdout episodes, we cannot substantiate any final checkpoint selection; assertion would be speculative.

Checkpoint candidate:
- code/model.ckpt-resume.pkl

Current context:
- Current git hash/branch: 7734013 on branch win_YJY
- Recent prime commits considered for benchmarking: T4 5d578b9; T5 88ad9c8; T6 1d90960; T10 7734013
- Evidence: T3_REAL_EXECUTION_UNSUPPORTED_IN_T2; NO_EPISODES; NEED_MORE_DATA; HOLDOUT_BENCHMARK_BASELINE shows NOT_IMPLEMENTED

Next evidence requirement:
- Real holdout runtime path or inference-only runner must be connected to generate real episodes for maps [4,7], 10 episodes per map.
- After runtime path exists, re-run baseline and analyzer; then decide whether to upgrade to reward, refactor, or still no change.

Gate conditions for future final selection:
- Real maps exist for [4,7], 10 episodes/map
- Combined average > 900
- Stability gates healthy; variance within acceptable range
- Benchmark confirms availability and reproducibility

Packaging note (do not run yet):
- Packaging command template (do not execute): bash train/package_and_sign.sh code/model.ckpt-resume.pkl 9339
- The packaging step must wait until threshold is met.

Monitoring:
- Monitor URL: http://127.0.0.1:${MONITOR_TRPC_PORT}

Rollback point and recent commits:
- T4: 5d578b9
- T5: 88ad9c8
- T6: 1d90960
- T10: 7734013

Notes:
- This gate reflects a NOT_READY posture; no final success claim is made.
- The plan will await the required data and real episodes to progress.
