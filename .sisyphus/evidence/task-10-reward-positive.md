# T10 Reward-Positive Gate

Status: **NOT SATISFIED**

T10 did not add or rewrite rewards. A reward-positive change is currently unsupported because the fixed T3 real baseline has no executed episodes and no before/after benchmark.

Evidence:

- T3 non-dry-run produced blocker `REAL_EXECUTION_UNSUPPORTED_IN_T2`.
- `train/context/HOLDOUT_ANALYSIS_BASELINE_20260425_0522.md` reports combined status `NO_EPISODES` and episode count `0`.
- `train/context/HOLDOUT_BENCHMARK_BASELINE_20260425_0522.json` has `episodes=[]`, planned episodes marked `NOT_EXECUTED`, and `baseline_established=false`.
- T8 baseline analysis returns `NEED_MORE_DATA`; analyzed failure count is `0` because there are no real episodes/death replays.

Conclusion: reward escalation is blocked until real fixed `[4,7]`, 10 episodes/map holdout data exists and shows a targeted failure prevalence plus before/after improvement.

Next evidence step:

```bash
python train/run_holdout_benchmark.py --maps 4,7 --episodes-per-map 10 --checkpoint code/model.ckpt-resume.pkl --output train/context/HOLDOUT_BENCHMARK_BASELINE_20260425_0522.json
```

T4 commit `5d578b9`, T5 commit `88ad9c8`, and T6 commit `1d90960` are separately benchmarkable candidates once runtime works; no T10 reward behavior change was made.
