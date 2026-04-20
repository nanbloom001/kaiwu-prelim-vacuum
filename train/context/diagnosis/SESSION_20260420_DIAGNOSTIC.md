# Training Session Diagnostic Report — 2026-04-20

**Session**: `20260420-091040` (aisrv logs start 09:43, data covers ~2h to ~11:34)  
**Workers**: pid440 (resume_writer=1, GPU1), pid441 (GPU1), pid445 (dual-helper, GPU2)  
**Learner checkpoint range**: 0 → 67,023 steps

---

## 1. Overall Training Health

### Metrics Trajectory (from pid440 training_metrics, most consistent)

| Time  | Step    | SPCRatio | Reward | Loss | Entropy | clean_score |
|-------|---------|----------|--------|------|---------|-------------|
| 09:43 | 0       | —        | 18.89  | 4.82 | 1.99    | 139.85      |
| 09:53 | 3,524   | 870.8    | 8.35   | 3.20 | 1.98    | 204.38      |
| 10:05 | 11,001  | 407.4    | 23.86  | 4.00 | 1.98    | 352.15      |
| 10:23 | 22,269  | 354.0    | 43.98  | 3.27 | 1.95    | 304.97      |
| 10:37 | 31,422  | 357.5    | 58.04  | 5.34 | 1.91    | 349.97      |
| 10:50 | 39,787  | 340.5    | 47.10  | 4.79 | 1.76    | 340.87      |
| 10:57 | 44,133  | 335.7    | 63.01  | 6.28 | 1.77    | 360.68      |
| 11:03 | 47,846  | 321.7    | 82.52  | 5.71 | 1.68    | 409.81      |
| 11:15 | 55,297  | 323.0    | 90.98  | 6.12 | 1.66    | 383.68      |
| 11:22 | 59,647  | 318.5    | 93.64  | 6.48 | 1.62    | 405.47      |
| 11:29 | 63,343  | 314.3    | 98.28  | 7.48 | 1.55    | 390.81      |

**Verdict**: Reward and clean_score are climbing steadily. Entropy is declining at a healthy pace (1.99→1.55). Loss is rising (4.82→7.48) — expected with increasing reward magnitude but worth monitoring if it exceeds ~10.

---

## 2. Episode Outcome Summary

### PID 440 (12 completed episodes)
| EP | Result | Steps | Score | Reward (eff) | Profile | Death |
|----|--------|-------|-------|-------------|---------|-------|
| 1  | FAIL   | 778   | 251   | 21.36       | mild    | collision |
| 2  | WIN    | 1100  | 333   | 6.63        | broad   | — |
| 3  | WIN    | 1000  | 364   | 3.15        | anchor  | — |
| 4  | WIN    | 500   | 248   | 85.08       | broad   | — |
| 5  | WIN    | 1000  | 389   | 69.41       | anchor  | — |
| 6  | WIN    | 1000  | 475   | 89.59       | mild    | — |
| 7  | FAIL   | 300   | 178   | 0.87        | anchor  | battery |
| 8  | WIN    | 700   | 365   | 96.56       | broad   | — |
| 9  | FAIL   | 907   | 477   | 6.68        | anchor  | battery |
| 10 | WIN    | 1000  | 506   | 197.95      | anchor  | — |
| 11 | WIN    | 1000  | 501   | 125.24      | anchor  | — |
| 12 | WIN    | 1000  | 467   | 156.71      | mild    | — |

**Win rate**: 9/12 = **75%**

### PID 441 (14 completed episodes)
| EP | Result | Steps | Score | Reward (eff) | Profile | Death |
|----|--------|-------|-------|-------------|---------|-------|
| 1  | WIN    | 1250  | 508   | 67.15       | mild    | — |
| 2  | WIN    | 1100  | 574   | 93.02       | broad   | — |
| 3  | WIN    | 1000  | 295   | -4.75       | anchor  | — |
| 4  | WIN    | 500   | 318   | 144.98      | broad   | — |
| 5  | WIN    | 1000  | 461   | 63.98       | anchor  | — |
| 6  | FAIL   | 300   | 74    | -23.95      | mild    | battery |
| 7  | FAIL   | 300   | 175   | 3.40        | anchor  | battery |
| 8  | FAIL   | 249   | 118   | 1.13        | broad   | collision |
| 9  | WIN    | 1000  | 416   | 106.96      | anchor  | — |
| 10 | WIN    | 1000  | 567   | 151.99      | anchor  | — |
| 11 | WIN    | 1000  | 414   | 59.86       | anchor  | — |
| 12 | FAIL   | 420   | 230   | 3.71        | mild    | battery |
| 13 | WIN    | 1000  | 428   | 106.98      | anchor  | — |
| 14 | FAIL   | 1011  | 467   | 43.85       | mild    | battery |

**Win rate**: 9/14 = **64.3%**

### PID 445 (dual-helper, ~22 completed episodes combined)
Win rate: approximately **60–65%**

### Aggregate Outcome
- **Overall win rate**: ~67% (28 wins / 42 episodes)
- **Battery death rate**: ~29% (12/42)
- **Collision death rate**: ~5% (2/42)
- **Battery death is the dominant failure mode** — 6x more frequent than collision

---

## 3. Battery Death Pattern Analysis

### Common characteristics across all battery death trajectories:

1. **Mode 4 lock-in**: Every battery death shows the robot stuck in `mode=4` (emergency/charge-seek mode) for the final 20+ steps. It never exits this mode.

2. **Negative slack throughout**: Slack values are consistently negative (-5 to -40), meaning the robot has already overshot the safe-return window by the time it starts seeking a charger.

3. **Action diversity in emergency**: During mode=4, the robot tries varied actions (0-7) but fails to converge on a path to a charger. This suggests the policy has not learned effective charge-return navigation.

4. **Profile correlation**:
   - **anchor** (battery_max=300, 1 robot): 7 battery deaths, ~50% failure rate in this profile
   - **mild** (battery_max=300-420, 1-2 robots): 4 battery deaths
   - **broad** (battery_max=160-720, 3-4 robots): 1 battery death (battery_max=160)

5. **Critical finding**: Battery deaths at step=300 (battery_max=300) indicate the robot never charges at all — it depletes its full battery in one go. Deaths at higher steps (907, 946, 1011) indicate the robot charges at least once but fails on a subsequent charge cycle.

### Root cause: The robot does not learn to initiate charging proactively. It waits until `slack < 0` (too late) before seeking a charger.

---

## 4. Reward Component Analysis

### Top negative signals (from REWARD_TOP across all episodes):

| Component | Frequency in top-3 neg | Typical range | Interpretation |
|-----------|----------------------|---------------|----------------|
| coverage_tangle_penalty | ~90% of episodes | -0.009 to -0.054 | Re-visiting already-cleaned cells |
| charge_detour_cost | ~80% of episodes | -0.004 to -0.020 | Inefficient paths to chargers |
| planner_alignment | ~50% of episodes | -0.005 to -0.013 | Deviation from optimal coverage plan |
| npc (collision risk) | ~15% of episodes | -0.019 to -0.032 | Near-miss or collision with NPCs |
| idle | ~15% of episodes | -0.004 to -0.010 | Standing still or spinning |
| charge_interrupt_cost | ~10% of episodes | -0.004 to -0.011 | Leaving charger prematurely |

### Top positive signals:
| Component | Frequency in top-3 pos | Typical range |
|-----------|----------------------|---------------|
| cleaning | 100% | 0.021 to 0.180 |
| explore | ~95% | 0.011 to 0.097 |
| streak | ~85% | 0.008 to 0.063 |
| charge_route_progress_bonus | ~10% | 0.007 to 0.014 |
| cps_bonus | ~5% | 0.023 |

### Key insight: `coverage_tangle_penalty` is the single most persistent behavioral problem. Even in high-reward episodes (e.g., 197.95 reward, ep10 pid440), it remains at -0.023. The robot's exploration is improving (explore reward growing) but it still wastes significant time revisiting areas.

---

## 5. Replay Pipeline Stability

| Metric | Start (09:43) | End (~11:30) | Trend |
|--------|--------------|-------------|-------|
| sample_production_and_consumption_ratio | 870.8 | 314.3 | ↓ (improving) |
| sample_receive_cnt growth rate | ~500/ep | ~2000/ep | ↑ (healthy) |
| predict_succ_cnt | 81,777 | 340,000+ | ↑ (healthy) |

**SPCRatio**: Dropped from 870→314 over the session. Still very high (ideal ~1-10). This confirms the known data pipeline bottleneck: aisrvs produce samples much faster than the learner consumes them. The ratio is improving but remains 30-80x above ideal.

**Model sync**: Working correctly. Checkpoint IDs advance smoothly (3500→63000 for pid440). Both `models/` and `models_new/` paths are used, indicating the double-buffer checkpoint system is operational. Cache hits occur when the same checkpoint is still current between short episodes.

---

## 6. Answers to Key Questions

### Q1: Is the model improving?
**Yes.** Average reward climbed from 18.89 to 98.28 (5.2x). clean_score increased from 139.85 to 390.81 (2.8x). Win rate in later episodes is higher than early ones. The improvement is real and substantial.

### Q2: What is the primary behavioral problem?
**Coverage tangling** — the robot frequently revisits already-cleaned areas. `coverage_tangle_penalty` is the #1 negative reward signal in ~90% of episodes and persists even in late-training high-reward episodes.

### Q3: Why do episodes fail?
**Battery exhaustion** accounts for 86% of failures (12/14). The robot fails to learn proactive charging — it depletes its battery before seeking a charger. The `anchor` profile (battery_max=300, 1 robot) is the hardest, with ~50% failure rate.

### Q4: Is the replay/reverb pipeline stable?
**Improving but still bottlenecked.** SPCRatio dropped from 870→314, meaning production still outpaces consumption by 300:1. No crashes, timeouts, or data corruption observed. The pipeline is functionally stable but throughput-limited on the learner side.

### Q5: Is entropy collapsing?
**No.** Entropy declined from 1.99 to 1.55 over ~67k steps. For 8 actions, max entropy = ln(8) ≈ 2.08. Current 1.55 represents ~75% of max entropy — the policy is specializing but still exploring healthily. No signs of premature collapse.

### Q6: Are there infrastructure issues?
**None detected.** All 3 GPU workers are running, model sync is working, checkpoint saving (time + step triggers) is functioning, no error-level log entries, no container restarts. pid445 runs dual-helper (two gamecore connections) which is working correctly.

---

## 7. Recommendations

1. **Battery management**: The anchor profile (300 battery, 1 robot) drives most failures. Consider:
   - Increasing `charge_detour_cost` penalty magnitude to force earlier charging
   - Adding a "low battery threshold" reward signal that triggers before slack goes negative
   - Curriculum: temporarily reduce anchor profile frequency until charging behavior improves

2. **Coverage tangle**: This is the ceiling on score improvement. Consider:
   - Strengthening the `planner_alignment` positive reward to guide systematic coverage
   - Adding a per-cell revisit counter that scales the tangle penalty with revisit count

3. **SPCRatio**: At 314:1, the learner is severely sample-starved relative to production. This is a known issue (per CLAUDE.md) — do not change until replay pipeline is characterized.

4. **Loss monitoring**: Total loss rose from 3→7.5. If it exceeds 10, consider reducing learning rate. Currently acceptable given the reward magnitude increase.
