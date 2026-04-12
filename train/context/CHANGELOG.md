# Changelog

## 2026-04-12

18:15 | Diagnosis & optimization plan written to DIAGNOSIS_AND_DIRECTIONS_20260412.md. Covers: entropy collapse analysis, reward-reward alignment critique, evaluation of another AI's proposal (7 accept/modify/reject decisions), final execution plan (6 changes across 4 files). Key decisions: BETA 0.012 (not 0.02), replay buffer 4096 (not 10000), add per-step CPS_ema reward (missing from other AI), keep cleaning_reward 1.5 (not 2.0).

17:10 | Fix resume checkpoint loading: learner used /data/projects/ path where resume file didn't exist, model sync overwrote aisrv's correct weights with learner's random checkpoints. Added multi-path fallback. Verified: entropy 0.98 (not 2.0), avg_score 838.

16:27 | Expert charging overhaul: state machine with hysteresis + A* actual distance + blocked cell memory (TTL=8) + visit count penalty in A* cost + path caching + dynamic margin. BETA 0.005->0.008. NPC cleaned tracking in preprocessor. (commit 9c1083b)

~14:00 | Coordinate bug fix (3 functions in preprocessor.py) + predict() layer reorder (NPC->Expert->Anti-stuck->RL) + expert uses model softmax probability + charging rewards 2x + BETA 0.007->0.005. (commit 2715e42)
