# Changelog

## 2026-04-12

~23:30 | Fix resume control: add RESUME_CHECKPOINT config in conf.py (None=fresh, filename=resume). Replace agent.py hardcoded auto-resume with config-driven logic. Remove redundant resume in train_workflow.py. Root cause: model.ckpt-resume.pkl always existed → every platform start triggered resume.

## 2026-04-12 v4

~01:00 | Comprehensive optimization v4 — parameter revert + reward rebalance + Expert Logit Bias. Selective revert to backup (7012) params: LR=0.0001, BETA=0.008, CLIP=0.2, buffer=10000/Uniform. Remove logit noise + adaptive entropy. Reward fixes: (1) wall bonus↓ revisit↑ prevents wall-looping, (2) NPC penalty coeff 0.5→1.5 range 6→8, (3) new npc_cleaned_penalty -0.3, (4) charge_reward efficiency-based (base+need-freq), (5) frontier_reward 0.10→0.15. Expert Logit Bias: new get_logit_bias() soft guidance during training, get_override() hard override during evaluation. Collision death -4.0→-6.0. 7 files modified.
~23:58 | Verified: local compose with `-p kaiwu-train` starts all 17 containers independently. Learner starts fresh (no RESUME msg, entropy=2.07, step=-1). Key: `-p kaiwu-train` makes container names match framework's hardcoded hostname resolution. Training fully independent of official platform.

## 2026-04-12

~21:00 | Comprehensive optimization v3 applied (5 files). conf.py: BETA 0.012, LR 0.00005, CLIP 0.15. algorithm.py: adaptive entropy [0.3,0.7]. preprocessor.py: +dirty_approach_reward +CPS_ema efficiency reward, edge_bonus halved, explore_reward 0.03. train_workflow.py: efficiency_bonus CPS weight 0.3→0.6, cap 1.5→1.0. configure_app.toml: replay buffer 4096, FIFO sampler. Deferred Expert logit bias to round 2.

18:15 | Diagnosis & optimization plan written to DIAGNOSIS_AND_DIRECTIONS_20260412.md. Covers: entropy collapse analysis, reward-reward alignment critique, evaluation of another AI's proposal (7 accept/modify/reject decisions), final execution plan (6 changes across 4 files). Key decisions: BETA 0.012 (not 0.02), replay buffer 4096 (not 10000), add per-step CPS_ema reward (missing from other AI), keep cleaning_reward 1.5 (not 2.0).

17:10 | Fix resume checkpoint loading: learner used /data/projects/ path where resume file didn't exist, model sync overwrote aisrv's correct weights with learner's random checkpoints. Added multi-path fallback. Verified: entropy 0.98 (not 2.0), avg_score 838.

16:27 | Expert charging overhaul: state machine with hysteresis + A* actual distance + blocked cell memory (TTL=8) + visit count penalty in A* cost + path caching + dynamic margin. BETA 0.005->0.008. NPC cleaned tracking in preprocessor. (commit 9c1083b)

~14:00 | Coordinate bug fix (3 functions in preprocessor.py) + predict() layer reorder (NPC->Expert->Anti-stuck->RL) + expert uses model softmax probability + charging rewards 2x + BETA 0.007->0.005. (commit 2715e42)
