# Changelog

## 2026-04-13

~20:30 | 撤销步数对齐保存（框架签名 .zip 不受我们代码控制），改为将 RESUME_TIME_SNAPSHOT_INTERVAL 从 15 分钟缩短到 10 分钟。
~20:05 | 手动保存最佳 checkpoint：v51-step4900。选择理由：CC=7.0（充电策略最佳）、Entropy=0.37（健康）、CompRate=1.00、CS=1035。在 10 个候选点中 balanced score 排名第 2（1553），仅次于 ckpt-1000（早期不稳定）。保存至 code/saved_models/v51-step4900/。
~17:30 | v5.1 充电效率修复（仅改 preprocessor.py）。(1) 新增 pre_charge_battery 追踪充电前电量；(2) charge_reward 替换为效率公式 3.0*(charge_received/battery_max)，消除"充满后 reward 恒 2.0"的 bug；(3) 新增 urgency_penalty：charger_slack<0 时 -0.4*min(-slack/8,1)，强化本地紧急信号替代依赖 GAE 传播的弱死亡惩罚。
~16:35 | v5 优化实施完成（基于外部 AI 专家方案 + 3-agent 验证团队核查）。4 文件修改：(1) agent.py predict() 修复 prob 存储——biased采样+clean存储，消除 PPO 梯度对抗；(2) preprocessor.py charge_reward 0.3→2.0+1.5*need, charger_approach 0.15→0.40, 删除 freq_penalty, clip [-3,4]→[-5,5];(3) algorithm.py 加 adaptive entropy floor;(4) conf.py BETA=0.012, CLIP=0.15, LR=5e-5, ENTROPY_FLOOR=0.5, COEF=0.1, RESUME_CHECKPOINT 启用。Resume 从 v5-step4300。详见 V5_OPTIMIZATION_ANALYSIS.md。
~12:00 | v4 训练深度分析完成。核心发现：Expert Logit Bias 机制失败——soft bias 与 PPO 梯度对抗，模型学会"充电不好"。Entropy 2.0→0.08 塌缩。推荐新方案：Hard Override + PPO 梯度隔离（override 时不参与 policy loss）+ Entropy floor。详细分析见 DIAGNOSIS_v4_analysis.md。
~11:50 | 选定 resume checkpoint：resume-episode-ep000200 (step~4300, CPS=0.862, comp=100%)，迁移至 saved_models/v5-step4300/。
~04:15 | v4 训练停止（step 7606, 2736 ep）。趋势：CPS 0.29→0.89 但 comp rate 100%→68%，entropy 2.0→0.08 塌缩，charge_count 14→1。

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
