# Changelog

2026-04-25 03:45 | 初始化分层 AGENTS.md 知识库，聚焦比赛规划、win_YJY 当前分支、PPO 主线与训练/提交链路。
2026-04-25 05:15 | Added fixed [4,7] holdout benchmark dry-run/analyzer contract with mutation guard and NO_EPISODES-safe reporting.
2026-04-25 05:24 | Attempted non-dry-run BASELINE holdout benchmark; runner exited 3 REAL_EXECUTION_UNSUPPORTED_IN_T2, produced failure JSON/NO_EPISODES analysis with clean mutation guard.
2026-04-25 05:58 | Enhanced holdout analyzer with failure classification, missing replay fallback, and single next_step NEED_MORE_DATA/actionable diagnostics.
2026-04-25 06:07 | Fixed holdout analyzer reason predicates so battery_fail/out_of_battery/battery_death and collision_fail aliases count/classify correctly.

2026-04-25 07:04 | T0: Adapt LTSPPO charge constraint benchmark design; local holdout benchmark scaffolding prepared (inference-only evaluation, per-episode env config injection, and metric artifacts).
2026-04-25 07:15 | T1: Baseline capture anchored at HEAD a34e9aa with origin/win_YJY; validate resume_best latest alignment and mutation-guard inventory.
2026-04-25 08:05 | T4: Guarded best checkpoint promotion so only completed episodes can update robust best state; failure telemetry remains intact.
2026-04-25 08:30 | T5: Added coverage-target-only return buffer in algorithm.py to tighten battery gating without changing global return margins or charge-mode A*.
2026-04-25 08:45 | T6: Added safe closed-loop dry-run/decision runner for Train→Benchmark→Analyze→Accept/Rollback gating without starting Docker.
2026-04-25 09:00 | T6-fix: Corrected closed-loop accept gating so >900 score still requires completed-rate and clean_per_step stability checks.
2026-04-25 09:20 | T6-fix: Narrowed infrastructure mutation rejection so informational MODEL_MUTATION_GUARD risks no longer trigger REJECT while real mutation errors still do.
2026-04-25 09:40 | T6-fix: Tightened mutation guard again so warning/info artifact-change language rejects while unchanged MODEL_MUTATION_GUARD info stays non-blocking.
2026-04-25 06:20 | T10: Recorded evidence gate blocking reward/refactor/network escalation until real holdout episodes resolve REAL_EXECUTION_UNSUPPORTED_IN_T2 and NEED_MORE_DATA.
2026-04-25 09:10 | T9: Recorded NOT_READY final checkpoint selection gate; no real >900 holdout evidence exists yet.
2026-04-25 11:24 | T4: Added holdout episode lifecycle summaries/evidence windows and taught analyzer to classify charger-unknown and optimistic-route-budget failures from episode-level diagnostics; synthetic fixtures passed.
2026-04-25 11:44 | T6: Verified holdout compile, synthetic analyzer fixtures, and dry-run benchmark outputs; wrote fresh QA evidence without touching model artifacts.
2026-04-25 11:58 | F1 fix: added holdout contract.fixed_config alias plus pre-action decision_context fields step, pos_before, and last_action for deterministic rejection-gap closure.
2026-04-25 13:35 | T3: runner now aggregates shard-local holdout results itself, waits for every shard marker/result, fails closed on shard coverage/schema issues, and ships no-Docker aggregation evidence.

## 2026-04-15

~14:00 | 训练全周期瓶颈分析报告完成。覆盖 v4→v5.4 全部 7 个瓶颈：entropy 塨缩（已解决）、Expert-RL 梯度对抗（核心结构性问题）、碰撞死亡（v5.4 修复 89%）、电池死亡（当前主瓶颈，bias 3-8 太弱）、GAE 长周期 credit 衰减（γλ^50=4.7%）、Reward 失衡（清扫:充电=60:1）、Peak-Then-Decline 训练曲线。含 18 例碰撞 + 14 例电池死亡完整日志。评估 4 个网络框架改动方向：LSTM 时序层、n-step return、势函数 reward shaping、分层策略。详见 BOTTLENECK_ANALYSIS_FULL_20260415.md。

## 2026-04-14

~22:05 | v5.4 电池死亡根因诊断完成。通过在 Expert `_evaluate_return()` 添加文件诊断日志（3214 条），推翻初始假设（充电桩未发现/A* 无路径），确认真正原因：Expert A* 寻路和 action 提供均正常（path=20-52, act 非 None），但 non-emergency bias（3-8）太弱，73% 的电池死亡场景中模型清扫偏好压过充电引导，直到 ratio≤0.10 才触发 emergency bias=100 为时已晚。v5.4 碰撞死亡 -89%（18→2），WinRate 85.2%→88.2%。修复方向：提高 bias 强度 + 降低 emergency 阈值。详见 LOG_20260414_v54_battery_death_diagnosis.md。
~21:30 | v5.4 Expert 充电逻辑修复（5 项，仅改 expert.py）。Fix 1: filter_actions dist≤3 时拦截所有靠近 NPC 的动作（原仅拦截正对方向）；Fix 2: 移除 A* npc_weight=0.0 fallback（2 处，消除穿过 NPC 路径）；Fix 3: get_logit_bias NPC dist≤4 时抑制充电 bias（解决躲避/充电双系统冲突）；Fix 4: return_mode 触发修复——高电量 85% 强制退出（修复 ep 204 持久化 bug，专家方案盲区）+ 动态 LOW_BATTERY_RATIO（小电池提前触发）+ 0.65 距离触发守卫；Fix 5: UNEXPLORED_COST 3.0→1.8（改善 A* 早期寻路）。详见 iterative-orbiting-goblet.md。
~21:10 | Expert 充电逻辑完整审查报告完成。独立分析 18 碰撞 + 14 电池死亡根因，提出 5 项具体修改方案（优先级排序）：(1) filter_actions dist≤3 时扩展拦截所有靠近动作，(2) 删除 A* npc_weight=0.0 无 NPC 避让 fallback，(3) 返回模式加 0.65 电量守卫+动态 LOW_BATTERY_RATIO，(4) NPC 近距距离时抑制充电 bias 解决冲突，(5) 降低 UNEXPLORED_COST 3.0→1.8。并评估及时性（低/中/低）、预期收益（-70% 碰撞、-50% 电池死），中长期简化 Expert 路线。详见 EXPERT_REVIEW_20260414.md。
~19:15 | v5.3 训练趋势分析（step ~5500, ep ~228）。ENTROPY_FLOOR_COEF 修复成功（0.12-0.21 稳定无塌缩）。WinRate 85.2%。DEATH_TRAJ 精准定位两个 Expert bug：(1) 碰撞死亡 18 例全部在 mode=2（充电模式）—— A* fallback 过度降级到 npc_weight=0.0 + filter_actions 漏洞；(2) 电池死亡 14 例全部从未触发充电（mode=1 到死）—— return_mode 触发条件不鲁棒 + A* 路径 inf + 多机器人争抢。详见 LOG_20260414_v53_training_analysis.md。
~15:00 | v5.3 综合优化实施（基于双专家方案）。4 文件修改：(1) conf.py ENTROPY_FLOOR_COEF 0.2→1.0 修复 entropy 塌缩；(2) preprocessor.py reward_process 返回分量字典 + urgency 三段式（-0.3/-0.6/-1.2）；(3) agent.py 传递 reward_components；(4) train_workflow.py 死亡轨迹日志 + 每 config 失败率追踪 + outcome_bonus 按剩余步数缩放（k=1.5）。Resume from v52-step8500。详见 OPT_PLAN_v53_20260414.md。
~12:20 | 完整训练趋势分析（CPS+存活率标准）。结论：v52-step8500 已是最佳可用 checkpoint。训练在 step 8k-10k 达峰（CPS=0.895, Surv=0.818, CS×Surv=941），step 10k-12k CS×Surv 最高（1055）但该 checkpoint 已被 learner 自动清理，仅剩 session best 快照也已随 aisrv 重启丢失。step 12k 后所有指标持续下降，至 step 64k 无回升。v52-step8500 位于 peak zone 且已双备份保存。清理了之前用 rejected robust_score 标准保存的 ep733 模型。
~02:45 | 手动保存最佳 checkpoint：v52-step8500。选择理由：综合评分第 1（N≥8 可靠样本）。AvgCS=1320（最高）、MinCS=871（无灾难局）、ColRate=0.101（全场最低）、Entropy=0.149（健康）、VLoss=147。在 12 个候选点中以 balanced score 1210.8 排名第 1。已保存至 saved_models/v52-step8500/ + backup_model/ 双重备份。
~02:10 | 修复监控面板 "fail to fetch"：fe-monitor-service 容器 `server_req_base_url` 被旧版 compose 缓存为 Docker 内部主机名，`--force-recreate` 重建后正确注入 `http://127.0.0.1:11001`。
~00:28 | ENTROPY_FLOOR_COEF 0.1→0.2，温和推高 entropy（effective_beta 从 0.05→0.09）。重启 learner+aisrv 生效。

## 2026-04-13

~22:30 | v5.2 综合优化实施完成（基于外部 AI 专家方案批注 + 精细调整）。3 文件修改：(1) preprocessor.py：移除 wall_adjacent edge_bonus、dirty_approach 加守卫条件系数 0.08、revisit 三组件系统（只罚干净格+区域感知+模式豁免）、NPC penalty 范围 10 系数 -3.0 +方向分量；(2) expert.py：NPC filter 距离 3→5、logit_bias NPC 负偏置 -2.0、LOW_BATTERY_RATIO 0.26→0.32 + BASE_RETURN_MARGIN 14→18；(3) train_workflow.py：4 阶段课程(eval_hard ep401+ 37.5%强制2000步)、outcome_bonus 碰撞 -8.0/电池 -3.0、robust_score 碰撞权重 -30。Resume from v51-step4900。详见 OPT_PLAN_v52_20260413.md。
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

2026-04-25 10:23 | Adapted linux-LTSPPO benchmark runtime to win_YJY. New files: code/agent_ppo/eval/holdout_benchmark.py (inference-only benchmark), train/.docker-compose.benchmark.yaml. Modified: train_workflow.py (KAIWU_BENCHMARK_MODE dispatch), run_holdout_benchmark.py (docker compose launch). Smoke test passed: map4=372(battery fail), map7=896(completed), avg=634.
2026-04-25 12:40 | Fixed holdout diagnostic mask counting to avoid numpy truth-value crashes by using safe mask counters and first-value extraction.
2026-04-25 12:58 | Added shard-worker holdout benchmark execution with strict hostname assignment validation, shard-local result/done outputs, and benchmark gate relaxation for non-primary AISRV workers only in sharded mode.
2026-04-25 16:16 | Task 4 sharded smoke blocked: benchmark runner failed before JSON generation because hostname resolution inside kaiwu-train-aisrv-1 tried to exec missing 'hostname' binary.
2026-04-25 16:26 | Fix sharded holdout runner hostname resolution via docker inspect; Task 4 smoke now passes on maps 4/7 with 2 shards.
2026-04-25 16:31 | Task 5: real sharded 2x10 baseline finished below target at avg 652.4 (map4 605.8, map7 699.0; completed 0.65, battery fail 0.30, collision fail 0.05); analyzer marked ACTIONABLE with repeated invalid-move/stuck diagnostics first, and the closed-loop helper emitted CONTINUE for Task 6 candidate selection.
2026-04-25 23:02 | Fixed holdout benchmark parallel execution defaults: runner now shards by default, map-partitions maps 4/7 across AISRV workers, waits for shard assignments safely, and reports wall-clock shard timing.
2026-04-25 23:20 | Extended holdout benchmark parallelism to multi-GC per AISRV: runner now sizes gamecore as AISRV shards x workers-per-AISRV and benchmark workers bind independent env/agent pairs when available.
2026-04-25 23:31 | Set holdout benchmark default to 5 gamecore/env workers per AISRV so the fixed 10 episodes per map can complete in two worker waves when runtime exposes enough env/agent pairs.
2026-04-25 23:39 | Reverted benchmark default to 4 episodes per map and 4 gamecore/env workers per AISRV after real 2x5 attempt showed framework exposed only one env/agent pair per AISRV.
2026-04-25 23:46 | Investigated AISRV env handle exposure: multi-gamecore connects at AISRV server layer, but workflow still receives one env/agent handle; Linux branch confirms speedup should come from multi-AISRV task distribution.
2026-04-26 00:03 | Implemented 2x4 dual-concurrency holdout benchmark path with AISRV env-count TOML patching, dynamic task queue execution, dry-run contract, and observed-worker validation.
2026-04-26 00:10 | Validated real 2x4 dynamic holdout benchmark: 8/8 tasks completed, 8 logical workers observed across 2 AISRV x 4 process indexes, done marker reached at ~190s.
2026-04-26 00:12 | Documented 2x4 dynamic holdout benchmark as the README standard and corrected benchmark overlay default episode comment.
