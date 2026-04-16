# Changelog

## 2026-04-16

~17:50 | 补齐 LTSPPO 诊断指标接线。训练侧在 `train_workflow.py` 中为每个 episode 新增并汇总 `late_return_rate`、`target_switch_rate`、`mode_usage_{clean,prepare_return,return,evade}`；monitor 面板在 `conf/monitor_builder.py` 中补齐对应曲线；benchmark 在 `eval/benchmark.py` 中新增同批指标的 episode / per-round / overall 聚合与摘要打印。完成后再次复核训练栈，learner 已稳定推进到 `global step = 2015`，并持续输出 `policy_loss / value_clean_loss / value_survive_loss / mode_teacher_loss / target_teacher_loss` 日志，说明诊断补丁未打断 LTSPPO 训练链。

~17:35 | LTSPPO Docker/Kaiwu smoke test 已打通。排障链路分 3 步：1) learner 首轮被 `mem_buffer.append: Sample size 168129 exceeds max_sample_size 100000` 阻塞，已将 recurrent chunk 从 `32/8/24` 缩小到 `16/4/12`；2) 随后 learner 卡在 replay buffer 预热，定位为 `configure_app.toml` 仍沿用旧单步 sample 吞吐参数，已下调为 `replay_buffer_capacity=1024`、`preload_ratio=0.25`、`send_sample_size=512`、`train_batch_size=128`；3) learner 真正进入训练后命中 `AttributeError: 'Tensor' object has no attribute 'obs'`，定位为 Kaiwu 实际向 `algorithm.learn()` 传入的是 tensor / tensor sequence，而非纯 `SampleData` 对象列表，已在 `agent.py` 与 `algorithm.py` 增加 batch 兼容解包层，支持对象列表、平铺 batch tensor、字段张量列表/样本张量序列。最终结果：learner 成功训练到 `train count = 279`、`global step = 279`，训练耗时约 `69.62 ms/step`（`data_fetch ≈ 4.33 ms`, `real_train ≈ 65.27 ms`），并成功保存/推送 `model.ckpt-500.pkl` 与 `kaiwu_checkpoint_robot_vacuum_ppo_500.tar.gz`。这标志 LTSPPO 已通过 “收样 -> chunk -> learner 解包 -> sequence PPO 反向传播 -> 存模/推模” 的端到端验证。详细排查日志见 `sessions/LTSPPO_SMOKETEST_DEBUG_LOG_20260416.md`。

~17:45 | 新增 `optimization/V6_SYNTHESIS_LONG_TERM_RECOMMENDATION_20260416.md`，作为两版 v6 方案与新终审文档的综合收敛版。该文档不再简单二选一，而是在吸收 `V6_DEFINITIVE_ARCHITECTURE_20260416.md` 的 `GRU + Dual Critic + 16×16 Global + Expert 充电控制降级` 主轴，以及原 `UNIFIED_TASK_MODELING_AND_V6_ARCHITECTURE_20260416.md` 的 `mode / target / entity / history` 显式建模思想后，收敛为 `V6-LTSPPO`：`Entity-aware + Target Pointer + GRU(192) + Dual Critic + sequence-aware PPO` 的长期主线方案。同时单独指出前面多份方案共同遗漏的 4 个关键点：`legal_action` 不应继续并入主状态表征、recurrent 训练不能停留在单步 hidden replay、target 应做对象 pointer 而非固定 charger 槽位分类、teacher/Expert 信号必须做可靠性门控。
~17:20 | 重写 `optimization/UNIFIED_TASK_MODELING_AND_V6_ARCHITECTURE_20260416.md`，将其从“折中式升级建议”改为“明确大改模型的 v6 正式重设计报告”。新版本不再沿用当前前馈骨架的保守前提，而是在官方任务建模、统一 checkpoint eval 结果、以及 `external_ai` 下三份独立分析的共同结论上，收敛为一套统一推荐架构：`Local 21×21×8 + Global 16×16×7 + Entity Set + Scalar/History + Action History` 的观察体系，配合 `多分支编码器 + GRU(256) + mode head + charger target head + mode/target conditioned action head + main/survival/clean 多价值头 + 4 个辅助头` 的 `V6-SRMTA` 主结构，同时明确 Expert 只保留硬安全过滤、极端低电 fallback 和早期 teacher signal 三种职责。
~16:55 | 继续整理 `analysis/OFFICIAL_TASK_MODELING_20260416.md` 正文，不删减原有信息，重点把稀疏段落压成高信息密度表格：`3.2 每个配置项到底在改变什么` 改为统一配置解释表，`13.1-13.7` 的指标定义、用途、判读方式和面板建议改为矩阵式表格，便于快速扫描，同时保留原有结论不变。
~16:45 | 继续整理 `analysis/OFFICIAL_TASK_MODELING_20260416.md`。在不删减原内容的前提下，为第 5-12 节补充了章节级总表，包括“形式化对象速览”“难度轴总表”“场景族总表”“核心子问题总表”“核心难点总表”“建模结论总表”“用途总表”，使后半部分也能像前面的速读区一样快速扫描。
~16:35 | 对 `analysis/OFFICIAL_TASK_MODELING_20260416.md` 做阅读友好化整理，在不改变结论的前提下新增“快速总览”表格区，将章节导航、核心结论、官方可配置轴、推荐评价指标集中成一页速读入口，便于人类和模型快速定位重点。
~16:20 | 扩充 `analysis/OFFICIAL_TASK_MODELING_20260416.md`，新增“合理的评价指标设计”章节。将指标分为三层：正式主指标（`AvgCleanScore`、`SurvivalRate`、`CPS_all`、`BatteryFailRate`、`CollisionFailRate`、`P10CleanScore`）、分解型客观指标（如 `CPS_win`、`ChargeEfficiency`、`InvalidMoveRate`、`PerMapVariance`、`ProfileSurvival`），以及训练判读辅助指标（如 `entropy`、`policy_loss`、`value_loss`、`avg_expert_weight`）。同时明确不建议把 `shaped reward`、`AvgChargeCount`、`entropy`、`PolicyLoss/ValueLoss` 直接当成最终性能主指标。
~16:00 | 新增基于官方文档的赛题建模报告 `analysis/OFFICIAL_TASK_MODELING_20260416.md`。该文档不讨论具体网络实现，专门从 `tencentarena-docs` 出发，整理了赛题固定规则、6 个官方可配环境轴（`map`、`map_random`、`robot_count`、`charger_count`、`max_step`、`battery_max`）、混合可观测信息结构、由配置轴诱导出的难度空间，以及当前仓库训练/benchmark 如何映射到官方赛题空间。用途是作为后续观察设计、网络设计、reward 设计和 benchmark 设计的统一任务模型基线。
~15:30 | 完成 v6 设计稿《统一赛题建模与网络结构方案》。在三份外部专家分析、v5.3 以来训练/诊断结论、以及 5 个手动 checkpoint 的统一 eval 基础上，重新把赛题建模为“资源受限、部分可观测、分阶段的生存约束清扫任务”，并给出统一推荐结构：`Local 21×21×8 + Global 16×16×6 + Entity Set + Scalar/History` 的观察设计，配合 `多分支编码器 + GRU(192) + mode head + 条件动作头 + 主/辅多价值头` 的 v6 网络方案，同时明确 Expert 重新定位、sequence-aware PPO 接口和辅助目标要求。详见 `optimization/UNIFIED_TASK_MODELING_AND_V6_ARCHITECTURE_20260416.md`。
~14:35 | 对全部手动保存点做统一 `4×10` 并行 benchmark 评估，覆盖 `v5-step4300`、`v51-step4900`、`v52-step10000`、`v52-step70000`、`v53-robust3450`。结论：`v52-step10000` 为本次统一口径下的综合最优（WR 70%, AvgCS 867, battery 8, collision 4），`v53-robust3450` 在保持同样 WR 的前提下把 collision 压到最低（2），但 battery 仍是主瓶颈（10）。`v52-step70000` 相比 `v52-step10000` 有明显回退，主要体现在 collision 恶化和 round_4 崩盘。详见 `benchmark/MANUAL_CHECKPOINT_EVAL_REPORT_20260416.md`。
~13:40 | 整理并行 benchmark 与仓库文档体系。新增根目录 README，补齐 Linux 下训练、串行评估、并行评估的主要入口、命令和结果路径。并行 benchmark 当前推荐固定使用 `4×10`（`--workers 4 --envs-per-worker 10`），完整 40 局墙钟时间约 108s，结果可稳定落地且脚本结束后容器可自动清理。并行 benchmark 实现与问题排查已沉淀到 `benchmark/BENCHMARK_PARALLEL_HANDOFF_20260416.md`。
~13:55 | 清理明显过期的已跟踪文件：移除 `code/eval_logs/**`、`code/eval_results.json`、`train/.docker-compose.yaml.bak`、`train/backup_20260414/**`、`code/reverb_dataset_v1_optimized.py.disabled`、`train/simple_dashboard.py`、`train/simple_monitor.py`、`log-412996-12336239.zip`。同时在 `.gitignore` 中补充 `code/eval_logs/`、`code/eval_results.json`、`code/.benchmark_done`，避免 benchmark 运行产物再次污染仓库。
~14:05 | 重组 `train/context/` 为二级分类目录：`analysis/`、`benchmark/`、`data/`、`diagnosis/`、`handoff/`、`operations/`、`optimization/`、`sessions/`，根目录仅保留 `README.md` 和 `CHANGELOG.md`。同步更新 `CLAUDE.md`、根 README 和上下文文档中的主要引用路径。后续涉及仓库结构、主要入口、benchmark 工作流或重要运维变更时，默认同时更新本文件。

## 2026-04-15

~14:00 | 训练全周期瓶颈分析报告完成。覆盖 v4→v5.4 全部 7 个瓶颈：entropy 塨缩（已解决）、Expert-RL 梯度对抗（核心结构性问题）、碰撞死亡（v5.4 修复 89%）、电池死亡（当前主瓶颈，bias 3-8 太弱）、GAE 长周期 credit 衰减（γλ^50=4.7%）、Reward 失衡（清扫:充电=60:1）、Peak-Then-Decline 训练曲线。含 18 例碰撞 + 14 例电池死亡完整日志。评估 4 个网络框架改动方向：LSTM 时序层、n-step return、势函数 reward shaping、分层策略。详见 BOTTLENECK_ANALYSIS_FULL_20260415.md。

# 2026-04-16

~17:35 | LTSPPO 主链路代码迁移进入“可静态联调”状态（linux-LTSPPO 分支）。已完成 `conf.py` / `feature/definition.py` / `model.py` / `expert.py` 的 LTSPPO 骨架重构，并继续接通 `preprocessor.py` / `agent.py` / `workflow/train_workflow.py` / `algorithm.py`。当前实现包括：Observation 拆分为 `Local 21x21x8 + Global 16x16x6 + Entity 8x7 + Scalar 88 + Action History 16`；模型为多分支编码器 + `GRU(192)` + `mode head` + `target pointer` + `dual critic` + `aux heads`；训练侧改为 chunked recurrent PPO（`chunk=32 / burn-in=8 / stride=24`）；Expert 收缩为 `NPC safety filter + emergency fallback + teacher reliability helpers`；workflow 已切到 `step-record -> sample_process(chunk)` 流程；benchmark 兼容新的 reward dict。使用 `PYTHONPYCACHEPREFIX=/tmp/pycache-ltsppo python3 -m py_compile ...` 对核心链路文件完成语法检查通过。新增 `code/tests/test_ltsppo_contracts.py`，当前本地因环境缺 `numpy` 自动 skip，但不再直接 import 崩溃。

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
