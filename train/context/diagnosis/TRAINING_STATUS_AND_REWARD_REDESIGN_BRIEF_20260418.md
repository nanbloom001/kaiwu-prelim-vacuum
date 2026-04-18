# 训练现状与 Reward 系统性微调评估简报

日期：2026-04-18  
面向对象：外部专家 / 第二个 AI 评审  
目的：基于最近几轮训练、课程机制、preload 恢复、lite benchmark 起始分层与日志证据，判断是否需要对 reward 函数进行全局系统性微调，甚至从头训练一个更健康的新模型。

---

## 1. 本文要回答的问题

当前团队考虑的不再是“继续小修小补”，而是判断：

1. 是否应该对当前 reward 结构做一次系统性微调，而不是继续局部打补丁。
2. 是否应该在微调后，从头训练一个更健康的新模型，而不是继续在现有策略上硬续训。
3. 如果重新出发，目标应该明确围绕这三件事：
   - 足够高的存活率
   - 足够高的 `CPS / clean_per_step`
   - 学会预先规划与有规律的清扫，而不是局部保守求活

---

## 2. 最近已经做过的核心改动

以下改动都已经进入当前代码，但尚未证明“整体训练已经健康”。

### 2.1 训练入口：本地 resume 改为 preload

目标：
- 不再依赖 `Agent.__init__()` 里的本地 `RESUME_CHECKPOINT` 自动加载
- 改为 framework preload 统一加载 learner 与 aisrv

现状：
- 当前训练入口已经从 `preload` 启动
- 当前 preload 起点来自 `429` 主基线：
  - `/home/user/TcKaiwuFinal/code/saved_models/v6-geo-bestmodel-576/model.ckpt-resume.pkl`
- 当前 preload 元数据：
  - `/home/user/TcKaiwuFinal/code/agent_ppo/ckpt/latest_preload.json`

意义：
- 训练恢复更统一
- Learner/AISRV 起点一致
- 版本语义比旧本地 resume 更清晰

### 2.2 课程：从本地 helper 状态改为全局共享状态

目标：
- 不再让每个 helper 各自维护课程阶段
- 用全局 episode / 全局 recent window 共同决定 stage 和 progress

现状：
- 当前共享课程状态文件：
  - `/home/user/TcKaiwuFinal/code/curriculum_state.json`
- 已经能看到：
  - `global_episode_count`
  - `global_step_since_resume`
  - `last_bootstrap_metrics`
  - `last_global_metrics`

意义：
- 课程不再完全被单 helper 本地局数卡死
- 课程进度比早期版本真实得多

### 2.3 课程起始分层：新增 lite benchmark

目标：
- 如果当前 checkpoint 没有 benchmark 记录，训练启动前先跑一轮轻量 benchmark
- 根据结果决定初始 stage

现状：
- lite benchmark 缓存：
  - `/home/user/TcKaiwuFinal/code/agent_ppo/ckpt/latest_lite_benchmark.json`
- 当前 lite benchmark 结果：
  - `completed_rate = 1.0`
  - `battery_fail_rate = 0.0`
  - `collision_fail_rate = 0.0`
  - `return_stall_rate = 0.6614`
  - `recommended_initial_stage = warmup`

意义：
- 当前 run 并不是按人工预设 `blend` 起步，而是被 lite benchmark 判回了 `warmup`
- 这件事对当前行为风格影响很大

### 2.4 采样稳定性：修了概率非法导致的 aisrv 崩溃

之前出现过：
- `predict() Exception pvals < 0, pvals > 1 or pvals contains NaNs`
- 随后 workflow 因 `predict()[0]` 触发 `NoneType` 级联崩溃

现在已经加上：
- 概率清洗
- fallback action
- workflow 级保护

目标是：
- 不再让单次概率异常直接打停 learner

### 2.5 保存评分体系：从单一 robust_score 升级为多分项

当前评分不再只看旧 `robust_score`，而是拆成：
- `resume_readiness_score`
- `submission_score`
- `checkpoint_preservation_score`

并进一步暴露分项：
- `resume_score_safety`
- `resume_score_efficiency`
- `resume_score_behavior`
- `resume_score_learning`
- `submission_score_completion`
- `submission_score_efficiency`
- `submission_score_stability`
- `submission_score_behavior`

目标：
- 更准确地区分“可继续训练的点”和“可提交的点”
- 也为课程起点分层提供辅助信号

---

## 3. 当前最关键的运行状态

### 3.1 当前课程状态

来自：
- `/home/user/TcKaiwuFinal/code/curriculum_state.json`

当前摘要：

- `stage = warmup`
- `initial_stage = warmup`
- `lite_benchmark_used = true`
- `curriculum_progress = 0.4694`
- `global_step_since_resume = 19333`
- `global_episode_count = 120`

这说明：
- 课程没有卡死在“没生效”
- 但并没有进入 `blend/robust`
- 当前 run 的真实状态是：lite benchmark 判级保守，课程也确实认为模型还没达到出 warmup 的行为质量

### 3.2 当前全局 40 局窗口

同样来自：
- `/home/user/TcKaiwuFinal/code/curriculum_state.json`

关键指标：

- `win_rate = 0.85`
- `broad_win_rate = 0.7143`
- `battery_fail_rate = 0.15`
- `collision_fail_rate = 0.0`
- `avg_clean_per_step = 0.3915`
- `avg_clean_score = 323.25`
- `avg_charge_count = 23.35`
- `avg_remaining_charge = 199.3`
- `mode_usage_contract = 0.6909`
- `mode_usage_return = 0.1076`
- `mode_usage_expand = 0.000036`
- `return_stall_rate = 0.5574`
- `planner_policy_divergence_rate = 0.8533`
- `return_efficiency_ratio = 0.1082`

这组数值说明：

1. 成功率和 broad 胜率并不差  
2. 但清扫效率不高  
3. 充电次数极高，结束剩余电量也偏高  
4. 几乎没有 `expand`，大量时间都在 `contract`  
5. `return_stall_rate` 和 `planner_policy_divergence_rate` 依然非常高

结论：
- 当前策略不是“完全失败”
- 而是明显偏向“保守生存策略”

---

## 4. 当前训练趋势的核心判断

### 4.1 环境分数在变好，但 reward 长期偏负

从 aisrv 训练日志中可以看到：

日志来源：
- `/home/user/TcKaiwuFinal/train/log/aisrv/aisrv_kaiwu_rl_helper_pid443_log_2026-04-18-13.log`
- `/home/user/TcKaiwuFinal/train/log/aisrv/aisrv_kaiwu_rl_helper_pid444_log_2026-04-18-13.log`

典型窗口：

- `env.total_score` 大约从 `230` 上升到 `327`
- `algorithm.reward` 多数仍是负值，常见区间：
  - `-31.21`
  - `-22.30`
  - `-18.37`
  - `-15.88`
  - `-5.12`
  - `-2.03`
  - 也有少量回正，如 `1.13 / 4.13 / 8.96`

这意味着：
- 模型不是完全没有学到东西
- 但当前“赢”的方式，并不符合 reward 所鼓励的高质量行为

换句话说：
- 环境分数上升
- 不等于 reward 对齐
- 当前策略很可能是以“保守生存 + 频繁回充”换取 win，而不是高质量清扫

### 4.2 过度充电现象有明确证据

当前窗口里的：

- `avg_charge_count = 23.35`
- `avg_remaining_charge = 199.3`

同时：

- `mode_usage_contract ≈ 0.69`
- `mode_usage_expand ≈ 0`

这非常像：
- 早早进入 contract
- 频繁补能
- 带着较高剩余电量结束
- 实际探索和规整覆盖不足

因此“过度充电”不是错觉，而是当前策略结构性偏保守的表现。

### 4.3 返航质量仍然差

最值得重视的指标仍然是：

- `return_stall_rate ≈ 0.56`
- `planner_policy_divergence_rate ≈ 0.85`
- `return_efficiency_ratio ≈ 0.11`

这说明：
- 当前并不是“返航意识没有了”
- 而是：
  - 返航开始得早
  - 但返航执行并不高效
  - policy 仍经常不跟 planner

这会带来两个后果：

1. 即使保守，也不一定真的优雅  
2. 会把 reward shaping 长期压成负值

---

## 5. 近期日志中的代表性样本

以下样本足以说明当前问题不是偶发。

### 5.1 `broad` 早期直接 battery fail

日志：
- `aisrv_kaiwu_rl_helper_pid443_log_2026-04-18-13.log`

样本：
- `13:37:59` 窗口显示：
  - `reward = -31.21`
  - `charge_count = 20.43`
  - `remaining_charge = 122.09`
- `13:38:30`：
  - `ep1`
  - `profile=broad`
  - `FAIL battery`
  - `clean_score = 103`
  - 末端 `DEATH_TRAJ` 中 `mode=4` 持续负 slack

意义：
- 即使在当前已经相对保守的策略下，broad 仍然存在返航失败
- 说明“保守”并没有彻底解决 survivability

### 5.2 `mild` 里也出现 collision fail

日志：
- `aisrv_kaiwu_rl_helper_pid443_log_2026-04-18-13.log`

样本：
- `13:59:48`
  - `ep12`
  - `profile=mild`
  - `FAIL collision`
  - `clean_score = 27`

意义：
- 不是只有 hardest broad 才掉点
- 中等 profile 下仍然可能因为局部行为不稳直接失败

### 5.3 `anchor` 里也出现 battery fail

日志：
- `aisrv_kaiwu_rl_helper_pid444_log_2026-04-18-13.log`

样本：
- `14:03:38`
  - `ep14`
  - `profile=anchor`
  - `FAIL battery`
  - `clean_score = 360`
  - `DEATH_TRAJ` 中最后连续处于 `mode=4` 且 slack 一直为负

意义：
- 这非常关键
- 说明当前 survivability 问题不只是 hardest case 的尾部问题
- 连 anchor 也还会发生典型晚返航死亡

---

## 6. learner 侧数值趋势

日志来源：
- `/home/user/TcKaiwuFinal/train/log/learner/learner_train_pid333_log_2026-04-18-13.log`

当前整体判断：

- 训练链本身是正常在跑的
- `global step` 从 `0` 增长到 `20k+`
- `sample_production_and_consumption_ratio` 大约从 `137` 回落到 `110`
- `entropy_loss` 长期在 `1.93 ~ 2.00`
- `value_clean_loss` 波动较大，常见在 `4 ~ 16`
- `value_survive_loss` 常见在 `3.6 ~ 10`
- `route_anchor_teacher_active_rate / target_teacher_active_rate` 普遍较高

这些数值的含义是：

1. 训练没崩，也没有完全停滞  
2. 但 `entropy` 依然高，策略分布不够收紧  
3. value 头并没有表现出“已经极稳定收敛”的状态  

因此：
- 当前不是“训练坏掉”
- 但也远谈不上“已经收敛到健康策略”

---

## 7. 目前最重要的诊断结论

### 7.1 当前模型更像“过度保守求活”，不是“高质量规划清扫”

核心证据：
- 高成功率
- 高 broad 胜率
- 但高 charge_count
- 高 remaining_charge
- 极低 expand
- 高 contract
- 高 return stall
- 高 planner divergence
- reward 长期偏负

这不是一个理想的长期解。

### 7.2 lite benchmark 可能把起点判得过于保守

当前 lite benchmark 结果是：

- `completed_rate = 1.0`
- `battery_fail_rate = 0.0`
- `collision_fail_rate = 0.0`
- 但 `return_stall_rate = 0.6614`
- 最终判级：
  - `recommended_initial_stage = warmup`

这意味着当前课程起点非常依赖 `return_stall_rate`。  
这种分层虽然安全，但可能过度保守，并进一步放大了：

- 频繁回充
- 过早 contract
- 低 CPS

### 7.3 当前 reward 结构与“想要的最终行为”可能存在错位

用户希望的最终模型应当：

1. 有足够高的存活率  
2. 有足够高的 CPS  
3. 学会预先规划和有规律的清扫  

但当前观察到的策略更像：

- 存活率在改善
- CPS 不高
- 规划与执行仍然分裂
- 清扫行为不规整，几乎没有 expand

因此当前 reward 结构可能存在如下问题之一或多项叠加：

1. 对 survivability 的 shaping 太强，压过了清扫效率和规整覆盖  
2. 对“返航效率差但提早返航”的行为惩罚不够区分  
3. 对 planning / coverage quality 的长期激励不够强  
4. 使模型学会了“保守求活”，但没有学会“高质量完成任务”

---

## 8. 建议专家重点回答的问题

请重点判断以下问题：

### A. 是否需要系统性重构 reward，而不是继续局部微调

要点：
- 当前 reward 与 env score 已出现明显背离
- 当前 reward 似乎更擅长压风险，但不擅长引导出高质量规划清扫

### B. 是否应该从头训练，而不是继续沿当前策略修补

理由：
- 当前策略已经形成明显的保守生存偏置
- 如果 reward 主结构本身有问题，继续续训可能只是在固化偏差

### C. 如果从头训练，reward 目标应如何分层

建议专家围绕以下三目标给出 reward 结构建议：

1. **Survivability**
   - 不晚回充
   - 不撞
   - 不在 return 中长期 stall

2. **Efficiency**
   - 高 `CPS / clean_per_step`
   - 低无效重复
   - 不用高频补能换取表面存活

3. **Structured Planning / Coverage**
   - 学会 expand -> harvest -> contract -> return 的合理节奏
   - 学会更规整、少交叉、少回头的清扫
   - 学会更稳定地跟 planner 对齐，而不是长期 divergence

### D. lite benchmark 分层门槛是否过于保守

尤其请判断：
- 在 `completed_rate = 1.0` 且无 battery/collision fail 的情况下
- 是否仅凭 `return_stall_rate = 0.6614` 就把初始 stage 判回 `warmup`
- 这样是否会系统性把成熟 checkpoint 推回过度保守区间

---

## 9. 建议专家重点阅读的文件

### 当前状态与窗口
- `/home/user/TcKaiwuFinal/code/curriculum_state.json`
- `/home/user/TcKaiwuFinal/code/agent_ppo/ckpt/latest_lite_benchmark.json`

### 训练日志
- `/home/user/TcKaiwuFinal/train/log/learner/learner_train_pid333_log_2026-04-18-13.log`
- `/home/user/TcKaiwuFinal/train/log/aisrv/aisrv_kaiwu_rl_helper_pid443_log_2026-04-18-13.log`
- `/home/user/TcKaiwuFinal/train/log/aisrv/aisrv_kaiwu_rl_helper_pid444_log_2026-04-18-13.log`

### 当前方案与背景文档
- `/home/user/TcKaiwuFinal/train/context/optimization/CURRICULUM_REDESIGN_20260418.md`
- `/home/user/TcKaiwuFinal/train/context/optimization/CHECKPOINT_SELECTION_SCORE_V2_20260418.md`
- `/home/user/TcKaiwuFinal/train/context/diagnosis/PRELOAD_VS_RESUME_RESEARCH.md`

### 关键代码
- `/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py`
- `/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py`
- `/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_policy.py`
- `/home/user/TcKaiwuFinal/code/agent_ppo/workflow/checkpoint_score.py`
- `/home/user/TcKaiwuFinal/code/agent_ppo/eval/lite_benchmark_bootstrap.py`
- `/home/user/TcKaiwuFinal/code/agent_ppo/agent.py`
- `/home/user/TcKaiwuFinal/code/agent_ppo/utils/policy_sampling.py`

---

## 10. 最终摘要

当前这轮训练的核心特征不是“模型完全学坏了”，而是：

- **成功率正在恢复**
- **但恢复方式明显偏向过度保守求活**
- **高频补能、早期 contract、极低 expand、低 CPS、高 return stall、高 planner divergence 共同出现**

这意味着：

1. 当前模型不适合作为“健康最终策略”的直接延长线  
2. 当前 reward 结构很可能需要系统性复盘  
3. 当前问题不只在课程或 preload，而可能已经涉及 reward 对长期行为的导向是否合理  

因此，本报告希望专家重点回答：

- 是否应该对 reward 函数的构成做全局系统性微调
- 是否应该在此基础上从头训练一个真正以：
  - 高存活率
  - 高 CPS
  - 预先规划与规律清扫
为目标的新模型

