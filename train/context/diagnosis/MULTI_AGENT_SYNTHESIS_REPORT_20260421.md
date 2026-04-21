# 多子代理综合诊断报告 — 2026-04-21

> 目的：基于整夜长时间训练结果，汇总三路独立子代理分析、当前运行态证据、既有改造背景与现有规划，形成一份统一的主诊断报告，作为下一轮训练修正与实验设计的主参考。
>
> 纳入的独立分析：
> - `Kuhn` — `gpt-5.4 high`
> - `Anscombe` — `gpt-5.2 high`，作为原计划 `gpt-5.1-codex-max` 的替代
> - `Descartes` — `gpt-5.3-codex high`，精简上下文重试版
>
> 不纳入：
> - `Linnaeus` — 因上下文窗口溢出失败
> - `McClintock` — 因模型不可用失败

## 1. 背景与前情

当前分支是：

- `linux-LTSPPO-charge-constraint`

这不是原始基线仓库，而是已经做过一整轮系统性改造后的训练分支。当前主线中已经落地的关键改动包括：

1. `scratch / resume` 启动协议隔离与 `runtime_state` 新布局
2. stop sentinel、`model_file_sync`、run-session 初始化竞态等工程问题修复
3. 充电 / 回充 / route-phase 相关 reward 与 episode 统计重构
4. `route_phase_planner_divergence_rate`、`reliable_planner_divergence_rate`、`route_phase_return_stall_rate` 等新指标链
5. curriculum / checkpoint gate 开始引入 route-phase / reliable 指标
6. `v1-lite` 行为闭环修复尝试
7. 第一阶段 `s1_survival` 的 phase 化启动入口

本次综合报告之前的主背景文档：

- [UNIFIED_PROBLEM_DIAGNOSIS_REPORT_20260420.md](./UNIFIED_PROBLEM_DIAGNOSIS_REPORT_20260420.md)
- [DEEP_PROBLEM_DIAGNOSIS_REPORT_20260420.md](./DEEP_PROBLEM_DIAGNOSIS_REPORT_20260420.md)
- [IMPLEMENTATION_PROGRESS_AND_NEXT_STEPS_20260420.md](./IMPLEMENTATION_PROGRESS_AND_NEXT_STEPS_20260420.md)
- [VNEXT_LITE_BEHAVIOR_SIMPLIFICATION_PLAN_20260420.md](../optimization/VNEXT_LITE_BEHAVIOR_SIMPLIFICATION_PLAN_20260420.md)
- [MULTI_STAGE_TRAINING_ADJUSTMENT_PLAN_20260420.md](../optimization/MULTI_STAGE_TRAINING_ADJUSTMENT_PLAN_20260420.md)

另外，08:20 左右的参考模型已单独归档为：

- [reference-phase1-20260421-step272500/README.txt](../../../code/saved_models/reference-phase1-20260421-step272500/README.txt)

它的定位是：

- `experimental_reference`
- 用于后续课程继续训练的效果对照
- **不是**主线 `resume-main` / `preload-main`

## 2. 当前训练现态

当前主参考运行态文件：

- [curriculum_state.json](../../../code/runtime_state/current/curriculum_state.json)

截至本次综合报告撰写时，当前窗口的核心事实是：

- `stage = warmup`
- `curriculum_stagnation_level = 3`
- `curriculum_stagnation_reason = ["charge", "reward"]`
- `degraded_mainline = true`
- `curriculum_profile_weights = anchor 0.6 / mild 0.3 / broad 0.1`
- `global_episode_count = 120`
- `global_step_since_resume = 430606`

### 当前 `bootstrap 20`

- `win_rate = 0.70`
- `battery_fail_rate = 0.30`
- `zero_charge_battery_fail_rate = 0.6667`
- `battery_positive_reward_rate = 1.0`
- `avg_clean_per_step = 0.8155`
- `mode_usage_contract = 0.0101`
- `mode_usage_expand = 0.0222`
- `mode_usage_harvest = 0.7349`
- `planner_policy_divergence_rate = 0.8486`
- `route_phase_planner_divergence_rate = 0.6303`
- `reliable_planner_divergence_rate = 0.0`
- `route_phase_return_stall_rate = 0.3754`

### 当前 `global 40`

- `win_rate = 0.625`
- `battery_fail_rate = 0.275`
- `zero_charge_battery_fail_rate = 0.6364`
- `battery_positive_reward_rate = 1.0`
- `avg_clean_per_step = 0.7721`
- `mode_usage_contract = 0.0076`
- `mode_usage_expand = 0.0382`
- `mode_usage_harvest = 0.7187`
- `planner_policy_divergence_rate = 0.8588`
- `route_phase_planner_divergence_rate = 0.5926`
- `reliable_planner_divergence_rate = 0.0`
- `route_phase_return_stall_rate = 0.3361`

### 当前训练是否“已经收敛”

从训练步数和表现形态看，这轮训练已经足够长，当前结果已经可以被视为一种**稳定局部最优**，而不再适合用“样本还不够”解释。

这不是“链路没跑起来”，而是：

- 清扫效率已经较高
- 完成率不差
- 但电量失败轨迹仍然常常是正收益
- `contract` 几乎消失
- raw / route-phase planner 偏离长期高位

换句话说，当前训练已经不是“未成熟”，而是**成熟地学歪了**。

## 3. 三路子代理结论对照

### 3.1 共识结论

三路独立分析高度一致地支持以下结论：

1. **当前主问题不是表达能力不足，也不是 backbone 太弱。**
2. **最危险的现象是 `battery_positive_reward_rate` 极高。**
3. **当前已经形成“高CPS / 高清扫 / 高planner偏离 / 低contract”的坏局部最优。**
4. **raw `planner_policy_divergence_rate` 不应再被视为唯一主矛盾或主门槛。**
5. **reward/objective 与动作控制路径之间存在结构性错配。**
6. **`contract` 的极低占比更像 gate / heuristic 控制问题，而不是 mode head 自然学出的分布。**
7. **不应优先动 backbone 或大规模 architecture 重构。**
8. **下一轮最优先修的应是目标函数、route-phase 动作闭环与相关监控，而不是继续原样长训。**

### 3.2 分歧结论

三路的主要分歧不在“诊断方向”，而在“第二优先级修法”：

- `Kuhn` 更强调：
  - 先修 `objective mismatch`
  - 再修 route-phase 动作级监督与监控
  - `s1_survival` 只能做短期止损，不宜长期 phase lock

- `Anscombe` 更强调：
  - planner 建议被“强统计、弱训练纠偏”
  - `CLEANING_RETURN_SCALE = 0.25` 让 contract/return 变成奖励洼地
  - 需要扩大 route-phase action teacher 覆盖并提升 planner 对齐奖励量级

- `Descartes` 更强调：
  - 当前系统存在“heuristic runtime control 与 learned heads 双轨错位”
  - 应尽快明确单一主控面
  - 中期应考虑把“模型主控、heuristic 仅紧急兜底”作为演进方向

### 3.3 我的核查结论

核查后我认为：

- `Kuhn` 的结论与当前运行态最一致，优先级判断最稳
- `Anscombe` 补充了非常关键的一点：
  - `CLEANING_RETURN_SCALE = 0.25` 使 contract/return 阶段的短期梯度天然吃亏
- `Descartes` 对“heuristic 与 learned control 双轨错位”的判断成立，但“切到模型主控”应列为**中期方向**，不适合作为下一轮最小变更起点

## 4. 证据核查与我的独立分析

下面按“明确支持 / 合理推断 / 待验证”分层。

### 4.1 明确支持

#### A. `battery_positive_reward_rate` 是当前最危险的硬证据

当前 `bootstrap 20` 和 `global 40` 的：

- `battery_positive_reward_rate = 1.0`

这意味着：

> 当前窗口内，所有 battery fail 都仍然是正有效收益轨迹。

这条证据足以说明：  
当前目标函数**没有把失败学成真正的失败**。

#### B. `contract` 几乎消失，而 `harvest` 占比异常高

当前：

- `mode_usage_contract ≈ 0.008 ~ 0.010`
- `mode_usage_harvest ≈ 0.72 ~ 0.73`
- `mode_usage_expand ≈ 0.02 ~ 0.04`

这说明当前运行时分布更像：

- 大量时间留在高清扫收益区
- 很少进入预备收缩缓冲态
- 风险上来后直接压给 `return` 或电量失败

#### C. raw planner divergence 长期高位，但不是最强因果指标

当前：

- `planner_policy_divergence_rate ≈ 0.85`
- `route_phase_planner_divergence_rate ≈ 0.59 ~ 0.63`
- `reliable_planner_divergence_rate = 0.0`

这说明：

- raw divergence 确实很高
- 但它和有效训练信号、可靠上下文覆盖并不一一对应

因此它仍然是问题信号，但不能再充当唯一主门槛。

#### D. 当前训练奖励的主导项仍然是清扫 / CPS / streak

当前 reward share 里，正项主导仍是：

- `cleaning`
- `cps_bonus`
- `streak`

同时负项里：

- `skip_needed_charge_penalty`
- `planner_alignment`
- `high_need_return_stall_penalty`

都还不足以压过前者。  
这与三路代理“当前奖励经济学仍在鼓励高CPS晚失败”的判断一致。

### 4.2 合理推断

#### A. 当前局部最优的本质

我认同 `Kuhn` 的表述：当前更接近

- `objective mismatch`
- `action-path mismatch`

而不是“planner 太差”。

更具体地说，当前策略是在学：

> “尽量多清扫、多拿CPS与 streak，晚一些失败也未必亏。”

#### B. `contract` 低并不意味着“模型不喜欢 contract”

当前运行态的 `mode_usage_*` 统计主要来自 runtime heuristic `current_mode`。  
因此极低 `contract` 首先说明：

- `_infer_mode()` 的门和 reward economics 共同把样本分布挤出了 `contract`

而不是：

- mode head 自发学出“讨厌 contract”

#### C. 当前 direct guidance 还没有形成真正可见的主闭环

虽然 `route_phase_policy_teacher_loss` 已经接进主 policy loss，  
但当前主监控和阶段判断仍然没有把：

- `route_phase_policy_teacher_loss`
- `route_phase_action_teacher_active_rate`

作为一等指标。  
这使得“loss 已接入”和“行为已改变”之间仍然断裂。

### 4.3 待验证

以下判断方向值得保留，但不应在没有实验的情况下直接定性为事实：

1. 是否应该中期切到“模型主控、heuristic 仅兜底”
2. 是否应该显著放宽 `CLEANING_RETURN_SCALE`
3. `route_phase_reliable_coverage` 的真实稀疏程度
4. raw planner divergence 在 completed / battery / zero-charge battery 三类 episode 中的分布差异是否稳定

## 5. 统一根因排序

综合三路代理、当前代码与运行态，我给出以下统一根因排序：

### 1. 奖励 / 目标函数错配

这是当前头号问题。

核心表现：

- `battery_positive_reward_rate = 1.0`
- battery fail 仍是正收益轨迹
- 清扫 / CPS / streak 的收益总量高于终局失败成本

结论：

> 当前 objective 没有把“失败”塑形成策略上不可接受的结果。

### 2. 动作控制路径与监督路径错配

动作最终由主 `policy_logits` 决定，但：

- planner 建议的奖励量级偏小
- action-level teacher 覆盖偏窄
- `return_action` 仍是 aux head
- 关键 route-phase teacher 指标没有进入主监控与 checkpoint 选择

结论：

> 训练信号并没有充分压在真正的动作主路径上。

### 3. heuristic mode gate 与 reward economics 共同塑造了坏样本分布

当前 `mode_usage_contract` 极低、`harvest` 极高，不是偶发。

这说明：

- gate 逻辑
- reward 结构
- 生存 / 清扫权衡

三者共同把策略推到了：

- 少 contract
- 多 harvest
- 晚 return
- 晚失败

的坏平衡。

### 4. curriculum / checkpoint 仍存在代理指标滞后问题

虽然这条线上已经开始引入 route-phase / reliable 指标，  
但从现态看，系统仍然：

- 太晚承认 reward pathology
- 过度依赖 raw divergence 这类粗代理

结论：

> curriculum 不是主因，但仍在放大问题和延迟止损。

### 5. 可观测性不足，导致行为改造与指标改造脱节

当前最关键的新训练闭环指标，还没有成为训练与 checkpoint 的一等观测对象。  
这让很多改动处于：

- “代码已经接了”
- “日志看不出来”
- “checkpoint 也不用”

的尴尬状态。

## 6. 统一优化路线

### 6.1 推荐路线：先修目标，再修动作路径，再恢复分布

这是本报告的最终推荐路线。

#### 阶段 1：Objective Reset

目标：

- 先把 battery fail，尤其 zero-charge battery fail，彻底打成负样本

验收主指标：

- `battery_positive_reward_rate`
- `zero_charge_battery_fail_rate`
- `battery_fail_rate`
- `avg_clean_per_step`

这一阶段不追求：

- raw planner divergence 立刻大幅下降
- 立刻 promotion

#### 阶段 2：Route-Phase Control Path

目标：

- 让 route-phase 的训练信号真正压到主动作路径

关键动作：

- 强化 route-phase action teacher 覆盖
- 将 `route_phase_policy_teacher_loss`、`route_phase_action_teacher_active_rate` 纳入主监控和 checkpoint 评估
- 使用 route-phase / reliable 指标作为主要行为验收

#### 阶段 3：Reopen CPS / Curriculum

前提：

- 只有阶段 1 和 2 都通过后，才进入这一阶段

目标：

- 在不反弹 fail-positive 的前提下恢复扩张和更高 CPS

这时才允许：

- 放松 contract gate
- 调整 profile weights
- 恢复更积极的课程分布

### 6.2 备选路线：heuristic-first 止血路线

备选路线可以作为短期止血方案：

- 继续强化 `_infer_mode()`
- 必要时增加更强 heuristic override
- 快速压零充电失败与晚失败

但这条路线的主要风险是：

- 更容易固化手工控制器
- 后续更难恢复 exploration 与泛化

因此：

> 它只适合短期止血，不适合作为主线长期方案。

## 7. 禁止动作

当前最不应该做的事：

1. **继续原样长时间硬跑当前训练。**
只要 `battery_positive_reward_rate` 还远高于 `0.20`，继续长训大概率只会把坏局部最优训得更稳。

2. **继续把 raw `planner_policy_divergence_rate` 当成单一主门槛。**
它仍是重要信号，但不是当前最强因果指标。

3. **先动 backbone / 主网络结构 / 大规模 architecture-lite。**
当前没有证据表明瓶颈在表达能力。

4. **通过放宽 warmup / curriculum 门槛来“让系统看起来能晋级”。**
这会把结构性问题带入下一阶段。

5. **只加 teacher 权重、不修 reward economics 和样本分布。**
这会把辅助头训得更好看，但不一定改变主行为。

## 8. 下一轮验证计划

### 8.1 固定基线

后续实验固定以当前：

- [curriculum_state.json](../../../code/runtime_state/current/curriculum_state.json)

作为主基线，必要时同时参考：

- `2026-04-21 11:59` 左右的 resume snapshot
- 08:20 左右参考模型归档

### 8.2 实验优先级

下一轮最小可验证实验应是：

1. fresh scratch run
2. 只动 `objective reset` 与 route-phase 关键监控
3. 不同时叠大批 gate / curriculum / architecture 改动

### 8.3 成功标准

下一轮实验至少要看到：

- `battery_positive_reward_rate` 显著下降
- `zero_charge_battery_fail_rate` 显著下降
- `battery_fail_rate` 不反弹
- `avg_clean_per_step` 不崩
- `route_phase_return_stall_rate` 不恶化

只有在这些成立后，才值得继续推进更强 planner / curriculum / CPS 恢复方案。

## 9. 最终结论

一句话总结：

> 当前整夜长训暴露的主问题，不是“模型不会清扫”，也不是“planner 指标不够漂亮”，而是训练系统仍在奖励一种“高CPS、高清扫、低contract、晚失败也不亏”的坏局部最优；因此下一轮最优先修的不是 backbone，而是 reward/objective、route-phase 主动作闭环以及与之对应的主监控与阶段判断。

当前主线建议是：

- **停止原样长训**
- **以“先修目标，再修动作路径，再恢复分布”的路线进入下一轮实验**
- **把 08:20 左右的参考模型只作为实验锚点，而不是主线 resume 基线**

