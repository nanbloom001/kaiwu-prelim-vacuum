# 训练修正行动方案 V2

> 日期: 2026-04-21  
> 目标: 用 2 到 3 次受控改动，把当前模型从“高 CPS、高清扫、但行为结构错误”的局部最优拉回到正确方向。  
> 原则: 不追求一步到位；每次只动一类主杠杆；每次都必须做明确验收。

## 1. 当前主判断

当前最关键的问题优先级如下：

### P0

1. `battery_positive_reward_rate` 极高，说明模型仍在学“失败也赚钱”
2. route-phase 的动作级约束不够强，planner 建议没有真正进入主行为闭环
3. 当前训练已经形成“高 CPS / 高清扫 / 高 planner 偏离 / 低 contract”的稳定坏局部最优

### P1

1. `contract` 过低、`harvest` 过高，样本分布与行为阶段失衡
2. curriculum / checkpoint 对粗代理指标仍有依赖
3. 关键 route-phase 训练信号可观测性不足

因此当前主线不应继续原样长跑，而应按下面顺序推进：

> **先修 objective，再修 route-phase 主动作闭环，最后修样本分布与课程门控。**

## 2. 总体执行规则

每一轮改动都必须满足以下规则：

1. 新开 **scratch run**
2. 至少跑到：
   - `bootstrap 20`
   - `global 40`
3. 每轮只改一类主杠杆
4. 不允许把下一轮改动叠加到当前轮一起做
5. 每轮都必须对比上一轮最佳结果

每轮都必须记录以下固定指标：

- `win_rate`
- `battery_fail_rate`
- `zero_charge_battery_fail_rate`
- `battery_positive_reward_rate`
- `avg_clean_per_step`
- `mode_usage_contract`
- `mode_usage_expand`
- `mode_usage_harvest`
- `planner_policy_divergence_rate`
- `route_phase_planner_divergence_rate`
- `reliable_planner_divergence_rate`
- `route_phase_return_stall_rate`
- `route_phase_action_teacher_active_rate`
- `route_phase_policy_teacher_loss`

## 3. 改动 1：Objective Reset

### 目的

先打掉当前最危险的坏局部最优：

> **高 CPS / 高清扫 / 晚失败仍然正收益**

如果这一步不做，后续再怎么修 `contract`、planner、teacher，都会被当前 reward economics 抵消。

### 具体改动

1. 提高 battery fail 终局代价
   - 提高 `BATTERY_TERMINAL_COST_SCALE`
   - 保持 `zero-charge battery fail` 比普通 battery fail 更重
   - 必要时增加固定 episode 级 battery fail penalty

2. 收紧 battery fail 的有效奖励缩放
   - 重新检查 `task_reward_scale` 在 battery fail / zero-charge fail 下的取值
   - 目标是让 battery fail 的 `effective_total_reward` **大多数为负**

3. 暂不新增大量 reward 项
   - 不继续叠加新 shaping
   - 避免一次改太多无法归因

4. 保持其它主链不动
   - 不改 backbone
   - 不改 teacher 体系
   - 不改大结构
   - 不重做 curriculum

### 本轮不做

- 不先调 `contract gate`
- 不先调 planner teacher weight
- 不先做 supervision-lite
- 不先放宽课程门槛

### 成功标准

用 `bootstrap 20` 和 `global 40` 两个窗口验收。

#### 必须全部满足

- `battery_positive_reward_rate <= 0.20`
- `zero_charge_battery_fail_rate <= 0.40`
- `battery_fail_rate <= 0.25`
- `avg_clean_per_step >= baseline * 0.92`

#### 理想标准

- `battery_positive_reward_rate <= 0.10`
- `zero_charge_battery_fail_rate <= 0.30`
- `battery_fail_rate <= 0.20`
- `avg_clean_per_step >= baseline * 0.95`

### 失败标准

出现任一情况即视为失败：

- `battery_positive_reward_rate > 0.35`
- `zero_charge_battery_fail_rate` 无明显下降
- `avg_clean_per_step < baseline * 0.88`
- `win_rate` 崩塌且 `battery_positive_reward_rate` 仍高

### 失败后的回退

- 不进入改动 2
- 继续只调：
  - battery terminal cost
  - battery fail task reward scale
  - zero-charge fail 终局处理
- 不叠加 planner / gate 改动

## 4. 改动 2：Route-Phase 主动作闭环

### 目的

在“失败真正亏钱”之后，开始修：

> **planner / route-phase 的训练信号没有真正压到主动作路径上**

当前真正决定行为的是主 `policy_logits`，不是 `return_action` aux head。

### 具体改动

1. 强化 `route_phase_policy_teacher_loss` 的训练地位
   - 确保进入 learner 主监控
   - 记录并输出：
     - `route_phase_policy_teacher_loss`
     - `route_phase_action_teacher_active_rate`

2. 扩大 route-phase teacher 覆盖
   - 不再只依赖过窄的 reliable mask
   - 改成可靠度分层：
     - 高可靠：权重 `1.0`
     - 中可靠：权重 `0.3 ~ 0.6`
     - 低可靠：不上 teacher

3. 提升 planner 对齐信号的相对量级
   - 当前 `PLANNER_ALIGNMENT_REWARD / PENALTY` 量级过低
   - 需要提升到能和主任务 reward 竞争
   - 这一轮只调这一个方向，不叠其它 reward 项

4. route-phase 指标转为主验收指标
   - 主看：
     - `route_phase_planner_divergence_rate`
     - `reliable_planner_divergence_rate`
     - `route_phase_return_stall_rate`
   - raw `planner_policy_divergence_rate` 降级为观察指标

### 本轮不做

- 不动 backbone
- 不删 head
- 不做 supervision-lite
- 不大改 `_infer_mode()`

### 成功标准

#### 必须全部满足

- `route_phase_planner_divergence_rate <= 0.45`
- `route_phase_return_stall_rate <= 0.25`
- `route_phase_action_teacher_active_rate >= 0.15`
- `battery_positive_reward_rate` 不高于改动 1 结果 `+0.05`
- `avg_clean_per_step >= 改动1结果 * 0.95`

#### 理想标准

- `route_phase_planner_divergence_rate <= 0.35`
- `reliable_planner_divergence_rate <= 0.25`
- `route_phase_return_stall_rate <= 0.20`
- `route_phase_action_teacher_active_rate >= 0.20`

### 失败标准

- `route_phase_action_teacher_active_rate < 0.10`
- `route_phase_planner_divergence_rate` 无明显下降
- `route_phase_return_stall_rate` 不降反升
- `battery_positive_reward_rate > 0.25`

### 失败后的回退

- 不进入改动 3
- 继续只调：
  - route-phase teacher mask 覆盖
  - route-phase teacher weight
  - planner alignment reward / penalty 量级
- 不叠加新的课程或 gate 改动

## 5. 改动 3：样本分布与课程门控修正

### 目的

只有前两步通过后，才处理：

> **`contract` 太低、`harvest` 太高、课程门槛看错问题**

这一轮的目标是：

- 恢复合理的预备收缩层
- 让课程系统看真正的因果指标
- 为后续恢复更高 CPS 做准备

### 具体改动

1. 重新调整 `contract gate`
   - 不再追求把 `contract` 压到接近 `0`
   - 让它重新承担 `harvest/expand -> return` 的缓冲态
   - 重点调：
     - `PREPARE_RETURN_SLACK_THRESHOLD`
     - `CONTRACT_BATTERY_RATIO`
     - `CONTRACT_RECOVERABILITY_THRESHOLD`
     - `CHARGE_MARGIN_WARN`
     - `CONTRACT_ROUTE_PRESSURE_THRESHOLD`
     - soft trigger hit 规则

2. 重新审视 `CLEANING_RETURN_SCALE`
   - 如果证据继续支持 contract/return 是奖励洼地，就把它从 `0.25` 回调
   - 但只在前两步通过后做，避免归因污染

3. curriculum / checkpoint 改成真正的主验收口径
   - 主门槛使用：
     - `battery_positive_reward_rate`
     - `zero_charge_battery_fail_rate`
     - `route_phase_planner_divergence_rate`
     - `route_phase_return_stall_rate`
   - raw planner divergence 只做辅助观察

4. 允许逐步恢复 profile 暴露
   - 在结构健康的前提下恢复更多 `mild / broad`
   - 为 CPS 恢复做准备

### 本轮不做

- 不重写整个 reward 系统
- 不做 architecture-lite
- 不做大规模 head 移除

### 成功标准

#### 必须全部满足

- `mode_usage_contract` 回到 `0.05 ~ 0.15`
- `mode_usage_expand >= 0.05`
- `battery_positive_reward_rate <= 0.15`
- `route_phase_return_stall_rate <= 0.22`
- `avg_clean_per_step >= 改动2结果`
- `win_rate >= 改动2结果`

#### 理想标准

- `mode_usage_contract` 回到 `0.06 ~ 0.18`
- `mode_usage_expand >= 0.08`
- `battery_fail_rate <= 0.18`
- `zero_charge_battery_fail_rate <= 0.25`
- `avg_clean_per_step >= 0.80` 或显著高于改动 2

### 失败标准

- `contract` 仍接近 `0`
- `expand` 没恢复
- `battery_positive_reward_rate` 重新抬高
- `avg_clean_per_step` 下降但行为结构没有明显改善

### 失败后的回退

- 回退到改动 2 最佳配置
- 暂停继续放宽 profile 或继续调 gate
- 不进入结构层面的精简或大重构

## 6. 决策规则

### 何时进入下一轮

- 当前轮“必须全部满足”标准成立
- 且没有触发失败标准

### 何时停止继续改

- 连续两轮都无法压低 `battery_positive_reward_rate`
- 或 route-phase 覆盖始终无法提升
- 或 reward 调整后清扫效率显著崩塌但行为结构也没改善

### 何时考虑更大改动

只有在这 3 次改动都做过后，仍然出现以下情况，才考虑更大规模调整：

- `battery_positive_reward_rate` 高
- `contract` 极低
- route-phase 指标不改善
- 但训练链和 reward 目标已经明确修过

这时才考虑：

- 更激进的 heuristic / learned control 面统一
- supervision-lite
- 或 architecture 层面的调整

## 7. Assumptions

- 当前最核心的问题是 `objective mismatch + action-path mismatch`，不是 backbone 容量不足。
- 当前 `s1_survival`、route-phase metrics、route-phase teacher 主链已经有工程基础，可以复用。
- raw `planner_policy_divergence_rate` 仍有观察价值，但不应作为唯一主验收指标。
- 这份方案优先追求“每次改动都能验证因果”，而不是一次改很多希望撞对。
