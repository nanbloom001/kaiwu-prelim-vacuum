# vNext-Lite 行为闭环简化方案 V1

> 日期: 2026-04-20  
> 目的: 在不回退到旧简单模型、也不继续原样硬跑当前复杂体系的前提下，优先修复 `planner 偏离 + return stall + 首充闭环不稳`，形成一个可直接实施的 `v1-lite` 路线。

## 1. 更新后的主判断

这版方案吸收独立审查后的核心修正如下：

1. 当前最值得先动的主杠杆不是“先削弱 mode / route / target teacher”，而是：
   - `contract` 进入逻辑
   - route-phase 的直接动作约束
   - curriculum / checkpoint 的 metric gate
2. 当前 `mode_usage_contract` 偏高，**更直接的上游原因是 `_infer_mode()` 的 heuristic gate 太宽**，不是 learned `mode` head 本身主导了行为分布。
3. 当前 `return_action` 虽然方向对，但它还是 aux head，**并不直接参与动作采样**；所以“单纯加强 return_action teacher”不能当主方案。
4. `route_anchor / target / mode` 目前仍直接参与 actor 输入，因此不能直接把它们降成“只做日志输出”，否则会削弱 policy 还在依赖的 latent context。

因此，`v1-lite` 的正式目标改成：

> **先修 runtime 行为分布和 route-phase 动作约束，再做有限度 supervision-lite。**

换句话说：

- 第一阶段是 **gate-lite + guidance-lite**
- 不是 **architecture-lite**

## 2. v1-lite 要保留的部分

以下内容继续保留，不纳入第一轮收缩：

- `value_clean + value_survive` 双 value
- 当前 scratch / resume / runtime state 工程主链
- 当前 charge / return reward 主框架
- 当前 planner / stall / battery fail 监控与诊断体系
- 当前多头 actor 结构本身

理由：

- 当前主问题不是 backbone 或双 value 表达不够
- 当前 reward 方向已经比旧版更对
- 现在最缺的是行为闭环打通，而不是继续推翻底层结构

## 3. 第一阶段真正要先改的 4 件事

### 3.1 第一主杠杆：重写 `contract` 进入逻辑

这是 `v1-lite` 的第一优先级。

当前问题不是“contract 这个 mode 不该存在”，而是：

- `contract` 的进入条件过宽
- 它会在很多“尚可继续 expand / harvest”的时刻提前吸走行为
- 结果造成：
  - `mode_usage_contract` 长期高
  - `mode_usage_expand` 极低
  - `return` 还没学好时，大量步数已经被锁进保守阶段

#### v1 的具体原则

- 保留 `return` 作为明确回充态
- 保留 `contract` 作为回充预备态
- 但把 `contract` 触发从“风险稍高就进”改成“需要明显进入回充预备”才进

#### v1 建议修改方向

优先收紧以下条件：

- `route_contract_pressure >= 0.5`
- `margin <= CHARGE_MARGIN_WARN`
- `CONTRACT_BATTERY_RATIO`
- `CONTRACT_RECOVERABILITY_THRESHOLD`
- `PREPARE_RETURN_SLACK_THRESHOLD`

原则：

- `return` 的硬门槛尽量不放松
- `contract` 的软门槛适度收紧
- 避免把大量原本应属于 `expand / harvest` 的时间过早吸进 `contract`

#### v1 预期效果

- `mode_usage_contract` 回落
- `mode_usage_expand` 抬升
- CPS 有恢复空间
- 同时不直接牺牲必要 return

### 3.2 第二主杠杆：强化 route-phase 对主动作的直接约束

当前 `return_action` 辅助头本身不是直接动作路径，所以仅调大它的 teacher weight，不足以成为主修复项。

v1 要做的不是“只强化 aux head”，而是：

> 在 route-phase 里，把 planner / teacher 对主动作 logits 的直接约束加强。

#### 具体做法方向

只在以下场景下生效：

- `planning / critical`
- `contract / return`
- route / return guidance reliable

对主动作 head 做更直接的 route-phase guidance，例如：

- 更强的 planner-consistency shaping
- 更直接的 route-phase action preference
- 更明确的 stall-to-progress 行为压力

而不是继续把主要希望寄托在：

- `mode_teacher`
- `route_anchor_teacher`
- `target_teacher`
- `return_action_teacher`

这几层间接中间语义上。

#### 预期效果

- 更直接降低 `planner_policy_divergence`
- 更直接降低 `return_stall_rate`
- 减少“中间语义学到了，动作还是没学会”的路径损失

### 3.3 第三主杠杆：同步重构 metric / curriculum gate

如果只改行为，不改 metric gate，当前系统仍会继续被：

- raw `planner_policy_divergence_rate`
- raw `return_stall_rate`

锁在 warmup / stagnation 里。

#### v1 的判断

当前 raw divergence 口径太粗，不能继续单独作为强锁门项。

#### v1 要补的 metric

新增并优先使用：

- `route_phase_planner_divergence_rate`
- `reliable_planner_divergence_rate`
- `route_phase_return_stall_rate`
- `battery_positive_reward_rate`
- `zero_charge_battery_fail_rate`

#### v1 要改的 gate 方向

- `curriculum_stagnation_reason` 不再由 raw planner divergence 主导
- promotion / degraded mainline 更看：
  - `zero_charge_battery_fail_rate`
  - `battery_positive_reward_rate`
  - `route_phase_return_stall_rate`
  - reliability-gated divergence

#### 预期效果

- 避免行为已改善但课程仍因粗指标卡死
- 避免系统继续因为“全步长 raw mismatch”过度保守化

### 3.4 第四主杠杆：reward 只做小调，不做重构

当前 reward 已经开始起作用，不适合再次大改。

#### v1 保持不变的部分

- `charge_route_progress_bonus`
- `charger_access_discovery_bonus`
- `charger_access_probe_bonus`
- `skip_needed_charge_penalty`
- `high_need_return_stall_penalty`
- `charge_detour_cost`
- `charge_interrupt_cost`

#### v1 只做两类小调

1. 适度增强 route-phase 正向推进收益
   - 优先增强 `return_progress_shaping_bonus`
2. 不放松“该充不充”的惩罚
   - 保持 `skip_needed_charge_penalty`

原则：

- 不再新加一批 reward 项
- 不推翻现有 reward 结构
- 只让 route-phase 的直接推进更占优

## 4. 第一阶段明确不做的事

为了避免误伤主链，以下内容不作为 v1 主方案。

### 4.1 不做 architecture-lite

也就是：

- 不移除 `route_anchor / target / mode` head
- 不把它们从 actor path 中摘掉
- 不修改 backbone 结构

原因：

- 当前还没有证据证明 actor 结构本身是主因
- 贸然摘掉这些上下文，风险高于收益

### 4.2 不把 `return_action` 提升成唯一核心辅助监督

原因：

- 它目前仍是 aux head
- 不在主动作采样路径上
- 单独加大权重不够形成主杠杆

### 4.3 不立即大幅下调 `route_anchor / target / mode` teacher

原因：

- 当前这三路 teacher 虽然可疑，但 actor 仍依赖对应 latent
- 在没有做 controlled ablation 前，不应该直接把它们大幅砍掉

它们后续只作为：

- 第二阶段 supervision-lite 消融项

而不是第一阶段主方案。

## 5. 第二阶段：受控 supervision-lite 消融

只有在第一阶段完成后，才进入这一步。

第二阶段的目标是验证：

> 当前多 teacher 是否真的在制造主要训练冲突。

### 5.1 第二阶段只允许做 loss-lite，不做 architecture-lite

也就是说：

- actor 结构不变
- head 保留
- 只调 loss 权重

### 5.2 消融顺序

建议严格按这个顺序：

1. `contract gate only`
2. `contract gate + route-phase direct guidance`
3. `contract gate + route-phase direct guidance + gate/metric redesign`
4. 在上述基础上，再做：
   - `ROUTE_ANCHOR_TEACHER_WEIGHT` 下调
   - `TARGET_TEACHER_WEIGHT` 下调
   - `MODE_TEACHER_WEIGHT` 轻度下调

### 5.3 第二阶段判断标准

如果在第一阶段后：

- `planner_policy_divergence`
- `return_stall`
- `zero_charge_battery_fail_rate`

已经明显改善，就不需要急着大幅简化 supervision。

## 6. v1-lite 的验收指标

这版方案的主验收不看总 win rate，而看结构指标。

### 主验收指标

1. `zero_charge_battery_fail_rate`
2. `battery_fail_rate`
3. `planner_policy_divergence_rate`
4. `route_phase_planner_divergence_rate`
5. `return_stall_rate`
6. `route_phase_return_stall_rate`
7. `mode_usage_expand`
8. `mode_usage_contract`
9. `avg_clean_per_step`
10. `battery_positive_reward_rate`

### 期望方向

- `zero_charge_battery_fail_rate` 明显下降
- `battery_fail_rate` 下降或至少不恶化
- `planner_policy_divergence_rate` 下行
- `return_stall_rate` 下行
- `mode_usage_expand` 回升
- `mode_usage_contract` 回落
- `avg_clean_per_step` 不崩
- `battery_positive_reward_rate` 继续下降

## 7. 需要在进入实现前定死的 Top 5 决策

这 5 个决定现在必须明确，才适合进入实现。

### 决策 1：`contract` gate 具体改哪些阈值、改多少

必须明确：

- 哪些 threshold 收紧
- 收紧多少
- 哪些条件保持不变

### 决策 2：route-phase direct guidance 的实现方式

必须明确：

- 是 planner-consistency reward 增强
- 还是对主动作 logits 做蒸馏/约束
- 还是 legal-action / route-phase mask 辅助

### 决策 3：新的 reliability-gated metrics 口径

必须明确：

- `route_phase_planner_divergence_rate` 怎么算
- `reliable_planner_divergence_rate` 怎么算
- `route_phase_return_stall_rate` 怎么算

### 决策 4：curriculum / checkpoint gate 的切换规则

必须明确：

- 哪些 gate 从 raw divergence 切走
- 切到哪些新指标
- 旧指标保留为诊断还是辅助约束

### 决策 5：第二阶段 supervision-lite 的回退条件

必须明确：

- 什么情况下才允许动 teacher weights
- 什么情况下立即回退

## 8. 一句话版本

> `v1-lite` 的第一主线不是“先砍多头、多 teacher”，而是**先重写 `contract` gate，增强 route-phase 对主动作的直接约束，并同步把 curriculum / checkpoint gate 从粗糙 raw planner-stall 指标改成更因果的 route-phase 指标**；只有第一阶段验证过后，才进入 supervision-lite 消融。

