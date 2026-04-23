# Charge Need And Return Context Retune 2026-04-23

## Summary

当前 `slice2a` 的 live 数据说明：

- `route_phase_risk_growth_penalty` 已经开始非零
- 但亮得仍然偏弱
- `route_phase_return_stall_rate` 仍高
- `avg_charge_need_score` 偏低

前一版方案的问题在于：它试图直接重定义 `charge_need_score`，这会把当前的 slice2a route-phase 对齐问题扩大成整条生存成本主链重写。

本修订版只做更窄、更稳的事情：

1. `charge_need_score` 继续代表“生存/充电必要性”
2. `battery_process_cost` 主链不大动
3. 新增一个 **slice2a 专用** 的 `route_phase_shadow_risk`
4. 把 `return_context_reliable` 收窄成 `route_phase_reward_ready`
5. 只让这两个新信号影响：
   - `risk_release_reward`
   - `route_phase_risk_growth_penalty`

## Root Cause Reframed

当前的主要问题不是：

- `charge_need_score` 作为全局 survival urgency 完全错误

而是：

- 它对于 slice2a 当前关心的 route-phase 失败解释来说，太保守、太晚
- 但如果直接重写它，就会连带改动：
  - `battery_state`
  - `battery_process_cost`
  - 生存惩罚主链

这不是当前最小、最稳的主线。

当前最需要修正的是：

> slice2a 的 route-phase reward gate 偏晚，而不是整个 survival score 必须重写。

## Design Principles

1. 不把 slice2a shadow 对齐问题扩大成 survival 主链重写
2. `charge_need_score` 与 `battery_process_cost` 继续保留现有主语义
3. slice2a 单独新增 route-phase shadow 风险量
4. 让 route-phase reward 由 “route 风险激活 + route 上下文可信” 决定
5. 尽量只改 `preprocessor.py`，降低实现面

## Proposed Mainline

### A. `charge_need_score` 只做小修，不做重写

#### A1. 保留主语义

`charge_need_score` 继续只表示：

- survival urgency
- 不是 slice2a 的 route-phase shadow 风险代理

因此：

- 不引入新的 `route_available`
- 不改 `battery_state` 阈值结构
- 不重写 unknown-route / known-route 分支拓扑

#### A2. 允许的唯一小修

只允许一处非常受控的小修：

- known-route 分支给 recoverability 一个轻微提前项

建议：

- 保持当前 `margin_term / battery_term / recoverability_term` 结构
- 仅把 recoverability 从：
  - `clip01(-future_recoverability_score)`
  调成：
  - 对 `0.0 ~ 0.35` 区间略提前感知

但要求：

- 这项单独不能把 `safe` 直接推成 `critical`
- 只作为弱修正，不改 score 的主导逻辑

如果做不到这一点，则本轮连这条小修也不做。

### B. 新增 `route_phase_shadow_risk`

这是本轮真正的核心新增量。

它只服务 slice2a route-phase reward，不进入：

- `battery_state`
- `battery_process_cost`
- global survival 主链

#### B1. 输入

`route_phase_shadow_risk` 由以下量构成：

- `min_recoverability = min(future_recoverability_score, planner_multi_route_recoverability)`
- `charger_slack`
- `charge_margin_now`
- `unknown_target_ratio`
- `route_contract_pressure`

#### B2. 语义

它表达的是：

> 在 route-phase 已经启动或即将启动时，
> 当前 route 质量是否仍在朝“更难回充”方向恶化。

这与 `charge_need_score` 不同：

- `charge_need_score` 是 survival urgency
- `route_phase_shadow_risk` 是 route-phase reward activation / worsening signal

#### B3. 建议公式

保持 max-merge，便于解释：

- `recoverability_warn_term`
- `slack_warn_term`
- `margin_warn_term`
- `unknown_route_term`
- `route_pressure_term`

建议：

`route_phase_shadow_risk = max(`
`  1.00 * recoverability_warn_term,`
`  0.90 * slack_warn_term,`
`  0.75 * margin_warn_term,`
`  0.55 * unknown_route_term,`
`  0.65 * route_pressure_term`
`)`

其中每一项都必须写死归一化公式：

- `recoverability_warn_term`
  - `clip01((RECOVERABILITY_WARN - min_recoverability) / RECOVERABILITY_SPAN)`
  - 默认：
    - `RECOVERABILITY_WARN = 0.35`
    - `RECOVERABILITY_SPAN = 0.70`
- `slack_warn_term`
  - `clip01((PREPARE_RETURN_SLACK_THRESHOLD - charger_slack) / max(PREPARE_RETURN_SLACK_THRESHOLD, 1.0))`
- `margin_warn_term`
  - `clip01((CHARGE_MARGIN_WARN - charge_margin_now) / max(CHARGE_MARGIN_WARN, 1.0))`
- `no_reachable_route_term`
  - `1.0 if planner_topk_reachable_count <= 0 else 0.0`
- `unknown_path_term`
  - `unknown_target_ratio`
  - 仅当存在 path 时解释为“路径未知比例”
- `unknown_route_term`
  - `max(no_reachable_route_term, unknown_path_term)`
- `route_pressure_term`
  - 直接复用 `route_contract_pressure`

并固定：

- `route_phase_shadow_risk_threshold = 0.12`

理由：

- 该量既要做 gate 又要做 delta
- 因此 span 和 threshold 不能留给实现时自由发挥

### C. 把 `return_context_reliable` 改成 `route_phase_reward_ready`

当前问题不在“route existence 不够”，而在：

- reward 触发被 `battery_state in {planning, critical}` 卡得过晚

因此本轮不重做 route existence，而是新增一个更贴切的 gate：

`route_phase_reward_ready`

#### C1. 触发条件

`route_phase_reward_ready = (`
`  current_mode in {MODE_CONTRACT, MODE_RETURN}`
`  and route_context_available`
`  and route_phase_shadow_risk >= route_phase_shadow_risk_threshold`
`)`

其中：

`route_context_available = (`
`  route_phase_reliable_active`
`  or return_action_reliable`
`  or anchor_reliable`
`)`

优先使用当前已有但没被消费的：

- `route_phase_reliable_active`

再用：

- `return_action_reliable`
- `anchor_reliable`

兜底。

#### C2. 为什么不用 `battery_state`

因为当前 `battery_state` 仍来自偏保守的 `charge_need_score`。  
本轮要解决的是：

- slice2a route-phase reward 激活偏晚

所以不能继续用它做主门。

### D. reward 接线方式

#### D1. `risk_release_reward`

这条 reward 必须明确拆成两条**监控上可见、训练上合并**的通路：

1. `just_charged` 事件通路
   - 继续使用：
     - `positive(prev_charge_need_score - charge_need_score)`
   - 原因：
     - 这是 survival urgency 真实下降

2. route-phase progress 通路
   - 改用：
     - `positive(prev_route_phase_shadow_risk - route_phase_shadow_risk)`
   - 原因：
     - 这是 slice2a 真正想修的 route-phase 风险释放

激活条件：

- route-phase progress 通路使用：
  - `route_phase_reward_ready`

最终 runtime reward 仍只有一条：

- `risk_release_reward = risk_release_from_charge_need + risk_release_from_route_shadow`

但监控必须拆成：

- `risk_release_from_charge_need_mean`
- `risk_release_from_route_shadow_mean`

#### D2. `route_phase_risk_growth_penalty`

从：

- `positive(charge_need_score - prev_charge_need_score)`

改为：

- `positive(route_phase_shadow_risk - prev_route_phase_shadow_risk)`

激活条件：

- `route_phase_reward_ready`

保持当前主结构：

`route_phase_risk_growth_penalty = - w_route * positive(delta_route_shadow_risk) * (0.65 + 0.35 * urgency_like)`

其中：

- `urgency_like` 优先继续使用当前 `need_term/urgency`
- 本轮不再新增新的 stall multiplier

### E. 诊断要求

新增监控：

- `avg_route_phase_shadow_risk`
- `avg_route_phase_reward_ready_rate`
- `avg_route_phase_shadow_risk_delta_positive`

保留现有：

- `avg_charge_need_score`
- `avg_reward_route_phase_risk_growth_penalty`
- `avg_reward_risk_release_reward`
- `avg_reward_risk_growth_while_clean_penalty`

这样可以直接回答：

- 是 survival urgency 太低？
- 还是 slice2a route reward gate 太晚？

## Why This Is Better

相比重写 `charge_need_score` 主链，这版更优，因为：

1. 不会顺手放大 `battery_process_cost`
2. 不会改坏 `battery_state` 的全局语义
3. 直接命中 slice2a 当前真正的问题：
   - route-phase reward 激活偏晚
4. 与当前 strong heuristic 架构兼容：
   - mode helper 不动
   - expert planner 不动
   - 只在 `preprocessor.py` 里补 route-phase shadow gate

## Implementation Scope

本轮优先改：

- `preprocessor.py`
- 监控 / compare / tests

本轮尽量不改：

- `constraint_utils.py`

除非需要做前面 A2 提到的 recoverability 小修，而且该小修必须非常受控。

## Acceptance Checks

1. `avg_reward_route_phase_risk_growth_penalty` 比当前更稳定非零
2. `avg_reward_risk_release_reward` 不再过度稀疏
3. `route_phase_return_stall_rate` 下降
4. `battery_fail_rate / zero_charge_battery_fail_rate` 下降
5. `avg_clean_per_step` 不因为 gate 提前而明显塌陷

若出现：

- `route_phase_reward_ready_rate` 大幅抬升，但 `route_phase_return_stall_rate` 不降
  - 说明问题不在 gate，而在 planner/progress 质量
- `avg_charge_need_score` 基本不变，但 route-phase reward 明显更有存在感
  - 这是预期中的正确结果，不是问题
