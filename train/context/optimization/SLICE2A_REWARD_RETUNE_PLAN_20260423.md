# Slice 2A Reward Retune Plan 2026-04-23

## Summary

当前 `s1_survival_strong_heuristic_slice2a_v1` 的早期 live 数据说明：

- `avg_reward_risk_growth_while_clean_penalty` 持续为 `0.0`
- `avg_reward_risk_release_reward` 仅在 `1e-4 ~ 1e-3`
- `avg_reward_charge_opportunity_cost_penalty` 仅在 `-1e-4 ~ -4e-4`
- `route_phase_return_stall_rate` 常在 `0.28 ~ 0.55`
- `mode_usage_contract + mode_usage_return` 不低，约 `0.24 ~ 0.46`
- battery fail 已出现，但失败局的 `REWARD_TOP` 仍主要由旧 shadow charging terrain 提供解释

这说明当前 Slice 2A 的新 reward 主链有两个核心问题：

1. `risk_growth_while_clean_penalty` 的作用域过窄，未覆盖真实主失败阶段
2. 新主链整体强度偏弱，当前更像“诊断性残差”而不是“行为塑形项”

本回调方案的目标不是重回旧 charging terrain，而是在保持短链路、强归因的前提下，让：

- “继续贪扫导致风险恶化” 能真的被罚到
- “已经进入预回充/回充但风险仍恶化” 也被主链直接惩罚
- 正向项仍然只奖励“风险真的下降”

## Current Failure Attribution

### 1. `risk_growth_while_clean_penalty` 没打中主失败段

当前实现只在：

- 非 `MODE_CONTRACT`
- 非 `MODE_RETURN`
- 非 `MODE_EVADE`
- `not just_charged`
- 且 `charge_need_score > prev_charge_need_score`

时触发。

但 strong heuristic 已经会较早把策略切入：

- `PRE_RETURN -> MODE_CONTRACT`
- `RETURN -> MODE_RETURN`

因此很多真实失败并不是发生在“仍处于 clean 的贪扫阶段”，而是发生在：

- 已经进入 `PRE_RETURN/RETURN`
- 但 `progress` 不好
- `slack` 继续恶化
- `charge_need_score` 继续上升

所以 clean-only penalty 当前长期为 0，不应解释为“行为没有晚充电问题”，而应解释为：

> 当前 reward 口径没有覆盖真实主失败段。

### 2. 新 reward 主项整体存在感偏弱

当前 scale：

- `KAIWU_SLICE2_RISK_RELEASE_SCALE=0.28`
- `KAIWU_SLICE2_RISK_GROWTH_CLEAN_PENALTY_SCALE=0.22`
- `KAIWU_SLICE2_CHARGE_OPPORTUNITY_COST_PENALTY_SCALE=0.18`

而实际 episode 均值里：

- `risk_release_reward` 约 `1e-4 ~ 1e-3`
- `risk_growth_while_clean_penalty` 为 `0`
- `charge_opportunity_cost_penalty` 约 `-1e-4 ~ -4e-4`

这说明当前主链仍然偏弱，至少在 early window 中尚不足以成为主导行为塑形信号。

## Retune Principles

1. 不恢复旧的多 charging terrain 主链
2. 不让一条 clean penalty 承担全部晚充电归因
3. 对 reward 的修改必须仍保持：
   - 语义短
   - 容易解释
   - 能与 shadow diagnostics 对照
4. 新增 reward 项数量最多增加 1 条

## Proposed Retune

### A. 保持 Slice 2A 仍然只有 3 条 runtime reward

回调后的主链仍然固定为：

1. `risk_release_reward`
2. `route_phase_risk_growth_penalty`
3. `charge_opportunity_cost_penalty`

不再新增第 4 条主 reward。

### B. 用 `route_phase_risk_growth_penalty` 直接替换当前 clean-only 项

当前项：

- `risk_growth_while_clean_penalty`

问题：

- 它把主失败错误地定位在 clean 阶段
- 但当前 live 证据显示：
  - `mode_usage_contract + mode_usage_return` 不低
  - `route_phase_return_stall_rate` 高
  - battery fail 主要发生在 route phase 已经启动之后

因此本回调不再继续扩大 clean penalty 的适用范围，而是：

- 将当前主负项直接改为：
  - `route_phase_risk_growth_penalty`

触发条件：

- `current_mode in {MODE_CONTRACT, MODE_RETURN}`
- `return_context_reliable == true`
- `not just_charged`
- `charge_need_score > prev_charge_need_score`

第一轮公式刻意保持单一，不引入额外复合 shaping：

`route_phase_risk_growth_penalty = - w_route * positive(charge_need_score - prev_charge_need_score) * (0.65 + 0.35 * urgency)`

建议初值：

- `w_route = 0.34`
- 合理调参区间：
  - `0.32 ~ 0.38`

明确不在第一轮加入：

- `progress <= 0` 的附加乘子
- `slack_worsening`
- `charger_distance_worsening`

原因：

- 当前第一优先是把主失败段直接打中
- 而不是重新做回旧 `high_need_return_stall_penalty` 的复杂度

### C. 原 clean penalty 退回 shadow diagnostics

当前 `risk_growth_while_clean_penalty` 不再作为 runtime 主 reward。

它保留为：

- `risk_growth_while_clean_shadow_mean`
- `clean_phase_need_growth_active_rate`

仅做诊断用途，不参与 reward 累加。

这样可以继续验证最初假设：

- 如果 clean shadow 仍接近 `0`
- 但 route-phase penalty 开始非零并与 fail/stall 同步

则说明当前主问题确实不在 clean 阶段，而在 route phase。

### D. 轻微提高 `risk_release_reward`

当前：

- `w_release = 0.28`

建议：

- `0.28 -> 0.31`

原则：

- 不新增 `charge_success_bonus`
- 不再拆过程/事件双重记账
- 只强化“风险真的下降”这条统一正项

### E. 暂不加强 `charge_opportunity_cost_penalty`

当前证据没有显示“过早充电”是主矛盾，因此：

- 保持 `0.18`
- 不在本轮放大

### F. 旧 shadow diagnostics 继续保留，不回流主 reward

保留诊断字段：

- `skip_needed_charge_penalty`
- `high_need_return_stall_penalty`
- `charge_detour_cost`
- `charge_interrupt_cost`

但仍不进入主 reward。

这样可以继续对照：

- 新的 `route_phase_risk_growth_penalty`

是否真的接管了旧诊断项原本解释的失败段。

## Why This Is Better

相比当前设计，这版改动的优点是：

1. 不再错误地让 clean-only penalty 承担所有晚充电问题
2. 不把同一个 `delta_need` 核心信号拆成两条主负项
3. 保持 reward 主链仍然短：
   - `risk_release_reward`
   - `route_phase_risk_growth_penalty`
   - `charge_opportunity_cost_penalty`
4. clean-phase 风险增长与旧 charging terrain 继续留在 shadow diagnostics，归因能力不会丢

## Acceptance Checks

回调后的主验证目标：

- `avg_reward_route_phase_risk_growth_penalty` 在 battery fail / stall run 中应明显非零
- `avg_reward_risk_growth_while_clean_shadow_mean` 可继续为低值，但应保留可观测性
- `route_phase_return_stall_rate` 下降
- `battery_fail_rate` 和 `zero_charge_battery_fail_rate` 下降
- `avg_clean_per_step` 不因惩罚上调而大幅下滑

若出现以下现象，则回调失败：

- `avg_reward_route_phase_risk_growth_penalty` 明显升高，但 `battery_fail_rate` 与 `route_phase_return_stall_rate` 都不降
  - 说明 `charge_need_score` 或 `return_context_reliable` 的口径本身有问题
- survival 改善但 `avg_clean_per_step` 明显塌陷
  - 说明 `w_route` 过强
