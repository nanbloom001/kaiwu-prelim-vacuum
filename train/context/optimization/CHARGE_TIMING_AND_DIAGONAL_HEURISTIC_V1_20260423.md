# Charge Timing And Diagonal Heuristic v1

## Summary

本方案只解决两个问题：

1. 让模型学到“不要太晚充电，也不要过度充电”
2. 用最小复杂度利用“斜着走”在回充、规避、开图中的几何优势

设计原则固定为：

- 不新增复杂状态机，不把系统重新做成重控制栈
- 不新增超过 3 条充电时机相关 reward
- 不把 diagonal 做成全局 reward
- 充电时机主要靠“连续风险变化 + 事件归因”
- 斜走启发主要靠“path/action soft bias”

## Charge Timing Reward

### Core State

继续保留一个压缩后的 `charge_need_score`，它由以下量组成：

- `charger_slack`
- `future_recoverability_score`
- `battery_ratio`

`charge_need_score` 只负责表达“继续清扫的可恢复性风险”，不直接等价于 reward。

不再引入新的 reward zone 或第二套时机状态机。  
`charge_need_score` 只作为连续风险量，不再在 reward 层切成 `free / need / late` 阈值区间。

### Kept / Added Reward Terms

只保留 3 条与充电时机直接相关的主项。

#### 1. `risk_release_reward`

只在：

- `current_mode in {PRE_RETURN, RETURN}`
- 或 `just_charged == 1`

时生效。

主公式：

`risk_release_reward = w_release * positive(prev_charge_need_score - charge_need_score)`

解释：

- 不再把“过程推进”与“结果充电成功”拆成两条 reward
- 统一只奖励“充电风险真的下降了”

用到的观测：

- `prev_charge_need_score`
- `charge_need_score`
- `current_mode`
- `just_charged`

预期效果：

- 避免 `charger_progress_reward` 与 `charge_success_bonus` 双重记账
- 用单条连续 reward 表达“回充链是否真的让局面变好”

实现后的解释性通过监控来补：

- `risk_release_from_progress_mean`
- `risk_release_from_charge_event_mean`

这两个量只进监控，不作为额外 reward。

#### 2. `risk_growth_while_clean_penalty`

只在：

- 逻辑 `CLEAN`
- 且 `not just_charged`

时生效。

主公式：

`risk_growth_while_clean_penalty = - w_growth * positive(charge_need_score - prev_charge_need_score) * low_task_value_gate`

其中：

- `low_task_value_gate`
  - 当前局部任务价值越低，惩罚越强
  - 当前局部任务价值越高，惩罚越弱，但不为 0

`low_task_value_gate` 使用：

- `local_dirt_density`
- `dirty_adjacent`
- `local_frontier_density`

构造一个简单值，不新增新地图结构。

预期效果：

- 防止“继续清扫让风险持续恶化”
- 同时避免“眼前任务价值很高时被硬拉走”

#### 3. `charge_opportunity_cost_penalty`

只在：

- `just_charged == 1`
- 且 `prev_charge_need_score` 很低

时生效。

主公式：

`charge_opportunity_cost_penalty = - w_early * just_charged * (1 - prev_charge_need_score) * max(prev_charge_detour_proxy, prev_charge_interrupt_proxy)`

用到的观测：

- `just_charged`
- `prev_charge_need_score`
- `prev_charge_detour_proxy`
- `prev_charge_interrupt_proxy`

预期效果：

- 显式打击“明显过早充电”
- 直接表达“为了这次充电绕路/打断清扫”的真实机会成本

### Explicit Non-Goals

本方案不新增：

- 新的 reward-zone 状态机
- 多层 route-phase timing reward
- 基于多 charger family 的 timing 归因
- 额外 teacher / curriculum 门控

## Diagonal Heuristic

### Design Goal

只利用 diagonal 的一个明确收益：

1. `RETURN` 时更快逼近 charger，减少交叉重叠

### Rule A: `RETURN` Diagonal Bias

只在：

- `current_mode == RETURN`
- 且现有 `suggested_action` 本身就是 diagonal

下生效。

不新增独立 diagonal 选边逻辑，只对现有 bias 做倍率修正：

- `RETURN`: `return_bias_scale *= 1.15`

用到的观测：

- 现有 `suggested_action`
- `legal_action`

预期效果：

- 保留 diagonal 在目标导向移动中的几何优势
- 不和现有 planner action selection 打架

### Rule B: `CLEAN` Early-Explore Diagonal Bias

本版默认**不启用**。

原因：

- clean 期更需要 coverage 规律性
- 斜向偏置容易把 clean 轨迹打碎
- 当前阶段更应先验证 `RETURN` 的 diagonal 几何收益
- 当前实现里 `EVADE` 没有现成的“escape suggested_action 正 bias”主链，因此也不纳入 v1

如果后续要做，只能作为单独可选增强：

- 默认关闭
- 单独实验
- 不与切片二主 reward 绑在一起

### Explicit Non-Goals

本方案不新增：

- diagonal reward
- diagonal mode
- diagonal teacher
- 全局强制 diagonal 规则

diagonal 只作为：

- `path/action soft bias`

## Complexity Guardrails

为防止重回复杂控制栈，本方案强制约束：

1. 充电时机新增 reward 固定为 3 条
2. diagonal 启发只允许通过已有 `soft logit bias` 框架接入
3. 不新增新的 teacher 头、mode、sample schema
4. 每一条新增逻辑必须能用一句话解释清楚

## Acceptance Checks

实现后优先检查：

- `battery_fail_rate`
- `zero_charge_battery_fail_rate`
- `late_return_rate`
- `missed_charge_opportunity_rate`
- `mode_usage_return`
- `avg_clean_per_step`
- `route_phase_return_stall_rate`

若 timing 改动后：

- survival 改善但 CPS 明显下滑
  - 说明 `risk_growth_while_clean_penalty` 过强
- survival 仍差且 late return 不降
  - 说明 `charge_need_score` 定义或 `risk_release_reward` 过弱
- 过早充电明显变多
  - 说明 `charge_opportunity_cost_penalty` 过弱
- `RETURN` 轨迹没有改善
  - 说明 diagonal scale modifier 太弱或根本未触发
