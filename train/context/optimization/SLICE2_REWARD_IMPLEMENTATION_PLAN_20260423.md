# Slice 2 Reward Implementation Plan 2026-04-23

## Summary

切片二拆成两个连续微切片：

1. `Slice 2A`
   - 只做 reward contraction
   - 不接 diagonal bias
2. `Slice 2B`
   - 在 `2A` 验证通过后
   - 再单独接 `RETURN` diagonal scale modifier

本轮明确**不做**：

- mode/teacher/sample schema 改动
- 新的 reward-zone 状态机
- `CLEAN` 或 `EVADE` 的 diagonal bias
- 新的 planner family / target reliability 逻辑
- 额外 curriculum / stagnation 控制

切片二的目标不是直接冲分，而是让 reward 更短链路、更易归因，并验证：

- 是否能减少 “顶着风险继续扫”
- 是否能减少 “明显过早充电”
- 是否能让 `RETURN` 路径更少交叉、更少停滞

## Implementation Changes

### A. Slice 2A: Reward 主线收缩

在 strong heuristic phase 下，仅保留以下充电时机相关 reward 主项：

#### 1. `risk_release_reward`

用途：

- 奖励“这一步之后充电风险真的下降了”

公式：

- `risk_release_reward = w_release * positive(prev_charge_need_score - charge_need_score)`

触发：

- `just_charged == 1`
- 或：
  - `current_mode in {MODE_CONTRACT, MODE_RETURN}`
  - 且 `return_context_reliable == true`

实现要求：

- 新增独立 reward component 名称：`risk_release_reward`
- 同时在 step diagnostics 中补两个监控拆解项：
  - `risk_release_from_progress`
  - `risk_release_from_charge_event`
- 这两个量只进监控与 compare，不参与 reward 累加

#### 2. `risk_growth_while_clean_penalty`

用途：

- 惩罚“还在 clean，但充电风险继续变坏”

公式：

- `risk_growth_while_clean_penalty = - w_growth * positive(charge_need_score - prev_charge_need_score) * low_task_value_gate`

其中：

- `low_task_value_gate = 0.35 + 0.65 * (1 - task_value_here)`
- `task_value_here` 沿用当前已有定义：
  - `local_dirt_density`
  - `dirty_adjacent`
  - `local_frontier_density`
  - `new_explored_cells`

触发：

- 逻辑 `CLEAN`
- `not just_charged`

实现要求：

- 逻辑 `CLEAN` 在新 phase 下按 strong heuristic 四态 helper 判断，不新增新的 mode id
- 若当前 phase 不是 strong heuristic v1，则此项恒为 0

#### 3. `charge_opportunity_cost_penalty`

用途：

- 惩罚“明明不太需要充，还为了这次充电付出了明显 detour / interrupt 成本”

公式：

- `charge_opportunity_cost_penalty = - w_early * just_charged * (1 - prev_charge_need_score) * max(prev_charge_detour_proxy, prev_charge_interrupt_proxy)`

触发：

- `just_charged == 1`

实现要求：

- 不再使用 `task_value_here` 来表达过早充电成本
- 必须直接复用当前已有：
  - `prev_charge_detour_proxy`
  - `prev_charge_interrupt_proxy`

### B. Slice 2A: 旧 charging terrain 退为 shadow diagnostics

在 strong heuristic phase 下，以下旧项从 reward 主链中移除，不再累计到 `task_reward` 或 `gain_reward`：

- `route_progress_bonus`
- `return_progress_shaping_bonus`
- `necessary_charge_bonus`
- `unnecessary_charge_penalty`
- `charge_detour_cost`
- `charge_interrupt_cost`
- `skip_needed_charge_penalty`
- `high_need_return_stall_penalty`
- `charger_access_discovery_bonus`
- `charger_access_probe_bonus`

保留但不在本轮修改：

- `battery_process_cost`
- `collision_process_cost`
- `coverage_tangle_penalty`
- `npc_penalty`
- `stuck_penalty`
- `idle_penalty`
- terminal battery/collision handling

shadow diagnostics 要求：

- 上述旧 charging 项在 `step payload / episode diagnostics / window aggregation / compare` 中继续保留一轮
- 但必须显式标记为：
  - `shadow_only`
  - `not_counted_in_reward`
- 允许：
  - 值继续计算
  - 但不得再进入 `task_reward` / `gain_reward`

这样本轮可以继续比较：

- 新 3 条 reward 是否真的替代了旧 charging terrain
- 而不是单纯把 compare 口径切断

### C. Slice 2A: `charge_need_score` 与缓存

切片二不修改 `compute_charge_need_score()` 的输入结构和 state 分类规则，只要求：

- 每步保留：
  - `charge_need_score`
  - `prev_charge_need_score`
- compare / diagnostics 继续输出：
  - `battery_state_idx`
  - `recoverability_score_avg`
  - `late_return_rate`
  - `missed_charge_opportunity_rate`

若实现中发现 `charge_need_score` 需要改公式，必须作为切片二之后的单独变更，不与本轮混做。

### D. Slice 2B: `RETURN` Diagonal Bias

只在 `Slice 2A` 的 reward-only 版本验证通过后启用。

规则：

- 仅当 `current_mode == MODE_RETURN`
- 且现有 `suggested_action` 本身属于 diagonal action
- 才对现有 return bias 做 scale modifier

公式：

- `effective_return_bias_scale = base_return_bias_scale * 1.15`

实现要求：

- 不能新增新的 diagonal 选边逻辑
- 不能绕过现有 `suggested_action`
- 不能新增 diagonal reward
- 不能对 `CLEAN` 或 `EVADE` 施加 diagonal bias

监控要求：

- 新增：
  - `return_diagonal_bias_active_rate`
- 以及：
  - `return_suggested_action_diagonal_rate`

两者只进 monitor / curriculum_state / comparison_samples，不新增训练头

### E. 版本区分

切片二不能继续复用“无区分的 strong heuristic v1”口径。

必须新增一个可区分的新版本标识，二选一均可，但实现时只能选一种：

- 新 phase：
  - `s1_survival_strong_heuristic_slice2a_v1`
  - `s1_survival_strong_heuristic_slice2b_v1`
- 或 phase 不变，但新增：
  - `KAIWU_STRONG_HEURISTIC_REWARD_VERSION=slice2a`
  - `KAIWU_STRONG_HEURISTIC_REWARD_VERSION=slice2b`

要求：

- comparison_samples
- curriculum_state
- compare_training_runs.py
- 诊断文档

都能明确区分：

- 原 strong heuristic v1
- slice2a reward-only contraction
- slice2b reward + return diagonal

## Test Plan

### 单元测试

至少新增或修改以下测试：

1. `risk_release_reward`
- `charge_need_score` 下降时为正
- `charge_need_score` 不下降时为 0
- 在 `RETURN / PRE_RETURN` 与 `just_charged` 下分别覆盖

2. `risk_growth_while_clean_penalty`
- 逻辑 `CLEAN` 且风险升高时触发
- 非 `CLEAN` 时不触发
- `task_value_here` 低时惩罚更强，高时更弱

3. `charge_opportunity_cost_penalty`
- `just_charged == 1` 且 `prev_charge_need_score` 低时触发
- `detour_proxy` / `interrupt_proxy` 越高，惩罚越强
- 任何一项缺失时按 0 处理，不得崩溃

4. 旧 charging terrain 退为 shadow diagnostics
- strong heuristic slice2a 下：
  - 上述旧项不进入 reward 累加
  - 但仍进入 step payload / diagnostics / compare
- 非 slice2a phase 下旧行为不变

5. `RETURN` diagonal bias
- `suggested_action` 为 diagonal 时，bias scale 被放大
- `suggested_action` 为 orthogonal 时，不触发
- `CLEAN / EVADE` 下不触发

### Dry-run / 非回归验证

必须通过：

- `python3 train/run_training_phase.py s1_survival_strong_heuristic_v1 --seed-label dry --dry-run`
- 现有强启发式主测试集

### 训练验收

先跑 scratch，不做 resume。

主观察窗口：

- `Slice 2A`
  - `bootstrap_20`
  - `global_40`
- `Slice 2B`
  - 在 `2A` 通过后，再看 `bootstrap_20`

必须重点看：

- `avg_clean_per_step`
- `battery_fail_rate`
- `zero_charge_battery_fail_rate`
- `late_return_rate`
- `missed_charge_opportunity_rate`
- `route_phase_return_stall_rate`
- `return_diagonal_bias_active_rate`（仅 `2B`）

判定原则：

`Slice 2A`

- 如果 `late_return_rate` 下降、`missed_charge_opportunity_rate` 下降，但 `avg_clean_per_step` 明显下滑
  - 说明 `risk_growth_while_clean_penalty` 过强
- 如果 `zero_charge_battery_fail_rate` 不降
  - 说明 `risk_release_reward` 太弱或 `charge_need_score` 本身定义不够好
- 如果旧 shadow metrics 全部下降但主结果不改善
  - 说明 reward-only contraction 没有真正改变决策质量，只改变了记账方式

`Slice 2B`

- 只在 `Slice 2A` 已证明 reward 主线健康后才启动
- 如果 `return_diagonal_bias_active_rate` 几乎为 0
  - 说明 diagonal 启发未真正接入行为主链
- 如果 `route_phase_return_stall_rate` 不降且 `avg_clean_per_step` 不升
  - 说明 diagonal modifier 边际价值不足，可直接回退

## Assumptions

- 本轮默认切片二优先选择：
  - 新 version 标识
  - 或新 phase 名称
  但必须二选一显式区分，不允许与原 strong heuristic v1 混口径
- 逻辑 `CLEAN` 继续通过 strong heuristic 四态 helper 映射到现有训练 mode，不改单独 mode 头
- 监控字段允许新增，但不能引入新的 learner 头或 sample schema 结构
- 当前最重要的是 reward 主链去冲突和可解释，不追求一次性把所有 CPS 问题解决
- diagonal 启发在本轮默认是后续微切片，不是 reward contraction 的组成部分
