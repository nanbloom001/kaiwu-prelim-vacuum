# 多充电桩多路径 Planner 设计方案 V1.1

> 日期: 2026-04-19
> 目标: 用更丰富但仍可控的 planner 信号层，提升“找到更多可达充电路径”的能力，同时与“清扫规整度”协同，而不给 policy 过强硬控制。

## 1. 结论先行

这个方向是值得做的，但前提是：

- planner 必须继续保持为**弱信号层**
- planner 内部必须显式计算代价，且是**多目标代价**
- policy / reward / teacher 不应直接消费完整路径或完整代价图，只应消费**压缩后的中间层特征**

也就是说，要求的不是“planner 非常巧妙”，而是**planner 分层必须清楚**：

1. 内部规划层负责认真算多目标代价
2. 候选压缩层负责把路径族压缩成低噪声摘要
3. 策略接口层只暴露结构化中间信号

如果这三层边界清楚，planner 不需要“神奇”，而是需要“稳定、可解释、可压缩”。

V1.1 在此基础上新增一条原则：

- planner 不只评估“能不能回去”，还要评估“为了回去，是否会把当前清扫几何结构破坏得太严重”

但这种“规整度”只应该作为**软代价**进入 planner 内部，而不应把 planner 变成新的 coverage controller。

---

## 2. 当前 planner 的主要局限

当前系统虽然会评估多个 charger candidate，但对训练链真正暴露出来的，仍然近似是：

- 单一 `charger_target`
- 单一 `route_anchor`
- 单一路径语义

这带来 4 个问题：

### 2.1 可达性表示太薄

现在更像是在回答：

- 当前最优桩是哪一个
- 当前最优路径是哪一条

而不是回答：

- 当前至少有哪些桩具有潜在可达性
- 对每个桩还保留多少备选回收路径

### 2.2 初期探索空间被压窄

在 `all_charger_known_path_count == 0` 时，系统仍然会很快收缩到单目标桩。  
这会让 reward / teacher 更早围绕一个 target 收缩，而不是先建立多个可回收选项。

### 2.3 recoverability 估计脆弱

当前 recoverability/slack 基本围绕单 anchor path 计算。  
如果该路径恰好差、堵、未知比高，就会低估真实的“还有别的路可回去”的空间。

### 2.4 reward 不容易直接表达“建立更多可达路径”

现在 reward 更容易奖励：

- 往当前 target 方向推进
- 充电成功

而不容易奖励：

- 已知可达 charger 数增加
- 某个 charger 的 route diversity 提升
- 未知路径知识改善

---

## 3. V1 总体原则

V1 不做“更强控制器”，只做“更强信号层”。

### 原则 A: planner 内部强，外部接口弱

内部可以维护多个 charger、每个 charger 多条 route family。  
对外只输出压缩后的结构化特征，不输出全量 path 集合。

### 原则 B: 可达性优先于最短路

planner 的首要目标不是“选最短回桩路径”，而是：

1. 是否存在稳定可达路径
2. 是否存在备选路径
3. 当前探索是否在增加未来可达性

### 原则 C: route family 比 path list 更重要

V1 不保存很多条完整路径，而是对每个 charger 维护少量不同偏好的**路径族**：

- 最低总代价路径
- 最低未知比例路径
- 最低风险路径

这比“很多条近似相同的 path”更稳定、更有信息价值。

### 原则 D: planner 不直接主导日常动作

planner 仍然只用于：

- guidance
- target/anchor 候选
- reward shaping
- teacher masks
- extreme emergency fallback

不把日常动作决策重新变成 hard-coded controller。

### 原则 E: 清扫规整度进入 planner，但只作为次级软代价

planner 应该考虑：

- 这条回充路会不会明显增加交叉与回头
- 会不会把当前边缘推进切断
- 会不会把当前正在连续处理的局部区域撕裂

但它不应直接优化：

- 哪条路能清更多格
- 哪条路能赚更多 `cleaning/streak/cps`

也就是说，planner 只评估**任务扰动成本**，不直接评估**任务收益**。

---

## 4. V1 结构设计

## 4.1 三层架构

### Layer 1: Internal Planner State

内部维护：

- Top-K charger targets
- 每个 target 的 2~3 条 route family
- 每条 route family 的多目标代价分解

建议：

- `K = 3`
- 每个 charger 最多维护 3 条 route family

### Layer 2: Candidate Compression

把每个 charger 的 route family 压成固定长度摘要：

- 是否可达
- 最优总代价
- 最优安全代价
- 最低未知比例
- 路径多样性
- 最佳 slack
- route family gap

### Layer 3: Policy-Facing Signals

最终只给 policy/reward/teacher 使用这些全局摘要：

- `known_route_count_total`
- `topk_reachable_count`
- `best_target_best_cost`
- `best_target_safe_cost`
- `best_target_unknown_ratio`
- `best_target_route_diversity`
- `best_vs_second_gap`
- `multi_route_recoverability_score`
- `charger_access_discovery_delta`
- `planner_best_target_tangle_cost`
- `planner_best_target_edge_break_cost`
- `planner_best_target_region_fragment_cost`
- `planner_current_task_continuity_cost`

---

## 4.2 Charger Target Set

新增内部结构：

```text
PlannerTargetSet = [
  target_1,
  target_2,
  target_3,
]
```

每个 target 至少包含：

- `center`
- `reachable_any`
- `known_route_count`
- `best_total_cost`
- `best_safe_cost`
- `best_unknown_ratio`
- `best_slack`
- `route_diversity`
- `corridor_coverage`
- `best_tangle_cost`
- `best_edge_break_cost`
- `best_region_fragment_cost`
- `last_improved_step`
- `selection_score`

这里的 `selection_score` 不是直接给 policy 的，而是 planner 内部用于 target 排序的综合分数。

---

## 4.3 Route Family Set

对每个 charger 维护 3 条 route family。

### Route Family A: Best-Cost Route

目标：

- 当前综合代价最低

适合：

- 已知路径较多
- 风险可控

### Route Family B: Low-Unknown Route

目标：

- 未知比例最低

适合：

- 当前处于“建立可达路径知识”的阶段
- 不一定最短，但更稳

### Route Family C: Safe Route

目标：

- NPC/blocked 风险最低

适合：

- contract/return 阶段
- 高风险环境

每条 route family 保持统一字段：

- `path`
- `path_source`
- `reachable`
- `cost_total`
- `cost_length`
- `cost_unknown`
- `cost_safety`
- `cost_revisit`
- `cost_task_interrupt`
- `cost_tangle`
- `cost_edge_break`
- `cost_region_fragment`
- `slack`
- `unknown_ratio`
- `corridor_mask_ref`

V1 不要求 route families 完全不同，但要求至少代表不同 cost 偏好。

---

## 5. 代价模型

## 5.1 必须显式计算代价

这一层 planner 不能只给“可行路径”，必须算代价。  
但这些代价主要用于 planner 内部与压缩摘要，不直接原样暴露给 policy。

## 5.2 代价分层

建议代价分成四类：

### 一级: 生存代价

- `cost_length`
- `cost_slack_risk`
- `cost_unreachable`

这是最高优先级。

### 二级: 安全代价

- `cost_npc_risk`
- `cost_blocked_memory`

这比效率更重要。

### 三级: 未知性代价

- `cost_unknown`
- `cost_route_knowledge_gap`

用于衡量该路径是否真的建立了稳定可达性。

### 四级: 任务扰动代价

- `cost_revisit`
- `cost_task_interrupt`
- `cost_tangle`
- `cost_edge_break`
- `cost_region_fragment`

这层不应压过生存和安全。

## 5.3 V1 建议总分公式

```text
cost_total
  = 1.00 * cost_length
  + 1.20 * cost_slack_risk
  + 1.00 * cost_npc_risk
  + 0.80 * cost_unknown
  + 0.20 * cost_revisit
  + 0.15 * cost_task_interrupt
  + 0.10 * cost_tangle
  + 0.08 * cost_edge_break
  + 0.07 * cost_region_fragment
  + unreachable_penalty
```

其中：

- `unreachable_penalty` 应维持强惩罚
- `cost_revisit`、`cost_task_interrupt`、`cost_tangle`、`cost_edge_break`、`cost_region_fragment` 只做软代价，不做硬约束

## 5.4 对“经过已清扫路面”和“经过机器人”的判断

### 经过已清扫路面

要算，但作为软代价：

- 不是禁止走
- 只是让“全靠重复踩旧路”的路径略微变贵

### 经过 NPC / 动态风险区域

要强算，属于安全代价：

- 重要性高于 revisit
- 应继续通过 danger field + blocked memory 体现

## 5.5 对清扫规整度的量化

V1.1 新增 3 个 planner 内部规整度代价。这些量不直接取代 reward，而是与 reward 协同：

### A. `cost_tangle`

表达：

- 这条回充路是否会明显增加交叉、回头和打结

建议使用现有几何量近似：

```text
cost_tangle
  = 0.50 * norm(path_cross_count_50, 10)
  + 0.30 * (1 - coverage_efficiency_20)
  + 0.20 * loop_proxy
```

这里直接复用 orderly coverage 方案里的 `coverage_tangle_penalty` 核心代理量，但只在 planner 内部作为软成本。

### B. `cost_edge_break`

表达：

- 当前如果正在沿边推进，这条回充路会不会把边缘清扫连续性硬切断

可用近似：

```text
cost_edge_break
  = frontier_presence
  * heading_conflict_with_route
  * edge_follow_active
```

其中：

- `frontier_presence` 由 `local_frontier_density` 给出
- `heading_conflict_with_route` 衡量当前顺边方向与回充路径前几步方向是否冲突
- `edge_follow_active` 表示当前是否处于 expand/harvest 且最近动作稳定

### C. `cost_region_fragment`

表达：

- 这条回充路会不会把当前正在连续处理的局部区域切碎

V1 用局部代理，不做复杂连通域：

```text
cost_region_fragment
  = local_task_value
  * route_separation
```

其中：

- `local_task_value`
  - `0.45 * local_dirt_density`
  - `+ 0.25 * norm(dirty_adjacent, 4)`
  - `+ 0.20 * local_frontier_density`
  - `+ 0.10 * norm(new_explored_cells, 6)`
- `route_separation`
  - 回充路径前 `6~10` 步是否快速脱离当前高价值局部区域

这 3 个成本的作用不是让 planner 追求“最漂亮的清扫路线”，而是在多条都能活着回去的路径里，优先选**更少破坏当前清扫几何结构**的那条。

---

## 6. 对外接口设计

## 6.1 不直接暴露什么

以下内容不直接进入 observation：

- 全部完整 path
- 全部 cost map
- 每格的 planner corridor 热图
- 全量 route family 明细

## 6.2 直接暴露什么

建议新增 planner 中间层特征：

- `planner_known_route_count_total`
- `planner_topk_reachable_count`
- `planner_best_target_best_cost`
- `planner_best_target_safe_cost`
- `planner_best_target_unknown_ratio`
- `planner_best_target_slack`
- `planner_best_target_route_diversity`
- `planner_best_vs_second_gap`
- `planner_multi_route_recoverability`
- `planner_access_discovery_delta`
- `planner_best_target_tangle_cost`
- `planner_best_target_edge_break_cost`
- `planner_best_target_region_fragment_cost`
- `planner_current_task_continuity_cost`

另外，对 top-3 charger 每个 charger 可以提供固定槽位摘要：

- `reachable_any`
- `known_route_count`
- `best_cost`
- `safe_cost`
- `best_unknown_ratio`
- `best_slack`
- `route_diversity`
- `best_tangle_cost`
- `best_edge_break_cost`
- `best_region_fragment_cost`

这样 policy 能看到“候选结构”，但不会过拟合 planner 细节。

这些新特征的定位是：

- 给 policy 一个“这条回充路会不会把当前清扫结构切烂”的摘要
- 而不是直接让 policy 学 planner 内部的完整路径偏好

---

## 7. 与 reward / teacher / mode 的接入

## 7.1 Reward

V1 最重要的新 reward 方向，不应是“跟随 A* 动作”，而应是：

- `charger_access_discovery_bonus`

它直接奖励：

- `known_route_count_total` 增加
- 当前 target 的 `unknown_ratio` 下降
- `reachable_any` 从 false/weak 变为 true/strong

一个较弱的辅助奖励可以是：

- `charger_corridor_exploration_bonus`

但只在 `all known routes == 0` 时启用，而且必须是一次性增量，不做持续奖励。

与清扫规整度的协同建议如下：

- planner 内部使用：
  - `cost_tangle`
  - `cost_edge_break`
  - `cost_region_fragment`
- reward 层使用：
  - `coverage_tangle_penalty`
  - `edge_follow_bonus`

两层职责必须分开：

### planner 层负责

- 评估“哪条回充路更少破坏当前清扫结构”

### reward 层负责

- 直接塑造“别交叉乱窜”
- 直接塑造“在边缘上下文中顺边推进”

注意：

- planner 不直接输出一个新的“coverage_shape_bonus”
- 不把“清扫收益”重新编码进 planner
- 只把“清扫几何扰动代价”编码进 planner

## 7.2 Teacher

teacher 仍然只在 planner 信号可靠时使用。  
升级后 teacher 可以使用：

- 更稳的 target choice
- 更稳的 route anchor
- 更好的 return action mask

但 teacher 不直接监督“走某条完整 path”。

## 7.3 Mode

mode 不再只依赖单一 anchor path，而是依赖：

- `multi_route_recoverability_score`
- `topk_reachable_count`
- `best_vs_second_gap`
- `planner_current_task_continuity_cost`

这样 contract/return 的切换会更平滑，不会过度依赖单条 path。

额外约束：

- 当 `battery_state == critical` 时，规整度相关成本应自动弱化或关闭
- 当 `battery_state == planning` 时，只保留弱影响
- 当 `battery_state == safe` 时，规整度代价正常参与 target/routing 排序

---

## 8. V1 实施顺序

建议按 4 步做，不要一次性全上。

### Step 1: 内部数据结构升级

只在 `expert.py` 内部新增：

- `PlannerTargetSet`
- `RouteFamilySet`
- 多目标代价分解
- 规整度代价分解：
  - `cost_tangle`
  - `cost_edge_break`
  - `cost_region_fragment`

先不改 reward / teacher / observation。

### Step 2: 中间层特征导出

在 `expert.get_charger_signal()` 中新增：

- target-set 摘要
- route-family 摘要
- 全局压缩特征
- 清扫规整度摘要特征

然后在 `preprocessor` 里挂到 feature 和 diagnostics。

### Step 3: 用新特征替换旧的单路径语义

逐步替换：

- `recoverability_score`
- `route_contract_pressure`
- `route_anchor_margin`
- 部分 `mode` 推断逻辑
- 部分 task continuity 相关 heuristics

但不删除旧字段，先做兼容期。

### Step 4: 接入新的 reward

新增：

- `charger_access_discovery_bonus`

必要时再加：

- 很弱的 `charger_corridor_exploration_bonus`

### Step 5: 与 orderly coverage reward 联动

把 planner 侧规整度代价与 reward 侧几何奖励明确配对：

- `cost_tangle` ↔ `coverage_tangle_penalty`
- `cost_edge_break` ↔ `edge_follow_bonus`
- `cost_region_fragment` ↔ 仅保留在 planner 内部，V1 不单独做 reward

V1.1 不建议再新增独立的 `coverage_shape_bonus`，避免和：

- `streak`
- `cps_bonus`
- `coverage_efficiency_bonus`

产生重复奖励。

---

## 9. 风险与保护

### 风险 A: target 抖动变大

保护：

- 继续保留 hysteresis
- 新增 `best_vs_second_gap` 稳定门槛
- route family 更新不直接强制换 target

### 风险 B: reward 更容易被刷

保护：

- discovery 类奖励只按增量发放
- 不做持续占位奖励
- corridor 奖励只作为弱辅助
- 规整度相关奖励不做持续占位奖励，只做几何状态或增量代理

### 风险 C: planner 偏差压过 policy

保护：

- planner 仍然是弱信号层
- 不直接奖励动作级 A* 对齐
- observation 只暴露压缩特征
- planner 内部可以考虑规整度，但不直接优化“清扫收益”

---

## 10. 测试与验收

## 10.1 单元测试

至少覆盖：

- 单 charger / 多 charger 下 target set 正确生成
- 每个 charger 至少有 2~3 条 route family 槽位
- known route count 与 reachable_any 逻辑正确
- best/safe/low-unknown route 的 cost 排序合理
- 压缩特征在空路径、未知路径、可达路径下数值正确

## 10.2 训练期观测

重点新增监控：

- `planner_topk_reachable_count`
- `planner_known_route_count_total`
- `planner_best_target_unknown_ratio`
- `planner_best_target_route_diversity`
- `planner_multi_route_recoverability`
- `charger_access_discovery_rate`
- `planner_best_target_tangle_cost`
- `planner_best_target_edge_break_cost`
- `planner_best_target_region_fragment_cost`

## 10.3 行为验收

如果 V1 生效，预期应看到：

- `zero_charge_battery_fail_rate` 下降
- `avg_all_charger_known_path_count` 上升更快
- `unknown_on_target_path_ratio` 下降
- `return_stall_rate` 下降
- `battery_fail_rate` 下降
- `path_cross_count_50` 下降
- `coverage_efficiency_20` 上升
- 边缘推进时的轨迹更连贯

同时：

- `avg_charge_count` 不应重新飙回旧版高频充电区

---

## 11. 最终建议

V1 最适合定义为：

> 一个“多充电桩、多路径族、多目标代价”的中层 planner 信号系统。

它不是更强的硬控制器，也不是只给可行路径的弱启发式，而是：

- 内部认真规划
- 外部谨慎暴露
- 主要增强：
  - route knowledge
  - recoverability estimation
  - charging-aware exploration shaping
  - 与清扫规整度兼容的回充路径选择

这条路线最适合当前项目，因为它既能提升“找到更多可达路径”的能力，又不会把训练重新拖回 hand-crafted controller 主导。
