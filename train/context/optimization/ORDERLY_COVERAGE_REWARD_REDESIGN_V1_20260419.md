# 有规律清扫奖励重构方案 V1.1

> 日期: 2026-04-19
> 目标: 把当前“高 CPS 但可能交叉乱窜”的清扫行为，重构成更连贯、更少交叉、更像条带式/贪吃蛇式填满区域的覆盖行为。

---

## 1. 问题定义

当前系统已经有一些和“有序清扫”相关的信号，但它们更多是在鼓励：

- 清到就有分
- 连续清到有分
- 清扫效率高有分
- 靠近未知边界有分
- 太乱、太绕、太停滞会亏

这套结构能把模型推向“更有效率”，但**不能直接表达“路径几何形状是否优美、是否连贯、是否像填满一个区域”**。

当前最主要的问题有 4 个：

1. `cps_bonus` 只看结果效率，不看路径形状。
2. `frontier_reward` 的定义更接近“边界附近探索激励”，不是“沿边有序清扫奖励”。
3. `coverage_efficiency_20`、`path_cross_count_50` 等几何量目前主要被用于负向成本或诊断，没有被组织成一套显性的“覆盖形状奖励”。
4. 当前没有直接奖励：
   - 条带推进
   - 区域填充完整性
   - 少交叉少回头
   - 边缘清扫连续性

这导致模型可能学成：

- CPS 还不错
- clean_score 也不差
- 但轨迹丑、交叉多、区域被切碎、边界锯齿状、清扫不整洁

---

## 2. 设计目标

新方案的目标不是简单提高分数，而是显式塑造以下 5 种“有规律清扫”的行为：

1. **条带推进**
   - 在局部区域内保持方向连续，不频繁大幅折返。

2. **区域填充**
   - 进入一块区域后尽量连续处理，而不是到处打补丁。

3. **少交叉**
   - 明显减少路径交叉和回头。

4. **边缘整洁**
   - 沿已知/未知边界清扫，但不是围着 frontier 打转。

5. **模式切换平滑**
   - 清扫路径几何优化不能压过回充/生存优先级。

因此本方案的基本原则是：

- 新增显性的“覆盖几何奖励”
- 重构而不是继续叠加旧 `frontier` 语义
- 与 charging reward 分层，不让“优美路径”压过 survival

---

## 3. 总体策略

V1.1 建议采用“二加一降级”的结构：

### 新增 2 类显性覆盖几何项

1. `coverage_tangle_penalty`
   - 惩罚交叉、回头、打结、重复切割

2. `edge_follow_bonus`
   - 真正表达“沿已知边缘顺边推进”的奖励

### 降级 1 类现有项

1. `frontier_reward`
   - 不再承担“有序覆盖”的职责
   - 保留为弱探索项，改名更合适，例如 `frontier_presence_bonus`

### 保留但重新定位的项

- `cleaning`
  - 基础主奖励，继续保留
- `streak`
  - 继续保留，但只表达“连续清扫”，不表达路径美学
- `cps_bonus`
  - 继续保留，作为结果性效率项
- `coverage_efficiency_bonus`
  - 保留，但只做结果性覆盖效率项，不再承担新的几何奖励职责
- `dirty_approach_reward`
  - 保留，但明确只做局部 dirt 方向诱导
  - 不再被视为“形状奖励”

---

## 4. 新奖励项设计

## 4.1 `coverage_tangle_penalty`

### 作用

直接惩罚“交叉乱窜”和“路径打结”。

这是 V1 最优先引入的项，因为：

- 语义简单
- 最不容易被误解
- 最容易用现有字段稳定实现

### 现有可直接复用的量

- `path_cross_count_50`
- `coverage_efficiency_20`
- `recent_unique_cells_20`
- `same_region_streak`

### 建议公式

```text
tangle_raw
  = 0.50 * norm(path_cross_count_50, 10)
  + 0.30 * (1 - coverage_efficiency_20)
  + 0.20 * loop_proxy
```

其中：

```text
loop_proxy = clip(
  same_region_streak / 8
  + (1 - recent_unique_cells_20 / 20),
  0, 1
)
```

最终：

```text
coverage_tangle_penalty = -S_tangle * tangle_raw
```

推荐初值：

- `S_tangle = 0.10`

### 激活条件

- `battery_state == safe`：全量
- `battery_state == planning`：`0.5x`
- `battery_state == critical`：`0`

理由：
- 临近回充时不要因为“保持路径漂亮”而晚充

---

## 4.2 `edge_follow_bonus`

### 作用

这项专门实现原始目标：

> 鼓励机器人沿着已知边缘进行清扫，提高整洁性

它不是 frontier presence，而是：

- 当前处于边缘上下文
- 当前动作方向与边缘走向一致
- 最近若干步沿边连续推进

### 为什么当前 `frontier` 不够

当前 `frontier_reward` 定义为：

```text
frontier_reward
  = frontier_scale
  * local_frontier_density
  * dirt_progress_term
```

它只表达：

- 当前附近 frontier 多不多

不表达：

- 动作是不是沿边走
- 边缘推进是否连续
- 边界是否更平滑

### V1.1 近似量化

不做复杂边界几何拟合，先做一个稳定近似。

#### A. 边缘存在性

直接用：

- `local_frontier_density`

并要求其高于阈值，比如：

- `local_frontier_density >= 0.10`

#### B. 顺边推进

若最近若干步：

- 方向切换率较低
- `same_region_streak` 不高到像打转
- `path_cross_count_50` 较低

则认为是在顺边推进，而不是围边乱窜。

#### C. 清扫发生

若：

- `cleaned_this_step > 0`
- `local_frontier_density` 持续存在

则给正奖励。

### 建议公式

```text
edge_follow_raw
  = frontier_presence
  * heading_consistency
  * low_tangle_mask
  * clean_step_mask
```

最终：

```text
edge_follow_bonus = S_edge * edge_follow_raw
```

推荐初值：

- `S_edge = 0.06`

### 激活条件

- `current_mode in {expand, harvest}`
- `battery_state == safe`
- `local_frontier_density >= 0.10`

理由：
- 这项只表达“在边缘上下文中顺边推进”
- 不再承担“区域填满”或“整体覆盖形状”的职责

---

## 5. 现有奖励的取舍建议

## 5.1 `frontier_reward`

### 当前问题

它现在承担了太多“有序覆盖”的想象，但实际只表达“边界附近探索”。

### 建议

不要直接删除，但要**降级重定位**：

- 语义改成：`frontier_presence_bonus`
- 只负责鼓励不要缩在内部区域
- 权重下调

建议：

- `FRONTIER_REWARD_SCALE: 0.18 -> 0.08`

并承认它不再负责“路径整洁性”。

---

## 5.2 `explore_reward`

保留。

理由：
- 它仍然是一般性未知区域开拓信号
- 不应由 coverage 几何奖励替代

但它也不应承担“清扫路径优美性”的职责。

---

## 5.3 `streak_bonus`

保留。

理由：
- 它能支持“连续工作”语义
- 但它只表达是否连续清到，不表达几何结构

不建议删除，只建议把“形状奖励”从它身上独立出去。

---

## 5.4 `coverage_efficiency_bonus`

保留，但重定位为：

- 结果性覆盖效率项

建议：

- 不再把它嵌入新的独立几何正项中，避免与新项重复奖励
- 允许 `coverage_tangle_penalty` 把它作为负向代理使用

---

## 5.5 `dirty_approach_reward`

保留为局部 dirt 方向感引导，但不视为“有序清扫奖励”。

它解决的是：

- 下一步该不该朝脏格方向靠

不是：

- 覆盖几何是否优美

---

## 6. V1 推荐的总体 reward 结构

建议未来的正负项结构改成：

### 基础清扫主线

- `cleaning`
- `streak`
- `cps_bonus`
- `explore`
- 弱化后的 `frontier_presence_bonus`

### 新的几何清扫主线

- `coverage_tangle_penalty`
- `edge_follow_bonus`

### 充电与生存主线

- 保留当前 charging 系统
- 这些新覆盖项在 `planning/critical` 下弱化或关闭

这样职责就清楚：

- `cleaning/cps`：结果效率
- `explore/frontier_presence`：一般探索
- `coverage_tangle/edge_follow`：路径几何与边缘整洁性
- charging rewards：回充与生存

---

## 7. 状态门控建议

覆盖几何奖励必须有状态门控，否则会和 survival 冲突。

### `safe`

- `edge_follow_bonus`：全量
- `coverage_tangle_penalty`：全量

### `planning`

- `edge_follow_bonus`：`0.3x`
- `coverage_tangle_penalty`：`0.5x`

### `critical`

- 全部关闭

理由：
- 临近没电时，先活下来
- 不要为了“路径好看”而晚充

---

## 8. 实施顺序

不要一次替换所有东西，建议三步走。

### Step 1

新增：

- `coverage_tangle_penalty`

同时：

- 把 `frontier_reward` 降级为弱探索项

这是最稳的一步。

### Step 2

新增：

- `edge_follow_bonus`

同时继续观察：

- 是否更沿边推进
- 是否减少边缘锯齿和局部碎裂

---

## 9. 测试与验收

## 9.1 单元测试

至少覆盖：

- 高 `path_cross_count_50` 时 `coverage_tangle_penalty` 更强
- `local_frontier_density` 高、方向稳定、且清扫发生时，`edge_follow_bonus` 为正
- `planning/critical` 状态下，这两项按门控衰减或关闭

## 9.2 训练指标验收

新增或重点观察：

- `avg_path_cross_count_50`
- `avg_coverage_efficiency_20`
- `same_region_streak` 分布
- `recent_unique_cells_20`
- 新 reward 分项占比

预期趋势：

- `path_cross_count_50` 下降
- `coverage_efficiency_20` 上升
- `cps` 不显著恶化
- `battery_fail_rate` 不明显恶化

## 9.3 轨迹人工验收

必须做。

至少人工抽查：

- 高 CPS completed 局
- 中期 expand/harvest 局
- 边缘推进明显的局

目标是视觉上确认：

- 交叉更少
- 回头更少
- 局部区域填充更完整
- 边缘清扫更整洁

---

## 10. 最终建议

如果只回答“是否要取代一部分现有奖励”，我的判断是：

### 要部分取代

尤其是：

- `frontier_reward` 不应继续承担“有序清扫”的职责

### 不要全替代

以下仍应保留：

- `cleaning`
- `streak`
- `cps_bonus`
- `explore`

### 最合理的方向

不是删除旧奖励重来，而是：

1. 让旧奖励回到各自更单纯的职责
2. 新增一个更窄、更可解释的“覆盖几何奖励层”

这样才能把“高效率”与“清扫轨迹更优美、更整洁”同时学出来。
