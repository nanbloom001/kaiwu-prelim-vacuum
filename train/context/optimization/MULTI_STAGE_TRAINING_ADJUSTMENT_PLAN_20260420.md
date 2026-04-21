# 多阶段训练调整方案 V1

> 日期: 2026-04-20  
> 目标: 用分阶段方式调整当前模型训练方法，不要求一步到位，但每一步都必须有明确目的、明确成功标准、明确失败回退逻辑。  
> 最终目标: 让模型逐步学会 **高 CPS、能自主充电、能按 planner 规划清扫路径**。

## 1. 当前基线判断

当前这轮 scratch 训练已经不是“链路没跑起来”，而是进入了**行为结构不健康**的状态。

最近窗口里，最关键的问题组合是：

- `battery_fail_rate` 仍高，约 `0.35 ~ 0.40`
- `zero_charge_battery_fail_rate` 仍高，约 `0.57 ~ 0.63`
- `battery_positive_reward_rate` 偏高，约 `0.50 ~ 0.64`
- `avg_clean_per_step` 没崩，但仍不够高，约 `0.507`
- `mode_usage_expand` 很低，约 `0.04 ~ 0.05`
- `mode_usage_contract` 几乎消失，约 `0.004`
- `planner_policy_divergence_rate` 长期偏高，约 `0.83`
- `route_phase_planner_divergence_rate` 仍偏高，约 `0.58`
- `route_phase_return_stall_rate` 约 `0.32 ~ 0.34`
- curriculum 已重新进入：
  - `curriculum_stagnation_level = 3`
  - `curriculum_stagnation_reason = ["charge", "reward"]`
  - `degraded_mainline = true`

这说明当前模型不是不会清扫，而是：

1. **首充闭环没有稳住**
2. **`contract` 缓冲态被压没了**
3. **planner 约束还没有真正落到 route-phase 动作层**
4. **坏的 battery fail 轨迹仍能拿到过多正收益**

因此后续训练调整必须按下面顺序推进：

> **先保生存，再保 planner，最后追高 CPS。**

## 2. 总体执行原则

### 2.1 阶段边界

每个阶段都采用：

- **新的 scratch run**
- 不从上一阶段续训
- 不混用阶段配置

原因：

- 方便归因
- 避免“这一阶段看起来有效，但其实是上一阶段残留”
- 便于明确判断“哪一步真的有效”

### 2.2 阶段验收窗口

每个阶段统一看两个窗口：

- `bootstrap 20`
- `global 40`

不允许只看单局或最近几局做结论。

### 2.3 允许的改动强度

本方案默认分两类杠杆：

#### 前三阶段允许

- `contract gate`
- `curriculum gate`
- `profile weights`
- `route-phase guidance`
- `reward 小调`

#### 第四阶段才允许

- 轻度 `supervision-lite`
- teacher weight 下调

明确不做：

- 不改 backbone
- 不删 head
- 不重写整体 reward 体系

## 3. 阶段 0：基线锁定

### 目标

在进入任何阶段调参前，先把当前基线固定下来，避免后续“没有统一对照组”。

### 本阶段做什么

1. 固定当前基线窗口：
   - `battery_fail_rate`
   - `zero_charge_battery_fail_rate`
   - `battery_positive_reward_rate`
   - `avg_clean_per_step`
   - `mode_usage_expand`
   - `mode_usage_contract`
   - `mode_usage_return`
   - `planner_policy_divergence_rate`
   - `route_phase_planner_divergence_rate`
   - `reliable_planner_divergence_rate`
   - `route_phase_return_stall_rate`
2. 为后续每个阶段准备独立配置名，例如：
   - `s1_survival`
   - `s2_planner`
   - `s3_cps`
   - `s4_light_ablation`
3. 约定统一的阶段报表模板

### 成功标准

- 基线快照完整
- 阶段命名和配置命名固定
- 后续所有 run 都能明确标注属于哪个阶段

### 失败标准

- 指标口径还不统一
- 阶段配置还依赖手工散改常量

### 进入下一阶段条件

- 基线快照和阶段模板已经固定

## 4. 阶段 1：生存优先，恢复“预备收缩层”

### 目标

优先解决：

- 一次电都没充上就死
- `contract` 缓冲态消失
- 坏的 battery fail 轨迹还能拿高正收益

本阶段目标不是高 CPS，而是：

> **先让模型活下来，并能更稳定地形成首充闭环。**

### 本阶段允许调整的杠杆

1. `contract gate` 回调
   - 让 `contract` 不再接近 `0`
   - 恢复它作为 `expand/harvest -> return` 的缓冲态
2. `curriculum` 的 stop-loss 继续以：
   - `charge`
   - `reward`
   为主
3. profile 分布偏 survival-first
   - 但不走纯 `anchor-heavy`

### 本阶段不追求

- `planner_policy_divergence_rate` 立刻变漂亮
- `expand` 大幅提升
- `avg_clean_per_step` 立刻上很高

### 成功标准

满足以下全部条件：

- `zero_charge_battery_fail_rate <= 0.40`
- `battery_fail_rate <= 0.28`
- `battery_positive_reward_rate <= 0.20`
- `mode_usage_contract` 回到 `0.06 ~ 0.18`
- `avg_clean_per_step >= 0.46`

### 失败标准

出现以下任一情况：

- `zero_charge_battery_fail_rate > 0.50`
- `mode_usage_contract < 0.03`
- `battery_positive_reward_rate > 0.30`

### 失败后的回退策略

- 不进入阶段 2
- 继续只调：
  - `contract gate`
  - `charge/reward` 相关 curriculum stop-loss 阈值

## 5. 阶段 2：规划优先，把 planner 压到 route-phase 动作层

### 目标

在首充闭环基本形成之后，让模型在：

- `contract`
- `return`

阶段真正更贴 planner，而不是只学会“进入某个 mode”。

### 本阶段允许调整的杠杆

1. 保留阶段 1 中已经验证有效的生存设置
2. 强化 route-phase 对主动作的直接约束
   - 主看 `route_phase_policy_teacher_loss`
3. curriculum 和 checkpoint gate 主看：
   - `reliable_planner_divergence_rate`
   - `route_phase_planner_divergence_rate`
   - `route_phase_return_stall_rate`

### 本阶段不追求

- 立刻把 `expand` 拉到很高
- 过早放宽 profile
- 追求最终 CPS 峰值

### 成功标准

满足以下全部条件：

- `reliable_planner_divergence_rate <= 0.30`
- `route_phase_planner_divergence_rate <= 0.45`
- `route_phase_return_stall_rate <= 0.25`
- `zero_charge_battery_fail_rate <= 0.35`
- `battery_fail_rate <= 0.25`
- `avg_clean_per_step >= 阶段1的95%`

### 失败标准

出现以下任一情况：

- `reliable_planner_divergence_rate > 0.45`
- `route_phase_return_stall_rate > 0.35`
- `battery_fail_rate` 相比阶段 1 反弹超过 `+0.05`

### 失败后的回退策略

- 不进入阶段 3
- 只调 route-phase guidance 权重和相关 gate

## 6. 阶段 3：恢复高 CPS，但不丢生存闭环

### 目标

在已经“会活、会充、会回”的前提下，把真正的高质量清扫效率拉起来。

本阶段关注的是：

- 让 `expand` 恢复
- 让清扫效率继续提高
- 但不能因此重新回到高零充失败、高 battery fail

### 本阶段允许调整的杠杆

1. profile 分布逐步放开：
   - 提高 `mild / broad` 暴露
2. 轻度放宽保守收缩
   - 让 `expand` 恢复空间
3. reward 只做小调，不加新项

### 本阶段不追求

- 再次大改 `contract` 逻辑
- 重写 reward
- 动结构

### 成功标准

满足以下全部条件：

- `avg_clean_per_step >= 0.56`
- `win_rate >= 0.68`
- `mode_usage_expand >= 0.08`
- `mode_usage_contract <= 0.12`
- `battery_fail_rate <= 0.25`
- `zero_charge_battery_fail_rate <= 0.30`
- `route_phase_planner_divergence_rate <= 0.40`

### 失败标准

出现以下任一情况：

- `avg_clean_per_step < 0.52`
- `battery_fail_rate > 0.30`
- `zero_charge_battery_fail_rate > 0.40`

### 失败后的回退策略

- 回退到阶段 2 最优配置
- 不继续放宽 profile

## 7. 阶段 4：轻度结构改动，只做 supervision-lite

### 目标

如果前三阶段已经尽量用训练方式把问题解决了，但 planner/stall 仍压不下去，再验证是否是多路 teacher 监督制造了训练冲突。

### 本阶段允许的唯一结构性动作

只允许轻度 `supervision-lite`：

- `MODE_TEACHER_WEIGHT` 下调
- `ROUTE_ANCHOR_TEACHER_WEIGHT` 下调
- `TARGET_TEACHER_WEIGHT` 下调

不允许：

- 删除 head
- 改 backbone
- 改 actor 输入结构

### 进入本阶段条件

仅当阶段 3 已经基本达标，但仍存在以下任一问题时才进入：

- `planner_policy_divergence_rate > 0.70`
- `route_phase_planner_divergence_rate > 0.35`
- `route_phase_return_stall_rate > 0.22`

### 成功标准

满足以下全部条件：

- `route_phase_planner_divergence_rate <= 0.35`
- `route_phase_return_stall_rate <= 0.22`
- `avg_clean_per_step` 不低于阶段 3 的 `97%`
- `battery_fail_rate` 不高于阶段 3 的 `+0.03`

### 失败标准

出现以下任一情况：

- `avg_clean_per_step` 明显回落
- `battery_fail_rate` 反弹
- `zero_charge_battery_fail_rate` 反弹

### 失败后的回退策略

- 直接回退到阶段 3 最优配置
- 不再继续做结构类探索

## 8. 阶段配置接口建议

为了让这套方案可执行，训练入口必须支持阶段化配置，不允许继续靠手工散改常量。

建议统一引入：

```text
TRAIN_PHASE = baseline | s1_survival | s2_planner | s3_cps | s4_light_ablation
```

每个阶段配置至少覆盖以下参数组：

- `contract gate thresholds`
- `curriculum thresholds`
- `profile weights`
- `route-phase guidance weights`
- `teacher/loss weights`

## 9. 固定验收指标

每个阶段统一导出并对比这些指标：

- `battery_fail_rate`
- `zero_charge_battery_fail_rate`
- `battery_positive_reward_rate`
- `avg_clean_per_step`
- `win_rate`
- `mode_usage_expand`
- `mode_usage_contract`
- `mode_usage_return`
- `planner_policy_divergence_rate`
- `route_phase_planner_divergence_rate`
- `reliable_planner_divergence_rate`
- `route_phase_return_stall_rate`

## 10. 执行顺序总结

统一执行顺序为：

1. **阶段 0：锁基线**
2. **阶段 1：保生存 / 恢复 contract 缓冲层**
3. **阶段 2：压 planner 到 route-phase 动作层**
4. **阶段 3：恢复 expand 和高 CPS**
5. **阶段 4：仅在必要时做轻度 supervision-lite**

核心原则：

> **每一阶段只解决一类主问题。**
> **每一阶段都必须有成功标准和失败回退。**
> **不允许在前一阶段还没跑通时叠加后一阶段改动。**

