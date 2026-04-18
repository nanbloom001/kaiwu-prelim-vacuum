# 独立评审：V6 清扫机器人策略诊断与优化方案

> 评审基线：`v6-geo-bestmodel-576` checkpoint  
> 评审数据：`20260417-200128` / `20260417-212241` 两轮标准 benchmark  
> 重点轨迹：`round_4_map2`、`round_4_map4`、`round_4_map8`、`round_4_map10`  
> 评审日期：2026-04-17  

---

## 一、核心发现：reward attribution 揭示了真正的主因

### 1.1 数据事实

| 异常行为 | 样本数 | 平均 total reward | 说明 |
|---------|--------|------------------|------|
| wall_hugging_clean_floor | 1400~1840 | **+0.22 ~ +0.43** | **正奖励** |
| narrow_unknown_commit | 1454~1586 | **+2.47 ~ +2.51** | **强正奖励** |
| missed_charge_opportunity | 213~245 | **+2.23** | **强正奖励** |
| suboptimal_target_hold | 8393~8505 | **+2.16 ~ +2.20** | **强正奖励** |
| revisit_on_clean_floor | 2567~3095 | **+0.13 ~ +0.16** | **正奖励** |
| return_stall_window | 1933~2020 | **+1.59 ~ +1.91** | **强正奖励（因 cleaning 项）** |
| loop | 190~288 | -0.04 ~ -0.08 | 微弱负奖励 |
| corner_loop | 64~108 | -0.07 ~ -0.11 | 微弱负奖励 |

**关键结论：几乎所有被标记为"异常"的行为，在当前 reward 函数下都收到了正奖励。** 模型并未"拒绝纠正"——它从 reward 角度看，这些行为就是正确的。这不是策略学习能力的问题，而是 reward 信号本身在鼓励错误行为。

### 1.2 正奖励的来源分解

当发生 `wall_hugging_clean_floor` 时，正奖励主要来自：
- `reward_charge`：+3.1 ~ +9.4（充电时距墙近，充电所得奖励被错误归因到沿墙步）
- `reward_astar_potential`：+2.1 ~ +2.9（沿墙移动经常使 A* 距离小幅变化，产生正 shaping）
- `reward_anchor_consistency`：+1.3（沿墙不切桩 → 拿到一致性奖励）

当发生 `missed_charge_opportunity` 时，正奖励来自：
- `reward_cleaning`：+33 ~ +38（路过桩时通常在清扫区域，清扫奖励远大于充电需求）
- `reward_streak`：+11 ~ +15

当发生 `return_stall_window`（返航停滞）时：
- `reward_cleaning`：+54（返航途中继续清扫，获得巨大正奖励）
- `reward_streak`：+25

**结论：cleaning reward（1.5/step）+ streak bonus 是支配整个 reward 函数的绝对主力项，其量级远超所有惩罚项的总和。任何与"当前清扫一块脏格"冲突的行为信号，在 PPO 的 advantage 计算中都被淹没。**

---

## 二、问题因果排序（从根因到表象）

### Tier 0 — 根因

**R0: Cleaning reward 绝对支配，导致所有非清扫行为信号被淹没**

- cleaning_reward = 1.5/step，streak 最高 0.75/step → 峰值 2.25/step
- idle_penalty 最大 -0.1，return_stall 仅 -0.08，anchor_consistency 仅 0.05
- **cleaning 与 stall 的比值约 28:1**
- 在这个比值下，即使 planner 建议回桩，只要前方有一格脏地，policy 就会选择清扫

这一个根因直接产生了以下全部表象：

### Tier 1 — 由 R0 直接派生的行为

| 表象 | 因果链 |
|------|--------|
| 沿墙走已清扫路径 | 墙边是 dirt boundary 的天然邻居 → 沿墙常能碰到脏格 → 清扫收益 > 离开代价 |
| 返航停滞 (return_stall_rate=54%) | 返航路径上有脏格 → 清扫收益 > return_stall_penalty(-0.08) |
| 路过桩不充 (missed_charge) | 正在清扫中 → 清扫收益 > 充电的机会成本 |
| planner 被否决 (78%) | planner 说"去桩" → 但附近有脏格 → 清扫优先 → diverge |
| target 粘滞 (subopt_target=20%) | 切桩 = 远离当前清扫区 → 短期 cleaning loss → 不切 |
| 路线交叉/重复覆盖 | 回到旧区域有残留脏格 → 重扫获得正奖励 → 路线交叉 |

### Tier 2 — 结构性短板（非 reward 直接导致但加剧后果）

| 问题 | 机制 |
|------|------|
| all_charger_known_path_count 始终 = 1 | A* 只搜最近桩，其余桩仅用 chebyshev 估距 → 多桩场景下信息不足 |
| unknown_on_target_path_ratio 高 | 前期只知道一个桩的路径 → 切桩时路径不可靠 → 不切 |
| 短局电池死亡 (map4/map8, 200步) | 初始探索方向恰好远离桩 → 无 A* 路径 → charge_margin 不准 → 回不来 |

### 不是主要问题（与现有方案看法一致）

- 充电桩竞争：`charger_contested_rate ≈ 0`
- 切比雪夫 fallback：`fallback_to_chebyshev_rate = 0`
- 碰撞：`collision_fail_rate = 0`

---

## 三、对现有方案（V6_BEHAVIOR_REMEDIATION_PLAN）的评价与分歧

### 3.1 同意的部分

1. 墙边低价值跟随确实是高优先级问题
2. target 粘滞和 planner 偏离是真实问题
3. 充电桩竞争和切比雪夫 fallback 确实不是当前主矛盾
4. 问题分优先级、分阶段修复的思路是合理的

### 3.2 不同意的部分

**分歧 1：现有方案把"边界语义重构"列为最高优先级，我认为这是治标不治本。**

方案 A 提议引入"有效边界"概念、增加 `stale_boundary_penalty`、`wall_hugging_clean_floor_penalty` 等新惩罚项。问题是：

- 这些新惩罚项必须与 cleaning reward 的量级竞争（2.25/step），需要很强的惩罚力度
- 强惩罚又容易产生副作用（真正需要沿墙清扫时也被惩罚）
- 增加了 reward 函数的复杂度和调参难度
- **正确的做法是削弱 cleaning reward 在不该清扫时的强度，而不是在 cleaning reward 之上叠加越来越多的反向惩罚**

**分歧 2：现有方案提出 6 个独立方案，分 3 个阶段落地。我认为改动过多、风险过高。**

一次引入"边界语义重构 + 旧路代价 + planner 强接管"（仅阶段一就 3 个方案），会让后续 benchmark 结果难以归因。不知道是哪个改动起了作用、哪个产生了副作用。

**分歧 3：方案 F 提出"病态循环下 planner 强接管"。我反对这个方向。**

- 强制 override 策略输出是在绕过 RL 训练本身
- 如果 reward 是正确的，policy 应该自然学会跟随 planner
- 强制介入会导致 on-policy 与 off-policy 不一致，PPO 的 importance ratio 失效
- 正确做法是修好 reward 信号，让 policy 自然收敛到正确行为

**分歧 4：现有方案没有触及 cleaning reward 的量级问题。**

整个方案围绕"增加新惩罚"和"增加新信号"展开，但从未质疑过 cleaning_reward=1.5 这个基础设定是否合理。这是最大的盲点。

### 3.3 总结

| 项目 | 现有方案 | 本评审 |
|------|---------|--------|
| 主因 | 边界语义缺失 + planner 被否决 | cleaning reward 支配性过强 |
| 治疗 | 增加 5+ 新惩罚/信号项 | 修正 1~2 个现有 reward 项的条件逻辑 |
| 复杂度 | 高（6 方案 / 3 阶段） | 低（3 个精准改动） |
| 风险 | 中高（新惩罚副作用难控） | 低（不增加新 reward 项，只改条件） |
| 对 planner | 提议 hard override | 反对 override，让 reward 驱动收敛 |

---

## 四、优化方案（按优先级排序）

### 方案 1：上下文感知的清扫奖励缩放（Reward 层）

**原理**

当前 `cleaning_reward = 1.5 * cleaned_this_step`，`streak_bonus = 0.15 * ... * consecutive_clean_steps`——无论在什么 mode、什么位置、什么电量下，清扫一格脏地都拿同样的奖励。这导致所有导航/安全信号都被淹没。

**改动位置**

`preprocessor.py` → `reward_process()` 方法内，cleaning_reward 和 streak_bonus 的计算处。

**具体改法**

```python
# --- 清扫奖励上下文缩放 ---
cleaning_scale = 1.0

# 返航/收缩模式下大幅降低清扫奖励
if self.current_mode in (self.MODE_CONTRACT, self.MODE_RETURN):
    cleaning_scale = 0.25

# 重访已清扫区域时降低
elif self.cur_visit_count >= 3:
    cleaning_scale = 0.4
elif self.cur_visit_count >= 2:
    cleaning_scale = 0.7

# 沿墙且周围无脏格时进一步降低
if self.wall_adjacent >= 2 and self.dirty_adjacent == 0 and self.local_frontier_density < 0.05:
    cleaning_scale *= 0.5

cleaning_reward = 1.5 * cleaning_scale * float(self.cleaned_this_step)
streak_bonus = 0.15 * cleaning_scale * min(float(self.cleaned_this_step > 0), 1.0) * min(self.consecutive_clean_steps, 5)
```

**预期收益**

- 返航模式下 cleaning reward 从 2.25 降到 ~0.56 → return_stall_penalty(-0.08) 终于能生效
- 重访区域 cleaning reward 降低 → 减少路线交叉和重复覆盖
- 墙边无价值区域 cleaning reward 降低 → wall_hugging 自然减少
- planner 的 return/contract 建议变得更有吸引力 → divergence 率下降

**风险**

- 低：不删除任何 reward 项，只调整 scale
- 可能的副作用：返航路上丢弃过多脏格 → 但当前 return_stall_rate=54%，即使矫枉过正也是改善

**验证指标**

- `return_stall_rate`：应从 54% 降至 < 35%
- `wall_hugging_clean_floor_rate`：应下降
- `planner_policy_divergence`：应从 78% 降至 < 60%

---

### 方案 2：增强返航停滞惩罚（Reward 层）

**原理**

当前 `return_stall_penalty = -0.08` 是固定值。在 cleaning reward 被缩放后，仍然需要一个递进的惩罚来防止长时间停滞。

**改动位置**

`preprocessor.py` → `reward_process()` 中 stall_penalty 计算处。

**具体改法**

```python
if self.current_mode in (self.MODE_CONTRACT, self.MODE_RETURN):
    progress = float(self._last_target_distance - self.current_target_dist)
    return_progress_reward = Config.RETURN_PROGRESS_REWARD_SCALE * progress
    if progress <= 0.0:
        # 递增的停滞惩罚：连续无进展时增强
        base_stall = 0.12                     # 提高基础值 (原 0.08)
        escalation = 0.02 * min(self.return_stall_ema * 5, 3.0)  # 根据 EMA 递增
        stall_penalty = -(base_stall + escalation)
    # ... 其余不变
```

**预期收益**

- 返航停滞的代价从 -0.08 提升到 -0.12 ~ -0.18 → 与缩放后的 cleaning reward (~0.56) 更匹配
- 长期停滞会受到更强惩罚 → 减少"原地磨蹭"

**风险**

- 低：仅在 return/contract 模式生效，不影响正常清扫
- 可能副作用：agent 回程时走更直接的路径，但可能忽略回程路上的小型脏区 → 实际上是期望行为

**验证指标**

- `return_stall_rate`：进一步下降
- `return_efficiency_ratio`：应从 0.08~0.12 提升至 > 0.20

---

### 方案 3：低电量模式下的充电裕量感知（Reward 层）

**原理**

当前 reward 函数通过 `recoverability_reward`（delta-based）和 `astar_potential`（delta-based）间接反映电量安全性。但这些都是差分信号——只有在安全性变化时才有信号，稳定在低安全状态时没有持续压力。

map4 和 map8 的 200 步电池死亡案例显示：agent 可以在 charge_margin < 0 的状态下持续运行 60~70 步而不感到任何紧迫性。

**改动位置**

`preprocessor.py` → `reward_process()` 中，在现有 recover_reward 计算之后添加。

**具体改法**

```python
# --- 充电裕量持续压力 ---
charge_margin_pressure = 0.0
if not self.just_charged and self.current_mode not in (self.MODE_DEPART,):
    margin = self.charger_slack  # battery - astar_dist - reserve
    if margin < 0:
        charge_margin_pressure = -0.20  # 已经回不来了，强惩罚
    elif margin < 10:
        charge_margin_pressure = -0.08  # 裕量很低，给压力
    elif margin < 20 and battery_ratio < 0.4:
        charge_margin_pressure = -0.03  # 轻微压力

recover_reward += charge_margin_pressure
```

**预期收益**

- 当 charger_slack < 0 时（意味着可能回不到桩），agent 每步受到 -0.20 的持续惩罚
- 配合方案 1（降低 cleaning 奖励），agent 会更早触发 contract/return
- map4/map8 类型的短局电池死亡应显著减少

**风险**

- 低：仅在低裕量时生效
- 可能副作用：agent 可能过于保守，过早回充 → 但当前 battery_fail_rate=30%~40%（round1），过于保守远好于电池死亡

**验证指标**

- `battery_fail_rate`：Round 1 应从 30%~40% 降至 < 20%
- `charge_margin_now` 分布：负值占比应大幅下降
- `missed_charge_opportunity_rate`：应下降

---

## 五、不建议优先做的事

### 5.1 不建议新增独立惩罚项名目

"wall_hugging_penalty"、"stale_boundary_penalty"、"redundant_path_penalty" 等——这些都是在不改变 cleaning reward 支配性的前提下叠加反力。不仅增加调参复杂度，而且在 PPO 的 advantage 计算中，多个互相冲突的 reward 项会让 value function 更难拟合。

### 5.2 不建议 hard override planner

在 pathological 循环时强制接管策略输出会破坏 PPO 的 on-policy 假设。如果 reward 设计正确，policy gradient 自然会把循环行为的 advantage 降为负值。

### 5.3 不建议在当前阶段改变模型结构或 teacher weight

模型本身的表达能力不是瓶颈（它已经学会了很复杂的清扫策略）。问题在于 reward 信号告诉它错误的优先级。先修 reward，再看是否需要调整架构。

### 5.4 不建议改动 target/anchor 机制

现有方案 D 提议改变 target 选择的粘滞逻辑。但从 reward attribution 看，`suboptimal_target_hold` 收到 +2.2 正奖励——target 粘滞不是因为机制有缺陷，而是因为切桩的短期代价（cleaning reward loss）太高。修正 cleaning reward 后，agent 自然会在代价更低时切桩。

---

## 六、建议立刻验证的最小改动集

**只改 `preprocessor.py` 中的 `reward_process()` 方法，共 3 处，约 20 行代码。**

### 改动 1（核心）

在 `cleaning_reward` 和 `streak_bonus` 赋值前，加入上下文 scale 逻辑：

```python
cleaning_scale = 1.0
if self.current_mode in (self.MODE_CONTRACT, self.MODE_RETURN):
    cleaning_scale = 0.25
elif self.cur_visit_count >= 3:
    cleaning_scale = 0.4
elif self.cur_visit_count >= 2:
    cleaning_scale = 0.7
```

将 `cleaning_reward` 和 `streak_bonus` 乘以 `cleaning_scale`。

### 改动 2

将 `return_stall_penalty` 从 `-0.08` 改为 `-0.12`，并加入基于 `return_stall_ema` 的递增。

### 改动 3

在 `recover_reward` 之后，增加 `charge_margin_pressure` 项（margin < 0 时 -0.20，margin < 10 时 -0.08）。

### 验证流程

```bash
# 1. 修改 preprocessor.py（仅 reward_process）
# 2. 重新训练 500~1000 episode
# 3. 标准 benchmark
cd train
bash run_benchmark_parallel.sh <new_checkpoint> \
  --workers 4 --envs-per-worker 10 --max-wait 1800 --policy-mode eval
```

### 阶段一验收标准

| 指标 | 当前值 | 目标值 |
|------|--------|--------|
| return_stall_rate | 54~56% | < 35% |
| wall_hugging_clean_floor_rate | 2.5~2.8% | < 1.5% |
| planner_policy_divergence (per traj) | ~78% | < 60% |
| battery_fail_rate (round_1) | 30~40% | < 20% |
| missed_charge_opportunity_rate | 0.4~0.5% | < 0.2% |
| win_rate | 63~65% | ≥ 65%（不降） |

如果阶段一验收通过，再考虑：
- 多桩 A* 路径扩展（改动 Tier 2 结构性问题）
- Target 切换条件优化

---

## 七、改动层归属总结

| 方案 | 修改层 | 修改文件 | 代码量 | 风险 |
|------|--------|---------|--------|------|
| 方案 1：清扫奖励上下文缩放 | Reward | preprocessor.py:reward_process | ~8 行 | 低 |
| 方案 2：增强返航停滞惩罚 | Reward | preprocessor.py:reward_process | ~5 行 | 低 |
| 方案 3：充电裕量持续压力 | Reward | preprocessor.py:reward_process | ~8 行 | 低 |

**不需要修改：** feature（observation 构造无问题）、model（表达能力足够）、planner/expert（信号已存在只是被淹没）、target 机制（reward 修正后自然改善）、训练参数（PPO 超参数合理）。

---

## 八、补充：对 map4/map8 短局电池死亡的专项分析

这两个 200 步电池死亡案例值得单独分析：

- 两者均是 2 charger / 2000 step 场景
- 200 步即死意味着：agent 从初始位置出发后一直没找到桩
- `all_charger_known_path_count = 1`（map8）但 A* 路径可能经过未知区域
- `charge_margin_now` 在 72/200 步（map4）和 66/200 步（map8）时已为负值

根因分析：
1. 初始 depart 方向恰好远离最近桩（depart 阶段 20 步，方向随机性强）
2. 进入 harvest/expand 后，cleaning reward 主导 → 继续远离桩
3. `charge_margin` 变负后没有足够强的信号逼回
4. A* 路径经过未知区域 → 实际距离被低估 → 回不来

方案 3（charge_margin_pressure）可以直接缓解这个问题。更根本的改善需要：
- depart 阶段优先向已知桩附近探索
- 或在 `all_charger_known_path_count = 1` 时提高探索压力

但这是 Tier 2 问题，应在方案 1~3 验证通过后再处理。

---

## 九、结论

**当前策略的核心矛盾不是"缺少惩罚信号"或"planner 被忽略"，而是 cleaning reward 的绝对主导性使得所有其他信号都失效。**

修复路径应该是：**先把 cleaning reward 做上下文缩放，让已有的导航/安全信号能被 policy 感知到**。而不是在一个已经过强的 cleaning reward 之上叠加越来越多的反向惩罚——那只会让 reward 函数更复杂、value function 更难拟合、训练更不稳定。

建议的最小验证集只涉及 `reward_process()` 中约 20 行代码的修改，没有架构变动、没有新 feature、没有 hard override。这是"最少改动、最大收益"的路径。
