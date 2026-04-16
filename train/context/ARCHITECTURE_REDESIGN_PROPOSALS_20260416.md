# 网络架构改进方案: 形式化建模 + 多方案设计

> 日期: 2026-04-16
> 基于: v5.4 (WinRate 88.2%, battery death 85%占比)
> 约束: PPO on-policy, forward → [logits, value] (可增加辅助输出), ≤3h训练, 8方向动作空间不变

---

## 第一部分: 形式化问题建模

### 1.1 任务本质

本竞赛是一个**有限时域·资源受限·部分可观测·多目标序贯决策问题**（Resource-Constrained POMDP with Multiple Objectives）。可形式化表示为:

$$\max_\pi \; \mathbb{E}_\pi \left[ \sum_{t=0}^{T} \mathbf{1}_{\text{cleaned}}(s_t, a_t) \right]$$
$$\text{s.t.} \quad B_t > 0 \; \forall t \leq T, \quad \text{dist}(p_t, n_t^{(i)}) > 0 \; \forall i$$

其中 $T = \min(T_{\max}, T_{\text{death}})$。优化目标是最大化清扫格数（即 clean_score），硬约束是电池不归零（存活）和不碰撞 NPC。

这个问题的**核心难度**在于: 优化目标（清扫）和存活约束（充电）之间不是简单的 trade-off，而是**阶段性结构冲突**——任意时刻，agent 要么在清扫 (increasing score)，要么在充电 (zero score but extending horizon)，两者天然互斥。

### 1.2 状态空间规模与关键维度

| 维度 | 规模 | 描述 |
|------|------|------|
| 地图状态 | $3^{128 \times 128}$ | 每格三态 (障碍/已清/未清)。实际可达约 4000-6000 格 |
| 电池 | $[0, B_{\max}]$ | 连续整数, 典型 $B_{\max} \in [120, 720]$ |
| 位置 | $128 \times 128$ | agent 坐标 |
| NPC 位置 | $(128 \times 128)^{N_{npc}}$ | $N_{npc} \in [1,4]$, 运动模式未知 |
| 充电桩 | $(128 \times 128)^{N_{charger}}$ | $N_{charger} \in [1,4]$, 静态 |
| 时间步 | $[0, T_{\max}]$ | 关键: 剩余时间决定清扫价值 |

**可观测量**（当前实现 1597D）：

| 部分 | 维度 | 信息完备度 |
|------|------|------------|
| Local view 21×21×3 | 1323 | 高——但无时序关联 |
| Global memory 8×8×3 | 192 | 低——16×16 压缩丢失空间细节 |
| Scalar features 74D | 74 | 中——涵盖关键数值但缺乏时序 |
| Legal action 8D | 8 | 完整 |

**关键缺失信息**:
- 时序轨迹: 最近 K 步的位置、动作、电量变化趋势
- NPC 运动模式: 无法预测 NPC 下一步方向
- 高分辨率全局记忆: 8×8 的全局地图无法精确表达已清扫区域分布
- 充电效率历史: agent 不知道上一次充电的时机是否合理

### 1.3 动作冲突结构

Agent 在每个 timestep 面临多层动作冲突:

```
Layer 0: 安全过滤 ← filter_actions (dist≤3 禁止接近)
Layer 1: 生存 vs 效率 ← 充电 or 清扫
Layer 2: 探索 vs 利用 ← 去未知区 or 清已知脏区
Layer 3: 前进 vs 回避 ← 朝目标走 or 绕 NPC
```

这些冲突有不同的时间尺度:

| 冲突 | 时间尺度 | 当前处理方式 | 瓶颈 |
|------|----------|-------------|------|
| 安全 (NPC) | 1-5 步 | Expert filter_actions ✅ | 已解决 89% |
| 充电决策 | 50-200 步 | Expert logit_bias (3-8) ❌ | **主瓶颈**: bias 太弱 |
| 清扫路径 | 10-100 步 | RL 自学 ⚠️ | 缺乏全局规划 |
| 探索 vs 利用 | 全局 | 无显式机制 | 后期 revisit |

### 1.4 当前系统的结构性极限

**(1) GAE 信用衰减致命性**

GAE 衰减因子: $(\gamma \lambda)^k = 0.9405^k$

| 步数 k | 衰减到 | 含义 |
|--------|--------|------|
| 10 | 54.0% | 尚可 — NPC 回避信号可传播 |
| 30 | 15.7% | 弱 — 充电决策信号几乎不可见 |
| 50 | 4.7% | 极弱 — 等价于噪声 |
| 100 | 0.2% | 零 — 与随机无异 |

充电决策的典型 lookahead 是 50-200 步: 看到电池低 → 决定回充 → 走到充电桩 → 充电 → 离开。这整个序列的 GAE credit 只能传播 4.7% 到决策起点，**模型从奖励信号中根本学不到"何时该开始回充"**。

**(2) 奖励失衡的量化分析**

单局 1000 步的典型奖励分布:
- 清扫奖励总量: $\sim 1.5 \times 800 = 1200$（假设清 800 格）
- 电量消耗惩罚: urgency 三档 $(-0.3) \times 30 + (-0.6) \times 20 + (-1.2) \times 10 = -33$
- 充电奖励: $\sim 3.0 \times e \times 5 = 15$（5 次充电，效率 e ≈ 1.0）
- NPC 惩罚: 稀疏，$\sim -3.0 \times 5 = -15$

**清扫:充电 ≈ 60:1** — critic 几乎完全由清扫梯度主导。Value function 学到的是 "proximity to dirt" 而非 "battery safety margin"。

**(3) Expert-RL 耦合的结构悖论**

```
Expert 越强 → 模型不需要学充电 → 模型不学充电 → 必须依赖 Expert → 性能上限 = Expert 质量
```

v5.4 的 clean_prob 存储修复了梯度中毒 (v4 bug)，但没有解决**探索缺失**:
- Expert bias 3-8 在 logit 空间内被模型的清扫偏好 (+5) 部分抵消
- 73% 的电池死亡场景中，Expert bias 不足以把充电方向的采样概率提到足够高
- 模型在 Expert 保护下的行为分布里，永远看不到 "不充电 → 死亡" 这一因果链

**(4) Peak-Then-Decline 的动力学解释**

训练曲线反复观察到 10-20% 训练量处出现峰值后衰退:

```
阶段 1 (0-10%): 模型快速学到 "走向脏格" → 分数上升
阶段 2 (10-20%): 模型开始优化路径效率 → 达到峰值
阶段 3 (20%+): 模型过度优化清扫 → 忽视充电 → battery death 增加 → 分数下降
```

这是因为清扫梯度的量级是充电梯度的 60 倍。当模型 "学会清扫" 后，继续训练只会让它更关注清扫而更忽视充电。entropy floor 阻止了策略崩溃，但没阻止**价值函数偏移**（critic 越来越认为 "清扫 > 一切"）。

### 1.5 网络架构能解决什么 vs 不能解决什么

| 问题 | 架构能否解决 | 原因 |
|------|-------------|------|
| 时序记忆缺失 | **可以** — RNN/attention/frame-stacking | 给模型历史信息 |
| 空间规划能力弱 | **可以** — 更高分辨率 global map + attention | 更精确的全局视野 |
| 充电信用衰减 | **不能直接解决** — 需奖励/n-step 配合 | GAE 是训练算法问题，不是网络问题 |
| 奖励失衡 | **不能解决** — 需奖励重设计 | 信号本身不平衡 |
| Expert 耦合 | **部分** — dual critic 可分离存活/清扫信号 | 但根源在于 Expert 代替模型决策 |
| NPC 碰撞 | **不需要** — filter_actions 已足够 | 规则系统处理即可 |

### 1.6 理想信息需求 vs 实际可用

**理想观测 (oracle agent 所需)**:
1. 完整 128×128 地图 (所有格子当前状态) → 当前: 21×21 local + 8×8 global
2. NPC 运动轨迹预测 (下 10 步位置) → 当前: 仅当前位置
3. 最优充电时机 (基于剩余步数和脏格分布) → 当前: 无
4. 清扫覆盖进度热力图 → 当前: 仅有 visit_count 但未被充分利用
5. 每步动作的长期价值精确估计 → 当前: GAE (50+ 步后 ≈ 0)

**可立即提升的信息**:
- 全局地图分辨率: 8×8 → 16×16 (无需环境改动, preprocessor 内部修改)
- 时序特征: 最近 K 步的电量变化率、位置变化率、模式持续时间
- visit count 作为 CNN channel (当前仅在 cost map 中使用, 未输入网络)
- 充电效率历史 (上次充电的清扫收益/步数比)

---

## 第二部分: 方案设计

---

### 方案 A: "奖励外科手术" — 势函数整形 + 梯度隔离

**设计动机**: 在不修改网络架构的前提下，直接攻击两个最大瓶颈: (1) 充电信号被 GAE 衰减到 0, (2) 清扫:充电 = 60:1 的奖励失衡。利用 Ng et al. (1999) 的势函数整形定理保证策略最优性不变。

#### A1. 网络架构设计

**不变.** 继续使用当前 CNN+MLP，完全兼容 resume 加载。

```
输入 (1597D) → [不变的 CNN+MLP] → logits(8D) + value(1D)
```

参数量: 不变 (~800K)

#### A2. 输入特征设计

在 scalar features 中新增 6D 时序派生特征 (Preprocessor 中维护 ring buffer):

| 新特征 | 维度 | 计算方式 |
|--------|------|----------|
| `battery_delta_ema` | 1 | 最近 10 步电池变化的 EMA |
| `clean_rate_ema` | 1 | 最近 20 步清扫速率的 EMA |
| `time_since_last_charge` | 1 | 上次充电距今步数 / max_step |
| `movement_entropy` | 1 | 最近 10 步方向的 entropy (检测震荡) |
| `charge_round_count` | 1 | 累计充电次数 / 预估需充次数 |
| `remaining_steps_ratio` | 1 | 剩余步数 / max_step |

总特征: 1597 → 1603D

> 注: scalar_dim 74 → 80, FEATURE_LEN 1597 → 1603。影响 Config、Preprocessor、Model 三个文件。由于仅改 scalar_proj 输入维度 (82→88)，可以部分 resume: 对 scalar_proj.0.weight 做 pad-zero 初始化，其余层权重不变。

#### A3. 奖励设计

**核心变更: 势函数整形 (Potential-Based Reward Shaping)**

定义势函数:
$$\Phi(s) = \alpha \cdot \text{battery\_safety\_margin}(s)$$

其中:
$$\text{battery\_safety\_margin}(s) = \min\left(\frac{B_t}{B_{\max}}, 1.0\right) - \frac{d_{\text{charger}}}{B_t + 1}$$

势函数的直觉: 电量越高、离充电桩越近 → 势越高。电量低且离充电桩远 → 势很低。

整形奖励:
$$r_{\text{shaped}}(s, a, s') = \gamma \cdot \Phi(s') - \Phi(s)$$

取 $\alpha = 10.0$ 使整形信号量级 ≈ 0.1~0.3/步，与清扫信号同阶。

**Ng 定理保证**: 对任意 $\Phi(s)$，添加 $\gamma\Phi(s') - \Phi(s)$ 后，最优策略不变。这意味着我们可以放心地注入充电信号而不会扭曲最优行为。

**附加奖惩调整**:

| 奖励项 | 当前值 | 新值 | 理由 |
|--------|--------|------|------|
| `cleaning_reward` | 1.5 | $1.5 \times \max(0.4, B_t/B_{\max})$ | 低电时清扫价值递减 |
| `urgency_penalty` tier3 | -1.2 | -3.0 | 与死亡惩罚同量级 |
| `battery_death_penalty` (episode end) | -3.0 | -6.0 | 加强死亡信号 |
| `charge_reward` | $3.0 \times e$ | $5.0 \times e$ | 提高充电正反馈 |
| `potential_shaping` | 无 | $\gamma \Phi(s') - \Phi(s)$ | 新增密集充电信号 |

**预期效果**: 清扫:充电 从 60:1 降至 ~8:1。

#### A4. Expert 系统协调

**关键变更: 梯度隔离 + bias 增强**

```python
# algorithm.py: 当 sample 标记为 "expert_override" 时，policy_loss 权重为 0.1
if expert_override_mask[i]:
    policy_loss_weight = 0.1  # 不完全屏蔽，保留微弱学习信号
else:
    policy_loss_weight = 1.0
```

Expert bias 调整:

| 参数 | 当前值 | 新值 | 理由 |
|------|--------|------|------|
| emergency_bias | 100 | 100 (不变) | 已足够 |
| normal_bias | [3, 8] | [8, 15] | 确保 P(charge_dir) > 50% |
| emergency_threshold | ratio ≤ 0.10 | ratio ≤ 0.20 | 更早启动硬覆盖 |
| NPC 抑制 (dist≤4 不加 charge bias) | 是 | NPC dist ≤ 2 时才抑制 | 缩小抑制触发距离 |

#### A5. 训练成本估计

| 指标 | 当前 | 预估 |
|------|------|------|
| 参数量 | ~800K | ~801K (+600) |
| 单步推理 | ~28ms | ~28ms (无变化) |
| 单步训练 | ~28ms | ~29ms (reward shaping 计算) |
| 收敛时间 | 3小时训练3500步 | 同量级, 期望 peak 延后到 30-40% |
| 可 resume | ✅ 可 | scalar_proj 需 pad-zero |

#### A6. 风险点

1. **势函数系数 α 敏感**: 过大会使 agent 过度保守 (整局在充电桩旁), 过小无效。需要 grid search α ∈ [5, 10, 20]。
2. **清扫 reward 衰减可能降低初期学习速度**: 低电时 cleaning_reward 打折可能让 warmup 阶段偏慢。
3. **Expert bias 增强可能在 NPC 密集场景引入新的路径冲突**: 充电 bias 15 在 NPC 方向上会形成强拉力，与 filter_actions 的安全过滤可能产生 "想去又不能去" 的震荡。

#### A7. 实现复杂度

| 文件 | 改动 | 行数 |
|------|------|------|
| `preprocessor.py` | 新增势函数计算, 6D 时序特征, ring buffer | ~60 行 |
| `conf.py` | SCALAR_DIM, FEATURE_LEN, POTENTIAL_ALPHA | ~10 行 |
| `model.py` | scalar_proj 输入维度 82→88 | ~2 行 |
| `algorithm.py` | expert_override 梯度权重 | ~15 行 |
| `agent.py` | 传递 expert_override 标记到 SampleData | ~10 行 |
| `expert.py` | bias 范围调整 | ~5 行 |
| **总计** | | **~100 行** |

**实现优先级: ⭐⭐⭐⭐⭐ (最高)** — 风险最低, 效果最直接, 可立即部署。

---

### 方案 B: "时序增强感知" — 帧堆叠 + 高精度全局图 + 双头 Critic

**设计动机**: Agent 当前的两个信息缺陷——(1) 全局地图分辨率太低 (8×8, 16 倍压缩), (2) 完全没有时序记忆——是纯 MLP 无法学到长程充电策略的底层原因。本方案通过增强感知来提高决策质量。

#### B1. 网络架构设计

```
输入 (2853D)
  ├─ Local view 21×21×4 (新增 visit_count channel)
  │    → Conv(4→16)→ReLU→Conv(16→32)→ReLU→MaxPool→Conv(32→32)→ReLU
  │    → Flatten(32×10×10=3200)→FC(3200→256)
  │    → 256D
  │
  ├─ Global memory 16×16×6 (分辨率提升 + 帧堆叠)
  │    → Conv(6→16)→ReLU→Conv(16→32)→ReLU→MaxPool(2,2)
  │    → Conv(32→32)→ReLU
  │    → Flatten(32×8×8=2048)→FC(2048→128)
  │    → 128D
  │
  ├─ Scalar features 86D (80+6 temporal) + legal 8D
  │    → FC(94→64)→ReLU→FC(64→64)→ReLU
  │    → 64D
  │
  └─ Concat → 448D
       → FC(448→256)→ReLU→FC(256→128)→ReLU
       → 128D backbone
       ├─ Actor head: FC(128→8) → logits
       ├─ Survival critic: FC(128→64)→ReLU→FC(64→1) → V_survival
       └─ Cleaning critic: FC(128→64)→ReLU→FC(64→1) → V_clean
```

**关键架构变化**:

| 组件 | 当前 | 新 | 理由 |
|------|------|-----|------|
| Local encoder | 3 ch | 4 ch | +visit_count 通道, 告诉模型 "走过没" |
| Global memory | 8×8×3 | 16×16×6 | 分辨率 ×4, channel ×2 (当前帧 + 2 帧前 diff) |
| Global encoder | 2 conv, 1 FC | 3 conv + MaxPool, 1 FC | 处理更大输入 |
| Critic | 1 head | 2 heads | 分离存活价值和清扫价值 |
| Backbone output | 128 | 128 (不变) | |

参数量: ~800K → ~1.2M (增加 ~50%)

#### B2. 输入特征设计

**Local view (21×21×4)**:
- Channel 0: 障碍图 (不变)
- Channel 1: 已清扫图 (不变)
- Channel 2: 脏格图 (不变)
- Channel 3: **visit_count (归一化)** — `min(visit_count / 10.0, 1.0)`, 颜色越深 = 访问越多

**Global memory (16×16×6)**:
- Channel 0-2: 当前帧 explored / dirty / visit_heat (不变, 但分辨率 128→16 而非 128→8)
- Channel 3-5: **2 步前的 explored / dirty / visit_heat** → 让模型感知全局变化趋势

preprocessor 维护 `global_memory_prev` buffer (仅保留 1 帧历史), 每步更新:
```python
global_memory_stacked = np.concatenate([current_global, prev_global], axis=0)  # 6 channels
self.global_memory_prev = current_global.copy()
```

**Scalar features (86D)**:
- 原 74D + 方案 A 的 6D 时序特征 + 6D 新增:

| 新特征 | 维度 | 计算 |
|--------|------|------|
| `battery_delta_ema` | 1 | 最近 10 步电池变化 EMA |
| `clean_rate_ema` | 1 | 最近 20 步清扫速率 EMA |
| `time_since_last_charge` | 1 | 距上次充电步数 / max_step |
| `movement_entropy` | 1 | 最近 10 步方向 entropy |
| `charge_round_count` | 1 | 充电次数 / 预估需充次数 |
| `remaining_steps_ratio` | 1 | 剩余步数 / max_step |
| `charger_path_safety` | 1 | Expert 规划路径是否安全 (0/1) |
| `optimal_charge_battery` | 1 | Expert 计算的理想触发电量 / battery_max |
| `dirt_density_ahead` | 1 | 当前方向 10 格内脏格密度 |
| `global_clean_progress_delta` | 1 | 最近 20 步清扫进度变化 |
| `mode_duration_normalized` | 1 | 当前模式持续步数 / 50 |
| `estimated_rounds_left` | 1 | 剩余步数 / (电池最大 × 充电距离估计) |

Scalar 总维度: 74 + 12 = 86D

**特征总维度**: 21×21×4 + 16×16×6 + 86 + 8 = 1764 + 1536 + 86 + 8 = **3394D**

#### B3. 奖励设计

继承方案 A 全部奖励变更, 另加:

**双 Critic 的分离目标**:
- $V_{\text{clean}}$: 仅用清扫相关奖励训练 (cleaning_reward, efficiency_reward, CPS)
- $V_{\text{survival}}$: 仅用存活相关奖励训练 (urgency_penalty, charge_reward, potential_shaping, death_penalty)

总 value = $V_{\text{clean}} + V_{\text{survival}}$, 用于 GAE 计算。

分离的核心意义: **清扫 critic 不再被充电信号干扰, 充电 critic 也不被清扫信号淹没**。这本质上是把一个 "信号量级差 60 倍" 的单一回归问题拆成两个量级内聚的子问题。

**辅助损失: 存活预测**

额外添加二元分类头 (从 backbone 128D):
```
death_predictor: FC(128→32)→ReLU→FC(32→1)→Sigmoid
```

Label: 该 episode 最终是否 battery death (1=死, 0=活), 作为常量 label 回传到每一步。

$$L_{\text{aux}} = \text{BCE}(\hat{p}_{\text{death}}, y_{\text{death}})$$

辅助损失系数: 0.1。作用: 强迫 backbone 学到 "当前状态距离电池死亡有多远" 的表征。

#### B4. Expert 系统协调

继承方案 A 的全部 Expert 调整 (bias 增强, 梯度隔离)。

额外: Expert 输出的 `charger_path_safety` 和 `optimal_charge_battery` 作为 scalar 特征**直接注入网络**。这让模型可以 "看到" Expert 的判断，并学习何时与 Expert 一致 vs 何时有更好的选择。

长期目标: 随着模型自主学到充电策略, Expert bias 可逐步退火 (bias *= 0.999/step), 最终让模型独立决策。

#### B5. 训练成本估计

| 指标 | 当前 | 预估 |
|------|------|------|
| 参数量 | ~800K | ~1.2M (+50%) |
| 特征维度 | 1597D | 3394D |
| 单步推理 | ~28ms | ~35ms (+25%, 主因: Global encoder 变大) |
| 单步训练 | ~28ms | ~38ms (+36%, 特征+参数均增) |
| 收敛时间 | 3h → 3500 步 | 3h → ~2800 步 |
| 可 resume | — | ❌ 需重新训练 (架构改变太大) |

#### B6. 风险点

1. **特征维度翻倍, 训练效率降低**: 3394D 输入可能需要更大 batch 来稳定梯度。当前 batch=4, 可能需提至 8。
2. **16×16 全局图需要 preprocessor mean_pool 精度**: 从 128→16 的压缩 (8 倍) 比 128→8 精确得多, 但仍可能丢失单格级别信息。
3. **双 Critic 的奖励分配需要精心设计**: 如果分配不当 (某些奖励项同时影响两个 critic), 可能比单 critic 更差。
4. **帧堆叠引入 temporal stale 风险**: 第一步的 prev_global 是 zero-initialized, 可能在 episode 起始产生异常预测。

#### B7. 实现复杂度

| 文件 | 改动 | 行数 |
|------|------|------|
| `preprocessor.py` | 16×16 global, visit channel, 帧堆叠, 12D 新特征 | ~120 行 |
| `conf.py` | 所有维度参数, GLOBAL_MEMORY_SIZE=16, LOCAL_VIEW_CHANNELS=4 等 | ~25 行 |
| `model.py` | global_encoder 重写, dual critic head, death_predictor, forward 返回 4 输出 | ~80 行 |
| `algorithm.py` | dual critic loss, aux death loss, reward 拆分 | ~60 行 |
| `agent.py` | death label 传递, SampleData 扩展 | ~25 行 |
| `definition.py` | SampleData 新增 death_label, survival_reward, clean_reward | ~20 行 |
| **总计** | | **~330 行** |

**实现优先级: ⭐⭐⭐⭐ (高)** — 中等风险, 但直击信息缺陷。建议在方案 A 验证后实施。

---

### 方案 C: "GRU 记忆核心" — 循环时序网络 + N-Step Return + 辅助任务

**设计动机**: 彻底解决时序记忆缺失问题。当前 MLP 的每一步决策都是独立的——它不知道自己上一步做了什么、不知道过去 10 步的运动模式、不知道 NPC 的运动趋势。GRU 让模型拥有真正的工作记忆。

#### C1. 网络架构设计

```
输入 (1603D, 同方案 A 特征)
  ├─ Local view 21×21×3 (不变)
  │    → [不变的 local encoder] → 256D
  │
  ├─ Global memory 8×8×3 (不变)
  │    → [不变的 global encoder] → 64D
  │
  ├─ Scalar 80D + legal 8D
  │    → [不变的 scalar_proj] → 64D
  │
  └─ Concat → 384D
       → FC(384→256)→ReLU→FC(256→128)→ReLU
       → 128D pre-GRU
       ↓
       GRU(input=128, hidden=128, num_layers=1)
       → 128D post-GRU
       ├─ Actor head: FC(128→8) → logits
       ├─ Critic head: FC(128→1) → value
       ├─ Aux battery predictor: FC(128→32)→ReLU→FC(32→1)→Sigmoid → B̂(t+50)
       └─ Aux death predictor: FC(128→32)→ReLU→FC(32→1)→Sigmoid → p̂_death
```

**关键**: GRU hidden state (128D) 在 episode 内 step-by-step 传递, episode 开始时 zero-init。

GRU 的作用:
- 隐式编码 "最近 30-50 步的轨迹摘要"
- 自动捕捉 NPC 运动模式 (是追逐还是巡逻?)
- 记住 "我正在赶往充电桩" 的内部状态
- 检测 "我在原地打转" 的异常模式

#### C2. KaiwuDRL 框架兼容性方案

**核心问题**: KaiwuDRL 训练数据管线 (aisrv → reverb → learner) 是按个体 SampleData 发送, 不保证时序顺序。直接用 GRU 处理 random batch 等价于 "每步 zero-init 的 GRU" 即退化为 MLP。

**解决方案: Truncated BPTT with Stored Hidden States**

1. **推理端 (aisrv)**:
   - Agent 维护 `self.gru_hidden` (128D tensor), episode 开始时 zero-init
   - 每步 `predict()` 传入 hidden, 得到 new_hidden + logits + value
   - 将 hidden state 打包进 SampleData.obs (附加 128D → obs 变 1731D)

2. **训练端 (learner)**:
   - 收到 batch 后, 从 obs 中拆出 128D stored_hidden
   - 用 stored_hidden 作为 GRU 输入的初始状态 (detach, 不回传梯度)
   - 计算单步 GRU 输出用于 loss

```python
# 训练时
stored_hidden = obs[:, -128:]         # 取出存储的 hidden
obs_clean = obs[:, :-128]             # 原始 1603D 观测
pre_gru = backbone(encode(obs_clean)) # 128D
gru_out, _ = gru(pre_gru.unsqueeze(0), stored_hidden.unsqueeze(0))
gru_out = gru_out.squeeze(0)          # 128D
logits = actor(gru_out)
value = critic(gru_out)
```

这种方案是 **Stored-State BPTT**: 推理时完整传递 hidden, 训练时用存储的 hidden state 做 1-step update。虽然训练梯度只回传 1 步而非完整序列, 但 GRU 在推理时实际运行了完整的 episode 长度序列。

> R2D2 (Kapturowski et al., 2019) 使用类似策略处理分布式训练中的过期 hidden state 问题。

#### C3. 奖励设计

继承方案 A 全部奖励变更, 加上:

**N-Step Return 替代 GAE**:

用 $n=20$ 的 n-step return 替代 GAE:
$$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})$$

n-step 的优势: 对于 20 步内的信号, 完整保留（不乘 $\lambda^k$ 衰减）。当 $n=20$ 时, 16 步外的充电奖励保留 $0.99^{16} = 85.1\%$, 而 GAE 仅保留 $0.9405^{16} = 36.7\%$。

代价: n-step return 方差更大 (不做 $\lambda$-加权平均), 可能增加训练噪声。

**辅助任务**:
- $L_{\text{battery\_predict}} = \text{MSE}(\hat{B}_{t+50}, B_{t+50} / B_{\max})$：预测 50 步后电量
- $L_{\text{death}} = \text{BCE}(\hat{p}_{\text{death}}, y_{\text{death}})$：预测 episode 是否 battery death

辅助 loss 系数各 0.1。

总损失:
$$L = L_{\text{PPO}} + 0.1 \cdot L_{\text{battery}} + 0.1 \cdot L_{\text{death}}$$

#### C4. Expert 系统协调

继承方案 A。额外变化:

随着 GRU 使模型具备时序记忆, 模型有望自主学到充电时机。建议训练后 50% 阶段逐步降低 Expert bias:

```python
# episode_cnt > 400 后, bias 逐步退火
decay = max(0.3, 1.0 - (episode_cnt - 400) / 600)
expert_bias *= decay
```

目标: 最终 Expert 只负责 filter_actions (安全过滤), 不再干预充电策略。

#### C5. 训练成本估计

| 指标 | 当前 | 预估 |
|------|------|------|
| 参数量 | ~800K | ~870K (+9%, GRU 128×128×3 gates ≈ 66K) |
| 特征维度 | 1597D | 1731D (1603+128 hidden) |
| 单步推理 | ~28ms | ~30ms (+7%, GRU 一步计算量小) |
| 单步训练 | ~28ms | ~32ms (+14%, GRU + aux losses) |
| 收敛时间 | 3h → 3500 步 | 3h → ~3200 步 (轻微降低) |
| 可 resume | — | ❌ 需重新训练 |

#### C6. 风险点

1. **Stored-State BPTT 的 hidden state staleness**: 训练时的 hidden state 来自可能已过时的模型版本 (aisrv 到 learner 有延迟)。PPO 的 clip ratio 可以部分缓解, 但极端情况下可能引入训练不稳定。
2. **N-step return 方差**: 当 $n=20$ 时, return 的方差显著高于 GAE($\lambda$=0.95), 可能需要降低学习率 (5e-5 → 3e-5)。
3. **GRU hidden state 与 on-policy 的紧张关系**: PPO 要求 behavior policy ≈ current policy。GRU 的 hidden state 编码了历史, 如果模型更新后 hidden state 的 "含义" 改变, 会导致 on-policy 假设轻微违反。
4. **辅助任务标签的正确性**: battery_predict 需要 "50 步后的电量" 作为标签, 但 episode 最后 50 步没有 label。需要特殊处理 (mask 或用当前值填充)。
5. **框架改动风险**: SampleData 增加 128D 会影响 reverb buffer 大小和 learner batch 的内存占用。

#### C7. 实现复杂度

| 文件 | 改动 | 行数 |
|------|------|------|
| `model.py` | 添加 GRU, aux heads, 修改 forward 签名 | ~60 行 |
| `agent.py` | 维护 gru_hidden, 打包到 obs, predict 改造 | ~50 行 |
| `algorithm.py` | 拆分 hidden, aux losses, n-step return | ~80 行 |
| `definition.py` | SampleData 扩展, sample_process 改用 n-step | ~60 行 |
| `preprocessor.py` | 同方案 A 的特征改动 | ~60 行 |
| `conf.py` | GRU 相关参数, N_STEP, FEATURE_LEN | ~15 行 |
| **总计** | | **~325 行** |

**实现优先级: ⭐⭐⭐ (中)** — 理论潜力最高, 但工程风险大。建议在 A+B 组合验证后作为 "冲刺" 方案。

---

### 方案 D: "空间注意力 + 模式条件策略" — Cross-Attention + Soft Option

**设计动机**: 当前模型的决策完全依赖固定感受野 (local 21×21 + global 8×8)。在清扫末期, 关键脏格可能分散在地图各处, 模型需要**主动关注**特定区域。同时, 清扫/充电/避敌三种行为模式差异极大, 用单一策略同时表达三种模式导致 "策略内冲突"。

#### D1. 网络架构设计

```
输入 (2853D)
  ├─ Local view 21×21×4 (含 visit_count)
  │    → Conv(4→16)→ReLU→Conv(16→32)→ReLU→MaxPool→Conv(32→32)→ReLU
  │    → Feature map: 32×10×10
  │    → Flatten + FC(3200→256)
  │    → 256D local_feature
  │
  ├─ Global memory 16×16×3
  │    → Conv(3→16)→ReLU→Conv(16→32)→ReLU→MaxPool(2,2)
  │    → Feature map: 32×8×8
  │    → Reshape: 64 positions × 32 channels (序列化)
  │    → 64-length sequence of 32D tokens
  │
  ├─ Cross-Attention: local_feature attends to global tokens
  │    Q: FC(256→64) from local_feature
  │    K: FC(32→64) from global tokens
  │    V: FC(32→64) from global tokens
  │    Attention(Q, K, V) → 64D attended_global
  │
  ├─ Scalar 86D + legal 8D → FC(94→64)→ReLU→FC(64→64) → 64D
  │
  └─ Concat [256 + 64 + 64] → 384D
       → FC(384→256)→ReLU→FC(256→128)→ReLU
       → 128D backbone
       │
       ├─ Mode gate: FC(128→3)→Softmax → π_mode ∈ {clean, charge, evade}
       │
       ├─ Actor_clean:  FC(128→64)→ReLU→FC(64→8) → logits_clean
       ├─ Actor_charge: FC(128→64)→ReLU→FC(64→8) → logits_charge
       ├─ Actor_evade:  FC(128→64)→ReLU→FC(64→8) → logits_evade
       │
       ├─ Final logits = Σ_m π_mode[m] × logits_m  (加权混合)
       │
       └─ Critic: FC(128→1) → value
```

**Cross-Attention 的直觉**: Local encoder 产生 "我在哪里, 周围有什么" (Query)。Global encoder 产生 "地图各处是什么状态" (Key-Value)。Attention 让模型可以**查询**特定方向/区域的全局脏格分布, 而不是被固定的 mean-pooling 压缩掉。

**Mode-Conditioned Policy 的直觉**: 三种模式 (clean/charge/evade) 各自有一个完整的 actor head。Mode gate 决定 "当前应该处于什么模式"。Final logits 是三个模式的加权混合。这等价于一个**软分层策略**——模型无需用单一 FC 同时表达三种截然不同的行为模式。

> 这比硬分层 (Option Framework) 更适合端到端训练, 因为所有参数通过 PPO 梯度联合优化, 不需要额外的 option termination/initiation 学习。

#### D2. 输入特征设计

与方案 B 的特征方案相同, 但**不做帧堆叠**:
- Local: 21×21×4 (加 visit_count channel) = 1764D
- Global: 16×16×3 (不堆叠, attention 代替了 temporal 需求) = 768D
- Scalar: 86D (含 12D 新增时序特征)
- Legal: 8D

总输入: 1764 + 768 + 86 + 8 = **2626D**

#### D3. 奖励设计

继承方案 A 全部变更。

额外: **模式一致性辅助奖励**

当 Expert 在 return_mode 时, 如果 mode gate 的 charge 概率 < 0.5, 给予 $-0.1$ 惩罚:
$$r_{\text{mode}} = -0.1 \times \mathbb{1}[\text{Expert\_in\_return} \wedge \pi_{\text{mode}}[\text{charge}] < 0.5]$$

这个辅助信号帮助 mode gate 快速学到 "现在应该是充电模式" 的判断。

注意: 这不是势函数整形, 会轻微改变最优策略。但效果是 "对齐模型意图与 Expert 知识", 在初始训练阶段有利于加速收敛。后期可退火至 0。

#### D4. Expert 系统协调

继承方案 A 的 bias/梯度隔离调整。

额外: 当 mode gate 输出 charge > 0.7 时, Expert 自动启用 return mode (即使 heuristic 尚未触发):

```python
if mode_probs[1] > 0.7 and not expert.return_mode:
    expert.force_return("model_requested")
```

这实现了 **模型 → Expert 的反向通信**: 模型判断该充电了, Expert 提供精确路径执行。当前单向 Expert → 模型的架构变成了双向协作。

#### D5. 训练成本估计

| 指标 | 当前 | 预估 |
|------|------|------|
| 参数量 | ~800K | ~1.5M (+87%) |
| Attention 计算 | 无 | Q(256)×K(64×64)→softmax→V: ~0.6ms/step |
| 3×actor head | 1 head | 3 heads (+30% actor 参数) |
| 单步推理 | ~28ms | ~38ms (+36%) |
| 单步训练 | ~28ms | ~42ms (+50%) |
| 收敛时间 | 3h → 3500 步 | 3h → ~2500 步 (-29%) |
| 可 resume | — | ❌ 需重新训练 |

#### D6. 风险点

1. **Attention 过拟合**: 只有 64 个 global tokens, attention 可能快速过拟合到固定的 "充电桩位置"。需要足够的 map randomization。
2. **Mode gate 退化**: soft mode gate 可能退化为 "永远选同一个模式"。需要 mode entropy 正则 (类似策略 entropy):
   $$L_{\text{mode\_ent}} = -0.005 \cdot H(\pi_{\text{mode}})$$
3. **3 actor head 的梯度共享不足**: 三个 head 独立学习, 但它们共享 backbone。可能出现 "充电 head 需要的 backbone 特征 vs 清扫 head 需要的特征" 冲突, 导致 backbone 表征退化。
4. **训练吞吐降低 30%**: 在 3 小时限制内, 只能跑约 2500 步。如果收敛需要 3000+ 步, 可能不够。
5. **Mode-Expert 双向通信的时序问题**: 模型 mode gate 在 predict 时输出, Expert 在同一个 predict 调用中使用。需要仔细设计调用顺序。

#### D7. 实现复杂度

| 文件 | 改动 | 行数 |
|------|------|------|
| `model.py` | Cross-attention, mode gate, 3×actor head, forward 重构 | ~120 行 |
| `preprocessor.py` | 16×16 global, visit channel, 12D 新特征 | ~80 行 |
| `conf.py` | 维度参数, mode 相关参数 | ~20 行 |
| `algorithm.py` | mode entropy loss, mode consistency aux loss | ~40 行 |
| `agent.py` | mode_probs 传递给 Expert | ~25 行 |
| `expert.py` | force_return 接口, 反向通信 | ~20 行 |
| **总计** | | **~305 行** |

**实现优先级: ⭐⭐ (低)** — 架构最前沿但风险最高。适合作为 "不需额外工程复杂度" 的 B 方案验证失败后的备选。

---

## 第三部分: 方案对比与推荐路线

### 3.1 多维对比矩阵

| 维度 | 方案 A | 方案 B | 方案 C | 方案 D |
|------|--------|--------|--------|--------|
| **核心理念** | 奖励外科手术 | 增强感知 | 时序记忆 | 空间注意力+分层 |
| **解决的瓶颈** | 奖励失衡, GAE 衰减 | 信息缺陷, 信号隔离 | 时序依赖, 全流程 | 空间推理, 模式冲突 |
| **参数增量** | +600 (+0.1%) | +400K (+50%) | +70K (+9%) | +700K (+87%) |
| **推理延迟** | 28ms (±0) | 35ms (+25%) | 30ms (+7%) | 38ms (+36%) |
| **训练吞吐** | 3500步/3h | 2800步/3h | 3200步/3h | 2500步/3h |
| **可 Resume** | ✅ 部分 | ❌ 不可 | ❌ 不可 | ❌ 不可 |
| **理论上限** | 中 (不改架构) | 高 | 最高 (真正记忆) | 高 (但工程风险大) |
| **实现复杂度** | ~100 行 | ~330 行 | ~325 行 | ~305 行 |
| **失败风险** | 低 | 中 | 高 | 中高 |
| **预期 WinRate 增益** | 88→91-93% | 88→92-95% | 88→93-97% | 88→91-94% |

### 3.2 推荐实施路线

```
阶段 1: 方案 A (1-2天)
  └─ 验证势函数整形 + 奖励再平衡的效果
  └─ 期望: WinRate 88% → 91%+, battery death 降至 50%以下
  └─ 如果 ≥91%: 进入阶段 2
  └─ 如果 <91%: 调参 α, 检查势函数设计

阶段 2: 方案 A + B 组合 (2-3天)
  └─ 在 A 的奖励基础上, 叠加 B 的架构改进
  └─ 16×16 global, visit channel, dual critic
  └─ 期望: WinRate 93%+
  └─ 如果 ≥93%: 微调, 准备提交
  └─ 如果 <93%: 考虑方案 C 冲刺

阶段 3 (可选): 方案 C 冲刺 (3-4天)
  └─ GRU + stored-state BPTT
  └─ 只在 A+B 组合已接近极限时尝试
  └─ 期望: WinRate 95%+
```

**不推荐**直接从方案 C 或 D 开始, 因为:
1. 方案 A 的改动量最小, 可以最快验证 "奖励失衡是否是主因";
2. 如果奖励重设计就能到 91%+, 换网络可能是不必要的;
3. 方案 C/D 一旦失败, 调试成本远高于 A/B。

### 3.3 各方案的 "杀手级改动" (单独验证价值最高的子改动)

如果时间极度紧张, 只能做一个改动, 优先级:

1. **势函数整形** (方案 A): 1 小时实现, 直接解决 GAE 衰减 → **最高 ROI**
2. **Expert bias 增强到 [8,15]** (方案 A): 30 分钟实现, 直接降低 battery death
3. **16×16 全局图** (方案 B): 全局感知精度 ×4 → 更好的路径规划
4. **Dual critic** (方案 B): 信号隔离 → 充电 critic 不被清扫淹没
5. **Visit count 作为 CNN channel** (方案 B): 模型直接 "看到" 哪里走过 → 减少 revisit

---

## 附录: 关键参数推荐值

### 方案 A 推荐参数
```python
# conf.py
POTENTIAL_ALPHA = 10.0          # 势函数系数
CLEANING_REWARD_DECAY = True     # 低电时清扫奖励衰减
URGENCY_TIER3 = -3.0            # 严重低电惩罚 (原 -1.2)
BATTERY_DEATH_PENALTY = -6.0     # 电池死亡终局惩罚 (原 -3.0)
CHARGE_REWARD_BASE = 5.0         # 充电奖励基值 (原 3.0)
EXPERT_NORMAL_BIAS_MIN = 8       # 常规 bias 最小值 (原 3)
EXPERT_NORMAL_BIAS_MAX = 15      # 常规 bias 最大值 (原 8)
EXPERT_EMERGENCY_RATIO = 0.20    # 紧急阈值 (原 0.10)
EXPERT_NPC_SUPPRESS_DIST = 2     # NPC 抑制触发距离 (原 4)
GRADIENT_ISOLATION_WEIGHT = 0.1  # Expert override 样本的 policy loss 权重
```

### 方案 B 新增参数
```python
# conf.py
GLOBAL_MEMORY_SIZE = 16          # 全局地图分辨率 (原 8)
GLOBAL_MEMORY_CHANNELS = 6      # 含帧堆叠 (原 3)
LOCAL_VIEW_CHANNELS = 4          # 含 visit_count (原 3)
SCALAR_DIM = 86                  # 原 74 + 12 新增
DUAL_CRITIC = True
AUX_DEATH_LOSS_COEF = 0.1
SURVIVAL_REWARD_KEYS = ["urgency_penalty", "charge_reward", "potential_shaping", "death_penalty"]
CLEANING_REWARD_KEYS = ["cleaning_reward", "efficiency_reward", "cps_reward"]
```

### 方案 C 新增参数
```python
# conf.py
GRU_HIDDEN_DIM = 128
GRU_NUM_LAYERS = 1
N_STEP_RETURN = 20               # 替代 GAE
AUX_BATTERY_PREDICT_HORIZON = 50
AUX_BATTERY_LOSS_COEF = 0.1
AUX_DEATH_LOSS_COEF = 0.1
EXPERT_BIAS_DECAY_START_EPISODE = 400
EXPERT_BIAS_DECAY_RATE = 0.999
```

---

## 第四部分: 更深一层的综合判断

### 4.1 我对当前问题的最终定性

如果只用一句话概括当前系统的根本矛盾，那就是：

> **模型被要求用单步即时奖励去学习一个本质上需要 50-200 步规划的生存决策，同时这个决策又被 Expert 在外部半接管，最终形成“模型没学会、Expert 又不够强”的结构性夹缝。**

这意味着当前问题并不适合被理解为“某个 reward 系数不对”或者“某个卷积层太浅”。真正的问题位于 4 个层面同时失配：

1. **任务定义失配**：赛题本质是“带硬约束的多目标决策”，但当前训练目标更接近“最大化累计清扫 shaped reward”。
2. **时间尺度失配**：充电决策需要长期前瞻，但 PPO + GAE 主要擅长近程 credit assignment。
3. **表示能力失配**：当前前馈网络没有内部记忆，无法形成“计划进行中”的隐变量表示。
4. **控制接口失配**：Expert 决定何时充电，RL 决定如何清扫，两者没有共享统一的价值分解和训练信号。

这也是为什么当前系统会呈现出非常典型的现象：

- entropy 问题能被 floor 暂时压住，但这不等于学会了更好的策略；
- 碰撞问题通过 Expert 规则可以显著修复，但电池死亡依旧存在；
- 模型会在训练早期快速学会“更会扫”，随后因为过度偏向即时清扫而逐渐丢失长期生存意识；
- 最优 checkpoint 常常出现在训练前 10-20% 区间，而不是越训越好。

### 4.2 这个赛题的真正解法应该长什么样

一个理想的高水平策略，不会把“清扫”和“回充”当作同一个层次上的同质动作选择，而是至少包含下面 3 层隐式能力：

1. **生存评估层**：判断当前电池安全边际是否足够继续清扫。
2. **阶段切换层**：在 clean / charge / evade 三种模式之间切换。
3. **局部执行层**：在给定模式下输出 8 方向动作。

当前模型只有第 3 层，且第 1、2 层主要由 Expert 在代码外部硬编码承担。这不是不能工作，但上限会被 Expert 的规则质量锁死。

### 4.3 为什么“完美策略”需要的不是更多参数，而是更好的状态表示

一个完美策略至少需要以下信息：

1. **未来 20-50 步的 NPC 风险趋势**，而不是只看当前距离。
2. **所有充电桩的真实可达代价和竞争风险**，而不是只看最近一个的几何距离。
3. **过去一段时间的覆盖轨迹**，用来减少 revisit 和 orbiting。
4. **当前清扫路线与回充路线之间的机会成本**，即“如果我现在多扫 15 步，会不会失去安全回充窗口”。
5. **任务阶段上下文**，例如“我已经回充了 8 步，还差 15 步到桩”，“我刚离开充电桩，不应再次进入 return mode”。

当前特征空间虽然有 visit_count、charger_slack、nearest_npc_dist 等局部近似量，但这些量不是在统一的时序表征中被组织的，而是散落在 74D scalar 和 coarse map 里，模型需要自己从单帧中“猜出”阶段。这是当前最核心的认知瓶颈。

### 4.4 哪些问题优先级最高

从收益 / 成本比来看，我认为优先级严格如下：

1. **先修训练目标和 Expert 接口**，因为这是当前最大 ROI。
2. **再修观测表示**，尤其是全局分辨率和时序信息。
3. **最后才考虑更复杂的策略架构**，例如 attention 或软分层。

原因很简单：如果 reward 和 credit 还是错的，换更强网络只会更快地把错误目标学得更好。

---

## 第五部分: 对各方案的深层考量与取舍理由

### 5.1 为什么方案 A 是最值得先做的

方案 A 的价值不在于“它最先进”，而在于它先解决了最不应该继续带着前进的结构错误：

1. **让充电从稀疏终局信号变成密集局部信号**。
2. **让 Expert 从“隐形训练控制器”退回“明确边界的安全网”**。
3. **尽量保住现有 checkpoint 和训练产物，不浪费 v5.2/v5.4 这条线的经验**。

这条路线的强项是：

- 最快验证“当前 88.2% 的天花板到底是 reward 问题还是表示问题”；
- 可以直接在现有基础上继续训练，不需要承担全重训风险；
- 如果只做 1-2 个高价值子改动，也有较大概率把 battery death 从主瓶颈打到次瓶颈。

它的弱点也很明确：

- 它不会根治“模型内部没有记忆”；
- 它仍然依赖 Expert 作为 safety net；
- 它更像是“把当前系统调到合理区间”，而不是构建下一代架构。

### 5.2 为什么方案 B 比方案 D 更适合作为第二阶段主线

从表面看，方案 D 的 cross-attention + mode-conditioned policy 更优雅，也更接近“结构正确”。但工程上我仍然更推荐先做 B，再做 D，原因有三点：

1. **B 改的是信息带宽**，而 D 改的是决策结构。信息不足时，先扩带宽通常比先加复杂控制器更稳。
2. **B 的失败更容易诊断**：如果 16×16 global + visit channel + dual critic 后仍无提升，可以相对明确地判断“问题不只在观测”。而 D 如果失败，很难知道是 attention、gate、mode mixing 还是训练不稳导致。
3. **B 与现有代码风格更连续**：本仓库当前就是 CNN + MLP actor-critic，B 只是在这个范式上加容量和信号分离；D 相当于引入新的策略组织方式，调试成本更高。

因此，从工程演进角度看：

- A 是修正方向；
- B 是扩大视野；
- C 才是补上记忆；
- D 更像是有一定风险的“跃迁式尝试”。

### 5.3 为什么方案 C 的理论上限最高，但不适合作为第一步

GRU 方案是我认为**从问题本体出发最合理**的网络升级，因为当前很多失败模式都明显需要时序记忆：

- 反复经过已清区域后继续绕圈；
- 充电路径被局部阻断后忘记“这条路不通”；
- NPC 在视野边缘反复逼近，模型只看当前帧无法判断趋势；
- 当前 return mode 是由 Expert 内部状态维持，而不是模型自己的隐状态。

但它不适合作为第一步，理由不是理论，而是基础设施：

1. KaiwuDRL 当前采样和 learner 管线天然是单步样本流，不是 sequence-first 设计。
2. 你当前训练吞吐并不宽裕，一旦重训 GRU 发现实现不稳定，时间成本会很高。
3. Stored-state BPTT 虽然可行，但它是工程妥协方案，不是理想的 full-sequence training。

所以我对 C 的判断是：

- **战略价值最高**；
- **战术风险也最高**；
- **应该在 A 或 A+B 把目标函数校正后再上**。

### 5.4 为什么我暂时不建议“彻底去掉 Expert”

这是一个很容易产生误判的点。很多人看到 Expert 和 RL 冲突，就会自然想到“既然冲突，那就去掉 Expert”。但对当前系统，这几乎一定是错误方向。

原因有三条：

1. 现有训练数据已经证明，模型不会自主充电，至少在当前 reward 和观测设计下不会。
2. 充电本身就是长周期规划问题，而当前算法和网络都不擅长这个尺度。
3. Expert 现在的主要问题不是“存在本身”，而是“作用边界没划清”。

因此更合理的方向不是 remove Expert，而是 **shrink Expert**：

- 保留它处理极端安全问题；
- 逐步撤销它对常规回充的主导权；
- 把它变成 teacher / planner / fallback，而不是常态控制器。

---

## 第六部分: 可直接执行的实验矩阵

### 6.1 总体实验原则

为了避免再次出现“多个改动叠加后无法归因”的问题，建议每一轮实验坚持三条原则：

1. **单轮只验证一个主假设**。
2. **保留与当前最优 checkpoint 可比的评估协议**。
3. **明确 stop condition，避免在明显退化的路线继续浪费训练时间**。

### 6.2 实验阶段划分

#### Phase 1: 目标函数修正实验

目标：验证 battery death 是否主要来自 reward / bias 失衡，而不是表示不足。

实验组：

1. Baseline: 当前 v5.4。
2. A1: 势函数整形 only。
3. A2: Expert bias 增强 only。
4. A3: 势函数整形 + bias 增强。
5. A4: 势函数整形 + bias 增强 + gradient isolation。

主要指标：

1. WinRate。
2. Battery death 占比。
3. Charge count 分布。
4. AvgCS 与 MinCS。
5. 低电量窗口内的 charge-mode 触发率。

成功标准：

- WinRate ≥ 91%；
- battery death 占失败原因比例下降到 50% 以下；
- AvgCS 不下降超过 5%。

若失败，则说明目标函数修正不足以支撑下一步，必须更快进入表示升级。

#### Phase 2: 观测表示增强实验

目标：验证当前失败是否主要来自“看不清”和“记不住”。

实验组：

1. B1: 仅 global 8→16。
2. B2: B1 + visit_count local channel。
3. B3: B2 + dual critic。
4. B4: B3 + death aux loss。

主要指标：

1. revisit_rate。
2. 在回充阶段的路径冗余长度。
3. broad 配置下的 battery death 变化。
4. map 6 / map 2 等难图的稳定性。

成功标准：

- broad profile 的 fail rate 显著下降；
- MinCS 提升明显；
- revisit 和 orbiting 轨迹减少。

#### Phase 3: 时序记忆冲刺实验

目标：验证“有记忆的策略”是否能显著改善 return timing 和回充执行一致性。

实验组：

1. C1: GRU + GAE。
2. C2: GRU + n-step return。
3. C3: C2 + battery predict aux。
4. C4: C3 + Expert bias decay。

主要指标：

1. low-battery 窗口下的正确回充起始率。
2. 进入 return mode 后成功到达 charger 的比例。
3. battery death 与 collision 共同变化。
4. hidden-state 引导下的 mode 连贯性。

成功标准：

- WinRate ≥ 94%；
- battery death 降到接近碰撞同量级；
- broad / low battery / low charger 配置不再极端脆弱。

### 6.3 每轮训练建议时长与早停规则

在当前训练资源下，不建议每个实验都完整跑满 3 小时后再判断。更高效的做法是分三级判定：

1. **20 分钟早筛**：看是否明显退化，例如 battery death 激增、AvgCS 快速下滑。
2. **60 分钟中筛**：看 WinRate 和失败原因结构是否进入正确方向。
3. **180 分钟终筛**：只对通过前两轮的候选继续跑满。

建议 early stop 条件：

- WinRate 较 baseline 下滑超过 5%；
- AvgCS 下滑超过 10%；
- battery death 占比不降反升；
- entropy 异常抬高并导致 charge_count 爆炸。

### 6.4 重点日志与可解释性指标

下一轮实验建议固定记录以下日志，不再只看最终 clean_score：

1. `return_trigger_step_ratio`：episode 中首次进入 return mode 的归一化时刻。
2. `battery_margin_histogram`：进入回充时的 slack 分布。
3. `charge_commit_success_rate`：进入回充后是否成功到桩。
4. `charge_abort_count`：进入回充后又切回清扫的次数。
5. `revisit_heatmap_score`：高访问区域占比。
6. `mode_switch_count`：模式切换频率，过高说明策略不稳定。

这些指标会比单纯 WinRate 更快揭示“改动是在修结构，还是只是碰巧刷分”。

---

## 第七部分: 文件级实施顺序建议

### 7.1 如果执行方案 A

建议改动顺序：

1. 先改 `preprocessor.py` 中的 reward 和新特征。
2. 再改 `conf.py` 使特征维度对齐。
3. 再改 `model.py` 的 scalar 输入维度。
4. 再改 `expert.py` 的 bias 区间。
5. 最后改 `algorithm.py` 做梯度隔离。

这样做的原因是 reward 和 feature 先落地后，模型维度和训练逻辑才有稳定接口。

### 7.2 如果执行方案 B

建议改动顺序：

1. 先只把 global 8×8 提到 16×16，确认 preprocessor 输出稳定。
2. 再加 local visit_count channel。
3. 再改模型 encoder。
4. 最后引入 dual critic 与 loss 拆分。

不要一次性把 global、visit、dual critic、aux loss 同时上完，否则很难判断问题来自哪一层。

### 7.3 如果执行方案 C

建议先做一个最小可运行版：

1. 先接入 GRU，但不改 return，仍用 GAE。
2. 确认 hidden state 在 episode 内正确传递。
3. 再切 n-step return。
4. 最后才加 auxiliary heads。

GRU 方案最怕的是一开始把所有复杂度都叠满，导致出问题后难以定位是 hidden state、loss 还是数据流。

---

## 第八部分: 最终结论

### 8.1 结论摘要

我对当前系统的判断是：

1. **当前 88.2% 的瓶颈仍主要是结构性目标错位，而不是单纯网络容量不足。**
2. **最值得优先做的不是“更复杂的模型”，而是“把充电决策重新纳入一个能被训练信号正确支持的框架”。**
3. **中期最值得投入的架构升级是时序记忆，而不是更花哨的 attention。**

### 8.2 最推荐的两条路线

如果你要一条最稳、最现实、最适合当前竞赛周期的路线：

1. **主线**：方案 A。
2. **下一阶段**：方案 A + B。

如果你准备开一条真正的“下一代模型”线：

1. **首选重训方案**：方案 C。
2. **备选高风险方案**：方案 D。

### 8.3 最不建议的误区

1. 不建议继续只调 reward 系数而不改结构接口。
2. 不建议在当前目标函数未修正前直接堆更大网络。
3. 不建议现在彻底移除 Expert。
4. 不建议同时启动 3 条高复杂度重训线，当前训练预算不支持大范围并行试错。

### 8.4 一句话建议

> **先把“模型为什么应该去充电”这件事讲清楚，再去提升“模型如何更聪明地行动”。**

这是当前系统最缺的一步，也是最有可能把 WinRate 从 88% 推到 91-93%，再进一步打开 95% 空间的关键前提。
---

## 第二部分：架构改动方案设计

基于上述对MDP破缺、信度分配衰减以及优化视界错位的形式化分析，以下提供三套具体的架构与特征升级方案。

### 方案一：势能导向与多帧堆叠 (The "Memory & Potential" Baseline)

**定位**：最具性价比的结构修补，针对“无记忆导致的原地打转”和“奖励稀疏导致的不懂回充”两大痛点。

1. **架构设计**：
   - 保持现有 CNN + FC 的基本前馈(Feed-forward)主干。
   - **电量门控(Battery-Gating)**：在进入 Actor 和 Critic 之前，提取网络的一层中间特征 $h$，计算门控特征 $h' = h \odot (1 + f(battery\_ratio))$，让网络显式区分高低电量子空间。
2. **输入特征设计**：
   - **Frame Stacking**：将 Local View 的输入通道从 `[3, 21, 21]` 扩展为 `[12, 21, 21]`（即缓存并堆叠最近 4 帧的局部观测）。
   - **Visited Map（轨迹热图）**：在 Local View 中增加 1 个 Channel，记录过去 50 步的移动轨迹（当前格值为1，每过一步衰减0.02），打破局部盲区。
   - **精确相对向位**：在 Scalar 中直接注入“最近充电桩的直线距离”和“相对角度的正弦/余弦值”。
3. **奖励函数设计 (Reward Shaping)**：
   - 取消极度稀疏的“充上电才给分”机制。
   - **A* 势能奖励**：令 $D_t$ 为 A* 算法算出的到达最近充电桩的真实路径距离。在低电量模式（<30%）下，每走一步给予 $R_{pot} = \alpha (D_{t-1} - D_t)$ 的奖励。只要往充电桩靠近，每一步都得正分。
4. **动机**：用 Frame Stacking 和 Visited Map 在空间中硬拼出时间记忆，解决死胡同打转；用单步势能差（Potential Reward）跨越 GAE 的衰减极限，把几十步之后的回充收益拉平到每一步。
5. **训练成本估算**：
   - 参数量增加 < 2%，由于只是多几个通道，单步前向时间估计从 28ms 升至 ~32ms。
   - 训练收敛速度保持一致（~2-3小时可验证）。
6. **风险点**：Reverb Replay Buffer 内存占用可能会由于特征变厚而增加；Visited Map 和 Stacking 需要在特征构建逻辑中维护状态缓存。
7. **实现复杂度**：低（改动 ~150 行）。主要修改 `preprocessor.py` 的状态存储与计算，`model.py` 更改输入通道数。
8. **与 Expert 的配合**：由于势能奖励的存在，RL 将自主学会靠向充电桩，因此可以将 Expert Logit Bias 逐步调低，转为仅在电量极度危急时作为兜底接管。

### 方案二：价值解耦双头 Critic (The "Multi-Head Critic" Architecture)

**定位**：从损失函数和价值评估层面强行解耦“清扫”与“生存”的数值打架问题。

1. **架构设计**：
   - 双头 Critic：提取 Backbone 后，保留 1 个 Actor 输出流，但将 1 个 Critic 拆解为两个并行的全连接头：
     - $V_{clean}(s)$：预测未来的清扫收益。
     - $V_{survive}(s)$：预测未来的存活收益（及回充概率）。
   - 计算 Advantage 时动态融合：$A = w(B_t) \cdot A_{clean} + (1 - w(B_t)) \cdot A_{survive}$，其中权重 $w$ 随电量 $B_t$ 下降而急剧降低。
2. **输入特征设计**：
   - 维持当前 1597D 输入基本不变，可引入方案一的“精确相对向位”。
3. **奖励函数设计**：
   - 将现有 Reward 强行劈开送入两条 GAE 管道：
     - $R_{clean}$ 仅包含扫灰得分（+1.0、+1.5）。
     - $R_{survive}$ 包含每步存活分（+0.01），电量衰减分（-0.05），成功充电巨分（+10.0），死亡惩罚（-50.0）。
4. **动机**：当前 Critic 极度容易被庞大的清扫得分淹没，导致它对低电量死亡的预测极度不敏锐。分离 Critic 使网络对两类完全不同量级的信号分别建立精确的值函数。
5. **训练成本估算**：
   - 网络仅增加一个平行的 FC 层分支，前向与反向时间基本不变。
   - `algorithm.py` 中 Advantage 计算开销增加约 10%。
6. **风险点**：双路 Critic 的 Loss 加权和 GAE 合并可能会引发训练曲线早期的剧烈波动；融合权重 $w(B_t)$ 的超参数极为敏感，需细致调校。
7. **实现复杂度**：中（改动 ~250 行）。需要修改 `model.py` 输出维度，修改 `agent.py` 的打包逻辑，并在 `algorithm.py` 重写 PPO 损失函数，确保向后兼容 Kaiwu 框架的通信流。
8. **与 Expert 的配合**：完美契合。当 $V_{survive}$ 骤降时，可以直接用此信号作为触发 Expert 接管的硬阈值，比看绝对电量更具前瞻性。

### 方案三：时序上下文与空间注意力 (The "GRU-lite + Spatial Attention" Solution)

**定位**：结构最强，上限最高，直接为网络赋予内生记忆和空间规划突破能力。

1. **架构设计**：
   - **Coordinate Spatial Attention**：在 Local Encoder 后接入一个简单的空间注意力模块。利用充电桩的相对坐标生成一个 Mask，对 Local 局部视图（例如32通道特征图）的特定通道和区域进行加权高亮。
   - **循环神经网络层 (GRU)**：将 Attention 后的 Local特征、Global特征和 Scalar特征拼接为 512 维隐向量，送入单层 GRU (hidden_size=256)。
   - GRU 的隐状态 $h_t$ 随每一个 Episode 的重置而归零。
2. **输入特征设计**：
   - 仅需单帧观察（摒弃冗余的特征堆叠），额外增加 $Action_{t-1}$ 的 One-hot 编码输入给 GRU，帮助网络建立动作到状态变迁的因果关联。
3. **奖励函数设计**：
   - 基本保持现状。可引入“同态打转惩罚”（如果连续出现互逆动作，如左→右→左，给予中等强度的惩罚）。
4. **动机**：纯正的 POMDP 解法。GRU 能够记住那些“滑出局部视野边界”的墙壁分布与充电桩方位，结合注意力机制打破局部死角卡死的僵局。
5. **训练成本估算**：
   - RNN 模块在 GPU 上带来的计算延迟更明显，单步计算时间可能上升 20%-30%。
6. **风险点**：KaiwuDRL 框架本身对 RNN hidden state 的截断拼接、重置及 Reverb 数据回放阶段的 Batch/Sequence 排布的兼容性是一大挑战。如果在 Linux/Docker 的数据流水线中已有瓶颈，增加复杂张量传递极易引发进程卡死或 OOM。
7. **实现复杂度**：极高。需要改动模型图，重构 `agent.py` 对环境 `state` 的打包装箱逻辑，并且要求底层框架支持 RNN 的 BPTT 训练链路。
8. **与 Expert 的配合**：不仅不冲突，还可以将 Expert 的当前 Logit 或指令直接作为一个 Feature 丢给 GRU 观察，让网络学会“此时此刻是否该采纳 Expert 意见”。

---

## 结论与动作建议 (Executive Summary)

考虑到你目前处于从 Windows 向 Linux/Docker 迁移、且数据流水线 (Reverb/Data Fetch) 在压力下已有瓶颈的情况，**必须规避基础设施层面的重构债务（方案三）**。同时，由于单次训练耗时已经长达 1 小时级别，必须保证迭代的稳妥性。

🏆 **首选推荐：方案一（势能导向 + 轨迹热图）**

- **实施门槛最低**：只需在 `preprocessor.py` 内部维护简单的 NumPy Array（记录过去的路径位置），并在构建特征词典时 `np.concatenate` 层叠起来，绝大部分修改仅在业务代码内发生，对 Kaiwu 框架的 Buffer 与 Loss 零侵入。
- **见效最直接**：单步 A* 势能奖励（Potential Reward）可以立刻将“回充策略学习”从 GAE 的超长步限制中解救出来。配合上 Visited Map，能解决你专家审查阶段发现的那 56% 的充电中途打转和磕碰的死亡案件。

如果你决定实施【方案一】，我们可以立即动手修改 `feature/preprocessor.py` 和 `model/model.py` 这两个关键文件。
