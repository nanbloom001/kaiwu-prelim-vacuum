# 独立评估：是否从头训练 — 从 Reward 结构到课程策略的系统性诊断

> 评审人：Claude (外部 RL 专家视角)  
> 输入文档：`TRAINING_STATUS_AND_REWARD_REDESIGN_BRIEF_20260418.md`  
> 对照基线：`CLAUDE_REVIEW_20260417.md`（前序 reward attribution 分析）  
> 评审日期：2026-04-18  

---

## 〇、总结论（先说结论再展开）

| 问题 | 判断 |
|------|------|
| 是否应从头训练 | **是，但条件不是"推翻一切"，而是"修好 reward 再从 preload 冷启动"** |
| 当前问题的首要来源 | **reward 结构本身**（不是课程、不是 preload、不是 planner） |
| lite benchmark 分层是否过于保守 | **逻辑正确，但门槛设置对强 checkpoint 过严** |
| 哪些机制应保留 | 双 value head、context-aware cleaning scale、preload、课程框架、teacher 可靠性门控 |
| 哪些机制应推翻 | cleaning/streak 的基础量级、return stall 惩罚量级、课程 warmup→blend 门槛中对 return_stall 的依赖权重 |

---

## 一、当前训练到底卡在哪——三因素归因

### 1.1 因素拆解

当前策略呈现出典型的"高存活、低效率"局部最优：

| 指标 | 当前值 | 健康目标 | 差距 |
|------|--------|----------|------|
| win_rate | 0.625~0.85 | ≥0.75 | 表面不差 |
| broad_win_rate | 0.38~0.44 | ≥0.65 | **差距大** |
| avg_clean_per_step (CPS) | 0.39~0.47 | ≥0.55 | **显著低** |
| avg_charge_count | 23~26 | 10~15 | **2 倍过高** |
| avg_remaining_charge | 137~199 | 30~80 | **4 倍过高** |
| mode_usage_expand | ≈0.001 | ≥0.10 | **接近零** |
| mode_usage_contract | ≈0.63~0.82 | 0.25~0.40 | **2 倍过高** |
| return_stall_rate | 0.55~0.66 | ≤0.30 | **2 倍过高** |
| planner_policy_divergence_rate | 0.83~0.87 | ≤0.25 | **4 倍过高** |
| return_efficiency_ratio | 0.10~0.16 | ≥0.50 | **极低** |

这组数据呈现出一个一致的画面：**模型学会了"频繁返桩、保守存活"，但没有学会"高效覆盖、规划清扫"。**

### 1.2 归因权重

我的判断是三因素叠加，但权重不同：

| 因素 | 权重 | 理由 |
|------|------|------|
| **Reward 结构性错位** | **60%** | cleaning reward 绝对支配（28:1 比值）是根因，所有行为偏差都在被正奖励强化 |
| **课程起点过保守** | **20%** | lite benchmark 把成熟 checkpoint 判回 warmup → 80% anchor+mild → 模型得不到足够 broad 压力 → 保守策略在简单环境中被进一步固化 |
| **Planner 与 Policy 长期偏离** | **20%** | divergence_rate=85% 说明 planner guidance 信号在 reward 中的权重太弱（alignment=0.03，divergence=-0.06），无法对抗 cleaning 的正反馈 |

**核心判断：reward 是"为什么会这样"的主因，课程是"为什么修不回来"的助力，planner divergence 是结果而非原因。**

### 1.3 为什么续训无法修复

观察当前 120 局 / 19k+ step 的训练窗口：

1. **env_total_score 在缓慢上升**（230→367），说明 PPO 梯度在工作
2. **但 algorithm.reward 长期偏负**（-31 到 +9 的宽区间），说明 reward 函数的期望行为与 env 评分所奖励的行为之间存在系统性偏差
3. **entropy 维持在 1.91~2.00**，没有收敛趋势，说明策略分布仍然很散

当 reward 结构本身在鼓励保守行为时（因为清扫正奖励远超任何导航惩罚），续训只会让 PPO 更坚定地学习"找到下一个脏格"而忽略长期规划。value head 学到的是"只要继续清扫就有正 return"的错误基线，新的 penalty 信号很难逆转已经固化的 value landscape。

---

## 二、四个核心问题的明确回答

### Q1: 是否应从头训练

**是。但不是"扔掉一切"，而是"修好 reward → preload 冷启动 → 让 PPO 在干净的 reward 信号下重新学习"。**

理由：
1. 当前 value network 已经编码了"cleaning 压倒一切"的错误价值观。在不修 reward 的情况下续训等于在固化偏差。
2. 即使修了 reward，当前 value head 的 bootstrap 估计与新 reward 的 scale/结构不一致，续训初期会产生大量 advantage 估计噪声。
3. preload 冷启动（保留 policy network 权重，重置 optimizer state）可以保留策略网络的有用知识（如地图理解、基础导航），同时让 value head 在新 reward 下重新拟合。

**具体建议**：
- 用当前最好的 checkpoint 作为 preload 起点（保留 policy weights）
- 重置 optimizer state（momentum, variance estimates）
- 让 value heads 在新 reward 下从头学习 value function
- 这不是从随机初始化重训，而是"reward 修复后的 fine-tune 冷启动"

### Q2: Reward 应如何围绕三大目标重新组织

#### 目标 1：存活率（Survivability）

当前存活率问题不是"惩罚不够"，而是"正奖励太强"。模型已经有 win_rate=0.65~0.85，但存活策略极其低效（频繁充电 23 次/局，剩余电量 199）。

**修改方向**：不需要增加新的存活惩罚项，需要做的是降低 cleaning reward 的量级，让现有的 charge_margin_pressure、late_contract_penalty、return_stall_penalty 等信号能在 advantage 计算中显现出来。

| 修改 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| cleaning_reward 基础系数 | 1.5 | **0.6** | 降到与 charge_reward(3.0*eff≈0.6~1.5)、return_stall_penalty(-0.12~-0.18) 同量级 |
| streak_bonus 系数 | 0.15/step | **0.06/step** | 同比例缩减，保持与 cleaning 的相对比例 |
| per-step reward clip | [-5, +5] | **[-3, +3]** | 收窄 clip 范围，避免充电/清扫峰值主导 |

**关键洞察**：reward 缩放是一个全局操作，所有现有的 penalty/bonus 项都不需要改——它们的绝对值没问题，问题在于 cleaning 项太大把它们淹没了。

#### 目标 2：清扫效率（CPS / Efficiency）

当前 CPS = 0.39~0.47，目标 ≥ 0.55。问题不是"没有清扫"，而是"清扫路线低效、大量时间浪费在返航和充电上"。

| 修改 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| 过度充电惩罚 | **无** | **-0.15 × max(0, charge_count/step_no - 0.015)** | 每步充电频率超过 1.5% 时开始惩罚，抑制频繁补能 |
| coverage_efficiency bonus | **无** | **+0.10 × (coverage_efficiency_20 - 0.5) when >0.5** | 鼓励高效覆盖（独立格/步比 > 0.5） |
| remaining_charge_penalty (终局) | **无** | 不加 | 终局奖惩在 episode 级别，PPO step-level reward 无法直接利用 |

**注意**：我不建议引入太多新的 reward 项。reward 项越多，credit assignment 越难。核心操作是"降低 cleaning 绝对量级"，让现有的 navigate/return/charge 信号浮出水面。

#### 目标 3：预先规划与规律清扫（Structured Coverage）

这是最难用 reward 直接塑造的目标。当前 expand 使用率 ≈ 0 的根本原因是 mode inference 的阈值设置过松——`CONTRACT_BATTERY_RATIO = 0.35` 和 `PREPARE_RETURN_SLACK_THRESHOLD = 12.0` 使得模型在电量还很充裕时就进入 contract，而 contract 模式下 explore/frontier 奖励被压到 0.25x。

| 修改 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| CONTRACT_BATTERY_RATIO | 0.35 | **0.28** | 延后进入 contract 的电量阈值 |
| PREPARE_RETURN_SLACK_THRESHOLD | 12.0 | **8.0** | 降低"准备返航"的距离余量，让模型有更多 expand 时间 |
| explore_reward 在 expand 中的 scale | 0.03 | **0.06** | 在 expand 模式下加倍鼓励探索 |
| planner alignment/divergence | 0.03/-0.06 | **0.08/-0.15** | 让 planner 信号在 advantage 中可见（尤其 cleaning 降到 0.6 后，0.15 的 divergence penalty 约等于 cleaning 的 25%） |

**结构性观点**：真正的"规律清扫"需要 policy 学会一个隐式的区域划分策略。这不太可能仅靠 reward shaping 做到——但可以通过降低 cleaning 的支配性，让 planner 的 guidance 信号发挥作用（planner 已经做了 A* 路径规划和桩选择，只是 policy 一直在否决它）。

### Q3: lite benchmark 与课程分层是否过于保守

**分层逻辑正确，但门槛对"已有能力、返航不够好"的 checkpoint 过于严格。**

当前 lite benchmark 结果：
- `completed_rate = 1.0` ✓
- `battery_fail_rate = 0.0` ✓  
- `collision_fail_rate = 0.0` ✓
- `return_stall_rate = 0.6614` ✗ (需要 ≤ 0.40)
- → **判级 warmup**

问题在于 `_recommended_initial_stage()` 的"blend"条件中没有 `return_stall_rate` 检查，但 `_meets_s0_exit()`（warmup→blend 推进条件）却要求 `return_stall_rate ≤ 0.40`。这意味着：

1. lite benchmark 判了 warmup
2. 进入训练后，要在 warmup 里攒够 3000 step + return_stall ≤ 0.40 才能升级
3. 但在 warmup 配置下 80% 的局是 anchor+mild → 保守策略会被进一步固化 → return_stall 不会自然改善
4. 形成**死循环**：保守策略 → 高 return_stall → 卡在 warmup → 更保守

**具体建议**：

```
# lite benchmark: 当 completed=1.0 且 battery_fail=0 且 collision_fail=0 时
# 即使 return_stall 高，也应判为 blend（而非 warmup）
# 因为这样的 checkpoint 不需要 warmup 的安全性保护，它需要的是更多 harder 场景压力
```

修改 `_recommended_initial_stage()`：

| 条件组 | 当前判级 | 建议判级 | 理由 |
|--------|----------|----------|------|
| completed≥0.70, battery≤0.10, collision≤0.05, broad_win≥0.65, stall≤0.40 | robust | robust | 不变 |
| completed≥0.55, battery≤0.22, collision≤0.10 | blend | blend | 不变 |
| **completed≥0.90, battery≤0.05, collision≤0.05, stall>0.40** | warmup | **blend** | **新增：高存活但 stall 高的 checkpoint 应进 blend 而非退回 warmup** |
| 其他 | warmup | warmup | 不变 |

同时修改 warmup→blend 推进条件（`_meets_s0_exit`）：将 `return_stall_rate ≤ 0.40` 放宽到 `≤ 0.50`，或加入一个备选条件 `win_rate ≥ 0.80 and battery_fail ≤ 0.10`。

### Q4: 哪些机制应保留、哪些应推翻

#### 明确保留

| 机制 | 理由 |
|------|------|
| **双 value head** (value_clean, value_survive) | 清扫/存活的 GAE 分离是对的，不应合并。问题在 reward 量级不均，不是 head 分离有误 |
| **Context-aware cleaning scale** | CLEANING_RETURN_SCALE=0.25、CLEANING_REVISIT 等机制方向完全正确，是当前代码中最有价值的设计 |
| **Preload 恢复** | 统一的 learner/aisrv 初始化，比旧 resume 更可靠 |
| **课程框架** | 4 stage + profile 权重 + 行为指标门槛的设计是合理的，只需调参 |
| **Teacher 可靠性门控** | mode_reliable、target_reliable 等条件化的 teacher mask 避免了低质量 teacher 信号 |
| **A* 充电路径规划** | expert.py 中的路径规划逻辑是正确的，只是 policy 不跟 |
| **route_anchor 粘滞机制** | 防止频繁切桩的机制方向对，只需让 sticky_penalty 在 reward 中可见 |
| **概率清洗与 fallback** | 修复了 NaN 概率导致的 aisrv 崩溃，必须保留 |
| **多分项评分体系** | resume_readiness vs submission_score 的分离有价值 |

#### 应该推翻 / 重大修改

| 机制 | 原因 | 建议 |
|------|------|------|
| **cleaning_reward = 1.5** | 绝对支配一切信号，是保守局部最优的根因 | 降至 0.6 |
| **streak_bonus = 0.15** | 与 cleaning 叠加后更加主导 | 降至 0.06 |
| **return_stall 在课程中的一票否决权** | 高 stall 的 checkpoint 被永远锁在 warmup | 降低权重，加备选路径 |
| **reward clip [-5, +5]** | 过宽，允许充电峰值(3.0×eff)主导 | 收窄到 [-3, +3] |
| **planner alignment 量级** | 0.03 / -0.06 在 cleaning=1.5 下完全不可见 | 提至 0.08 / -0.15（在 cleaning=0.6 下约占 25%） |
| **CONTRACT_BATTERY_RATIO 和 PREPARE_RETURN_SLACK_THRESHOLD** | 过早触发 contract 导致 expand≈0 | 分别降至 0.28 和 8.0 |

#### 不确定，建议观察

| 机制 | 理由 |
|------|------|
| **过度充电惩罚** | 理论上有价值，但新增 reward 项增加 credit assignment 负担。建议先看 cleaning 降低后 charge_count 是否自然下降 |
| **charge_reward = 3.0** | 绝对值看起来高，但降低 cleaning 后它的相对比例会更合理。如果 charge_count 仍然高，再降到 2.0 |
| **lite benchmark 的样本量** | 当前只跑 4 局。样本太少 → 方差大。建议增加到 8~12 局 |

---

## 三、Reward 结构偏差的量化证据

### 3.1 单步 reward 量级对比（当前值）

在一个"正常清扫步"中，各 reward 项的典型绝对值：

| 项 | 典型值/step | 相对于 cleaning 的比例 |
|----|-------------|----------------------|
| cleaning_reward | +1.50 | 100% (基线) |
| streak_bonus | +0.15~0.75 | 10~50% |
| explore_reward | +0.01~0.12 | 0.7~8% |
| frontier_reward | +0.01~0.15 | 0.7~10% |
| return_stall_penalty | -0.12~-0.18 | 8~12% |
| planner_divergence_penalty | -0.06 | 4% |
| charge_margin_warn_penalty | -0.03 | 2% |
| charge_margin_low_penalty | -0.10 | 7% |
| anchor_consistency_reward | +0.05 | 3% |
| idle_penalty | -0.10 | 7% |

**结论**：除了 cleaning 和 charge (一次性 3.0×eff)，所有其他 reward 项都在 cleaning 的 10% 以内。在 PPO 的 advantage 计算中，这些信号被 cleaning 的方差完全淹没。

### 3.2 修正后的量级对比（建议值 cleaning=0.6）

| 项 | 典型值/step | 相对于 cleaning 的比例 |
|----|-------------|----------------------|
| cleaning_reward | +0.60 | 100% (新基线) |
| streak_bonus | +0.06~0.30 | 10~50% |
| return_stall_penalty | -0.12~-0.18 | **20~30%** ← 可见了 |
| planner_divergence_penalty | -0.15 | **25%** ← 可见了 |
| charge_margin_low_penalty | -0.10 | **17%** ← 可见了 |
| late_contract_penalty | -0.35 | **58%** ← 非常显著 |
| astar_potential | ±0.35 | **58%** ← 非常显著 |

**修正后，所有导航/安全信号的比例从 <10% 提升到 20~58%，PPO 的 advantage 估计将首次能"看到"这些信号。**

### 3.3 当前 algorithm.reward 为什么长期偏负

文稿中记录 aisrv 日志显示 `algorithm.reward` 常见 -31 到 +9。这个值是 episode 累积 reward，不是 per-step reward。

在每步 clip 到 [-5, +5] 的情况下，1000 步的理论范围是 [-5000, +5000]。但实际 reward 在 [-31, +9]，说明：
- 每步平均 reward 约在 -0.03 到 +0.01 之间
- cleaning 的正奖励（~1.5/step × 清扫概率）大部分被 survival 侧的惩罚抵消了
- **但惩罚主要来自 npc_penalty(-3.0×risk)、stuck_penalty、charge_margin 等定额惩罚，而不是来自"引导模型改进行为"的 shaping 信号**

这正是问题所在：reward 函数产生了大量噪声性惩罚（NPC 距离近就扣分，无论是否真的有碰撞风险），但有意义的导向性信号（return 进度、planner 对齐）被 cleaning 淹没。

---

## 四、推荐的重训方案

### 4.1 最小改动集（8 处参数修改 + 1 处逻辑修改）

**reward 参数修改（conf.py，6 处）**：

```python
# 核心缩放
cleaning_reward_base = 0.6        # 原 1.5
streak_bonus_base = 0.06          # 原 0.15

# planner alignment 增强
PLANNER_ALIGNMENT_REWARD = 0.08   # 原 0.03
PLANNER_DIVERGENCE_PENALTY = 0.15 # 原 0.06

# 延后 contract 进入
CONTRACT_BATTERY_RATIO = 0.28     # 原 0.35
PREPARE_RETURN_SLACK_THRESHOLD = 8.0  # 原 12.0
```

**reward clip 修改（preprocessor.py，1 处）**：

```python
reward_total = float(np.clip(gain_reward + recover_reward, -3.0, 3.0))
# 组件 clip 也收窄：
"reward_clean": float(np.clip(gain_reward, -3.0, 3.0)),
"reward_survive": float(np.clip(recover_reward, -3.0, 3.0)),
```

**课程分层修改（lite_benchmark_bootstrap.py，1 处逻辑）**：

```python
def _recommended_initial_stage(metrics):
    # ... 现有 robust 条件不变 ...
    # ... 现有 blend 条件不变 ...
    # 新增：高存活但 stall 高的 checkpoint 应进 blend
    if (
        float(metrics.get("completed_rate", 0.0)) >= 0.90
        and float(metrics.get("battery_fail_rate", 1.0)) <= 0.05
        and float(metrics.get("collision_fail_rate", 1.0)) <= 0.05
    ):
        return "blend"
    return "warmup"
```

### 4.2 训练流程

1. 修改上述参数
2. 选择当前最佳 checkpoint 作为 preload 起点
3. 重置 optimizer state（如果 framework 支持）
4. 启动训练
5. 前 20 局观察：
   - `algorithm.reward` 的均值是否上升（预期：从 -20 级别回到 -5~+5）
   - `mode_usage_expand` 是否从 0 开始上升
   - `charge_count` 是否开始下降
6. 40 局窗口观察 curriculum 是否能升级到 blend

### 4.3 不建议的修改

| 修改 | 为什么不建议 |
|------|------------|
| 引入 5+ 新 reward 项 | 增加 credit assignment 难度，调参空间爆炸 |
| Planner hard override | 违反 on-policy 假设，PPO gradient 会矛盾 |
| 移除 dual value head | 当前 clean/survive 分离是正确的设计 |
| 移除 teacher guidance | teacher 方向正确，只是 reward 量级让 policy 忽略它 |
| 从随机初始化训练 | 当前 policy network 有大量有用知识（地图理解、基础导航），不应丢弃 |
| 修改 PPO 超参数 | LR=3e-5, CLIP=0.15, ENTROPY_FLOOR=0.15 都是合理值，不是瓶颈 |

---

## 五、关于"过度保守求活"vs"高质量规划清扫"的最终判断

### 5.1 当前状态确实是"过度保守求活"

证据链：
1. `mode_usage_expand ≈ 0` → 模型从不主动探索新区域
2. `mode_usage_contract ≈ 0.63~0.82` → 大部分时间在"收缩"模式
3. `avg_charge_count = 23~26` → 每局充电 23 次 ≈ 每 40 步充一次
4. `avg_remaining_charge = 137~199` → 结束时电量还剩 70~100%
5. `return_stall_rate = 0.55` → 一半的返航步骤没有进度
6. `planner_divergence = 0.83~0.87` → 几乎完全无视 planner
7. `reward 长期偏负` → 在当前 reward 下，保守策略也无法积累正 return

**模型学会了一个最保守的存活策略：在桩附近做小范围清扫、频繁充电、不探索、不远离。** 这个策略在简单环境下能赢（win_rate=0.85），但在 broad 环境下因为覆盖不足而输（broad_win_rate=0.38）。

### 5.2 这不是"需要更多训练"能解决的

关键理解：**当前 reward 函数在数学上就是在鼓励这种行为。**

- 在桩附近清扫 1 格 → reward = +0.6~1.5
- 走 10 步去远处探索 → 10 步 idle_penalty + 无 cleaning = -1.0
- PPO advantage：清扫 > 探索

除非改变 cleaning reward 的量级，否则 PPO 不会学到"先探索、后收割"的策略，因为探索的即时 reward 为负。

### 5.3 但也不需要"推翻一切重来"

当前代码中有大量正确的设计：
- Context-aware cleaning scale 已经识别了所有需要降权的场景
- 双 value head 的 GAE 分离是正确的
- Teacher 可靠性门控避免了垃圾 teacher 信号
- A* 路径规划提供了准确的充电距离估计
- 课程框架的 4 stage 设计是合理的

**需要改变的只是一个核心数字和它的几个配套参数。** cleaning_reward 从 1.5 降到 0.6，就是整个修复的 70%。

---

## 六、与前序评审 (CLAUDE_REVIEW_20260417) 的一致性验证

| 前序结论 | 本次验证 | 状态 |
|---------|---------|------|
| cleaning reward 28:1 支配 → 根因 | 训练日志确认 algorithm.reward 偏负但 env_score 上升 → reward 与 env 目标确实脱节 | ✓ 一致 |
| 所有异常行为获得正奖励 | 当前 contract_usage=0.82, expand=0 → 保守行为被持续强化 | ✓ 一致 |
| 建议 cleaning 降到 0.6 | 本次独立分析得出同一数字，且增加了课程和 clip 的配套修改 | ✓ 一致 |
| 不建议增加 5+ 新 penalty | 本次继续维持，强调 credit assignment 难度 | ✓ 一致 |
| planner divergence 是果不是因 | divergence=85% 在新 planner_alignment=0.08/-0.15 + cleaning=0.6 下有望改善 | ✓ 一致 |

**新增发现（前序未覆盖）**：
1. lite benchmark 的 warmup 判级形成了死循环（保守→高 stall→卡 warmup→更保守）
2. CONTRACT_BATTERY_RATIO=0.35 和 PREPARE_RETURN_SLACK_THRESHOLD=12.0 过早触发 contract，是 expand≈0 的直接原因
3. reward clip [-5,+5] 过宽，充电事件的峰值(3.0×eff)可能扭曲 advantage 估计

---

## 附录：建议的验证指标与时间线

### 冷启动后 20 局（约 2~3 小时）

| 指标 | 预期变化 | 未达标则 |
|------|---------|---------|
| mode_usage_expand | >0.03 | 检查 CONTRACT_BATTERY_RATIO 是否生效 |
| algorithm.reward 均值 | >-10 | 检查 reward clip 是否生效 |
| charge_count | <20 | 正常，新 reward 需要时间传播 |

### 冷启动后 60 局（约 8~10 小时）

| 指标 | 预期变化 | 未达标则 |
|------|---------|---------|
| curriculum_stage | blend 或 robust | 检查课程门槛 |
| CPS | >0.45 | 检查 cleaning scale 是否过低 |
| planner_divergence | <0.70 | 检查 planner alignment 量级 |
| return_stall_rate | <0.45 | 预期自然改善 |
| charge_count | <18 | 如果仍 >20 考虑加过度充电惩罚 |

### 冷启动后 120 局（约 16~20 小时）

| 指标 | 预期变化 | 未达标则 |
|------|---------|---------|
| broad_win_rate | >0.50 | 当前 0.38，需要显著提升 |
| CPS | >0.50 | 接近目标 |
| expand usage | >0.05 | 说明策略学会了主动探索 |
| entropy | <1.85 | 说明策略在收敛 |
