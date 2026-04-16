# 训练现状总结 + 顶层优化方向建议

> 本文档为另一个 AI 提供完整的系统现状分析，用于顶层设计方案优化。
> 目标：**提升 CPS（每步清理效率）** 和 **完成率（存活率）**。

---

## 一、系统架构概览

### 1.1 模型架构 (model.py)

纯前馈 CNN+MLP，无循环记忆：

```
Input (1597D)
├── local_view:   21×21×3 = 1323D  → CNN(3层) → 256D
├── global_mem:   8×8×3   = 192D   → CNN(2层) → 64D
├── scalar:       74D                → MLP(2层) → 64D
└── legal_action: 8D                 ┘
                                             ↓ concat = 384D
                                        Backbone MLP → 128D
                                             ↓
                                   Policy Head → 8 actions
                                   Value Head  → 1 value
```

- **local_view (21×21×3)**: hero 周围 10 格范围，通道 [wall, clean, dirty]
- **global_memory (8×8×3)**: 整张 128×128 地图压缩到 8×8，通道 [explored, dirty, visit_heatmap]
- **scalar (74D)**: 39 维原始 + 26 维扩展（NPC距离×4, charger距离×4, directional_dirty×8）+ 9 维 one-hot (last_action)
- **legal_action (8D)**: 合法动作掩码

**关键限制**：无记忆机制（无 LSTM/GRU），模型只能通过 8×8 global_memory 和 74D scalar 感知历史状态。

### 1.2 决策层级 (agent.py predict)

```
Layer 1: NPC Filter — 过滤朝向 NPC 的动作
Layer 2: Expert Override — 充电状态机（return_mode hysteresis）
Layer 3: Anti-stuck — 10 步卡住后随机合法动作
Layer 4: RL Normal — 模型 softmax 采样
```

Expert 使用 A* 实际路径距离（非 Chebyshev）计算充电阈值，有 blocked cell 记忆 (TTL=8) 和路径缓存。

### 1.3 PPO 训练配置 (conf.py)

| 参数 | 值 |
|------|-----|
| GAMMA | 0.99 |
| LAMDA (GAE) | 0.95 |
| LR | 0.0001 |
| BETA (entropy) | 0.008 |
| CLIP_PARAM | 0.2 |
| VF_COEF | 0.5 |
| train_batch_size | 2048 |
| replay_buffer | 10000 |
| dump_model_freq | 100 steps |

---

## 二、奖励系统详解

### 2.1 每步奖励 (preprocessor.py reward_process, 11 个分量)

| 分量 | 公式 | 典型值 | 说明 |
|------|------|--------|------|
| **cleaning_reward** | `1.5 × cleaned_this_step` | 0 或 1.5 | **主奖励**，踩到脏格子才触发 |
| streak_bonus | `0.15 × min(consecutive_clean, 5)` | 0-0.75 | 连续清扫加成 |
| edge_bonus | `0.08×wall_adj + 0.12×dirty_adj` | 0-0.32 | 沿墙/脏边界行走 |
| explore_reward | `0.05 × new_cells` (cap 6) | 0-0.30 | 探索新区域 |
| frontier_reward | `0.10 × frontier_density × f(clean_ratio)` | 0-0.10 | 探索边界区域 |
| charger_reward | `0.15 × pressure × delta_slack` | ±0.15 | 接近充电桩 |
| charger_path_explore | `0.12 × new_cells × closer` | 0-0.48 | 边充边探索 |
| charge_bonus | `1.0 × just_charged` | 0 或 1.0 | 到达充电桩 |
| npc_penalty | `-0.5 × risk²` | 0 to -0.5 | NPC 距离惩罚 |
| revisit_penalty | `-0.05~0.08 × visit_count` | 0 to -0.24 | 重访惩罚 |
| stuck_penalty | `-0.5×invalid - 0.25×stuck_norm` | 0 to -0.75 | 卡住惩罚 |
| idle_penalty | `-0.1 × no_progress_ratio` | 0 to -0.10 | 无进展惩罚 |

**总奖励裁剪到 [-3.0, 4.0]**

### 2.2 Episode 结束奖励 (_handle_episode_end)

```python
outcome_bonus = {"completed": +1.5, "battery": -2.5, "collision": -4.0, "unknown": -3.0}
efficiency_bonus = 0.6 × cleaning_ratio + 0.3 × min(clean_score / max(step, 1), 1.5)
final_reward = outcome_bonus + efficiency_bonus
```

- `efficiency_bonus` 中 `clean_score / step` 就是 **CPS**，但权重只有 0.3，上限 1.5
- `cleaning_ratio` (已清扫/总脏格) 权重 0.6

### 2.3 奖励与得分的关键关系

```
env_score = 环境清扫的脏格子数（只增不减）
step_reward = reward_process() 的 shaped reward（包含清扫、探索、充电等）
final_reward = outcome_bonus + efficiency_bonus

总 reward = Σ(step_reward) + final_reward
```

**核心问题**：
- `cleaning_reward = 1.5 × cleaned_this_step` — 每步最多 1.5 分，但这是 **绝对值**，不除以步数
- 2000 步 episode 的总 cleaning_reward 上限 = 1.5 × 2000 = 3000
- 500 步 episode 的总 cleaning_reward 上限 = 1.5 × 500 = 750
- 但实际上 CPS 0.8 vs 0.9 的差异远比 episode 长度的影响小
- **模型没有动机提升 CPS**——只要存活够久，总分自然高

---

## 三、训练现状数据 (step 0-5255, episode 0-944)

### 3.1 关键指标趋势

```
Step   Ep   Entropy  Score   CPS    Survival  Charges  V_loss
    1    4    0.98     838   0.856   78.3%     3.8      484
  400   88   0.82     853   0.853  100.0%     4.2      499
  868  170   0.67     732   0.771  100.0%     5.2      467
 1332  254   0.44     926   0.842  100.0%     4.8      508
 1606  309   0.40     952   0.810  100.0%     6.8      438
 2062  383   0.29     868   0.868  100.0%     4.8      505
 4975  890   0.13     635   0.792   75.5%     3.5      311
 5255  944   0.19     837   0.733   79.5%     6.5      266
```

### 3.2 问题诊断

| 问题 | 现象 | 严重程度 |
|------|------|----------|
| **Entropy 坍缩** | 0.98 → 0.13 in 5000 steps | 严重 — 策略已接近确定性 |
| **CPS 停滞** | 0.84 → 0.73 (反而下降) | 严重 — 模型在变差 |
| **存活率下降** | 100% → 75-80% | 严重 — 模型丧失了 resume 模型的生存能力 |
| **Value loss 下降** | 484 → 266 | 看似改善，但说明 value function 在拟合一个退化策略 |

### 3.3 Resume 基线 vs 当前

| 指标 | Resume 模型 (step 0) | 当前 (step 5255) | 变化 |
|------|---------------------|-----------------|------|
| Entropy | 0.98 | 0.19 | -0.79 |
| CPS | 0.84-0.87 | 0.73-0.80 | -0.07 |
| Survival | 95-100% | 75-80% | -20% |

**结论：训练在使模型退化**。Resume 预训练模型的策略已经是一个还不错的局部最优，但 PPO 的 reward signal 不足以引导模型找到更好的策略，反而把原有的好策略"遗忘"了。

---

## 四、根因分析

### 4.1 为什么 CPS 停滞/退化

1. **奖励信号不对齐**：模型获得的 reward 主要来自 `cleaning_reward = 1.5 × cleaned_this_step`。这个奖励奖励"踩到脏格子"，但不奖励"高效地前往脏格密集区"。模型只需要随机漫游就能偶尔踩到脏格子，获得正 reward。

2. **Episode 长度混淆**：长 episode (max_step=2000) 自然比短 episode (max_step=500) 获得更多总 reward，即使 CPS 更低。模型被激励去"存活更久"而非"更高效地清扫"。

3. **Entropy 过快收敛**：BETA=0.008 不足以维持探索。模型快速收敛到"随机漫步+偶尔清扫"的策略，然后就停止探索新策略了。

4. **Expert 系统的副作用**：Expert 充电状态机确保 100% 存活率，但这也意味着：
   - RL 永远体验不到充电失败的后果
   - 充电决策完全由 Expert 接管，RL 在这方面没有学习动力
   - Expert 的 A* 寻路偶尔和 RL 的决策冲突，导致不稳定

### 4.2 为什么存活率从 100% 下降到 75%

1. **Expert 充电状态机依赖 visit_count 和 passable_map**：这些全局状态在 Resume 模型下是合理的，但随着训练进行，RL 的行为模式变了（策略从探索变为利用），导致 Expert 的输入分布偏移。

2. **Entropy 降低 → Expert 接管时 RL 给出的 prob 分布极端**：当 entropy 很低时，RL 对非 Expert 动作的概率接近 0，PPO importance ratio 失真。

3. **Replay buffer 混合新旧数据**：buffer 容量 10000，包含旧策略的数据。随着策略快速变化，off-policy 偏差增大。

---

## 五、方向性启发

以下是一些可能的优化方向，供另一个 AI 分析和深入：

### 启发 1：奖励归一化 — 按步平均

**问题**：当前 reward 是绝对值，长 episode 自然得分高。
**方向**：将部分 reward 除以当前步数或预期步数：
```python
# 方案 A: cleaning_reward 归一化
cleaning_reward = 1.5 * cleaned_this_step / max(sqrt(step), 1)

# 方案 B: efficiency bonus 增强
efficiency_bonus = alpha * (cleaned_this_step) + beta * (CPS_running_avg - baseline)

# 方案 C: 用 return-to-go (remaining steps) 归一化
normalized_reward = cleaning_reward * (max_step - step) / max_step
```
这样模型会被激励在有限的步数内尽可能高效地清扫。

### 启发 2：方向性奖励 — 引导走向脏格密集区

**问题**：当前只有"踩到脏格子"才给 reward，没有"走向脏格密集区"的引导。
**方向**：利用已有的 `directional_dirty` 8 维特征（8 个方向上的脏格计数），当 agent 朝脏格密度最高的方向移动时给奖励：
```python
# directional_dirty[act] 是动作 act 方向上的脏格密度
dirty_direction_reward = 0.3 * directional_dirty[action] / max(max(directional_dirty), 1)
```

### 启发 3：Entropy 管理 — 防止过早收敛

**问题**：BETA=0.008 太低，5000 步 entropy 从 0.98 降到 0.13。
**方向**：
- 提高 BETA 到 0.015-0.02
- 或使用 entropy schedule（前期高 0.02，后期低 0.005）
- 或使用 adaptive entropy（当 entropy < 0.3 时自动增加 beta）

### 启发 4：全局记忆增强 — 提升路径规划能力

**问题**：8×8 global_memory 太粗糙（128×128 压缩 16 倍），无法表达精细的清扫路径。
**方向**：
- 增大到 16×16（4x 精度，内存开销可接受）
- 或增加通道：加入 npc_cleaned（NPC 清扫过的格子）和 frontier（边界格）通道
- 这不改模型架构，只改 global_memory 的生成逻辑和通道数

### 启发 5：CPS 直接作为 reward component

**问题**：模型不直接优化 CPS。
**方向**：在每步 reward 中加入 running CPS 的改善量：
```python
# running CPS 的指数移动平均
cps_ema = 0.99 * cps_ema + 0.01 * cleaned_this_step
cps_improvement = cps_ema - cps_baseline  # baseline 可以是 0.8
cps_reward = 0.5 * cps_improvement  # 正值表示效率提升
```

### 启发 6：Episode 级别奖励重分配

**问题**：当前 episode 结束的 `efficiency_bonus` 权重太小（0.3），且只在结束时给。
**方向**：将 outcome bonus 和 efficiency bonus 按比例分配到每一步（reward redistribution）：
```python
# 在 episode 结束时，将 final_bonus 均匀分配到每一步
per_step_bonus = final_bonus / total_steps
# 这使得每一步的 reward 都受到最终效率的影响
```

### 启发 7：学习率 / 训练稳定性

**问题**：Value loss 在下降（484→266），但策略在变差——说明 value function 在拟合一个退化的策略。
**方向**：
- 降低学习率 (0.0001 → 0.00003) 减慢策略更新
- 增加 VF_COEF (0.5 → 1.0) 让 value function 更准确
- 减小 CLIP_PARAM (0.2 → 0.1) 限制策略更新幅度

---

## 六、关键文件索引

| 文件 | 路径 | 核心功能 |
|------|------|----------|
| 模型 | `code/agent_ppo/model/model.py` | CNN+MLP 架构 |
| Agent | `code/agent_ppo/agent.py` | 决策层级 + resume 加载 |
| Preprocessor | `code/agent_ppo/feature/preprocessor.py` | 特征提取 + 奖励计算 |
| Expert | `code/agent_ppo/feature/expert.py` | A* 充电状态机 |
| 算法 | `code/agent_ppo/algorithm/algorithm.py` | PPO 实现 |
| 配置 | `code/agent_ppo/conf/conf.py` | 超参数 |
| Workflow | `code/agent_ppo/workflow/train_workflow.py` | 训练循环 + episode 结束奖励 |
| 定义 | `code/agent_ppo/feature/definition.py` | 数据结构 |
| 全局配置 | `code/conf/configure_app.toml` | 框架级配置 |

---

## 七、约束条件

1. **不能改模型架构维度**（改了需要重新训练）
2. **可以改**：reward 函数、超参数、Expert 逻辑、特征计算（不改维度）
3. **可以改 global_memory 的通道语义**（通道 0,1 改了 CNN 能适应，但不能增减通道数）
4. **训练框架**：Docker 分布式，8 个并行环境，~100 steps/min
5. **当前模型**：从 model.ckpt-resume.pkl fine-tune，已训练 ~5255 步
6. **竞赛评估**：最终提交 best_model.pkl，在固定配置下评估 clean_score

---

## 八、另一 AI 方案评估 + 执行方案

> 另一 AI 给出了 Phase 1-3 方案。以下逐项评估，标注 ✅采纳 / ⚠️修改 / ❌不采纳，并给出理由。
> 最终执行方案在末尾汇总。

### 8.1 Phase 1: Entropy & 训练稳定性

#### BETA_START 0.008 → 0.02 (2.5x) — ⚠️ 改为 0.012

**另一 AI 理由**：2.5x 防止 entropy 自由坍缩。

**我的评估**：0.02 过激进。用历史数据推算：
- BETA=0.005 时 entropy 底部 ≈ 0.35（9000 步训练）
- BETA=0.008 时 entropy 底部 ≈ 0.13（5000 步训练）
- 每次 BETA +0.003 大约提升 entropy 底部 0.2
- BETA=0.012 预计 entropy 底部 ≈ 0.35-0.45

加上自适应 entropy 兜底（<0.3 时 double），0.012 已经足够。0.02 会让 entropy 长期在 0.7+，策略难以收敛到任何有用策略，浪费训练步数。

**关键教训**：entropy bonus 不是越高越好。过高的 entropy = 模型在随机探索 = 不学习。目标是 **维持适度的探索**（0.3-0.7），不是最大化 entropy。

#### INIT_LEARNING_RATE 0.0001 → 0.00006 — ✅ 采纳，改为 0.00005

**理由**：Resume 微调应保守，减慢策略漂移。0.00005 比 0.00006 更保守一点点，配合 CLIP 0.15 进一步保护。

#### CLIP_PARAM 0.2 → 0.15 — ✅ 采纳

**理由**：限制策略更新幅度，标准做法。

#### VF_COEF 0.5 → 0.8 — ❌ 不采纳

**另一 AI 理由**：增强 value function，改善 advantage 估计。

**我的评估**：Value loss 已在正常下降（484→266），说明 value function 在学。问题是 **policy 的探索不够** 和 **reward signal 不对齐**，不是 value function 估计不准。增加 VF_COEF 只会让 total_loss 更多地被 value_loss 主导，不解决核心问题。

此外，PPO 中 value function 的作用是提供 baseline（计算 advantage），不是直接优化策略。过度强调 value loss 可能导致 optimizer 把梯度预算花在拟合 value 上而非改善策略。

#### 自适应 Entropy Bonus — ✅ 采纳，调整目标区间

**另一 AI 方案**：目标 [0.3, 0.8]，低于 0.3 时 ×2.0，高于 0.8 时 ×0.7。

**我的调整**：目标 [0.3, 0.7]，低于 0.3 时 ×2.0，高于 0.7 时 ×0.6。
- 上限从 0.8 降到 0.7：0.8 的 entropy 对 8-action 空间（max 2.08）来说已经偏高
- 上限倍率从 0.7 降到 0.6：更快地让策略收敛到有用策略

**实现位置**：`algorithm.py _compute_loss()`，在 total_loss 计算前插入 adaptive beta 逻辑。

### 8.2 Phase 2: 奖励函数

#### dirty_approach_reward = 0.15 — ⚠️ 改为 0.10

**另一 AI 理由**：利用 directional_dirty 引导走向脏格密集区。

**我的评估**：方向正确，但系数需要谨慎。考虑reward 量纲：
- cleaning_reward = 1.5（实际清扫）
- dirty_approach_reward = 0.15（走向脏格方向）

0.15 / 1.5 = 10% 看似合理，但问题在于：directional_dirty 是归一化后的密度（0-1），不是二值的。agent 每步都朝"某个方向"移动，那个方向几乎总是有一些脏格的（除非已完全清扫）。

这意味着 dirty_approach_reward 几乎每步都是正的（只是大小不同），可能变成一种"移动就给奖"的噪声。0.10 更保守，减少这种噪声。

**需要额外保护**：只有当 `max(directional_dirty) > 0.01`（确实有脏格在附近）时才给奖，否则为 0。

#### cleaning_reward 1.5 → 2.0 — ❌ 不采纳

**另一 AI 理由**：强化主信号，清扫为绝对优先。

**我的评估**：这是 **治标不治本**。问题不是 cleaning_reward 太弱，而是：
1. cleaning_reward 是绝对值，不反映效率
2. 1.5→2.0 只是放大倍率，长 episode 仍然天然得分高
3. CPS 0.8 和 CPS 0.9 的 reward 差距在 1.5x 和 2.0x 下比例完全一样

真正需要的是让 **效率本身** 影响 reward（见下文的 CPS_ema 奖励），而不是简单放大绝对值。

#### streak_bonus 0.15 → 0.25 — ❌ 不采纳

**另一 AI 理由**：奖励连续清扫节奏。

**我的评估**：streak_bonus 已经是正反馈（连续清扫 → 越扫越多 → 越容易继续扫），不需要额外加强。0.15→0.25 的增量 (~0.10/step) 足以让模型贪心地追逐连续清扫而忽略更好的路径（比如先绕远去密集区再连续清扫）。

#### edge_bonus 减半 — ✅ 采纳

**理由**：edge_bonus 本身是 shaping（辅助引导），不需要太强。减半合理。

#### explore_reward 0.05 → 0.03 — ✅ 采纳

**理由**：探索应服务于清扫目标，适当降低权重合理。

#### revisit_penalty -0.08 → -0.12 — ❌ 不采纳

**另一 AI 理由**：更强惩罚无效重访。

**我的评估**：128×128 地图有墙壁和障碍，某些路径 **必须** 重访（比如从死胡同返回、绕过 NPC 后回原路）。更强的重访惩罚会让 agent 走"避免重访"的路径而非"清扫效率高"的路径。

一个具体的风险场景：agent 发现一片脏格密集区，扫了一部分后被 NPC 赶走，等 NPC 离开后需要返回。更强的 revisit_penalty 会惩罚这个返回行为，导致那片区域永远扫不完。

#### Episode-end CPS 权重 0.3→0.6 — ✅ 采纳

**另一 AI 理由**：增强 CPS 信号。

**我的评估**：方向正确。但需注意：
- Episode-end bonus 只影响最后一步的 reward（通过 GAE 会部分传播到前面，但衰减很快）
- 单纯提高 episode-end 权重不够，**必须配合 per-step CPS 信号**（见我的补充方案）
- cap 从 1.5 降到 1.0：实际 CPS 通常 < 1.0，1.5 的 cap 让奖励梯度太平

#### 我的补充：Per-step CPS EMA 奖励

这是另一 AI 方案中 **缺失的关键环节**。

问题：Episode-end CPS 权重即使提到 0.6，对 GAE 的影响也只有最后几十步。前面 900+ 步的 reward 完全不受 CPS 影响。

方案：在每步 reward 中加入 running CPS 的指数移动平均（EMA）：
```python
# 在 Preprocessor.__init__ / reset() 中:
self._cps_ema = 0.5  # 初始基线

# 在 reward_process() 中:
if self.cleaned_this_step > 0:
    self._cps_ema = 0.95 * self._cps_ema + 0.05 * 1.0
else:
    self._cps_ema = 0.95 * self._cps_ema + 0.05 * 0.0
efficiency_reward = 0.3 * max(self._cps_ema - 0.75, 0)  # 超过基线才有奖
```

设计考量：
- **EMA 而非瞬时值**：单步 CPS 要么 0 要么 1，太噪声。EMA 平滑后反映趋势。
- **基线 0.75**：resume 模型的 CPS ≈ 0.84-0.87，基线设在 0.75 意味着"比最差情况好一点"才有奖。不会对已有策略产生太大扰动。
- **系数 0.3**：EMA 稳定在 0.85 时，efficiency_reward = 0.3 × (0.85-0.75) = 0.03。不大，但持续正反馈引导方向。
- **max(..., 0)**：低于基线不给负惩罚，避免干扰正常学习。CPS 下降时靠 entropy 探索来恢复。

### 8.3 Phase 3: 课程域随机化

#### 重写 EnvConfigSampler — ❌ 不采纳

**另一 AI 理由**：替换离散 3 阶段为连续课程。

**我的评估**：
1. **收益不明确**：当前 3-profile 系统（anchor/mild/broad）已经提供了课程。训练数据表明 anchor profile 的 WIN 率 75.5%，broad 的 57.7%——梯度从易到难。
2. **代码风险**：重写采样逻辑涉及 ~50 行新代码 + 新的参数边界处理（如 charger_count=0 的异常情况）。另一 AI 的方案中 `charger_count mean=2, std=1.8` 可能采样出 0 甚至负数。
3. **低优先级**：当前最大问题是 entropy 坍缩和 reward 不对齐，不是课程不够好。
4. **验证困难**：连续课程的参数空间太大，很难判断"课程改了"还是"其他改动"导致了变化。

**替代方案**：保持当前 3-profile 系统不变。如果后续需要调整，只需修改 profile 采样比例（如增加 mild 比例），不重写整个逻辑。

### 8.4 另一 AI 遗漏的关键点：Replay Buffer

另一 AI 在 "Further Considerations" 中提到但未纳入主方案。我认为这是 **仅次于 entropy 修复的第二优先级**。

**问题分析**：
- Buffer 容量 10000，采样方式 Uniform
- 训练 ~5000 步，策略从 entropy=0.98 变到 0.13
- Buffer 中最旧的数据来自 ~5000 步前的策略
- PPO 用 old_prob / new_prob 计算 importance ratio，old_prob 来自旧策略
- 旧策略的 advantage 估计 = 旧 value function 的残差，和当前策略完全不一致
- 结果：off-policy 偏差 → 梯度方向错误 → 策略退化

**解决方案**：`configure_app.toml` 中 `replay_buffer_capacity` 从 10000 缩到 4096。

- 4096 约等于 3-4 个 episode 的数据
- 策略在 3-4 个 episode 内变化不大，off-policy 偏差可控
- 训练吞吐不受影响（每步仍然有数据可训练，buffer 只是控制"多旧的数据"）

---

## 九、最终执行方案

### 改动清单

| # | 文件 | 改动内容 | 优先级 |
|---|------|----------|--------|
| 1 | `conf.py` | BETA 0.008→0.012, LR 0.0001→0.00005, CLIP 0.2→0.15 | P0 |
| 2 | `algorithm.py` | `_compute_loss()` 添加自适应 entropy（目标 0.3-0.7） | P0 |
| 3 | `preprocessor.py` | `reward_process()` 添加 dirty_approach_reward (0.10) + CPS_ema 奖励 (0.3) | P0 |
| 4 | `preprocessor.py` | edge_bonus 减半, explore_reward 0.05→0.03 | P1 |
| 5 | `train_workflow.py` | efficiency_bonus 权重: 0.6×ratio + 0.6×min(CPS,1.0) | P1 |
| 6 | `configure_app.toml` | replay_buffer_capacity 10000→4096 | P0 |

### 不改的文件

- `model.py` — 架构不变
- `expert.py` — 充电状态机刚修好，不动
- `agent.py` — 决策层级已正确
- `definition.py` — 数据结构不变
- `conf.py` 中 GAMMA, LAMDA, VF_COEF — 不变

### 不改的参数

| 参数 | 另一 AI 建议 | 决定 | 理由 |
|------|-------------|------|------|
| cleaning_reward | 1.5→2.0 | **保持 1.5** | 放大绝对值不解决 CPS 对齐问题 |
| streak_bonus | 0.15→0.25 | **保持 0.15** | 避免贪心追逐连续清扫 |
| revisit_penalty | -0.08→-0.12 | **保持 -0.08** | 128×128 地图必须重访某些路径 |
| VF_COEF | 0.5→0.8 | **保持 0.5** | Value function 不是瓶颈 |
| 课程系统 | 重写采样逻辑 | **保持现状** | 风险高收益低 |

### 执行步骤

1. **停止当前训练**（模型已退化，继续无意义）
2. 应用全部 6 项改动
3. **从 resume checkpoint 重启**（不用 step 5255 的退化模型）
4. 清除 Docker volumes（`down -v`）避免旧 checkpoint 干扰
5. 重启容器，移除 process_stop.done

### 验证指标

| 指标 | 目标 | 检查方式 |
|------|------|----------|
| Entropy | 0.3-0.7（不低于 0.25） | learner log 每 min |
| CPS | ≥ 0.85（超过 resume 基线 0.84） | aisrv metrics 每 min |
| Survival | ≥ 95% | aisrv metrics finished_steps/max_steps |
| CPS_ema | 日志中出现，值合理 | episode log |
| dirty_approach_reward | 非零步数占比 > 50% | 需要临时日志或观察 reward 分布 |
