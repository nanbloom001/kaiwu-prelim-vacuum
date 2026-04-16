# v5 优化全面分析：充电机制问题诊断与方案建议

> 日期：2026-04-13
> 目的：为外部 AI 专家提供完整的训练数据、代码审计、问题根因分析和优化方向建议
> 约束：不恢复 hard override，保持 Logit Bias 框架并修复其缺陷

---

## 一、项目背景

- **项目**：Robot Vacuum 清扫大作战 PPO 训练
- **目标**：最大化 CPS (Clean Per Step = clean_score / finished_steps) 和 comp_rate (completed rate = finished_steps / max_steps)
- **环境**：8 方向移动，128x128 网格，4 NPC 敌人，4 充电桩，battery_max=200
- **Expert Policy**：A* 路径规划 + 状态机充电导航（hysteresis + 动态 margin）
- **当前机制**：Expert Logit Bias（soft guidance），v4 训练 7606 步后充电行为退化

### 核心文件结构

| 文件 | 职责 |
|------|------|
| `code/agent_ppo/agent.py` | Agent 主类，predict() 决策流程 |
| `code/agent_ppo/algorithm/algorithm.py` | PPO 算法，_compute_loss() |
| `code/agent_ppo/conf/conf.py` | 超参数配置 |
| `code/agent_ppo/feature/preprocessor.py` | 特征处理 + reward 计算 |
| `code/agent_ppo/feature/expert.py` | Expert Policy（充电导航） |
| `code/agent_ppo/model/model.py` | 神经网络模型 |

---

## 二、v4 训练数据详细记录

v4 训练共 84 个数据点，step 28->7606，episode 14->2736。

### 2.1 按阶段分组数据

| Phase | Entropy Range | Steps | Pts | Avg CPS | Avg Comp | Avg ChgCnt | Avg Entropy | Avg Clean |
|-------|---------------|-------|-----|---------|----------|------------|-------------|-----------|
| Healthy | 1.5-2.0 | 99-2261 | 25 | 0.586 | 94.9% | 8.9 | 1.763 | 596 |
| Declining | 1.0-1.5 | 2338-3886 | 18 | 0.760 | 87.4% | 4.7 | 1.259 | 780 |
| Collapsing | 0.5-1.0 | 3964-4882 | 11 | 0.851 | 87.1% | 3.2 | 0.730 | 822 |
| Collapsed | 0.0-0.5 | 5047-7606 | 29 | 0.845 | 81.7% | 3.5 | 0.217 | 841 |

### 2.2 关键快照（采样）

```
#   Step    EP    CPS   Clean   CompRate  ChgCnt  Entropy
 1     28    14  0.294     312    100.0%    14.9   2.000
 6    460   201  0.488     503    100.0%     9.2   1.930
11    916   350  0.644     528     84.2%     4.6   1.870
16   1350   488  0.572     780    100.0%    10.1   1.730
21   1879   646  0.666     653     96.8%     5.9   1.650
26   2261   794  0.702     661     94.8%     5.5   1.500
31   2735   923  0.743     752     83.5%     5.9   1.420
36   3139  1064  0.814     559     61.7%     3.1   1.210
41   3604  1229  0.727     927     90.7%     7.2   1.120
46   4030  1408  0.797     915     92.8%     4.1   0.890
51   4486  1574  0.877     780     93.6%     3.4   0.620
56   5047  1749  0.860     814     92.3%     3.1   0.450
61   5443  1939  0.852     925     75.5%     3.1   0.340
66   5911  2107  0.843    1023     83.8%     4.6   0.240
71   6400  2311  0.863     893     76.0%     2.1   0.170
76   6880  2493  0.833     609     64.3%     2.5   0.120
81   7369  2658  0.837     908     76.4%     4.2   0.090
```

### 2.3 最优点

| 指标 | Step | CPS | CompRate | ChgCnt | Entropy | Clean |
|------|------|-----|----------|--------|---------|-------|
| Best CPS | 5992 | 0.914 | 69.9% | 1.9 | 0.200 | - |
| Best CompRate | 28 | 0.294 | 100.0% | 14.9 | 2.000 | - |
| Best CPS*Comp | 4261 | 0.862 | 100.0% | - | 0.840 | - |

### 2.4 选定的 Resume Checkpoint

- **路径**: `code/saved_models/v5-step4300/model.pkl`
- **来源**: step 约 4300, episode 约 1500
- **指标**: CPS=0.862, comp=100%, entropy=0.84
- **选择理由**: entropy 塌缩前、comp_rate 仍为 100% 的最佳时刻

---

## 三、备份版本（Hard Override）对比数据

备份版本使用 hard override（训练时也强制执行 Expert 充电决策），共 82 个数据点，step 0->3466，episode 4->5509。

### 3.1 按阶段分组

| Phase | Entropy Range | Steps | Pts | Avg CPS | Avg Comp | Avg ChgCnt | Avg Entropy |
|-------|---------------|-------|-----|---------|----------|------------|-------------|
| Healthy | 1.5-2.0 | 417-1565 | 33 | 0.674 | 50.1% | 8.0 | 1.761 |
| Declining | 1.0-1.5 | 1327-2731 | 23 | 0.602 | 61.6% | 94.4 | 1.305 |
| Collapsing | 0.5-1.0 | 2674-3466 | 10 | 0.279 | 81.4% | 252.6 | 0.753 |
| Collapsed | 0.0-0.5 | 0-3409 | 3 | 0.218 | 75.0% | 204.2 | 0.283 |

### 3.2 关键快照

```
#   Step    EP    CPS   Clean   CompRate  ChgCnt  Entropy
 1      0     4  0.269     108     40.0%     0.0   0.000
 6    136   261  0.315     158     50.2%     2.0   2.070
16    418   849  0.602     269     44.7%     0.2   1.940
26    703  1451  0.754     330     43.7%     0.4   1.790
36   1009  2056  0.549     349     63.5%    25.8   1.760
46   1384  2897  0.859     372     43.4%     0.1   1.480
56   1727  3174  0.435     237     54.5%    95.8   1.450
66   2393  4201  0.390     330     84.5%   206.8   1.100
76   3073  5046  0.150     128     85.0%   320.0   0.590
81   3409  5449  0.222     205     92.5%   269.1   0.400
```

### 3.3 充电趋势对比

| 版本 | Early ChgCnt | Mid ChgCnt | Late ChgCnt | 趋势 |
|------|-------------|-----------|-------------|------|
| v4 (Soft Bias) | 8.9 | 4.7 | 3.5 | 下降（学"不充电"） |
| Backup (Hard Override) | 1.0 | 15.1 | 183.8 | 暴涨（过度充电） |

### 3.4 对比结论

**两个极端**：
- **Hard Override**: 充电过多（charge_count 达 300+），CPS 暴跌至 0.15，comp_rate 反而提升到 85%（因为一直在充电没死）
- **Soft Bias**: 充电过少（charge_count 降至 1），CPS 提升到 0.86，comp_rate 下降到 68%（电池耗尽死亡）

这证明：**简单的 hard override 和 soft bias 都不是最优方案。需要让 RL 自主学会正确的充电时机。**

---

## 四、三重崩溃分析

### 4.1 Entropy 塌缩

```
Phase 1 [Healthy]:     entropy 1.76 → CPS 0.59 → comp 94.9%
Phase 2 [Declining]:   entropy 1.26 → CPS 0.76 → comp 87.4%
Phase 3 [Collapsing]:  entropy 0.73 → CPS 0.85 → comp 87.1%
Phase 4 [Collapsed]:   entropy 0.22 → CPS 0.85 → comp 81.7%
```

- Entropy 从 2.0 降到 0.08，策略从"均匀探索"变成"只走一条路"
- cleaning reward (+1.5/cell) 形成强一致性梯度，把模型推向确定性策略
- BETA=0.008 的 entropy 正则化不够强（只贡献约 0.1 梯度量）

### 4.2 充电行为退化

- charge_count: Phase1=8.9 -> Phase2=4.7 -> Phase3=3.2 -> Phase4=3.5
- 模型发现局部最优：不充电 -> 多清扫 -> 更高 reward -> 强化不充电
- 充电的 benefit 是延迟的（花 20 步去充电桩 -> 之后能多扫 100 格）
- PPO short horizon + discount gamma=0.99 看不到延迟回报

### 4.3 Logit Bias 梯度对抗（核心问题）

当前 predict() 中 logit bias 的处理方式：

```python
# agent.py predict() 当前实现（简化）
bias = expert.get_logit_bias(...)        # emergency=100, 非紧急=3~8
logits = logits + bias                    # 加入 bias
prob = self._legal_soft_max(logits, ...)  # biased prob -> 存为 old_prob
action = self._legal_sample(prob)         # 按 biased prob 采样
return ActData(action=[action], prob=list(prob), ...)
```

**问题链**：

```
采样阶段: prob_old = softmax(model_logits + bias)     <- bias 起作用，选了充电
学习阶段: prob_new = softmax(model_logits)              <- 没有 bias，自然 logits
PPO ratio: new_prob / old_prob                          <- ratio != 1

对于充电动作:
  old_prob (有bias) > new_prob (无bias)  =>  ratio < 1
  如果 advantage > 0 (充电后结果好) => PPO 增加充电概率
  如果 advantage < 0 (充电花费了时间) => PPO 减少充电概率

  充电的即时 reward 远不如清扫 => advantage 大概率 < 0
  => PPO 主动学习"不要充电"
```

**核心矛盾**：Logit Bias 在采样时推动了充电行为，但 PPO 梯度根据 advantage 反而惩罚了充电行为。Bias 越强 -> 更多充电采样 -> 更多"充电不好"的梯度信号 -> 策略越抗拒充电。

---

## 五、Expert 充电判断逻辑详解

### 5.1 状态机参数 (expert.py)

| 参数 | 值 | 含义 |
|------|-----|------|
| `EXIT_RETURN_RATIO` | 0.95 | 充到 95% + 站在充电桩才退出 return_mode |
| `LOW_BATTERY_RATIO` | 0.26 | 电量 < 26% 强制进入 return_mode |
| `BASE_RETURN_MARGIN` | 14.0 | 路径距离 + 14 = 触发阈值 |
| `BLOCKED_TTL` | 8 | 障碍物记忆衰减步数 |

**触发公式**: `battery <= charger_dist + margin` 或 `battery_ratio <= 0.26`

**动态 margin**: `BASE_RETURN_MARGIN(14) + 0.35 * turns + 1.2 * blocked_count`，上限 40

**A* 导航**: 3 级 NPC 避让降级（权重 1.0 -> 0.3 -> 0），有路径缓存 + blocked cell memory

### 5.2 Logit Bias 实现 (expert.py `get_logit_bias`)

```python
if should_return and expert_action is not None:
    slack = prep.battery - charger_dist
    urgency = float(np.clip(1 - slack / max(margin, 1), 0.2, 1.0))

    if slack <= 3 or (prep.battery / max(prep.battery_max, 1)) <= 0.10:
        bias[expert_action] = 100.0     # 紧急：等效 hard override
    else:
        bias[expert_action] = 3.0 + 5.0 * urgency  # 非紧急：[3.0, 8.0]
```

### 5.3 当前 predict() 决策流程 (agent.py)

```
Layer 1: NPC safety filter (block dangerous NPC directions, always active)
Layer 2: Training -> logit_bias / Eval -> hard_override
Layer 3: Anti-stuck (random action if stuck >= 10 steps, skip during return_mode)
Layer 4: RL decision (sample from softmax(logits + bias))
```

---

## 六、奖励函数全面审计

### 6.1 充电相关 reward (preprocessor.py `reward_process`)

| Reward | 代码 | 公式 | 最大值/step | 触发条件 |
|--------|------|------|-------------|----------|
| charger_reward | L596-602 | `0.15 * pressure * delta_slack` | ~0.15 | slack < 8 且接近中 |
| charger_path_explore | L606-612 | `0.12 * new_cells * delta_dist/3` | ~0.48 | 探索新格且靠近充电桩 |
| charge_reward | L615-626 | `0.3 + 0.4*need - freq_penalty` | ~0.7 | just_charged 时 |

charge_reward 详细公式：
```python
if self.just_charged:
    battery_ratio = self.battery / max(self.battery_max, 1)
    charge_base = 0.3
    need = max(0.0, 1.0 - battery_ratio)           # 0~1
    charge_eff = 0.4 * need                         # 0~0.4
    freq_penalty = -0.15 * max(total_recent - 3, 0) # 惩罚频繁充电
    charge_reward = max(charge_base + charge_eff + freq_penalty, -0.3)
```

### 6.2 清扫相关 reward

| Reward | 代码行 | 公式 | 最大值/step |
|--------|--------|------|-------------|
| cleaning_reward | L580 | `1.5 * cleaned_count` | 1.5/cell |
| streak_bonus | L583 | `0.15 * min(has_clean,1) * min(consec,5)` | 0.75 |
| frontier_reward | L592-593 | `0.15 * frontier * (0.5+0.5*progress)` | ~0.15 |
| dirty_approach_reward | L654-655 | `0.10 * directional_dirty[action]` | ~0.10 |
| explore_reward | L589 | `0.05 * min(new_cells, 6)` | 0.30 |
| edge_bonus | L586 | `0.02*min(wall,2) + 0.08*min(dirty/2,1)` | ~0.12 |

### 6.3 惩罚项

| Penalty | 公式 | 最大负值 |
|---------|------|----------|
| npc_penalty | `-1.5 * risk^2` (range 8) | -1.5 |
| npc_cleaned_penalty | `-0.3 * npc_cleaned_here` | -0.3 |
| revisit_penalty | `-0.10~-0.15 * min(visit-1, 2~3)` | ~-0.45 |
| stuck_penalty | `-0.5 * invalid - 0.25 * stuck_ratio` | ~-0.75 |
| idle_penalty | `-0.1 * clip(no_progress/15, 0, 1)` | -0.1 |

### 6.4 量级对比分析

**充电路径 (20步) 的总 reward**:
- charger_reward: 0.15 * 20 = ~3.0
- charge_event: ~0.7
- **总计: 约 3.7**

**继续清扫 (20步) 的总 reward**:
- cleaning_reward: 1.5 * 20 = 30.0 (假设每步清 1 格)
- streak_bonus: 0.75 * 20 = 15.0
- frontier + explore + approach: ~0.5 * 20 = 10.0
- **总计: 约 55**

**机会成本比 15:1**，RL 理性选择"不充电"。

### 6.5 关键发现

1. **没有显式的死亡惩罚**: battery death 只终止 episode（未来 reward 归零），无 per-step penalty
2. **reward clip [-3.0, 4.0]**: 死亡带来的负 reward 上限太小
3. **freq_penalty 实际上在惩罚频繁充电**: `freq_penalty = -0.15 * max(recent_charges - 3, 0)`
4. **charger_reward 系数 0.15 远不够**: 对比 cleaning_reward 1.5，差 10 倍

---

## 七、PPO 训练参数 (conf.py)

```python
GAMMA = 0.99              # Discount factor
LAMDA = 0.95              # GAE lambda
INIT_LEARNING_RATE_START = 0.0001
BETA_START = 0.008        # Entropy coefficient
CLIP_PARAM = 0.2          # PPO clip range
VF_COEF = 0.5             # Value loss coefficient
USE_GRAD_CLIP = True
GRAD_CLIP_RANGE = 0.5
```

### algorithm.py loss 计算：

```python
total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss
```

标准 PPO 三项 loss，无 entropy floor，无额外正则化。

---

## 八、优化方向建议

### 方向 A: 修复 Logit Bias 的 Prob 存储（最小改动，机制创新）

**问题**: biased prob 存为 old_prob -> PPO ratio 被扭曲

**修复方案**: bias 只用于采样，存储无 bias 的原始 prob

```python
# 当前（有缺陷）: agent.py predict()
bias = expert.get_logit_bias(...)
logits = logits + bias
prob = softmax(logits, legal)       # biased -> 存为 old_prob
action = sample(prob)

# 修复方案
biased_logits = logits + bias
biased_prob = softmax(biased_logits, legal)  # 仅用于采样
action = sample(biased_prob)
clean_prob = softmax(logits, legal)           # 无 bias -> 存为 old_prob
return ActData(action=[action], prob=list(clean_prob), ...)
```

**效果**: PPO ratio 约 1（old_prob 约等于 new_prob），bias 不再扭曲梯度。biased sampling 选中充电且结果好时 -> advantage > 0 -> PPO 正确增加该动作概率。

**改动量**: agent.py 约 5 行

### 方向 B: 增强充电 reward 量级

充电 reward 需要匹配"放弃清扫的机会成本"在同一量级：

| 参数 | 当前值 | 建议值 | 理由 |
|------|--------|--------|------|
| charge_reward base | 0.3 | 2.0 | 匹配约 1.3 步的清扫价值 |
| charge_reward need 系数 | 0.4 | 1.5 | 低电量时大幅增加充电价值 |
| charger_reward 接近系数 | 0.15 | 0.4 | 增强充电路径的 per-step signal |
| freq_penalty | -0.15*(recent-3) | 0 或 -0.05*(recent-5) | 不惩罚（或大幅放宽）主动充电 |
| battery_death penalty | 无 | -5.0 | 新增，充到这一步的 advantage 直接为强负 |

### 方向 C: Entropy Floor（机制创新）

标准 PPO 无 entropy 下限，可塌缩到 0。加 floor 提供持续探索梯度：

```python
# conf.py 新增
ENTROPY_FLOOR = 0.5          # 最低 entropy
ENTROPY_FLOOR_COEF = 0.05    # floor 惩罚系数

# algorithm.py _compute_loss() 末尾
if entropy_loss.item() < Config.ENTROPY_FLOOR:
    floor_gap = Config.ENTROPY_FLOOR - entropy_loss.item()
    floor_penalty = Config.ENTROPY_FLOOR_COEF * floor_gap
else:
    floor_penalty = 0.0
total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss + floor_penalty
```

同时 BETA 从 0.008 提到 0.012-0.015。

**改动量**: conf.py 约 3 行，algorithm.py 约 5 行

### 方向 D: 未来清扫潜力 Shaping（进阶）

充电时 reward 按恢复电量 x 当前 CPS 估计解锁的未来清扫价值：

```python
if self.just_charged:
    battery_gained = self.battery_max * (1.0 - old_battery_ratio)
    future_potential = battery_gained * self._cps_ema
    charge_reward = shaping_coeff * future_potential
```

**优点**: 直接衡量充电的"清扫投资回报率"
**风险**: shaping 不当可能导致 reward hacking
**建议**: 作为后续优化，先验证 A+B+C 效果

### 方向 E: 其他微调建议

1. **Expert 触发时机**: `LOW_BATTERY_RATIO=0.26` 可能触发太晚，考虑 0.30-0.35
2. **BASE_RETURN_MARGIN=14**: 可以根据实际 A* 路径成功率微调
3. **reward clip**: [-3.0, 4.0] 可以放宽到 [-5.0, 6.0] 以容纳更大的 charge_reward

---

## 九、推荐方案

**A + B + C 组合**，理由：

1. **方向 A**（修复 prob 存储）是最小的机制修复，消除 logit bias 的梯度对抗。不改架构，不改 reward，只修正了一个实现 bug
2. **方向 B**（增强充电 reward）解决量级不匹配的根本问题。当前充电 reward 和清扫 reward 差 15 倍，RL 不可能学"充电"
3. **方向 C**（entropy floor）防止策略塌缩，和 A 协同（A 修复了 bias 问题，C 保证模型保持探索能力去利用修复后的 bias）

**改动文件和预估行数**:

| 文件 | 修改内容 | 行数 |
|------|---------|------|
| `agent.py` predict() | 分离 biased sampling 和 clean prob 存储 | 约 5 行 |
| `algorithm.py` _compute_loss() | 加 entropy floor | 约 5 行 |
| `conf.py` | BETA=0.012, ENTROPY_FLOOR=0.5, COEF=0.05 | 约 3 行 |
| `preprocessor.py` reward_process() | 增大充电 reward + 加 death penalty | 约 10 行 |

**总计约 23 行代码修改，4 个文件。**

---

## 十、Resume 兼容性

v5-step4300 checkpoint 的模型架构完全不变（Model, Preprocessor, ExpertPolicy）。

修改仅涉及：
- agent.py predict() 的推理逻辑（不改模型结构）
- algorithm.py 的 loss 计算（不改模型结构）
- conf.py 的超参数
- preprocessor.py 的 reward 函数（reward 不存入模型权重）

设置 `RESUME_CHECKPOINT = "model.ckpt-resume.pkl"`，将 `saved_models/v5-step4300/model.pkl` 复制过去即可。

---

## 十一、验证标准

### 启动后确认
- learner log 显示 `[RESUME] Loaded from ...`
- entropy 从约 0.84 开始（不是 2.0）
- entropy 逐步回升到 0.5+（floor 生效）

### 前 100 episodes
- comp rate >= 95%（Expert 保证）
- charge_count >= 5（reward 增强 + bias 修复）
- CPS >= 0.7（resume 基础）

### 500+ episodes
- CPS >= 0.85
- comp rate >= 95%
- entropy 维持在 0.5-1.0（floor 效果）
- charge_count 稳定在 5-15（不过多也不过少）
