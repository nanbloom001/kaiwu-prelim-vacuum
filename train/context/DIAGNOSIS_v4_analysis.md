# v4 训练深度分析：为什么 Logit Bias 失败，以及机制创新方向

## 一、v4 训练的三重崩溃

### 1.1 Entropy 塌缩（根本原因）

```
Step    0→ 2261:  entropy 1.5-2.0   →  CPS 0.59   comp 94.9%   正常探索期
Step 2338→ 3886:  entropy 1.0-1.5   →  CPS 0.76   comp 87.4%   开始衰退
Step 3964→ 4882:  entropy 0.5-1.0   →  CPS 0.85   comp 87.0%   策略趋确定
Step 5047→ 7606:  entropy 0.0-0.5   →  CPS 0.85   comp 81.7%   完全塌缩
```

Entropy 从 2.0 降到 0.08，意味着策略从"均匀探索"变成了"只走一条路"。

**这不是简单的 BETA 太小的问题。** 即使 BETA=0.008，entropy loss 也只有 ~0.1 的梯度贡献。
真正的问题是：cleaning reward (+1.5/cell) 形成了强一致性的梯度信号，把模型推向确定性策略。

### 1.2 充电行为退化

```
Charge count: 14.88 → 3.88 → 1.38 → 2.25（沿训练时间持续下降）
```

模型发现了一个局部最优：**不充电 → 多清扫 → 更高 reward → 策略梯度强化不充电行为**。

充电 reward（base=0.3, max=0.62）相比 cleaning reward（1.5/cell × 可能扫几十格）微不足道。
更关键的是，充电的 benefit 是延迟的（现在花 20 步走到充电桩 → 之后能多扫 100 格），
PPO 的 short horizon + discount factor 让模型无法看到这个延迟回报。

### 1.3 Expert Logit Bias 失效

设计的意图：用 soft bias (3-8 logits) 引导充电，RL 可以偏离但大部分时候会跟随。

实际失败原因——**Bias 与 PPO 梯度的对抗**：

```
采样阶段：prob_old = softmax(model_logits + bias)     ← bias 起作用，选了充电
学习阶段：prob_new = softmax(model_logits)              ← 没有 bias，自然 logits
PPO ratio: new_prob / old_prob                          ← ratio ≠ 1

对于充电动作：prob_old > prob_new → ratio < 1
如果 advantage > 0（充电后结果好）→ PPO 增加充电概率 ✓
如果 advantage < 0（充电花费了时间）→ PPO 减少充电概率 ✗

问题是：充电的即时 reward 是负的（放弃清扫去充电），
所以 advantage 大概率 < 0 → PPO 主动学习"不要充电"
```

**核心矛盾：Logit Bias 在采样时推动了充电行为，但 PPO 梯度根据 advantage 反而惩罚了充电行为。**
Bias 越强 → 更多充电采样 → 更多"充电不好"的梯度信号 → 策略越抗拒充电。

### 1.4 对比备份版本（Hard Override + robust=3051）

备份版本用 hard override：充电时直接执行 expert 动作，PPO 梯度信号来自 model 的原始概率。
由于 override 动作的实际 reward 在大多数情况下是正的（避免了死亡），
PPO 虽然采样分布被扭曲，但 value function 学到了正确的长期价值。

**结论：Hard Override 的 "梯度污染" 比预期轻微得多，而 Soft Bias 的 "梯度对抗" 反而更致命。**

---

## 二、为什么简单调参无法修复

| 调参方案 | 为什么不够 |
|----------|-----------|
| 提高 BETA | 减缓 entropy 塌缩速度，但不改变 cleaning reward 的主导地位 |
| 加强 bias (3-8 → 10-20) | 更强的 bias → 更多充电采样 → 更多负 advantage → 更强的反充电梯度 |
| 增加充电 reward | 即使 charge_reward 从 0.62 提高到 5.0，仍远不如一局 100+ 格的 cleaning reward |
| 降低 cleaning reward | 会削弱模型学到的清扫效率，得不偿失 |

**根本问题：充电和清扫是不同时间尺度的目标，用同一个 reward + 同一个 policy 优化，必然偏向即时回报的清扫。**

---

## 三、机制创新方向

### 方案 A：Expert Override + PPO 梯度隔离（推荐，最务实）

**核心思想**：Expert 负责 survival，RL 负责 cleaning，两者不互相干扰。

```
1. Expert override 时机不变（battery <= charger_dist + margin）
2. Override 时，在 sample 数据中标记 expert_override=True
3. PPO 训练时，跳过 expert_override=True 的 sample（不参与 policy loss 计算）
4. Value function 仍然在这些 sample 上训练（学习 survival 的长期价值）
```

**优点**：
- Expert 保证生存（100% comp rate，如备份版本）
- PPO 只在自主决策的 step 上学习，梯度干净
- Value function 从 expert 的正确决策中学习充电的价值
- 实现简单：只需在 sample 数据中加一个 flag + algorithm.py 中一行条件判断

**原理**：这等价于一个分层策略——底层 RL 学清扫，高层 Expert 管 survival。
PPO 的梯度不会对抗 expert 的决策，因为 expert 的决策根本不参与 policy loss。

### 方案 B：Dual-Policy with Gating（创新但复杂）

**核心思想**：训练两个独立的 policy head，一个管清扫一个管充电，用 gate 切换。

```
Model 输出:
  cleaning_logits (8D) — 清扫方向
  charging_logits (8D) — 充电路径
  gate_value (1D) — 何时切换

决策逻辑:
  if gate > threshold or battery_critical:
      action = argmax(charging_logits)  # 充电模式
  else:
      action = sample(cleaning_logits)  # 清扫模式
```

**优点**：两个目标完全解耦，互不干扰
**缺点**：架构改动大，训练复杂，需要大量数据才能学好 gate

### 方案 C：Entropy Floor + 动态 Bias 缩放（最小改动）

**核心思想**：不改变架构，但让 bias 适应模型的置信度。

```python
# 动态 bias：基于模型自身 logits 的幅度
def get_logit_bias(self, prep, logits, legal_action, last_action=-1):
    ...
    if should_return and expert_action is not None:
        max_logit = float(np.max(np.abs(logits)))
        # Bias 与模型置信度成正比，确保始终有影响力
        if slack <= 3 or battery_ratio <= 0.10:
            bias[expert_action] = max_logit * 1.5  # 紧急：超过模型置信度
        else:
            bias[expert_action] = max_logit * 0.5  # 非紧急：达到模型一半

    # Entropy floor：如果策略太确定，给所有合法动作加均匀噪声
    if entropy_estimate < 0.3:
        bias += np.random.uniform(-0.5, 0.5, size=8) * legal_action

    return bias
```

```python
# algorithm.py: adaptive entropy floor
entropy_loss = -(prob_dist * log(prob_dist)).sum(1).mean()
if entropy_loss < ENTROPY_FLOOR:
    # 额外惩罚低 entropy
    entropy_penalty = ENTROPY_FLOOR - entropy_loss
    total_loss -= FLOOR_COEF * entropy_penalty
```

**优点**：改动小，可以和方案 A 结合
**缺点**：没有从根本上解决 reward 时间尺度问题

---

## 四、推荐方案：A + C 组合

### Phase 1：Expert Override + PPO 梯度隔离（方案 A）

这是最核心的机制改进：

1. `agent.py predict()`: 恢复 training 时也使用 hard override
2. 在 ActData 中新增 `expert_override` 字段
3. `algorithm.py learn()`: 计算 policy loss 时，mask 掉 expert_override=True 的 sample
4. Value function 正常训练（不受 mask 影响）

预期效果：
- Comp rate 恢复 100%（expert 保证生存）
- PPO 梯度干净（只在 RL 自主决策时学习）
- 清扫效率继续提升（和备份版本一样）

### Phase 2：Entropy Floor + 更高 BETA（方案 C 简化版）

在 Phase 1 基础上，防止 entropy 在非 override 区域塌缩：

1. BETA 从 0.008 提高到 0.015
2. 在 algorithm.py 中加 entropy floor（最低 0.5）
3. 保持探索能力，防止策略过度确定

### 实现复杂度

| 修改 | 文件 | 行数 |
|------|------|------|
| 恢复 training hard override | agent.py | ~10 行 |
| ActData 加 override flag | definition.py | ~3 行 |
| PPO loss mask override samples | algorithm.py | ~5 行 |
| BETA + entropy floor | conf.py + algorithm.py | ~8 行 |

总计约 25 行代码修改，4 个文件。

### 与备份版本的本质区别

备份版本：hard override + 全部 sample 参与 PPO → gradient 污染（但实际影响小）
新方案：hard override + override sample 不参与 policy loss → gradient 隔离

这是一个有意义的机制改进：保留 expert 保证生存的优点，同时消除对 PPO 策略梯度的干扰。
