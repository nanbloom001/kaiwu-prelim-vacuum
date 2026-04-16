# 训练全周期瓶颈分析报告

**日期**: 2026-04-15
**目的**: 为网络框架改动评估提供完整数据支撑
**训练周期**: v4 (2026-04-12) → v5 → v5.1 → v5.2 → v5.3 → v5.4 (2026-04-14)
**当前最佳 WinRate**: 88.2% (v5.4, 110 ep)
**目标 WinRate**: >95%

---

## 一、训练概览与版本迭代

### 1.1 版本谱系

```
随机权重
  └─ v4: 参数回调 + reward 重平衡 + Expert Logit Bias
       │  step 0→7606, 2736 ep, 1.5h
       │  CPS 0.29→0.89, comp 100%→68%, entropy 2.0→0.08 塌缩
       │
       └─ v5-step4300 (resume)
            │  CPS=0.862, comp=100%, entropy=0.84
            │
            ├─ v5: prob 存储修复 + reward 增强 + entropy floor
            │    └─ v5.1: 充电效率修复
            │         └─ v51-step4900 (resume)
            │              │  CC=7.0, entropy=0.37, CS=1035
            │              │
            │              └─ v5.2: NPC 惩罚 + revisit 三组件 + 课程训练
            │                   │  step 0→64k
            │                   │  Peak at step 8k-12k
            │                   │
            │                   └─ v52-step8500 (resume)
            │                        │  AvgCS=1320, MinCS=871, balanced score #1
            │                        │
            │                        ├─ v5.3: entropy floor + urgency 三段式 + outcome bonus
            │                        │    216 ep, WinRate 85.2%
            │                        │    碰撞 18 例(56%), 电池死 14 例(44%)
            │                        │
            │                        └─ v5.4: Expert 5 项 bug 修复
            │                             110 ep, WinRate 88.2%
            │                             碰撞 2 例(15%), 电池死 11 例(85%)
            └─ 当前 model.ckpt-resume.pkl (v5.4 自动快照)
                 clean_score=903, ep=120, entropy ~0.18
```

### 1.2 各版本关键指标

| 版本 | 训练步数 | CPS | WinRate | Entropy | 碰撞死 | 电池死 | 核心改动 |
|------|---------|-----|---------|---------|-------|-------|---------|
| v4 | 7606 | 0.89 | 82%* | 0.08(塌缩) | N/A | N/A | Logit Bias 首次引入 |
| v5 | ~4300 | 0.862 | — | 0.84 | — | — | prob 修复 + reward 增强 |
| v5.2 | 64k | 0.895 | 81.8% | 0.149 | — | — | NPC + revisit + 课程 |
| v5.3 | ~5500 | — | 85.2% | 0.12-0.21 | 18(56%) | 14(44%) | entropy floor + urgency |
| v5.4 | ~2000 | — | 88.2% | 0.18-0.21 | 2(15%) | 11(85%) | Expert 5 项 bug 修复 |

*v4 的 comp_rate 68% 不等于 WinRate，但反映相同趋势

---

## 二、瓶颈 1：Entropy 塌缩（已解决但仍有隐患）

### 2.1 问题描述

训练过程中策略 entropy 从 2.0（均匀随机）持续下降到 0.08（近确定性），导致模型丧失探索能力，无法适应 reward 变化或学习新行为。

### 2.2 发生数据

**v4 完整训练周期（step 0→7606）**：

| 阶段 | Step 区间 | Entropy | CPS | CompRate | ChargeCnt | 诊断 |
|------|----------|---------|-----|----------|-----------|------|
| 健康探索 | 28-2261 | 1.5-2.0 | 0.59 | 94.9% | 8.8 | 正常学习 |
| 开始衰退 | 2338-3886 | 1.0-1.5 | 0.76 | 87.4% | 4.4 | CPS 升但少充电 |
| 策略趋确定 | 3964-4882 | 0.5-1.0 | 0.85 | 87.0% | 2.9 | 探索减少 |
| 完全塌缩 | 5047-7606 | 0.08-0.5 | 0.85 | 81.7% | 3.2 | 18% 电死，无法恢复 |

**v5.2 再次塌缩（ENTROPY_FLOOR_COEF=0.2 太弱）**：
- Step 0-8k: entropy 0.15-0.20（表面稳定）
- Step 12k 后: entropy 持续降至 0.08（floor 太弱，无法阻止）

### 2.3 根因分析

```
Entropy 塌缩因果链：

cleaning_reward = 1.5 × cleaned  （每步，量级大）
  ↓
模型发现确定性清扫策略 → reward 最高
  ↓
entropy bonus (BETA=0.008) 梯度 << policy gradient
  ↓
策略收敛到 near-deterministic
  ↓
entropy 从 2.0 → 0.08
  ↓
模型无法探索替代策略（如充电、绕行）
  ↓
面对 Expert 引导的新行为时无法适应
```

**数学解释**：
- Policy gradient 量级 ≈ `∇log π(a|s) × A(s,a)` ≈ `5.0 × 1.5` = 7.5
- Entropy bonus 梯度量级 ≈ `BETA × ∇H(π)` ≈ `0.008 × 0.5` = 0.004
- Policy gradient 是 entropy 梯度的 **1875 倍**
- 即使 BETA 提到 0.012，比例仍有 625:1

### 2.4 解决方案与效果

| 方案 | 版本 | 效果 |
|------|------|------|
| BETA 0.005→0.008 | v4 | 无效，仍然塌缩 |
| BETA 0.008→0.012 | v5 | 延缓但未阻止 |
| ENTROPY_FLOOR_COEF=0.2 | v5.2 | 延缓但 step 12k 后仍塌缩 |
| **ENTROPY_FLOOR_COEF=1.0** | **v5.3** | **成功：entropy 稳定 0.12-0.21** |

当前 entropy floor 机制（algorithm.py L156-164）：

```python
if entropy_val < Config.ENTROPY_FLOOR:  # 0.5
    floor_gap = Config.ENTROPY_FLOOR - entropy_val  # e.g., 0.5 - 0.1 = 0.4
    effective_beta = self.var_beta + Config.ENTROPY_FLOOR_COEF * floor_gap
    # = 0.012 + 1.0 * 0.4 = 0.412（比基础 beta 强 34 倍）
```

### 2.5 残留隐患

1. **entropy 被强制维持，不是自然产生**：模型本身倾向确定性，靠外部力量维持探索
2. **entropy 0.12-0.21 区间太低**：模型难以学习全新行为（如自主充电）
3. **峰值策略 entropy ≈ 0.15**：只够跟随 Expert 引导，不够独立探索

---

## 三、瓶颈 2：Expert Logit Bias 与 PPO 梯度对抗（核心结构性问题）

### 3.1 问题描述

Expert 通过 logit bias 引导模型充电，但 PPO 的策略梯度与 Expert 引导方向相反，导致"模型被引导去充电，但 PPO 学到的是充电不好"。

### 3.2 机制详解

```
训练时 predict() 流程：

Step 1: MLP → logits = [0.2, 5.1, 0.3, 0.1, -5.0, 0.2, 0.1, 0.3]
        模型认为清扫(action 1)最好，充电(action 4)最差

Step 2: Expert logit_bias = [0, 0, 0, 0, +6.0, 0, 0, 0]
        Expert 说应该充电

Step 3: 合并 logits = [0.2, 5.1, 0.3, 0.1, 1.0, 0.2, 0.1, 0.3]
        充电方向仍然不如清扫（5.1 > 1.0）

Step 4: 采样 → action = 1（清扫），prob 存 clean_prob（不含 bias）

Step 5: PPO 更新：
        - 模型选了 action 1（清扫），reward = +1.5（清扫 reward）
        - advantage > 0 → 增强清扫概率
        - action 4（充电）没被选，模型不会增强充电
        → PPO 持续强化"清扫好、充电差"
```

**关键冲突**：

| 系统 | 目标 | 方向 |
|------|------|------|
| Expert logit bias | 引导充电 | 充电 action logit ↑ |
| PPO policy gradient | 最大化 reward | 清扫 action logit ↑（因为清扫 reward 恒正）|
| Entropy bonus | 维持探索 | 所有 action logit 趋均匀 |

三个系统在互相打架。

### 3.3 已采取的措施

**v5 修复：clean prob 存储（agent.py L169-183）**

```python
if not use_hard_override and np.any(expert_bias > 0):
    # 从 biased 分布采样行为
    biased_logits = logits + np.array(expert_bias)
    biased_prob = self._legal_soft_max(biased_logits, legal_arr)
    action = self._legal_sample(biased_prob)  # 采样用 biased
else:
    action = self._legal_sample(clean_prob)

# 但 PPO 存储的是 clean_prob（不含 bias）
return ActData(
    action=[action],
    prob=list(clean_prob),  # ← 关键：PPO ratio 用 clean_prob
)
```

效果：PPO 不再直接惩罚 Expert 引导的行为。但间接问题仍在——模型自身 logits 没有学到充电偏好。

### 3.4 未解决的根本问题

**即使 prob 存储正确，模型仍然不学充电**：

1. Expert bias 3-8 太弱，压制不过模型清扫 logits（5+）
2. 模型 80% 时间在清扫 → 清扫 logits 持续增强
3. 充电只在 Expert 强制时发生 → 充电 logits 从不被自然强化
4. 当 Expert 触发条件失败时 → 模型完全没有充电意识

---

## 四、瓶颈 3：碰撞死亡（v5.4 修复 89%，剩余 15%）

### 4.1 v5.3 碰撞死亡详细日志（18 例）

| ep | 地图类型 | bat/max | 比例 | NPC距离 | mode | 最后5步modes | 充电量 |
|----|---------|---------|------|---------|------|-------------|--------|
| 11 | anchor | 277/300 | 92% | 2 | 2 | {2} | 高 |
| 22 | anchor | 204/300 | 68% | 3 | 2 | {2} | 中 |
| 61 | mild | 128/300 | 43% | 2 | 2 | {0,2} | 中 |
| 66 | mild | 216/220 | 98% | 3 | 2 | {0,2} | 高 |
| 73 | mild | 338/420 | 80% | 2 | 2 | {0,2} | 高 |
| 92 | broad | 203/420 | 48% | 2 | 2 | {0,2} | 中 |
| 101 | anchor | 105/260 | 40% | 2 | 2 | {2} | 低 |
| 104 | anchor | 117/300 | 39% | 2 | 2 | {2} | 低 |
| 110 | anchor | 215/300 | 72% | 2 | 2 | {2} | 高 |
| 128 | broad | 213/420 | 51% | 2 | 2 | {2} | 中 |
| 134 | mild | 188/260 | 72% | 2 | 2 | {2} | 中 |
| 172 | broad | 272/340 | 80% | 2 | 2 | {0,2} | 高 |
| 189 | anchor | 32/300 | 11% | 2 | 2 | {1,2} | 极低 |
| 191 | mild | 212/220 | 96% | 3 | 2 | {0,2} | 高 |
| 195 | broad | 68/300 | 23% | 3 | 2 | {0,2} | 低 |
| **204** | **broad** | **667/720** | **93%** | **2** | **2** | **{0,2}** | **极高** |
| 205 | broad | 111/560 | 20% | 3 | 2 | {0,2} | 低 |
| 214 | mild | 225/260 | 87% | 2 | 2 | {0,2} | 中 |

**关键特征**：
- **全部 18 例** 在 mode=2（Expert 充电导航）下发生
- NPC 距离全部 ≤ 3（近距碰撞）
- ep 204 电量 93% 完全不需要充电 → Expert 误触发

### 4.2 典型碰撞轨迹

**ep 204（93% 电量误触发，broad map 10，4 机器人）**：

```
s35:  bat=686/720  slack=640  npc=59  mode=0  act=0  ← 正常清扫
s41:  bat=680/720  slack=631  npc=10  mode=0  act=0  ← NPC 进入视野
s42:  bat=679/720  slack=631  npc=8   mode=0  act=0  ← NPC 在 8 格处
...
s53:  bat=668/720  slack=631  npc=5   mode=2  act=0  ← Expert 强制进入充电模式！
s54:  bat=667/720  slack=630  npc=2   mode=2  act=0  ← NPC 在 2 格 → 碰撞死亡
```

**ep 189（低电量冲向充电桩，anchor map 2）**：

```
s250: bat=51/300  slack=30.0  npc=9  mode=0  act=1  ← 正常清扫
s251: bat=50/300  slack=30.0  npc=7  mode=0  act=1  ← NPC 靠近
s252: bat=49/300  slack=28.0  npc=7  mode=0  act=5  ← 继续清扫
s253: bat=48/300  slack=28.0  npc=9  mode=1  act=4  ← 进入充电模式
...
s268: bat=33/300  slack=17.0  npc=4  mode=2  act=2  ← Expert 导航冲向充电桩
s269: bat=32/300  slack=17.0  npc=2  mode=2  act=2  ← NPC 在 2 格！继续冲 → 碰撞
```

### 4.3 根因分析（三层防线失效）

**链条 1：filter_actions 漏洞**

```python
# 原代码 (expert.py L98-109)
if npc_dist <= 5:
    sx = (1 if ndx > 0 else -1) if ndx != 0 else 0
    sz = (1 if ndz > 0 else -1) if ndz != 0 else 0
    if dx == sx and dz == sz:  # ← 只拦截方向完全一致的动作
        legal[idx] = 0
```

问题：NPC 在东北 (1,1)，走北 (0,1) 或东 (1,0) 不被拦截，但都会靠近 NPC。

**链条 2：A* 第 3 级 fallback 放弃 NPC 避让**

```python
# npc_weight=0.0 → 完全忽略 NPC
cost_map = self._build_cost_map(prep, npc_weight=0.0)
# A* 规划的路径直接穿过 NPC 旁边
```

**链条 3：充电 bias 压过安全 bias**

充电 bias 3-100 >> NPC 负 bias -2。NPC dist=2 时，充电 action bias=8，NPC action bias=-2，仍选充电方向。

**链条 4：return_mode 持久化**

ep 204：电量 93% 时 return_mode 被激活（因为 broad 大地图 charger_dist+margin > battery），充电后 return_mode 无法退出（需 on_charger AND ≥ 95%），持续导航导致碰撞。

### 4.4 v5.4 修复效果

| 修复项 | 改动 | 效果 |
|-------|------|------|
| Fix 1: filter_actions 扩展 | dist ≤ 3 拦截所有缩短距离动作 | — |
| Fix 2: 移除 npc_weight=0.0 | 删除第 3 级 fallback | — |
| Fix 3: NPC 近距抑制充电 bias | dist ≤ 4 时不施充电正 bias | — |
| Fix 4: return_mode 修复 | 85% 高电量退出 + 动态 LOW_BATTERY_RATIO | — |
| Fix 5: UNEXPLORED_COST 3.0→1.8 | A* 早期更容易找到路径 | — |
| **综合效果** | **碰撞 18→2（-89%）** | **WinRate 85.2%→88.2%** |

### 4.5 剩余问题

v5.4 仍有 2 例碰撞（15%），可能原因：
- NPC dist=1（相邻格）时 filter_actions 可能无合法动作兜底
- 多机器人场景中其他机器人推挤导致

---

## 五、瓶颈 4：电池死亡（当前主要瓶颈）

### 5.1 v5.3 电池死亡详细日志（14 例）

| ep | bat/max | slack | 最后20步充电数 | 最后20步modes | profile | 充电桩数 | 机器人数 |
|----|---------|-------|--------------|-------------|---------|---------|---------|
| 5 | 1/300 | -14.0 | 2/20 | {1,2} | anchor | 4 | 1 |
| 49 | 1/260 | -9.4 | **0/20** | {1} | mild | 3 | 1 |
| 68 | 1/300 | -16.0 | **0/20** | {1} | anchor | 4 | 1 |
| 83 | 1/380 | -15.2 | **0/20** | {1} | mild | 4 | 2 |
| 97 | 1/300 | -11.0 | **0/20** | {1} | anchor | 4 | 1 |
| 130 | 1/120 | **-47.0** | **0/20** | {1} | broad | 1 | 4 |
| 135 | 1/320 | -20.8 | **0/20** | {1} | broad | 1 | 3 |
| 136 | 1/200 | **-73.0** | **0/20** | {1} | broad | 1 | 2 |
| 143 | 1/200 | -28.0 | **0/20** | {1} | broad | 3 | 4 |
| 149 | 1/300 | -12.0 | **0/20** | {1} | anchor | 4 | 1 |
| 159 | 1/300 | -16.0 | **0/20** | {1} | anchor | 4 | 1 |
| 161 | 1/120 | -8.0 | **0/20** | {0,1} | broad | 3 | 1 |
| 224 | 1/120 | -33.0 | **0/20** | {1} | broad | 2 | 2 |
| 232 | 1/260 | -9.4 | **0/20** | {1} | broad | 2 | 3 |

**关键特征**：
- **全部 14 例** 在 mode=1（清扫模式）下到电量归零
- **13/14 例** 最后 20 步充电次数 = 0（Expert 完全未触发充电）
- broad 配置最多（7/14），且多为 charger_count=1 + robot_count=3-4
- 小电池（battery_max=120-200）尤其严重：slack -47、-73 表明充电桩远在可达距离外

### 5.2 v5.4 电池死亡诊断（3214 条日志）

v5.4 修复碰撞后，通过在 Expert `_evaluate_return()` 添加诊断日志收集 3214 条数据。

**推翻的假设**：

| 假设 | 验证结果 |
|------|---------|
| 充电桩 organ 未出现 → charger_list 空 | ❌ chargers=3，始终找到充电桩 |
| A* 找不到路径 → expert_action=None | ❌ 全部有有效 action，path=20-52 |
| NPC 太近 → Fix 3 抑制充电 bias | ❌ NPC 距离 59-98（远），未被抑制 |
| 充电桩格子 impassable | ❌ path 非空，A* 成功规划 |

**确认的根因：non-emergency bias 太弱**

battery_ratio 分布：

| 区间 | 数量 | 占比 | bias 值 | vs 模型清扫 logit |
|------|------|------|---------|-----------------|
| ≤0.10（emergency） | 1105 | 27% | 100.0 | 足够强 ✅ |
| 0.10-0.15 | 1271 | 31% | 3-8 | **太弱 ❌** |
| 0.15-0.20 | 1664 | 41% | 3-8 | **太弱 ❌** |

**73% 的电池死亡场景中，bias（3-8）无法覆盖模型的清扫偏好（logit 5+）**

**典型计算**（来自诊断日志）：

```
bat=68/340 ratio=0.200 act=0 chargers=3 path=52 dist=51 margin=22.6
→ slack = 68 - 51 = 17
→ urgency = clip(1 - 17/22.6, 0.2, 1.0) = 0.25
→ bias = 3.0 + 5.0 × 0.25 = 4.25

模型清扫方向 logit ≈ +5.0
Expert 充电方向 logit ≈ -5.0（模型认为充电不好）
Effective：-5.0 + 4.25 = -0.75（仍低于清扫 +5.0）
→ 模型忽略 Expert，继续清扫
→ 电量下降到 0.10 → emergency bias=100 → 但只剩 ~34 步 → 可能来不及
```

### 5.3 电池死亡根因链

```
Expert return_mode 触发（ratio=0.32，bat≈109/340）
→ A* 找到路径（path=52，dist=51）
→ expert_action 有效（act=0，方向正确）
→ bias = 3-8（非 emergency，因为 ratio > 0.10）
→ bias 太弱，模型清扫 logit（+5）压过充电 bias（+4.25）
→ 机器人继续清扫而非充电
→ 电量持续下降到 ratio=0.10
→ emergency bias=100 终于触发
→ 但只剩 ~34 步，可能来不及
→ 电池死亡
```

### 5.4 v5.4 修复后状态

碰撞从 18→2（-89%），但电池死亡从 14→11，成为 85% 的死因。WinRate 85.2%→88.2%。

---

## 六、瓶颈 5：GAE 长周期 Credit 衰减（结构性硬限制）

### 6.1 问题描述

充电决策需要提前 50-200 步规划（发现电量低 → 导航到充电桩 → 等待充电），但 PPO 的 GAE（Generalized Advantage Estimation）在长距离上衰减到接近零，使得充电决策几乎无法获得来自"存活到终点"的正向 credit。

### 6.2 数学分析

```
GAE 参数：γ = 0.99, λ = 0.95
每步衰减因子：γλ = 0.9405

充电决策的 credit 传播：

Step 0:    决定去充电（battery_ratio=0.30）
Step 30:   走到充电桩        GAE 衰减 = 0.9405^30 = 15.8%
Step 50:   开始充电          GAE 衰减 = 0.9405^50 = 4.7%
Step 80:   充电完成          GAE 衰减 = 0.9405^80 = 0.7%
Step 2000: episode 结束存活  GAE 衰减 = 0.9405^2000 ≈ 0%

→ 充电完成时的 reward (+3.0) 传回决策步骤只剩 3.0 × 0.7% = 0.021
→ 而同一时刻的清扫 reward (+1.5) 全额计入
→ 清扫 reward 在 credit 传播中比充电 reward 强 71 倍
```

### 6.3 影响

**这不是 bug，是 PPO + MLP 架构的内在限制**：

1. **模型无法从 episode 终点的存活 reward 反向学习充电决策**——信号衰减太快
2. **只能依赖局部信号**（urgency_penalty、charge_reward）——但这些信号太弱
3. **Expert logit bias 是必要的补偿**——用外部知识填补 PPO 长周期 credit 的空白
4. **但 Expert bias 本身又不参与 PPO 梯度**——导致模型永远不内化充电决策

### 6.4 当前缓解措施及局限

| 措施 | 机制 | 局限 |
|------|------|------|
| urgency_penalty (-0.3 ~ -1.2) | 局部信号，不依赖 GAE | 太弱（vs 清扫 +1.5）|
| charge_reward (+3.0) | 充电时的即时 reward | 只在充电时触发（稀疏）|
| Expert logit bias (3-100) | 外部引导充电决策 | 不参与 PPO 梯度 |
| outcome_bonus (死亡惩罚) | episode 结束时的全局信号 | GAE 衰减后几乎为零 |

---

## 七、瓶颈 6：Reward 结构失衡

### 7.1 当前 Reward 16 分量

```python
reward = (
    cleaning_reward       # 1.5 × cleaned        量级: 0 ~ +1.5/步   频率: 每步
    + streak_bonus        # 0.15 × streak        量级: 0 ~ +0.75/步   频率: 连续清扫
    + edge_bonus          # 0.06 × dirty_adj     量级: 0 ~ +0.06/步   频率: 常见
    + explore_reward      # 0.05 × explored      量级: 0 ~ +0.30/步   频率: 早期多
    + frontier_reward     # 0.15 × frontier      量级: 0 ~ +0.15/步   频率: 常见
    + charger_reward      # 0.40 × approach      量级: 0 ~ +0.40/步   频率: 稀少
    + charger_path_explore # 0.12 × explore      量级: 0 ~ +0.48/步   频率: 稀少
    + charge_reward       # 3.0 × efficiency     量级: 0 ~ +3.0/步    频率: 极稀少
    + npc_penalty         # -3.0 × risk^1.5      量级: -3.0 ~ 0/步    频率: NPC近
    + npc_cleaned_penalty # -0.3 × cleaned       量级: -0.3 ~ 0/步    频率: 偶尔
    + revisit_penalty     # -0.12 × visits       量级: -0.36 ~ 0/步   频率: 常见
    + stuck_penalty       # -0.5 × invalid       量级: -0.5 ~ 0/步    频率: 偶尔
    + idle_penalty        # -0.1 × idle          量级: -0.1 ~ 0/步    频率: 少
    + dirty_approach      # 0.08 × direction     量级: 0 ~ +0.08/步   频率: 常见
    + efficiency_reward   # 0.3 × CPS_ema        量级: 0 ~ +0.075/步  频率: 后期
    + urgency_penalty     # -0.3 ~ -1.2          量级: -1.2 ~ 0/步    频率: 低电量
)
# clip to [-5.0, 5.0]
```

### 7.2 Reward 总量对比（典型 2000 步 episode）

| 分量 | 单步均值 | 总量 | 占比 |
|------|---------|------|------|
| cleaning + streak + edge + frontier + dirty_approach | ~+1.8 | **+3600** | **87%** |
| charge_reward | ~+0.01 | +15 | 0.4% |
| charger_reward + path_explore | ~+0.02 | +40 | 1.0% |
| urgency_penalty | ~-0.01（仅低电量时）| -20 | 0.5% |
| npc_penalty | ~-0.02 | -40 | 1.0% |
| 其他 | ~+0.1 | +200 | 5% |

**清扫 reward 总量是充电 reward 的 60 倍。** 从模型视角，最优策略就是不停清扫。

### 7.3 核心矛盾

```
从模型视角的 reward 景观：

不停清扫（永远不充电）：
  → 2000 步 × 1.5 = +3000 reward
  → 电池耗尽死亡：outcome_bonus 惩罚 -8.0（但 GAE 衰减到几乎为零）
  → 净 reward ≈ +3000

合理清扫 + 适时充电：
  → 1940 步 × 1.5 + 60 步充电 ≈ +2910 + +18 = +2928 reward
  → 存活到结束：+1.0（bonus）
  → 净 reward ≈ +2929

差距：3000 vs 2929 = 2.4%
→ PPO 判断"不停清扫"策略更好（因为充电时无法清扫）
→ 模型不学充电
```

---

## 八、瓶颈 7：模型架构限制

### 8.1 当前架构

```
输入特征 (74D 标量 + 21×21×3 局部地图 + 8×8×3 全局记忆)
  ↓
  MLP (隐藏层)
  ↓
  输出: 8D logits (动作) + 1D value (状态价值)
```

### 8.2 架构限制

| 限制 | 影响 |
|------|------|
| **无时序记忆（MLP，非 RNN/LSTM）** | 模型无法记住历史轨迹，只能根据当前观测做决策。无法形成"我已经走了 30 步还没到充电桩"的概念 |
| **74D 标量特征无充电历史** | 特征包含 battery_ratio、nearest_charger_dist，但不包含"上次充电是多久前"、"充电桩是否被占"等关键信息 |
| **单头 Value** | Value 网络预测的是单一标量，无法区分"清扫得分高但快没电"和"清扫得分低但电量充足"的不同价值结构 |
| **8D 动作空间无层次** | 只有原始方向动作，无法表达"先走到充电桩再充电"这样的复合动作 |
| **Resume 模式架构冻结** | 只能微调权重，不能改网络结构 |

### 8.3 与瓶颈的关联

```
无时序记忆
  → 无法追踪"去充电桩的路走了多少步"→ 无法判断是否该放弃当前路径
  → 无法记住"上次经过充电桩时被占"→ 可能重复去同一个被占的充电桩

74D 无充电历史
  → 模型不知道"我已经很久没充电了"
  → 只能靠 battery_ratio（当前值）判断 → 可能来不及

单头 Value
  → Value 网络无法学习"低电量 + 远离充电桩 = 极其危险"
  → 因为平均来看这个状态很少被采样到（大多数时候电量够）
```

---

## 九、训练曲线特征：Peak-Then-Decline 模式

### 9.1 现象

多次训练出现同一模式：训练初期指标快速上升，在某个 peak zone 达到最优，然后持续下降，无法恢复。

**v5.2 完整训练数据（step 0→64k）**：

| Step 区间 | CPS | Survival | CS×Surv | 趋势 |
|-----------|-----|----------|---------|------|
| 0-4k | 0.80→0.89 | 0.75→0.85 | 640→757 | 上升 |
| **4k-8k** | **0.89→0.90** | **0.85→0.82** | **757→738** | **Peak Zone** |
| 8k-12k | 0.90→0.88 | 0.82→0.78 | 738→686 | 下降 |
| 12k-64k | 0.88→0.60 | 0.78→0.50 | 686→300 | 持续恶化 |

### 9.2 Peak Zone 期间的关键指标

v52-step8500（peak zone 最佳点，在 12 个候选点中排名第 1）：

| 指标 | 值 |
|------|-----|
| AvgCS | 1320（最高） |
| MinCS | 871（无灾难局） |
| ColRate | 0.101（全场最低） |
| Entropy | 0.149 |
| VLoss | 147 |
| balanced_score | 1210.8 |

### 9.3 下降原因

```
Peak-Then-Decline 因果链：

模型在 peak zone 学到最优清扫策略
  ↓
继续训练 → 模型进一步优化清扫（确定性策略）
  ↓
Entropy 持续下降（尽管有 entropy floor）
  ↓
模型开始过拟合到高频场景 → 低频场景（低电量、NPC 近距）表现恶化
  ↓
WinRate 下降 → MinCS 下降 → CPS 下降
  ↓
训练后期策略质量不如 peak zone
```

**教训**：不应该"训练越久越好"。最佳 checkpoint 通常在训练曲线的 peak zone（总步数的 10-20% 处）。

---

## 十、综合瓶颈图谱

```
                         ┌──────────────────────────────────────────────┐
                         │           根本结构性问题                      │
                         │                                              │
                         │  1. Expert 主导充电 → 模型不学充电决策        │
                         │  2. 清扫 reward >> 充电 reward (60:1)        │
                         │  3. GAE 衰减使长周期 credit 为零              │
                         │  4. MLP 无时序记忆 → 无法追踪充电规划进度     │
                         │                                              │
                         └──────────┬───────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ↓               ↓               ↓
             ┌──────────┐   ┌──────────┐   ┌──────────────┐
             │ Entropy  │   │ Expert   │   │  Reward      │
             │ 塌缩     │   │ Bug      │   │  失衡        │
             │          │   │          │   │              │
             │ 已缓解   │   │ 碰撞     │   │ 清扫主导     │
             │ (floor)  │   │ 18→2     │   │ 充电弱       │
             │          │   │          │   │              │
             │ 残留:    │   │ 电池死亡 │   │urgency太弱   │
             │ 0.12-0.21│   │ 14→11    │   │ GAE衰减      │
             │ 无法学   │   │ bias太弱 │   │ credit ≈ 0   │
             │ 新行为   │   │          │   │              │
             └──────────┘   └──────────┘   └──────────────┘
                    │               │               │
                    └───────────────┼───────────────┘
                                    ↓
                         ┌──────────────────────┐
                         │  WinRate 天花板       │
                         │  当前: 88.2%         │
                         │  Expert 修复后: ~92% │
                         │  理论上限(当前架构):  │
                         │  Expert 精修: ~95%   │
                         │  需框架改动: >95%    │
                         └──────────────────────┘
```

---

## 十一、针对网络框架改动的评估

### 11.1 当前架构能解决什么

| 问题 | 当前架构能解决吗 | 方案 |
|------|----------------|------|
| 碰撞死亡 | ✅ 已基本解决 | Expert filter_actions 修复 |
| 电池死亡（bias 太弱） | ✅ 可以 | 提高 bias + 降低 emergency 阈值 |
| Entropy 塌缩 | ✅ 已解决 | entropy floor |
| Reward 失衡 | ⚠️ 部分可以 | 提高 urgency + 降低清扫权重 |
| **模型自主充电** | **❌ 不行** | 需要 credit 传播或架构改动 |
| **多步决策规划** | **❌ 不行** | 需要时序记忆 |
| **Peak-Then-Decline** | **❌ 不行** | 需要更好的 exploration 策略 |

### 11.2 网络框架可能改动方向

#### 方向 A：加入 LSTM/GRU 时序层

**解决的问题**：
- 模型可以记住"去充电桩走了多少步"
- 可以记住"上次充电桩被占"
- Value 网络可以学习基于历史的"电量趋势"判断

**风险**：
- Resume 模式下无法改架构（输入输出维度会变）
- 需要从随机权重重新训练
- LSTM 训练更不稳定（梯度消失/爆炸）
- 推理延迟增加

**可行性**：如果框架允许修改网络结构，这是最高价值改动。

#### 方向 B：多步 Return 或 Monte Carlo Return

**解决的问题**：
- 减轻 GAE 衰减问题
- 充电决策可以获得更多来自后续步骤的 credit

**实现**：
- 使用 n-step return（n=50 或 n=100）替代 GAE 的 1-step TD
- 或者使用 Monte Carlo return（整个 episode 的累计 reward）
- 只需修改 algorithm.py 的 loss 计算，不改网络结构

**风险**：
- 高方差（单个 episode 的 return 波动大）
- 需要 episode 结束才能更新（当前已是 on-policy PPO）
- 可能需要更大的 batch size

**可行性**：不改网络结构，Resume 模式可行。推荐优先尝试。

#### 方向 C：Reward Shaping（基于势函数）

**解决的问题**：
- 在不改最优策略的前提下，提供更密集的充电引导信号
- 不依赖 GAE 传播长周期 credit

**实现**：
- 基于 battery_ratio 和 nearest_charger_dist 设计势函数 F(s)
- reward_shaped = reward + γ × F(s') - F(s)
- 理论保证不改变最优策略（Ng et al., 1999）

**风险**：
- 设计好的势函数需要领域知识
- 如果势函数设计不当，可能引入新的 bias

**可行性**：只改 reward 函数，不改网络。推荐尝试。

#### 方向 D：分层策略（Hierarchical RL）

**解决的问题**：
- 高层策略决定"清扫/充电/躲避"模式
- 低层策略在模式内选择具体动作
- 充电决策变为高层的一个离散选择，credit 传播更直接

**风险**：
- 架构改动大
- 训练复杂度显著增加
- Resume 模式不可行

**可行性**：需要从头训练，长期方向。

### 11.3 推荐优先级

| 优先级 | 改动 | 改动范围 | 预期收益 | 可行性 |
|--------|------|---------|---------|--------|
| **1** | Reward 重构（电量感知衰减 + urgency 放大） | preprocessor.py ~30 行 | WinRate +3-5% | Resume 可行 |
| **2** | Expert bias 上限降低（emergency 100→15） | expert.py ~10 行 | 让 RL 接管充电 | Resume 可行 |
| **3** | n-step return 或 MC return | algorithm.py ~50 行 | 减轻 GAE 衰减 | Resume 可行 |
| **4** | 势函数 reward shaping | preprocessor.py ~40 行 | 密集充电信号 | Resume 可行 |
| **5** | LSTM 时序层 | 网络结构重构 | 根本性解决时序问题 | 需重新训练 |

**建议**：先做 1-2（50 行代码），观察 RL 能否学会自主充电。如果仍不行，再做 3-4。如果还不行，考虑 5（但需要评估是否值得重新训练）。

---

## 十二、附录：训练数据汇总

### A. v5.3 完整 episode 指标

| 窗口 | Ep | WinRate | AvgCS | MinCS | 失败数 |
|------|----|---------|-------|-------|--------|
| ep 1-25 | 23 | 87.0% | 863 | 474 | 3 |
| ep 26-50 | 24 | **95.8%** | 959 | 431 | 1 |
| ep 51-75 | 22 | 81.8% | 918 | 295 | 4 |
| ep 76-100 | 25 | 88.0% | 832 | 210 | 3 |
| ep 101-125 | 24 | 87.5% | 914 | 82 | 3 |
| ep 126-150 | 23 | **69.6%** | 766 | 120 | **7** |
| ep 151-175 | 23 | 87.0% | 888 | 114 | 3 |
| ep 176-200 | 23 | 87.0% | 887 | 213 | 3 |
| ep 201-225 | 22 | 81.8% | 784 | 53 | 4 |
| ep 226-250 | 7 | 85.7% | 756 | 247 | 1 |

### B. v5.3 Training Metrics 采样

| Step | CS | Entropy | VLoss | PLoss | ChgCnt | EpCnt |
|------|-----|---------|-------|-------|--------|-------|
| 31 | 1042 | 0.14 | 91 | -6.4 | 4.75 | 4 |
| 222 | 891 | 0.13 | 110 | -6.0 | 4.75 | 46 |
| 562 | 966 | 0.15 | 119 | -6.8 | 5.50 | 110 |
| 1089 | 1224 | 0.17 | 162 | -7.5 | 11.75 | 195 |
| 1576 | 943 | 0.14 | 127 | -5.1 | 3.50 | 269 |
| 2686 | 955 | 0.13 | 145 | -5.9 | 5.00 | 454 |
| 3820 | 993 | 0.12 | 84 | -7.1 | 6.50 | 638 |
| 4486 | 954 | 0.18 | 124 | -3.9 | 7.25 | 754 |
| 5410 | 1107 | 0.19 | 119 | -5.7 | 7.00 | 915 |

### C. 地图难度分析（v5.3）

| 地图 | Episode | WIN | FAIL | AvgCS | 死亡率 |
|------|---------|-----|------|-------|--------|
| map 3 | 13 | 13 | 0 | 918 | 0% |
| map 5 | 15 | 15 | 0 | 952 | 0% |
| map 8 | 13 | 13 | 0 | 895 | 0% |
| map 4 | 16 | 15 | 1 | 840 | 6.3% |
| map 10 | 9 | 8 | 1 | 917 | 11.1% |
| map 7 | 15 | 13 | 2 | 867 | 13.3% |
| map 1 | 17 | 14 | 3 | 829 | 17.6% |
| map 9 | 11 | 9 | 2 | 818 | 18.2% |
| map 2 | 18 | 14 | **4** | 822 | **22.2%** |
| map 6 | 8 | 5 | **3** | **656** | **37.5%** |

### D. 当前模型架构参数

```
输入:
  - local_map:   21×21×3 = 1323D (obstacle, cleaned, dirt)
  - global_mem:  8×8×3   = 192D  (explored, dirt, visit_heat)
  - scalar:      74D (battery, NPC, charger, mode, etc.)
  - legal_action: 8D
  总计: 1597D

网络: MLP (具体层数由框架定义)
输出: 8D logits + 1D value

PPO 参数:
  γ = 0.99, λ = 0.95
  LR = 5e-5, BETA = 0.012, CLIP = 0.15
  ENTROPY_FLOOR = 0.5, ENTROPY_FLOOR_COEF = 1.0
  VF_COEF = 0.5, GRAD_CLIP = 0.5
```

---

**报告完成日期**: 2026-04-15
**数据来源**: v4-v5.4 全周期训练日志、Expert 诊断日志（3214 条）、死亡轨迹日志、checkpoint 评估
