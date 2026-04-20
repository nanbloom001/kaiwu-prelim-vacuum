# 深度问题诊断报告 — 2026-04-20

> **报告版本**: v1.0  
> **分析会话**: `20260420-091040`（learner 67K steps, 120+ episodes, ~52K global_step_since_resume）  
> **数据来源**: curriculum_state.json, learner log, aisrv helper logs (pid440/441/445), curriculum signal JSONs, 完整 reward/constraint/curriculum 源码

---

## 第一章：Executive Summary（执行摘要）

训练系统当前处于 **warmup 阶段锁死** 状态。尽管表面指标（win_rate=72.5%, 奖励稳步上升, entropy 健康下降）显示"有改善"，但两个结构性缺陷——**planner-policy 偏离率 85%** 和 **return_stall_rate 55%**——形成了一个**自我强化的恶性循环**，使系统永远无法通过 warmup 出口门（要求 planner≤80%, return_stall≤50%）。

**核心诊断**：这不是一个超参调优问题，而是一个**奖励结构盲区问题**。Agent 在 return 模式（mode=4）下没有任何有效的学习信号来纠正其导航行为，导致它进入返回模式后反复原地打转或偏离充电站，最终耗尽电池死亡。

**关键证据**：所有 DEATH_TRAJ 记录均显示 mode=4 且 slack 持续为负（-5 到 -40），NPC 计数不降反升（agent 远离充电站），return_action_teacher_active_rate 仅 8-14%，charging net reward 仅 0.006（vs cleaning 0.090，信号弱 15 倍）。

---

## 第二章：当前训练趋势

### 2.1 积极信号

| 指标 | 初始值 | 当前值 | 趋势 |
|------|--------|--------|------|
| episode_reward | 18.89 | 98.28 | ▲ 持续上升 |
| clean_score | 139.85 | 390.81 | ▲ 稳定上升 |
| entropy_loss | 1.99 | 1.55→1.43 | ▼ 健康下降 |
| win_rate (40-ep) | — | 0.725 | 中等 |
| policy_loss | — | -0.02 ~ -0.07 | 正常范围 |
| data_fetch | — | 5-8ms | ▲ pipeline 健康 |
| real_train | — | 83-98ms | 正常 |

### 2.2 停滞信号

| 指标 | 当前值 | 门限 | 偏差 | 状态 |
|------|--------|------|------|------|
| planner_policy_divergence_rate | **0.850** | ≤0.80 (S0 exit) | +0.050 | **🔴 阻塞** |
| return_stall_rate | **0.550** | ≤0.50 (S0 exit) | +0.050 | **🔴 阻塞** |
| zero_charge_battery_fail_rate | **0.545** | ≤0.50 (strict) | +0.045 | **🔴 阻塞** |
| return_efficiency_ratio | **0.090** | — | 极低 | **🔴 结构性** |
| charger_access_probe_bonus | **0.000** | >0 | 永不触发 | **🔴 信号缺失** |
| charging net reward | **0.006** | — | 微弱 | **🔴 信号不足** |
| curriculum_stagnation_level | **3** (max) | — | ≥8 windows | **🔴 锁定** |

### 2.3 训练健康但行为学习停滞

训练 pipeline 本身完全健康：
- nan_skip_rate=0.0004，gradient 稳定
- checkpoint 每 500 步正常保存
- data_fetch 仅 5-8ms（reverb pipeline 畅通）
- policy_loss、value_loss 在正常范围内波动

**结论**：问题不在训练 infra，而在 agent 的行为策略——它学会了"清扫赚分"这个局部最优，但未学会"管理电池以存活"这个关键能力。

---

## 第三章：主要问题分类

### 3.1 问题类型判定

**分类：奖励结构盲区 + 教师覆盖缺口 → 导致 return 导航能力缺失 → 形成局部最优锁定**

这不属于以下任何一种常见分类：
- ❌ 不是"训练不稳定"（gradient 正常，loss 合理）
- ❌ 不是"探索不足"（entropy 1.43 仍有合理探索）
- ❌ 不是"数据管道问题"（data_fetch 健康）
- ❌ 不是"超参不当"（PPO 参数合理，lr=3e-5, clip=0.15）
- ✅ **是"奖励信号缺失"**：return 模式下无 planner alignment 信号
- ✅ **是"教师覆盖缺口"**：return_action_teacher 活跃率仅 8-14%
- ✅ **是"系统耦合锁死"**：stagnation → conservative weights → reduced exposure → slower learning → stagnation

### 3.2 问题架构图

```
┌─────────────────────────────────────────────────────────┐
│                 WARMUP 阶段锁死                         │
│                                                         │
│  planner_divergence=0.85 ──┐                            │
│  return_stall=0.55 ────────┤                            │
│        ↓                   │                            │
│  stagnation_level=3 ───────┤                            │
│        ↓                   ↓                            │
│  invalid_for_promotion ← requires_reward_revision       │
│        ↓                                                │
│  conservative weights ← anchor=0.60, mild=0.35          │
│        ↓                                                │
│  reduced scenario diversity → slower behavioral change  │
│        ↓                                                │
│  planner_divergence stays high → stagnation persists    │
│                                                         │
│  ──── 恶性循环 ────────────────────────────────          │
└─────────────────────────────────────────────────────────┘
```

---

## 第四章：局部最优评估

### 4.1 是否陷入局部最优？

**是的。** Agent 找到了一个稳定但次优的策略：

**当前策略（局部最优）**：
> "尽可能多清扫 → 偶尔充电（靠运气撞到充电站）→ 有时存活到完成（72.5%）→ 充电失败时死亡（27.5%）"

**全局最优策略应该是**：
> "清扫 + 主动监控电量 → 在 planning 阶段沿 planner 路径返回 → 高效充电 → 继续清扫 → 高存活率（>85%）"

### 4.2 局部最优的行为特征

| 特征 | 证据 |
|------|------|
| **清扫能力已学会** | avg_clean_per_step=0.448, cleaning 占正向奖励 43.4% |
| **返回能力未学会** | return_efficiency_ratio=0.090, return_stall_rate=0.550 |
| **充电依赖运气** | zero_charge_battery_fail_rate=0.545 — 过半电池死亡根本没充过电 |
| **返回时原地打转** | DEATH_TRAJ: mode=4, npc 计数上升, slack 持续为负 |
| **planner 信号被忽略** | planner_policy_divergence_rate=0.850 |

### 4.3 为什么 agent 卡在这个局部最优？

Agent 发现了一个"能拿正奖励"的策略：
- cleaning（0.090/step）是最大的正信号
- explore（0.040）和 streak（0.032）紧随其后
- 这些信号**仅在清扫模式下触发**

而"改进返回行为"的信号几乎不存在：
- charging net total 仅 0.006/step — 只有 cleaning 的 1/15
- planner_alignment 在 return 模式下**完全不生效**
- charger_access_probe_bonus = 0.0 — 永远不触发

Agent 的 gradient 信号几乎全部来自"如何更好地清扫"，没有信号告诉它"如何更好地返回"。

---

## 第五章：Planner-Policy 偏离 & Return Stall 模式分析

### 5.1 Planner-Policy 偏离的结构性原因

**关键发现**：`planner_alignment_reward` 的生效条件是：

```python
if battery_state == "safe" and self.current_mode not in (self.MODE_CONTRACT, self.MODE_RETURN):
    if planner_matches:
        planner_alignment_reward = +0.03
    elif planner_diverges:
        planner_alignment_reward = -0.03
```

这意味着：
1. **Return 模式（mode=4）下没有 planner alignment 信号** — 这恰恰是最需要跟随 planner 的时刻
2. **Contract 模式（mode=5）下也没有** — 前往充电站的过渡阶段同样缺失
3. **Battery 非 safe 状态下也没有** — 而需要返回时 battery 通常已经进入 planning/critical
4. **三个条件取交集** → planner alignment 信号几乎只在"安全电量 + 清扫模式"时生效

**影响量化**：mode_usage 分布为 harvest=0.359, contract=0.421, return=0.149, expand=0.003。即约 57% 的时间（contract+return）完全没有 planner alignment 信号。而 planner_policy_divergence_rate 是全模式统计的，导致这 57% 的"无信号时间"里的偏离行为永远不会被纠正。

### 5.2 Return Stall 的行为病理

**DEATH_TRAJ 一致性模式**（所有 battery failure 均来自 pid441/445）：

| Episode | 死亡步数 | slack 范围 | NPC 趋势 | 行为描述 |
|---------|----------|-----------|----------|---------|
| ep6 (pid441) | s281-300 | -5 → -33 | 85→48（后半降） | mode=4, act 混乱(5,6,7,0 交替) |
| ep7 (pid441) | s281-300 | -9 → -36 | 79→85（上升！） | mode=4, act=1 重复（直线前行远离） |
| ep12 (pid441) | s401-420 | -15.8 固定 | 102→87（微降但高） | mode=4, act=5 重复（单一方向） |
| ep14 (pid441) | s992-1011 | -11.6 固定 | 90→90（不变） | mode=4, act=6 重复（原地打转） |
| ep18 (pid441) | s580-599 | -3 → -16 | 67→48（降但仍远） | mode=4, act=1 重复（直线远离） |
| ep6 (pid445) | 类似 | 类似 | 类似 | 类似 |

**共性**：
1. **全部 mode=4（return）**：agent 知道要返回，但不会导航
2. **slack 持续为负**：距充电站越来越远或原地打转
3. **NPC 计数高或上升**：agent 在远离充电站的区域徘徊
4. **action 重复单一**：act=1（直行远离）或 act=5/6（单方向移动），没有修正行为
5. **没有路径调整**：agent 不会根据 planner 的 anchor/target 信息调整方向

### 5.3 Return Stall 的根因链

```
return_action_teacher_active_rate = 8-14% （教师极少覆盖 return 情况）
        ↓
planner_alignment 在 return 模式不生效 （奖励信号空白）
        ↓
agent 进入 return 模式后无学习信号
        ↓
行为随机化 → 原地打转或直线远离
        ↓
return_efficiency_ratio = 0.09 → return_stall_rate = 0.55
        ↓
电池耗尽 → battery_fail_rate = 0.275
        ↓
zero_charge_battery_fail_rate = 0.545
        ↓
stagnation 触发 → conservative weights → 学习更慢
```

---

## 第六章：充电奖励结构与错误局部最优

### 6.1 充电奖励信号强度分析

| 充电奖励组件 | 平均值/step | 占正向奖励比例 | 实际效果 |
|-------------|-------------|---------------|----------|
| charge_route_progress_bonus | 0.013 | 8.0% | 微弱 |
| charger_access_discovery_bonus | 0.004 | 3.8% | 微弱 |
| charger_access_probe_bonus | **0.000** | **0%** | **完全无效** |
| necessary_charge_bonus | 0.002 | 1.4% | 微弱 |
| **充电正向总计** | **0.015** | — | 远低于清扫 |
| charge_detour_cost | -0.014 | 23.3% (负向) | 惩罚较重 |
| charge_interrupt_cost | -0.004 | 7.7% (负向) | 轻微 |
| unnecessary_charge_penalty | -0.001 | 0.9% (负向) | 轻微 |
| **充电负向总计** | **-0.019** | — | — |
| **充电净总计** | **-0.004** | — | **净负！** |

**关键发现**：在 bootstrap 窗口中，充电奖励的净效果实际上是**负的**（-0.004/step），意味着当前奖励结构实际上在**惩罚**与充电相关的行为，而非鼓励它。

对比主导信号：
- cleaning: 0.049/step — **是充电正向信号的 3.3 倍**
- explore: 0.028/step — **是充电正向信号的 1.9 倍**
- coverage_tangle_penalty: -0.020/step — 负向也比充电重

### 6.2 charger_access_probe_bonus 为何永远为零

```python
probe_gate = float(
    access_state_gate > 0.0           # battery 必须 safe 或 planning
    and not self.just_charged         # 刚充完电不触发
    and self.charge_count <= 0        # 必须一次都没充过
    and all_known_paths <= 0.0        # 必须没有已知充电路径
    and self.current_mode in (MODE_EXPAND, MODE_HARVEST)  # 必须在探索/清扫模式
    and self.new_explored_cells > 0   # 必须正在探索新区域
    and self.local_frontier_density >= 0.05  # 前沿密度够高
    and unknown_target_ratio > 0.12   # 未知区域比例够高
)
```

问题在于 `all_known_paths <= 0.0` 这个条件。当 `avg_planner_known_route_count_total = 7.06`（当前值）时，`all_known_paths` 几乎总是 >0。即使在新地图开始时，planner 很快就能发现至少一条路径。这意味着 **probe bonus 仅在极短暂的初始探索窗口（几十步内）有理论触发可能**，但实际上由于其他条件（frontier_density, unknown_ratio）的共同约束，它**从未实际触发过**。

### 6.3 奖励结构推动了错误的策略

当前奖励结构的信号梯度：

```
清扫相关    ████████████████████████ 0.090/step (cleaning + streak + cps)
探索相关    ████████████        0.040/step (explore + frontier)
充电正向    ██                  0.015/step
充电负向    ███                -0.019/step
充电净      ▏                  -0.004/step （净负！）
planner     ██                 -0.006/step （仅在 safe+非return 时）
tangle      ████               -0.020/step
```

Agent 的 gradient 信号明确告诉它：**多清扫，少跑路（tangle惩罚），不要绕路充电（detour_cost）**。在这个梯度场下，agent 形成"激进清扫 + 忽略电池"的策略是**奖励结构导向的必然结果**。

---

## 第七章：为何 Stagnation Level 持续为 3（最高）

### 7.1 Stagnation 检测机制

```python
# warmup 阶段 stagnation 判定
thresholds = {
    "cps": 0.32,        # avg_clean_per_step < 0.32 → 触发
    "planner": 0.80,    # planner_divergence > 0.80 → 触发 ✅ (0.85)
    "expand": 0.01,     # mode_usage_expand < 0.01 → 触发
    "stall": 0.52,      # return_stall_rate > 0.52 → 触发 ✅ (0.55)
}
# 触发 ≥2 项 → stagnation active
# stagnant_windows ≥8 → level=3
```

当前触发项：**planner (0.85>0.80)** 和 **stall (0.55>0.52)** → 满足 ≥2 条件 → stagnation active。

由于这两个指标的改善需要 agent 学会在 return 模式下跟随 planner（但该模式没有学习信号），stagnation windows 持续累加至 ≥8 → level=3。

### 7.2 Stagnation Level 3 的系统效应

```python
state["invalid_for_promotion"] = True      # 禁止阶段晋升
state["requires_reward_revision"] = True   # 标记需要奖励修订（但没有自动机制！）

# profile_plan_for_runtime 中：
if stagnation_level > 0:
    selected = conservative  # anchor=0.60, mild=0.35, broad=0.05
```

**效应链**：
1. **晋升完全阻塞** — 即使 win_rate 达到 0.60，也无法通过
2. **requires_reward_revision = True** — 系统正确诊断需要奖励修订，但这是一个被动标志，**没有自动修复机制**
3. **conservative weights 锁定** — broad=0.05 意味着 agent 95% 时间在简单场景中
4. **简单场景进一步强化当前策略** — anchor/mild 场景充电站距离近、地图小，agent 可以"偶尔撞到充电站"就能存活，不需要学会真正的返回导航

### 7.3 为什么 stagnation 无法自行解除

解除条件：planner_divergence ≤ 0.80 且 return_stall ≤ 0.52。

但要降低这两个指标需要 agent 改善 return 模式下的行为 → 而改善 return 行为需要 return 模式下的学习信号 → 而 planner_alignment 在 return 模式下不生效 → **逻辑死结**。

这就是为什么 stagnation level 3 是一个**吸收态**（absorbing state）——一旦进入，系统缺乏自我恢复机制。

---

## 第八章：为何反复调整未见效果——遗漏了什么

### 8.1 历史调整回顾

根据 diagnosis 目录中的历史报告和 curriculum signal 的时间线：

| 调整轮次 | 调整内容 | 效果 | 未解决的问题 |
|---------|---------|------|-------------|
| 初始迁移 | Linux 热补丁、分布式拓扑 | infra 正常运行 | 行为学习未开始 |
| 奖励重构 | 正向奖励体系、charging 组件 | 从全负奖励(-287)→正奖励(+100) | 充电信号过弱 |
| Curriculum 系统 | 4阶段、stagnation 检测 | 防止过快推进 | stagnation 成为死锁 |
| 教师系统 | mode/anchor/target/return teacher | 部分行为引导 | return teacher 覆盖率仅 8-14% |
| 约束优化 | dual multiplier (lambda_battery) | battery cost 信号存在 | cost 不等于行为指导 |
| 轮廓权重 | conservative/observation profiles | 稳定性保护 | 限制了学习多样性 |

### 8.2 被遗漏的三个关键盲区

#### 盲区 1：Return 模式的奖励真空

**所有历史调整都专注于"清扫阶段"和"充电决策"，但没有任何调整给 return 模式的导航行为提供直接学习信号。**

当前 return 模式下唯一的信号是 `charge_route_progress_bonus`（0.013/step），但这是一个基于 progress 的弱信号，它的前提是 agent 已经在正确方向上移动。对于一个"不知道怎么返回"的 agent，这个信号几乎没有指导作用——agent 不移动或随机移动时，progress=0，bonus=0，没有负向信号告诉它"你在做错事"。

#### 盲区 2：Return Action Teacher 的覆盖率严重不足

`return_action_teacher_active_rate = 8-14%` 意味着只有 8-14% 的训练 step 中 return action teacher 在提供教师信号。考虑到 return 模式占总时间的 ~15%，且 teacher 仅在其中的 ~60-90% 时间活跃，这意味着 **agent 在 return 模式的关键决策时刻大部分时间没有教师引导**。

对比：`mode_teacher_active_rate = 46-64%`（模式选择教师有充分覆盖），`target_teacher_active_rate = 19-35%`（目标教师有适度覆盖）。Return action teacher 是覆盖率最低的教师组件，恰恰覆盖的是最关键的能力缺口。

#### 盲区 3：Conservative Weights 的反作用力

当 stagnation 触发后，系统切换到 conservative weights（anchor=0.60, mild=0.35, broad=0.05）。这在逻辑上是"先稳定再提升"的思路，但在实际效果上：

- **Anchor 场景**（60%）：充电站位置相对固定，agent 已经在这类场景中有 66.7% 胜率
- **Mild 场景**（35%）：中等难度，agent 有 77.8% 胜率
- **Broad 场景**（5%）：高挑战但 agent 反而有 75% 胜率

**问题**：agent 在简单场景中的高胜率恰恰来自"运气充电"（charger 距离近），而非学会了返回导航。Conservative weights 让 agent 在"不需要学返回"的场景上花 95% 时间，进一步固化了当前策略。

---

## 第九章：Top 3 Root Causes（按优先级排序）

### 🔴 Root Cause #1: Return 模式的奖励信号真空（最关键）

**严重度**：Critical  
**影响范围**：planner_divergence, return_stall, battery_fail, zero_charge_fail — 所有阻塞指标的上游原因  
**证据强度**：确定性 — 源码级验证 + DEATH_TRAJ 行为一致性

**机制**：
```python
# preprocessor.py L1612-1617
# planner_alignment 仅在 safe + 非 return/contract 模式下生效
if battery_state == "safe" and self.current_mode not in (MODE_CONTRACT, MODE_RETURN):
    ...
```

当 agent 进入 return 模式时（通常 battery 已不是 safe），**三个条件全部失败**：
1. battery_state ≠ safe（planning 或 critical）
2. current_mode = MODE_RETURN
3. 结果：planner_alignment_reward = 0.0

Agent 在 return 模式下的唯一正向信号是 `charge_route_progress_bonus`（0.013/step，且前提是 agent 已在正确方向移动）。没有负向信号告诉 agent "你偏离了正确返回路径"。

**修复方向**：
- 在 return 模式下引入 planner alignment 信号（可以使用不同的 scale 或基于 slack 的调制）
- 或引入 return-specific 的导航奖励（如 charger_distance_reduction_bonus）
- 或增加 return_stall_penalty（agent 在 return 模式下 progress 为 0 时的惩罚）

---

### 🔴 Root Cause #2: Return Action Teacher 覆盖率不足

**严重度**：High  
**影响范围**：return 模式下的行为探索和学习速度  
**证据强度**：高 — learner log 中 return_action_teacher_active_rate 一致在 8-14%

**机制**：
- mode_teacher_active_rate: 46-64%（充分）
- route_anchor_teacher_active_rate: 19-35%（适中）
- target_teacher_active_rate: 19-35%（适中）
- return_action_teacher_active_rate: **8-14%**（严重不足）

Return action teacher 本应在 agent 处于 return 模式时提供"正确方向"的教师信号。但它的活跃率意味着在 return 的大部分时间里，agent 依靠自己的（未训练的）策略来做决策 → 行为随机化 → stall。

**修复方向**：
- 提高 return action teacher 的活跃率（降低其退出条件的阈值，或在 battery_state != safe 时强制启用）
- 或增加 return 模式下的 expert demonstration 数据
- 确保 teacher 在 battery critical 时有更高的覆盖率

---

### 🟡 Root Cause #3: Stagnation → Conservative Weights 恶性循环

**严重度**：Medium-High  
**影响范围**：学习速度和策略多样化  
**证据强度**：高 — curriculum_policy.py 源码 + curriculum_state.json 状态

**机制**：
```
stagnation_level ≥ 1 → conservative weights → anchor=0.60, broad=0.05
    → 95% 时间在简单场景 → agent 用"运气充电"策略也能存活
    → 没有 gradient 压力去改善返回行为
    → planner_divergence 和 return_stall 不下降
    → stagnation 维持 → 循环
```

**修复方向**：
- 在 stagnation level 3 + requires_reward_revision 时，**自动调整奖励结构**（而非仅设标志位）
- 或在 stagnation 持续时逐步释放 broad 场景比例（如 broad=0.05→0.15），迫使 agent 面对更有挑战的场景
- 或添加 stagnation_escape 机制：当 stagnation_level=3 持续 N windows 后，临时提高充电/返回相关奖励的 scale

---

## 第十章：最终结论

### 10.1 一句话诊断

**Agent 的返回导航能力处于"奖励结构真空"中——没有信号引导它学会在 return 模式下跟随 planner 到达充电站，导致它陷入"激进清扫 + 随机返回"的局部最优，而 curriculum 系统的自我保护机制（stagnation → conservative weights）进一步固化了这个局部最优。**

### 10.2 当前系统状态的定量画像

```
                    当前值      warmup 出口要求    缺口
win_rate            0.725       ≥0.60            ✅ 通过
battery_fail_rate   0.275       ≤0.20            ❌ -0.075
collision_fail_rate 0.100       ≤0.10            ⚠️ 边界
return_stall_rate   0.550       ≤0.50            ❌ -0.050
planner_divergence  0.850       ≤0.80            ❌ -0.050
zero_charge_fail    0.545       ≤0.50            ❌ -0.045
entropy_loss        1.430       ≤0.92 (×max)     ✅ 通过
```

五项指标中三项不达标，且三项之间有强因果关系（planner_divergence → return_stall → battery_fail）。修复 Root Cause #1 应能级联改善所有三项。

### 10.3 预期修复优先级

| 优先级 | 修复目标 | 预期影响 | 风险 |
|--------|---------|---------|------|
| **P0** | Return 模式引入 planner alignment | planner_div 下降 → return_stall 下降 → battery_fail 下降 | 需要小心平衡 scale，避免过度惩罚 |
| **P1** | 提高 return_action_teacher 覆盖率 | 加速 return 行为学习 | 低风险 |
| **P2** | Stagnation 解锁机制 | 打破恶性循环 | 中等风险，需要防止过早解锁 |

### 10.4 不建议的操作

- ❌ 不要调整 PPO 超参（lr, clip, gamma 等）— 训练动力学正常
- ❌ 不要增加 cleaning/explore 奖励 — 已经是最强信号，增加会进一步偏向清扫
- ❌ 不要减少 coverage_tangle_penalty — 它在防止 agent 重复覆盖，是有价值的信号
- ❌ 不要直接放松 stagnation threshold — 会掩盖问题而非解决
- ❌ 不要绕过 warmup 出口门 — 门限检测到了真实的能力缺陷

---

*报告生成时间: 2026-04-20*  
*数据截止: learner step ~67K, global_step_since_resume ~52K, 120+ episodes*  
*分析覆盖: curriculum_state.json, learner logs, aisrv helper logs (pid440/441/445), 10+ curriculum signal files, preprocessor.py, constraint_utils.py, curriculum_policy.py, curriculum_state.py, conf.py*
