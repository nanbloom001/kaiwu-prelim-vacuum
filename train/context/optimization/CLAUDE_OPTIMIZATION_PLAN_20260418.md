# 进一步优化方案：Reward-Curriculum-Behavior 联合修正

> 评审人：Claude (RL 专家视角)
> 前序文档：`CLAUDE_REVIEW_20260417.md`（reward attribution 分析）、`CLAUDE_RETRAIN_EVALUATION_20260418.md`（重训评估）
> 依据数据：16:xx 时段 learner 日志 (pid331)、4 组 aisrv helper 日志 (pid438/441/443/446)、`curriculum_state.json`
> 日期：2026-04-18

---

## 〇、四个核心问题回答

### Q1: 当前 reward 各项量级/排序是否已接近训练目标？

**否。量级/排序存在根本性错位，且 16:xx 日志证实这一结构正在制造训练崩溃。**

来自 16:xx 新训练日志的直接证据：

| 指标 | 早期 (step<3000) | 晚期 (step>10000) | 趋势 |
|------|-----------------|------------------|------|
| clean_score (每集) | 440~503 | 37~165 | **暴跌 70%+** |
| episode_reward (WIN 场) | +65~+105 | -14~-124 | **WIN 也深度负奖** |
| charge_count (每集) | 7~12 | 45~60 | **膨胀 5 倍** |
| remaining_charge | 66~130 | 240~286 | **过量电量翻倍** |
| mode_usage_contract | ~0.80 | ~0.82 | **80%+ 时间在收缩** |
| mode_usage_expand | ~0.001 | ~0.0001 | **探索趋近于零** |

所有 GAMEOVER 日志的 `[REWARD_TOP]` 汇总一致显示：

- **planner_alignment 恒定 -0.04~-0.055**（过强且无法改善，因为 divergence rate 88% 是结构性的）
- **return_stall 持续 -0.02~-0.063**（惩罚在累积但无法引导行为修正）
- **cleaning 从 +0.12 衰减至 +0.003~+0.019**（清扫奖励在训练过程中自我塌缩）

**核心错位**：agent 发现"反复充电+收缩模式不做任何事"是当前 reward 下的最优策略——因为这样能把 cleaning 的上下文衰减乘子 (CLEANING_RETURN_SCALE=0.25, revisit=0.40~0.70) 全部规避，且 charge reward (+3.0*eff) 几乎不受惩罚。但一旦被课程强制推到 anchor profile（1 robot, 300 battery, 1000 steps），cleaning 收益无法覆盖 planner_alignment + return_stall + idle 的累积惩罚，导致 **WIN 也拿到 -124 的步奖。**

---

### Q2: 主因归因——Reward vs 课程 vs Teacher vs Planner-Policy

**归因占比：Reward 结构 60% → Teacher 信号衰减 20% → 课程死循环 15% → Planner 5%**

#### Reward 结构 (60%)
1. **cleaning → 0 的自我塌缩循环**：agent 学到反复充电 → 每步清扫下降 → context-aware scale 给更低乘子 → cleaning reward 更低 → 更依赖充电以获得稳定的 charge reward → 进一步不清扫。这是一个正反馈崩溃环路。
2. **planner_alignment 惩罚无法被减少**：divergence rate 88%→89% 不降反升。根本原因是 planner 输出的 route_anchor/target 和 agent 实际行为在当前 reward 下无法对齐——agent 选择了一条 planner 不理解的"充电+收缩"路径。
3. **charge reward (+3.0*eff) 是唯一稳定正信号**：charge_count 从 7→60 就是 agent 对此信号的精确优化。

#### Teacher 信号衰减 (20%)
从 learner 日志的 teacher active rate 演化：

| Teacher Head | Step 0 | Step 18000 | 趋势 |
|---|---|---|---|
| mode_teacher_active_rate | 0.60 | 0.08~0.15 | **急剧下降 → 几乎不再教学** |
| route_anchor_teacher_active_rate | 0.24 | 0.88~0.96 | **反常上升 → 更多介入** |
| target_teacher_active_rate | 0.24 | 0.88~0.96 | **反常上升 → 更多介入** |
| return_action_teacher_active_rate | 0.003 | 0.24~0.28 | **显著上升** |

**关键异常**：route_anchor 和 target 的 teacher active rate 在上升意味着：这两个 teacher 认为 agent 的选择越来越差，需要更多干预。但 mode_teacher 几乎退出了。这造成了一个矛盾信号：mode teacher 不再纠正 mode 选择（agent 用 80% contract），但 target teacher 又在持续试图引导目标选择——agent 的行为模式和目标规划被两个 teacher 拉向不同方向。

#### 课程死循环 (15%)
- warmup 要求 `return_stall ≤ 0.40`，当前 0.60
- 但 warmup profile 权重 (anchor=0.6, mild=0.35, broad=0.05) 反而强化了保守行为
- 短暂进入 blend 后又立刻被拉回 warmup（pid438 ep:6 一次性闪现 blend）
- 结果：课程无法推进 → agent 只见保守场景 → 更加保守

#### Planner (5%)
Planner 本身是正确的；问题在于 agent 的行为模式（80% contract, 0% expand）使得 planner 的 route_anchor 和 target 建议本质上不可执行。这是结果不是原因。

---

### Q3: 增量微调还是系统性重构？

**需要系统性重构 reward 结构，但保持架构和课程框架不变。**

理由：
1. 16:xx 日志显示 **训练在主动崩溃**（clean_score 单集跌到 37，episode_reward 跌到 -124），这不是停滞——是向错误方向收敛
2. 增量调参无法修复"正反馈崩溃环路"：只调低 cleaning 系数而不重新设计 charge reward 和 context scale 的交互关系，环路仍然存在
3. value head 可以继续用（value_clean_loss 1.0~3.0, value_survive_loss 1.3~4.9 都在下降，说明 value 网络有学习能力）
4. GRU 权重可以通过 preload 保留——重构的是 reward 信号而非网络结构

**判定：系统性 reward 重构 + preload 冷启动，保留网络权重和课程框架。**

---

### Q4: 具体优先级排序的优化方案

见下方完整方案。

---

## 一、优化方案：分阶段执行

### Phase 1: 紧急修正 — 阻断崩溃环路 [优先级 CRITICAL]

> 目标：消除 "充电→不清扫→cleaning 崩塌→更多充电" 的正反馈环路

#### 1.1 Cleaning Reward 压缩 + Charge Reward 重设计

```python
# conf.py 修改

# --- Cleaning ---
# 旧: CLEANING_REWARD_BASE = 1.5
CLEANING_REWARD_BASE = 0.6        # ↓60%：消除 cleaning 对其他信号的绝对压制
# 旧: STREAK_BONUS_BASE = 0.15
STREAK_BONUS_BASE = 0.06          # ↓60%：与 cleaning 同步缩放

# --- Charge ---
# 旧: CHARGE_REWARD_BASE = 3.0
CHARGE_REWARD_BASE = 0.8          # ↓73%：从"稳定正信号"变为"必要补给"
# 新增：充电次数递减
CHARGE_DIMINISHING_RATE = 0.92    # 每次充电后下一次 reward *= 0.92
CHARGE_COUNT_SOFT_CAP = 15        # 超过 15 次充电后 reward *= 0.5
```

**预期效果**：charge_count 从 60 回到 10~20 范围；cleaning 不再被 charge 主导。

#### 1.2 Context-Aware Cleaning Scale 简化

```python
# conf.py 修改

# 旧: CLEANING_RETURN_SCALE = 0.25
CLEANING_RETURN_SCALE = 0.50      # ↑100%：返航时清扫不应打 75% 折
# 旧: CLEANING_REVISIT_SOFT_SCALE = 0.70
CLEANING_REVISIT_SOFT_SCALE = 0.80  # 轻微放宽
# 旧: CLEANING_LOOP_SCALE = 0.25
CLEANING_LOOP_SCALE = 0.40        # ↑60%：loop 中清扫仍有价值
```

**预期效果**：cleaning reward 不再因 context 叠加乘子而自我消解到 +0.003/step。

#### 1.3 Reward Clip 收窄

```python
# conf.py 修改

# 旧: REWARD_CLIP_MIN, REWARD_CLIP_MAX = -5.0, 5.0
REWARD_CLIP_MIN = -3.0
REWARD_CLIP_MAX = 3.0
```

**预期效果**：限制单步极端奖励，防止 charge 事件（一次 +3.0*eff）产生 distortion。

---

### Phase 2: Planner 对齐修正 [优先级 HIGH]

> 目标：将 planner 信号从"恒定惩罚噪声"变为"可优化的引导信号"

#### 2.1 Planner Alignment 奖惩再平衡

```python
# conf.py 修改

# 旧: PLANNER_ALIGNMENT_REWARD = 0.03
PLANNER_ALIGNMENT_REWARD = 0.10   # ↑233%：对齐时正向激励显著增加
# 旧: PLANNER_DIVERGENCE_PENALTY = -0.06
PLANNER_DIVERGENCE_PENALTY = -0.15 # ↑150%：偏离时惩罚加强

# 新增：divergence 惩罚按严重程度分级
PLANNER_DIVERGENCE_MILD_PENALTY = -0.05   # route_anchor 不一致但方向合理
PLANNER_DIVERGENCE_SEVERE_PENALTY = -0.20 # route_anchor + target 同时不一致
```

**预期效果**：planner_alignment 从恒定的 -0.04~-0.05 变成有梯度的信号——对齐时拿 +0.10，偏离时根据严重程度惩罚 -0.05~-0.20。

#### 2.2 解耦 Mode Teacher 和 Action 信号

当前 mode_teacher_active_rate 急降到 0.08 但 agent 依然 80% contract。这说明 mode teacher 过早"满意"并退出了。

**修改建议**（preprocessor.py 或 expert.py）：
- 当 `mode_usage_contract > 0.60` 时，mode teacher 不应退出——保持 active_rate ≥ 0.30
- 增加 expand mode 的 teacher 权重，使 teacher 主动示范 explore 行为

---

### Phase 3: 课程门槛与电池管理 [优先级 HIGH]

> 目标：打破 warmup 死循环，让优质 checkpoint 能进入 blend

#### 3.1 Warmup → Blend 门槛调整

```python
# curriculum_policy.py 修改

# warmup → blend 退出条件
# 旧: return_stall ≤ 0.40
return_stall_threshold = 0.50    # 放宽 25%：当前 0.60 差距从 0.20 降到 0.10
# 旧: win_rate ≥ 0.60 
win_rate_threshold = 0.70        # 收紧 16%：当前 0.93 远超，提高门槛防虚假进入
# 新增：CPS 门槛
cps_threshold = 0.25             # 新增：每步清扫率需≥0.25，防止"高 win 低效率"通过

# blend → robust 门槛也相应调整
# 旧: return_stall ≤ 0.35
return_stall_threshold_robust = 0.40
```

#### 3.2 电池管理参数收紧

```python
# conf.py 修改

# 旧: CONTRACT_BATTERY_RATIO = 0.35
CONTRACT_BATTERY_RATIO = 0.28    # 更早进入返航准备
# 旧: PREPARE_RETURN_SLACK_THRESHOLD = 12.0
PREPARE_RETURN_SLACK_THRESHOLD = 8.0  # 更紧的返航余量
# 旧: RETURN_SLACK_THRESHOLD = 4.0
RETURN_SLACK_THRESHOLD = 3.0     # 更紧的返航触发
```

**预期效果**：减少 battery fail（当前 0.075），让 agent 更早做出返航决策而非拖到 slack=-22~-33。

---

### Phase 4: Return Stall 与 Idle 惩罚校准 [优先级 MEDIUM]

> 目标：让 return_stall 从 0.60 降至 ≤0.35

#### 4.1 Return Stall 惩罚升级

```python
# conf.py 修改

# 旧: RETURN_STALL_BASE_PENALTY = -0.12
RETURN_STALL_BASE_PENALTY = -0.20    # ↑67%
# 旧: RETURN_STALL_EMA_PENALTY = -0.06
RETURN_STALL_EMA_PENALTY = -0.10     # ↑67%

# 新增：连续 stall 的指数惩罚
RETURN_STALL_CONSECUTIVE_ESCALATION = 1.15  # 连续 stall 每步 *= 1.15
```

#### 4.2 Idle 惩罚细化

```python
# conf.py 修改

# 旧: IDLE_PENALTY = -0.1
IDLE_PENALTY = -0.15                 # ↑50%
# 新增：区分"在充电桩附近 idle"和"在地图中心 idle"
IDLE_AT_CHARGER_PENALTY = -0.25      # 在充电桩附近空转重罚
```

---

### Phase 5: Entropy 管理与探索激励 [优先级 MEDIUM]

> 当前 entropy 从 2.00 → 1.90，下降过慢，说明策略几乎没有形成明确偏好

#### 5.1 Entropy 目标调整

```python
# conf.py 修改

# 旧: ENTROPY_FLOOR = 0.15
ENTROPY_FLOOR = 0.20              # 保持更高的最低探索
# 旧: ENTROPY_COEFF = 0.01 (推测)
ENTROPY_COEFF = 0.005             # 降低 entropy bonus，让策略更快分化
```

#### 5.2 Expand Mode 正向激励

```python
# preprocessor.py reward_process 修改

# 新增：expand mode 直接奖励
EXPAND_MODE_BONUS = 0.05          # 选择 expand mode 时给予基础奖励
EXPAND_CLEANING_MULTIPLIER = 1.5  # expand 模式下 cleaning 额外加成 50%
```

---

## 二、执行顺序与检验标准

### 执行流水线

```
Phase 1 (CRITICAL) ──→ Phase 2 (HIGH) ──→ Phase 3 (HIGH) ──→ Phase 4+5 (MEDIUM)
      ↓                    ↓                    ↓                    ↓
  修改 conf.py         修改 conf.py +         修改 curriculum_     修改 conf.py +
  (5 个参数)           preprocessor.py        policy.py            preprocessor.py
                       (3 个参数 +            (4 个参数)           (4 个参数)
                        分级逻辑)
```

**Phase 1 + Phase 2 可同时实施**（无依赖关系），Phase 3 在 Phase 1 验证后实施。

### 每阶段检验里程碑

| 阶段完成 | 检验指标 | 通过标准 | 预期 step |
|---------|---------|---------|----------|
| Phase 1 | charge_count, cleaning/step, episode_reward | charge <20, cleaning>0.03/step, WIN 场 reward>0 | step 3000 |
| Phase 2 | planner_divergence_rate, reward_planner_alignment | divergence <0.60, alignment 奖惩比 >1:3 | step 5000 |
| Phase 3 | curriculum_stage, return_stall | 进入 blend, stall <0.45 | step 8000 |
| Phase 4+5 | mode_usage_expand, entropy, CPS | expand>0.05, entropy 1.6~1.8, CPS>0.35 | step 12000 |

### Lite Benchmark 评估触发

每个 Phase 完成后跑一轮 lite benchmark（4 episodes），对比：

| 指标 | 当前基线 | 目标 |
|------|---------|------|
| completed_rate | 1.0 | 保持 ≥0.90 |
| return_stall | 0.66 | ≤0.40 |
| broad_win_rate | 0.38→1.0(最新) | 保持 ≥0.80 |
| avg_clean_per_step | 0.19 | ≥0.35 |
| avg_charge_count | 57 | ≤20 |
| recommended_stage | warmup | blend 或 robust |

---

## 三、风险与回退

### 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| Cleaning 降幅过大导致 agent 完全忽略清扫 | 低 | 高 | cleaning 仍是最大正信号(0.6 vs charge 0.8, 但 cleaning 每步触发而 charge 低频), 观察 CPS 指标 |
| Charge 惩罚过强导致 battery fail 飙升 | 中 | 中 | 保持 charge_margin_pressure 分级惩罚, 若 battery_fail>0.20 回退 CHARGE_REWARD_BASE 至 1.2 |
| Return stall 惩罚过强导致 agent 过早放弃清扫 | 中 | 中 | 监控 finished_steps/max_steps 比率, 若<0.50 减半 RETURN_STALL_CONSECUTIVE_ESCALATION |
| 课程门槛放宽后 agent 过早进入困难场景 | 低 | 低 | blend 保留 anchor+mild 比重, 仅增加 broad_eval 占比 |

### 回退计划

- **Phase 1 回退**：若 step 3000 时 battery_fail > 0.25 或 win_rate < 0.50，回退 CHARGE_REWARD_BASE 到 1.5，仅保留 cleaning 修改
- **Phase 2 回退**：若 planner_divergence 不降反升到 >0.95，取消分级惩罚，回退到原始 -0.06 统一惩罚
- **全局回退**：保存当前 preload checkpoint 的完整备份；若所有修改后 step 5000 时 clean_score 均值 <100，回退全部修改并重新诊断

---

## 四、补充：新日志暴露的额外问题

### 4.1 DEATH_TRAJ 模式分析

16:xx 日志中的 7 条 DEATH_TRAJ 全部显示相同模式：

```
mode=4 (contract) 在 battery < 20 时持续执行
slack 深度负值 (-9 到 -33)
action 在 0~7 之间看似随机（无明确返航意图）
```

这证实：**agent 在 contract 模式下没有学会返航**。contract 模式应该触发充电寻路，但 agent 在 mode=4 下执行的 action 是无目的的（act=0/5/6/7 随机分布）。

**修改建议**：在 contract 模式下，当 slack < 0 时，增加一个硬约束奖惩：
```python
if mode == 4 and slack < 0:
    # 强制惩罚非朝向充电桩的 action
    if not moving_toward_nearest_charger:
        reward -= 0.30  # 远大于其他所有信号
```

### 4.2 Blend 阶段闪现问题

pid438 ep:6 和 pid443 ep:4-5 短暂进入了 blend 阶段：
```
Episode 6 start stage=blend global_progress=1.00 profile=broad
Episode 4 start stage=blend global_progress=0.69 profile=broad_eval
```

但随后立刻回退到 warmup（ep:7 又是 warmup）。说明 curriculum 的状态跳转不稳定，可能需要增加退出条件的 hysteresis（迟滞门槛）。

**修改建议**：增加 `consecutive_pass_windows >= 2` 才允许进入下一阶段，防止单窗口偶然达标导致的闪入闪出。

### 4.3 训练吞吐正常

data_fetch 5~9ms, real_train 55~82ms, 总计 60~95ms/step。sample_production_and_consumption_ratio 稳定在 120~130。**数据管线无瓶颈，所有问题都在 reward/behavior 层面。**

---

## 五、总结

| 维度 | 现状 | 修正方向 | 优先级 |
|------|------|---------|--------|
| Cleaning 量级 | 1.5 → 自我塌缩到 +0.003/step | 基数 0.6 + context scale 放宽 | CRITICAL |
| Charge 量级 | 3.0 → agent 充电 60 次/集 | 基数 0.8 + 递减 + 软上限 | CRITICAL |
| Reward clip | [-5,+5] | [-3,+3] | CRITICAL |
| Planner 对齐 | 恒定 -0.05/step 噪声 | 分级惩罚 + 对齐奖励增强 | HIGH |
| Teacher 退出 | mode teacher 过早退出 | 按 mode_usage 条件保持 active | HIGH |
| 课程门槛 | return_stall≤0.40 卡死 | 放宽到 0.50 + 加 CPS 门槛 | HIGH |
| 电池管理 | CONTRACT_RATIO 0.35 太松 | 收紧到 0.28 | HIGH |
| Return stall | 0.60, 缓降 | 基础惩罚增强 + 连续指数 | MEDIUM |
| 探索激励 | expand mode ≈ 0% | expand 直接奖励 + cleaning 加成 | MEDIUM |

**执行路径**：Phase 1+2 同步修改 → preload 冷启动 → 3000 step 检验 → Phase 3 → 8000 step 检验 → Phase 4+5 → 12000 step 最终检验。
