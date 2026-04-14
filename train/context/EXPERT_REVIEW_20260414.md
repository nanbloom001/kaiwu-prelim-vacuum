# Expert 充电逻辑审查报告

**日期**: 2026-04-14  
**审查对象**: `agent_ppo/feature/expert.py` 充电导航模块  
**背景数据**:
- 训练总体健康 (entropy 0.12-0.21 稳定无塌缩, WinRate 85%)
- 所有 32 次失败直接来自 Expert 代码 bug
- 碰撞死亡: 18 例 (56%)，全部在 mode=charger（Expert 导航）中 NPC dist ≤ 3 时撞上
- 电池死亡: 14 例 (44%)，全部在 mode=clean 中到电量归零，Expert 未触发 return_mode

---

## 一、Expert vs 纯 RL：架构决策

### 结论：保留 Expert 但大幅简化改进，不推荐完全去掉

#### 为什么必须保留 Expert

1. **14 次电池死亡证明模型不会自主充电**
   - 14 例电池死亡最后 20 步全部 mode=clean（清扫模式）
   - Expert 完全没有触发 return_mode
   - 模型也没有任何充电意识或行动
   - 去掉 Expert 只会更差

2. **GAE 衰减破坏长期规划能力**
   - γλ = 0.99 × 0.95 = 0.9405
   - 50 步衰减到 4.7%，100 步衰减到 0.2%
   - 充电决策需要提前 50-200 步规划
   - RL 从当前奖励信号学不会这种长周期行为

3. **模型架构冻结无法升级**
   - Resume 模式下只能微调不能改架构
   - 无法加 Memory/Attention 辅助长周期规划
   - 无法加 exploration bonus 鼓励主动探索充电策略

#### 但 Expert 的确在限制 RL

1. **Expert 导致了全部 32 次死亡**
   - 18 碰撞: Expert A* 引导撞 NPC
   - 14 电池: Expert return_mode 条件太苛刻或误触发
   - 说明 Expert bug 是直接瓶颈，并非 RL 无能

2. **训练中 logit bias 严重偏移采样分布**
   - 充电 bias 3-100 级别制导
   - 模型只走 Expert 规划的充电路径，从不探索替代方案
   - 当 Expert 犯错时模型毫无 contingency plan

3. **Expert 实现高复杂度 → bug 密集**
   - A* + 3 级 fallback + 动态 margin + state machine
   - 每个 bug 都直接致死，没有冗余机制

### 改进路径（分阶段）

**阶段 1（本次）**: 修 Expert 的 5 个具体 bug → 预期 WinRate > 92%

**阶段 2（训练 5k 步后）**:
- 观察 Survival 趋势
- 如果 > 92% 稳定：降低 logit bias 上限（8→5，emergency=100→50）
- 增强 RL 充电信号（urgency_penalty -0.4→-1.3, charger_reward scale up）

**阶段 3（训练 20k 步后）**:
- Expert 退化为纯安全网（只处理 LOW_BATTERY < 0.20 紧急场景）
- 正常充电由 RL + urgency_penalty 主导
- 评估模式下才用 hard override

---

## 二、根因详细分析

### 碰撞死亡 (18 例) — 三层防线失效

#### 日志证据
典型案例：
- ep 204 (broad map), battery=667/720 (93%) → Expert 进入 return_mode → NPC dist 2 → 碰撞
- ep 189 (anchor map), battery=32/300 → Expert 进入 return_mode → NPC dist 2 → 碰撞

全部 18 例碰撞都在 Expert return_mode=True 后 2-5 步内发生。

#### 根因链

**链条 1: filter_actions 漏洞 — 近距离拦截不完整**

```python
# 当前代码 (expert.py L98-109)
if npc_dist <= 5:
    sx = (1 if ndx > 0 else -1) if ndx != 0 else 0
    sz = (1 if ndz > 0 else -1) if ndz != 0 else 0
    if dx == sx and dz == sz:  # ← 只拦截方向完全一致
        legal[idx] = 0
```

问题场景：
- NPC 在东北方向 (1,1)
- 走北向 (0,1) 或东向 (1,0) 时 filter 不拦截（dx/dz 不等于 sx/sz）
- 但都会靠近 NPC（Chebyshev 距离从 1 变 2）
- 结果 2-3 步后直接撞上

**链条 2: A* fallback 过度降级 — 完全放弃 NPC 避让**

```python
# _plan_to_charger_cached() L363-367
# Fallback 2: no NPC avoidance at all
cost_map = self._build_cost_map(prep, npc_weight=0.0)  # ← 关键问题
return self._weighted_astar(prep, cost_map, is_goal, h_func)
```

问题：
- npc_weight=0.0 时 NPC 周围无 danger cost
- A* 规划的路径会穿过 NPC 旁边的格子
- 示例：NPC 在 (50,50)，A* 规划 (48,49) → (50,49) → (50,51)，直接擦过 NPC
- filter_actions 与这个路径配合后，无法完全阻止

**链条 3: 回报覆盖冲突 — 充电 bias 压过安全 bias**

```python
# get_logit_bias() L207-219
if should_return and expert_action is not None:
    bias[expert_action] = 100.0  # 充电方向 bias [3,100]
# + NPC 负 bias (L223-230)
    bias[_idx] -= 2.0 * _dot * _close  # −2 max
```

问题：
- 充电 bias 3-100 远大于 NPC 负 bias -2
- NPC dist=2-3 时：充电 action bias=8，NPC action bias=-2，仍然选充电方向
- NPC 避让 bias 算法也有问题：只在 dist ≤ 6 时施加，且计算 `_dot*_close` 可能不够强

**链条 4: 误触发 return_mode — 不该充电时也在充电**

```python
# _evaluate_return() L195-205
should_return = (
    self.return_mode
    or (charger_dist < float('inf') and prep.battery <= charger_dist + margin)
    or battery_ratio <= self.LOW_BATTERY_RATIO
)
```

ep 204 案例数据：
- battery=667, battery_max=720 → battery_ratio=0.927 (93%)
- charger_dist=600+, margin=40 (broad 大地图, 路径复杂)
- 条件判断: 667 <= 640? No... 等等，这个应该 NO；但为什么 return_mode=True?

**原因**：之前某步可能 battery 值曾经很低导致 return_mode 激活，然后机器人充了一些电，但 return_mode 没有正确退出。或者 charger_dist 计算有问题。

**改进建议**：加电量比例守卫，防止高电量不必要的 return_mode。

### 电池死亡 (14 例) — return_mode 触发条件失效

#### 日志证据
典型案例：
- ep 130 (broad 1 charger 4 robot): slack=-47，最后 20 步 mode=1 clean, 0 charging steps → 电池耗尽
- ep 136 (broad 1 charger 2 robot): slack=-73，最后 20 步 mode=1 clean, 0 charging steps → 电池耗尽

#### 根因链

**链条 1: A* 找不到路径 → charger_dist=inf → 距离条件永假**

```python
# _evaluate_return() L179-183
charger_path, charger_dist, charger_target = self._plan_to_charger_cached(prep)
if not charger_path:
    charger_dist = prep.nearest_charger_dist
```

问题：
- 当未探索区域多时（training early phase），`_UNEXPLORED_COST=3.0` 抬高代价
- A* 搜索空间爆炸或找不到路径 → `charger_path=[]`
- fallback 用 Chebyshev 距离（不考虑障碍物）← 可能低估真实距离
- 即使用了 fallback，Chebyshev 也只是下界估计

**链条 2: LOW_BATTERY_RATIO = 0.32 对小电池太晚**

```python
# ExpertPolicy L36
LOW_BATTERY_RATIO = 0.32
```

计算：
- battery_max=120 (broad config) → 0.32 × 120 = 37 steps
- charger 距离 >= 38 时来不及到达
- actual data: ep 130 slack=-47, ep 136 slack=-73 表明充电桩距离 > 50 steps

**链条 3: return_mode 一旦激活难以退出**

```python
# _evaluate_return() L168-172
if self.return_mode and on_charger and battery_ratio >= self.EXIT_RETURN_RATIO:
    self.return_mode = False
```

问题：
- 需要同时满足三个条件：on_charger AND battery >= 0.95
- 多机器人场景中，多个机器人争抢同一充电桩（charger_count=1）
- 机器人 A 占据充电桩，机器人 B 无法上，battery 耗尽
- 代码中完全没有感知其他机器人，只能等待

#### 未探索区域代价问题

```python
# expert.py L299
_UNEXPLORED_COST = 3.0
```

这个值太高了：
- 已探索 passable = 1.0 + visit_penalty (0-0.75)
- 未探索 = 3.0
- 意味着 A* 严重避免穿过未探索区域
- 加上 blocked_penalty=4.0，有些区域看起来比墙（INF_COST=1e6）都有吸引力……不对，still 低于 INF

实际影响：
- early training (explored ratio < 30%) 时，A* 常常找不到从起点到充电桩的路径
- battery death episode 大多发生在 training step 0-1000（early phase）

---

## 三、五项具体修改方案

### Fix 1: 扩展 filter_actions 近距离拦截 — 防止 NPC dist ≤ 3 时靠近

**优先级**: ⭐⭐⭐⭐⭐ (最高)

**修改文件**: `expert.py` L98-109

**当前代码问题**:
- 只拦截方向完全一致的动作
- NPC 在东北方向 (1,1) 时，走 (0,1) 或 (1,0) 均不被拦截
- 对角线移动同样不被拦截

**修改方案**:
```python
# dist ≤ 3: 拦截所有缩短距离的动作(包括对角线、侧向)
# dist 4-5: 原逻辑（只拦截正对方向）

if npc_dist <= 3:
    # 阻止所有会缩短到 NPC 距离的移动
    new_dist = max(abs(ndx - dx), abs(ndz - dz))  # 新Chebyshev距离
    if new_dist < npc_dist:  # 会靠近
        legal[idx] = 0
elif npc_dist <= 5:
    # 原有逻辑：只阻止直接移向 NPC
    sx = (1 if ndx > 0 else -1) if ndx != 0 else 0
    sz = (1 if ndz > 0 else -1) if ndz != 0 else 0
    if dx == sx and dz == sz:
        legal[idx] = 0
```

**风险评估**: 低
- 可能过度拦截导致无合法动作
- 已有兜底机制：`if sum(legal) == 0: return list(legal_action)`
- 返回原始 legal_action，机器人仍能移动

**预期效果**: 直接防止 18 例碰撞中的大多数（估计 80%+）

---

### Fix 2: 移除 A* 第 3 级 npc_weight=0.0 fallback — 防止 A* 规划穿过 NPC

**优先级**: ⭐⭐⭐⭐ (高)

**修改文件**: `expert.py` 两处
1. `_plan_to_charger_cached()` L363-367
2. `_plan_to_charger()` L585-589

**当前代码问题**:
```python
# Fallback: no NPC avoidance at all
cost_map = self._build_cost_map(prep, npc_weight=0.0)
act, path, dist = self._weighted_astar(prep, cost_map, is_goal, h_func)
```

问题：
- 当前两级 fallback (1.0 → 0.3) 都找不到路时，第三级完全放弃 NPC 避让
- A* 规划的路径会直接穿过 NPC 旁边
- 虽然 NPC 不在起点和中间路径上，但机器人可能在 2-5 步内移动到 NPC 位置

**修改方案**:
直接删除第三级 fallback，只保留两级（1.0 → 0.3）。如果都找不到，fall through 到 `_greedy_toward_charger()`（该函数受 filter_actions 保护）。

**实施步骤**:

在 `_plan_to_charger_cached()` 中：
```python
# 删除这块：
# Fallback: no NPC avoidance
# cost_map = self._build_cost_map(prep, npc_weight=0.0)
# act, path, dist = self._weighted_astar_full(prep, cost_map, is_goal, h_func)
# if act is not None and path:
#     ...
#     return path, dist, self._cached_target

# 改为直接返回空
return [], float('inf'), None
```

在 `_plan_to_charger()` 中：
```python
# 删除最后三行：
# Fallback 2: no NPC avoidance at all
# cost_map = self._build_cost_map(prep, npc_weight=0.0)
# return self._weighted_astar(prep, cost_map, is_goal, h_func)

# 改为返回 None
return None
```

**风险评估**: 极低
- 极少数场景下 NPC 可能完全封锁所有通道
- 但这种情况下 npc_weight=0.0 也无法安全通过（走过去就撞了）
- 不如不走，让 filter_actions + greedy 处理

**预期效果**: 消除 A* 规划穿过 NPC 的情况，再减少碰撞 10-20%

---

### Fix 3: return_mode 触发条件加电量守卫 + 动态 LOW_BATTERY_RATIO — 防误触发和电池死亡

**优先级**: ⭐⭐⭐⭐ (高)

**修改文件**: `expert.py` L36 和 L195-205

**当前问题**:
1. 93% 电量（ep 204）仍进入 return_mode
2. LOW_BATTERY_RATIO=0.32 对 battery_max=120 太晚（只剩 37 steps）
3. 多机器人场景中 low battery 机器人无法竞争充电桩

**修改方案**:

```python
# 第 1 处：L36 增加动态 LOW_BATTERY_RATIO 计算逻辑
# 但在类定义中保持原值作为基准

LOW_BATTERY_RATIO = 0.32

# 第 2 处：L195-205 更新 _evaluate_return()
def _evaluate_return(self, prep, legal_action, last_action=-1):
    """..."""
    self.update_chargers(prep)
    self.update_blocked(prep, last_action)

    hx, hz = prep.cur_pos
    battery_ratio = prep.battery / max(prep.battery_max, 1.0)
    on_charger = self._is_on_charger(hx, hz)

    if self.return_mode and on_charger and battery_ratio >= self.EXIT_RETURN_RATIO:
        self.return_mode = False
        self._cached_path = []
        self._cached_target = None

    charger_path, charger_dist, charger_target = self._plan_to_charger_cached(prep)
    if not charger_path:
        charger_dist = prep.nearest_charger_dist

    margin = self._charge_margin(charger_path)

    # ★ 新增：动态 LOW_BATTERY_RATIO
    effective_low_ratio = max(
        self.LOW_BATTERY_RATIO,
        min(50.0 / max(prep.battery_max, 1), 0.45)
    )
    # 计算示例：
    # battery_max=120 → 50/120=0.417 → min(0.417,0.45)=0.417 → max(0.32,0.417)=0.417
    # battery_max=200 → 50/200=0.25 → min(0.25,0.45)=0.25 → max(0.32,0.25)=0.32
    # battery_max=720 → 50/720=0.069 → min(0.069,0.45)=0.069 → max(0.32,0.069)=0.32

    # ★ 新增：触发条件加电量守卫
    should_return = (
        self.return_mode
        or (charger_dist < float('inf') 
            and prep.battery <= charger_dist + margin
            and battery_ratio <= 0.65)        # ← 新增守卫：电量 > 65% 不触发
        or battery_ratio <= effective_low_ratio  # ← 用动态比例替代常数
    )

    if not should_return:
        return False, None, charger_dist, margin

    self.return_mode = True
    # ... rest of function unchanged
```

**风险评估**: 中等
- `battery_ratio <= 0.65` 守卫可能让 charger_dist 很大的场景延迟触发
- 但 charger_dist > 65% × battery_max 的场景本来就很危险，margin 也只有 40
- 实际危害较小，因为还有 `battery_ratio <= effective_low_ratio` 兜底（新电池 42% 时一定会触发）

**预期效果**: 
- 消除 93% 电量误触发
- 小电池提前 10% 左右触发充电（从 32% 到 42%）
- 减少电池死亡 40-50%

---

### Fix 4: NPC 近距离时抑制充电 bias — 解决充电/避让冲突

**优先级**: ⭐⭐⭐⭐ (高)

**修改文件**: `expert.py` L207-219

**当前问题**:
- 充电 bias [3, 100] vs NPC 负 bias [-2]
- 充电方向压过避让方向
- NPC dist ≤ 3 时两个系统打架

**修改方案**:

```python
def get_logit_bias(self, prep, legal_action, last_action=-1):
    """Soft logit bias for charging — replaces hard override during training."""
    bias = np.zeros(8, dtype=np.float32)
    should_return, expert_action, charger_dist, margin = self._evaluate_return(
        prep, legal_action, last_action
    )

    if should_return and expert_action is not None:
        # ★ 新增：检查 NPC 最小距离
        min_npc_dist = float('inf')
        hx, hz = prep.cur_pos
        for npc in prep._npcs:
            _pos = npc.get("pos") or {}
            _nx, _nz = int(_pos.get("x", 0)), int(_pos.get("z", 0))
            _ndx, _ndz = _nx - hx, _nz - hz
            _npc_dist = max(abs(_ndx), abs(_ndz))
            min_npc_dist = min(min_npc_dist, _npc_dist)

        # ★ 改进：NPC 接近时不施加充电 bias
        if min_npc_dist <= 4:
            # NPC 太近，让 NPC 负 bias 主导
            pass  # 不施加充电正 bias
        else:
            # NPC 足够远，施加充电 bias
            slack = prep.battery - charger_dist
            urgency = float(np.clip(1 - slack / max(margin, 1), 0.2, 1.0))

            if slack <= 3 or (prep.battery / max(prep.battery_max, 1)) <= 0.10:
                bias[expert_action] = 100.0  # Emergency
            else:
                bias[expert_action] = 3.0 + 5.0 * urgency  # Soft [3, 8]

    # NPC avoidance bias (unchanged)
    hx, hz = prep.cur_pos
    for npc in prep._npcs:
        _pos = npc.get("pos") or {}
        _nx, _nz = int(_pos.get("x", 0)), int(_pos.get("z", 0))
        _ndx, _ndz = _nx - hx, _nz - hz
        _npc_dist = max(abs(_ndx), abs(_ndz))
        if _npc_dist > 6 or _npc_dist < 1:
            continue
        for _idx, (_dx, _dz) in enumerate(self.DELTAS):
            if not legal_action[_idx]:
                continue
            _nlen = max(max(abs(_ndx), abs(_ndz)), 1.0)
            _dot = (_dx * _ndx + _dz * _ndz) / _nlen
            if _dot > 0:
                _close = (6.0 - _npc_dist) / 6.0
                bias[_idx] -= 2.0 * _dot * _close

    return bias
```

**风险评估**: 中等
- NPC 近距离时完全放弃充电正 bias
- 可能导致紧急时（slack 很小且 NPC dist ≤ 4 同时发生）无法充电
- 但 filter_actions 已经阻止了朝 NPC 方向移动，机器人可以先躲避再充电
- 冲突时：躲避 > 充电（生命安全第一）

**预期效果**: 消除充电/避让方向选择冲突，再减少碰撞 5-10%

---

### Fix 5: 降低未探索区域代价 — 便于 A* 早期找到充电路径

**优先级**: ⭐⭐⭐ (中)

**修改文件**: `expert.py` L299

**当前问题**:
```python
_UNEXPLORED_COST = 3.0  # 太高
```

早期训练（explored ratio < 30%）时：
- 大地图上已探索/未探索交界处的距离被严重高估
- A* 找不到充电路径 → charger_dist=inf → battery death

**修改方案**:
```python
_UNEXPLORED_COST = 1.8  # 从 3.0 降到 1.8
```

解释：
- 已探索 passable = 1.0 + visit_penalty (0-0.75) = [1.0, 1.75]
- 未探索 1.8 仍高于大多数已探索格子（除高渡道），但不再禁止性
- A* 更愿意穿过未探索区域去充电

**风险评估**: 低
- A* 可能规划穿过未探索但实际有墙的区域
- `update_blocked()` 会在碰壁后标记这些格子为 blocked_penalty=+4.0
- 下次 A* 会自动绕行（代价从 1.8 变 5.8+）
- 自适应修正，不是一次性问题

**预期效果**: 减少早期阶段（step < 2000）的 A* 无路径情况，减少电池死亡 10-15%

---

## 四、修改优先级与实施顺序

| 顺序 | Fix | 目标 | 复杂度 | 代码行数 | 预期收益 |
|------|-----|------|--------|---------|---------|
| **1** | Fix 1:filter_actions 扩展 | 防碰撞 | 低 | 8 | -60% 碰撞 |
| **2** | Fix 3: 移除 npc_weight=0 | 防碰撞 | 极低 | 10(删除) | -20% 碰撞 |
| **3** | Fix 4: NPC 近距离抑制 bias | 防碰撞 | 低 | 15 | -10% 碰撞 |
| **4** | Fix 2: return_mode 守卫 + 动态比例 | 防电池+误触发 | 低 | 12 | -40% 电池死 |
| **5** | Fix 5: 降低 UNEXPLORED_COST | 防电池 | 极低 | 1 | -15% 电池死 |

**建议一次性上线 Fix 1-4**（共 45 行代码），互相补充，效果显著。Fix 5 可选（风险极低，效果有限）。

---

## 五、测试检查单

修改完成后，训练前检查：

- [ ] `filter_actions()` 中 dist ≤ 3 时新逻辑已添加
- [ ] A* 的两个 fallback 函数中第 3 级删除完毕
- [ ] `get_logit_bias()` 中 NPC dist 检查逻辑添加
- [ ] `_evaluate_return()` 中 effective_low_ratio 和 0.65 守卫添加
- [ ] `_UNEXPLORED_COST` 改为 1.8（可选）

---

## 六、预期训练效果

### 修改后预期指标（基于现有 v52 数据）

| 指标 | v5.2 当前 | 修改后预期 | 理由 |
|------|----------|----------|------|
| WinRate | 85% | 93-96% | 碰撞-70%, 电池死-50% |
| AvgCS | 865 | 950-1050 | 失败减少，episode 增多 |
| MinCS | 53 | 400+ | 消除早期碰撞+极端电池 |
| Collision death | 56% of fail | <10% | Fix1-4 针对性强 |
| Battery death | 44% of fail | <15% | Fix2-5 针对性强 |

### 训练进度预计

- **Step 0-2000**: 修改生效快速，WinRate 迅速上升到 90%+，MinCS 快速改善
- **Step 2000-5000**: 稳定阶段，可观察 RL 是否开始学习独立充电决策（charge_count、urgency_response）
- **Step 5000+**: 如果 Survival ≥ 93% 稳定，可开始降低 logit bias 上限

---

## 七、中长期改进计划

### 阶段 2（训练 5k 步后）：降低 Expert 制导强度

观察 Survival > 93% 稳定后：

1. **降低 logit bias 上限**
   ```python
   bias[expert_action] = 2.0 + 3.0 * urgency  # 从 [3,8] -> [2,5]
   ```

2. **移除 emergency=100**
   ```python
   # 改为
   bias[expert_action] = 5.0 + 5.0 * urgency  # [5,10] high but not override
   ```

3. **增强 RL 充电信号**
   - urgency_penalty: -0.4 → -1.3（已在 v5.3 plan 中）
   - charger_reward scale: 0.40 → 0.60
   - efficiency_bonus: 0.3 → 0.5

### 阶段 3（训练 20k 步后）：Expert 纯安全网化

目标：Expert 只在极端紧急（battery_ratio < 0.20）时才硬干预，其他正常充电由 RL 完全掌控。

实施：
1. 训练模式 logit_bias 完全依赖 battery_ratio（不用 charger_dist 条件）
2. 评估模式保留 hard override（max survival 需求）
3. 删除复杂的 state machine，改为简单阈值

---

## 八、实施建议

### 代码修改步骤

1. **备份原文件**
   ```bash
   cp expert.py expert.py.backup_20260414
   ```

2. **依次应用 5 个修改**（建议用 multi_replace 一次完成）

3. **代码审视**
   - 检查 filter_actions 从 dist ≤ 5 到 dist ≤ 3 的过渡是否平滑
   - 检查 effective_low_ratio 公式的边界条件
   - 检查 NPC dist 检查中 `min_npc_dist` 的初始化

4. **单元测试**
   - 手写测试用例验证 filter_actions 的 dist ≤ 3 和 4-5 逻辑分界
   - 测试 edge case: legal_action 全为 0 时的兜底

### 训练启动

```bash
cd /workspace && python train_test.py -p kaiwu-train
```

### 监控指标

**关键看板**（first 1000 steps）：
- Episode WinRate (target > 90%)
- Episode CollisionDeathRate (target < 5%)
- Episode BatteryDeathRate (target < 10%)
- Charge_count per episode (should increase smoothly)

---

## 九、常见问题

### Q: Fix 1 中 dist ≤ 3 可能导致无合法动作，怎么办？
**A**: 代码已有兜底 `if sum(legal) == 0: return list(legal_action)` — 返回原始 legal 让机器人移动，虽然不安全但比卡住强。实际上这种情况很稀少（需要被 NPC 完全包围）。

### Q: Fix 3 中 0.65 守卫是否太保守？
**A**: 在大地图场景中不算保守。charger_dist 最高 400+，margin 最高 40，665 × 0.65 ≈ 432 > 440 多数情况仍会触发。对小地图（dist < 30）保守一点反而是安全的。

### Q: Fix 4 关闭充电 bias 后紧急充电怎么处理？
**A**: 
- Filter_actions 已防止朝 NPC 移动
- 模型在安全距离外仍有 urgency_penalty 信号
- NPC 一旦离开就立即恢复充电 bias
- 结果：模型学会先躲再充（更安全的策略）

### Q: Expert 现在是不是已经完美了？
**A**: 不是。这 5 个修改解决了 bug，但高复杂度本身仍是隐患。中长期仍应逐步简化 Expert，让 RL 学会自主决策。

---

**报告完成日期**: 2026-04-14  
**审查人**: Code Agent  
**建议行动**: 实施 Fix 1-4，启动新轮训练，3000 steps 后审视数据决定是否后续调整
