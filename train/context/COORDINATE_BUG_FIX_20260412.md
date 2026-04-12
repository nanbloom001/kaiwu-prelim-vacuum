# 坐标Bug排查与修复记录 — 2026-04-12

## 一、Bug发现过程

另一个AI session分析了训练数据中agent表现不佳的根因，定位到 `preprocessor.py` 中存在系统性的坐标映射错误。经本session逐一验证确认bug确实存在。

## 二、Bug根因分析

### 坐标系约定（来自竞赛文档）
- x轴向右为正，z轴向下为正，原点(0,0)在左上角
- `map_info[row][col]` = `map_info[z][x]`（row是行=z方向，col是列=x方向）
- `cur_pos = (hero.pos.x, hero.pos.z)` = `(x, z)`

### Bug产生原因
代码在 `_view_map` 的 `row/col` 索引与全局坐标 `x/z` 之间建立了错误的映射关系：
- `row`（z方向的行偏移）被错误地加到了 `x` 坐标上
- `col`（x方向的列偏移）被错误地加到了 `z` 坐标上

这导致了全局地图的x/z轴被互换（转置）。

## 三、受影响的3个函数

### 1. `_update_memory` (原line 270-271)
```python
# 修改前 (BUG)
gx = hx - half + row   # row=z偏移,却加到x上
gz = hz - half + col   # col=x偏移,却加到z上

# 修复后
gx = hx - half + col   # col=x方向偏移
gz = hz - half + row   # row=z方向偏移
```
**影响**: explored_map, passable_map, dirty_memory 三个全局地图被转置

### 2. `_calc_local_frontier_density` (原line 372-373)
```python
# 同样的bug，同步修复
gx = hx - self.VIEW_HALF + col  # 原为 + row
gz = hz - self.VIEW_HALF + row  # 原为 + col
```
**影响**: 边界密度计算使用了错误的全局坐标（当前与_update_memory互相抵消，修一个必须修另一个）

### 3. `_cell_passable_local` (原line 334-335)
```python
# 修改前 (BUG)
row = self.VIEW_HALF + dx   # dx=x偏移,却用作行(=z)索引
col = self.VIEW_HALF + dz   # dz=z偏移,却用作列(=x)索引

# 修复后
row = self.VIEW_HALF + dz   # z偏移 → 行索引
col = self.VIEW_HALF + dx   # x偏移 → 列索引
```
**影响**: 合法动作掩码被旋转90°，映射关系: 0↔6(right↔down), 1↔5, 2↔4(up↔left), 3和7对称不受影响

## 四、Bug传导链

```
_update_memory (转置) → explored_map/passable_map/dirty_memory 全部转置
    ↓
    ├─ global_memory特征 (8×8×3通道) → 通道0,1(探索/脏)转置，通道2(访问热力图)未转置 → 不一致
    ├─ _compute_directional_dirty → 读dirty_memory时index正确但数据在错误位置
    ├─ expert.py A*路径规划 → cost_map从转置地图构建 → A*在错误地形上寻路
    └─ _calc_local_frontier_density → 与_update_memory互相抵消(都转置)

_cell_passable_local (转置) → _actual_legal_act (旋转90°)
    ↓
    └─ get_legal_action: env_legal AND rotated_legal → 可能错误阻挡合法动作
       (但不会放行非法动作，env的_legal_act是底线)
```

## 五、为什么模型仍能训练到1063分

1. **local_view (1323/1597 = 83%)不受影响** — 直接从_view_map提取，没有坐标变换
2. **环境的_legal_act兜底** — AND操作只会多阻挡合法动作，不会放行非法动作
3. **Chebyshev距离不受影响** — max(|dx|,|dz|)在x/z互换后不变
4. **CNN能学习旋转特征** — bug是一致性的转置，CNN能适应
5. **充电桩位置正确** — 直接从organ读取(x,z)，未经过转置map

## 六、本次修复的完整变更

### preprocessor.py
1. `_update_memory`: gx/gz的row/col互换
2. `_calc_local_frontier_density`: 同上
3. `_cell_passable_local`: row/col的dx/dz互换
4. 添加坐标验证断言（agent位置必须passable_map≥0.5）
5. 充电奖励增强:
   - charger_path_explore: 0.06 → 0.12
   - charge_bonus: 0.5 → 1.0
   - charger_reward: 0.10 → 0.15

### agent.py
1. predict()层级重排: Expert→NPC Filter → NPC Filter→Expert→Anti-stuck→RL
2. Expert override使用模型softmax概率替代uniform（修复PPO importance ratio）

### conf.py
1. BETA_START: 0.007 → 0.005

### 不需要修改的文件
- expert.py — Phase 1修复后A*自动在正确地图上运行
- algorithm.py — PPO算法未涉及坐标
- model.py — 网络结构未改

## 七、验证断言

在pb2struct末尾添加了断言：
```python
assert self.passable_map[hx, hz] >= 0.5
```
如果坐标系假设错误（即竞赛文档不正确），此断言会在第一步就触发。

## 八、训练策略

使用当前model.ckpt-resume.pkl（step ~34500）fine-tune继续训练。
- local_view(83%)不受影响，权重可复用
- global_memory语义变了，预期初期有波动
- 如果5k步后仍未恢复，考虑从头训练
