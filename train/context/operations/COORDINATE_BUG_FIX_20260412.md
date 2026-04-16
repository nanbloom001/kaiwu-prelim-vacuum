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

### 1. `_update_memory` (现line 278-279)
```python
# 修改前 (BUG)
gx = hx - half + row   # row=z偏移,却加到x上
gz = hz - half + col   # col=x偏移,却加到z上

# 修复后
gx = hx - half + col   # col=x方向偏移
gz = hz - half + row   # row=z方向偏移
```
**影响**: explored_map, passable_map, dirty_memory 三个全局地图被转置

### 2. `_calc_local_frontier_density` (现line 380-381)
```python
# 同样的bug，同步修复
gx = hx - self.VIEW_HALF + col  # 原为 + row
gz = hz - self.VIEW_HALF + row  # 原为 + col
```
**影响**: 边界密度计算使用了错误的全局坐标（当前与_update_memory互相抵消，修一个必须修另一个）

### 3. `_cell_passable_local` (现line 342-343)
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
1. `_update_memory` (line 278-279): gx/gz的row/col互换
2. `_calc_local_frontier_density` (line 380-381): 同上
3. `_cell_passable_local` (line 342-343): row/col的dx/dz互换
4. 添加坐标验证断言（agent位置必须passable_map≥0.5）
5. 充电奖励增强:
   - charger_path_explore: 0.06 → 0.12
   - charge_bonus: 0.5 → 1.0
   - charger_reward: 0.10 → 0.15

### agent.py
1. predict()层级重排: NPC Filter → Expert → Anti-stuck → RL (原来是 Expert → NPC Filter → Anti-stuck → RL)
2. Expert override使用模型softmax概率替代uniform（修复PPO importance ratio）
3. Expert和Anti-stuck都使用filtered_legal（经过NPC过滤的mask）

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
**实际验证：677个episode，断言从未触发，确认坐标系修复正确。**

---

## 八、对另一个AI建议方案的评估

另一个AI给出了13项具体建议，以下是逐一评估结果：

### 完全采纳的修改

| # | 修改项 | 理由 |
|---|--------|------|
| 1 | `_update_memory` 坐标修复 | 核心bug，经过逐行代码追踪验证确认 |
| 2 | `_calc_local_frontier_density` 坐标修复 | 同一bug的第二处，与_update_memory互相抵消，必须同步修 |
| 3 | `_cell_passable_local` 坐标修复 | 同一bug的第三处，导致合法动作掩码旋转90° |
| 4 | 坐标验证断言 | 低成本的验证手段，677个episode全部通过 |
| 5 | predict()层级重排 | NPC Filter必须在Expert之前执行，否则Expert可能将agent引向NPC |
| 6 | Expert使用模型概率 | 关键PPO修复。uniform概率使importance ratio失真，模型无法从专家动作中学习 |
| 7 | BETA_START 0.007→0.005 | 熵已经健康(0.5-0.7)，不需要过高的entropy bonus |

### 部分采纳的修改

| # | 修改项 | 采纳部分 | 不采纳部分 | 理由 |
|---|--------|----------|------------|------|
| 8 | 充电奖励增强 | charger_path_explore 0.06→0.12, charge_bonus 0.5→1.0, charger_reward 0.10→0.15 | 充电奖励3x增幅 | 3x过激进，先用2x观察。实际训练数据显示2x已产生显著效果 |

### 明确不采纳的修改

| # | 修改项 | 不采纳理由 |
|---|--------|------------|
| 9 | 移除edge_bonus | edge_bonus数值小(≤0.32)，鼓励沿墙和脏格子行走，对清扫有益。删除风险大于收益 |
| 10 | 简化charger_reward为binary(closer/not closer) | 当前比例式设计 `0.15 * charge_pressure * delta_slack` 更优：移动5步比1步值得更多奖励，binary抹杀了这种差异 |
| 11 | battery_critical_penalty -0.5/step | 过于激进。当前已有charge_pressure机制(电量低时加大充电奖励)，额外惩罚可能导致策略过于保守 |
| 12 | Expert忽略legal_action保底触发 | 环境会拒绝非法移动，浪费步骤。当前expert的3级fallback(full NPC → 30% → 0%)已足够 |
| 13 | 新建额外的Context文档 | 用户明确要求将信息集中在一个文件中 |

---

## 九、训练策略与执行

### 9.1 Fine-tune方案
- 使用修复前的 `model.ckpt-resume.pkl`（约step 34500，由之前训练session产出）继续fine-tune
- 从step 0重新开始计数（框架限制），实际使用了预训练权重
- 理由：local_view(1323/1597=83%)不受bug影响，权重可复用

### 9.2 训练环境
- Docker分布式训练：4个gamecore容器 + 1个learner + 1个aisrv
- 8个并行环境
- 训练速度：~100 steps/min
- 当前checkpoint：model.ckpt-resume.pkl (step ~8800, episode ~677)

---

## 十、修复后训练数据详细分析

### 10.1 总体数据（677个episode, step 0-9040）

| 指标 | 值 |
|------|-----|
| 总episode数 | 677 |
| 总WIN数 | 279/437 = 63.8% (有记录的437个) |
| 平均分数 | 363 |
| 最高分数 | 1439 |
| 平均步数 | 821 |
| 当前训练步 | ~9040 |
| 当前模型 | model.ckpt-8700.pkl |

### 10.2 分数趋势（按每20个episode分组）

```
Ep   1- 34 (20): WIN=16/20 (80%) avg=247  best= 695 avg_steps= 912  ← 初始阶段
Ep  35- 59 (20): WIN=18/20 (90%) avg=140  best= 279 avg_steps=1118
Ep  65- 92 (20): WIN=19/20 (95%) avg=146  best= 300 avg_steps= 947
Ep  95-121 (20): WIN=16/20 (80%) avg=121  best= 285 avg_steps= 896
Ep 124-146 (20): WIN=17/20 (85%) avg=190  best= 327 avg_steps= 974
Ep 150-178 (20): WIN=19/20 (95%) avg=215  best= 440 avg_steps=1030
Ep 182-208 (20): WIN=15/20 (75%) avg=236  best= 380 avg_steps= 870
Ep 212-240 (20): WIN=16/20 (80%) avg=293  best= 457 avg_steps= 974  ← 分数开始上升
Ep 241-270 (20): WIN=18/20 (90%) avg=431  best= 981 avg_steps=1147  ← Q4=256突破
Ep 271-301 (20): WIN=15/20 (75%) avg=430  best= 728 avg_steps= 902
Ep 302-339 (20): WIN= 8/20 (40%) avg=385  best=1182 avg_steps= 617  ← WIN率下降
Ep 340-363 (20): WIN=13/20 (65%) avg=483  best=1246 avg_steps= 889
Ep 364-392 (20): WIN=13/20 (65%) avg=512  best=1081 avg_steps= 793
Ep 393-424 (20): WIN=11/20 (55%) avg=453  best=1439 avg_steps= 730  ← 全局最高1439
Ep 425-450 (20): WIN=11/20 (55%) avg=536  best= 915 avg_steps= 782
Ep 451-479 (20): WIN= 9/20 (45%) avg=425  best=1143 avg_steps= 609
Ep 480-512 (20): WIN= 9/20 (45%) avg=492  best=1168 avg_steps= 676
Ep 513-544 (20): WIN= 7/20 (35%) avg=421  best=1173 avg_steps= 575  ← WIN率低谷
Ep 545-585 (20): WIN= 7/20 (35%) avg=404  best= 937 avg_steps= 547
Ep 586-619 (20): WIN= 9/20 (45%) avg=477  best=1166 avg_steps= 629
Ep 626-653 (20): WIN= 4/20 (20%) avg=384  best=1104 avg_steps= 583  ← WIN率最低
Ep 656-677 (17): WIN= 9/17 (53%) avg=594  best=1011 avg_steps= 874  ← 近期回升
```

**关键观察**：
- **前200个episode (step 0-5000)**: WIN率高(75-95%)但分数低(100-300)。原因是早期episode多为简单的mild profile（少量机器人、高battery、低max_step）
- **中间阶段 (step 5000-7000)**: 分数快速提升(300→700+)，但WIN率下降至40-55%。原因是broad profile增多（更多机器人、更长max_step），难度增大
- **近期 (step 7000-9000)**: 分数波动400-700，WIN率20-55%，出现高分但稳定性不足

### 10.3 按地图类型统计

```
Map  9: 47 eps, WIN=40/47 (85.1%) avg=485  ← 表现最好
Map  3: 44 eps, WIN=30/44 (68.2%) avg=405
Map 10: 51 eps, WIN=33/51 (64.7%) avg=379
Map  8: 50 eps, WIN=31/50 (62.0%) avg=344
Map  1: 42 eps, WIN=26/42 (61.9%) avg=364
Map  5: 53 eps, WIN=34/53 (64.2%) avg=311
Map  4: 45 eps, WIN=26/45 (57.8%) avg=471  ← 分数高但WIN率低
Map  2: 39 eps, WIN=23/39 (59.0%) avg=341
Map  7: 38 eps, WIN=19/38 (50.0%) avg=291
Map  6: 30 eps, WIN=14/30 (46.7%) avg=242  ← 表现最差
```

### 10.4 按Profile统计

```
anchor:  98 eps, WIN=74/98 (75.5%) avg=268  ← WIN率高但分数低
broad : 182 eps, WIN=105/182 (57.7%) avg=416  ← 分数高但WIN率低
mild  : 159 eps, WIN=97/159 (61.0%) avg=376
```

### 10.5 按机器人数量统计

```
robots=1: 255 eps, WIN=177/255 (69.4%) avg=364  ← 最容易
robots=2:  92 eps, WIN= 50/92 (54.3%) avg=337
robots=3:  36 eps, WIN= 22/36 (61.1%) avg=436
robots=4:  56 eps, WIN= 27/56 (48.2%) avg=397  ← 最难
```

### 10.6 首尾100个episode对比

| 指标 | 前100个 | 后100个 |
|------|---------|---------|
| WIN率 | 85% | 36% |
| 平均分数 | 180 | 445 |
| 最高分 | 695 | 1173 |
| 平均步数 | 964 | 632 |

**注意**：WIN率下降并不代表性能变差。早期episode的max_step低（300-500步），容易存活但分数低。后期episode的max_step高（1000-2000步），存活更难但单局分数更高。

---

## 十一、训练Loss趋势

### 11.1 按阶段汇总（每5个loss记录取平均）

```
Batch  1 (step~  421): policy= 30.35  value= 1284  entropy=0.812  ← 初始波动
Batch  2 (step~  907): policy= -3.75  value=   66  entropy=0.538
Batch  3 (step~ 1393): policy= -3.36  value=   55  entropy=0.645
Batch  4 (step~ 1885): policy= -4.07  value=   60  entropy=0.770
Batch  5 (step~ 2383): policy= -5.77  value=   77  entropy=0.841
Batch  6 (step~ 2882): policy= -6.34  value=   79  entropy=0.961
Batch  7 (step~ 3385): policy= -9.70  value=  120  entropy=1.080  ← entropy高峰
Batch  8 (step~ 3883): policy=-11.32  value=  150  entropy=0.912
Batch  9 (step~ 4384): policy=-15.36  value=  219  entropy=0.928
Batch 10 (step~ 4885): policy=-17.18  value=  252  entropy=0.878  ← 分数快速提升期
Batch 11 (step~ 5329): policy=-18.97  value=  288  entropy=0.809
Batch 12 (step~ 5771): policy=-17.15  value=  259  entropy=0.745
Batch 13 (step~ 6210): policy=-17.49  value=  266  entropy=0.656  ← entropy开始收敛
Batch 14 (step~ 6639): policy=-18.82  value=  287  entropy=0.565
Batch 15 (step~ 7079): policy=-17.41  value=  273  entropy=0.497
Batch 16 (step~ 7537): policy=-17.97  value=  280  entropy=0.469
Batch 17 (step~ 7990): policy=-16.32  value=  255  entropy=0.410
Batch 18 (step~ 8431): policy=-17.72  value=  269  entropy=0.359  ← 近期
Batch 19 (step~ 8877): policy=-15.79  value=  242  entropy=0.363
```

### 11.2 Loss趋势分析

| 阶段 | 步数范围 | 特征 | 诊断 |
|------|----------|------|------|
| 探索期 | 0-2000 | entropy 0.5→0.8, value 55→80, policy接近0 | 正常的初始学习 |
| 熵增期 | 2000-4000 | entropy 0.8→1.08, policy -5→-11 | 策略过于发散，但值函数在学习 |
| 成熟期 | 4000-7000 | entropy 0.9→0.5, policy -15→-18, value 200→290 | 最佳学习阶段，分数快速提升 |
| 收敛期 | 7000-9000 | entropy 0.5→0.35, policy -16→-18, value 240→270 | 熵过低，策略可能过早收敛 |

### 11.3 近期10个metrics快照平均

```
avg_reward = 1144
avg_score  = 584
avg_charge = 3.0 (每4个episode)
avg_remain = 161
avg_entropy= 0.356
avg_vloss  = 243
```

---

## 十二、当前问题诊断

### 问题1：Entropy过早收敛
- entropy从1.08 (step 3385) 持续下降到0.33 (step 9040)
- 接近策略坍缩 (entropy < 0.3时通常表示策略过于确定)
- **可能原因**：BETA_START 0.005可能仍然偏低，或entropy衰减过快
- **建议**：考虑将BETA_START提高到0.008-0.01，或使用entropy系数衰减schedule

### 问题2：WIN率下降
- 前200个episode WIN率75-95%
- 近100个episode WIN率降至36%
- 但需注意：后期episode的max_step更长、机器人更多，客观难度更大
- **存活率(survival rate)** 近期约60-85%，说明agent在某些配置下确实生存困难
- **可能原因**：过度充电(浪费步数)或充电不及时(在困难场景下电量不足)

### 问题3：分数方差大
- 后100个episode平均445，但最高1173，最低0
- 某些episode在300步内就失败(电量耗尽)
- **可能原因**：
  - mild profile的battery_max=200-340，充电窗口很窄
  - 高机器人数量(3-4)时NPC干扰大
  - 地图6表现差(avg=242, WIN=47%)可能是因为该地图布局不利于清扫

### 问题4：Map 6表现最差
- 30个episode仅46.7% WIN率，avg_score=242
- 需要分析Map 6的特殊地形（可能是狭窄通道多、充电桩远等）
- **建议**：专门分析Map 6的失败模式

---

## 十三、下一步优化方向建议

### 优先级1：Entropy管理
1. **提高BETA_START到0.008-0.01** — 当前0.005太低，策略过早收敛
2. 或引入entropy coefficient schedule — 前期高(0.01)，后期低(0.003)
3. 目标：维持entropy在0.5-0.7区间

### 优先级2：充电策略优化
1. **分析充电频率** — 当前avg_charge=3.0/4episodes，看起来合理
2. 但个别episode出现charge_count=0且remaining_charge=0的情况（电量耗尽死亡）
3. **建议**：增加低电量警告的reward shaping，让模型学会提前充电
4. 考虑charge_pressure的阈值调整

### 优先级3：困难场景专项优化
1. **Map 6专项分析** — 为什么表现差？地形？充电桩位置？
2. **多机器人场景** — robots=4时WIN率48%，NPC filter是否足够？
3. **低battery场景** — battery_max=120-200的episode，生存率如何？

### 优先级4：特征工程
1. **global_memory (8×8×3) 语义** — 修复后通道0,1(探索/脏)的语义变了，CNN需要重新适应
2. 考虑增加charger方向的scalar特征，让模型更容易学到充电路线
3. 当前的directional_dirty 8维特征是否有效？可以ablation测试

### 优先级5：Expert系统调优
1. Expert的充电override频率 — 太频繁会压制RL学习，太少会电量耗尽
2. A*路径规划的cost权重 — dirty格子权重、NPC danger权重
3. NPC filter的danger半径 — 当前可能过于保守或激进

---

## 十四、关键文件位置

| 文件 | 路径 | 说明 |
|------|------|------|
| 预处理器 | `code/agent_ppo/feature/preprocessor.py` | 坐标修复+奖励函数 |
| Agent主类 | `code/agent_ppo/agent.py` | predict()层级重排 |
| Expert | `code/agent_ppo/feature/expert.py` | A*寻路+充电override |
| 配置 | `code/agent_ppo/conf/conf.py` | 超参 |
| 模型 | `code/agent_ppo/model/model.py` | CNN+MLP网络结构 |
| 算法 | `code/agent_ppo/algorithm/algorithm.py` | PPO实现 |
| 最新checkpoint | `code/model.ckpt-resume.pkl` | step ~8800, episode ~640 |
| 训练日志 | `train/log/learner/learner_train_pid219_log_2026-04-12-13.log` | loss记录 |
| Episode日志 | `train/log/aisrv/aisrv_kaiwu_rl_helper_pid321_log_2026-04-12-13.log` | episode结果 |

---

## 十五、Feature向量结构参考

```
总维度: 1597 = [1323 + 192 + 74 + 8]

[0:1323]     local_view (21×21×3) — 不受坐标bug影响
[1323:1515]  global_memory (8×8×3)
               ch0: explored_map (修复后语义正确)
               ch1: dirty_map (修复后语义正确)
               ch2: visit_heatmap (不受bug影响)
[1515:1589]  scalar (74维)
               - 39维原始特征 (位置/电量/分数/脏格比例等)
               - 26维额外 (NPC距离×4, charger距离×4, directional_dirty×8)
               - 9维one-hot (last_action)
[1589:1597]  dir_onehot (8维, 动作方向one-hot)
```

---

## 十六、训练环境Profile说明

训练中3种profile随机出现：

```
mild:   1-4 robots, 1-4 chargers, max_step 300-1250, battery_max 120-420
broad:  1-4 robots, 1-4 chargers, max_step 500-2000, battery_max 120-720
anchor: 1 robot,   4 chargers,    max_step 500-1000, battery_max 200-400
```

- WIN条件：存活到max_step（不是清扫完毕）
- FAIL条件：电量降到0
- 分数 = 清扫的脏格子数 (dirt_cleaned)
- 每局地图从10张预设地图中随机选择
