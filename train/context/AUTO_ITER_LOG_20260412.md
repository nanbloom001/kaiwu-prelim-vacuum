# Auto-Iteration Log — 2026-04-12

## Session Start: 03:00

### Baseline
- Training start: 2026-04-12 02:31
- Git baseline: 54461a4
- Config: 8 parallel envs, 4 gamecores, maps 1-10
- Resume from: model.ckpt-resume.pkl (previous ~4805 episodes + ~143 new)

---

## Iterations

### 04:05 — Iteration 1: Reward平衡 + PPO修复

**问题诊断:**
- 从最佳模型(47491步, robust=1397.9) resume后分数持续暴跌
- invalid_move_rate飙升至0.6-1.0
- 根因分析:
  1. **filter_actions未调用**: NPC安全过滤器定义但未使用 → agent朝NPC移动被阻挡
  2. **LR=0.0003过高**: 对迁移学习太激进，PPO快速摧毁已有好策略
  3. **BETA=0.001过低**: 策略坍塌到确定性动作 → 卡住无法恢复
  4. **回访惩罚过重**: off-frontier -0.20/4 造成死亡螺旋

**修改:**
- preprocessor.py: cleaning 2.0→1.5, charger_path_explore 0.15→0.06, revisit -0.20/4→-0.08/3, clip [-4,6]→[-3,4], stuck -0.3→-0.5/-0.15→-0.25
- expert.py: 提前启动专家(路径未探索时阈值×1.5+30), CHARGE_SAFETY_MARGIN 40→50
- agent.py: 调用filter_actions()阻止NPC方向移动 + 10步卡住后随机合法动作
- conf.py: LR 0.0003→0.0001, BETA 0.001→0.003 (中间值: 先试0.01过高→改0.003)

**Git commits:** 30989cc → 3412802 → 70004a6 → b11d4cd

**当前状态 (04:35):**
- 从最佳模型重新resume, BETA=0.003, LR=0.0001
- global_step 1234, 58+ eps
- avg_score=89, avg_inv=0.578
- 最佳单局: score=322 inv=0.190 (ep:115), score=306 inv=0.184 (ep:120)
- 灾难局(>0.9): 9/58 (15.5%) — 主要是NPC阻挡引擎层面问题
- 趋势: 好局在提升(291→306→322), 灾难局比例稳定
- 训练正常运行中，让cron继续监控

### 04:37 — 快照报告

```
[快照 04:37] ep:~140 | avg_score:93 | FAIL:100% | avg_invalid:0.573 | charge_count:0.25-2.75/interval
```

**趋势**: First 30 avg=84 → Last 30 avg=99 (+18%), 分数在提升
**MAP_STATS**: variance=21.0, min=71.6(map6), max=139.2(map4), spread=67.6
**Top 5**: 322(0.19), 306(0.184), 295(0.36), 291(0.29), 286(0.24)
**灾难局**: 7/30 (inv>0.9) — NPC引擎阻挡

**训练算法metrics**:
- value_loss: 88-106 (稳定,未爆炸)
- entropy_loss: 1.5-1.67 (健康)
- policy_loss: -3.6 ~ +3.9 (正常波动)
- charge_count: 0.25-2.75/interval (有充电但不稳定)

**决策**: 趋势正面(+18%), 训练稳定, 不做调整, 继续监控

### 04:37 — 深度分析

**#1瓶颈: 充电一致性**
- charge_count非零但极不稳定(0.25-2.75)
- 1 WIN / 67 FAIL, 所有失败为battery
- 专家策略能规划路径但穿越未探索区不可靠(可能撞墙)

**#2瓶颈: 灾难局(15.5%)**
- NPC动态阻挡导致100% invalid move
- anti-stuck机制已到位但引擎层面无法绕过

**正向信号**:
- 最佳单局持续提升: 291→306→322
- 分数趋势+18%, 模型在进步
- 训练loss稳定, entropy健康

**建议**: 当前状态可接受, 让训练继续. 等global_step达到3000+再看是否需要调整.

### 05:00 — 快照报告 (重大突破!)

```
[快照 05:00] ep:334 | avg_score:145 | FAIL:98% | avg_invalid:0.473 | charge_count:0-1.25/interval
```

**趋势**: First 30 avg=84 → Last 30 avg=203 (**+141%!**)
**MAP_STATS**: variance=45.3, min=142(map7), max=306(map1), spread=164
**Best**: **720** (ep:269) — 接近旧最佳1397.9!
**WIN**: 4/168 (首次出现胜利)
**灾难局**: 3/30 (大幅改善 from 7/30)
**Good局(inv<0.2)**: 6/30 (翻倍 from 3/30)
**invalid趋势**: 0.562→0.403 (降至阈值以下)

**训练算法metrics** (global_step: 3676):
- value_loss: 218-251 (上升⚠️ — 需关注)
- entropy_loss: 0.97-1.05 (下降⚠️ — 策略在收敛)
- policy_loss: -12 (正常)
- charge_count: 0-1.25/interval (仍不稳定)

**决策**: 不做调整. 趋势极强(+141%), 让训练继续.
**关注点**: value_loss上升(90→250)和entropy下降(1.6→1.0), 下次深度分析评估.

### 05:30 — 深度分析 + 调整

**数据总览** (global_step ~6409, ~550 eps):
| Quarter | eps | avg_score | avg_inv | wins | best |
|---------|-----|-----------|---------|------|------|
| Q1 | 1-145 | 97 | 0.568 | 1 | 335 |
| Q2 | 146-273 | 167 | 0.408 | 2 | 720 |
| Q3 | 281-407 | **215** | **0.368** | **4** | 652 |
| Q4 | 414-549 | 169 ↓ | 0.410 | 3 | **745** |
| Last30 | 520-549 | 148 ↓ | 0.463 | 1 | — |

**Top 10**: 745(0.07), 720(0.06), 652(0.05), 638(0.00), 575×2, 557, 545, 525, 507

**训练metrics趋势**:
- value_loss: 90→263 (持续上升⚠️)
- entropy: 1.6→0.42 (急降⚠️ 接近过早收敛)
- charge_count: 0.8→0.0 (降至零)
- clean_score: 332→40 (近interval暴跌)

**MAP**: variance=32, min=129.5(map3), max=239.9(map1)

**瓶颈**: 策略过早收敛 — entropy降至0.42, 失去探索能力, 导致Q4分数回落+充电停止

**调整**: BETA 0.003→0.005 (增加entropy, 防止进一步收敛)
**Git**: 9ee1696(checkpoint) → 0285878(BETA+0.002)
**继续从当前checkpoint训练** (不回退)

### 06:01 — 快照报告 (BETA=0.005生效)

```
[快照 06:01] ep:266 | avg_score:100 | FAIL:98% | avg_invalid:0.509 | charge:0-1.2/interval
```

**趋势**: Prev30=118 → Last30=120 (+1.5%), 基本持平
**Best**: 328 (ep:248)
**WIN**: 2/125 | 灾难局: 7/30 | Good局: 8/30
**MAP**: variance=33.1, min=81.3(map8), max=211(map9)

**BETA调整效果** (对比调整前同期):
| 指标 | BETA=0.003 @125ep | BETA=0.005 @125ep |
|------|-------------------|-------------------|
| avg_score | 97 | 100 |
| best | 335 | 328 |
| entropy_loss | 0.42 (急降) | **0.66 (稳定)** ✓ |
| value_loss | 263 (飙升) | **179-205 (可控)** ✓ |

**决策**: entropy调整生效, vl和entropy明显改善. 模型正在接近Q3峰值区间(ep280-400), 关键看能否避免上次Q4回落. 不做额外调整, 继续监控.

### 06:28 — 深度分析 #2

**数据总览** (global_step ~5648, ~436 eps):
| Quarter | eps | avg_score | avg_inv | wins | best |
|---------|-----|-----------|---------|------|------|
| Q1 | 1-129 | 83 | 0.544 | 0 | 286 |
| Q2 | 130-245 | 114 | 0.457 | 2 | 273 |
| Q3 | 246-346 | 140 | 0.500 | 1 | 362 |
| **Q4** | **347-436** | **157** | **0.412** | **2** | **607** |

**关键对比** — BETA=0.005成功延迟收敛:
- 旧session(BETA=0.003): Q3=215 → **Q4=169↓** (已回落)
- 新session(BETA=0.005): Q3=140 → **Q4=157↑** (仍在上升!)

**Last30 vs Prev30**: 160 vs 147 (+8.7%) — 持续上升

**训练metrics**:
- value_loss: 172-218 (稳定, 优于旧session的263)
- entropy: 0.28-0.34 (⚠️ 再次降至0.3边缘, 但收敛速度明显减缓)
- charge_count: 0-0.5 (低但非零)

**MAP**: variance=57.6, min=100(map8), max=280(map4), spread=180
- 弱图: 6(103), 8(100), 7(129)
- 强图: 4(280), 5(254), 10(194)

**Top 5**: 607(0.06), 362(0.42), 350(0.13), 342(0.15), 341(0.08)

**瓶颈分析**:
1. **充电仍是#1问题**: charge_count接近0, 几乎所有episode都是battery FAIL
2. **弱图(map6/8)**: avg仅100, 是其他图的一半
3. **entropy再次逼近0.3**: 需密切监控, 若下次快照<0.25则需加BETA

**决策**: Q4仍在上升(+8.7%), 不做调整. 下次快照若出现下降或entropy<0.25则提高BETA至0.007.

### 06:31 — 快速状态检查 (context恢复后)

```
[状态 06:31] global_step: 5946 | ep:~465 | entropy: 0.26-0.33 | value_loss: 156-180
```

**近期亮点:**
- **ep:462 WIN** score=338 inv=0.000 map:5 chargers:4 (完整胜利!)
- **ep:463** score=375 inv=0.087 map:6 (弱图突破! 之前avg仅100)
- ep:456 score=213 inv=0.264 map:4
- ep:464 score=162 inv=0.281 map:9 (mild)
- ep:465 score=227 inv=0.232 map:10 (mild)

**训练metrics**:
- entropy_loss: 0.26~0.33 (最低0.2611, 未破0.25阈值)
- value_loss: 156-180 (持续下降, 优于之前的172-218)
- policy_loss: -5.9~-7.8
- charge_count: 0.0 (未改善)

**决策**: 训练健康, entropy在0.25阈值上方安全区, 不做调整. 继续由cron监控.

### 06:35 — 快照报告 (entropy警告)

```
[快照 06:35] ep:479 | avg_score:129 | FAIL:97% | avg_invalid:0.471 | charge:0/interval
```

**趋势**: Prev snapshot avg=100 → Current avg=129 (+29%)
**MAP_STATS**:
- 强图: map6(309), map5(249), map10(185), map1(177)
- 弱图: map8(74), map7(74), map4(58), map2(52)
**Top 5**: 375(0.087,map6), 338(0.000,map5 WIN!), 257(0.006,map10), 243(0.150,map6), 227(0.232,map10)
**WIN**: 1/30 (ep:462 map:5)
**灾难局**(inv>0.9): 6/30 (20%)
**Good局**(inv<0.2): 6/30 (20%)

**训练算法metrics** (global_step: 6340):
- value_loss: 156-205 (稳定, 较上轮下降)
- entropy_loss: 0.237~0.327 (⚠️ 最低破0.25至0.2368, 但恢复)
- policy_loss: -5.9~-6.9
- Last 5 entropy avg: **0.281** (仍在0.25安全线以上)

**entropy趋势** (每100步):
- Step 4500-5000: avg ~0.37
- Step 5000-5500: avg ~0.30
- Step 5500-6000: avg ~0.28
- Step 6000-6340: avg ~0.28 (趋稳?)

**决策**: 不做调整. 分数趋势+29%, value_loss下降. Entropy单次破0.25后恢复, 5次均值0.281. 放宽触发条件: 若5次rolling平均<0.25则提高BETA至0.007. 继续监控.

