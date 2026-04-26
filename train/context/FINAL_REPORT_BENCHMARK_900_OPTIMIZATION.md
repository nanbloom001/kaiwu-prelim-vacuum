# Benchmark-900 优化全记录

**分支**: `win_YJY` | **目标**: holdout dynamic 2×8 avg ≥ 900 | **状态**: 未达成（最高 baseline avg 803.6）

---

## 一、实验时间线总览

### 阶段 0：基础设施 & 基线（2026-04-25）

| 时间 | 事件 | 关键产出 |
|------|------|---------|
| 03:45 | 建立 AGENTS.md 知识库 | 项目结构/约束文档化 |
| 05:15–09:40 | holdout benchmark 基建搭建 | dry-run、分片执行器、analyzer、mutation guard |
| 23:31–00:12 | dynamic 2×4/2×8 benchmark 定型 | 固定 2×8 为最终 gate，2×4 为 fast screen |

**基线结果**:

| 基准 | Overall | Map4 | Map7 | Complete | Battery | Collision |
|------|---------|------|------|----------|---------|-----------|
| 2×4 baseline | **838.9** | 793.2 | 884.5 | 8/8 (100%) | 0 | 0 |
| 2×8 baseline | **803.6** | 714.2 | 892.9 | 15/16 (93.75%) | 1 (6.25%) | 0 |

> 注：分片基线 (`SHARDED_BASELINE`) 为 early-stage 测试：overall 652.4，13/16 complete，battery 6，collision 1，不作为正式基线。

---

### 阶段 1：早期规则 tweak 线（4月25日）

这些在正式 benchmark 基础设施完成前进行，主要依赖训练日志评估：

| 方案 | 方向 | 结论 |
|------|------|------|
| T0: coverage-target return buffer | algorithm.py 增加覆盖目标专属返航余量 | 未独立 benchmark 测试 |
| v4→v5.4 系列 | Expert 充电逻辑修复、bias 调整、battery death 根因诊断 | 关键经验：battery death 是主瓶颈，expert A* bias 3-8 太弱 |

---

### 阶段 2：系统性候选实验（2026-04-26）

以下为所有经 2×4 fast screen + 部分经 2×8 gate 验证的候选方案。

#### 1. T3 — hard emergency charge-follow

**修改**: `CoveragePlanner` 硬性跟桩评分 + override

| 基准 | Overall | Map4 | Map7 | Complete | Battery | Collision | 
|------|---------|------|------|----------|---------|-----------|
| 2×4 | **740.5** | 600.8 | 880.2 | 7/8 | 0 | 1 |

**判定**: ❌ 回滚。Map4 崩盘（−192 vs baseline），新增 collision。硬 override 不可取。

---

#### 2. yjy reward retune

**修改**: 吸收 yjy 分支 charger arrival / fresh-path / no-clean revisit 奖励

| 基准 | Overall | Map4 | Map7 | Complete | Battery | Collision |
|------|---------|------|------|----------|---------|-----------|
| 2×4 | **825.1** | 767.0 | 883.2 | 8/8 | 0 | 0 |

**判定**: ❌ 回滚。安全但无净收益（−13.8 vs baseline），Map4 反而退步。

---

#### 3. frontier energy-gating

**修改**: `CoveragePlanner` 加入 charger path distance BFS + 电量门控

| 基准 | Overall | Map4 | Map7 | Complete | Battery | Collision |
|------|---------|------|------|----------|---------|-----------|
| 2×4 | **799.4** | 696.2 | 902.5 | 7/8 | 0 | 1 |

**判定**: ❌ 回滚。Map7 +18（好），但 Map4 −97 + new collision。高度地图敏感。

---

#### 4. FE-lite v1 & v2

**修改**: CoveragePlanner 反抖 tie-breaker（v1 宽激活 / v2 窄激活）

| 版本 | Overall | Map4 | Map7 | Complete | Battery | Collision |
|------|---------|------|------|----------|---------|-----------|
| v1 (approx) | **~757.4** | ~818.0 | ~696.8 | 7/8 | 0 | 1 |
| v2 | **717.1** | 528.8 | 905.5 | 7/8 | 0 | 1 |

**判定**: ❌ 双版本均回滚。v1 Map7 崩，v2 Map4 崩且 Map7 正收益。高度地图敏感。

> ⚠️ **证据缺口**: v1 的 HOLDOUT JSON 被 v2 覆盖，原始数据仅存于 `code/eval_logs/20260426-114033-*` 的 worker 目录中，`holdout_detail_logs` 中也只剩 `schema.json`。v1 数据来源于当时的运行时内存记录。

---

#### 5. T9 — no-clean revisit penalty（非饥饿型去重罚分）

**修改**: reward 中惩罚非脏格重扫（只罚非饥饿/非探索路径上的无意义振荡）

| 基准 | Overall | Map4 | Map7 | Complete | Battery | Collision |
|------|---------|------|------|----------|---------|-----------|
| 2×4 (20m) | **859.1** | 821.5 | 896.8 | 8/8 | 0 | 0 |
| 2×8 (20m) | **776.9** | 682.2 | 871.6 | 15/16 | 0 | 1 |
| 2×8 (50m) | **727.9** | 614.4 | 841.4 | 13/16 | 0 | 3 |

**判定**: ❌ 回滚。2×4 假阳性（+20.2），2×8 短期微跌（−26.7），50m 崩盘（collision×3）。无实际净收益。

📦 **手动保存**: `code/manual_checkpoints/T9_noclean_revisit_20min_20260426-124030_*`

---

#### 6. T9 — BC_COEF_MIN 调整

**修改**: BC 正则化最小系数

| 基准 | Overall | Map4 | Map7 | Complete | Battery | Collision |
|------|---------|------|------|----------|---------|-----------|
| BC0.32 2×4 (20m) | **834.1** | 770.2 | 898.0 | 8/8 | 0 | 0 |
| BC0.30 2×4 (20m) | **846.9** | 828.2 | 865.5 | 8/8 | 0 | 0 |
| BC0.30 2×8 (20m) | **673.4** | 647.1 | 699.6 | 12/16 | 2 | 2 |
| BC0.30 2×8 (50m) | **711.6** | 723.6 | 699.6 | 14/16 | 1 | 1 |

**判定**: ❌ BC0.32 安全但 Map4 退步；BC0.30 2×4 通过但 2×8 彻底崩盘。

📦 **手动保存**: `code/manual_checkpoints/T9_BC030_20min_20260426-153858_*`

---

#### 7. hard-charge one-hot planner prior

**修改**: 只当 residual alpha 已为 0 时做 charge 模式显式安全检查

| 基准 | Overall | Map4 | Map7 | Complete | Battery | Collision |
|------|---------|------|------|----------|---------|-----------|
| 2×4 | **814.9** | 783.2 | 846.5 | 8/8 | 0 | 0 |

**判定**: ❌ 回滚。安全但无增益，作得太窄。

---

#### 8. eval alpha alignment

**修改**: `Agent.exploit()` 使用 `RESIDUAL_ALPHA_WARMUP_TARGET` (0.18) 替代 `MAX` (0.45)

| 基准 | Overall | Map4 | Map7 | Complete | Battery | Collision |
|------|---------|------|------|----------|---------|-----------|
| 2×4 | **738.8** | 745.8 | 731.8 | 6/8 | 2 | 0 |

**判定**: ❌ 回滚。简单缩小 eval alpha 直接炸 Map7 battery tail（battery×2）。

> ⚠️ **代码回滚不完整**: 当前 worktree 中 `agent.py` 仍保留此修改（`RESIDUAL_ALPHA_MAX` → `WARMUP_TARGET`），需在新阶段开始前确认是否回退。

---

#### 9. planner target contract — ⭐ 关键实验

**修改**: `CoveragePlanner._goal_passes_coverage_contract()` + 强化 stale-goal 复用契约

| 基准 | Overall | Map4 | Map7 | Complete | Battery | Collision |
|------|---------|------|------|----------|---------|-----------|
| 2×4 | **846.6** | 835.5 | 857.8 | 8/8 | 0 | 0 |
| 2×8 | **801.8** | 820.5 | 783.0 | 15/16 | 1 | 0 |

**判定**: ⚠️ **重要发现但未通过 gate**。
- **Map4 +106.3 vs baseline** — 首次强证据表明 stale target commitment 是 Map4 核心根因。
- Map7 −109.9 + 1 battery death — excursion / authority 边界问题暴露。
- 整体分数 −1.8 vs baseline，但失败模式从电池死变为电池死+Map7扫不干净。

**状态**: 代码未提交（当前 worktree dirty），作为下一阶段最重要的代码基线。

---

#### 10. active-goal A* return-cost refinement

**修改**: 在 active-goal 复用决策中引入真实 A* 返航代价

| 基准 | Overall | Map4 | Map7 | Complete | Battery | Collision |
|------|---------|------|------|----------|---------|-----------|
| 2×4 | **630.9** | 422.0 | 839.8 | 8/8 | 0 | 0 |

**判定**: ❌ 回滚。A* 代价过激导致过度返航（avg_charge_count 119.38，Map4 212），coverage 崩坏。但证实了 contract 语义方向正确时机不对。

---

## 二、保留的 Checkpoints

| 名称 | 路径 | 条件 | 说明 |
|------|------|------|------|
| T9_noclean_revisit_20min | `code/manual_checkpoints/T9_noclean_revisit_20min_20260426-124030_*` | 20min 训练 | 2×4 假阳性但 reward tweak 方向可参考 |
| T9_BC030_20min | `code/manual_checkpoints/T9_BC030_20min_20260426-153858_*` | 20min 训练 | BC0.30 在 2×4 安全，但 2×8 崩 |

---

## 三、Git 备份记录

共 2 次 benchmark 相关的正式 commit（均在 `win_YJY` 分支）：

| Commit | 日期 | 说明 |
|--------|------|------|
| `c7689a9` | 2026-04-26 | Improve holdout stability with stale revisit penalty |
| `3458f7d` | 2026-04-26 | Stabilize BC0.30 holdout backup (HEAD) |

**当前 worktree 状态**: dirty
- `algorithm.py` — 包含 `_goal_passes_coverage_contract()` 目标契约改动 (target contract)
- `agent.py` — 仍包含 eval alpha alignment 改动 (应回滚但未回滚)
- `CHANGELOG.md` — 包含未提交的追加记录
- 大量 `.sisyphus/evidence/`、`eval_logs/`、`holdout_shards/` 等 untracked 证据文件

---

## 四、系统性诊断结论

经过 10+ 轮候选实验 + 远端 yjy 分支比较 + 多次 delegate agent 讨论，核心结论如下：

### 4.1 已证实的事实

1. **Stale target commitment 是 Map4 核心根因之一**
   - 证据：target contract 实验 Map4 从 714.2 → 820.5 (+106)，且 Map4 battery/collision 清零。
   - 机制：CoveragePlanner 在脏格被扫清后仍复用旧目标，浪费步数和电量。

2. **前期规则主导、后期 RL 影响渐增**
   - 训练前期 reward/planner 规则完全控制行为。
   - 随着训练深入，RL policy 接管度上升，planner safety contract 的软性不足被放大。

3. **局部 heuristic / reward / BC / alpha patch 已收益递减**
   - 10 轮中没有任何一个 patch 突破 2×8 gate。
   - 每次 patch 获得的局部收益都被另一个维度的退化抵消。

4. **Map4 vs Map7 对不同改动的敏感性差异巨大**
   - frontier-energy：Map7+18 / Map4−97
   - FE-lite v1：Map4+25 / Map7−188
   - FE-lite v2：Map7+13 / Map4−185
   - 原因：map4 (8×8) 的探索空间更小，planner 错误决策代价更高。

### 4.2 根因总结

- **Planner safety contract 太软** — `_goal_is_still_valid` 过于宽松，缺乏电量预算、时间预算、进度监控等不变量。
- **Excursion authority 边界不清** — coverage 模式下 planner 可以无限远+无限久地让 agent 探索，没有强制返航合约。
- **Coverage→Charge 切换缺乏结构性边界** — 当前靠 residual alpha 混合 policy prob + planner 评分，但 alpha 只控制"混合比例"不控制"条件约束"。

### 4.3 下一阶段方向

基于证据积累，建议按优先级：

1. **实现 planner option contract（最高优先）**
   - 给 coverage goal 加上时间预算 (`COVERAGE_GOAL_MAX_COMMIT_STEPS`) 和进度监控 (`COVERAGE_GOAL_STALL_STEPS`)
   - 强制返航条件：电量 < 去程 + 返程估计 + 余量时中断 coverage 目标
   - 切入点：在已有 `_goal_passes_coverage_contract()` 基础上扩展

2. **Reward 结构性改动（次优先）**
   - 不是继续调 scalar weight，而是改 reward 的信号结构
   - 例如：引入 coverage progress delta 作为正向激励，而非只用全局 cleaning score

3. **Model 侧改动（如 1+2 仍不力）**
   - 考虑在 feature 中增加 planner disagreement signal 作为输入
   - 或引入 subgoal-conditioned policy

4. **超参系统性搜索（最后手段）**
   - 仅在结构性改动稳定后，用 grid search 微调

---

## 五、证据文件清单

### HOLDOUT benchmark artifacts (30 文件)

所有位于 `train/context/HOLDOUT_*.json`。

14 个为 dryrun/scaffold（无 episode 评分），16 个有实际执行结果：

| 文件 | Timestamp | Overall | Map4 | Map7 | C | B | Coll |
|------|-----------|---------|------|------|---|---|------|
| DYNAMIC_2X4 | 002507 | 838.9 | 793.2 | 884.5 | 8 | 0 | 0 |
| DYNAMIC_2X4_REAL | 000653 | 833.1 | 773.8 | 892.5 | 7 | 1 | 0 |
| DYNAMIC_2X8 | 003549 | 803.6 | 714.2 | 892.9 | 15 | 1 | 0 |
| SHARDED_BASELINE | 163028 | 652.4 | 605.8 | 699.0 | 13 | 6 | 1 |
| T3_EMERGENCY_2X4 | 021809 | 740.5 | 600.8 | 880.2 | 7 | 0 | 1 |
| YJY_REWARD_2X4 | 025501 | 825.1 | 767.0 | 883.2 | 8 | 0 | 0 |
| FRONTIER_ENERGY_2X4 | 111223 | 799.4 | 696.2 | 902.5 | 7 | 0 | 1 |
| FE_LITE_2X4 | 115858 | 717.1 | 528.8 | 905.5 | 7 | 0 | 1 |
| T9_NOCLEAN_REVISIT_20MIN_2X4 | 124030 | 859.1 | 821.5 | 896.8 | 8 | 0 | 0 |
| T9_NOCLEAN_REVISIT_20MIN_2X8 | 135739 | 776.9 | 682.2 | 871.6 | 15 | 0 | 1 |
| T9_NOCLEAN_REVISIT_50MIN_2X8 | 134444 | 727.9 | 614.4 | 841.4 | 13 | 0 | 3 |
| T9_BC_20MIN_2X4 | 150733 | 834.1 | 770.2 | 898.0 | 8 | 0 | 0 |
| T9_BC_030_20MIN_2X4 | 154732 | 846.9 | 828.2 | 865.5 | 8 | 0 | 0 |
| T9_BC_030_20MIN_2X8 | 164039 | 673.4 | 647.1 | 699.6 | 12 | 2 | 2 |
| T9_BC_030_50MIN_2X8 | 162802 | 711.6 | 723.6 | 699.6 | 14 | 1 | 1 |
| CHARGE_CONTRACT_2X4 | 195528 | 814.9 | 783.2 | 846.5 | 8 | 0 | 0 |
| EVAL_ALPHA_ALIGN_2X4 | 200951 | 738.8 | 745.8 | 731.8 | 6 | 2 | 0 |
| TARGET_CONTRACT_2X4 | 203255 | 846.6 | 835.5 | 857.8 | 8 | 0 | 0 |
| TARGET_CONTRACT_2X8 | 203908 | 801.8 | 820.5 | 783.0 | 15 | 1 | 0 |
| TARGET_CONTRACT_ASTAR_2X4 | 210740 | 630.9 | 422.0 | 839.8 | 8 | 0 | 0 |

> C=Complete count(out of 8 for 2×4, 16 for 2×8), B=Battery deaths, Coll=Collision deaths

### 其他证据

- `train/context/CHANGELOG.md` — 107 行完整时间线
- `.sisyphus/notepads/benchmark-900-optimization/` — learnings.md, issues.md, decisions.md, problems.md
- `.sisyphus/evidence/` — 110 文件（task 执行证据、QA 验证、dry-run 快照）
- `code/eval_logs/` — 207 目录（每次 benchmark 的 worker 级别执行日志）
- `code/holdout_shards/` — 分片分配 metadata
- `train/holdout_detail_logs/` — 各实验的聚合结果 schema

---

## 六、完整性声明

| 维度 | 状态 |
|------|------|
| 16 个有效 benchmark 实验 JSON | ✅ 全部保存于 `train/context/HOLDOUT_*.json` |
| 107 行 CHANGELOG | ✅ 持续记录 |
| 10+ 次回滚 | ✅ 代码状态可追溯（git diff 确认） |
| FE-lite v1 独立 JSON | ⚠️ 被 v2 覆盖，episode 数据在 `code/eval_logs/` 可恢复 |
| target contract 代码 | ⚠️ 当前 dirty，未 commit |
| eval alpha alignment 代码 | ⚠️ 应回滚但仍在 worktree 中 |
| 手动保存 checkpoint×2 | ✅ 在 `code/manual_checkpoints/` |
| Git 备份×2 | ✅ `c7689a9`, `3458f7d` |

---

*报告生成: 2026-04-26 | 版本: v1.0 | 作者: Sisyphus agent orchestration*
