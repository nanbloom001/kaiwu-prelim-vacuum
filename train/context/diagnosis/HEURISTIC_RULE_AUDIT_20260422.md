# Heuristic Rule Audit 2026-04-22

## Summary

目标：围绕 `origin/yjy` 深入盘点“有效启发式规则”，并扩展到 `linux`、`origin/win`、当前分支 `linux-LTSPPO-control-stack-simplify`，为后续构建“强启发式主导、弱学习补细节”的新结构提供规则底座。

本次审计的核心结论是：

1. `origin/yjy` 的优势不是“没有启发式”，而是**启发式非常直接**：
   - 直接把 charger / evade / coverage 优先级写进 feature
   - 直接把 low-battery、NPC、revisit、stuck、terminal penalty 写进 reward
   - 没有额外的复杂 expert / teacher / curriculum 控制层
2. `origin/win` 和 `linux` 的优势是**存在明确的 direct expert control**：
   - `NPC safety filter`
   - `return_mode` 状态机
   - `A* charger planning`
   - `hard override` / `soft logit bias`
3. 当前分支不是“启发式不够”，而是**启发式太多、太系统化、太间接**：
   - 大量规则被转译成 `mode / route_anchor / target / teacher / curriculum / compare`
   - 行为解释链路过长，稳定性和可归因性都明显变差
4. 如果要回到“强启发式结构”，最值得保留的是：
   - `NPC safety filter`
   - `A* charger planner + blocked-cell memory`
   - `extreme-emergency fallback`
   - 较薄的 `pre-return / return` 两阶段边界
   - 有限的 `mode/target/return-action` teacher
5. 不建议继续保留为主骨架的，是当前分支那种：
   - 过重的 `contract/return/readiness` 多阈值链
   - 过细的 reward terrain
   - 过多的 route-phase / curriculum / compare 解释性联动

## Scope

本次盘点聚焦以下分支和文件：

- `origin/yjy`
- `origin/win`
- `linux`
- `linux-LTSPPO-control-stack-simplify`

核心文件：

- `code/agent_ppo/feature/preprocessor.py`
- `code/agent_ppo/feature/expert.py`
- `code/agent_ppo/conf/conf.py`
- `code/agent_ppo/workflow/train_workflow.py`
- `code/agent_ppo/algorithm/algorithm.py`

## Branch-Level Characterization

### `origin/yjy`

特点：

- 没有独立 `expert.py`
- 没有复杂 curriculum / teacher / compare / checkpoint 评分控制
- 核心都在 `preprocessor.py`

方法论：

> 用强 reward shaping 和强动作优先级特征，直接把“该充电 / 该避 NPC / 该覆盖新区域”写进训练信号。

### `origin/win`

特点：

- 有独立 `expert.py`
- 有 `hard override` 与 `soft logit bias`
- 有 A* charger path、NPC 安全过滤、return-mode 状态机

方法论：

> 由 direct expert 规则先定义关键安全和充电边界，再让模型在边界内学习。

### `linux`

特点：

- 基本继承 `win` 的 direct expert 主体
- 同时加入更多 reward shaping、expert annealing 和动态 curriculum

方法论：

> direct expert control 仍在，但 reward / curriculum 比 `win` 更重。

### 当前分支 `linux-LTSPPO-control-stack-simplify`

特点：

- 常规 charging override / bias 已收缩
- 仍保留 `NPC filter`、A* charger planner、blocked-cell memory、极端低电 fallback
- 主启发式转为：
  - `mode / route_anchor / target / return_action` teacher
  - battery/collision risk label
  - 复杂的 return/contract/readiness 结构
  - 更细碎的 reward terrain

方法论：

> 启发式没有减少，而是被系统化、层级化、间接化了。

## Effective Heuristic Rules By Category

### 1. Charger / Return / Battery

#### `origin/yjy`

有效规则：

- `action_priority_feature` 里显式计算：
  - `charger_weight`
  - `evade_weight`
  - `coverage_weight`
- 电量低时，charger 方向分数直接抬高
- reward 中直接有：
  - low-battery charger approach reward
  - charger distance progress reward
  - 充电成功 bonus
  - battery death terminal penalty

评价：

- 这是**直接启发式主导**
- 没有显式状态机，但行为边界通过 feature + reward 直接压出来

#### `origin/win`

有效规则：

- `return_mode` 状态机
- `LOW_BATTERY_RATIO` / `EXIT_RETURN_RATIO`
- charger A* path with caching
- hard override in eval mode
- soft logit bias in train mode
- greedy-to-charger fallback

评价：

- 这是**最典型的 direct heuristic control**

#### `linux`

有效规则：

- 延续 `win` 的 `return_mode + A* + override + bias`
- 加上：
  - `charger_proximity_reward`
  - `charger_path_explore`
  - `A* potential shaping`
  - 更强的 expert bias / anneal

评价：

- 仍是 direct heuristic 主导
- 但开始出现“规则 + reward + curriculum”耦合加重

#### 当前分支

有效规则：

- `evaluate_simplified_return_readiness()`
- charger candidate ranking
- route-family planner
- `get_charger_signal()`
- `get_teacher_guidance()`
- extreme-emergency fallback
- current `expert.get_logit_bias()` 只保留 NPC avoid bias

评价：

- 常规 return control 已经从 direct override/bias 退化为：
  - planning signal
  - teacher target
  - readiness chain
- **只有 extreme emergency fallback 还算 direct heuristic**

### 2. NPC Avoidance / Safety

#### `origin/yjy`

有效规则：

- NPC 距离进入 feature
- NPC 距离越近惩罚越强
- 远离 NPC movement 获得正反馈
- 极近 NPC 时额外重罚

评价：

- 这是**直接 reward-level heuristic**

#### `origin/win` / `linux`

有效规则：

- `filter_actions()` / NPC safety filter
- NPC danger field in cost map
- logit bias 对“朝 NPC 方向移动”直接减分

评价：

- 这是**direct action-space control**
- 也是跨分支最稳定、最值得保留的一类规则

#### 当前分支

有效规则：

- `NPC safety filter` 还保留
- planner cost map 仍包含 NPC danger
- `get_logit_bias()` 只保留 NPC avoidance bias

评价：

- 当前分支里最清晰、最没被污染的 direct heuristic 之一

### 3. Coverage / Revisit / Path Efficiency

#### `origin/yjy`

有效规则：

- `coverage_rate`
- `repeat_visit_ratio`
- `visit_count == 1` 奖励
- 高 revisit 次数额外惩罚
- `unique_visit_ratio` 奖励
- `directional_coverage_scores`

评价：

- 这类规则非常直接
- 是 `yjy` 里最像“高 CPS 导向”的部分

#### `origin/win` / `linux`

有效规则：

- `frontier_reward`
- revisit penalty 三段式
- local cleaned density 处罚
- evade/charge 模式下 revisit penalty 抑制
- `_cps_ema` 派生 efficiency reward

评价：

- 这已经比 `yjy` 更复杂
- 仍然能解释，但开始出现“代理量太多”的苗头

#### 当前分支

有效规则：

- `coverage_efficiency_20`
- `path_cross_count_50`
- `coverage_tangle_penalty`
- `clean_floor_revisit_penalty`
- `effective_coverage_bonus`

评价：

- 这类规则现在过于分散
- 启发式本身没有完全失效，但解释链已经过长

### 4. Anti-Stuck

#### `origin/yjy`

有效规则：

- `recent_positions` history
- `_is_stuck()`
- stuck penalty
- 离开 stuck 状态时小奖励

评价：

- 直接、清晰、容易保留

#### `origin/win` / `linux`

有效规则：

- `stuck_steps`
- `last_move_invalid`
- `stuck_penalty`

评价：

- 同样属于高解释性启发式

#### 当前分支

有效规则：

- 仍有 stuck / invalid move tracking
- 但已被覆盖在更大的一套路径、coverage、teacher 规则里

评价：

- 可以保留，但应被重新收缩成简单局部规则

### 5. Expert Override / Teacher / Curriculum

#### `origin/yjy`

- 基本没有独立 expert
- 没有 teacher mask
- 没有复杂 curriculum gate

评价：

- 这是它“解释性强、链路短”的重要原因

#### `origin/win`

有效规则：

- hard override
- soft logit bias
- charger A* fallback

评价：

- 这是典型的 direct expert system

#### `linux`

有效规则：

- direct expert system 仍在
- 再叠加 curriculum 阶段推进
- expert anneal

评价：

- 开始进入“direct heuristic + training control”混合形态

#### 当前分支

有效规则：

- `mode_teacher`
- `target_teacher`
- `route_anchor_teacher`
- `return_action_teacher`
- `route_phase teacher`
- curriculum-lite / phase / stagnation / compare / checkpoint score

评价：

- 这部分已经不是“简单启发式”
- 而是**训练控制系统**
- 可观察性强，但解释性、可归因性和鲁棒性都显著下降

## Direct Heuristic Control vs Weak Proxy

### Direct heuristic control

这些规则直接约束动作或强制边界，最有资格进入“强启发式结构”：

- `NPC safety filter`
- `hard override`
- `soft logit bias`
- `return_mode` / hysteresis
- `A* charger planner`
- `blocked-cell memory`
- `greedy-to-charger fallback`
- 当前分支里的 `extreme-emergency fallback`

### Weak proxy

这些规则更像训练代理，不适合作为主骨架：

- `coverage_tangle_penalty`
- `frontier_reward`
- `_cps_ema`
- `clean_floor_revisit_penalty`
- `effective_coverage_bonus`
- `charger_path_explore`
- `charger_proximity_reward`
- 大部分 curriculum / stagnation / compare 指标

### Intermediate / hybrid

这些规则介于两者之间，适合保留但要瘦身：

- `mode_teacher`
- `target_teacher`
- `route_anchor_teacher`
- `return_action_teacher`
- 较薄的 `pre-return / return` 双阶段边界

## Most Valuable Rules To Preserve

### Tier 1: 必保留

这些是跨分支重复出现、解释性强、对题最直接的规则：

1. `NPC safety filter`
2. `A* charger planner`
3. `blocked-cell / dangerous-cell memory`
4. `extreme-emergency charger fallback`
5. 简单 hysteresis：
   - 低电进入 return
   - 充电后高电退出 return

### Tier 2: 建议保留但收缩

1. `pre-return / return` 两阶段，而不是单一 low-battery switch
2. `mode_teacher + target/route_anchor_teacher + return_action_teacher`
3. revisit / anti-stuck 的简洁局部惩罚

### Tier 3: 只保留为辅助项

1. coverage / frontier / cps proxy reward
2. charger proximity / charger path exploration reward
3. 当前分支里复杂的 readiness / route-phase / return-stall 解释链

## Candidate Strong-Heuristic Structure

建议的新结构不该继续沿用当前“重控制栈”，而应该回到：

> direct heuristic boundary + thin learning layer

### Suggested shape

1. **Direct safety layer**
   - `NPC safety filter`
   - illegal move filter
   - blocked-cell memory

2. **Direct charger control layer**
   - A* to charger
   - low-battery hysteresis
   - emergency fallback
   - charger target selection

3. **Thin phase layer**
   - 只保留：
     - `clean / pre-return / return / evade`
   - 不再保留当前这么重的：
     - `depart / expand / harvest / contract / return / route-phase / readiness` 多层转译

4. **Thin teacher layer**
   - 只保留最硬的：
     - mode
     - target
     - return action
   - 去掉多余 reliability / route-phase 细枝末节

5. **Auxiliary reward layer**
   - 保留少数局部项：
     - clean
     - revisit
     - anti-stuck
     - low-battery charger progress
     - NPC avoidance
   - 去掉大量细碎 terrain

## What To Cut From Current Branch

当前分支中最应该砍掉的是：

1. 过重的 `contract / return / readiness` 多阈值链
2. 过细的 route-phase / teacher mask 细分
3. 大量“为了可解释而可解释”的观察链直接耦合到行为控制
4. 过于碎片化的 charging / coverage reward terrain
5. 依赖 `compare / stagnation / curriculum-lite` 来间接解释行为对错的做法

## Final Judgment

如果目标是回到高分、强稳定、强可解释的路线，那么最值得借鉴的不是当前分支，而是：

- `yjy` 的直接启发式 reward + action-priority
- `win/linux` 的 direct expert control

最不值得继续扩张的，是当前分支这类：

> “启发式很多，但都经过 mode / teacher / curriculum / compare 层层转译”的系统。

更直接的说法是：

> 当前真正需要的，不是“更多 heuristics”，而是“更短、更硬、更直接的 heuristics”。

## Independent Review

本报告额外合并了一条独立 `gpt-5.4 high` 调研结论，结论与主线代码审计一致：

- `origin/yjy` 最有效的规则集中在：
  - low-battery charging shaping
  - NPC 距离惩罚 / 远离奖励
  - coverage / revisit
  - anti-stuck
  - `action_priority` 的 `charge > evade > coverage`
- `origin/win` / `linux` 最有效的是：
  - `NPC safety filter`
  - `return_mode`
  - A* 到 charger
  - `hard override` / `soft logit bias`
- 当前分支最值得保留的是：
  - `NPC safety filter`
  - A* charger planner
  - `extreme-emergency fallback`
  - 较薄的 `mode/target/return-action teacher`
- 不建议继续把：
  - `coverage/revisit/CPS` 的细碎 reward
  - 以及复杂 curriculum / compare / route-phase 解释链
 作为“强启发式骨架”的主体

