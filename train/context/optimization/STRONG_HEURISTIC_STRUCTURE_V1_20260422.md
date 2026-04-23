# Strong Heuristic Structure v1.2 2026-04-22

## Summary

本方案目标不是继续在当前 `linux-LTSPPO-control-stack-simplify` 分支的重控制栈内做小步调参，而是明确转向：

> **强启发式定边界，PPO 学局部细节。**

新结构的核心判断是：

1. 当前比赛中，`yjy` / `win` / `linux` 这几条更成功的路线，都依赖较强的 direct heuristics。
2. 当前分支不是“启发式不够”，而是“启发式太多、太系统化、太间接”，导致行为链路失去解释性，稳定性也持续下降。
3. 下一轮不再追求通过 `contract / readiness / route-phase / teacher reliability / curriculum stagnation` 这类复杂机制去间接塑形，而是直接恢复：
   - `NPC safety filter`
   - `A* charger planner`
   - `soft logit bias`
   - `extreme-emergency hard override`
   - 简化后的 `clean / pre-return / return / evade` 四态结构
4. 这不是“一轮删除所有 teacher/aux”的方案，而是一次**分层回退**：
   - 先保留 learner schema、风险辅助项和少量关键 teacher
   - 只砍最重、最难解释的 route-phase / reliability / curriculum 反向控制链

本文件是实施方案，不是代码审计。规则来源与分支盘点见：

- [HEURISTIC_RULE_AUDIT_20260422.md](/home/user/TcKaiwuFinal/train/context/diagnosis/HEURISTIC_RULE_AUDIT_20260422.md:1)

## Design Principles

### 1. 保留 direct heuristics，砍掉复杂转译层

保留：

- `NPC safety filter`
- `A* charger planner`
- `blocked-cell / dangerous-cell memory`
- `soft logit bias`
- `extreme-emergency hard override`

砍掉或停用：

- 复杂 `contract / readiness` 多阈值训练控制链
- `route_anchor / target / route_phase` 的训练期 teacher 体系
- 细碎 reward terrain
- 课程停滞观察器对行为设计的反向约束

明确**不整包删除**的项：

- planner 的核心目标选择信号
  - `known_path_count`
  - `unknown_path_ratio`
  - `planner_multi_route_recoverability`
  - `target_gap`
  - `selected_target_rank`
- battery / collision auxiliary 风险头
- 样本 schema 中已有的 teacher / aux 字段

### 2. 行为边界由启发式定死，PPO 只学局部选择

PPO 不再负责：

- 什么时候该回充
- 是否应该强行继续探索
- 极端低电量下是否还能赌一段路径

PPO 只负责：

- 在当前允许的方向中做局部选择
- 在 `clean` 阶段做更高效 coverage
- 在 `pre-return / return` 阶段做更平滑的局部移动

### 3. 只保留少数可以直接解释的奖励项

保留 reward 主项：

- `cleaning`
- `time_penalty`
- `revisit_penalty`
- `anti_stuck_penalty`
- `charger_progress_reward`
- `charge_success_bonus`
- `npc_penalty`
- `terminal_penalty`

其他复杂项全部降级或关闭。

## Target Architecture

### A. 四态行为结构

新 phase 下，行为结构统一为：

- `CLEAN`
- `PRE_RETURN`
- `RETURN`
- `EVADE`

不再保留：

- `DEPART`
- `EXPAND`
- `HARVEST`
- `CONTRACT`
- route-phase 子状态

状态触发固定为：

#### `EVADE`

满足任一条件：

- `nearest_npc_dist <= 4`
- `NPC safety filter` 判定大部分合法动作危险

#### `RETURN`

满足任一条件：

- `battery_ratio <= 0.32`
- `charger_slack <= 0`
- `battery <= charger_dist + margin`

退出条件固定为：

- `on_charger == true` 且 `battery_ratio >= 0.85`

额外 hysteresis：

- 一旦进入 `RETURN`，除非满足退出条件，否则不回退到 `PRE_RETURN / CLEAN`
- `on_charger == true` 但 `battery_ratio < 0.85` 时仍保持 `RETURN`

#### `PRE_RETURN`

满足任一条件，且尚未触发 `RETURN`：

- `battery_ratio <= 0.45`
- `charger_slack <= 6`
- `planner_multi_route_recoverability <= 0.35`
- `known_path_count < min(total_charger, 2)` 且 `unknown_path_ratio >= 0.20`
- `route_contract_pressure >= 0.50`
- `margin <= charge_margin_warn`

#### `CLEAN`

其余所有情况。

### A.1 实现映射：四态逻辑，六态头兼容

第一轮**不修改 `MODE_NUM`**，不改 learner 头维度。

实现映射固定为：

- 逻辑 `CLEAN` -> 训练标签 `MODE_EXPAND`
- 逻辑 `PRE_RETURN` -> 训练标签 `MODE_CONTRACT`
- 逻辑 `RETURN` -> 训练标签 `MODE_RETURN`
- 逻辑 `EVADE` -> 训练标签 `MODE_EVADE`

以下旧 mode 在新 phase 下不再参与主逻辑分支：

- `MODE_DEPART`
- `MODE_HARVEST`

处理方式固定为：

- `_infer_mode()` 在新 phase 下只返回上述 4 个映射后的 mode id
- mode one-hot 仍保持 6 维
- teacher / monitor / sample schema 继续使用现有 6 维结构
- 四态到六态的映射必须抽成**单独 helper**
  - `preprocessor` 的 mode 判定
  - teacher 标签生成
  - 相关 monitor/diagnostics
  必须共用同一份映射逻辑

这样可以避免：

- 改 `MODE_NUM`
- 改模型头数量
- 改 sample schema

### B. Direct safety layer

#### NPC safety filter

永远启用，优先级最高。

作用：

- 过滤明显朝 NPC 靠近的危险动作
- 在极近距离时直接压制危险方向

来源参考：

- `win/linux/current` 中的 `filter_actions()` / NPC direction penalty

#### Illegal move filter

永远启用，不再在 reward 中补救非法移动。

#### Blocked-cell / dangerous-cell memory

保留于 planner cost map 中，用于 charger routing。

### C. Direct charger layer

#### A* charger planner

保留：

- cached A* path
- blocked-cell memory
- danger-weighted cost map
- charger candidate ranking
- 目标选择所需的关键 planner 信号

停用：

- training-time `target_reliable / anchor_reliable / mode_reliable / return_action_reliable` 这套 reliability mask 链
- route-anchor 作为训练目标的额外解释层

planner 输出统一保留为：

- `charger_target`
- `charger_path`
- `charger_dist`
- `charger_margin`
- `slack`
- `suggested_action`
- `known_path_count`
- `unknown_path_ratio`
- `planner_multi_route_recoverability`
- `target_gap`
- `selected_target_rank`

说明：

> 本轮只砍“训练控制复杂度”，不砍“多 charger 目标选择能力”。

#### Soft logit bias

恢复为训练主启发式之一。

生效条件：

- 当前状态为 `PRE_RETURN` 或 `RETURN`
- NPC safety filter 未进入极端规避

bias 强度固定两档：

- `PRE_RETURN`：中等强度
- `RETURN`：高强度

不再保留当前复杂的 teacher mask / route-phase reliability 链来解释 bias 是否启用。

### C.1 Soft logit bias 的真实接入点

这一步不是只改 `expert.py`，必须同时改：

- `code/agent_ppo/agent.py`

当前训练态动作主路径只用了：

- `filter_actions()`
- `get_emergency_fallback()`

而没有使用 `get_logit_bias()`。

因此新 phase 的接入要求固定为：

1. 在训练态 `predict()` 中，读取 `expert.get_logit_bias(...)`
2. 只在 `s1_survival_strong_heuristic_v1` 下启用
3. 在 model logits 上加 bias 后，再进入采样
4. `NPC safety filter` 仍先于 bias 生效
5. emergency fallback 优先级高于 soft bias

同时要求：

- bias 强度不要硬编码成全局常量
- 先通过新 phase env 设置两档 scale：
  - `PRE_RETURN_BIAS_SCALE`
  - `RETURN_BIAS_SCALE`

### C.2 Soft bias 监控要求

新 phase 必须把 bias 实际生效情况纳入验收，而不是只验证代码路径存在。

优先复用当前仓库已有的 `expert_weight` 统计链路，并在 monitor / recent metrics 中显式关注：

- `avg_expert_weight`
- `expert_weight_nonzero_rate`
- `pre_return_bias_active_rate`
- `return_bias_active_rate`

要求：

- `PRE_RETURN` 场景下 bias 不能长期近似 0
- `RETURN` 场景下 bias active rate 必须显著高于 `PRE_RETURN`
- 若训练通过但上述监控长期接近 0，则视为“设计未真正接入动作主链”

#### Hard override

仅在以下极端条件生效：

- `charger_slack < 0`
- 或 `battery_ratio < 0.20`
- 且当前不在 charger 上

这是唯一允许保留的 runtime hard override。

### D. 动作优先级特征

重新引入 `yjy` 风格的 8 维 `action_priority_feature`。

组成：

- `charger_priority`
- `evade_priority`
- `coverage_priority`
- `revisit_suppression`

各状态下的组合：

#### `RETURN`

- `charger_priority + evade_priority`

#### `PRE_RETURN`

- `0.7 * charger_priority + 0.3 * coverage_priority + evade_priority`

#### `CLEAN`

- `coverage_priority + evade_priority`

#### `EVADE`

- `evade_priority` 主导

这部分直接进入 observation，不通过 teacher 转译。

## Reward Simplification

### 保留项

#### `cleaning_reward`

主正奖励，不再做复杂 mode 调制。

#### `time_penalty`

常驻，抑制无效游走。

#### `revisit_penalty`

仅对 low-value revisit 生效：

- 已清扫
- 无新 dirt
- 无 frontier
- 非 `RETURN / EVADE`

#### `anti_stuck_penalty`

基于：

- 最近位置重复
- invalid move
- stuck state transition

#### `charger_progress_reward`

仅在 `PRE_RETURN / RETURN` 下生效。

#### `charge_success_bonus`

充电成功后的明确正奖励。

#### `npc_penalty`

保持强惩罚。

#### `terminal_penalty`

- battery fail
- collision fail
分开计。

### 关闭项

以下项在新 phase 下明确关闭：

- `effective_coverage_bonus`
- `clean_floor_revisit_penalty`
- `return_progress_shaping_bonus`
- `route_phase_return_stall_penalty`
- `contract` 专属 shaping
- 复杂 charger candidate 派生 bonus
- 过细 coverage / frontier / edge-follow 微项
- 当前 `cps_align` 两步中的额外局部塑形

### 保留为辅助负项

- `coverage_tangle_penalty`
- `skip_needed_charge_penalty`

但它们不再承载主行为引导，只作为辅助约束。

## Teacher / Auxiliary Loss Policy

### 第一轮保留 / 降权 / 关闭策略

本轮不改 sample schema，不裁剪 learner heads，只通过 **phase-aware loss multipliers** 收缩训练目标。

#### 保留不变

- `aux_battery_loss`
- `aux_collision_loss`

理由：

- 当前 survival 信号仍然稀疏
- 这些头已经并入总损失，不适合和行为重构同一轮同时删除

#### 保留但降权

- `mode_teacher`
- `return_action_teacher`

推荐策略：

- `MODE_TEACHER_WEIGHT` 降到当前默认值的 `25%`
- `RETURN_ACTION_TEACHER_WEIGHT` 降到当前默认值的 `35%`

#### 关闭

- `target_teacher`
- `route_anchor_teacher`
- `route_phase teacher`

原因：

- 这三项最依赖当前复杂 route-phase / reliability 链
- 是当前训练解释性最差的一层

### 实施要求

- `algorithm.py` 不改网络头数量
- 只加 phase-aware multiplier
- 新 phase 下将上述关闭项 multiplier 置零，将降权项乘以固定缩放系数

### 字段语义固定

新 phase 下，teacher 相关字段不删除，但语义固定为：

- `mode_teacher`
  - 使用新的四态逻辑映射后生成
- `return_action_teacher`
  - 仅在逻辑 `RETURN` 下生成
- `target_teacher`
  - 不再作为训练目标，但字段值仍直接映射为当前 `charger_target`
- `route_anchor_teacher`
  - 与 `charger_target` 保持同值占位，不再单独定义 route-anchor 语义
- `route_phase_action_teacher`
  - 统一输出默认值 / inactive mask

具体要求：

- 字段存在
- sample schema 不变
- multiplier 为 0 的项必须仍输出稳定、可序列化的默认值
- 不允许“某些 batch 有字段、某些 batch 没字段”

## Curriculum / Phase Policy

新增 phase：

- `s1_survival_strong_heuristic_v1`

该 phase：

- 继续使用 `curriculum-lite`
- 固定 `warmup`
- 固定 profile `0.20 / 0.40 / 0.40`

但课程系统角色被限制为：

- 提供固定训练分布
- 提供观测指标

明确不允许：

- curriculum 反向决定行为设计
- stagnation 规则主导 reward / mode 结构

### 接入要求

新增 phase 不是只加 `.env` 文件，需要同时改：

- `curriculum_policy.py`
  - 让新 phase 明确命中 fixed-profile + lite-stage 锁
- `preprocessor.py`
  - 新 phase 走 strong-heuristic 行为分支
- `expert.py`
  - 新 phase 启用 direct charger bias / emergency fallback / 简化 teacher
- `algorithm.py`
  - 新 phase 应用 teacher/aux multiplier

不允许仅通过 phase 名称隐式复用现有 `control_stack_simplify` / `cps_align` 逻辑。

## Files / Subsystems To Change

这轮实现主要集中在 4 个核心文件和 1 个 phase 文件：

- `code/agent_ppo/feature/preprocessor.py`
- `code/agent_ppo/feature/expert.py`
- `code/agent_ppo/agent.py`
- `code/agent_ppo/conf/conf.py`
- `code/agent_ppo/algorithm/algorithm.py`
- `train/phases/s1_survival_strong_heuristic_v1.env`

必要时附带：

- `code/agent_ppo/workflow/curriculum_policy.py`
  - 用于注册新 phase 的行为和 profile 接入
- `code/tests/test_ltsppo_contracts.py`
- `code/tests/test_curriculum_and_checkpoint_score.py`

## Validation Strategy

### 阶段 1：结构正确性

必须验证：

- 四态逻辑互斥且优先级正确
- `NPC safety filter` 永远生效
- `PRE_RETURN / RETURN` 下 soft bias 正常
- `agent.py` 训练态主路径实际使用了 `get_logit_bias()`
- emergency override 只在极端条件触发
- `target/route_anchor/route_phase teacher` 在新 phase 下失活
- `mode_teacher / return_action_teacher` 在新 phase 下仅降权，不消失
- battery/collision aux loss 在新 phase 下保持存在
- `RETURN` 在 charger 上可以稳定退出，不会自锁

### 阶段 1.1：mode 优先级测试表

必须把以下 case 写成显式单测，不允许只做泛化断言：

1. `nearest_npc_dist <= 4`
   - 即使同时满足 `RETURN` 条件，也必须进入 `EVADE`
2. `charger_slack <= 0`
   - 即使同时满足 `PRE_RETURN` 条件，也必须进入 `RETURN`
3. `on_charger == true` 且 `battery_ratio < 0.85`
   - 必须保持 `RETURN`
4. `on_charger == true` 且 `battery_ratio >= 0.85`
   - 必须退出 `RETURN`
5. `known_path_count < min(total_charger, 2)` 且 `unknown_path_ratio` 高，但 `charger_slack > 0`
   - 只能进入 `PRE_RETURN`，不能直接进 `RETURN`
6. 普通清扫场景：
   - 不满足 NPC / return / pre-return 条件时，必须进入逻辑 `CLEAN`（实现上映射到 `MODE_EXPAND`）

### 阶段 1.2：bias 监控落地要求

以下字段不能只停留在文档中，必须实际进入 rolling metrics / monitor：

- `avg_expert_weight`
- `expert_weight_nonzero_rate`
- `pre_return_bias_active_rate`
- `return_bias_active_rate`

推荐接入位置：

- `train_workflow.py` 的 recent metrics / rolling diagnostics

验收要求：

- 新 phase 训练中，这 4 个字段必须可见
- `return_bias_active_rate` 必须显著高于 `pre_return_bias_active_rate`
- 若字段不存在或长期为 0，则视为 bias 没有真正接入

### 阶段 2：训练启动

先跑 dry-run：

```bash
python3 train/run_training_phase.py s1_survival_strong_heuristic_v1 --seed-label dry --dry-run
```

然后只跑 **scratch**，不跑 resume。

### 阶段 3：训练验收

观察点：

- `bootstrap_20`
- `global_40`

主指标：

- `avg_clean_per_step`
- `battery_fail_rate`
- `zero_charge_battery_fail_rate`
- `win_rate`
- `mode_usage_return`
- `return_stall_rate`
- `late_return_rate`
- `missed_charge_opportunity_rate`

第一阶段通过线：

- `global_40`
  - `avg_clean_per_step >= 0.60`
  - `battery_fail_rate <= 0.20`
  - `zero_charge_battery_fail_rate <= 0.15`
  - `win_rate >= 0.75`

### 阶段 3.1：benchmark 前行为健康门

在跑 benchmark 之前，先检查行为指标是否至少不劣于当前 `s1_survival` 基线。

必须检查：

- `late_return_rate`
- `missed_charge_opportunity_rate`
- `return_stall_rate`
- `planner_policy_divergence_rate`

通过要求：

- `late_return_rate` 不高于当前 `s1_survival` 参考 run
- `missed_charge_opportunity_rate` 不高于当前 `s1_survival` 参考 run
- `return_stall_rate` 不高于当前 `s1_survival` 参考 run
- `planner_policy_divergence_rate` 不显著恶化

若 clean score / CPS 上升，但上述行为门显著变差，则判定为：

> 行为退化型假改善

这种情况不允许直接进入 benchmark 验收。

### 阶段 3.2：benchmark 前置条件

只有同时满足以下两类条件，才允许进入 benchmark：

1. 主结果门：
   - `global_40` 通过本文件定义的主线指标
2. 行为健康门：
   - `late_return_rate`
   - `missed_charge_opportunity_rate`
   - `return_stall_rate`
   - `planner_policy_divergence_rate`
   全部不劣化

### 阶段 4：benchmark 对照

训练窗口通过后，再跑 benchmark。

第一目标：

- benchmark `mean avg_clean_score >= 768`

这是当前仓库中我能直接验证到的 `linux` 本地 benchmark 基线。

第二目标才是：

- 向你提到的 `888+`
- 或接近满分路线

## Explicit Cuts

当前分支中以下机制默认应视为待删或待停用：

1. `contract / readiness` 多阈值链
2. `target / route_anchor / route_phase` 的复杂 teacher mask 链
3. 当前 `cps_align` 的额外局部塑形项
4. 过细的 coverage / charging terrain
5. 用 `curriculum_stagnation` / `compare` 指标反向塑造行为设计的做法

明确不直接删除：

- route-family / charger candidate ranking 所依赖的核心 planner 信号
- battery / collision auxiliary risk 头
- mode / return_action 的最薄 teacher 通道

## Assumptions

- 当前比赛阶段更适合 direct heuristics 主导，而不是继续追求更复杂的端到端学习控制。
- 新结构默认优先借鉴：
  - `yjy` 的 direct reward / action-priority
  - `win/linux` 的 direct expert control
- 这轮目标不是立刻满分，而是先证明“强启发式结构”能显著优于当前复杂控制栈。
- 当前 compare / runtime_state / resume 工具链保留，因为它们属于诊断底座，不是本轮主要裁剪对象。
