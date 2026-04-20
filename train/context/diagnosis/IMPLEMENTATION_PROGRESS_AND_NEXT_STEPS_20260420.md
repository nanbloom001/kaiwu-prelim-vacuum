# 实施进度与后续计划追踪 — 2026-04-20

> 目的：记录当前这轮训练修复的**完整规划**、**已经落实的改动**、**尚未落实的改动**、**测试环境现状**与**下一步验证顺序**，避免后续遗忘或重复判断。
> 当前主参考报告：
> - [UNIFIED_PROBLEM_DIAGNOSIS_REPORT_20260420.md](/home/user/TcKaiwuFinal/train/context/diagnosis/UNIFIED_PROBLEM_DIAGNOSIS_REPORT_20260420.md)
> - [DEEP_PROBLEM_DIAGNOSIS_REPORT_20260420.md](/home/user/TcKaiwuFinal/train/context/diagnosis/DEEP_PROBLEM_DIAGNOSIS_REPORT_20260420.md)
> - [FULL_REPO_AUDIT_FINDINGS_20260420.md](/home/user/TcKaiwuFinal/train/context/diagnosis/FULL_REPO_AUDIT_FINDINGS_20260420.md)

## 1. 当前问题总判断

当前训练主问题不是链路损坏，而是：

- 策略进入了“能继续清扫、能赢一部分局，但回充闭环学不出来”的局部最优
- `planner_policy_divergence_rate` 长期高位
- `return_stall_rate` 长期高位
- `battery_fail_rate` 偏高
- `zero_charge_battery_fail_rate` 偏高
- curriculum 因 `planner + stall` 长期高位而把系统锁在 `warmup`

当前共识性的主根因是：

1. planner 约束没有在关键 `planning / contract / return` 阶段真正落到动作层  
2. 充电/回充相关 shaping 太弱或 gate 太窄，尤其 `charger_access_probe_bonus` 近乎失效  
3. curriculum 在识别问题上基本正确，但正在放大当前坏局部最优

## 2. 原始完整规划

这轮完整规划原本分成两步：

### 第一步：止损 + 低风险高确定性修复

目标：

- 先把已经证据充分的问题直接打掉
- 避免训练继续在明显错误方向上强化

原计划包括：

1. 把 `battery fail`，尤其 `zero-charge battery fail`，明确打成强负样本
2. 让 `charger_access_probe_bonus` 真正触发
3. 增强“该充不充”的惩罚，不再让 `skip_needed_charge_penalty` 近乎失效
4. 暂停继续把 curriculum 收缩到更保守的 `anchor-heavy` 分布

### 第二步：关键阶段学习信号重构

目标：

- 修复 `planner + return/charge` 的关键学习缺口
- 让策略真正学会 return 动作模板，而不是只学会一个 return mode 标签

原计划包括：

1. 把 planner / return teacher 覆盖扩到 `planning / contract / return`
2. 重构 return 阶段的连续进展 shaping
3. 让 curriculum 的主门槛从 raw `planner_divergence` 逐步切向更贴近因果的 return/charge 健康指标
4. 增加新的验证指标，而不是只看总 `win_rate`

## 3. 已经落实的改动

以下是**已经落到代码里**的内容。

### 3.1 battery fail / zero-charge fail 止损

已落实：

- 在 [train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py) 中新增 `_compute_battery_fail_outcome(...)`
- `battery fail` 的终局处理现在区分：
  - 普通 battery fail
  - `zero-charge battery fail`
- `zero-charge battery fail` 会：
  - 额外增加 terminal cost
  - 使用更保守的 `task_reward_scale`
- episode summary 中已增加：
  - `zero_charge_battery_fail`
  - `battery_positive_reward`

影响目标：

- `battery_positive_reward_rate`
- `zero_charge_battery_fail_rate`
- `avg_charge_count_battery_fail`

状态：**已落实**

### 3.2 `charger_access_probe_bonus` gate 修正

已落实：

- 在 [preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py) 中放宽了 `probe` 的触发条件
- 不再强依赖 `all_known_paths <= 0`
- 新增了弱路径知识条件：
  - `CHARGER_ACCESS_PROBE_WEAK_ROUTE_MAX`
  - `CHARGER_ACCESS_PROBE_SLACK_CONFIDENCE_MAX`
- 允许在 `MODE_CONTRACT` 下按较低系数触发
- 新增了 `CHARGER_ACCESS_PROBE_CONTRACT_SCALE`

影响目标：

- `avg_reward_charger_access_probe_bonus`
- `reward_positive_share_charger_access_probe_bonus`
- `zero_charge_battery_fail_rate`

状态：**已落实**

### 3.3 `skip_needed_charge_penalty` 增强

已落实：

- 在 [preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py) 中扩大了 `skip_needed_charge_penalty` 的触发范围
- 不再只限于“贴近 charger 但没充”
- 对 `planning/critical` 且路径/上下文已经足够明确，但：
  - 模式仍不对
  - 或进入 `contract/return` 但没有推进
  的情况也会触发

影响目标：

- `avg_reward_skip_needed_charge_penalty`
- `reward_negative_share_skip_needed_charge_penalty`
- battery fail 中 `charge_count=0` 的占比

状态：**已落实**

### 3.4 planner / return 阶段覆盖扩展（第一阶段）

已落实：

- 在 [preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py) 中：
  - `planner_alignment_reward` 不再只在 `battery_state == safe` 且不处于 `CONTRACT/RETURN` 时生效
  - 对 `CONTRACT/RETURN` 且 `return_action_reliable` 的场景加入了更高权重的 planner 对齐信号
- 在 [expert.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/expert.py) 中：
  - 即使 `target_reliable/mode_reliable` 为假，只要 `return_action_reliable` 为真，也不再直接放弃 teacher guidance
- 在 [preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py) 中：
  - `planning/critical + CONTRACT/RETURN` 下的 `return_action_teacher_mask` 提升到了更高下限

影响目标：

- `return_action_teacher_active_rate`
- `planner_policy_divergence_rate`
- `return_stall_rate`

状态：**已落实（第一阶段覆盖扩展已做，第二阶段连续 shaping 还没做）**

### 3.5 curriculum 不再继续往更保守 warmup profile 收缩

已落实：

- 在 [curriculum_policy.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_policy.py) 中新增：
  - `DEGRADED_MAINLINE_PROFILE_WEIGHTS`
- 对 `warmup + degraded_mainline/stagnation_level>=2` 的情况：
  - 不再沿用更保守的 `CONSERVATIVE_PROFILE_WEIGHTS["warmup"]`
  - 改为更软的 `anchor=0.52 / mild=0.33 / broad=0.15`

目标：

- 避免坏局部最优在 `anchor-heavy` 分布里继续被强化

状态：**已落实**

### 3.6 新增统计指标

已落实：

- 在 [curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py) 中新增：
  - `battery_fail_count`
  - `battery_positive_reward_count`
  - `battery_positive_reward_rate`
- 同时更新了聚合逻辑，避免这些计数被错误按平均数合并

目标：

- 后续验证“坏样本有没有真的被打坏”

状态：**已落实**

### 3.7 宿主机 conda 测试环境

已落实：

- 新建环境：`/home/user/TcKaiwuFinal/.conda-numpy`
- 已安装：
  - `python 3.11`
  - `numpy`
  - `torch 2.11.0+cpu`
  - `pytest`
  - `pyyaml`
  - `toml`

默认规则：

- **今后宿主机测试默认使用**：
  - `/home/user/TcKaiwuFinal/.conda-numpy/bin/python`

状态：**已落实**

## 4. 已完成的验证

### 4.1 已通过的目标用例

以下这批是这轮改动直接相关、已经通过的目标测试：

- `test_profile_plan_for_runtime_uses_softer_weights_when_warmup_is_degraded`
- `test_apply_terminal_outcome_to_step_records_writes_collision_cost_into_last_sample`
- `test_battery_fail_outcome_adds_extra_cost_for_zero_charge_fail`
- `test_recent_episode_metrics_use_charged_only_efficiency_and_zero_charge_fail_rate`
- `test_benchmark_aggregate_overall_contains_completed_and_failure_rates`
- `test_reward_process_emits_charger_access_bonuses_when_route_knowledge_improves`
- `test_reward_process_emits_probe_bonus_when_route_knowledge_is_weak`
- `test_reward_process_penalizes_skipping_needed_charge_when_critical_even_away_from_charger`
- `test_expert_teacher_guidance_keeps_return_action_only_signal`

### 4.2 当前测试结论

可以确认：

- 第一步止损项已基本落地
- `probe` 和 `skip-needed-charge` 的新逻辑至少在契约层可触发
- `zero-charge battery fail` 的额外惩罚逻辑已落地
- warmup degraded profile 的软化逻辑已落地
- return 阶段连续 shaping 已落地：
  - `return_progress_shaping_bonus`
  - `high_need_return_stall_penalty`
- `return_progress_per_step / return_efficiency_ratio / high_need_return_stall_rate` 继续沿用现有 episode 聚合链，无需再补基础口径
- 新增的 reward contribution 口径已纳入：
  - `reward_positive_share_return_progress_shaping_bonus`
  - `reward_negative_share_high_need_return_stall_penalty`
  - `reward_charging_positive_share_return_progress_shaping_bonus`
  - `reward_charging_negative_share_high_need_return_stall_penalty`

最新宿主机 conda 回归结果：

- `tests.test_curriculum_and_checkpoint_score`：`41 tests OK`
- `tests.test_ltsppo_contracts` 新增定向用例：
  - `test_reward_process_adds_continuous_return_progress_shaping_when_return_is_reliable`
  - `test_reward_process_penalizes_high_need_return_stall_without_progress`
  - 均已通过

已知旧问题：

- `tests.test_ltsppo_contracts.LtsppoResumeCompatibleBehaviorTests.test_reward_process_applies_contextual_cleaning_scale_and_margin_pressure`
  仍因历史字段 `charge_margin_pressure` 缺失而失败
- 该问题不是本轮新增回归，也不阻断当前这批 return/stall 修复

### 4.3 运行态主阻断修复：`process_stop.done` 不再导致 learner 秒退

已落实：

- 新增 [run_scoped_stop_patch.py](/home/user/TcKaiwuFinal/code/agent_ppo/utils/run_scoped_stop_patch.py)
- 在 [train/.docker-compose.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.yaml) 的 `learner` / `aisrv` 启动注入层中：
  - 导出 `KAIWU_RUN_BOOT_TS`
  - 导出 `KAIWU_CURRENT_RUN_MANIFEST`
  - 启动时自动 patch：
    - `/root/tools/common.sh`
    - `kaiwudrl/common/utils/train_test_utils.py`
- `check_process_stop_done()` 现在会：
  - 忽略 **早于当前容器启动时间** 的 stale `process_stop.done`
  - 若存在 `process_stop.meta.json`，则可进一步忽略 **不属于当前 run_session** 的 stop token
  - 在忽略 stale stop 时同步清空 `sigterm_pids_file`

已完成验证：

- 宿主机单测：
  - [test_run_scoped_stop_patch.py](/home/user/TcKaiwuFinal/code/tests/test_run_scoped_stop_patch.py) `3 tests OK`
- 运行态验证：
  - 重新 `force-recreate learner` 后，learner 不再在 `check_process_stop_done` 后立即退出
  - 重新 `force-recreate aisrv` 后，4 个 aisrv 均保持在线
  - learner 当前日志已出现真实训练输出：
    - `train count = 587`
    - `global step = 587`
    - `sample_production_and_consumption_ratio = 14.24`
    - `replay buffer monitor = (638, 5275)`
  - 说明此前“`train global step` 不动”的主因链已经被切断

当前结论：

- **已确认修复：**
  - 全局 stale `process_stop.done` 导致 learner 秒退
- **尚未完全修复：**
  - `avg_return_progress_per_step / avg_return_efficiency_ratio / avg_high_need_return_stall_rate` 在**没有形成当前 session bootstrap 窗口**时仍会是 `null`
  - 这不是字段缺失，而是“当前 session 尚无 20 局窗口”的正常状态
  - 训练推进链已经恢复，剩余问题主要是 state/聚合展示时要区分“无窗口”和“字段缺失”

## 5. 还未落实的改动

以下是还没有做完、但仍属于原计划中的内容。

### 5.1 return 阶段连续进展 shaping（第二步主项）

已落实：

- 在 `preprocessor.py` 中新增：
  - `return_progress_shaping_bonus`
  - `high_need_return_stall_penalty`
- `return_progress_shaping_bonus` 现在显式围绕：
  - target progress
  - slack 恢复
  - nearest charger distance 恢复
- `high_need_return_stall_penalty` 现在显式围绕：
  - 无 progress

### 5.2 learner / state 聚合可观测性补齐

已落实：

1. `curriculum_state.last_learning_metrics.global_step` 现在可通过 learner 日志真值兜底，不再只依赖 helper signal 的 `learning_metrics`
   - 兼容两种日志布局：
     - 宿主机：`train/log/learner`
     - 容器：`/workspace/log/learner`
   - 运行态已验证：
     - 容器内手动 `refresh_state()` 后，`global_step` 已从 `0.0` 变为 `7377.0`

2. return 指标已补充 `avg_*` 别名：
   - `avg_return_progress_per_step`
   - `avg_return_efficiency_ratio`
   - `avg_high_need_return_stall_rate`
   - 它们会在**形成 bootstrap/global 窗口后**与原始键同步出现

仍需继续核对：

3. `battery_positive_reward`
4. `zero_charge_battery_fail`
   是否稳定出现在 `curriculum_signals` 的 episode payload 中

状态：**主体已落实；剩余是当前 session 继续积累窗口后的验证项**
  - slack 恶化
  - charger distance 恶化
- 两者都限定在 `planning/critical + CONTRACT/RETURN + 可靠回充上下文` 下生效

状态：**已落实**

### 5.2 curriculum 主门槛重构

尚未落实：

- 还没有把 curriculum 的主门槛从 raw `planner_policy_divergence_rate + return_stall_rate` 切向更贴近因果的指标组合
- 例如：
  - `battery_positive_reward_rate`
  - `zero_charge_battery_fail_rate`
  - `high_need_return_stall_rate`
  - 可信 return 场景下的 divergence

原因：

- 当前先做的是 stop-loss 和分布止损，不是状态机重构

状态：**未落实**

### 5.3 更完整的运行态验证

尚未落实：

- 还没有重新启动训练做新一轮短窗口结构验证
- 当前训练已经手动停掉，适合在完成第二步前继续补测试和验证方案

后续应该重点看：

- `battery_positive_reward_rate`
- `avg_reward_charger_access_probe_bonus`
- `zero_charge_battery_fail_rate`
- `return_action_teacher_active_rate`
- `return_stall_rate`
- `high_need_return_stall_rate`

状态：**未落实**

### 5.4 宿主机更完整的 framework 级测试链

现状：

- `tests.test_runtime_optimizations` 这类更靠近框架集成的宿主机用例，仍然会因为宿主机缺：
  - `kaiwudrl`
  - 完整 `common_python`
  - 以及部分 runtime framework 依赖
  而无法作为本轮主验证源

结论：

- 不应把这些环境导入失败当成本轮实现回归
- 但如果后续要做更完整的宿主机测试，需要再补一层 framework shim 或复用容器中的完整代码挂载

状态：**未落实**

## 6. 当前推荐的下一步顺序

### 优先级 1：补运行态短窗口验证

先做：

1. 保持当前代码不再扩散
2. 启动训练短窗口
3. 只看结构指标，不先看总 `win_rate`

原因：

- 第一阶段止损项和第二阶段第一批 return/stall shaping 已经落地
- 当前最缺的是新 run 下的真实结构反馈，而不是继续堆代码

### 优先级 2：视短窗口结果决定是否继续增强 return 信号

如果短窗口显示：

- `probe` 仍然是 0
- `skip-needed-charge` 仍然几乎不触发
- `return_progress_per_step / return_efficiency_ratio` 没明显改善
- `high_need_return_stall_rate` 不下降

则下一步继续增强 return 动作模板学习信号，而不是先动 curriculum。

### 优先级 3：再动 curriculum 门槛

只有在确认：

- `probe` 活了
- `battery_positive_reward_rate` 下来了
- `return_action_teacher_active_rate` 提升了
- `return_stall_rate` 开始下降

之后，才适合重构 curriculum 主门槛。

## 7. 默认测试方式（后续约定）

后续在这个仓库里，**默认使用 conda 环境跑宿主机测试**：

```bash
/home/user/TcKaiwuFinal/.conda-numpy/bin/python -m unittest ...
```

如果是本轮 reward / curriculum / teacher 改动，优先跑：

```bash
/home/user/TcKaiwuFinal/.conda-numpy/bin/python -m unittest \
  tests.test_curriculum_and_checkpoint_score \
  tests.test_ltsppo_contracts
```

如果是更靠近框架 runtime 的宿主机测试：

- 先确认是否缺 `kaiwudrl/common_python` 等外部框架模块
- 没补齐前，不把这类导入失败当作业务逻辑回归

## 8. 当前训练状态备注

当前训练已手动停止，状态是：

- `learner`：已停
- `aisrv`：已停
- `gamecore`：已停
- `monitor-service / fe-monitor-service / vector / greptimedb / pushgateway / backup_model`：仍在运行

所以现在适合：

- 继续补代码与测试
- 做实现收口
- 不适合直接从现有运行态继续读训练趋势

## 9. 一句话总结

这轮修复中，**第一步的止损项已经基本落实并通过目标测试**；当前还没做完的是：

- return 阶段连续 shaping
- curriculum 主门槛重构
- 新一轮短窗口运行态验证

后续不要再忘记的核心顺序是：

> **先补 return 动作模板学习信号，再做短窗口结构验证，最后才重构 curriculum 门槛。**
