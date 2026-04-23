# Slice 2A Batch A Co-Primary Fix Plan 2026-04-23

## Summary

当前 `s1_survival_strong_heuristic_slice2a_v1` 的主问题已经收敛为两条 **co-primary**：

1. `profile` 固定分布在运行时被 **helper-local 固定随机种子** 同步偏置，导致当前 active history 长期表现为：
   - `anchor ≈ 0`
   - `mild ≈ 0.10 ~ 0.20`
   - `broad ≈ 0.80 ~ 0.90`
   尽管 phase 明确配置为：
   - `anchor / mild / broad = 0.20 / 0.40 / 0.40`

2. `battery-fail terminal economics` 仍允许一部分 `battery fail` episode 的 `effective_total_reward > 0`，即：
   - `battery_positive_reward_rate` 不是解释噪声，而是 objective 仍然残留错误激励

这两条问题必须在 **同一批次** 修复。  
如果只先修 route-phase reward gate / planner divergence，相当于在：

- 错误采样分布
- 错误终局经济学

之上继续做局部调参，结论会继续漂移。

## Confirmed Evidence

### A. Fixed profile mix is not being realized online

当前 live 容器环境已确认：

- `KAIWU_CURRICULUM_LITE=1`
- `KAIWU_CURRICULUM_FIXED_STAGE=warmup`
- `KAIWU_CURRICULUM_PROFILE_ANCHOR=0.20`
- `KAIWU_CURRICULUM_PROFILE_MILD=0.40`
- `KAIWU_CURRICULUM_PROFILE_BROAD=0.40`

但当前 `recent_episodes` 的 active history 分布为：

- 最近 `20` 局：
  - `broad=16, mild=4, anchor=0`
- 最近 `40` 局：
  - `broad=35, mild=5, anchor=0`
- 最近 `120` 局：
  - `broad=100, mild=20, anchor=0`

已经远离 intended fixed mix。

### B. Root cause of profile drift is reproducible

`EnvConfigSampler` 当前在初始化时固定：

- `self.rng = np.random.default_rng(seed=20260409)`

并且每个 helper 都会单独构造一个新的 sampler。  
当按真实 `sample()` 里的随机消耗路径复现时，前 12 次 profile 序列是：

- `broad, broad, mild, broad, broad, mild, broad, broad, broad, broad, mild, mild`

这与当前 live `aisrv Episode 1..12 start` 日志里的 profile 序列一致。

同时当前 active history 中，大部分 `source_id` 的 `episode_cnt_local` 只到：

- `8 ~ 13`

这意味着：

- 各 helper 一直在重复相同 RNG 前缀
- 很多 helper 根本还没运行到第一次 `anchor`

因此当前 broad-heavy active history 不是聚合误差，而是：

> **同步的 per-helper RNG 前缀偏差。**

### C. Battery fail still allows positive return

当前 runtime artifact 中已经确认存在：

1. `battery fail`
   - `task_reward_scale = 0.22`
   - `clean_score = 365`
   - `finished_steps = 625`
   - `charge_count = 6`
   - `total_reward = +7.43`

2. `battery fail`
   - `task_reward_scale = 0.22`
   - `clean_score = 499`
   - `finished_steps = 1428`
   - `charge_count = 22`
   - `total_reward = +8.86`

所以当前：

- `battery_positive_reward_rate ≈ 0.22`

反映的是 objective 仍然允许：

> “clean enough before dying” 作为可赚钱结果存在。

这必须被视为 real bug，而不是指标污染。

## Batch A Scope

Batch A 只做这两项 co-primary 修复：

1. 修 `profile sampling synchronization bias`
2. 修 `battery-fail terminal economics`

明确不在本批次修改：

- `route_phase_reward_ready`
- `route_phase_shadow_risk`
- `planner/teacher divergence`
- Slice 2A 其它 reward 权重

这些问题会留到 **Batch B**，前提是 Batch A 先把：

- 数据分布
- 终局目标

拉回正确状态。

## Change Set A1: Fix synchronized profile sampling

### Goal

让不同 helper 的 profile sampling 不再共享相同 RNG 前缀。  
目标不是“做出真正随机”，而是：

- **在同一 run_session 内可复现**
- **不同 helper / source 不共享前缀**

### Current Problematic Code

文件：

- `code/agent_ppo/workflow/train_workflow.py`

当前：

- `EnvConfigSampler.__init__()`
  - 固定 `seed=20260409`

### Implementation Plan

#### A1.1 Introduce helper-scoped deterministic seed derivation

在 `EnvConfigSampler.__init__()` 中，不再直接写死：

- `np.random.default_rng(seed=20260409)`

改为基于以下上下文派生 deterministic seed：

- `run_session_id`
- `source_id`

推荐顺序：

1. 新增一个 helper：
   - `_derive_sampler_seed(base_seed, run_session_id, source_id)`
2. 使用稳定 hash 得到 `uint32` seed
3. 保留一个 base seed 常量，例如：
   - `20260409`

输出要求：

- 同一个 `run_session_id + source_id` 下可复现
- 不同 helper 不同 seed

明确不使用：

- `helper_session_id`

原因：

- 它当前包含时间戳语义
- 会破坏同一 `run_session_id + source_id` 下的可复现性

#### A1.2 Seed source should come from actual runtime identity

优先从 runtime 已有信息取：

- `EpisodeRunner.run_session_id`
- `source_id`

如果 `EnvConfigSampler` 当前构造时拿不到这些值，则做最小重构：

- 在创建 `EpisodeRunner` 后注入 sampler seed
- 或把 sampler 初始化从 workflow 顶层下沉到 `EpisodeRunner`

原则：

- 不要引入全局随机态
- 不要在每次 `sample()` 时重新 seed

#### A1.3 Add runtime diagnostics for sampled profile distribution

当前只有 completed-episode 的 `profile`。  
Batch A 必须新增一组 sampler-level diagnostics：

- `sampled_profile_anchor_count`
- `sampled_profile_mild_count`
- `sampled_profile_broad_count`
- `sampled_profile_anchor_rate`
- `sampled_profile_mild_rate`
- `sampled_profile_broad_rate`

在 Batch A 当前 fixed stage 下：

- `broad_eval` 不应出现

因此本批次 diagnostics 只记录：

- `anchor / mild / broad`

如果后续该基础设施推广到非 fixed warmup phase，再补：

- `sampled_profile_broad_eval_*`

这些值按 helper / window 聚合后进入：

- `train_workflow.py` diagnostics
- `curriculum_state.py`
- monitor payload

这样可以区分：

1. 采样本身就偏了
2. 采样没偏，但完成局窗口偏了

#### A1.4 Add episode-start logging field for sampler seed

在 `Episode start` 日志里增加：

- `profile_seed=<derived_seed>`

只要在 debug 期可见即可。  
之后可以视情况降级为 debug-only。

### Files Expected to Change

- `code/agent_ppo/workflow/train_workflow.py`
- `code/agent_ppo/workflow/curriculum_state.py`
- `code/agent_ppo/conf/monitor_builder.py`
- 可能新增一个很小的 helper file，但优先不新增

### Validation for A1

#### Unit / deterministic tests

新增测试：

1. 相同 `run_session_id + source_id` 派生相同 seed
2. 不同 `source_id` 派生不同 seed
3. 两个 helper 的前 12 profile 序列不再相同

#### Runtime acceptance

修复后，最近 `40` sampled-profile 分布应接近：

- `anchor ≈ 0.20`
- `mild ≈ 0.40`
- `broad ≈ 0.40`

允许短窗口偏差，但不能再出现：

- `anchor = 0`
- `broad > 0.80`

此外必须增加两条更强的 per-source 验收：

1. 活跃 helper 的 `derived_seed` 必须两两不同
2. 对最近活跃的 `source_id`，其本地前缀不能再长期重复同一 profile 序列

也就是说，Batch A 不能只看全局 sampled mix 过线；  
必须证明：

> 同步前缀 bug 已经被打散。

## Change Set A2: Fix battery-fail terminal economics

### Goal

明确消除：

- `battery fail` episode 的 `effective_total_reward > 0`

也就是：

> battery fail 不能再作为净正收益 outcome 存在。

### Current Problematic Code

文件：

- `code/agent_ppo/workflow/train_workflow.py`

当前路径：

1. `_compute_battery_fail_outcome()`
   - 给出：
     - `battery_terminal_cost`
     - `task_reward_scale`
2. 最终：
   - `effective_total_reward = total_reward * task_reward_scale + final_reward`

问题在于：

- 对 `mid_recoverability_loss`
- 当前 `task_reward_scale = 0.22`
- 仍然允许高 `clean_score` 局在 fail 后净正

### Implementation Plan

#### A2.1 Keep existing fail typing, but add hard non-positive clamp for battery fail

最小修法优先采用：

- 保留当前：
  - `battery_fail_type`
  - `battery_fail_severity`
  - `task_reward_scale`
  - `battery_terminal_cost`
- 但这次 **不能只改 episode summary scalar**
- 必须改到会进入 learner 的 terminal outcome path

因此修法应落在：

- `_compute_battery_fail_outcome()`
- `_apply_terminal_outcome_to_step_records()`
- 以及最终 `effective_total_reward` 计算保持一致

推荐语义：

1. 先得到：
   - `task_reward_scale`
   - `battery_terminal_cost`
2. 再在 terminal outcome 中新增一个明确字段：
   - `force_non_positive_battery_fail = True`
3. 在 `_apply_terminal_outcome_to_step_records()` 所在的 terminal path 中，保证：
   - 电池失败 episode 对应的 step-level 回报路径也被修正
4. episode summary 使用同一条修正后的结果，不允许只在 monitor 层裁剪

可以使用的实现形式包括：

- 在 terminal outcome 中增加额外负向 terminal adjust，使最终 learner-path return 一定不为正
- 或在 terminal outcome 应用阶段，对 battery fail episode 的最终累计 return 做统一 cap

但无论具体写法如何，都必须满足：

> 修复必须进入 `step_records -> sample_process -> PPO advantage` 这条真实学习路径。

只有在这个前提下，才允许使用：

```text
if fail_reason == "battery":
    effective_total_reward = min(effective_total_reward, 0.0)
```

理由：

- 这是最稳的 objective 修正
- 不需要猜测“scale 应该调到多少才够”
- 直接消除错误激励

#### A2.2 Additionally tighten non-early battery-fail scaling

除了 hard clamp，还应把：

- `scheduled_battery_fail_task_reward_scale`

向下收紧。  
当前代码里：

- `early_unrecoverable`
  走 `scheduled_early_battery_fail_task_reward_scale`
- 所有其它 battery fail
  走 `scheduled_battery_fail_task_reward_scale`

也就是说：

- 当前 `0.22` 不是 `mid_recoverability_loss` 专属
- 而是 **所有 non-early battery fail 共用**

建议：

- `0.22 -> 0.12 ~ 0.15`

这样即便后续有人临时关掉 clamp，economics 也更保守。

如果后续真的要做 subtype-specific scaling：

- `mid_recoverability_loss`
- `late_near_completion`

则放到 Batch A.1 或 Batch B，不在本批次扩大实现面。

#### A2.3 Keep stronger treatment for zero-charge fail

保留：

- `ZERO_CHARGE_BATTERY_FAIL_EXTRA_COST`

并继续保证：

- `zero_charge_battery_fail`
  一定不可能有正收益

#### A2.4 Surface positive battery-fail evidence more directly

当前已有：

- `battery_positive_reward_rate`
- `battery_positive_reward_count`

因此本批次不新增新的 count 字段，而是要把已有字段更明确地暴露到：

- diagnostics
- compare
- 需要时的 monitor / report

至少在日志中增加一条当场告警：

- `[BATTERY_POSITIVE_FAIL] ep:... total_reward:... clean_score:... scale:...`

便于 live 复核。

### Files Expected to Change

- `code/agent_ppo/workflow/train_workflow.py`
- 可能轻微影响：
  - `code/agent_ppo/workflow/curriculum_state.py`
  - compare / monitor 只需保持兼容即可

### Validation for A2

#### Unit tests

新增测试：

1. `battery fail` 且原始 scaled reward 为正时，最终进入 learner-path 的 terminal outcome 也保证非正
2. `zero_charge battery fail` 仍保持更强负向
3. `collision fail` 逻辑不受影响

#### Runtime acceptance

修复后要求：

- `battery_positive_reward_rate == 0.0`

如果窗口里仍然 > 0：

- 视为 Batch A 未通过

## Batch A Acceptance Criteria

Batch A 必须同时满足：

1. `sampled_profile_anchor_rate` 不再接近 0，且 active window 不再表现为 broad-heavy 同步前缀
2. `battery_positive_reward_rate == 0.0`
3. `battery fail` episode 不再出现 `effective_total_reward > 0`
4. 通过专门测试确认：step-level terminal reward path 也不再允许 battery fail 正收益漏进 learner

Batch A 不要求立刻解决：

- `route_phase_return_stall_rate`
- `planner_policy_divergence_rate`
- `route_phase_reward_ready_rate`

这些是 Batch B 的目标。

## Batch B Preview (Not In This Change)

只有当 Batch A 通过后，才进入 Batch B：

1. 修 `planner/teacher divergence`
2. 放宽 `route_phase_reward_ready`
3. 重新校正 route-phase penalty 强度

否则当前 Batch B 的结果仍然会被：

- 错误采样分布
- 错误终局经济学

污染。

## Recommended Execution Order

1. 先改 `EnvConfigSampler` seed 策略和 sampled-profile diagnostics
2. 再改 battery-fail terminal economics 和 hard clamp
3. 补测试
4. dry-run
5. 子代理 code review
6. 重新 `scratch` 启动
7. 先看：
   - `sampled_profile_*_rate`
   - `battery_positive_reward_rate`
8. 只有这两个通过，才继续看 Batch B
