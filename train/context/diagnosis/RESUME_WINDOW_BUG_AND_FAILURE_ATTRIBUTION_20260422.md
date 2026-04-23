# Resume Window Bug And Failure Attribution 2026-04-22

## Summary

- Active resumed run: `20260422-175940`
- Resume source bundle: `resume-bundle-20260422-151541-step92149`
- Training startup: success
- `aisrv` main loop status: healthy
- Main bug: **resume 后课程窗口和 fixed/local compare 语义错位**
- Main training issue: **当前失败主因仍是 battery fail，且多数是 zero-charge / late-return battery fail**

这次问题分成两层：

1. **系统层 bug**
   - `aisrv` 没有新的 runtime exception，但 resumed run 的 `curriculum_state` / `comparison_samples` 语义不自洽。
   - 原因是 resume seed 时保留了旧 run 的 `global_episode_count`，同时又把 `recent_episodes` 清空，导致 resumed run 看起来像“已经跑了 839 局”，但本地窗口只有几十局。
   - 结果是：
     - rolling 还能算出部分窗口
     - `comparison_samples.json` 不落点
     - scratch run 与 resumed run 的 fixed/local compare 失效

2. **训练行为失败**
   - 当前不是 reward-sign 残留问题，`battery_positive_reward_rate = 0.0`
   - 当前主失败是 survival 退化：
     - `battery_fail_rate` 高
     - `zero_charge_battery_fail_rate` 高
     - `expand` 极低
     - `return` 占比偏高但回充路径质量一般

## Verified Runtime Facts

### `aisrv` status

从 `train/log/aisrv/aisrv_kaiwu_rl_helper_pid440_log_2026-04-22-17.log` 看：

- 能持续看到 `Episode N start`
- 能持续看到 `[GAMEOVER]`
- 能持续看到 `[REWARD_TOP]`
- 能持续看到 `[DEATH_TRAJ]`
- 未看到新的 `Traceback / Exception / KeyError / RuntimeError`

结论：

> 当前不是 `aisrv` 主循环崩溃，也不是 helper 线程再次抛异常。

### Resume state mismatch

恢复源 bundle `resume-bundle-20260422-151541-step92149` 中：

- `global_episode_count = 839`
- `global_step_since_resume = 93981`

而当前 resumed run 一度表现为：

- `training_start_mode = resume`
- `restored_from_session_id = 20260422-151541`
- `global_episode_count = 839`
- `recent_episodes_len` 只有几十局
- `comparison_samples.json.sample_points = {}`

这说明 resumed run 混用了两套语义：

- 一套是“恢复前的历史全局累计”
- 一套是“恢复后重新积累的本地窗口”

## Root Cause

根因在 `code/agent_ppo/workflow/curriculum_state.py`：

- `seed_initial_state()` 的 resume 路径会从 bundle snapshot 恢复：
  - `global_episode_count`
  - `global_step_since_resume`
  - `last_bootstrap_metrics`
  - `last_global_metrics`
  - 以及部分观察状态
- 但同时把：
  - `recent_episodes = []`

这样进入 `refresh_state()` 后，会出现：

- `global_episode_count` 还是旧历史值
- `recent_episodes` 却是恢复后新 run 的本地窗口

后续 `_update_comparison_samples_payload()` 又要求：

- `global_episode_count == len(recent_episodes)`

才能安全地创建 fixed/local sample points。

因此 resumed run 会出现：

- rolling 指标部分可见
- `comparison_samples` 长期为空
- `compare_training_runs.py` 只能得到 `missing_local`

## Fix Applied

### 1. Resume local window reset

在 `curriculum_state.py` 里，把 resumed run 的窗口语义改为：

- **恢复来源元数据单独保存**
  - `restored_from_session_id`
  - `restored_global_episode_count`
  - `restored_global_step_since_resume`
- **恢复后本地窗口重新计数**
  - `global_episode_count = 0`
  - `global_step_since_resume = 0`
  - `recent_episodes = []`
  - `last_bootstrap_metrics = {}`
  - `last_global_metrics = {}`
  - `sample_window_metrics = {}`
  - 观察器状态清零
  - `window_origin = resumed_local`

这样 resumed run 不再伪装成 scratch 前缀，而是明确表示：

> “这是从旧 bundle 恢复出来的新局部训练窗口”

### 2. Comparison payload metadata

`comparison_samples.json` 现在会显式写：

- `training_start_mode`
- `window_origin`
- `resumed_from_session_id`

这样 compare 侧可以知道：

- 这是 scratch-local 还是 resumed-local

### 3. Compare fallback reconstruction

`train/compare_training_runs.py` 增加了 resumed-local fallback：

- 对旧 resumed run snapshot，即使 `global_episode_count != len(recent_episodes)`
- 只要 snapshot 带有 resume 语义，就按：
  - `effective_episode_count = len(recent_episodes)`
  来做 resumed-local 重建

这让当前 run `20260422-175940` 不用重启训练，也能恢复 fixed/local compare。

## Current Failure Attribution

### A. Fixed/local compare after the fix

`python3 train/compare_training_runs.py 20260422-151541 20260422-175940`

当前恢复出的关键点位：

- `bootstrap_10`
  - `zero_charge_battery_fail_rate = 0.20`
  - `battery_fail_rate = 0.40`
  - `avg_clean_per_step = 0.3583`
  - `status = direction_normal`
- `bootstrap_20`
  - `zero_charge_battery_fail_rate = 0.60`
  - `battery_fail_rate = 0.60`
  - `avg_clean_per_step = 0.4900`
  - `status = early_stop_warning`
- `global_40`
  - `zero_charge_battery_fail_rate = 0.50`
  - `battery_fail_rate = 0.65`
  - `avg_clean_per_step = 0.4967`
  - `status = main_fail`

这说明：

> resumed run 的正式 fixed/local 主判已经能重新落出来，而且结论偏负面。

### B. Current rolling 40 window

当前 `curriculum_state.json` 的最近 40 局：

- `avg_clean_per_step = 0.4533`
- `battery_fail_rate = 0.525`
- `zero_charge_battery_fail_rate = 0.425`
- `battery_positive_reward_rate = 0.0`
- `win_rate = 0.475`
- `mode_usage_expand = 0.0131`
- `mode_usage_contract = 0.0073`
- `mode_usage_return = 0.1591`
- `route_phase_return_stall_rate = 0.3419`
- `clean_floor_revisit_rate = 0.1313`
- `clean_floor_revisit_penalty_mean = -0.0051`
- `effective_coverage_bonus_mean = 0.00176`

含义：

- 不是“fail 还在赚钱”
- 而是 **真实的高 battery fail / 高 zero-charge fail**
- `expand` 极低，`return` 不低，但 `return` 质量仍一般
- CPS 对齐项已经生效，但量级很小，无法主导当前结果

### C. Reward composition from live `REWARD_TOP`

从当前 resume run 的 `REWARD_TOP` 看，正负项形态很稳定：

正项反复出现：

- `cleaning`
- `explore`
- `streak`

负项反复出现：

- `coverage_tangle_penalty`
- `skip_needed_charge_penalty`
- `planner_alignment`
- 部分局还会出现 `clean_floor_revisit_penalty`

这说明：

> 当前不是 reward 方向完全错了，而是“清扫正收益仍在，但结构性失败成本更大”。

尤其是：

- `clean_floor_revisit_penalty` 已经进入真实训练，例如：
  - `ep:4 neg:[..., clean_floor_revisit_penalty=-0.010]`
  - `ep:7 neg:[..., clean_floor_revisit_penalty=-0.018]`

说明 CPS 对齐两步已接到线上训练，不是空实现。

### D. Death replay

当前最关键的死亡回放特征：

- `DEATH_TRAJ` 末 20 步几乎持续处于 `mode=4 (RETURN)`
- `slack` 长时间为负，并且会进一步恶化

代表样本：

- `ep:1`
  - `steps=200`
  - `clean_score=124`
  - 最后 20 步全部 `mode=4`
  - `slack` 长期 `-25` 左右
- `ep:2`
  - `steps=320`
  - `clean_score=156`
  - 最后 20 步全部 `mode=4`
  - `slack` 从 `-5.8` 一路恶化到 `-11.8`
- `ep:13`
  - `steps=481`
  - `clean_score=267`
  - 最后 20 步大部分 `mode=4`
  - `slack` 从正值逐步跌到 `-8.5`

这说明：

> 当前主失败不是“完全没进 return”，而是“进入了 return，但已经太晚，进入时 slack 不够，无法形成首充闭环”。

## Final Attribution

当前失败可以归为四条：

1. **resume 系统本身可用，但观察链原先有 bug**
   - 训练确实 resumed 成功
   - 但 fixed/local compare 的窗口语义原先失真

2. **当前主失败不是 reward-sign residual**
   - `battery_positive_reward_rate = 0.0`

3. **当前主失败是 late-return / zero-charge battery fail**
   - `battery_fail_rate` 高
   - `zero_charge_battery_fail_rate` 高
   - `DEATH_TRAJ` 证明 return 往往进得太晚

4. **CPS 对齐项已经接入，但当前强度不足以扭转主失败结构**
   - `clean_floor_revisit_penalty` 已真实生效
   - `effective_coverage_bonus_mean` 有值
   - 但当前更大的问题仍是 survival 失败

## Practical Meaning

当前最准确的工程判断是：

> resume 观察链 bug 已经修复，当前 resumed run 现在可以被正式比较。
> 但比较结果本身并不好：当前 resumed run 的失败主因仍然是高 zero-charge / high battery-fail，与其继续猜测 `aisrv` 是否崩掉，不如把下一轮注意力放到“如何让 return 更早、更稳地闭合首充”。

