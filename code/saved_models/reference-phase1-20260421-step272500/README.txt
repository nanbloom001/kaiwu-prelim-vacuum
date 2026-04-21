参考留档模型：reference-phase1-20260421-step272500
===============================================

保存时间
--------
2026-04-21

用途定位
--------
这是一个“实验参考模型”，不是正式主线存档模型。

它适合用于：
1. 观察 curriculum 继续训练后的效果变化
2. 对比后续阶段是否真的打掉当前坏局部最优
3. 做 benchmark / 日志层面的阶段性横向比较

它不适合默认用于：
1. preload-main
2. resume-main
3. 主线最终 archive

为什么保存这个点
----------------
用户要求查看 2026-04-21 08:20 左右的模型质量，并保留一个可供后续课程实验对照的参考点。

当时最接近的训练点是：
- step 272500
- step 273000
- step 273500

其中最接近 08:20 且证据链最完整的“原始意图点”是 step 272500。

但是实际排查时发现：
- 原始训练 checkpoint `model.ckpt-272500.pkl` 已经不再保存在可访问的宿主机/挂载目录里
- framework 侧旧 step 文件已经被清理

因此这次留档采用了**最近仍可用的 fallback 资产**：
- `code/runtime_state/runs/20260420-205346/session_best/20260421-010416-pid-443/best_model.pkl`

这个 fallback 资产的更新时间是：
- `2026-04-21 08:19:44`

它与 08:20 窗口非常接近，而且有完整运行态证据支撑，因此适合做“08:20 参考模型”的替代留档。

证据摘要
--------
1. learner 在 08:18:03 保存了 `model.ckpt-272500.pkl`
2. aisrv 在 08:19:08 成功加载了 checkpoint 272500
3. 同一时间窗内，helper `20260421-010416-pid-443` 的 `best_score.json` 在 08:19:44 更新
4. 随后在 08:22:41 跑出：
   - `WIN`
   - `clean_score = 802`
   - `steps = 1000`
   - `profile = mild`

这说明这个参考点不是“刚保存但还没被环境验证”的死快照，而是被实际运行过的近邻资产。

为什么它不适合作为正式主线 archive
-----------------------------------
虽然这批模型在 08:20 左右：
- Win Rate 高
- Avg Clean Score 高
- learner 训练稳定

但它的行为结构仍不健康，主要问题包括：
- `zero_charge_battery_fail_rate` 仍高
- `battery_positive_reward_rate` 仍高
- `planner_policy_divergence_rate` 仍高
- `route_phase_planner_divergence_rate` 仍高
- `mode_usage_contract` 仍几乎为零

这意味着它更像：
- 高分但行为结构仍偏坏的参考点

而不是：
- 已经学出健康 charge / planner / return 闭环的正式基线

当前窗口参考（同一 run 的最近可用窗口）
----------------------------------------
- win_rate: 0.80
- battery_fail_rate: 0.125
- zero_charge_battery_fail_rate: 0.60
- battery_positive_reward_rate: 0.80
- avg_clean_per_step: 0.6735
- mode_usage_expand: 0.0374
- mode_usage_contract: 0.0050
- mode_usage_return: 0.1447
- planner_policy_divergence_rate: 0.8353
- route_phase_planner_divergence_rate: 0.5814
- route_phase_return_stall_rate: 0.3422

如何使用
--------
建议把这个目录当作：
- “课程继续训练前的坏局部最优参考锚点”

后续如果要比较新实验是否真的进步，优先对比：
1. zero_charge_battery_fail_rate 是否下降
2. battery_positive_reward_rate 是否下降
3. mode_usage_contract 是否恢复到合理区间
4. planner / route-phase 偏离率是否下降
5. avg_clean_per_step 是否在不回到高充电保守策略的前提下继续保持

目录内容
--------
- `best_model.pkl`
  - 仍可用于评测型对照
- `model.ckpt-resume.pkl`
  - 为后续工具兼容保留的同一份模型文件别名
- `best_score.json`
  - 保存该 helper 的 best score 摘要
- `archive_manifest.json`
  - 说明本次留档来源、限制和选择理由
