# Full Repo Audit Findings 2026-04-20

本文件记录 2026-04-20 针对 `/home/user/TcKaiwuFinal` 的发布前全量代码审查首批已确认问题。

当前状态：

- 审查仍在进行中
- 这里先记录已由主审查线程复核确认的问题
- 后续如有新增高优先级问题，应继续追加到本文件

## 已确认高优先级问题

### P1-01 `global_step_since_resume` 在正常训练路径中会被错误写成 0

位置：

- [code/agent_ppo/agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py:358)
- [code/agent_ppo/agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py:361)

现象：

- `observation_process()` 从 `current_model_ref["checkpoint_id"]` 推导 `global_step_since_resume`
- 正常训练路径里 `run_episodes()` 每局调用的是 `load_model(id="latest")`
- 此时 `checkpoint_id` 很容易是字符串 `"latest"`
- 代码随后落入 `ValueError` 分支，把 `global_step_since_resume` 设成 `0`

影响：

- `Preprocessor.feature_process()` 里的进度相关逻辑会长期停留在 warm-start 行为
- 包括 `get_reward_schedule()`、阶段性 teacher mask、进度相关 reward/gating

结论：

- 这是 correctness 回归
- 会持续污染训练行为，不是只影响监控

### P1-02 `collision/unknown` 终局惩罚没有真正写回 learner 样本

位置：

- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:1364)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:1367)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:1467)

现象：

- episode 结束后，最后一个 step 只把 `task_terminal_bonus` 写回 `reward_clean`
- 只把 `battery_terminal_cost` 写回 `reward_survive`
- `collision_terminal_cost` 虽然参与日志里的 `effective_total_reward`
- 但没有写回 `step_records[-1]`

影响：

- `collision` 和 `unknown` 失败在日志里看起来被惩罚了
- 但 learner 实际训练 batch 没吃到对应 terminal penalty
- 会系统性低估这两类失败

结论：

- 这是 correctness 回归
- 会直接改变训练目标

### P1-03 invalid gradient 处理在 AMP 关闭时仍然会执行优化器步进

位置：

- [code/agent_ppo/algorithm/algorithm.py](/home/user/TcKaiwuFinal/code/agent_ppo/algorithm/algorithm.py:174)

现象：

- `_finalize_invalid_after_unscale()` 在确认梯度非有限后仍然调用 `self.scaler.step(self.optimizer)`
- 当 `use_amp=False` 时，disabled `GradScaler` 会退化成直接执行 `optimizer.step()`

影响：

- 本应“跳过坏 batch”
- 实际上会用 NaN/Inf 梯度更新参数
- 在非 AMP 训练环境中会造成真实权重污染

结论：

- 这是 correctness 回归
- 严重时会让训练表面继续推进，但结果已被污染

### P2-01 `load_model_calls/reloads/cache_hits` 未真正进入监控 payload

位置：

- [code/agent_ppo/agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py:648)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:2126)

现象：

- `agent.get_runtime_metrics()` 已经暴露：
  - `load_model_calls`
  - `load_model_reloads`
  - `load_model_cache_hits`
- 但 `_build_monitor_payload()` 只回填：
  - `predict_fallback_count`
  - `predict_error_count`

影响：

- reload 平滑化虽然进了代码
- 但运行时无法从统一监控上直接验证是否生效
- 会削弱训练过程可观测性

结论：

- 这是 observability 缺口
- 不是训练目标回归，但会影响上线后判断

### P1-04 多 helper 会并发覆盖同一组 latest resume 工件，恢复链可能回退或自相矛盾

位置：

- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:671)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:687)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:691)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:765)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:779)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:784)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:1092)

现象：

- 多个 helper 都会在本地触发器里无条件重写 shared `latest resume` 工件
- 没有 bundle 级锁
- 没有“只接受更大 step”的保护
- 训练指标又是按固定间隔拉取，writer 容易拿过期 step 落盘

运行态证据：

- [code/model.ckpt-resume.state.json](/home/user/TcKaiwuFinal/code/model.ckpt-resume.state.json:7) 中的 `global_step`
- [code/curriculum_state.resume_snapshot.json](/home/user/TcKaiwuFinal/code/curriculum_state.resume_snapshot.json:23) 中的 `global_step_since_resume`
- [code/curriculum_state.resume_snapshot.json](/home/user/TcKaiwuFinal/code/curriculum_state.resume_snapshot.json:149) 中的 `last_learning_metrics.global_step`

这些值已经出现无法同时成立的组合。

影响：

- resume 恢复点可能回退
- sidecar 与 pkl 可能不对应
- checkpoint 选择与恢复行为可能被历史脏状态污染

结论：

- 这是 correctness + recovery chain 高优先级问题

### P1-05 `session_id` 被错误地当成全局训练 owner，导致多 helper 分叉共享课程状态

位置：

- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:247)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:444)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:615)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:618)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:669)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:682)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:717)

现象：

- `session_id` 是每个 helper 本地按时间生成的
- 但共享 `curriculum_state` 却把它当成全局唯一 owner
- 只要 helper 不是同一秒启动，就会出现多套当前 session
- `curriculum_state` 刷新时还会按 `session_id` 丢弃其他 helper 的 signal

运行态证据：

- `code/curriculum_signals/` 中当前同一轮并存多组：
  - `20260420-012242`
  - `20260420-012243`
  - `20260420-012246`

影响：

- 全局窗口聚合会分叉
- 恢复 sidecar 与当前 helper 视角不一致
- dashboard、课程推进、resume 恢复都可能基于不完整的 helper 子集做判断

结论：

- 这是 shared state 语义错误
- 会持续制造跨 helper 不一致

### P2-02 `global_episode_count` 的语义与展示不一致，运行久后会封顶在 120

位置：

- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:52)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:299)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:740)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:909)
- [train/local_monitor_dashboard.py](/home/user/TcKaiwuFinal/train/local_monitor_dashboard.py:277)
- [train/local_monitor_dashboard.py](/home/user/TcKaiwuFinal/train/local_monitor_dashboard.py:330)
- [train/local_monitor_dashboard.py](/home/user/TcKaiwuFinal/train/local_monitor_dashboard.py:520)

现象：

- `global_episode_count` 实际上只是 `recent_episodes` 的长度
- `recent_episodes` 被硬截到 120 条
- dashboard 却把它展示成“当前 session 总 episode 数”

影响：

- 长时间运行后该值会封顶
- 会误导人工判断训练推进与窗口稳定性

结论：

- 这是 observability 语义错误

### P2-03 dashboard 的 Recent Episodes 表在多 helper 场景下会系统性丢数据

位置：

- [train/local_monitor_dashboard.py](/home/user/TcKaiwuFinal/train/local_monitor_dashboard.py:345)
- [train/local_monitor_dashboard.py](/home/user/TcKaiwuFinal/train/local_monitor_dashboard.py:363)
- [train/local_monitor_dashboard.py](/home/user/TcKaiwuFinal/train/local_monitor_dashboard.py:367)

现象：

- 当前表格只扫描最近两个 aisrv 日志文件
- 再按本地 `ep` 去重
- 但 `ep` 是 helper 本地计数，不是全局唯一键

影响：

- 不同 helper 的同号 episode 会相互覆盖
- 表格会随机丢掉部分 helper 的最近样本

结论：

- 这是 observability mismatch

### P2-04 `seed_preload_from_resume()` 复制 sidecar 时没有同步修正 `checkpoint_path`

位置：

- [code/agent_ppo/workflow/preload_checkpoint.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/preload_checkpoint.py:199)
- [code/agent_ppo/workflow/preload_checkpoint.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/preload_checkpoint.py:216)
- [code/agent_ppo/workflow/preload_checkpoint.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/preload_checkpoint.py:104)

现象：

- 复制 snapshot sidecar 时，只修正了 `curriculum_state_snapshot_path`
- 没有把 `checkpoint_path` 改成新落地的本地工件路径
- 后续恢复又直接信任这个字段

影响：

- 原 snapshot 被清理后，state sidecar 可能仍在
- 但其中 `checkpoint_path` 已经悬空
- 恢复逻辑会沿错误路径继续走

结论：

- 这是 resume 健壮性问题

### P1-06 多个训练关键环境变量定义在 `.env`，但没有真正进入容器运行态

位置：

- [train/.env](/home/user/TcKaiwuFinal/train/.env:90)
- [train/.env](/home/user/TcKaiwuFinal/train/.env:91)
- [train/.env](/home/user/TcKaiwuFinal/train/.env:100)
- [train/.env](/home/user/TcKaiwuFinal/train/.env:102)
- [train/.env](/home/user/TcKaiwuFinal/train/.env:119)
- [train/.env](/home/user/TcKaiwuFinal/train/.env:127)
- [train/.docker-compose.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.yaml:18)
- [code/agent_ppo/conf/conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py:225)
- [code/agent_ppo/conf/conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py:254)
- [code/agent_ppo/conf/conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py:339)
- [code/agent_ppo/conf/conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py:343)
- [code/agent_ppo/conf/conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py:377)

现象：

- `.env` 里定义了多项 curriculum / reward / resume / snapshot 关键参数
- 但 compose 的共享 `environment` 块没有把这些键全部导出到 learner/aisrv
- 代码侧又直接用 `os.getenv()` 读取这些值

影响：

- 运维以为改了 `.env`
- 实际运行仍在静默使用代码默认值
- 配置意图与真实训练行为不一致

结论：

- 这是配置传播链高优先级问题

### P1-07 archive 目录挂载路径与运行时代码默认路径不一致

位置：

- [train/.env](/home/user/TcKaiwuFinal/train/.env:26)
- [train/.docker-compose.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.yaml:148)
- [train/.docker-compose.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.yaml:1005)
- [code/agent_ppo/utils/experiment_archive.py](/home/user/TcKaiwuFinal/code/agent_ppo/utils/experiment_archive.py:68)

现象：

- 主机侧 `KAIWU_ARCHIVE_DIR` 被挂载到了容器的 `/workspace/archive`
- 但该变量本身没有稳定进入运行态环境
- `ExperimentArchive` 因此会退回到默认路径 `/workspace/train/archive`

影响：

- 表面上 archive 卷已经挂载
- 实际运行元数据可能写到了另一条默认路径
- 归档与运维预期位置不一致，容易造成“文件存在但不是写到想要的地方”

结论：

- 这是配置与运行时路径语义不一致的高优先级问题

### P2-05 `KAIWU_RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS` 当前更像硬编码默认，不像可验证的可调配置

位置：

- [train/.env](/home/user/TcKaiwuFinal/train/.env:102)
- [code/agent_ppo/conf/conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py:378)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:475)
- [code/tests/test_curriculum_and_checkpoint_score.py](/home/user/TcKaiwuFinal/code/tests/test_curriculum_and_checkpoint_score.py:1352)

现象：

- `.env` 暴露了可调旋钮
- `Config` 里直接写死默认 `600`
- 测试只验证默认值，不验证 env override

影响：

- 这个配置是否真的可调缺少保护
- 即使 compose 透传修好，后续仍可能被误判为“已覆盖”

结论：

- 这是配置可验证性缺口

### P2-06 现有测试与 runtime probe 没覆盖 `.env -> compose -> container -> os.getenv()` 传播链

位置：

- [code/agent_ppo/agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py:46)
- [code/agent_ppo/agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py:57)
- [code/tests/test_runtime_optimizations.py](/home/user/TcKaiwuFinal/code/tests/test_runtime_optimizations.py:142)
- [code/tests/test_curriculum_and_checkpoint_score.py](/home/user/TcKaiwuFinal/code/tests/test_curriculum_and_checkpoint_score.py:1253)
- [code/tests/test_curriculum_and_checkpoint_score.py](/home/user/TcKaiwuFinal/code/tests/test_curriculum_and_checkpoint_score.py:1352)

现象：

- runtime probe 只记录少量 replay/batch/service 相关 env
- 测试也主要断言默认值
- 没有任何测试覆盖训练关键 env 的容器传播链

影响：

- 上述环境变量丢失问题可以静默穿过 CI
- 配置回归难以及时发现

结论：

- 这是测试覆盖缺口

### P2-07 `expert_weight` 在 sample -> learner 链路中被静默丢失

位置：

- [code/agent_ppo/feature/definition.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/definition.py:64)
- [code/agent_ppo/feature/definition.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/definition.py:222)
- [code/agent_ppo/algorithm/algorithm.py](/home/user/TcKaiwuFinal/code/agent_ppo/algorithm/algorithm.py:22)
- [code/agent_ppo/algorithm/algorithm.py](/home/user/TcKaiwuFinal/code/agent_ppo/algorithm/algorithm.py:469)
- [code/agent_ppo/algorithm/algorithm.py](/home/user/TcKaiwuFinal/code/agent_ppo/algorithm/algorithm.py:545)

现象：

- `sample_process()` 会把 `expert_weight` 写进 `SampleData`
- `SAMPLE_FIELD_ORDER` 里也保留了该字段
- 但 learner unpack/build batch 时没有把它带回 batch dict

影响：

- 这是 producer/consumer 合约漂移
- 训练侧会静默丢失一个本应可学习或可观测的信号

结论：

- 这是中优先级数据链回归

### P3-01 safe fallback 的 `route_anchor_prob` 形状与配置不一致

位置：

- [code/agent_ppo/conf/conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py:47)
- [code/agent_ppo/agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py:783)
- [code/agent_ppo/agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py:811)

现象：

- `ROUTE_ANCHOR_DIM = CHARGER_SLOTS + 1`
- 但 safe fallback 里硬编码了 `neutral_anchor = np.array([0.5, 0.5])`

影响：

- fallback 路径下 `route_anchor_prob` 维度错误
- 下游分析和日志会得到错误形状的数据
- fallback 时 `route_anchor` 只能落在很小的子空间

结论：

- 这是低优先级数据质量/可观测性问题

### P1-08 课程门槛使用的是模型辅助头预测标签，而不是真实 planner/runtime 行为

位置：

- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:1306)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:1321)
- [code/agent_ppo/agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py:781)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:1891)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:1907)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:1915)
- [code/agent_ppo/workflow/curriculum_policy.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_policy.py:85)
- [code/agent_ppo/workflow/curriculum_policy.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_policy.py:500)

现象：

- `step_records["mode"]`、`["target"]`、`["route_anchor"]` 使用的是 `ActData` 的预测结果
- `_episode_sequence_diagnostics()` 再基于这些预测标签推导：
  - `late_return_rate`
  - `return_stall_rate`
  - `mode_usage_*`
  - `anchor_switch_rate`
  - `target_switch_rate`
- 这些指标随后又进入课程晋级/停滞门槛

影响：

- 课程推进和回退可能由 noisy head prediction 驱动
- 而不是由真实的 planner/runtime 行为驱动
- 这会让课程状态机对“预测抖动”过敏

结论：

- 这是课程诊断与 gating 语义错误，高优先级

### P1-09 resume stabilization 可能把恢复出的高阶段静默推回 `warmup`

位置：

- [code/agent_ppo/workflow/curriculum_policy.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_policy.py:260)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:784)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:793)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:627)

现象：

- `choose_stage_decision()` 在 resume stabilization 窗口内会无条件给出 `proposed_stage = "warmup"`
- `refresh_state()` 只要发现 `proposed_stage != current_stage`，就走“推进”路径
- 这里没有检查 stage 顺序方向

影响：

- 如果 resume 恢复的是 `blend / robust / eval_hard`
- 该 run 可能仅因为处于 stabilization window，就被静默推回 `warmup`
- 而这一步还会被当成正常课程切换处理

结论：

- 这是课程恢复语义错误，高优先级

### P1-10 `_infer_mode()` 用了错误量做 contract gate：拿距离阈值去比 slack 阈值

位置：

- [code/agent_ppo/feature/preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py:1208)
- [code/agent_ppo/feature/expert.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/expert.py:380)

现象：

- `_infer_mode()` 用 `anchor_return_dist` 去比较 `PREPARE_RETURN_SLACK_THRESHOLD`
- 这个阈值语义上是 slack threshold
- teacher 逻辑里对应比较的是 `slack`

影响：

- 只要 charger 物理上离得近，就可能过早进入 `MODE_CONTRACT`
- 即使电量 slack 其实很健康
- 同时 runtime mode 会与 teacher 监督不一致

结论：

- 这是 mode 推断逻辑错误，高优先级

### P2-08 在 expert 明确标记不可靠的状态下，代码仍然强行提高 target/route-anchor teacher mask

位置：

- [code/agent_ppo/feature/expert.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/expert.py:356)
- [code/agent_ppo/feature/expert.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/expert.py:407)
- [code/agent_ppo/feature/preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py:1671)
- [code/agent_ppo/feature/preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py:1677)
- [code/agent_ppo/algorithm/algorithm.py](/home/user/TcKaiwuFinal/code/agent_ppo/algorithm/algorithm.py:699)
- [code/agent_ppo/algorithm/algorithm.py](/home/user/TcKaiwuFinal/code/agent_ppo/algorithm/algorithm.py:704)

现象：

- expert 只在 `target_reliable / anchor_reliable` 成立时暴露 teacher masks
- 但 `reward_process()` 在 `planning/critical` 状态下会把这两个 mask 至少抬到 `0.65`
- learner 又把这些 mask 当真实 loss 权重使用

影响：

- 原本“不可靠、不应监督”的 teacher signal
- 变成了被强行监督的 noisy label

结论：

- 这是 teacher gating 语义错误，中优先级

### P2-09 reward attribution 不完整，课程和 reward 分析看到的贡献分解不是 agent 实际训练到的 reward

位置：

- [code/agent_ppo/feature/preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py:1329)
- [code/agent_ppo/feature/preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py:1340)
- [code/agent_ppo/feature/preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py:1706)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:53)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:79)
- [code/agent_ppo/utils/reward_metrics.py](/home/user/TcKaiwuFinal/code/agent_ppo/utils/reward_metrics.py:12)
- [code/agent_ppo/workflow/curriculum_state.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/curriculum_state.py:266)

现象：

- `dirty_approach_reward` 被加进了 `gain_reward` 和 `reward_total`
- 但没有被导出到 `components`
- 后续 reward-share / curriculum window 统计又只跟踪 components 子集

影响：

- 分析面板里看到的 reward contribution totals
- 不是 agent 实际训练到的完整 reward 分解
- 很容易误导 reward 调优判断

结论：

- 这是 reward observability 与分析口径错误，中优先级

### P3-02 `planner_topk_reachable_count` 当前不是独立信息，而是与 `all_known_path_count` 重复

位置：

- [code/agent_ppo/feature/expert.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/expert.py:262)
- [code/agent_ppo/feature/expert.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/expert.py:272)
- [code/agent_ppo/feature/preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py:1192)
- [code/agent_ppo/feature/preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py:1216)

现象：

- `get_charger_signal()` 里两个计数在相同条件下同时递增
- `_infer_mode()` 却把它们当成两个独立条件使用

影响：

- 当前 `planner_topk_reachable_count` 没提供额外信息
- 会增加指标与门槛的表面复杂度

结论：

- 这是低优先级指标冗余问题

### P1-11 benchmark checkpoint 加载路径同样会把 actor 侧进度传播打成 0

位置：

- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:1811)
- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:1814)
- [code/agent_ppo/agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py:359)

现象：

- benchmark 加载特定 checkpoint 后，会手动写：
  - `current_model_ref["checkpoint_id"] = os.path.basename(resolved)`
- 例如值会变成 `model.ckpt-12345.pkl`
- 但 `observation_process()` 后续会直接 `int(checkpoint_id)`

影响：

- benchmark 路径下 actor 侧 `global_step_since_resume` 同样会落回 `0`
- 评测时使用的 reward schedule / staged gating / teacher mask 也会被固定在 early phase
- 这会让 benchmark 结果与真实训练语义不一致

结论：

- 这是评测正确性问题，高优先级

### P1-12 `agent_diy` 被注册成可运行算法，但实现基本是空骨架，属于高风险误接入口

位置：

- [code/conf/app_conf_robot_vacuum.toml](/home/user/TcKaiwuFinal/code/conf/app_conf_robot_vacuum.toml:6)
- [code/conf/algo_conf_robot_vacuum.toml](/home/user/TcKaiwuFinal/code/conf/algo_conf_robot_vacuum.toml:9)
- [code/train_test.py](/home/user/TcKaiwuFinal/code/train_test.py:15)
- [code/agent_diy/agent.py](/home/user/TcKaiwuFinal/code/agent_diy/agent.py:28)
- [code/agent_diy/algorithm/algorithm.py](/home/user/TcKaiwuFinal/code/agent_diy/algorithm/algorithm.py:20)
- [code/agent_diy/feature/definition.py](/home/user/TcKaiwuFinal/code/agent_diy/feature/definition.py:46)
- [code/agent_diy/model/model.py](/home/user/TcKaiwuFinal/code/agent_diy/model/model.py:20)
- [code/agent_diy/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_diy/workflow/train_workflow.py:22)

现象：

- `diy` 已经被配置成可选算法入口
- 但从 agent、algorithm、reward/sample、model 到 workflow 基本都是 `pass` 或空骨架

影响：

- 只要有人把主线算法切到 `diy`
- 就会进入“能启动但实际上没有训练能力”的空实现路径

结论：

- 这不是当前 `ppo` 主线 bug
- 但属于高风险误接入口，发布前应明确冻结或移除

### P1-13 `run_speed_experiments.py` 会直接永久改写 `train/.env`，没有备份/恢复

位置：

- [train/run_speed_experiments.py](/home/user/TcKaiwuFinal/train/run_speed_experiments.py:62)
- [train/run_speed_experiments.py](/home/user/TcKaiwuFinal/train/run_speed_experiments.py:171)
- [train/.docker-compose.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.yaml:33)
- [train/.docker-compose.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.yaml:56)
- [.gitignore](/home/user/TcKaiwuFinal/.gitignore:35)

现象：

- 脚本会直接写 `train/.env`
- 然后立即重启训练栈
- 结束后没有回滚逻辑
- `.env` 还被 `.gitignore` 忽略

影响：

- 实验参数会永久污染后续正常训练
- 由于 `.env` 不进 `git status`，这种漂移很难被察觉

结论：

- 这是高优先级运维/实验污染风险

### P2-10 `resume_best.py` 的 checkpoint 路径绑定与当前仓库布局不一致

位置：

- [train/resume_best.py](/home/user/TcKaiwuFinal/train/resume_best.py:20)
- [train/resume_best.py](/home/user/TcKaiwuFinal/train/resume_best.py:29)

现象：

- 脚本扫描的是 `train/backup_model/*.zip.json`
- 但当前仓库实际 checkpoint 元数据主要在 `train/backup_model/signed/`

影响：

- `list / best / prepare / clean` 很可能看不到现有模型
- 脚本会误报“没有 checkpoint”

结论：

- 这是中优先级路径错绑

### P2-11 多个实验脚本把结果写进了错误的 context 目录层级

位置：

- [train/run_datafetch_benchmark.py](/home/user/TcKaiwuFinal/train/run_datafetch_benchmark.py:30)
- [train/run_env_scaling_experiment.py](/home/user/TcKaiwuFinal/train/run_env_scaling_experiment.py:24)
- [train/run_replay_stability_experiments.py](/home/user/TcKaiwuFinal/train/run_replay_stability_experiments.py:23)
- [train/run_speed_experiments.py](/home/user/TcKaiwuFinal/train/run_speed_experiments.py:18)
- [train/context/data/DATAFETCH_BENCHMARK_RESULTS.json](/home/user/TcKaiwuFinal/train/context/data/DATAFETCH_BENCHMARK_RESULTS.json:1)
- [train/context/data/ENV_SCALING_RESULTS.json](/home/user/TcKaiwuFinal/train/context/data/ENV_SCALING_RESULTS.json:1)
- [train/context/data/REPLAY_STABILITY_RESULTS.json](/home/user/TcKaiwuFinal/train/context/data/REPLAY_STABILITY_RESULTS.json:1)

现象：

- 这些脚本把新结果写到 `train/context/*.json`
- 但现有结构化结果文件实际约定存放在 `train/context/data/`

影响：

- 会持续制造散落 JSON
- 结果文件位置分叉，文档和消费脚本难以对齐

结论：

- 这是中优先级产物治理问题

### P3-03 `collect_data.py` 会混扫所有历史 aisrv 日志，导出结果容易混入旧 session

位置：

- [train/collect_data.py](/home/user/TcKaiwuFinal/train/collect_data.py:11)
- [train/collect_data.py](/home/user/TcKaiwuFinal/train/collect_data.py:33)
- [train/collect_data.py](/home/user/TcKaiwuFinal/train/collect_data.py:191)

现象：

- 脚本直接扫描 `train/log/aisrv` 下所有历史日志
- 没有按 session 或时间边界隔离

影响：

- 导出的 `TRAINING_DATA.json` 容易混入旧 run 数据
- 属于低风险但高误导性的分析脚本问题

结论：

- 这是低优先级分析口径问题

### P3-04 `benchmark_report.py` 基于当前工作树代码分析历史 archive，结果不可复现

位置：

- [train/benchmark_report.py](/home/user/TcKaiwuFinal/train/benchmark_report.py:20)
- [train/benchmark_report.py](/home/user/TcKaiwuFinal/train/benchmark_report.py:25)

现象：

- 脚本在分析 `train/archive/<run_id>` 时
- 不是用归档时对应版本的分析逻辑
- 而是把当前工作树的 `archive_analysis` 注入 `sys.path` 后执行

影响：

- 历史 run 的分析结果会随当前代码变化
- 不利于做严肃的历史对比

结论：

- 这是低优先级可复现性问题

### P1-14 benchmark 指定 checkpoint 不存在时会静默回退到另一份模型，但结果元数据仍写成原 checkpoint

位置：

- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:282)
- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:306)
- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:358)
- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:1802)
- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:1804)
- [code/agent_ppo/agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py:597)
- [code/agent_ppo/eval/benchmark_parallel.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark_parallel.py:339)
- [code/agent_ppo/eval/benchmark_parallel.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark_parallel.py:425)

现象：

- benchmark 入口先解析并记录目标 checkpoint
- 如果文件不存在，真正加载时会 silently fallback 到 `agent.load_model(id="latest")`
- 但 manifest/result 仍保留原先那个 checkpoint 字符串

影响：

- 日志和结果文件写的是 A
- 实际跑的模型却可能是 B
- A/B benchmark 比较会直接失真

结论：

- 这是评测正确性高优先级问题

### P1-15 `_run_eval_episode()` 在灾备分支中可能使用未初始化或过期的 `terminated/truncated`

位置：

- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:453)
- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:457)
- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:829)

现象：

- `handle_disaster_recovery(env_obs, logger)` 触发后立即 `break`
- 但 `terminated/truncated` 要在后面才赋值
- 循环结束后又会用这两个变量去推断 `fail_reason`

影响：

- 如果第一步就触发灾备，可能直接引用未定义变量
- 如果中途触发，可能复用上一 step 的旧值
- 终局归因会错

结论：

- 这是 benchmark episode runner 正确性高优先级问题

### P1-16 benchmark `overall` 口径缺少关键字段，下游会把缺失值按 0 参与决策

位置：

- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:918)
- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py:998)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:804)
- [code/agent_ppo/eval/lite_benchmark_bootstrap.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/lite_benchmark_bootstrap.py:200)
- [code/agent_ppo/workflow/checkpoint_score.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/checkpoint_score.py:162)
- [code/agent_ppo/workflow/checkpoint_score.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/checkpoint_score.py:217)

现象：

- `per_round` 里有 `battery_fail_rate` / `collision_fail_rate`
- 但 `overall` 缺少一批训练主线和下游决策会读的字段
- lite benchmark / checkpoint scoring 又直接从 `overall` 取这些值

影响：

- 缺失字段被静默按 `0` 参与 stage 选择和 checkpoint 打分
- benchmark-based curriculum/bootstrap/scoring 会被错误乐观或错误悲观地驱动

结论：

- 这是 benchmark 聚合口径高优先级问题

### P2-12 并行 benchmark 最终 `result.json` 会把实际运行 rounds 覆盖成代码默认 `ROUNDS`

位置：

- [code/agent_ppo/eval/benchmark_parallel.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark_parallel.py:107)
- [code/agent_ppo/eval/benchmark_parallel.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark_parallel.py:300)
- [code/agent_ppo/eval/benchmark_parallel.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark_parallel.py:429)

现象：

- 任务生成阶段按 `_rounds_and_maps()` / manifest 运行
- finalize 时又把 `rounds` 写回成 `benchmark_mod.ROUNDS`

影响：

- 如果运行时通过 env 覆盖了 rounds
- 结果文件里的 scenario 描述会和实际跑过的任务不一致

结论：

- 这是并行 benchmark 元数据一致性问题

### P2-13 lite benchmark 缓存只按 `checkpoint_path` 命中，无法识别同一路径新内容或 benchmark 口径变化

位置：

- [code/agent_ppo/eval/lite_benchmark_bootstrap.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/lite_benchmark_bootstrap.py:112)
- [code/agent_ppo/eval/lite_benchmark_bootstrap.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/lite_benchmark_bootstrap.py:195)

现象：

- 缓存命中只比较 `checkpoint_path`
- 没有记录 mtime/hash/schema/rounds/policy mode

影响：

- `model.ckpt-resume.pkl` 这类固定路径只要内容更新，缓存也可能被错误复用
- 调整 lite rounds/阈值后也可能继续读旧结果

结论：

- 这是 lite benchmark 缓存一致性问题

### P2-14 `resume_best.py prepare` 只替换 pkl，不同步 resume state / curriculum sidecar

位置：

- [train/resume_best.py](/home/user/TcKaiwuFinal/train/resume_best.py:109)
- [train/resume_best.py](/home/user/TcKaiwuFinal/train/resume_best.py:170)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:559)
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py:612)

现象：

- `prepare` 只拷 `model.ckpt-resume.pkl`
- 不同步 `model.ckpt-resume.state.json`
- 也不处理 `curriculum_state.resume_snapshot.json`

影响：

- 容易出现“权重来自 A，resume 元数据来自 B”的错配
- 恢复后的 `global_step_since_resume`、课程状态、learning metrics 都可能对不上

结论：

- 这是中优先级恢复工具链问题

### P2-15 `ArchiveAgent` 后处理链当前可能根本没有被任何编排拉起

位置：

- [code/agent_ppo/utils/archive_agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/utils/archive_agent.py:300)

现象：

- 代码里存在 `ArchiveAgent` 及其 `main()`
- 但当前仓库中没有看到明确的 compose / 脚本 / workflow 引用去启动它

影响：

- `summary.json`
- `checkpoint_ranking.md`
- raw log 压缩等 archive 后处理产物  
可能根本不会生成

结论：

- 这是中优先级归档链完整性问题

### P3-05 `determine_effective_slot_count()` 在 env/agent 为空时仍至少返回 1，启动期 guard 形同虚设

位置：

- [code/agent_ppo/eval/benchmark_parallel.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark_parallel.py:225)
- [code/agent_ppo/eval/benchmark_parallel.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark_parallel.py:165)

现象：

- slot 计数函数在 env/agent 为空时仍会返回至少 `1`
- 后续马上按这个 slot 去索引 `envs[slot_index]` 和 `agents[slot_index]`

影响：

- 启动保护逻辑名义存在
- 但边界情况下并不能真正拦住非法状态

结论：

- 这是低优先级启动期 guard 缺口

### P2-16 当前 benchmark / routing / zmq 测试主要覆盖工具函数，不覆盖本次已确认的高优先级回归

位置：

- [code/tests/test_benchmark_parallel.py](/home/user/TcKaiwuFinal/code/tests/test_benchmark_parallel.py:29)
- [code/tests/test_container_routing.py](/home/user/TcKaiwuFinal/code/tests/test_container_routing.py:23)
- [code/tests/test_zmq_patch.py](/home/user/TcKaiwuFinal/code/tests/test_zmq_patch.py:23)
- [code/tests/test_ltsppo_contracts.py](/home/user/TcKaiwuFinal/code/tests/test_ltsppo_contracts.py:479)

现象：

- `test_benchmark_parallel.py` 主要测 task queue / reclaim / slot counting
- `test_container_routing.py` 主要测 GPU/容器路由工具
- `test_zmq_patch.py` 主要测 patch 注入是否幂等
- `test_ltsppo_contracts.py` 覆盖了部分 reward/curriculum contract，但没有覆盖：
  - `latest` 路径下 actor step 传播
  - collision/unknown terminal penalty 是否进 learner batch
  - invalid gradient 在 AMP off 时是否真的 skip
  - benchmark missing checkpoint fallback 与结果元数据一致性
  - multi-helper resume sidecar 并发覆盖

影响：

- 当前多个 P1/P2 可以直接穿过测试网
- CI 通过并不能说明主链正确性成立

结论：

- 这是测试覆盖缺口，中优先级

## 当前审查结论

截至本文件写入时，已经确认：

- 至少有 3 条会影响训练正确性的高优先级回归
- 至少有 1 条会影响监控与验证的中优先级问题

因此当前代码状态不应视为“可直接发布”。

## 后续审查方向

后续继续重点覆盖：

1. reward / curriculum / planner 链
2. resume / sidecar / signal / dashboard 链
3. tests / config / docker env 一致性
