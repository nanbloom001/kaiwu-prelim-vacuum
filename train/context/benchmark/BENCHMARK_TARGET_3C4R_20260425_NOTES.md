# Fixed Target Benchmark Notes 20260425

## 背景

本记录沉淀当前固定 benchmark 口径、运行方式、遇到的问题和已采取的修复。该 benchmark 用于评估模型在固定难度下的稳定表现，并为 AI 诊断训练/策略问题提供足够结构化的日志。

Canonical 成功口径：

- 官方地图 `1-10`
- 每张地图 `3` 轮，共 `30 episodes`
- `max_step=1000`
- `battery_max=150`
- `charger_count=3`
- `robot_count=4`
- runner 默认 `serial`；serial-30 是最终 900+ 成功声明的唯一 canonical 口径
- policy mode 默认 `eval`

推荐调用：

```bash
bash train/run_target_benchmark_900.sh --runner serial
```

dry-run 校验：

```bash
bash train/run_target_benchmark_900.sh --dry-run --runner serial

python3 train/tools/validate_target_benchmark_manifest.py \
  .sisyphus/evidence/benchmark-900/task-1-dry-run-manifest.json \
  --episodes 30 \
  --rounds-per-map 3 \
  --maps 1,2,3,4,5,6,7,8,9,10 \
  --charger-count 3 \
  --robot-count 4 \
  --max-step 1000 \
  --battery-max 150
```

Operational/noncanonical 并行诊断口径仍保留，但必须显式选择：

```bash
bash train/run_target_benchmark_900.sh --profile target-parallel --runner parallel --max-wait 1800
```

该口径使用 maps 1-10 × 4 = 40 episodes 和默认 `4 AISRV × 10 gamecore = 40` 拓扑，只能用于运行效率/诊断回归；没有 matching serial-30 结果时不能作为最终成功证据。

## 本次问题与解决

### 1. 固定难度口径没有独立入口

问题：原有并行 benchmark 支持自定义 rounds，但固定 `1000 / 150 / 3 / 4 / 每图3轮` canonical 口径，以及额外 operational parallel-40 口径，需要人工传环境变量或临时改配置，容易在多轮对比时混入口径差异。

解决：`train/run_target_benchmark_900.sh --profile target` 固定选择 canonical `target_3c4r_1000_150_30`；如需 40 局并行诊断，必须显式使用 operational/noncanonical `--profile target-parallel`。该入口 dry-run 会生成 manifest，正式运行会调用所选 runner。

### 2. `4×10` 并行配置曾被旧容器状态干扰

问题：曾观察到旧 `kaiwu-train-*` 容器残留，容器环境仍是旧的 `32 gamecore / 8 envs_per_aisrv` 口径，导致看起来“指定 4 AISRV、每个 10 gamecore 没生效”。

解决：运行前清理旧 benchmark/训练栈，并在 `train/run_benchmark_parallel.sh` 中加入启动后断言，检查容器环境变量和 TOML：

- `KAIWU_BENCHMARK_PARALLEL_MODE=1`
- `KAIWU_AISRV_NUM=4`
- `KAIWU_GAMECORE_NUM=40`
- `KAIWU_PARALLEL_ENV_PER_AISRV=10`
- `KAIWU_BENCHMARK_WORKER_COUNT=4`
- `KAIWU_BENCHMARK_ENVS_PER_WORKER=10`
- `aisrv_connect_to_kaiwu_env_count = 10`

后续完整运行确认：4 个 AISRV、40 个 gamecore、40 个 logical worker 均实际生效。

### 3. Host 侧日志不是运行中实时目录

问题：benchmark 运行时，host 上的 `train/eval_parallel_logs/<session>/` 通常要到脚本收尾复制后才完整出现；运行中如果直接在 host 查任务目录，可能误判为没有日志。

解决：运行中状态应从容器内 runtime 目录查看：

```bash
docker exec kaiwu-train-aisrv-1 bash -lc \
  'find /workspace/train/benchmark_runtime/<session>/tasks -maxdepth 3 -type f'
```

脚本完成后，结果会复制到 host：

- `train/eval_parallel_logs/<session>/result.json`
- `train/eval_parallel_logs/<session>/ai_summary.json`
- `train/eval_parallel_logs/<session>/episodes/*.jsonl`
- `train/eval_parallel_results.json`

### 4. 日志足够记录行为，但 AI 诊断摘要不够直接

问题：原始 step jsonl 已经有动作、模式、电量、slack、planner 对齐、奖励分解和异常标签，但 AI 每次诊断仍要从大量 step 中重新推断关键转折点和奖励冲突。

解决：benchmark schema 升到 `5`，新增面向 AI 诊断的结构化字段：

- step 级：`battery_ratio`、`charge_need_zone`、`just_charged`、`risk_worsening_while_cleaning`
- episode 级：`phase_events`
- episode 级：`charge_timing_summary`
- reward attribution：`positive_reward_sum_mean`、`negative_reward_sum_mean`、`positive_total_reward_rate`、`conflict_score`
- `ai_summary.json`：`diagnosis_cards`

这些字段只用于 benchmark/eval 日志，不参与训练、不改变 reward、不改变 action。

### 5. `diagnosis_cards.first_evidence_window` 指针不够精确

问题：首次实现时，诊断卡会取第一个非空 evidence window。某些 issue 例如 `wall_hugging` 或 `low_value_revisit` 可能被指向 `first_late_return_window`，虽然窗口非空，但不是该 issue 最直接证据。

解决：为 issue 增加专属 evidence window 映射，例如：

- `low_value_revisit` → `first_low_value_revisit_window`
- `wall_hugging` → `first_wall_hugging_window`
- `missed_charge_opportunity` → `first_missed_charge_window`
- `return_stall` → `first_return_stall_window`
- `battery_fail` → `last_battery_fail_window`

并补充缺失窗口。该修复已通过单元测试；在修复之后的新 benchmark 产物中会体现更准确的 evidence 指针。

### 6. 长尾 episode 仍然存在，但不是并行配置失败

现象：完整 benchmark 中经常最后停在 37/40 或 39/40，主要由 map2/map4 的长 episode 拖尾。

已确认：worker 日志持续推进，例如 `target_round_2/map4` 能从 step 400、500、600、700、800 继续前进，不是死锁。该现象不影响 benchmark 结果有效性，也不属于本次“AI 诊断日志”优化范围。

记录：

- `20260425-155300`：`WR=70%`，`Avg CS=386.4`，内部耗时 `978.4s`
- `20260425-173603`：`WR=65%`，`Avg CS=395.5`，内部耗时 `1310.9s`，脚本墙钟 `1375s`

## 当前验证命令

```bash
cd code
/home/user/TcKaiwuFinal/.conda-numpy/bin/python -m unittest \
  tests.test_benchmark_diagnostics \
  tests.test_benchmark_parallel \
  tests.test_summarize_benchmark_failures

/home/user/TcKaiwuFinal/.conda-numpy/bin/python -m py_compile \
  agent_ppo/eval/benchmark.py \
  tests/test_benchmark_diagnostics.py
```

已验证：

- `tests.test_benchmark_diagnostics`
- `tests.test_benchmark_parallel`
- `tests.test_summarize_benchmark_failures`
- `benchmark.py` 和诊断测试文件语法编译

## 使用注意

- 不要与正常训练栈同时运行，benchmark 复用 `kaiwu-train` compose project。
- 对比模型时必须固定同一 profile、同一并行拓扑和同一 policy mode。
- AI 诊断优先读取 `ai_summary.json` 的 `top_anomalies`、`diagnosis_cards` 和 `reward_attribution`，再进入对应 episode 的 `evidence_windows` 和 `episodes/*.jsonl`。
- 如果要验证并行是否生效，优先看 `result.json.execution` 中的 `logical_worker_count`、`effective_envs_per_aisrv`、`completed_task_count`，以及容器环境断言输出。
