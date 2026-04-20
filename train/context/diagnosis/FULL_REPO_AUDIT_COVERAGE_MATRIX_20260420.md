# Full Repo Audit Coverage Matrix 2026-04-20

本文件记录本次“全分支代码审查”的覆盖矩阵。

说明：

- `已深审`：已经进入高风险主审，发现或排除过关键问题
- `已补审`：已纳入边缘/补漏审查，但不是高风险主线
- `待补审`：仍需继续补证据或复核
- 运行态产物、日志、checkpoint、signals 不计入“代码覆盖文件”，但可作为证据源

## 主线高风险模块（已深审）

### Training Core

- `code/agent_ppo/agent.py`
- `code/agent_ppo/algorithm/algorithm.py`
- `code/agent_ppo/model/model.py`
- `code/agent_ppo/feature/definition.py`
- `code/agent_ppo/feature/preprocessor.py`
- `code/agent_ppo/feature/expert.py`

### Workflow / Curriculum / Resume

- `code/agent_ppo/workflow/train_workflow.py`
- `code/agent_ppo/workflow/curriculum_policy.py`
- `code/agent_ppo/workflow/curriculum_state.py`
- `code/agent_ppo/workflow/preload_checkpoint.py`
- `code/agent_ppo/workflow/checkpoint_score.py`

### Reward / Constraint / Monitoring Semantics

- `code/agent_ppo/utils/reward_schedule.py`
- `code/agent_ppo/utils/reward_metrics.py`
- `code/agent_ppo/utils/constraint_utils.py`
- `code/agent_ppo/conf/conf.py`
- `code/agent_ppo/conf/monitor_builder.py`
- `train/local_monitor_dashboard.py`

### Container / Env Wiring

- `train/.docker-compose.yaml`
- `train/.env`

### Tests (主线相关)

- `code/tests/test_curriculum_and_checkpoint_score.py`
- `code/tests/test_ltsppo_contracts.py`
- `code/tests/test_runtime_optimizations.py`
- `code/tests/test_algorithm_stability.py`

## 评测与运维链（部分已深审，部分待补审）

### Eval / Benchmark

- `code/agent_ppo/eval/benchmark.py` — 已深审
- `code/agent_ppo/eval/benchmark_parallel.py` — 已深审
- `code/agent_ppo/eval/lite_benchmark_bootstrap.py` — 已深审

### Archive / Runtime Utilities

- `code/agent_ppo/utils/experiment_archive.py` — 已深审
- `code/agent_ppo/utils/archive_agent.py` — 已补审
- `code/agent_ppo/utils/archive_analysis.py` — 已补审
- `code/agent_ppo/utils/container_routing.py` — 已补审
- `code/agent_ppo/utils/model_signer.py` — 已补审
- `code/agent_ppo/utils/policy_sampling.py` — 已补审
- `code/agent_ppo/utils/zmq_patch.py` — 已补审

### Benchmark / Ops Scripts

- `train/run_benchmark.sh` — 已深审
- `train/run_benchmark_parallel.sh` — 已深审
- `train/benchmark_report.py` — 已补审
- `train/compare_benchmarks.py` — 已补审
- `train/resume_best.py` — 已深审
- `train/run_env_scaling_experiment.py` — 已补审
- `train/run_datafetch_benchmark.py` — 已补审
- `train/run_replay_stability_experiments.py` — 已补审
- `train/run_speed_experiments.py` — 已深审
- `train/collect_data.py` — 已补审

## 配置与框架接线（待补审）

- `code/conf/algo_conf_robot_vacuum.toml` — 已补审
- `code/conf/app_conf_robot_vacuum.toml` — 已补审
- `code/conf/configure_app.toml` — 已补审
- `code/kaiwu.json` — 已补审
- `code/train_test.py` — 已补审

## agent_diy / 边缘代码（补漏审查）

说明：当前判断 `agent_diy` 不是主运行线，但仍然纳入全覆盖，防止误接主线。

- `code/agent_diy/agent.py` — 已补审
- `code/agent_diy/algorithm/algorithm.py` — 已补审
- `code/agent_diy/model/model.py` — 已补审
- `code/agent_diy/feature/definition.py` — 已补审
- `code/agent_diy/workflow/train_workflow.py` — 已补审
- `code/agent_diy/conf/conf.py` — 已补审
- `code/agent_diy/conf/monitor_builder.py` — 已补审
- `code/agent_diy/conf/train_env_conf.toml` — 已补审

## 其他测试（待补审）

- `code/tests/test_benchmark_parallel.py` — 已补审
- `code/tests/test_container_routing.py` — 已补审
- `code/tests/test_zmq_patch.py` — 已补审

## 当前 worktree 中也应纳入审查的新代码

这些文件虽然当前未纳入 `git ls-files`，但属于本次运行逻辑的一部分，也必须算 in-scope：

- `code/agent_ppo/utils/reward_schedule.py`
- `code/agent_ppo/utils/reward_metrics.py`
- `code/agent_ppo/utils/constraint_utils.py`
- `code/tests/test_algorithm_stability.py`

## 非代码覆盖对象，但必须作为证据源

- `code/curriculum_state.json`
- `code/curriculum_state.resume_snapshot.json`
- `code/model.ckpt-resume*.json`
- `code/curriculum_signals/**`
- `code/resume_snapshots/**`
- `code/saved_models/**`
- `train/log/**`
- `train/eval_logs/**`
- `train/eval_parallel_logs/**`
