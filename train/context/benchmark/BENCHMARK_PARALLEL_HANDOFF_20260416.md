# 并行 Benchmark 实现与排障交接文档

> 日期: 2026-04-16
> 分支: `linux`
> 状态: 并行 benchmark 已可用，推荐默认配置为 `4×10`

## 一、目标与当前结论

本轮工作的目标，是在**不影响正常训练**的前提下，为现有 benchmark 系统增加一套**并行评估能力**，缩短固定 40 局评估的墙钟时间，并保留可落地的结构化结果。

当前结论：

- 并行 benchmark 已实现，入口为 `train/run_benchmark_parallel.sh`
- workflow 已新增独立分支，只有 `KAIWU_BENCHMARK_PARALLEL_MODE=1` 时才走并行评估
- 默认建议固定使用 `4×10`
- `4×10` 已完整验证：
  - 40 局可正常完成
  - 结果可正确写回 `train/eval_parallel_logs/<session>/result.json`
  - 脚本结束后容器可自动清理干净

本轮完整验证样例：

- session: `20260416-130524`
- 命令: `bash train/run_benchmark_parallel.sh --workers 4 --envs-per-worker 10 --max-wait 1800`
- 脚本总耗时: `108.09s`
- benchmark 内部耗时: `40.2s`
- 结果:
  - WR `67.5%`
  - Avg CS `788.8`
  - Wins `27/40`

## 二、本轮新增/修改的主要文件

### 2.1 新增并行 benchmark runner

- `code/agent_ppo/eval/benchmark_parallel.py`

职责：

- 创建 session runtime 目录
- 生成 40 个固定 benchmark 任务
- 使用文件队列做动态任务分发
- `aisrv-1` 兼 coordinator，负责聚合结果并写 `done.json`
- 所有 worker 从共享 runtime 目录中 claim / complete 任务

### 2.2 workflow 并行模式分流

- `code/agent_ppo/workflow/train_workflow.py`

新增：

- `KAIWU_BENCHMARK_PARALLEL_MODE=1` 时，直接进入并行 benchmark runner
- 不影响正常训练
- 不影响原串行 benchmark 入口

### 2.3 并行 benchmark compose override

- `train/.docker-compose.benchmark.yaml`

用途：

- 对 benchmark 模式单独注入环境变量
- 对 benchmark 使用到的服务覆盖 `restart: "no"`
- 不改动正常训练 compose 语义

### 2.4 并行 benchmark 启动脚本

- `train/run_benchmark_parallel.sh`

职责：

- 设置 worker / envs-per-worker / runtime 目录
- 起 benchmark 所需服务
- 轮询 `done.json`
- 拷回结果
- 脚本结束时自动 cleanup

### 2.5 单元测试

- `code/tests/test_benchmark_parallel.py`

覆盖：

- 任务生成
- claim / complete
- stale claim recovery
- slot 数裁剪逻辑

## 三、实现过程中发现的关键问题

以下问题按时间顺序和影响程度整理。

### 3.1 独立 compose project 名不可用

#### 现象

最开始尝试把并行 benchmark 做成独立项目名，例如 `kaiwu-benchmark`，希望完全与训练栈隔离。

但实际启动失败，框架内部脚本仍然在查找：

- `kaiwu-train-learner-1`
- `kaiwu-train-aisrv-*`

而不是新的 project 名。

#### 根因

Kaiwu 框架启动链内部存在对默认项目名前缀 `kaiwu-train-*` 的强依赖，服务发现不是完全由外层 compose project 名驱动。

#### 解决思路

- 并行 benchmark 仍然复用 `COMPOSE_PROJECT_NAME=kaiwu-train`
- 启动脚本增加保护：
  - 如发现已有 `kaiwu-train-*` 容器在运行，则拒绝启动 benchmark
  - 避免与正常训练并发

#### 当前结论

这是一个**框架约束**，不是本轮脚本的小 bug。

因此当前 benchmark 方案是：

- 运行隔离通过**独立入口、独立结果目录、独立 runtime 目录**实现
- 而不是通过 compose project 名彻底隔离

---

### 3.2 指定 checkpoint 最初没有真正接到 benchmark 模型加载

#### 现象

最初 `run_benchmark.sh <checkpoint>` 只是在 metadata / 日志层记录了 checkpoint，但实际加载路径仍可能回到 `conf.py` 默认值。

#### 根因

- benchmark 入口与 agent 默认 resume 链路耦合
- checkpoint 参数没有强制覆盖最终 load 行为

#### 解决思路

在串行 benchmark 入口中显式处理：

- checkpoint fallback
- 指定 checkpoint 加载
- 完成标记文件 `.benchmark_done`

这样 benchmark 的“传参”和“实际加载”才一致。

#### 当前结论

串行 benchmark 现在可稳定加载指定 checkpoint，并为并行 benchmark 的加载方式提供了可复用基础。

---

### 3.3 两个 aisrv 同时跑 benchmark，结果重复

#### 现象

串行 benchmark 模式下，两个 `aisrv` 会各自跑一套 benchmark，导致：

- 进度计数超过 `40/40`
- 结果目录重复
- 汇总逻辑互相干扰

#### 根因

benchmark 分支写在 workflow 层，但没有做“只允许一个 evaluator”约束。

#### 解决思路

在串行 benchmark workflow 中加入：

- 仅 `aisrv-1` 真正执行 benchmark
- 其他 aisrv 等待完成标记后退出

#### 当前结论

该问题已修复，串行 benchmark 可用。

---

### 3.4 并行 benchmark 的真实加速来自“多 aisrv 外层并发”，不是 Python workflow 层显式多 env 句柄

#### 现象

在并行 benchmark manifest 中经常看到：

- `available_env_handles = 1`
- `available_agent_handles = 1`
- `effective_envs_per_worker = 1`

但实际吞吐又明显高于 `1×1`。

#### 根因

Kaiwu 框架在容器内部存在自己的多环境并发调度，不一定以多组 `env` / `agent` Python 句柄形式暴露给 workflow 层。

#### 结论

更准确的理解是：

- 一个 `aisrv` 容器**可以有效承载多路环境推进**
- 但当前 workflow 并没有显式拿到多组 `env/agent` 句柄

因此 benchmark 设计上不应假设：

- “Python 看到 1 个 env handle” 就等于容器只能串行跑 1 局

---

### 3.5 8 个 aisrv 只有 3-4 个 CPU 很高，其余基本空转

#### 现象

在 `8×1` 早期测试中，CPU 分布很不均匀：

- 少数 `aisrv` CPU 很高
- 其余大多在 `10%` 左右

#### 根因

并行 benchmark 脚本最初只改了：

- `KAIWU_AISRV_NUM`
- `KAIWU_GAMECORE_NUM`

但**没有同步覆盖**：

- `KAIWU_PARALLEL_ENV_PER_AISRV`

结果容器内仍然沿用了 `.env` 默认值 `4`，导致：

- 外层以为在跑 `8×1`
- 框架内层仍按 `4 env per aisrv` 去绑定 gamecore
- 负载分配畸形

#### 修复

在 `train/run_benchmark_parallel.sh` 中显式导出：

```bash
export KAIWU_PARALLEL_ENV_PER_AISRV="${ENVS_PER_WORKER}"
```

#### 验证

修复后：

- `aisrv` 启动日志只绑定预期数量的 gamecore
- `8×1` 的 CPU 分布恢复到较均匀状态

#### 当前结论

`KAIWU_PARALLEL_ENV_PER_AISRV` 是并行 benchmark 的关键环境变量，必须与脚本参数保持一致。

---

### 3.6 `2×20` 失败不是“速度太慢”，而是启动链路的路由竞态

#### 现象

测试：

```bash
bash train/run_benchmark_parallel.sh --workers 2 --envs-per-worker 20 --max-wait 1800
```

表现：

- 超过 `110s` 仍然 `completed=0/40`
- 没有形成有效结果
- `aisrv` 日志报错：

```text
Failed to resolve aisrv routing ...
[aisrv-routing] failed to resolve container index or GPU target
```

#### 根因链条

1. 脚本启动 benchmark 栈
2. 只等待固定 `5s`
3. 立刻 `docker exec kaiwu-train-aisrv-1 printenv ...`
4. 如果拿到空值，就认为 env 未传播，触发 `--force-recreate`
5. 但在大规模 gamecore 启动下，这个 `5s` 探针很容易误判
6. recreate 过程中产生了 `aisrv-3/4`
7. `container_routing.py` 只按 `service_count=2` 去匹配：
   - `kaiwu-train-aisrv-1`
   - `kaiwu-train-aisrv-2`
8. 因此 `aisrv-3/4` 无法被路由解析，启动失败

#### 结论

`2×20` 当前失败的根因是：

- 脚本启动探针过脆
- recreate 后副本编号漂移
- `container_routing.py` 严格依赖副本名落在 `1..N`

#### 当前状态

- `2×20` **不建议使用**
- 该问题暂未进一步修复，因为当前已决定固定使用 `4×10`

---

### 3.7 benchmark 结束后容器清理不彻底

#### 现象

在多轮测试中，benchmark 跑完后经常仍残留：

- `kaiwu-train-aisrv-*`
- `kaiwu-train-learner-1`

#### 根因

主要有两层：

1. benchmark 脚本原来只做：

```bash
docker compose down
```

没有：

- `--remove-orphans`
- 二次兜底清理

而 benchmark 在不同 worker 数之间切换时，容易留下旧副本。

2. base compose 中部分服务默认带：

- `restart: always`

这使得 benchmark 容器在某些收尾时机会被再次拉起，或至少让 cleanup 更不稳定。

#### 修复

本轮做了两层修复。

第一层：benchmark compose override 中，对 benchmark 用到的服务统一覆盖：

- `pushgateway`
- `backup_model`
- `learner`
- `aisrv`
- `gamecore`

设置：

```yaml
restart: "no"
```

第二层：在 `run_benchmark_parallel.sh` 中新增 `cleanup_stack()`：

- `docker compose down --remove-orphans --timeout 10`
- 再按 `com.docker.compose.project=${COMPOSE_PROJECT_NAME}` label 强制 `docker rm -f`
- 最后尝试删除默认 network
- 并通过 `trap cleanup_stack EXIT` 保证：
  - 正常结束
  - 超时
  - 中途失败

都能执行 cleanup

#### 验证

短 smoke test 和完整 `4×10` benchmark 都已验证：

- 脚本退出后 `docker ps -a | grep '^kaiwu-train-'` 为空

#### 当前结论

cleanup 问题已闭环。

## 四、测速与配置对比结论

### 4.1 已测配置

短时间吞吐测试曾试过：

- `2×1`
- `4×1`
- `8×1`
- `1×4`
- `10×4`

完整 end-to-end 对比重点看了：

- `4×10`
- `10×4`

### 4.2 完整测试结论

#### `4×10`

- session: `20260416-123439`
  - benchmark 内部耗时 `56.1s`
  - WR `57.5%`
  - Avg CS `687.6`
- session: `20260416-130524`
  - benchmark 内部耗时 `40.2s`
  - 脚本总耗时 `108.09s`
  - WR `67.5%`
  - Avg CS `788.8`

#### `10×4`

- session: `20260416-123920`
  - benchmark 内部耗时 `52.1s`
  - 脚本总耗时 `113.17s`
  - WR `72.5%`
  - Avg CS `832.6`

### 4.3 为什么最终固定使用 `4×10`

虽然 `10×4` 在某次 benchmark 内部执行时间上略快，但综合考虑：

- 脚本总耗时
- 容器数量
- 启动/收尾负担
- 配置简单度
- 当前已验证的稳定性

最终更适合作为默认配置的是：

```bash
--workers 4 --envs-per-worker 10
```

注意：

- 不同并发拓扑会对结果本身产生影响
- 因此模型之间做 A/B 比较时，必须固定同一 benchmark 拓扑

## 五、当前推荐使用方式

### 5.1 默认并行 benchmark

```bash
cd train
bash run_benchmark_parallel.sh --workers 4 --envs-per-worker 10 --max-wait 1800
```

### 5.2 使用指定 checkpoint

```bash
cd train
bash run_benchmark_parallel.sh saved_models/<your_model>/model.ckpt-resume.pkl \
  --workers 4 \
  --envs-per-worker 10 \
  --max-wait 1800
```

### 5.3 结果位置

- 详细结果：
  - `train/eval_parallel_logs/<session>/result.json`
- 最新汇总：
  - `train/eval_parallel_results.json`

## 六、未决事项

当前仍有两个未解决但不阻塞默认使用的问题：

### 6.1 启动探针仍偏脆

`run_benchmark_parallel.sh` 目前仍保留：

- 启动后等待固定 `5s`
- 再检查 `aisrv-1` 的 env

虽然对 `4×10` 当前足够，但该逻辑在更重配置下仍可能误判。

后续可改进为：

- 轮询容器状态直到 `docker exec` 可用
- 或检查 benchmark runtime 目录是否已创建

### 6.2 `2×20` 之类极端缩容配置的路由鲁棒性不足

如果未来仍要探索少 worker / 高 env 的配置，需要考虑：

- 减少或消除 `force-recreate`
- 或增强 `container_routing.py` 对副本编号漂移的容忍度

## 七、最终建议

当前建议非常明确：

1. 并行 benchmark 默认固定使用 `4×10`
2. 不再继续尝试 `2×20`
3. A/B 对比时不要混用不同 benchmark 拓扑
4. 当前脚本 cleanup 已修好，可以作为日常评估入口使用

