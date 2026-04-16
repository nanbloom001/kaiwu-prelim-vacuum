# KaiwuDRL 框架训练加速接口分析

> 日期: 2026-04-13
> 来源: 容器镜像 `kaiwu-pub.tencentcloudcr.com/project/robot_vacuum/edu/win_gpu:13.0.1`
> 框架路径: `/data/projects/robot_vacuum/kaiwudrl/`

## 前提：当前配置现状

当前训练瓶颈在 `data_fetch` 方差和较高的 `sample_production_and_consumption_ratio`，业务层 `real_train` 时间已优化至 ~25ms。因此加速重点应放在**样本数据管道**和**on-policy流程效率**上。

当前 `.env` 中的关键覆盖值：
- `KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE=1`（使用 reverb_dataset_v1）
- `KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE=2048`
- `KAIWU_EXPERIMENT_DUMP_MODEL_FREQ=100`
- `KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER=16`
- `KAIWU_EXPERIMENT_REVERB_RATE_LIMITER=MinSize`
- `KAIWU_EXPERIMENT_FILL_THREADS=3`

---

## 一、替换 Replay Buffer 类型（影响最大）

### 框架支持的 5 种 replay buffer 类型

| 类型 | 配置值 | 传输方式 | 特点 | 适用场景 |
|------|--------|----------|------|----------|
| **Reverb** | `reverb` | TCP/gRPC | 框架默认，最稳定但最重 | 当前使用 |
| **ZMQ** | `zmq` | ZMQ + 共享内存 | 零拷贝批量读取，有后台预取 | **首选推荐** |
| **共享内存** | `shared_memory` | POSIX 共享内存 + 信号量 | 动态发现 producer，极低延迟 | 极致低延迟场景 |
| **文件映射** | `file_mmap` | 内存映射文件 | 大容量但非最快 | 样本量极大时 |
| **TF Uniform** | `tf_uniform` | TensorFlow 内部 | 仅 TensorFlow 框架 | 不适用 |

### 为什么切换到 ZMQ 能显著加速

当前 reverb 方案的数据路径：
```
aisrv → [TCP/gRPC序列化] → reverb_server(独立进程) → [TCP/gRPC反序列化] → learner
```

ZMQ 方案的数据路径：
```
aisrv → [共享内存直接写入] → learner(后台预取线程批量读取)
```

关键差异：
1. **reverb** 是独立 C++ 进程，数据经过 TCP/gRPC 序列化和反序列化，有选择器（selector）和限流器（rate limiter）开销
2. **ZMQ** 使用共享内存 `Array`（`multiprocessing.Array`）直接传递 numpy 数组，零序列化
3. `ZmqReplayBuffer.next_by_batch_size()` 已内置**后台预取线程 + 双缓冲队列**，训练线程取到 numpy 后才做 `torch.from_numpy().to(device)`，不阻塞训练
4. 支持 `batch_process_for_batch_manager` 配置预取进程数

### 配置方式

在 `learner.toml` 或 hot-patch 中设置：
```toml
replay_buffer_type = "zmq"
```

**注意**：切换 replay buffer 类型是较大变更，需要验证 on-policy 的样本过滤逻辑（`model_version` 匹配）在 ZMQ 方案下是否正常工作。建议先在小规模实验中验证。

### Reverb Dataset v1 vs v2

当前使用 v1。框架还提供了 v2（`reverb_dataset_v2.py`），v2 比 v1 多了：
- `pin_memory` 支持（CPU tensor 固定内存页，加速 GPU 传输）
- CUDA 异步拷贝流（`torch.cuda.Stream`），实现 H2D 双缓冲流水线

如果你保持 reverb 方案不变，切换到 v2 可能有少量提升。配置：`pytorch_read_data_from_reverb_type = 2`。

---

## 二、提高样本生成吞吐

### `predict_batch_size`（configure.toml）

```
当前值: 32
```

actor 每次预测处理的批量大小。提高此值可以减少通信开销占比，让 actor 更高效地处理样本。

**建议**：提高到 64 或 128。需要同时确保 `proxy_batch_size` 也匹配（当前为 32）。

### `actor_receive_cost_time_ms`（configure.toml）

```
当前值: 1ms
```

actor 等待 aisrv 请求的批处理时间窗口。太短会导致 batch 未满就执行预测（低效空转），太长会增加延迟。

**建议**：提高到 3-5ms，让 actor 有更多时间收集满一个 batch。

### `send_sample_size`（aisrv.toml）

```
当前值: 10000
```

aisrv 积攒多少条样本后一次性发给 learner。在 on-policy 场景下，样本只用一次，10000 远大于 `train_batch_size=2048`，意味着 aisrv 会积攒大量样本才发送，造成不必要的延迟。

**建议**：降低到 `train_batch_size` 的 1-2 倍（如 2048 或 4096），让样本更快到达 learner。

---

## 三、减少数据类型转换开销

### `sample_data_return_data_type`（configure.toml）

```
当前默认: "numpy"
可选值: "numpy" / "tensor"
```

控制 replay buffer 返回数据的类型：
- `"numpy"`：返回 numpy 数组，训练时需要 `torch.from_numpy().float().to(device)` 转换
- `"tensor"`：直接返回 GPU tensor，零转换

**建议**：设为 `"tensor"`。hot-patch 中覆盖此配置即可。

---

## 四、减少模型保存阻塞

### `dump_model_freq`（configure.toml）

```
当前覆盖值: 100（从默认 1000 降低）
```

每 N 步训练保存一次模型。模型保存是**同步阻塞操作**（在训练线程中执行），包含：
1. 调用 `agent.save_model()`（CPU 上 clone state_dict + torch.save）
2. 生成 json 元数据文件
3. 生成 tar.gz 压缩包（`after_save_param`）
4. 维护 `id_list` 和 `file_queue`

频率从 1000 降到 100 意味着保存频率提高了 10 倍，训练阻塞也增加了 10 倍。

**建议**：对于日常训练，200-500 是更合理的值。只在需要精细监控模型演进时才用 100。

### `max_save_model_file_count`（configure.toml）

```
当前默认: 107（质数，避免和 save_model 频率冲突）
```

框架保存的最大模型文件数量（FIFO 淘汰）。107 足够大，不需要调整。

---

## 五、on-policy 流程优化

### on-policy 训练流程耗时分析

每次训练成功后的 on-policy 流程：
1. `replay_buffer.reset()` — 清空样本池
2. `learner_push_model_to_modelpool()` — 推送模型到 modelpool（有重试，最多 `on_policy_error_retry_count_when_modelpool` 次）
3. 发送 `MODEL_VERSION_CHANGE_REQUEST` 给所有 aisrv/actor
4. 等待所有 aisrv/actor 确认加载完毕
5. 心跳保活

### 可调参数

| 参数 | 位置 | 当前值 | 作用 | 建议 |
|------|------|--------|------|------|
| `on_policy_timeout_seconds` | configure.toml | 5 | 单次 on-policy 超时 | 保持不变 |
| `on_policy_error_retry_count` | configure.toml | 10000 | 每轮重试中的等待次数 | 保持不变 |
| `on_policy_error_max_retry_rounds` | configure.toml | 3 | 最大重试轮数 | 保持不变 |
| `on_policy_error_retry_count_when_modelpool` | configure.toml | 10 | modelpool 推送重试次数 | 保持不变 |

on-policy 参数当前值合理，不建议调整。真正的瓶颈在模型保存和样本管道。

---

## 六、CPU 线程与空转控制

### `torch_num_threads`（configure.toml）

```
当前值: 1
```

框架会据此设置以下环境变量（`torch_utils.py`）：
- `OMP_NUM_THREADS`
- `MKL_NUM_THREADS`
- `NUMEXPR_NUM_THREADS`
- `VECLIB_MAXIMUM_THREADS`
- `OPENBLAS_NUM_THREADS`

值为 1 可以最大限度减少 CPU 和 GPU 的资源争用。如果你的 GPU 利用率已接近 100%，1 是正确的。

**建议**：如果 GPU 利用率只有 60-70%，可以尝试提高到 2-4，让 CPU 侧的数据预处理更快。

### `idle_sleep_second` / `idle_sleep_count`（configure.toml）

```
idle_sleep_second = 0.001  # 1ms
idle_sleep_count = 5        # 每 5 次空转 sleep 一次
```

用于 aisrv/actor 进程的空转控制，避免 CPU 100%。learner 的训练循环不受此参数控制（learner 等待在 reverb_dataset 的 `__iter__` 中，使用 `CONFIG.idle_sleep_second`）。

**建议**：保持不变。

---

## 七、其他已确认最优的配置

| 配置项 | 当前值 | 说明 |
|--------|--------|------|
| `aisrv_actor_protocol` | `msgpack` | 已是最快序列化方案（比 pickle 快，protobuf 需要编译 schema） |
| `reverb_rate_limiter` | `MinSize` | on-policy 场景最优（等待足够样本才训练） |
| `reverb_data_cache` | `false` | 正确，PyTorch 场景不需要 TF 数据缓存层 |
| `use_compress_decompress` | `true` | 已启用 lz4 压缩，减少通信量 |
| `algorithm_on_policy_or_off_policy` | `on_policy` | PPO 标准配置 |
| `wrapper_type` | `remote` | 正确，分布式模式 |
| `remote_agent_default_runtime_mode` | `local_aisrv_workflow` | 小规模环境最优（<8 个环境/aisrv，函数调用代替进程间通信） |

---

## 八、暂不适用的接口

| 接口 | 原因 |
|------|------|
| FSDP 分布式训练 (`dist_fsdp_utils.py`) | 当前是单 learner 单 GPU，aisrv 各用一张 GPU，无法利用多 learner DDP |
| `use_pipeline_predict` | 仅 TensorRT 模式支持 |
| `actor_server_async` | 仅 C++ actor 场景 |
| `torch.compile` | 框架未集成，需要在业务层 agent 中手动使用 |
| `enable_mixed_precision` / `max_grad_norm` | 仅 TensorFlow 框架层实现，PyTorch 版本由业务层 `agent.py` 自行控制（你已实现 AMP） |

---

## 九、推荐变更优先级

| 优先级 | 变更 | 预期效果 | 风险 |
|--------|------|----------|------|
| P0 | 切换 `replay_buffer_type` 到 `zmq` | 大幅降低 data_fetch 耗时和方差 | 需验证 on-policy 样本过滤 |
| P1 | 提高 `predict_batch_size`（32→64/128） | 提高样本生成吞吐 | 低 |
| P1 | 降低 `send_sample_size`（10000→2048/4096） | 加快样本到 learner 速度 | 低 |
| P2 | 设置 `sample_data_return_data_type = "tensor"` | 消除 numpy→tensor 转换 | 低 |
| P2 | 提高 `dump_model_freq`（100→200/500） | 减少模型保存阻塞 | 中（减少检查点粒度） |
| P3 | 调整 `actor_receive_cost_time_ms`（1→3-5） | 减少 actor 空转 | 低 |
| P3 | 尝试提高 `torch_num_threads`（1→2） | 加速 CPU 预处理 | 需观察 GPU 利用率 |

**建议实施顺序**：先单独测试 P0（切换 replay buffer），因为这是最可能的瓶颈。P1 和 P2 可以一起调整。
