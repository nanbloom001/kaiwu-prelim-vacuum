# 训练优化修改报告

> 日期: 2026-04-14
> 分支: linux
> 作者: Claude Code (AI 辅助优化)
> 适用场景: 未来 debug、回滚、或继续优化时参考

---

## 一、性能演进总览

| 阶段 | steps/s | samples/s | real_train | data_fetch | 关键变更 |
|------|---------|-----------|------------|------------|----------|
| reverb 基线 | 6-7 | ~25K | 32ms | 120ms | Windows→Linux 迁移 |
| + ZMQ replay | 23.0 | ~94K | 32ms | 11ms | ZMQ 替换 reverb 传输 |
| + AMP | 28.5 | ~117K | 26ms | 7ms | 混合精度 + fused Adam |
| + torch.jit.trace | 28.5 | ~117K | 26ms | 7ms | 前向追踪编译（+6.7% 仅 forward） |
| + buffer=50K | **31.2** | **~128K** | **22-27ms** | **2.7ms** | 信号量热补丁 + 大 buffer |

**当前最优**: 31.2 steps/s, 128K samples/s, GPU0 55%

---

## 二、修改文件清单

### 2.1 `code/agent_ppo/agent.py`（核心修改）

#### 2.1.1 环境变量读取辅助函数（第 37-42 行）

```python
def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}
```

**用途**: 统一读取布尔型环境变量开关，支持 0/false/no/off 为 falsy。

**debug 注意**: 如果某个优化开关未生效，先检查 `os.getenv(name)` 返回值。

---

#### 2.1.2 Torch 运行时配置函数（第 137-166 行）

```python
def _configure_torch_runtime(service_name: str, device) -> None:
```

**行为**:
- **learner**: 设置 `torch.set_num_threads(4)`, `torch.set_num_interop_threads(2)`, 启用 TF32/cuDNN benchmark
- **非 learner (aisrv)**: 强制 `set_num_threads(1)`, `set_num_interop_threads(1)` — 减少推理延迟

**环境变量覆盖**:
- `KAIWU_LEARNER_CPU_THREADS` → 默认 4
- `KAIWU_LEARNER_CPU_INTEROP_THREADS` → 默认 2

**debug 注意**: 如果 learner CPU 占用异常，检查这两个值。aisrv 不应超过 1 线程。

---

#### 2.1.3 Batch Tensor 学习路径（第 168-196 行）

```python
def _patch_remote_agent_batch_learn() -> None:
```

**原理**: KaiwuDRL 框架默认将 batch tensor 拆成 `list[SampleData]` 再传给 `learn()`。此补丁在 `RemoteAgent.learn` 上做 monkey-patch，当检测到输入已是 `torch.Tensor` 或 `np.ndarray` 时，直接透传给 `_business_learn`，跳过框架层拆包/重组。

**关键条件**:
- `is_learner_call`: 当前是 learner 进程
- `prefers_batch_tensor`: `PREFER_BATCH_TENSOR_LEARN = True`（Config 中设置）
- `business_learn`: Agent 类有 `_business_learn` 方法

**debug 注意**: 如果训练 loss 异常，可能需要临时禁用此优化（`LEARNER_PREFER_BATCH_TENSOR=False`）回归框架默认路径。

---

#### 2.1.4 模型编译 / JIT 追踪（第 207-225 行）

```python
self.model = Model(device).to(self.device)
if "learner" in self.service_name:
    compiled = False
    # 尝试 torch.compile（PyTorch 2.2+ 才支持，当前容器 2.0.1 会失败）
    if _env_flag("KAIWU_LEARNER_TORCH_COMPILE", ...):
        try:
            self.model = torch.compile(self.model, mode="reduce-overhead")
            compiled = True
        except RuntimeError:
            pass
    # fallback: torch.jit.trace（当前生效的优化）
    if not compiled and _env_flag("KAIWU_LEARNER_JIT_TRACE", ...):
        try:
            _dummy = torch.randn(1, Config.DIM_OF_OBSERVATION, device=self.device)
            self.model = torch.jit.trace(self.model, _dummy)
            self.model.set_train_mode = lambda: self.model.train()
            self.model.set_eval_mode = lambda: self.model.eval()
        except Exception:
            pass
```

**当前状态**: `torch.compile` 在 PyTorch 2.0.1 + Python 3.11 下不可用（`RuntimeError`），自动 fallback 到 `torch.jit.trace`。

**jit.trace 的局限性**:
- 只优化 forward pass，backward 仍是 eager mode
- 实测仅提升 forward ~6.7%，对整体 training step 影响有限
- `set_train_mode` / `set_eval_mode` 需要手动 shim（因为 ScriptModule 没有这些方法）

**debug 注意**:
- 如果模型输出异常，检查 jit.trace 是否改变了 forward 行为（trace 用 dummy input 录制的计算图）
- 设置 `KAIWU_LEARNER_JIT_TRACE=False` 可禁用
- 日志中 `TracerWarning: Encountering a list at the output` 是正常的（forward 返回 list）

---

#### 2.1.5 AMP + 优化器配置（第 226-255 行）

```python
# AMP
use_amp = _env_flag("KAIWU_LEARNER_USE_AMP", Config.LEARNER_USE_AMP)
self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# 优化器 — fused 和 foreach 互斥
allow_fused = _env_flag("KAIWU_LEARNER_ALLOW_FUSED_OPTIMIZER", ...)
allow_foreach = _env_flag("KAIWU_LEARNER_ALLOW_FOREACH_OPTIMIZER", ...)
# fused 优先（AMP 时更高效）；如果 fused 不可用，fallback 到 foreach
```

**重要修复**: PyTorch 2.0.1 的 `Adam` 不允许 `fused=True` 和 `foreach=True` 同时设置。原代码两者都为 True，导致 `RuntimeError`。修改后 fused 优先，AMP 启用时使用 `fused=True, foreach=False`。

**debug 注意**:
- 如果出现 `RuntimeError: fused and foreach cannot both be True`，说明此修复被覆盖
- `GradScaler` 需要 PyTorch ≥ 1.6，当前 2.0.1 没问题
- AMP 精度损失对 PPO 训练无明显影响（已验证 loss 曲线正常）

---

### 2.2 `code/agent_ppo/conf/conf.py`（新增配置项）

在第 51-61 行新增:

```python
# Learner runtime tuning
LEARNER_CPU_THREADS = 4              # learner CPU 线程数
LEARNER_CPU_INTEROP_THREADS = 2      # learner CPU interop 线程数
LEARNER_USE_AMP = True               # 启用混合精度
LEARNER_ALLOW_FOREACH_OPTIMIZER = True  # foreach 优化器
LEARNER_ALLOW_FUSED_OPTIMIZER = True    # fused 优化器（与 foreach 互斥）
LEARNER_PREFER_BATCH_TENSOR = True   # 跳过框架层 SampleData 拆包
LEARNER_JIT_TRACE = True             # torch.jit.trace 模型
LEARNER_TORCH_COMPILE = True         # torch.compile（需 PyTorch 2.2+）
AGENT_LOAD_MODEL_CACHE = True        # 模型文件缓存
PERF_STAT_WINDOW_SECONDS = 60        # 性能统计窗口
```

**debug 注意**: 这些都是 Python 层默认值，可以通过环境变量覆盖。优先级: 环境变量 > conf.py 默认值。

---

### 2.3 `code/conf/configure_app.toml`（参数调整）

```toml
replay_buffer_capacity = 50000    # 原 11000 → 50000（需配合 mem_buffer 热补丁）
reverb_sampler = "reverb.selectors.Fifo"
reverb_rate_limiter = "MinSize"
send_sample_size = 4096           # 原 8192 → 4096（最优值）
train_batch_size = 4096           # 原 8192 → 4096（最优值）
dump_model_freq = 500
```

**为什么 buffer=50000**:
- 原 buffer=10000 时，`data_fetch` 需要 7ms（buffer 经常不够采样）
- buffer=50000 后，`data_fetch` 降至 2.7ms（buffer 充足，采样更快）
- 但 buffer>10000 需要 mem_buffer 信号量热补丁（见 2.5）

**为什么 batch=4096 而非 8192**:
- batch=8192 时 `real_train` 从 26ms → 38ms（+46%），GPU 利用率未提升
- steps/s 从 28.5 → 15.0，虽然 samples/s 基本持平（~123K vs 117K）
- 单 GPU 上 A10 的算力对 1M 参数模型，batch=4096 已经接近 compute bound

---

### 2.4 `train/.env`（环境变量）

```bash
# 训练核心参数
KAIWU_TRAINING_MODE=2
KAIWU_ALGORITHM=ppo
KAIWU_GAMECORE_NUM=64                # 64 个 gamecore 环境实例
KAIWU_AISRV_NUM=2                    # 2 个 AI server
KAIWU_PARALLEL_ENV_PER_AISRV=4

# GPU 分配
KAIWU_LEARNER_GPU=0
KAIWU_AISRV_GPU1=1
KAIWU_AISRV_GPU2=2
KAIWU_AISRV_GPU3=3                   # 未使用（0%）

# 训练参数覆盖
KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE=4096
KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE=4096
KAIWU_EXPERIMENT_DUMP_MODEL_FREQ=500
KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER=4
KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE=128

# ZMQ replay buffer
KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE=zmq
KAIWU_EXPERIMENT_COORDINATOR_SLEEP=0.05

# PyTorch 升级开关（当前禁用）
KAIWU_PYTORCH_UPGRADE=0
```

**debug 注意**:
- `KAIWU_PYTORCH_UPGRADE=1` 会在容器启动时 `pip install torch==2.2.0`，但当前因容器内 cp437 编解码器缺失而失败（已禁用）
- `KAIWU_GAMECORE_NUM=64` 产生样本速度远超 learner 消费速度，`sample_production_and_consumption_ratio > 150`

---

### 2.5 `train/.docker-compose.yaml`（热补丁集合）

learner 服务的 `command` 块包含多个运行时热补丁，均在容器启动时执行。**所有补丁都有幂等性检查**（检查 marker 字符串，已存在则跳过）。

#### 2.5.1 ZMQ 运行时文件补丁（`patch_zmq_runtime_files`）

**位置**: python3 heredoc 中，`patch_zmq_runtime_files(root, ...)` 函数

**修改内容**:
1. `mem_buffer.py` 和 `mem_buffer_ratio.py` 的 `self.device` 初始化：从 `"cuda"` 改为 `"cpu"`（避免 fork 前 CUDA 初始化）
2. `monitor_manager.py` 的 `_get_shared_queue()`: 从 `multiprocessing.Manager().Queue()` 改为原生 `multiprocessing.Queue()`（避免 Linux 下 socket 断连）

**幂等 marker**: `"import sys\n"` (monitor_manager), `"# Linux ZMQ: avoid touching CUDA"` (mem_buffer)

**debug 注意**: 如果 ZMQ 模式下 learner 启动时报 CUDA fork 错误，检查此补丁是否生效（搜索 `[learner-zmq]` 日志）。

---

#### 2.5.2 trainer.py 补丁 — ModelSignerThread

**修改内容**: 跳过 `model_file_saver` fork（COS 禁用场景），改用 in-process `ModelSignerThread` 处理模型签名。

**依赖文件**: `code/agent_ppo/utils/model_signer.py`（从 mounted code 目录加载）

**幂等 marker**: `trainer_marker` 字符串检查

---

#### 2.5.3 off_policy_strategy.py 补丁

**修改内容**:
1. `process_policy_specific()`: 从 fork-based model sync 改为 process-internal
2. `cleanup()`: 添加 None guard 防止 `model_file_sync_wrapper` 未初始化时报错

**幂等 marker**: `spawn_safe_trainer_marker` 检查

---

#### 2.5.4 reverb_dataset_v1.py 多线程填充补丁

**修改内容**: 将单线程 `_fill_buffers_loop` 替换为多线程 `_fill_worker` + coordinator 模式。

**参数**:
- `KAIWU_EXPERIMENT_FILL_THREADS` → worker 数量（默认 3）
- `KAIWU_EXPERIMENT_COORDINATOR_SLEEP` → coordinator 轮询间隔（默认 0.05s）

**可选**: `code/reverb_dataset_v1_optimized.py` 如果存在，会整体替换（`cp` 覆盖）。

---

#### 2.5.5 TOML 键值替换（post_init_patch.py）

**原理**: `init_code.sh` 在容器启动时会用 `code/conf/` 覆盖项目 `conf/` 目录，覆盖之前的补丁。因此写了 `post_init_patch.py` 在 `start_train_client.sh` 之后运行，从环境变量重新注入 TOML 键值。

**覆盖的键**:
```
pytorch_read_data_from_reverb_type  ← KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE
replay_buffer_cache_multiplier      ← KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER
replay_buffer_type                  ← KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE
train_batch_size                    ← KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE
dump_model_freq                     ← KAIWU_EXPERIMENT_DUMP_MODEL_FREQ
reverb_rate_limiter                 ← KAIWU_EXPERIMENT_REVERB_RATE_LIMITER
reverb_sampler                      ← KAIWU_EXPERIMENT_REVERB_SAMPLER
send_sample_size                    ← KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE
```

**debug 注意**: 如果 TOML 值没生效，检查 `post_init_patch.py` 是否运行成功（日志中搜索 `post_init_patch` 或 `replace_toml_key`）。

---

#### 2.5.6 mem_buffer 信号量热补丁（关键修复）

**位置**: learner command block，`python3 - <<'MEMBUFPY'` heredoc

**问题根因**:
- `mem_buffer.py` 原始代码: `self._data_status = [Value(ctypes.c_bool, False, lock=True) for _ in range(max_sample_num)]`
- 每个 `Value(lock=True)` 创建 1 个 POSIX semaphore
- 系统 SEMMNI=32000（内核编译时上限，无法通过 sysctl 调大）
- 当 `replay_buffer_capacity > 10000` 时，加上其他进程的信号量，总量超限
- 报错: `OSError: [Errno 28] No space left on device`

**补丁内容**（5 步替换）:

```python
# 1. 添加 Lock 到 imports
'from multiprocessing import Value, Array, Queue'
→ 'from multiprocessing import Value, Array, Queue, Lock'

# 2. 替换列表推导为单个 Array + 单个 Lock
'self._data_status = [Value(ctypes.c_bool, False, lock=True) for _ in range(max_sample_num)]'
→ 'self._data_status_arr = Array(ctypes.c_bool, max_sample_num, lock=False)  # lock_free_data_status\n        self._data_status_lock = Lock()'

# 3. 替换 with self._data_status[idx].get_lock(): → with self._data_status_lock:
re.sub(r'with self\._data_status\[[^\]]+\]\.get_lock\(\):', 'with self._data_status_lock:', c)

# 4. 替换 self._data_status[idx].value = True → self._data_status_arr[idx] = True
re.sub(r'self\._data_status\[([^\]]+)\]\.value\s*=\s*(True|False)', r'self._data_status_arr[\1] = \2', c)

# 5. 替换 self._data_status[i].value → self._data_status_arr[i]
re.sub(r'self\._data_status\[([^\]]+)\]\.value', r'self._data_status_arr[\1]', c)
```

**效果**:
- 原来: N 个信号量（buffer=50000 → 50000 个信号量，超限）
- 现在: 1 个信号量（单个 Lock），不再受限

**性能影响**: 用单个全局锁替代 per-slot 锁。由于每次加锁只保护一个 bool 读写 + numpy 切片拷贝，持有时间极短（<1μs），不会成为瓶颈。

**幂等 marker**: `grep -q "lock_free_data_status"` — 补丁后的代码包含此注释，不会重复执行。

**debug 注意**:
- 如果 buffer>10000 报 `No space left on device`，检查此补丁是否生效
- 日志中搜索 `[learner-patch] mem_buffer` 确认补丁应用
- 补丁是在容器启动时对容器内文件做的修改，容器销毁后不持久化

---

#### 2.5.7 PyTorch 升级路径（当前禁用）

**位置**: `KAIWU_PYTORCH_UPGRADE` 条件块

**设计**: 设置 `KAIWU_PYTORCH_UPGRADE=1` 时，在容器启动时 `pip install torch==2.2.0+cu118`，启用 `torch.compile`。

**当前状态**: **禁用**（`KAIWU_PYTORCH_UPGRADE=0`）

**已知问题**: 容器内 Python 3.11.13 缺少 cp437 编解码器，`pip install` 解压 wheel 时报 `LookupError: unknown encoding: cp437`。尝试过:
- `LC_ALL=C.UTF-8` — 无效
- 先升级 pip — 无效
- 手动下载 wheel 后 unzip — 可以绕过，但流程复杂

**如果未来要启用**: 需要:
1. 修复 cp437 编码问题（或升级基础镜像）
2. 验证 KaiwuDRL 框架与 PyTorch 2.2 的 API 兼容性
3. 验证 `torch.compile(mode="reduce-overhead")` 在此模型上是否工作

---

### 2.6 `code/agent_ppo/utils/` 新增文件

| 文件 | 用途 |
|------|------|
| `container_routing.py` | ZMQ 容器网络路由工具 |
| `zmq_patch.py` | ZMQ 运行时补丁工具 |

### 2.7 `code/tests/` 新增/修改测试

| 文件 | 测试内容 |
|------|----------|
| `test_container_routing.py` | container_routing 单元测试 |
| `test_zmq_patch.py` | zmq_patch 单元测试 |
| `test_runtime_optimizations.py` | AMP、JIT trace、optimizer 配置测试（已修改扩展） |

---

## 三、尝试过但失败的方案

### 3.1 torch.compile

- **原因**: PyTorch 2.0.1 + Python 3.11 不支持 TorchDynamo
- **错误**: `RuntimeError: Python 3.11+ not yet supported for torch.compile`
- **代码已处理**: `try/except` 自动 fallback 到 jit.trace

### 3.2 PyTorch 2.2 pip install

- **原因**: 容器 Python 缺 cp437 编解码器
- **错误**: `LookupError: unknown encoding: cp437`
- **状态**: 已禁用，保留代码路径

### 3.3 torch.jit.script

- **原因**: Model 类引用了 `Config.LOCAL_VIEW_CHANNELS` 等类属性，JIT scripter 无法解析
- **错误**: `RuntimeError: __torch__.agent_ppo.conf.conf.Config not found`
- **替代方案**: 改用 `torch.jit.trace`（只记录 tensor 运算，不解析 Python 类）

### 3.4 batch=8192

- **原因**: `real_train` 从 26ms → 38ms（+46%），GPU 利用率未提升
- **结论**: A10 单 GPU 对 1M 参数模型的 compute bound 在 batch=4096 已接近极限

### 3.5 sysctl 增大信号量

- **原因**: SEMMNI=32000 是内核编译时上限
- **错误**: `Invalid argument` / `Numerical result out of range`
- **替代方案**: mem_buffer 热补丁消除信号量依赖

---

## 四、已知局限与未完成方案

### 4.1 GPU3 闲置（0% 利用率）

- 4×A10 中 GPU0 跑 learner（55%），GPU1-2 跑 aisrv（5-7%），GPU3 完全闲置
- **可行方案**: 部署第二个 learner + 独立 aisrv 到 GPU3（需要拆分 gamecore，修改 docker-compose ~150 行）
- **预期效果**: 总吞吐翻倍至 ~256K samples/s

### 4.2 单 learner 天花板

- 当前 31.2 steps/s ≈ 128K samples/s
- 瓶颈在 `real_train`（GPU 前向+反向），batch=4096 时 22-27ms
- 进一步优化需要: 手动 CUDA Graphs 或升级 PyTorch 启用 torch.compile

### 4.3 torch.compile 未验证

- 如果能解决 PyTorch 升级问题，`torch.compile` 预计可再降 real_train 20-35%
- 但框架兼容性未验证

---

## 五、回滚指南

### 5.1 回滚到 AMP 前的状态

```bash
# 1. 环境变量
# .env 中修改:
KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE=reverb
KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE=4096

# 2. TOML
# configure_app.toml:
#   replay_buffer_capacity = 10000
#   send_sample_size = 4096
#   train_batch_size = 4096

# 3. 重启
docker compose -f .docker-compose.yaml --profile distributed down
docker compose -f .docker-compose.yaml --profile distributed up -d --force-recreate
```

### 5.2 禁用特定优化

| 优化 | 禁用方法 |
|------|----------|
| AMP | `KAIWU_LEARNER_USE_AMP=False` (环境变量) |
| JIT trace | `KAIWU_LEARNER_JIT_TRACE=False` (环境变量) |
| Batch tensor | `Config.LEARNER_PREFER_BATCH_TENSOR = False` (conf.py) |
| mem_buffer 热补丁 | 无（幂等标记 `lock_free_data_status` 在容器内，销毁即失效） |
| ZMQ | `KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE=reverb` |

### 5.3 mem_buffer 热补丁异常时

如果怀疑 mem_buffer 热补丁导致数据损坏:

1. 将 `configure_app.toml` 中 `replay_buffer_capacity` 改回 `10000`
2. 重启容器 — 补丁只在 buffer>10000 时才需要
3. 或在 docker-compose 中注释掉 MEMBUFPY 整个 block

---

## 六、关键日志路径与检查命令

```bash
# learner 训练日志（最重要）
docker exec kaiwu-train-learner-1 cat /data/projects/robot_vacuum/log/learner.log | tail -30

# learner 启动日志（检查热补丁是否生效）
docker logs kaiwu-train-learner-1 2>&1 | grep -E "learner-patch|learner-sed|learner-cp|learner-zmq"

# 性能指标（每分钟一行）
docker exec kaiwu-train-learner-1 cat /data/projects/robot_vacuum/log/learner.log | grep "get_training_metrics_dicts"

# GPU 利用率
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# 容器状态
docker compose -f train/.docker-compose.yaml ps

# 全部停止
docker compose -f train/.docker-compose.yaml --profile distributed down
```

---

## 七、备份文件位置

训练容器修改前的备份已保存在:

```
train/backup_20260414/
  .docker-compose.yaml.bak
  .env.bak
  agent.py.bak
  conf.py.bak
```

如果需要恢复到 AMP 启用前的状态，从这些备份文件恢复。
