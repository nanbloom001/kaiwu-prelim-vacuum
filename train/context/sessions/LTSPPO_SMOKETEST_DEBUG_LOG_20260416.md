# LTSPPO Smoke Test Debug Log

日期：2026-04-16  
分支：`linux-LTSPPO`  
目标：验证 LTSPPO 是否已从“代码骨架接通”推进到“Docker / Kaiwu 环境下可真实启动训练”

## 1. 测试目标

本轮 smoke test 的验证目标不是看分数，而是看 LTSPPO 是否已经打通以下整条链路：

1. `aisrv` 正常加载新 observation 与新模型。
2. `train_workflow.py` 能正常收集 step records。
3. `sample_process()` 能生成 recurrent chunk。
4. learner 能接收 chunk sample，不被 replay buffer / mem buffer 拒绝。
5. `algorithm.learn()` 能正确解包 batch，并执行 sequence-aware PPO 反向传播。
6. learner 能正常存模并把 checkpoint 推送给 `aisrv`。

## 2. 涉及的核心改动文件

本轮排查和修复中，直接参与链路打通的文件主要有：

| 文件 | 作用 |
|---|---|
| `code/agent_ppo/conf/conf.py` | LTSPPO 模型、观察、chunk 配置 |
| `code/conf/configure_app.toml` | Kaiwu 框架 replay / preload / batch 吞吐参数 |
| `code/agent_ppo/feature/definition.py` | `SampleData` 与 recurrent chunk 协议 |
| `code/agent_ppo/feature/preprocessor.py` | 新 observation、teacher labels、dual reward |
| `code/agent_ppo/model/model.py` | LTSPPO 多分支 + GRU + dual critic 模型 |
| `code/agent_ppo/agent.py` | 新 runtime、`rnn_state`、batch learn 兼容入口 |
| `code/agent_ppo/workflow/train_workflow.py` | `step-record -> sample_process(chunk)` |
| `code/agent_ppo/algorithm/algorithm.py` | recurrent PPO 解包、loss、反向传播 |

## 3. 测试方法

统一使用当前训练栈：

```bash
cd train
docker compose -f .docker-compose.yaml --profile distributed up -d --force-recreate
```

主要观察 4 类日志：

- `aisrv.log`
- `learner.log`
- `train/log/learner/learner_train_pid*_log_*.log`
- `docker logs kaiwu-train-learner-1`

重点判据：

- `aisrv` 是否开始 `Episode`
- `learner_server` 是否收到样本
- learner 是否从 `train count = 0` 进入真实训练
- 是否能落出 `model.ckpt-500.pkl`

## 4. Round 1：样本过大，mem_buffer 直接拒收

### 4.1 现象

第一轮训练栈启动后：

- `aisrv` 可以进入 episode
- 新模型前向正常
- `learner` 初始化正常

但 learner 日志里出现硬错误：

```text
mem_buffer.append: Sample size 168129 exceeds max_sample_size 100000
```

这意味着 LTSPPO 的 chunk sample 在进入 Kaiwu 的共享内存缓冲区之前就被拒绝，训练不可能开始。

### 4.2 当时的结构背景

LTSPPO 当时使用的是第一版 recurrent 配置：

- `SEQ_CHUNK_LEN = 32`
- `SEQ_BURN_IN = 8`
- `SEQ_STRIDE = 24`

同时 observation 主向量维度已经扩大到：

- `DIM_OF_OBSERVATION = 5224`

于是单个样本的主要体积约为：

| 项 | 估算 |
|---|---:|
| `obs` | `5224 × 32 = 167168` |
| 其他字段 | `~1000` |
| 合计 | `~168129` |

这已经超过框架 `max_sample_size = 100000` 的硬上限。

### 4.3 根因

根因不是代码语法问题，也不是 learner 解包逻辑，而是：

- LTSPPO 把 observation 大幅扩展了
- 仍沿用 `32` 步 chunk
- 导致单个 sample 平铺后超出 Kaiwu 共享内存样本上限

也就是说，这是 **“新 observation 规模 + 旧 chunk 长度”之间的物理不兼容**。

### 4.4 修复

在 `code/agent_ppo/conf/conf.py` 中将 recurrent chunk 缩小到：

| 参数 | 旧值 | 新值 |
|---|---:|---:|
| `SEQ_CHUNK_LEN` | 32 | 16 |
| `SEQ_BURN_IN` | 8 | 4 |
| `SEQ_STRIDE` | 24 | 12 |

这样单个 sample 体积下降到 roughly `84k` 量级，重新回到框架可接受区间。

### 4.5 结果

第二轮启动后，`mem_buffer.append ... exceeds max_sample_size` 错误不再出现。  
这说明 **样本过大问题已经完全解决**。

## 5. Round 2：learner 不报错，但一直不训练

### 5.1 现象

在 chunk 缩小后，learner 虽然不再报错，但日志长期停留在：

```text
input ready size is 50000, train process now train count is 0, global step is 0
```

同时：

- `learner_server` 已经在收样
- `aisrv` 也在正常跑 episode

也就是说：

- 样本生产是正常的
- 但 learner 迟迟不进入真实训练

### 5.2 初始误判

当时一度怀疑是：

- learner 挂了
- replay 断连
- 或者框架对 recurrent sample 还有额外限制

但继续看日志后发现，不是这些。

### 5.3 根因

真正的根因是：`code/conf/configure_app.toml` 仍沿用旧单步 PPO 的吞吐参数。

当时的关键值是：

| 参数 | 旧值 |
|---|---:|
| `replay_buffer_capacity` | 50000 |
| `preload_ratio` | 1.0 |
| `send_sample_size` | 4096 |
| `train_batch_size` | 4096 |

这套参数在单步 sample 时代是合理的，但对 LTSPPO 的 recurrent chunk 来说过大了：

- chunk sample 体积更大
- chunk 产样速度更慢
- learner 必须等 buffer 填到 `50000 × 1.0 = 50000` 才启动训练

这会导致 learner 长时间停在“预热阶段”。

### 5.4 修复

在 `code/conf/configure_app.toml` 中，把吞吐参数改成更适合 recurrent chunk 的量级：

| 参数 | 旧值 | 新值 |
|---|---:|---:|
| `replay_buffer_capacity` | 50000 | 1024 |
| `preload_ratio` | 1.0 | 0.25 |
| `send_sample_size` | 4096 | 512 |
| `train_batch_size` | 4096 | 128 |

### 5.5 一个容易误读的点

learner runtime probe 中仍会打印环境变量：

- `KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE = 4096`
- `KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE = 4096`

但同一条 probe 里的生效配置已经变成：

- `send_sample_size = 512`
- `train_batch_size = 128`

这说明当前最终生效的是 `configure_app.toml` 修改后的参数，而不是环境变量文字本身。  
这个现象很容易让人误以为“配置没生效”，实际上是生效了。

### 5.6 结果

下一轮 learner 不再卡在“永远预热”。  
但新的训练期错误暴露了出来，说明系统已经推进到了下一层。

## 6. Round 3：learner 退出，表面像 stop sentinel，实则是训练异常

### 6.1 现象

调完 `configure_app.toml` 后，learner 容器出现：

- `Exited (0)`

容器启动脚本日志里还能看到：

```text
train success, sending SIGTERM to sigterm_pids processes.
```

以及对 `/data/ckpt/robot_vacuum_ppo/process_stop.done` 的检测逻辑。

### 6.2 初步判断

第一眼看起来像是：

- 历史的 `process_stop.done` 文件残留
- 导致 learner 自己把训练进程停掉

### 6.3 后续核查后的结论

这个判断 **不完整**。

进一步查看宿主机映射出来的详细 learner 训练日志：

- `train/log/learner/learner_train_pid332_log_2026-04-16-17.log`

才发现真正先发生的是训练异常：

```text
AttributeError: 'Tensor' object has no attribute 'obs'
```

随后框架才写出 stop 标记并停止训练进程。

所以：

- `process_stop.done` 不是根因
- 它是 **下游结果**

这点非常重要，因为它说明：

- learner 脚本层面的“正常退出”
- 并不等于业务训练逻辑正常

## 7. Round 4：真正训练期错误，batch 解包协议假设不对

### 7.1 现象

详细 learner 训练日志里，错误稳定复现：

```text
AttributeError: 'Tensor' object has no attribute 'obs'
```

调用链是：

```text
standard_agent_wrapper_pytorch.py -> agent.learn() -> algorithm.learn() -> _unpack_train_batch()
```

LTSPPO 当时的 `_unpack_train_batch()` 假设输入是：

- `list[SampleData]`

并直接做：

```python
for s in list_sample_data:
    s.obs
```

### 7.2 根因

但 Kaiwu 在 learner 真实训练时传进来的并不一定是：

- 纯 Python `SampleData` 对象列表

而是可能是：

- 平铺 batch tensor
- 字段张量列表
- 样本张量序列

也就是说，**框架真正的 batch 协议比实现里假设的更复杂**。

这是 LTSPPO 接入 Kaiwu 时第二个最关键的兼容性问题。

### 7.3 修复策略

修复分两层：

#### A. `agent.py`

补强 remote agent 的 `patched_learn()` 入口，让 learner 场景下不只接受：

- `torch.Tensor`
- `np.ndarray`

还接受：

- `list[Tensor]`
- `tuple[Tensor]`

只要属于 batch tensor 路径，就直接转给新的业务 learn。

#### B. `algorithm.py`

把 `_unpack_train_batch()` 改成同时兼容三类输入：

| 输入形态 | 处理方式 |
|---|---|
| `SampleData` 对象列表 | 逐字段 stack |
| 平铺 batch tensor | 按 `SampleData` 字段顺序切片还原 |
| 字段张量列表 / 样本张量序列 | 统一转成 field map，再构建 batch |

这样就不再依赖 Kaiwu “一定传 Python 对象列表”这一假设。

### 7.4 结果

补完兼容层后，重新启动训练栈，learner 保持运行，没有再次出现：

```text
'Tensor' object has no attribute 'obs'
```

说明 **batch 解包协议已经和框架真实行为对齐**。

## 8. 最终成功状态

第三轮修复完成后，再次启动训练栈并等待约 1 分钟以上，learner 日志中出现了真正的训练推进信息：

```text
train process now input ready size is 1024,
train process now train count is 279,
global step is 279,
train once cost time is 69.62 ms
(data_fetch: 4.33 ms, real_train: 65.27 ms)
```

同时继续出现：

```text
save model ... model.ckpt-500.pkl successfully
push checkpoint 500 to modelpool success
```

这表明以下链路都已经成功：

| 链路 | 状态 |
|---|---|
| `aisrv` 新 observation / 新模型前向 | 成功 |
| `step-record -> sample_process(chunk)` | 成功 |
| learner 接收 recurrent chunk | 成功 |
| batch 解包 | 成功 |
| sequence-aware PPO 反向传播 | 成功 |
| checkpoint 保存 | 成功 |
| checkpoint 推送 modelpool | 成功 |

## 9. 本轮排查中最关键的 4 个经验

### 9.1 recurrent PPO 接入时，先检查“样本物理体积”

新 observation 一旦变大，`chunk_len` 不能凭经验设。  
先算单 sample 平铺后的体积，否则会直接撞上框架 `max_sample_size`。

### 9.2 Kaiwu 吞吐参数不能沿用单步 PPO 的默认量级

LTSPPO 的 chunk sample 和旧单步 sample 在“数量级”和“节奏”上完全不同。  
`replay_buffer_capacity`、`preload_ratio`、`send_sample_size`、`train_batch_size` 必须重新配。

### 9.3 `process_stop.done` 容易误导判断

看到 learner 因 stop sentinel 退出时，不能直接认定是“残留文件问题”。  
必须回头看训练日志，确认它是不是业务错误的下游结果。

### 9.4 框架传给 learner 的 batch 形态必须以真实日志为准

不能只按本地 `SampleData` 设计来写解包器。  
要直接兼容：

- 对象列表
- 平铺 tensor
- 张量列表

否则很容易在第一轮真实训练时崩掉。

## 10. 当前结论

本轮 smoke test 的最终结论是：

**LTSPPO 已经通过了从 `aisrv` 产样到 learner 训练、再到存模/推模的端到端验证。**

更准确地说，已经确认：

1. 代码骨架不是“只能静态通过”的状态。
2. LTSPPO 现在已经能在 Kaiwu 训练栈里真正跑起来。
3. 当前已跨过的不是单点语法错误，而是三类系统级阻塞：
   - sample 物理尺寸
   - replay / preload 吞吐参数
   - learner batch 协议兼容

## 11. 建议的下一步

下一步不该继续大改结构，而应按以下顺序推进：

1. 做 10-30 分钟短训练稳定性验证  
   重点看：
   - `policy_loss`
   - `value_clean_loss`
   - `value_survive_loss`
   - `entropy`
   - 是否出现 NaN / OOM / hidden-state 漏 reset

2. 补 LTSPPO 监控指标  
   建议新增：
   - `mode_usage`
   - `target_switch_rate`
   - `late_return_rate`
   - `battery_fail_rate`
   - `collision_fail_rate`

3. 做一次短 benchmark / eval  
   目标不是追求高分，而是确认：
   - reset 后 `rnn_state` 真正清空
   - eval 流程不会因为新 reward dict / 新模型输出契约崩掉

4. 之后再进入训练策略层调优  
   例如：
   - teacher 权重
   - target teacher 可靠性门控
   - dual critic 权重平衡
   - recurrent chunk 进一步优化

