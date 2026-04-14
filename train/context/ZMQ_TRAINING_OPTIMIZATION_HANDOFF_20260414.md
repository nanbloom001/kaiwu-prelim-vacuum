# 训练优化完整交接说明

> 日期: 2026-04-14
> 状态: ZMQ 训练链路已跑通并稳定产出训练指标
> 适用对象: 下一位继续接手本仓库训练优化工作的 AI / 工程师

## 1. 文档目的

这份文档统一整理 2026-04-14 本轮训练优化工作的完整过程，包括：

- 为什么要从 reverb 继续推进到 ZMQ
- 本轮实际修改了哪些仓库文件
- 容器启动时额外对框架源码做了哪些运行时 hot patch
- 真实训练测试里依次遇到了哪些问题
- 每个问题是怎么定位、怎么验证、为什么有些方案被放弃
- 当前已经验证成立的性能结论是什么
- 下一位 AI 接手时最应该优先做什么

目标不是给出零散结论，而是让下一位接手者不需要重新 reconstruct 这一轮技术决策。

---

## 2. 执行摘要

本轮工作的核心结论只有三条：

1. GPU 路由修复后，reverb baseline 重新验证过，主瓶颈依旧是 replay / data_fetch，而不是 learner 计算。
2. ZMQ 路径不是“配置没生效”，而是最初真的被激活了，但 learner 侧存在 Linux 下 fork 后重入 CUDA 的兼容性问题。
3. 最终有效解法不是全局强制 `spawn`，而是保留 fork 友好的主流程，同时避免父进程里的 shared mem_buffer 在 fork 之前触发 CUDA 初始化。修复后，ZMQ 训练已稳定跑通，训练速度从 reverb 的约 6-7 steps/s 提升到当前约 23.3 steps/s。

当前已经验证的稳定窗口指标：

- 平均 `train once cost time`: 46.44 ms
- 平均 `data_fetch`: 11.15 ms
- 平均 `real_train`: 35.27 ms
- 平均 `steps/s`: 23.3
- 估算 `samples/s`: 95,419

这说明 replay 管道瓶颈已经大幅缓解，下一阶段重点应从“继续修 replay”转为“把 learner 计算吞吐再抬高”。

---

## 3. 开始这轮工作前的已知背景

开始本轮工作前，仓库已经具备以下基础状态：

- 项目为腾讯 Arena 扫地机器人 PPO 训练系统，运行在独立 Linux + Docker Compose 环境。
- 4 张 A10 GPU 中，learner 固定在物理 GPU0，aisrv 预期固定在物理 GPU1 / GPU2。
- 之前已经确认：
  - aisrv GPU 路由原先有问题，后续已修正；
  - reverb 可以稳定训练，但 `data_fetch` 明显高于 `real_train`；
  - 第一轮 ZMQ smoke test 失败，错误为 `Cannot re-initialize CUDA in forked subprocess`。

因此，本轮任务不是“从零开始调训练”，而是：

1. 把观测能力补齐，避免继续盲调。
2. 在修好 GPU 路由后的前提下重采 reverb baseline。
3. 如果 reverb 仍是 replay bottleneck，就继续把 ZMQ 路径真的跑通。

---

## 4. 本轮实际修改过的仓库文件

下面只列和训练优化直接相关的代码/配置文件；训练过程中产生的模型文件变化单独放到最后说明。

### 4.1 `code/agent_ppo/agent.py`

新增了运行时取证能力 `runtime_probe`，目的不是提速，而是避免“看起来都在 cuda:0，但实际并没有绑到正确物理卡”这种假象再次出现。

关键改动：

- 新增 `_build_runtime_probe_payload()` 和 `_emit_runtime_probe_once()`。
- 在 `init`、`learn`、`predict` 阶段各输出一次结构化 JSON 日志。
- 记录内容包括：
  - `CUDA_VISIBLE_DEVICES`
  - `NVIDIA_VISIBLE_DEVICES`
  - `KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE`
  - `predict_batch_size` / `train_batch_size` / `send_sample_size`
  - `requested_device`
  - `algorithm_device`
  - `model_param_device`
  - 输入输出张量的 device / dtype / shape

这一步的价值：

- 验证 learner 是否只看见一张卡
- 验证 aisrv 是否真的绑到物理 GPU1 / GPU2
- 验证 replay_buffer_type 是否真的从配置层传到业务层
- 之后做参数实验时，可以判断配置有没有真的落地

### 4.2 `code/tests/test_runtime_optimizations.py`

给 `runtime_probe` 加了单测，主要覆盖：

- probe payload 是否包含关键设备与配置字段
- 同一阶段是否只打一次日志，避免刷屏

### 4.3 `code/agent_ppo/utils/container_routing.py`

新增了一个 aisrv 容器路由 helper，解决“用 hostname 后缀推副本编号”在 Docker 实际环境下失效的问题。

设计逻辑：

- 读当前容器名 / 当前容器 IP
- 用 service DNS 解析 `kaiwu-train-aisrv-1/2/...` 的 IP
- 用当前容器 IP 匹配 service IP，得出稳定的 `aisrv_index`
- 再根据 `KAIWU_AISRV_GPU1_NUM` / `KAIWU_AISRV_GPU2_NUM` / `KAIWU_AISRV_GPU3_NUM` 映射到目标物理 GPU
- 输出一组 shell `export` 语句供 compose 启动脚本 `eval`

### 4.4 `code/tests/test_container_routing.py`

给 `container_routing.py` 补了完整单测，覆盖：

- index 解析
- GPU 分组映射
- DNS 解析成功 / 失败路径
- shell exports 输出格式
- CLI 入参校验

### 4.5 `code/agent_ppo/utils/zmq_patch.py`

新增 ZMQ 运行时 patch helper，用来在容器启动时对框架源码注入最小补丁，而不是直接把容器内 framework 文件硬编码改爆。

当前它主要提供两类能力：

- `patch_zmq_runtime_files()`
  - 给 `zmq_replay_buffer.py` 和 `learner_server.py` 注入 spawn-context prelude
  - 目标是让这些内部 ZMQ worker 路径在 Linux 下使用 spawn context 的 `Process/Queue/...`
- `patch_zmq_entrypoints()`
  - 曾用于给 `learner.py` / `aisrv.py` 注入全局 `set_start_method("spawn")`
  - 这是中途尝试过但最终放弃的方案，目前函数还在，属于遗留代码

这个文件是为了把“字符串拼接式框架补丁”从 compose 里抽离一部分，至少让 ZMQ 相关 patch 具备可测试性和幂等性。

### 4.6 `code/tests/test_zmq_patch.py`

新增标准库即可运行的独立测试文件，原因是宿主环境没有完整 torch 依赖，不能把所有测试都塞进原来的 runtime 测试模块。

该测试覆盖：

- 非法 Python 源文件时的 fallback 注入
- 编码声明保留
- `__future__` import 顺序保留
- idempotency
- 只有 `replay_buffer_type=zmq` 时才进行 patch

最终宿主机执行结果为 7 个测试全部通过。

### 4.7 `train/.docker-compose.yaml`

这是本轮改动最多、也是最关键的文件。它承担了三类任务：

1. learner / aisrv 启动前环境 banner
2. 容器启动时对框架源码做 hot patch
3. 把实验参数从 `.env` 传入运行时 TOML / Python 路径

本轮与训练优化直接相关的关键改动包括：

- 把以下 aisrv GPU 路由变量显式透传进共享环境：
  - `KAIWU_AISRV_GPU1`
  - `KAIWU_AISRV_GPU2`
  - `KAIWU_AISRV_GPU3`
  - `KAIWU_AISRV_GPU1_NUM`
  - `KAIWU_AISRV_GPU2_NUM`
  - `KAIWU_AISRV_GPU3_NUM`
- learner 启动时：
  - 打印 runtime banner
  - 调用 `patch_zmq_runtime_files()`
  - 对 `mem_buffer.py` / `mem_buffer_ratio.py` 做 CPU-only 设备初始化 patch
  - 在 post-init patch 中把 `replay_buffer_type` 写回 `configure_app.toml`
- aisrv 启动时：
  - 调用 `container_routing.py` 得到自己的 replica index 和目标物理 GPU
  - 输出 aisrv runtime banner
  - 同样确保 `replay_buffer_type` 能写进 app 配置
- 中间为了排障，还临时加入过：
  - trainer spawn-safe 初始化 patch
  - learner / learner_server 对 shared mem buffer 的 logger 剥离与恢复 patch

这些补丁最终大多仍保留在 compose 热补丁逻辑中，因为它们参与了实际排障和最终可运行状态的形成。

### 4.8 `train/.env`

当前已经切到 ZMQ：

- `KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE=zmq`

当前关键实验参数为：

- `KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE=1`
- `KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE=4096`
- `KAIWU_EXPERIMENT_DUMP_MODEL_FREQ=500`
- `KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE=4096`
- `KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE=128`
- `KAIWU_EXPERIMENT_REVERB_SAMPLER=reverb.selectors.Fifo`

注意：虽然部分变量名还保留 `REVERB` 字样，但当前有效 replay backend 已经是 ZMQ，不要被变量名误导。

---

## 5. 本轮真实排障过程，按时间顺序展开

下面是这次训练优化实际发生的顺序。这个部分最重要，因为很多“看起来可行”的方案后来被证明是错方向。

### 5.1 第一步：补观测，确认 GPU 路由和配置落地

先做了 runtime probe 和 aisrv routing 修复，原因很简单：

- 如果连“现在到底跑在哪张卡上、到底是不是 zmq、参数是不是 4096/128”都不能直接在日志里看到，后面的性能分析不可信。

验证结果：

- learner 只看到自己的单卡，业务侧设备为 `cuda:0`
- aisrv-1 / aisrv-2 分别只看到物理 GPU1 / GPU2，但容器内部依然显示本地 `cuda:0`
- 这说明容器内 `cuda:0` 的语义正常，本质是单卡 namespace，不是“大家都挤在同一张物理卡”

### 5.2 第二步：重采 reverb baseline，确认还值不值得继续推进 ZMQ

GPU 路由修好后，重新看 reverb 训练结果，发现结论没有变：

- `train once cost time` 依然很大
- `data_fetch` 长时间显著高于 `real_train`
- GPU 利用率并不高

结论：

- learner 算力还没被吃满
- replay / sample delivery 仍然是主瓶颈
- 继续推进 ZMQ 是有价值的

### 5.3 第三步：第一次真实切换到 ZMQ，暴露新的配置问题

把 `.env` 切到 `zmq` 之后重新起分布式训练栈，首先暴露的不是 CUDA 问题，而是 aisrv 启动直接失败。

根因：

- `train/.docker-compose.yaml` 里新的路由脚本要求 `KAIWU_AISRV_GPU1_NUM` 等变量必须存在
- 但这些变量并没有实际从 compose 共享环境层传进去

修复：

- 在 compose 的 `x-kaiwu-client` 环境块里补齐所有 GPU 路由相关变量透传

修复后：

- 两路 aisrv 容器都能稳定起来

### 5.4 第四步：ZMQ 真正跑起来后，复现原始 blocker

在 aisrv 能启动后继续看 learner，日志明确显示：

- `learner ZmqReplayBuffer using shared mem_buffer from external source`
- `learner train replay_buff, use zmq`

说明 ZMQ 配置已经真实进入框架主链路，不是“改了 `.env` 但没用”。

随后 learner 失败，错误为：

- `Cannot re-initialize CUDA in forked subprocess`

关键认识：

- 问题不是 ZMQ 配置没有切进去
- 问题是 learner 某个进程路径在 Linux 下 fork 了一个已经触发过 CUDA lazy init 的父进程

### 5.5 第五步：第一次修复思路，尝试全局强制 `spawn`

这是最直觉的方案：

- 如果 fork 后不能重进 CUDA，那就改成 spawn

具体做法：

- 在 `zmq_patch.py` 中加入 entrypoint 级的 `set_start_method("spawn", force=True)` prelude 注入能力
- 在 compose 启动补丁里对 `learner.py` / `aisrv.py` 注入这个 prelude

结果：

- 原来的 CUDA/fork 报错确实消失了
- 但出现新的 `PicklingError`：`KaiwuLogger` 无法被序列化

这一步很关键，因为它证明：

- 全局改 `spawn` 并不是“对了只是没改全”
- 它虽然绕开了 CUDA/fork，但会把框架里所有不可 pickling 的运行态对象都暴露出来

### 5.6 第六步：继续沿着 `spawn` 方向做 trainer spawn-safe 改造

为验证 spawn 方案是不是“只差一点”，又进一步在 compose 里做了几件事：

- 把 `Trainer.__init__` 里对 `KaiwuLogger` / `ReplayBufferWrapper` / `strategy` 的初始化挪迟到 `before_run`
- 在 parent 里把 `shared_mem_buffer` 上的 logger 剥离掉，避免随进程序列化
- 在 child 里再把 logger 恢复回 shared buffer

这样做之后：

- `PicklingError` 消失了
- 但 learner 启动出现长时间卡住
- 伴随 `resource_tracker` 和僵尸子进程，表现为进程间对象传输/启动链路异常

到这里可以得出更本质的结论：

- 对这个框架，`spawn` 不是低风险修复；
- shared mem_buffer 体量和结构太重，全局 spawn 会带来额外的进程启动与序列化复杂度。

### 5.7 第七步：修正方向，回到 fork 友好模型，避免父进程先碰 CUDA

这是本轮真正的转折点。

重新审视问题后，结论是：

- 真正要避免的不是 fork 本身
- 而是“父进程在 fork 之前先初始化了 CUDA”

于是最终改法变成：

1. 保留 ZMQ 相关内部 runtime 文件的 spawn-context patch
   - 仅用于 `zmq_replay_buffer.py` / `learner_server.py` 这些局部 worker 路径
2. 不再依赖全局 entrypoint `spawn` 作为主方案
3. 在 compose 热补丁里改 `mem_buffer.py` / `mem_buffer_ratio.py`
   - 把它们的 device 初始化强制留在 CPU
   - 明确注释：trainer 之后会显式把 tensor / model 放到 CUDA

这一步的本质：

- 父进程里 shared mem_buffer 只负责共享内存与样本搬运，不再碰 CUDA
- 真正需要 CUDA 的地方仍然在 trainer/agent 路径里显式初始化

### 5.8 第八步：再次起栈，ZMQ 训练真正跑通

应用 CPU-only mem_buffer patch 后重新启动训练栈，结果如下：

- learner 存活
- trainer 子进程存活
- learner_server 子进程存活
- aisrv learner_proxy 与 learner 的 9997 / 9998 端口连接成功
- learner_server ZMQ worker 持续收到来自 aisrv 的样本
- trainer 日志持续产出 `train once cost time`

这意味着：

- ZMQ 路径已经不是 smoke test，而是真实训练在跑
- 原始 blocker 已被跨过

---

## 6. 当前稳定运行状态

### 6.1 当前 replay backend

- 当前有效 replay backend: `zmq`
- learner `runtime_probe` 已经记录到 `replay_buffer_type = "zmq"`

### 6.2 当前训练关键配置

- `train_batch_size = 4096`
- `predict_batch_size = 128`
- `send_sample_size = 4096`
- `dump_model_freq = 500`

### 6.3 当前训练日志状态

当前 trainer 日志已经稳定从以下文件持续产出：

- `train/log/learner/learner_train_pid329_log_2026-04-14-12.log`

ZMQ 两个 worker 的收样统计日志位于：

- `train/log/learner/learner_server_zmq_container740ba8ef_pid408_log_2026-04-14-12.log`
- `train/log/learner/learner_server_zmq_container740ba8ef_pid409_log_2026-04-14-12.log`

aisrv learner_proxy 连接日志位于：

- `train/log/aisrv/learner_proxy_container4e152f38_pid419_log_2026-04-14-12.log`
- `train/log/aisrv/learner_proxy_container234ec4f2_pid426_log_2026-04-14-12.log`

---

## 7. 真实性能结果

### 7.1 reverb baseline（修好 GPU 路由后重采）

此前 handoff 已记录过：

- `steps/s` 大致在 6.4 - 7.1
- `data_fetch` 大致在 109 - 717 ms，均值量级在 114 - 185 ms
- `real_train` 大多在 36 - 72 ms

结论：

- replay 管道占主导瓶颈
- learner 算力未吃满

### 7.2 当前 ZMQ 稳定窗口采样

已对 trainer 日志最近稳定区间做过量化汇总。采样窗口：

- 2026-04-14 12:56:54 到 13:05:54
- 样本点数：10

聚合结果：

- 平均 `train once cost time`: 46.44 ms
- 平均 `data_fetch`: 11.15 ms
- 平均 `real_train`: 35.27 ms
- `data_fetch` 最小值: 2.78 ms
- `data_fetch` 最大值: 25.2 ms
- 平均 `sample_production_and_consumption_ratio`: 208.5
- 平均 `steps/s`: 23.3
- 估算 `samples/s`: 95,419

### 7.3 与 reverb 的对比结论

保守比较也能得到：

- `steps/s` 从约 6-7 提升到约 23.3，约为 3.3 倍提升
- `samples/s` 从约 2.8 万提升到约 9.5 万
- `data_fetch` 从百毫秒量级下降到十毫秒量级，下降接近一个数量级

当前瓶颈结构已经明显改变：

- 以前是 `data_fetch >> real_train`
- 现在是 `real_train > data_fetch`

这意味着：

- replay 管道优化已经取得决定性收益
- 后续继续提速的主方向不该再是 reverb/ZMQ 迁移本身，而应该是 learner 计算吞吐

---

## 8. 这轮排障中验证失败、已经不建议继续作为主方案的方向

### 8.1 不建议再把“全局强制 spawn”当作默认主线

原因：

- 它确实能绕过原始 CUDA/fork 错误
- 但会引入新的不可 pickling 运行态对象问题
- 为了配合 spawn，又会牵引出 Trainer 初始化顺序、logger、shared_mem_buffer、strategy 等一串额外改造
- 最后还出现进程卡住 / 僵尸子进程 / resource tracker 异常

如果未来必须再试 spawn，需要先明确目标是：

- 做更彻底的框架级进程模型重构

而不是把它当作小补丁。

### 8.2 不要把 ZMQ 失败误判为“配置没切进去”

这次已经明确证实过：

- 当时 learner 日志里已有 `use zmq`
- 真正失败点在 learner 侧 CUDA / multiprocessing 初始化顺序

所以以后如果再见到 ZMQ 相关异常，先去看训练链路的具体报错，不要退回到“是不是 env 没生效”的旧怀疑上。

---

## 9. 当前残留问题和已知技术债

### 9.1 `zmq_patch.py` 里有遗留函数

- `patch_zmq_entrypoints()` 代表的是中途尝试过、最终被放弃的 entrypoint spawn 路线
- 当前最终有效方案并不依赖它
- 后续可以考虑删除，避免误导后人

### 9.2 `train/.docker-compose.yaml` 的运行时 patch 逻辑已经很重

当前 compose 文件承担了太多运行时源码修补逻辑，风险在于：

- anchor 一旦跟框架源码版本不匹配，就会 patch 失败
- 后续继续堆字符串替换会越来越脆弱

建议后续如果确认 ZMQ 路线长期保留，可以考虑把这批 hot patch 收敛成：

- 独立的容器内 patch script
- 或者干脆把 framework 层修改固化到可控源码层，而不是一直在 compose 里做字符串替换

### 9.3 当前仍存在配置名和实际语义不一致的问题

例如：

- `KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE`

变量名仍有 `REVERB`，但当前 replay backend 已经是 ZMQ。后续如果做长期维护，建议逐步把这类历史命名隔离清楚，否则容易影响阅读和实验判断。

### 9.4 `predict_batch_size` 仍值得继续核对

本轮观察到：

- 业务环境变量里 `KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE=128`
- 但 learner 某次 runtime_probe 中 `predict_batch_size` 仍显示为 1，而 `proxy_batch_size=128`

这说明配置链路里可能仍有一段映射语义不一致或未完全落地，需要专门核实，不要默认相信 env 已经完整生效。

---

## 10. 下一位 AI 接手时的优先级建议

### P0: 固化 ZMQ 新 baseline，不要立刻再大改

建议先做 30-60 分钟稳定性观测，确认：

- learner 不会中途挂掉
- `train once cost time` 没有持续漂移恶化
- learner_server 两路收样持续稳定
- checkpoint / model_signer / modelpool 同步都持续正常

为什么先做这个：

- 目前系统刚跨过 replay bottleneck，最值钱的是先证明“它不是一次性的 lucky run”

### P1: 优化重点从 `data_fetch` 转向 `real_train`

当前稳定窗口里：

- 平均总耗时 46.44 ms
- 其中 `real_train` 已占 35.27 ms

说明主瓶颈已经转到 learner 计算侧。优先建议做：

1. `train_batch_size=6144`
2. `train_batch_size=8192`

并和现有 4096 做真实对比，重点看：

- `samples/s`
- `steps/s`
- `real_train`
- 总体 reward / 训练稳定性

### P2: 核对 `predict_batch_size` 真正落地位置

需要确认：

- learner business config 的 `predict_batch_size`
- proxy 层 `proxy_batch_size`
- aisrv 侧对应取值

是否是同一条链路，还是历史上存在两个不同语义的字段。

### P3: 如果目标是“尽量吃满整台 4 卡机器”，下一步不是继续抠 replay，而是并行实验

当前框架限制下，单 learner 很难自然扩展到多卡同步训练。现实可行的下一步是：

- 让 GPU3 跑第二套独立实验
- 做 batch / predict_batch / 采样配置的并行对比

这比在单栈里继续挤几毫秒更有产出。

---

## 11. 推荐给下一位 AI 的工作顺序

最建议的接手顺序如下：

1. 先读这份文档
2. 再读 `train/context/HANDOFF_20260413.md`
3. 然后确认当前容器是否还在跑、当前日志是否仍持续增长
4. 做 30-60 分钟 ZMQ 稳定性采样
5. 再开始 batch size 对比实验

不建议上来就做的事：

- 不要第一步就继续重写 compose 热补丁
- 不要第一步就回头换回 reverb
- 不要默认使用全局 `spawn`

---

## 12. 本轮验证证据清单

### 12.1 关键代码文件

- `code/agent_ppo/agent.py`
- `code/agent_ppo/utils/container_routing.py`
- `code/agent_ppo/utils/zmq_patch.py`
- `code/tests/test_container_routing.py`
- `code/tests/test_runtime_optimizations.py`
- `code/tests/test_zmq_patch.py`
- `train/.docker-compose.yaml`
- `train/.env`

### 12.2 关键上下文文件

- `train/context/HANDOFF_20260413.md`
- `train/context/DIAGNOSIS_REMEDIATION_REPORT_20260409.md`

### 12.3 关键日志文件

- `train/log/learner/learner_train_pid329_log_2026-04-14-12.log`
- `train/log/learner/learner_server_zmq_container740ba8ef_pid408_log_2026-04-14-12.log`
- `train/log/learner/learner_server_zmq_container740ba8ef_pid409_log_2026-04-14-12.log`
- `train/log/aisrv/learner_proxy_container4e152f38_pid419_log_2026-04-14-12.log`
- `train/log/aisrv/learner_proxy_container234ec4f2_pid426_log_2026-04-14-12.log`

---

## 13. 训练过程中产生的非代码文件变化

以下变化是训练运行结果，不是本轮优化逻辑本身：

- `code/latest_model.pkl`
- `code/model.ckpt-resume.pkl`
- `code/model.ckpt-resume.meta.json`

这些文件变化反映了训练继续推进和 resume 快照更新，不应被误读为“优化逻辑改动”。

---

## 14. 最终结论

一句话总结这轮工作：

**ZMQ replay 路线已经从“理论上可能更快”变成“在当前仓库里已真实跑通并带来 3 倍以上训练吞吐提升”，而且最关键的经验是：要避免父进程预先触发 CUDA，而不是简单粗暴地全局切 `spawn`。**

如果下一位 AI 要继续推进，最正确的起点已经不是“修 ZMQ 能不能用”，而是：

- 把 ZMQ 作为新 baseline 固化下来；
- 然后把优化主战场切到 `real_train` 和更大 batch 的吞吐实验。