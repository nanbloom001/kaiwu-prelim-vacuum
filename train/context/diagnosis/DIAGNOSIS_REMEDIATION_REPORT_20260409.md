# TcKaiwuFinal 独立训练链路诊断修复报告

## 1. 报告目的

本文档记录 `/home/user/TcKaiwuFinal` 在脱离开悟平台、从 Windows 11 迁移到独立 Linux Docker 环境后的完整诊断与修复过程，重点覆盖以下链路：

- 容器编排与进程状态
- learner 训练主链路
- AISRV 样本发送与模型加载链路
- reverb 经验回放链路
- checkpoint 落盘与 modelpool 同步链路
- GPU 使用情况
- 18080 监控面板与 GreptimeDB 数据查询链路

本报告基于 2026-04-09 晚间的实际在线排查结果撰写，结论为：

- 训练链路已恢复正常
- 当前无持续性阻塞错误
- 监控链路已恢复出数
- 剩余现象均为非阻塞、可接受或后续优化项

---

## 2. 环境与约束

### 2.1 仓库与运行环境

- 仓库路径：`/home/user/TcKaiwuFinal`
- 当前运行方式：独立 Linux Docker 分布式训练
- 原始背景：
  - 代码仓库从 Windows 11 平台直接迁移
  - 原始训练流程依赖开悟平台
  - 当前目标是在脱离开悟平台的本地环境中直接训练

### 2.2 已确认环境约束

- 主机存在 Clash 代理
  - 本地 HTTP 检查需要显式绕过代理
- 主机 `4000` 端口被 NoMachine 占用
  - 主机访问 GreptimeDB 需要使用 `14000`
  - 容器内部访问 `greptimedb:4000` 仍然是正常行为

### 2.3 关键容器状态

本次复核时以下核心容器均为 `Up`：

- `kaiwu-train-learner-1`
- `kaiwu-train-aisrv-1`
- `kaiwu-train-aisrv-2`
- `kaiwu-train-gamecore-1` ~ `kaiwu-train-gamecore-8`
- `kaiwu-train-pushgateway-1`
- `kaiwu-train-vector-1`
- `kaiwu-train-monitor-service-1`
- `kaiwu-train-fe-monitor-service-1`
- `kaiwu-train-backup_model-1`

说明基础容器编排已健康。

---

## 3. 初始故障现象

在本轮修复前，系统存在以下严重问题：

### 3.1 learner 卡在启动阶段

- learner 日志只停留在启动阶段
- 仅生成过一次初始模型：
  - `model.ckpt-0.pkl`
- `/data/ckpt/robot_vacuum_ppo/` 长时间没有后续 checkpoint

### 3.2 AISRV 持续报错

- `policy.send_train_data failed, please check`
- `model_file_sync current_available_model_files is empty`

### 3.3 learner 辅助进程异常

- `model_file_save` 持续 `FileNotFoundError`
- `monitor_proxy` 持续 `Broken pipe`

### 3.4 18080 监控面板曾经大量显示 `n/a`

该问题后续被确认是 GreptimeDB 网络接入问题，已先行修复，不是这次训练主链路阻塞的根因。

---

## 4. 根因分析

## 4.1 不是 GPU 不可用

排查中确认：

- learner 容器可见 CUDA
- GPU 0 为 `NVIDIA A10`
- 训练过程中显存稳定占用约 4GB
- GPU 利用率持续有波动

因此问题并不是“没有调用 GPU”。

## 4.2 不是环境仿真链路中断

排查中确认：

- gamecore 正在运行
- AISRV 有持续 episode 启停
- reverb buffer 有持续写入

因此 rollout / 环境交互链路本身是活的。

## 4.3 `spawn` 不是正确修复方向

曾尝试将 learner/aisrv 的 multiprocessing 强制改成 `spawn`，结果 learner 报出：

- `_pickle.PicklingError: Can't pickle <class 'common_python.logging.kaiwu_logger.KaiwuLogger'>`

这说明在当前框架结构下，直接全局改 `spawn` 会引入新的不可序列化问题，并不适合作为最小修复方案。

## 4.4 Linux 下 `Manager().Queue()` 是不稳定点之一

框架中 `monitor_manager.py` 原先使用：

- `multiprocessing.Manager().Queue()`

在独立 Linux 训练下，这类 manager/proxy 队列更容易在 fork 之后出现连接问题，和早前的 `Broken pipe`/代理异常现象一致。

因此后续改成：

- Linux 下使用 `multiprocessing.Queue(CONFIG.queue_size)`
- 非 Linux 再保留 `Manager().Queue()`

## 4.5 最终核心根因：`clear_user_ckpt_dir()` 删除了挂载根目录

这是本次问题的真正阻塞点。

框架函数：

- `kaiwudrl/common/checkpoint/model_file_common.py`
- `clear_user_ckpt_dir()`

原逻辑是：

```python
def clear_user_ckpt_dir():
    model_path = CONFIG.user_ckpt_dir
    if os.path.islink(model_path):
        return
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
```

而当前 compose 中：

- 主机目录 `${KAIWU_TRAIN_LOG}/framework_ckpt`
- 挂载到容器 `/data/user_ckpt_dir`

于是 learner 启动后在第一次保存 `model.ckpt-0.pkl` 之后，执行：

- `clear_user_ckpt_dir()`

等价于试图删除一个挂载根目录，最终抛出：

- `OSError: [Errno 16] Device or resource busy: '/data/user_ckpt_dir'`

这与 Windows/平台内原始运行假设不一致，但在独立 Linux Docker 挂载场景下是必然失败的。

这就是 learner 一直卡在初始保存之后、不再继续初始化训练主链路的核心原因。

---

## 5. 修复方案

## 5.1 修复原则

采用最小侵入方案，不直接改镜像，而是通过 compose 启动注入脚本在容器内修补运行时代码。原因如下：

- 框架源码主要存在容器内部 `/data/projects/robot_vacuum/...`
- 主仓库不是完整运行时源码镜像
- 需要在不重建镜像的前提下快速恢复训练

## 5.2 已实施修复项

### 修复 1：启动脚本启用失败即停

修改文件：

- [train/.docker-compose.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.yaml#L72)

为 learner / aisrv 的启动脚本增加：

```bash
set -euo pipefail
```

作用：

- 如果运行时补丁脚本失败，容器直接退出
- 避免“补丁没生效但容器还在跑”的假健康状态

### 修复 2：去掉错误的 `spawn` 注入

在启动补丁中清理了先前尝试插入到：

- `learner.py`
- `aisrv.py`

中的 `mp.set_start_method("spawn")` 片段。

作用：

- 避免再次触发 `PicklingError`

### 修复 3：Linux 下 monitor queue 改用原生 Queue

在启动补丁中重写：

- `common_python/monitor/monitor_manager.py`

让 Linux 使用：

```python
multiprocessing.Queue(CONFIG.queue_size)
```

而不是：

```python
multiprocessing.Manager().Queue()
```

作用：

- 降低 manager socket/proxy 断连风险
- 对应修复早前 `monitor_proxy` 相关不稳定问题

### 修复 4：禁止删除挂载根目录，只清空目录内容

在启动补丁中重写：

- `kaiwudrl/common/checkpoint/model_file_common.py`
- `clear_user_ckpt_dir()`

新逻辑不再 `rmtree(CONFIG.user_ckpt_dir)`，而是：

- 若目录不存在则返回
- 若是软链接则返回
- 遍历目录内容
- 删除文件/软链接
- 递归删除子目录
- 保留挂载根目录本身

作用：

- 修复 learner 启动时的硬阻塞
- 兼容 Linux 挂载点语义

### 修复 5：无 COS 场景跳过 `model_file_saver` 额外进程

在启动补丁中重写：

- `kaiwudrl/server/learner/trainer.py`

当：

- `push_to_cos = 0`

时不再启动 `ModelFileSave()` 进程。

作用：

- 避免无意义的额外 fork
- 对应消除早前 `model_file_save FileNotFoundError` 风险路径

### 修复 6：off-policy 模型同步改为进程内同步

在启动补丁中重写：

- `kaiwudrl/server/learner/off_policy_strategy.py`

使 learner 直接在进程内使用 `ModelFileSync()` 完成 checkpoint 推送到 modelpool，而不再依赖额外 fork 出来的包装进程。

作用：

- 避免 Linux 独立训练环境下多进程同步链路不稳定
- 确保首个模型和后续 checkpoint 都能及时推送到 modelpool

### 修复 7：修正用户自定义 monitor 面板命名校验失败

修改文件：

- [code/agent_ppo/conf/monitor_builder.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/monitor_builder.py#L46)

将不合法的 panel 名称：

- `Avg Invalid Move Rate`
- `Avg Charge Efficiency`

改为：

- `平均无效移动率`
- `平均充电效率`

作用：

- 消除 learner 初始化阶段最后一条 `ERROR`
- 恢复用户自定义监控配置加载

### 修复 8：18080 面板数据链路恢复

此前已完成：

- 将 `train-greptimedb-1` 接入 `kaiwu-train_default`
- 为其添加 `greptimedb` 网络别名
- 重启 `vector`

作用：

- 恢复 `pushgateway -> vector -> greptimedb -> 18080` 的数据链路

---

## 6. 复核结果

以下为本次最终全链路复核结果。

## 6.1 learner 训练主链路

当前 learner 已稳定运行，并持续推进训练：

- `global step` 已从启动后的 91 增长到 1201+
- 训练日志持续输出：
  - `policy_loss`
  - `value_loss`
  - `entropy_loss`
  - `sample_production_and_consumption_ratio`

关键日志现象：

- `train process start success`
- `start_background_filler success`
- 周期性输出 `global step is ...`

结论：

- learner 主训练循环正常

## 6.2 reverb 链路

容器内实测：

- `current_size = 10000`
- `insert_completed = 303677`
- `sample_completed = 2562786`

解释：

- reverb 中样本在持续插入
- learner 在持续消费样本
- `sample_completed` 大于 0 且持续增长，证明训练真实发生

结论：

- reverb 链路正常

## 6.3 checkpoint 落盘链路

当前目录：

- `/data/ckpt/robot_vacuum_ppo/`

可见文件包括：

- `model.ckpt-0.pkl`
- `model.ckpt-100.pkl`
- `model.ckpt-200.pkl`
- `model.ckpt-300.pkl`
- `model.ckpt-400.pkl`
- `model.ckpt-500.pkl`
- `model.ckpt-600.pkl`
- `model.ckpt-700.pkl`
- `model.ckpt-800.pkl`
- `model.ckpt-900.pkl`
- `model.ckpt-1000.pkl`
- `model.ckpt-1100.pkl`
- `model.ckpt-1200.pkl`

结论：

- checkpoint 已连续生成，不再停留在 `model.ckpt-0.pkl`

## 6.4 learner -> modelpool -> AISRV 模型下发链路

learner 日志确认：

- `train first model file push to modelpool success`
- 后续 `train push checkpoint 100/200/.../1200 to modelpool success`

AISRV 日志确认：

- 持续加载：
  - `model.ckpt-300`
  - `model.ckpt-400`
  - `model.ckpt-600`
  - `model.ckpt-700`
  - `model.ckpt-900`
  - `model.ckpt-1000`
  - `model.ckpt-1100`

且 AISRV 训练指标中：

- `load_model_succ_cnt` 已增长到 820 / 910

结论：

- model sync 链路正常
- AISRV 可以持续获取并加载新模型

## 6.5 AISRV 样本发送链路

AISRV 指标中：

- `sample_receive_cnt` 已增长到 `457184` / `489610`
- `train_global_step` 与 learner 同步增长到 `1845` / `1985`

同时不再出现：

- `policy.send_train_data failed, please check`

结论：

- AISRV 到 learner 的样本发送链路正常

## 6.6 monitor / 辅助进程链路

当前未再扫描到以下异常：

- `Broken pipe`
- `FileNotFoundError`
- learner/AISRV `ERROR`

且 monitor 相关进程有启动成功日志：

- `monitor_proxy process start success`

结论：

- 辅助进程链路已恢复稳定

## 6.7 GPU 使用情况

当前主机 `nvidia-smi` 显示：

- GPU 0: `NVIDIA A10`
- 利用率约 `21%`
- 显存占用约 `4024 MiB / 23028 MiB`

其余 GPU 空闲。

结论：

- 训练已实际调用 GPU

## 6.8 18080 面板与数据查询链路

### 18080 页面响应

访问：

- `http://127.0.0.1:18080/`

已返回完整 HTML 页面，不再是异常页。

### GreptimeDB 查询结果

通过主机侧查询：

- `sum(kaiwu_episode_cnt{}) = 1963`
- `avg(kaiwu_reward{}) = 27.8634...`

说明：

- `vector -> greptimedb` 写入是有效的
- 18080 面板查询基础数据已恢复

### 说明

`pushgateway:9091/metrics` 上并不一定直接暴露所有最终聚合指标，这并不代表 18080 异常。当前更可信的数据入口是：

- GreptimeDB Prometheus 查询接口
- 18080 面板本身

结论：

- 监控数据链路正常

---

## 7. 当前未见的问题

本次最终复核中，未再发现以下持续性问题：

- learner 卡死在启动阶段
- 只有 `model.ckpt-0.pkl` 没有后续 checkpoint
- AISRV `policy.send_train_data failed`
- AISRV `model_file_sync` 长期空模型且无法加载
- learner `FileNotFoundError`
- learner `Broken pipe`
- monitor 自定义配置加载 `ERROR`
- GPU 未被使用
- 18080 面板无响应或完全无数据

---

## 8. 仍需说明的现象

以下现象存在，但不构成当前阻塞：

### 8.1 AISRV 启动初期偶发 `current_available_model_files is empty`

在重启后的很短时间窗口内，AISRV 可能先于 learner 完成首个模型推送，因此会出现少量：

- `model_file_sync current_available_model_files is empty`

但随后会紧接着出现：

- `load model ... success`

因此这是启动瞬态现象，不是持续故障。

### 8.2 训练效果层面仍需单独评估

当前报告确认的是：

- “训练链路正常”

并不等价于：

- “策略效果已经收敛到理想水平”

目前从 AISRV episode 日志看：

- 仍有较多失败 episode
- 但也已经出现 `WIN`
- clean score 在部分 episode 中可达到 `100+`

这属于后续算法调参与 reward 设计问题，不属于本轮基础设施故障。

---

## 9. 最终结论

## 9.1 根因结论

本次严重故障的根因不是 dashboard，也不是 GPU，也不是 gamecore，而是：

- 从 Windows/开悟平台环境迁移到独立 Linux Docker 之后
- 框架仍沿用平台化假设
- 在 learner 启动阶段错误地对挂载根目录 `/data/user_ckpt_dir` 执行 `shutil.rmtree()`
- 从而导致 learner 在初始模型保存后直接阻塞，训练主链路无法继续推进

同时，Linux 下多进程 manager queue、无 COS 场景的额外进程、off-policy 多进程 model sync，也放大了不稳定性。

## 9.2 当前状态结论

截至本次复核结束，系统状态为：

- 分布式容器全体健康
- learner 正在稳定训练
- reverb 正在持续供样与采样
- checkpoint 正在持续生成
- modelpool 正在持续接收 checkpoint
- AISRV 正在持续加载新模型
- AISRV 正在持续向 learner 发送样本
- GPU 正在被实际使用
- 18080 面板和 GreptimeDB 查询均正常出数
- 最新日志中未发现持续性错误

即：

- 当前已经恢复到“可正常训练且无持续报错”的状态

---

## 10. 本次涉及的关键文件

- [train/.docker-compose.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.yaml)
- [code/agent_ppo/conf/monitor_builder.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/monitor_builder.py)
- [train/context/sessions/SESSION_LOG_20260409.md](/home/user/TcKaiwuFinal/train/context/sessions/SESSION_LOG_20260409.md)

---

## 11. 后续建议

建议接下来按以下顺序继续工作：

1. 保持当前训练运行一段时间，继续观察：
   - `global step`
   - `sample_receive_cnt`
   - `load_model_succ_cnt`
   - 新 checkpoint 是否继续增长
2. 用 18080 与 GreptimeDB 查询继续验证指标连续性，避免只看单点值
3. 单独进入“训练效果优化”阶段，评估：
   - reward 设计
   - charging 行为
   - invalid move rate
   - win rate 与 clean score 趋势
4. 如后续需要长期维护，建议把当前 compose 内的运行时热补丁逐步固化到正式源码/镜像中，减少后续重启依赖

---

## 12. 2026-04-10 Learner 吞吐专项诊断与修复记录

本章节用于持续追加 learner 吞吐、并行扩容、多 AISRV 链路、GPU 利用率相关问题的诊断结果。

后续若继续排查同类问题，应继续追加到本文件，不再新建独立诊断文件。

### 12.1 问题现象

本轮排查期间，核心现象如下：

- 并行环境数量增加后，训练速度没有随之提升，甚至出现下降
- `global step/min` 基本停留在 `135 ~ 148 step/min` 附近
- AISRV 侧 `sample_receive_cnt` 随并行环境和 AISRV 数量上升而显著增长
- learner 侧 GPU 利用率始终较低，主要只使用 `GPU0`
- learner 训练日志中，`data_fetch` 明显高于 `real_train`

已观察到的典型日志形态：

- 优化前常见窗口：
  - `train once cost time` 约 `918 ~ 1202 ms`
  - `data_fetch` 约 `791 ~ 825 ms`
  - `real_train` 约 `127 ~ 377 ms`
- 优化后稳定训练窗口：
  - `train once cost time` 约 `751 ~ 1059 ms`
  - `data_fetch` 约 `725 ~ 1032 ms`
  - `real_train` 约 `26 ~ 27 ms`

这说明：

- learner 真正的前向/反向训练已经不是主要瓶颈
- 当前主瓶颈在样本获取与 batch 组装链路，而不是 GPU 算力本身

### 12.2 关键结论

#### 12.2.1 `global step/min` 是正确的训练速度指标

从业务算法代码看，`global step` 对应真实的 learner 参数更新次数，而不是环境步数或 episode 数。

定位位置：

- [algorithm.py](/home/user/TcKaiwuFinal/code/agent_ppo/algorithm/algorithm.py)

关键行为：

- `Algorithm.learn()` 每执行一次真实优化步骤后执行 `self.train_step += 1`

因此：

- `global step/min` 可以直接作为 learner 训练更新吞吐指标

#### 12.2.2 并行环境增加但训练速度不升，主因不是统计口径，而是 learner / replay 数据链路先成为瓶颈

已经做过的对照实验显示：

- 单 AISRV 扩容：
  - `4 env` 约 `148 step/min`
  - `8 env` 约 `144 ~ 145 step/min`
  - `16 env` 约 `137 ~ 140 step/min`
- 多 AISRV 扩容：
  - `1x4` 时 `sample_receive/min` 约 `8170`
  - `2x4` 时 `sample_receive/min` 约 `33375`
  - `2x8` 时 `sample_receive/min` 约 `64141`
  - `3x8` 时 `sample_receive/min` 约 `136636`

结论：

- 样本产出吞吐显著增长
- learner 更新吞吐没有同步增长
- 因此系统已经进入“样本生成远快于样本消费”的状态

### 12.3 问题位置

#### 12.3.1 业务侧训练入口存在“拆开再拼回去”的重复开销

定位位置：

- [algorithm.py](/home/user/TcKaiwuFinal/code/agent_ppo/algorithm/algorithm.py)
- 容器内框架代码：`/data/projects/robot_vacuum/kaiwudrl/interface/remote_agent.py`

原始链路是：

1. reverb dataset 返回 batch tensor
2. 框架层 `remote_agent.py` 把 batch tensor 反序列化成 `list[SampleData]`
3. 业务侧 `Algorithm.learn()` 再对 `obs / act / prob / value / reward_sum / advantage` 逐项 `torch.stack()`

这导致 learner 侧每个 batch 都要经历一轮：

- `tensor -> Python 对象列表 -> tensor`

这部分属于纯 CPU / Python 开销，对 GPU 无贡献，但会体现在 `data_fetch` 阶段。

#### 12.3.2 reverb dataset 预取链路本身是单线程、大批量、串行拉取

定位位置：

- 容器内框架代码：`/data/projects/robot_vacuum/kaiwudrl/common/replay_buffer/reverb_dataset_v1.py`

关键实现特征：

- 只有一个后台填充线程
- 每次通过 `client.sample()` 按大 batch 拉取数据
- 拉取后再做 Python 侧 `raw_batch -> process_batch -> tensor`
- 只有当 `active_buffer` 满足 `train_batch_size` 时 learner 才能继续推进

这导致：

- learner 很容易在等待 replay buffer 组 batch
- `data_fetch` 时间显著高于 `real_train`

#### 12.3.3 当前架构仍是单 learner 单卡

定位位置：

- [train/.docker-compose.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.yaml)

当前运行方式决定了：

- learner 只有一个
- 实际训练只在单张 GPU 上进行
- 即使宿主机有 4 张显卡，也不会自动转化为 4 卡训练

因此：

- 当前 GPU0 低占用不是“显卡坏了”
- 而是单 learner 结构下，数据链路先卡住了，GPU 自然无法吃满

### 12.4 已实施并验证有效的修复

以下修复已保留在业务代码中，且实测可以正常训练：

#### 12.4.1 learner 侧 PyTorch runtime 调优

修改位置：

- [agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py)
- [conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py)

已做内容：

- learner 不再强制固定 `torch.set_num_threads(1)`
- learner 改为使用可配置 CPU 线程数
- CUDA 下开启：
  - `torch.set_float32_matmul_precision("high")`
  - `cudnn.benchmark = True`
  - TF32 相关开关
- optimizer 优先尝试：
  - `foreach=True`
  - `fused=True`
- learner CUDA 训练启用 AMP

当前保留配置：

- `LEARNER_CPU_THREADS = 4`
- `LEARNER_CPU_INTEROP_THREADS = 2`
- `LEARNER_USE_AMP = True`
- `LEARNER_ALLOW_FOREACH_OPTIMIZER = True`
- `LEARNER_ALLOW_FUSED_OPTIMIZER = True`

#### 12.4.2 业务侧改为直接消费 batch tensor，跳过 `SampleData` 列表重组开销

修改位置：

- [agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py)
- [algorithm.py](/home/user/TcKaiwuFinal/code/agent_ppo/algorithm/algorithm.py)
- [conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py)

已做内容：

- 在 `Agent` 中为 learner 打开 `PREFER_BATCH_TENSOR_LEARN`
- 对框架 `RemoteAgent.learn()` 做定向 patch：
  - learner 收到 batch tensor 时，直接把 tensor 交给业务 `learn()`
  - 不再反序列化成 `list[SampleData]`
- 在 `Algorithm.learn()` 中增加：
  - `_unpack_train_batch()`
  - `_unpack_batch_tensor()`
  - `_unpack_sample_objects()`

这样业务侧就支持两种输入：

- 直接 batch tensor
- 兼容旧的 `list[SampleData]`

实测效果：

- `real_train` 从之前常见的 `127 ~ 377 ms` 压到了约 `26 ms`
- 说明 learner 真正训练阶段的 CPU 与张量整理开销已经明显下降

#### 12.4.3 训练已恢复正常，checkpoint 持续推进

实测结果：

- learner 正常启动并训练
- checkpoint 已持续生成到：
  - `model.ckpt-100.pkl`
  - `model.ckpt-200.pkl`
  - 更高编号 checkpoint 在不同窗口中也持续生成
- AISRV `learner_proxy` 日志显示：
  - `send sample stat, succ_cnt ...`
  - `error_cnt is 0`

结论：

- 当前系统处于“可持续训练”的状态
- 没有出现此前那种 learner 卡死、checkpoint 不增长、AISRV 持续报错的严重故障

### 12.5 已尝试但未保留的修复

#### 12.5.1 尝试对 reverb dataset 做多线程并行预取

尝试内容：

- 在业务层对容器内 `ReverbDataset` 进行 monkey patch
- 试图把单线程预取改为多线程分块预取

结果：

- learner 重启后出现：
  - `input ready size is 0`
  - `global step is 0`
  - 训练未真正开始
- AISRV 出现 reverb 写入异常：
  - `Item confirmation worker were stopped when 2 unconfirmed items ...`

结论：

- 该尝试没有通过验证
- 已回退
- 当前代码库中没有保留这部分修改

### 12.6 当前状态判断

截至本次记录，当前状态为：

- 训练链路正常
- checkpoint 持续生成
- AISRV 正常发送样本
- learner 正常消耗样本并训练
- GPU0 被实际使用
- GPU1/2/3 基本空闲

但同时也应明确：

- learner 真正计算开销已经压低
- 主要瓶颈仍是 `data_fetch`
- 当前并不是“GPU 算不满”，而是“数据链路供给不到位”

### 12.7 后续修复建议

如果后续继续优化 learner 吞吐，建议按以下优先级推进：

1. 不要继续在业务层堆叠 monkey patch 到 replay/reverb 核心逻辑。
2. 应直接修改框架层正式实现：
   - 容器内 `reverb_dataset_v1.py`
   - 容器内 `remote_agent.py`
   - 容器内 `replay_buffer_wrapper.py`
3. 优先目标不是再压 `real_train`，而是降低：
   - `data_fetch`
   - replay batch 等待
   - Python 对象拆装成本
4. 建议做正式对照实验：
   - `train_batch_size = 2048`
   - `train_batch_size = 1024`
   - 对比 `global step/min`
   - 对比 `data_fetch_ms`
   - 对比 `sample_production_and_consumption_ratio`
5. 如果目标是充分利用 4 张 GPU，则需要架构升级，而不是继续挤压当前单 learner：
   - 多 learner
   - DDP / 多卡训练
   - 或明确的参数服务器 / 多进程训练方案

### 12.8 本章节涉及的关键文件

- [agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py)
- [algorithm.py](/home/user/TcKaiwuFinal/code/agent_ppo/algorithm/algorithm.py)
- [conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py)
- [train/.docker-compose.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.yaml)
- `/data/projects/robot_vacuum/kaiwudrl/interface/remote_agent.py`
- `/data/projects/robot_vacuum/kaiwudrl/common/replay_buffer/reverb_dataset_v1.py`

### 12.9 追加约定

后续如果继续发现以下问题，统一追加到本文件本章节之后：

- learner 吞吐问题
- replay / reverb 取数瓶颈
- 多 AISRV 并行递减问题
- GPU 利用率偏低问题
- batch size / 并行环境 / learner 架构相关实验结果

---

## 13. 2026-04-10 第一阶段业务层优化实施记录

本节记录“先业务后框架”路线中的第一阶段实际实施内容、验证结果和新的观察。

### 13.1 本次已实施的改动

#### 13.1.1 Agent.load_model 增加同文件缓存跳过逻辑

修改位置：

- [agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py)

实现内容：

- 新增 `AGENT_LOAD_MODEL_CACHE` 配置
- 为 `load_model()` 增加基于：
  - `model_file_path`
  - `stat().st_mtime_ns`
  的缓存判断
- 当 episode 间重复加载同一 checkpoint 文件且文件未变化时：
  - 跳过 `torch.load()`
  - 不重复执行 `model.load_state_dict()`

新增运行时统计项：

- `load_model_calls`
- `load_model_reloads`
- `load_model_cache_hits`

原因：

- AISRV 业务 workflow 每局开始都会执行 `agent.load_model(id="latest")`
- 在模型未变化时重复从磁盘读模型是纯额外开销

#### 13.1.2 增加 AISRV 业务侧性能窗口统计

修改位置：

- [train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py)

实现内容：

- 新增 `PerfWindow`
- 对以下路径增加耗时累计：
  - `load_model`
  - `observation_process`
  - `predict`
  - `sample_process`
  - `send_sample_data`
- 对以下计数增加窗口统计：
  - `samples_built`
  - `samples_sent`
  - `episodes_yielded`

说明：

- 当前这些窗口统计已经写入业务归档链路
- 由于 archive agent 与宿主目录同步有时间差，短时间内不一定立刻在宿主 `train/archive` 下可见

#### 13.1.3 训练快照相关参数改为环境变量可覆盖

修改位置：

- [conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py)
- [train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py)

已支持的运行时覆盖项：

- `KAIWU_SAVE_MODEL_INTERVAL_EPISODES`
- `KAIWU_RESUME_EPISODE_SNAPSHOT_INTERVAL`
- `KAIWU_RESUME_LATEST_SYNC_INTERVAL_EPISODES`
- `KAIWU_RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS`
- `KAIWU_PERF_STAT_WINDOW_SECONDS`

目的：

- 后续做吞吐实验时，不必每次改业务源码
- 可以直接通过环境变量快速调整试验参数

#### 13.1.4 为新增逻辑补充轻量单测

新增文件：

- [test_runtime_optimizations.py](/home/user/TcKaiwuFinal/code/tests/test_runtime_optimizations.py)

已覆盖内容：

- `load_model` 缓存跳过逻辑
- `Algorithm._unpack_batch_tensor()` 的字段切片正确性
- `PerfWindow` 统计与 flush 行为

执行结果：

- 容器内 `unittest` 通过，`3/3 OK`

### 13.2 本次实施后的验证结果

实施后已重启 `learner` 和 `aisrv`，训练恢复正常，关键结果如下：

- learner 正常训练
- AISRV 正常 rollout
- checkpoint 持续增长：
  - `model.ckpt-100.pkl`
  - `model.ckpt-200.pkl`
  - `model.ckpt-300.pkl`
  - `model.ckpt-400.pkl`
- AISRV `training_metrics` 持续出数
- GPU0 继续被实际使用

### 13.3 新的关键观察

本次实施后，learner 日志出现了一个重要现象：

- `12:47:32` 窗口：
  - `train once cost time = 706.19 ms`
  - `data_fetch = 681.10 ms`
  - `real_train = 25.02 ms`
- `12:48:32` 窗口：
  - `train once cost time = 47.63 ms`
  - `data_fetch = 23.39 ms`
  - `real_train = 24.21 ms`
- `12:49:33` 窗口：
  - `train once cost time = 815.37 ms`
  - `data_fetch = 789.67 ms`
  - `real_train = 25.60 ms`

这说明：

- 业务侧轻量优化已经能在部分窗口显著减少等待
- `real_train` 仍稳定维持在 `25ms` 左右
- 但 `data_fetch` 波动仍然很大
- replay / reverb 数据供给仍然是主要不稳定因素

也就是说：

- 当前第一阶段优化有效
- 但它只能降低一部分上层无效开销
- 还不能根治 replay 数据链路的抖动

### 13.4 当前结论更新

截至本次实施后，结论更新为：

1. 业务层“同模型重复加载”确实是可消除开销，已做掉。
2. learner 真正训练算子开销已稳定压缩到很低，仍不是主瓶颈。
3. 当前剩余的主要问题是：
   - replay buffer 供给波动
   - reverb dataset 取样链路不稳定
   - `data_fetch` 在不同窗口中出现明显抖动
4. 下一阶段应优先推进：
   - `train_batch_size` 对照实验
   - replay / reverb 框架层热路径改造

### 13.5 下一步建议

建议严格按以下顺序继续：

1. 先基于当前代码做 `train_batch_size = 2048 / 1536 / 1024` 的对照实验。
2. 若 `data_fetch` 仍长期主导，则进入框架层改造：
   - `reverb_dataset_v1.py`
   - `remote_agent.py`
   - `replay_buffer_wrapper.py`
3. 在 replay 热路径未稳定前，不建议直接推进多 GPU 或多 learner。

---

## 14. 2026-04-10 第二阶段：基础设施修复 + 环境缩放实验

### 14.1 基础设施修复

本轮发现实验基础设施本身存在两个阻断性 bug，训练根本没在跑：

#### 14.1.1 `replace_toml_key()` 热补丁失效导致 TOMLDecodeError

**文件**：`train/.docker-compose.yaml`

**现象**：learner 容器每次启动即崩溃，报 `dynaconf.vendor.tomllib.TOMLDecodeError: Cannot overwrite a value (at line 59, column 39)`。

**根因**：`replace_toml_key()` 用正则 `^key\s*=.*$` (MULTILINE) 尝试删除 TOML 中的同名 key，但未能匹配原始行。函数只做了追加，导致 `pytorch_read_data_from_reverb_type` 和 `replay_buffer_cache_multiplier` 各出现两次。

**修复**：改为逐行 `split("=", 1)[0].strip() != key` 过滤所有同名行，再末尾追加唯一新值。

#### 14.1.2 实验脚本日志采集路径错误

**文件**：`train/run_replay_stability_experiments.py`

**现象**：`collect_rows()` 从宿主机 `train/log/learner/` 目录读日志（空目录），导致实验结果 `row_count=0`。

**根因**：容器内日志写在 `/data/projects/robot_vacuum/log/learner.log`（KaiwuDRL 框架路径），不在 Docker 挂载的 `/workspace/log/` 下。

**修复**：改为 `docker exec kaiwu-train-learner-1 cat /data/projects/robot_vacuum/log/learner.log` 读取，同时适配纯文本格式（原代码假设 JSON lines）。

### 14.2 环境数量递增缩放实验

**脚本**：`train/run_env_scaling_experiment.py`（新建）

**固定参数**：reverb_type=2, batch_size=2048, cache_multiplier=4, rate_limiter=MinSize

**实验矩阵**：4 / 6 / 8 / 12 环境数，每组 180 秒

**结果**：

| 环境数 | AISRV | step/min | mean_fetch | mean_train | mean_ratio | buffer |
|--------|-------|----------|-----------|-----------|-----------|--------|
| 4 | 1 | N/A (1行) | 747ms | 23ms | 9.9 | 2048/8192 |
| 6 | 2 | 118 | 838ms | 22ms | 8.3 | 2048/8192 |
| 8 | 2 | 130 | 869ms | 24ms | 8.2 | 2048/8192 |
| 12 | 3 | 120 | 823ms | 26ms | 5.7 | 2048/8192 |

完整数据：`train/context/data/ENV_SCALING_RESULTS.json`

### 14.3 核心结论

1. **data_fetch 不随环境数下降**：4→12 环境，fetch 始终在 700-1000ms 波动。
2. **step_per_min 基本持平**：118~130，增加环境数无显著吞吐增益。
3. **buffer 只用了 25%** (2048/8192)：空间充裕，不是瓶颈。
4. **real_train 稳定 22-27ms**：确认业务层训练不是瓶颈。
5. **瓶颈在 reverb 读路径本身**，不在数据生产端。

### 14.4 下一步建议

1. 不要再靠加环境数优化吞吐，已证明无效。
2. 优先排查 reverb dataset 读路径（`reverb_dataset_v1.py` 单线程预取、`remote_agent.py` 反序列化）。
3. batch_size 对照实验（1024 vs 2048）仍值得做。
4. 在 replay 读路径稳定前，不建议推进多 GPU / 多 learner。

### 14.5 涉及文件

- `train/run_env_scaling_experiment.py`（新建）
- `train/run_replay_stability_experiments.py`（修复 collect_rows）
- `train/.docker-compose.yaml`（修复 replace_toml_key）
- `train/context/data/ENV_SCALING_RESULTS.json`
