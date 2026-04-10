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
- [train/context/SESSION_LOG_20260409.md](/home/user/TcKaiwuFinal/train/context/SESSION_LOG_20260409.md)

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

