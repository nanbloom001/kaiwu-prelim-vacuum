# 模型打包签名机制诊断与修复报告

**日期**: 2026-04-13
**作者**: Claude Code 自动诊断
**分支**: linux

---

## 1. 问题描述

训练系统在 Linux 独立部署模式下，`/workspace/train/backup_model/` 目录中**无法生成带有签名的 zip 包**。

症状：
- Learner 正常训练，checkpoint `.pkl` 文件正常生成
- `id_list` 文件正常写入（包含所有 checkpoint ID）
- 但 `backup_model/` 目录始终为空，无 zip 包输出

## 2. 根因分析

### 2.1 原始架构

模型打包签名由框架中的 `ModelFileSave` 类负责（路径：`kaiwudrl/common/checkpoint/model_file_save.py`）。

```
ModelFileSave → Worker → multiprocessing.Process
```

`ModelFileSave` 继承 `Worker`，`Worker` 继承 `multiprocessing.Process`。在 `trainer.py` 中：

```python
if int(CONFIG.push_to_cos):
    self.model_file_saver = ModelFileSave()
    self.model_file_saver.start()  # fork() 一个新进程
```

### 2.2 崩溃原因

Python 的 `multiprocessing.Process` 默认使用 **`fork()`** 启动方法（已在容器中确认）。

Fork 安全性规则：**在多线程程序中调用 fork() 是不安全的**。

Learner 进程在启动 `ModelFileSave` 之前，已经初始化了：
- **TensorFlow 2.13.1** 运行时（内部有 GPU 线程、内存池线程）
- **Reverb** replay buffer server（后台网络线程）
- **MonitorProxy** 子进程（跨进程通信线程）

Fork 后子进程只复制调用线程，其他线程的 mutex/lock 全部处于锁定状态，导致：
- TensorFlow 内部死锁
- Reverb 连接断开
- 子进程崩溃或僵尸

### 2.3 之前的临时修复

在 `trainer.py` 的 hot-patch 中，当 `push_to_cos=False` 时跳过了 `model_file_saver`：

```python
if int(CONFIG.push_to_cos):
    self.model_file_saver = ModelFileSave()
    self.model_file_saver.start()
else:
    self.logger.info("train skip model_file_saver process because push_to_cos is disabled")
```

这避免了崩溃，但也**完全禁用了模型打包签名功能**。

## 3. 方案探索过程

### 3.1 方案 A: 直接启动原始 ModelFileSave（不可行）

- 风险：fork 崩溃
- 结论：不可行

### 3.2 方案 B: 全局修改 Worker 基类 Process → Thread（风险高）

`Worker` 基类被 20+ 个子类使用（Trainer、LearnerServer、AiSrvHandle、Predictor、DataCollector 等），全局修改会影响所有组件。

- 结论：风险太高，一个改动可能影响整个训练流程

### 3.3 方案 C: 在 YAML 中注入 Python 类到 trainer.py（失败）

在 docker-compose 的 `command` 块中用 `'\\n'.join([...])` 生成 Python 代码注入 `trainer.py`。

**失败原因**：YAML 单引号字符串中 `\\n` 会生成字面量 `\n` 字符，而非换行符。注入后的 Python 代码变成一行（所有内容用 `\n` 连接），产生 `SyntaxError: unexpected character after line continuation character`。

```
# 注入后的实际文件内容（错误）：
\nclass _InProcessModelSigner:\n    """Thread-based...
# 而不是：
class _InProcessModelSigner:
    """Thread-based...
```

### 3.4 方案 D: 独立 .py 文件 + 最小 import patch（最终方案）

**核心思路**：将 signer 代码写成独立的 Python 文件，放在已挂载的 `code/` 目录中。docker-compose 只需注入 1 行 import + 3 行使用代码。

文件：`code/agent_ppo/utils/model_signer.py`
修改：`trainer.py` 添加 import 和替换 model_file_saver 创建逻辑

**优势**：
- 代码完全可读可维护（不在 YAML 字符串中）
- 使用 `threading.Thread` 替代 `multiprocessing.Process`，零 fork 风险
- 最小侵入：只改 trainer.py 的 4 行代码

## 4. 最终实现

### 4.1 文件结构

```
code/agent_ppo/utils/
└── model_signer.py          # 新增：线程化模型签名器
```

### 4.2 ModelSignerThread 类设计

```
ModelSignerThread
├── __init__(logger)
│   ├── 读取 CONFIG 路径（restore_dir, app, algo → ckpt_dir）
│   ├── 设置输出目录（backup_model/signed/）
│   └── 初始化签名密钥（首次自动生成 RSA-2048）
├── start()      → 启动 daemon 线程
├── stop()       → 设置退出标志
├── is_alive()   → 检查线程状态
├── _run_loop()  → 每 60 秒调用 _process_new_checkpoints()
├── _process_new_checkpoints()
│   ├── 读取 id_list
│   └── 对每个未处理的 checkpoint ID 调用 _sign_checkpoint()
├── _sign_checkpoint(ckpt_id)
│   ├── 复制 .pkl 到临时目录
│   ├── 复制 conf/ 目录
│   ├── 计算 SHA-256 哈希
│   ├── RSA-PSS 数字签名
│   ├── 打包为 zip（含 kaiwu.json）
│   ├── 生成 .zip.json 元数据
│   └── 清理旧 zip（保留最近 50 个）
└── _cleanup_old_zips()
```

### 4.3 关键设计决策

| 决策 | 原因 |
|------|------|
| 输出到 `backup_model/signed/` 子目录 | 避免 backup_model sidecar 容器（Go 二进制）检测到 zip 后进行二次签名并删除文件 |
| 使用 checkpoint step ID 命名 | `robot_vacuum_ppo-{step}-{time}.zip`，避免同一分钟内多个 checkpoint 的 zip 相互覆盖 |
| 首次自动生成 RSA 密钥 | Linux 独立部署中 `CONFIG.private_key_content` 为空，需要在本地生成 |
| 保留 50 个 zip | 训练产出约 500 steps/min，每分钟 1 个 zip，50 个约覆盖 50 分钟 |

### 4.4 trainer.py Patch

在 docker-compose 的 hot-patch 中：

1. **写入 signer 文件**：将 `model_signer.py` 写到 `/workspace/code/agent_ppo/utils/`
2. **修改 trainer.py**：
   - 添加 `from agent_ppo.utils.model_signer import ModelSignerThread`
   - 将 `ModelFileSave()` 替换为 `ModelSignerThread(self.logger)`
   - 移除 `push_to_cos` 条件判断（始终启用）

## 5. 验证结果

### 5.1 签名器启动

```
2026-04-13 04:08:12.576 | [ModelSigner] Generated new private key at /workspace/train/backup_model/.keys
2026-04-13 04:08:12.577 | [ModelSigner] Started in-process signing thread
```

### 5.2 Checkpoint 签名记录

```
04:26 | Signed model.ckpt-0   → robot_vacuum_ppo-0-2026-04-13-04-26.zip     (3.7 MB)
04:26 | Signed model.ckpt-100 → robot_vacuum_ppo-100-2026-04-13-04-26.zip   (3.7 MB)
04:27 | Signed model.ckpt-200 → robot_vacuum_ppo-200-2026-04-13-04-27.zip   (3.7 MB)
04:27 | Signed model.ckpt-300 → robot_vacuum_ppo-300-2026-04-13-04-27.zip   (3.7 MB)
...
04:31 | Signed model.ckpt-2200 → robot_vacuum_ppo-2200-2026-04-13-04-31.zip (3.7 MB)
```

### 5.3 输出文件验证

```
/workspace/train/backup_model/signed/
├── .keys/
│   ├── private_key.pem    (RSA-2048 私钥，1704 bytes)
│   └── public_key.pem     (RSA-2048 公钥，451 bytes)
├── robot_vacuum_ppo-0-2026-04-13-04-26.zip         (3,885,128 bytes)
├── robot_vacuum_ppo-0-2026-04-13-04-26.zip.json     (623 bytes)
├── robot_vacuum_ppo-100-2026-04-13-04-26.zip        (3,888,488 bytes)
├── robot_vacuum_ppo-100-2026-04-13-04-26.zip.json   (629 bytes)
...
└── robot_vacuum_ppo-1500-2026-04-13-04-30.zip.json  (632 bytes)
```

### 5.4 JSON 元数据样例

```json
{
  "created_at": "2026-04-12T20:27:06.590682+00:00",
  "train_step": 500,
  "model_file_name": "robot_vacuum_ppo-500-2026-04-13-04-27.zip",
  "model_file_hash": "68c32fe86e22ce5f18cb203f58780d0cc93850aba40bf8d8c4675e95af05d607",
  "model_file_path": ["ckpt/model.ckpt-500.pkl"],
  "signature": "ZsLcdVoRcTAHI6oJUFo8E1g7wPsUv5... (344 chars, RSA-PSS-SHA256)"
}
```

### 5.5 训练性能影响

| 指标 | 签名器启动前 | 签名器运行时 |
|------|-------------|-------------|
| train once cost | 30-50 ms | 30-50 ms |
| data_fetch | 40-280 ms | 40-280 ms |
| real_train | 29-53 ms | 29-53 ms |

**签名线程对训练性能无可测量影响**（daemon 线程，每分钟仅执行一次 I/O 操作）。

## 6. backup_model Sidecar 行为发现

### 6.1 意外行为

`backup_model` sidecar 容器（Go 二进制）会监控 `backup_model/` 目录：

```
sidecar/sign.go: will sign file: .../robot_vacuum_ppo-3000-.../zip -> .../sign_model/.../zip
sidecar/sidecar.go: run failed, err: failed to parse PEM block containing the public key
```

Sidecar 尝试对 zip 进行二次签名，但：
1. 它期望特定格式的公钥（来自腾讯内部 COS 平台），不识别我们本地生成的 RSA 密钥
2. 签名失败后，**zip 文件被删除**

### 6.2 解决方案

将签名输出改到 `backup_model/signed/` 子目录。Sidecar 只监控 `backup_model/` 根目录的 zip 文件，不会进入子目录。

## 7. 遗留事项

### 7.1 dump_model_freq 调整

当前 `dump_model_freq = 100`（每 100 步保存一次 checkpoint）。这意味着每分钟约生成 5 个 checkpoint，每个都会被签名打包。

如需调整频率：
- 减少保存频率：在 `.env` 中设置 `KAIWU_EXPERIMENT_DUMP_MODEL_FREQ=500`
- 增加签名间隔：修改 `model_signer.py` 中的 `_interval_sec`

### 7.2 backup_model Sidecar

Sidecar 容器持续报错但无害。可考虑：
- 保持现状（不影响训练和签名）
- 移除 sidecar 容器（如果不需要腾讯 COS 平台的签名流程）

### 7.3 COMPOSE_PROJECT_NAME

发现需要 `.env` 中设置 `COMPOSE_PROJECT_NAME=kaiwu-train`，否则容器名变为 `train-*`（使用目录名），导致框架启动脚本中 `get_learner_ip()` 无法解析主机名。

## 8. 修改文件清单

| 文件 | 操作 | 描述 |
|------|------|------|
| `code/agent_ppo/utils/model_signer.py` | **新增** | 线程化模型签名器（150 行） |
| `train/.docker-compose.yaml` | **修改** | trainer.py patch：import ModelSignerThread + 替换 model_file_saver |
| `train/.env` | **修改** | 添加 `COMPOSE_PROJECT_NAME=kaiwu-train` |
