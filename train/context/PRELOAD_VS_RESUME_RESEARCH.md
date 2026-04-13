# Preload Model 替代 Resume 可行性研究报告

> 日期: 2026-04-13
> 状态: 研究完成，待决策

---

## 一、核心发现：初始化时序冲突

**这是最关键的发现**——preload 和 resume 不是等效的替代关系，它们的执行时序完全不同：

```
Learner 启动流程（按时间顺序）：

1. Agent.__init__() 执行
   ├── 模型创建（随机初始化）
   ├── 【自定义 Resume 加载】← agent.py:166-182，torch.load + load_state_dict
   └── super().__init__() 设置 framework_handler

2. create_standard_agent_wrapper() 执行
   ├── 创建 agent_wrapper，包装 agent
   └── agent.framework_handler = agent_wrapper

3. trainer.before_run() 执行
   ├── 【框架 Preload 加载】← trainer.py:360，agent_wrapper.preload_model_file()
   │   └── 调用 agent.load_model(path, id, framework=True)
   │       └── 触发 RemoteAgent 拦截器 → _business_load_model
   ├── PyTorch 初始化：save_param_by_source(id=0)
   └── update_id_list(preload_model_id)
```

**关键问题：如果两者同时启用，模型会被加载两次。**

- Resume 先加载（步骤 1），将自定义路径的权重灌入模型
- Preload 后加载（步骤 3），会**覆盖** Resume 的结果，从 `preload_model_dir` 重新加载

这意味着：**不能用 preload 直接替代 resume，必须二选一**。如果启用了 preload，resume 的加载会被覆盖，变成无效操作。

---

## 二、Preload Model 机制详解

### 2.1 完整调用链

```
trainer.py:360            CONFIG.preload_model == true 时触发
  → LoadModelCommon.preload_model_file(policy_agent_wrapper_maps)
    → check_path_id_valid(dir, id)              # 验证目录存在、id ≥ 0
    → agent_wrapper.preload_model_file(dir, id)  # PyTorch 版
      → agent.load_model(path=dir, id=id, framework=True)  # 调用业务的 load_model
      → self.train_count = preload_model_id
      → self.preload_model_train_count = preload_model_id  # 修正样本统计
```

### 2.2 Learner 端的初始化后操作

Preload 完成后，trainer 做了两件关键事：

```python
# trainer.py:393-404
# 1. 保存一份初始模型文件（id=0）
self.agent_wrapper.save_param_by_source(source=FRAMEWORK)

# 2. 更新 id_list 文件
if int(CONFIG.preload_model):
    update_id_list(CONFIG.preload_model_id, framework=True)  # 用 preload_id
else:
    update_id_list(0, framework=True)
```

这确保了 on-policy 的模型版本追踪从正确的 ID 开始。

### 2.3 AISRV 端的 Preload

```python
# predictor_local.py:612-618
if int(CONFIG.preload_model):
    if CONFIG.run_mode == KaiwuDRLDefine.RUN_MODE_TRAIN:
        self.load_model_common_object.preload_model_file(self.policy_agent_wrapper_maps)
```

**AISRV 也会执行 preload**，这意味着 learner 和 aisrv 在训练开始时就拥有相同的模型权重，避免了随机初始化产生的低质量样本。

### 2.4 On-Policy 兼容性

Preload 与 on-policy 流程完全兼容：

1. `save_param_by_source(id=0)` → 保存初始模型
2. `update_id_list(preload_model_id)` → 设置起始版本号
3. 第一次 `after_train()` → `train_count` 从 `preload_model_id` 递增
4. `process_policy_specific(model_file_id)` → 通知 aisrv/actor 更新版本

`preload_model_train_count` 用于修正样本消耗比：
```python
# trainer.py:224
ratio = (train_count - preload_model_train_count) * batch_size / sample_receive_cnt
```

---

## 三、自定义 Resume 机制详解

### 3.1 完整快照系统

当前 resume 系统不只是一个加载点，而是一套**持续快照管理系统**：

| 触发类型 | 配置项 | 默认值 | 保存内容 |
|---------|--------|--------|---------|
| Episode 快照 | `RESUME_EPISODE_SNAPSHOT_INTERVAL` | 50 episodes | state_dict + meta.json |
| Time 快照 | `RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS` | 15 分钟 | state_dict + meta.json |
| Best 快照 | 当 `is_new_best=True` | 自动 | state_dict 到 session_best/ |
| Latest 同步 | `RESUME_LATEST_SYNC_INTERVAL_EPISODES` | 20 episodes | 覆盖 model.ckpt-resume.pkl |
| Manual 检查点 | `SAVE_MODEL_INTERVAL_EPISODES` | 50 episodes | 调用 agent.save_model() |

### 3.2 保存的元数据

```json
// model.ckpt-resume.meta.json
{
  "trigger": "episode",
  "episode_cnt": 1600,
  "clean_score": 1123.0,
  "saved_at": "2026-04-13 13:47:49",
  "pid": 426
}
```

### 3.3 目录结构

```
code/
├── model.ckpt-resume.pkl          # 主恢复点（持续覆盖更新）
├── model.ckpt-resume.meta.json    # 元数据
├── latest_model.pkl               # 别名（与 resume.pkl 同步）
├── resume_snapshots/               # 快照目录
│   ├── resume-episode-ep1600.pkl  # Episode 快照（保留 8 个）
│   ├── resume-episode-ep1650.pkl
│   ├── resume-time-20260413-134345.pkl  # Time 快照（保留 6 个）
│   └── ...
├── session_best/20260413-053807/  # 最佳模型（按 session 组织）
│   ├── best-ep719-score1505.pkl   # 当前最佳
│   └── ...
└── manual_checkpoints/             # 框架检查点（空）
```

### 3.4 加载行为

```python
# agent.py:166-182
if Config.RESUME_CHECKPOINT:        # 如 "model.ckpt-resume.pkl"
    for path in candidates:
        if os.path.isfile(path):
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            break
```

特点：
- 仅加载模型权重（state_dict），不加载 optimizer 状态
- 不恢复 episode_cnt（由 EpisodeRunner 单独维护）
- 文件不存在时静默继续（从零训练）
- 加载失败时打印警告，继续从随机权重训练
- 仅在 Learner 端加载，AISRV 端不加载

---

## 四、对比分析

### 4.1 功能对比

| 维度 | Preload Model | 自定义 Resume |
|------|--------------|---------------|
| **加载时机** | Agent 创建后、训练循环前 | Agent `__init__` 中（更早） |
| **加载位置** | Learner + AISRV 都加载 | 仅 Learner |
| **文件命名** | `model.ckpt-{整数id}.pkl` | 任意文件名 |
| **文件路径** | `preload_model_dir`（需是存在的目录） | 任意路径 |
| **训练计数修正** | 自动（`train_count = preload_model_id`） | 不修正（从 0 开始计数） |
| **样本统计修正** | 自动（`preload_model_train_count`） | 不修正（导致消耗比偏高） |
| **AISRV 同步** | 框架自动在两端加载 | 不同步（aisrv 用随机权重直到第一次模型更新） |
| **失败处理** | 硬终止（`learner_process_stop`） | 软失败（打印警告，继续训练） |
| **On-Policy 兼容** | 完全兼容（修正 model_version 起始值） | 兼容但 model_version 从 0 开始 |
| **多次快照** | 不支持（只加载一个文件） | 支持（episode/time/best/latest 四种触发器） |
| **元数据** | 不保存 | 保存 trigger/episode_cnt/score/timestamp |
| **连续备份** | 不支持 | 训练中持续保存快照 |
| **配置方式** | `configure_app.toml` | `conf.py` + 环境变量 |

### 4.2 Preload 优于 Resume 的方面

1. **AISRV 同步加载**：Preload 在 learner 和 aisrv 两端都加载模型。当前 resume 只在 learner 加载，aisrv 从随机权重开始预测，直到第一次 on-policy 模型同步（可能需要数十个训练步），这段时间产出的样本质量极低。

2. **训练计数统计修正**：Preload 设置 `train_count` 和 `preload_model_train_count`，使得样本消耗比指标正确。当前 resume 不修正，导致指标虚高。

3. **On-Policy 版本追踪**：Preload 通过 `update_id_list(preload_model_id)` 让 on-policy 流程从正确的版本号开始。当前 resume 从版本 0 开始，第一轮 on-policy 同步时 aisrv 可能版本不匹配。

4. **硬终止保护**：Preload 失败时直接退出，避免用随机权重继续训练浪费资源。Resume 的软失败可能导致用户不知道训练实际上是从零开始的。

### 4.3 Resume 优于 Preload 的方面

1. **持续快照**：Resume 在训练过程中持续保存多种快照（episode/time/best），提供多个恢复点。Preload 只在启动时加载一次。

2. **元数据追踪**：Resume 保存 episode_cnt、clean_score、触发类型等上下文信息。Preload 无元数据。

3. **灵活的文件命名**：Resume 不受 `model.ckpt-{int_id}.pkl` 命名约束。

4. **Best 模型管理**：Resume 有独立的 session_best 目录，追踪历史最佳模型。Preload 无此功能。

5. **多恢复点**：保留 8 个 episode 快照 + 6 个 time 快照 + 5 个 best 快照，可从任意一个恢复。Preload 只有单文件。

---

## 五、关键问题与风险

### 5.1 时序冲突（致命问题）

如前所述，resume 在 `Agent.__init__` 中执行（步骤 1），preload 在 trainer 初始化后执行（步骤 3）。如果同时启用：
- Resume 先加载权重 A
- Preload 后加载权重 B，覆盖 A

**结论：不能同时启用，必须选择一个或设计协调机制。**

### 5.2 文件命名约束

Preload 要求文件名格式为 `model.ckpt-{整数id}.pkl`。当前 resume 快照命名为：
- `model.ckpt-resume.pkl` → 不兼容（"resume" 不是整数）
- `resume-episode-ep1600.pkl` → 不兼容
- `best-ep719-score1505.pkl` → 不兼容

如要使用 preload，需要在保存快照时改为 `model.ckpt-{episode_cnt}.pkl` 格式。

### 5.3 配置动态化

当前 preload 通过 `configure_app.toml` 配置，是静态的。需要通过 docker-compose hot-patch 的 `replace_toml_key()` 动态注入：

```python
# 需要添加到 learner 和 aisrv 的 hot-patch 中
replace_toml_key(app_conf_target, "preload_model", os.environ.get("KAIWU_PRELOAD_MODEL", "false"))
replace_toml_key(app_conf_target, "preload_model_dir", os.environ.get("KAIWU_PRELOAD_MODEL_DIR", ""))
replace_toml_key(app_conf_target, "preload_model_id", os.environ.get("KAIWU_PRELOAD_MODEL_ID", "0"))
```

### 5.4 Episode 计数丢失

无论是 preload 还是当前 resume，都不保存/恢复 optimizer 状态和 episode 计数。Episode 计数由 `EpisodeRunner.episode_cnt` 维护，存储在 Python 进程内存中，进程重启后归零。

这意味着**即使使用 preload 恢复模型权重，训练日志中的 episode 编号也会从 0 重新开始**，无法与历史训练连续。

---

## 六、改进方向

### 方案 A：纯 Preload 替代（简单但不完整）

**思路**：禁用自定义 resume，完全使用 preload。

**需要改动**：
1. `conf.py`：`RESUME_CHECKPOINT = None`
2. 保存快照时，将文件命名为 `model.ckpt-{episode_cnt}.pkl` 格式
3. 放入 `{agent_name}/ckpt/` 目录
4. 在 `.env` 中配置 `KAIWU_PRELOAD_MODEL=true`，`KAIWU_PRELOAD_MODEL_DIR`，`KAIWU_PRELOAD_MODEL_ID`
5. 添加 hot-patch 注入 preload 配置

**优点**：简单，获得 AISRV 同步和统计修正
**缺点**：丢失所有快照管理、元数据追踪、多恢复点功能

### 方案 B：Preload + 增强版快照系统（推荐）

**思路**：保留自定义快照系统用于持续备份，但用 preload 作为实际的恢复加载机制。

**具体方案**：
1. 保留当前快照系统的所有触发器（episode/time/best/latest）
2. 每次保存快照时，**额外**保存一份符合 preload 命名约定的文件到 `agent_ppo/ckpt/` 目录
3. 恢复时通过 `.env` 配置 `preload_model_id` 指向目标 episode 的快照
4. 保留 `.meta.json` 用于人工参考，但实际加载由 preload 完成

**代码改动示意**（在 `_save_resume_artifacts` 中）：
```python
def _save_resume_artifacts(self, trigger, clean_score, with_named_snapshot=False):
    state_dict = self._snapshot_state_dict()
    # 现有的保存逻辑...
    self._write_state_dict(self.resume_latest_path, state_dict)
    
    # 新增：保存一份符合 preload 命名约定的文件
    preload_dir = self.code_path / "agent_ppo" / "ckpt"
    preload_dir.mkdir(exist_ok=True)
    preload_path = preload_dir / f"model.ckpt-{self.episode_cnt}.pkl"
    self._write_state_dict(preload_path, state_dict)
    
    # 清理旧的 preload 文件（只保留最新的）
    for f in preload_dir.glob("model.ckpt-*.pkl"):
        if f != preload_path:
            f.unlink(missing_ok=True)
```

**恢复流程**：
1. 找到最新或最佳的 `.meta.json`，获取 `episode_cnt`
2. 确认 `agent_ppo/ckpt/model.ckpt-{episode_cnt}.pkl` 存在
3. 在 `.env` 中设置：
   ```
   KAIWU_PRELOAD_MODEL=true
   KAIWU_PRELOAD_MODEL_DIR=agent_ppo/ckpt
   KAIWU_PRELOAD_MODEL_ID={episode_cnt}
   ```
4. 在 `conf.py` 中设置 `RESUME_CHECKPOINT = None`（禁用自定义 resume）
5. 重启容器

**优点**：
- 获得 preload 的所有优势（AISRV 同步、统计修正、硬终止保护）
- 保留快照管理系统（多恢复点、元数据、best 追踪）
- 文件命名自动适配 preload 约定

### 方案 C：优化现有 Resume（保守方案）

**思路**：不用 preload，但在现有 resume 基础上补齐 preload 的优势。

**需要补齐的功能**：
1. **AISRV 同步**：在 workflow 中，resume 加载后立即通过 `agent.save_model()` 保存一份到框架路径，触发 on-policy 同步
2. **训练计数修正**：在 resume 加载后，通过 `agent.framework_handler` 设置 `train_count`
3. **硬终止保护**：在 resume 加载失败时退出进程而非静默继续

**限制**：AISRV 端的 resume 需要在 `predictor_local.py` 的 `before_run` 中也加载一次，但业务代码在 aisrv 端无法访问 framework_handler（因为 resume 在 `__init__` 中执行时 framework_handler 尚未设置）。

---

## 七、最终建议

**推荐方案 B（Preload + 增强版快照系统）**，理由：

| 评估维度 | 方案 A（纯 Preload） | 方案 B（混合） | 方案 C（优化 Resume） |
|---------|-------------------|--------------|-------------------|
| 实现复杂度 | 低 | 中 | 高 |
| AISRV 同步 | 有 | 有 | 难以实现 |
| 统计修正 | 有 | 有 | 需要额外实现 |
| 多恢复点 | 无 | 有 | 有 |
| 元数据追踪 | 无 | 有 | 有 |
| On-Policy 兼容 | 完全兼容 | 完全兼容 | 需要额外处理 |
| 风险 | 中（丢失快照功能） | 低 | 高（AISRV 同步难解决） |

**方案 B 的实施步骤**：

1. 在 `_save_resume_artifacts` 中添加 preload 格式文件保存
2. 禁用 `agent.py` 中的自定义 resume（`RESUME_CHECKPOINT = None`）
3. 在 docker-compose hot-patch 中添加 preload 配置注入
4. 在 `.env` 中添加 `KAIWU_PRELOAD_MODEL`、`KAIWU_PRELOAD_MODEL_DIR`、`KAIWU_PRELOAD_MODEL_ID`
5. 编写测试验证 preload 加载 + 快照系统联动
6. 创建恢复脚本：读取最新 `.meta.json`，自动配置 `.env` 中的 preload 参数

**预估工作量**：约 2-3 小时（代码改动量小，主要是配置和测试）
