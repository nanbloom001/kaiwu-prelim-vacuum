# Resume 控制修复 + 本地独立训练验证

> 日期：2026-04-13 00:00 ~ 00:30
> 触发：用户报告官方平台启动训练时总是触发 resume，entropy 几乎为 0

---

## 1. 问题诊断

### 现象
- 从开悟官方平台启动训练后，训练总是自动 resume（加载历史 checkpoint）
- Entropy 在 0.0 ~ 0.9 之间剧烈振荡（非持续为 0，而是 adaptive beta 的阶跃式调整导致过冲）

### 根因
在之前的 session 中，我添加了两处自动加载 `model.ckpt-resume.pkl` 的代码：

**位置 A：`code/agent_ppo/agent.py` 第 55-73 行**
```python
# Auto-load resume checkpoint if available (for fine-tuning)
_resume_candidates = [
    os.path.join(os.path.dirname(__file__), "..", "model.ckpt-resume.pkl"),
    "/workspace/code/model.ckpt-resume.pkl",
]
for _resume_path in _resume_candidates:
    if os.path.isfile(_resume_path):
        state_dict = torch.load(_resume_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        break
```
- 在 `Agent.__init__()` 中执行
- learner 和 aisrv **都会执行**（两边各创建一个 Agent 实例）
- 只要文件存在就无条件加载，无配置开关

**位置 B：`code/agent_ppo/workflow/train_workflow.py` 第 57-65 行**
```python
resume_ckpt = shared_code_dir / "model.ckpt-resume.pkl"
if resume_ckpt.exists():
    state_dict = torch.load(resume_ckpt, map_location=agent.device, weights_only=True)
    agent.model.load_state_dict(state_dict)
```
- 在 `workflow()` 函数中执行
- 仅 aisrv 侧运行
- 实际上**冗余**（agent.py 已经加载过）

**问题链**：`model.ckpt-resume.pkl` 始终存在于 code 目录（训练中 `_save_resume_artifacts` 会持续覆写）→ 每次启动都自动 resume → 无法从零训练

---

## 2. 修复方案

### 改动 1：`conf.py` 新增配置开关

```python
# code/agent_ppo/conf/conf.py
RESUME_CHECKPOINT = None  # None = 从零训练; "model.ckpt-resume.pkl" = resume
```

### 改动 2：`agent.py` 替换为配置驱动

将硬编码的自动检测逻辑替换为读取 `Config.RESUME_CHECKPOINT`：
- `None` → 不加载任何文件，从随机权重开始
- 非空字符串 → 尝试从两个候选路径加载该文件

### 改动 3：`train_workflow.py` 删除冗余 resume

删除 workflow 级的 resume 代码，仅保留 agent.py 中的一处加载逻辑。

### 验证结果
- Learner 日志无 `[RESUME]` 消息
- Entropy = 2.07（接近 ln(8) ≈ 2.08，完全随机，正常的初始值）
- global step = -1（全新训练）

---

## 3. 本地独立训练架构发现

### 核心发现：容器命名前缀

| 启动方式 | 容器前缀 | hostname 可解析 | 能否训练 |
|---------|---------|----------------|---------|
| 官方平台 | `kaiwu-train-learner-1` | 是 | 能 |
| 本地 compose（默认） | `train-learner-1` | 否 | 崩溃 |
| 本地 compose（`-p kaiwu-train`） | `kaiwu-train-learner-1` | 是 | **能** |

框架启动脚本 `start_train_client.sh` 内部硬编码：
```bash
getent hosts kaiwu-train-learner-1 kaiwu-train_learner_1
```
用 `-p kaiwu-train` 指定 compose 项目名即可匹配。

### 之前 resume 能工作的原因
之前的操作是 `docker stop/start` 官方平台已创建的容器（不改变容器名），而非创建新容器。所以 hostname 始终能解析。

### 完整独立启动命令

```bash
# 停止并清理
cd D:/TcKaiwuFinal/train
docker compose -f .docker-compose.yaml --profile distributed -p kaiwu-train down -v

# 启动（从零训练，conf.py 中 RESUME_CHECKPOINT = None）
docker compose -f .docker-compose.yaml --profile distributed -p kaiwu-train up -d

# 启动（resume 训练，先改 conf.py: RESUME_CHECKPOINT = "model.ckpt-resume.pkl"）
docker compose -f .docker-compose.yaml --profile distributed -p kaiwu-train up -d
```

共启动 17 个容器：learner(1) + aisrv(2) + gamecore(8) + pushgateway + greptimedb + vector + monitor-service + fe-monitor-service + backup_model

### 验证结果
- 17 个容器全部 `Up` 且稳定运行
- Learner 正常训练（entropy=2.05, step=28）
- 完全独立于开悟平台

---

## 4. 监控面板

### 访问地址
```
http://127.0.0.1:11000/p/v5/exp/monitor?domain_id=1&exp_id=1&task_uuid=1&task_id=0&platform=competition_stage&lang=zh&train_create_at=2026-04-12T13:48:46.843Z
```

### 关于 train_create_at 参数
- 仅影响前端展示（如"训练开始时间"标签），**不影响数据查询**
- 监控数据来自 greptimedb 时序数据库，始终查询最新数据
- 因此填任意时间都能看到当前训练的实时指标

### server_req_base_url 注意事项
- 原始配置 `http://127.0.0.1:${MONITOR_TRPC_PORT}`（即 `http://127.0.0.1:11001`）是**正确的，不要修改**
- 原因：`fe-monitor-service` 的前端是 Next.js 应用，页面中的 JS 在**用户浏览器（宿主机）** 中执行
- 浏览器访问 `127.0.0.1:11001` → 通过 Docker 端口映射 `11001:8040` → 到达 `monitor-service` 容器 → 正常工作
- 如果改成 `http://monitor-service:8040`（Docker 内部 DNS），浏览器无法解析该域名 → fail to fetch

### 面板打不开的常见原因
- 容器未完全就绪，greptimedb healthcheck 需要约 30 秒
- 等待所有容器 `Up` 后再访问

---

## 5. 修改文件清单

| 文件 | 修改内容 |
|------|---------|
| `code/agent_ppo/conf/conf.py` | 新增 `RESUME_CHECKPOINT = None` |
| `code/agent_ppo/agent.py` | 替换硬编码 resume 为 Config 驱动 |
| `code/agent_ppo/workflow/train_workflow.py` | 删除冗余的 workflow 级 resume 代码 |
| `train/.docker-compose.yaml` | 未修改（`server_req_base_url` 回退为原值） |
