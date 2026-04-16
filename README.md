# TcKaiwuFinal

腾讯 Arena「扫地机器人」强化学习训练仓库，基于 KaiwuDRL 框架和 PPO 算法，当前主工作流运行在 **Linux + Docker Compose**。

这个仓库的目标不是只保存 agent 代码，而是把**训练、评估、日志、实验脚本和上下文文档**放在一起，方便直接在本地服务器上启动和迭代。

## 快速开始

最常用的入口都在 `train/` 目录。

### 1. 启动训练

```bash
cd train

# 启动完整分布式训练栈
docker compose -f .docker-compose.yaml --profile distributed up -d

# 修改了 .env / compose / patch 后，强制重建
docker compose -f .docker-compose.yaml --profile distributed up -d --force-recreate

# 停止训练
docker compose -f .docker-compose.yaml --profile distributed down
```

### 2. 看训练状态

```bash
# learner 日志最重要
docker logs -f kaiwu-train-learner-1

# 查看容器状态
docker compose -f .docker-compose.yaml ps
```

### 3. 跑串行 benchmark

```bash
cd train

# 使用 conf.py 默认 checkpoint
bash run_benchmark.sh

# 指定 checkpoint
bash run_benchmark.sh saved_models/v53-robust3450/model.ckpt-resume.pkl

# benchmark 后自动重启训练
RESTART=1 bash run_benchmark.sh
```

### 4. 跑并行 benchmark

当前推荐固定使用 `4×10`：

```bash
cd train
bash run_benchmark_parallel.sh --workers 4 --envs-per-worker 10 --max-wait 1800
```

指定 checkpoint：

```bash
cd train
bash run_benchmark_parallel.sh saved_models/v53-robust3450/model.ckpt-resume.pkl \
  --workers 4 \
  --envs-per-worker 10 \
  --max-wait 1800
```

### 5. 查看 benchmark 结果

```bash
cd train

# 串行 benchmark 结果
python3 compare_benchmarks.py latest

# 对比两次串行 benchmark
python3 compare_benchmarks.py 0 1
```

结果文件位置：

- 串行 benchmark 汇总：`train/eval_results.json`
- 串行 benchmark 明细：`train/eval_logs/<session_id>/`
- 并行 benchmark 汇总：`train/eval_parallel_results.json`
- 并行 benchmark 明细：`train/eval_parallel_logs/<session_id>/`

## 仓库结构

```text
TcKaiwuFinal/
├── README.md
├── CLAUDE.md                      # 当前仓库的工程说明与常用命令
├── code/                          # 挂载到容器 /workspace/code 的业务代码
│   ├── agent_ppo/                 # 主 PPO agent
│   ├── agent_diy/                 # 框架要求保留的自定义 agent 入口
│   ├── conf/                      # Kaiwu 框架侧 TOML 配置
│   ├── saved_models/              # 手动保存/评估用模型
│   └── tests/                     # 单元测试
├── train/                         # Docker、训练脚本、评估脚本、日志和 context
│   ├── .docker-compose.yaml
│   ├── .docker-compose.benchmark.yaml
│   ├── .env
│   ├── run_benchmark.sh
│   ├── run_benchmark_parallel.sh
│   ├── compare_benchmarks.py
│   ├── context/
│   ├── eval_logs/
│   ├── eval_parallel_logs/
│   └── log/
├── tencentarena-docs/             # 官方赛事/框架文档
└── license.dat                    # 训练授权文件
```

## 主要入口文件

如果你第一次接手这个仓库，建议按下面顺序读。

### 训练与评估入口

- [train/.docker-compose.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.yaml)
  - 正常训练栈定义，包含 learner / aisrv / gamecore / backup_model / monitor 链路。
- [train/.docker-compose.benchmark.yaml](/home/user/TcKaiwuFinal/train/.docker-compose.benchmark.yaml)
  - 并行 benchmark 专用 override，只覆盖 benchmark 需要的环境变量和重启策略。
- [train/run_benchmark.sh](/home/user/TcKaiwuFinal/train/run_benchmark.sh)
  - 串行 benchmark 入口。
- [train/run_benchmark_parallel.sh](/home/user/TcKaiwuFinal/train/run_benchmark_parallel.sh)
  - 并行 benchmark 入口，当前默认建议用 `4×10`。
- [train/compare_benchmarks.py](/home/user/TcKaiwuFinal/train/compare_benchmarks.py)
  - benchmark 对比工具。

### Agent 核心代码

- [code/agent_ppo/agent.py](/home/user/TcKaiwuFinal/code/agent_ppo/agent.py)
  - PPO agent 主入口，含预测、学习、模型加载、runtime 优化。
- [code/agent_ppo/workflow/train_workflow.py](/home/user/TcKaiwuFinal/code/agent_ppo/workflow/train_workflow.py)
  - workflow 主流程，决定训练、串行 benchmark、并行 benchmark 的分流入口。
- [code/agent_ppo/eval/benchmark.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark.py)
  - 串行 benchmark 实现。
- [code/agent_ppo/eval/benchmark_parallel.py](/home/user/TcKaiwuFinal/code/agent_ppo/eval/benchmark_parallel.py)
  - 并行 benchmark 实现。

### 特征、规则与超参

- [code/agent_ppo/feature/preprocessor.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/preprocessor.py)
  - 状态特征构造、reward shaping、地图/充电桩相关特征。
- [code/agent_ppo/feature/expert.py](/home/user/TcKaiwuFinal/code/agent_ppo/feature/expert.py)
  - expert bias / safety filter / A* 充电策略。
- [code/agent_ppo/conf/conf.py](/home/user/TcKaiwuFinal/code/agent_ppo/conf/conf.py)
  - Python 侧训练超参与模型结构配置。
- [code/conf/configure_app.toml](/home/user/TcKaiwuFinal/code/conf/configure_app.toml)
  - Kaiwu 框架侧配置，如 batch、dump_model_freq、replay 相关项。

### 上下文文档

- [train/context/README.md](/home/user/TcKaiwuFinal/train/context/README.md)
  - context 文档索引。
- [train/context/benchmark/BENCHMARK_SYSTEM.md](/home/user/TcKaiwuFinal/train/context/benchmark/BENCHMARK_SYSTEM.md)
  - 串行 benchmark 说明。
- [train/context/benchmark/BENCHMARK_PARALLEL_HANDOFF_20260416.md](/home/user/TcKaiwuFinal/train/context/benchmark/BENCHMARK_PARALLEL_HANDOFF_20260416.md)
  - 并行 benchmark 的实现、问题、排障和推荐配置。

## 当前默认工作流

### 日常训练

1. 调整 `train/.env`、`code/agent_ppo/conf/conf.py` 或相关特征/规则代码。
2. 在 `train/` 下执行：

```bash
docker compose -f .docker-compose.yaml --profile distributed up -d --force-recreate
```

3. 用 `docker logs -f kaiwu-train-learner-1` 观察：
   - loss
   - entropy
   - data_fetch / real_train
   - 保存 checkpoint 的节奏
4. 训练一段时间后停栈，再跑 benchmark 做 A/B 对比。

### 日常评估

如果只想要稳定、可比较的标准评估：

```bash
cd train
bash run_benchmark.sh saved_models/<your_model>/model.ckpt-resume.pkl
```

如果想缩短评估墙钟时间：

```bash
cd train
bash run_benchmark_parallel.sh saved_models/<your_model>/model.ckpt-resume.pkl \
  --workers 4 \
  --envs-per-worker 10 \
  --max-wait 1800
```

注意：

- 并行 benchmark 会复用 `kaiwu-train` 这个 compose project 名，不能和正常训练并发启动。
- 模型之间做对比时，benchmark 拓扑要固定，不要这次用 `4×10`、下次用 `10×4`。

## 常用命令速查

```bash
# 启动训练
cd train && docker compose -f .docker-compose.yaml --profile distributed up -d

# 强制重建训练栈
cd train && docker compose -f .docker-compose.yaml --profile distributed up -d --force-recreate

# 停止训练
cd train && docker compose -f .docker-compose.yaml --profile distributed down

# learner 日志
docker logs -f kaiwu-train-learner-1

# 串行 benchmark
cd train && bash run_benchmark.sh saved_models/v53-robust3450/model.ckpt-resume.pkl

# 并行 benchmark（推荐）
cd train && bash run_benchmark_parallel.sh --workers 4 --envs-per-worker 10 --max-wait 1800

# 查看最新 benchmark 对比
cd train && python3 compare_benchmarks.py latest

# 运行单测
cd code && python3 -m pytest tests/ -v
```

## 环境要求

- Linux
- Docker + Docker Compose
- NVIDIA GPU 驱动与容器运行时
- Python 3.10+ 用于本地脚本和测试

## 补充说明

- 当前 `linux` 分支与早期 `win` 工作流已经不完全一致，优先以本 README、`CLAUDE.md` 和 `train/context/` 中的最新文档为准。
- 如果只是想快速上手，先看本 README 的“快速开始”和“主要入口文件”两节就够了。
