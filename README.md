# TcKaiwuFinal — 腾讯开悟机器人吸尘器 PPO 训练项目

基于腾讯开悟 DRL 框架的机器人吸尘器强化学习训练项目，使用 PPO 算法。

## 仓库结构

```
TcKaiwuFinal/
├── code/                          # AI 代码（挂载到容器 /workspace/code）
│   ├── agent_ppo/                 # PPO Agent 实现
│   │   ├── agent.py               # Agent 主类，含 expert override
│   │   ├── feature/
│   │   │   ├── preprocessor.py    # 状态特征预处理
│   │   │   ├── definition.py      # 特征定义常量
│   │   │   └── expert.py          # ExpertPolicy：低电量充电导航
│   │   ├── model/model.py         # 神经网络结构定义
│   │   ├── conf/conf.py           # 超参数配置（SCALAR_DIM=74 等）
│   │   ├── algorithm/             # PPO 算法实现
│   │   ├── utils/                 # 工具函数
│   │   └── workflow/
│   │       └── train_workflow.py  # 训练主流程（含 per-session best model）
│   ├── agent_diy/                 # 自定义 Agent（框架要求）
│   ├── conf/
│   │   ├── configure_app.toml     # 框架配置（dump_model_freq, batch_size 等）
│   │   ├── algo_conf_robot_vacuum.toml
│   │   └── app_conf_robot_vacuum.toml
│   ├── ckpt/                      # 官方模型保存目录（容器运行时）
│   ├── resume_snapshots/          # 断点续训快照
│   ├── session_best/              # Per-session 最优模型
│   │   ├── manifest.json          # 所有 session 摘要汇总
│   │   └── <session_id>/          # 每个训练 session 独立目录
│   │       ├── best_model.pkl     # 该 session 最优模型
│   │       └── best_score.json    # 该 session 最高分记录
│   ├── manual_checkpoints/        # 手动保存的检查点
│   ├── model.ckpt-resume.pkl      # 当前断点续训文件
│   ├── model.ckpt-resume.meta.json
│   ├── best_model.pkl             # 全局 best（兼容旧逻辑）
│   ├── latest_model.pkl           # 最新模型
│   └── kaiwu.json                 # 框架版本信息
├── train/                         # 训练配置与工具
│   ├── .docker-compose.yaml       # Docker Compose 编排文件
│   ├── .env                       # 环境变量配置
│   ├── backup_model/              # sidecar 签名后的提交 zip
│   ├── log/                       # 训练日志（aisrv/learner/kaiwu_env）
│   ├── tb_logs/                   # TensorBoard 事件文件
│   ├── archive/                   # 历史训练归档
│   ├── package_model.py           # 手动打包模型为提交 zip
│   ├── resume_best.py             # 检查点管理工具
│   ├── tb_writer.py               # 日志 → TensorBoard 事件转换
│   ├── local_monitor_dashboard.py # 本地训练监控面板（端口 18080）
│   ├── benchmark_report.py        # 训练稳健性报告
│   ├── collect_data.py            # 训练数据采集
│   └── auto_monitor.sh            # 自动监控迭代脚本
├── dev/                           # 开发辅助资源
│   └── images/                    # Docker 镜像文件
├── tencentarena-docs/             # 开悟框架文档
├── license.dat                    # 训练授权文件
└── code-robot_vacuum-public-13.0.1-comp-normal-lite.26comp.zip  # 官方初始代码包
```

## 手动开启容器训练

```bash
cd train

# 1. 首次加载镜像（仅第一次需要）
docker load -i ../dev/images/kaiwu-images-13.0.1.tar.zst

# 2. 启动分布式训练（learner + aisrv + gamecore + backup_model）
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d

# 3. 查看训练日志
docker logs -f kaiwu-train-learner-1
docker logs -f kaiwu-train-aisrv-1

# 4. 停止训练
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed down

# 5. 重启训练（需先删除旧容器以清除 process_stop.done）
docker rm kaiwu-train-learner-1 kaiwu-train-aisrv-1
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d
```

## 打包模型提交

```bash
# 一键打包 + 签名（推荐）
bash train/package_and_sign.sh <pkl路径> <step>

# 示例：从 session_best 打包
bash train/package_and_sign.sh code/session_best/20260411-154102/best_model.pkl 9339

# 示例：从断点续训文件打包
bash train/package_and_sign.sh code/model.ckpt-resume.pkl 9339

# 签名完成后文件位于 train/backup_model/<zip文件>.zip，可直接提交
```

如需手动分步操作：

```bash
# 1. 打包
python train/package_model.py --pkl <pkl路径> --step <step>

# 2. 投递到 sidecar 签名
docker cp train/_package_tmp/<zip文件>.zip kaiwu-train-backup_model-1:/workspace/train/backup_model/
docker cp train/_package_tmp/<zip文件>.zip.json kaiwu-train-backup_model-1:/workspace/train/backup_model/

# 3. 等待约 5 秒，签名后文件自动出现在 train/backup_model/
```

## 手动开启监控面板

```bash
# 1. 本地训练监控面板（端口 18080）
#    显示 episode 分数、胜率、训练进度等图表
python train/local_monitor_dashboard.py --port 18080
# 浏览器访问: http://127.0.0.1:18080

# 2. TensorBoard 可视化（端口 18081）
#    先将训练日志转换为 TensorBoard 事件
python train/tb_writer.py
# 启动 TensorBoard
tensorboard --logdir train/tb_logs --port 18081 --bind_all
# 浏览器访问: http://127.0.0.1:18081

# 3. 官方监控面板（端口 11000，需容器运行中）
# 浏览器访问: http://127.0.0.1:11000/p/v5/exp/monitor?domain_id=1&exp_id=1&task_uuid=1&task_id=0&platform=competition_stage
```

## 检查点管理工具

```bash
# 列出所有检查点
python train/resume_best.py list

# 查看最优检查点信息
python train/resume_best.py best

# 查看当前断点续训信息
python train/resume_best.py latest
```

## 环境要求

- Docker + Docker Compose（含 NVIDIA GPU 支持）
- Python 3.10+（用于打包脚本和监控面板）
- 依赖：`tensorboard`（用于 tb_writer.py）
## Benchmark 标配

当前 holdout benchmark 标配为 `2 x 8` 双重并发：

- `AISRV = 2`
- `env/gamecore per AISRV = 4`
- `GAMECORE = 8`
- `scheduler = dynamic`
- `maps = 4,7`
- `episodes_per_map = 8`

标准命令：

```bash
python train/run_holdout_benchmark.py \
  --checkpoint code/model.ckpt-resume.pkl \
  --output train/context/HOLDOUT_BENCHMARK_DYNAMIC_2X8.json
```

该命令默认会设置并验证：

- `KAIWU_AISRV_NUM=2`
- `KAIWU_PARALLEL_ENV_PER_AISRV=4`
- `KAIWU_GAMECORE_NUM=8`
- `KAIWU_BENCHMARK_PARALLEL_MODE=1`
- `KAIWU_BENCHMARK_SCHEDULER=dynamic`
- `aisrv_connect_to_kaiwu_env_count = 4`

真实验证记录：`train/context/HOLDOUT_BENCHMARK_DYNAMIC_2X8.json`，16/16 task completed。注意每个 workflow 进程内 `visible_env_handles=1` 是正常现象；并发来自 Kaiwu 框架启动的多个 helper process，而不是单个 Python workflow 里拿到多个 env 句柄。
