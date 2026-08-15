# kaiwuFinal — 腾讯开悟「清扫大作战」强化学习项目

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/nanbloom001/kaiwuFinal/ci.yml?branch=master&label=CI)](https://github.com/nanbloom001/kaiwuFinal/actions)

Reinforcement learning agent for the **Robot Vacuum** ("清扫大作战") competition
on the Tencent Kaiwu platform (KaiwuDRL). The **PPO agent** (`code/agent_ppo/`)
is the primary implementation; the repository also contains the training
orchestration stack used for local Linux training.

> **平台依赖声明**：本项目运行于腾讯开悟平台（KaiwuDRL）。训练需要平台提供的
> `kaiwudrl` SDK、容器镜像与 `license.dat` 授权文件，**本仓库不包含平台代码**，
> 单独 clone 本仓库无法直接训练。官方文档见 <https://tencentarena.com>。
>
> **Platform dependency**: this project runs on the Tencent Kaiwu platform.
> Training requires the platform SDK (`kaiwudrl`), container images and a
> `license.dat` license — none of which are distributed in this repository.

---

## 目录结构 / Layout

```
kaiwuFinal/
├── code/                          # 开悟训练代码包（容器挂载到 /workspace/code）
│   ├── agent_ppo/                 # PPO 智能体（主力实现）
│   │   ├── agent.py               # Agent 主类（决策/训练/存档入口）
│   │   ├── algorithm/             # PPO 算法（value loss + policy loss - β·entropy）
│   │   ├── model/                 # 混合 CNN + MLP 策略网络
│   │   ├── feature/               # 观测/动作数据定义（GAE）、特征预处理
│   │   ├── conf/                  # 超参数（Config）、监控面板、环境配置
│   │   ├── workflow/              # 训练工作流
│   │   └── utils/                 # 实验归档、checkpoint 分析、归档代理
│   ├── agent_diy/                 # DIY 智能体模板骨架（平台要求的起始脚手架）
│   ├── conf/                      # 框架级配置（configure_app / algo / app）
│   ├── kaiwu.json                 # 开悟版本与项目代码（robot_vacuum）
│   ├── train_test.py              # 训练入口（修改 algorithm_name 后运行）
│   └── robot_vacuum-ppo-577.zip   # 最终提交模型包（唯一入库的模型产物，见模型策略）
├── train/                         # 训练运维侧（编排、工具脚本、交接文档）
│   ├── .docker-compose.yaml       # 训练栈编排（learner/aisrv/gamecore/监控等）
│   ├── collect_data.py            # 训练数据采集（GAMEOVER/训练指标 → TRAINING_DATA.json）
│   ├── resume_best.py             # checkpoint 管理（list/best/latest/prepare/clean）
│   ├── local_monitor_dashboard.py # 自托管轻量监控面板（GreptimeDB 指标）
│   ├── benchmark_report.py        # 归档 checkpoint 鲁棒性评估报告
│   └── context/                   # 运维交接文档（服务器 AI 提示词、同步与监控说明）
├── .github/workflows/ci.yml       # 开源 CI（语法检查 + 文件守卫）
├── LICENSE / NOTICE.md            # 许可证与第三方声明
└── CONTRIBUTING.md / CODE_OF_CONDUCT.md / SECURITY.md
```

---

## 快速开始 / Quick Start

### 1. 训练入口（需开悟平台环境）

```bash
# 修改 code/train_test.py 中的 algorithm_name（"ppo" 或 "diy"）后运行：
python code/train_test.py
```

### 2. 容器化训练（Linux 服务器）

```bash
cd train
# 准备 .env（提供 KAIWU_* 环境变量，变量清单见 .docker-compose.yaml；
# 服务器操作规范参考 train/context/SERVER_AI_PROMPT.md）
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d
# 停止：
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed down
```

> `train/.docker-compose.yaml` 编排了 learner / aisrv / gamecore / kaiwu_env /
> backup_model / pushgateway / greptimedb / vector / monitor 全套服务；
> `code/agent_ppo/utils/archive_agent.py` 会读取该文件归档训练产物，**请勿重命名**。

### 3. 本地监测

```bash
python train/local_monitor_dashboard.py   # 从 GreptimeDB 读取指标，自托管 HTML 面板
```

---

## 模型文件策略 / Model Policy（重要）

**唯一入库的模型产物**：`code/robot_vacuum-ppo-577.zip` — 最终提交平台的模型包
（含平台签名 `.kaiwu.sign` 与最终 checkpoint `ckpt/model.ckpt-5777.pkl`）。

除此之外的模型文件（`.pkl` / `.meta.json`）是体积大、高频变化的训练产物，
**不入库**：`code/best_model.pkl`、`code/latest_model.pkl`、
`code/model.ckpt-resume.pkl` + `.meta.json` 等请通过 `scp` / `rsync`、共享存储
或 GitHub Releases 分发（详见 `train/context/SERVER_SYNC_AND_MONITOR.md`）。
`.gitignore` 已全局忽略这些文件。

---

## 配置一览 / Configuration

| 文件 | 作用 |
| --- | --- |
| `code/conf/configure_app.toml` | 训练框架主配置：样本池容量、采样策略、批大小、模型同步频率等 |
| `code/conf/algo_conf_robot_vacuum.toml` | 算法注册表：ppo / diy 的 agent 与 workflow 类路径 |
| `code/conf/app_conf_robot_vacuum.toml` | 应用注册：rl_helper、策略构建器 |
| `code/agent_ppo/conf/train_env_conf.toml` | 训练环境：地图、机器人/充电桩数量、电量、最大步数 |
| `code/agent_ppo/conf/conf.py` | PPO 超参数与特征布局（Config 类） |
| `code/kaiwu.json` | 开悟版本号与项目代码 |

---

## 分支与贡献 / Branches & Contributing

- `master` 为默认分支，始终保持可发布状态；开发请走 `feat/*` / `fix/*` 分支并提交
  Pull Request。
- 贡献指引见 [CONTRIBUTING.md](CONTRIBUTING.md)，行为准则见
  [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)，安全问题上报见 [SECURITY.md](SECURITY.md)。

---

## 文档 / Documentation

- `train/README.md` — train/ 目录工具脚本使用说明
- `train/context/` — 训练运维交接文档（服务器操作提示词、同步与监控 SOP）
- 比赛官方文档（开发指南 / 框架 / 智能体）：<https://tencentarena.com>

---

## 许可证 / License

MIT License — 见 [LICENSE](LICENSE)。本仓库基于腾讯开悟平台模板开发，
第三方来源与依赖声明见 [NOTICE.md](NOTICE.md)。
