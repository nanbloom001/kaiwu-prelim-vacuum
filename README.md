# kaiwuFinal — 腾讯开悟「清扫大作战」强化学习项目

> 基于腾讯开悟平台（KaiwuDRL）的机器人清扫对战（Robot Vacuum）强化学习比赛项目。
> 当前以 **PPO 智能体**（`code/agent_ppo/`）为主力实现，仓库同时沉淀了训练运维编排、比赛官方文档存档与每日分支工作摘要自动化。

---

## 目录结构

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
│   ├── agent_diy/                 # DIY 智能体模板骨架（新算法从这份脚手架开始）
│   ├── conf/                      # 框架级配置
│   │   ├── configure_app.toml     # 训练框架主配置（样本池/批大小/模型同步等）
│   │   ├── algo_conf_robot_vacuum.toml  # 算法注册（ppo / diy）
│   │   └── app_conf_robot_vacuum.toml   # 应用注册（rl_helper 等）
│   ├── kaiwu.json                 # 开悟版本与项目代码（robot_vacuum）
│   ├── train_test.py              # 训练入口（修改 algorithm_name 后运行）
│   └── best_model.pkl 等          # 跨分支交接的核心模型文件（见下方约定）
├── train/                         # 训练运维侧（编排、工具脚本、交接文档）
│   ├── .docker-compose.yaml       # 训练栈编排（learner/aisrv/gamecore/监控等）
│   ├── collect_data.py            # 训练数据采集（GAMEOVER/训练指标 → TRAINING_DATA.json）
│   ├── resume_best.py             # checkpoint 管理（list/best/latest/prepare/clean）
│   ├── local_monitor_dashboard.py # 自托管轻量监控面板（GreptimeDB 指标）
│   ├── benchmark_report.py        # 归档 checkpoint 鲁棒性评估报告
│   └── context/                   # 高价值交接文档（服务器 AI 提示词、同步与监控说明等）
├── tencentarena-docs/             # 比赛官方文档存档（开发指南 + 框架文档 + 智能体文档）
├── branch_summaries/              # 每日分支工作摘要（由 GitHub Actions 自动生成）
└── .github/                       # 每日摘要自动化工作流与脚本
```

---

## 快速开始

### 1. 训练入口

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

## 模型文件约定（重要）

以下文件是跨分支交接的核心交付物，**随 Git 跟踪、请勿删除或改名**（`.gitignore`
已显式放行，运维规范见 `train/context/SERVER_AI_PROMPT.md`）：

- `code/best_model.pkl` — 当前最佳模型
- `code/latest_model.pkl` — 最新模型
- `code/model.ckpt-resume.pkl` + `code/model.ckpt-resume.meta.json` — 断点续训模型

---

## 配置一览

| 文件 | 作用 |
| --- | --- |
| `code/conf/configure_app.toml` | 训练框架主配置：样本池容量、采样策略、批大小、模型同步频率等 |
| `code/conf/algo_conf_robot_vacuum.toml` | 算法注册表：ppo / diy 的 agent 与 workflow 类路径 |
| `code/conf/app_conf_robot_vacuum.toml` | 应用注册：rl_helper、策略构建器 |
| `code/agent_ppo/conf/train_env_conf.toml` | 训练环境：地图、机器人/充电桩数量、电量、最大步数 |
| `code/agent_ppo/conf/conf.py` | PPO 超参数与特征布局（Config 类） |
| `code/kaiwu.json` | 开悟版本号与项目代码 |

---

## 每日分支摘要自动化

- 工作流：`.github/workflows/daily-summary.yml`（每日 00:00 UTC 定时 + 手动 `workflow_dispatch`）
- 脚本：`.github/scripts/daily_summary.py`（调用 LLM API 汇总各分支 24h 提交）
- 产物：`branch_summaries/<yyyy-mm-dd>/`（每分支一个 md + OVERALL_SUMMARY.md，无提交时为 no_updates.md）
- 密钥：需在仓库 Secrets 中配置 `OPENCLAW_API_KEY`

---

## 分支约定

- `master`：集成主线（每日摘要自动提交到 master）
- 开发分支：`cyy` `hjc` `hjc-ppo` `linux` `linux-LTSPPO` `linux-yjy` 等，
  命名大体遵循 `<平台>-<负责人>[-<主题>]` 的约定

---

## 相关文档

- `tencentarena-docs/README.md` — 比赛官方文档索引（开发指南 / 框架 / 智能体）
- `train/context/` — 训练运维交接文档（服务器 AI 提示词、同步与监控 SOP、会话记录）
- `train/README.md` — train/ 目录工具脚本使用说明
