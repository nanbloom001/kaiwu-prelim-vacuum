# train/ — 训练运维目录

本目录存放训练编排、工具脚本与运维交接文档，供 Linux 训练服务器上的操作者/运维 AI 使用。

> 约定：**重产物不进 Git**（train/log/、train/archive/、train/backup_model/、train/checkpoint/
> 等已在根 .gitignore 中忽略）；小而高价值的上下文文档放入 context/。

## 编排与启动

- **.docker-compose.yaml** — 训练栈编排（learner / aisrv / gamecore / kaiwu_env /
  backup_model / pushgateway / greptimedb / vector / monitor）。
  使用前需准备 .env（含 KAIWU_* 环境变量），启动方式见仓库根 README.md。
  注意：该文件名被 code/agent_ppo/utils/archive_agent.py 引用，**请勿重命名**。

## 工具脚本

| 脚本 | 用途 | 示例 |
| --- | --- | --- |
| resume_best.py | checkpoint 管理：查看、准备、清理最佳/最新模型 | python resume_best.py list / prepare best / clean --keep 3 |
| collect_data.py | 从 aisrv 日志采集 GAMEOVER 与训练指标，输出 TRAINING_DATA.json | python collect_data.py |
| local_monitor_dashboard.py | 自托管轻量训练监控面板（读取 GreptimeDB 的 Prometheus 指标） | python local_monitor_dashboard.py |
| benchmark_report.py | 对归档训练 run 生成 checkpoint 鲁棒性评估报告 | python benchmark_report.py --run-dir train/archive/<run_id> |

> 所有脚本路径均以仓库根为基准解析（不要依赖绝对路径或硬编码盘符）。

## 交接文档（context/）

| 文件 | 内容 |
| --- | --- |
| README.md | context 目录的收纳规范 |
| SERVER_AI_PROMPT.md | 交给 Linux 服务器运维 AI 的完整提示词与工作规则 |
| SERVER_SYNC_AND_MONITOR.md | 服务器同步与监控 SOP |
| SESSION_LOG_20260409.md | 历史会话记录（供回溯决策） |

## 模型归档（archive 机制）

训练期间的日志/checkpoint/配置由 code/agent_ppo/utils/archive_agent.py（容器内后台进程）
定期同步到 train/archive/<run_id>/，并按空闲时间自动 finalize 归档；
archive_analysis.py / benchmark_report.py 负责归档后的评估分析。
