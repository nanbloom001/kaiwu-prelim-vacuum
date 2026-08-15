# 腾讯开悟「清扫大作战」强化学习智能体

基于腾讯开悟平台（KaiwuDRL）的机器人清扫对战（Robot Vacuum）强化学习比赛项目。智能体通过 PPO 算法学习清扫策略：在未知地图上清扫待清扫格、自主规划充电时机，与官方机器人同场竞技。当前已完成最终提交模型包（robot_vacuum-ppo-577）并沉淀了完整的自托管训练、监控与评估工具链。

**平台：** 腾讯开悟平台 · Robot Vacuum 清扫环境 · Linux 容器化训练栈（learner / aisrv / gamecore）· GreptimeDB 监控

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/nanbloom001/kaiwuFinal/ci.yml?branch=master&label=CI)](https://github.com/nanbloom001/kaiwuFinal/actions)

## 项目展示

游戏地图与自研监控面板的实机演示（65 秒）：

<video controls width="100%" src="https://raw.githubusercontent.com/nanbloom001/kaiwuFinal/master/assets/demo.mp4"></video>

<!-- TODO：补充更多 GIF / 真机图片 -->

## 已完成功能

### 智能体核心
- [x] PPO 训练算法：`value + policy − β·entropy` 损失组合（γ 0.99 / λ 0.95 / clip 0.2 / 梯度裁剪）
- [x] 混合 CNN + MLP 策略网络：21×21 局部视野 + 全局记忆 + 48 维标量特征，正交初始化
- [x] 观测与动作定义（GAE）：8 动作 / 3 模式 / 合法动作掩码，特征预处理流水线
- [x] 断点续训与快照：episode / 时间双轨快照，自动清理保留策略

### 奖励塑形与训练演进
- [x] 充电引导奖励体系：强化充电奖励 → 修复刷分 exploit → 阈值与惩罚再平衡
- [x] 电量约束阶梯调优（battery_max 400 → 300 → 150），迫使智能体学会主动充电
- [x] Curriculum Learning 训练方案（终局奖励再平衡 + 课程推进）

### 训练工程与运维
- [x] 容器化训练栈编排：learner / aisrv / gamecore / 监控一键启停
- [x] 自研本地监控面板：GreptimeDB 指标直读，替代官方监控页面
- [x] 训练工具链：checkpoint 管理、数据采集、实验归档与鲁棒性评估

### 交付
- [x] 最终提交模型包 `code/robot_vacuum-ppo-577.zip`（平台签名，唯一入库模型产物）

### 下一步

<!-- TODO：后续人工补充 -->

## 系统架构

<!-- TODO：补充系统架构图 -->

## 关键问题

### 1. 充电奖励失衡导致的刷分行为

强化充电奖励后智能体出现刷分 exploit，反复充电而不清扫。通过奖励阈值、惩罚项与电量约束的联合调整逐步修正，最终收敛出主动规划充电的稳定策略。

**解决方式**
- 充电引导阈值 30% → 50%，奖励归一化至 1.0
- 增加低电量惩罚（−8），修复充电 exploit
- battery_max 阶梯下调 400 → 300 → 150，压缩错误策略空间

### 2. 官方监控面板不可用

官方本地监控页面不稳定，训练健康度无法及时确认，影响长时间训练的排障效率。

**解决方式**
- 自研轻量监控面板，直读 GreptimeDB 的 Prometheus 指标
- 核心指标统一可视化：全局步数、局数、清扫分、剩余电量、充电次数、样本生产/消费比
- 监控服务纳入训练栈容器编排，随训练一键启动

<details>
<summary>更多工程问题</summary>

- **模型大文件不入 Git**：checkpoint 体积大且高频变化，跨机交接走 `scp` / `rsync` / GitHub Releases，避免仓库膨胀
- **每日分支摘要自动化空转**：外部 LLM API 不稳定，产生大量无实际内容的提交，已整体移除
- **环境配置校验**：机器人数量配置为 0 会导致环境验证失败，已强制最小值为 1

</details>

## 使用说明

### 环境要求

运行本仓库需要**腾讯开悟平台环境**：平台提供的 `kaiwudrl` SDK、容器镜像与 `license.dat` 授权文件，本仓库不包含平台代码。官方文档见 <https://tencentarena.com>。

### 快速开始

```bash
# 修改 code/train_test.py 中的 algorithm_name（"ppo" 或 "diy"）后运行
python code/train_test.py
```

### 训练（Linux 服务器）

```bash
cd train
# 准备 .env（KAIWU_* 变量清单见 .docker-compose.yaml，SOP 见 train/context/）
docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d
```

### 监控

```bash
python train/local_monitor_dashboard.py   # 自托管 HTML 面板，读取 GreptimeDB 指标
```

### 配置

| 文件 | 作用 |
| --- | --- |
| `code/conf/configure_app.toml` | 训练框架主配置（样本池 / 批大小 / 模型同步） |
| `code/conf/algo_conf_robot_vacuum.toml` | 算法注册（ppo / diy） |
| `code/conf/app_conf_robot_vacuum.toml` | 应用注册（rl_helper 等） |
| `code/agent_ppo/conf/train_env_conf.toml` | 训练环境（地图 / 机器人 / 电量 / 步数） |
| `code/agent_ppo/conf/conf.py` | PPO 超参数与特征布局 |

### 仓库结构

- `code/` — 开悟平台上传单元：`agent_ppo/`（主力实现）、`agent_diy/`（平台模板骨架）、`conf/`、`kaiwu.json`、`train_test.py`
- `train/` — 训练运维：编排、监控、采集、恢复、评估脚本与交接文档（详见 `train/README.md`）
- `assets/` — README 演示媒体
- 模型策略：仓库仅保留最终提交包 `code/robot_vacuum-ppo-577.zip`，其余 checkpoint 不入库

### 常见问题

- **为什么单独 clone 无法训练？** 依赖平台 SDK 与授权，本仓库只含智能体与工具代码。
- **监控面板没有数据？** 确认 `greptimedb` 已随训练栈启动且 4000 端口可达（见 `train/context/`）。

## 许可证

MIT License，详见 [LICENSE](LICENSE)。基于腾讯开悟平台模板开发，来源与依赖声明见 [NOTICE.md](NOTICE.md)。
