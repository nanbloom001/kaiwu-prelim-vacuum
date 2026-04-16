# Expert 充电逻辑审查请求

## 任务

请审查一个 PPO 机器人清扫 AI 的 Expert 充电导航模块代码，并给出详细的改进方案。

## 背景

这个 AI 在训练中总体健康（entropy 稳定、无策略退化、WinRate 85%），但所有失败都集中在两种模式。我们通过死亡轨迹日志定位到失败全部来自 Expert 充电逻辑，而非 reward 设计或 PPO 训练问题。

## 你需要做的

1. 阅读 Expert 代码 `expert.py`（下面附完整代码）
2. 阅读训练失败分析报告 `LOG_20260414_v53_training_analysis.md`
3. 独立分析失败原因，给出你的改进方案，包括：
   - 每个问题的根因判断（可以同意或不同意报告中的分析）
   - 具体修改方案（改哪个函数、怎么改、为什么这样改）
   - 修改的风险评估
   - 优先级排序

## 关键约束

- Expert 在训练模式下通过 `get_logit_bias()` 提供软引导（bias 3-100），评估模式下通过 `get_override()` 硬覆盖
- 充电导航用 A* 寻路 + 3 级 fallback + NPC danger cost map
- 充电触发用 state machine（return_mode 持续到电量 ≥ 95% 且在充电桩上）
- `filter_actions()` 在 agent.py 的 predict() 中最先执行，是 NPC 安全防线
- 修改不应改变 agent.py 的调用接口

## 失败数据摘要

**碰撞死亡（18 例）**：全部在 mode=2（充电导航模式），NPC 距离 2-3 时碰撞。包括电量 93%（667/720）时也进入充电模式后撞 NPC 的案例。

**电池死亡（14 例）**：全部在 mode=1（清扫模式）到电量归零，最后 20 步零充电。包括 charger_count=1、robot_count=4 的极端配置（slack 低至 -73）。

请输出你的完整改进方案。
