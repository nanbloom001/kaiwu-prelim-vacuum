Resume Checkpoint: v52-step10000
==================================

保存时间: 2026-04-15
来源: ppo-10000 (Phase 1 训练第 ~250 个 episode 对应的 learner step 10000)
训练轮次: v52 Phase 1 (A* 势函数奖励 + 轨迹热图 + Expert Bias 增强 + 梯度隔离)

选定指标 (对应 training_metrics 采样点 step ~9714)
-----------
- Learner Global Step: ~10000
- 累计 Episode: ~236
- Clean Score: 915
- Charge Count: 5.2
- Entropy: 0.15
- Reward: 2041
- Value Loss: 442

选择理由
--------
1. Entropy 0.15 是整个训练过程中最后一个"策略仍有充分探索能力"的点
2. 此阶段 Win Rate 接近 100%，无电池死亡、无碰撞死亡
3. ChargeCount 5.2 说明 A* 势函数奖励已生效，模型开始自主回充
4. Step 11332 之后 entropy 快速跌至 0.09-0.06，进入和 v51 相同的塌缩模式

训练退化时间线
--------------
阶段            Steps       Entropy    CS      CC    诊断
健康探索        0-10000     0.33→0.15  836→915 5.2   A* potential 生效, 100% WIN
开始退化        10000-18000 0.13→0.09  915→950 6.5   entropy 下降, broad 偶发高分
波动期          18000-23000 0.09→0.11  1150→950 4.5   ChargeCount 骤降, 回充退化
严重退化        23000-28000 0.09→0.07  950→785 3.4   策略确定性, 清扫/充电双退化
塌缩            28000-38000 0.06→0.07  890→785 3.8   与 v51 相同问题

退化根因
--------
ENTROPY_FLOOR_COEF=3.0 仍不够强:
  - entropy=0.06 时, floor_gap=0.44, extra_beta=1.32, effective_beta=1.33
  - entropy 项 = 1.33 * 0.06 = 0.08, 相对 policy_loss=-24 和 value_loss=400 可忽略
  - 需要将 ENTROPY_FLOOR_COEF 提高到 8-10 才能产生足够恢复力

架构变更 (相对 v51)
-------------------
- Local view: 3ch → 4ch (新增轨迹热图, TRAJECTORY_LENGTH=50, DECAY=0.02)
- Feature dim: 1597D → 2038D
- Batch tensor: 1620 → 2062 (新增 expert_weight 1D)
- A* potential reward: ALPHA=0.25, 阈值 65% 电量
- Expert bias: [3,8] → [5,15]
- Gradient isolation: 1/(1+w/10) 对 expert 接管样本降权
- Checkpoint migration: 自动 3ch→4ch conv 权重填充

注意
----
此 checkpoint 是 4ch 模型权重, 不需要再做 3ch→4ch 迁移。
文件权限为 user:rw, 框架可直接 torch.load 读取。
