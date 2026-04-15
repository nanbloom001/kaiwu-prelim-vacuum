Resume Checkpoint: resume-v5-step4300
======================================

保存时间: 2026-04-13
来源: resume-episode-ep000200.pkl (本轮训练第 200 个快照点)
训练轮次: v4 综合优化（奖励重平衡 + Expert Logit Bias + 参数回调）

训练概要
--------
- 总训练: step 0→7606, 2736 episodes, 约 1.5 小时
- 起点: 随机权重（无 resume）
- 框架: PPO + Expert Logit Bias（训练时软引导，评估时硬覆盖）

选定 Checkpoint 指标
--------------------
- 训练步数: ~4300 (learner global step)
- 累计 Episode: ~1465
- CPS (clean per step): 0.862 ← 达标（目标 ≥ 0.85）
- Completed Rate: 100% ← 达标（目标 ≥ 95%）
- Clean Score: 1293
- Entropy: 0.84
- Charge Count: 3.0
- Remaining Charge: 114

选择理由
--------
1. CPS 和 CompRate 同时达标的最优点（step 4261 在完整日志中 CPS=0.862, comp=100%）
2. Entropy 0.84 仍在可恢复范围（尚未塌缩），下一轮配合更高 BETA 可重新激发探索
3. 模型已学会高效清扫路径，无需从零开始
4. 比早期 checkpoint（ep100, CPS=0.73）效率高 18%

完整训练趋势（4 个阶段）
------------------------
阶段         Steps      CPS    CompRate  ChargeCnt  Entropy   诊断
健康探索     28-2261    0.59   94.9%     8.8        1.5-2.0   充电充足,存活高
开始衰退     2338-3886  0.76   87.4%     4.4        1.0-1.5   CPS升但少充电
策略趋确定   3964-4882  0.85   87.0%     2.9        0.5-1.0   CPS高但趋确定
完全塌缩     5047-7606  0.85   81.7%     3.2        0.0-0.5   18%电死

主要问题（下一轮需修复）
------------------------
1. Entropy 塌缩: 2.0 → 0.08，策略变为确定性，丧失探索能力
   修复: BETA 0.008→0.015, 加 entropy floor 0.3-0.5

2. Expert Logit Bias 太弱: soft bias 3-8 被高置信度 RL logits 覆盖
   修复: bias 范围从 [3,8] 提升到 [5,15]

3. 充电行为退化: charge_count 从 14 降至 1-3
   根因: 模型发现"不充电→多清扫→更高 reward"，soft bias 无法纠正
   修复: 同上 bias 增强 + 效率充电奖励已到位

本轮最优评估成绩
----------------
- session_best robust_score: 2506.8
- session_best avg_score: 845.1
- 历史最佳（v3, 20260412-170437）: robust=3051.6, avg=917.0

对比历史最佳
------------
此 checkpoint 的 CPS(0.86) 已超过历史最佳版本的表现，
但 comp rate 和 score 尚未超越（历史最佳 robust=3051）。
下一轮修复 entropy 和 bias 后有望全面超越。

使用方法
--------
将 model.pkl 复制为 model.ckpt-resume.pkl，并在 conf.py 中设置:
  RESUME_CHECKPOINT = "model.ckpt-resume.pkl"
