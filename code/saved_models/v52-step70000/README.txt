Resume Checkpoint: v52-step70000
==================================

保存时间: 2026-04-15
来源: model.ckpt-70000 (Phase 2 比例式 Entropy 保护 + 动态课程训练)
训练轮次: v52 Phase 2 (从 v52-step10000 resume)

选定指标 (对应 episode ~150-179 区间, learner step ~70000)
-----------
- Learner Global Step: 70000
- 累计 Episode: ~150-179 (aisrv-1)
- Entropy: 0.14 (稳定, 未塌缩)
- Value Loss: 285 (从 438 持续下降)
- Policy Loss: -19 (从 -26 持续收敛)

Anchor 配置表现 (ep 150-179)
-----------
- WinRate: 100% (26/26 全部完成)
- CPS (WIN avg): 0.879
- Avg CS (WIN): 879
- Charge Count: 4.3

Mild 配置表现 (ep 150-179)
-----------
- WinRate: 77.8%
- CPS (WIN avg): 0.875
- Avg CS (WIN): 924

选择理由
--------
1. Anchor WinRate 100%, CPS 0.879 — 基础配置完全掌握
2. Entropy 0.14 稳定不塌缩, 比例式保护生效
3. 位于 ep 150-179 性能巅峰区间, 之后 mild WinRate 开始下滑
4. Value Loss 285, Critic 仍在持续学习但已大幅收敛

Phase 2 改动摘要
---------------
- 比例式 Entropy Floor: ENTROPY_FLOOR=0.15, COEF=0.05
  entropy bonus = COEF * value_loss * floor_gap, 与 loss 量级解耦
- 动态课程跃迁: 基于 win_rate/avg_cs/avg_cc 指标驱动
  (本轮训练未触发提前跃迁, WinRate 始终 <90%)
- Entropy 从 v51 的 0.06 塌缩改善为 0.13-0.17 稳定震荡

训练整体时间线
--------------
阶段            Steps       Entropy    CPS(W)  WinRate  诊断
warmup          10k-30k     0.15→0.13  0.870   84%      稳定, anchor 为主
blend 早期      30k-50k     0.14→0.16  0.873   83%      加入 mild/broad
blend 巅峰      50k-70k     0.14→0.17  0.876   83%     ★ 最佳区间
blend 后期      70k-95k     0.13→0.17  0.854   81%      CPS 微降, mild 退化

已解决的问题
-----------
- Entropy 塌缩: v51 在 step 28k 后 entropy 跌至 0.06
  → v52 Phase 2 比例式保护, step 95k 仍保持 0.13-0.17

未解决的问题
-----------
- Mild 配置 WinRate 下降: ep 90-119 的 71.4% 到 ep 180-209 的 70%
- Broad 配置仍有大量 FAIL (battery 死亡)
- WinRate 停滞在 81-82%, 无法达到 90% 触发课程跃迁
- CPS 收敛于 0.85-0.88, 无上升趋势

架构
----
- 4ch local view (obstacle/cleaned/dirt/trajectory_heatmap)
- Feature dim: 2038D
- A* potential reward (ALPHA=0.25)
- Expert bias [5,15], gradient isolation 1/(1+w/10)
- Batch tensor: 2062D

注意
----
此 checkpoint 是 4ch 模型权重, 不需要做 3ch→4ch 迁移。
文件权限为 user:rw, 框架可直接 torch.load 读取。
