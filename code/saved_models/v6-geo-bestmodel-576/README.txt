模型归档：v6-geo-bestmodel-576

来源检查点
- 原始文件：/home/user/TcKaiwuFinal/code/session_best/20260417-110429/best_model.pkl
- 归档文件：model.ckpt-resume.pkl

选择该检查点的原因
- 该模型来自本轮几何感知长期规划训练的后期成熟阶段，不是单局偶然高分点，而是 session 级综合最优模型。
- 选择时对应的 session 摘要为：
  - best_robust_score：3494.05
  - best_avg_score：1228.2
  - updated_at：2026-04-17 16:40:18
  - episode_cnt：576

手工筛选依据
- 本次筛选没有只看单局 clean score，而是综合评估了以下日志数值：
  - eval_hard 阶段的 win_rate、avg_cs、avg_cc
  - training_metrics 中的 total_score、charge_count、remaining_charge
  - MAP_STATS 中的 variance、min_avg、spread
  - learner 侧的 entropy、teacher active rate、loss 稳定性
  - 失败轨迹中 mode=4 时的负 slack 幅度
- 该 checkpoint 所处窗口具备较强的持续性：
  - eval_hard 下连续出现多局 broad_eval 高分胜局
  - total_score 长时间维持在较高水平
  - charge_count 较高，说明补能链路已经较成熟
  - teacher 没有掉线，长期规划监督仍在参与训练

为什么它适合作为 resume 点
- 它比单局峰值点更稳，更适合作为下一轮继续训练的主基线。
- 它来自训练后期，代表的是当前策略已经学成的综合能力，而不是高方差高峰。
- 当前核心问题仍然存在，但该点在“成绩、课程阶段、地图统计、teacher 在线状态”四个维度上更平衡。

已知问题
- battery fail 仍然是当前主失败模式。
- 一部分失败轨迹仍表现为：
  - 已进入 return 模式
  - slack 已经明显为负
  - 最终耗死在返航阶段
- 这说明回充时机与返航动作效率还没有被完全解决。

建议用途
- 作为下一轮继续训练的主 resume 检查点。
- 如果要做固定 benchmark，对照点建议同时保留一个高峰型 best-ep 模型，用于比较“稳态继续训练”和“激进峰值起点”的差异。
