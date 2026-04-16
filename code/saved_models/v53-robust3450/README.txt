Resume Checkpoint: v53-robust3450
===================================

保存时间: 2026-04-15
来源: session_best/20260415-214339 (本轮训练最佳 episode 36, score=1103)
训练轮次: v52+ 代码改进后的 robust 阶段训练

选定指标 (session_best)
-----------
- Best Robust Score: 3450.2 (历史最佳)
- Best Avg Score: 909.2
- Episode Count: 40 (session_best 窗口)
- 最好单局: ep 36, CS=1103

训练上下文 (aisrv-1)
-----------
- 训练阶段: robust (全程未触发课程跃迁)
- Episode 110: WR=0.85, avg_cs=848, avg_cc=6.3
- Episode 150: WR=0.75, avg_cs=800, avg_cc=4.8 (开始退化)
- 最佳在 ep 40 附近，之后缓慢退化

课程配置 (robust 阶段)
-----------
- anchor: 1机器人/4充电站/1000步/300电池 或类似
- mild: 1-2机器人/3-4充电站/750-1100步/220-420电池
- broad: 2-4机器人/1-4充电站/700-2000步/120-260电池

选择理由
--------
1. 历史最高 robust_score=3450.2，超越所有之前版本
2. session_best 中有多个 episode CS > 1000 (最高 1144)
3. 位于训练早期巅峰，尚未退化

对比其他版本
-----------
- v52-step70000: robust~3100, Anchor WR=100%, CPS=0.879
- v53 (本版): robust=3450, 整体更强但 anchor/mild/broad 细分数据需 benchmark 确认
- v5-step4300: robust~2507, 3ch 架构，已过时

使用方法
--------
在 conf.py 中设置:
  RESUME_CHECKPOINT = "saved_models/v53-robust3450/model.ckpt-resume.pkl"

或用于 benchmark 评估:
  bash run_benchmark.sh saved_models/v53-robust3450/model.ckpt-resume.pkl
