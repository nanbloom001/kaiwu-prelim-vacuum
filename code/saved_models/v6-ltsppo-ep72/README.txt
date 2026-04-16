Resume Checkpoint: v6-ltsppo-ep72
=================================

保存时间: 2026-04-16
来源: LTSPPO session_best 快照
源文件: code/session_best/20260416-183426/best-ep000072-score00832.pkl
当前保存文件: code/saved_models/v6-ltsppo-ep72/model.ckpt-resume.pkl
训练主线: linux-LTSPPO (LTSPPO / GRU + Dual Critic)

选定依据
--------
这是当前 LTSPPO 训练中，经手动 benchmark 复核后最适合作为 resume 候选基线的 checkpoint。

统一评估命令:
  bash train/run_benchmark_parallel.sh <checkpoint> --workers 4 --envs-per-worker 10 --max-wait 1800

对应评估结果:
  结果目录: train/eval_parallel_logs/20260416-191815/result.json
  Win Rate: 30.0% (12/40)
  Avg Clean Score: 460.0

分轮结果:
  round_1: WR 40.0%, CS 504.7, BatteryFail 60.0%, CollisionFail 0.0%
  round_2: WR 60.0%, CS 629.5, BatteryFail 40.0%, CollisionFail 0.0%
  round_3: WR 20.0%, CS 476.1, BatteryFail 80.0%, CollisionFail 0.0%
  round_4: WR 0.0%,  CS 229.8, BatteryFail 90.0%, CollisionFail 10.0%

为什么保存这个点
----------------
1. 在本轮手动评估的 LTSPPO 候选点中，它的综合表现最好。
2. 相比同时期的常规 learner checkpoint (`model.ckpt-10000/19000/24000.pkl`)，它明显更强。
3. 相比同目录下的 `best_model.pkl`，它的固定 benchmark Win Rate 更高，泛化更均衡。
4. 它位于在线分数上升阶段的峰值附近，适合作为后续调参和继续训练的起点。

本轮对比结果摘要
----------------
1. best-ep000072-score00832.pkl
   WR 30.0%, CS 460.0
2. session_best/20260416-183426/best_model.pkl
   WR 25.0%, CS 479.4
3. session_best/20260416-183427/best_model.pkl
   WR 17.5%, CS 453.4
4. learner model.ckpt-19000.pkl
   WR 17.5%, CS 193.8
5. learner model.ckpt-10000.pkl
   WR 17.5%, CS 176.1
6. learner model.ckpt-24000.pkl
   WR 12.5%, CS 168.9

当前局限
--------
1. 这个点仍然不算强基线，只能说是“当前 LTSPPO 中最值得保留的点”。
2. 主要失败来源仍然是 battery fail，不是 collision。
3. harder / long-horizon 场景 (`round_3 / round_4`) 仍明显偏弱。
4. 因此它更适合作为“继续训练与调参的 resume 基线”，而不是最终提交模型。

后续使用建议
------------
1. 如果继续当前 LTSPPO 训练，优先从这个 checkpoint resume。
2. 继续重点观察:
   - battery_fail_rate
   - late_return_rate
   - target_switch_rate
   - anchor/mild/broad 分 profile 表现
3. 若后续出现更高 Win Rate 且更低 battery fail 的固定 benchmark 结果，应替换该基线。
