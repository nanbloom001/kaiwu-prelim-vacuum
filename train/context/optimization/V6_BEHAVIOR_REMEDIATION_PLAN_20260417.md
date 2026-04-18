# V6 行为缺陷修复方案（基于 2026-04-17 多轮标准 benchmark）

## 摘要

本方案基于以下标准 benchmark 日志与对照分析：

- `train/eval_parallel_logs/20260417-200128/`
- `train/eval_parallel_logs/20260417-212241/`
- `train/eval_parallel_logs/20260417-192632/`
- `train/eval_parallel_logs/20260417-192809/`

当前模型的主问题已经比较明确，不再适合继续只做泛泛的“更早回充”调参。真正的核心矛盾是 4 件事叠加：

1. 墙边/旧边界低价值跟随过强
2. 窄路与未知路径承诺过强
3. 充电提前量不足，存在“路过不充/卡点回充”
4. `route_anchor / target` 粘滞，且策略经常偏离 planner

同时明确排除两条容易误判的方向：

- **充电桩竞争**：当前 benchmark 中不是主要矛盾
- **显式切比雪夫 fallback**：当前 benchmark 中不是主因

最严重的异常样本已在 `20260417-212241` 中抓到：

- `round_4_map2`
- 存在约 `162` 步的异常段
- 其中在 `(47,123) <-> (48,123)` 间来回往返
- 同时满足：
  - `wall_hugging_clean_floor`
  - `stale_boundary_follow`
  - `revisit_on_clean_floor`
  - `redundant_clean_path`
  - `loop_suspect`
  - `corner_loop_suspect`
  - `planner_policy_divergence`

因此本方案的目标不是“继续提分”，而是：

- 先把明显病态行为压下去
- 再提升长局和少充电桩场景稳定性
- 最终让高分局不再建立在“带病运行”的局部最优上

---

## 一、问题诊断与优先级

## P0：必须优先修的结构性问题

### P0-1. 墙边低价值跟随 / 旧边界打磨

日志证据：

- `wall_hugging_clean_floor_rate`
- `stale_boundary_follow_rate`
- `revisit_on_clean_floor_rate`
- `redundant_clean_path_rate`

典型强样本：

- `round_4_map2`
  - `wall_hugging_clean_floor_rate = 0.315`
  - `stale_boundary_follow_rate = 0.298`
  - `revisit_on_clean_floor_rate = 0.381`
  - `redundant_clean_path_rate = 0.2765`
  - `avg_wall_follow_streak = 23.1755`
  - `avg_path_cross_count_50 = 12.817`

问题本质：

- 模型仍将“墙边”当作强导向结构
- 但已清扫、低 frontier 的墙边没有被强烈降权
- 在局部路径价值耗尽后仍会反复打磨旧边界

### P0-2. Planner 与策略长期不一致

日志证据：

- 多个高分/完成局中 `planner_policy_divergence_rate` 长期在 `0.7 ~ 0.8`

代表性局：

- `round_4_map2`: `0.782`
- `round_4_map6`: `0.7355`
- `round_4_map10`: `0.777`

问题本质：

- planner 并非完全失效
- 但策略形成了与 planner 冲突的局部习惯
- 这会直接放大：
  - 墙边打磨
  - 旧桩不切
  - 返航停滞

### P0-3. `target / route_anchor` 粘滞过强

日志证据：

- `suboptimal_target_hold_rate`
- `avg_target_selection_gap`

代表性局：

- `round_4_map10`
  - `suboptimal_target_hold_rate = 0.3785`
  - `avg_target_selection_gap = 11.049`

问题本质：

- 不是单步误选桩
- 而是持续持有非最优 charger target
- 导致路径质量、返航时机和补能容错同时恶化

## P1：强影响问题，应在 P0 后紧接修

### P1-1. 未知路径 / 窄路承诺过强

日志证据：

- `narrow_unknown_commit_rate`
- `unknown_on_target_path_ratio`
- `all_charger_known_path_count`

代表性失败局：

- `round_4_map4`
- `round_4_map8`
- `round_4_map6`

问题本质：

- 模型在路径不明确时仍继续前推
- 对未知路径风险惩罚不足
- 对“先确认回桩网络再探索”没有形成偏好

### P1-2. 充电提前量不足 / 路过不充

日志证据：

- `missed_charge_opportunity_rate`
- `charger_nearby_not_charged_rate`
- `charge_margin_now`
- `first_late_return_window`

问题本质：

- 不是总是负 slack 才开始回
- 但经常是小提前量才回
- 路过桩时继续拿清扫/streak 奖励，使“现在去充”不够有吸引力

## P2：当前不是主因，但后续应继续观察

### P2-1. 充电桩竞争

当前 benchmark 中：

- `charger_contested_rate ≈ 0`
- `target_charger_robot_count` 很低

结论：

- 当前不是主要矛盾
- 暂不作为优先修复目标

### P2-2. 显式切比雪夫 fallback

当前 benchmark 中：

- `fallback_to_chebyshev_rate = 0`

结论：

- 当前不是主因
- 不应优先围绕 fallback 本身重构

---

## 二、优化方案

## 方案 A：边界语义重构（优先级最高）

目标：

- 让“墙边”不再天然等于高价值边界
- 让“已清扫且低 frontier 的墙边”快速失效

建议动作：

1. 引入“有效边界”概念，替换当前隐式的“墙边即边界”
2. 满足以下条件的格子视为 **失效边界**：
   - 已清扫
   - `dirty_adjacent == 0`
   - `local_frontier_density` 低
   - `cur_visit_count >= 2`
3. 对失效边界增加显式负项：
   - `stale_boundary_penalty`
4. 对沿失效边界连续移动增加递增惩罚：
   - `wall_hugging_clean_floor_penalty`
   - 强度随 `wall_follow_streak / same_region_streak / path_cross_count_50` 增强
5. 将 `frontier_reward` 的生效条件收紧：
   - 如果当前是旧边界、重复路径，则不应继续给 `frontier` 正奖励

预期效果：

- 压低 `wall_hugging_clean_floor_rate`
- 压低 `stale_boundary_follow_rate`
- 减少“沿墙往返磨损”

## 方案 B：把“已清扫旧路”明确纳入代价

目标：

- 减少重复覆盖和路线交叉

建议动作：

1. 对 `revisit_on_clean_floor` 增加强惩罚
2. 对 `redundant_clean_path` 增加强惩罚
3. 对 `low_value_revisit` 增加强惩罚
4. 将 `coverage_efficiency_20` 纳入 shaping：
   - 低效率时给负项
5. 将 `path_cross_count_50` 作为轻度惩罚源
6. 削弱以下正奖励在重访场景中的效果：
   - `cleaning`
   - `streak`
   - `frontier`

说明：

- 不是完全禁止重访
- 而是防止“没有新增价值的重访”继续被当成正行为

## 方案 C：充电时机从“卡点效率”改为“留足提前量”

目标：

- 让模型不再偏好 razor-margin return
- 对“路过不充”和“过晚充电”形成更强约束

建议动作：

1. 引入 `charge_margin_reward`
   - 提前量健康时给正反馈
   - 提前量过小时给负反馈
2. 增加 `missed_charge_opportunity_penalty`
   - 当近桩、低电、且未充电时触发
3. 增加 `charger_nearby_not_charged_penalty`
   - 当近桩且 charge_margin 已低时仍略过，负奖励更强
4. 降低“卡点回充成功”的纯效率奖励权重
   - 减少只强调 `charge efficiency` 的偏向
5. 对 `return` 触发前的 `contract` 阶段给予更保守偏好
   - 提前进入 `contract`
   - 不是等到快没电时再切

预期效果：

- 压低 missed charge 行为
- 抬高安全提前量
- 减少“2~5 格电回桩”的极窄裕量行为

## 方案 D：charger target 选择从“粘滞稳定”改为“全局比较后再稳定”

目标：

- 解决“更近桩不切”
- 让模型具备更合理的多桩规划能力

建议动作：

1. 每步维护所有 charger 的统一代价视图：
   - `astar_dist`
   - `dist`
   - `unknown_path_ratio`
   - `reachable`
   - `priority`
2. target 选择不再只看旧 anchor 稳定性
   - 增加“明显更优目标”强制切换条件
3. 当 `target_selection_gap` 超过阈值时：
   - 不允许继续保留旧 target
4. `route_anchor` 的“稳定”只作为二级约束
   - 不能覆盖显著更优的新桩
5. 对 `suboptimal_target_hold` 增加惩罚

预期效果：

- 降低 `suboptimal_target_hold_rate`
- 降低 `avg_target_selection_gap`
- 让“更近桩不切”显著减少

## 方案 E：让多 charger 路径认知更宽，而不是只盯一个已知桩

目标：

- 解决长局/少桩场景中的“路径认知太窄”
- 逼出“蛙跳式”补能探索

建议动作：

1. 同时维护所有 charger 的 A* 可达信息
2. 当 `all_charger_known_path_count < 2` 且场景为长局/少桩：
   - 额外奖励“尽早明确另一个桩的路径”
3. 对“无明确已知回桩网络却继续深探索”增加惩罚
4. 对 `unknown_on_target_path_ratio` 加强惩罚
5. 在 `2 charger / 2000 step` 等长局场景中：
   - 将“扩图”目标调整为“扩桩网络认知”优先于“随机 frontier”

预期效果：

- 减少“只掌握一个桩”的脆弱状态
- 更接近“从一个桩探索另一个桩”的蛙跳式路线

## 方案 F：在特定病态状态下，planner 需要更强接管

目标：

- 避免策略在明显病态循环中继续拒绝 planner

当前证据：

- `planner_policy_divergence_rate` 长期过高
- 极端墙角循环样本中，planner 长时间被否决

建议动作：

当同时满足以下条件时，触发更强控制：

- `wall_hugging_clean_floor`
- `stale_boundary_follow`
- `no_clean_no_return_progress`
- `same_region_streak` 高
- 或 `loop_suspect / corner_loop_suspect`

在这些状态下：

1. 提高 planner logit bias
2. 或进入短时 hard override 安全模式
3. 并在脱离病态状态后恢复正常策略控制

说明：

- 不建议常态强接管
- 只在明确病态行为中做“急救式接管”

---

## 三、实施优先级

建议按以下顺序落地，不要同时大改：

### 第一阶段（必须先做）

1. 方案 A：边界语义重构
2. 方案 B：已清扫旧路代价
3. 方案 F：病态循环下 planner 强接管

目标：

- 先压掉墙角/墙边病态循环

### 第二阶段

4. 方案 C：充电提前量重构
5. 方案 D：target 选择去粘滞

目标：

- 解决卡点回充与旧桩不切

### 第三阶段

6. 方案 E：多 charger 路径认知扩展

目标：

- 解决长局/少桩场景的结构性脆弱

---

## 四、验证方案

每完成一阶段，都必须重新跑标准 benchmark：

```bash
cd train
bash run_benchmark_parallel.sh code/saved_models/v6-geo-bestmodel-576/model.ckpt-resume.pkl \
  --workers 4 \
  --envs-per-worker 10 \
  --max-wait 1800 \
  --policy-mode eval
```

### 阶段一验收指标

- `wall_hugging_clean_floor_rate` 下降
- `stale_boundary_follow_rate` 下降
- `revisit_on_clean_floor_rate` 下降
- `avg_path_cross_count_50` 下降
- `loop_suspect_rate / corner_loop_rate` 明显下降

### 阶段二验收指标

- `missed_charge_opportunity_rate` 下降
- `charger_nearby_not_charged_rate` 下降
- `late_return_rate` 和极小 `charge_margin_now` 减少
- `suboptimal_target_hold_rate` 下降
- `avg_target_selection_gap` 下降

### 阶段三验收指标

- `all_charger_known_path_count` 提高
- `unknown_on_target_path_ratio` 下降
- `2 charger / 2000 step` 场景下 `win_rate` 提升

---

## 五、当前不建议优先做的事

基于本轮日志，以下方向不建议放前面：

1. 不优先围绕“充电桩竞争”做大改
2. 不优先围绕“显式切比雪夫 fallback”重构
3. 不优先扩大模型结构
4. 不优先改 AMP / 并行评测等基础设施

这些都不是当前 benchmark 暴露出的主要矛盾。

---

## 结论

经过两轮标准 benchmark 和一次针对角落循环的补跑，当前最该优先处理的，不是单一的“回充太晚”，而是：

- **旧墙边低价值打磨**
- **planner 被策略长期否决**
- **target 粘滞**
- **充电裕量过小**

其中最紧急的，是先把：

- `wall_hugging_clean_floor`
- `stale_boundary_follow`
- `loop_suspect / corner_loop_suspect`

压下去。否则即使继续提升分数，模型仍会在完成局中“带病运行”，只是把问题藏起来而已。
