# Checkpoint 选择评分体系 V2（面向 Resume / 提交点）

## 1. 目标

本评分体系的唯一目的不是“解释训练”，也不是“衡量单局上限”，而是：

- **筛选最适合继续 resume 的保存点**
- **筛选最适合最终提交的保存点**

这两个目标高度相关，但并不完全相同：

- 适合继续 resume 的点，需要：
  - 当前策略健康
  - 学习状态稳定
  - 没有明显发散
  - 未来继续训练的风险小

- 适合最终提交的点，需要：
  - 在固定 benchmark 下更稳
  - 长尾风险更低
  - battery / collision 更少
  - 行为病灶更少

因此新体系不再试图用一个简单标量粗暴解决全部问题，而是设计成：

1. **硬门槛（Eligibility Gates）**
2. **Resume 分支评分**
3. **Submission 分支评分**
4. **统一保留决策分数**


## 2. 旧 `robust_score` 的问题

当前代码中的 `robust_score` 定义为：

```python
robust_score = (
    rolling_avg
    + 3.0 * _percentile(scores, 0.10)
    - 8.0 * invalid_move_rate
    - 20.0 * (1.0 if fail_reason == "battery" else 0.0)
    - 30.0 * (1.0 if fail_reason == "collision" else 0.0)
)
```

它在早期是合理的，因为当时主要问题集中在：
- battery fail
- collision fail
- invalid move
- 低尾分数

但放到现在，它已经不够用了，原因有 5 个：

### 2.1 混合了窗口信号和单局信号

- `rolling_avg` 和 `10%分位数` 是窗口级
- `invalid_move_rate` / `battery` / `collision` 是当前局级

这会导致：
- 最近窗口整体很好
- 但刚好当前一局碰撞死
- 分数被一次性大幅打掉

### 2.2 过度依赖 raw clean score

你已经指出这一点了，而且是对的。

`clean_score` 受以下因素强烈影响：
- `max_step`
- `battery_max`
- `robot_count`
- `charger_count`

所以如果任务长度和资源配置不固定，只看 `CS` 会天然偏向：
- 长局
- 高电量
- 易生存配置

### 2.3 看不到“带病运行”

当前旧分数几乎不看这些行为问题：
- `return_stall`
- `late_return`
- `wall_hugging_clean_floor`
- `stale_boundary_follow`
- `narrow_unknown_commit`
- `missed_charge_opportunity`
- `suboptimal_target_hold`
- `planner_policy_divergence`

也就是说，模型即使：
- 不太死
- 分数也还不错
- 但一直沿墙打磨、旧桩不切、返航停滞

旧 `robust_score` 也几乎看不出来。

### 2.4 不区分“适合继续训练”和“适合直接提交”

一个 checkpoint 可能：
- benchmark 结果不错
- 但 learner 已经进入高熵漂移期，不适合继续 resume

也可能：
- learner 状态很好，适合继续训
- 但固定 benchmark 还不是当前最优提交点

旧分数没有区分这两种用途。

### 2.5 不可扩展

每次新增一个指标，都需要重新人工改公式。

这会导致：
- 新增指标越多，旧公式越失真
- 每次都得推翻重写


## 3. 新评分体系的设计原则

### 原则 A：先过门槛，再算分

先做 hard gates，再做加权评分。  
因为有些模型不应该进入候选池：

- 例如 battery fail 太高
- collision fail 太高
- learner 明显失稳

### 原则 B：`CS` 不再做主指标

在训练侧和课程侧，`CS` 不应是主指标。  
主效率指标改为：

- `clean_per_step`
- `cps_win`

也就是：
- 单位步数清扫效率
- 完成局里的单位步数清扫效率

`clean_score` 只保留为：
- 固定 benchmark 拓扑下的弱辅助指标
- 不能作为主导项

### 原则 C：分离 Resume 价值和 Submission 价值

新体系明确拆成两条分支：

- `ResumeReadinessScore`
- `SubmissionScore`

然后再组合成：

- `CheckpointPreservationScore`

### 原则 D：核心指标固定，扩展指标注册化

评分体系要兼容未来新增指标，不应该每次重写公式。

因此设计成两层：

1. **Core Metrics**
   - 默认长期存在
   - 构成基础分

2. **Extension Metrics**
   - 新增行为诊断项
   - 作为扩展惩罚/奖励
   - 缺失时自动中性，不破坏总分结构

### 原则 E：固定 benchmark 优先于训练内局部窗口

用于最终提交的决策时，优先级必须是：

1. 固定 benchmark
2. 训练窗口
3. 单局高分


## 4. 评分框架总览

新体系分四层：

### Layer 1: Eligibility Gates

决定 checkpoint 是否有资格进入候选池。

### Layer 2: ResumeReadinessScore

判断它是否适合作为“继续训练”的起点。

### Layer 3: SubmissionScore

判断它是否适合作为“最终提交”的候选。

### Layer 4: CheckpointPreservationScore

统一产出一个保存优先级分数，用于：
- 是否归档
- 是否升级为主 resume
- 是否升级为主 submit 候选


## 5. Layer 1：Eligibility Gates

硬门槛不追求精细排序，只负责：
- 剔除明显不适合保留的点
- 限制极端坏模型混入

### 5.1 Resume 候选硬门槛

建议：

- `completed_rate >= 0.55`
- `battery_fail_rate <= 0.25`
- `collision_fail_rate <= 0.12`
- `entropy_loss <= 1.05`
- `value_clean_loss` 不能连续恶化
- `teacher_active_rate` 不得掉线

若不满足，则：
- 不作为主 resume 候选
- 只能作为“实验快照”保存

### 5.2 Submission 候选硬门槛

建议在固定 benchmark 下要求：

- `WR >= 0.60`
- `battery_fail_rate <= 0.20`
- `collision_fail_rate <= 0.08`

如果不满足：
- 不进入提交候选池


## 6. Layer 2：ResumeReadinessScore

这条分支只回答一个问题：

**这个点适不适合继续训练？**

### 6.1 类别权重

总分 100，分成 4 类：

- Safety / Survivability：30
- Efficiency / Yield：20
- Behavior Health：25
- Learning Health：25

### 6.2 Resume 核心指标

#### A. Safety / Survivability（30）

- `completed_rate`
- `battery_fail_rate`
- `collision_fail_rate`
- `late_return_rate`
- `return_stall_rate`

#### B. Efficiency / Yield（20）

- `avg_clean_per_step`
- `cps_win`
- `avg_remaining_charge`

说明：
- 不用 raw `avg_clean_score` 做主导
- 因为 `max_step` 和配置不固定

#### C. Behavior Health（25）

优先使用当前已稳定存在的行为指标：

- `late_return_rate`
- `late_contract_rate`
- `return_progress_per_step`
- `return_efficiency_ratio`
- `recoverability_violation_rate`

如果新增指标存在，则作为扩展项加入：
- `wall_hugging_clean_floor_rate`
- `stale_boundary_follow_rate`
- `narrow_unknown_commit_rate`
- `missed_charge_opportunity_rate`
- `suboptimal_target_hold_rate`
- `planner_policy_divergence_rate`

#### D. Learning Health（25）

这是旧体系完全没有的，但对 resume 很关键。

建议使用：

- `entropy_loss_band_score`
- `value_clean_loss_trend_score`
- `value_survive_loss_trend_score`
- `mode_teacher_active_rate`
- `route_anchor_teacher_active_rate`
- `target_teacher_active_rate`
- `return_action_teacher_active_rate`

### 6.3 Entropy 处理方式

`entropy` 不应越低越好，也不应越高越好。  
它应使用“目标区间”评分：

- 过低：说明过早僵化
- 过高：说明策略发散或还未重收敛

建议目标带：

- 最优区间：`0.65 ~ 0.85`
- 超出后按距离衰减

### 6.4 Resume 分数解释

高 `ResumeReadinessScore` 代表：

- 策略行为健康
- learner 数值稳定
- 即使不是当前 benchmark 最优，也值得继续训


## 7. Layer 3：SubmissionScore

这条分支只回答一个问题：

**这个点是不是最适合提交？**

### 7.1 类别权重

总分 100，分成 4 类：

- Completion & Safety：40
- Efficiency：25
- Stability：15
- Behavior Quality：20

### 7.2 Submission 核心指标

#### A. Completion & Safety（40）

优先级最高：

- `WR / completed_rate`
- `battery_fail_rate`
- `collision_fail_rate`

这是提交点最核心的部分，因为：
- 失败会直接吞掉后续收益

#### B. Efficiency（25）

主要使用：

- `cps_win`
- `avg_clean_per_step`

弱辅助项：

- `avg_clean_score_win`

注意：
- `avg_clean_score_win` 只能作为固定 benchmark 拓扑下的辅助项
- 如果 benchmark round 结构变化，应将其权重降到最低或重新标定

#### C. Stability（15）

这里体现“低尾稳定性”，替代旧 `robust_score` 的那部分直觉。

建议使用：

- `10%分位 completed-episode CPS`
- `10%分位 clean_per_step`
- `MAP_STATS.spread`（若有）
- round 间方差

即：
- 不只看平均
- 也看低尾和不同 round 是否稳定

#### D. Behavior Quality（20）

用于区分“高分但带病运行”和“高分且更干净”的模型。

建议使用：

- `return_stall_rate`
- `late_return_rate`
- `recoverability_violation_rate`

扩展项：

- `wall_hugging_clean_floor_rate`
- `stale_boundary_follow_rate`
- `narrow_unknown_commit_rate`
- `missed_charge_opportunity_rate`
- `suboptimal_target_hold_rate`
- `planner_policy_divergence_rate`


## 8. Layer 4：CheckpointPreservationScore

这是最终的统一保留分数。

### 8.1 当 benchmark 不存在时

只做训练内 provisional 评分：

```text
CheckpointPreservationScore = ResumeReadinessScore
status = provisional
```

用途：
- 决定是否保留为候选 resume 点
- 不能直接升级为最终提交点

### 8.2 当 benchmark 存在时

统一分数：

```text
CheckpointPreservationScore =
    0.45 * ResumeReadinessScore
  + 0.55 * SubmissionScore
```

为什么 benchmark 更高权重：
- 最终保存点的主要目标是“值得保留为主线”
- benchmark 比单纯训练窗口更接近真实使用价值

### 8.3 推荐状态标签

不要只输出一个分数，建议输出：

- `provisional_resume`
- `promoted_resume`
- `submit_candidate`
- `archived_reference`

这样分数和用途能分开。


## 9. 指标注册机制（兼容未来新增指标）

为避免以后每加一个指标就重写公式，建议把评分体系做成**注册表驱动**。

每个指标都用统一 schema 描述：

```python
{
  "name": "battery_fail_rate",
  "branch": ["resume", "submission"],
  "category": "safety",
  "direction": "lower_better",
  "required": True,
  "weight": 12.0,
  "normalizer": {"bad": 0.25, "good": 0.05},
  "neutral_if_missing": False,
}
```

### 注册表字段建议

- `name`
- `branch`
- `category`
- `direction`
- `required`
- `weight`
- `normalizer`
- `neutral_if_missing`
- `source`
  - `training`
  - `benchmark`
  - `both`

### 新增指标时怎么处理

如果是行为诊断类新指标，例如：
- `wall_hugging_clean_floor_rate`
- `planner_policy_divergence_rate`
- 未来的新 anomaly

做法是：

1. 把它注册到 `behavior` 类
2. 给它一个小到中等权重
3. 如果某些旧 checkpoint 没有这个指标，就：
   - `neutral_if_missing=True`
   - 缺失时按中性分处理，不直接打死

这样新增指标就不会破坏旧分数体系。


## 10. 关于 `CS` 和 `CPS` 的最终处理原则

你特别强调这一点是对的。

### 10.1 训练侧

在训练侧，**`CS` 不能做主指标**。  
因为：
- `max_step` 不固定
- 配置难度不固定

因此训练侧主效率指标应是：

- `avg_clean_per_step`
- `cps_win`

### 10.2 benchmark 侧

固定 benchmark 拓扑下，可以保留：

- `avg_clean_score_win`

但只作为弱辅助项，原因是：
- 即使 benchmark 固定，round 之间的 `max_step` 也不同

### 10.3 结论

新的评分体系里：

- `CPS / clean_per_step` 是主效率指标
- `CS` 只做辅助参考，不再主导


## 11. 推荐的最终保存决策流程

### 11.1 训练中

每隔固定窗口计算：

- `ResumeReadinessScore`

如果：
- 超过当前 best provisional
- 且通过 resume hard gates

则保存：
- `provisional_resume`

### 11.2 固定 benchmark 后

计算：

- `ResumeReadinessScore`
- `SubmissionScore`
- `CheckpointPreservationScore`

然后按用途决定：

- `CheckpointPreservationScore` 最高 -> 主线归档点
- `SubmissionScore` 最高 -> 提交优先候选
- `ResumeReadinessScore` 最高 -> 继续训练优先候选

如果三者不一致，可以同时保留 2~3 个点，而不是硬选一个。


## 12. 最终建议

一句话总结：

**新的保存评分体系不应再依赖单一 `robust_score`，而应改成“硬门槛 + ResumeReadinessScore + SubmissionScore + 可扩展指标注册表”的结构。这样既能避免 raw CS 因 `max_step` 可变而误导，也能在未来新增行为指标时保持兼容，真正服务于“挑出最值得保留为 resume/提交点的 checkpoint”。**

最重要的三条落地原则是：

1. **训练侧主看 `CPS / clean_per_step`，而不是 raw `CS`。**
2. **最终提交侧以固定 benchmark 为主，训练窗口为辅。**
3. **新增行为指标一律通过注册表扩展，不再重写总公式。**

