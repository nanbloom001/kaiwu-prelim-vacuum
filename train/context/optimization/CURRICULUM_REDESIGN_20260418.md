# V6 动态课程重设计方案（Resume 友好版）

## 1. 背景与目标

当前课程设计已经不再适配这轮训练条件，主要原因有三点：

1. 行为主线已经发生变化。
当前训练不再只是“继续提分”，而是在修复：
- 沿墙低价值跟随
- 窄路/未知路径承诺过强
- 卡点回充
- 旧桩不切
- 墙角/墙边循环
- planner / policy 长期分歧

2. 训练拓扑已经变化。
`aisrv` 从 2 个扩到 4 个后，课程推进仍然按单个 `EpisodeRunner.episode_cnt` 驱动，而不是按全局训练进度驱动。这会导致：
- 全局已经训练很多
- 本地 runner 还停在 `warmup`
- 课程推进速度和真实训练进度脱钩

3. 当前课程只优化结果，不优化行为。
现有推进门槛只有：
- `win_rate`
- `avg_clean_score`
- `avg_charge_count`

这不足以约束当前真正想修复的行为问题。模型可能：
- 分数还可以
- 但仍在墙边打磨、旧桩不切、卡点回充、返航停滞

因此，新的课程设计目标不是“更快进入 harder”，而是：

1. 让课程真正服务于行为修复与稳定进步。
2. 对 resume 模型更友好，能快速跳过已掌握的前期。
3. 用更综合的指标判断是否推进，而不是只看分数。
4. 保留足够的 harder 场景压力，帮助模型在简单任务上也变稳，而不是只会刷 easy case。


## 2. 当前课程的主要问题

### 2.1 课程推进单位错误

当前代码在 [train_workflow.py](../../../code/agent_ppo/workflow/train_workflow.py) 中使用：

- `self.episode_cnt + 1`

来驱动 `EnvConfigSampler.sample()`，并在 `_stage_name()` 中使用本地 `episode_idx` 强制阶段：

- `episode_idx <= 40 -> warmup`
- `episode_idx <= 200 -> blend`
- `episode_idx <= 400 -> robust`

这在单 `aisrv` 或低并行时还勉强可用，但在现在 `4 aisrv` 下会导致：

- 课程推进按“每个 runner 自己的局数”走
- 而不是按全局训练已见样本量走

结果是：
- 你看到全局训练已经很多步/很多局
- 课程却依然停在 `warmup`

### 2.2 推进条件过于单一

现有门槛在 [conf.py](../../../code/agent_ppo/conf/conf.py) 中是：

- `CURRICULUM_ADVANCE_WIN_RATE = 0.80`
- `CURRICULUM_ADVANCE_AVG_CS = 800`
- `CURRICULUM_ADVANCE_CHARGE = 2.5`

问题在于：

- `avg_charge_count` 不是行为健康指标，只是结果相关代理
- 这套门槛无法识别“带病运行”

例如模型可能：
- `win_rate` 不低
- `avg_cs` 也够
- 但仍然存在：
  - `battery_fail`
  - `return_stall`
  - `wall_hugging_clean_floor`
  - `stale_boundary_follow`
  - `suboptimal_target_hold`
  - `planner_policy_divergence`

### 2.3 课程是硬切换，不是连续混合

当前 `_pick_profile()` 仍然是基于阶段名的一组固定概率。
这会带来两个问题：

1. 阶段内变化不够细。
2. 模型一旦被推进到 harder 分布，没有行为保护机制。

更合理的方式应该是：
- 课程本质是 profile 概率混合
- 这些概率由全局训练状态动态调整


## 3. 新课程设计原则

新的课程应遵守 5 条原则：

### 原则 A：用全局训练进度，而不是本地 episode

课程主时钟优先使用：

- `train_global_step`
- 最近窗口行为指标

而不是单个 `EpisodeRunner.episode_cnt`。

### 原则 B：先稳行为，再提难度

课程推进必须把行为质量放在结果之前。

结果指标是必要条件，但不是充分条件。

### 原则 C：课程应是“连续配比调整”

不要把课程理解成互斥的 `warmup / blend / robust / eval_hard` 四挡。

更合理的是：
- 每个阶段对应一组 profile 配比
- 训练过程中逐步调整 harder 占比

### 原则 D：要允许 resume 模型快速跳过前期

如果 resume 点已经具备：
- 基本生存能力
- 基本清扫能力
- 基本回充能力

那课程不应强制再跑长时间 `warmup`。

### 原则 E：更严格任务可以用于“行为塑形”，但不能压垮训练

“用更严格标准训练，是否能在简单任务上取得较好成绩”这个想法是成立的，但前提是：

1. harder 任务暴露的是同一类真实行为缺陷。
2. harder 占比不能过高，不能把模型直接打散。
3. harder 任务必须作为行为塑形约束，而不是全盘替代 easy/anchor。


## 4. 新课程总体结构

建议把课程改成四个阶段，但这些阶段不再是硬切换，而是：

- 用于定义一组 profile 概率
- 用于定义推进和回退门槛

### Stage S0：Recovery / Resume Stabilize

用途：
- 刚 resume 后的重适应阶段
- 让新 reward / guidance / target 规则先稳定下来

目标：
- 不是冲分
- 是防止策略被新规则直接打散

建议 profile 配比：
- `anchor`: 45%
- `mild`: 35%
- `broad`: 20%
- `broad_eval`: 0%

重点监控：
- `value_clean_loss`
- `entropy_loss`
- `battery_fail_rate`
- `return_stall_rate`
- `wall_hugging_clean_floor_rate`
- `planner_policy_divergence_rate`

退出条件：
- 不是按 episode 数
- 而是满足最小全局步数 + 行为不恶化

建议硬门槛：
- `global_step_since_resume >= 3000`
- `battery_fail_rate <= 0.20`
- `return_stall_rate <= 当前基线 * 1.05`
- `entropy_loss` 不再继续快速上升

如果 resume 点很强，可快速越过此阶段。

### Stage S1：Behavior Repair

用途：
- 集中修复行为病灶

目标：
- 降低病态行为
- 不是只提高 `avg_cs`

建议 profile 配比：
- `anchor`: 25%
- `mild`: 35%
- `broad`: 30%
- `broad_eval`: 10%

重点优化：
- `wall_hugging_clean_floor_rate`
- `stale_boundary_follow_rate`
- `narrow_unknown_commit_rate`
- `missed_charge_opportunity_rate`
- `suboptimal_target_hold_rate`
- `planner_policy_divergence_rate`

进入 S2 的建议硬门槛：
- `win_rate >= 0.72`
- `avg_clean_score >= 720`
- `battery_fail_rate <= 0.12`
- `return_stall_rate <= 0.35`
- `wall_hugging_clean_floor_rate <= 0.06`
- `suboptimal_target_hold_rate <= 0.08`

### Stage S2：Robustness Build

用途：
- 把修好的行为扩展到更宽、更苛刻的任务族

目标：
- 让模型不只会在 easy/anchor 稳定
- 也能在 harder 分布下保持正确行为

建议 profile 配比：
- `anchor`: 10%
- `mild`: 25%
- `broad`: 40%
- `broad_eval`: 25%

重点监控：
- `2000步 / 2 charger` 下的 `battery_fail_rate`
- `unknown_on_target_path_ratio`
- `all_charger_known_path_count`
- `avg_target_selection_gap`

进入 S3 的建议硬门槛：
- `broad_win_rate >= 0.65`
- `battery_fail_rate <= 0.08`
- `suboptimal_target_hold_rate <= 0.05`
- `unknown_path` 相关异常继续下降

### Stage S3：Hard Eval Prep

用途：
- 最终提交前的稳定性与长尾控制

建议 profile 配比：
- `anchor`: 5%
- `mild`: 15%
- `broad`: 35%
- `broad_eval`: 45%

这里不再追求快速学习，而是：
- 让 hardest 配置暴露剩余问题
- 同时保留少量 easy/mild 防止基本策略漂移


## 5. 新的推进逻辑

### 5.1 课程主时钟

建议废弃“本地前 40 局强制 warmup”这种写法。

新的主时钟采用：

- `global_step`
- 最近 `N` 局 rolling metrics

推荐：
- `CURRICULUM_WINDOW = 40`

原因：
- 当前 20 局窗口太短，波动大
- 40 局更能看出行为指标变化趋势

### 5.2 推进不是只看结果，要分两层

每个阶段都使用：

1. 结果层条件
2. 行为层条件

只有两层同时满足，才允许提高 harder 占比。

#### 结果层建议指标
- `win_rate`
- `avg_clean_score`
- `avg_remaining_charge`

#### 行为层建议指标
- `battery_fail_rate`
- `return_stall_rate`
- `wall_hugging_clean_floor_rate`
- `stale_boundary_follow_rate`
- `narrow_unknown_commit_rate`
- `suboptimal_target_hold_rate`
- `planner_policy_divergence_rate`

### 5.3 使用 hysteresis，避免课程抖动

进入下一阶段与回退上一阶段的阈值应不同。

例如：
- 进入下一阶段要求 `battery_fail_rate <= 0.10`
- 回退上一阶段只在 `battery_fail_rate >= 0.18` 时触发

这样避免：
- 刚达标就推进
- 一波坏样本又马上打回


## 6. Resume 模型如何快速越过前期

这是当前最重要的要求之一。

建议增加一个“resume 启动判定”逻辑：

### 如果检测到存在 resume checkpoint 且：
- `global_step > 0`
- 或模型来源不是 fresh init

则不再使用固定前期阶段，而是做一轮 `Resume Qualification`：

使用最近 40 局或最近若干训练窗口，计算：

- `win_rate`
- `avg_clean_score`
- `battery_fail_rate`
- `return_stall_rate`
- `wall_hugging_clean_floor_rate`
- `suboptimal_target_hold_rate`

然后直接映射到最近似阶段：

#### 可直接进 S1 的条件
- `win_rate >= 0.65`
- `avg_clean_score >= 650`
- `battery_fail_rate <= 0.20`

#### 可直接进 S2 的条件
- `win_rate >= 0.75`
- `avg_clean_score >= 760`
- `battery_fail_rate <= 0.12`
- `return_stall_rate <= 0.40`

#### 可直接进 S3 的条件
- `win_rate >= 0.82`
- `avg_clean_score >= 850`
- `battery_fail_rate <= 0.08`
- `suboptimal_target_hold_rate <= 0.05`

这样 resume 模型就可以：
- 快速跳过前期
- 不再被“本地 episode <= 40”锁死


## 7. 更严格训练是否能帮助简单任务

### 结论

**可以，但前提是设计成“行为塑形型 harder 课程”，而不是把全部训练切到 hardest。**

### 为什么可能有效

因为你现在最想修的很多问题，本质上是“在简单任务里被掩盖、在严格任务里才暴露”的：

- 卡点回充
- 旧桩不切
- 墙边打磨
- 未知路径承诺过强
- 返航停滞

这些问题在：
- 少机器人
- 多充电桩
- 高电量
- 短局

里经常不会立刻失败，所以模型会学会“带病成功”。

而在更严格任务里：
- 少 charger
- 长 `max_step`
- 中低电量
- 多轮清扫-回充循环

这些病灶会被放大，训练才会真正有动力修正它们。

### 为什么不能全盘 hardest

如果一上来把 harder 任务占比拉太高，会有两个风险：

1. 旧策略被直接打散，`entropy` 长期过高
2. 模型学不到稳定的基础 coverage / return 节奏

所以正确方式不是“用 hardest 代替 easy”，而是：

- 用 harder 任务作为行为塑形约束
- 同时保留足够 anchor/mild 做稳定器

### 实际建议

在 S1 / S2 阶段，保留：

- 至少 10%~25% 的 `anchor`
- 至少 20%~35% 的 `mild`

同时逐步提高：

- `broad`
- `broad_eval`

这样模型既能：
- 在 harder 中学正确行为
- 又不至于丢掉简单任务上的稳定表现


## 8. 当前结构是否支持 `2000` 步以上

### 当前结论

**当前结构对长 episode 本身是支持的，但对 `2000` 步以上并没有真正完整支持。**

更准确地说：

#### 支持的部分

1. 训练样本切块机制不依赖 episode 总长度。
在 [definition.py](../../../code/agent_ppo/feature/definition.py) 中：
- `sample_process()` 按 `SEQ_CHUNK_LEN = 16`
- 用滑窗把整局切成 recurrent chunks

因此从算法角度讲：
- episode 是 1000 步、2000 步、甚至更长
- 都可以被切成 chunk 训练

2. benchmark 和现有课程已经实际支持到 `2000`。
代码里多处已经把：
- `max_step = 2000`

作为 hardest 配置。

#### 不完全支持的部分

1. 配置文档明确把 `max_step` 范围写成 `1~2000`
见 [train_env_conf.toml](../../../code/agent_ppo/conf/train_env_conf.toml:22)。

2. 课程采样上限是 `2000`
见 [train_workflow.py](../../../code/agent_ppo/workflow/train_workflow.py:258) 附近：
- `levels = [500, 700, 900, 1100, 1400, 1700, 2000]`
- `broad_eval` 最大也是 `2000`

3. 标量归一化把 `2000` 当成时间尺度
见 [preprocessor.py](../../../code/agent_ppo/feature/preprocessor.py:979)：
- `_norm(self.step_no, 2000)`
- `_norm(max(self.max_step - self.step_no, 0), 2000)`

这意味着如果你真的把 `max_step` 提到 `2500 / 3000`：
- 相关 step / remaining_step 特征会失真
- 超过 2000 后会进入饱和区

4. 配置统计分桶上限是 `2000`
见 [train_workflow.py](../../../code/agent_ppo/workflow/train_workflow.py:976)：
- `max_step_bin = min(int(actual_max_step / 500) * 500, 2000)`

这会让 `2000+` 的场景都被压进同一个 bin。

### 因此应如何判断

#### 如果你的问题是：
“当前训练结构能不能稳定处理 2000 步长局？”

答案是：
- **能**
- 而且当前设计已经在做

#### 如果你的问题是：
“能不能直接把课程上限拉到 2500/3000 继续用？”

答案是：
- **不建议直接做**
- 因为现有结构把 `2000` 当成设计上限
- 要支持 `2000+`，至少要同步改：
  - step 标量归一化
  - `max_step` 采样上限
  - 配置分桶
  - benchmark round 定义

### 推荐结论

当前阶段不建议直接把训练上限改到 `2000+`。

更合理的是：
- 先把课程和行为修复做好
- 把 `2000` 作为 hardest 任务充分利用
- 等 `2000 / 2 charger / 多机器人` 真正稳定后，再评估是否需要更长局


## 9. 具体落地建议

### 第一阶段：只改课程推进逻辑，保持 resume 兼容

修改点：
- [train_workflow.py](../../../code/agent_ppo/workflow/train_workflow.py)
- [conf.py](../../../code/agent_ppo/conf/conf.py)

原则：
- 不改 observation
- 不改模型结构
- 不改 checkpoint 形状

因此：
- **完全支持继续 resume**

### 第二阶段：把行为指标接到课程判断

把以下指标纳入课程门槛：
- `battery_fail_rate`
- `return_stall_rate`
- `wall_hugging_clean_floor_rate`
- `stale_boundary_follow_rate`
- `suboptimal_target_hold_rate`
- `planner_policy_divergence_rate`

### 第三阶段：把课程改成 profile 混合器

不要再把 `_pick_profile()` 写成按 stage 固定表，而要：
- 根据阶段输出 profile 概率
- 再根据行为健康度动态微调


## 10. 最终建议

一句话总结：

**现在应该重新设计课程，而且优先级很高。新的课程必须从“按本地 episode 粗暴切阶段、只看分数”升级成“按全局训练进度 + 行为质量 + 混合 profile 配比”来动态推进。这样既能让 resume 模型快速跳过前期，又能真正服务于当前行为修复目标。**

更具体地说：

1. 课程推进应以全局训练步数和 rolling metrics 为主，不再按本地 `episode_idx<=40` 强制 warmup。
2. 推进条件必须同时看结果和行为，而不是只看 `win_rate / avg_cs / avg_cc`。
3. 更严格任务是有价值的，但它应当作为行为塑形工具，而不是全盘替代简单任务。
4. 当前结构适合充分利用 `2000` 步 hardest 场景，但不建议在现阶段直接推到 `2000+`。

