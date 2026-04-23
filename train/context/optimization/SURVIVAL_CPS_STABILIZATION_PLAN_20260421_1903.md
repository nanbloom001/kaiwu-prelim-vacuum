# 生存率与 CPS 稳定提效方案

> 日期: 2026-04-21 19:03:08 CST  
> 修订: 2026-04-21 20:46:00 CST  
> 适用阶段: `Objective Reset` 与 `v1-lite / minimal closed loop` 多轮验证后形成的新主线  
> 目标: 在不贸然简化模型的前提下，优先把当前训练拉回到“高生存率 + 高 CPS + 可解释过渡行为”的稳定区间。  
> 过期提醒: 若后续出现更新日期更晚的同主题方案，应优先读取更新版；但在没有更新版前，**不要跳过本方案继续引用 2026-04-20 或更早的旧路线**。

## 0. 与旧方案的关系

本方案是当前阶段的**主参考方案**，用于接管：

1. `VNEXT_LITE_BEHAVIOR_SIMPLIFICATION_PLAN_20260420.md` 中“降低训练框架复杂度”的旧主线表述
2. `TRAINING_ACTION_PLAN_V2_20260421.md` 中“先纯 Objective Reset，再进入 Route-Phase 主动作闭环”的旧顺序

当前统一口径是：

1. `Objective Reset` 并非完全无效，而是**已验证方向成立，但不足以作为独立第一阶段闭环**
2. `battery-fail economics` 继续保留为底座，但不再作为当前阶段的首要主修复杠杆
3. `TRAINING_ACTION_PLAN_V2_20260421.md` 不作废：
   - 其中的 `scratch run`
   - 每轮单主杠杆
   - 固定窗口对比
   - route-phase 指标链
   仍然有效，继续沿用
4. 本方案实际替代的是：
   - 旧的主线优先级
   - 旧的“先把 Objective Reset 完整做通，再做后续阶段”的判断方式

## 1. 当前总判断

当前阶段不应再把“降低训练框架复杂度”当作主线目标。

原因已经比较清楚：

1. 现有证据更支持当前主问题是：
   - reward economics 只能部分修正方向
   - contract/return 过渡层仍不稳定
   - route-phase 主动作约束与行为分布之间存在耦合
2. 过去几轮实验说明：
   - 只拉 reward，容易把 CPS 压塌
   - 只拉 gate，容易让 survival 或 zero-charge fail 反弹
   - 只拉 teacher，容易形成局部纠偏但不能稳定闭环
3. 当前还没有足够证据证明：
   - backbone 太弱
   - 多头结构本身就是第一主因
   - 必须立即做 architecture-lite 或 supervision-lite 才能继续推进

因此当前主线应改成：

> **先把“生存率 + CPS + 行为过渡”做成稳定闭环；只有在连续多轮联动微调后仍表现出强不可解释性和强参数敏感性，才把模型简化提升为主线。**

## 2. 新的阶段目标

这一阶段只看三类核心目标，不再先追求 planner 指标“好看”。

### 2.1 硬目标

1. 简单图生存率 `>= 0.95`
2. 难图 / 少充电桩图生存率 `>= 0.85`
3. `avg_clean_per_step >= 0.95`

### 2.2 同时必须守住的健康约束

1. `battery_positive_reward_rate` 必须稳定处于低位，不能重新回到“失败仍然赚钱”
2. `zero_charge_battery_fail_rate` 必须持续下降。这里的定义是：
   - `zero_charge battery fail / all episodes`
   - 它表示总体零充电失败风险，而不是“battery fail 中零充电占比”
3. `zero_charge_among_battery_fail_rate` 继续保留，但只作为失败结构诊断指标：
   - `zero_charge battery fail / battery fail`
   - 当它偏高时，优先表示“失败结构仍偏向没形成首充闭环”，而不是直接表示总体 survival 更差
3. `mode_usage_contract` 与 `mode_usage_return` 必须恢复为可解释的过渡层，而不是：
   - `contract` 几乎消失
   - 或过早进入保守态导致 CPS 被吃掉

### 2.3 行为层面的目标表述

本阶段真正要形成的不是某个单指标峰值，而是以下闭环：

`expand / harvest -> contract -> return -> charge`

要求：

1. 简单图上该闭环不能过早触发，否则 CPS 会被压掉
2. 难图 / 少充电桩图上该闭环必须更早进入稳定回充准备，否则 survival 达不到底线
3. 这条闭环必须主要靠当前 policy 学到，而不是靠越来越重的 heuristic 强拉

## 3. 主线排序

后续所有实验与修正，统一按这个顺序理解：

### 3.1 第一主线：稳定生存闭环

目标：

- 压住 battery fail，尤其是 zero-charge battery fail
- 但不再通过持续加大 terminal cost 来硬压

原则：

- 当前 battery-fail economics 已经足够作为底座
- 后续不再把 reward economics 当主修复杠杆

### 3.2 第二主线：同步恢复 CPS

目标：

- 不接受“生存上升但 CPS 明显掉队”的方案
- survival 与 CPS 必须同步看，而不是先牺牲一个、再补另一个

原则：

- contract/return 不能过早触发
- route-phase teacher 不能强到主导全部中期行为

### 3.3 第三主线：恢复行为可解释性

目标：

- 让 `contract` 成为真实过渡层
- 让 `return` 成为必要回充主路径
- 让当前主动作 teacher 形成“中等强度稳定约束”，而不是脉冲式强干预

原则：

- 先追求闭环稳定
- 再追求 planner 对齐更漂亮
- 暂不把 raw `planner_policy_divergence_rate` 当主裁决指标

## 3.4 实验纪律

从本方案开始，仍然默认沿用 `TRAINING_ACTION_PLAN_V2_20260421.md` 的实验纪律：

1. 每轮必须新开 `scratch run`
2. 每轮只允许改一类主杠杆
3. 不把下一轮改动叠加到当前轮
4. 每轮至少跑到：
   - `bootstrap_20`
   - `global_40`
5. 所有结论都必须基于同窗口比较，而不是不同训练阶段的快照混比
6. 每轮不再做“纯单杠杆孤立实验”，而是只允许：
   - `1` 个主杠杆
   - `+ 1` 个必要配套杠杆
   两者必须共同服务于同一个闭环目标
7. 每轮都必须给出失败归因模板，只能归到以下四类之一：
   - `economics_residual`
   - `transition_gate_issue`
   - `action_path_issue`
   - `bucket_split_issue`

## 3.5 训练观察时间表

当前阶段不再等到很后面才判断 run 好坏，而是固定按以下时间表看趋势。

### 3.5.1 观察节点

1. `bootstrap_10`
2. `bootstrap_20`
3. `global_40`
4. `global_80`
5. `global_120`
6. `global_160`
7. `global_200`

### 3.5.2 每个节点的用途

1. `bootstrap_10`
   - 只做早期预警
   - 判断这轮是否已经明显跑歪
   - 不做最终通过/失败结论
2. `bootstrap_20`
   - 做第一次方向判断
   - 同时执行第一轮的机械分流：
     - `economics_residual`
     - `transition_gate_issue`
     - `action_path_issue`
     - `bucket_split_issue`
3. `global_40`
   - 作为正式主判窗口
   - 是否继续跑、是否判失败、是否进入下一轮，都以这里为准
4. `global_80`
   - 只做趋势复核
   - 判断 run 是在恢复，还是在稳定化成某种局部策略形态
5. `global_120`
   - 用于确认该 run 是否已经收敛到某种稳定局部最优
   - 不用于洗白一个已经在 `global_40` 明确失败的 run
6. `global_160`
   - 只做晚恢复复核
   - 用于判断 run 是否存在“前缀失败但后段恢复”的 targeted-resume 价值
7. `global_200`
   - 只做晚恢复 / targeted-resume 复核
   - 不改变 `global_40` 的主判结论

### 3.5.3 经验判断口径

对当前这条训练线，统一采用以下经验判断：

1. `10 ~ 20` 局：
   - 已足够暴露方向是否明显错误
2. `40` 局左右：
   - 已足够形成稳定行为倾向
   - 也是当前最重要的正式主判点
3. `80 ~ 120` 局：
   - 可以判断模型是否开始收敛到某种局部策略形态
4. `160 ~ 200` 局：
   - 只用于判断是否存在晚恢复价值
   - 不用于洗白前缀已失败的 run

### 3.5.3A fixed-window 口径

当前固定采样统一采用双口径，但主判只看 local：

1. `bootstrap_10 / bootstrap_20`
   - 使用 `local_10`
   - 即记录点前 10 局平均
2. `global_40 / 80 / 120 / 160 / 200`
   - 使用 `local_20`
   - 即记录点前 20 局平均
3. 旧 `prefix` 指标继续保留：
   - 只做辅助诊断
   - 不再作为主判依据

### 3.5.4 当前阶段的默认动作

1. 若 `bootstrap_10` 已明显失衡：
   - 记为早期预警，但继续看 `bootstrap_20`
2. 若 `bootstrap_20` 已明显落入某类失败归因：
   - 不继续幻想“多跑一点自然会好”
   - 最晚到 `global_40` 做正式主判
3. 若 `global_40` 明确失败：
   - 当前 run 不再作为继续长跑候选
   - `global_80 / 120` 只用于确认它是否稳定收敛成坏局部最优

## 4. 核心策略

## 4A. 控制栈简化 v1 已实现

> 记录时间：2026-04-22  
> 对应分支：`linux-LTSPPO-control-stack-simplify`

当前主方案已经落下第一版“控制栈简化”实验实现，目的不是继续在现有 `charge/return` 重控制栈里微调，而是先验证：

> 在保持 LTSPPO 行为底座、`curriculum-lite` 和中等强度 battery-fail economics 不变的前提下，收缩 `contract/return` 控制链本身，是否能更稳地恢复 CPS。

### 4A.1 已实现内容

1. 新增独立实验 phase：
   - `s1_survival_control_simplify_v1`
2. 新增统一的 readiness 判定逻辑：
   - `return_now`
   - `pre_return_ready`
3. `preprocessor._infer_mode()` 与 `expert.get_teacher_guidance()` 在该 phase 下共用同一套 readiness 判定，不再各自维护一套近似但不一致的 `contract/return` 触发逻辑。
4. 在该 phase 下，`contract` 不再像 `return` 那样强压：
   - `cleaning`
   - `explore`
   - `frontier`
5. 在该 phase 下，charging local terrain 只保留核心项：
   - `return_progress`
   - `skip_needed_charge`
   - `high_need_return_stall`
6. teacher 观测链已补强：
   - `mode/route_anchor/target/return_action/route_phase` active rate
   - 对应 teacher loss
7. 新增控制栈简化专用行为观测项：
   - `pre_return_readiness_hit_rate`
   - `readiness_to_return_transition_rate`
   - `direct_return_without_readiness_rate`
   - `return_entry_count`
   - `readiness_supported_return_entry_count`

### 4A.2 本轮实验边界

这一版实现明确不做以下事情：

1. 不简化 LTSPPO 主网络
2. 不回退 `curriculum-lite`
3. 不重新设计 compare 主判口径
4. 不继续强化 battery-fail economics
5. 不扩大 teacher 权重，只收缩其触发语义与观测链

### 4A.3 本轮成功定义

控制栈简化 v1 的成功，不看 `mode_usage_contract` 是否提升，而看：

1. `global_40 / 80` 的 local CPS 是否高于当前重控制栈参考 run
2. `route_phase_return_stall_rate` 是否下降
3. `battery_fail_rate` 与 `zero_charge_battery_fail_rate` 是否不明显恶化
4. `pre_return_readiness_hit_rate` 与 `readiness_to_return_transition_rate` 是否提供出更可解释的切换链路
5. 解释这些 readiness 比率时，必须同时结合：
   - `return_entry_count`
   - `readiness_supported_return_entry_count`
   若当前窗口没有 `return entry`，相关比率应视为 `n/a`，而不是 `0.0`

### 4A.4 当前状态

截至当前，这一版已经：

1. 完成代码实现
2. 完成单测
3. 完成 `run_training_phase.py --dry-run`

但还没有正式启动新的 scratch run。  
后续实验应使用该独立 phase，而不是继续在旧 `s1_survival` phase 上追加小步参数回调。

### 4.1 reward economics：只做稳定底座，不再做主驱动

本阶段原则：

1. 保留已经验证有效的“失败不能轻易赚钱”约束
2. 不再继续升级：
   - `BATTERY_TERMINAL_COST_SCALE`
   - `BATTERY_FAIL_TASK_REWARD_SCALE`
   - `ZERO_CHARGE_BATTERY_FAIL_EXTRA_COST`
   等主经济学杠杆
3. reward 只允许做小幅微调，不承担主修复职责

目标：

- 避免重复出现“方向被拉正了，但 CPS 同时塌掉”的情况

### 4.2 contract / return 过渡层：作为当前首要可调层

本阶段最值得优先动的是 `_infer_mode()` 这一层的阶段切换逻辑。

重点关注：

- `PREPARE_RETURN_SLACK_THRESHOLD`
- `CONTRACT_BATTERY_RATIO`
- `CONTRACT_RECOVERABILITY_THRESHOLD`
- `CHARGE_MARGIN_WARN`
- `CONTRACT_ROUTE_PRESSURE_THRESHOLD`

本阶段调参原则不是“更保守”，而是“更分场景”：

1. 简单图：
   - 允许略晚进入 `contract / return`
   - 给清扫与扩张更大的空间
2. 难图 / 少充电桩图：
   - 更早进入稳定回充准备
   - survival 优先于局部 CPS

如果现有训练链尚未按地图桶分开展示指标，那么这不是“可以以后再补”的问题，而是当前阶段必须补齐的观测基础。

### 4.3 route-phase 主动作 teacher：保留，但保持中等强度

当前不应回退到“只靠 aux head”的旧状态，但也不应继续提升到强压式纠偏。

本阶段策略：

1. 保留 contract/return 场景下对主动作的直接 teacher 覆盖
2. 维持中等 teacher weight，不继续放大
3. 重点看：
   - 它是否能稳定守住回充主路径
   - 而不是它能否在所有中期行为上都主导策略

因此它的定位是：

> 保证关键回充路径不漂掉，但不取代整个中期行为分配逻辑。

### 4.4 课程与 profile：改成分桶化评估，而不是统一退化收缩

既然当前目标已经明确分为：

- 简单图 survival `0.95`
- 难图 / 少充电桩图 survival `0.85+`
- 全局 CPS `0.95`

那么课程与 phase 评估不应再只看统一全局退化指标。

建议改成：

1. 简单图：
   - 主看 completion / survival / CPS
   - 作为恢复 `0.95` CPS 的主要场景
2. 难图 / 少充电桩图：
   - 主看 survival / zero-charge fail / 必要 return 闭环
   - 作为保 `0.85+` survival 底线的主要场景
3. profile 暴露调节：
   - 不能只由全局 fail rate 决定
   - 必须结合分桶表现是否失衡

当前阶段对课程系统的统一态度是：

> **不彻底取消课程统计，但停止让课程系统自动控制训练节奏。**

也就是说，本方案不推荐：

1. 保留当前这套硬课程控流：
   - stage promotion / demotion
   - stagnation level 驱动的自动收缩
   - degraded_mainline 驱动的自动 profile 回调
2. 也不推荐直接切到完全纯随机

本方案推荐的是一个小改版：

> **curriculum-lite = observe-only + fixed-profile**

具体定义：

1. `stage` 保留字段与日志，但不再自动推进、不再自动回退
2. `stagnation_level`、`stagnation_reason`、`degraded_mainline` 继续计算，但只做观察，不再控流
3. `profile` 继续保留，但改为 phase/run 级固定权重，不再根据窗口指标自动变化
4. bucket 继续保留，用于评估与报告，不直接驱动 run 内采样权重的动态变化

这样做的直接目的不是“更优雅”，而是先拿回：

1. `归因能力`
2. `节奏控制权`
3. run 内训练分布稳定性

### 4.4.1 `curriculum-lite` 的最小落实形态

如果当前阶段要以最小修改面落地课程改造，优先使用以下形态：

1. 停止 `stage` 自动 promotion / demotion
2. 停止 `stagnation_level` 对训练分布的硬控制
3. 停止 `degraded_mainline` 对 profile 的自动收缩
4. 保留：
   - `curriculum_state`
   - `stagnation_reason`
   - `degraded_mainline`
   - `checkpoint scoring`
   但它们只做观察或 checkpoint 评估，不再改训练流
5. `profile` 权重在一个 run 内固定，由 phase/env 明确给定

这意味着当前阶段的课程系统定位是：

> **观察器与分布说明器，而不是训练控制器。**

### 4.4.2 `curriculum-lite` 的具体实施清单

这一版不要求重写课程系统，只要求把“自动控流”停掉，同时保留统计与报告链。

优先实施顺序如下：

1. 在课程策略层新增一个显式开关，例如：
   - `KAIWU_CURRICULUM_LITE=1`
2. 当该开关开启时，训练链遵守以下规则：
   - `stage` 固定为 phase/run 指定值
   - `choose_stage_decision()` 不再 promotion / demotion
   - `profile_plan_for_runtime()` 不再根据 `stagnation_level` / `degraded_mainline` / adaptive profile 自动改权重
   - `stagnation_status()` 继续算，但只写状态，不再驱动控流
3. `curriculum_state` 继续落盘，保留：
   - `curriculum_stagnation_level`
   - `curriculum_stagnation_reason`
   - `degraded_mainline`
   - `curriculum_profile_weights`
   但这些字段只用于观察与报告
4. `checkpoint_score` 与 benchmark / resume 评分链先不重写：
   - 当前阶段仍允许它们做 checkpoint 评价
   - 但不允许反向改训练流

### 4.4.3 `fixed-profile` 的具体定义

当前阶段不直接做“完整 fixed-bucket random”，先做更小的 `fixed-profile`。

定义如下：

1. 一个 run 内只使用一组固定 profile 权重
2. 权重必须由 phase/env 显式给定，而不是由窗口指标动态推导
3. 默认仍沿用当前 profile 体系：
   - `anchor`
   - `mild`
   - `broad`
4. 推荐第一版固定权重：
   - `anchor = 0.60`
   - `mild = 0.30`
   - `broad = 0.10`
5. 若后续需要调节节奏：
   - 只允许在下一轮 run 启动前手动改权重
   - 不允许 run 内自动漂移

### 4.4.4 为什么这仍然算小修改面

因为它不要求你：

1. 重写 `curriculum_state`
2. 重写 benchmark / checkpoint score
3. 立即实现严格意义上的 map bucket sampler
4. 移除现有课程统计链

它主要只要求在当前集中控制点上“停控流、保观测”：

1. `choose_stage_decision()`
2. `profile_plan_for_runtime()`
3. `stagnation_status()` 的后续使用方式

因此它属于：

> **保留基础设施，收回自动控制权。**

## 4.5 地图分桶的临时机械口径

在正式的地图难度分级体系落地前，本方案统一使用 episode summary 中已有字段做临时分桶，避免实现者各自解释。

使用字段：

- `robot_count`
- `charger_count`
- `battery_max`
- `train_profile`
- `map_id`

临时定义如下：

1. 简单图桶：
   - `robot_count == 1`
   - `charger_count >= 3`
   - `battery_max >= 300`
2. 难图 / 少充电桩桶：
   - `charger_count <= 2`
   - 或 `robot_count >= 3`
   - 或 `battery_max <= 240`
3. 如同一 episode 同时命中多个条件：
   - 优先归入“难图 / 少充电桩桶”
4. `train_profile` 与 `map_id` 继续保留在报告中，作为二级定位信息
5. 若某个 episode 不命中任一桶：
   - 归入“中间桶 / observation-only”
   - 只做观察，不参与当前阶段三大硬目标的主判

说明：

- 这是当前阶段的**执行口径**，不是永久性的地图学术分类
- 后续若形成更正式的 map bucket 映射表，应在更新版方案中显式替换本段

## 5. 三轮执行方案

### 5.1 第一轮：过渡层闭环启动

目标：

- 先把最容易导致 survival / CPS 对冲的过渡层打通
- 不再做几乎纯 gate-first 的实验
- 第一轮就让 `gate + direct guidance + local reward terrain` 形成一个小联动闭环

允许改动：

1. `contract / return gate` 相关阈值
2. contract / return 场景下的 route-phase 主动作 direct guidance
3. `contract / return` 局部 reward 地形的小回调

当前已选定的第一轮 phase overlay 参数为：

1. `PREPARE_RETURN_SLACK_THRESHOLD = 7.5`
2. `CONTRACT_BATTERY_RATIO = 0.27`
3. `CONTRACT_RECOVERABILITY_THRESHOLD = 0.12`
4. `CHARGE_MARGIN_WARN = 17.0`
5. `CONTRACT_ROUTE_PRESSURE_THRESHOLD = 0.52`
6. `ROUTE_PHASE_POLICY_TEACHER_WEIGHT = 0.50`
7. `BATTERY_TERMINAL_COST_SCALE = 41.0`
8. `BATTERY_FAIL_TASK_REWARD_SCALE = 0.22`
9. `EARLY_BATTERY_FAIL_TASK_REWARD_SCALE = 0.07`
10. `BATTERY_FAIL_TASK_REWARD_SCALE_PEAK = 0.14`
11. `EARLY_BATTERY_FAIL_TASK_REWARD_SCALE_PEAK = 0.04`
12. `RETURN_PROGRESS_SHAPING_SCALE = 0.28`
13. `SKIP_NEEDED_CHARGE_PENALTY = 0.20`
14. `HIGH_NEED_RETURN_STALL_PENALTY = 0.12`
15. `NECESSARY_CHARGE_BONUS_SCALE = 1.00`

这一组选值的意图是：

1. 保留当前 `curriculum-lite + economics cleanup` 形态，不再回到旧硬课程
2. battery economics 暂时固定，优先恢复 CPS
3. 通过轻度推迟 charge/return 触发，减少过度保守和过度充电

这里的“局部 reward 地形”仅指：

- 影响 `contract -> return -> charge` 过渡质量的局部 shaping
- `contract -> return` 的推进收益
- `return` 阶段的 stall 惩罚
- 与 `contract/return` 短期梯度直接相关的局部 shaping
- 不包括 battery-fail 主经济学主轴

不允许改动：

1. backbone
2. actor / head 结构
3. `BATTERY_TERMINAL_COST_SCALE`、`BATTERY_FAIL_TASK_REWARD_SCALE`、`ZERO_CHARGE_BATTERY_FAIL_EXTRA_COST` 等 battery-fail 主经济学
4. 大规模 teacher 体系扩张

主判指标：

1. 简单图 survival
2. 难图 / 少充电桩图 survival
3. `avg_clean_per_step`
4. `battery_positive_reward_rate`
5. `zero_charge_battery_fail_rate`
6. `zero_charge_among_battery_fail_rate`
7. `mode_usage_contract`
8. `route_phase_action_teacher_active_rate`

早期分流规则：

1. `bootstrap_20` 时先做一次机械分流，不等到 `global_40` 再解释
2. 若 `battery_positive_reward_rate > 0.15`：
   - 直接判为 `economics_residual`
   - 不继续按“过渡层问题”解释
   - 进入一个小型 objective 回调旁路
3. 若 `battery_positive_reward_rate <= 0.10`，但 `mode_usage_contract < 0.02`：
   - 判为 `transition_gate_issue`
4. 若 `battery_positive_reward_rate <= 0.10`、`mode_usage_contract >= 0.02`，但 `route_phase_action_teacher_active_rate < 0.08`：
   - 判为 `action_path_issue`
5. 若简单图与难图 / 少充电桩图 survival 差值在 `bootstrap_20` 已经 `> 0.15`：
   - 预标记为 `bucket_split_issue`
6. 若 `battery_positive_reward_rate <= 0.10` 且 `zero_charge_battery_fail_rate <= 0.15`，但 `zero_charge_among_battery_fail_rate > 0.60`：
   - 预标记为“失败结构仍偏首充闭环缺失”
   - 在 `transition_gate_issue` 与 `action_path_issue` 中优先排查：
     - `contract -> return` 进入过晚
     - `return` 主路径仍不稳定
     - route-phase direct guidance 可靠覆盖不足

`economics_residual` 旁路规则：

1. 该旁路只允许做 battery-fail 主经济学的小幅回调
2. 只允许动：
   - `BATTERY_TERMINAL_COST_SCALE`
   - `BATTERY_FAIL_TASK_REWARD_SCALE`
   - `EARLY_BATTERY_FAIL_TASK_REWARD_SCALE`
   - `BATTERY_FAIL_TASK_REWARD_SCALE_PEAK`
   - `EARLY_BATTERY_FAIL_TASK_REWARD_SCALE_PEAK`
   - `ZERO_CHARGE_BATTERY_FAIL_EXTRA_COST`
3. 不允许同时继续改第一轮的 gate / direct guidance / local reward terrain
4. 若进入该旁路，则当前 run 不再判作“过渡层闭环启动成功”，而是回到主经济学残余修正

成功标准：

1. `global_40` 时，简单图 survival `>= 0.90`
2. `global_40` 时，难图 / 少充电桩图 survival `>= 0.78`
3. `global_40` 时，`avg_clean_per_step >= 0.75`
4. `battery_positive_reward_rate <= 0.10`
5. `zero_charge_battery_fail_rate <= 0.15`
6. `mode_usage_contract` 处于 `0.02 ~ 0.12`
7. `route_phase_action_teacher_active_rate >= 0.10`

失败信号：

1. 简单图 survival `< 0.88`
2. 难图 / 少充电桩图 survival `< 0.72`
3. `avg_clean_per_step < 0.70`
4. `zero_charge_battery_fail_rate > 0.18`
5. `battery_positive_reward_rate > 0.15`
6. `route_phase_action_teacher_active_rate < 0.08`

失败归因模板：

1. 若 `battery_positive_reward_rate > 0.15`：
   - 归因为 `economics_residual`
2. 若 `battery_positive_reward_rate <= 0.10`，但 `mode_usage_contract` 仍接近 `0` 或 survival/CPS 同时不改善：
   - 归因为 `transition_gate_issue`
3. 若 `battery_positive_reward_rate <= 0.10`、`mode_usage_contract` 回升，但 `route_phase_action_teacher_active_rate` 偏低或回充路径仍漂：
   - 归因为 `action_path_issue`
4. 若简单图指标恢复明显，但难图 / 少充电桩图持续明显偏弱：
   - 归因为 `bucket_split_issue`
5. 若 `zero_charge_battery_fail_rate` 已回到低位，但 `zero_charge_among_battery_fail_rate` 仍高：
   - 不再优先怀疑 `economics_residual`
   - 优先在以下两条里细分：
     - `transition_gate_issue`：`contract` 进入过晚，导致少量失败仍集中在“没形成首充闭环”
     - `action_path_issue`：已进入 `contract/return`，但回充路径学习仍漂，导致一旦 fail 就更容易是 zero-charge 类型

### 5.2 第二轮：分桶稳定化

触发条件：

- `global_40` 时满足以下任一条件：
  - 简单图 survival `>= 0.90`，但难图 / 少充电桩图 survival `< 0.78`
  - `avg_clean_per_step >= 0.75`，但难图 / 少充电桩图 survival 没有同步改善
  - 全局指标改善，但简单图与难图 / 少充电桩图 survival 差值 `> 0.12`

目标：

- 把简单图与难图承担的训练职责真正分开
- 让简单图承担 CPS 恢复
- 让难图 / 少充电桩图承担 survival floor
- 同时把课程系统从“自动控流器”降级成 `curriculum-lite`

允许改动：

1. `stage` 自动 promotion / demotion 停用
2. `stagnation_level` / `degraded_mainline` 保留计算但停止控流
3. `profile` 采样改为固定权重
4. 课程与 profile 的分桶化评估逻辑
5. 运行态对 map bucket 的对比输出

第二轮实施要求：

1. 优先先落 `curriculum-lite = observe-only + fixed-profile`
2. 只有在 run 内分布已经稳定后，才继续细化 map bucket 报告
3. 不要求第二轮就完成严格的 bucket sampler 重构
4. 第二轮的重点是：
   - 停掉课程自动控流
   - 稳定训练分布
   - 提升归因能力

不允许改动：

1. backbone
2. battery-fail 主经济学
3. route-phase teacher 量级主设置
4. 大规模 heuristic 重写
5. 直接切到完全纯随机

成功标准：

1. `global_40` 时，简单图 survival `>= 0.93`
2. `global_40` 时，难图 / 少充电桩图 survival `>= 0.82`
3. `global_40` 时，`avg_clean_per_step >= 0.85`
4. 简单图与难图 / 少充电桩图 survival 差值 `<= 0.12`
5. 同一 run 内 profile 分布不再因停滞/退化信号自动漂移

失败归因模板：

1. 若简单图显著改善、难图仍弱：
   - 归因为 `bucket_split_issue`
2. 若分桶后两类桶仍同步低迷：
   - 回看第一轮，优先按 `transition_gate_issue` 或 `action_path_issue` 重判

### 5.3 第三轮：主动作路径稳定化

触发条件：

- 前两轮后 survival / CPS 已接近目标
- 且满足以下任一条件：
  - `route_phase_planner_divergence_rate > 0.45`
  - `reliable_planner_divergence_rate > 0.30`
  - `route_phase_action_teacher_active_rate < 0.12`

目标：

- 不再第一次接入动作主路径闭环
- 而是在第一轮已经前移 direct guidance 的基础上，做稳定化微调

允许改动：

1. `ROUTE_PHASE_POLICY_TEACHER_WEIGHT`
2. route-phase teacher 的可靠度分层阈值
3. 与主动作 teacher 激活率直接相关的 phase 参数

不允许改动：

1. 重新增加大量 teacher 类型
2. 再次改 battery-fail 主经济学
3. 直接进入 architecture-lite

成功标准：

1. `global_40` 时，简单图 survival `>= 0.95`
2. `global_40` 时，难图 / 少充电桩图 survival `>= 0.85`
3. `global_40` 时，`avg_clean_per_step >= 0.95`
4. `route_phase_action_teacher_active_rate >= 0.12`
5. `route_phase_planner_divergence_rate <= 0.45`
6. `reliable_planner_divergence_rate <= 0.30`
7. 相比第二轮，简单图 survival、难图 / 少充电桩图 survival、`avg_clean_per_step` 三项都不得下降超过 `0.03`

失败信号：

1. `route_phase_action_teacher_active_rate < 0.08`
2. `route_phase_planner_divergence_rate` 相比第二轮下降不足 `0.02`
3. 简单图 survival、难图 / 少充电桩图 survival、`avg_clean_per_step` 任一项相比第二轮下降超过 `0.03`

## 6. 统一验收口径

从本方案开始，后续每一轮必须统一输出以下结果，不再只给一个全局平均：

### 6.1 地图分桶指标

1. 简单图 survival
2. 简单图 CPS
3. 难图 / 少充电桩图 survival
4. 难图 / 少充电桩图 CPS

### 6.2 全局行为与失败指标

1. `avg_clean_per_step`
2. `battery_positive_reward_rate`
3. `battery_fail_rate`
4. `zero_charge_battery_fail_rate`
5. `zero_charge_among_battery_fail_rate`
6. `mode_usage_contract`
7. `mode_usage_return`
8. `route_phase_return_stall_rate`
9. `route_phase_planner_divergence_rate`
10. `reliable_planner_divergence_rate`
11. `route_phase_action_teacher_active_rate`
12. `route_phase_policy_teacher_loss`

### 6.2.1 这两个 zero-charge 指标的职责

为了避免后续再次误判，统一按下面方式理解：

1. `zero_charge_battery_fail_rate`
   - 主风险指标
   - 看总体零充电失败风险
   - 可直接参与课程、停滞、主验收和主比较
2. `zero_charge_among_battery_fail_rate`
   - 结构诊断指标
   - 看 battery fail 内部结构是否仍偏向“完全没形成首充闭环”
   - 不单独作为主控流指标
   - 只在失败归因时辅助判断：
     - 更像 `transition_gate_issue`
     - 还是更像 `action_path_issue`

### 6.3 固定比较方式

1. 固定使用同窗口比较
2. 固定使用：
   - `bootstrap_10`
   - `bootstrap_20`
   - `global_40`
   - `global_80`
   - `global_120`
3. `global_40` 作为主判窗口
4. `bootstrap_10 / 20` 只看早期方向
5. `global_80 / 120` 只做趋势复核

## 7. 何时才考虑“简化模型”

当前不把简化模型当主线。

只有出现以下至少 3 条，才允许把“模型简化”提升为正式方向：

1. 连续 `2 ~ 3` 轮小范围联动微调后，结果仍高度不稳定
2. 同方向的小改动会反复引发明显反向结果，参数极敏感
3. `route_anchor / target / mode / return_action` 等中间头与最终动作长期脱节，可解释性很弱
4. 同窗口比较中，改善无法稳定复现
5. 需要依赖越来越多 heuristic / gate 才能勉强稳定

即便进入“简化模型”阶段，也必须遵守这个顺序：

1. 先简化 supervision 与 auxiliary influence
2. 再减少 actor path 中的中间语义依赖
3. 最后才考虑 backbone / 主结构级简化

## 8. 禁止动作

当前阶段明确不做：

1. 重新把“框架降复杂度”本身当成主目标
2. 直接做 architecture-lite
3. 贸然删 head 或拔掉当前 policy 仍在使用的上下文
4. 再次把 reward economics 当唯一主修复项
5. 只看全局平均，不看地图分桶结果
6. 因为某个单次 run 的 planner 指标难看，就提前推翻整个 survival/CPS 主线

## 9. 一句话总结

> 当前最合理的主线不是继续推进“训练框架降复杂度”，而是用“稳定 reward 底座 + 回调 contract/return 过渡层 + 中等强度 route-phase 主动作 teacher + 分地图桶课程/评估”四件事，先把简单图 `0.95` survival、难图 / 少充电桩图 `0.85+` survival 和 `0.95` CPS 做成稳定闭环；只有连续多轮都证明这条路仍然高度不可解释、极度敏感，才转入模型简化。
