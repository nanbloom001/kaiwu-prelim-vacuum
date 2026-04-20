# 统一问题定位报告 — 2026-04-20

> 基准会话：`20260420-094316`
> 主要证据源：`code/curriculum_state.json`、`curriculum_signals/*.json`、`train/log/learner/learner_train_pid*_log_*.log`、`train/log/aisrv/aisrv_kaiwu_rl_helper_pid*_log_*.log`
> 参考材料：`DEEP_PROBLEM_DIAGNOSIS_REPORT_20260420.md`、`SESSION_20260420_DIAGNOSTIC.md`、两份独立子代理结论（Bohr / Euclid）以及之前的全仓审查结论

## 1. Executive Summary

当前训练的主问题，不是训练链路损坏，也不是单一监控假象，而是**策略已经进入一个稳定但不健康的局部最优**：它可以维持一定的清扫效率和部分胜率，但没有学会高质量的回充闭环，也没有在关键阶段真正跟随 planner。结果表现为：

- `planner_policy_divergence_rate` 长期高位，最新 40 局约 `0.844`
- `return_stall_rate` 长期高位，最新 40 局约 `0.536`
- `battery_fail_rate` 高，最新 40 局约 `0.425`
- `zero_charge_battery_fail_rate` 极高，最新 40 局约 `0.706`
- `curriculum_stagnation_level = 3`
- `invalid_for_promotion = true`
- `requires_reward_revision = true`
- `degraded_mainline = true`

这说明系统当前不是“没在学”，而是**在错误方向上持续学**。更准确地说，它学会了“继续清扫并维持一定收益”，但没有学会“在 planning / return 阶段稳定形成可执行的回充动作链”。curriculum 对这个状态的判定基本是对的：当前高停滞不是假警报，而是真实的结构性停滞。

## 2. 当前训练趋势的真实判断

只看当前最新 session `20260420-094316` 的最新窗口：

### 2.1 最新 40 局窗口

- `win_rate = 0.575`
- `battery_fail_rate = 0.425`
- `zero_charge_battery_fail_rate = 0.7059`
- `avg_clean_per_step = 0.5101`
- `cps_win = 0.4653`
- `avg_charge_count = 2.85`
- `avg_charge_count_completed = 4.3478`
- `planner_policy_divergence_rate = 0.8440`
- `return_stall_rate = 0.5358`

### 2.2 最新 20 局窗口

- `win_rate = 0.60`
- `battery_fail_rate = 0.40`
- `zero_charge_battery_fail_rate = 0.75`
- `avg_clean_per_step = 0.5288`
- `cps_win = 0.4882`
- `avg_charge_count = 2.4`
- `avg_charge_count_completed = 3.3333`
- `planner_policy_divergence_rate = 0.8461`
- `return_stall_rate = 0.5465`

### 2.3 结论

当前训练不能简单归类为“明显向好”。更准确的判断是：

- 训练基础设施是健康的，learner 在推进，`global_step_since_resume` 已到 `96k+`
- 清扫效率没有崩，`avg_clean_per_step` 和 `cps_win` 仍在可接受区间
- 但结果质量正在被回充链路问题主导
- 最新窗口的胜率、battery fail、zero-charge fail 都不支持“整体正在自然修复”的乐观判断

所以当前趋势应定性为：

> **训练链路健康，但策略趋势不健康；不是纯坏线，但也不是正在稳定收敛的好线。**

## 3. 当前主问题定性

我的独立判断是：

> **当前主问题是“planner 约束无法在关键回充阶段落到动作层”，叠加“充电相关 shaping 对正确回充闭环的驱动力不足”，最终形成高 divergence、高 stall、高 zero-charge fail 的局部最优，并被 curriculum 锁死在 warmup。**

这不是单一 reward 权重问题，也不是单一 curriculum 门槛问题，而是一条连续因果链：

1. planner 信号在统计上很强，但在关键阶段对动作的实际塑形不足
2. return / charge 阶段缺乏足够强、足够持续的正向学习驱动
3. 策略因此学会了继续清扫和赚主任务分，但没有学会高质量回充
4. 回充失败集中表现为 return stall 和 zero-charge battery death
5. curriculum 再把这种结构问题稳定识别为 `planner + stall`，长期禁止晋级

## 4. 当前是否陷入局部最优

是，已经陷入，而且这个局部最优的行为模式很清楚。

### 4.1 当前局部最优的具体形态

当前策略更像在执行下面这个替代解：

> **尽量继续清扫，依靠 cleaning/explore/streak/cps 奖励维持正收益；在需要回充时进入 return 模式，但经常回充推进质量差；有些局靠多次充电或运气完成，有些局则在一次都没充上的情况下电池耗尽。**

### 4.2 证据

- 40 局窗口里 `avg_charge_count_completed = 4.3478`，说明完成局仍依赖不少充电行为
- 但 `avg_charge_count_battery_fail = 0.8235`，而 `zero_charge_battery_fail_rate = 0.7059`，说明失败局大量根本没形成有效充电闭环
- `return_stall_rate` 在 40 局和 20 局都稳定 > `0.53`
- `planner_policy_divergence_rate` 在 40 局和 20 局都稳定 > `0.84`

也就是说，模型不是完全不会赢，而是在靠一个“行为结构不健康”的替代策略赢。

## 5. planner-policy divergence 与 return stall 的共同模式

这是当前最核心的一组症状。

### 5.1 planner 信号不是没有，而是没有在关键阶段变成动作主导

从代码和运行态结合看：

- `train_workflow.py` 统计 `planner_policy_divergence` 的口径很硬，基本是 planner 建议动作与实际动作是否一致
- 但 `preprocessor.py` 里的 `planner_alignment_reward` 只在 `battery_state == "safe"` 且不处于 `CONTRACT/RETURN` 时才生效

这意味着：

- 偏离率统计覆盖的是大量关键阶段
- 但真正用来纠偏的 reward 并不覆盖这些关键阶段

结果就是：

> **“偏离被记账了，但没有在最需要的时候被充分训练纠正。”**

### 5.2 return 已经成为一个 mode 标签，但还没有成为一个高质量动作模板

从之前多次抽样的 `DEATH_TRAJ` 和当前状态一致可以判断：

- battery death 末段经常是在 `mode=4` 下发生
- slack 持续为负
- 动作表现为重复、来回摆动、推进差

这说明策略知道“现在该回了”，但没有学会“怎么按 planner 或可靠 route 真正回去”。

所以偏离主要不是发生在“要不要进入 return”这一层，而是：

> **发生在 return 之后的动作执行层。**

### 5.3 为什么完成局里 divergence 仍然很高

因为当前完成局并不等于“高质量贴 planner”。

完成局的常见真实结构更可能是：

- 主任务奖励先维持住正收益
- 充电次数不低
- return 阶段并不顺滑，但通过更多步数、更多尝试把局做完

所以 win rate 的存在并不反证 planner/stall 问题，反而说明：

> **模型已经找到一种“不太跟 planner、但仍能完成一部分任务”的替代策略。**

## 6. 奖励结构是否在支持错误局部最优

结论是：**是的，当前奖励结构没有充分打断这个错误局部最优。**

### 6.1 当前正向奖励主导项

40 局窗口正项占比：

- `cleaning`: `42.93%`
- `streak`: `17.43%`
- `explore`: `17.03%`
- `cps_bonus`: `15.24%`
- `charge_route_progress_bonus`: `4.83%`
- `charger_access_discovery_bonus`: `1.70%`
- `necessary_charge_bonus`: `0.62%`
- `charger_access_probe_bonus`: `0%`

也就是说，主任务相关正项远强于充电相关正项。

### 6.2 当前负向奖励主导项

40 局窗口负项占比：

- `coverage_tangle_penalty`: `43.30%`
- `charge_detour_cost`: `22.62%`
- `planner_alignment`: `10.84%`
- `idle`: `10.26%`
- `charge_interrupt_cost`: `10.08%`

这里最关键的不是某个负项高，而是：

- `charge_detour_cost` 这类“低质量回充成本”很显眼
- 但像 `skip_needed_charge_penalty` 这种“该充不充”的直击型惩罚几乎为零

### 6.3 probe 奖励基本形同虚设

当前最新窗口：

- `avg_reward_charger_access_probe_bonus = 0.0`
- `reward_positive_share_charger_access_probe_bonus = 0.0`

这与代码设计是一致的：`probe` 的 gate 需要极窄条件，尤其是 `all_known_paths <= 0`，而当前 `avg_all_charger_known_path_count` 在 `2.5` 左右，因此几乎恒不满足。

这意味着：

> **在最需要稠密引导“先把充电路径知识补出来”的阶段，系统实际上没有提供有效 reward。**

### 6.4 充电相关净效果太弱，且方向不够清晰

当前 40 局窗口：

- `reward_charging_positive_total = 0.01863`
- `reward_charging_negative_total = 0.01503`
- `reward_charging_net_total = 0.00360`

这里不能粗暴地说“系统在惩罚充电本身”，因为 detour / interrupt 惩罚的是低质量充电行为，而不是所有充电行为。但可以明确说：

> **当前充电相关信号并没有形成足够强的正向学习驱动，去压过 cleaning/explore/streak 这类更稳定、更容易拿到的正收益。**

这正是错误局部最优能长期存活的原因之一。

## 7. 为什么停滞等级长期高

这一点上，我认同深度报告和两个子代理的共同判断：**高停滞是真问题，不是假警报。**

### 7.1 当前状态

- `curriculum_stagnation_level = 3`
- `curriculum_stagnation_reason = ["planner", "stall"]`
- `invalid_for_promotion = true`
- `requires_reward_revision = true`
- `degraded_mainline = true`
- `degraded_mainline_windows = 328`

### 7.2 为什么被锁在 warmup

当前核心门外指标始终高于可接受区：

- `planner_policy_divergence_rate ≈ 0.84`
- `return_stall_rate ≈ 0.53`

所以 curriculum 并不是在错杀一个本来已经健康的策略，而是在稳定识别：

- policy 不贴 planner
- return 阶段推进质量差

这一判断和运行态日志、`DEATH_TRAJ` 模式、battery fail 结构是相互印证的。

### 7.3 为什么越训越难自己出来

一旦进入高停滞：

- profile 会保持保守
- promotion 被禁止
- 训练继续集中在同类分布下
- 行为模式没有足够外部冲击打破

所以系统进入了：

> **行为问题 -> 课程识别为停滞 -> 分布更保守 -> 行为问题继续维持**

这就是当前的锁死机制。

## 8. 为什么反复调整后仍然不理想

这是这份报告最重要的结论之一。

我的判断是：**之前的调整并非完全无效，而是多数调整只打到了“辅助层”和“结果层”，没有真正打中“return 动作模板如何学出来”这个核心点。**

### 8.1 之前调整真正解决了什么

- 训练 correctness 和 resume/shared-state 这类底层问题被修复了不少
- learner/aisrv 链路已经健康，不是 infra 卡住
- session / resume / monitor 的大坑被部分填上
- reward 里加入了 discovery、route_progress 等充电相关 shaping，至少让充电信号不再完全缺席

### 8.2 但没有打中的点

1. **planner 对齐信号没有覆盖到最关键的 return/contract 阶段**
2. **return action teacher 覆盖仍然偏低，关键动作阶段监督不够密**
3. **probe 类早期 charger-access 稠密奖励事实上没起作用**
4. **skip-needed-charge 这类“该充不充”的直击型惩罚太弱**
5. **curriculum 在识别问题上是对的，但它本身不能替代行为学习信号**

所以出现了一个很典型的结果：

> 你调了很多东西，训练确实发生了变化，但变化主要体现在“表层分布和部分结果”上，而没有让“return 阶段该怎么走、如何稳定上桩”变成一等学习目标。

## 9. 观测假象与真实问题的区分

需要明确区分以下两类问题：

### 9.1 观测假象 / 次要问题

- dashboard 某些累计数口径曾经不一致，这会误导人工判断
- `planner_policy_divergence` 的统计定义可能存在时序/口径放大效应
- 部分旧报告基于旧 session `20260420-091040`，只能当历史背景，不能代表当前状态

### 9.2 真实影响训练效果的问题

- `battery_fail_rate` 高
- `zero_charge_battery_fail_rate` 高
- `return_stall_rate` 高
- `planner_policy_divergence_rate` 高
- `probe` 奖励完全不触发
- `planner_alignment` 在关键回充阶段缺乏有效覆盖
- curriculum 长期锁死在 warmup

这里尤其要强调：

> 即使把所有监控口径噪声都扣掉，当前回充行为质量差、zero-charge fail 高、planner/stall 长期越线这几个问题仍然成立。  
> 所以主问题是真实存在的，不是面板幻觉。

## 10. 主根因 Top 3

### Root Cause #1：planner 约束无法在关键回充阶段有效落到动作层

这是我认为最核心的根因。

原因：

- `planner_policy_divergence` 统计强，但 `planner_alignment_reward` 覆盖窄
- return/contract 阶段恰恰是最需要 planner 纠偏的时候
- 结果是“planner 被用来打分和卡门”，但没有被充分用来训练

直接后果：

- divergence 长期高位
- return 阶段动作推进差
- curriculum 长期判定 `planner + stall`

### Root Cause #2：充电相关 shaping 对“形成稳定回充闭环”的驱动力不足

具体表现：

- `probe` 完全不触发
- `skip_needed_charge_penalty` 太弱
- `charge_route_progress` / `discovery` 有作用，但不足以成为主导
- 主任务奖励更稳定、更强，导致模型更愿意继续清扫

直接后果：

- 模型更容易学成“继续扫 + 充电碰运气”的次优策略
- zero-charge battery fail 长期居高不下

### Root Cause #3：curriculum 正确识别了问题，但也把系统锁在了当前坏状态

这一条不是首因，但已经成为重要放大器。

具体表现：

- `stagnation_level = 3`
- `invalid_for_promotion = true`
- `degraded_mainline = true`

直接后果：

- 训练分布更保守
- 晋级被阻断
- 策略继续在同类经验中强化已有次优模式

## 11. 对现有几份报告的取舍

### 11.1 `SESSION_20260420_DIAGNOSTIC.md`

价值：

- 对 battery death 和 `REWARD_TOP` 有一些原始摘要价值

局限：

- 基于旧 session `20260420-091040`
- 过度乐观
- 主问题抓偏到了 coverage tangle

结论：

> 只能作为历史现象摘要，不能作为当前主判断依据。

### 11.2 `DEEP_PROBLEM_DIAGNOSIS_REPORT_20260420.md`

价值：

- 已经抓到了 planner/stall 与 reward 结构盲区
- 比前一份更接近根因

局限：

- 仍然基于旧 session `20260420-091040`
- 对 `planner_alignment` 缺失和 charging net 的归因略偏绝对

结论：

> 可以作为重要中间分析材料，但必须用最新 session 重新校正。

### 11.3 两份独立子代理报告

价值：

- Bohr 更准确地抓到了 reward 结构失配
- Euclid 更准确地抓到了 curriculum 锁死与指标链
- Euler 进一步确认了最新 40 局里 zero-charge fail 与 planner/stall 的集中性

结论：

> 三者结合后，已经能稳定支持本报告的主判断。

## 12. 最终结论

当前训练反复调整后仍然不理想，**不是因为没有训练、不是因为系统坏了、也不是因为单一监控口径问题**。真正的问题是：

> **策略没有学会“在 planning / return 阶段稳定地按规划形成回充闭环”，而奖励与教师信号又没有在关键阶段给出足够强、足够持续的纠偏，因此模型形成了一个“能继续清扫、能赢一部分局、但经常在回充阶段失败”的局部最优。curriculum 正确识别了这一点，并把系统锁在 warmup。**

如果只保留一句最关键的话：

> **当前主矛盾不是清扫能力，而是回充执行能力；不是有没有 planner，而是 planner 没有变成关键阶段的动作主导。**
