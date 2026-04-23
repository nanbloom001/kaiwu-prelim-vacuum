# 分支价值审计报告

> 日期：2026-04-22  
> 审计对象：`linux`、`linux-LTSPPO`、当前工作区（`linux-LTSPPO-charge-constraint` + 未提交 survival/CPS 改造）  
> 审计目标：围绕当前总目标，识别哪些改动是纯负面收益，哪些改动具有高潜力且应被保留。

## 1. 结论先行

当前三条线的关系可以概括为：

1. `linux`
   - 强项是较高的任务效率上限和较轻的系统复杂度
   - 弱项是信用归因、死亡尾部和规划一致性长期不稳
2. `linux-LTSPPO`
   - 强项是把长期规划、rich feature、多模式行为和 checkpoint/benchmark 体系真正搭起来
   - 弱项是 battery fail 仍然是主失败模式，整体 benchmark 还不够强
3. 当前工作区
   - 强项是生存相关控制、启动模式、比较口径、运行态观测能力显著增强
   - 弱项是 `charge/return` 控制栈显著过重，并持续呈现“fail 下降但 CPS 被长期压低”的高概率 trade-off

本次审计的主结论是：

> 当前系统里最值得保留的是 **LTSPPO 行为底座 + scratch/resume/compare/观测基础设施 + 有限度的 battery-fail economics**；最应判定为高优先级裁剪对象的是 **以显式 `contract` 优化为中心的重型 `charge/return` 控制栈，以及原始的重课程自动控流逻辑**。

## 2. 版本层次与证据摘要

### 2.1 `linux`

这个阶段最核心的效果证据来自 `v52/v53` 训练与诊断文档。

- `v52-step8500` 在方案文档中被记录为：
  - `AvgCS=1320`
  - `MinCS=871`
  - `CPS=0.895`
  - `Survival=0.818`
- `v53-robust3450` 归档 README 记录：
  - `Best Robust Score = 3450.2`
  - `Best Avg Score = 909.2`
  - `Episode 110: WR=0.85, avg_cs=848`
  - 随后在 `ep 150` 开始退化

这说明 `linux` 的核心特征是：

- 任务质量与 CPS 上限高
- 训练会阶段性达到较强表现
- 但死亡尾部、信用归因和训练后期退化问题明显

### 2.2 `linux-LTSPPO`

这个阶段最核心的效果证据来自 `v6-ltsppo` 与 `v6-geo` 模型归档和 benchmark 报告。

- `v6-ltsppo-ep188` README 记录：
  - `Win Rate = 47.5%`
  - `Avg Clean Score = 634.0`
  - 明确写明 battery fail 仍是主失败模式
- `v6-geo-bestmodel-1051` README 记录：
  - `best_robust_score = 3266.05`
  - `best_avg_score = 1154.65`
  - 强调硬分布 ceiling 更高、适合作为继续训练点
  - 同样明确写明 battery fail 仍然主导失败

这说明 `linux-LTSPPO` 的价值不在于“已经打穿结果”，而在于：

- 行为建模能力明显增强
- checkpoint 选择与 benchmark 更成熟
- 在 harder 分布上开始出现更高上限
- 但 survival/return timing 问题并没有被彻底解决

### 2.3 当前工作区

当前工作区必须看两类证据：

1. 文档与工具链
   - `SURVIVAL_CPS_STABILIZATION_PLAN_20260421_1903.md`
   - `compare_training_runs.py`
   - `run_training_phase.py`
   - `curriculum-lite`
2. 当前真实 run 趋势
   - 当前 active run `20260422-105212`
   - 最新 rolling 40 局：
     - `win_rate = 0.95`
     - `avg_clean_per_step = 0.4623`
     - `battery_fail_rate = 0.05`
     - `zero_charge_battery_fail_rate = 0.05`
     - `battery_positive_reward_rate = 0.0`
     - `mode_usage_contract ≈ 0.0`
     - `mode_usage_return ≈ 0.07`
     - `route_phase_return_stall_rate ≈ 0.29`
   - 同日中午窗口曾出现：
     - `battery_positive_reward_rate` 明显回升
     - `CPS` 约 `0.47 ~ 0.50`
     说明 economics 与 CPS 之间曾出现过明显摆动，但该形态不应和“最新状态”混写
   - fixed local 对比：
     - `global_40` 的 local CPS 为 `0.3330`
     - `global_80` 的 local CPS 为 `0.3451`
     - `global_120` 的 local CPS 为 `0.3836`
     - 这些低 CPS 数值是硬证据；脚本输出的状态标签可作为辅助描述，但不应单独作为本报告的唯一依据

这说明当前工作区的真实状态是：

- fail 与 zero-charge 比较容易被压下去
- 但 CPS 长期低于目标，且明显低于高效版本
- `contract` 基本贴地，`return` 不低但 stall 偏高
- 当前系统已经显露出“保 fail，不保 CPS”的稳定 trade-off

## 3. 能力包审计结论

下表中的“引入版本”按第一次成型出现的版本写，不严格等同于第一次提交文件的分支。

| 能力包 | 引入版本 | 审计结论 | 结论摘要 |
|---|---|---:|---|
| benchmark / 手工 checkpoint 留档 / model README 体系 | `linux` | `Keep` | 这是后续所有横向比较和回溯的证据底座，复杂度低、收益高。 |
| reward 分量追踪 / 死亡轨迹 / config 风险诊断 | `linux` | `Keep` | 明显提升可解释性，没有看到负面副作用。 |
| entropy floor / 训练稳定性补丁 | `linux` | `Keep but Simplify` | 曾经解决 entropy collapse，但不是当前主瓶颈，不宜继续扩大其角色。 |
| LTSPPO richer feature / 多模式行为结构 / planner-aware guidance | `linux-LTSPPO` | `Keep` | 这是当前系统行为表达能力的底座，尚无证据表明它本身是主负担。 |
| geometry-aware / multi-route planning 增强 | `linux-LTSPPO` | `Unproven` | 有 ceiling 提升与 harder-round 潜力，但尚未被当前 survival/CPS 主目标单独验证。 |
| checkpoint readiness / benchmark stability / resume 选择评分 | `linux-LTSPPO` | `Keep` | 对训练延续、模型留档和后续实验组织有明显正收益。 |
| 原始重型课程自动控流（promotion/demotion + adaptive profile） | `linux-LTSPPO` | `Pure Negative` | 对当前目标看，复杂度高且多次阻碍归因，已被 `curriculum-lite` 事实性替代。 |
| scratch/preload/startup mode 显式控制 | 当前线 | `Keep` | 当前实验链能快速重启、隔离、验证，离不开这一层。 |
| `curriculum-lite = observe-only + fixed-profile` | 当前线 | `Keep` | 已经证明能拿回节奏控制权和归因能力，当前不应回退。 |
| zero-charge 双指标定义修正 | 当前线 | `Keep` | 明显减少了误判风险，是当前 survival 诊断必需项。 |
| fixed compare + local_10/local_20 主判链 | 当前线 | `Keep` | 相比 rolling-only 或 prefix-only，当前口径更贴近你的决策需求。 |
| phase overlay + `run_training_phase.py` | 当前线 | `Keep` | 这是当前快速实验与定向切换的核心基础设施。 |
| battery-fail economics 强化（Objective Reset 系列） | 当前线 | `Keep but Simplify` | 已证明能压住 `battery_positive_reward_rate`，但继续加重会稳定压塌 CPS。应保留为底座，而不是主杠杆。 |
| 将 `contract` 作为显式优化目标的框架 | 当前线 | `Pure Negative` | 当前所有证据都显示：把 `mode_usage_contract` 拉高本身，并不能稳定换来更好的 survival/CPS。 |
| 多阈值 `charge/return` gate 栈 | 当前线 | `Pure Negative` | 阈值多、耦合高、调参面大，当前稳定制造 fail/CPS trade-off。 |
| charging local terrain 的碎片化 shaping 套件 | 当前线 | `Keep but Simplify` | 其中少数项有价值，但整体过碎，应收缩到少数核心项。 |
| route-phase teacher 扩展与复杂可靠度分支 | 当前线 | `Keep but Simplify` | 方向对，但当前在线证据链仍不完整，暂不能把它单独判成主负担。 |

## 4. 纯负面收益项

本节只列“可以明确判负”的项，不把“暂未证明有用”的项混进去。

### 4.1 原始重型课程自动控流

判定：`Pure Negative`

原因：

1. 对当前目标，它显著增加了训练分布的隐藏漂移
2. 使 run 内 stage/profile 自动变化，直接削弱归因能力
3. 当前你已经用 `curriculum-lite` 实际替代了它，并且没有看到回退需求

结论：

- 不应回归原始重课程主线
- 后续只保留观察功能，不恢复自动控流

### 4.2 以 `contract` 为中心的显式中间阶段优化

判定：`Pure Negative`

原因：

1. 当前工作区多轮实验都显示：
   - `contract` 经常几乎贴地
   - 一旦强行拉起，又伴随更高 stall 或更低 CPS
2. 当前 run `20260422-105212` 里：
   - 最新 rolling 40 局中 `mode_usage_contract` 近乎贴地
   - `mode_usage_return` 维持在不低水平
   - `route_phase_return_stall_rate` 仍偏高
   - `avg_clean_per_step` 仍明显低于目标
   这说明系统已经不再缺“回家的意愿”，而是缺“高效回家与继续清扫的平衡”
3. 继续围绕 `contract` 去优化，实际是在强化一个人为定义但不稳定的中间代理目标
4. 需要强调的是：这里判负的是“当前这版显式 `contract` 优化框架”，而不是否认任何形式的 pre-return readiness 都没有价值

结论：

- 下一阶段不应再把 `mode_usage_contract` 提升本身当成成功信号
- 更合理的是用连续的 `return_readiness` 替代显式 `contract` 阶段

### 4.3 多阈值 `charge/return` gate 栈

判定：`Pure Negative`

原因：

1. 当前 `_infer_mode()` 里，`contract/return` 已经同时依赖：
   - slack
   - battery ratio
   - recoverability
   - route pressure
   - charge margin
2. 你最近几轮已经反复验证：
   - tightening 一档：fail 侧改善，但 CPS 掉
   - rollback 一档：fail 反弹或 return 走不稳
   - economics-only：fail 更稳，但 CPS 依旧不回
3. 这说明它已经不是一个“调对就行”的低维参数块，而是一个复杂度已经超过收益的控制栈

结论：

- 下一阶段不应继续在这组 gate 上做细颗粒调参搜索
- 应直接收缩成更少阈值、更连续的 readiness 结构

## 5. 高潜力且值得保留的项

### 5.1 LTSPPO richer feature / planner-aware 行为底座

判定：`Keep`

原因：

1. `linux-LTSPPO` 相比 `linux`，并不是结果立刻全面更强，但明确带来了更强的行为表达能力和 harder-round ceiling 潜力
2. 当前没有证据表明 backbone、多头结构、planner-aware feature 本身是第一主因
3. 当前主问题更多出在“控制栈怎么驱动这些能力”，而不是“模型没有这些能力”

结论：

- 不建议直接回退到 `linux` 的轻底座
- 也不建议先砍主网络或多头结构

### 5.2 startup mode / phase overlay / scratch-resume 工具链

判定：`Keep`

原因：

1. 这是当前快速试验和定向验证的核心基础设施
2. `run_training_phase.py` + `s1_survival.env` 让实验可以按 phase 干净重启
3. 这层复杂度低、收益高，而且与当前问题诊断强相关

结论：

- 必须保留
- 后续无论简化控制栈还是回退实验，都应继续用这套入口

### 5.3 `curriculum-lite`

判定：`Keep`

原因：

1. 当前已经证明，课程复杂度本身会显著削弱归因能力
2. `curriculum-lite` 至少把这个变量从主问题里排除了
3. 当前 run 的主要问题已经不是样本分布被偷偷改掉，而是控制逻辑本身在起作用

结论：

- 保留 lite
- 不要在下一轮把课程重新变成主变量

### 5.4 zero-charge 双指标 + fixed local compare

判定：`Keep`

原因：

1. 这两组改动显著减少了口径误判
2. 当前很多判断之所以能更清楚，就是因为：
   - `zero_charge_battery_fail_rate` 与 `zero_charge_among_battery_fail_rate` 已分离
   - `local_10/local_20` 和 rolling 的职责已拆开
3. 它们属于观测与决策底座，不是当前复杂度主负担

结论：

- 保留，不回退
- 后续继续围绕这套统一口径做实验结论

### 5.5 battery-fail economics 作为底座

判定：`Keep but Simplify`

原因：

1. Objective Reset 线证明了：不处理 `battery_positive_reward_rate`，系统会稳定学到坏局部最优
2. 但也同样证明了：继续加大 terminal/fail 经济学，会很容易把 CPS 压住
3. 当前它的正确定位不是“继续调”，而是“作为底座冻结”

结论：

- 保留一个中等强度版本
- 从主修复杠杆降级成基础约束

## 6. 当前工作区中潜力大但必须收缩的项

### 6.1 charging local terrain

判定：`Keep but Simplify`

保留潜力：

- `skip_needed_charge` 惩罚
- `return_progress` 正向 shaping
- 一个必要的 stall 惩罚

应删减部分：

- discovery/probe 类零碎 bonus
- 过于细碎的局部地形修补项

原因：

- 当前不是完全不需要 charging shaping，而是 shaping 组件太多、太碎，形成了高度耦合的 reward 地形

### 6.2 route-phase teacher / direct guidance

判定：`Keep but Simplify`

保留潜力：

- 当前结果说明，单靠 gate 和 economics 不够，return-path 仍需要直接行为约束

必须简化的原因：

- 当前 teacher 触发依赖太多可靠度条件
- 多分支 mask 逻辑增加了复杂度，但当前 learner log / runtime state 里的在线证据链还不足以证明它是第一主因

结论：

- 保留“return 场景下需要更直接主动作监督”这个思想
- 但后续应改成更少条件、更稳定的监督结构
- 在 teacher 指标链补齐之前，不应把它单独当成下一轮唯一主变量

## 7. 下一阶段建议保留的最小系统

这是本次审计最重要的产物：下一轮如果你要做“控制栈简化 v1”，建议保留的系统应当是下面这套，而不是完整继承当前复杂结构。

### 7.1 直接保留

1. LTSPPO 主网络 / richer feature / planner-aware 行为底座
2. startup mode / phase overlay / scratch-resume 工具链
3. `curriculum-lite`
4. zero-charge 双指标
5. fixed local compare
6. 中等强度的 battery-fail economics 底座

### 7.2 简化后保留

1. `charge/return` 结构
   - 从 `harvest -> contract -> return`
   - 收缩成更连续的 `harvest -> return_readiness -> return`
2. charging reward 地形
   - 收缩到：
     - battery fail 主经济学
     - `skip_needed_charge`
     - `return_progress`
     - 一个 stall 惩罚
3. return-path supervision
   - 保留 direct guidance 思想
   - 但减少条件分支和可靠度分叉

### 7.3 直接冻结或剔除

1. 原始重课程自动控流
2. 以 `contract` 为中心的显式阶段优化
3. 多阈值 `charge/return` gate 栈的继续微调
4. 将 `mode_usage_contract` 本身当成优化目标

## 8. 最终判断

这次审计之后，可以明确形成以下判断：

1. `linux` 的价值在于：
   - 它证明了轻系统可以打出更高 CPS 和更高任务效率上限
   - 但缺乏长期稳定性和更强行为建模
2. `linux-LTSPPO` 的价值在于：
   - 它提供了当前仍然值得保留的行为与表示底座
   - 不应被误判成“复杂就是坏”
3. 当前工作区的真正问题不在主网络，而在：
   - 叠加在 LTSPPO 之上的 `charge/return` 控制栈太复杂
   - 已经持续制造出“保 fail，不保 CPS”的高概率 trade-off

因此，下一阶段最值得做的不是：

- 回退到 `linux`
- 继续在当前控制栈里调阈值
- 立刻先砍主网络

而是：

> **保留 LTSPPO 行为底座和当前工具链，优先清理当前分支新增的 `charge/return` 重控制栈。**

## 9. 审计边界与证据强度说明

为避免把方向判断写成“已经被完全证明的绝对结论”，本报告补充以下边界：

1. `global_40/80/120` 的低 CPS 数值是硬证据；compare 脚本给出的状态标签可参考，但不是唯一依据。
2. 当前工作区关于 `battery_positive_reward_rate` 的判断必须区分：
   - 同日中午某段窗口的高位摆动
   - 最新 rolling 40 局的低位状态
   报告中的 economics 结论以“曾出现摆动 + 最新仍未带来高 CPS”为依据，而不是单点窗口。
3. `contract` 被判负，针对的是：
   - 当前显式阶段定义
   - 围绕 `mode_usage_contract` 提升本身做优化
   不等于否认一个更薄、更连续的 readiness 层可能有价值。
4. route-phase teacher 当前属于“需要简化且需补证据链”的项，而不是已经被完全证实的主负担。
