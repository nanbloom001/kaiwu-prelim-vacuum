# v6 统一架构重设计报告

日期：2026-04-16

定位：

- 这不是“下一轮小修小补建议”，而是一份面向 `v6` 的正式重设计报告
- 目标是：在当前 PPO / KaiwuDRL 基础设施内，做一次**明确的大改模型**
- 这份文档不再维持“尽量少动网络”的保守前提，而是重新回答：
  - 这个赛题究竟需要什么样的策略结构
  - 哪些组件必须一次性补齐
  - 哪些复杂度必须克制，避免把系统推到训不起来

主要依据：

- 官方任务建模：
  - `train/context/analysis/OFFICIAL_TASK_MODELING_20260416.md`
- 外部三份独立方案：
  - `train/context/analysis/external_ai/Gpt.md`
  - `train/context/analysis/external_ai/Gemini.md`
  - `train/context/analysis/external_ai/Opus.md`
- 当前统一评估结果：
  - `train/context/benchmark/MANUAL_CHECKPOINT_EVAL_REPORT_20260416.md`
- 现有实现：
  - `code/agent_ppo/model/model.py`
  - `code/agent_ppo/feature/preprocessor.py`
  - `code/agent_ppo/feature/expert.py`
  - `code/agent_ppo/algorithm/algorithm.py`
  - `code/agent_ppo/agent.py`
  - `code/agent_ppo/conf/conf.py`

---

## 0. 快速结论

### 0.1 一句话结论

当前任务的主瓶颈已经明确不是“卷积不够深”或“奖励少调一点”，而是：

- 任务本质需要**阶段切换**
- 决策本质需要**时序记忆**
- 回充问题本质需要**目标选择 + 执行保持**
- 当前网络却仍然是**单帧前馈 + 单动作头 + 单 critic**

所以 `v6` 不应再沿用“当前骨架上再补几个通道”的路线，而应升级为：

> **结构化时序策略网络**  
> 即：`多分支观察编码 + GRU 记忆核心 + 模式头 + 充电目标头 + 条件动作头 + 多价值头`

### 0.2 最终推荐结构

我对 `v6` 的最终推荐是：

`V6 Structured Recurrent Mode-and-Target Actor-Critic`

简称：

`V6-SRMTA`

这套结构的核心不是“更大”，而是把过去分散在 reward / Expert / scalar / heuristic 里的关键决策对象显式化：

- `mode`
  - 当前到底是在 `clean / prepare_return / return / evade`
- `target`
  - 当前回充时的目标充电桩是谁
- `memory`
  - 当前阶段已经持续了多久、风险趋势如何、最近是不是在绕圈
- `value decomposition`
  - 清扫价值和生存价值不再强行挤在一个 critic 里

### 0.3 这次采纳了哪些专家方案

| 来源 | 采纳的核心思想 | 在 v6 中如何落地 |
|---|---|---|
| `Gpt.md` | 任务是“带生存约束的多目标 POMDP”，问题在结构错位，不在单点超参 | 直接采用“阶段 + 时序 + 价值分解”的主框架 |
| `Gemini.md` | 不要直接上 full attention / full hierarchy；先解决状态表示、模式、记忆 | 采用 GRU + 软分层，而不是 full transformer / hard option |
| `Opus.md` | A/B/C/D 四类改动里，A 的目标修正、B 的信息带宽、C 的 GRU、D 的 mode conditioning 都有价值 | 选取 A+B+C+D 的核心部件，做成一套统一架构 |

### 0.4 这次不采纳什么

| 不采纳项 | 原因 |
|---|---|
| 继续沿用当前 `CNN + MLP + 单 critic` 主骨架 | 已证明提升空间开始递减，且主失败模式不再匹配这个结构 |
| full transformer 全替换 | 训练成本和调试成本过高，不是当前资源下的最优平衡点 |
| full hierarchical RL / option framework | 工程复杂度过高，信用分配更难，当前基础设施不适合直接跳这一步 |
| 完全移除 Expert | 现阶段不现实，硬安全和紧急回充仍需要安全网 |

---

## 1. 为什么现在应当明确“大改模型”

### 1.1 来自任务建模的结论

基于 `OFFICIAL_TASK_MODELING_20260416.md`，这个赛题有 4 个关键事实：

| 事实 | 含义 |
|---|---|
| `128×128` 大图，局部 `21×21` 观测 | 单帧局部图不可能天然包含全局决策所需信息 |
| 电池耗尽与 NPC 碰撞都是硬失败 | 生存价值不是附属项，而是主约束 |
| 充电是“路径问题 + 时机问题”，不是“站桩问题” | 决策关键在何时切回、回哪个桩、如何持续执行 |
| 官方机器人会持续抢污渍并制造风险 | 任务不是单纯 coverage，而是带弱对抗的机会竞争 |

因此这不是一个“看当前局部图 -> 输出方向”的普通反应式控制问题，而是一个：

- 有全局结构
- 有长期预算
- 有显式阶段
- 有弱对抗竞争
- 有部分可观测性

的资源受限 POMDP。

### 1.2 来自实验的结论

基于 `MANUAL_CHECKPOINT_EVAL_REPORT_20260416.md`，当前版本线已经说明：

| 结论 | 证据 |
|---|---|
| 训练确实能学到东西 | `v5 -> v51 -> v52-step10000` 有明显提升 |
| 提升不是单调的 | `v52-step10000 -> v52-step70000` 明显回退 |
| 碰撞可以被压下去 | `v53-robust3450` 的 collision 已降到很低 |
| battery 仍未根治 | 最优点仍有大量 battery fail |
| 主失败模式是“回得太晚”而不是“完全不会回” | 失败局普遍出现 `charge_count=0`、首次 return 太晚、`charger_slack < 0` |

这说明当前系统的问题不是“行为完全随机”，而是：

- 能学局部清扫
- 能靠 Expert 压短时风险
- 但不会稳定地把**中长期回充阶段**学成内部能力

### 1.3 来自三份专家分析的共识

三份分析虽然表述不同，但共识非常高：

| 共识 | 来源 |
|---|---|
| 主瓶颈是 battery / return timing，不再是 collision | `Gpt.md`、`Gemini.md`、`Opus.md` |
| 当前问题是“任务目标、训练信号、状态表示、Expert 接口”多处错位 | `Gpt.md`、`Gemini.md` |
| 继续只调 reward 系数收益递减 | 三份都同意 |
| 需要更高分辨率全局表示、visit/revisit 建模、时序记忆 | 三份都同意 |
| 最值得补的是 GRU/阶段表达，而不是一上来 full attention | `Gpt.md`、`Gemini.md` |

所以此时继续保守迭代当前骨架，已经不是“稳”，而是“继续在错误表示上加码”。

---

## 2. 重新定义 v6 的设计目标

### 2.1 v6 不是“更大的清扫网络”

`v6` 的目标不应表述为：

- 更大的 CNN
- 更多通道
- 更多特征
- 更高的 logits 质量

而应表述为：

> 构建一个能在单策略 PPO 框架内，同时表达  
> `阶段判断`、`回充目标选择`、`时序记忆`、`局部执行`、`价值分解`  
> 的结构化策略网络。

### 2.2 v6 必须满足的 7 个条件

| 条件 | 说明 |
|---|---|
| 1. 全局带宽要提升 | `8×8` 过粗，无法稳定表达收益版图与回充代价场 |
| 2. 局部图要显式包含 revisit / return 信息 | 否则继续靠网络自己猜阶段 |
| 3. 必须有内部记忆 | return 执行和 risk trend 不能继续外包给 Expert 状态机 |
| 4. 必须显式表达“当前阶段” | clean / prepare_return / return / evade 不是同质动作选择 |
| 5. 必须显式表达“当前回哪个桩” | 仅靠 8 方向动作难以学出稳定回充计划 |
| 6. 必须把清扫价值和生存价值拆开 | 单 critic 会继续被清扫信号淹没 |
| 7. 复杂度必须可训练 | 不能直接引入 full transformer + hard hierarchy + 全序列 PPO 的三重风险 |

### 2.3 v6 的定位

| 方案类型 | 是否采用 |
|---|---|
| 小修补版本 | 否 |
| 极端前沿版本 | 否 |
| 有明显结构升级、但仍能落到现有训练栈 | 是 |

换句话说，`v6` 应该是：

- **大改模型**
- 但不是“推翻 PPO / Kaiwu / Expert / benchmark 的全部基础设施”

---

## 3. v6 的统一设计思路

### 3.1 先把策略拆成 3 层能力

从任务本体出发，策略实际上包含 3 层：

| 层级 | 作用 | 当前系统主要靠谁承担 |
|---|---|---|
| 生存评估层 | 判断现在还能不能继续扫 | reward / heuristic / scalar 混合承担 |
| 阶段与目标层 | 判断当前 mode，决定是否回充、回哪个桩 | Expert 主导 |
| 局部执行层 | 给定当前 mode 和目标，输出 8 方向动作 | 网络主导 |

当前网络只真正在学第 3 层，这就是错位来源。

### 3.2 v6 的核心改法

v6 要做的不是“把 Expert 的东西全塞回 reward”，而是：

- 把第 1 层和第 2 层显式地放进网络结构
- 保留 Expert 只做边界安全网
- 让 policy 真正学到：
  - 什么时候该回
  - 回哪个桩
  - 在 return 过程中如何保持决策一致

### 3.3 为什么要加“目标头”

过去讨论里更常提的是 `mode head`，但基于现在的任务理解，我认为仅有 `mode head` 还不够。

原因很简单：

- `return` 不是一个完整动作
- 它至少还包含：
  - 回哪一个 charger
  - 是否坚持执行当前 return 计划

所以 `v6` 应增加一个显式的：

- `charger target head`

这样网络内部就不再只是“知道该回了”，而是“知道回哪个桩”。

---

## 4. v6 统一观察设计

这部分不是列“还能再喂什么特征”，而是重新按任务对象组织观察。

### 4.1 总体结构

v6 观察由 5 个分支组成：

| 分支 | 目标 |
|---|---|
| `Local Spatial` | 解决局部执行、局部风险、局部 return guidance |
| `Global Spatial` | 解决全局收益版图与回充代价场 |
| `Entity Set` | 解决 NPC / Charger 的关系建模 |
| `Scalar + Short History` | 解决阶段摘要和趋势摘要 |
| `Action History` | 解决振荡、卡住、回充执行一致性 |

### 4.2 Local Spatial：`21×21×8`

推荐 8 个通道：

| 通道 | 含义 | 目的 |
|---|---|---|
| 1 | `obstacle_or_passable` | 局部拓扑基础 |
| 2 | `dirty_mask` | 当前直接收益 |
| 3 | `charger_mask` | 本地可见桩位置 |
| 4 | `npc_danger` | 空间化的短时风险 |
| 5 | `visit_heat` | 抑制 revisit |
| 6 | `trajectory_heat` | 抑制局部震荡与来回折返 |
| 7 | `frontier_mask` | 区分新区域和低价值旧区域 |
| 8 | `return_guidance` | 指向当前选定 charger 的局部引导势场 |

说明：

- 这里保留了当前 heatmap 路线的正确方向
- 但把它从“一个辅助通道”升级为与 `visit` / `return` 配套的局部结构

### 4.3 Global Spatial：`16×16×7`

推荐 7 个通道：

| 通道 | 含义 | 目的 |
|---|---|---|
| 1 | `explored_ratio` | 哪些区域已经看过 |
| 2 | `dirty_density` | 哪些区域仍值得去 |
| 3 | `visit_density` | 哪些区域已过度访问 |
| 4 | `charger_presence` | 桩在粗粒度空间中的分布 |
| 5 | `npc_risk_density` | 竞争和风险热点 |
| 6 | `competition_loss_density` | 被 NPC 抢收益更快的区域 |
| 7 | `return_cost_field` | 到最近 charger 的粗粒度可达代价 |

这一步是这次重设计的关键之一：

- `8×8 -> 16×16`
- 不是为了更“高清”
- 而是为了让全局图真正有资格参与决策，而不是只当一个粗提示

### 4.4 Entity Set：`NPC + Charger` 显式实体分支

实体分支建议固定编码：

- `NPC top-4`
- `Charger top-4`

每个实体建议 10 到 12 维：

| 实体特征 | 含义 |
|---|---|
| `dx, dz` | 相对位移 |
| `dist_chebyshev` | 几何距离 |
| `dist_astar_or_reachable_cost` | 更接近真实可达代价 |
| `bearing_sin, bearing_cos` | 方向 |
| `risk_or_priority` | NPC 风险或 charger 优先级 |
| `is_nearest` | 是否当前最近目标 |
| `in_local_view` | 是否在视野内 |
| `last_seen_delta` | 距离最近一次精确观测过去了多久 |

这一步的作用是：

- 不再把 charger/NPC 关系全部散在 scalar 里
- 让网络明确处理“对象关系”，尤其是：
  - 哪个 charger 更合适
  - 哪个 NPC 更危险

### 4.5 Scalar + Short History：目标维度 `96-112D`

建议保留原始关键 scalar，并新增趋势摘要：

| 特征类别 | 具体建议 |
|---|---|
| 电量与步数 | `battery_ratio`、`remaining_step_ratio`、`time_since_last_charge` |
| 清扫趋势 | `clean_rate_ema`、`score_rate_ema`、`steps_since_last_new_area` |
| 回充摘要 | `charger_slack(astar)`、`nearest_charger_astar`、`estimated_return_margin` |
| 风险摘要 | `nearest_npc_dist`、`npc_risk_summary`、`recent_collision_margin_min` |
| 行为稳定性 | `recent_invalid_move_ratio`、`recent_oscillation_entropy`、`recent_revisit_ratio` |
| 阶段摘要 | `mode_duration`、`on_charger_flag`、`recent_charge_count`、`recent_return_abort_count` |

原则是：

- 不把网络能自己学的东西重复手工编码太多
- 但必须把“趋势”和“阶段摘要”补齐

### 4.6 Action History：`last action + 4-step summary`

推荐保留：

- `last_action_onehot`
- `recent_action_histogram(4 steps)`

这是低成本高价值特征，能帮助网络判断：

- 当前是否在振荡
- 当前是否在 return 过程中不断修正
- 当前是否处于“想走但走不出去”的局部模式

---

## 5. v6 统一网络结构

### 5.1 总体结构图

```text
Local 21x21x8   -> Local ResCNN      -> 256
Global 16x16x7  -> Global ResCNN     -> 160
Entity Set      -> Shared MLP + Pool -> 128
Scalar+History  -> MLP               -> 128
Action History  -> MLP               ->  32

Fuse -> FC -> FC -> GRU(256)
     -> shared recurrent trunk
        -> mode head (4)
        -> charger target head (5 = none + 4 chargers)
        -> mode-and-target conditioned action head (8)
        -> value_main
        -> value_survival
        -> value_clean
        -> auxiliary heads
```

### 5.2 为什么是这个结构

| 组件 | 必要性 |
|---|---|
| 多分支编码器 | 任务对象本来就是局部图 / 全局图 / 实体 / 摘要混合，不应继续强行压成单路 MLP |
| GRU | 当前最缺的就是内部时序状态 |
| mode head | 当前策略缺显式阶段表达 |
| charger target head | 当前 return 缺“目标选择”这一层 |
| conditioned action head | 动作语义在不同模式下不同 |
| 多价值头 | 清扫价值和生存价值必须拆开 |

---

## 6. 各编码器设计

### 6.1 Local Encoder

推荐：

- 轻量 residual CNN
- 输出 `256D`

建议结构：

| 层 | 规格 |
|---|---|
| stem | `8 -> 32` conv |
| block1 | residual block × 2 |
| downsample | stride 2 |
| block2 | residual block × 2 |
| head | GAP + FC -> `256` |

它要负责的不是“记地图”，而是：

- 局部 dirt / charger / danger / revisit / guidance 的联合解释

### 6.2 Global Encoder

推荐：

- 中等深度 residual CNN
- 输出 `160D`

建议结构：

| 层 | 规格 |
|---|---|
| stem | `7 -> 24` conv |
| block1 | residual block × 2 |
| block2 | residual block × 2 |
| head | GAP + FC -> `160` |

它不是主执行层，但它要能表达：

- 哪些区域还值得去
- 哪些区域已经过度访问
- 哪些区域在低电时已不值得继续深入

### 6.3 Entity Encoder

推荐：

- 每个实体走共享 MLP
- NPC / Charger 分别池化
- 再拼接成 `128D`

原因：

- 实体数量小
- 对象类型明确
- 不需要上 full self-attention

但要显式保留：

- `npc_summary`
- `charger_summary`

因为两者语义完全不同：

- 一个是风险对象
- 一个是回充对象

### 6.4 Scalar Encoder

建议：

- 两层 MLP
- 输出 `128D`

注意重点不在深度，而在于：

- 这里要包含“趋势摘要”和“阶段摘要”

### 6.5 Action History Encoder

建议：

- 一层小 MLP
- 输出 `32D`

它不是大头，但对抑制振荡非常有价值。

---

## 7. 时序核心：GRU

### 7.1 为什么必须是 GRU

从当前任务看，最缺失的不是“更强的空间混合”，而是：

- 回充计划是否已开始
- 当前 return 是否已经执行了若干步
- 最近风险趋势是否在升高
- 最近是否一直在重复无效动作

这些都不是单帧能稳定表示的。

### 7.2 为什么选 GRU，不选 LSTM / Transformer

| 方案 | 判断 |
|---|---|
| GRU | 最平衡，足够表达短中程状态，又不把实现复杂度推太高 |
| LSTM | 可以，但额外门控开销和实现复杂度不一定换来对应收益 |
| full transformer | 过重，而且当前输入并不是天然高质量长序列 |

### 7.3 推荐规格

建议：

- 单层 `GRU`
- hidden size `256`

为什么比上一版建议更大：

- 这次是明确大改模型
- 又同时承载 mode、target、局部执行和价值分解
- `192` 偏保守，`256` 更合适

### 7.4 训练接口建议

v6 不应继续停留在“完全单步样本思维”。

建议最小升级为：

- rollout chunk length: `16` 或 `32`
- 每个 env 持有 GRU hidden state
- learner 端做 truncated BPTT

这是 `v6` 能否真正学到阶段持续性的关键接口之一。

---

## 8. 决策头设计

### 8.1 Mode Head：4 类

推荐 mode：

| mode | 含义 |
|---|---|
| `clean` | 电量安全，主要目标是效率清扫 |
| `prepare_return` | 继续扫已经开始侵蚀安全边际 |
| `return` | 已明确进入回桩执行 |
| `evade` | 局部高风险，短期以保命为主 |

为什么要有 `prepare_return`：

- 当前系统最大的失败正是“清扫 -> return”切换太晚
- 如果只有 `clean` 和 `return`，模型仍然容易在边界点抖动

### 8.2 Charger Target Head：`5 类`

输出：

- `none`
- `charger_1`
- `charger_2`
- `charger_3`
- `charger_4`

说明：

- 在 `clean` 模式下通常偏向 `none`
- 在 `prepare_return / return` 中必须选出目标 charger

这是 v6 和上一版建议的最大区别之一。  
我现在认为，不补这层，mode 仍然会太抽象。

### 8.3 Action Head：Mode-and-Target Conditioned

动作空间仍然是 8 方向，不改。

但推荐做成：

- 共享 action trunk
- 拼接 `mode embedding`
- 拼接 `selected charger embedding / target summary`
- 最终输出 8 个 logits

这样可以实现：

- `clean` 时动作偏收益
- `return` 时动作偏路径执行
- `evade` 时动作偏安全

### 8.4 为什么不用 4 套完全独立 actor head

不建议直接上 4 套独立 actor，因为：

- 参数量膨胀快
- 模式样本分布不均
- 训练更容易不稳定

条件化比完全分叉更平衡。

---

## 9. 价值头与辅助头设计

### 9.1 多价值头

建议保留 3 个价值头：

| 价值头 | 用途 |
|---|---|
| `value_main` | PPO 主损失使用 |
| `value_survival` | 学习生存/回充相关回报 |
| `value_clean` | 学习清扫/效率相关回报 |

### 9.2 为什么必须拆

当前系统的核心问题之一就是：

- 单 critic 同时回归“多扫一格”和“50-200 步后不死”

这在信号量级和时间尺度上都不合理。

拆分的目标不是多任务花活，而是让 trunk 真的能学到：

- 生存边际
- 清扫效率

是两类不同价值。

### 9.3 辅助头

建议保留 4 个高价值辅助头：

| 辅助头 | 含义 |
|---|---|
| `return_success_head` | 现在进入 return 后能否安全到桩 |
| `collision_risk_head` | 未来短窗口碰撞风险 |
| `battery_delta_head` | 近未来电量变化趋势 |
| `coverage_gain_head` | 近未来新增清扫收益 |

这些头的作用是把 trunk 拉向：

- 和真实任务结构一致的表征

而不是单纯让 PPO 自己从主损失里慢慢猜。

---

## 10. Expert 的重新定位

### 10.1 不移除，但彻底换职责

v6 中 Expert 不应再是：

- 常态下的隐形 mode controller

而应退回到：

| 职责 | 是否保留 |
|---|---|
| NPC 硬安全过滤 | 保留 |
| 极端低电硬接管 | 保留 |
| 常态非紧急回充主导 | 不保留 |
| 长期弱 bias 牵引 | 大幅削弱或仅早期保留 |

### 10.2 Expert 在 v6 中更像什么

更合理的定位是：

- safety shield
- emergency fallback
- early-training teacher

而不是“策略主体”。

### 10.3 建议增加的 teacher signal

为了降低大改模型的训练难度，建议早期加入弱监督：

| teacher signal | 来源 |
|---|---|
| `mode pseudo label` | Expert 当前状态机 |
| `charger target pseudo label` | Expert 的当前 charger 选择 |

注意：

- 这不是永久 imitation
- 而是早期 bootstrap，后续逐步退火到 0

这样做的目的，是避免大模型在初期把 mode/target 空间学塌。

---

## 11. 训练设计建议

### 11.1 PPO 仍保留，但接口必须升级

我不建议为了 v6 一次性改算法到 SAC / IMPALA / 完整 options。

当前更合理的是：

- 仍保留 PPO 主框架
- 但把数据接口升级为 sequence-aware PPO

### 11.2 推荐训练接口

| 项目 | 建议 |
|---|---|
| rollout 组织 | sequence chunk |
| GRU hidden state | env 内维护，episode reset 清零 |
| learner 训练 | truncated BPTT |
| 主 value | `value_main` |
| 辅助 loss | `value_survival`、`value_clean`、aux heads |
| mode/target bootstrap | 早期 teacher loss，后期退火 |

### 11.3 reward 方面的立场

虽然这份文档是“大改模型”方案，但我不建议把 `A` 类改动丢掉。

换句话说：

- `v6` 不是“网络升级后就不需要 reward / Expert 接口修正”

相反，应该继承已经验证有效的那部分：

- A* potential shaping
- 充电相关 signal 统一
- expert 只做边界安全网

否则大模型只会更快地把错误目标学得更好。

### 11.4 训练难度控制

为了平衡“表达能力 / 上限 / 训练难度”，`v6` 需要克制以下事情：

| 不做的事 | 原因 |
|---|---|
| 不一次性上 full transformer | 风险太高 |
| 不做 full hierarchical RL | 调试复杂度太高 |
| 不做 4 套完全独立策略头 | 模式不均衡，容易训崩 |
| 不彻底移除 Expert | 当前不现实 |

---

## 12. 为什么这套结构是当前最合理的平衡点

### 12.1 相比当前结构，它补齐了什么

| 当前缺口 | v6 对应补法 |
|---|---|
| 全局太粗 | `16×16×7 global` |
| revisit / orbiting 只靠弱特征 | `visit + trajectory + action history` |
| 无内部记忆 | `GRU(256)` |
| 无阶段表达 | `mode head` |
| 无回充目标选择 | `charger target head` |
| 单 critic 被清扫信号淹没 | `main + survival + clean` 三头 |

### 12.2 相比 full transformer，它保留了什么优势

| 优势 | 说明 |
|---|---|
| 更稳 | CNN 仍保留空间归纳偏置 |
| 更便宜 | 参数量和训练代价更可控 |
| 更容易落地 | 和现有代码风格、训练栈更连续 |

### 12.3 相比继续修当前骨架，它的本质优势

当前骨架的问题不是“缺几个特征”，而是：

- 决策层级错位
- 时序缺位
- 价值分解缺位

v6 正是对这三件事一次性补齐。

---

## 13. 建议的 v6 正式定义

### 13.1 v6 架构摘要

| 模块 | 规格 |
|---|---|
| Local | `21×21×8` |
| Global | `16×16×7` |
| Entity | `NPC top-4 + Charger top-4` |
| Scalar | `96-112D` |
| Action history | `last action + 4-step summary` |
| Local encoder | ResCNN -> `256` |
| Global encoder | ResCNN -> `160` |
| Entity encoder | shared MLP + pool -> `128` |
| Scalar encoder | MLP -> `128` |
| Action-history encoder | MLP -> `32` |
| Recurrent core | `GRU(256)` |
| Decision heads | `mode(4) + charger_target(5) + action(8)` |
| Value heads | `main + survival + clean` |
| Aux heads | `return_success + collision_risk + battery_delta + coverage_gain` |

### 13.2 预期收益

| 问题 | 预期改善 |
|---|---|
| battery fail | 明显下降，尤其是“回得太晚”的类型 |
| collision | 维持在低位，不因模型更复杂而回退 |
| revisit / orbiting | 明显下降 |
| checkpoint 波动 | 应比当前单 critic 前馈结构更小 |
| hard profile 稳定性 | 应明显优于当前版本 |

### 13.3 主要风险

| 风险 | 说明 |
|---|---|
| 训练接口升级成本 | sequence-aware PPO 需要改 learner/rollout 打包 |
| mode/target 学习初期不稳 | 需要 teacher bootstrap |
| 多头 loss 配平 | 需要控制 loss 权重 |
| 完全不兼容现有 checkpoint | 需要接受 `v6` 从新线重训 |

---

## 14. 最终建议

### 14.1 是否继续在当前网络上修

不建议把当前网络继续作为下一代主线。

原因不是它“完全没用”，而是：

- 它更适合继续作为旧线 baseline
- 不适合继续承担下一代能力上限

### 14.2 是否另起炉灶

是，但不是另起整个训练系统的炉灶，而是：

- 保留 PPO / Kaiwu / benchmark / Expert 基础设施
- 重写 observation 组织、model 骨架和训练接口

### 14.3 我对 v6 的最终判断

如果你现在明确希望：

- 不再沿用当前前馈骨架
- 一步到位补齐真正需要的组件
- 但又不想直接跳进 full transformer / full hierarchy 的高风险区

那么这份报告给出的 `V6-SRMTA` 就是我现在最推荐的折中点：

- 结构上已经是明显换代
- 任务表达上已经对准当前主瓶颈
- 工程上仍然能在你现有训练系统里落地

---

## 15. 下一步实施顺序

如果后续按这份 v6 报告落代码，我建议顺序是：

1. 先改 observation schema 和 `preprocessor.py`
2. 再改 `model.py`，先搭起多分支编码器 + GRU + heads skeleton
3. 再改 `agent.py` / sample packing，把 GRU state、mode/target 输出接通
4. 再改 `algorithm.py`，接多 value / aux losses / teacher bootstrap
5. 最后调 `expert.py`，把 Expert 角色收缩到边界安全网

原因很简单：

- v6 的关键不是某一层网络，而是“观察、状态、决策头、训练接口”四者一起重新对齐

这才是这次大改模型真正要做的事。
