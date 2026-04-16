# v6 综合收敛方案：长期主线架构建议

日期：2026-04-16

目标：

- 不是继续做边际优化
- 也不是在两份 v6 文档之间二选一站队
- 而是吸收：
  - 官方赛题建模
  - 当前实现与统一 benchmark 结果
  - `external_ai/` 三份分析
  - `V6_DEFINITIVE_ARCHITECTURE_20260416.md`
  - `UNIFIED_TASK_MODELING_AND_V6_ARCHITECTURE_20260416.md`
- 最终收敛成一份更适合作为**长期主线**的 v6 方案

这份文档的立场是：

- 允许明显的大改模型
- 接受一定训练与工程成本
- 不接受收益明显边际化的小修补路线
- 目标是得到一套：
  - 能比较快追平当前成绩
  - 后续可以持续 resume
  - 上限明显高于当前前馈骨架
  - 不会很快再次推翻重来

---

## 0. 最终结论

### 0.1 是否应该一步到位

结论：**应该一步到位做主线重构，但不能无节制地把所有“看起来先进”的组件都塞进去。**

当前已经可以明确排除两条路：

1. **继续修当前前馈骨架**
   - 已经出现明显边际递减
   - battery / return timing 仍是主瓶颈
   - 再做 reward / bias 微调，更多是在减少退化，而不是提升上限

2. **直接做 full transformer / full hierarchy**
   - 表达能力更强，但当前工程、训练接口和训练预算并不支持这样一步跳太远
   - 过高复杂度会让第一版 v6 很难训稳，也很难定位问题

因此最合理的路线不是“保守继续修”，也不是“极端前沿全都上”，而是：

> **一步到位做结构化重构，但控制在当前 PPO / Kaiwu 基础设施还能稳落地的复杂度里。**

### 0.2 最终推荐框架

我最终推荐的新主线是：

`V6-LTSPPO`

全称：

`Long-Term Structured Pointer Recurrent Dual-Critic PPO`

这是一个吸收两边优点后的折中点：

- 保留我方案里的：
  - 任务结构显式化
  - mode / target / entity / history 的建模思想
- 吸收新文档里的：
  - GRU + Dual Critic 作为最核心、最先必须落地的主轴
  - 对工程复杂度的克制
  - 对 Expert 充电决策强依赖的明确切断

但最终不完全采用任何一边的极端版本。

---

## 1. 为什么不直接采用任一旧版本

### 1.1 为什么不直接采用“极简 GRU + Dual Critic”版

`V6_DEFINITIVE_ARCHITECTURE_20260416.md` 的优点很明显：

- 更务实
- 更像一版第一时间能开工的实现
- 明确抓住了：
  - `GRU`
  - `Dual Critic`
  - `16×16 Global`
  - `Expert 充电控制权下放`

但它也有 3 个长期问题：

1. **阶段与目标仍然过于隐式**
   - 它希望 GRU 自己隐式学出 mode 和 target
   - 这对“尽快起量”是有利的
   - 但对“长期主线的结构上限”不够稳

2. **观察改造偏保守**
   - 没有真正把 entity / history / return guidance 独立成结构对象
   - 更像“当前 observation 的增强版”，不是“任务对象重构版”

3. **它对 recurrent training 的接口还不够彻底**
   - 文档里最核心的 recurrent 训练方案仍偏向 stored-state 单步更新
   - 这能工作，但不是最适合长期主线的训练方式

### 1.2 为什么不直接采用“完整 mode/target 显式化”版

我自己之前那版 `UNIFIED_TASK_MODELING_AND_V6_ARCHITECTURE_20260416.md` 的优点是：

- 更贴近任务本体
- 把 `mode / target / entity / history / value decomposition` 一次性定义完整
- 更像长期终局结构

但它也有 3 个现实问题：

1. **第一版复杂度偏高**
   - `mode head + target head + conditioned actor + 多 value + 多 aux`
   - 首版就同时改这么多，训练不稳定时很难快速定位问题

2. **有些显式结构可以先软化，而不必一开始就硬接入主控制流**
   - 比如 mode，不一定要第一版就做 hard gate

3. **辅助头数量偏多**
   - 在任务本质正确的前提下，过多 aux 头更像调试成本，而不是第一优先级收益

因此，两份旧方案都不是最终最优点：

- 前者太克制，可能很快再次撞天花板
- 后者太完整，第一版风险偏高

---

## 2. 综合之后的最终判断

### 2.1 必须保留的“硬核心”

以下组件我认为在新主线里是**必须有**的：

| 组件 | 是否必须 | 理由 |
|---|---|---|
| `16×16 Global` | 必须 | 当前 8×8 明显太粗 |
| `GRU` | 必须 | 时序记忆是主问题核心 |
| `Dual Critic` | 必须 | clean / survive 不拆就继续被淹没 |
| `Entity 建模` | 必须 | NPC / Charger 是任务里的关键对象，不该继续散在 scalar 里 |
| `Target 选择机制` | 必须 | 回充不是纯 mode 切换，还包含“回哪个桩” |
| `Expert 角色重构` | 必须 | 不切断 charging 控制依赖，模型不会真正学会回充 |

### 2.2 不必第一版就做成“硬控制器”的部分

以下组件我认为**需要，但第一版不必过度硬化**：

| 组件 | 结论 | 说明 |
|---|---|---|
| `mode` | 需要，但先做辅助头/表征约束，不做 hard gate | 先让 hidden state 学会阶段表达，再决定是否把它接入主控制 |
| `charger target` | 需要，而且应该直接进入主路径 | 这是回充问题的关键，不建议只做辅助预测 |
| `aux heads` | 需要，但数量要少 | 只留最关键的 1-2 个 |

### 2.3 我认为前面几份方案共同遗漏的关键点

这里是最重要的一部分。

我认为前面的多份方案，虽然已经覆盖了绝大多数核心问题，但仍共同遗漏了 4 个非常关键的设计点：

#### 漏点 1：**Legal Action Mask 不应继续作为状态语义输入主分支**

当前实现里：

- `legal_action` 不仅在 actor 采样/训练时做了 mask
- 还被直接拼进 observation，当成 scalar 分支输入的一部分

这是不理想的，因为：

- legal mask 是**当前时刻的控制约束**
- 不是稳定的环境状态语义
- 既作为 mask 又作为输入，会让网络表征混入过强的短时控制噪声

新主线里应改为：

- `legal_action` 只在 policy 端做 masked softmax / masked logits
- 不再并入主状态编码
- 如确有需要，只保留一个低维摘要，例如 `actual_legal_ratio`

这个点前面几份方案基本都没正面处理。

#### 漏点 2：**Recurrent 训练不能停留在“单步 hidden state 回放”**

很多方案提到了 stored-state BPTT，但没有把它上升为“主线训练接口必须升级”的级别。

如果只做：

- 推理时传递 hidden
- 训练时单步取出 hidden 做一步 PPO

那么 GRU 会“有记忆地推理”，但“没有足够序列感地训练”。

对长期主线来说，这不够。

因此新主线应该直接采用：

- `sequence-aware PPO`
- `chunk length = 32`
- `burn-in = 8`
- `truncated BPTT`

这件事不是可选优化，而是 recurrent 主线真正成立的条件。

#### 漏点 3：**Target 选择必须做成“对象指针”，不能做成静态 charger 槽位分类**

如果做 target head，不能简单理解成：

- `charger_1 / charger_2 / charger_3 / charger_4`

因为这会把 target 绑定到静态槽位，而不是“当前最合适的充电对象”。

更合理的方式是：

- 对 charger entity set 做动态排序 / 编码
- 用 recurrent state 作为 query
- 对 charger set 做 pointer / attention selection

也就是：

- **target 是对对象集合的选择**
- 不是对固定 ID 的分类

这个点也基本没有被前面方案完整讲透。

#### 漏点 4：**Teacher / Expert 信号必须做可靠性门控，否则会污染主线**

你特别强调的一点是对的：

- 如果 Expert 自己会错
- 那么把它直接当 teacher，可能反而伤害模型

所以新主线里：

- Expert 的 pseudo label 不能默认全量可信

必须做：

- `confidence-gated teacher signal`

只有在以下条件同时满足时，才允许 Expert 提供 teacher supervision：

- A* 路径可达且不为 `inf`
- 当前 charger 选择在最近若干步内稳定一致
- 当前局部路径安全性通过检查
- 当前场景不处于明显冲突态（例如 NPC 极近且 heuristic 自己也不稳定）

否则：

- 不提供 teacher loss
- 只保留 safety fallback

---

## 3. 最终推荐架构：V6-LTSPPO

## 3.1 观察设计

### A. Local Spatial：`21×21×8`

推荐通道：

1. `passable_or_obstacle`
2. `dirty_mask`
3. `charger_mask`
4. `npc_danger`
5. `visit_heat`
6. `trajectory_heat`
7. `frontier_mask`
8. `return_guidance`

理由：

- local 执行仍然是主要动作依据
- 必须把 revisit、risk、return guidance 一起空间化

### B. Global Spatial：`16×16×6`

推荐通道：

1. `explored_ratio`
2. `dirty_density`
3. `visit_density`
4. `charger_presence`
5. `npc_risk_density`
6. `return_cost_field`

我最终没有保留我上一版里的全部 7 通道，是为了控制首版复杂度。  
`competition_loss_density` 可以留到后续增强，不必第一版强上。

### C. Entity Set：对象分支必须保留

分成两组：

- `NPC top-4`
- `Charger top-4`

每个实体建议包含：

- `dx, dz`
- `dist_chebyshev`
- `dist_astar_or_reachable`
- `bearing_sin, bearing_cos`
- `in_local_view`
- `risk_or_priority`
- `is_nearest`

### D. Scalar + Short History：`80-96D`

建议保留并增强：

- `battery_ratio`
- `remaining_step_ratio`
- `time_since_last_charge`
- `clean_rate_ema`
- `steps_since_last_new_area`
- `charger_slack(astar)`
- `nearest_charger_astar`
- `npc_risk_summary`
- `recent_invalid_move_ratio`
- `recent_oscillation_entropy`
- `recent_revisit_ratio`
- `mode_duration_like_feature`

### E. Action History：保留

建议保留：

- `last_action_onehot`
- `recent_action_histogram(4)`

### F. Legal Action：**从 observation 主干里拆出去**

这是这版方案明确新增的决定：

- `legal_action` 不再拼入主特征
- 只在 actor 端进行 masked logits / masked softmax

---

## 4. 网络结构设计

### 4.1 编码器

| 分支 | 结构 | 输出维度 |
|---|---|---|
| Local | 轻量 ResCNN | `192` |
| Global | 轻量 ResCNN | `128` |
| Entity | Shared MLP + separate pooling | `128` |
| Scalar/History | 2-layer MLP | `96` |
| Action History | small MLP | `32` |

融合后：

- `192 + 128 + 128 + 96 + 32 = 576`
- FC -> `256`
- FC -> `192`

### 4.2 Recurrent Core

最终建议：

- 单层 `GRU`
- hidden size = `192`

这是对两份旧方案的折中：

- 比 `128` 更有余量
- 比 `256` 更稳、更省

### 4.3 Target Pointer Head：**保留，并进入主路径**

这次我明确认为：

- target 选择不能只做辅助头
- 它必须进入主路径

但形式不采用固定 5 类 hard 分类，而是：

- `recurrent state` 作为 query
- 对 charger entity set 做 pointer / attention
- 输出：
  - `target distribution`
  - `target context vector`

然后：

- actor head 使用 `recurrent_state + target_context`

这能直接把“回哪个桩”进入决策，而不引入过于死板的 charger slot 分类。

### 4.4 Mode Head：**保留，但第一版不做 hard gate**

mode 仍然重要，但我不建议第一版就让它成为强控制器。

推荐：

- `4-class mode head`
  - `clean`
  - `prepare_return`
  - `return`
  - `evade`

但第一版用途是：

- 表征约束
- 早期 teacher bootstrap
- 诊断指标

而不是：

- 直接 hard gate actor logits

这样可以吸收我原方案里“显式阶段表达”的价值，又避免新文档担心的 mode collapse 风险。

### 4.5 Actor Head

actor 输入：

- `recurrent_state`
- `target_context`
- `mode_probs` 或 `mode_embedding`

输出：

- 8 方向 logits

注意：

- 这里是 soft conditioning，不是 4 套独立 actor

### 4.6 Critic

最终采用：

- `V_clean`
- `V_survive`

不保留 triple critic。

因为新文档这里的判断是对的：

- dual critic 已经抓住主要矛盾
- triple critic 对第一版大改不是最高收益项

### 4.7 Auxiliary Heads：只留 2 个

我不再坚持 4 个 auxiliary heads。

最终只建议保留 2 个：

1. `battery_fail_risk_head`
   - 近未来电池失败风险
2. `collision_risk_head`
   - 近未来碰撞风险

理由：

- 这两个和任务主失败模式最直接相关
- 标签相对更容易定义
- 比 `coverage_gain / return_success / battery_delta` 更值得优先保留

---

## 5. 训练接口设计

### 5.1 算法主框架

保留：

- PPO

但升级为：

- `sequence-aware recurrent PPO`

### 5.2 训练组织

建议直接采用：

- sequence chunk length: `32`
- burn-in: `8`
- rollout hidden-state carry
- truncated BPTT

而不是只做单步 hidden replay。

### 5.3 Advantage / Value

保留总 advantage 的 PPO 主损失，但：

- `V_total = V_clean + V_survive`
- 两个 critic 各自回归自己的 reward stream

这点吸收新文档的 dual critic 设计，不再做更复杂的 triple value。

### 5.4 Teacher Signal

保留 teacher，但只在前期、且必须 gated：

| teacher 目标 | 来源 |
|---|---|
| mode pseudo label | Expert / heuristic 状态 |
| target pseudo label | A* 选出的 charger target |

但只在 confidence 条件满足时启用。

并且：

- teacher loss 线性退火到 `0`
- 不是永久监督

---

## 6. Expert 的最终定位

### 6.1 保留什么

必须保留：

- `NPC hard safety filter`
- `extreme battery emergency fallback`

### 6.2 移除什么

移除：

- 常态 `get_logit_bias()` 充电主导作用
- 常态 `_evaluate_return()` 对策略的控制权

### 6.3 保留但降格为信号源的部分

保留：

- A* 距离
- return 可达性
- heuristic target

但用途改为：

- reward shaping
- scalar/entity feature
- confidence-gated teacher

### 6.4 最关键的原则

> Expert 不能再是常态控制器，只能是  
> `安全网 + 极端 fallback + 高置信 teacher + 信号源`

---

## 7. 这版方案和前面两版的关系

### 它保留了“极简 GRU 方案”的优点

- 抓住 `GRU + Dual Critic + 16×16 Global` 这个主轴
- 不做过度膨胀的 aux heads
- 不让复杂度失控

### 它保留了“结构化 v6 方案”的优点

- 明确保留 `mode`
- 明确保留 `target`
- 明确保留 `entity`
- 明确保留 `history`
- 不接受继续完全隐式地让网络自己猜所有结构

### 它修正了前面共同漏掉的点

- `legal mask` 从主表征拆出
- recurrent 训练升级为真正 sequence-aware
- target 做成对象 pointer，而不是 charger slot 分类
- teacher 信号必须做可靠性门控

---

## 8. 最终实施建议

### 8.1 这套方案适不适合作为长期主线

我的判断是：

- **适合**

原因：

1. 它已经不是当前前馈骨架的小改版
2. 它已经把任务的核心结构对象补齐
3. 它的复杂度仍然没有高到需要换掉整个训练体系
4. 它足够支撑后续继续 resume

### 8.2 第一批必须落地的组件

| 必须项 | 原因 |
|---|---|
| `16×16 Global` | 基础带宽升级 |
| `Entity branch` | 对象关系必须显式化 |
| `GRU(192)` | 时序能力核心 |
| `Dual Critic` | 价值分解核心 |
| `Target pointer` | 回充目标选择核心 |
| `Legal mask 脱离 observation` | 修正表征接口 |
| `sequence-aware PPO` | recurrent 主线成立条件 |
| `Expert 降级` | 切断 charging 控制依赖 |

### 8.3 第一批不必硬上的组件

| 可稍后增强 | 原因 |
|---|---|
| mode hard gating | 第一版先做 soft/aux 即可 |
| 更多 auxiliary heads | 第一版收益不如稳定性重要 |
| competition_loss global channel | 可后续补 |
| 更强 attention | 后续再增强 |

---

## 9. 一句话结论

如果目标是：

- 不再做边际收益明显的小修补
- 允许明显大改模型
- 希望后续都围绕同一框架持续 resume

那么我最终推荐的不是“极简 GRU 替换版”，也不是“完整 mode/target/aux 全开版”，而是这份综合后的：

> **V6-LTSPPO：Structured Entity-Aware Recurrent Dual-Critic PPO with Target Pointer**

它比极简版更像长期主线，  
也比全量结构版更适合作为第一版真正训起来的系统。
