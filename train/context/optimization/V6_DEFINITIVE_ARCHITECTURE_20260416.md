# V6 统一架构终审报告：唯一推荐框架

日期：2026-04-16  
基于：v5.4 代码实际状态 + 4 份独立专家分析 + 官方赛题建模 + 统一 benchmark 评估  
约束：PPO on-policy, forward → [logits, value] (可扩展辅助输出), ≤3h 训练窗口, 8 方向动作空间不变

---

## 0. 总判断

**支持大改模型。不支持继续修补。**

理由很简单：当前系统已经把"在错误目标函数上训练一个结构错误的前馈网络"这条路走到了极限。所有 5 个手动 checkpoint（v5→v53）的 benchmark 都卡在 WR 70%、battery fail 8-10/40 的同一区间。v52-step10000 和 v53 用完全不同的策略风格（一个靠 battery 少，一个靠 collision 少）达到了几乎相同的天花板。这不是调参能突破的，而是结构性极限。

当前代码已经做了部分 A 类修正（势函数整形 α=0.35、Expert bias 5-15、梯度隔离、trajectory heatmap），但 battery fail 依然是所有版本的第一死因。这证实了三位外部专家的共同判断：**先修目标函数是必要的，但远远不够。** 问题的根在于前馈单帧网络无法学会"何时该开始回充"这个天然需要 50-200 步前瞻的决策。

---

## 1. 唯一主推荐框架：GRU-Dual-Critic Actor-Critic

### 1.1 为什么是 GRU + Dual Critic，不是其他

三位外部专家（GPT、Gemini、Opus）在分析路线上高度一致，但在推荐优先级上分歧明显：

| 专家 | 首推 | 次推 | 理由 |
|------|------|------|------|
| GPT | 方案 A (先修 reward) | 方案 C (GRU) | "先纠正目标，再升级网络" |
| Gemini | 方案一 (势能+帧堆叠) | 方案二 (双头 Critic) | "必须规避基础设施层面的重构债务" |
| Opus | 方案 A (势函数整形) | 方案 B (帧堆叠+双 Critic) | "分阶段实施，A→B→C" |

三者的共同盲点是：**他们分析时假设方案 A 还没做。** 但实际代码已经实施了方案 A 的核心元素：

- `ASTAR_POTENTIAL_ALPHA = 0.35` — 势函数整形已落地
- `EXPERT_BIAS_MIN = 5.0, MAX = 15.0` — Expert bias 已增强
- `USE_EXPERT_GRADIENT_ISOLATION = True` — 梯度隔离已启用
- `LOCAL_VIEW_CHANNELS = 4` — trajectory heatmap 已加入

**方案 A 已经做了，但结果并没有突破天花板。** 所以继续沿 A→B→C 的渐进路线走，只是在浪费时间。

### 1.2 框架核心设计

```
输入 (FEATURE_LEN)
  ├─ Local view 21×21×4 (保留 trajectory heatmap)
  │    → Conv(4→16)→ReLU→Conv(16→32)→ReLU→MaxPool→Conv(32→32)→ReLU
  │    → Flatten(32×10×10) → FC(3200→256) → 256D
  │
  ├─ Global memory 16×16×4 (分辨率提升 + 新增 dirty_memory channel)
  │    → Conv(4→16)→ReLU→Conv(16→32)→ReLU→MaxPool(2,2)→Conv(32→32)→ReLU
  │    → Flatten(32×8×8) → FC(2048→128) → 128D
  │
  ├─ Scalar features (SCALAR_DIM + 8 legal) → FC→ReLU→FC→ReLU → 64D
  │
  └─ Concat → 448D
       → FC(448→256)→ReLU → FC(256→128)→ReLU → 128D (pre-GRU)
       ↓
       GRU(input=128, hidden=128, num_layers=1)
       → 128D (post-GRU)
       │
       ├─ Actor: FC(128→8) → logits
       ├─ Clean Critic: FC(128→64)→ReLU→FC(64→1) → V_clean
       └─ Survive Critic: FC(128→64)→ReLU→FC(64→1) → V_survive
```

**改动清单（对比当前 model.py）**：

| 组件 | 当前 | 新 | 改动原因 |
|------|------|-----|----------|
| Global encoder | 8×8×3, 2 conv | 16×16×4, 3 conv + MaxPool | 8×8 丢失太多空间细节，16 倍→8 倍压缩显著改善路径规划 |
| GRU | 无 | GRU(128, 128, 1) | 核心升级：赋予模型工作记忆 |
| Critic | 1 head | 2 heads (clean + survive) | 打破 60:1 信号淹没 |
| 参数量 | ~800K | ~1.1M (+37%) | 主要来自 global encoder 扩大和 dual critic |

### 1.3 为什么选 GRU(128) 而不是 GRU(256) 或 Transformer

1. **GRU(128) 是最小有效记忆单元**。当前 backbone 输出就是 128D，GRU 维度匹配意味着不需要额外的投影层，参数增量仅 ~66K（3 个 gate × 128 × 128 + bias）。GRU(256) 参数量翻 4 倍（~260K），但实际需要记住的信息量有限（mode 持续步数、大致运动方向、上次充电时间），128D 足够编码这些。

2. **Transformer 在这个问题上是错误的选择**。Transformer 需要序列输入，但 KaiwuDRL 的数据管线（aisrv → reverb → learner）是按单步 SampleData 发送的，不保证时序顺序。要跑 Transformer 必须重写整个训练管线，这在 3 小时训练窗口+有限工程资源下不现实。GRU 的 stored-state BPTT 方案（R2D2 风格）可以在现有管线上直接工作。

3. **帧堆叠不是真正的记忆**。Gemini 的方案一用 4 帧堆叠来"硬拼"时序，但这只能看到 4 步历史（局部视图），无法表达"我已经在赶往充电桩的路上走了 30 步"这类长程状态。GRU 的衰减记忆窗口有效跨度约 30-50 步，恰好覆盖充电决策的时间尺度。

### 1.4 为什么 Dual Critic 是必须的

当前 reward 结构（已含势函数整形后）的单局量级估算：

| 信号类型 | 单局累积量 | 占比 |
|----------|-----------|------|
| 清扫奖励 | ~1200 | 92% |
| 势函数整形 | ~30-50 | 3-4% |
| 充电/urgency 相关 | ~50-80 | 4-6% |

单个 critic 的 value 预测和 GAE advantage 计算被清扫信号主导。即使势函数已经落地，充电信号仍然是噪声级别。

**Dual Critic 的做法**：

- `V_clean` 只训练清扫相关 reward（cleaning_reward, efficiency bonus）
- `V_survive` 只训练生存相关 reward（urgency_penalty, charge_reward, potential_shaping, death_penalty）
- 总 value = `V_clean + V_survive`，用于 GAE 计算
- 但两个 critic head 各自用自己的 reward 流做独立的 value 回归

这让 survive critic 不再被清扫信号淹没，可以真正学到"当前电池安全边际有多大"。

### 1.5 GRU 的 KaiwuDRL 兼容性方案

这是所有外部专家都标记为"高风险"的部分。以下是具体工程方案：

**推理端（aisrv / agent.py）**：

```python
class Agent:
    def __init__(self):
        self.gru_hidden = torch.zeros(1, 1, 128)  # (num_layers, batch, hidden)
    
    def predict(self, obs):
        # ... existing encode + backbone ...
        gru_out, self.gru_hidden = self.model.gru(
            pre_gru.unsqueeze(0),          # (1, 1, 128)
            self.gru_hidden.detach()       # detach 防止反向传播到旧步
        )
        logits = self.model.actor_head(gru_out.squeeze(0))
        v_clean = self.model.clean_critic(gru_out.squeeze(0))
        v_survive = self.model.survive_critic(gru_out.squeeze(0))
        
        # 把 hidden state 打包进 SampleData.obs
        hidden_flat = self.gru_hidden.squeeze(0).squeeze(0)  # (128,)
        packed_obs = np.concatenate([obs_flat, hidden_flat.cpu().numpy()])
        # ... 发送 packed_obs 到 reverb ...
    
    def on_episode_start(self):
        self.gru_hidden = torch.zeros(1, 1, 128)
```

**训练端（learner / algorithm.py）**：

```python
def learn(self, batch):
    obs = batch['obs']
    stored_hidden = obs[:, -128:]           # 取出存储的 hidden
    obs_clean = obs[:, :-128]               # 原始观测
    
    # Encode
    pre_gru = self.model.backbone(self.model.encode(obs_clean))  # (B, 128)
    
    # GRU 单步，用 stored hidden（detached，不回传梯度到过期模型）
    gru_out, _ = self.model.gru(
        pre_gru.unsqueeze(0),                           # (1, B, 128)
        stored_hidden.unsqueeze(0).detach()              # (1, B, 128)
    )
    gru_out = gru_out.squeeze(0)  # (B, 128)
    
    logits = self.model.actor_head(gru_out)
    v_clean = self.model.clean_critic(gru_out)
    v_survive = self.model.survive_critic(gru_out)
    # ... PPO loss with dual critic ...
```

这就是 **Stored-State BPTT**（R2D2 风格）：推理时 GRU 完整传递整个 episode 的 hidden state，训练时只回传 1 步梯度。GRU 在推理时看到了完整的 episode 历史，但训练梯度只更新当前步的映射。

**关键约束**：hidden state 增加了 128D 到 obs 维度（FEATURE_LEN 变为原始观测 + 128），需要修改 reverb buffer 的特征维度配置。这在 KaiwuDRL 中对应 `definition.py` 中的 SampleData 定义。

### 1.6 观测增强：16×16 全局图

当前 8×8 全局图（128→8，16 倍压缩）丢失了太多空间信息。改为 16×16（8 倍压缩）：

- 分辨率提升 4 倍，一个 16×16 像素代表 8×8 格而非 16×16 格
- 新增第 4 通道：dirty_memory（归一化残余脏格密度），让模型"看到"哪些区域还值得去
- preprocessor 中 mean_pool 的 pool_size 从 16 改为 8

**不做帧堆叠**。GRU 已经提供了时序记忆，帧堆叠是冗余的（Gemini 方案一的帧堆叠是因为没有 GRU 才需要的替代品）。

### 1.7 训练算法配套调整

| 项目 | 当前 | 新 | 理由 |
|------|------|-----|------|
| 优势估计 | GAE(λ=0.95) | GAE + N-step(n=16) 混合 | N-step 在 16 步内保留完整信号（γ^16=85%，vs GAE 的 (γλ)^16=37%）|
| 学习率 | 5e-5 | 3e-5 | GRU + dual critic 增加训练复杂度，降低 LR 稳定收敛 |
| Entropy 系数 | β=0.012 | β=0.015 | 防止 GRU 导致的过早确定性策略 |
| Expert bias | 线性退火 ep50→300 | 保持，但 MIN_SCALE 从 0.2 提高到 0.3 | 过渡期保留更多安全网 |

**N-step + GAE 混合方案**：

$$A_t = \alpha \cdot A_t^{\text{n-step}} + (1-\alpha) \cdot A_t^{\text{GAE}}$$

取 $\alpha = 0.5$。N-step 部分保留短程完整信号（充电奖励传播），GAE 部分保留长程方差降低。这比完全切换到 N-step 更稳定。

### 1.8 Reward 分流设计

Dual Critic 需要把 reward 分成两条流：

**Clean reward 流**（送入 V_clean 的 GAE）：
- `cleaning_reward`（清扫新格得分）
- `efficiency_bonus`（每步清扫效率）
- `cps_reward`（单步清扫密度）

**Survive reward 流**（送入 V_survive 的 GAE）：
- `urgency_penalty`（低电三档惩罚）
- `charge_reward`（成功充电奖励）
- `potential_shaping`（A* 势函数差分）
- `battery_death_penalty`（终局死亡惩罚）
- `collision_penalty`（碰撞惩罚）
- `alive_bonus`（每步存活微奖励 +0.01）

**Actor 的 advantage** 使用总 `V = V_clean + V_survive` 计算 GAE，然后统一 PPO clipped loss。两个 critic head 各自只对自己的 reward 流做 value loss。

### 1.9 参数与性能预估

| 指标 | 当前 | 预估 | 变化 |
|------|------|------|------|
| 参数量 | ~800K | ~1.1M | +37% |
| 特征维度 | 1956D | ~3000D + 128 hidden | +60% |
| 单步推理 | ~28ms | ~33ms | +18% |
| 单步训练 | ~28ms | ~36ms | +29% |
| 3h 可训步数 | ~3500 | ~2800 | -20% |
| 可 resume | — | ❌ 不可 | 架构差异太大 |

---

## 2. Expert 是否保留：保留，但彻底重新定位

### 2.1 Expert 现在在做什么

从当前代码看，Expert 实际承担了三个角色：

1. **NPC 安全过滤**（filter_actions）：dist≤3 禁止接近、dist 4-5 抑制直接方向。这是纯规则层安全网，工作良好，碰撞降低 89%。
2. **充电决策**（_evaluate_return + get_logit_bias）：基于 slack 和 battery ratio 的启发式 return mode，通过 logit bias 施加充电方向引导。这是当前 Expert-RL 耦合的核心问题。
3. **A* 路径规划**（get_expert_action）：用 A* 计算到充电桩的路径，作为 logit bias 的方向来源。

### 2.2 Expert 是在帮忙还是在制造结构性依赖

**三个角色的价值判断完全不同：**

| 角色 | 判断 | 理由 |
|------|------|------|
| NPC 安全过滤 | **必须保留** | 1-5 步内的碰撞回避是纯规则问题，不需要也不应该让 RL 从头学。RL 从"碰撞→死→负 reward→学会避开"的反馈链太长、太昂贵。规则系统在这里的 ROI 远高于学习。 |
| 充电决策 | **必须移除** | 这是当前閉环依赖的根源。Expert 做了高层 mode selection（何时充电），RL 被降格为低层执行者（往哪走）。但训练目标没有把这两个层次统一起来。结果是：Expert 越强 → 模型越不学充电 → 必须依赖 Expert → 性能上限 = Expert 规则质量。benchmark 数据直接证明了这一点：在所有版本上，battery fail 都是 8-10/40，Expert 的 logit bias（即使已增强到 5-15）仍然无法稳定地压过模型的清扫偏好。 |
| A* 路径规划 | **降格为 reward shaping 信号源** | A* 算出的路径距离是高质量信息，但不应该通过 logit bias 直接控制动作。应该把 A* 距离差分作为 reward shaping 的输入（当前已在做：potential_shaping），并作为 scalar feature 输入网络（例如 `astar_dist_to_nearest_charger`），让模型自己在 GRU 记忆的帮助下学会"该何时转向充电"。 |

### 2.3 Expert 的新角色定义

```
Expert v6 角色：
┌──────────────────────────────┐
│ NPC 安全过滤 (filter_actions) │ ← 保留，不变
│ - dist ≤ 3: 禁止接近方向     │
│ - dist 4-5: 抑制直接方向     │
└──────────────────────────────┘
┌──────────────────────────────┐
│ Reward 信号源                │ ← 保留，但从控制端移到信号端
│ - A* dist 差分 → potential   │
│ - A* dist → scalar feature   │
│ - return_mode flag → 不再用  │
└──────────────────────────────┘
┌──────────────────────────────┐
│ Emergency Fallback (新增)     │ ← 仅在极端情况硬接管
│ - battery_ratio ≤ 0.05       │
│   且 charger_slack ≤ 0       │
│ - 此时 Expert 完全接管       │
│ - 样本从 PPO 训练中排除      │
└──────────────────────────────┘
```

**关键变化**：删除 `get_logit_bias()` 和 `_evaluate_return()` 的常规调用路径。Expert 不再在正常情况下施加 logit bias。GRU + survive critic 组合负责让模型自主学会充电时机。

**只有在 battery_ratio ≤ 0.05 且 slack ≤ 0 时**（已经几乎必死的边缘情况），Expert 才硬接管，此时的样本完全从 PPO 训练中排除（不计入 policy loss 和 value loss），避免污染梯度。

### 2.4 "错误 Expert" 对模型的伤害量化

从 benchmark 数据可以直接推算 Expert 充电决策的失败率：

- v52-step10000: battery fail 8/40 = 20%
- v53-robust3450: battery fail 10/40 = 25%
- 失败局中 charge_count=0 的比例约 50%

这意味着 **Expert 在至少 10-12.5% 的局里完全没能触发充电**（charge_count=0）。原因不是 Expert 代码有 bug，而是：

- Expert 的 return threshold（LOW_BATTERY_RATIO）在某些地图/配置下计算的安全窗口不够早
- 当 Expert 触发时，bias 5-15 仍然无法稳定压过模型在那些方向上的清扫 logit
- RL 从未真正学过充电，所以当 Expert 边缘失效时，模型完全接不住

GRU + 移除 logit bias 之后，模型必须自己学充电。初期可能看到 battery fail 暂时上升，但 survive critic 的分离训练信号 + GRU 的时序记忆将确保模型最终学到比 Expert 启发式更好的充电时机。

### 2.5 Expert 退火过渡方案

直接砍掉 Expert logit bias 是安全的做法（因为 NPC filter 保留了），但为降低训练初期风险：

```
Epoch 0-100:   Expert logit bias 保持原样（过渡期）
Epoch 100-300: Expert logit bias × linear_decay(1.0 → 0.0)
Epoch 300+:    Expert 仅保留 NPC filter + emergency fallback
```

---

## 3. 为什么不是其他路线

### 3.1 为什么不是"只做 Reward 修正"（各专家的方案 A）

**已经做了，效果到顶了。** 当前代码已实施：
- 势函数整形（α=0.35，在 battery < 80% 时激活）
- Expert bias 已增强到 [5, 15]
- 梯度隔离已启用
- trajectory heatmap 已作为第 4 通道

benchmark 结果：最好的两个 checkpoint 仍然有 20-25% 的 battery fail rate。Reward 修正是必要条件，不是充分条件。瓶颈已经转移到"网络没有记忆"和"value 被清扫信号淹没"。

### 3.2 为什么不是"帧堆叠 + 电量门控"（Gemini 方案一）

帧堆叠是一种"假记忆"：

- 4 帧堆叠只看 4 步历史（~4ms 时间窗），但充电决策需要 50-200 步的上下文
- 帧堆叠把 local view 从 21×21×4 增加到 21×21×16（4 帧×4 通道），这将 local_dim 翻 4 倍，训练吞吐严重下降
- 帧堆叠需要 preprocessor 维护 frame buffer，但 reverb 单步采样仍然丢失了帧间连续性

电量门控（$h' = h \odot (1 + f(\text{battery\_ratio}))$）的想法合理，但这只是一个 3 行代码的无条件改进，不是架构选择。可以在 GRU-Dual-Critic 框架中直接加入，作为 scalar 特征的一部分即可。

### 3.3 为什么不是 Cross-Attention + Mode-Conditioned Policy（Opus 方案 D / V6-SRMTA 的变体）

之前的 V6-SRMTA 文档提出了 Mode head (4 模式) + Charger target head (5 目标) + 3 条件化 actor head 的复杂分层策略。这个设计在学术上很漂亮，但在工程上有致命问题：

1. **Mode gate 容易退化**。soft mode gate 训练不稳定，容易塌缩为"永远选同一个模式"。需要 mode entropy 正则来对抗，但这又引入了新的超参数和调试复杂度。

2. **3 个独立 actor head 共享 backbone 时的梯度冲突**。clean actor 和 charge actor 需要的 backbone 特征可能相互矛盾，导致表征退化。

3. **训练吞吐降低 30-36%**（Opus 估算）。在 3 小时窗口内只能跑 ~2500 步，如果收敛需要 3000+，可能不够。

4. **辅助头（return_success, collision_risk, battery_delta, coverage_gain）的标签生成不trivial**。`return_success` 需要回溯判断"这次回充是否成功"，`collision_risk` 需要前瞻 NPC 轨迹。这些标签要么是嘈杂的（用 heuristic 生成），要么昂贵。

**GRU-Dual-Critic 用更少的复杂度解决了同样的问题**：GRU 的 hidden state 自然编码了"当前处于什么模式"（clean 模式下 hidden state 趋近某个子空间，charge 模式下趋近另一个），不需要显式的 mode gate。Dual critic 已经分离了 clean/survive 信号，不需要 mode-conditioned value heads。

### 3.4 为什么不是完全去掉 Expert

三位专家一致同意：**现阶段不能完全去掉 Expert。** 理由：

- 模型当前在任何 reward 设计下都没有展示过自主充电能力
- NPC 安全过滤是成本极低、效果极好的规则系统，RL 不需要从头学
- 完全去掉 Expert 意味着训练初期 battery death rate 可能接近 100%，使训练信号几乎全是负的

但"不完全去掉"不等于"保持现状"。Expert 的充电决策职能必须移除，否则 GRU 永远没有机会学习充电时机。

### 3.5 为什么不是"先 A 再 B 再 C"的渐进路线

这是 GPT 和 Opus 推荐的路线。它在逻辑上是对的（先验证假设再逐步加复杂度），但在时间约束下是错的：

- **每次架构变更都需要从头训练**。方案 B 和方案 C 都不与当前 checkpoint 兼容。所以 A→B→C 不是"递进式 resume"，而是"三次推倒重训"。
- **每次重训至少 3-6 小时**（含调试、观察、调参）。A→B→C 的总耗时可能超过一周。
- **方案 A 与方案 B 的组合不稳定于单独的方案 C**。如果最终目标是 GRU，先花时间调好 A+B 再切 C 等于浪费了 A+B 的调参时间，因为 C 的网络结构完全不同。

**一步到位做 GRU + Dual Critic + 16×16 Global**，把三者的核心收益合并在一次架构变更中，是工程上最高效的路径。

---

## 4. 第一批必须改的模块

### 4.1 改动文件清单与优先顺序

| 优先级 | 文件 | 改动内容 | 修改行数估计 |
|--------|------|----------|-------------|
| P0 | `model.py` | 新增 GRU、dual critic heads、修改 forward 返回 `[logits, v_clean, v_survive]`、global encoder 扩容 | ~100 行 |
| P0 | `conf.py` | 新增 GRU_HIDDEN_DIM=128、修改 GLOBAL_MEMORY_SIZE=16、GLOBAL_MEMORY_CHANNELS=4、FEATURE_LEN 更新 | ~20 行 |
| P0 | `preprocessor.py` | global memory 从 8×8→16×16、新增 dirty_memory channel、hidden state 打包/解包、reward 分流标记 | ~80 行 |
| P0 | `algorithm.py` | dual critic loss、hidden state 拆分、PPO loss 适配（从 obs 中拆 hidden）、survive/clean reward 分流 | ~80 行 |
| P1 | `agent.py` | 维护 gru_hidden、predict() 中的 GRU 调用、hidden 打包到 obs、on_episode_start 重置 | ~50 行 |
| P1 | `expert.py` | 删除常规 logit_bias 路径、保留 NPC filter + emergency fallback、删除 _evaluate_return 常规调用 | ~40 行 |
| P2 | `train_workflow.py` | episode 重置时清零 hidden state、metric 日志增加 v_clean/v_survive 分拆 | ~20 行 |

**总改动量：~390 行**

### 4.2 实施顺序建议

```
Day 1: model.py + conf.py
  - 写好完整的 GRU-Dual-Critic Model
  - 确保 forward() 能跑通、输出维度正确
  - 单元测试：随机输入 → 输出 shape 验证

Day 2: preprocessor.py + algorithm.py
  - 16×16 全局图 + dirty_memory channel
  - reward 分流逻辑
  - dual critic loss + hidden state 拆分
  - 确保 learner 端能正确训练一个 batch

Day 3: agent.py + expert.py
  - predict 流程改造（GRU hidden 维护和打包）
  - Expert 角色缩减
  - 端到端 docker compose 启动验证

Day 4: 首轮训练 + 观察
  - 跑满 3 小时
  - 观察 v_clean 和 v_survive 的分布变化
  - 观察 battery fail rate 变化趋势
  - 如果 train 跑不动，debugfixed state BPTT 兼容性
```

### 4.3 验证标准

**成功训练的早期信号（20 分钟内）**：
- v_survive 能产生明显的低电量区间负值
- entropy 稳定在 0.8+ 以上（不塌缩）
- 无 NaN / inf（GRU hidden state 正常传递）

**60 分钟中筛**：
- battery fail 比率开始下降（相比第一轮 episode）
- v_clean 和 v_survive 走分化趋势

**3 小时终筛**：
- 跑完整 benchmark（同口径 4×10）
- 目标：WR ≥ 70%（不低于当前最好），battery fail ≤ 8
- 如果 WR 达 75%+ 且 battery fail ≤ 5，说明方向正确

---

## 5. 风险评估与缓解

### 5.1 最高风险：GRU hidden state 在 KaiwuDRL 管线中的兼容性

**风险描述**：KaiwuDRL 的 reverb buffer 是按固定特征维度存储的。增加 128D hidden state 后：
- SampleData 的 obs 维度变化
- learner 端 batch 拼接时 hidden state 的对齐
- 如果多个 aisrv 用不同版本的模型（distributed training 中的 staleness），hidden state 含义可能不一致

**缓解措施**：
1. 所有维度变更在 conf.py 中集中管理，FEATURE_LEN = 原始 obs dim，GRU_HIDDEN_DIM = 128，TOTAL_OBS_DIM = FEATURE_LEN + GRU_HIDDEN_DIM
2. algorithm.py 中用 `obs[:, :FEATURE_LEN]` 和 `obs[:, FEATURE_LEN:]` 明确拆分
3. PPO 的 clip ratio 天然对 stale hidden state 有容忍性（这是 R2D2 论文验证过的）

### 5.2 中等风险：Dual Critic 的 reward 分配不当

**风险描述**：如果某些 reward 被分错了流（例如 potential_shaping 应该归 survive 但被归了 clean），dual critic 可能比单 critic 更差。

**缓解措施**：
1. reward 分流在 preprocessor.py 中用 dict 显式管理，每个 reward key 明确标记归属
2. 总 value = V_clean + V_survive 作为 sanity check，应约等于单 critic 输出的量级
3. 如果 dual critic 明显不稳定，可以快速退回单 critic（只改 algorithm.py 的 loss 计算）

### 5.3 低风险：训练吞吐下降过多

**风险描述**：预估 3h 跑 ~2800 步。如果编码器变大后实际吞吐更低（例如 <2500 步），可能训练不充分。

**缓解措施**：
1. 保持 local encoder 不变（已优化过）
2. global encoder 只增加一层 conv + MaxPool，不做 attention
3. 如果吞吐太低，可临时把 global 从 16×16 降回 12×12

---

## 6. 关于 V6-SRMTA 文档中其他组件的取舍

之前的 UNIFIED_TASK_MODELING_AND_V6_ARCHITECTURE_20260416.md 提出了更多组件。以下是逐项判断：

| 组件 | 判断 | 理由 |
|------|------|------|
| Multi-branch encoding (Local+Global+Entity Set+Scalar+Action History) | **部分采纳** | Local+Global+Scalar 保留并增强。Entity Set encoding（NPC 用 set network）不采纳——当前 NPC 信息已在 scalar 中，且 NPC 数量固定 ≤4，不需要 set/attention。Action History 不采纳——GRU 自然记忆动作历史。 |
| GRU(256) recurrent core | **改为 GRU(128)** | 见 1.3 的理由。128 足够，256 参数量翻 4 倍但收益边际递减。 |
| Mode head (4: clean/prepare_return/return/evade) | **不采纳** | 显式 mode gate 工程复杂度高且容易退化。GRU hidden state 隐式编码模式。 |
| Charger target head (5: none + 4 chargers) | **不采纳** | 固定 4 charger slot 限制了泛化。A* 距离作为 scalar feature 已经传递了 charger routing 信息。 |
| Triple value heads (main + survival + clean) | **简化为 dual (clean + survive)** | 3 个 head 增加调参维度但边际收益低。clean + survive 的 2 头已经覆盖核心信号分离需求。|
| 4 auxiliary heads (return_success, collision_risk, battery_delta, coverage_gain) | **不采纳** | 标签生成成本高，且容易引入噪声。Dual critic 本身已经提供了 survive/clean 的分离表征压力。如果后期需要，可以加一个简单的 death prediction head（binary，label = episode 是否 battery death），这比 4 个 aux head 简单得多。 |
| Expert 作为 safety shield + emergency fallback + early teacher | **部分采纳** | safety shield (NPC filter) + emergency fallback 保留。early teacher (初期 logit bias) 通过退火过渡实现，而非长期保留。 |

---

## 7. 总结

| 项目 | 结论 |
|------|------|
| **总判断** | 支持大改。当前系统在 A 类修正已落地的情况下仍然卡在 WR 70%，结构性极限明确。 |
| **唯一主推荐框架** | GRU-Dual-Critic Actor-Critic：GRU(128) 时序记忆 + clean/survive 双头 Critic + 16×16 全局图 |
| **Expert 是否保留** | 保留 NPC filter 和 emergency fallback。移除充电 logit bias（从控制接口降格为 reward/feature 信号源）。 |
| **为什么不是其他路线** | (1) 纯 Reward 修正已做到位仍不够；(2) 帧堆叠是假记忆；(3) Mode-Conditioned Policy 工程复杂度过高且容易退化；(4) 渐进 A→B→C 在时间上不经济。 |
| **第一批必须改的模块** | model.py, conf.py, preprocessor.py, algorithm.py (P0)。agent.py, expert.py (P1)。~390 行总改动。 |
| **预期结果** | 首轮训练目标 WR ≥ 70%（不退化）、battery fail ≤ 8；迭代 2-3 轮后目标 WR ≥ 80%、battery fail ≤ 5 |
