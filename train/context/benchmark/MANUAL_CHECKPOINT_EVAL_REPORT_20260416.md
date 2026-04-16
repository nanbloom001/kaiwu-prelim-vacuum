# 手动保存点统一 Eval 分析报告

日期：2026-04-16

## 目的

对仓库中所有手动保存的 checkpoint 用同一套 benchmark 口径做统一评估，回答三个问题：

1. 模型是否随着版本推进而真实进步。
2. 最好的手动保存点是哪一个。
3. 当前主要问题是碰撞、安全性，还是回充/电池管理。

## 评估口径

为避免 benchmark 拓扑本身带来偏差，本次全部 checkpoint 都使用相同配置：

- 入口：`bash train/run_benchmark_parallel.sh <checkpoint> --workers 4 --envs-per-worker 10 --max-wait 1800`
- benchmark 拓扑：`4×10`
- 场景集：固定 40 局
- 说明：此前已验证不同 benchmark 拓扑会影响结果，因此本次只做**同拓扑横向比较**

## 评估对象

1. `code/saved_models/v5-step4300/model.pkl`
2. `code/saved_models/v51-step4900/model.ckpt-resume.pkl`
3. `code/saved_models/v52-step10000/model.ckpt-resume.pkl`
4. `code/saved_models/v52-step70000/model.ckpt-resume.pkl`
5. `code/saved_models/v53-robust3450/model.ckpt-resume.pkl`

对应 session：

- `v5-step4300` -> `20260416-134009`
- `v51-step4900` -> `20260416-134213`
- `v52-step10000` -> `20260416-134402`
- `v52-step70000` -> `20260416-134558`
- `v53-robust3450` -> `20260416-134758`

结果目录位于 `train/eval_parallel_logs/<session_id>/`。

## 总表

| Checkpoint | WR | Avg CS | Wins | Elapsed | Battery | Collision |
|---|---:|---:|---:|---:|---:|---:|
| `v5-step4300` | 52.5% | 682.3 | 21/40 | 40.0s | 14 | 5 |
| `v51-step4900` | 55.0% | 724.0 | 22/40 | 36.1s | 13 | 5 |
| `v52-step10000` | 70.0% | 867.0 | 28/40 | 48.1s | 8 | 4 |
| `v52-step70000` | 60.0% | 715.0 | 24/40 | 48.1s | 9 | 7 |
| `v53-robust3450` | 70.0% | 864.0 | 28/40 | 52.9s | 10 | 2 |

## 逐版本趋势

### 1. `v5-step4300 -> v51-step4900`

有改善，但幅度有限：

- WR：`52.5% -> 55.0%`
- Avg CS：`682.3 -> 724.0`
- Battery fail：`14 -> 13`
- Collision fail：`5 -> 5`

说明：

- 早期版本主要问题就是**大量 battery fail**
- `v51` 比 `v5` 更稳，但没有质变，仍然处于“能清扫，但经常死于回充失败”的阶段

### 2. `v51-step4900 -> v52-step10000`

这是本轮手动保存点里最明显的一次提升：

- WR：`55.0% -> 70.0%`
- Avg CS：`724.0 -> 867.0`
- Battery fail：`13 -> 8`
- Collision fail：`5 -> 4`

说明：

- `v52-step10000` 是本次统一 eval 下的**综合最优点**
- 它不只是更能活下来，而且成功局的质量也更高：
  - `completed` 局平均 clean score `1084.0`
  - `completed` 局平均 steps `1371.4`

### 3. `v52-step10000 -> v52-step70000`

这里出现了明显回退：

- WR：`70.0% -> 60.0%`
- Avg CS：`867.0 -> 715.0`
- Battery fail：`8 -> 9`
- Collision fail：`4 -> 7`

说明：

- 这不是轻微波动，而是**实质性退化**
- 主要问题不是单纯 battery，而是**collision 明显恶化**
- `v52-step70000` 不适合作为“当前最强基线”的默认结论

### 4. `v52-step70000 -> v53-robust3450`

`v53` 把成绩重新拉回高位，但改善方式和 `v52-step10000` 不一样：

- WR：`60.0% -> 70.0%`
- Avg CS：`715.0 -> 864.0`
- Battery fail：`9 -> 10`
- Collision fail：`7 -> 2`

说明：

- `v53` 最突出的优点是**安全性恢复明显**
- 它几乎把 collision 压回最低水平
- 但 battery fail 没有同步解决，甚至略高于 `v52-step10000`

## 分轮结果

### `v5-step4300`

- `round_1`: WR `0.7`, CS `656.1`
- `round_2`: WR `0.8`, CS `899.2`
- `round_3`: WR `0.4`, CS `531.8`
- `round_4`: WR `0.2`, CS `642.0`

### `v51-step4900`

- `round_1`: WR `0.9`, CS `799.0`
- `round_2`: WR `0.7`, CS `835.9`
- `round_3`: WR `0.4`, CS `623.4`
- `round_4`: WR `0.2`, CS `637.7`

### `v52-step10000`

- `round_1`: WR `0.8`, CS `729.2`
- `round_2`: WR `0.8`, CS `865.5`
- `round_3`: WR `0.8`, CS `1082.6`
- `round_4`: WR `0.4`, CS `790.7`

### `v52-step70000`

- `round_1`: WR `0.8`, CS `724.2`
- `round_2`: WR `0.7`, CS `755.9`
- `round_3`: WR `0.7`, CS `816.6`
- `round_4`: WR `0.2`, CS `563.3`

### `v53-robust3450`

- `round_1`: WR `0.9`, CS `807.4`
- `round_2`: WR `0.8`, CS `968.0`
- `round_3`: WR `0.4`, CS `595.0`
- `round_4`: WR `0.7`, CS `1085.6`

## 关键观察

### 观察 1：进步不是单调上升，而是“先峰值、后回退、再换方向恢复”

如果按整体成绩看：

- `v5 -> v51`：缓慢改善
- `v51 -> v52-step10000`：显著提升
- `v52-step10000 -> v52-step70000`：明显回退
- `v52-step70000 -> v53`：重新回升

因此，当前仓库里的手动保存点不是一条单调更优的轨迹，而是：

- **`v52-step10000` 达到第一个综合峰值**
- **`v52-step70000` 出现回退**
- **`v53` 通过压碰撞把总成绩拉回高位**

### 观察 2：`v52-step10000` 和 `v53` 是两种不同风格的“最优”

两者总体非常接近：

- `v52-step10000`: `WR 70%`, `Avg CS 867.0`, `battery 8`, `collision 4`
- `v53-robust3450`: `WR 70%`, `Avg CS 864.0`, `battery 10`, `collision 2`

差别在于：

- `v52-step10000` 的优势在 **battery 更少、round_3 更强**
- `v53` 的优势在 **collision 更少、round_4 更强**

也就是说：

- 如果优先看**综合平均成绩**，`v52-step10000` 略胜
- 如果优先看**安全性/碰撞控制**，`v53` 更好

### 观察 3：`v52-step70000` 是明显回退点，主要问题是 collision 恶化

`v52-step70000` 的 collision fail 达到 `7`，是后期模型里最差的。

典型现象：

- `round_3/map9` 在 `19` 步就撞死
- `round_4/map2` 在 `7` 步就撞死
- `round_4/map9` 在 `18` 步就撞死

这类失败不是“后程决策失误”，而是**很早就失去安全边际**。

这说明在这段训练后期：

- 模型对 hard profile 的避障稳定性变差
- 不是单纯变得更激进，而是确实出现了**安全策略退化**

### 观察 4：battery fail 仍然是贯穿所有版本的主问题

即使最佳模型也没有彻底解决 battery：

- `v52-step10000`: battery `8`
- `v53`: battery `10`

battery fail 的共同特征非常一致：

- 大量失败局 `charge_count = 0`
- 很多失败局直到 episode 后半程才进入 `mode 1/2`
- 进入回充模式时 `charger_slack` 已经为负，说明已经“来不及”

代表性案例：

- `v53 round_3/map2`
  - `200` 步电池死
  - `charge_count = 0`
  - 第一次进入 return mode 在 `145/200`
  - 最终 `charger_slack = -7`

- `v52-step10000 round_4/map4`
  - `200` 步电池死
  - `charge_count = 0`
  - 第一次进入 return mode 在 `119/200`
  - 最终 `charger_slack = -67`

这说明当前 battery fail 不只是“没找到充电桩”，更常见的是：

- **回充触发太晚**
- **触发后已经无路可救**
- **在若干 hard profile 下模型仍会优先清扫而不是提前回充**

### 观察 5：`v53` 的主要收益是“降碰撞”，不是“降电池死”

`v53` 的 collision 从 `7` 降到 `2`，这是最明显的改进。

但 battery 没同步改善：

- `v52-step70000`: `battery 9`
- `v53`: `battery 10`

因此 `v53` 更像是：

- **安全性修复版本**
- 而不是**回充策略突破版本**

### 观察 6：hard case 仍集中在 `round_3` 和 `round_4`

跨模型对比后，重复失败较多的 case 主要集中在：

- `round_3`: map `2/4/5/6/9`
- `round_4`: map `1/2/3/4/6/7/8/9/10`

其中最顽固的是：

- `round_4/map4`
- `round_4/map6`
- `round_4/map8`

这些 case 在至少 3 个 checkpoint 上失败，说明它们不是单次噪声，而是**当前策略的稳定盲区**。

## 失败模式拆解

### Battery fail

Battery fail 可以分成两类：

1. **短局早死型**
   - 常见于 `battery_max=200` 的 hard case
   - `charge_count=0`
   - 在 `100-170` 步左右才进入回充模式
   - 典型问题是“回充判断滞后”

2. **长局后程耗尽型**
   - 已经充过几次电，但在长 horizon 下最终仍没撑住
   - 如：
     - `v53 round_4/map1`: `940` 步，`charge_count=5`
     - `v53 round_4/map7`: `1512` 步，`charge_count=9`
     - `v52-step70000 round_4/map7`: `1844` 步，`charge_count=15`
   - 这类问题更像：
     - 长程 credit assignment 不足
     - 充电后再次出发的策略边际不够
     - 多次回充场景下的时机管理仍然偏贪心

### Collision fail

Collision fail 也分两类：

1. **超早期撞死**
   - 如：
     - `v52-step70000 round_4/map2`: `7` 步
     - `v52-step70000 round_3/map9`: `19` 步
     - `v53 round_3/map10`: `2` 步
   - 更像是：
     - 初始布局的安全过滤不够
     - 某些局面下 expert soft avoidance 仍有漏洞

2. **中后程碰撞**
   - 如：
     - `v52-step10000 round_4/map10`: `1195` 步
     - `v52-step10000 round_1/map10`: `465` 步
   - 这类通常不是“起手爆炸”，而是长局中的路径冲突或回充/躲避博弈失误

## 最终判断

### 1. 是否有进步

有，但不是单调进步。

更准确的判断是：

- 训练在 `v52-step10000` 左右达到了第一波综合峰值
- 之后出现过明显回退（`v52-step70000`）
- `v53` 通过减少 collision 把整体成绩重新拉回高位

### 2. 当前最好的手动保存点是哪一个

如果只看这次统一 eval：

- **综合最优**：`v52-step10000`
- **安全性最优**：`v53-robust3450`

推荐：

- 做后续基线对比时，可以同时保留两者：
  - `v52-step10000` 作为**综合成绩基线**
  - `v53-robust3450` 作为**低碰撞安全基线**

### 3. 当前主要问题在哪里

主问题仍然是 **battery / 回充策略**，不是 collision。

理由：

- 最好的两个模型依然有 `8-10` 个 battery fail
- 这些失败在很多局里表现为：
  - `charge_count = 0`
  - 回充模式触发太晚
  - 进入回充时 `charger_slack < 0`

collision 不是消失了，但：

- `v53` 已经证明 collision 可以压到很低
- battery 则在所有版本上都没有被真正解决

## 下一步建议

1. 后续训练和 benchmark 不要只盯 `v52-step70000`
   - 它不是本次统一 eval 下的最佳点
   - 至少应把 `v52-step10000` 和 `v53-robust3450` 一起作为双基线

2. 后续所有 checkpoint 保存都建议固定跑一次同口径 `4×10` benchmark
   - 否则很容易把训练曲线误判成真实进步

3. 如果下一轮继续做算法优化，优先方向仍是：
   - 提前回充触发
   - 长程 battery credit assignment
   - hard profile 下的 return-mode 稳定性

4. 如果要做更细的 case study，优先盯这些 persistent hard cases：
   - `round_4/map4`
   - `round_4/map6`
   - `round_4/map8`
   - `round_3/map2`
   - `round_3/map4`
   - `round_3/map5`
   - `round_3/map6`
   - `round_3/map9`
