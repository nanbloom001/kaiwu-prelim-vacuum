# v5.4 电池死亡根因诊断

**日期**: 2026-04-14
**方法**: 在 Expert `_evaluate_return()` 中添加文件诊断日志，捕获 battery_ratio ≤ 0.20 时的内部状态

---

## 一、初始假设（均被推翻）

| 假设 | 验证结果 |
|------|---------|
| 充电桩 organ 未出现 → _charger_list 空 | ❌ `chargers=3`，始终找到充电桩 |
| A* 找不到路径 → expert_action=None | ❌ 3214 条日志全部有有效 action，path=20-52 |
| NPC 太近 → Fix 3 抑制充电 bias | ❌ NPC 距离 59-98（远），未被抑制 |
| 充电桩格子 impassable → A* 不可达 | ❌ path 非空，A* 成功规划 |

---

## 二、诊断方法

在 `expert.py` `_evaluate_return()` 中添加诊断代码：

```python
if battery_ratio <= 0.20:
    _diag = (
        f"[EXPERT_DIAG] bat={prep.battery}/{prep.battery_max} ratio={battery_ratio:.3f}"
        f" act={'None' if expert_action is None else expert_action}"
        f" chargers={len(self._charger_list)}"
        f" path={len(charger_path) if charger_path else 0}"
        f" dist={charger_dist:.0f} margin={margin:.1f}"
        f" nearest_cd={prep.nearest_charger_dist:.0f}"
        f" cdx={prep.nearest_charger_dx:.0f} cdz={prep.nearest_charger_dz:.0f}"
        f" stage={_fallback_stage}"
        f" legal={sum(1 for l in legal_action if l)}"
        f" return={self.return_mode}\n"
    )
    with open("/workspace/log/expert_diag.log", "a") as _f:
        _f.write(_diag)
```

写入 `/workspace/log/expert_diag.log`（stderr 在 Docker 中不可见），重启 aisrv 后收集 3214 条日志。

---

## 三、诊断数据摘要

### 3.1 Expert 内部状态

- **expert_action**: 全部非 None（8 个方向均有），分布：act=0(634) > act=2(624) > act=4(532) > act=6(463)
- **chargers**: 3（始终找到充电桩）
- **path**: 20-52（A* 路径长度，非空）
- **stage**: 0（全部来自缓存路径，最可靠的一级 fallback）
- **return**: True（return_mode 正确触发）

### 3.2 battery_ratio 分布

| 区间 | 数量 | 占比 | bias 值 | 效果 |
|------|------|------|---------|------|
| ≤0.10（emergency） | 1105 | 27% | 100.0 | 足够强 |
| 0.10-0.15 | 1271 | 31% | 3-8 | **太弱** |
| 0.15-0.20 | 1664 | 41% | 3-8 | **太弱** |

### 3.3 典型案例

```
bat=68/340 ratio=0.200 act=0 chargers=3 path=52 dist=51 margin=22.6 nearest_cd=15
```

计算 bias：
- slack = 68 - 51 = 17
- urgency = clip(1 - 17/22.6, 0.2, 1.0) = 0.25
- bias = 3.0 + 5.0 * 0.25 = **4.25**

模型的清扫方向 logit 可能是 +5，Expert 推荐方向 logit 可能是 -5。
Effective：-5 + 4.25 = -0.75，仍低于清扫方向的 +5。
**模型忽略 Expert 方向，继续清扫。**

---

## 四、根因链

```
Expert return_mode 触发（ratio=0.32，bat≈109/340）
→ A* 找到路径（path=52，dist=51）
→ expert_action 有效（act=0，方向正确）
→ bias = 3-8（非 emergency，因为 ratio > 0.10）
→ bias 太弱，模型 logit 中清扫偏好压过充电 bias
→ 机器人继续清扫而非充电
→ 电量持续下降到 ratio=0.10
→ emergency bias=100 终于触发
→ 但此时只剩 bat≈34 步，可能来不及
→ 电池死亡
```

---

## 五、修复方向

**核心问题：non-emergency bias（3-8）远不足以覆盖模型的清扫偏好。**

方案 A（推荐）：提高 bias 强度 + 降低 emergency 阈值
- emergency 阈值：0.10 → 0.25
- non-emergency 范围：[3, 8] → [8, 20]
- 效果：ratio 0.10-0.25 时 bias=20+，足够覆盖 10 点 logit 差

方案 B：更早触发 return_mode
- effective_low_ratio：0.32 → 0.45
- 但这只是更早触发，如果 bias 仍然太弱则无效

方案 C：A+B 组合（最稳健）

---

## 六、v5.4 训练数据汇总

| 指标 | v5.3 | v5.4 | 变化 |
|------|------|------|------|
| WinRate | 85.2% (216 ep) | 88.2% (110 ep) | +3% |
| 碰撞死亡 | 18 例 (56%) | 2 例 (15%) | **-89%** |
| 电池死亡 | 14 例 (44%) | 11 例 (85%) | -21% |
| Entropy | 0.12-0.21 稳定 | 0.18-0.21 稳定 | 健康 |

v5.4 Fix 1-3 对碰撞的修复效果极为显著（-89%）。电池死亡成为唯一剩余瓶颈。
