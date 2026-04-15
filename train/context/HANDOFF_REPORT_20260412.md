# 训练交接报告 — 2026-04-12

**生成时间**: 2026-04-12 ~11:20
**撰写者**: Claude Code (8小时自动监控session)
**接手者**: 下一个AI session

---

## 一、当前训练状态 (实时)

| 指标 | 当前值 |
|------|--------|
| 容器状态 | 12/12 运行中 |
| Global step | ~34,500 (仍在训练) |
| 最新entropy | 0.51-0.70 (非常健康) |
| 最新value_loss | 128-228 (正常波动) |
| 最新checkpoint | model.ckpt-34500.pkl |
| resume模型 | model.ckpt-resume.pkl (episode_cnt=2560) |
| 分支 | win |
| 训练速度 | ~100 steps/min |

**训练正在正常运行，没有崩溃或异常。**

---

## 二、项目结构

```
D:\TcKaiwuFinal\
├── code\                          # Agent代码 (挂载到容器内 /data/projects/robot_vacuum/)
│   ├── agent_ppo\
│   │   ├── agent.py               # Agent主类 (predict/learn/save/load)
│   │   ├── algorithm\algorithm.py # PPO算法实现
│   │   ├── conf\conf.py           # 超参数配置
│   │   ├── feature\
│   │   │   ├── preprocessor.py    # 特征提取 + reward函数
│   │   │   ├── expert.py          # 专家策略 (NPC过滤 + A*充电导航)
│   │   │   └── definition.py      # ObsData/ActData定义
│   │   ├── model\model.py         # 神经网络
│   │   └── utils\                 # 工具类
│   ├── model.ckpt-resume.pkl      # 当前resume模型 (自动同步最新)
│   ├── latest_model.pkl           # 最新模型快照
│   └── model.ckpt-resume.meta.json # resume元数据
├── train\
│   ├── .docker-compose.yaml       # Docker Compose配置
│   ├── .env                       # KAIWU_GAMECORE_NUM=4
│   ├── tb_logs\                   # TensorBoard日志
│   ├── log\                       # 运行日志
│   │   ├── learner\               # Learner训练日志
│   │   ├── kaiwu_env\             # 环境日志 (含episode finish数据)
│   │   └── aisrv\                 # AI服务日志
│   └── context\                   # 上下文/报告
│       ├── AUTO_ITER_LOG_20260412.md  # 完整8小时迭代日志 (重要!)
│       └── HANDOFF_REPORT_20260412.md # 本文件
```

### 关键命令

```bash
# 查看容器状态
docker ps --filter "name=kaiwu-train" --format "{{.Names}} {{.Status}}"

# 查看最新训练指标
grep "policy_loss" train/log/learner/learner_train_pid*_log_2026-04-12-*.log | tail -5

# 查看最新global step
grep "global step" train/log/learner/learner_train_pid*_log_2026-04-12-*.log | tail -1

# 提取episode数据 (从所有4个env容器)
for f in train/log/kaiwu_env/env_container*_log_2026-04-12-*.log; do grep "finish monitor_data" "$f" 2>/dev/null; done

# 重启训练 (应用代码修改)
cd train && docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed down
cd train && docker compose -p kaiwu-train -f .docker-compose.yaml --profile distributed up -d

# 仅重启learner (应用BETA等参数)
docker restart kaiwu-train-learner-1
```

---

## 三、代码修改历史 (本次session)

### 基线commit: 54461a4

### 修改1: 紧急修复策略坍塌 (04:05, commits 30989cc→b11d4cd)

**问题**: 从最佳模型(step 47491, robust=1397.9) resume后分数暴跌, invalid_move率0.6-1.0

**4个根因及修复**:

#### conf.py
```python
# 修改前
INIT_LEARNING_RATE_START = 0.0003
BETA_START = 0.001

# 修改后
INIT_LEARNING_RATE_START = 0.0001  # 迁移学习需要低LR
BETA_START = 0.003                 # 增加entropy防止策略坍塌
```

#### agent.py — 启用NPC安全过滤 + 反卡死
```python
# 修改前: filter_actions()已定义但未调用
# 修改后: 在predict()中调用

# Layer 1: NPC安全过滤 — 阻止朝NPC方向移动
filtered_legal = expert.filter_actions(self.preprocessor, legal_action)

# Layer 3: 反卡死 — 10步卡住后随机合法动作
if self.preprocessor.stuck_steps >= 10:
    legal_indices = [i for i, l in enumerate(filtered_legal) if l]
    if legal_indices:
        random_action = int(np.random.choice(legal_indices))
        # ... 返回随机动作
```

#### preprocessor.py — Reward调整
```python
# 修改前→修改后:
cleaning_reward: 2.0 → 1.5          # 降低清扫奖励防止过度贪婪
revisit penalty: -0.20/4 → -0.08/3  # 减轻回访惩罚防止死亡螺旋
stuck_penalty: -0.3/-0.15 → -0.5/-0.25  # 加强卡住惩罚
clip: [-4,6] → [-3,4]              # 收窄reward范围
# 新增: charger_path_explore = 0.06  # 朝充电桩方向探索新区域奖励
```

#### expert.py — 充电专家改进
```python
# 修改:
CHARGE_SAFETY_MARGIN: 40 → 50      # 提前更多触发充电

# 新增: 路径未探索时提前激活专家
if self._path_unexplored_ratio(prep) > 0.3:
    threshold = int(threshold * 1.5) + 30

# 新增: cost_map将未探索区域设为可通行(代价3.0)而非不可通行
# 之前: 未探索=INF (不可通行, A*无法规划)
# 之后: 未探索=3.0 (可通行但代价高, 允许规划)
```

### 修改2: BETA 0.003→0.005 (05:30, commit 0285878)

**原因**: entropy从1.6降至0.42, Q4分数回落-21%

### 修改3: BETA 0.005→0.007 (06:42, commit 769987a)

**原因**: entropy破0.25安全线降至0.215
**状态**: 代码已commit但**容器未重启, 此修改未生效!**

---

## 四、当前生效的配置

```python
# conf.py (当前生效)
INIT_LEARNING_RATE_START = 0.0001    # 实际生效
BETA_START = 0.005                   # 实际生效 (0.007已commit但未重启)
CLIP_PARAM = 0.2
VF_COEF = 0.5
GAMMA = 0.99
LAMDA = 0.95
GRAD_CLIP_RANGE = 0.5
ACTION_NUM = 8

# 网络输入维度 (不可修改, 会破坏权重兼容性)
SCALAR_DIM = 74
FEATURES = [1323, 192, 74, 8]  # [local_view, global_memory, scalar, dir_onehot]
FEATURE_LEN = sum(FEATURES) = 1597
```

---

## 五、训练进展数据

### 8小时里程碑

| 时间 | Step | Best Score | Avg Score | Entropy | 关键事件 |
|------|------|-----------|-----------|---------|----------|
| 04:35 | 1,234 | 322 | 89 | 0.58 | 初始修复后恢复训练 |
| 05:00 | 3,676 | 720 | 203 | 1.0 | 首次突破700 |
| 06:28 | 5,648 | 607 | 157 | 0.28 | BETA调至0.005 |
| 07:30 | 11,500 | 450 | 179 | 0.24 | entropy最低点 |
| 08:03 | 13,900 | 415 | 210 | 0.28 | 灾难局首次清零 |
| 08:28 | 17,434 | 660 | 232 | 0.27 | 历史突破660 |
| 09:33 | 23,385 | 865 | 259 | 0.40 | 历史突破865 |
| 09:43 | 25,584 | **1063** | 191 | 0.32 | **ALL-TIME BEST** |
| 10:33 | 29,800 | 1063 | 297 | 0.38 | 8小时监控结束 |

### 分数分布 (1023 episodes)

| 等级 | 范围 | 占比 |
|------|------|------|
| 灾难 (0-10) | 14.4% | NPC阻挡导致 |
| 很低 (10-50) | 10.9% | |
| 低 (50-200) | 38.2% | |
| 中 (200-400) | 28.6% | 主力分布 |
| 好 (400-600) | 6.8% | |
| 精英 (600+) | 1.1% | |

### ALL-TIME Top 10

| 排名 | Score | Map | Charges | 是否WIN |
|------|-------|-----|---------|---------|
| 1 | 1063 | map6 | 1 | 否(battery耗尽) |
| 2 | 906 | map5 | 1 | 否 |
| 3 | 899 | map5 | 1 | WIN (remain=291) |
| 4 | 798 | map5 | 1 | 否 |
| 5 | 792 | map10 | 2 | WIN (remain=60) |
| 6 | 782 | map10 | 6 | WIN (remain=126) |
| 7 | 775 | map10 | 3 | WIN (remain=49) |
| 8 | 768 | map2 | 1 | 否 |
| 9 | 766 | map2 | 1 | 否 |
| 10 | 609 | map6 | 0 | 否 |

### Per-Map表现 (1023 eps)

| Map | Avg Score | Best | Episode数 | Charge_avg |
|-----|-----------|------|-----------|------------|
| map6 | 205 | 1063 | 65 | 0.2 |
| map8 | 191 | 589 | 59 | 0.0 |
| map10 | 178 | 792 | **437** | 0.3 |
| map5 | 171 | 906 | 79 | 0.1 |
| map7 | 167 | 608 | 63 | 3.5* |
| map4 | 156 | 534 | 67 | 0.1 |
| map1 | 152 | 581 | 59 | 0.1 |
| map2 | 152 | 768 | 68 | 0.1 |
| map3 | 113 | 434 | 52 | 0.2 |
| map9 | 132 | 594 | 61 | 0.1 |

*map7的3.5是异常值, 某个episode有217次充电

### WIN统计
- 总WINs: 41/1023 (4.0%)
- WIN条件: 完成所有max_steps且remaining_charge > 0
- 最高WIN: 899 (map5)

---

## 六、三大瓶颈分析

### 瓶颈 #1: 充电策略 (最大杠杆)

**数据**:
- 有充电的episode: 109/1023 (10.8%), 平均得分 **362**
- 无充电的episode: 914/1023 (89.2%), 平均得分 **147**
- 差距: **2.5倍**

**根因**:
1. 专家override时使用uniform概率(非模型概率), PPO ratio计算失真, agent从override中学不到东西
2. 充电reward太弱 (charger_path_explore=0.06, charge_bonus=0.5), 相比cleaning_reward=1.5微不足道
3. 模型在entropy高时主要学清扫, entropy低后失去探索充电路线的能力
4. A*路径规划在未探索区域不可靠

**可能的优化方向**:
- 增加充电相关reward权重 (charger_path_explore 0.06→0.15, charge_bonus 0.5→1.0)
- 专家override时保留模型概率分布(而非uniform), 让PPO能从override中学习
- 增加battery低时的紧迫感reward
- 考虑curriculum: 先训练充电, 再训练清扫

### 瓶颈 #2: 灾难局 (14.4%)

- 14.4%的episode得分<10
- 主要原因: NPC动态阻挡导致agent卡住
- 已有机制: filter_actions(阻止朝NPC方向移动) + anti-stuck(10步后随机)
- 引擎层面问题, 代码层面改进空间有限

### 瓶颈 #3: Map方差

- 强图(map6=205, map8=191) vs 弱图(map3=113, map9=132), 差距近2x
- map10占比43.3% (437/1023), 异常集中, 可能是训练配置问题
- 原因待查: 检查.env和gamecore配置中map选择逻辑

---

## 七、训练算法健康状况

| 指标 | 当前值 | 评价 |
|------|--------|------|
| Entropy | 0.51-0.70 | 非常健康 (阈值: >0.25安全, <0.20危险) |
| Value Loss | 128-228 | 正常波动, 未发散 |
| Policy Loss | -3~+0.4 | 正常 |
| 学习率 | 0.0001 | 稳定 |
| 训练速度 | ~100 steps/min | 正常 |

### Entropy历史
```
Step 0:      0.58 (初始)
Step 5000:   0.28 (下降)
Step 7000:   0.24 (危险低点)
Step 10000:  0.24 (BETA=0.005生效, 开始恢复)
Step 20000:  0.35 (自然恢复)
Step 30000:  0.38 (健康)
Step 34000:  0.51-0.70 (非常健康, 可能偏高)
```

---

## 八、待办事项 (优先级排序)

### [紧急] 重启容器应用BETA=0.007
- Commit: 769987a
- 当前运行BETA=0.005, 代码已改为0.007但容器未重启
- 命令: `docker restart kaiwu-train-learner-1`
- 注意: 当前entropy已升至0.5-0.7, 重启后BETA=0.007可能使entropy更高. 如果entropy持续>0.7导致策略过于随机, 可考虑降回0.005或0.006.

### [重要] 调查map10占比43%
- 437/1023 episodes出现在map10, 远超其他map的5-8%
- 检查: `.env`, `train_env_conf.toml`, gamecore配置中的map选择逻辑
- 可能是配置中map10权重过高, 或某种profile组合总是选择map10

### [重要] 优化充电策略
- 这是提升分数的最大杠杆 (2.5x差距)
- 具体方向见上方"瓶颈#1"分析
- 建议优先尝试: 增大充电相关reward权重

### [可选] Entropy监控
- 如果继续自动监控, 建议设阈值:
  - entropy < 0.25 持续5次 → 提高BETA +0.002
  - entropy > 0.70 持续5次 → 降低BETA -0.002
- 当前entropy 0.5-0.7, 处于较高水平

---

## 九、文件变更清单

### 已修改的代码文件 (相比baseline 54461a4)
| 文件 | 修改内容 |
|------|----------|
| `code/agent_ppo/conf/conf.py` | LR 0.0003→0.0001, BETA 0.001→0.007(未生效) |
| `code/agent_ppo/agent.py` | 启用filter_actions() + anti-stuck机制 |
| `code/agent_ppo/feature/expert.py` | CHARGE_SAFETY_MARGIN 40→50, 新增_path_unexplored_ratio(), cost_map未探索区可通行 |
| `code/agent_ppo/feature/preprocessor.py` | reward调整: cleaning→1.5, revisit→-0.08/3, stuck→-0.5/-0.25, clip→[-3,4], 新增charger_path_explore |

### 未修改的文件 (重要但未动)
| 文件 | 说明 |
|------|------|
| `code/agent_ppo/algorithm/algorithm.py` | PPO算法, loss计算, var_beta从Config读取 |
| `code/agent_ppo/model/model.py` | 网络结构, 未改动 |
| `code/agent_ppo/feature/definition.py` | 数据结构定义 |
| `code/agent_ppo/workflow/train_workflow.py` | 训练流程 |

### 新增文件
| 文件 | 说明 |
|------|------|
| `train/context/AUTO_ITER_LOG_20260412.md` | 完整8小时迭代日志, 含6次深度分析和10+次快照 |
| `train/context/episodes_*.txt` | 原始episode数据 |
| `train/context/HANDOFF_REPORT_20260412.md` | 本文件 |

---

## 十、重要约束

1. **不可修改的维度**: `SCALAR_DIM=74`, `FEATURES=[1323,192,74,8]` — 改了会破坏已训练权重
2. **不可修改网络结构** — 同上
3. **Git分支**: 当前在 `win` 分支
4. **Docker**: 使用 `kaiwu-train` 项目名
5. **模型同步**: `code/model.ckpt-resume.pkl` 是自动同步的最新模型, 不要手动删除

---

*报告结束 — 2026-04-12 11:20*
