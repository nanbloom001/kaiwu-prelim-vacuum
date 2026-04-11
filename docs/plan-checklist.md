# 清扫大作战执行清单（基于 plan.md）

生成日期：2026-04-08（2026-04-09按当前仓库状态修订）  
适用范围：`E:\competition\26fwwb\agent_ppo`

## 0. 训练执行策略（当前）

当前以**腾讯官方平台训练**为主，本地 Docker 镜像问题不再作为阻塞项。  
本仓库重点工作是 `agent_ppo` 算法与特征/奖励优化，训练验证以平台结果为准。

本地容器仅用于可选调试（非必需）：

```powershell
# 可选：本地调试时，先同步到容器挂载目录
Set-Location E:\competition\26fwwb
Copy-Item .\agent_ppo\conf\conf.py .\code\agent_ppo\conf\conf.py -Force
Copy-Item .\agent_ppo\feature\preprocessor.py .\code\agent_ppo\feature\preprocessor.py -Force

# 可选：固定项目名，避免容器脚本解析地址失败
Set-Location .\dev
docker compose -p kaiwu-dev --env-file .env -f .docker-compose.yaml up -d
# 若需要环境容器，再加 profile（可能受镜像可用性影响）
# docker compose -p kaiwu-dev --env-file .env -f .docker-compose.yaml --profile packing up -d
```

## 1. 使用说明

本清单用于把 [plan.md](./plan.md) 转换为可直接执行的实施手册。  
每个环节统一包含：
- 设计目标
- 设计思路
- 执行步骤
- 执行指令
- 验证方法

执行原则：
- 每次只改一个主因素，避免无法归因。
- 每个阶段必须先完成验证，再进入下一阶段。
- 所有实验必须保留记录。

## 2. Phase 1：基础修复（P0）

## 2.1 环节一：修复配置维度

### 设计目标
让 PPO 配置与环境真实动作空间、观测空间一致，消除致命配置错误。

### 设计思路
当前仓库（`E:\competition\26fwwb`）中，P0 配置已修正到 `ACTION_NUM=8`、`DIM_OF_OBSERVATION=470`。  
本环节主要用于复核配置与后续改动的一致性。

### 执行步骤
1. 打开 `agent_ppo/conf/conf.py`
2. 将动作维度改为 8
3. 将观测维度改为 470
4. 同步检查 `FEATURES`、`FEATURE_SPLIT_SHAPE`、`FEATURE_LEN`

### 执行指令
```powershell
Get-Content -LiteralPath E:\competition\26fwwb\agent_ppo\conf\conf.py
```

修改完成后可用以下命令快速复核：
```powershell
Select-String -Path E:\competition\26fwwb\agent_ppo\conf\conf.py -Pattern "ACTION_NUM|DIM_OF_OBSERVATION|FEATURE"
```

### 验证方法
- 确认 `ACTION_NUM = 8`
- 确认 `DIM_OF_OBSERVATION = 470`
- 确认 `FEATURE_LEN` 与观测维度一致

### 验证标准
- 配置字段无冲突
- 后续模型初始化不会因为输入维度报错

## 2.2 环节二：实现特征工程

### 设计目标
将环境原始观测转成可学习的结构化输入，让 PPO 能感知地图、风险、资源和历史行为。

### 设计思路
采用方案02的 470 维特征：
- 地图平铺特征 441 维
- 电量、步数、位置、清扫进度 5 维
- 充电桩距离 4 维
- NPC 距离 4 维
- NPC 方向 8 维
- 上一步动作 8 维

设计原则：
- 优先保证信息正确，其次再优化表达方式
- 全部特征归一化，避免数值尺度差异过大
- 缺失 NPC/充电桩时要补零，保证维度固定

### 执行步骤
1. 打开 `agent_ppo/feature/preprocessor.py`
2. 保留 `Preprocessor` 结构，替换 blank feature 逻辑
3. 读取 `env_obs["map_info"]` 并 flatten
4. 读取角色状态、电量、步数、位置、得分进度
5. 计算到充电桩的距离特征，长度固定为 4
6. 计算 NPC 距离和方向特征，长度固定为 12
7. 构造 last action one-hot，长度为 8
8. 拼接成固定 470 维向量
9. 读取 `env_obs["legal_act"]` 作为合法动作掩码

### 执行指令
查看文件：
```powershell
Get-Content -LiteralPath E:\competition\26fwwb\agent_ppo\feature\preprocessor.py
```

静态检查关键字段：
```powershell
Select-String -Path E:\competition\26fwwb\agent_ppo\feature\preprocessor.py -Pattern "map_info|legal_action|legal_act|battery|organs|npcs|last_action"
```

### 验证方法
- 打印或临时记录特征长度
- 检查特征中是否存在 NaN / Inf
- 检查 `legal_action` 长度是否为 8
- 缺失 NPC 或 organ 时仍能正常返回固定维度

### 验证标准
- `len(feature) == 470`
- `len(legal_action) == 8`
- 特征非全零
- 不因空列表或字段缺失崩溃

## 2.3 环节三：实现奖励函数

### 设计目标
让智能体获得足够清晰的学习信号，兼顾清扫收益、生存能力和电量管理。

### 设计思路
奖励由五部分组成：
- 主奖励：清扫新污渍
- 时间惩罚：抑制无效移动
- 低电量充电引导：避免远离充电桩
- NPC 安全惩罚：避免贴近危险源
- 终局惩罚：对失败结局做明显负反馈

设计原则：
- 主目标权重最高
- 辅助奖励只负责“引导”，不能压过主奖励
- 奖励范围要可控，避免梯度极端波动

### 执行步骤
1. 在 `preprocessor.py` 中增加 reward 计算逻辑
2. 使用 `step_cleaned_cells` 或等价字段计算本步清扫收益
3. 每步固定减去小惩罚
4. 在低电量状态下，计算与最近充电桩的关系
5. 在接近 NPC 时增加惩罚
6. 在终止局面加入失败惩罚
7. reward 返回格式保持与训练流程兼容

### 执行指令
查看奖励相关字段：
```powershell
Select-String -Path E:\competition\26fwwb\agent_ppo\feature\preprocessor.py -Pattern "reward|terminated|truncated|step_cleaned_cells"
```

### 验证方法
- 运行时检查 reward 是否有波动
- 验证清扫成功时 reward 增加
- 验证靠近 NPC 或失败时 reward 下降
- 验证 reward 不是恒定值

### 验证标准
- reward 非零且有正负变化
- 清扫主奖励贡献清晰
- 失败终局能被明显区分

## 2.4 环节四：P0 冒烟运行

### 设计目标
确认“配置 + 特征 + 奖励”三者已经形成最小可训练闭环。

### 设计思路
先不追求高分，只验证训练链路正确、日志有反馈、行为开始偏离纯随机。

### 执行步骤
1. 进入仓库根目录 `E:\competition\26fwwb`
2. 运行 `train_test.py`
3. 观察是否有模型初始化错误、维度错误、动作越界
4. 观察 reward、loss、episode 是否正常输出
5. 保存本轮日志并记录结论

### 执行指令
```powershell
Set-Location E:\competition\26fwwb
python train_test.py
```

### 验证方法
- 检查程序是否持续运行
- 检查是否出现非零 reward
- 检查是否出现 loss 数值更新
- 检查是否有维度不匹配、索引越界、非法动作错误

### 验证标准
- 训练可持续运行
- 无关键异常
- 具备继续进入 P1 的前提

## 3. Phase 2：模型与超参优化（P1）

## 3.1 环节一：增强 MLP 模型

### 设计目标
提升网络表达能力，让 PPO 能更好吸收 470 维输入特征。

### 设计思路
先走低风险路线，优先增强 MLP，不直接切换复杂结构。  
推荐结构：
- 输入：470
- 隐层：256
- 中间层：128
- 输出：Actor 8，Critic 1

可加入：
- `LayerNorm`
- `Dropout(0.1)`（可选）

### 执行步骤
1. 打开 `agent_ppo/model/model.py`
2. 将 backbone 从 `64 -> 32` 扩展到 `256 -> 128`
3. 视训练稳定性增加 LayerNorm
4. 保持 actor/critic 双头输出结构不变

### 执行指令
```powershell
Get-Content -LiteralPath E:\competition\26fwwb\agent_ppo\model\model.py
```

复核关键层：
```powershell
Select-String -Path E:\competition\26fwwb\agent_ppo\model\model.py -Pattern "Linear|LayerNorm|Dropout|actor_head|critic_head"
```

### 验证方法
- 运行训练，检查 loss 是否可正常回传
- 观察是否出现梯度爆炸或数值异常
- 对比 P0 的 reward 变化趋势

### 验证标准
- 模型可稳定初始化和训练
- 相比 P0，得分趋势更平滑或更快上升

## 3.2 环节二：超参数小步调优

### 设计目标
在不破坏稳定性的前提下，提高探索能力与收敛速度。

### 设计思路
优先调两个核心参数：
- 学习率 `INIT_LEARNING_RATE_START`
- 熵系数 `BETA_START`

建议实验矩阵：
- 学习率：`5e-4`、`7e-4`、`1e-3`
- 熵系数：`0.01`、`0.02`、`0.05`

### 执行步骤
1. 打开 `agent_ppo/conf/conf.py`
2. 每次只调整一个主参数
3. 固定训练时长和随机种子
4. 每组实验记录同样的评估指标

### 执行指令
查看配置：
```powershell
Get-Content -LiteralPath E:\competition\26fwwb\agent_ppo\conf\conf.py
```

运行单次实验：
```powershell
Set-Location E:\competition\26fwwb
python train_test.py
```

### 验证方法
- 对比平均得分
- 对比碰撞率、电量耗尽率
- 对比 entropy 是否快速塌缩

### 验证标准
- 至少选出 1 组优于 P0 的超参组合
- 不出现明显训练发散

## 3.3 环节三：P1 对比实验与结论固化

### 设计目标
从多个候选版本中确定 P1 最优基线。

### 设计思路
不要凭单次训练感受决策，必须用统一表格对比。

### 执行步骤
1. 选取 3 组模型/超参组合
2. 各自运行同样训练时长
3. 记录得分、成功率、碰撞率、电量耗尽率
4. 选择综合指标最优方案

### 执行指令
重复执行：
```powershell
Set-Location E:\competition\26fwwb
python train_test.py
```

### 验证方法
- 生成实验对比表
- 检查最优方案是否在多次运行下仍稳定

### 验证标准
- 明确选出 P1 固化版本
- 输出下一阶段输入基线

## 4. Phase 3：进阶优化与泛化（P2）

## 4.1 环节一：CNN+MLP 混合模型

### 设计目标
更有效利用地图空间结构，提高感知能力和泛化上限。

### 设计思路
将地图特征和标量特征分开编码：
- 地图分支：CNN 提取空间模式
- 标量分支：MLP 提取状态特征
- 融合后输出动作与价值

### 执行步骤
1. 在 `model.py` 中拆分地图输入与标量输入
2. 地图分支使用 Conv2D 编码
3. 标量分支使用小型 MLP
4. 两分支融合后接 actor/critic
5. 同步调整特征输入格式

### 执行指令
```powershell
Get-Content -LiteralPath E:\competition\26fwwb\agent_ppo\model\model.py
Get-Content -LiteralPath E:\competition\26fwwb\agent_ppo\feature\preprocessor.py
```

### 验证方法
- 检查张量 shape 是否一致
- 检查 forward 是否成功
- 比较 P1 与 P2 的多地图表现

### 验证标准
- 不出现 shape mismatch
- 多地图平均成绩优于 P1

## 4.2 环节二：课程学习

### 设计目标
降低训练难度，使策略逐步学会从简单场景迁移到复杂场景。

### 设计思路
先简单后复杂：
- 单图 / 低机器人数量 / 较短步数
- 多图 / 高机器人数量 / 标准步数

### 执行步骤
1. 梳理环境配置中可控制的地图与难度参数
2. 设计课程阶段表
3. 每个阶段训练完成后再进入下一阶段

### 执行指令
查看环境配置：
```powershell
Get-Content -LiteralPath E:\competition\26fwwb\agent_ppo\conf\train_env_conf.toml
```

运行训练：
```powershell
Set-Location E:\competition\26fwwb
python train_test.py
```

### 验证方法
- 检查每阶段平均得分是否提升
- 检查训练是否在切换阶段后崩溃

### 验证标准
- 课程训练优于直接全难度训练

## 4.3 环节三：多地图评估

### 设计目标
判断模型是否真正具备泛化能力，而不是只在局部地图上过拟合。

### 设计思路
输出逐地图指标，而不是只看整体均值。

### 执行步骤
1. 固定评估流程
2. 分地图统计平均得分、失败率、存活步数
3. 计算均值与方差
4. 输出评估报告

### 执行指令
```powershell
Set-Location E:\competition\26fwwb
python train_test.py
```

### 验证方法
- 检查是否存在“单图高分、多图崩溃”
- 检查跨地图方差是否下降

### 验证标准
- 泛化表现优于 P1
- 结果具备稳定性

## 5. Phase 4：冲刺优化（P3）

## 5.1 环节一：超参数网格搜索

### 设计目标
在已稳定的模型架构上进一步压榨性能上限。

### 设计思路
只在有限范围内搜索，避免资源浪费。

### 执行步骤
1. 固定最优结构
2. 选择 2-3 个核心参数建立小规模网格
3. 汇总结果，保留最优组

### 执行指令
```powershell
Set-Location E:\competition\26fwwb
python train_test.py
```

### 验证方法
- 对比各组实验指标表

### 验证标准
- 找到最终候选参数组合

## 5.2 环节二：探索增强试点

### 设计目标
提高对未探索区域的覆盖能力。

### 设计思路
RND 和 ICM 只试一个，避免一次性引入两个变量。

### 执行步骤
1. 先保持现有主奖励不变
2. 引入一种探索增强模块
3. 观察是否提高远区域探索和总体得分

### 执行指令
无统一固定命令，按实现后执行：
```powershell
Set-Location E:\competition\26fwwb
python train_test.py
```

### 验证方法
- 检查探索范围是否扩大
- 检查得分是否实质提升

### 验证标准
- 增益成立才保留，否则回滚

## 5.3 环节三：鲁棒性回归

### 设计目标
确保冲刺优化没有破坏已有稳定性。

### 设计思路
冲刺优化必须经过回归验证，不能只看单次峰值。

### 执行步骤
1. 用最终候选版本回跑历史关键场景
2. 比较与 P1/P2 基线的差异
3. 记录回归结论

### 执行指令
```powershell
Set-Location E:\competition\26fwwb
python train_test.py
```

### 验证方法
- 回归测试通过
- 无关键指标显著恶化

### 验证标准
- 版本可作为最终提交候选

## 6. 实验记录模板

每次实验必须记录：

```markdown
# 实验编号：exp_YYYYMMDD_XX

## 1. 目标
- 本次验证什么假设

## 2. 改动项
- 配置：
- 特征：
- 奖励：
- 模型：
- 超参：

## 3. 执行指令
```powershell
cd E:\competition\26fwwb
python train_test.py
```

## 4. 结果
- 平均得分：
- 成功率：
- NPC碰撞率：
- 电量耗尽率：
- 平均存活步数：
- 备注：

## 5. 结论
- 是否保留：
- 下一步：
```

## 7. 今日执行顺序建议

1. 复核 `agent_ppo/conf/conf.py` 与 `agent_ppo/feature/preprocessor.py` 的 470D 与奖励逻辑
2. 开始 P1：增强 `agent_ppo/model/model.py`（`256 -> 128`，保持 actor/critic 双头）
3. 做超参小步实验（优先 `INIT_LEARNING_RATE_START`、`BETA_START`）
4. 在腾讯官方平台执行训练并记录结果（建议 `docs/experiments/exp_20260409_01.md`）
5. 固化一组优于当前基线的 P1 版本，再进入 P2

## 8. 完成定义（总验收）

满足以下条件后，认为本轮方案执行完成：
- 训练链路可跑通
- reward 非零且可学习
- 单地图分数较基线上升
- 多地图泛化不明显退化
- 所有实验有记录、可复现、可回滚
