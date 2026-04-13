# DIY 当前策略文档

最后更新时间：2026-04-12

这份文档用于记录 `kaiwu2` 中 `agent_diy` 的当前运行流程、规划策略、训练设计和关键约束。
它是当前版本的统一说明入口。后续只要代码里的流程、策略或算法发生变化，都应同步更新这份文档。

## 1. 总体架构

当前方案是一个 `分层强化学习 + 规则执行器` 的混合系统。

- 规则层负责：
  - 世界模型与全局地图记忆
  - 拓扑提取
  - `room / tail / bridge` 区域识别
  - 充电桩相关推理
  - BFS / A* 路径搜索
  - NPC 硬避让与软风险场
  - 低层动作合法性判断与最终动作打分
- RL 层负责：
  - 在候选决策中做高层选择
  - 选择 `path_style`
  - 学习何时探索、何时深扫、何时返航、何时脱困

当前配置仍保留 `TEACHER_ONLY = True`，所以默认推理仍是教师规则优先。学生网络和训练链路已经接好，后续可以切换到训练版本。

## 2. 运行流程

主流程如下：

1. `agent.py::observation_process`
   - 规范化环境观测
   - 调用 `preprocessor.feature_process`
   - 打包状态特征、候选特征、候选 mask、教师候选、教师路径风格等信息
2. `preprocessor.py::feature_process`
   - 解析环境输入
   - 更新全局地图、访问统计、清扫统计、充电桩、NPC 和记忆状态
   - 在需要时更新拓扑和区域
   - 构造低层合法动作
   - 选择动作或触发高层决策事件
   - 构造模型输入特征
   - 构造 reward
3. `agent.py::predict`
   - 如果当前不是决策事件，直接执行当前执行器动作
   - 如果是决策事件且 `TEACHER_ONLY = True`，直接使用教师候选和教师 `path_style`
   - 如果启用学生模型，则由模型选择候选和路径风格，再交给规则执行器生成真实路径和真实动作

## 3. 高层决策策略

旧版学生头是 `feature -> 8 个底层动作`。

当前已经改为：

- 输入：
  - 全局状态特征
  - 展平后的候选特征
  - 候选 mask
- 输出：
  - 候选决策选择
  - 路径风格选择
  - 状态价值

当前候选集合包括：

- `keep_plan`
- `top-K clean regions`
- `top-K explore mouths`
- `top-K charger choices`
- `fallback frontier`

当前路径风格包括：

- `aggressive_explore`
- `balanced`
- `deep_clean`
- `safe_return`
- `escape`

教师监督不是再靠“当前 goal 反推候选”，而是在每次 `_replan()` 后保存本次教师实际选中的高层候选类型、区域或充电桩，再在候选集中显式匹配。
这样可以避免 `keep_plan` 在决策事件里错误吞掉原本属于 `clean_region / explore_mouth / charger` 的 teacher 标签。

## 4. 候选生成器

候选仍然由 `preprocessor.py` 中的规则逻辑生成。

每个候选会携带一组结构化特征，例如：

- 候选类型
- dirty 质量 / frontier 质量 / 面积 / 深度
- entry 距离 / anchor 距离 / goal 距离
- 充电桩距离 / 返回预算 / 电量余量
- spine 占比
- NPC 风险
- blocked 风险
- `npc_cleaned` 占比
- 是否与当前活跃区域一致
- 是否属于 dirty / frontier / charger 类型
- 当前规则评分

这样做的好处是：

- 区域提取和路径安全仍由规则层稳定控制
- RL 只学习高层取舍，而不是从原始视野里硬学区域结构

## 5. 规划模式

当前规划器保留 3 个主模式：

- `explore`
- `clean_region`
- `return_charge`

三个模式的职责分别是：

- `explore`
  - 打开地图拓扑
  - 找 mouth、找新房间
  - 尽量用较少代价补全视野
- `clean_region`
  - 进入一个选定房间或尾支
  - 按较稳定的覆盖顺序清扫
  - 尽量不要半途离开
- `return_charge`
  - 优先保证安全返航
  - 压低充电桩附近和返航通道的浪费步数

## 6. 房间覆盖顺序

当前版本不再只依赖弱蛇形偏置。

对于 `room` 和 `tail` 区域，系统现在会显式构造 `cover_sequence`，并同步维护条带级状态。

核心思路是：

- 先根据区域主轴，把区域拆成若干条带
- 条带顺序从 mouth 向更深处排列
- 每条带内部再生成一个覆盖顺序
- 活跃区域会维护：
  - `active_cover_sequence`
  - `active_cover_index`
  - `active_cover_strip`
- 选区域内目标时，优先使用下一个“仍然有活可做”的覆盖点，而不是重新做通用点评分
- 低层动作打分时，再叠加：
  - 留在区域内的奖励
  - 朝下一条有效条带推进的奖励
  - 向深处推进的奖励
  - 过早离开区域的强惩罚
  - 房间内重复空刷的惩罚

这不是严格数学意义上的全局一笔画，而是一个更贴近比赛需求的“房间级有序覆盖”策略。

## 7. 充电桩协议

当前版本新增了明确的充电桩协议，而不是只靠一个笼统的“桩冷却”。

### 7.1 回桩阶段

在评估回桩候选时：

- 每个充电桩会先选择一个 `spoke` 方向，也就是 8 个方向中的一个
- `spoke` 的评分会优先考虑：
  - 前方未知 / frontier / dirty 更多
  - 重复经过更少
  - blocked 风险更低
  - NPC 风险更低
  - `npc_cleaned` 更低
- 选定 `spoke` 后，再从充电桩格子里选一个最符合该方向、同时代价较低的落桩点

这样做的目的，是让回桩不再是“随便从哪一边进去都行”，而是更有方向感，也更方便后续离桩。

### 7.2 离桩阶段

当机器人在充电桩内充满电后：

- 当前充电桩会被记录为 `launch_lock_source`
- 对应的 `spoke` 会成为本次离桩的优先射线
- 系统会启动 `launch_lock`
- 只有当机器人离这个充电桩至少 `CHARGER_LAUNCH_MIN_DIST` 格后，才会释放这层约束

在 `launch_lock` 期间：

- 优先选择“与充电桩距离严格变大”的动作
- 优先选择和 `spoke` 更一致的动作
- 优先选择更少重复、更少访问过的格子
- 被 `blocked_cells`、NPC 风险或 `npc_cleaned` 标记过重的位置会被压低

这相当于在规则层增加了一条硬协议：先把机器人从桩区弹出去，再恢复正常规划。

## 8. 桩区重复惩罚

旧逻辑主要依赖两件事：

- 同一充电桩级别的统一冷却
- 一个统一的 post-charge halo penalty

当前版本已经把它扩展成更细粒度的“按位置惩罚”，并与 post-charge 离桩协议叠加使用。

新的桩区重复惩罚会综合考虑：

- 该格到充电桩区域的距离
- 该格的 visit 次数
- 该格的 clean-pass 次数
- 局部重复密度
- 该格附近的几何结构

几何结构相关规则是：

- 贴墙的重复走位，惩罚更低
- 像角落那样的重复走位，惩罚更高
- 开放区的重复走位，介于二者之间

这样做是因为户型图里有些贴墙通道是返航必经，而角落磨蹭通常就是纯浪费。

## 9. 拓扑与区域提取

当前地图被拆成 4 类结构：

- `transit_spine`
- `room`
- `tail`
- `bridge`

核心逻辑是：

- 先用充电桩附近和已知走廊，构建 `transit_spine`
- 再在 `passable - transit_spine` 上生长区域
- `bridge` 默认不作为优先深扫区域
- `room` 和 `tail` 更适合作为主动清扫目标

这也是当前“走廊偏交通、房间偏得分”的主要实现基础。

## 10. NPC 与失败恢复

当前版本保留并融合了几类很实战的机制：

- `blocked_cells TTL`
  - 短时间内卡住、失败或不好走的位置会被临时降权
- `npc_cleaned`
  - 被官方机器人反复扫过的位置会被显式降权
- NPC 动态硬阻塞
- NPC 软风险场

这些量会进入：

- 候选打分
- 路径代价
- 动作打分
- 充电桩 `spoke` 选择

## 11. 路径搜索与执行器

低层路径生成仍然是规则驱动的。

- 全局路径：带权 A* / BFS
- 低层动作优先级：
  - 先看 `launch_lock`
  - 再看 `path commit`
  - 最后才用局部动作打分

这个分层很重要，因为它意味着：

- RL 不直接学习逐格路径
- RL 不直接学习精确回桩路线
- RL 不直接从原始视野里学房间切分

RL 学的是“当前该信哪个候选、该用哪种路径风格”，具体执行仍由规则层兜底。

## 12. 训练设计

当前高层学生模型采用“事件驱动采样”，而不是“每走一格就采一个训练样本”。

决策事件包括：

- 需要重新规划
- 当前目标完成
- 进入或退出充电模式
- 卡住恢复
- 软风险相对上次规划基线出现显著突增
- 区域完成或区域被放弃

每条高层样本会保存：

- 状态特征
- 候选特征
- 候选 mask
- 选中的候选动作
- 选中的路径风格
- 教师候选监督
- 教师路径风格监督
- value 相关目标

损失结构包括：

- 候选动作的 PPO 风格策略损失
- 路径风格的 PPO 风格策略损失
- value loss
- entropy 正则
- 对教师候选 / 教师风格的 imitation loss

训练监控指标同时兼容 PPO 面板口径，当前标准字段包括：

- `reward`
- `total_loss`
- `value_loss`
- `policy_loss`
- `entropy_loss`

其中 `policy_loss` 是高层 `decision_policy_loss + STYLE_POLICY_COEF * style_policy_loss` 的聚合口径，便于和 `agent_ppo` 的监控面板对齐；同时仍保留：

- `decision_policy_loss`
- `style_policy_loss`
- `imitation_loss`
- `teacher_mix`
- `imitation_coef`
- `teacher_weight`
- `policy_weight`

## 13. 奖励概览

当前有两层奖励。

低层 reward shaping：

- clean reward
- step penalty
- stuck penalty
- charge reward
- NPC danger penalty
- repeat clean penalty
- frontier reward

高层决策奖励：

- score delta reward
- efficiency reward
- region completion reward
- halo waste penalty
- spine waste penalty
- frontier skip penalty
- blocked penalty
- stuck penalty

## 14. 监控上报约束

当前训练与评测监控上报只允许标量数值。

这意味着：

- 直接送给 monitor / Prometheus 的字典必须是扁平的
- 不能把嵌套 `dict`、原始候选结构、复杂对象直接塞进监控字段
- 如果外部训练指标返回了嵌套结构，必须先做扁平化和标量化

当前实现已经在工作流监控层增加了监控字段清洗，避免 `dict` 类型直接进入 Prometheus。

## 15. 优先阅读的文件

- `code/agent_diy/feature/preprocessor.py`
- `code/agent_diy/model/model.py`
- `code/agent_diy/algorithm/algorithm.py`
- `code/agent_diy/workflow/train_workflow.py`
- `code/agent_diy/agent.py`
- `code/agent_diy/conf/conf.py`

## 16. 更新规则

只要以下任何一项发生变化，这份文档都应与代码同一次更新一起修改：

- 规划流程
- 候选策略
- 充电桩协议
- 房间覆盖顺序
- 训练目标
- 奖励设计
