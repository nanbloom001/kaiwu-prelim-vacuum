# 代码编写实施方案

基于最终技术方案（混合规则 + 分层规划 + PPO 微调的清扫架构），制定如下代码实施计划。

---

## 1. 项目目录结构

```
agent_diy/
├── __init__.py
├── config.py                    # 全局配置参数
├── perception/                  # 感知层
│   ├── __init__.py
│   ├── feature_pyramid.py       # 多尺度局部特征编码
│   └── coarse_map.py            # 32×32全局粗网格管理
├── planning/                    # 规划层
│   ├── __init__.py
│   ├── region_manager.py        # 区域分割与管理
│   ├── path_planner.py          # A* / BFS 路径规划
│   └── lookahead_planner.py     # 前瞻规划（3-5步）
├── decision/                    # 决策层
│   ├── __init__.py
│   ├── mode_selector.py         # 模式优先队列管理
│   └── reward_shaper.py         # 自适应奖励设计
├── navigation/                  # 导航控制层
│   ├── __init__.py
│   ├── unstuck_handler.py       # 脱困处理
│   └── return_to_charger.py     # 低电返航
└── learning/                    # 学习层（Phase3）
    ├── __init__.py
    ├── model.py                 # CNN+MLP 融合网络
    ├── behavior_clone.py        # 行为克隆预训练
    └── ppo_trainer.py           # PPO 微调训练
```

---

## 2. 模块实施顺序与依赖关系

```
Phase 1: 基础设施
    ├── config.py
    ├── perception/feature_pyramid.py
    └── perception/coarse_map.py
        
Phase 2: 规划与决策核心
    ├── planning/path_planner.py
    ├── planning/region_manager.py
    ├── navigation/return_to_charger.py
    ├── navigation/unstuck_handler.py
    └── decision/mode_selector.py
        
Phase 3: 智能增强
    ├── planning/lookahead_planner.py
    ├── decision/reward_shaper.py
    └── agent_diy/main_agent.py (整合)
        
Phase 4: 学习模块（可选增强）
    ├── learning/model.py
    ├── learning/behavior_clone.py
    └── learning/ppo_trainer.py
```

---

## 3. 各模块详细设计

### 3.1 config.py - 全局配置

**功能**：集中管理所有超参数，支持不同阶段的配置切换。

**核心配置项**：
```python
# 地图与感知参数
MAP_SIZE = 128
LOCAL_VIEW_SIZE = 21
COARSE_MAP_SIZE = 32
FEATURE_SCALES = [21, 11, 7]  # L0, L1, L2

# 资源约束
MAX_BATTERY = 200
MAX_STEPS = 2000
CHARGER_POS = (0, 0)

# 模式优先级（数值越大优先级越高）
MODE_PRIORITY = {
    'RETURN_TO_CHARGER': 100,
    'UNSTUCK': 90,
    'DIRECT_CLEANING': 70,
    'EXPLORATION': 50,
    'NEURAL_MICRO': 30
}

# 电量阈值
BATTERY_RETURN_THRESHOLD = 0.3  # 30%电量强制返航
BATTERY_LOW_THRESHOLD = 0.5     # 50%电量预警

# A* / BFS 参数
ASTAR_MAX_ITER = 10000
PATH_CACHE_SIZE = 50

# 前瞻规划参数
LOOKAHEAD_DEPTH = 5
LOOKAHEAD_GAMMA = 0.95

# 训练参数（Phase3）
PPO_CONFIG = {
    'clip_epsilon': 0.2,
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'vf_coef': 0.5,
    'entropy_coef': 0.01,
    'lr': 3e-4,
    'grad_clip': 0.5
}
```

---

### 3.2 perception/feature_pyramid.py - 多尺度局部特征

**功能**：从原始 21×21 局部视野提取多尺度特征。

**接口设计**：
```python
class FeaturePyramid:
    def __init__(self, scales=[21, 11, 7]):
        """
        初始化多尺度特征提取器
        Args:
            scales: 三个尺度的视野大小 [L0, L1, L2]
        """
        pass
    
    def extract(self, local_view: np.ndarray, robot_pos: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """
        提取多尺度特征
        Args:
            local_view: 21×21 的局部视野数组
            robot_pos: 机器人在局部视野中的位置 (x, y)
        Returns:
            {
                'L0': np.ndarray(shape=(21, 21, channels)),  # 精细层
                'L1': np.ndarray(shape=(11, 11, channels)),  # 中尺度
                'L2': np.ndarray(shape=(7, 7, channels)),    # 粗粒度
                'features': np.ndarray(shape=(160,))         # 展平融合特征
            }
        """
        pass
```

**实现要点**：
- L0(21×21): 完整局部视野，用于精细避障和近距离清扫
- L1(11×11): 中心裁剪，用于中尺度目标选择
- L2(7×7): 进一步裁剪，用于姿态估计与安全边界判断
- 每个尺度提取：障碍通道、脏点通道、已访问通道、距离变换

---

### 3.3 perception/coarse_map.py - 全局粗网格

**功能**：维护 32×32 全局粗网格地图，存储统计信息。

**接口设计**：
```python
class CoarseMap:
    def __init__(self, map_size=128, coarse_size=32):
        """
        初始化全局粗网格
        Args:
            map_size: 原始地图大小 128
            coarse_size: 粗网格大小 32
        """
        pass
    
    def update(self, robot_pos: Tuple[int, int], local_view: np.ndarray):
        """
        根据当前局部视野更新粗网格（指数滑动平均）
        Args:
            robot_pos: 机器人在全局地图中的位置
            local_view: 21×21 局部视野
        """
        pass
    
    def get_region_stats(self, region_idx: int) -> Dict:
        """
        获取指定区域的统计信息
        Returns:
            {
                'obstacle_density': float,  # 障碍密度
                'dirt_density': float,      # 脏点密度
                'visit_freq': float,        # 访问频率
                'frontier_score': float,    # 前沿探索得分
                'danger_level': float       # 危险等级（远离充电站+障碍多）
            }
        """
        pass
    
    def get_cell_info(self, coarse_x: int, coarse_y: int) -> Dict:
        """获取指定粗网格单元的信息"""
        pass
    
    def get_state_vector(self) -> np.ndarray:
        """
        获取全局状态向量（展平所有粗网格信息）
        Returns: np.ndarray(shape=(32*32*5,))
        """
        pass
```

**实现要点**：
- 比例尺：128×128 原始地图 → 32×32 粗网格（每格 4×4）
- 更新时使用指数滑动平均：new_value = α * old + (1-α) * observed
- 定期衰减 visit_freq 以支持重探索

---

### 3.4 planning/path_planner.py - 路径规划

**功能**：A* / BFS 路径规划，支持启发式优化。

**接口设计**：
```python
class PathPlanner:
    def __init__(self, coarse_map: CoarseMap):
        self.coarse_map = coarse_map
    
    def astar(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int],
        visit_penalty: float = 0.1,
        max_iter: int = 10000
    ) -> List[Tuple[int, int]]:
        """
        A*路径规划
        Args:
            start: 起点 (x, y)
            goal: 终点 (x, y)
            visit_penalty: 对高访问频率区域的惩罚系数
            max_iter: 最大迭代次数
        Returns:
            路径点列表，失败返回空列表
        """
        pass
    
    def bfs(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int],
        max_depth: int = 100
    ) -> List[Tuple[int, int]]:
        """
        BFS回退路径规划（当A*失败时使用）
        """
        pass
    
    def get_action_from_path(
        self, 
        path: List[Tuple[int, int]], 
        robot_pos: Tuple[int, int],
        legal_actions: List[int]
    ) -> int:
        """
        从路径中提取下一步动作
        Args:
            path: 路径点列表
            robot_pos: 当前位置
            legal_actions: 合法动作列表 [0-7]
        Returns:
            动作编号 0-7，失败返回 None
        """
        pass
```

**启发式函数**：
```python
def heuristic(pos, goal, coarse_map):
    # 欧氏距离
    dist = euclidean_distance(pos, goal)
    # 访问频率惩罚
    cell = coarse_map.get_cell_at(pos)
    penalty = 0.1 * cell.visit_freq
    return dist + penalty
```

---

### 3.5 planning/region_manager.py - 区域管理

**功能**：将地图划分为 4×4=16 个区域，动态选择目标区域。

**接口设计**：
```python
class RegionManager:
    def __init__(self, map_size=128, region_split=4):
        """
        初始化区域管理器
        Args:
            map_size: 地图大小
            region_split: 每边分割数，4表示4×4=16个区域
        """
        pass
    
    def pos_to_region(self, pos: Tuple[int, int]) -> int:
        """将全局坐标转换为区域索引 0-15"""
        pass
    
    def region_to_center(self, region_idx: int) -> Tuple[int, int]:
        """获取区域中心坐标"""
        pass
    
    def score_regions(self, coarse_map: CoarseMap, robot_pos: Tuple[int, int]) -> List[Tuple[int, float]]:
        """
        计算所有区域的得分并排序
        得分 = dirt_density×0.5 + (1-visit_freq)×0.3 + frontier_score×0.2 - distance_penalty
        Returns:
            [(region_idx, score), ...] 按得分降序排列
        """
        pass
    
    def select_best_region(
        self, 
        coarse_map: CoarseMap, 
        robot_pos: Tuple[int, int],
        battery_ratio: float
    ) -> Tuple[int, Tuple[int, int]]:
        """
        选择最佳目标区域
        Returns:
            (region_idx, target_pos)
        """
        pass
```

---

### 3.6 navigation/return_to_charger.py - 低电返航

**功能**：电量不足时安全返回充电站。

**接口设计**：
```python
class ReturnToCharger:
    def __init__(self, charger_pos=(0, 0)):
        self.charger_pos = charger_pos
    
    def should_return(
        self, 
        battery: int, 
        robot_pos: Tuple[int, int],
        coarse_map: CoarseMap
    ) -> bool:
        """
        判断是否应该返航
        考虑因素：当前电量、返回所需距离、安全余量
        """
        pass
    
    def get_return_path(
        self, 
        robot_pos: Tuple[int, int],
        path_planner: PathPlanner
    ) -> List[Tuple[int, int]]:
        """规划返航路径"""
        pass
    
    def estimate_return_cost(
        self, 
        robot_pos: Tuple[int, int],
        coarse_map: CoarseMap
    ) -> int:
        """估计返航所需步数"""
        pass
```

**返航触发条件**：
- 硬约束：`battery <= distance_to_charger + safety_margin`
- 软约束：`battery_ratio < BATTERY_RETURN_THRESHOLD (30%)`

---

### 3.7 navigation/unstuck_handler.py - 脱困处理

**功能**：检测卡死并执行脱困动作。

**接口设计**：
```python
class UnstuckHandler:
    def __init__(self, history_size=10):
        self.position_history = deque(maxlen=history_size)
        self.stuck_threshold = 5  # 5步内移动距离小于阈值则认为卡住
    
    def update(self, robot_pos: Tuple[int, int]):
        """更新位置历史"""
        pass
    
    def is_stuck(self) -> bool:
        """检测是否卡住"""
        pass
    
    def get_unstuck_action(
        self, 
        local_view: np.ndarray,
        legal_actions: List[int]
    ) -> int:
        """
        获取脱困动作
        策略：尝试随机合法动作、回退、旋转等
        """
        pass
    
    def reset(self):
        """重置状态"""
        pass
```

**卡死检测逻辑**：
- 最近N步位置变化很小（欧氏距离 < 2）
- 连续多次执行相同动作但位置未变
- 长时间未清扫到新脏点

---

### 3.8 decision/mode_selector.py - 模式选择器

**功能**：管理清扫模式的优先队列。

**接口设计**：
```python
class ModeSelector:
    """
    模式优先级（从高到低）：
    1. RETURN_TO_CHARGER - 电量不足强制返航
    2. UNSTUCK - 卡死脱困
    3. DIRECT_CLEANING - 视野内有脏点直接清扫
    4. EXPLORATION - 探索前沿区域
    5. NEURAL_MICRO - 神经网络微调（Phase3）
    """
    
    def __init__(self):
        self.current_mode = 'EXPLORATION'
        self.mode_cooldown = {}  # 模式切换冷却时间
    
    def select_mode(
        self,
        battery: int,
        robot_pos: Tuple[int, int],
        local_view: np.ndarray,
        coarse_map: CoarseMap,
        unstuck_handler: UnstuckHandler,
        return_handler: ReturnToCharger
    ) -> str:
        """
        根据当前状态选择最佳模式
        Returns: 模式名称字符串
        """
        pass
    
    def has_dirt_in_view(self, local_view: np.ndarray) -> bool:
        """检查视野内是否有脏点"""
        pass
    
    def get_nearest_dirt_pos(
        self, 
        local_view: np.ndarray,
        robot_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """获取视野内最近的脏点位置"""
        pass
```

---

### 3.9 planning/lookahead_planner.py - 前瞻规划

**功能**：在关键决策点进行 3-5 步前瞻规划。

**接口设计**：
```python
class LookaheadPlanner:
    def __init__(self, depth=5, gamma=0.95):
        self.depth = depth
        self.gamma = gamma
    
    def evaluate_action_sequence(
        self,
        actions: List[int],
        current_state: Dict,
        coarse_map: CoarseMap
    ) -> float:
        """
        评估动作序列的长期价值
        使用简单模型模拟执行，累加折扣奖励
        """
        pass
    
    def lookahead_plan(
        self,
        current_state: Dict,
        coarse_map: CoarseMap,
        path_planner: PathPlanner,
        depth: int = None
    ) -> Tuple[int, float]:
        """
        前瞻规划选择最佳动作
        Returns:
            (best_action, expected_value)
        """
        pass
    
    def should_trigger_lookahead(
        self, 
        mode: str, 
        step_count: int,
        last_trigger_step: int
    ) -> bool:
        """
        判断是否触发前瞻规划（控制算力）
        触发条件：模式切换、每N步、目标区域切换
        """
        pass
```

---

### 3.10 decision/reward_shaper.py - 自适应奖励设计

**功能**：根据训练阶段动态调整奖励 shaping。

**接口设计**：
```python
class RewardShaper:
    def __init__(self):
        self.phase = 'EXPLORATION'  # EXPLORATION / BALANCE / CONVERGENCE
        self.phase_thresholds = [0.33, 0.67]  # 阶段切换阈值
    
    def update_phase(self, clean_progress: float):
        """根据清扫进度更新阶段"""
        pass
    
    def shape_reward(
        self,
        raw_reward: float,
        new_dirt_count: int,
        old_dirt_dist: float,
        new_dirt_dist: float,
        battery_ratio: float,
        action_legal: bool,
        is_stuck: bool,
        step_count: int
    ) -> float:
        """
        计算 shaping 后的奖励
        
        基础奖励：+0.5 × 新清扫脏点数
        阶段惩罚：探索期 -0.001/步，平衡期 -0.005/步，收敛期 -0.01/步
        接近奖励：0.01 × (旧脏点距离 - 新距离)
        电量惩罚：battery < 阈值时 -0.05
        违规惩罚：撞障、非法动作、卡死
        """
        pass
    
    def get_step_penalty(self) -> float:
        """获取当前阶段的每步惩罚"""
        penalties = {
            'EXPLORATION': -0.001,
            'BALANCE': -0.005,
            'CONVERGENCE': -0.01
        }
        return penalties[self.phase]
```

---

### 3.11 learning/model.py - 神经网络模型（Phase3）

**功能**：Actor-Critic 网络，融合多尺度视觉特征与全局标量。

**接口设计**：
```python
class CleaningAgentNet(nn.Module):
    """
    清扫智能体网络结构
    
    输入：
    - L0: (batch, 3, 21, 21) - 精细局部视野
    - L1: (batch, 3, 11, 11) - 中尺度视野
    - L2: (batch, 3, 7, 7) - 粗粒度视野
    - scalar: (batch, 10) - 标量状态向量
    
    输出：
    - policy_logits: (batch, 8) - 8个动作的对数概率
    - value: (batch, 1) - 状态价值估计
    """
    
    def __init__(self):
        super().__init__()
        # 三个尺度的CNN编码器
        self.encoder_l0 = self._make_encoder(3, 32, 3)  # 21x21 -> 7x7
        self.encoder_l1 = self._make_encoder(3, 32, 2)  # 11x11 -> 5x5
        self.encoder_l2 = self._make_encoder(3, 32, 1)  # 7x7 -> 5x5
        
        # 融合层：CNN特征 + 标量特征 -> 160维
        self.fusion = nn.Sequential(
            nn.Linear(32*7*7 + 32*5*5 + 32*5*5 + 10, 256),
            nn.ReLU(),
            nn.Linear(256, 160),
            nn.ReLU()
        )
        
        # Actor头
        self.actor = nn.Linear(160, 8)
        
        # Critic头
        self.critic = nn.Linear(160, 1)
        
        # 正交初始化
        self.apply(self._ortho_init)
    
    def forward(self, l0, l1, l2, scalar):
        """前向传播"""
        pass
    
    def get_action_and_value(self, obs, legal_mask=None, action=None):
        """
        获取动作和价值，用于PPO训练
        Args:
            legal_mask: (batch, 8) 合法动作掩码
            action: 实际执行的动作（用于计算log_prob）
        """
        pass
```

---

### 3.12 agent_diy/main_agent.py - 主智能体

**功能**：整合所有模块，实现完整的决策流程。

**接口设计**：
```python
class CleaningAgent:
    """
    清扫智能体主类
    
    决策流程：
    1. 更新感知（局部特征 + 全局粗网格）
    2. 检测卡死/返航需求
    3. 模式选择（优先队列）
    4. 目标区域选择（区域管理器）
    5. 路径规划（A* / BFS）
    6. 前瞻规划（可选，关键决策点）
    7. 执行动作并更新状态
    """
    
    def __init__(self):
        # 初始化所有模块
        self.feature_pyramid = FeaturePyramid()
        self.coarse_map = CoarseMap()
        self.path_planner = PathPlanner(self.coarse_map)
        self.region_manager = RegionManager()
        self.return_handler = ReturnToCharger()
        self.unstuck_handler = UnstuckHandler()
        self.mode_selector = ModeSelector()
        self.lookahead = LookaheadPlanner()
        self.reward_shaper = RewardShaper()
        
        # 状态跟踪
        self.current_mode = 'EXPLORATION'
        self.target_region = None
        self.current_path = []
        self.step_count = 0
    
    def act(self, obs: Dict) -> int:
        """
        主决策函数
        Args:
            obs: 环境观测 {
                'agent_maps': np.ndarray,  # 局部视野
                'agent_pos': Tuple[int, int],
                'battery': int,
                'step': int,
                ...
            }
        Returns:
            action: 0-7 的动作编号
        """
        # 1. 更新感知
        # 2. 模式选择
        # 3. 模式对应的决策逻辑
        # 4. 返回动作
        pass
    
    def _mode_return_to_charger(self) -> int:
        """返航模式决策"""
        pass
    
    def _mode_unstuck(self) -> int:
        """脱困模式决策"""
        pass
    
    def _mode_direct_cleaning(self) -> int:
        """直接清扫模式决策"""
        pass
    
    def _mode_exploration(self) -> int:
        """探索模式决策"""
        pass
```

---

## 4. 实施阶段计划

### Phase 1: 基础设施搭建（预计 2-3 天）

**目标**：实现基础感知与规划能力

**任务清单**：
- [ ] `config.py` - 全局配置
- [ ] `perception/feature_pyramid.py` - 多尺度特征提取
- [ ] `perception/coarse_map.py` - 全局粗网格
- [ ] `planning/path_planner.py` - A*/BFS 路径规划
- [ ] 编写单元测试验证基础功能

**验收标准**：
- 能够从局部视野正确提取多尺度特征
- A*路径规划在简单地图上正确运行
- 粗网格能够正确统计和更新

---

### Phase 2: 核心决策逻辑（预计 3-4 天）

**目标**：实现完整的模式切换和区域管理

**任务清单**：
- [ ] `planning/region_manager.py` - 区域管理
- [ ] `navigation/return_to_charger.py` - 返航逻辑
- [ ] `navigation/unstuck_handler.py` - 脱困处理
- [ ] `decision/mode_selector.py` - 模式选择
- [ ] `agent_diy/main_agent.py` - 基础版本（无前瞻、无学习）
- [ ] 集成测试

**验收标准**：
- 模式切换逻辑正确，优先级符合设计
- 返航逻辑能够在电量不足时安全返回
- 脱困功能能够识别并处理卡死
- 区域选择能够覆盖未清扫区域

---

### Phase 3: 高级规划与奖励（预计 2-3 天）

**目标**：增加前瞻规划和自适应奖励

**任务清单**：
- [ ] `planning/lookahead_planner.py` - 前瞻规划
- [ ] `decision/reward_shaper.py` - 自适应奖励
- [ ] 更新 `main_agent.py` 集成前瞻规划
- [ ] 性能优化（控制计算开销）
- [ ] 全面测试与调优

**验收标准**：
- 前瞻规划在关键决策点触发
- 计算开销在可接受范围内
- 奖励 shaping 能够正确反映训练阶段

---

### Phase 4: 学习模块（Phase3，预计 4-5 天）

**目标**：实现 PPO 微调能力

**任务清单**：
- [ ] `learning/model.py` - 神经网络模型
- [ ] `learning/behavior_clone.py` - 行为克隆
- [ ] `learning/ppo_trainer.py` - PPO 训练
- [ ] 数据收集脚本（规则策略生成轨迹）
- [ ] 训练流程整合
- [ ] 模型评估与迭代

**验收标准**：
- 行为克隆能够收敛
- PPO 训练稳定，奖励提升
- 微调后的策略性能优于纯规则

---

## 5. 接口约定

### 5.1 与环境交互的观测格式

```python
obs = {
    # 局部视野 (21, 21, 3)
    # 通道0: 障碍 (0=空, 1=障碍, 2=未知)
    # 通道1: 脏点 (0=无, 1=有脏)
    # 通道2: 已访问 (0=未访问, 1=已访问, 2=充电站)
    'agent_maps': np.ndarray,
    
    # 机器人在全局地图中的位置
    'agent_pos': (x, y),
    
    # 当前电量 0-200
    'battery': int,
    
    # 当前步数
    'step': int,
    
    # 合法动作掩码 (8,) bool数组
    'legal_actions': np.ndarray,
    
    # 其他环境信息...
}
```

### 5.2 动作编号定义

```
0: 上 (0, -1)
1: 右上 (1, -1)
2: 右 (1, 0)
3: 右下 (1, 1)
4: 下 (0, 1)
5: 左下 (-1, 1)
6: 左 (-1, 0)
7: 左上 (-1, -1)
```

### 5.3 模块间通信

- 使用数据类或字典传递状态信息
- 避免模块间的循环依赖
- 关键接口提供类型注解

---

## 6. 测试策略

### 6.1 单元测试

每个模块独立测试：
```python
# test_feature_pyramid.py
def test_multiscale_extraction():
    fp = FeaturePyramid()
    view = np.random.randint(0, 3, (21, 21, 3))
    features = fp.extract(view, (10, 10))
    assert features['L0'].shape == (21, 21, 3)
    assert features['L1'].shape == (11, 11, 3)
    assert features['L2'].shape == (7, 7, 3)
```

### 6.2 集成测试

测试模块间协作：
```python
def test_full_pipeline():
    agent = CleaningAgent()
    # 模拟环境交互
    for step in range(100):
        obs = env.get_obs()
        action = agent.act(obs)
        env.step(action)
```

### 6.3 性能测试

确保满足计算约束：
```python
def test_decision_time():
    agent = CleaningAgent()
    obs = create_test_obs()
    start = time.time()
    for _ in range(1000):
        action = agent.act(obs)
    avg_time = (time.time() - start) / 1000
    assert avg_time < 0.01  # 每步决策 < 10ms
```

---

## 7. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| 计算超时 | 无法及时决策 | 限制A*迭代次数、前瞻深度，添加超时回退 |
| 内存溢出 | 程序崩溃 | 限制历史记录长度、使用生成器、定期清理缓存 |
| 模式切换震荡 | 效率低下 | 添加模式冷却时间、滞后切换机制 |
| 地图漂移 | 规划错误 | 指数平滑、定期重校准、置信度阈值 |
| 奖励 shaping 过拟合 | 泛化差 | 分阶段关闭 shaping、对齐环境原始奖励 |

---

## 8. 性能优化建议

1. **路径缓存**：缓存常用路径（如返回充电站），避免重复计算
2. **延迟更新**：粗网格不必每帧更新，可每N帧或移动一定距离后更新
3. **剪枝搜索**：A*搜索时设置最大节点扩展数，超限后回退BFS
4. **Numba加速**：关键计算（如A*）可使用Numba JIT加速
5. **向量化操作**：尽量使用NumPy向量化操作代替Python循环

---

## 9. 文档维护

- 每个模块需包含：功能说明、接口定义、使用示例
- 关键算法需添加注释说明原理
- 配置变更需同步更新此文档
