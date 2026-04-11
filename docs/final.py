#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
清扫大作战 - 混合规则 + 分层规划 + PPO微调智能体
Robot Vacuum - Hybrid Rule-based + Hierarchical Planning + PPO Fine-tuning Agent

基于最终技术方案实现，包含：
- 多尺度局部特征金字塔 (L0:21x21, L1:11x11, L2:7x7)
- 32x32全局粗网格地图管理
- 4x4区域分割与动态目标选择
- A*/BFS路径规划
- 优先级队列模式管理 (返航>脱困>直接清扫>探索>神经网络微调)
- 前瞻规划 (3-5步)
- 自适应奖励塑形
- 低电返航与脱困处理
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from typing import Dict, List, Tuple, Optional, Set
import heapq
import math

# ==================== 全局配置 ====================

class Config:
    """全局配置参数"""
    # 地图与感知参数
    MAP_SIZE = 128
    LOCAL_VIEW_SIZE = 21
    COARSE_MAP_SIZE = 32
    FEATURE_SCALES = [21, 11, 7]  # L0, L1, L2
    
    # 资源约束
    MAX_BATTERY = 200
    MAX_STEPS = 1000
    
    # 模式优先级（数值越大优先级越高）
    MODE_PRIORITY = {
        'EMERGENCY_EVADE': 120,
        'RETURN_TO_CHARGER': 100,
        'EARLY_CHARGE': 95,
        'STUCK_RECOVERY': 90,
        'CHARGER_TRANSFER': 80,
        'STRIPE_CLEANING': 70,
        'DIRECT_DIRT_PICKUP': 60,
        'SAFE_EXPLORATION': 50,
        'NEURAL_MICRO': 30,
    }

    # 安全电量参数（动态护栏）
    DETOUR_MARGIN = 6
    UNKNOWN_MARGIN = 4
    DOCK_BUFFER = 8
    ACTION_MOVE_COST = 1
    EARLY_CHARGE_THRESHOLD = 0.92
    NEAR_CHARGE_TOPUP_RATIO = 0.96
    PRESSURE_TOPUP_RATIO = 0.94
    CHARGE_NEAR_DISTANCE = 20
    CHARGE_SAFE_DISTANCE = 18
    CHARGE_RING_DISTANCE = 14
    CHARGE_PRESSURE_DISTANCE_BONUS = 6
    MIN_CLEAN_BEFORE_CHARGE = 3
    MIN_STEPS_BETWEEN_CHARGES = 14
    CHARGE_INTERVAL_STEPS = 40
    TARGET_CHARGE_COUNT = 24
    MIN_CHARGE_COUNT = 10
    FORCE_RETURN_AHEAD_MARGIN = 8
    CHARGE_EVENT_MIN_BATTERY_JUMP = 6
    NO_CHARGER_RESERVE = 80

    # 动态避碰参数
    ROBOT_TRACK_FRAMES = 5
    ROBOT_PREDICT_STEPS = 5
    ROBOT_DANGER_BLOCK_TH = 3.5
    DANGER_EMERGENCY_TH = 2.8
    DANGER_COST_SCALE = 0.8
    DANGER_DECAY = 0.88

    # 模式相关阈值
    MODE_CHANGE_COOLDOWN = 5
    FULL_BATTERY_RATIO = 0.90
    ANCHOR_TRANSFER_COVERAGE = 0.85
    MIN_SWITCH_GAIN = 0.15
    SAFE_EXPLORE_RADIUS = 24
    BATTERY_LOW_THRESHOLD = 0.5
    BATTERY_RETURN_THRESHOLD = 0.35
    CHARGER_POS = (0, 0)  # 兼容旧模块，主逻辑使用 charger_map 动态发现
    
    # A* / BFS 参数
    ASTAR_MAX_ITER = 10000
    PATH_CACHE_SIZE = 50
    
    # 前瞻规划参数
    LOOKAHEAD_DEPTH = 5
    LOOKAHEAD_GAMMA = 0.95
    LOOKAHEAD_INTERVAL = 10  # 每10步触发一次前瞻规划
    
    # 区域管理参数
    REGION_SPLIT = 4  # 4x4 = 16个区域

    # 条带清扫参数
    STRIPE_SHIFT = 1
    STRIPE_MAX_LEN = 52
    STRIPE_REJOIN_BUDGET = 5
    STRIPE_MIN_LEN = 6
    
    # 脱困检测参数
    STUCK_HISTORY_SIZE = 10
    STUCK_THRESHOLD = 3  # 3步内移动距离小于阈值则认为卡住
    
    # 训练参数（PPO）
    PPO_CONFIG = {
        'clip_epsilon': 0.2,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'vf_coef': 0.5,
        'entropy_coef': 0.01,
        'lr': 3e-4,
        'grad_clip': 0.5
    }
    
    # 奖励参数
    CLEANING_REWARD = 0.5
    EXPLORATION_BONUS = 0.1
    PROXIMITY_BONUS = 0.01
    BATTERY_PENALTY = -0.05
    
    # 动作定义 (8方向)
    ACTIONS = [
        (0, -1),   # 0: 上
        (1, -1),   # 1: 右上
        (1, 0),    # 2: 右
        (1, 1),    # 3: 右下
        (0, 1),    # 4: 下
        (-1, 1),   # 5: 左下
        (-1, 0),   # 6: 左
        (-1, -1),  # 7: 左上
    ]


# ==================== 感知层 ====================

class FeaturePyramid:
    """多尺度局部特征金字塔"""
    
    def __init__(self, scales=None):
        self.scales = scales or Config.FEATURE_SCALES
        self.center_pos = (Config.LOCAL_VIEW_SIZE // 2, Config.LOCAL_VIEW_SIZE // 2)
    
    def extract(self, local_view: np.ndarray, robot_pos: Tuple[int, int]) -> Dict[str, np.ndarray]:
        """提取多尺度特征"""
        features = {}
        
        # L0: 完整21x21视野
        l0 = local_view.copy()
        features['L0'] = self._normalize(l0)
        
        # L1: 中心11x11裁剪
        center = Config.LOCAL_VIEW_SIZE // 2
        half_l1 = self.scales[1] // 2
        l1 = local_view[
            center - half_l1:center + half_l1 + 1,
            center - half_l1:center + half_l1 + 1
        ].copy()
        features['L1'] = self._normalize(l1)
        
        # L2: 中心7x7裁剪
        half_l2 = self.scales[2] // 2
        l2 = local_view[
            center - half_l2:center + half_l2 + 1,
            center - half_l2:center + half_l2 + 1
        ].copy()
        features['L2'] = self._normalize(l2)
        
        # 展平融合特征
        features['features'] = self._flatten_features(features)
        
        return features
    
    def _normalize(self, view: np.ndarray) -> np.ndarray:
        """归一化处理"""
        if len(view.shape) == 2:
            view = view[..., np.newaxis]
        return view.astype(np.float32) / 2.0
    
    def _flatten_features(self, features: Dict) -> np.ndarray:
        """将多尺度特征展平为向量"""
        l0_flat = features['L0'].flatten()
        l1_flat = features['L1'].flatten()
        l2_flat = features['L2'].flatten()
        
        total_dim = 160
        combined = np.concatenate([l0_flat, l1_flat, l2_flat])
        
        if len(combined) >= total_dim:
            return combined[:total_dim]
        else:
            padding = np.zeros(total_dim - len(combined), dtype=np.float32)
            return np.concatenate([combined, padding])


class CoarseMap:
    """32x32全局粗网格地图管理"""
    
    def __init__(self, map_size=128, coarse_size=32):
        self.map_size = map_size
        self.coarse_size = coarse_size
        self.scale = map_size // coarse_size
        
        # 粗网格统计信息
        self.obstacle_density = np.zeros((coarse_size, coarse_size), dtype=np.float32)
        self.dirt_density = np.zeros((coarse_size, coarse_size), dtype=np.float32)
        self.visit_freq = np.zeros((coarse_size, coarse_size), dtype=np.float32)
        self.frontier_score = np.zeros((coarse_size, coarse_size), dtype=np.float32)
        self.danger_level = np.zeros((coarse_size, coarse_size), dtype=np.float32)
        
        self.alpha = 0.9
        self.last_update_step = 0
    
    def _global_to_coarse(self, global_pos: Tuple[int, int]) -> Tuple[int, int]:
        """全局坐标转粗网格坐标"""
        return (global_pos[0] // self.scale, global_pos[1] // self.scale)
    
    def update(self, robot_pos: Tuple[int, int], local_view: np.ndarray, step: int):
        """根据当前局部视野更新粗网格"""
        view_half = Config.LOCAL_VIEW_SIZE // 2
        
        for dy in range(-view_half, view_half + 1):
            for dx in range(-view_half, view_half + 1):
                gx, gy = robot_pos[0] + dx, robot_pos[1] + dy
                if 0 <= gx < self.map_size and 0 <= gy < self.map_size:
                    cx, cy = self._global_to_coarse((gx, gy))
                    vx, vy = dx + view_half, dy + view_half
                    if 0 <= vx < Config.LOCAL_VIEW_SIZE and 0 <= vy < Config.LOCAL_VIEW_SIZE:
                        cell = local_view[vy, vx]
                        
                        self.obstacle_density[cy, cx] = (
                            self.alpha * self.obstacle_density[cy, cx] +
                            (1 - self.alpha) * (1 if cell[0] == 1 else 0)
                        )
                        self.dirt_density[cy, cx] = (
                            self.alpha * self.dirt_density[cy, cx] +
                            (1 - self.alpha) * (1 if cell[1] == 1 else 0)
                        )
        
        # 更新访问频率
        cx, cy = self._global_to_coarse(robot_pos)
        if 0 <= cx < self.coarse_size and 0 <= cy < self.coarse_size:
            self.visit_freq[cy, cx] += 1
        
        self._update_frontier_scores()
        self.last_update_step = step
    
    def _update_frontier_scores(self):
        """更新前沿探索得分"""
        for y in range(self.coarse_size):
            for x in range(self.coarse_size):
                if self.visit_freq[y, x] > 0:
                    frontier = 0
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < self.coarse_size and 0 <= nx < self.coarse_size:
                            if self.visit_freq[ny, nx] == 0:
                                frontier += 1
                    self.frontier_score[y, x] = frontier
    
    def get_region_stats(self, region_idx: int) -> Dict:
        """获取指定区域的统计信息"""
        rx = region_idx % Config.REGION_SPLIT
        ry = region_idx // Config.REGION_SPLIT
        
        region_coarse_size = self.coarse_size // Config.REGION_SPLIT
        x_start = rx * region_coarse_size
        y_start = ry * region_coarse_size
        x_end = min(x_start + region_coarse_size, self.coarse_size)
        y_end = min(y_start + region_coarse_size, self.coarse_size)
        
        return {
            'obstacle_density': float(np.mean(self.obstacle_density[y_start:y_end, x_start:x_end])),
            'dirt_density': float(np.mean(self.dirt_density[y_start:y_end, x_start:x_end])),
            'visit_freq': float(np.mean(self.visit_freq[y_start:y_end, x_start:x_end])),
            'frontier_score': float(np.sum(self.frontier_score[y_start:y_end, x_start:x_end])),
            'danger_level': float(np.mean(self.danger_level[y_start:y_end, x_start:x_end]))
        }
    
    def get_state_vector(self) -> np.ndarray:
        """获取全局状态向量"""
        return np.concatenate([
            self.obstacle_density.flatten(),
            self.dirt_density.flatten(),
            self.visit_freq.flatten(),
            self.frontier_score.flatten(),
            self.danger_level.flatten()
        ])


# ==================== 规划层 ====================

class PathPlanner:
    """A*/BFS路径规划器"""
    
    def __init__(self, coarse_map: CoarseMap):
        self.coarse_map = coarse_map
        self.path_cache = {}
    
    def heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """启发式函数：欧氏距离 + 访问频率惩罚"""
        dist = math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)
        
        # 访问频率惩罚
        cx, cy = self.coarse_map._global_to_coarse(pos)
        if 0 <= cx < self.coarse_map.coarse_size and 0 <= cy < self.coarse_map.coarse_size:
            visit_penalty = 0.1 * self.coarse_map.visit_freq[cy, cx]
        else:
            visit_penalty = 0
        
        return dist + visit_penalty
    
    def astar(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacle_map: Optional[np.ndarray] = None,
        max_iter: int = None,
        danger_map: Optional[np.ndarray] = None,
        danger_as_block: bool = False
    ) -> List[Tuple[int, int]]:
        """A*路径规划"""
        max_iter = max_iter or Config.ASTAR_MAX_ITER
        
        if start == goal:
            return [start]
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        
        closed_set = set()
        iterations = 0
        
        while open_set and iterations < max_iter:
            iterations += 1
            _, current = heapq.heappop(open_set)
            
            if current in closed_set:
                continue
            closed_set.add(current)
            
            if current == goal:
                return self._reconstruct_path(came_from, current)
            
            for action_idx, (dx, dy) in enumerate(Config.ACTIONS):
                neighbor = (current[0] + dx, current[1] + dy)
                
                # 检查边界
                if not (0 <= neighbor[0] < Config.MAP_SIZE and 0 <= neighbor[1] < Config.MAP_SIZE):
                    continue
                
                # 检查障碍
                if obstacle_map is not None:
                    if obstacle_map[neighbor[1], neighbor[0]] == 1:
                        continue

                if danger_as_block and danger_map is not None:
                    if danger_map[neighbor[1], neighbor[0]] >= Config.ROBOT_DANGER_BLOCK_TH:
                        continue
                
                if neighbor in closed_set:
                    continue
                
                # 对角线移动代价为sqrt(2)，直线为1
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                if danger_map is not None:
                    move_cost += Config.DANGER_COST_SCALE * float(danger_map[neighbor[1], neighbor[0]])
                tentative_g = g_score[current] + move_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # 未找到路径
    
    def bfs(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        obstacle_map: Optional[np.ndarray] = None,
        max_depth: int = 100,
        danger_map: Optional[np.ndarray] = None,
        danger_as_block: bool = False
    ) -> List[Tuple[int, int]]:
        """BFS回退路径规划"""
        if start == goal:
            return [start]
        
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
            
            for dx, dy in Config.ACTIONS:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if not (0 <= neighbor[0] < Config.MAP_SIZE and 0 <= neighbor[1] < Config.MAP_SIZE):
                    continue
                
                if obstacle_map is not None:
                    if obstacle_map[neighbor[1], neighbor[0]] == 1:
                        continue

                if danger_as_block and danger_map is not None:
                    if danger_map[neighbor[1], neighbor[0]] >= Config.ROBOT_DANGER_BLOCK_TH:
                        continue
                
                if neighbor in visited:
                    continue
                
                new_path = path + [neighbor]
                
                if neighbor == goal:
                    return new_path
                
                visited.add(neighbor)
                queue.append((neighbor, new_path))
        
        return []
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def path_cost(self, path: List[Tuple[int, int]]) -> float:
        """计算路径总代价"""
        if not path or len(path) == 1:
            return 0.0
        cost = 0.0
        for idx in range(1, len(path)):
            dx = abs(path[idx][0] - path[idx - 1][0])
            dy = abs(path[idx][1] - path[idx - 1][1])
            cost += 1.414 if dx == 1 and dy == 1 else 1.0
        return cost
    
    def get_action_from_path(
        self,
        path: List[Tuple[int, int]],
        robot_pos: Tuple[int, int],
        legal_actions: List[int]
    ) -> Optional[int]:
        """从路径中提取下一步动作"""
        if len(path) < 2:
            return None
        
        next_pos = path[1]
        dx = next_pos[0] - robot_pos[0]
        dy = next_pos[1] - robot_pos[1]
        
        # 找到对应的动作
        for action_idx, (adx, ady) in enumerate(Config.ACTIONS):
            if dx == adx and dy == ady:
                if action_idx in legal_actions:
                    return action_idx
        
        return None


class RegionManager:
    """区域分割与管理器"""
    
    def __init__(self, map_size=128, region_split=4):
        self.map_size = map_size
        self.region_split = region_split
        self.region_size = map_size // region_split
        self.num_regions = region_split * region_split
    
    def pos_to_region(self, pos: Tuple[int, int]) -> int:
        """将全局坐标转换为区域索引"""
        rx = pos[0] // self.region_size
        ry = pos[1] // self.region_size
        return ry * self.region_split + rx
    
    def region_to_center(self, region_idx: int) -> Tuple[int, int]:
        """获取区域中心坐标"""
        rx = region_idx % self.region_split
        ry = region_idx // self.region_split
        center_x = rx * self.region_size + self.region_size // 2
        center_y = ry * self.region_size + self.region_size // 2
        return (center_x, center_y)
    
    def get_region_bounds(self, region_idx: int) -> Tuple[int, int, int, int]:
        """获取区域边界 (x_min, y_min, x_max, y_max)"""
        rx = region_idx % self.region_split
        ry = region_idx // self.region_split
        x_min = rx * self.region_size
        y_min = ry * self.region_size
        x_max = min(x_min + self.region_size, self.map_size)
        y_max = min(y_min + self.region_size, self.map_size)
        return (x_min, y_min, x_max, y_max)
    
    def score_regions(self, coarse_map: CoarseMap, robot_pos: Tuple[int, int]) -> List[Tuple[int, float]]:
        """计算所有区域的得分并排序"""
        scores = []
        
        for region_idx in range(self.num_regions):
            stats = coarse_map.get_region_stats(region_idx)
            region_center = self.region_to_center(region_idx)
            distance = math.sqrt(
                (robot_pos[0] - region_center[0])**2 +
                (robot_pos[1] - region_center[1])**2
            )
            
            # 得分公式
            score = (
                stats['dirt_density'] * 0.5 +
                (1 - min(stats['visit_freq'] / 10, 1)) * 0.3 +
                stats['frontier_score'] * 0.2
            ) / (1 + distance / 50)  # 距离惩罚
            
            scores.append((region_idx, score))
        
        # 按得分降序排列
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores
    
    def select_best_region(
        self,
        coarse_map: CoarseMap,
        robot_pos: Tuple[int, int],
        battery_ratio: float
    ) -> Tuple[int, Tuple[int, int]]:
        """选择最佳目标区域"""
        scored_regions = self.score_regions(coarse_map, robot_pos)
        
        # 根据电量调整策略
        if battery_ratio < Config.BATTERY_LOW_THRESHOLD:
            # 低电量时优先选择靠近充电站的区域
            charger = Config.CHARGER_POS
            best_region = None
            best_score = -float('inf')
            
            for region_idx, score in scored_regions[:5]:  # 只考虑前5个
                region_center = self.region_to_center(region_idx)
                dist_to_charger = math.sqrt(
                    (region_center[0] - charger[0])**2 +
                    (region_center[1] - charger[1])**2
                )
                adjusted_score = score - dist_to_charger / 100
                
                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_region = region_idx
        else:
            best_region = scored_regions[0][0]
        
        return (best_region, self.region_to_center(best_region))


class LookaheadPlanner:
    """前瞻规划器"""
    
    def __init__(self, depth=5, gamma=0.95):
        self.depth = depth
        self.gamma = gamma
    
    def should_trigger_lookahead(
        self,
        mode: str,
        step_count: int,
        last_trigger_step: int,
        target_changed: bool = False
    ) -> bool:
        """判断是否触发前瞻规划"""
        # 模式切换时触发
        if target_changed:
            return True
        
        # 每N步触发一次
        if step_count - last_trigger_step >= Config.LOOKAHEAD_INTERVAL:
            return True
        
        return False
    
    def evaluate_action_sequence(
        self,
        actions: List[int],
        current_state: Dict,
        coarse_map: CoarseMap
    ) -> float:
        """评估动作序列的长期价值"""
        value = 0
        gamma_power = 1.0
        
        # 简化模拟：基于当前状态估计未来收益
        robot_pos = current_state['robot_pos']
        
        for action in actions[:self.depth]:
            dx, dy = Config.ACTIONS[action]
            new_pos = (robot_pos[0] + dx, robot_pos[1] + dy)
            
            # 检查合法性
            if not (0 <= new_pos[0] < Config.MAP_SIZE and 0 <= new_pos[1] < Config.MAP_SIZE):
                value -= 10 * gamma_power  # 撞墙惩罚
                continue
            
            # 估计即时奖励
            cx, cy = coarse_map._global_to_coarse(new_pos)
            if 0 <= cx < coarse_map.coarse_size and 0 <= cy < coarse_map.coarse_size:
                dirt_reward = coarse_map.dirt_density[cy, cx] * Config.CLEANING_REWARD
                exploration_bonus = (1 - min(coarse_map.visit_freq[cy, cx] / 10, 1)) * Config.EXPLORATION_BONUS
                value += (dirt_reward + exploration_bonus) * gamma_power
            
            gamma_power *= self.gamma
            robot_pos = new_pos
        
        return value
    
    def lookahead_plan(
        self,
        current_state: Dict,
        coarse_map: CoarseMap,
        legal_actions: List[int],
        path_planner: PathPlanner
    ) -> Tuple[int, float]:
        """前瞻规划选择最佳动作"""
        best_action = legal_actions[0] if legal_actions else 0
        best_value = -float('inf')
        
        # 对前几个动作做简单前瞻
        for action in legal_actions[:4]:  # 限制分支因子
            # 生成候选动作序列 (当前动作 + 贪心后续)
            actions = [action]
            
            # 评估
            value = self.evaluate_action_sequence(actions, current_state, coarse_map)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action, best_value


# ==================== 导航控制层 ====================

class UnstuckHandler:
    """脱困处理器"""
    
    def __init__(self, history_size=10):
        self.position_history = deque(maxlen=history_size)
        self.action_history = deque(maxlen=history_size)
        self.stuck_count = 0
        self.last_dirt_count = 0
        self.no_progress_steps = 0
    
    def update(self, robot_pos: Tuple[int, int], action: int, current_dirt_count: int):
        """更新位置历史"""
        self.position_history.append(robot_pos)
        self.action_history.append(action)
        
        # 检测是否有清扫进展
        if current_dirt_count > self.last_dirt_count:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1
        
        self.last_dirt_count = current_dirt_count
    
    def is_stuck(self) -> bool:
        """检测是否卡住"""
        if len(self.position_history) < Config.STUCK_HISTORY_SIZE:
            return False
        
        # 检测1: 最近N步位置变化很小
        recent_positions = list(self.position_history)[-Config.STUCK_THRESHOLD:]
        if len(recent_positions) >= 2:
            total_dist = 0
            for i in range(1, len(recent_positions)):
                dist = math.sqrt(
                    (recent_positions[i][0] - recent_positions[i-1][0])**2 +
                    (recent_positions[i][1] - recent_positions[i-1][1])**2
                )
                total_dist += dist
            
            if total_dist < 1.5:  # 几乎没移动
                self.stuck_count += 1
                if self.stuck_count >= 2:
                    return True
        
        # 检测2: 长时间没有清扫进展
        if self.no_progress_steps > 30:
            return True
        
        # 检测3: 动作来回震荡
        if len(self.action_history) >= 4:
            recent_actions = list(self.action_history)[-4:]
            if recent_actions[0] == recent_actions[2] and recent_actions[1] == recent_actions[3]:
                if recent_actions[0] != recent_actions[1]:
                    return True
        
        return False
    
    def get_unstuck_action(
        self,
        local_view: np.ndarray,
        legal_actions: List[int],
        robot_pos: Tuple[int, int]
    ) -> int:
        """获取脱困动作"""
        # 策略1: 尝试随机合法动作
        if len(legal_actions) > 0:
            # 优先选择之前很少执行的动作
            action_counts = {}
            for a in legal_actions:
                action_counts[a] = list(self.action_history).count(a)
            
            # 选择执行次数最少的动作
            best_action = min(legal_actions, key=lambda a: action_counts.get(a, 0))
            return best_action
        
        return 0  # 默认动作
    
    def reset(self):
        """重置状态"""
        self.position_history.clear()
        self.action_history.clear()
        self.stuck_count = 0
        self.no_progress_steps = 0
        self.last_dirt_count = 0


class ReturnToCharger:
    """返航处理器（多充电桩 + 动态危险规避）"""

    @staticmethod
    def should_return(battery: int, safe_energy: int) -> bool:
        return battery <= safe_energy

    @staticmethod
    def get_return_path(
        robot_pos: Tuple[int, int],
        chargers: List[Tuple[int, int]],
        path_planner: PathPlanner,
        obstacle_map: Optional[np.ndarray] = None,
        danger_map: Optional[np.ndarray] = None,
    ) -> Tuple[List[Tuple[int, int]], Optional[Tuple[int, int]], float]:
        """规划到最近可达充电桩的路径"""
        best_path: List[Tuple[int, int]] = []
        best_charger: Optional[Tuple[int, int]] = None
        best_cost = float('inf')

        for charger in chargers:
            path = path_planner.astar(
                robot_pos,
                charger,
                obstacle_map=obstacle_map,
                danger_map=danger_map,
                danger_as_block=True,
            )
            if not path:
                path = path_planner.bfs(
                    robot_pos,
                    charger,
                    obstacle_map=obstacle_map,
                    danger_map=danger_map,
                    danger_as_block=True,
                    max_depth=220,
                )
            if not path:
                continue
            cost = path_planner.path_cost(path)
            if cost < best_cost:
                best_cost = cost
                best_path = path
                best_charger = charger

        return best_path, best_charger, best_cost


# ==================== 决策层 ====================

class ModeSelector:
    """模式选择器 - 优先级队列管理"""
    
    def __init__(self):
        self.current_mode = 'SAFE_EXPLORATION'
        self.mode_cooldown = {}
        self.last_mode_change_step = 0
        self.mode_change_cooldown = Config.MODE_CHANGE_COOLDOWN
    
    def select_mode(
        self,
        step_count: int,
        emergency: bool,
        need_return: bool,
        early_charge: bool,
        stuck: bool,
        need_transfer: bool,
        can_stripe: bool,
        has_direct_dirt: bool,
        need_safe_explore: bool
    ) -> str:
        """根据状态机优先级选择模式"""
        # P0/P1 硬切换，不受冷却约束
        if emergency:
            if self.current_mode != 'EMERGENCY_EVADE':
                self.current_mode = 'EMERGENCY_EVADE'
                self.last_mode_change_step = step_count
            return self.current_mode

        if need_return:
            if self.current_mode != 'RETURN_TO_CHARGER':
                self.current_mode = 'RETURN_TO_CHARGER'
                self.last_mode_change_step = step_count
            return self.current_mode

        # 检查模式切换冷却
        if step_count - self.last_mode_change_step < self.mode_change_cooldown:
            return self.current_mode

        next_mode = 'SAFE_EXPLORATION'
        if stuck:
            next_mode = 'STUCK_RECOVERY'
        elif need_transfer:
            next_mode = 'CHARGER_TRANSFER'
        elif can_stripe:
            next_mode = 'STRIPE_CLEANING'
        elif has_direct_dirt:
            next_mode = 'DIRECT_DIRT_PICKUP'
        elif need_safe_explore:
            next_mode = 'SAFE_EXPLORATION'

        if next_mode != self.current_mode:
            self.current_mode = next_mode
            self.last_mode_change_step = step_count

        return self.current_mode
    
    def has_dirt_in_view(self, local_view: np.ndarray) -> bool:
        """检查视野内是否有脏点"""
        # 通道1是脏点
        if len(local_view.shape) == 3:
            return np.any(local_view[:, :, 1] == 1)
        else:
            return np.any(local_view == 2)  # 假设2表示脏点
    
    def get_nearest_dirt_pos(
        self,
        local_view: np.ndarray,
        robot_pos: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        """获取视野内最近的脏点位置"""
        center = Config.LOCAL_VIEW_SIZE // 2
        
        if len(local_view.shape) == 3:
            dirt_channel = local_view[:, :, 1]
        else:
            dirt_channel = (local_view == 2).astype(int)
        
        dirt_positions = np.argwhere(dirt_channel == 1)
        
        if len(dirt_positions) == 0:
            return None
        
        # 找到距离中心最近的脏点
        min_dist = float('inf')
        nearest_pos = None
        
        for dy, dx in dirt_positions:
            dist = abs(dx - center) + abs(dy - center)
            if dist < min_dist:
                min_dist = dist
                # 转换为全局坐标
                nearest_pos = (
                    robot_pos[0] + (dx - center),
                    robot_pos[1] + (dy - center)
                )
        
        return nearest_pos


class RewardShaper:
    """自适应奖励塑形器"""
    
    def __init__(self):
        self.phase = 'EXPLORATION'  # EXPLORATION / BALANCE / CONVERGENCE
        self.phase_thresholds = [0.33, 0.67]
        self.cleaned_count = 0
        self.last_distance_to_dirt = None
    
    def update_phase(self, clean_progress: float):
        """根据清扫进度更新阶段"""
        if clean_progress < self.phase_thresholds[0]:
            self.phase = 'EXPLORATION'
        elif clean_progress < self.phase_thresholds[1]:
            self.phase = 'BALANCE'
        else:
            self.phase = 'CONVERGENCE'
    
    def shape_reward(
        self,
        raw_reward: float,
        new_dirt_count: int,
        old_dirt_dist: float,
        new_dirt_dist: float,
        battery_ratio: float,
        action_legal: bool,
        is_stuck: bool,
        step_count: int,
        is_new_area: bool = False
    ) -> float:
        """计算shaping后的奖励"""
        
        # 基础清扫奖励
        reward = raw_reward + new_dirt_count * Config.CLEANING_REWARD
        
        # 阶段惩罚
        reward += self.get_step_penalty()
        
        # 接近奖励
        if old_dirt_dist is not None and new_dirt_dist is not None:
            proximity_reward = Config.PROXIMITY_BONUS * (old_dirt_dist - new_dirt_dist)
            reward += proximity_reward
        
        # 探索奖励 (早期阶段)
        if self.phase == 'EXPLORATION' and is_new_area:
            reward += Config.EXPLORATION_BONUS
        
        # 电量惩罚
        if battery_ratio < 0.3:
            reward += Config.BATTERY_PENALTY
        
        # 违规惩罚
        if not action_legal:
            reward -= 1.0
        
        if is_stuck:
            reward -= 0.5
        
        return reward
    
    def get_step_penalty(self) -> float:
        """获取当前阶段的每步惩罚"""
        penalties = {
            'EXPLORATION': -0.001,
            'BALANCE': -0.005,
            'CONVERGENCE': -0.01
        }
        return penalties.get(self.phase, -0.005)


# ==================== 神经网络模型 (学习层) ====================

class CleaningAgentNet(nn.Module):
    """
    清扫智能体网络结构 (Actor-Critic)
    
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
        
        # 计算展平后的维度
        # L0: 32 * 7 * 7 = 1568
        # L1: 32 * 5 * 5 = 800
        # L2: 32 * 5 * 5 = 800
        # scalar: 10
        # total: 1568 + 800 + 800 + 10 = 3178
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(32*7*7 + 32*5*5 + 32*5*5 + 10, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
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
    
    def _make_encoder(self, in_channels, out_channels, num_layers):
        """创建CNN编码器"""
        layers = []
        channels = [in_channels, 16, out_channels]
        for i in range(num_layers):
            layers.extend([
                nn.Conv2d(channels[i], channels[i+1], kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2) if i < num_layers - 1 else nn.Identity()
            ])
        return nn.Sequential(*layers)
    
    def _ortho_init(self, m):
        """正交初始化"""
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight, gain=1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, l0, l1, l2, scalar):
        """前向传播"""
        # 编码各尺度特征
        f0 = self.encoder_l0(l0)
        f1 = self.encoder_l1(l1)
        f2 = self.encoder_l2(l2)
        
        # 展平
        f0_flat = f0.reshape(f0.size(0), -1)
        f1_flat = f1.reshape(f1.size(0), -1)
        f2_flat = f2.reshape(f2.size(0), -1)
        
        # 融合
        fused = torch.cat([f0_flat, f1_flat, f2_flat, scalar], dim=1)
        features = self.fusion(fused)
        
        # 输出
        policy_logits = self.actor(features)
        value = self.critic(features)
        
        return policy_logits, value
    
    def get_action_and_value(self, obs_dict, legal_mask=None, action=None):
        """
        获取动作和价值，用于PPO训练
        Args:
            obs_dict: 包含l0, l1, l2, scalar的观测字典
            legal_mask: (batch, 8) 合法动作掩码
            action: 实际执行的动作（用于计算log_prob）
        """
        policy_logits, value = self.forward(
            obs_dict['l0'],
            obs_dict['l1'],
            obs_dict['l2'],
            obs_dict['scalar']
        )
        
        # 应用合法动作掩码
        if legal_mask is not None:
            policy_logits = policy_logits.masked_fill(~legal_mask.bool(), float('-inf'))
        
        # 计算概率分布
        probs = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


# ==================== 主智能体 ====================

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
    
    def __init__(self, use_neural=False, device='cpu'):
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
        
        # 神经网络（可选）
        self.use_neural = use_neural
        self.device = device
        if use_neural:
            self.neural_net = CleaningAgentNet().to(device)
            self.neural_net.eval()
        
        # 状态跟踪
        self.current_mode = 'SAFE_EXPLORATION'
        self.target_region = None
        self.target_pos = None
        self.current_path = []
        self.step_count = 0
        self.last_lookahead_step = 0
        self.last_action = -1
        self.start_pos = None
        
        # 全局记忆地图（静态层 + 动态层）
        self.static_obstacle_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int8)
        self.free_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int8)
        self.known_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int8)
        self.dirt_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int8)
        self.cleaned_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int8)
        self.visit_count_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int32)
        self.anchor_region_map = np.full((Config.MAP_SIZE, Config.MAP_SIZE), -1, dtype=np.int16)
        self._obs_confidence = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int8)
        self._last_obs_state = np.full((Config.MAP_SIZE, Config.MAP_SIZE), -1, dtype=np.int8)

        # 充电桩/锚点
        self.charger_map: Set[Tuple[int, int]] = set()
        self.charger_list: List[Tuple[int, int]] = []
        self.current_anchor: Optional[Tuple[int, int]] = None
        self.return_cost_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], float] = {}

        # 官方机器人轨迹和危险图
        self.robot_tracks: deque = deque(maxlen=Config.ROBOT_TRACK_FRAMES)
        self.robot_danger_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)

        # 条带状态
        self.stripe_dir = 1
        self.stripe_shift_dir = 1
        self.stripe_run_len = 0
        self.stripe_rejoin_fail = 0
        
        # 统计
        self.total_cleaned = 0
        self.last_battery = Config.MAX_BATTERY
        self.last_pos = None
        self.steps_since_last_charge = 0
        self.charge_count = 0
        self.last_charge_step = -1000
        self.prev_on_charger = False
        self.battery_max_runtime = Config.MAX_BATTERY
        self.charge_guard_active = False
        self.last_charge_deficit = 0
    
    def act(self, obs: Dict) -> int:
        """
        主决策函数
        Args:
            obs: 环境观测 {
                'agent_maps': np.ndarray,  # 局部视野 (21, 21, 3)
                'agent_pos': Tuple[int, int],
                'battery': int,
                'step': int,
                'legal_actions': List[int],
                ...
            }
        Returns:
            action: 0-7 的动作编号
        """
        # 解析观测
        local_view = obs.get('agent_maps', np.zeros((21, 21, 3)))
        robot_pos = obs.get('agent_pos', (0, 0))
        battery = obs.get('battery', Config.MAX_BATTERY)
        step = obs.get('step', 0)
        legal_actions = obs.get('legal_actions', list(range(8)))
        
        self.step_count = step
        if self.start_pos is None:
            self.start_pos = robot_pos
            self._register_charger(robot_pos)
        
        # 更新全局记忆地图
        dynamic_candidates = self._update_memory_maps(robot_pos, local_view)
        self._refresh_dynamic_danger_map(dynamic_candidates)
        
        # 更新粗网格
        self.coarse_map.update(robot_pos, local_view, step)
        self._sync_coarse_danger()
        
        # 更新脱困处理器
        self.unstuck_handler.update(robot_pos, self.last_action, self.total_cleaned)
        self._select_current_anchor(robot_pos)

        base_legal_actions = [a for a in legal_actions if 0 <= a < len(Config.ACTIONS)]
        if not base_legal_actions:
            base_legal_actions = [0]

        safe_energy = self._calc_safe_energy(robot_pos)
        need_return = self.return_handler.should_return(battery, safe_energy)
        energy_actions = self._filter_energy_actions(base_legal_actions, robot_pos, battery)
        danger_safe_actions = self._filter_danger_actions(energy_actions, robot_pos)
        mode_actions = danger_safe_actions or energy_actions or base_legal_actions

        has_direct_dirt = self.mode_selector.has_dirt_in_view(local_view)
        need_safe_explore = len(self.charger_map) == 0
        can_stripe = not need_safe_explore and battery > safe_energy + Config.DETOUR_MARGIN
        need_transfer = self._should_transfer_anchor(robot_pos, battery)
        emergency = self._is_emergency_risk(robot_pos, energy_actions, danger_safe_actions)
        
        # 模式选择
        self.current_mode = self.mode_selector.select_mode(
            step_count=step,
            emergency=emergency,
            need_return=need_return,
            stuck=self.unstuck_handler.is_stuck(),
            need_transfer=need_transfer,
            can_stripe=can_stripe,
            has_direct_dirt=has_direct_dirt,
            need_safe_explore=need_safe_explore,
        )
        
        # 根据模式执行决策
        if self.current_mode == 'EMERGENCY_EVADE':
            action = self._mode_emergency_evade(robot_pos, mode_actions)
        elif self.current_mode == 'RETURN_TO_CHARGER':
            action = self._mode_return_to_charger(robot_pos, mode_actions)
        elif self.current_mode == 'STUCK_RECOVERY':
            action = self._mode_stuck_recovery(robot_pos, local_view, mode_actions)
        elif self.current_mode == 'CHARGER_TRANSFER':
            action = self._mode_charger_transfer(robot_pos, battery, mode_actions)
        elif self.current_mode == 'STRIPE_CLEANING':
            action = self._mode_stripe_cleaning(robot_pos, battery, mode_actions)
        elif self.current_mode == 'DIRECT_DIRT_PICKUP':
            action = self._mode_direct_dirt_pickup(robot_pos, local_view, battery, mode_actions)
        else:
            action = self._mode_safe_exploration(robot_pos, battery, mode_actions)

        if action not in mode_actions:
            action = mode_actions[0] if mode_actions else base_legal_actions[0]
        
        # 更新统计
        self.last_battery = battery
        self.last_pos = robot_pos
        self.last_action = action
        self.visit_count_map[robot_pos[1], robot_pos[0]] += 1
        
        return action
    
    def _sync_coarse_danger(self):
        """将细粒度危险图同步到粗网格"""
        scale = self.coarse_map.scale
        for cy in range(self.coarse_map.coarse_size):
            y0 = cy * scale
            y1 = min(y0 + scale, Config.MAP_SIZE)
            for cx in range(self.coarse_map.coarse_size):
                x0 = cx * scale
                x1 = min(x0 + scale, Config.MAP_SIZE)
                patch = self.robot_danger_map[y0:y1, x0:x1]
                self.coarse_map.danger_level[cy, cx] = float(np.max(patch)) if patch.size > 0 else 0.0

    def _extract_charger_from_cell(self, cell) -> bool:
        if isinstance(cell, np.ndarray) and cell.ndim > 0:
            if cell.shape[0] >= 3 and cell[2] >= 0.5:
                return True
            if np.max(cell) >= 3:
                return True
            return False
        return int(cell) == 3

    def _register_charger(self, pos: Tuple[int, int]):
        if pos not in self.charger_map:
            self.charger_map.add(pos)
            self.charger_list = sorted(list(self.charger_map))

    def _get_known_chargers(self) -> List[Tuple[int, int]]:
        if self.charger_list:
            return self.charger_list
        if self.start_pos is not None:
            return [self.start_pos]
        return []

    def _refresh_dynamic_danger_map(self, dynamic_candidates: List[Tuple[int, int]]):
        """维护机器人轨迹并预测 3-5 步危险走廊"""
        self.robot_danger_map *= Config.DANGER_DECAY
        frame = dynamic_candidates[:40]
        self.robot_tracks.append(frame)
        if not frame:
            return

        prev_frame = self.robot_tracks[-2] if len(self.robot_tracks) >= 2 else []
        for x, y in frame:
            vx, vy = 0, 0
            if prev_frame:
                nearest = min(prev_frame, key=lambda p: abs(p[0] - x) + abs(p[1] - y))
                if abs(nearest[0] - x) + abs(nearest[1] - y) <= 4:
                    vx = x - nearest[0]
                    vy = y - nearest[1]

            for t in range(Config.ROBOT_PREDICT_STEPS):
                px = int(round(x + vx * t))
                py = int(round(y + vy * t))
                if not (0 <= px < Config.MAP_SIZE and 0 <= py < Config.MAP_SIZE):
                    continue
                radius = 2 if t <= 2 else 1
                risk = 2.4 - 0.35 * t
                for ry in range(-radius, radius + 1):
                    for rx in range(-radius, radius + 1):
                        nx, ny = px + rx, py + ry
                        if 0 <= nx < Config.MAP_SIZE and 0 <= ny < Config.MAP_SIZE:
                            dist = math.sqrt(rx * rx + ry * ry)
                            if dist <= radius:
                                self.robot_danger_map[ny, nx] = max(
                                    self.robot_danger_map[ny, nx],
                                    max(0.0, risk - 0.45 * dist),
                                )

    def _update_memory_maps(self, robot_pos: Tuple[int, int], local_view: np.ndarray) -> List[Tuple[int, int]]:
        """更新静态/动态地图并返回动态障碍候选点"""
        center = Config.LOCAL_VIEW_SIZE // 2
        dynamic_candidates: List[Tuple[int, int]] = []

        for dy in range(-center, center + 1):
            for dx in range(-center, center + 1):
                gx, gy = robot_pos[0] + dx, robot_pos[1] + dy
                if 0 <= gx < Config.MAP_SIZE and 0 <= gy < Config.MAP_SIZE:
                    vx, vy = dx + center, dy + center
                    if 0 <= vx < Config.LOCAL_VIEW_SIZE and 0 <= vy < Config.LOCAL_VIEW_SIZE:
                        cell = local_view[vy, vx]
                        obstacle = (cell[0] == 1) if len(cell.shape) > 0 else (int(cell) == 0)
                        dirt = (cell[1] == 1) if len(cell.shape) > 0 and cell.shape[0] >= 2 else (int(cell) == 2)
                        charger = self._extract_charger_from_cell(cell)

                        self.known_map[gy, gx] = 1
                        prev_obs = int(self._last_obs_state[gy, gx])
                        curr_obs = 1 if obstacle else 0
                        if prev_obs != -1 and prev_obs != curr_obs and self.static_obstacle_map[gy, gx] == 0:
                            dynamic_candidates.append((gx, gy))
                        self._last_obs_state[gy, gx] = curr_obs

                        if obstacle:
                            self._obs_confidence[gy, gx] = min(6, self._obs_confidence[gy, gx] + 1)
                            self.free_map[gy, gx] = 0
                            if self._obs_confidence[gy, gx] >= 3:
                                self.static_obstacle_map[gy, gx] = 1
                        else:
                            self._obs_confidence[gy, gx] = max(0, self._obs_confidence[gy, gx] - 2)
                            self.free_map[gy, gx] = 1
                            if self._obs_confidence[gy, gx] == 0:
                                self.static_obstacle_map[gy, gx] = 0

                        if dirt:
                            self.dirt_map[gy, gx] = 1
                        elif self.dirt_map[gy, gx] == 1:
                            self.dirt_map[gy, gx] = 0
                            self.cleaned_map[gy, gx] = 1
                            self.total_cleaned += 1

                        if charger:
                            self._register_charger((gx, gy))

                        if self.charger_list and self.free_map[gy, gx] == 1:
                            best_idx = min(
                                range(len(self.charger_list)),
                                key=lambda idx: abs(self.charger_list[idx][0] - gx) + abs(self.charger_list[idx][1] - gy),
                            )
                            self.anchor_region_map[gy, gx] = int(best_idx)

        return dynamic_candidates

    def _next_pos(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        dx, dy = Config.ACTIONS[action]
        return pos[0] + dx, pos[1] + dy

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < Config.MAP_SIZE and 0 <= pos[1] < Config.MAP_SIZE

    def _estimate_path_cost(self, start: Tuple[int, int], goal: Tuple[int, int], use_danger: bool = True) -> float:
        key = (start, goal)
        if key in self.return_cost_cache and self.step_count % 4 != 0:
            return self.return_cost_cache[key]

        path = self.path_planner.astar(
            start,
            goal,
            obstacle_map=self.static_obstacle_map,
            danger_map=self.robot_danger_map if use_danger else None,
            danger_as_block=False,
        )
        if not path:
            path = self.path_planner.bfs(
                start,
                goal,
                obstacle_map=self.static_obstacle_map,
                danger_map=self.robot_danger_map if use_danger else None,
                danger_as_block=False,
                max_depth=240,
            )
        if path:
            cost = self.path_planner.path_cost(path)
        else:
            cost = float(abs(start[0] - goal[0]) + abs(start[1] - goal[1]) + Config.DETOUR_MARGIN)

        self.return_cost_cache[key] = cost
        return cost

    def _calc_min_return_cost(self, pos: Tuple[int, int]) -> int:
        chargers = self._get_known_chargers()
        if not chargers:
            return Config.MAX_BATTERY // 2
        costs = [self._estimate_path_cost(pos, c, use_danger=True) for c in chargers]
        return int(math.ceil(min(costs)))

    def _calc_safe_energy(self, pos: Tuple[int, int]) -> int:
        min_return = self._calc_min_return_cost(pos)
        unknown_margin = Config.UNKNOWN_MARGIN if self.known_map[pos[1], pos[0]] == 0 else max(2, Config.UNKNOWN_MARGIN // 2)
        safe_energy = min_return + Config.DETOUR_MARGIN + unknown_margin + Config.DOCK_BUFFER
        return min(Config.MAX_BATTERY - 1, int(safe_energy))

    def _action_energy_feasible(self, robot_pos: Tuple[int, int], action: int, battery: int) -> bool:
        nxt = self._next_pos(robot_pos, action)
        if not self._is_valid_pos(nxt):
            return False
        if self.static_obstacle_map[nxt[1], nxt[0]] == 1:
            return False
        battery_after = battery - Config.ACTION_MOVE_COST
        if battery_after <= 0:
            return False
        return battery_after >= self._calc_safe_energy(nxt)

    def _filter_energy_actions(self, legal_actions: List[int], robot_pos: Tuple[int, int], battery: int) -> List[int]:
        return [a for a in legal_actions if self._action_energy_feasible(robot_pos, a, battery)]

    def _filter_danger_actions(self, legal_actions: List[int], robot_pos: Tuple[int, int]) -> List[int]:
        safe = []
        for action in legal_actions:
            nxt = self._next_pos(robot_pos, action)
            if self._is_valid_pos(nxt) and self.robot_danger_map[nxt[1], nxt[0]] < Config.ROBOT_DANGER_BLOCK_TH:
                safe.append(action)
        return safe

    def _is_emergency_risk(self, robot_pos: Tuple[int, int], energy_actions: List[int], danger_safe_actions: List[int]) -> bool:
        current_danger = float(self.robot_danger_map[robot_pos[1], robot_pos[0]])
        if current_danger >= Config.ROBOT_DANGER_BLOCK_TH * 0.8:
            return True
        if energy_actions and not danger_safe_actions:
            return True
        return False

    def _select_current_anchor(self, robot_pos: Tuple[int, int]):
        chargers = self._get_known_chargers()
        if not chargers:
            self.current_anchor = None
            return
        if self.current_anchor not in chargers or self.step_count % 20 == 0:
            self.current_anchor = min(chargers, key=lambda c: self._estimate_path_cost(robot_pos, c, use_danger=True))

    def _anchor_index(self, anchor: Optional[Tuple[int, int]]) -> int:
        if anchor is None or anchor not in self.charger_list:
            return -1
        return self.charger_list.index(anchor)

    def _cell_in_current_anchor(self, pos: Tuple[int, int]) -> bool:
        idx = self._anchor_index(self.current_anchor)
        if idx < 0:
            return True
        tag = int(self.anchor_region_map[pos[1], pos[0]])
        return tag < 0 or tag == idx

    def _anchor_coverage(self) -> float:
        idx = self._anchor_index(self.current_anchor)
        if idx < 0:
            return 0.0
        mask = (self.anchor_region_map == idx) & (self.known_map == 1)
        known = int(np.sum(mask))
        if known == 0:
            return 0.0
        cleaned = int(np.sum(self.cleaned_map[mask] == 1))
        dirt_left = int(np.sum(self.dirt_map[mask] == 1))
        return cleaned / max(1, cleaned + dirt_left)

    def _score_anchor(self, anchor: Tuple[int, int], robot_pos: Tuple[int, int]) -> float:
        if anchor not in self.charger_list:
            return -1e9
        idx = self.charger_list.index(anchor)
        mask = (self.anchor_region_map == idx) & (self.known_map == 1)
        known = int(np.sum(mask))
        if known == 0:
            return 0.2 - 0.01 * self._estimate_path_cost(robot_pos, anchor, use_danger=True)

        dirt_density = float(np.mean(self.dirt_map[mask]))
        uncleaned_ratio = float(np.sum(self.dirt_map[mask] == 1)) / max(1, known)
        frontier_score = float(np.sum((self.visit_count_map[mask] == 0).astype(np.float32))) / max(1, known)
        transfer_cost = self._estimate_path_cost(robot_pos, anchor, use_danger=True)
        robot_risk = float(self.robot_danger_map[anchor[1], anchor[0]])

        return 2.0 * uncleaned_ratio + 1.2 * dirt_density + 0.8 * frontier_score - 0.03 * transfer_cost - 0.8 * robot_risk

    def _select_next_anchor(self, robot_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        chargers = self._get_known_chargers()
        if len(chargers) <= 1:
            return None
        scored = sorted(chargers, key=lambda c: self._score_anchor(c, robot_pos), reverse=True)
        return scored[0] if scored else None

    def _should_transfer_anchor(self, robot_pos: Tuple[int, int], battery: int) -> bool:
        chargers = self._get_known_chargers()
        if len(chargers) <= 1 or self.current_anchor is None:
            return False
        if battery / Config.MAX_BATTERY < Config.FULL_BATTERY_RATIO:
            return False
        if abs(robot_pos[0] - self.current_anchor[0]) + abs(robot_pos[1] - self.current_anchor[1]) > 2:
            return False
        if self._anchor_coverage() < Config.ANCHOR_TRANSFER_COVERAGE:
            return False
        nxt = self._select_next_anchor(robot_pos)
        if nxt is None or nxt == self.current_anchor:
            return False
        return self._score_anchor(nxt, robot_pos) > self._score_anchor(self.current_anchor, robot_pos) + Config.MIN_SWITCH_GAIN

    def _best_action_towards(self, robot_pos: Tuple[int, int], target: Tuple[int, int], legal_actions: List[int]) -> int:
        best = legal_actions[0] if legal_actions else 0
        best_score = -float('inf')
        for action in legal_actions:
            nxt = self._next_pos(robot_pos, action)
            if not self._is_valid_pos(nxt):
                continue
            dist = abs(nxt[0] - target[0]) + abs(nxt[1] - target[1])
            danger = float(self.robot_danger_map[nxt[1], nxt[0]])
            revisit = float(self.visit_count_map[nxt[1], nxt[0]])
            anchor_bonus = 0.2 if self._cell_in_current_anchor(nxt) else -0.2
            score = -dist - 1.8 * danger - 0.03 * revisit + anchor_bonus
            if score > best_score:
                best_score = score
                best = action
        return best

    def _choose_safest_action(self, robot_pos: Tuple[int, int], legal_actions: List[int]) -> int:
        best = legal_actions[0] if legal_actions else 0
        best_score = -float('inf')
        for action in legal_actions:
            nxt = self._next_pos(robot_pos, action)
            if not self._is_valid_pos(nxt):
                continue
            danger = float(self.robot_danger_map[nxt[1], nxt[0]])
            revisit = float(self.visit_count_map[nxt[1], nxt[0]])
            anchor_bonus = 0.2 if self._cell_in_current_anchor(nxt) else -0.2
            score = -2.0 * danger - 0.05 * revisit + anchor_bonus
            if score > best_score:
                best_score = score
                best = action
        return best

    def _mode_emergency_evade(self, robot_pos: Tuple[int, int], legal_actions: List[int]) -> int:
        return self._choose_safest_action(robot_pos, legal_actions)

    def _mode_return_to_charger(self, robot_pos: Tuple[int, int], legal_actions: List[int]) -> int:
        chargers = self._get_known_chargers()
        path, best_charger, _ = self.return_handler.get_return_path(
            robot_pos,
            chargers,
            self.path_planner,
            obstacle_map=self.static_obstacle_map,
            danger_map=self.robot_danger_map,
        )
        if best_charger is not None:
            self.current_anchor = best_charger
        if path and len(path) > 1:
            action = self.path_planner.get_action_from_path(path, robot_pos, legal_actions)
            if action is not None:
                self.current_path = path
                return action
        if self.current_anchor is not None:
            return self._best_action_towards(robot_pos, self.current_anchor, legal_actions)
        return self._choose_safest_action(robot_pos, legal_actions)

    def _mode_stuck_recovery(self, robot_pos: Tuple[int, int], local_view: np.ndarray, legal_actions: List[int]) -> int:
        action = self.unstuck_handler.get_unstuck_action(local_view, legal_actions, robot_pos)
        if action not in legal_actions:
            action = self._choose_safest_action(robot_pos, legal_actions)
        if not self.unstuck_handler.is_stuck():
            self.unstuck_handler.reset()
        return action

    def _mode_charger_transfer(self, robot_pos: Tuple[int, int], battery: int, legal_actions: List[int]) -> int:
        next_anchor = self._select_next_anchor(robot_pos)
        if next_anchor is None:
            return self._mode_stripe_cleaning(robot_pos, battery, legal_actions)
        path = self.path_planner.astar(
            robot_pos,
            next_anchor,
            obstacle_map=self.static_obstacle_map,
            danger_map=self.robot_danger_map,
            danger_as_block=True,
        )
        if not path:
            path = self.path_planner.bfs(
                robot_pos,
                next_anchor,
                obstacle_map=self.static_obstacle_map,
                danger_map=self.robot_danger_map,
                danger_as_block=True,
                max_depth=240,
            )
        if path and len(path) > 1:
            action = self.path_planner.get_action_from_path(path, robot_pos, legal_actions)
            if action is not None:
                self.current_anchor = next_anchor
                self.current_path = path
                self.stripe_run_len = 0
                return action
        return self._best_action_towards(robot_pos, next_anchor, legal_actions)

    def _mode_stripe_cleaning(self, robot_pos: Tuple[int, int], battery: int, legal_actions: List[int]) -> int:
        primary = 2 if self.stripe_dir > 0 else 6
        if (
            self.stripe_run_len < Config.STRIPE_MAX_LEN
            and primary in legal_actions
            and self._action_energy_feasible(robot_pos, primary, battery)
            and self._cell_in_current_anchor(self._next_pos(robot_pos, primary))
        ):
            self.stripe_run_len += 1
            return primary

        shift_pref = 4 if self.stripe_shift_dir > 0 else 0
        shift_alt = 0 if shift_pref == 4 else 4
        for shift in [shift_pref, shift_alt]:
            if (
                shift in legal_actions
                and self._action_energy_feasible(robot_pos, shift, battery)
                and self._cell_in_current_anchor(self._next_pos(robot_pos, shift))
            ):
                if shift == shift_alt:
                    self.stripe_shift_dir *= -1
                self.stripe_dir *= -1
                self.stripe_run_len = 0
                self.stripe_rejoin_fail = 0
                return shift

        self.stripe_rejoin_fail += 1
        if self.stripe_rejoin_fail >= Config.STRIPE_REJOIN_BUDGET:
            self.stripe_rejoin_fail = 0
            return self._mode_safe_exploration(robot_pos, battery, legal_actions)
        return self._choose_safest_action(robot_pos, legal_actions)

    def _mode_direct_dirt_pickup(
        self,
        robot_pos: Tuple[int, int],
        local_view: np.ndarray,
        battery: int,
        legal_actions: List[int],
    ) -> int:
        nearest_dirt = self.mode_selector.get_nearest_dirt_pos(local_view, robot_pos)
        if nearest_dirt is None:
            return self._mode_safe_exploration(robot_pos, battery, legal_actions)

        path = self.path_planner.astar(
            robot_pos,
            nearest_dirt,
            obstacle_map=self.static_obstacle_map,
            danger_map=self.robot_danger_map,
            danger_as_block=False,
        )
        if path and len(path) > 1:
            action = self.path_planner.get_action_from_path(path, robot_pos, legal_actions)
            if action is not None and self._action_energy_feasible(robot_pos, action, battery):
                return action
        return self._best_action_towards(robot_pos, nearest_dirt, legal_actions)

    def _pick_frontier_target(self, robot_pos: Tuple[int, int], anchor_only: bool) -> Optional[Tuple[int, int]]:
        best_target = None
        best_score = -float('inf')
        for y in range(Config.MAP_SIZE):
            for x in range(Config.MAP_SIZE):
                if self.known_map[y, x] != 1 or self.free_map[y, x] != 1:
                    continue
                if anchor_only and not self._cell_in_current_anchor((x, y)):
                    continue
                if len(self.charger_map) == 0 and self.start_pos is not None:
                    if abs(x - self.start_pos[0]) + abs(y - self.start_pos[1]) > Config.SAFE_EXPLORE_RADIUS:
                        continue

                unknown_neighbors = 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < Config.MAP_SIZE and 0 <= ny < Config.MAP_SIZE and self.known_map[ny, nx] == 0:
                        unknown_neighbors += 1

                dist = abs(x - robot_pos[0]) + abs(y - robot_pos[1])
                if dist > 48:
                    continue

                score = (
                    2.0 * float(self.dirt_map[y, x])
                    + 0.9 * unknown_neighbors
                    - 0.04 * float(self.visit_count_map[y, x])
                    - 0.02 * dist
                    - 0.7 * float(self.robot_danger_map[y, x])
                )
                if score > best_score:
                    best_score = score
                    best_target = (x, y)
        return best_target

    def _mode_safe_exploration(self, robot_pos: Tuple[int, int], battery: int, legal_actions: List[int]) -> int:
        anchor_only = self.current_anchor is not None
        target = self._pick_frontier_target(robot_pos, anchor_only=anchor_only)
        if target is None and anchor_only:
            target = self._pick_frontier_target(robot_pos, anchor_only=False)
        if target is None:
            return self._choose_safest_action(robot_pos, legal_actions)

        self.target_pos = target
        path = self.path_planner.astar(
            robot_pos,
            target,
            obstacle_map=self.static_obstacle_map,
            danger_map=self.robot_danger_map,
            danger_as_block=False,
        )
        if not path:
            path = self.path_planner.bfs(
                robot_pos,
                target,
                obstacle_map=self.static_obstacle_map,
                danger_map=self.robot_danger_map,
                danger_as_block=False,
                max_depth=200,
            )
        if path and len(path) > 1:
            action = self.path_planner.get_action_from_path(path, robot_pos, legal_actions)
            if action is not None:
                self.current_path = path
                return action
        return self._best_action_towards(robot_pos, target, legal_actions)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_cleaned': self.total_cleaned,
            'current_mode': self.current_mode,
            'target_region': self.target_region,
            'step_count': self.step_count,
            'known_chargers': len(self.charger_map),
            'safe_energy': self._calc_safe_energy(self.last_pos) if self.last_pos is not None else 0,
            'anchor_coverage': self._anchor_coverage(),
        }


# ==================== 工具函数 ====================

def create_observation_dict(agent_maps, agent_pos, battery, step, legal_actions, **kwargs):
    """创建观测字典"""
    return {
        'agent_maps': agent_maps,
        'agent_pos': agent_pos,
        'battery': battery,
        'step': step,
        'legal_actions': legal_actions,
        **kwargs
    }


def preprocess_local_view(raw_view: np.ndarray) -> np.ndarray:
    """预处理局部视野"""
    # 假设 raw_view 是环境返回的原始数据
    # 转换为 (21, 21, 3) 格式
    if raw_view.shape != (21, 21, 3):
        # 需要reshape或填充
        processed = np.zeros((21, 21, 3), dtype=np.float32)
        h, w = min(raw_view.shape[0], 21), min(raw_view.shape[1], 21)
        processed[:h, :w] = raw_view[:h, :w]
        return processed
    return raw_view


# ==================== 主入口 ====================

def main():
    """主函数 - 用于测试"""
    print("=" * 60)
    print("清扫大作战 - 混合规则智能体")
    print("Robot Vacuum - Hybrid Rule-based Agent")
    print("=" * 60)
    
    # 创建智能体
    agent = CleaningAgent(use_neural=False)
    
    print("\n智能体模块初始化完成:")
    print(f"  - 特征金字塔: L0={Config.FEATURE_SCALES[0]}, L1={Config.FEATURE_SCALES[1]}, L2={Config.FEATURE_SCALES[2]}")
    print(f"  - 粗网格地图: {Config.COARSE_MAP_SIZE}x{Config.COARSE_MAP_SIZE}")
    print(f"  - 区域分割: {Config.REGION_SPLIT}x{Config.REGION_SPLIT} = {Config.REGION_SPLIT**2}个区域")
    print(f"  - 模式优先级: {Config.MODE_PRIORITY}")
    print(f"  - 前瞻深度: {Config.LOOKAHEAD_DEPTH}")
    
    # 模拟测试
    print("\n模拟测试运行...")
    
    # 创建模拟观测
    mock_view = np.random.randint(0, 3, (21, 21, 3))
    mock_obs = create_observation_dict(
        agent_maps=mock_view,
        agent_pos=(10, 10),
        battery=150,
        step=1,
        legal_actions=list(range(8))
    )
    
    # 执行决策
    action = agent.act(mock_obs)
    print(f"测试动作输出: {action} ({Config.ACTIONS[action]})")
    
    print("\n智能体准备就绪!")
    print("=" * 60)
    
    return agent


if __name__ == "__main__":
    agent = main()
