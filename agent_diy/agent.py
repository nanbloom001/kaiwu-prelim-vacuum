#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import heapq
import math
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from kaiwudrl.interface.agent import BaseAgent


class Config:
    MAP_SIZE = 128
    LOCAL_VIEW_SIZE = 21
    CHARGER_RADIUS = 1

    MAX_BATTERY = 200
    MAX_STEPS = 1000

    DETOUR_MARGIN = 6
    UNKNOWN_MARGIN = 4
    DOCK_BUFFER = 8
    ACTION_MOVE_COST = 1

    EARLY_CHARGE_THRESHOLD = 0.78
    NEAR_CHARGE_TOPUP_RATIO = 0.84
    PRESSURE_TOPUP_RATIO = 0.80
    CHARGE_NEAR_DISTANCE = 10
    CHARGE_SAFE_DISTANCE = 10
    CHARGE_RING_DISTANCE = 8
    CHARGE_PRESSURE_DISTANCE_BONUS = 3
    MIN_CLEAN_BEFORE_CHARGE = 8
    MIN_STEPS_BETWEEN_CHARGES = 24
    CHARGE_INTERVAL_STEPS = 44
    TARGET_CHARGE_COUNT = 22
    MIN_CHARGE_COUNT = 10
    FORCE_RETURN_AHEAD_MARGIN = 8
    NO_CHARGER_RESERVE = 80
    MIN_EXCURSION_BEFORE_RETURN = 18
    CHARGE_WORK_DISTANCE = 14
    CHARGE_HARD_LIMIT_DISTANCE = 22
    CHARGER_ORBIT_PENALTY_RADIUS = 6
    CHARGER_ORBIT_VISIT_TH = 2

    ROBOT_COLLISION_RADIUS = 1
    ROBOT_DANGER_BLOCK_TH = 1.2
    DANGER_EMERGENCY_TH = 0.9
    DANGER_DECAY = 0.92
    ROBOT_SAFE_MARGIN = 2

    SAFE_EXPLORE_RADIUS = 25
    STRIPE_MAX_LEN = 52
    STUCK_NO_PROGRESS_LIMIT = 3
    STUCK_LOOP_WINDOW = 12
    STUCK_LOOP_UNIQUE_LIMIT = 3
    STUCK_ESCAPE_STEPS = 5

    MODE_PRIORITY = {
        "EMERGENCY_EVADE": 120,
        "RETURN_TO_CHARGER": 100,
        "EARLY_CHARGE": 95,
        "STUCK_RECOVERY": 90,
        "CHARGER_TRANSFER": 80,
        "STRIPE_CLEANING": 70,
        "DIRECT_DIRT_PICKUP": 60,
        "SAFE_EXPLORATION": 50,
    }

    # Official action protocol.
    ACTIONS = [
        (1, 0),    # 0 right
        (1, -1),   # 1 right-up
        (0, -1),   # 2 up
        (-1, -1),  # 3 left-up
        (-1, 0),   # 4 left
        (-1, 1),   # 5 left-down
        (0, 1),    # 6 down
        (1, 1),    # 7 right-down
    ]


class Agent(BaseAgent):
    """Rule controller used directly by DIY or as PPO safety layer."""

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        super().__init__(agent_type, device, logger, monitor)
        self.logger = logger
        self.monitor = monitor
        self.device = device
        self._reset_episode_state()

    def _reset_episode_state(self):
        self.free_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int8)
        self.known_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int8)
        self.static_obstacle_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int8)
        self.visit_count_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int32)
        self.robot_danger_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        self.dirt_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.int8)

        self.charger_list: List[Tuple[int, int]] = []
        self.charger_set = set()
        self.robot_positions: List[Tuple[int, int]] = []
        self.prev_robot_positions: List[Tuple[int, int]] = []
        self.position_history: deque = deque(maxlen=Config.STUCK_LOOP_WINDOW)

        self.current_mode = "SAFE_EXPLORATION"
        self.current_target: Optional[Tuple[int, int]] = None
        self.current_target_kind = ""
        self.current_path: List[Tuple[int, int]] = []
        self.step_count = 0
        self.max_steps_runtime = Config.MAX_STEPS
        self.start_pos: Optional[Tuple[int, int]] = None
        self.last_pos: Optional[Tuple[int, int]] = None
        self.last_battery = Config.MAX_BATTERY
        self.battery_max_runtime = Config.MAX_BATTERY
        self.last_action = -1

        self.steps_since_last_charge = 0
        self.no_progress_steps = 0
        self.stuck_escape_steps = 0
        self.charge_count = 0
        self.prev_on_charger = False
        self.charge_guard_active = False
        self.last_charge_deficit = 0
        self.last_mode_actions = list(range(8))
        self.last_recommended_action = 0
        self.last_safe_energy = 0
        self.last_need_return = False
        self.last_early_charge = False
        self.last_emergency = False
        self.return_cost_cache = {}

    def _normalize_legal_actions(self, raw_legal) -> List[int]:
        if raw_legal is None:
            return list(range(8))
        vals = [int(v) for v in raw_legal]
        if len(vals) == 8 and all(v in (0, 1) for v in vals):
            return [i for i, v in enumerate(vals) if v == 1] or [0]
        acts = [a for a in vals if 0 <= a < 8]
        return acts or [0]

    def _build_local_grid(self, map_info) -> np.ndarray:
        if map_info is None:
            return np.ones((Config.LOCAL_VIEW_SIZE, Config.LOCAL_VIEW_SIZE), dtype=np.int8)
        arr = np.array(map_info)
        if arr.ndim == 2:
            h = min(arr.shape[0], Config.LOCAL_VIEW_SIZE)
            w = min(arr.shape[1], Config.LOCAL_VIEW_SIZE)
            out = np.ones((Config.LOCAL_VIEW_SIZE, Config.LOCAL_VIEW_SIZE), dtype=np.int8)
            out[:h, :w] = arr[:h, :w]
            return out.astype(np.int8)

        out = np.ones((Config.LOCAL_VIEW_SIZE, Config.LOCAL_VIEW_SIZE), dtype=np.int8)
        h = min(arr.shape[0], Config.LOCAL_VIEW_SIZE)
        w = min(arr.shape[1], Config.LOCAL_VIEW_SIZE)
        crop = arr[:h, :w]
        obstacle = crop[:, :, 0] >= 0.5
        dirt = crop[:, :, 1] >= 0.5 if crop.shape[2] >= 2 else np.zeros((h, w), dtype=bool)
        out[:h, :w] = 1
        out[:h, :w][obstacle] = 0
        out[:h, :w][dirt] = 2
        return out

    def _extract_obs(self, env_obs: Dict):
        observation = env_obs.get("observation", env_obs)
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        hero = frame_state.get("heroes", {})
        if isinstance(hero, list):
            hero = hero[0] if hero else {}

        pos_info = hero.get("pos", {})
        pos = (int(pos_info.get("x", 0)), int(pos_info.get("z", 0)))
        battery = int(hero.get("battery", env_info.get("remaining_charge", self.last_battery)))
        battery_max = int(hero.get("battery_max", env_info.get("battery_max", Config.MAX_BATTERY)))
        step = int(observation.get("step_no", env_obs.get("frame_no", 0)))
        legal_actions = self._normalize_legal_actions(
            observation.get("legal_act", observation.get("legal_action", env_obs.get("legal_actions")))
        )
        local_grid = self._build_local_grid(observation.get("map_info", env_obs.get("map_info")))

        chargers = []
        for organ in frame_state.get("organs", []) or []:
            if int(organ.get("sub_type", -1)) != 1:
                continue
            p = organ.get("pos", {})
            chargers.append((int(p.get("x", 0)), int(p.get("z", 0))))

        robots = []
        for npc in frame_state.get("npcs", []) or []:
            p = npc.get("pos", {})
            robots.append((int(p.get("x", 0)), int(p.get("z", 0))))

        env_charge_count = int(env_info.get("charge_count", self.charge_count))
        max_steps = int(env_info.get("max_step", env_info.get("max_steps", self.max_steps_runtime)))
        return pos, battery, battery_max, step, legal_actions, local_grid, chargers, robots, env_charge_count, max_steps

    def _next_pos(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        dx, dy = Config.ACTIONS[action]
        return pos[0] + dx, pos[1] + dy

    def _is_valid_pos(self, pos: Tuple[int, int]) -> bool:
        return 0 <= pos[0] < Config.MAP_SIZE and 0 <= pos[1] < Config.MAP_SIZE

    def _global_action_passable(self, pos: Tuple[int, int], action: int, block_danger: bool) -> bool:
        nxt = self._next_pos(pos, action)
        if not self._is_valid_pos(nxt):
            return False
        if self.static_obstacle_map[nxt[1], nxt[0]] == 1:
            return False
        if block_danger and self.robot_danger_map[nxt[1], nxt[0]] >= Config.ROBOT_DANGER_BLOCK_TH:
            return False
        if self._is_in_robot_danger_zone(nxt):
            return False

        dx, dy = Config.ACTIONS[action]
        if dx != 0 and dy != 0:
            side_a = (pos[0] + dx, pos[1])
            side_b = (pos[0], pos[1] + dy)
            side_a_blocked = self._is_valid_pos(side_a) and self.static_obstacle_map[side_a[1], side_a[0]] == 1
            side_b_blocked = self._is_valid_pos(side_b) and self.static_obstacle_map[side_b[1], side_b[0]] == 1
            if side_a_blocked and side_b_blocked:
                return False
        return True

    def _heuristic(self, start: Tuple[int, int], goal: Tuple[int, int]) -> float:
        return float(max(abs(start[0] - goal[0]), abs(start[1] - goal[1])))

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _astar_path(self, start: Tuple[int, int], goal: Tuple[int, int], block_danger: bool = True, max_iter: int = 12000) -> List[Tuple[int, int]]:
        if start == goal:
            return [start]
        open_heap = []
        heapq.heappush(open_heap, (0.0, start))
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score = {start: 0.0}
        closed = set()
        iterations = 0

        while open_heap and iterations < max_iter:
            iterations += 1
            _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self._reconstruct_path(came_from, current)
            closed.add(current)

            for action in range(len(Config.ACTIONS)):
                if not self._global_action_passable(current, action, block_danger):
                    continue
                nxt = self._next_pos(current, action)
                if nxt in closed:
                    continue
                tentative = g_score[current] + 1.0 + 0.03 * min(self.visit_count_map[nxt[1], nxt[0]], 6) + 0.25 * float(self.robot_danger_map[nxt[1], nxt[0]])
                if tentative < g_score.get(nxt, 1e18):
                    came_from[nxt] = current
                    g_score[nxt] = tentative
                    f_score = tentative + self._heuristic(nxt, goal)
                    heapq.heappush(open_heap, (f_score, nxt))
        return []

    def _action_from_path(self, path: List[Tuple[int, int]], pos: Tuple[int, int], legal_actions: List[int]) -> Optional[int]:
        if len(path) < 2:
            return None
        nxt = path[1]
        dx = nxt[0] - pos[0]
        dy = nxt[1] - pos[1]
        for action, (adx, ady) in enumerate(Config.ACTIONS):
            if dx == adx and dy == ady and action in legal_actions:
                return action
        return None

    def _estimate_path_cost(self, start: Tuple[int, int], goal: Tuple[int, int], block_danger: bool = True) -> float:
        key = (start, goal, block_danger)
        if key in self.return_cost_cache and self.step_count % 4 != 0:
            return self.return_cost_cache[key]
        path = self._astar_path(start, goal, block_danger=block_danger)
        if path:
            cost = float(len(path) - 1)
        else:
            cost = float(abs(start[0] - goal[0]) + abs(start[1] - goal[1]) + Config.DETOUR_MARGIN)
        self.return_cost_cache[key] = cost
        return cost

    def _register_charger(self, pos: Tuple[int, int]):
        for cx, cy in self.charger_list:
            if abs(pos[0] - cx) <= 2 and abs(pos[1] - cy) <= 2:
                return
        self.charger_list.append(pos)
        self.charger_set.add(pos)

    def _is_on_charger(self, pos: Tuple[int, int]) -> bool:
        for cx, cy in self.charger_list:
            if abs(pos[0] - cx) <= Config.CHARGER_RADIUS and abs(pos[1] - cy) <= Config.CHARGER_RADIUS:
                return True
        return False

    def _distance_to_nearest_charger(self, pos: Tuple[int, int]) -> int:
        if not self.charger_list:
            return 999
        best = 999
        for cx, cy in self.charger_list:
            dx = max(0, abs(pos[0] - cx) - Config.CHARGER_RADIUS)
            dy = max(0, abs(pos[1] - cy) - Config.CHARGER_RADIUS)
            best = min(best, dx + dy)
        return best

    def _target_charge_count(self) -> int:
        warmup_steps = 120
        effective_steps = max(0, self.step_count - warmup_steps)
        interval = max(1, int((self.max_steps_runtime - warmup_steps) / max(1, Config.TARGET_CHARGE_COUNT)))
        pace = effective_steps // interval
        return min(Config.TARGET_CHARGE_COUNT, pace)

    def _charge_deficit(self) -> int:
        return max(0, self._target_charge_count() - self.charge_count)

    def _charger_safe_distance(self) -> int:
        deficit = self._charge_deficit()
        if deficit <= 0:
            return Config.CHARGE_SAFE_DISTANCE
        return min(Config.CHARGE_SAFE_DISTANCE + Config.CHARGE_PRESSURE_DISTANCE_BONUS, Config.CHARGE_SAFE_DISTANCE + deficit)

    def _in_charger_safe_zone(self, pos: Tuple[int, int]) -> bool:
        return self._distance_to_nearest_charger(pos) <= self._charger_safe_distance()

    def _calc_safe_energy(self, pos: Tuple[int, int]) -> int:
        if not self.charger_list:
            return Config.NO_CHARGER_RESERVE
        min_return = min(self._estimate_path_cost(pos, charger, block_danger=True) for charger in self.charger_list)
        unknown_margin = Config.UNKNOWN_MARGIN if self.known_map[pos[1], pos[0]] == 0 else max(2, Config.UNKNOWN_MARGIN // 2)
        return min(self.battery_max_runtime - 1, int(min_return + Config.DETOUR_MARGIN + unknown_margin + Config.DOCK_BUFFER))

    def _should_enforce_charge_guard(self, pos: Tuple[int, int], battery: int, safe_energy: int) -> bool:
        if not self.charger_list:
            return False
        if (
            self.steps_since_last_charge < Config.MIN_EXCURSION_BEFORE_RETURN
            and battery > safe_energy + Config.DETOUR_MARGIN + Config.FORCE_RETURN_AHEAD_MARGIN + 8
        ):
            return False
        if self._charge_deficit() >= 2:
            return True
        if battery <= safe_energy + Config.DETOUR_MARGIN + Config.FORCE_RETURN_AHEAD_MARGIN:
            return True
        if self.step_count > self.max_steps_runtime * 0.8 and self.charge_count < Config.MIN_CHARGE_COUNT:
            return True
        return False

    def _must_return_now(self, battery: int, safe_energy: int) -> bool:
        margin = Config.FORCE_RETURN_AHEAD_MARGIN if self._charge_deficit() > 0 else 0
        return battery <= safe_energy + margin

    def _should_early_charge(self, pos: Tuple[int, int], battery: int, safe_energy: int) -> bool:
        if self._is_on_charger(pos) or not self.charger_list:
            return False
        if self.steps_since_last_charge < Config.MIN_CLEAN_BEFORE_CHARGE:
            return False
        if (
            self.steps_since_last_charge < Config.MIN_EXCURSION_BEFORE_RETURN
            and battery > safe_energy + Config.DETOUR_MARGIN + Config.FORCE_RETURN_AHEAD_MARGIN + 8
        ):
            return False
        dist_to_charger = self._distance_to_nearest_charger(pos)
        battery_ratio = float(battery) / max(1.0, float(self.battery_max_runtime))
        since_last = self.steps_since_last_charge
        deficit = self._charge_deficit()

        if battery <= safe_energy + Config.DETOUR_MARGIN + Config.FORCE_RETURN_AHEAD_MARGIN:
            return True
        if deficit > 0:
            if since_last >= Config.MIN_STEPS_BETWEEN_CHARGES and dist_to_charger <= Config.CHARGE_NEAR_DISTANCE + Config.CHARGE_PRESSURE_DISTANCE_BONUS:
                return True
            if since_last >= Config.MIN_STEPS_BETWEEN_CHARGES and battery_ratio <= Config.PRESSURE_TOPUP_RATIO:
                return True
        if dist_to_charger <= Config.CHARGE_NEAR_DISTANCE and battery_ratio <= Config.NEAR_CHARGE_TOPUP_RATIO:
            return True
        interval = max(1, int((self.max_steps_runtime - 120) / max(1, Config.TARGET_CHARGE_COUNT)))
        if since_last >= interval and battery_ratio <= Config.EARLY_CHARGE_THRESHOLD:
            return True
        return False

    def _update_maps(self, pos: Tuple[int, int], local_grid: np.ndarray):
        c = Config.LOCAL_VIEW_SIZE // 2
        for dy in range(-c, c + 1):
            for dx in range(-c, c + 1):
                gx, gy = pos[0] + dx, pos[1] + dy
                if not self._is_valid_pos((gx, gy)):
                    continue
                cell = int(local_grid[dy + c, dx + c])
                self.known_map[gy, gx] = 1
                if cell == 0:
                    self.static_obstacle_map[gy, gx] = 1
                    self.free_map[gy, gx] = 0
                    self.dirt_map[gy, gx] = 0
                else:
                    self.static_obstacle_map[gy, gx] = 0
                    self.free_map[gy, gx] = 1
                    self.dirt_map[gy, gx] = 1 if cell == 2 else 0

    def _update_robot_danger(self, robots: List[Tuple[int, int]]):
        self.robot_danger_map *= Config.DANGER_DECAY
        self.prev_robot_positions = self.robot_positions
        self.robot_positions = list(robots)
        for x, y in robots:
            vx, vy = 0, 0
            if self.prev_robot_positions:
                nearest = min(
                    self.prev_robot_positions,
                    key=lambda p: abs(p[0] - x) + abs(p[1] - y),
                )
                if abs(nearest[0] - x) + abs(nearest[1] - y) <= 3:
                    vx = max(-1, min(1, x - nearest[0]))
                    vy = max(-1, min(1, y - nearest[1]))

            for t in range(3):
                px = x + vx * t
                py = y + vy * t
                if not self._is_valid_pos((px, py)):
                    continue
                for ry in range(-3, 4):
                    for rx in range(-3, 4):
                        nx, ny = px + rx, py + ry
                        if not self._is_valid_pos((nx, ny)):
                            continue
                        dist = max(abs(rx), abs(ry))
                        risk = 0.0
                        if dist <= 1:
                            risk = 4.5 - 0.5 * t
                        elif dist == 2:
                            risk = 2.5 - 0.35 * t
                        elif dist == 3:
                            risk = 1.2 - 0.2 * t
                        if risk > 0:
                            self.robot_danger_map[ny, nx] = max(self.robot_danger_map[ny, nx], risk)

    def _is_in_robot_danger_zone(self, pos: Tuple[int, int]) -> bool:
        for rx, ry in self.robot_positions:
            if abs(pos[0] - rx) <= Config.ROBOT_COLLISION_RADIUS and abs(pos[1] - ry) <= Config.ROBOT_COLLISION_RADIUS:
                return True
        return False

    def _local_action_passable(self, local_grid: np.ndarray, action: int) -> bool:
        c = Config.LOCAL_VIEW_SIZE // 2
        dx, dy = Config.ACTIONS[action]
        tx, ty = c + dx, c + dy
        if not (0 <= tx < Config.LOCAL_VIEW_SIZE and 0 <= ty < Config.LOCAL_VIEW_SIZE):
            return False
        if int(local_grid[ty, tx]) == 0:
            return False
        if dx != 0 and dy != 0:
            a = int(local_grid[c, c + dx]) if 0 <= c + dx < Config.LOCAL_VIEW_SIZE else 0
            b = int(local_grid[c + dy, c]) if 0 <= c + dy < Config.LOCAL_VIEW_SIZE else 0
            if a == 0 and b == 0:
                return False
        return True

    def _safe_actions(self, pos: Tuple[int, int], battery: int, local_grid: np.ndarray, legal_actions: List[int]) -> List[int]:
        safe = []
        for action in legal_actions:
            nxt = self._next_pos(pos, action)
            if not self._is_valid_pos(nxt):
                continue
            if not self._local_action_passable(local_grid, action):
                continue
            if self._is_in_robot_danger_zone(nxt):
                continue
            if self.robot_danger_map[nxt[1], nxt[0]] >= Config.ROBOT_DANGER_BLOCK_TH:
                continue
            near_robot = min(
                [max(abs(nxt[0] - rx), abs(nxt[1] - ry)) for rx, ry in self.robot_positions] or [10]
            )
            if near_robot <= 2:
                continue
            battery_after = battery - Config.ACTION_MOVE_COST
            if battery_after <= 0:
                continue
            if self.charger_list and battery_after < self._calc_safe_energy(nxt):
                continue
            safe.append(action)
        return safe or legal_actions

    def _nearest_charger(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        if not self.charger_list:
            return None
        return min(self.charger_list, key=lambda c: abs(c[0] - pos[0]) + abs(c[1] - pos[1]))

    def _nearest_dirt(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        ys, xs = np.where(self.dirt_map == 1)
        if len(xs) == 0:
            return None
        idx = min(range(len(xs)), key=lambda i: abs(xs[i] - pos[0]) + abs(ys[i] - pos[1]))
        return int(xs[idx]), int(ys[idx])

    def _best_action_towards(self, pos: Tuple[int, int], target: Tuple[int, int], candidates: List[int]) -> int:
        path = self._astar_path(pos, target, block_danger=True)
        action = self._action_from_path(path, pos, candidates)
        if action is not None:
            return action

        best = candidates[0]
        best_score = -1e18
        for action in candidates:
            nxt = self._next_pos(pos, action)
            dist = abs(nxt[0] - target[0]) + abs(nxt[1] - target[1])
            danger = float(self.robot_danger_map[nxt[1], nxt[0]]) if self._is_valid_pos(nxt) else 10.0
            revisit = float(self.visit_count_map[nxt[1], nxt[0]]) if self._is_valid_pos(nxt) else 10.0
            robot_margin = min(
                [max(abs(nxt[0] - rx), abs(nxt[1] - ry)) for rx, ry in self.robot_positions] or [10]
            )
            charger_penalty = 0.0
            if self.charge_guard_active and not self._in_charger_safe_zone(nxt):
                charger_penalty = 6.0
            elif (not self.charge_guard_active) and self.charger_list:
                charger_dist = self._distance_to_nearest_charger(nxt)
                if charger_dist <= Config.CHARGER_ORBIT_PENALTY_RADIUS and self.last_battery > self.last_safe_energy + 12:
                    charger_penalty += 0.8 * (Config.CHARGER_ORBIT_PENALTY_RADIUS - charger_dist + 1)
                if charger_dist > Config.CHARGE_HARD_LIMIT_DISTANCE:
                    charger_penalty += 1.2 * (charger_dist - Config.CHARGE_HARD_LIMIT_DISTANCE)
            robot_penalty = 3.5 * max(0, 3 - robot_margin)
            score = -dist - 3.0 * danger - 0.08 * revisit - charger_penalty - robot_penalty
            if score > best_score:
                best_score = score
                best = action
        return best

    def _pick_frontier_target(self, pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        best = None
        best_score = -1e18
        for y in range(Config.MAP_SIZE):
            for x in range(Config.MAP_SIZE):
                if self.known_map[y, x] != 1 or self.free_map[y, x] != 1:
                    continue
                if self.charge_guard_active and not self._in_charger_safe_zone((x, y)):
                    continue
                unknown_neighbors = 0
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < Config.MAP_SIZE and 0 <= ny < Config.MAP_SIZE and self.known_map[ny, nx] == 0:
                        unknown_neighbors += 1
                if unknown_neighbors == 0:
                    continue
                dist = abs(x - pos[0]) + abs(y - pos[1])
                if dist > 60:
                    continue
                score = 0.9 * unknown_neighbors - 0.02 * dist - 0.12 * self.visit_count_map[y, x] - 0.7 * self.robot_danger_map[y, x]
                if self._charge_deficit() > 0:
                    score += 0.12 * max(0, Config.CHARGE_SAFE_DISTANCE - self._distance_to_nearest_charger((x, y)))
                elif self.charger_list:
                    charger_dist = self._distance_to_nearest_charger((x, y))
                    ring_bonus = 1.1 - 0.08 * abs(charger_dist - Config.CHARGE_WORK_DISTANCE)
                    score += max(-1.0, ring_bonus)
                    if charger_dist <= Config.CHARGER_ORBIT_PENALTY_RADIUS:
                        score -= 0.6 * (Config.CHARGER_ORBIT_PENALTY_RADIUS - charger_dist + 1)
                    if charger_dist > Config.CHARGE_HARD_LIMIT_DISTANCE:
                        score -= 0.5 * (charger_dist - Config.CHARGE_HARD_LIMIT_DISTANCE)
                if score > best_score:
                    best_score = score
                    best = (x, y)
        return best

    def _safe_target_reachable(self, pos: Tuple[int, int], target: Optional[Tuple[int, int]], battery: int) -> bool:
        if target is None:
            return False
        path_cost = self._estimate_path_cost(pos, target, block_danger=True)
        if not math.isfinite(path_cost):
            return False
        battery_after = battery - path_cost
        if battery_after <= 0:
            return False
        target_safe_energy = self._calc_safe_energy(target)
        return battery_after >= target_safe_energy

    def _mode_emergency(self, pos: Tuple[int, int], candidates: List[int]) -> int:
        best = candidates[0]
        best_score = -1e18
        for action in candidates:
            nxt = self._next_pos(pos, action)
            if not self._is_valid_pos(nxt):
                continue
            robot_dist = min(
                [max(abs(nxt[0] - rx), abs(nxt[1] - ry)) for rx, ry in self.robot_positions] or [10]
            )
            danger = float(self.robot_danger_map[nxt[1], nxt[0]])
            score = 3.5 * robot_dist - 2.0 * danger - 0.05 * self.visit_count_map[nxt[1], nxt[0]]
            if score > best_score:
                best_score = score
                best = action
        return best

    def _is_stuck(self, pos: Tuple[int, int]) -> bool:
        if self.no_progress_steps >= Config.STUCK_NO_PROGRESS_LIMIT:
            return True
        if len(self.position_history) >= Config.STUCK_LOOP_WINDOW:
            recent = list(self.position_history)
            if len(set(recent)) <= Config.STUCK_LOOP_UNIQUE_LIMIT:
                return True
        return False

    def _mode_stuck(self, pos: Tuple[int, int], battery: int, local_grid: np.ndarray, legal_actions: List[int]) -> int:
        best = legal_actions[0]
        best_score = -1e18
        recent = set(self.position_history)
        charger = self._nearest_charger(pos)
        for action in legal_actions:
            if not self._local_action_passable(local_grid, action):
                continue
            nxt = self._next_pos(pos, action)
            if not self._is_valid_pos(nxt):
                continue
            if self._is_in_robot_danger_zone(nxt):
                continue
            if self.robot_danger_map[nxt[1], nxt[0]] >= Config.ROBOT_DANGER_BLOCK_TH:
                continue
            battery_after = battery - Config.ACTION_MOVE_COST
            if battery_after <= 0:
                continue
            if self.charger_list and battery_after < self._calc_safe_energy(nxt):
                continue
            novelty = 2.5 if nxt not in recent else -1.5
            wall_clear = 0.0
            for ax, ay in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = nxt[0] + ax, nxt[1] + ay
                if self._is_valid_pos((nx, ny)) and self.free_map[ny, nx] == 1:
                    wall_clear += 0.5
            charger_bias = 0.0
            if charger is not None:
                charger_dist = self._distance_to_nearest_charger(nxt)
                if self.charge_guard_active:
                    charger_bias += min(charger_dist, Config.CHARGE_WORK_DISTANCE) * 0.05
                else:
                    charger_bias += 0.2 * min(abs(charger_dist - Config.CHARGE_WORK_DISTANCE), 10)
            danger = float(self.robot_danger_map[nxt[1], nxt[0]])
            score = novelty + wall_clear - 1.5 * danger - 0.1 * self.visit_count_map[nxt[1], nxt[0]] - charger_bias
            if score > best_score:
                best_score = score
                best = action
        return best

    def act(self, env_obs: Dict) -> int:
        pos, battery, battery_max, step, legal_actions, local_grid, chargers, robots, env_charge_count, max_steps = self._extract_obs(env_obs)
        if step <= 1 and self.step_count > 10:
            self._reset_episode_state()

        self.step_count = step
        self.max_steps_runtime = max(1, max_steps)
        self.battery_max_runtime = max(1, battery_max)
        if self.start_pos is None:
            self.start_pos = pos
        for charger in chargers:
            self._register_charger(charger)

        self._update_maps(pos, local_grid)
        self._update_robot_danger(robots)
        self.charge_count = max(self.charge_count, env_charge_count)
        self.position_history.append(pos)

        if self.last_pos is not None and pos == self.last_pos and self.last_action >= 0:
            self.no_progress_steps += 1
        else:
            self.no_progress_steps = max(0, self.no_progress_steps - 1)

        on_charger = self._is_on_charger(pos)
        if on_charger or self.charge_count > 0 and battery >= self.battery_max_runtime - 1 and self.last_battery < battery:
            self.steps_since_last_charge = 0
        else:
            self.steps_since_last_charge += 1

        safe_energy = self._calc_safe_energy(pos)
        self.last_safe_energy = safe_energy
        self.last_charge_deficit = self._charge_deficit()
        self.charge_guard_active = self._should_enforce_charge_guard(pos, battery, safe_energy)
        need_return = self._must_return_now(battery, safe_energy)
        early_charge = self._should_early_charge(pos, battery, safe_energy)
        emergency = self._is_in_robot_danger_zone(pos) or float(self.robot_danger_map[pos[1], pos[0]]) >= Config.DANGER_EMERGENCY_TH

        safe_actions = self._safe_actions(pos, battery, local_grid, legal_actions)
        self.last_mode_actions = list(safe_actions)
        stuck = self._is_stuck(pos)

        if emergency:
            self.current_mode = "EMERGENCY_EVADE"
            action = self._mode_emergency(pos, safe_actions)
        elif stuck:
            self.current_mode = "STUCK_RECOVERY"
            action = self._mode_stuck(pos, battery, local_grid, safe_actions)
        elif need_return:
            self.current_mode = "RETURN_TO_CHARGER"
            target = self._nearest_charger(pos) or pos
            action = self._best_action_towards(pos, target, safe_actions)
        elif early_charge:
            self.current_mode = "EARLY_CHARGE"
            target = self._nearest_charger(pos) or pos
            action = self._best_action_towards(pos, target, safe_actions)
        else:
            dirt_target = self._nearest_dirt(pos)
            if (
                dirt_target is not None
                and ((not self.charge_guard_active) or self._in_charger_safe_zone(dirt_target))
                and self._safe_target_reachable(pos, dirt_target, battery)
            ):
                self.current_mode = "DIRECT_DIRT_PICKUP"
                action = self._best_action_towards(pos, dirt_target, safe_actions)
            else:
                self.current_mode = "SAFE_EXPLORATION"
                frontier = self._pick_frontier_target(pos)
                if frontier is not None and not self._safe_target_reachable(pos, frontier, battery):
                    frontier = None
                if frontier is None and self.charge_guard_active and self.charger_list:
                    self.current_mode = "EARLY_CHARGE"
                    frontier = self._nearest_charger(pos)
                action = self._best_action_towards(pos, frontier or pos, safe_actions)

        self.last_recommended_action = int(action)
        self.last_need_return = need_return
        self.last_early_charge = early_charge
        self.last_emergency = emergency or stuck
        self.last_battery = battery
        self.last_pos = pos
        self.prev_on_charger = on_charger
        self.visit_count_map[pos[1], pos[0]] += 1
        self.last_action = int(action)
        return int(action)

    def get_hybrid_context(self, robot_pos=None, battery=None):
        pos = robot_pos if robot_pos is not None else (self.last_pos or (0, 0))
        battery_val = battery if battery is not None else self.last_battery
        return {
            "mode": self.current_mode,
            "recommended_action": self.last_recommended_action,
            "mode_actions": list(self.last_mode_actions),
            "safe_energy": self.last_safe_energy,
            "need_return": self.last_need_return,
            "early_charge": self.last_early_charge,
            "emergency": self.last_emergency,
            "charge_count": self.charge_count,
            "target_charge_count": self._target_charge_count(),
            "charge_deficit": self.last_charge_deficit,
            "charge_guard_active": self.charge_guard_active,
            "steps_since_last_charge": self.steps_since_last_charge,
            "known_chargers": len(self.charger_list),
            "distance_to_charger": self._distance_to_nearest_charger(pos),
            "in_charger_safe_zone": self._in_charger_safe_zone(pos),
            "on_charger": self._is_on_charger(pos),
            "battery_ratio": float(battery_val) / max(1.0, float(self.battery_max_runtime)),
            "danger_here": float(self.robot_danger_map[pos[1], pos[0]]) if self._is_valid_pos(pos) else 0.0,
        }

    # Stubs for DIY template compatibility.
    def predict(self, list_obs_data):
        out = []
        for obs in list_obs_data:
            raw = obs.obs if hasattr(obs, "obs") else obs
            out.append(self.act(raw))
        return out

    def exploit(self, list_obs_data):
        return self.predict(list_obs_data)

    def learn(self, list_sample_data):
        return None

    def save_model(self, path=None, id="1"):
        return None

    def load_model(self, path=None, id="1"):
        return None

    def observation_process(self, obs, preprocessor=None, extra_info=None):
        return obs, extra_info

    def action_process(self, act_data):
        return act_data
