#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Rule-based coverage planner with edge-first exploration, charger return and NPC avoidance.
"""

from __future__ import annotations

from dataclasses import dataclass
import heapq
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


Position = Tuple[int, int]


@dataclass
class PolicyInfo:
    safe_action_mask: np.ndarray
    action_scores: np.ndarray
    chosen_action: int
    greedy_action: int
    target_mode: str
    target_pos: Optional[Position]
    target_distance: float
    battery: float
    battery_ratio: float
    charger_distance: float
    charger_slack: float
    nearest_npc_distance: float
    frontier_density: float
    local_dirty_ratio: float
    local_unknown_ratio: float
    new_known_cells: int
    on_charger: bool
    should_charge: bool


class CoveragePlanner:
    UNKNOWN = -1
    OBSTACLE = 0
    CLEAN = 1
    DIRT = 2

    MAP_SIZE = 128
    VIEW_RADIUS = 10
    ACTION_TO_DELTA: Sequence[Position] = (
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )
    DIAGONAL_ACTIONS = {1, 3, 5, 7}
    PATH_ACTION_ORDER = (1, 3, 5, 7, 0, 2, 4, 6)

    BASE_RETURN_MARGIN = 26.0
    NPC_RETURN_MARGIN = 16.0
    RETURN_RETRY_MARGIN = 10.0
    LOW_BATTERY_RATIO = 0.36
    FORCE_CHARGE_RATIO = 0.66
    FORCE_CHARGE_MIN_STEP = 80
    FORCE_CHARGE_EXIT_RATIO = 0.82
    POST_CHARGE_EXPLORE_STEPS = 180
    EXIT_RETURN_RATIO = 0.98
    EXPANSION_KNOWN_RATIO = 0.78
    EXPANSION_STEP_LIMIT = 450
    HARD_NPC_RADIUS = 1
    PATH_RISK_RADIUS = 3
    AGGRESSIVE_EDGE_STEPS = 500
    CHARGE_APPROACH_RADIUS = 6
    DEAD_END_DEGREE = 1
    NARROW_DEGREE = 2
    DEAD_END_PENALTY = 3.0
    NARROW_PENALTY = 1.2
    NPC_BLOCK_RADIUS = 2
    CHARGER_PATH_TRIGGER_RATIO = 0.55
    CHARGER_REPLAN_INTERVAL = 12
    BLOCKED_CELL_TTL = 8

    def __init__(self):
        self.reset()

    def reset(self):
        self.global_map = np.full((self.MAP_SIZE, self.MAP_SIZE), self.UNKNOWN, dtype=np.int8)
        self.visit_count = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.int16)
        self.hero_cleaned = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.uint8)
        self.npc_cleaned = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.uint8)
        self.charger_regions: List[Set[Position]] = []
        self.step_no = 0
        self.return_mode = False
        self.force_charge_mode = False
        self.last_force_charge_exit_step = -100000
        self.current_goal: Optional[Position] = None
        self.current_mode = "explore"
        self.last_policy_info: Optional[PolicyInfo] = None
        self.current_step = 0
        self.current_path: List[Position] = []
        self.prev_hero_pos: Optional[Position] = None
        self.blocked_cells: Dict[Position, int] = {}
        self._unknown_prefix = np.zeros((self.MAP_SIZE + 1, self.MAP_SIZE + 1), dtype=np.int32)
        self._dirt_prefix = np.zeros((self.MAP_SIZE + 1, self.MAP_SIZE + 1), dtype=np.int32)
        self._frontier_map = np.zeros((self.MAP_SIZE, self.MAP_SIZE), dtype=np.uint8)
        self._known_bbox_cache: Optional[Tuple[int, int, int, int]] = None
        self._trap_penalty_cache: Dict[Position, float] = {}
        self._free_degree_cache: Dict[Position, int] = {}
        self._npc_distance_cache: Dict[Position, float] = {}
    def update(self, env_obs: Any, last_action: int = -1) -> PolicyInfo:
        obs = self._extract_observation(env_obs)
        env_info = self._get(obs, "env_info", {})
        frame_state = self._get(obs, "frame_state", {})
        hero = self._extract_hero(frame_state)
        npcs = self._extract_npcs(frame_state)
        organs = self._extract_organs(frame_state)

        hero_pos = self._extract_pos(self._get(hero, "pos", self._get(env_info, "pos", {})))
        map_grid = self._extract_map_grid(obs)
        battery = self._safe_float(self._get(hero, "battery", self._get(env_info, "remaining_charge", 0)), 0.0)
        battery_max = self._safe_float(self._get(hero, "battery_max", self._get(env_info, "battery_max", 200)), 200.0)
        battery_ratio = battery / max(battery_max, 1.0)
        self.current_step = int(self._safe_float(self._get(obs, "step_no", 0), 0.0))
        self._decay_blocked_cells()
        self._update_stuck_memory(hero_pos, last_action)

        self.charger_regions = self._extract_charger_regions(organs)
        cleaned_cells = self._extract_positions(self._get(env_info, "step_cleaned_cells", []))
        new_known_cells = self._update_global_memory(hero_pos, map_grid, cleaned_cells)
        self._mark_cleaned_cells(cleaned_cells)
        self._mark_charger_cells()
        self._rebuild_step_caches()
        self.visit_count[hero_pos[1], hero_pos[0]] = min(np.iinfo(np.int16).max, self.visit_count[hero_pos[1], hero_pos[0]] + 1)

        legal_mask = self._build_local_legal_mask(map_grid)
        safe_mask = self._apply_npc_safety_mask(hero_pos, legal_mask, npcs)

        on_charger = self._is_on_charger(hero_pos)
        if self.force_charge_mode and on_charger and battery_ratio >= self.FORCE_CHARGE_EXIT_RATIO:
            self.force_charge_mode = False
            self.return_mode = False
            self.current_goal = None
            self.current_mode = "explore"
            self.last_force_charge_exit_step = self.current_step
        if on_charger and battery_ratio >= self.EXIT_RETURN_RATIO:
            self.return_mode = False
            self.current_goal = None
            self.current_mode = "explore"

        charger_known = bool(self.charger_regions)
        charger_distance = float("inf")
        charger_path: List[Position] = []
        charger_target: Optional[Position] = None
        charger_heuristic = self._heuristic_to_closest_charger(hero_pos) if charger_known else float("inf")
        extra_return_margin = 0.0
        if charger_known:
            charger_path, charger_distance, charger_target = self._reuse_charge_path(hero_pos, npcs)
            if not charger_path:
                charger_path, charger_distance, charger_target = self._plan_to_charger(hero_pos, npcs)
        if not charger_path and charger_known:
            charger_distance = charger_heuristic

        retry_return_margin = self._charge_retry_margin(charger_path)
        force_charge = (
            charger_known
            and self.current_step >= self.FORCE_CHARGE_MIN_STEP
            and battery_ratio <= self.FORCE_CHARGE_RATIO
        )
        if force_charge:
            self.force_charge_mode = True
        force_charge = self.force_charge_mode and charger_known
        should_charge = (
            self.force_charge_mode
            or
            self.return_mode
            or (
                charger_known
                and np.isfinite(charger_distance)
                and battery <= charger_distance + self.BASE_RETURN_MARGIN + retry_return_margin + extra_return_margin
            )
            or force_charge
            or (charger_known and battery_ratio <= self.LOW_BATTERY_RATIO)
        )
        if should_charge:
            self.return_mode = True
            if not charger_path:
                charger_path, charger_distance, charger_target = self._plan_to_charger(hero_pos, npcs)
            if not charger_path and charger_known:
                charger_distance = charger_heuristic
            retry_return_margin = self._charge_retry_margin(charger_path)
            extra_return_margin = self.NPC_RETURN_MARGIN if self._path_has_npc_risk(charger_path, npcs) else 0.0
            force_charge = (
                charger_known
                and self.current_step >= self.FORCE_CHARGE_MIN_STEP
                and battery_ratio <= self.FORCE_CHARGE_RATIO
            )
            if force_charge:
                self.force_charge_mode = True
            force_charge = self.force_charge_mode and charger_known
            should_charge = (
                self.force_charge_mode
                or
                self.return_mode
                or (
                    charger_known
                    and np.isfinite(charger_distance)
                    and battery <= charger_distance + self.BASE_RETURN_MARGIN + retry_return_margin + extra_return_margin
                )
                or force_charge
                or (charger_known and battery_ratio <= self.LOW_BATTERY_RATIO)
            )

        if should_charge:
            if charger_path and np.isfinite(charger_distance):
                target_mode = "charge"
                target_pos = charger_target
                path = charger_path
                target_distance = charger_distance
            else:
                target_mode, target_pos, target_distance, path = self._select_charge_approach_target(hero_pos, npcs)
        else:
            target_mode, target_pos, target_distance, path = self._select_explore_target(hero_pos, battery, charger_distance, npcs)

        if target_mode in ("charge", "charge_approach"):
            safe_mask = self._tighten_corner_mask(map_grid, safe_mask)
            safe_mask = self._relax_charge_mask(hero_pos, map_grid, safe_mask, legal_mask, npcs)
            if self.force_charge_mode:
                safe_mask = self._lock_force_charge_mask(hero_pos, safe_mask, legal_mask, path, npcs)

        action_scores = self._score_actions(
            hero_pos=hero_pos,
            safe_mask=safe_mask,
            last_action=last_action,
            npcs=npcs,
            target_pos=target_pos,
            target_mode=target_mode,
            path=path,
            should_charge=should_charge,
        )

        if safe_mask.sum() <= 0:
            action_scores = self._stall_or_escape_scores(hero_pos, legal_mask, map_grid, npcs)
            fallback_action = int(np.argmax(action_scores))
            safe_mask = np.zeros((8,), dtype=np.float32)
            safe_mask[fallback_action] = 1.0

        chosen_action = self._choose_action(hero_pos, target_mode, path, safe_mask, action_scores)
        frontier_density = self._frontier_density(hero_pos)
        local_dirty_ratio = float(np.mean(map_grid == self.DIRT))
        local_unknown_ratio = float(self._count_unknown_near(hero_pos, 2)) / 25.0
        nearest_npc_distance = self._nearest_npc_distance(hero_pos, npcs)
        charger_slack = battery - charger_distance if np.isfinite(charger_distance) else battery

        policy_info = PolicyInfo(
            safe_action_mask=safe_mask.astype(np.float32),
            action_scores=action_scores.astype(np.float32),
            chosen_action=chosen_action,
            greedy_action=chosen_action,
            target_mode=target_mode,
            target_pos=target_pos,
            target_distance=float(target_distance if np.isfinite(target_distance) else 999.0),
            battery=float(battery),
            battery_ratio=float(battery_ratio),
            charger_distance=float(charger_distance if np.isfinite(charger_distance) else 999.0),
            charger_slack=float(charger_slack),
            nearest_npc_distance=float(nearest_npc_distance),
            frontier_density=float(frontier_density),
            local_dirty_ratio=float(local_dirty_ratio),
            local_unknown_ratio=float(local_unknown_ratio),
            new_known_cells=int(new_known_cells),
            on_charger=bool(on_charger),
            should_charge=bool(should_charge),
        )
        self.last_policy_info = policy_info
        self.current_goal = target_pos
        self.current_mode = target_mode
        self.current_path = list(path) if path else []
        self.prev_hero_pos = hero_pos
        return policy_info

    def _select_explore_target(
        self,
        hero_pos: Position,
        battery: float,
        charger_distance: float,
        npcs: Sequence[Position],
    ) -> Tuple[str, Optional[Position], float, List[Position]]:
        if self._goal_is_still_useful(self.current_goal, hero_pos):
            cached_path, cached_distance = self._use_cached_path(hero_pos, npcs)
            if cached_path:
                return self.current_mode, self.current_goal, cached_distance, cached_path
            path, distance = self._plan_to_targets(hero_pos, [self.current_goal], False, npcs)
            if path and np.isfinite(distance):
                return self.current_mode, self.current_goal, distance, path

        distance_map = self._distance_map(hero_pos)
        known_ratio = float(np.count_nonzero(self.global_map != self.UNKNOWN)) / float(self.MAP_SIZE * self.MAP_SIZE)
        charger_known = bool(self.charger_regions)
        post_charge_explore = (self.current_step - self.last_force_charge_exit_step) <= self.POST_CHARGE_EXPLORE_STEPS
        expansion_phase = (
            self.current_step <= self.AGGRESSIVE_EDGE_STEPS
            or known_ratio < self.EXPANSION_KNOWN_RATIO
            or not charger_known
            or post_charge_explore
        )
        known_bbox = self._known_bbox()

        best_score = -1e9
        best_target: Optional[Position] = None
        best_mode = "frontier"
        reserve = self.BASE_RETURN_MARGIN + 4.0

        candidate_mask = (distance_map >= 0) & ((self.global_map == self.DIRT) | (self._frontier_map > 0))
        candidate_coords = np.argwhere(candidate_mask)

        for z, x in candidate_coords:
            dist = distance_map[z, x]
            pos = (int(x), int(z))
            cell = int(self.global_map[z, x])
            is_frontier = bool(self._frontier_map[z, x])

            charger_need = self._heuristic_to_closest_charger(pos) if charger_known else 0.0
            if charger_known and battery <= dist + charger_need + reserve:
                continue

            info_gain = self._count_unknown_near(pos, 2)
            dirt_gain = self._count_dirt_near(pos, 2)
            visit_penalty = float(min(8, int(self.visit_count[z, x])))
            edge_bonus = self._edge_bonus(pos, known_bbox)
            npc_clean_penalty = 1.6 * float(self.npc_cleaned[z, x])
            trap_penalty = self._trap_penalty(pos)
            blocked_penalty = 2.0 if pos in self.blocked_cells else 0.0
            diagonal_bonus = 1.0 if abs(x - hero_pos[0]) == abs(z - hero_pos[1]) else 0.0

            score = (
                (4.8 if post_charge_explore else 3.6) * float(info_gain)
                + (1.8 if post_charge_explore else 2.6) * float(dirt_gain)
                + ((1.0 if post_charge_explore else 2.4) if cell == self.DIRT else 0.0)
                - (0.36 if post_charge_explore else 0.42) * float(dist)
                - (0.55 if post_charge_explore else 0.30) * visit_penalty
                - npc_clean_penalty
                - trap_penalty
                - blocked_penalty
                + 0.22 * diagonal_bonus
            )
            if expansion_phase:
                score += (6.8 if post_charge_explore else (4.2 if not charger_known else 2.8)) * edge_bonus
                score += (
                    5.0 if post_charge_explore and is_frontier else
                    (2.4 if not charger_known and is_frontier else (1.5 if is_frontier else 0.0))
                )
                if not charger_known and edge_bonus < 0.25 and cell != self.DIRT:
                    score -= 2.0
            else:
                score += 0.7 * edge_bonus
            if post_charge_explore:
                score += 2.4 * float(info_gain)
                score += 2.0 if is_frontier else 0.0
                score += 1.0 * self._map_edge_bonus(pos)
                if self.visit_count[z, x] > 0:
                    score -= 1.2

            if score > best_score:
                best_score = score
                best_target = pos
                best_mode = "find_charger_edge" if (not charger_known and expansion_phase) else ("edge_frontier" if expansion_phase and is_frontier else ("frontier" if is_frontier else "dirt"))

        if best_target is None:
            return "fallback", None, float("inf"), []
        path, distance = self._plan_to_targets(hero_pos, [best_target], False, npcs)
        return best_mode, best_target, distance, path

    def _select_charge_approach_target(
        self,
        hero_pos: Position,
        npcs: Sequence[Position],
    ) -> Tuple[str, Optional[Position], float, List[Position]]:
        if not self.charger_regions:
            return "fallback", None, float("inf"), []

        distance_map = self._distance_map(hero_pos)
        best_score = -1e9
        best_target: Optional[Position] = None

        for z in range(self.MAP_SIZE):
            for x in range(self.MAP_SIZE):
                dist = distance_map[z, x]
                if dist < 0:
                    continue
                pos = (x, z)
                cell = int(self.global_map[z, x])
                if cell == self.OBSTACLE:
                    continue

                charger_need = self._heuristic_to_closest_charger(pos)
                if charger_need > self.CHARGE_APPROACH_RADIUS and not self._is_frontier(pos):
                    continue

                score = 0.0
                score -= 0.40 * float(dist)
                score -= 0.85 * float(charger_need)
                score += 0.35 * float(self._count_unknown_near(pos, 1))
                score += 0.60 if self._is_frontier(pos) else 0.0
                score -= 1.4 * float(self.npc_cleaned[z, x])
                score -= self._trap_penalty(pos)
                score -= 2.0 if pos in self.blocked_cells else 0.0
                score -= 0.30 * max(0.0, 6.0 - self._nearest_npc_distance(pos, npcs))

                if score > best_score:
                    best_score = score
                    best_target = pos

        if best_target is None:
            return "fallback", None, float("inf"), []

        path, distance = self._plan_to_targets(hero_pos, [best_target], False, npcs)
        if path and np.isfinite(distance):
            return "charge_approach", best_target, distance, path
        return "fallback", None, float("inf"), []
    def _plan_to_charger(self, hero_pos: Position, npcs: Sequence[Position]) -> Tuple[List[Position], float, Optional[Position]]:
        charger_targets = self._charger_target_cells()
        if not charger_targets:
            return [], float("inf"), None

        path, distance = self._plan_to_targets(hero_pos, charger_targets, False, npcs)
        if path:
            return path, distance, path[-1]
        path, distance = self._plan_to_targets(hero_pos, charger_targets, True, npcs)
        return path, distance, (path[-1] if path else None)

    def _reuse_charge_path(self, hero_pos: Position, npcs: Sequence[Position]) -> Tuple[List[Position], float, Optional[Position]]:
        if self.current_mode not in ("charge", "charge_approach"):
            return [], float("inf"), None
        path, distance = self._use_cached_path(hero_pos, npcs)
        if not path:
            return [], float("inf"), None
        target = path[-1] if path else None
        if target is None:
            return [], float("inf"), None
        if self.current_mode == "charge" and not self._is_on_charger(target):
            return [], float("inf"), None
        return path, distance, target

    def _plan_to_targets(
        self,
        start: Position,
        targets: Sequence[Position],
        allow_unknown: bool,
        npcs: Sequence[Position],
    ) -> Tuple[List[Position], float]:
        target_set = {pos for pos in targets if self._in_bounds(pos)}
        if not target_set or not self._in_bounds(start):
            return [], float("inf")
        if start in target_set:
            return [start], 0.0

        pq: List[Tuple[float, float, Position]] = [(0.0, 0.0, start)]
        best_cost: Dict[Position, float] = {start: 0.0}
        parent: Dict[Position, Position] = {}

        while pq:
            _, cost, cur = heapq.heappop(pq)
            if cost > best_cost.get(cur, float("inf")) + 1e-6:
                continue
            if cur in target_set:
                path = self._reconstruct_path(parent, start, cur)
                return path, float(max(0, len(path) - 1))

            for action in self.PATH_ACTION_ORDER:
                nxt = self._move(cur, action)
                if not self._can_traverse(cur, nxt, allow_unknown):
                    continue
                if self._hard_npc_zone(nxt, npcs) and nxt not in target_set:
                    continue

                step_cost = 1.0
                if self.global_map[nxt[1], nxt[0]] == self.UNKNOWN:
                    step_cost += 2.0
                step_cost += 0.15 * min(5, int(self.visit_count[nxt[1], nxt[0]]))
                step_cost += 0.9 * float(self.npc_cleaned[nxt[1], nxt[0]])
                step_cost += 0.6 * self._trap_penalty(nxt)
                if nxt in self.blocked_cells:
                    step_cost += 4.0
                if action not in self.DIAGONAL_ACTIONS:
                    step_cost += 0.03
                npc_dist = self._nearest_npc_distance(nxt, npcs)
                if npc_dist < 5:
                    step_cost += (5.0 - npc_dist) * 1.5

                new_cost = cost + step_cost
                if new_cost + 1e-6 >= best_cost.get(nxt, float("inf")):
                    continue

                best_cost[nxt] = new_cost
                parent[nxt] = cur
                heuristic = min(self._chebyshev(nxt, tgt) for tgt in target_set)
                heapq.heappush(pq, (new_cost + heuristic, new_cost, nxt))

        return [], float("inf")

    def _score_actions(
        self,
        hero_pos: Position,
        safe_mask: np.ndarray,
        last_action: int,
        npcs: Sequence[Position],
        target_pos: Optional[Position],
        target_mode: str,
        path: Sequence[Position],
        should_charge: bool,
    ) -> np.ndarray:
        scores = np.full((8,), -1e9, dtype=np.float32)
        path_action = self._path_to_next_action(path) if path else None
        known_bbox = self._known_bbox()
        edge_mode = target_mode in ("edge_frontier", "find_charger_edge")
        current_npc_distance = self._nearest_npc_distance(hero_pos, npcs)

        for action in range(8):
            if safe_mask[action] <= 0.5:
                continue
            nxt = self._move(hero_pos, action)
            if not self._in_bounds(nxt):
                continue

            cell = int(self.global_map[nxt[1], nxt[0]])
            info_gain = self._count_unknown_near(nxt, 2)
            frontier_gain = self._count_unknown_near(nxt, 1)
            revisit_penalty = 0.18 * min(6, int(self.visit_count[nxt[1], nxt[0]]))
            npc_dist = self._nearest_npc_distance(nxt, npcs)
            npc_penalty = max(0.0, 6.0 - npc_dist) * (0.35 if target_mode in ("charge", "charge_approach") else 0.75)
            npc_clean_penalty = 0.9 * float(self.npc_cleaned[nxt[1], nxt[0]])
            trap_penalty = self._trap_penalty(nxt)
            blocked_penalty = 2.5 if nxt in self.blocked_cells else 0.0

            score = 0.0
            if action in self.DIAGONAL_ACTIONS:
                score += 0.15
            if action == last_action:
                score += 0.08
            if cell == self.DIRT:
                score += 1.25
            score += 0.07 * float(info_gain)
            score += 0.04 * float(frontier_gain)
            score -= revisit_penalty + npc_penalty + npc_clean_penalty + trap_penalty + blocked_penalty

            if path_action is not None and action == path_action:
                score += 2.6 if should_charge else 2.2
            if target_pos is not None and path_action is not None:
                progress = float(self._chebyshev(hero_pos, target_pos) - self._chebyshev(nxt, target_pos))
                score += (0.95 if should_charge else 0.45) * progress
            if edge_mode:
                score += 0.9 * self._edge_bonus(nxt, known_bbox)
            if target_mode in ("charge", "charge_approach") and self._is_on_charger(nxt):
                score += 4.0
            if target_mode in ("frontier", "dirt", "edge_frontier", "find_charger_edge") and self._is_frontier(nxt):
                score += 0.45
            if target_mode == "charge_approach":
                score += 0.20 * float(self._count_unknown_near(nxt, 1))
            if target_mode not in ("charge", "charge_approach") and npc_dist <= self.NPC_BLOCK_RADIUS and npc_dist < current_npc_distance:
                score -= 3.0
            scores[action] = score

        return scores

    def _escape_scores(self, hero_pos: Position, legal_mask: np.ndarray, npcs: Sequence[Position]) -> np.ndarray:
        scores = np.full((8,), -1e9, dtype=np.float32)
        for action in range(8):
            if legal_mask[action] <= 0.5:
                continue
            nxt = self._move(hero_pos, action)
            if not self._in_bounds(nxt):
                continue
            npc_dist = self._nearest_npc_distance(nxt, npcs)
            scores[action] = float(npc_dist) + (0.1 if action in self.DIAGONAL_ACTIONS else 0.0)
        return scores

    def _stall_or_escape_scores(
        self,
        hero_pos: Position,
        legal_mask: np.ndarray,
        map_grid: np.ndarray,
        npcs: Sequence[Position],
    ) -> np.ndarray:
        scores = self._escape_scores(hero_pos, legal_mask, npcs)
        center = self.VIEW_RADIUS
        for action, (dx, dz) in enumerate(self.ACTION_TO_DELTA):
            row = center + dz
            col = center + dx
            if row < 0 or row >= map_grid.shape[0] or col < 0 or col >= map_grid.shape[1]:
                continue
            blocked = int(map_grid[row, col]) == self.OBSTACLE
            if action in self.DIAGONAL_ACTIONS:
                side_h = int(map_grid[center, center + dx]) != self.OBSTACLE
                side_v = int(map_grid[center + dz, center]) != self.OBSTACLE
                blocked = blocked or not (side_h or side_v)
            if blocked:
                scores[action] = max(scores[action], 100.0 + self._nearest_npc_distance(hero_pos, npcs))
        return scores

    def _distance_map(self, start: Position) -> np.ndarray:
        dist = np.full((self.MAP_SIZE, self.MAP_SIZE), -1, dtype=np.int16)
        if not self._in_bounds(start):
            return dist

        queue: List[Position] = [start]
        dist[start[1], start[0]] = 0
        head = 0
        while head < len(queue):
            cur = queue[head]
            head += 1
            cur_dist = int(dist[cur[1], cur[0]])
            for action in self.PATH_ACTION_ORDER:
                nxt = self._move(cur, action)
                if not self._can_traverse(cur, nxt, allow_unknown=False):
                    continue
                if dist[nxt[1], nxt[0]] >= 0:
                    continue
                dist[nxt[1], nxt[0]] = cur_dist + 1
                queue.append(nxt)
        return dist

    def _update_global_memory(self, hero_pos: Position, map_grid: np.ndarray, cleaned_cells: Sequence[Position]) -> int:
        center = self.VIEW_RADIUS
        hero_cleaned_set = set(cleaned_cells)
        new_known = 0
        for row in range(map_grid.shape[0]):
            gz = hero_pos[1] + (row - center)
            if gz < 0 or gz >= self.MAP_SIZE:
                continue
            for col in range(map_grid.shape[1]):
                gx = hero_pos[0] + (col - center)
                if gx < 0 or gx >= self.MAP_SIZE:
                    continue
                prev = int(self.global_map[gz, gx])
                val = int(map_grid[row, col])
                if prev == self.UNKNOWN:
                    new_known += 1
                if prev == self.DIRT and val == self.CLEAN and (gx, gz) not in hero_cleaned_set:
                    self.npc_cleaned[gz, gx] = 1
                self.global_map[gz, gx] = val
        return new_known

    def _mark_cleaned_cells(self, cleaned_cells: Sequence[Position]) -> None:
        for pos in cleaned_cells:
            if self._in_bounds(pos):
                self.global_map[pos[1], pos[0]] = self.CLEAN
                self.hero_cleaned[pos[1], pos[0]] = 1
                self.npc_cleaned[pos[1], pos[0]] = 0

    def _mark_charger_cells(self) -> None:
        for region in self.charger_regions:
            for pos in region:
                if self._in_bounds(pos) and self.global_map[pos[1], pos[0]] == self.UNKNOWN:
                    self.global_map[pos[1], pos[0]] = self.CLEAN
    def _build_local_legal_mask(self, map_grid: np.ndarray) -> np.ndarray:
        mask = np.zeros((8,), dtype=np.float32)
        c = self.VIEW_RADIUS
        for action, (dx, dz) in enumerate(self.ACTION_TO_DELTA):
            row = c + dz
            col = c + dx
            if row < 0 or row >= map_grid.shape[0] or col < 0 or col >= map_grid.shape[1]:
                continue
            if int(map_grid[row, col]) == self.OBSTACLE:
                continue
            if action in self.DIAGONAL_ACTIONS:
                side_h = int(map_grid[c, c + dx]) != self.OBSTACLE
                side_v = int(map_grid[c + dz, c]) != self.OBSTACLE
                if not (side_h or side_v):
                    continue
            mask[action] = 1.0
        if mask.sum() <= 0:
            mask[:] = 1.0
        return mask

    def _tighten_corner_mask(self, map_grid: np.ndarray, safe_mask: np.ndarray) -> np.ndarray:
        tightened = safe_mask.copy()
        c = self.VIEW_RADIUS
        for action, (dx, dz) in enumerate(self.ACTION_TO_DELTA):
            if action not in self.DIAGONAL_ACTIONS or tightened[action] <= 0.5:
                continue
            side_h = int(map_grid[c, c + dx]) != self.OBSTACLE
            side_v = int(map_grid[c + dz, c]) != self.OBSTACLE
            if not (side_h and side_v):
                tightened[action] = 0.0
        if tightened.sum() <= 0:
            return safe_mask
        return tightened

    def _relax_charge_mask(
        self,
        hero_pos: Position,
        map_grid: np.ndarray,
        safe_mask: np.ndarray,
        legal_mask: np.ndarray,
        npcs: Sequence[Position],
    ) -> np.ndarray:
        relaxed = safe_mask.copy()
        current_charge = self._heuristic_to_closest_charger(hero_pos)
        for action in range(8):
            if relaxed[action] > 0.5 or legal_mask[action] <= 0.5:
                continue
            nxt = self._move(hero_pos, action)
            if self._hard_npc_zone(nxt, npcs):
                continue
            if action in self.DIAGONAL_ACTIONS and not self._is_strict_diagonal_open(map_grid, action):
                continue
            next_charge = self._heuristic_to_closest_charger(nxt)
            next_npc_distance = self._nearest_npc_distance(nxt, npcs)
            if self._is_on_charger(nxt) or (next_charge < current_charge and next_npc_distance > 1.0):
                relaxed[action] = 1.0
        return relaxed

    def _lock_force_charge_mask(
        self,
        hero_pos: Position,
        safe_mask: np.ndarray,
        legal_mask: np.ndarray,
        path: Sequence[Position],
        npcs: Sequence[Position],
    ) -> np.ndarray:
        locked = np.zeros_like(safe_mask)
        path_action = self._path_to_next_action(path) if path else None
        if path_action is not None and safe_mask[path_action] > 0.5:
            locked[path_action] = 1.0

            escape_scores = self._escape_scores(hero_pos, legal_mask, npcs)
            escape_scores[path_action] = -1e9
            escape_action = int(np.argmax(escape_scores))
            if escape_scores[escape_action] > -1e8 and safe_mask[escape_action] > 0.5:
                locked[escape_action] = 1.0

        if locked.sum() <= 0:
            return safe_mask
        return locked

    def _choose_action(
        self,
        hero_pos: Position,
        target_mode: str,
        path: Sequence[Position],
        safe_mask: np.ndarray,
        action_scores: np.ndarray,
    ) -> int:
        forced_action = self._forced_charge_action(hero_pos, target_mode, path, safe_mask)
        if forced_action is not None:
            finite_scores = action_scores[np.isfinite(action_scores)]
            baseline = float(np.max(finite_scores)) if finite_scores.size > 0 else 0.0
            action_scores[forced_action] = baseline + 8.0
            return forced_action
        return int(np.argmax(action_scores))

    def _forced_charge_action(
        self,
        hero_pos: Position,
        target_mode: str,
        path: Sequence[Position],
        safe_mask: np.ndarray,
    ) -> Optional[int]:
        if target_mode not in ("charge", "charge_approach"):
            return None
        path_action = self._path_to_next_action(path) if path else None
        if path_action is None or safe_mask[path_action] <= 0.5:
            return None
        nxt = self._move(hero_pos, path_action)
        if not self._in_bounds(nxt):
            return None
        return path_action

    def _apply_npc_safety_mask(self, hero_pos: Position, legal_mask: np.ndarray, npcs: Sequence[Position]) -> np.ndarray:
        safe_mask = legal_mask.copy()
        current_npc_distance = self._nearest_npc_distance(hero_pos, npcs)
        for action in range(8):
            if safe_mask[action] <= 0.5:
                continue
            nxt = self._move(hero_pos, action)
            next_npc_distance = self._nearest_npc_distance(nxt, npcs)
            if self._hard_npc_zone(nxt, npcs):
                safe_mask[action] = 0.0
            elif next_npc_distance <= self.NPC_BLOCK_RADIUS and next_npc_distance <= current_npc_distance:
                safe_mask[action] = 0.0
        return safe_mask

    def _hard_npc_zone(self, pos: Position, npcs: Sequence[Position]) -> bool:
        return any(self._chebyshev(pos, npc) <= self.HARD_NPC_RADIUS for npc in npcs)

    def _path_has_npc_risk(self, path: Sequence[Position], npcs: Sequence[Position]) -> bool:
        for pos in path:
            if any(self._chebyshev(pos, npc) <= self.PATH_RISK_RADIUS for npc in npcs):
                return True
        return False

    def _frontier_density(self, hero_pos: Position) -> float:
        total = 0
        frontier = 0
        for dz in range(-4, 5):
            gz = hero_pos[1] + dz
            if gz < 0 or gz >= self.MAP_SIZE:
                continue
            for dx in range(-4, 5):
                gx = hero_pos[0] + dx
                if gx < 0 or gx >= self.MAP_SIZE:
                    continue
                total += 1
                if self._frontier_map[gz, gx] > 0:
                    frontier += 1
        return float(frontier) / float(max(total, 1))

    def _is_frontier(self, pos: Position) -> bool:
        if not self._in_bounds(pos):
            return False
        return bool(self._frontier_map[pos[1], pos[0]])

    def _goal_is_still_useful(self, goal: Optional[Position], hero_pos: Position) -> bool:
        if goal is None or not self._in_bounds(goal) or goal == hero_pos:
            return False
        if self.current_mode in ("charge", "charge_approach", "edge_frontier", "find_charger_edge"):
            return True
        cell = int(self.global_map[goal[1], goal[0]])
        return cell == self.DIRT or self._is_frontier(goal)

    def _known_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self._known_bbox_cache

    def _edge_bonus(self, pos: Position, bbox: Optional[Tuple[int, int, int, int]]) -> float:
        if bbox is None:
            return 0.0
        min_x, max_x, min_z, max_z = bbox
        edge_dist = min(abs(pos[0] - min_x), abs(pos[0] - max_x), abs(pos[1] - min_z), abs(pos[1] - max_z))
        return max(0.0, 4.0 - float(edge_dist)) / 4.0

    def _map_edge_bonus(self, pos: Position) -> float:
        edge_dist = min(pos[0], pos[1], self.MAP_SIZE - 1 - pos[0], self.MAP_SIZE - 1 - pos[1])
        return max(0.0, 12.0 - float(edge_dist)) / 12.0

    def _free_neighbor_count(self, pos: Position, allow_unknown: bool = False) -> int:
        if not allow_unknown:
            cached = self._free_degree_cache.get(pos)
            if cached is not None:
                return cached
        count = 0
        for _, nxt in self._neighbors(pos):
            if self._can_traverse(pos, nxt, allow_unknown):
                count += 1
        if not allow_unknown:
            self._free_degree_cache[pos] = count
        return count

    def _trap_penalty(self, pos: Position) -> float:
        cached = self._trap_penalty_cache.get(pos)
        if cached is not None:
            return cached
        degree = self._free_neighbor_count(pos, allow_unknown=False)
        if degree <= self.DEAD_END_DEGREE:
            penalty = self.DEAD_END_PENALTY
        elif degree <= self.NARROW_DEGREE:
            penalty = self.NARROW_PENALTY
        else:
            penalty = 0.0
        self._trap_penalty_cache[pos] = penalty
        return penalty

    def _charge_retry_margin(self, path: Sequence[Position]) -> float:
        if len(path) < 2:
            return self.RETURN_RETRY_MARGIN + 4.0

        turns = 0
        narrow_steps = 0
        blocked_steps = 0
        prev_delta: Optional[Position] = None

        for idx in range(1, len(path)):
            cur = path[idx - 1]
            nxt = path[idx]
            delta = (
                int(np.clip(nxt[0] - cur[0], -1, 1)),
                int(np.clip(nxt[1] - cur[1], -1, 1)),
            )
            if prev_delta is not None and delta != prev_delta:
                turns += 1
            prev_delta = delta

            if self._trap_penalty(nxt) > 0.0:
                narrow_steps += 1
            if nxt in self.blocked_cells:
                blocked_steps += 1

        margin = 2.0 + 0.35 * float(turns) + 0.8 * float(narrow_steps) + 1.2 * float(blocked_steps)
        return min(self.BASE_RETURN_MARGIN, self.RETURN_RETRY_MARGIN + margin)

    def _extract_charger_regions(self, organs: Sequence[Any]) -> List[Set[Position]]:
        regions: List[Set[Position]] = []
        for organ in organs:
            sub_type = int(self._safe_float(self._get(organ, "sub_type", 0), 0.0))
            if sub_type != 1:
                continue
            pos = self._extract_pos(self._get(organ, "pos", {}))
            w = max(1, int(self._safe_float(self._get(organ, "w", 3), 3.0)))
            h = max(1, int(self._safe_float(self._get(organ, "h", 3), 3.0)))
            region: Set[Position] = set()
            for dx in range(w):
                for dz in range(h):
                    region.add((pos[0] + dx, pos[1] + dz))
            half_w = w // 2
            half_h = h // 2
            for dx in range(-half_w, half_w + 1):
                for dz in range(-half_h, half_h + 1):
                    region.add((pos[0] + dx, pos[1] + dz))
            region = {cell for cell in region if self._in_bounds(cell)}
            if region:
                regions.append(region)
        return regions

    def _charger_target_cells(self) -> List[Position]:
        cells: List[Position] = []
        for region in self.charger_regions:
            cells.extend(region)
        return cells

    def _is_on_charger(self, pos: Position) -> bool:
        return any(pos in region for region in self.charger_regions)

    def _heuristic_to_closest_charger(self, pos: Position) -> float:
        best = float("inf")
        for region in self.charger_regions:
            for cell in region:
                best = min(best, float(self._chebyshev(pos, cell)))
        return best if np.isfinite(best) else 0.0

    def _is_strict_diagonal_open(self, map_grid: np.ndarray, action: int) -> bool:
        if action not in self.DIAGONAL_ACTIONS:
            return True
        c = self.VIEW_RADIUS
        dx, dz = self.ACTION_TO_DELTA[action]
        side_h = int(map_grid[c, c + dx]) != self.OBSTACLE
        side_v = int(map_grid[c + dz, c]) != self.OBSTACLE
        return side_h and side_v

    def _count_unknown_near(self, pos: Position, radius: int) -> int:
        return self._rect_sum(self._unknown_prefix, pos, radius)

    def _count_dirt_near(self, pos: Position, radius: int) -> int:
        return self._rect_sum(self._dirt_prefix, pos, radius)

    def _path_to_next_action(self, path: Sequence[Position]) -> Optional[int]:
        if len(path) < 2:
            return None
        cur, nxt = path[0], path[1]
        dx = int(np.clip(nxt[0] - cur[0], -1, 1))
        dz = int(np.clip(nxt[1] - cur[1], -1, 1))
        for action, delta in enumerate(self.ACTION_TO_DELTA):
            if delta == (dx, dz):
                return action
        return None

    def _use_cached_path(self, hero_pos: Position, npcs: Sequence[Position]) -> Tuple[List[Position], float]:
        if not self.current_path:
            return [], float("inf")
        if hero_pos in self.current_path:
            idx = self.current_path.index(hero_pos)
            path = self.current_path[idx:]
        else:
            path = self.current_path
        if not path or path[0] != hero_pos:
            return [], float("inf")
        if len(path) >= 2:
            nxt = path[1]
            if self._hard_npc_zone(nxt, npcs) or nxt in self.blocked_cells or not self._can_traverse(hero_pos, nxt, allow_unknown=False):
                return [], float("inf")
        return list(path), float(max(0, len(path) - 1))

    def _decay_blocked_cells(self) -> None:
        if not self.blocked_cells:
            return
        next_map: Dict[Position, int] = {}
        for pos, ttl in self.blocked_cells.items():
            if ttl > 1:
                next_map[pos] = ttl - 1
        self.blocked_cells = next_map

    def _update_stuck_memory(self, hero_pos: Position, last_action: int) -> None:
        if self.prev_hero_pos is None or last_action < 0:
            return
        if hero_pos != self.prev_hero_pos:
            return
        attempted = self._move(hero_pos, last_action)
        if self._in_bounds(attempted):
            self.blocked_cells[attempted] = self.BLOCKED_CELL_TTL
        self.current_goal = None
        self.current_path = []
    def _can_traverse(self, cur: Position, nxt: Position, allow_unknown: bool) -> bool:
        if not self._in_bounds(cur) or not self._in_bounds(nxt):
            return False
        cell = int(self.global_map[nxt[1], nxt[0]])
        if cell == self.OBSTACLE:
            return False
        if cell == self.UNKNOWN and not allow_unknown:
            return False
        dx = int(np.clip(nxt[0] - cur[0], -1, 1))
        dz = int(np.clip(nxt[1] - cur[1], -1, 1))
        if dx != 0 and dz != 0:
            side_h = (cur[0] + dx, cur[1])
            side_v = (cur[0], cur[1] + dz)
            if not self._side_is_open(side_h, allow_unknown) and not self._side_is_open(side_v, allow_unknown):
                return False
        return True

    def _side_is_open(self, pos: Position, allow_unknown: bool) -> bool:
        if not self._in_bounds(pos):
            return False
        cell = int(self.global_map[pos[1], pos[0]])
        if cell == self.OBSTACLE:
            return False
        if cell == self.UNKNOWN and not allow_unknown:
            return False
        return True

    def _reconstruct_path(self, parent: Dict[Position, Position], start: Position, goal: Position) -> List[Position]:
        path = [goal]
        cur = goal
        while cur != start:
            cur = parent[cur]
            path.append(cur)
        path.reverse()
        return path

    def _neighbors(self, pos: Position) -> Iterable[Tuple[int, Position]]:
        for action in range(8):
            yield action, self._move(pos, action)

    def _move(self, pos: Position, action: int) -> Position:
        dx, dz = self.ACTION_TO_DELTA[action]
        return pos[0] + dx, pos[1] + dz

    def _nearest_npc_distance(self, pos: Position, npcs: Sequence[Position]) -> float:
        if not npcs:
            return 99.0
        cached = self._npc_distance_cache.get(pos)
        if cached is not None:
            return cached
        dist = min(float(self._chebyshev(pos, npc)) for npc in npcs)
        self._npc_distance_cache[pos] = dist
        return dist

    @staticmethod
    def _chebyshev(a: Position, b: Position) -> int:
        return max(abs(a[0] - b[0]), abs(a[1] - b[1]))

    def _extract_observation(self, env_obs: Any) -> Any:
        if isinstance(env_obs, dict) and "observation" in env_obs:
            return env_obs.get("observation")
        return env_obs

    def _extract_hero(self, frame_state: Any) -> Any:
        heroes = self._get(frame_state, "heroes", {})
        if isinstance(heroes, (list, tuple)):
            return heroes[0] if heroes else {}
        return heroes

    def _extract_npcs(self, frame_state: Any) -> List[Position]:
        npcs = self._get(frame_state, "npcs", [])
        if not isinstance(npcs, (list, tuple)):
            return []
        return [self._extract_pos(self._get(npc, "pos", {})) for npc in npcs if self._get(npc, "pos", None) is not None]

    def _extract_organs(self, frame_state: Any) -> List[Any]:
        organs = self._get(frame_state, "organs", [])
        return list(organs) if isinstance(organs, (list, tuple)) else []

    def _extract_map_grid(self, obs: Any) -> np.ndarray:
        map_info = self._get(obs, "map_info", None)
        if map_info is None:
            return np.ones((21, 21), dtype=np.int8)
        arr = np.asarray(map_info, dtype=np.int8)
        if arr.ndim != 2:
            return np.ones((21, 21), dtype=np.int8)
        return arr

    def _extract_pos(self, obj: Any) -> Position:
        x = int(self._safe_float(self._get(obj, "x", 0), 0.0))
        z = int(self._safe_float(self._get(obj, "z", 0), 0.0))
        return int(np.clip(x, 0, self.MAP_SIZE - 1)), int(np.clip(z, 0, self.MAP_SIZE - 1))

    def _extract_positions(self, items: Any) -> List[Position]:
        if not isinstance(items, (list, tuple)):
            return []
        return [self._extract_pos(item) for item in items]

    def _in_bounds(self, pos: Position) -> bool:
        return 0 <= pos[0] < self.MAP_SIZE and 0 <= pos[1] < self.MAP_SIZE

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        if obj is None:
            return default
        return getattr(obj, key, default)

    @staticmethod
    def _safe_float(v: Any, default: float) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

    def _rebuild_step_caches(self) -> None:
        unknown = (self.global_map == self.UNKNOWN).astype(np.int32)
        dirt = (self.global_map == self.DIRT).astype(np.int32)

        self._unknown_prefix.fill(0)
        self._dirt_prefix.fill(0)
        self._unknown_prefix[1:, 1:] = np.cumsum(np.cumsum(unknown, axis=0), axis=1)
        self._dirt_prefix[1:, 1:] = np.cumsum(np.cumsum(dirt, axis=0), axis=1)

        traversable = (self.global_map == self.CLEAN) | (self.global_map == self.DIRT)
        padded_unknown = np.pad(unknown, 1, mode="constant", constant_values=0)
        neighbor_unknown = np.zeros_like(unknown, dtype=bool)
        for dz in range(3):
            for dx in range(3):
                if dx == 1 and dz == 1:
                    continue
                neighbor_unknown |= padded_unknown[dz : dz + self.MAP_SIZE, dx : dx + self.MAP_SIZE] > 0
        self._frontier_map = (traversable & neighbor_unknown).astype(np.uint8)

        known = np.argwhere(self.global_map != self.UNKNOWN)
        if known.size == 0:
            self._known_bbox_cache = None
        else:
            self._known_bbox_cache = (
                int(np.min(known[:, 1])),
                int(np.max(known[:, 1])),
                int(np.min(known[:, 0])),
                int(np.max(known[:, 0])),
            )

        self._trap_penalty_cache.clear()
        self._free_degree_cache.clear()
        self._npc_distance_cache.clear()

    def _rect_sum(self, prefix: np.ndarray, pos: Position, radius: int) -> int:
        x0 = max(0, pos[0] - radius)
        x1 = min(self.MAP_SIZE - 1, pos[0] + radius)
        z0 = max(0, pos[1] - radius)
        z1 = min(self.MAP_SIZE - 1, pos[1] + radius)
        return int(
            prefix[z1 + 1, x1 + 1]
            - prefix[z0, x1 + 1]
            - prefix[z1 + 1, x0]
            + prefix[z0, x0]
        )






