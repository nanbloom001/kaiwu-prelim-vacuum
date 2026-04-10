#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Lightweight world-model preprocessor and rule-based planner for DIY mode.
"""

from collections import deque
import heapq

import numpy as np

from agent_diy.conf.conf import Config


def _norm(v, v_max, v_min=0.0):
    v = float(np.clip(v, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


def _signed_norm(v, max_abs):
    max_abs = float(max(max_abs, 1.0))
    return float(np.clip(v, -max_abs, max_abs) / max_abs)


class Preprocessor:
    UNKNOWN = -1
    OBSTACLE = 0
    CLEAN = 1
    DIRTY = 2

    VIEW_HALF = Config.VIEW_SIZE // 2
    LOCAL_HALF = Config.LOCAL_VIEW_SIZE // 2
    ACTION_DELTAS = [
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 2000
        self.battery = 200
        self.last_battery = 200
        self.battery_max = 200
        self.charge_count = 0

        self.cur_pos = (0, 0)
        self.last_pos = (0, 0)
        self.last_action = -1

        self.score = 0
        self.last_score = 0
        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        self.map_state = np.full((Config.GRID_SIZE, Config.GRID_SIZE), self.UNKNOWN, dtype=np.int8)
        self.visit_count = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=np.int16)
        self.clean_pass_count = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=np.int16)
        self.last_visit_step = np.full((Config.GRID_SIZE, Config.GRID_SIZE), -10000, dtype=np.int32)

        self._view_map = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.int8)
        self._env_legal_action = [1] * Config.ACTION_DIM
        self._legal_action = [1] * Config.ACTION_DIM
        self._current_step_cleaned = set()

        self.chargers = []
        self.npc_positions = {}
        self.npc_prev_positions = {}
        self.npc_centers = {}
        self.stuck_chain = 0
        self._charger_dist_map = None

        self.charge_mode = False
        self.on_charger = False
        self.last_on_charger = False
        self.goal = None
        self.goal_kind = None
        self.path = []
        self.last_plan_step = -999

        self.pending_action = 0
        self.pending_prob = np.full(Config.ACTION_DIM, 1.0 / Config.ACTION_DIM, dtype=np.float32)

    def feature_process(self, env_obs, last_action):
        self._parse_env_obs(env_obs, last_action)
        self._legal_action = self._build_legal_action()
        self.pending_action = self._select_action(self._legal_action)
        feature = self._build_feature()
        reward = self.reward_process()
        return feature, list(self._legal_action), reward

    def get_pending_action(self):
        return int(self.pending_action)

    def get_pending_prob(self):
        return self.pending_prob.copy()

    def should_force_teacher(self):
        charger_dist = self._current_charger_distance()
        charge_margin = self.battery - charger_dist
        risk = self._npc_zone_penalty(self.cur_pos)
        if self.stuck_chain >= Config.FORCE_TEACHER_STUCK:
            return True
        if risk >= Config.FORCE_TEACHER_RISK:
            return True
        if self.charge_mode and charge_margin <= Config.CRITICAL_CHARGE_MARGIN:
            return True
        return False

    def get_teacher_mix_bias(self):
        bias = 0.0
        risk = self._npc_zone_penalty(self.cur_pos)
        if risk >= 12.0:
            bias += Config.RISK_TEACHER_BIAS
        if self.charge_mode:
            bias += Config.CHARGE_TEACHER_BIAS
        if self.stuck_chain > 0:
            bias += Config.STUCK_TEACHER_BIAS * min(self.stuck_chain, 3)
        return float(np.clip(bias, 0.0, 0.35))

    def _parse_env_obs(self, env_obs, last_action):
        observation = self._normalize_observation(env_obs)
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})

        hero = frame_state.get("heroes", {})
        if isinstance(hero, list):
            hero = hero[0] if hero else {}

        self.step_no = int(observation.get("step_no", env_info.get("step_no", 0)))
        self.max_step = int(env_info.get("max_step", self.max_step))

        self.last_action = int(last_action)
        self.last_pos = self.cur_pos
        self.last_battery = self.battery
        self.last_score = self.score
        self.last_dirt_cleaned = self.dirt_cleaned
        self.last_on_charger = self.on_charger

        pos = hero.get("pos") or env_info.get("pos") or {}
        self.cur_pos = (
            int(pos.get("x", self.cur_pos[0])),
            int(pos.get("z", self.cur_pos[1])),
        )
        self.battery = int(hero.get("battery", env_info.get("remaining_charge", self.battery)))
        self.battery_max = max(int(hero.get("battery_max", env_info.get("battery_max", self.battery_max))), 1)
        self.score = int(hero.get("score", env_info.get("total_score", self.score)))
        self.dirt_cleaned = int(hero.get("dirt_cleaned", self.dirt_cleaned))
        self.total_dirt = max(int(env_info.get("total_dirt", self.total_dirt)), 1)
        self.charge_count = int(env_info.get("charge_count", self.charge_count))

        raw_legal_action = observation.get("legal_action")
        if raw_legal_action is None:
            raw_legal_action = observation.get("legal_act")
        self._env_legal_action = [int(x) for x in (raw_legal_action or [1] * Config.ACTION_DIM)]

        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.int8)
            self._integrate_local_view()
            self._update_stuck_state()

        self._current_step_cleaned = set()
        cleaned_cells = env_info.get("step_cleaned_cells") or []
        for cell in cleaned_cells:
            x = int(cell.get("x", -1))
            z = int(cell.get("z", -1))
            if self._in_bounds(x, z):
                self.map_state[x, z] = self.CLEAN
                self._current_step_cleaned.add((x, z))

        self._update_chargers(frame_state.get("organs") or [])
        self._update_npcs(frame_state.get("npcs") or [])

        self.on_charger = self._is_on_charger(self.cur_pos)
        cx, cz = self.cur_pos
        if self._in_bounds(cx, cz):
            self.visit_count[cx, cz] = min(32767, self.visit_count[cx, cz] + 1)
            self.last_visit_step[cx, cz] = self.step_no
            if self.map_state[cx, cz] == self.CLEAN and (cx, cz) not in self._current_step_cleaned:
                self.clean_pass_count[cx, cz] = min(32767, self.clean_pass_count[cx, cz] + 1)

    def _update_stuck_state(self):
        if self.cur_pos != self.last_pos or self.last_action < 0:
            self.stuck_chain = 0
            return

        self.stuck_chain = min(self.stuck_chain + 1, 8)
        self.path = []
        self.goal = None
        self.goal_kind = None

        dx, dz = self.ACTION_DELTAS[int(self.last_action)]
        next_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
        local_target = self._get_local_cell(dx, dz)
        if self._in_bounds(*next_pos) and local_target == self.OBSTACLE:
            self.map_state[next_pos[0], next_pos[1]] = self.OBSTACLE

        if dx != 0 and dz != 0:
            side_a = (self.cur_pos[0] + dx, self.cur_pos[1])
            side_b = (self.cur_pos[0], self.cur_pos[1] + dz)
            if self._in_bounds(*side_a) and self._get_local_cell(dx, 0) == self.OBSTACLE:
                self.map_state[side_a[0], side_a[1]] = self.OBSTACLE
            if self._in_bounds(*side_b) and self._get_local_cell(0, dz) == self.OBSTACLE:
                self.map_state[side_b[0], side_b[1]] = self.OBSTACLE

    def _normalize_observation(self, env_obs):
        if isinstance(env_obs, dict) and "observation" in env_obs:
            return env_obs["observation"]

        if isinstance(env_obs, dict):
            observation = dict(env_obs)
            if "env_info" not in observation and isinstance(observation.get("state"), dict):
                observation["env_info"] = observation["state"]
            return observation

        return {}

    def _integrate_local_view(self):
        hx, hz = self.cur_pos
        for row in range(Config.VIEW_SIZE):
            for col in range(Config.VIEW_SIZE):
                gx = hx - self.VIEW_HALF + col
                gz = hz - self.VIEW_HALF + row
                if not self._in_bounds(gx, gz):
                    continue
                self.map_state[gx, gz] = int(self._view_map[row, col])

    def _update_chargers(self, organs):
        self.chargers = []
        for organ in organs:
            if int(organ.get("sub_type", 0)) != 1:
                continue

            x = int(organ.get("pos", {}).get("x", 0))
            z = int(organ.get("pos", {}).get("z", 0))
            w = max(int(organ.get("w", 1)), 1)
            h = max(int(organ.get("h", 1)), 1)
            self.chargers.append({"x": x, "z": z, "w": w, "h": h})

            for gx in range(x, min(x + w, Config.GRID_SIZE)):
                for gz in range(z, min(z + h, Config.GRID_SIZE)):
                    if self.map_state[gx, gz] == self.UNKNOWN:
                        self.map_state[gx, gz] = self.CLEAN

    def _update_npcs(self, npcs):
        new_positions = {}
        for npc in npcs:
            idx = int(npc.get("idx", len(new_positions) + 1))
            pos = npc.get("pos", {})
            npc_pos = (int(pos.get("x", 0)), int(pos.get("z", 0)))
            new_positions[idx] = npc_pos
            self.npc_centers.setdefault(idx, npc_pos)

        self.npc_prev_positions = dict(self.npc_positions)
        self.npc_positions = new_positions

    def _build_legal_action(self):
        env_mask = list(self._env_legal_action)
        obstacle_mask = list(env_mask)
        safe_mask = list(env_mask)

        for act, (dx, dz) in enumerate(self.ACTION_DELTAS):
            if env_mask[act] == 0:
                obstacle_mask[act] = 0
                safe_mask[act] = 0
                continue

            if not self._is_local_move_passable(dx, dz):
                obstacle_mask[act] = 0
                safe_mask[act] = 0
                continue

            next_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
            if not self._is_global_cell_plannable(*next_pos, allow_unknown=False):
                obstacle_mask[act] = 0
                safe_mask[act] = 0
                continue
            if self._would_hit_npc(next_pos):
                safe_mask[act] = 0

        if sum(safe_mask) > 0:
            return safe_mask
        if sum(obstacle_mask) > 0:
            return obstacle_mask
        recovery_mask = self._build_recovery_mask(env_mask)
        if sum(recovery_mask) > 0:
            return recovery_mask
        return self._build_best_effort_mask(env_mask)

    def _build_recovery_mask(self, env_mask):
        mask = [0] * Config.ACTION_DIM
        for act, (dx, dz) in enumerate(self.ACTION_DELTAS):
            if env_mask[act] == 0:
                continue
            next_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
            if self._would_hit_npc(next_pos):
                continue
            if self._is_local_move_passable(dx, dz) and self._is_global_cell_plannable(*next_pos, allow_unknown=True):
                mask[act] = 1
        return mask

    def _build_best_effort_mask(self, env_mask):
        mask = [0] * Config.ACTION_DIM
        for act, ok in enumerate(env_mask):
            if not ok:
                continue
            dx, dz = self.ACTION_DELTAS[act]
            next_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
            if self._is_local_move_passable(dx, dz) and self._is_global_cell_plannable(*next_pos, allow_unknown=True):
                mask[act] = 1
        if sum(mask) > 0:
            return mask
        if sum(env_mask) > 0:
            return env_mask
        return [1] * Config.ACTION_DIM

    def _is_local_move_passable(self, dx, dz):
        if not self._is_local_passable(dx, dz):
            return False
        if dx != 0 and dz != 0:
            if not (self._is_local_passable(dx, 0) and self._is_local_passable(0, dz)):
                return False
        return True

    def _get_local_cell(self, dx, dz):
        row = self.VIEW_HALF + dz
        col = self.VIEW_HALF + dx
        if not (0 <= row < Config.VIEW_SIZE and 0 <= col < Config.VIEW_SIZE):
            return self.OBSTACLE
        return int(self._view_map[row, col])

    def _is_local_passable(self, dx, dz):
        return self._get_local_cell(dx, dz) != self.OBSTACLE

    def _would_hit_npc(self, pos):
        for idx, npc_pos in self.npc_positions.items():
            if self._chebyshev(pos, npc_pos) <= Config.NPC_COLLISION_RADIUS:
                return True

            prev_pos = self.npc_prev_positions.get(idx, npc_pos)
            pred = (
                npc_pos[0] + int(np.clip(npc_pos[0] - prev_pos[0], -1, 1)),
                npc_pos[1] + int(np.clip(npc_pos[1] - prev_pos[1], -1, 1)),
            )
            if self._chebyshev(pos, pred) <= Config.NPC_PREDICT_RADIUS:
                return True
        return False

    def _select_action(self, legal_action):
        self._trim_path_head()
        if self._need_charge():
            self.charge_mode = True
        elif self.on_charger and self.battery >= int(self.battery_max * 0.98):
            self.charge_mode = False

        if self._need_replan():
            self._replan()

        action_scores = self._score_actions(legal_action)
        if self.path:
            next_pos = self.path[0]
            act = self._pos_to_action(next_pos)
            if act is not None and legal_action[act]:
                action_scores[act] += Config.PATH_FOLLOW_BONUS

        self.pending_prob = self._scores_to_probs(action_scores, legal_action)
        if float(np.sum(self.pending_prob)) > 0.0:
            return int(np.argmax(self.pending_prob))

        for act, ok in enumerate(legal_action):
            if ok:
                return act
        return 0

    def _trim_path_head(self):
        while self.path and self.path[0] == self.cur_pos:
            self.path.pop(0)

    def _need_charge(self):
        if not self.chargers:
            return False
        charger_dist = self._current_charger_distance()
        if self.battery <= charger_dist + Config.RETURN_CHARGE_BUFFER:
            return True
        return self.battery / self.battery_max <= Config.LOW_BATTERY_RATIO

    def _need_replan(self):
        if not self.path:
            return True
        if self.step_no - self.last_plan_step >= Config.REPLAN_INTERVAL:
            return True
        if self.stuck_chain > 0:
            return True
        if self.goal is None:
            return True
        if self.goal_kind == "dirty":
            gx, gz = self.goal
            if not self._in_bounds(gx, gz) or self.map_state[gx, gz] != self.DIRTY:
                return True
        if self.goal_kind == "charger" and self.on_charger:
            return True
        return False

    def _replan(self):
        dist, prev_x, prev_z = self._build_bfs_tree(allow_unknown=True)
        self._charger_dist_map = self._build_multi_source_dist(self._charger_cells(), allow_unknown=True)

        goal_kind = None
        goal = None
        if self.charge_mode:
            goal_kind, goal = self._pick_best_charger_goal(dist)
        else:
            goal_kind, goal = self._pick_best_dirty_goal(dist, self._charger_dist_map)
            if goal is None:
                goal_kind, goal = self._pick_best_frontier_goal(dist, self._charger_dist_map)
            if goal is None and self.battery <= int(self.battery_max * 0.55):
                goal_kind, goal = self._pick_best_charger_goal(dist)

        self.goal_kind = goal_kind
        self.goal = goal
        self.last_plan_step = self.step_no

        if goal is None:
            self.path = []
            return

        path = self._plan_weighted_path(goal, allow_unknown=True)
        if not path:
            path = self._reconstruct_path(goal, prev_x, prev_z)
        self.path = path[: Config.MAX_TRACKED_PATH]

    def _build_bfs_tree(self, allow_unknown):
        dist = np.full((Config.GRID_SIZE, Config.GRID_SIZE), -1, dtype=np.int16)
        prev_x = np.full((Config.GRID_SIZE, Config.GRID_SIZE), -1, dtype=np.int16)
        prev_z = np.full((Config.GRID_SIZE, Config.GRID_SIZE), -1, dtype=np.int16)

        start_x, start_z = self.cur_pos
        queue = deque([(start_x, start_z)])
        dist[start_x, start_z] = 0

        while queue:
            x, z = queue.popleft()
            for dx, dz in self.ACTION_DELTAS:
                nx = x + dx
                nz = z + dz
                if not self._in_bounds(nx, nz):
                    continue
                if dist[nx, nz] != -1:
                    continue
                if not self._is_global_move_passable((x, z), (nx, nz), allow_unknown=allow_unknown):
                    continue
                if self._npc_zone_hard_block((nx, nz)):
                    continue

                dist[nx, nz] = dist[x, z] + 1
                prev_x[nx, nz] = x
                prev_z[nx, nz] = z
                queue.append((nx, nz))

        return dist, prev_x, prev_z

    def _build_multi_source_dist(self, starts, allow_unknown):
        dist = np.full((Config.GRID_SIZE, Config.GRID_SIZE), -1, dtype=np.int16)
        queue = deque()
        for sx, sz in starts:
            if not self._in_bounds(sx, sz):
                continue
            if dist[sx, sz] != -1:
                continue
            dist[sx, sz] = 0
            queue.append((sx, sz))

        while queue:
            x, z = queue.popleft()
            for dx, dz in self.ACTION_DELTAS:
                nx = x + dx
                nz = z + dz
                if not self._in_bounds(nx, nz):
                    continue
                if dist[nx, nz] != -1:
                    continue
                if not self._is_global_move_passable((x, z), (nx, nz), allow_unknown=allow_unknown):
                    continue
                if self._npc_zone_hard_block((nx, nz)):
                    continue
                dist[nx, nz] = dist[x, z] + 1
                queue.append((nx, nz))

        return dist

    def _is_global_move_passable(self, cur_pos, next_pos, allow_unknown):
        nx, nz = next_pos
        if not self._is_global_cell_plannable(nx, nz, allow_unknown=allow_unknown):
            return False

        dx = nx - cur_pos[0]
        dz = nz - cur_pos[1]
        if dx != 0 and dz != 0:
            side_a = (cur_pos[0] + dx, cur_pos[1])
            side_b = (cur_pos[0], cur_pos[1] + dz)
            if not (
                self._is_global_cell_plannable(*side_a, allow_unknown=allow_unknown)
                and self._is_global_cell_plannable(*side_b, allow_unknown=allow_unknown)
            ):
                return False
        return True

    def _is_global_cell_plannable(self, x, z, allow_unknown):
        if not self._in_bounds(x, z):
            return False
        cell = int(self.map_state[x, z])
        if cell == self.OBSTACLE:
            return False
        if cell == self.UNKNOWN:
            return bool(allow_unknown)
        return True

    def _plan_weighted_path(self, goal, allow_unknown):
        if goal == self.cur_pos:
            return []

        best_cost = np.full((Config.GRID_SIZE, Config.GRID_SIZE), np.inf, dtype=np.float32)
        prev_x = np.full((Config.GRID_SIZE, Config.GRID_SIZE), -1, dtype=np.int16)
        prev_z = np.full((Config.GRID_SIZE, Config.GRID_SIZE), -1, dtype=np.int16)

        sx, sz = self.cur_pos
        best_cost[sx, sz] = 0.0
        heap = [(float(self._chebyshev(self.cur_pos, goal)), 0.0, sx, sz)]

        while heap:
            _, cur_cost, x, z = heapq.heappop(heap)
            if cur_cost > float(best_cost[x, z]) + 1e-6:
                continue
            if (x, z) == goal:
                break

            for dx, dz in self.ACTION_DELTAS:
                nx = x + dx
                nz = z + dz
                if not self._in_bounds(nx, nz):
                    continue
                if not self._is_global_move_passable((x, z), (nx, nz), allow_unknown=allow_unknown):
                    continue
                if self._npc_zone_hard_block((nx, nz)):
                    continue

                move_cost = self._transition_cost((x, z), (nx, nz))
                new_cost = cur_cost + move_cost
                if new_cost + 1e-6 >= float(best_cost[nx, nz]):
                    continue

                best_cost[nx, nz] = new_cost
                prev_x[nx, nz] = x
                prev_z[nx, nz] = z
                heuristic = float(self._chebyshev((nx, nz), goal))
                heapq.heappush(heap, (new_cost + heuristic, new_cost, nx, nz))

        return self._reconstruct_path(goal, prev_x, prev_z)

    def _transition_cost(self, cur_pos, next_pos):
        nx, nz = next_pos
        cell = int(self.map_state[nx, nz]) if self._in_bounds(nx, nz) else self.OBSTACLE
        visit = float(self.visit_count[nx, nz]) if self._in_bounds(nx, nz) else float(Config.MAX_VISIT_CLIP)
        transit = (
            float(self.clean_pass_count[nx, nz]) if self._in_bounds(nx, nz) else float(Config.MAX_TRANSIT_CLIP)
        )

        base = 1.0 + (0.05 if next_pos[0] != cur_pos[0] and next_pos[1] != cur_pos[1] else 0.0)
        recent_gap = self.step_no - int(self.last_visit_step[nx, nz]) if self._in_bounds(nx, nz) else 0
        recent_penalty = 0.0 if recent_gap > 20 else 0.05 * max(0, 20 - recent_gap)
        repeat_penalty = 0.12 * min(visit, float(Config.MAX_VISIT_CLIP))
        transit_penalty = 0.22 * min(transit, float(Config.MAX_TRANSIT_CLIP))
        risk_penalty = 0.025 * min(self._npc_zone_penalty(next_pos), 40.0)
        interior_penalty = self._interior_clean_penalty(next_pos)

        dirty_bonus = 0.95 if cell == self.DIRTY else 0.0
        frontier_bonus = 0.0
        if cell == self.UNKNOWN:
            frontier_bonus += 0.20
        if self._has_unknown_neighbor(next_pos):
            frontier_bonus += 0.30
        frontier_bonus += 0.05 * min(self._unknown_density(next_pos, radius=2), 8)

        if next_pos == self.last_pos:
            recent_penalty += 0.8

        slack = self._charge_slack()
        if self.goal_kind == "charger" or self.charge_mode:
            dirty_bonus *= slack
            frontier_bonus *= 0.65 * slack

        cost = base + recent_penalty + repeat_penalty + transit_penalty + risk_penalty + interior_penalty
        cost -= dirty_bonus
        cost -= frontier_bonus
        return max(0.12, float(cost))

    def _charge_slack(self):
        charger_dist = max(self._current_charger_distance(), 1)
        margin = self.battery - charger_dist - Config.RETURN_CHARGE_BUFFER
        return float(np.clip(margin / max(self.battery_max, 1), 0.0, 1.0))

    def _interior_clean_penalty(self, pos):
        x, z = pos
        if not self._in_bounds(x, z):
            return 1.0
        if int(self.map_state[x, z]) != self.CLEAN:
            return 0.0

        nearby_unknown = self._unknown_density(pos, radius=2)
        nearby_dirty = self._dirty_density(x, z)
        if nearby_unknown > 0 or nearby_dirty > 1:
            return 0.0
        return 0.06 * max(0, self._local_clean_density(pos, radius=2) - 6)

    def _pick_best_dirty_goal(self, dist, charger_dist):
        dirty_cells = np.argwhere(self.map_state == self.DIRTY)
        if len(dirty_cells) == 0:
            return None, None

        candidates = self._cluster_cells(dirty_cells, Config.DIRTY_CLUSTER_RADIUS, self._dirty_density)
        if not candidates:
            return None, None

        best_score = -1e9
        best_goal = None
        for gx, gz in candidates:
            d = int(dist[gx, gz])
            if d <= 0:
                continue

            return_budget = d + self._charger_dist_at((gx, gz), charger_dist) + Config.TARGET_CHARGE_BUFFER
            if return_budget >= self.battery:
                continue

            score = (
                18.0
                + 4.2 * self._dirty_density(gx, gz)
                + 0.8 * self._frontier_gain(gx, gz)
                - 0.8 * d
                - 0.55 * self.visit_count[gx, gz]
                - 0.75 * self.clean_pass_count[gx, gz]
                - 0.25 * self._npc_zone_penalty((gx, gz))
            )

            if self.goal_kind == "dirty" and self.goal == (gx, gz):
                score += 1.5

            if score > best_score:
                best_score = score
                best_goal = (int(gx), int(gz))

        if best_goal is None:
            return None, None
        return "dirty", best_goal

    def _pick_best_frontier_goal(self, dist, charger_dist):
        frontier_cells = self._collect_frontiers()
        if not frontier_cells:
            return None, None

        candidates = self._cluster_cells(frontier_cells, Config.FRONTIER_CLUSTER_RADIUS, self._frontier_gain)
        if not candidates:
            return None, None

        best_score = -1e9
        best_goal = None
        for gx, gz in candidates:
            d = int(dist[gx, gz])
            if d <= 0:
                continue

            return_budget = d + self._charger_dist_at((gx, gz), charger_dist) + Config.TARGET_CHARGE_BUFFER
            if return_budget >= self.battery:
                continue

            unknown_gain = self._frontier_gain(gx, gz)
            score = (
                4.8 * unknown_gain
                - 0.9 * d
                - 0.55 * self.visit_count[gx, gz]
                - 0.85 * self.clean_pass_count[gx, gz]
                - 0.35 * self._npc_zone_penalty((gx, gz))
            )
            if self.goal_kind == "frontier" and self.goal == (gx, gz):
                score += 1.0

            if score > best_score:
                best_score = score
                best_goal = (int(gx), int(gz))

        if best_goal is None:
            return None, None
        return "frontier", best_goal

    def _pick_best_charger_goal(self, dist):
        if not self.chargers:
            return None, None

        best_dist = 1e9
        best_goal = None
        for charger in self.chargers:
            for gx in range(charger["x"], charger["x"] + charger["w"]):
                for gz in range(charger["z"], charger["z"] + charger["h"]):
                    if not self._in_bounds(gx, gz):
                        continue
                    d = int(dist[gx, gz])
                    if d == -1:
                        continue
                    cost = d + 0.30 * float(self.visit_count[gx, gz]) + 0.55 * float(self.clean_pass_count[gx, gz])
                    if cost < best_dist:
                        best_dist = cost
                        best_goal = (gx, gz)

        if best_goal is None:
            return "charger", None
        return "charger", best_goal

    def _charger_cells(self):
        cells = []
        for charger in self.chargers:
            for gx in range(charger["x"], charger["x"] + charger["w"]):
                for gz in range(charger["z"], charger["z"] + charger["h"]):
                    if self._in_bounds(gx, gz):
                        cells.append((gx, gz))
        return cells

    def _collect_frontiers(self):
        frontier_cells = []
        passable_coords = np.argwhere((self.map_state == self.CLEAN) | (self.map_state == self.DIRTY))
        for gx, gz in passable_coords:
            if self._has_unknown_neighbor((int(gx), int(gz))):
                frontier_cells.append((int(gx), int(gz)))
        return frontier_cells

    def _has_unknown_neighbor(self, pos):
        x, z = pos
        for dx, dz in self.ACTION_DELTAS:
            nx = x + dx
            nz = z + dz
            if self._in_bounds(nx, nz) and self.map_state[nx, nz] == self.UNKNOWN:
                return True
        return False

    def _cluster_cells(self, cells, radius, weight_fn):
        if len(cells) == 0:
            return []

        scored = []
        for cell in cells:
            gx = int(cell[0])
            gz = int(cell[1])
            scored.append((float(weight_fn(gx, gz)), gx, gz))
        scored.sort(reverse=True)

        selected = []
        for _, gx, gz in scored:
            too_close = False
            for sx, sz in selected:
                if self._chebyshev((gx, gz), (sx, sz)) <= radius:
                    too_close = True
                    break
            if too_close:
                continue
            selected.append((gx, gz))
            if len(selected) >= 64:
                break

        return selected

    def _frontier_gain(self, x, z):
        return self._unknown_density((x, z), radius=3)

    def _charger_dist_at(self, pos, charger_dist):
        x, z = pos
        if charger_dist is not None:
            d = int(charger_dist[x, z])
            if d >= 0:
                return d
        return int(np.ceil(self._nearest_charger_heuristic(pos) * Config.CHARGER_HEURISTIC_SCALE))

    def _current_charger_distance(self):
        if self.on_charger:
            return 0
        return self._charger_dist_at(self.cur_pos, self._charger_dist_map)

    def _reconstruct_path(self, goal, prev_x, prev_z):
        gx, gz = goal
        if not self._in_bounds(gx, gz) or prev_x[gx, gz] == -1 and (gx, gz) != self.cur_pos:
            return []

        path = []
        cur = (gx, gz)
        while cur != self.cur_pos:
            path.append(cur)
            px = int(prev_x[cur[0], cur[1]])
            pz = int(prev_z[cur[0], cur[1]])
            if px < 0 or pz < 0:
                return []
            cur = (px, pz)

        path.reverse()
        return path

    def _score_actions(self, legal_action):
        scores = np.full((Config.ACTION_DIM,), -1e9, dtype=np.float32)
        charger_dist_now = self._current_charger_distance()
        goal_dist_now = self._chebyshev(self.cur_pos, self.goal) if self.goal is not None else 0

        for act, (dx, dz) in enumerate(self.ACTION_DELTAS):
            if not legal_action[act]:
                continue

            nx = self.cur_pos[0] + dx
            nz = self.cur_pos[1] + dz
            next_pos = (nx, nz)

            score = -1.6 * self._transition_cost(self.cur_pos, next_pos)
            if self._in_bounds(nx, nz):
                if self.map_state[nx, nz] == self.DIRTY:
                    score += 8.0
                score += 1.5 * self._frontier_gain(nx, nz)
                score -= 0.45 * self.visit_count[nx, nz]
                score -= 0.75 * self.clean_pass_count[nx, nz]

            if next_pos == self.last_pos:
                score -= 2.0
            if self.stuck_chain > 0 and act == self.last_action:
                score -= 6.0

            if self.charge_mode:
                charger_dist_next = self._charger_dist_at(next_pos, self._charger_dist_map)
                score += 3.8 * (charger_dist_now - charger_dist_next)
                if self._is_on_charger(next_pos):
                    score += 7.0
            elif self.goal is not None:
                score += 1.35 * (goal_dist_now - self._chebyshev(next_pos, self.goal))

            scores[act] = float(score)
        return scores

    def _scores_to_probs(self, scores, legal_action):
        mask = np.array(legal_action, dtype=np.float32)
        if float(np.sum(mask)) <= 0.0:
            return np.full(Config.ACTION_DIM, 1.0 / Config.ACTION_DIM, dtype=np.float32)

        masked_scores = np.array(scores, dtype=np.float32)
        masked_scores[mask <= 0.0] = -1e9
        valid_scores = masked_scores[mask > 0.0]
        max_score = float(np.max(valid_scores)) if valid_scores.size > 0 else 0.0
        logits = np.full(Config.ACTION_DIM, -1e9, dtype=np.float32)
        logits[mask > 0.0] = (masked_scores[mask > 0.0] - max_score) / 1.35
        probs = np.exp(np.clip(logits, -30.0, 20.0)) * mask
        denom = float(np.sum(probs))
        if denom <= 0.0:
            return mask / np.sum(mask)
        return probs / denom

    def _dirty_density(self, x, z):
        x0 = max(0, x - 2)
        x1 = min(Config.GRID_SIZE, x + 3)
        z0 = max(0, z - 2)
        z1 = min(Config.GRID_SIZE, z + 3)
        return int(np.sum(self.map_state[x0:x1, z0:z1] == self.DIRTY))

    def _unknown_density(self, pos, radius=2):
        x, z = pos
        x0 = max(0, x - radius)
        x1 = min(Config.GRID_SIZE, x + radius + 1)
        z0 = max(0, z - radius)
        z1 = min(Config.GRID_SIZE, z + radius + 1)
        window = self.map_state[x0:x1, z0:z1]
        return int(np.sum(window == self.UNKNOWN))

    def _local_clean_density(self, pos, radius=2):
        x, z = pos
        x0 = max(0, x - radius)
        x1 = min(Config.GRID_SIZE, x + radius + 1)
        z0 = max(0, z - radius)
        z1 = min(Config.GRID_SIZE, z + radius + 1)
        window = self.map_state[x0:x1, z0:z1]
        return int(np.sum(window == self.CLEAN))

    def _nearest_charger_heuristic(self, pos):
        if not self.chargers:
            return Config.GRID_SIZE
        return min(self._dist_to_charger_rect(pos, charger) for charger in self.chargers)

    def _dist_to_charger_rect(self, pos, charger):
        x, z = pos
        x0 = charger["x"]
        z0 = charger["z"]
        x1 = x0 + charger["w"] - 1
        z1 = z0 + charger["h"] - 1
        near_x = min(max(x, x0), x1)
        near_z = min(max(z, z0), z1)
        return self._chebyshev(pos, (near_x, near_z))

    def _npc_zone_hard_block(self, pos):
        for center in self.npc_centers.values():
            if self._chebyshev(pos, center) <= Config.NPC_CENTER_HARD_RADIUS:
                return True
        return False

    def _npc_zone_penalty(self, pos):
        penalty = 0.0
        for center in self.npc_centers.values():
            dist = self._chebyshev(pos, center)
            if dist <= Config.NPC_CENTER_HARD_RADIUS:
                penalty += 25.0
            elif dist <= Config.NPC_CENTER_SOFT_RADIUS:
                penalty += float(Config.NPC_CENTER_SOFT_RADIUS - dist + 1)

        for idx, npc_pos in self.npc_positions.items():
            dist = self._chebyshev(pos, npc_pos)
            if dist <= 2:
                penalty += 30.0
            prev_pos = self.npc_prev_positions.get(idx, npc_pos)
            pred = (
                npc_pos[0] + int(np.clip(npc_pos[0] - prev_pos[0], -1, 1)),
                npc_pos[1] + int(np.clip(npc_pos[1] - prev_pos[1], -1, 1)),
            )
            if self._chebyshev(pos, pred) <= 2:
                penalty += 18.0
        return penalty

    def _is_on_charger(self, pos):
        x, z = pos
        for charger in self.chargers:
            if charger["x"] <= x < charger["x"] + charger["w"] and charger["z"] <= z < charger["z"] + charger["h"]:
                return True
        return False

    def _build_feature(self):
        local_map = np.zeros((Config.LOCAL_VIEW_SIZE, Config.LOCAL_VIEW_SIZE), dtype=np.float32)
        local_visit = np.zeros_like(local_map)
        local_transit = np.zeros_like(local_map)
        local_risk = np.zeros_like(local_map)
        local_frontier = np.zeros_like(local_map)

        for row in range(Config.LOCAL_VIEW_SIZE):
            for col in range(Config.LOCAL_VIEW_SIZE):
                gx = self.cur_pos[0] - self.LOCAL_HALF + col
                gz = self.cur_pos[1] - self.LOCAL_HALF + row
                if not self._in_bounds(gx, gz):
                    local_map[row, col] = -1.0
                    local_risk[row, col] = 1.0
                    continue

                cell = int(self.map_state[gx, gz])
                if cell == self.OBSTACLE:
                    local_map[row, col] = -1.0
                elif cell == self.UNKNOWN:
                    local_map[row, col] = -0.35
                elif cell == self.CLEAN:
                    local_map[row, col] = 0.25
                else:
                    local_map[row, col] = 1.0

                local_visit[row, col] = min(float(self.visit_count[gx, gz]) / float(Config.MAX_VISIT_CLIP), 1.0)
                local_transit[row, col] = min(
                    float(self.clean_pass_count[gx, gz]) / float(Config.MAX_TRANSIT_CLIP), 1.0
                )
                local_risk[row, col] = min(self._npc_zone_penalty((gx, gz)) / 40.0, 1.0)
                local_frontier[row, col] = min(self._frontier_gain(gx, gz) / 8.0, 1.0)

        local_feature = np.concatenate(
            [
                local_map.flatten(),
                local_visit.flatten(),
                local_transit.flatten(),
                local_risk.flatten(),
                local_frontier.flatten(),
            ]
        ).astype(np.float32)

        nearest_charger = self._current_charger_distance()
        nearest_npc = min(
            [self._chebyshev(self.cur_pos, pos) for pos in self.npc_positions.values()],
            default=Config.GRID_SIZE,
        )
        goal_dx, goal_dz = (0.0, 0.0)
        if self.goal is not None:
            goal_dx = _signed_norm(self.goal[0] - self.cur_pos[0], Config.GRID_SIZE)
            goal_dz = _signed_norm(self.goal[1] - self.cur_pos[1], Config.GRID_SIZE)

        cx, cz = self.cur_pos
        explored_ratio = float(np.mean(self.map_state != self.UNKNOWN))
        goal_dirty = 1.0 if self.goal_kind == "dirty" else 0.0
        goal_frontier = 1.0 if self.goal_kind == "frontier" else 0.0
        goal_charger = 1.0 if self.goal_kind == "charger" else 0.0

        global_feature = np.array(
            [
                _norm(self.step_no, max(self.max_step, 1)),
                _norm(self.battery, self.battery_max),
                _norm(self.dirt_cleaned, self.total_dirt),
                1.0 - _norm(self.dirt_cleaned, self.total_dirt),
                _norm(cx, Config.GRID_SIZE),
                _norm(cz, Config.GRID_SIZE),
                _norm(nearest_charger, Config.GRID_SIZE),
                _norm(nearest_npc, Config.GRID_SIZE),
                1.0 if self.charge_mode else 0.0,
                1.0 if self.on_charger else 0.0,
                _norm(len(self.path), Config.MAX_TRACKED_PATH),
                _norm(self._unknown_density(self.cur_pos, radius=3), 49),
                _norm(int(np.sum(self._view_map == self.DIRTY)), Config.VIEW_SIZE * Config.VIEW_SIZE),
                _norm(int(np.sum(self.map_state == self.DIRTY)), 512),
                goal_dx,
                goal_dz,
                _norm(int(self.visit_count[cx, cz]), Config.MAX_VISIT_CLIP),
                _norm(int(self.clean_pass_count[cx, cz]), Config.MAX_TRANSIT_CLIP),
                self._charge_slack(),
                _norm(self.stuck_chain, 8),
                explored_ratio,
                _norm(self._frontier_gain(cx, cz), 49),
                _norm(self.charge_count, 10),
                goal_dirty,
                goal_frontier,
                goal_charger,
            ],
            dtype=np.float32,
        )

        return np.concatenate([local_feature, global_feature]).astype(np.float32)

    def reward_process(self):
        reward = Config.STEP_PENALTY
        reward += Config.CLEAN_REWARD * float(len(self._current_step_cleaned))

        if self.cur_pos == self.last_pos and self.last_action != -1:
            reward += Config.STUCK_PENALTY

        if self.on_charger and not self.last_on_charger:
            reward += Config.CHARGE_REWARD

        if self._npc_zone_penalty(self.cur_pos) >= 25.0:
            reward += Config.NPC_DANGER_PENALTY

        cx, cz = self.cur_pos
        if (
            self._in_bounds(cx, cz)
            and self.map_state[cx, cz] == self.CLEAN
            and (cx, cz) not in self._current_step_cleaned
        ):
            reward -= Config.REPEAT_CLEAN_PENALTY * min(float(self.clean_pass_count[cx, cz]), 6.0)

        if self._has_unknown_neighbor(self.cur_pos):
            reward += Config.FRONTIER_REWARD

        return float(reward)

    def _pos_to_action(self, next_pos):
        dx = next_pos[0] - self.cur_pos[0]
        dz = next_pos[1] - self.cur_pos[1]
        for act, delta in enumerate(self.ACTION_DELTAS):
            if delta == (dx, dz):
                return act
        return None

    def _chebyshev(self, a, b):
        return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))

    def _in_bounds(self, x, z):
        return 0 <= x < Config.GRID_SIZE and 0 <= z < Config.GRID_SIZE
