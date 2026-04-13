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
    GRID_X, GRID_Z = np.indices((Config.GRID_SIZE, Config.GRID_SIZE), dtype=np.float32)

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
        self.npc_pred_positions = {}
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
        self.plan_mode = "explore"
        self.current_region_id = None
        self.region_seq = []
        self.region_graph = {}
        self.regions = {}
        self.region_revision_step = -999
        self.coverage_targets = []
        self._active_region_dist_map = None
        self._coverage_target_dist_map = None
        self.charge_exit_pending = False
        self.charge_exit_region_id = None

        self.pending_action = 0
        self.pending_prob = np.full(Config.ACTION_DIM, 1.0 / Config.ACTION_DIM, dtype=np.float32)
        self.region_lock_kind = None
        self.region_lock_center = None
        self._unknown_neighbor_mask = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        self._frontier_mask = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        self._npc_zone_hard_block_map = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        self._npc_dynamic_hard_block_map = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        self._npc_penalty_map = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=np.float32)
        self._plannable_unknown_map = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        self._plannable_known_map = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        self._dirty_integral = np.zeros((Config.GRID_SIZE + 1, Config.GRID_SIZE + 1), dtype=np.float32)
        self._unknown_integral = np.zeros((Config.GRID_SIZE + 1, Config.GRID_SIZE + 1), dtype=np.float32)
        self._clean_integral = np.zeros((Config.GRID_SIZE + 1, Config.GRID_SIZE + 1), dtype=np.float32)
        self._visit_integral = np.zeros((Config.GRID_SIZE + 1, Config.GRID_SIZE + 1), dtype=np.float32)
        self._transit_integral = np.zeros((Config.GRID_SIZE + 1, Config.GRID_SIZE + 1), dtype=np.float32)
        self._bottleneck_map = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)

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
        if risk >= 18.0:
            bias += Config.RISK_TEACHER_BIAS
        elif risk >= 12.0:
            bias += 0.5 * Config.RISK_TEACHER_BIAS
        if self.charge_mode and self._charge_slack() <= 0.18:
            bias += Config.CHARGE_TEACHER_BIAS
        if self.stuck_chain > 0:
            bias += Config.STUCK_TEACHER_BIAS * min(self.stuck_chain, 3)
        return float(np.clip(bias, 0.0, 0.22))

    def get_teacher_weight(self):
        if self.should_force_teacher():
            return 1.0

        risk = self._npc_zone_penalty(self.cur_pos)
        if risk >= 18.0:
            return 0.85
        if self.charge_mode and self._charge_slack() <= 0.12:
            return 0.55
        if self.stuck_chain > 0:
            return min(0.65, 0.25 + 0.15 * self.stuck_chain)
        return 0.0

    def get_policy_weight(self):
        if self.should_force_teacher():
            return 0.0

        risk = self._npc_zone_penalty(self.cur_pos)
        if risk >= 18.0:
            return 0.35
        if self.charge_mode and self._charge_slack() <= 0.12:
            return 0.55
        if self.stuck_chain > 0:
            return 0.6
        return 1.0

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

        self._refresh_spatial_caches()

    def _build_integral_grid(self, grid):
        padded = np.pad(grid.astype(np.float32, copy=False), ((1, 0), (1, 0)), mode="constant")
        return padded.cumsum(axis=0).cumsum(axis=1)

    def _window_sum(self, integral, x0, x1, z0, z1):
        return float(integral[x1, z1] - integral[x0, z1] - integral[x1, z0] + integral[x0, z0])

    def _stamp_chebyshev_disk(self, target, center, radius):
        if radius < 0:
            return
        x, z = int(center[0]), int(center[1])
        x0 = max(0, x - radius)
        x1 = min(Config.GRID_SIZE, x + radius + 1)
        z0 = max(0, z - radius)
        z1 = min(Config.GRID_SIZE, z + radius + 1)
        target[x0:x1, z0:z1] = True

    def _refresh_spatial_caches(self):
        dirty_mask = self.map_state == self.DIRTY
        unknown_mask = self.map_state == self.UNKNOWN
        clean_mask = self.map_state == self.CLEAN
        self._plannable_unknown_map = self.map_state != self.OBSTACLE
        self._plannable_known_map = clean_mask | dirty_mask

        self._dirty_integral = self._build_integral_grid(dirty_mask)
        self._unknown_integral = self._build_integral_grid(unknown_mask)
        self._clean_integral = self._build_integral_grid(clean_mask)
        self._visit_integral = self._build_integral_grid(self.visit_count)
        self._transit_integral = self._build_integral_grid(self.clean_pass_count)

        padded_unknown = np.pad(unknown_mask, 1, mode="constant", constant_values=False)
        neighbor_mask = np.zeros_like(unknown_mask, dtype=bool)
        for dx, dz in self.ACTION_DELTAS:
            neighbor_mask |= padded_unknown[1 + dx : 1 + dx + Config.GRID_SIZE, 1 + dz : 1 + dz + Config.GRID_SIZE]
        self._unknown_neighbor_mask = neighbor_mask
        self._frontier_mask = self._plannable_known_map & self._unknown_neighbor_mask

        zone_block = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        dynamic_block = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        penalty = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=np.float32)

        grid_x = self.GRID_X
        grid_z = self.GRID_Z
        for center in self.npc_centers.values():
            self._stamp_chebyshev_disk(zone_block, center, Config.NPC_CENTER_HARD_RADIUS)
            dx = grid_x - float(center[0])
            dz = grid_z - float(center[1])
            cheb = np.maximum(np.abs(dx), np.abs(dz))
            contrib = Config.NPC_FIELD_CENTER_SCALE / (dx * dx + dz * dz + (Config.NPC_FIELD_SOFTEN + 2.0))
            penalty += np.where(cheb <= Config.NPC_CENTER_SOFT_RADIUS, contrib, 0.0).astype(np.float32)

        for idx, npc_pos in self.npc_positions.items():
            pred = self.npc_pred_positions.get(idx, npc_pos)
            mid = ((npc_pos[0] + pred[0]) // 2, (npc_pos[1] + pred[1]) // 2)

            self._stamp_chebyshev_disk(dynamic_block, npc_pos, Config.NPC_COLLISION_RADIUS + 1)
            self._stamp_chebyshev_disk(dynamic_block, pred, Config.NPC_PREDICT_RADIUS + 1)
            self._stamp_chebyshev_disk(dynamic_block, mid, 1)

            for source, strength, soften in (
                (npc_pos, Config.NPC_FIELD_NPC_SCALE, Config.NPC_FIELD_SOFTEN),
                (pred, Config.NPC_FIELD_PRED_SCALE, Config.NPC_FIELD_SOFTEN),
                (mid, Config.NPC_FIELD_MID_SCALE, Config.NPC_FIELD_SOFTEN + 0.5),
            ):
                dx = grid_x - float(source[0])
                dz = grid_z - float(source[1])
                penalty += (strength / (dx * dx + dz * dz + soften)).astype(np.float32)

        self._npc_zone_hard_block_map = zone_block
        self._npc_dynamic_hard_block_map = dynamic_block
        self._npc_penalty_map = penalty
        self._bottleneck_map = self._build_bottleneck_map()

    def _build_bottleneck_map(self):
        bottleneck = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        passable = self._plannable_known_map & (~self._npc_zone_hard_block_map) & (~self._npc_dynamic_hard_block_map)
        for x in range(Config.GRID_SIZE):
            for z in range(Config.GRID_SIZE):
                if not passable[x, z]:
                    continue
                neighbors = []
                for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx = x + dx
                    nz = z + dz
                    neighbors.append(self._in_bounds(nx, nz) and passable[nx, nz])
                degree = int(sum(neighbors))
                if degree <= 1:
                    bottleneck[x, z] = True
                    continue
                if degree == 2:
                    left, right, up, down = neighbors[1], neighbors[0], neighbors[2], neighbors[3]
                    if (left and right and not up and not down) or (up and down and not left and not right):
                        bottleneck[x, z] = True
                        continue
                    # L-shaped corridors are also treated as narrow mandatory turns.
                    bottleneck[x, z] = True
        return bottleneck

    def _update_stuck_state(self):
        if self.cur_pos != self.last_pos or self.last_action < 0:
            self.stuck_chain = 0
            return

        self.stuck_chain = min(self.stuck_chain + 1, 8)
        self.path = []
        self.goal = None
        self.goal_kind = None
        self._clear_region_lock()

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
        self.npc_pred_positions = {}
        for idx, npc_pos in self.npc_positions.items():
            prev_pos = self.npc_prev_positions.get(idx, npc_pos)
            self.npc_pred_positions[idx] = (
                npc_pos[0] + int(np.clip(npc_pos[0] - prev_pos[0], -1, 1)),
                npc_pos[1] + int(np.clip(npc_pos[1] - prev_pos[1], -1, 1)),
            )

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
            if self._npc_dynamic_hard_block(next_pos):
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
            if self._npc_dynamic_hard_block(next_pos):
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
            if self._npc_dynamic_hard_block(next_pos):
                continue
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
            # Environment diagonal rule: at least one side corridor must be passable.
            if not (self._is_local_passable(dx, 0) or self._is_local_passable(0, dz)):
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

            pred = self.npc_pred_positions.get(idx, npc_pos)
            if self._chebyshev(pos, pred) <= Config.NPC_PREDICT_RADIUS:
                return True
            mid = ((npc_pos[0] + pred[0]) // 2, (npc_pos[1] + pred[1]) // 2)
            if self._chebyshev(pos, mid) <= 1:
                return True
        return False

    def _npc_dynamic_hard_block(self, pos):
        x, z = int(pos[0]), int(pos[1])
        if not self._in_bounds(x, z):
            return True
        return bool(self._npc_dynamic_hard_block_map[x, z])

    def _select_action(self, legal_action):
        self._trim_path_head()
        if self._need_charge():
            if not self.charge_mode:
                self._clear_active_region()
            self.charge_mode = True
            self.plan_mode = "charge"
            self._clear_region_lock()
        elif self.on_charger and self.battery >= int(self.battery_max * 0.98):
            self.charge_mode = False
            self.plan_mode = "transit" if self._in_fill_phase() else "explore"
            self._prepare_charge_exit_switch()
            if self.charge_exit_pending:
                self.path = []

        if self._need_replan():
            self._replan()

        action_scores = self._score_actions(legal_action)
        if self.path:
            next_pos = self.path[0]
            act = self._pos_to_action(next_pos)
            if act is not None and legal_action[act]:
                follow_scale = self._repeat_bias_decay(next_pos, radius=1)
                if self._in_explore_phase():
                    follow_scale = min(follow_scale, 0.72)
                action_scores[act] += Config.PATH_FOLLOW_BONUS * follow_scale

        self.pending_prob = self._scores_to_probs(action_scores, legal_action)
        if float(np.sum(self.pending_prob)) > 0.0:
            chosen = int(np.argmax(self.pending_prob))
            if self.charge_exit_pending and self.on_charger:
                dx, dz = self.ACTION_DELTAS[chosen]
                if dx == 0 or dz == 0:
                    exit_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
                    if not self._is_on_charger(exit_pos):
                        self.charge_exit_pending = False
                        self.charge_exit_region_id = None
                    self.path = []
            return chosen

        for act, ok in enumerate(legal_action):
            if ok:
                if self.charge_exit_pending and self.on_charger:
                    dx, dz = self.ACTION_DELTAS[act]
                    if dx == 0 or dz == 0:
                        exit_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
                        if not self._is_on_charger(exit_pos):
                            self.charge_exit_pending = False
                            self.charge_exit_region_id = None
                        self.path = []
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

    def _in_explore_phase(self):
        return self.step_no < Config.EXPLORE_PHASE_STEPS

    def _repeat_bias_decay(self, pos, radius=1):
        repeat_density = self._local_repeat_density(pos, radius=radius)
        return float(1.0 / (1.0 + 0.12 * repeat_density))

    def _in_fill_phase(self):
        return not self._in_explore_phase()

    def _clear_active_region(self):
        self.current_region_id = None
        self.region_seq = []
        self.coverage_targets = []
        self._active_region_dist_map = None
        self._coverage_target_dist_map = None
        self.region_lock_kind = None
        self.region_lock_center = None
        self.charge_exit_pending = False
        self.charge_exit_region_id = None

    def _current_region(self):
        return self.regions.get(self.current_region_id)

    def _cell_parity(self, pos):
        return (int(pos[0]) + int(pos[1])) & 1

    def _diag_stripe_id(self, pos, family):
        x = int(pos[0])
        z = int(pos[1])
        if family == "sum":
            return x + z
        return x - z

    def _diag_progress_key(self, pos, family):
        x = int(pos[0])
        z = int(pos[1])
        if family == "sum":
            return x - z
        return x + z

    def _choose_region_diag_family(self, cells):
        if not cells:
            return "diff"

        def family_score(family):
            stripes = {}
            for cell in cells:
                sid = self._diag_stripe_id(cell, family)
                stripes.setdefault(sid, []).append(self._diag_progress_key(cell, family))

            segments = 0
            stripe_count = len(stripes)
            mean_len = 0.0
            for keys in stripes.values():
                ordered = sorted(set(int(v) for v in keys))
                if not ordered:
                    continue
                mean_len += float(len(ordered))
                segments += 1
                for idx in range(1, len(ordered)):
                    if abs(int(ordered[idx]) - int(ordered[idx - 1])) > 2:
                        segments += 1
            if stripe_count > 0:
                mean_len /= float(stripe_count)
            return float(segments + 0.30 * stripe_count - 0.20 * mean_len)

        score_sum = family_score("sum")
        score_diff = family_score("diff")
        return "sum" if score_sum < score_diff else "diff"

    def _diag_direct_segment(self, start_pos, goal_pos, family, allowed_mask):
        if start_pos == goal_pos:
            return []
        if self._diag_stripe_id(start_pos, family) != self._diag_stripe_id(goal_pos, family):
            return None

        if family == "sum":
            step = (1, -1) if goal_pos[0] > start_pos[0] else (-1, 1)
        else:
            step = (1, 1) if goal_pos[0] > start_pos[0] else (-1, -1)

        path = []
        cur = start_pos
        while cur != goal_pos:
            cur = (cur[0] + step[0], cur[1] + step[1])
            if not self._in_bounds(*cur):
                return None
            if allowed_mask is not None and not allowed_mask[cur[0], cur[1]]:
                return None
            path.append(cur)
            if len(path) > Config.MAX_TRACKED_PATH:
                return None
        return path

    def _prepare_charge_exit_switch(self):
        if not self._in_fill_phase():
            self.charge_exit_pending = False
            self.charge_exit_region_id = None
            return

        region = self._current_region()
        if region is None:
            self.charge_exit_pending = False
            self.charge_exit_region_id = None
            return

        region["diag_phase"] = 1 - int(region.get("diag_phase", self._cell_parity(region["anchor"])))
        region["diag_forward"] = -1 * int(region.get("diag_forward", 1))
        self.charge_exit_pending = True
        self.charge_exit_region_id = int(region["id"])

    def _charge_exit_action_bonus(self, next_pos):
        if not self.charge_exit_pending or not self.on_charger:
            return 0.0

        dx = next_pos[0] - self.cur_pos[0]
        dz = next_pos[1] - self.cur_pos[1]
        if dx != 0 and dz != 0:
            return -Config.CHARGE_EXIT_DIAGONAL_PENALTY

        region = self.regions.get(self.charge_exit_region_id)
        bonus = Config.CHARGE_EXIT_STRAIGHT_BONUS
        if region is not None:
            desired_phase = int(region.get("diag_phase", self._cell_parity(next_pos)))
            if self._cell_parity(next_pos) == desired_phase:
                bonus += 1.4
            else:
                bonus -= 0.9
            bonus -= 0.10 * self._distance_to_region(next_pos, region["kind"], region["anchor"])
            bonus -= 0.06 * self._npc_zone_penalty(next_pos)
        return float(bonus)

    def _coverage_diagonal_action_bonus(self, next_pos, region):
        if region is None or self.plan_mode != "coverage" or self.charge_mode:
            return 0.0
        if not self._inside_region(region, next_pos):
            return 0.0

        dx = next_pos[0] - self.cur_pos[0]
        dz = next_pos[1] - self.cur_pos[1]
        is_diag = dx != 0 and dz != 0
        desired_phase = int(region.get("diag_phase", self._cell_parity(next_pos)))
        bonus = 0.0

        if is_diag:
            bonus += Config.COVERAGE_DIAGONAL_ACTION_BONUS
            if self._cell_parity(next_pos) == desired_phase:
                bonus += 0.45
        else:
            if self._cell_parity(next_pos) == desired_phase:
                bonus += Config.COVERAGE_ORTHO_SWITCH_BONUS
            else:
                bonus -= Config.COVERAGE_ORTHO_PENALTY

        family = region.get("diag_family", "diff")
        last_stripe_id = region.get("last_stripe_id")
        if last_stripe_id is not None:
            stripe_gap = abs(self._diag_stripe_id(next_pos, family) - int(last_stripe_id))
            bonus -= 0.08 * max(0, stripe_gap - 1)

        return float(bonus)

    def _inside_region(self, region, pos=None):
        if region is None:
            return False
        if pos is None:
            pos = self.cur_pos
        x, z = int(pos[0]), int(pos[1])
        if not self._in_bounds(x, z):
            return False
        return bool(region["travel_mask"][x, z])

    def _corridor_repeat_discount(self, pos):
        x, z = int(pos[0]), int(pos[1])
        if not self._in_bounds(x, z):
            return 1.0
        if self._bottleneck_map[x, z]:
            return Config.BOTTLENECK_REPEAT_DISCOUNT
        return 1.0

    def _path_overlap_penalty(self, path, transit_bias=1.0):
        overlap = 0.0
        risk = 0.0
        seen = set()
        for cell in path:
            x, z = int(cell[0]), int(cell[1])
            if not self._in_bounds(x, z):
                continue
            cell_penalty = 0.12 * float(self.visit_count[x, z]) + 0.18 * float(self.clean_pass_count[x, z])
            if cell in seen:
                cell_penalty += 0.35
            if self._bottleneck_map[x, z]:
                cell_penalty *= Config.BOTTLENECK_PATH_DISCOUNT
            overlap += cell_penalty
            risk += 0.03 * min(self._npc_zone_penalty(cell), 30.0)
            seen.add(cell)
        return float(transit_bias * overlap + risk), float(overlap), float(risk)

    def _path_transition_cost(self, start_pos, path, mode=None):
        total = 0.0
        cur = start_pos
        for pos in path:
            total += self._transition_cost(cur, pos, mode=mode)
            cur = pos
        return float(total)

    def _expand_mask_on_known(self, base_mask, radius):
        if radius <= 0:
            return base_mask.copy()
        result = base_mask.copy()
        passable = self._plannable_known_map & (~self._npc_zone_hard_block_map) & (~self._npc_dynamic_hard_block_map)
        cells = np.argwhere(base_mask)
        for gx, gz in cells:
            x0 = max(0, int(gx) - radius)
            x1 = min(Config.GRID_SIZE, int(gx) + radius + 1)
            z0 = max(0, int(gz) - radius)
            z1 = min(Config.GRID_SIZE, int(gz) + radius + 1)
            result[x0:x1, z0:z1] |= passable[x0:x1, z0:z1]
        return result

    def _choose_region_anchor(self, cells):
        if not cells:
            return None
        arr = np.array(cells, dtype=np.float32)
        center = np.mean(arr, axis=0)
        best_score = float("inf")
        best_cell = cells[0]
        for cell in cells:
            dist = abs(float(cell[0]) - center[0]) + abs(float(cell[1]) - center[1])
            score = dist + 0.12 * self._local_repeat_density(cell, radius=1) + 0.05 * self._npc_zone_penalty(cell)
            if self.map_state[cell[0], cell[1]] == self.DIRTY:
                score -= 0.35
            if score < best_score:
                best_score = score
                best_cell = cell
        return best_cell

    def _select_region_entry_cells(self, travel_mask, anchor):
        cells = [tuple(map(int, pos)) for pos in np.argwhere(travel_mask)]
        if not cells:
            return []
        candidates = []
        for cell in cells:
            x, z = cell
            is_border = False
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx = x + dx
                nz = z + dz
                if not self._in_bounds(nx, nz) or not travel_mask[nx, nz]:
                    is_border = True
                    break
            if not is_border:
                continue
            score = (
                self._local_repeat_density(cell, radius=1)
                + 0.08 * self._npc_zone_penalty(cell)
                + 0.06 * self._chebyshev(cell, anchor)
            )
            candidates.append((float(score), cell))
        if not candidates:
            candidates = [(0.0, cell) for cell in cells]
        candidates.sort(key=lambda item: item[0])
        return [cell for _, cell in candidates[: Config.REGION_ENTRY_CANDIDATES]]

    def _build_real_regions(self):
        old_regions = self.regions if isinstance(self.regions, dict) else {}
        seed_mask = (self.map_state == self.DIRTY) | self._frontier_mask
        seed_mask &= self._plannable_known_map
        seed_mask &= (~self._npc_zone_hard_block_map)
        seed_mask &= (~self._npc_dynamic_hard_block_map)

        visited = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        regions = {}
        region_id = 0

        for start in np.argwhere(seed_mask):
            sx = int(start[0])
            sz = int(start[1])
            if visited[sx, sz]:
                continue

            queue = deque([(sx, sz)])
            visited[sx, sz] = True
            component = []
            while queue:
                x, z = queue.popleft()
                component.append((x, z))
                for dx, dz in self.ACTION_DELTAS:
                    nx = x + dx
                    nz = z + dz
                    if not self._in_bounds(nx, nz):
                        continue
                    if visited[nx, nz] or not seed_mask[nx, nz]:
                        continue
                    visited[nx, nz] = True
                    queue.append((nx, nz))

            if not component:
                continue

            seed_component_mask = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
            for x, z in component:
                seed_component_mask[x, z] = True

            support_mask = self._expand_mask_on_known(seed_component_mask, Config.REGION_SUPPORT_RADIUS)
            travel_mask = self._expand_mask_on_known(support_mask, Config.REGION_TRAVEL_RADIUS)
            travel_mask &= self._plannable_known_map
            travel_mask &= (~self._npc_zone_hard_block_map)
            travel_mask &= (~self._npc_dynamic_hard_block_map)
            if not np.any(travel_mask):
                continue

            dirty_cells = [cell for cell in component if self.map_state[cell[0], cell[1]] == self.DIRTY]
            frontier_cells = [cell for cell in component if self._frontier_mask[cell[0], cell[1]]]
            anchor = self._choose_region_anchor(dirty_cells if dirty_cells else component)
            if anchor is None:
                continue

            entry_cells = self._select_region_entry_cells(travel_mask, anchor)
            region_kind = "dirty" if len(dirty_cells) >= len(frontier_cells) else "frontier"
            travel_cells = [tuple(map(int, pos)) for pos in np.argwhere(travel_mask)]
            dominant_cells = dirty_cells if dirty_cells else component
            parity_counts = [0, 0]
            for cell in dominant_cells:
                parity_counts[self._cell_parity(cell)] += 1
            default_phase = 0 if parity_counts[0] >= parity_counts[1] else 1
            diag_family = self._choose_region_diag_family(component)
            matched_old = None
            matched_dist = 1e9
            for old_region in old_regions.values():
                if old_region.get("kind") != region_kind:
                    continue
                old_anchor = old_region.get("anchor")
                if old_anchor is None:
                    continue
                dist = self._chebyshev(anchor, old_anchor)
                if dist < matched_dist:
                    matched_dist = dist
                    matched_old = old_region

            repeat_mean = float(np.mean([self._local_repeat_density(cell, radius=1) for cell in component]))
            risk_mean = float(np.mean([self._npc_zone_penalty(cell) for cell in component]))
            coverage_estimate = float(len(component) + 0.22 * len(travel_cells))
            region_value = (
                Config.REGION_VALUE_DIRTY_WEIGHT * float(len(dirty_cells))
                + Config.REGION_VALUE_FRONTIER_WEIGHT * float(len(frontier_cells))
                - Config.REGION_VALUE_REPEAT_WEIGHT * repeat_mean
                - Config.REGION_VALUE_RISK_WEIGHT * risk_mean
            )

            regions[region_id] = {
                "id": region_id,
                "kind": region_kind,
                "anchor": anchor,
                "seed_cells": component,
                "dirty_cells": dirty_cells,
                "frontier_cells": frontier_cells,
                "seed_mask": seed_component_mask,
                "travel_mask": travel_mask,
                "travel_cells": travel_cells,
                "entry_cells": entry_cells if entry_cells else [anchor],
                "value": float(region_value),
                "coverage_estimate": coverage_estimate,
                "charger_return": float(self._charger_dist_at(anchor, self._charger_dist_map)),
                "diag_family": matched_old.get("diag_family", diag_family) if matched_old is not None else diag_family,
                "diag_phase": int(matched_old.get("diag_phase", default_phase)) if matched_old is not None else int(default_phase),
                "diag_forward": int(matched_old.get("diag_forward", 1)) if matched_old is not None else 1,
                "last_stripe_id": matched_old.get("last_stripe_id") if matched_old is not None else None,
            }
            region_id += 1

        old_graph = self.region_graph if isinstance(self.region_graph, dict) else {}
        self.regions = regions
        self.region_graph = {rid: dict(old_graph.get(rid, {})) for rid in regions}

    def _match_region_by_anchor(self, anchor, kind=None):
        best_id = None
        best_score = 1e9
        for rid, region in self.regions.items():
            if kind is not None and region["kind"] != kind:
                continue
            dist = self._chebyshev(anchor, region["anchor"])
            if self._inside_region(region, anchor):
                return rid
            if dist < best_score:
                best_score = dist
                best_id = rid
        return best_id

    def _region_remaining_targets(self, region):
        if region is None:
            return []
        merged = {}
        for cell in region.get("travel_cells", []):
            x, z = int(cell[0]), int(cell[1])
            if self.map_state[x, z] == self.DIRTY:
                merged[(x, z)] = 2.0
            elif self._frontier_mask[x, z] and self._plannable_known_map[x, z]:
                merged[(x, z)] = 1.0
        return list(merged.keys())

    def _region_has_remaining_work(self, region):
        return len(self._region_remaining_targets(region)) > 0

    def _refresh_active_region_maps(self, region):
        if region is None:
            self._active_region_dist_map = None
            self._coverage_target_dist_map = None
            return
        self._active_region_dist_map = self._build_multi_source_dist(
            region["entry_cells"],
            allow_unknown=False,
        )
        remaining = self._region_remaining_targets(region)
        if remaining:
            self._coverage_target_dist_map = self._build_multi_source_dist(
                remaining,
                allow_unknown=False,
                allowed_mask=region["travel_mask"],
            )
        else:
            self._coverage_target_dist_map = None

    def _sync_region_inventory(self, force=False):
        if self.charge_mode or self._in_explore_phase():
            return
        if not force and self.regions and self.step_no - self.region_revision_step < Config.REGION_REBUILD_INTERVAL:
            return

        prev_region = self._current_region()
        prev_anchor = prev_region["anchor"] if prev_region is not None else None
        prev_kind = prev_region["kind"] if prev_region is not None else None

        self._build_real_regions()
        self.region_revision_step = self.step_no

        if prev_anchor is not None:
            self.current_region_id = self._match_region_by_anchor(prev_anchor, prev_kind)
        elif self.current_region_id not in self.regions:
            self.current_region_id = None

        if self.current_region_id is not None:
            region = self.regions.get(self.current_region_id)
            if region is None or not self._region_has_remaining_work(region):
                self.current_region_id = None

        self.region_seq = [
            rid
            for rid in self.region_seq
            if rid in self.regions and self._region_has_remaining_work(self.regions[rid]) and rid != self.current_region_id
        ]

        if self.current_region_id is not None:
            active = self.regions[self.current_region_id]
            self.region_lock_kind = active["kind"]
            self.region_lock_center = active["anchor"]
            self._refresh_active_region_maps(active)
        else:
            self._refresh_active_region_maps(None)

    def _estimate_position_to_region(self, start_pos, region):
        if region is None:
            return None

        allow_unknown = bool(region["kind"] == "frontier")
        best = None
        goal_cells = region["entry_cells"] if region["entry_cells"] else [region["anchor"]]
        for entry in goal_cells[: Config.REGION_ENTRY_CANDIDATES]:
            path = self._plan_path_to_goals(
                start_pos,
                [entry],
                allow_unknown=allow_unknown,
                mode="transit",
            )
            if path is None:
                continue
            transition_cost = self._path_transition_cost(start_pos, path, mode="transit")
            total_penalty, overlap_penalty, risk_penalty = self._path_overlap_penalty(path, transit_bias=1.0)
            score_cost = transition_cost + Config.PATH_OVERLAP_TRANSIT_WEIGHT * total_penalty
            if best is None or score_cost < best["cost"]:
                best = {
                    "entry": entry,
                    "path": path,
                    "steps": len(path),
                    "cost": float(score_cost),
                    "overlap": float(overlap_penalty),
                    "risk": float(risk_penalty),
                }

        if best is None and allow_unknown:
            path = self._plan_path_to_goals(start_pos, [region["anchor"]], allow_unknown=True, mode="transit")
            if path is not None:
                transition_cost = self._path_transition_cost(start_pos, path, mode="transit")
                total_penalty, overlap_penalty, risk_penalty = self._path_overlap_penalty(path, transit_bias=1.0)
                best = {
                    "entry": region["anchor"],
                    "path": path,
                    "steps": len(path),
                    "cost": float(transition_cost + Config.PATH_OVERLAP_TRANSIT_WEIGHT * total_penalty),
                    "overlap": float(overlap_penalty),
                    "risk": float(risk_penalty),
                }

        if best is None:
            return None

        best["return_budget"] = (
            float(best["steps"]) + region["coverage_estimate"] + region["charger_return"] + Config.TARGET_CHARGE_BUFFER
        )
        return best

    def _get_region_edge(self, src_region_id, dst_region_id, src_pos=None):
        if src_region_id is None:
            return self._estimate_position_to_region(src_pos if src_pos is not None else self.cur_pos, self.regions.get(dst_region_id))

        cache = self.region_graph.setdefault(src_region_id, {})
        if dst_region_id in cache:
            return cache[dst_region_id]

        src_region = self.regions.get(src_region_id)
        dst_region = self.regions.get(dst_region_id)
        if src_region is None or dst_region is None:
            return None

        edge = None
        start_candidates = src_region["entry_cells"] if src_region["entry_cells"] else [src_region["anchor"]]
        for start in start_candidates[: Config.REGION_ENTRY_CANDIDATES]:
            candidate = self._estimate_position_to_region(start, dst_region)
            if candidate is None:
                continue
            if edge is None or candidate["cost"] < edge["cost"]:
                edge = candidate
        if edge is not None:
            cache[dst_region_id] = edge
        return edge

    def _build_region_sequence(self):
        candidate_ids = [rid for rid, region in self.regions.items() if self._region_has_remaining_work(region)]
        if not candidate_ids:
            self.region_seq = []
            return

        seq = []
        used = set()
        cursor_region_id = self.current_region_id if self.current_region_id in candidate_ids else None
        cursor_pos = self.regions[cursor_region_id]["anchor"] if cursor_region_id is not None else self.cur_pos

        while len(seq) < Config.REGION_QUEUE_SIZE and len(used) < len(candidate_ids):
            best_rid = None
            best_score = -1e9
            for rid in candidate_ids:
                if rid == cursor_region_id or rid in used:
                    continue
                region = self.regions[rid]
                edge = self._get_region_edge(cursor_region_id, rid, src_pos=cursor_pos)
                if edge is None:
                    continue
                if edge["return_budget"] >= float(self.battery):
                    continue
                score = (
                    region["value"]
                    - Config.REGION_TRANSIT_COST_WEIGHT * edge["cost"]
                    - Config.REGION_TRANSIT_REPEAT_WEIGHT * edge["overlap"]
                    - Config.REGION_TRANSIT_RISK_WEIGHT * edge["risk"]
                    - 0.08 * region["coverage_estimate"]
                )
                if self.current_region_id is not None and rid == self.current_region_id:
                    score += 1.0
                if score > best_score:
                    best_score = score
                    best_rid = rid
            if best_rid is None:
                break
            seq.append(best_rid)
            used.add(best_rid)
            cursor_region_id = best_rid
            cursor_pos = self.regions[best_rid]["anchor"]

        self.region_seq = seq

    def _activate_region(self, region_id):
        region = self.regions.get(region_id)
        if region is None:
            self._clear_active_region()
            return None
        self.current_region_id = region_id
        self.region_lock_kind = region["kind"]
        self.region_lock_center = region["anchor"]
        self._refresh_active_region_maps(region)
        return region

    def _build_region_stripe_order(self, region, remaining_targets, start_pos):
        family = region.get("diag_family", "diff")
        stripes = {}
        for cell in remaining_targets:
            sid = self._diag_stripe_id(cell, family)
            stripes.setdefault(sid, []).append(cell)
        for sid, cells in stripes.items():
            stripes[sid] = sorted(cells, key=lambda cell: self._diag_progress_key(cell, family))

        if not stripes:
            return [], {}

        current_sid = self._diag_stripe_id(start_pos, family)
        desired_phase = int(region.get("diag_phase", self._cell_parity(start_pos)))
        if not any((sid & 1) == desired_phase for sid in stripes):
            nearest_sid = min(stripes.keys(), key=lambda sid: abs(int(sid) - int(current_sid)))
            desired_phase = int(nearest_sid & 1)
            region["diag_phase"] = desired_phase

        order = []
        remaining_ids = set(stripes.keys())
        cursor_sid = region.get("last_stripe_id")
        if cursor_sid is None:
            cursor_sid = current_sid
        phase = desired_phase

        while remaining_ids and len(order) < Config.COVERAGE_TARGET_BATCH:
            phase_ids = [sid for sid in remaining_ids if (sid & 1) == phase]
            if not phase_ids:
                phase ^= 1
                phase_ids = [sid for sid in remaining_ids if (sid & 1) == phase]
                if not phase_ids:
                    break
            best_sid = min(
                phase_ids,
                key=lambda sid: (abs(int(sid) - int(cursor_sid)), abs(int(sid) - int(current_sid))),
            )
            order.append(best_sid)
            remaining_ids.remove(best_sid)
            cursor_sid = best_sid
            phase ^= 1

        if not order:
            nearest_sid = min(stripes.keys(), key=lambda sid: abs(int(sid) - int(current_sid)))
            order.append(nearest_sid)
        return order, stripes

    def _order_region_stripe_cells(self, region, stripe_cells, current_pos):
        if not stripe_cells:
            return [], 1

        family = region.get("diag_family", "diff")
        ordered = sorted(stripe_cells, key=lambda cell: self._diag_progress_key(cell, family))
        forward = 1 if int(region.get("diag_forward", 1)) >= 0 else -1
        pref = ordered if forward > 0 else list(reversed(ordered))
        alt = list(reversed(pref))

        def entry_cost(cells):
            head = cells[0]
            return float(self._chebyshev(current_pos, head) + 0.12 * self._local_repeat_density(head, radius=1))

        if entry_cost(alt) + 0.25 < entry_cost(pref):
            return alt, -forward
        return pref, forward

    def _pick_phase_shift_step(self, region, current_pos, next_stripe_id):
        family = region.get("diag_family", "diff")
        current_sid = self._diag_stripe_id(current_pos, family)
        if int(current_sid) == int(next_stripe_id):
            return None

        direction = 1 if int(next_stripe_id) > int(current_sid) else -1
        if family == "sum":
            candidates = [(1, 0), (0, 1)] if direction > 0 else [(-1, 0), (0, -1)]
        else:
            candidates = [(1, 0), (0, -1)] if direction > 0 else [(-1, 0), (0, 1)]

        best_pos = None
        best_score = 1e9
        for dx, dz in candidates:
            nx = current_pos[0] + dx
            nz = current_pos[1] + dz
            if not self._in_bounds(nx, nz):
                continue
            if not region["travel_mask"][nx, nz]:
                continue
            if self._npc_zone_hard_block((nx, nz)) or self._npc_dynamic_hard_block((nx, nz)):
                continue
            score = (
                self._local_repeat_density((nx, nz), radius=1)
                + 0.08 * self._npc_zone_penalty((nx, nz))
                + 0.15 * abs(self._diag_stripe_id((nx, nz), family) - int(next_stripe_id))
            )
            if self._cell_parity((nx, nz)) == int(region.get("diag_phase", self._cell_parity((nx, nz)))):
                score -= 0.85
            if score < best_score:
                best_score = score
                best_pos = (nx, nz)

        if best_pos is None:
            return None
        return [best_pos]

    def _build_region_coverage_path(self, region):
        remaining_targets = self._region_remaining_targets(region)
        remaining = set(remaining_targets)
        if self.cur_pos in remaining:
            remaining.discard(self.cur_pos)
        if not remaining:
            self.coverage_targets = []
            return []

        stripe_order, stripe_map = self._build_region_stripe_order(region, list(remaining), self.cur_pos)
        if not stripe_order:
            self.coverage_targets = list(remaining)
            return []

        path = []
        current = self.cur_pos
        travel_mask = region["travel_mask"]
        last_sid = None

        for stripe_idx, sid in enumerate(stripe_order):
            stripe_cells = [cell for cell in stripe_map.get(sid, []) if cell in remaining]
            if not stripe_cells:
                continue

            ordered_cells, actual_forward = self._order_region_stripe_cells(region, stripe_cells, current)
            entry_segment = self._plan_path_to_goals(
                current,
                [ordered_cells[0]],
                allow_unknown=False,
                allowed_mask=travel_mask,
                mode="coverage",
            )
            if entry_segment is None:
                continue
            for cell in entry_segment:
                if not path or path[-1] != cell:
                    path.append(cell)
                remaining.discard(cell)
            current = ordered_cells[0]
            remaining.discard(current)

            family = region.get("diag_family", "diff")
            for cell in ordered_cells[1:]:
                if len(path) >= Config.MAX_TRACKED_PATH:
                    break
                segment = self._diag_direct_segment(current, cell, family, travel_mask)
                if segment is None:
                    segment = self._plan_path_to_goals(
                        current,
                        [cell],
                        allow_unknown=False,
                        allowed_mask=travel_mask,
                        mode="coverage",
                    )
                if segment is None:
                    continue
                for pos in segment:
                    if not path or path[-1] != pos:
                        path.append(pos)
                    remaining.discard(pos)
                current = cell
                remaining.discard(current)
                if len(path) >= Config.MAX_TRACKED_PATH:
                    break

            last_sid = sid
            region["diag_forward"] = -actual_forward
            if len(path) >= Config.MAX_TRACKED_PATH:
                break

            if stripe_idx + 1 < len(stripe_order):
                shift = self._pick_phase_shift_step(region, current, stripe_order[stripe_idx + 1])
                if shift is not None:
                    for pos in shift:
                        if not path or path[-1] != pos:
                            path.append(pos)
                        remaining.discard(pos)
                    current = shift[-1]
                    if len(path) >= Config.MAX_TRACKED_PATH:
                        break

        if last_sid is not None:
            region["last_stripe_id"] = int(last_sid)
            region["diag_phase"] = int(1 - (int(last_sid) & 1))
        self.coverage_targets = list(remaining)
        return path[: Config.MAX_TRACKED_PATH]

    def _plan_transit_to_region(self, region):
        edge = self._estimate_position_to_region(self.cur_pos, region)
        if edge is None:
            return False
        self.plan_mode = "transit"
        self.goal = edge["entry"]
        self.goal_kind = region["kind"]
        self.path = list(edge["path"][: Config.MAX_TRACKED_PATH])
        self.last_plan_step = self.step_no
        return True

    def _plan_coverage_for_region(self, region):
        self.plan_mode = "coverage"
        self.goal_kind = region["kind"]
        self.goal = region["anchor"]
        self._refresh_active_region_maps(region)
        path = self._build_region_coverage_path(region)
        if not path:
            self.path = []
            self.last_plan_step = self.step_no
            return False
        self.path = path[: Config.MAX_TRACKED_PATH]
        self.last_plan_step = self.step_no
        return True

    def _replan_fill(self):
        self._sync_region_inventory(force=not self.regions)
        attempts = 0
        max_attempts = max(2, len(self.regions) + 1)

        while attempts < max_attempts:
            region = self._current_region()
            if region is None or not self._region_has_remaining_work(region):
                self.current_region_id = None
                self._refresh_active_region_maps(None)
                if not self.region_seq:
                    self._build_region_sequence()
                next_region_id = None
                while self.region_seq:
                    rid = self.region_seq.pop(0)
                    if rid in self.regions and self._region_has_remaining_work(self.regions[rid]):
                        next_region_id = rid
                        break
                if next_region_id is None:
                    self._sync_region_inventory(force=True)
                    self._build_region_sequence()
                    while self.region_seq:
                        rid = self.region_seq.pop(0)
                        if rid in self.regions and self._region_has_remaining_work(self.regions[rid]):
                            next_region_id = rid
                            break
                if next_region_id is None:
                    self.plan_mode = "coverage"
                    self.goal = None
                    self.goal_kind = None
                    self.path = []
                    self.last_plan_step = self.step_no
                    return
                region = self._activate_region(next_region_id)

            if region is None:
                attempts += 1
                continue

            if self._inside_region(region):
                if self._plan_coverage_for_region(region):
                    return
                self.current_region_id = None
                attempts += 1
                continue

            if self._plan_transit_to_region(region):
                return

            self.current_region_id = None
            attempts += 1

        self.goal = None
        self.goal_kind = None
        self.path = []
        self.last_plan_step = self.step_no

    def _need_replan(self):
        if not self.path:
            return True
        if self.last_plan_step < Config.EXPLORE_PHASE_STEPS <= self.step_no:
            return True
        interval = Config.REPLAN_INTERVAL if self.charge_mode or self._in_explore_phase() else Config.FILL_REPLAN_INTERVAL
        if self.step_no - self.last_plan_step >= interval:
            return True
        if self.stuck_chain > 0:
            return True
        if self._path_is_npc_unsafe():
            return True

        if self._in_fill_phase() and not self.charge_mode:
            region = self._current_region()
            if region is None:
                return True
            if not self._region_has_remaining_work(region):
                return True
            if self.plan_mode == "transit" and self._inside_region(region):
                return True
            if self.plan_mode == "coverage" and not self._inside_region(region):
                return True
            return False

        if self.goal is None:
            return True
        if self.goal_kind == "dirty":
            gx, gz = self.goal
            if not self._in_bounds(gx, gz) or self.map_state[gx, gz] != self.DIRTY:
                return True
        if self.goal_kind == "charger" and self.on_charger:
            return True
        return False

    def _path_is_npc_unsafe(self):
        for pos in self.path[:6]:
            if self._npc_dynamic_hard_block(pos):
                return True
        return False

    def _clear_region_lock(self):
        self.region_lock_kind = None
        self.region_lock_center = None

    def _region_radius(self, kind):
        if kind == "dirty":
            return Config.REGION_LOCK_DIRTY_RADIUS
        if kind == "frontier":
            return Config.REGION_LOCK_FRONTIER_RADIUS
        return 0

    def _window_bounds(self, pos, radius):
        x, z = int(pos[0]), int(pos[1])
        x0 = max(0, x - radius)
        x1 = min(Config.GRID_SIZE, x + radius + 1)
        z0 = max(0, z - radius)
        z1 = min(Config.GRID_SIZE, z + radius + 1)
        return x0, x1, z0, z1

    def _region_mass(self, kind, center):
        if center is None:
            return 0.0

        radius = self._region_radius(kind)
        x0, x1, z0, z1 = self._window_bounds(center, radius)
        window = self.map_state[x0:x1, z0:z1]
        if window.size == 0:
            return 0.0

        dirty = float(np.sum(window == self.DIRTY))
        unknown = float(np.sum(window == self.UNKNOWN))
        if kind == "dirty":
            return dirty + 0.12 * unknown
        if kind == "frontier":
            return 0.75 * unknown + 0.35 * dirty
        return 0.0

    def _region_min_mass(self, kind):
        if kind == "dirty":
            return Config.REGION_LOCK_DIRTY_MIN_MASS
        if kind == "frontier":
            return Config.REGION_LOCK_FRONTIER_MIN_MASS
        return float("inf")

    def _set_region_lock(self, kind, center, mass):
        if kind not in {"dirty", "frontier"} or center is None:
            return
        if self.charge_mode or self._in_explore_phase():
            return
        if mass < self._region_min_mass(kind):
            return
        self.region_lock_kind = kind
        self.region_lock_center = (int(center[0]), int(center[1]))

    def _should_keep_region_lock(self):
        if self.charge_mode or self._in_explore_phase() or self.region_lock_center is None or self.region_lock_kind is None:
            return False

        dirty_mass = self._region_mass("dirty", self.region_lock_center)
        frontier_mass = self._region_mass("frontier", self.region_lock_center)
        return (
            dirty_mass >= Config.REGION_LOCK_DIRTY_MIN_MASS
            or frontier_mass >= Config.REGION_LOCK_FRONTIER_MIN_MASS
        )

    def _filter_candidates_by_region(self, candidates, kind, region_center, restrict_region):
        if region_center is None:
            return list(candidates)

        radius = self._region_radius(kind)
        filtered = [
            (int(gx), int(gz))
            for gx, gz in candidates
            if self._chebyshev((int(gx), int(gz)), region_center) <= radius
        ]
        if filtered:
            return filtered
        if restrict_region:
            return []
        return list(candidates)

    def _distance_to_region(self, pos, kind, center):
        if self.current_region_id is not None and self._active_region_dist_map is not None and self._in_bounds(pos[0], pos[1]):
            d = int(self._active_region_dist_map[pos[0], pos[1]])
            if d >= 0:
                return d
        if center is None or kind not in {"dirty", "frontier"}:
            return 0
        return max(0, self._chebyshev(pos, center) - self._region_radius(kind))

    def _pick_best_region_plan(self, dist, charger_dist):
        best_score = -1e9
        best_kind = None
        best_center = None

        dirty_cells = np.argwhere(self.map_state == self.DIRTY)
        dirty_candidates = self._cluster_cells(dirty_cells, Config.DIRTY_CLUSTER_RADIUS, self._dirty_density)
        for gx, gz in dirty_candidates[: Config.REGION_PLAN_CANDIDATES]:
            d = int(dist[gx, gz])
            if d <= 0:
                continue

            return_budget = d + self._charger_dist_at((gx, gz), charger_dist) + Config.TARGET_CHARGE_BUFFER
            if return_budget >= self.battery:
                continue

            mass = self._region_mass("dirty", (gx, gz))
            if mass < Config.REGION_LOCK_DIRTY_MIN_MASS:
                continue

            score = (
                Config.REGION_SCORE_DIRTY_WEIGHT * min(mass, 24.0)
                + 1.10 * self._dirty_density(gx, gz)
                + 0.35 * self._frontier_gain(gx, gz)
                - Config.REGION_SCORE_DISTANCE_WEIGHT * d
                - Config.REGION_SCORE_REPEAT_WEIGHT * self._local_repeat_density((gx, gz), radius=2)
                - Config.REGION_SCORE_RISK_WEIGHT * self._npc_zone_penalty((gx, gz))
            )
            if (
                self.region_lock_kind == "dirty"
                and self.region_lock_center is not None
                and self._chebyshev((gx, gz), self.region_lock_center) <= self._region_radius("dirty")
            ):
                score += Config.REGION_LOCK_STICKY_BONUS

            if score > best_score:
                best_score = score
                best_kind = "dirty"
                best_center = (int(gx), int(gz))

        frontier_candidates = self._cluster_cells(
            self._collect_frontiers(),
            Config.FRONTIER_CLUSTER_RADIUS,
            self._frontier_gain,
        )
        for gx, gz in frontier_candidates[: Config.REGION_PLAN_CANDIDATES]:
            d = int(dist[gx, gz])
            if d <= 0:
                continue

            return_budget = d + self._charger_dist_at((gx, gz), charger_dist) + Config.TARGET_CHARGE_BUFFER
            if return_budget >= self.battery:
                continue

            mass = self._region_mass("frontier", (gx, gz))
            if mass < Config.REGION_LOCK_FRONTIER_MIN_MASS:
                continue

            score = (
                Config.REGION_SCORE_FRONTIER_WEIGHT * min(mass, 28.0)
                + 0.80 * self._frontier_gain(gx, gz)
                + 0.25 * self._dirty_density(gx, gz)
                - Config.REGION_SCORE_DISTANCE_WEIGHT * d
                - Config.REGION_SCORE_REPEAT_WEIGHT * self._local_repeat_density((gx, gz), radius=2)
                - 1.05 * Config.REGION_SCORE_RISK_WEIGHT * self._npc_zone_penalty((gx, gz))
            )
            if (
                self.region_lock_kind == "frontier"
                and self.region_lock_center is not None
                and self._chebyshev((gx, gz), self.region_lock_center) <= self._region_radius("frontier")
            ):
                score += Config.REGION_LOCK_STICKY_BONUS

            if score > best_score:
                best_score = score
                best_kind = "frontier"
                best_center = (int(gx), int(gz))

        return best_kind, best_center

    def _replan(self):
        dist, prev_x, prev_z = self._build_bfs_tree(allow_unknown=True)
        self._charger_dist_map = self._build_multi_source_dist(self._charger_cells(), allow_unknown=True)

        goal_kind = None
        goal = None
        if self.charge_mode:
            self.plan_mode = "charge"
            goal_kind, goal = self._pick_best_charger_goal(dist)
        elif self._in_explore_phase():
            self.plan_mode = "explore"
            self._clear_region_lock()
            goal_kind, goal = self._pick_best_frontier_goal(dist, self._charger_dist_map)
            if goal is None:
                goal_kind, goal = self._pick_best_dirty_goal(dist, self._charger_dist_map)
        else:
            self._replan_fill()
            if self.goal is not None or self.path:
                return
            goal_kind, goal = self._pick_best_dirty_goal(dist, self._charger_dist_map)
            if goal is None:
                goal_kind, goal = self._pick_best_frontier_goal(dist, self._charger_dist_map)
            if goal is None and self.battery <= int(self.battery_max * 0.55):
                self._clear_region_lock()
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
        plannable = self._plannable_unknown_map if allow_unknown else self._plannable_known_map
        zone_block = self._npc_zone_hard_block_map
        dynamic_block = self._npc_dynamic_hard_block_map
        grid_size = Config.GRID_SIZE

        start_x, start_z = self.cur_pos
        queue = deque([(start_x, start_z)])
        dist[start_x, start_z] = 0

        while queue:
            x, z = queue.popleft()
            for dx, dz in self.ACTION_DELTAS:
                nx = x + dx
                nz = z + dz
                if nx < 0 or nx >= grid_size or nz < 0 or nz >= grid_size:
                    continue
                if dist[nx, nz] != -1:
                    continue
                if not plannable[nx, nz]:
                    continue
                if dx != 0 and dz != 0 and not (plannable[x + dx, z] or plannable[x, z + dz]):
                    continue
                if zone_block[nx, nz] or dynamic_block[nx, nz]:
                    continue

                dist[nx, nz] = dist[x, z] + 1
                prev_x[nx, nz] = x
                prev_z[nx, nz] = z
                queue.append((nx, nz))

        return dist, prev_x, prev_z

    def _build_multi_source_dist(self, starts, allow_unknown, allowed_mask=None):
        dist = np.full((Config.GRID_SIZE, Config.GRID_SIZE), -1, dtype=np.int16)
        queue = deque()
        plannable = self._plannable_unknown_map if allow_unknown else self._plannable_known_map
        zone_block = self._npc_zone_hard_block_map
        dynamic_block = self._npc_dynamic_hard_block_map
        grid_size = Config.GRID_SIZE
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
                if nx < 0 or nx >= grid_size or nz < 0 or nz >= grid_size:
                    continue
                if dist[nx, nz] != -1:
                    continue
                if not plannable[nx, nz]:
                    continue
                if allowed_mask is not None and not allowed_mask[nx, nz]:
                    continue
                if dx != 0 and dz != 0 and not (plannable[x + dx, z] or plannable[x, z + dz]):
                    continue
                if dx != 0 and dz != 0 and allowed_mask is not None and not (allowed_mask[x + dx, z] or allowed_mask[x, z + dz]):
                    continue
                if zone_block[nx, nz] or dynamic_block[nx, nz]:
                    continue
                dist[nx, nz] = dist[x, z] + 1
                queue.append((nx, nz))

        return dist

    def _is_global_move_passable(self, cur_pos, next_pos, allow_unknown, allowed_mask=None):
        nx, nz = next_pos
        plannable = self._plannable_unknown_map if allow_unknown else self._plannable_known_map
        if not self._in_bounds(nx, nz) or not plannable[nx, nz]:
            return False
        if allowed_mask is not None and not allowed_mask[nx, nz]:
            return False

        dx = nx - cur_pos[0]
        dz = nz - cur_pos[1]
        if dx != 0 and dz != 0:
            side_a = (cur_pos[0] + dx, cur_pos[1])
            side_b = (cur_pos[0], cur_pos[1] + dz)
            if not (plannable[side_a[0], side_a[1]] or plannable[side_b[0], side_b[1]]):
                return False
            if allowed_mask is not None and not (allowed_mask[side_a[0], side_a[1]] or allowed_mask[side_b[0], side_b[1]]):
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
        return self._plan_path_to_goals(self.cur_pos, [goal], allow_unknown=allow_unknown)

    def _plan_path_to_goals(self, start_pos, goals, allow_unknown, allowed_mask=None, mode=None):
        goal_list = [tuple(map(int, goal)) for goal in goals if goal is not None and self._in_bounds(int(goal[0]), int(goal[1]))]
        if not goal_list:
            return None
        goal_set = set(goal_list)
        if start_pos in goal_set:
            return []

        best_cost = np.full((Config.GRID_SIZE, Config.GRID_SIZE), np.inf, dtype=np.float32)
        prev_x = np.full((Config.GRID_SIZE, Config.GRID_SIZE), -1, dtype=np.int16)
        prev_z = np.full((Config.GRID_SIZE, Config.GRID_SIZE), -1, dtype=np.int16)

        sx, sz = int(start_pos[0]), int(start_pos[1])
        if not self._in_bounds(sx, sz):
            return None
        if allowed_mask is not None and not allowed_mask[sx, sz]:
            return None

        best_cost[sx, sz] = 0.0
        start_h = min(float(self._chebyshev(start_pos, goal)) for goal in goal_list)
        heap = [(start_h, 0.0, sx, sz)]
        reached_goal = None

        while heap:
            _, cur_cost, x, z = heapq.heappop(heap)
            if cur_cost > float(best_cost[x, z]) + 1e-6:
                continue
            if (x, z) in goal_set:
                reached_goal = (x, z)
                break

            for dx, dz in self.ACTION_DELTAS:
                nx = x + dx
                nz = z + dz
                if not self._in_bounds(nx, nz):
                    continue
                if not self._is_global_move_passable((x, z), (nx, nz), allow_unknown=allow_unknown, allowed_mask=allowed_mask):
                    continue
                if self._npc_zone_hard_block((nx, nz)) or self._npc_dynamic_hard_block((nx, nz)):
                    continue

                move_cost = self._transition_cost((x, z), (nx, nz), mode=mode)
                new_cost = cur_cost + move_cost
                if new_cost + 1e-6 >= float(best_cost[nx, nz]):
                    continue

                best_cost[nx, nz] = new_cost
                prev_x[nx, nz] = x
                prev_z[nx, nz] = z
                heuristic = min(float(self._chebyshev((nx, nz), goal)) for goal in goal_list)
                heapq.heappush(heap, (new_cost + heuristic, new_cost, nx, nz))

        if reached_goal is None:
            return None
        return self._reconstruct_path(reached_goal, prev_x, prev_z)

    def _transition_cost(self, cur_pos, next_pos, mode=None):
        if mode is None:
            mode = self.plan_mode
        nx, nz = next_pos
        cell = int(self.map_state[nx, nz]) if self._in_bounds(nx, nz) else self.OBSTACLE
        visit = float(self.visit_count[nx, nz]) if self._in_bounds(nx, nz) else float(Config.MAX_VISIT_CLIP)
        transit = (
            float(self.clean_pass_count[nx, nz]) if self._in_bounds(nx, nz) else float(Config.MAX_TRANSIT_CLIP)
        )
        charge_repeat_scale = Config.CHARGE_REPEAT_MULTIPLIER if self.goal_kind == "charger" or self.charge_mode else 1.0

        base = 1.0 + (0.05 if next_pos[0] != cur_pos[0] and next_pos[1] != cur_pos[1] else 0.0)
        recent_gap = self.step_no - int(self.last_visit_step[nx, nz]) if self._in_bounds(nx, nz) else 0
        recent_penalty = 0.0 if recent_gap > 20 else 0.08 * max(0, 20 - recent_gap)
        corridor_scale = self._corridor_repeat_discount(next_pos)
        repeat_penalty = 0.16 * charge_repeat_scale * corridor_scale * min(visit, float(Config.MAX_VISIT_CLIP))
        transit_penalty = 0.28 * charge_repeat_scale * corridor_scale * min(transit, float(Config.MAX_TRANSIT_CLIP))
        risk_penalty = 0.05 * min(self._npc_zone_penalty(next_pos), 60.0)
        interior_penalty = self._interior_clean_penalty(next_pos)
        snake_bias = self._serpentine_bias(cur_pos, next_pos)
        boundary_bonus = self._boundary_bonus(next_pos)
        repeat_decay = self._repeat_bias_decay(next_pos, radius=1)

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
        snake_scale = 1.0
        boundary_scale = 0.18 if self._in_explore_phase() else Config.FILL_BOUNDARY_WEIGHT_SCALE
        if self.goal_kind == "charger" or self.charge_mode:
            dirty_bonus *= slack
            frontier_bonus *= 0.65 * slack
            snake_scale = Config.CHARGE_SNAKE_SCALE * (0.35 + 0.65 * slack)
            boundary_scale *= 0.55 + 0.45 * slack

        if self._in_explore_phase():
            snake_scale = 0.0
        else:
            snake_scale *= Config.FILL_SNAKE_WEIGHT_SCALE * repeat_decay
            boundary_scale *= 0.65 + 0.35 * repeat_decay

        if mode == "transit":
            repeat_penalty *= 1.18
            transit_penalty *= 1.22
            boundary_scale *= 0.35
            snake_scale *= 0.18
            frontier_bonus *= 0.55
        elif mode == "coverage":
            dirty_bonus *= 1.10
            boundary_scale *= 0.75
            if self._current_region() is not None and not self._inside_region(self._current_region(), next_pos):
                interior_penalty += 1.0

        snake_penalty = 0.0
        if snake_bias >= 0.0:
            snake_penalty -= snake_scale * Config.SNAKE_PROGRESS_WEIGHT * snake_bias
        else:
            snake_penalty += snake_scale * Config.SNAKE_BACKTRACK_PENALTY * (-snake_bias)
        if next_pos[0] != cur_pos[0] and next_pos[1] != cur_pos[1]:
            snake_penalty += snake_scale * Config.SNAKE_DIAGONAL_PENALTY

        cost = (
            base
            + recent_penalty
            + repeat_penalty
            + transit_penalty
            + risk_penalty
            + interior_penalty
            + snake_penalty
            - Config.BOUNDARY_BONUS_WEIGHT * boundary_scale * boundary_bonus
        )
        cost -= dirty_bonus
        cost -= frontier_bonus
        if self._in_explore_phase() and not self.charge_mode:
            explore_bonus = Config.EXPLORE_FRONTIER_PATH_WEIGHT * min(float(self._frontier_gain(nx, nz)), 12.0)
            if cell == self.UNKNOWN:
                explore_bonus += 0.45 * Config.EXPLORE_UNKNOWN_CELL_BONUS
            if nx != cur_pos[0] and nz != cur_pos[1] and (cell == self.UNKNOWN or self._has_unknown_neighbor(next_pos)):
                explore_bonus += Config.EXPLORE_DIAGONAL_PATH_BONUS
            cost -= explore_bonus
        return max(0.12, float(cost))

    def _serpentine_index(self, pos):
        x, z = int(pos[0]), int(pos[1])
        if z % 2 == 0:
            return z * Config.GRID_SIZE + x
        return z * Config.GRID_SIZE + (Config.GRID_SIZE - 1 - x)

    def _row_forward_has_work(self, pos, lookahead=4):
        x, z = int(pos[0]), int(pos[1])
        row_dir = 1 if z % 2 == 0 else -1
        for step in range(1, lookahead + 1):
            nx = x + row_dir * step
            if not self._in_bounds(nx, z):
                break
            if self.map_state[nx, z] == self.OBSTACLE:
                break
            if self.map_state[nx, z] == self.DIRTY:
                return True
            if self._has_unknown_neighbor((nx, z)):
                return True
        return False

    def _serpentine_bias(self, cur_pos, next_pos):
        if self.goal_kind == "charger" or self.charge_mode:
            active = True
        else:
            active = self.goal_kind in {"dirty", "frontier"} or self._has_unknown_neighbor(cur_pos)
        if not active:
            return 0.0

        cur_idx = self._serpentine_index(cur_pos)
        next_idx = self._serpentine_index(next_pos)
        delta_idx = next_idx - cur_idx
        dx = next_pos[0] - cur_pos[0]
        dz = next_pos[1] - cur_pos[1]

        if delta_idx == 1:
            return 1.0
        if delta_idx == -1:
            return -1.0

        if abs(dz) == 1 and not self._row_forward_has_work(cur_pos):
            return 0.65
        if abs(dz) == 1 and self._row_forward_has_work(cur_pos):
            return -0.35
        if dx != 0 and dz == 0:
            return 0.2 if delta_idx > 0 else -0.55
        if dx != 0 and dz != 0:
            return -0.15 if delta_idx < 0 else 0.05
        return -0.2

    def _charge_slack(self):
        charger_dist = max(self._current_charger_distance(), 1)
        margin = self.battery - charger_dist - Config.RETURN_CHARGE_BUFFER
        return float(np.clip(margin / max(self.battery_max, 1), 0.0, 1.0))

    def _explore_action_bonus(self, cur_pos, next_pos):
        if self.charge_mode or not self._in_explore_phase() or not self._in_bounds(*next_pos):
            return 0.0

        nx, nz = next_pos
        cell = int(self.map_state[nx, nz])
        bonus = Config.EXPLORE_FRONTIER_ACTION_WEIGHT * min(float(self._frontier_gain(nx, nz)), 12.0) / 6.0
        if cell == self.UNKNOWN:
            bonus += Config.EXPLORE_UNKNOWN_CELL_BONUS
        elif cell == self.DIRTY:
            bonus += 0.45

        if nx != cur_pos[0] and nz != cur_pos[1] and (cell == self.UNKNOWN or self._has_unknown_neighbor(next_pos)):
            bonus += Config.EXPLORE_DIAGONAL_ACTION_BONUS

        bonus -= 0.10 * min(self._local_repeat_density(next_pos, radius=1), 10.0)
        return float(bonus)

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

    def _pick_best_dirty_goal(self, dist, charger_dist, region_center=None, restrict_region=False):
        dirty_cells = np.argwhere(self.map_state == self.DIRTY)
        if len(dirty_cells) == 0:
            return None, None

        candidates = self._cluster_cells(dirty_cells, Config.DIRTY_CLUSTER_RADIUS, self._dirty_density)
        candidates = self._filter_candidates_by_region(candidates, "dirty", region_center, restrict_region)
        if not candidates:
            return None, None

        best_score = -1e9
        best_goal = None
        best_region_anchor = None
        best_region_mass = 0.0
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
                - 0.45 * self._npc_zone_penalty((gx, gz))
            )
            region_anchor = region_center if region_center is not None else (gx, gz)
            region_mass = self._region_mass("dirty", region_anchor)
            score += Config.REGION_LOCK_VALUE_WEIGHT * min(region_mass, 18.0)

            if self.goal_kind == "dirty" and self.goal == (gx, gz):
                score += 1.5
            if (
                self.region_lock_kind == "dirty"
                and self.region_lock_center is not None
                and self._chebyshev((gx, gz), self.region_lock_center) <= self._region_radius("dirty")
            ):
                score += Config.REGION_LOCK_STICKY_BONUS

            if score > best_score:
                best_score = score
                best_goal = (int(gx), int(gz))
                best_region_anchor = region_anchor
                best_region_mass = region_mass

        if best_goal is None:
            return None, None
        self._set_region_lock("dirty", best_region_anchor, best_region_mass)
        return "dirty", best_goal

    def _pick_best_frontier_goal(self, dist, charger_dist, region_center=None, restrict_region=False):
        frontier_cells = self._collect_frontiers()
        if not frontier_cells:
            return None, None

        candidates = self._cluster_cells(frontier_cells, Config.FRONTIER_CLUSTER_RADIUS, self._frontier_gain)
        candidates = self._filter_candidates_by_region(candidates, "frontier", region_center, restrict_region)
        if not candidates:
            return None, None

        best_score = -1e9
        best_goal = None
        best_region_anchor = None
        best_region_mass = 0.0
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
                - 0.55 * self._npc_zone_penalty((gx, gz))
            )
            region_anchor = region_center if region_center is not None else (gx, gz)
            region_mass = self._region_mass("frontier", region_anchor)
            score += Config.REGION_LOCK_VALUE_WEIGHT * min(region_mass, 22.0)
            if self.goal_kind == "frontier" and self.goal == (gx, gz):
                score += 1.0
            if (
                self.region_lock_kind == "frontier"
                and self.region_lock_center is not None
                and self._chebyshev((gx, gz), self.region_lock_center) <= self._region_radius("frontier")
            ):
                score += Config.REGION_LOCK_STICKY_BONUS

            if score > best_score:
                best_score = score
                best_goal = (int(gx), int(gz))
                best_region_anchor = region_anchor
                best_region_mass = region_mass

        if best_goal is None:
            return None, None
        self._set_region_lock("frontier", best_region_anchor, best_region_mass)
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
                    corridor_penalty = 0.08 * float(self._local_repeat_density((gx, gz), radius=1))
                    cost = (
                        d
                        + 0.40 * float(self.visit_count[gx, gz])
                        + 0.90 * float(self.clean_pass_count[gx, gz])
                        + corridor_penalty
                    )
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
        return [tuple(map(int, pos)) for pos in np.argwhere(self._frontier_mask)]

    def _has_unknown_neighbor(self, pos):
        x, z = pos
        if not self._in_bounds(x, z):
            return False
        return bool(self._unknown_neighbor_mask[x, z])

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

    def _local_repeat_density(self, pos, radius=1):
        x, z = pos
        x0 = max(0, x - radius)
        x1 = min(Config.GRID_SIZE, x + radius + 1)
        z0 = max(0, z - radius)
        z1 = min(Config.GRID_SIZE, z + radius + 1)
        visit_sum = self._window_sum(self._visit_integral, x0, x1, z0, z1)
        transit_sum = self._window_sum(self._transit_integral, x0, x1, z0, z1)
        return visit_sum + 1.5 * transit_sum

    def _is_repeat_wall_cell(self, pos):
        x, z = pos
        if not self._in_bounds(x, z):
            return False
        if self.map_state[x, z] != self.CLEAN:
            return False
        return (
            int(self.visit_count[x, z]) >= Config.BOUNDARY_REPEAT_VISIT_THRESHOLD
            or int(self.clean_pass_count[x, z]) >= Config.BOUNDARY_REPEAT_TRANSIT_THRESHOLD
        )

    def _is_soft_wall(self, pos):
        x, z = pos
        if not self._in_bounds(x, z):
            return True
        if self.map_state[x, z] == self.OBSTACLE:
            return True
        if self._npc_zone_penalty(pos) >= Config.BOUNDARY_NPC_THRESHOLD:
            return True
        if self._is_repeat_wall_cell(pos):
            return True
        return False

    def _boundary_bonus(self, pos):
        x, z = pos
        if not self._in_bounds(x, z):
            return -1.0

        wall_neighbors = 0
        open_neighbors = 0
        value_neighbors = 0
        cardinal_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dz in cardinal_dirs:
            nx = x + dx
            nz = z + dz
            neighbor = (nx, nz)
            if self._is_soft_wall(neighbor):
                wall_neighbors += 1
                continue

            open_neighbors += 1
            if self._in_bounds(nx, nz):
                if self.map_state[nx, nz] == self.DIRTY or self.map_state[nx, nz] == self.UNKNOWN:
                    value_neighbors += 1
                elif self._has_unknown_neighbor(neighbor):
                    value_neighbors += 1
                elif not self._is_repeat_wall_cell(neighbor):
                    value_neighbors += 1

        bonus = 0.0
        if 1 <= wall_neighbors <= 2:
            bonus += 1.0 + 0.18 * wall_neighbors
        elif wall_neighbors == 3:
            bonus += 0.45
        elif wall_neighbors >= 4:
            bonus -= 0.55

        if value_neighbors >= 2:
            bonus += 0.35
        elif value_neighbors == 0:
            bonus -= 0.35

        if open_neighbors <= 1:
            bonus -= 0.25

        if self._is_repeat_wall_cell(pos):
            bonus -= 0.45
        if self._npc_zone_penalty(pos) >= Config.BOUNDARY_NPC_THRESHOLD:
            bonus -= 0.65

        return float(bonus)

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
        region_dist_now = self._distance_to_region(self.cur_pos, self.region_lock_kind, self.region_lock_center)
        active_region = self._current_region()
        target_dist_now = None
        if self._coverage_target_dist_map is not None and self._in_bounds(*self.cur_pos):
            cur_target_dist = int(self._coverage_target_dist_map[self.cur_pos[0], self.cur_pos[1]])
            if cur_target_dist >= 0:
                target_dist_now = cur_target_dist

        for act, (dx, dz) in enumerate(self.ACTION_DELTAS):
            if not legal_action[act]:
                continue

            nx = self.cur_pos[0] + dx
            nz = self.cur_pos[1] + dz
            next_pos = (nx, nz)

            score = -1.6 * self._transition_cost(self.cur_pos, next_pos)
            if self._in_bounds(nx, nz):
                local_gain_scale = 0.55 if self.plan_mode == "transit" and not self.charge_mode else 1.0
                if self.map_state[nx, nz] == self.DIRTY:
                    score += 8.0 * local_gain_scale
                score += 1.5 * local_gain_scale * self._frontier_gain(nx, nz)
                score -= 0.45 * self.visit_count[nx, nz]
                score -= 0.75 * self.clean_pass_count[nx, nz]
                score -= 0.08 * self._npc_zone_penalty(next_pos)
                repeat_decay = self._repeat_bias_decay(next_pos, radius=1)
                snake_scale = 0.0 if self._in_explore_phase() else Config.FILL_SNAKE_WEIGHT_SCALE * repeat_decay
                boundary_scale = 0.18 if self._in_explore_phase() else Config.FILL_BOUNDARY_WEIGHT_SCALE * (0.65 + 0.35 * repeat_decay)
                if self.charge_mode:
                    charge_scale = Config.CHARGE_SNAKE_SCALE * (0.35 + 0.65 * self._charge_slack())
                    snake_scale *= charge_scale
                    boundary_scale *= 0.55 + 0.45 * self._charge_slack()
                score += Config.SNAKE_ACTION_WEIGHT * snake_scale * self._serpentine_bias(self.cur_pos, next_pos)
                score += Config.BOUNDARY_ACTION_WEIGHT * boundary_scale * self._boundary_bonus(next_pos)
                score += self._explore_action_bonus(self.cur_pos, next_pos)
                if self.charge_exit_pending and self.on_charger:
                    score += self._charge_exit_action_bonus(next_pos)

                if active_region is not None and self._in_fill_phase() and not self.charge_mode:
                    in_region = bool(active_region["travel_mask"][nx, nz])
                    if self.plan_mode == "transit":
                        region_dist_next = self._distance_to_region(next_pos, self.region_lock_kind, self.region_lock_center)
                        score += Config.TRANSIT_REGION_ACTION_WEIGHT * (region_dist_now - region_dist_next)
                        if in_region:
                            score += 1.2
                    elif self.plan_mode == "coverage":
                        if not in_region:
                            score -= 2.8
                        if target_dist_now is not None and self._coverage_target_dist_map is not None:
                            next_target_dist = int(self._coverage_target_dist_map[nx, nz])
                            if next_target_dist >= 0:
                                score += Config.COVERAGE_REGION_ACTION_WEIGHT * (target_dist_now - next_target_dist)
                        score += self._coverage_diagonal_action_bonus(next_pos, active_region)

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
                region_dist_next = self._distance_to_region(next_pos, self.region_lock_kind, self.region_lock_center)
                score += Config.REGION_LOCK_ACTION_WEIGHT * (region_dist_now - region_dist_next)
                if region_dist_next == 0 and self.region_lock_kind in {"dirty", "frontier"}:
                    score += 0.25

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
        return int(self._window_sum(self._dirty_integral, x0, x1, z0, z1))

    def _unknown_density(self, pos, radius=2):
        x, z = pos
        x0 = max(0, x - radius)
        x1 = min(Config.GRID_SIZE, x + radius + 1)
        z0 = max(0, z - radius)
        z1 = min(Config.GRID_SIZE, z + radius + 1)
        return int(self._window_sum(self._unknown_integral, x0, x1, z0, z1))

    def _local_clean_density(self, pos, radius=2):
        x, z = pos
        x0 = max(0, x - radius)
        x1 = min(Config.GRID_SIZE, x + radius + 1)
        z0 = max(0, z - radius)
        z1 = min(Config.GRID_SIZE, z + radius + 1)
        return int(self._window_sum(self._clean_integral, x0, x1, z0, z1))

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
        x, z = int(pos[0]), int(pos[1])
        if not self._in_bounds(x, z):
            return True
        return bool(self._npc_zone_hard_block_map[x, z])

    def _npc_zone_penalty(self, pos):
        x, z = int(pos[0]), int(pos[1])
        if not self._in_bounds(x, z):
            return 1e9
        return float(self._npc_penalty_map[x, z])

    def _npc_field_value(self, pos, source, strength, soften):
        dx = float(pos[0] - source[0])
        dz = float(pos[1] - source[1])
        dist_sq = dx * dx + dz * dz
        return float(strength / (dist_sq + soften))

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

        if self._npc_zone_penalty(self.cur_pos) >= 35.0:
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
