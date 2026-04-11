#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessor for Robot Vacuum.
"""

from __future__ import annotations

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.expert import ExpertPolicy


def _norm(value, v_max, v_min=0.0):
    value = float(np.clip(value, v_min, v_max))
    if v_max == v_min:
        return 0.0
    return (value - v_min) / (v_max - v_min)


def _clip_signed(value, scale):
    if scale <= 0:
        return 0.0
    return float(np.clip(value / scale, -1.0, 1.0))


class Preprocessor:
    GRID_SIZE = 128
    VIEW_SIZE = Config.LOCAL_VIEW_SIZE
    VIEW_HALF = VIEW_SIZE // 2
    COARSE_SIZE = Config.GLOBAL_MEMORY_SIZE
    COARSE_BLOCK = GRID_SIZE // COARSE_SIZE
    ACTION_DIM = Config.ACTION_NUM
    LAST_ACTION_DIM = ACTION_DIM + 1
    ACTION_DELTAS = (
        (1, 0),
        (1, -1),
        (0, -1),
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    )
    MODE_CLEAN = 0
    MODE_CHARGE = 1
    MODE_EVADE = 2

    def __init__(self):
        self.expert = ExpertPolicy()
        self.reset()

    def reset(self):
        self.step_no = 0
        self.max_step = 1000
        self.battery = 200
        self.battery_max = 200
        self.total_charger = 4
        self.npc_count = 4
        self.map_random = 0
        self.cur_pos = (0, 0)
        self.last_pos = None

        self.score = 0
        self.last_score = 0
        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        self.charge_count = 0
        self.last_charge_count = 0
        self.just_charged = 0.0

        self.cleaned_this_step = 0
        self.new_explored_cells = 0
        self.stuck_steps = 0
        self.no_progress_steps = 0
        self.invalid_move_count = 0
        self.last_move_invalid = 0.0
        self.invalid_move_ema = 0.0

        self.cur_visit_count = 0

        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0
        self.nearest_charger_dist = 200.0
        self.last_nearest_charger_dist = 200.0
        self.nearest_charger_dx = 0.0
        self.nearest_charger_dz = 0.0
        self.charger_slack = 0.0
        self.last_charger_slack = 0.0
        self.nearest_npc_dist = 200.0
        self.nearest_npc_dx = 0.0
        self.nearest_npc_dz = 0.0

        self.all_npc_info = []
        self.all_charger_info = []
        self.directional_dirty = np.zeros(self.ACTION_DIM, dtype=np.float32)

        self.local_dirt_density = 0.0
        self.local_obstacle_density = 0.0
        self.local_frontier_density = 0.0
        self.explored_ratio = 0.0
        self.dirty_memory_ratio = 0.0
        self.actual_legal_ratio = 1.0

        self.current_mode = self.MODE_CLEAN

        self._view_map = np.ones((self.VIEW_SIZE, self.VIEW_SIZE), dtype=np.int8)
        self._legal_act = [1] * self.ACTION_DIM
        self._actual_legal_act = [1] * self.ACTION_DIM
        self._organs = []
        self._npcs = []

        self.explored_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.passable_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.dirty_memory = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.visit_count = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)

    def pb2struct(self, env_obs, last_action):
        observation = env_obs.get("observation") or {}
        frame_state = observation.get("frame_state") or {}
        env_info = observation.get("env_info") or {}
        hero = frame_state.get("heroes") or {}

        self.step_no = int(observation.get("step_no", env_info.get("step_no", self.step_no)))
        self.max_step = max(int(env_info.get("max_step", self.max_step)), 1)
        self.total_charger = max(int(env_info.get("total_charger", self.total_charger)), 1)
        self.npc_count = max(int(env_info.get("npc_count", len(frame_state.get("npcs") or []))), 1)
        self.map_random = int(env_info.get("map_random", self.map_random))

        self.last_pos = self.cur_pos
        self.cur_pos = (
            int((hero.get("pos") or {}).get("x", 0)),
            int((hero.get("pos") or {}).get("z", 0)),
        )

        self.battery = int(hero.get("battery", env_info.get("remaining_charge", self.battery)))
        self.battery_max = max(int(hero.get("battery_max", env_info.get("battery_max", self.battery_max))), 1)

        self.last_score = self.score
        self.score = int(hero.get("score", env_info.get("clean_score", self.score)))

        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero.get("dirt_cleaned", self.dirt_cleaned))
        self.total_dirt = max(int(env_info.get("total_dirt", self.total_dirt)), 1)

        self.last_charge_count = self.charge_count
        self.charge_count = int(env_info.get("charge_count", self.charge_count))
        self.just_charged = 1.0 if self.charge_count > self.last_charge_count else 0.0

        legal_action = observation.get("legal_action")
        if legal_action is None:
            legal_action = observation.get("legal_act")
        self._legal_act = [int(x) for x in (legal_action or [1] * self.ACTION_DIM)]

        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.int8)
            self.new_explored_cells = self._update_memory(*self.cur_pos)
        else:
            self.new_explored_cells = 0

        self._organs = list(frame_state.get("organs") or [])
        self._npcs = list(frame_state.get("npcs") or [])

        step_cleaned_cells = env_info.get("step_cleaned_cells") or []
        score_gain = max(0, self.score - self.last_score)
        dirt_gain = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        self.cleaned_this_step = max(len(step_cleaned_cells), score_gain, dirt_gain)

        if self.cleaned_this_step > 0:
            self.no_progress_steps = 0
        else:
            self.no_progress_steps += 1

        self.last_move_invalid = 0.0
        if last_action is not None and last_action >= 0:
            if self.last_pos == self.cur_pos:
                self.invalid_move_count += 1
                self.last_move_invalid = 1.0
                self.invalid_move_ema = 0.8 * self.invalid_move_ema + 0.2
            else:
                self.invalid_move_ema *= 0.8

        if self.last_pos == self.cur_pos:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        hx, hz = self.cur_pos
        if 0 <= hx < self.GRID_SIZE and 0 <= hz < self.GRID_SIZE:
            self.visit_count[hx, hz] += 1.0
            self.cur_visit_count = int(self.visit_count[hx, hz])
        else:
            self.cur_visit_count = 0

        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()

        self.last_nearest_charger_dist = self.nearest_charger_dist
        (
            self.nearest_charger_dist,
            self.nearest_charger_dx,
            self.nearest_charger_dz,
        ) = self._nearest_charger_metrics()

        self.last_charger_slack = self.charger_slack
        reserve = max(8.0, 0.04 * self.battery_max)
        self.charger_slack = float(self.battery - self.nearest_charger_dist - reserve)

        (
            self.nearest_npc_dist,
            self.nearest_npc_dx,
            self.nearest_npc_dz,
        ) = self._nearest_npc_metrics()

        # All NPC info: (dist, dx, dz) sorted by distance
        hx, hz = self.cur_pos
        self.all_npc_info = []
        for npc in self._npcs:
            pos = npc.get("pos") or {}
            nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
            dx, dz = nx - hx, nz - hz
            dist = float(max(abs(dx), abs(dz)))
            self.all_npc_info.append((dist, float(dx), float(dz)))
        self.all_npc_info.sort(key=lambda x: x[0])

        # All charger info: (dist, dx, dz) sorted by distance
        self.all_charger_info = []
        for organ in self._organs:
            if int(organ.get("sub_type", 1)) != 1:
                continue
            pos = organ.get("pos") or {}
            cx, cz = int(pos.get("x", 0)), int(pos.get("z", 0))
            half_w = max(int(organ.get("w", 3)) // 2, 0)
            half_h = max(int(organ.get("h", 3)) // 2, 0)
            dist_x = max(abs(hx - cx) - half_w, 0)
            dist_z = max(abs(hz - cz) - half_h, 0)
            dist = float(max(dist_x, dist_z))
            self.all_charger_info.append((dist, float(cx - hx), float(cz - hz)))
        self.all_charger_info.sort(key=lambda x: x[0])

        self.directional_dirty = self._compute_directional_dirty()

        self.local_dirt_density = float(np.mean(self._view_map == 2))
        self.local_obstacle_density = float(np.mean(self._view_map == 0))
        self.local_frontier_density = self._calc_local_frontier_density()
        self.explored_ratio = float(np.mean(self.explored_map))
        self.dirty_memory_ratio = float(np.mean(self.dirty_memory))
        self._actual_legal_act = self._compute_actual_legal_actions()
        self.actual_legal_ratio = float(np.mean(self._actual_legal_act))
        self.current_mode = self._infer_mode()

    def _update_memory(self, hx, hz):
        new_cells = 0
        half = self.VIEW_HALF
        for row in range(self.VIEW_SIZE):
            for col in range(self.VIEW_SIZE):
                gx = hx - half + row
                gz = hz - half + col
                if not (0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE):
                    continue
                if self.explored_map[gx, gz] == 0:
                    new_cells += 1
                self.explored_map[gx, gz] = 1.0
                cell = int(self._view_map[row, col])
                self.passable_map[gx, gz] = 1.0 if cell != 0 else 0.0
                if cell == 2:
                    self.dirty_memory[gx, gz] = 1.0
                elif cell == 1:
                    self.dirty_memory[gx, gz] = 0.0
        return new_cells

    def _calc_nearest_dirt_dist(self):
        dirt_coords = np.argwhere(self._view_map == 2)
        if len(dirt_coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        chebyshev = np.max(np.abs(dirt_coords - center), axis=1)
        return float(np.min(chebyshev))

    def _nearest_charger_metrics(self):
        hx, hz = self.cur_pos
        best_dist = 200.0
        best_dx = 0.0
        best_dz = 0.0
        for organ in self._organs:
            if int(organ.get("sub_type", 1)) != 1:
                continue
            pos = organ.get("pos") or {}
            cx = int(pos.get("x", 0))
            cz = int(pos.get("z", 0))
            half_w = max(int(organ.get("w", 3)) // 2, 0)
            half_h = max(int(organ.get("h", 3)) // 2, 0)
            dist_x = max(abs(hx - cx) - half_w, 0)
            dist_z = max(abs(hz - cz) - half_h, 0)
            dist = float(max(dist_x, dist_z))
            if dist < best_dist:
                best_dist = dist
                best_dx = float(cx - hx)
                best_dz = float(cz - hz)
        return best_dist, best_dx, best_dz

    def _nearest_npc_metrics(self):
        hx, hz = self.cur_pos
        best_dist = 200.0
        best_dx = 0.0
        best_dz = 0.0
        for npc in self._npcs:
            pos = npc.get("pos") or {}
            nx = int(pos.get("x", 0))
            nz = int(pos.get("z", 0))
            dx = nx - hx
            dz = nz - hz
            dist = float(max(abs(dx), abs(dz)))
            if dist < best_dist:
                best_dist = dist
                best_dx = float(dx)
                best_dz = float(dz)
        return best_dist, best_dx, best_dz

    def _cell_passable_local(self, dx, dz):
        row = self.VIEW_HALF + dx
        col = self.VIEW_HALF + dz
        if not (0 <= row < self.VIEW_SIZE and 0 <= col < self.VIEW_SIZE):
            return False
        return int(self._view_map[row, col]) != 0

    def _compute_actual_legal_actions(self):
        actual = []
        for dx, dz in self.ACTION_DELTAS:
            legal = self._cell_passable_local(dx, dz)
            if legal and dx != 0 and dz != 0:
                legal = self._cell_passable_local(dx, 0) or self._cell_passable_local(0, dz)
            actual.append(int(legal))
        if sum(actual) == 0:
            return [1] * self.ACTION_DIM
        return actual

    def _calc_local_frontier_density(self):
        hx, hz = self.cur_pos
        passable = 0
        frontier = 0
        for row in range(self.VIEW_SIZE):
            for col in range(self.VIEW_SIZE):
                gx = hx - self.VIEW_HALF + row
                gz = hz - self.VIEW_HALF + col
                if not (0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE):
                    continue
                if int(self._view_map[row, col]) == 0:
                    continue
                passable += 1
                for ndx, ndz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx = gx + ndx
                    nz = gz + ndz
                    if not (0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE):
                        frontier += 1
                        break
                    if self.explored_map[nx, nz] == 0:
                        frontier += 1
                        break
        if passable == 0:
            return 0.0
        return float(frontier / passable)

    def _get_local_map_feature(self):
        obstacle = (self._view_map == 0).astype(np.float32)
        cleaned = (self._view_map == 1).astype(np.float32)
        dirt = (self._view_map == 2).astype(np.float32)
        return np.stack([obstacle, cleaned, dirt], axis=0).reshape(-1)

    def _get_global_memory_feature(self):
        explored = self._pool_global_map(self.explored_map)
        dirt = self._pool_global_map(self.dirty_memory)
        visit_heat = self._pool_global_map(np.clip(self.visit_count / max(self.step_no + 1, 1), 0.0, 1.0))
        return np.stack([explored, dirt, visit_heat], axis=0).reshape(-1)

    def _pool_global_map(self, grid):
        reshaped = grid.reshape(self.COARSE_SIZE, self.COARSE_BLOCK, self.COARSE_SIZE, self.COARSE_BLOCK)
        return reshaped.mean(axis=(1, 3)).astype(np.float32)

    def _last_action_one_hot(self, last_action):
        encoded = np.zeros(self.LAST_ACTION_DIM, dtype=np.float32)
        if last_action is None or last_action < 0 or last_action >= self.ACTION_DIM:
            encoded[-1] = 1.0
        else:
            encoded[int(last_action)] = 1.0
        return encoded

    def _infer_mode(self):
        battery_ratio = self.battery / max(self.battery_max, 1)
        if self.nearest_npc_dist <= 2.0:
            return self.MODE_EVADE
        if self.charger_slack <= 4.0 or battery_ratio <= 0.16:
            return self.MODE_CHARGE
        return self.MODE_CLEAN

    def _mode_flags(self):
        return (
            1.0 if self.current_mode == self.MODE_CLEAN else 0.0,
            1.0 if self.current_mode == self.MODE_CHARGE else 0.0,
            1.0 if self.current_mode == self.MODE_EVADE else 0.0,
        )

    def _get_scalar_feature(self, last_action):
        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0
        battery_ratio = _norm(self.battery, self.battery_max)
        step_ratio = _norm(self.step_no, self.max_step)
        low_battery_flag = 1.0 if (battery_ratio <= 0.18 or self.charger_slack <= 4.0) else 0.0
        charge_pressure = float(np.clip((8.0 - self.charger_slack) / 8.0, 0.0, 1.0))
        npc_risk_flag = float(np.clip((4.0 - self.nearest_npc_dist) / 4.0, 0.0, 1.0))
        revisit_ratio = float(np.clip((self.cur_visit_count - 1) / 6.0, 0.0, 1.0))
        mode_clean, mode_charge, mode_evade = self._mode_flags()

        scalar = np.array(
            [
                _norm(self.step_no, 2000),
                step_ratio,
                battery_ratio,
                _norm(self.battery_max, 999.0, 100.0),
                _norm(self.dirt_cleaned, self.total_dirt),
                1.0 - _norm(self.dirt_cleaned, self.total_dirt),
                _norm(self.cur_pos[0], self.GRID_SIZE - 1),
                _norm(self.cur_pos[1], self.GRID_SIZE - 1),
                _norm(self.nearest_dirt_dist, self.VIEW_HALF),
                dirt_delta,
                self.local_dirt_density,
                self.local_obstacle_density,
                self.local_frontier_density,
                revisit_ratio,
                _norm(self.stuck_steps, 20),
                _norm(self.no_progress_steps, 80),
                self.invalid_move_ema,
                self.actual_legal_ratio,
                _norm(self.charge_count, 50),
                self.just_charged,
                _norm(self.nearest_charger_dist, self.GRID_SIZE),
                _clip_signed(self.nearest_charger_dx, self.GRID_SIZE),
                _clip_signed(self.nearest_charger_dz, self.GRID_SIZE),
                _clip_signed(self.charger_slack, max(self.battery_max, 1)),
                low_battery_flag,
                charge_pressure,
                _norm(self.nearest_npc_dist, self.GRID_SIZE),
                _clip_signed(self.nearest_npc_dx, self.GRID_SIZE),
                _clip_signed(self.nearest_npc_dz, self.GRID_SIZE),
                npc_risk_flag,
                _norm(self.total_charger, 4, 1),
                _norm(self.npc_count, 4, 1),
                _norm(self.max_step, 2000),
                self.explored_ratio,
                self.dirty_memory_ratio,
                _norm(self.cleaned_this_step, 12),
                mode_clean,
                mode_charge,
                mode_evade,
            ],
            dtype=np.float32,
        )

        # All 4 NPC slots (padding missing with far-away sentinel)
        PAD = (200.0, 0.0, 0.0)
        npc1 = self.all_npc_info[0] if len(self.all_npc_info) > 0 else PAD
        npc2 = self.all_npc_info[1] if len(self.all_npc_info) > 1 else PAD
        npc3 = self.all_npc_info[2] if len(self.all_npc_info) > 2 else PAD
        npc4 = self.all_npc_info[3] if len(self.all_npc_info) > 3 else PAD
        # All 4 charger slots
        ch1 = self.all_charger_info[0] if len(self.all_charger_info) > 0 else PAD
        ch2 = self.all_charger_info[1] if len(self.all_charger_info) > 1 else PAD
        ch3 = self.all_charger_info[2] if len(self.all_charger_info) > 2 else PAD
        ch4 = self.all_charger_info[3] if len(self.all_charger_info) > 3 else PAD

        extra = np.array(
            [
                _norm(npc2[0], self.GRID_SIZE),
                _clip_signed(npc2[1], self.GRID_SIZE),
                _clip_signed(npc2[2], self.GRID_SIZE),
                _norm(npc3[0], self.GRID_SIZE),
                _clip_signed(npc3[1], self.GRID_SIZE),
                _clip_signed(npc3[2], self.GRID_SIZE),
                _norm(npc4[0], self.GRID_SIZE),
                _clip_signed(npc4[1], self.GRID_SIZE),
                _clip_signed(npc4[2], self.GRID_SIZE),
                _norm(ch2[0], self.GRID_SIZE),
                _clip_signed(ch2[1], self.GRID_SIZE),
                _clip_signed(ch2[2], self.GRID_SIZE),
                _norm(ch3[0], self.GRID_SIZE),
                _clip_signed(ch3[1], self.GRID_SIZE),
                _clip_signed(ch3[2], self.GRID_SIZE),
                _norm(ch4[0], self.GRID_SIZE),
                _clip_signed(ch4[1], self.GRID_SIZE),
                _clip_signed(ch4[2], self.GRID_SIZE),
                *self.directional_dirty,
            ],
            dtype=np.float32,
        )

        return np.concatenate([scalar, extra, self._last_action_one_hot(last_action)], axis=0)

    def get_legal_action(self):
        merged = [int(a and b) for a, b in zip(self._legal_act, self._actual_legal_act)]
        if sum(merged) == 0:
            return list(self._actual_legal_act)
        return merged

    def _local_dirty_vector(self):
        dirt_coords = np.argwhere(self._view_map == 2)
        if len(dirt_coords) == 0:
            return None
        center = self.VIEW_HALF
        best = min(
            dirt_coords,
            key=lambda rc: (max(abs(int(rc[0]) - center), abs(int(rc[1]) - center)), abs(int(rc[0]) - center) + abs(int(rc[1]) - center)),
        )
        return float(int(best[0]) - center), float(int(best[1]) - center)

    def _alignment_scores(self, target_dx, target_dz):
        vector_norm = float(np.hypot(target_dx, target_dz))
        if vector_norm <= 1e-6:
            return np.zeros(self.ACTION_DIM, dtype=np.float32)
        scores = []
        for dx, dz in self.ACTION_DELTAS:
            denom = max(np.hypot(dx, dz), 1.0)
            scores.append((dx * target_dx + dz * target_dz) / (denom * vector_norm))
        return np.array(scores, dtype=np.float32)

    def _patch_mean(self, grid, cx, cz, radius=1):
        x0 = max(cx - radius, 0)
        x1 = min(cx + radius + 1, self.GRID_SIZE)
        z0 = max(cz - radius, 0)
        z1 = min(cz + radius + 1, self.GRID_SIZE)
        if x0 >= x1 or z0 >= z1:
            return 0.0
        patch = grid[x0:x1, z0:z1]
        if patch.size == 0:
            return 0.0
        return float(np.mean(patch))

    def _compute_directional_dirty(self):
        """Compute directional dirty density for each of 8 action directions."""
        hx, hz = self.cur_pos
        DIRS = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
        result = np.zeros(self.ACTION_DIM, dtype=np.float32)
        for i, (ddx, ddz) in enumerate(DIRS):
            total = 0.0
            count = 0
            for r in range(1, 6):
                gx = hx + ddx * r
                gz = hz + ddz * r
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    total += float(self.dirty_memory[gz, gx])
                    count += 1
            result[i] = total / max(count, 1)
        return result

    def _directional_patch_scores(self, grid, radii, weights, patch_radius=1):
        hx, hz = self.cur_pos
        scores = np.zeros(self.ACTION_DIM, dtype=np.float32)
        for idx, (dx, dz) in enumerate(self.ACTION_DELTAS):
            score = 0.0
            for radius, weight in zip(radii, weights):
                tx = int(np.clip(hx + dx * radius, 0, self.GRID_SIZE - 1))
                tz = int(np.clip(hz + dz * radius, 0, self.GRID_SIZE - 1))
                score += weight * self._patch_mean(grid, tx, tz, radius=patch_radius)
            scores[idx] = score
        return scores

    def get_action_biases(self):
        dirty_scores = self._directional_patch_scores(self.dirty_memory, radii=(3, 6, 10, 15), weights=(1.5, 1.2, 0.8, 0.4))
        frontier_scores = self._directional_patch_scores(
            1.0 - self.explored_map,
            radii=(5, 10, 15),
            weights=(0.8, 0.5, 0.3),
        )
        visit_scores = self._directional_patch_scores(
            np.clip(self.visit_count / max(self.step_no + 1, 1), 0.0, 1.0),
            radii=(3, 6, 9),
            weights=(0.9, 0.5, 0.2),
        )

        biases = 1.0 * dirty_scores + 0.4 * frontier_scores - 0.4 * visit_scores

        local_dirty_vec = self._local_dirty_vector()
        if local_dirty_vec is not None:
            biases += 1.5 * self._alignment_scores(*local_dirty_vec)

        charge_align = self._alignment_scores(self.nearest_charger_dx, self.nearest_charger_dz)
        npc_align = self._alignment_scores(self.nearest_npc_dx, self.nearest_npc_dz)
        evade_align = self._alignment_scores(-self.nearest_npc_dx, -self.nearest_npc_dz)

        npc_pressure = float(np.clip((8.0 - self.nearest_npc_dist) / 8.0, 0.0, 1.0))
        charge_pressure = float(np.clip((10.0 - self.charger_slack) / 10.0, 0.0, 1.0))

        if self.current_mode == self.MODE_CHARGE:
            dirt_in_path = self._directional_patch_scores(self.dirty_memory, radii=(3, 6), weights=(1.0, 0.6))
            biases = 1.5 * charge_align + 0.8 * dirt_in_path - 0.4 * visit_scores
        elif self.current_mode == self.MODE_EVADE:
            biases = 2.0 * evade_align + 0.4 * charge_pressure * charge_align - 0.2 * visit_scores
        else:
            biases += 0.45 * charge_pressure * charge_align

        biases -= 1.6 * npc_pressure * np.maximum(npc_align, 0.0)
        return biases.astype(np.float32)

    def feature_process(self, env_obs, last_action):
        self.pb2struct(env_obs, last_action)

        local_map = self._get_local_map_feature()
        global_memory = self._get_global_memory_feature()
        scalar_state = self._get_scalar_feature(last_action)
        legal_action = self.get_legal_action()
        legal_arr = np.array(legal_action, dtype=np.float32)

        feature = np.concatenate([local_map, global_memory, scalar_state, legal_arr], axis=0).astype(np.float32)
        reward = self.reward_process()
        return feature, legal_action, reward

    def reward_process(self):
        cleaning_reward = 2.0 * float(self.cleaned_this_step)
        explore_reward = 0.02 * float(min(self.new_explored_cells, 4))

        charge_pressure = float(np.clip((8.0 - self.charger_slack) / 8.0, 0.0, 1.0))
        delta_charger_slack = np.clip(
            (self.charger_slack - self.last_charger_slack) / max(self.battery_max, 1),
            -1.0,
            1.0,
        )
        charger_reward = 0.03 * charge_pressure * float(delta_charger_slack)

        npc_risk = float(np.clip((4.0 - self.nearest_npc_dist) / 4.0, 0.0, 1.0))
        npc_penalty = -0.04 * npc_risk
        revisit_penalty = -0.06 * float(np.clip(self.cur_visit_count - 1, 0.0, 3.0))
        stuck_penalty = -0.2 * self.last_move_invalid
        idle_penalty = -0.03 * float(np.clip(self.no_progress_steps / 30.0, 0.0, 1.0))

        reward = (
            cleaning_reward
            + explore_reward
            + charger_reward
            + npc_penalty
            + revisit_penalty
            + stuck_penalty
            + idle_penalty
        )
        return float(np.clip(reward, -4.0, 6.0))
