#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
LTSPPO feature preprocessor for Robot Vacuum.
"""

from __future__ import annotations

from collections import deque
import math

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


def _bearing(dx, dz):
    angle = math.atan2(float(dz), float(dx))
    return math.sin(angle), math.cos(angle)


class Preprocessor:
    GRID_SIZE = 128
    VIEW_SIZE = Config.LOCAL_VIEW_SIZE
    VIEW_HALF = VIEW_SIZE // 2
    COARSE_SIZE = Config.GLOBAL_MEMORY_SIZE
    COARSE_BLOCK = GRID_SIZE // COARSE_SIZE
    ACTION_DIM = Config.ACTION_NUM
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
    MODE_PREPARE_RETURN = 1
    MODE_RETURN = 2
    MODE_EVADE = 3
    MODE_NAME_TO_ID = {
        "clean": MODE_CLEAN,
        "prepare_return": MODE_PREPARE_RETURN,
        "return": MODE_RETURN,
        "evade": MODE_EVADE,
    }

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
        self.pre_charge_battery = 200

        self.cleaned_this_step = 0
        self.new_explored_cells = 0
        self.stuck_steps = 0
        self.consecutive_clean_steps = 0
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

        self.local_dirt_density = 0.0
        self.local_obstacle_density = 0.0
        self.local_frontier_density = 0.0
        self.explored_ratio = 0.0
        self.dirty_memory_ratio = 0.0
        self.actual_legal_ratio = 1.0
        self.wall_adjacent = 0
        self.dirty_adjacent = 0
        self.current_mode = self.MODE_CLEAN

        self.all_npc_info = []
        self.all_charger_info = []
        self.sorted_charger_candidates = []
        self.directional_dirty = np.zeros(self.ACTION_DIM, dtype=np.float32)

        self._view_map = np.ones((self.VIEW_SIZE, self.VIEW_SIZE), dtype=np.int8)
        self._legal_act = [1] * self.ACTION_DIM
        self._actual_legal_act = [1] * self.ACTION_DIM
        self._organs = []
        self._npcs = []

        self.explored_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.passable_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.dirty_memory = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.visit_count = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.npc_cleaned = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.charger_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)
        self.npc_risk_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.float32)

        self._cps_ema = 0.5
        self._last_action = -1
        self._recent_actions = deque(maxlen=Config.ACTION_HISTORY_WINDOW)
        self._trajectory = []
        self._astar_dist = float("inf")
        self._last_astar_dist = float("inf")
        self._guidance_cache_step = -1
        self._guidance_cache = None
        self._teacher_cache_step = -1
        self._teacher_cache = None
        self._teacher_target_history = deque(maxlen=Config.TEACHER_TARGET_STABILITY_WINDOW)

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

        self.pre_charge_battery = self.battery
        self.battery = int(hero.get("battery", env_info.get("remaining_charge", self.battery)))
        self.battery_max = max(int(hero.get("battery_max", env_info.get("battery_max", self.battery_max))), 1)

        self.last_score = self.score
        self.score = int(hero.get("score", env_info.get("clean_score", self.score)))
        self._guidance_cache_step = -1
        self._guidance_cache = None
        self._teacher_cache_step = -1
        self._teacher_cache = None

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
        self._refresh_static_maps()

        step_cleaned_cells = env_info.get("step_cleaned_cells") or []
        score_gain = max(0, self.score - self.last_score)
        dirt_gain = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        self.cleaned_this_step = max(len(step_cleaned_cells), score_gain, dirt_gain)

        if self.cleaned_this_step > 0:
            self.no_progress_steps = 0
            self.consecutive_clean_steps += 1
        else:
            self.no_progress_steps += 1
            self.consecutive_clean_steps = 0

        self.last_move_invalid = 0.0
        if last_action is not None and last_action >= 0:
            self._recent_actions.append(int(last_action))
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

        self._trajectory.append((hx, hz))
        if len(self._trajectory) > Config.TRAJECTORY_LENGTH:
            self._trajectory = self._trajectory[-Config.TRAJECTORY_LENGTH:]

        self.last_nearest_dirt_dist = self.nearest_dirt_dist
        self.nearest_dirt_dist = self._calc_nearest_dirt_dist()

        self.last_nearest_charger_dist = self.nearest_charger_dist
        self.nearest_charger_dist, self.nearest_charger_dx, self.nearest_charger_dz = self._nearest_charger_metrics()

        self._last_astar_dist = self._astar_dist
        self._astar_dist = self._compute_astar_charger_dist()
        self.last_charger_slack = self.charger_slack
        reserve = max(8.0, 0.04 * self.battery_max)
        base_dist = self._astar_dist if np.isfinite(self._astar_dist) else self.nearest_charger_dist
        self.charger_slack = float(self.battery - base_dist - reserve)

        self.nearest_npc_dist, self.nearest_npc_dx, self.nearest_npc_dz = self._nearest_npc_metrics()
        self.all_npc_info = self._collect_npc_info()
        self.all_charger_info = self._collect_charger_info()
        self.sorted_charger_candidates = self._sort_charger_candidates()
        self.directional_dirty = self._compute_directional_dirty()

        self.local_dirt_density = float(np.mean(self._view_map == 2))
        self.local_obstacle_density = float(np.mean(self._view_map == 0))
        self.local_frontier_density = self._calc_local_frontier_density()
        self.explored_ratio = float(np.mean(self.explored_map))
        self.dirty_memory_ratio = float(np.mean(self.dirty_memory))
        self._actual_legal_act = self._compute_actual_legal_actions()
        self.actual_legal_ratio = float(np.mean(self._actual_legal_act))
        self.wall_adjacent, self.dirty_adjacent = self._calc_adjacency()
        self.current_mode = self._infer_mode()

    def _refresh_static_maps(self):
        self.charger_map.fill(0.0)
        self.npc_risk_map.fill(0.0)

        for organ in self._organs:
            if int(organ.get("sub_type", 1)) != 1:
                continue
            pos = organ.get("pos") or {}
            cx, cz = int(pos.get("x", 0)), int(pos.get("z", 0))
            half_w = max(int(organ.get("w", 3)) // 2, 0)
            half_h = max(int(organ.get("h", 3)) // 2, 0)
            x0, x1 = max(cx - half_w, 0), min(cx + half_w + 1, self.GRID_SIZE)
            z0, z1 = max(cz - half_h, 0), min(cz + half_h + 1, self.GRID_SIZE)
            self.charger_map[x0:x1, z0:z1] = 1.0

        for npc in self._npcs:
            pos = npc.get("pos") or {}
            nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
            radius = 8
            for dx in range(-radius, radius + 1):
                for dz in range(-radius, radius + 1):
                    gx, gz = nx + dx, nz + dz
                    if not (0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE):
                        continue
                    dist = max(abs(dx), abs(dz))
                    if dist > radius:
                        continue
                    risk = max(0.0, 1.0 - dist / max(radius, 1))
                    self.npc_risk_map[gx, gz] = max(self.npc_risk_map[gx, gz], risk)

    def _update_memory(self, hx, hz):
        new_cells = 0
        half = self.VIEW_HALF
        for row in range(self.VIEW_SIZE):
            for col in range(self.VIEW_SIZE):
                gx = hx - half + col
                gz = hz - half + row
                if not (0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE):
                    continue
                if self.explored_map[gx, gz] == 0:
                    new_cells += 1
                self.explored_map[gx, gz] = 1.0
                cell = int(self._view_map[row, col])
                prev_dirty = self.dirty_memory[gx, gz]
                self.passable_map[gx, gz] = 1.0 if cell != 0 else 0.0
                if cell == 2:
                    self.dirty_memory[gx, gz] = 1.0
                elif cell == 1:
                    self.dirty_memory[gx, gz] = 0.0
                    if prev_dirty > 0.5 and not (gx == hx and gz == hz):
                        self.npc_cleaned[gx, gz] = 1.0
        return new_cells

    def _compute_astar_charger_dist(self):
        self.expert.update_chargers(self)
        path, dist, _ = self.expert._plan_to_charger_cached(self)
        return dist if path else self.nearest_charger_dist

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
            cx, cz = int(pos.get("x", 0)), int(pos.get("z", 0))
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

    def _collect_npc_info(self):
        hx, hz = self.cur_pos
        result = []
        for npc in self._npcs:
            pos = npc.get("pos") or {}
            nx, nz = int(pos.get("x", 0)), int(pos.get("z", 0))
            dx, dz = nx - hx, nz - hz
            dist = float(max(abs(dx), abs(dz)))
            result.append((dist, float(dx), float(dz), nx, nz))
        result.sort(key=lambda x: x[0])
        return result

    def _collect_charger_info(self):
        hx, hz = self.cur_pos
        result = []
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
            result.append((dist, float(cx - hx), float(cz - hz), cx, cz, half_w, half_h))
        result.sort(key=lambda x: x[0])
        return result

    def _sort_charger_candidates(self):
        hx, hz = self.cur_pos
        signal = self._get_guidance()
        target = signal.get("charger_target")
        candidates = []
        for idx, (_, dx, dz, cx, cz, half_w, half_h) in enumerate(self.all_charger_info):
            cheb = max(abs(dx), abs(dz))
            reachable = 1.0
            if target is not None and (cx, cz) == tuple(target):
                astar = float(signal.get("charger_dist", cheb))
                priority = 1.0
            else:
                astar = float(cheb)
                priority = 0.5
            in_local = 1.0 if abs(dx) <= self.VIEW_HALF and abs(dz) <= self.VIEW_HALF else 0.0
            candidates.append(
                {
                    "center": (cx, cz),
                    "dx": float(dx),
                    "dz": float(dz),
                    "dist": float(cheb),
                    "astar_dist": float(astar),
                    "reachable": reachable,
                    "priority": priority + 0.2 * in_local + (1.0 if idx == 0 else 0.0),
                }
            )
        candidates.sort(key=lambda item: (-item["reachable"], item["astar_dist"], item["dist"]))
        return candidates[: Config.CHARGER_SLOTS]

    def _get_guidance(self):
        if self._guidance_cache_step == self.step_no and self._guidance_cache is not None:
            return self._guidance_cache

        filtered_legal = self.expert.filter_actions(self, self.get_legal_action())
        guidance = self.expert.get_charger_signal(self, filtered_legal, self._last_action)
        target = guidance.get("charger_target")
        self._teacher_target_history.append(tuple(target) if target is not None else None)
        self._guidance_cache = guidance
        self._guidance_cache_step = self.step_no
        self._teacher_cache_step = -1
        self._teacher_cache = None
        return guidance

    def _get_teacher_guidance(self):
        if self._teacher_cache_step == self.step_no:
            return self._teacher_cache
        teacher = self.expert.get_teacher_guidance(
            self,
            self.get_legal_action(),
            self._last_action,
            signal=self._get_guidance(),
        )
        self._teacher_cache = teacher
        self._teacher_cache_step = self.step_no
        return teacher

    def is_teacher_target_stable(self, target):
        if target is None:
            return False
        target = tuple(target)
        history = list(self._teacher_target_history)
        window = Config.TEACHER_TARGET_STABILITY_WINDOW
        if len(history) < window:
            return False
        recent = history[-window:]
        return all(item == target for item in recent)

    def _cell_passable_local(self, dx, dz):
        row = self.VIEW_HALF + dz
        col = self.VIEW_HALF + dx
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

    def _calc_adjacency(self):
        c = self.VIEW_HALF
        wall = 0
        dirty = 0
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, col = c + dr, c + dc
            if 0 <= r < self.VIEW_SIZE and 0 <= col < self.VIEW_SIZE:
                cell = int(self._view_map[r, col])
                if cell == 0:
                    wall += 1
                elif cell == 2:
                    dirty += 1
        return wall, dirty

    def _calc_local_frontier_density(self):
        hx, hz = self.cur_pos
        passable = 0
        frontier = 0
        for row in range(self.VIEW_SIZE):
            for col in range(self.VIEW_SIZE):
                gx = hx - self.VIEW_HALF + col
                gz = hz - self.VIEW_HALF + row
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

    def _compute_directional_dirty(self):
        hx, hz = self.cur_pos
        result = np.zeros(self.ACTION_DIM, dtype=np.float32)
        for i, (ddx, ddz) in enumerate(self.ACTION_DELTAS):
            total = 0.0
            count = 0
            for r in range(1, 6):
                gx = hx + ddx * r
                gz = hz + ddz * r
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    total += float(self.dirty_memory[gx, gz])
                    count += 1
            result[i] = total / max(count, 1)
        return result

    def _build_local_channels(self):
        passable = (self._view_map != 0).astype(np.float32)
        dirt = (self._view_map == 2).astype(np.float32)
        charger = np.zeros((self.VIEW_SIZE, self.VIEW_SIZE), dtype=np.float32)
        npc_danger = np.zeros((self.VIEW_SIZE, self.VIEW_SIZE), dtype=np.float32)
        visit_heat = np.zeros((self.VIEW_SIZE, self.VIEW_SIZE), dtype=np.float32)
        trajectory_heat = np.zeros((self.VIEW_SIZE, self.VIEW_SIZE), dtype=np.float32)
        frontier = np.zeros((self.VIEW_SIZE, self.VIEW_SIZE), dtype=np.float32)
        return_guidance = np.zeros((self.VIEW_SIZE, self.VIEW_SIZE), dtype=np.float32)

        hx, hz = self.cur_pos
        half = self.VIEW_HALF

        for row in range(self.VIEW_SIZE):
            for col in range(self.VIEW_SIZE):
                gx = hx - half + col
                gz = hz - half + row
                if not (0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE):
                    continue
                charger[row, col] = self.charger_map[gx, gz]
                npc_danger[row, col] = self.npc_risk_map[gx, gz]
                visit_heat[row, col] = float(np.clip(self.visit_count[gx, gz] / 6.0, 0.0, 1.0))
                if passable[row, col] > 0.5:
                    for ndx, ndz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, nz = gx + ndx, gz + ndz
                        if not (0 <= nx < self.GRID_SIZE and 0 <= nz < self.GRID_SIZE) or self.explored_map[nx, nz] == 0:
                            frontier[row, col] = 1.0
                            break

        for age, (tx, tz) in enumerate(reversed(self._trajectory)):
            col = tx - (hx - half)
            row = tz - (hz - half)
            if 0 <= row < self.VIEW_SIZE and 0 <= col < self.VIEW_SIZE:
                val = max(0.0, 1.0 - age * Config.TRAJECTORY_DECAY)
                trajectory_heat[int(row), int(col)] = max(trajectory_heat[int(row), int(col)], val)

        guidance = self._get_guidance()
        path = guidance.get("charger_path") or []
        for age, (px, pz) in enumerate(path[1:8], start=1):
            col = px - (hx - half)
            row = pz - (hz - half)
            if 0 <= row < self.VIEW_SIZE and 0 <= col < self.VIEW_SIZE:
                return_guidance[int(row), int(col)] = max(return_guidance[int(row), int(col)], 1.0 - 0.12 * age)

        return np.stack(
            [passable, dirt, charger, npc_danger, visit_heat, trajectory_heat, frontier, return_guidance],
            axis=0,
        ).reshape(-1)

    def _pool_global_map(self, grid):
        reshaped = grid.reshape(self.COARSE_SIZE, self.COARSE_BLOCK, self.COARSE_SIZE, self.COARSE_BLOCK)
        return reshaped.mean(axis=(1, 3)).astype(np.float32)

    def _build_global_channels(self):
        explored = self._pool_global_map(self.explored_map)
        dirt = self._pool_global_map(self.dirty_memory)
        visit_density = self._pool_global_map(np.clip(self.visit_count / max(self.step_no + 1, 1), 0.0, 1.0))
        charger_presence = self._pool_global_map(self.charger_map)
        npc_risk_density = self._pool_global_map(self.npc_risk_map)
        if np.isfinite(self._astar_dist):
            return_cost_field = self._pool_global_map(
                np.clip(self.charger_map * (1.0 / max(self._astar_dist + 1.0, 1.0)), 0.0, 1.0)
            )
        else:
            return_cost_field = np.zeros_like(explored)
        return np.stack(
            [explored, dirt, visit_density, charger_presence, npc_risk_density, return_cost_field],
            axis=0,
        ).reshape(-1)

    def _build_entity_feature(self):
        features = []
        hx, hz = self.cur_pos

        for idx in range(Config.NPC_SLOTS):
            if idx < len(self.all_npc_info):
                dist, dx, dz, _, _ = self.all_npc_info[idx]
                sinv, cosv = _bearing(dx, dz)
                features.extend(
                    [
                        _clip_signed(dx, self.GRID_SIZE),
                        _clip_signed(dz, self.GRID_SIZE),
                        _norm(dist, self.GRID_SIZE),
                        sinv,
                        cosv,
                        1.0 if abs(dx) <= self.VIEW_HALF and abs(dz) <= self.VIEW_HALF else 0.0,
                        float(np.clip((8.0 - dist) / 8.0, 0.0, 1.0)),
                    ]
                )
            else:
                features.extend([0.0] * Config.ENTITY_FEATURE_DIM)

        for idx in range(Config.CHARGER_SLOTS):
            if idx < len(self.sorted_charger_candidates):
                cand = self.sorted_charger_candidates[idx]
                dx = cand["dx"]
                dz = cand["dz"]
                sinv, cosv = _bearing(dx, dz)
                features.extend(
                    [
                        _clip_signed(dx, self.GRID_SIZE),
                        _clip_signed(dz, self.GRID_SIZE),
                        _norm(cand["astar_dist"], self.GRID_SIZE),
                        sinv,
                        cosv,
                        1.0 if abs(dx) <= self.VIEW_HALF and abs(dz) <= self.VIEW_HALF else 0.0,
                        float(np.clip(cand["priority"], 0.0, 1.0)),
                    ]
                )
            else:
                features.extend([0.0] * Config.ENTITY_FEATURE_DIM)

        arr = np.asarray(features, dtype=np.float32)
        if arr.size != Config.ENTITY_DIM:
            raise ValueError(f"entity feature dim mismatch: {arr.size} != {Config.ENTITY_DIM}")
        return arr

    def _mode_onehot(self):
        vec = np.zeros(Config.MODE_NUM, dtype=np.float32)
        vec[int(np.clip(self.current_mode, 0, Config.MODE_NUM - 1))] = 1.0
        return vec

    def _build_scalar_feature(self, last_action):
        battery_ratio = _norm(self.battery, self.battery_max)
        clean_ratio = _norm(self.dirt_cleaned, self.total_dirt)
        guidance = self._get_guidance()
        slack = float(guidance.get("slack", self.charger_slack))
        min_npc_dist = float(guidance.get("min_npc_dist", self.nearest_npc_dist))

        base = [
            _norm(self.step_no, 2000),
            _norm(self.max_step, 2000),
            battery_ratio,
            _norm(self.battery_max, 999.0, 100.0),
            clean_ratio,
            1.0 - clean_ratio,
            _norm(self.cur_pos[0], self.GRID_SIZE - 1),
            _norm(self.cur_pos[1], self.GRID_SIZE - 1),
            _norm(self.nearest_dirt_dist, self.VIEW_HALF),
            self.local_dirt_density,
            self.local_obstacle_density,
            self.local_frontier_density,
            _norm(self.stuck_steps, 20),
            _norm(self.no_progress_steps, 80),
            self.invalid_move_ema,
            self.actual_legal_ratio,
            _norm(self.charge_count, 50),
            self.just_charged,
            _norm(self.nearest_charger_dist, self.GRID_SIZE),
            _clip_signed(self.nearest_charger_dx, self.GRID_SIZE),
            _clip_signed(self.nearest_charger_dz, self.GRID_SIZE),
            _clip_signed(slack, max(self.battery_max, 1)),
            float(np.clip((Config.PREPARE_RETURN_SLACK_THRESHOLD - slack) / max(Config.PREPARE_RETURN_SLACK_THRESHOLD, 1.0), 0.0, 1.0)),
            _norm(min_npc_dist, self.GRID_SIZE),
            _clip_signed(self.nearest_npc_dx, self.GRID_SIZE),
            _clip_signed(self.nearest_npc_dz, self.GRID_SIZE),
            float(np.clip((8.0 - min_npc_dist) / 8.0, 0.0, 1.0)),
            _norm(self.total_charger, 4, 1),
            _norm(self.npc_count, 4, 1),
            self.explored_ratio,
            self.dirty_memory_ratio,
            _norm(self.cleaned_this_step, 12),
            _norm(self.new_explored_cells, 12),
            _norm(self.cur_visit_count, 8),
            float(np.clip((self.cur_visit_count - 1) / 6.0, 0.0, 1.0)),
            float(guidance.get("reachable", False)),
            float(guidance.get("target_reliable", False)),
            float(guidance.get("mode_reliable", False)),
            float(guidance.get("on_charger", False)),
            _norm(float(guidance.get("charger_dist", self.nearest_charger_dist)), self.GRID_SIZE),
            _clip_signed(float(guidance.get("margin", 0.0)), 40.0),
            _clip_signed(float(self._astar_dist if np.isfinite(self._astar_dist) else self.nearest_charger_dist), self.GRID_SIZE),
            _clip_signed(float(self._last_astar_dist if np.isfinite(self._last_astar_dist) else self.last_nearest_charger_dist), self.GRID_SIZE),
            self.last_move_invalid,
            _norm(self.wall_adjacent, 4),
            _norm(self.dirty_adjacent, 4),
            self._cps_ema,
            float(last_action if last_action is not None and last_action >= 0 else -1) / max(Config.ACTION_NUM - 1, 1),
        ]
        base.extend(self._mode_onehot().tolist())
        base.extend(self.directional_dirty.tolist())

        for idx in range(Config.NPC_SLOTS):
            if idx < len(self.all_npc_info):
                dist, dx, dz, _, _ = self.all_npc_info[idx]
                base.extend([_norm(dist, self.GRID_SIZE), _clip_signed(dx, self.GRID_SIZE), _clip_signed(dz, self.GRID_SIZE)])
            else:
                base.extend([0.0, 0.0, 0.0])

        for idx in range(Config.CHARGER_SLOTS):
            if idx < len(self.sorted_charger_candidates):
                cand = self.sorted_charger_candidates[idx]
                base.extend(
                    [
                        _norm(cand["astar_dist"], self.GRID_SIZE),
                        _clip_signed(cand["dx"], self.GRID_SIZE),
                        _clip_signed(cand["dz"], self.GRID_SIZE),
                    ]
                )
            else:
                base.extend([0.0, 0.0, 0.0])

        arr = np.asarray(base, dtype=np.float32)
        if arr.size < Config.SCALAR_DIM:
            arr = np.concatenate([arr, np.zeros(Config.SCALAR_DIM - arr.size, dtype=np.float32)], axis=0)
        elif arr.size > Config.SCALAR_DIM:
            arr = arr[: Config.SCALAR_DIM]
        return arr

    def _build_action_history(self):
        last_action_onehot = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        if self._last_action is not None and 0 <= self._last_action < Config.ACTION_NUM:
            last_action_onehot[int(self._last_action)] = 1.0
        hist = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        if self._recent_actions:
            for action in self._recent_actions:
                if 0 <= action < Config.ACTION_NUM:
                    hist[action] += 1.0
            hist /= max(len(self._recent_actions), 1)
        return np.concatenate([last_action_onehot, hist], axis=0)

    def get_legal_action(self):
        merged = [int(a and b) for a, b in zip(self._legal_act, self._actual_legal_act)]
        if sum(merged) == 0:
            return list(self._actual_legal_act)
        return merged

    def _infer_mode(self):
        battery_ratio = self.battery / max(self.battery_max, 1)
        if self.nearest_npc_dist <= 3.0:
            return self.MODE_EVADE
        if self.charger_slack <= Config.RETURN_SLACK_THRESHOLD or battery_ratio <= Config.RETURN_BATTERY_RATIO:
            return self.MODE_RETURN
        if self.charger_slack <= Config.PREPARE_RETURN_SLACK_THRESHOLD or battery_ratio <= Config.PREPARE_RETURN_BATTERY_RATIO:
            return self.MODE_PREPARE_RETURN
        return self.MODE_CLEAN

    def feature_process(self, env_obs, last_action):
        self._last_action = last_action
        self.pb2struct(env_obs, last_action)

        local_map = self._build_local_channels()
        global_memory = self._build_global_channels()
        entity_state = self._build_entity_feature()
        scalar_state = self._build_scalar_feature(last_action)
        action_history = self._build_action_history()
        legal_action = self.expert.filter_actions(self, self.get_legal_action())

        feature = np.concatenate(
            [local_map, global_memory, entity_state, scalar_state, action_history],
            axis=0,
        ).astype(np.float32)
        if feature.size != Config.DIM_OF_OBSERVATION:
            raise ValueError(f"feature dim mismatch: {feature.size} != {Config.DIM_OF_OBSERVATION}")

        reward_total, reward_components = self.reward_process()
        return feature, legal_action, reward_total, reward_components

    def _target_teacher_from_guidance(self, guidance):
        target = guidance.get("target")
        if target is None:
            return 0
        target_tuple = tuple(target)
        for idx, cand in enumerate(self.sorted_charger_candidates, start=1):
            if tuple(cand["center"]) == target_tuple:
                return idx
        return 0

    def reward_process(self):
        clean_reward = 0.0
        survive_reward = 0.0

        cleaning_reward = 1.5 * float(self.cleaned_this_step)
        streak_bonus = 0.15 * min(float(self.cleaned_this_step > 0), 1.0) * min(self.consecutive_clean_steps, 5)
        explore_reward = (
            Config.EXPLORE_REWARD_SCALE
            * float(min(self.new_explored_cells, Config.EXPLORE_REWARD_CAP))
            * max(0.0, 1.0 - self.explored_ratio)
        )
        battery_ratio = self.battery / max(self.battery_max, 1)
        if battery_ratio < Config.FRONTIER_CRITICAL_BATTERY_RATIO:
            frontier_scale = 0.0
        elif battery_ratio < Config.FRONTIER_LOW_BATTERY_RATIO:
            frontier_scale = 0.5
        else:
            frontier_scale = 1.0
        frontier_reward = (
            Config.FRONTIER_REWARD_SCALE
            * frontier_scale
            * self.local_frontier_density
            * (0.5 + 0.5 * _norm(self.dirt_cleaned, self.total_dirt))
        )
        dirty_approach_reward = 0.0
        if self._last_action is not None and 0 <= self._last_action < Config.ACTION_NUM:
            dirty_approach_reward = 0.08 * float(self.directional_dirty[self._last_action])

        clean_reward += cleaning_reward + streak_bonus + explore_reward + frontier_reward + dirty_approach_reward

        guidance = self._get_guidance()
        slack = float(guidance.get("slack", self.charger_slack))
        charge_pressure = float(
            np.clip(
                (Config.PREPARE_RETURN_SLACK_THRESHOLD - slack) / max(Config.PREPARE_RETURN_SLACK_THRESHOLD, 1.0),
                0.0,
                1.0,
            )
        )
        delta_charger_slack = np.clip((slack - self.last_charger_slack) / max(self.battery_max, 1), -1.0, 1.0)
        charger_reward = 0.40 * charge_pressure * float(delta_charger_slack)
        charge_reward = 0.0
        if self.just_charged:
            charge_received = max(0.0, float(self.battery - self.pre_charge_battery + 1))
            efficiency = charge_received / max(self.battery_max, 1)
            charge_reward = 3.0 * efficiency

        npc_risk = float(np.clip((10.0 - self.nearest_npc_dist) / 10.0, 0.0, 1.0))
        npc_penalty = -3.0 * npc_risk ** 1.5
        stuck_penalty = -0.5 * self.last_move_invalid - 0.25 * _norm(self.stuck_steps, 10)
        idle_penalty = -0.1 * float(np.clip(self.no_progress_steps / 15.0, 0.0, 1.0))
        urgency_penalty = 0.0
        if not self.just_charged:
            if slack < -8:
                urgency_penalty = -1.2
            elif slack < 0:
                urgency_penalty = -0.6 * min(float(-slack) / 8.0, 1.0)
            elif slack < 5 and (self.battery / max(self.battery_max, 1)) < 0.20:
                urgency_penalty = -0.3

        astar_potential_reward = 0.0
        if battery_ratio < Config.ASTAR_POTENTIAL_BATTERY_THRESHOLD and np.isfinite(self._last_astar_dist):
            delta_dist = self._last_astar_dist - self._astar_dist
            astar_potential_reward = Config.ASTAR_POTENTIAL_ALPHA * float(delta_dist)

        survive_reward += charger_reward + charge_reward + npc_penalty + stuck_penalty + idle_penalty + urgency_penalty + astar_potential_reward

        if self.cleaned_this_step > 0:
            self._cps_ema = 0.95 * self._cps_ema + 0.05 * 1.0
        else:
            self._cps_ema = 0.95 * self._cps_ema + 0.05 * 0.0
        clean_reward += 0.3 * max(self._cps_ema - 0.75, 0.0)

        teacher = self._get_teacher_guidance()
        if teacher is None:
            mode_teacher = -1
            target_teacher = 0
            mode_teacher_mask = 0.0
            target_teacher_mask = 0.0
        else:
            mode_teacher = self.MODE_NAME_TO_ID.get(teacher.get("mode", "clean"), self.MODE_CLEAN)
            target_teacher = self._target_teacher_from_guidance(teacher)
            mode_teacher_mask = float(teacher.get("mode_teacher_mask", 0.0))
            target_teacher_mask = float(teacher.get("target_teacher_mask", 0.0))

        battery_risk_label = 1.0 if (battery_ratio <= 0.12 or slack < 0.0) else 0.0
        collision_risk_label = 1.0 if self.nearest_npc_dist <= 2.0 else 0.0

        reward_total = float(np.clip(clean_reward + survive_reward, -5.0, 5.0))
        components = {
            "reward_clean": float(np.clip(clean_reward, -5.0, 5.0)),
            "reward_survive": float(np.clip(survive_reward, -5.0, 5.0)),
            "reward_total": reward_total,
            "mode_teacher": int(mode_teacher),
            "target_teacher": int(target_teacher),
            "mode_teacher_mask": float(mode_teacher_mask),
            "target_teacher_mask": float(target_teacher_mask),
            "battery_risk_label": float(battery_risk_label),
            "collision_risk_label": float(collision_risk_label),
            "fallback_mask": 0.0,
            "expert_weight": 0.0,
            "cleaning": cleaning_reward,
            "streak": streak_bonus,
            "explore": explore_reward,
            "frontier": frontier_reward,
            "charger_approach": charger_reward,
            "charge": charge_reward,
            "npc": npc_penalty,
            "stuck": stuck_penalty,
            "idle": idle_penalty,
            "urgency": urgency_penalty,
            "astar_potential": astar_potential_reward,
        }
        return reward_total, components
