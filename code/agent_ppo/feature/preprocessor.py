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

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.battery = 200
        self.battery_max = 200
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

        self.cur_visit_count = 0

        self.nearest_dirt_dist = 200.0
        self.last_nearest_dirt_dist = 200.0
        self.nearest_charger_dist = 200.0
        self.last_nearest_charger_dist = 200.0
        self.charger_slack = 0.0
        self.last_charger_slack = 0.0
        self.nearest_npc_dist = 200.0

        self.local_dirt_density = 0.0
        self.local_obstacle_density = 0.0
        self.explored_ratio = 0.0
        self.dirty_memory_ratio = 0.0

        self._view_map = np.ones((self.VIEW_SIZE, self.VIEW_SIZE), dtype=np.int8)
        self._legal_act = [1] * self.ACTION_DIM
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

        self.step_no = int(observation.get("step_no", 0))

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
        self.nearest_charger_dist, _, _ = self._nearest_charger_metrics()

        self.last_charger_slack = self.charger_slack
        reserve = 12.0
        self.charger_slack = float(self.battery - self.nearest_charger_dist - reserve)

        self.nearest_npc_dist, _, _ = self._nearest_npc_metrics()

        self.local_dirt_density = float(np.mean(self._view_map == 2))
        self.local_obstacle_density = float(np.mean(self._view_map == 0))
        self.explored_ratio = float(np.mean(self.explored_map))
        self.dirty_memory_ratio = float(np.mean(self.dirty_memory))

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

    def _get_scalar_feature(self, last_action):
        hx, hz = self.cur_pos
        charger_dist, charger_dx, charger_dz = self._nearest_charger_metrics()
        npc_dist, npc_dx, npc_dz = self._nearest_npc_metrics()

        dirt_delta = 1.0 if self.nearest_dirt_dist < self.last_nearest_dirt_dist else 0.0
        battery_ratio = _norm(self.battery, self.battery_max)
        low_battery_flag = 1.0 if (battery_ratio <= 0.35 or self.charger_slack <= 16.0) else 0.0
        charger_progress_flag = 1.0 if charger_dist < self.last_nearest_charger_dist else 0.0
        npc_risk_flag = float(np.clip((4.0 - npc_dist) / 4.0, 0.0, 1.0))
        revisit_ratio = float(np.clip((self.cur_visit_count - 1) / 6.0, 0.0, 1.0))

        scalar = np.array(
            [
                _norm(self.step_no, 2000),
                battery_ratio,
                _norm(self.dirt_cleaned, self.total_dirt),
                1.0 - _norm(self.dirt_cleaned, self.total_dirt),
                _norm(hx, self.GRID_SIZE - 1),
                _norm(hz, self.GRID_SIZE - 1),
                _norm(self.nearest_dirt_dist, self.VIEW_HALF),
                dirt_delta,
                self.local_dirt_density,
                self.local_obstacle_density,
                revisit_ratio,
                _norm(self.stuck_steps, 20),
                _norm(self.no_progress_steps, 80),
                _norm(self.charge_count, 50),
                self.just_charged,
                _norm(charger_dist, self.GRID_SIZE),
                _clip_signed(charger_dx, self.GRID_SIZE),
                _clip_signed(charger_dz, self.GRID_SIZE),
                _clip_signed(self.charger_slack, self.battery_max),
                low_battery_flag,
                charger_progress_flag,
                _norm(npc_dist, self.GRID_SIZE),
                _clip_signed(npc_dx, self.GRID_SIZE),
                _clip_signed(npc_dz, self.GRID_SIZE),
                npc_risk_flag,
                self.explored_ratio,
                self.dirty_memory_ratio,
                _norm(self.cleaned_this_step, 12),
            ],
            dtype=np.float32,
        )

        return np.concatenate([scalar, self._last_action_one_hot(last_action)], axis=0)

    def get_legal_action(self):
        return list(self._legal_act)

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
        # 清扫是核心奖励，权重最高
        cleaning_reward = 0.12 * float(self.cleaned_this_step)
        explore_reward = 0.001 * float(self.new_explored_cells)

        battery_ratio = self.battery / max(self.battery_max, 1)
        # 仅在电量极低(<20%)时施加充电压力
        low_battery_pressure = float(np.clip((0.20 - battery_ratio) / 0.20, 0.0, 1.0))
        charger_progress = float(np.clip(self.last_nearest_charger_dist - self.nearest_charger_dist, -6.0, 6.0))
        slack_improve = float(np.clip(self.charger_slack - self.last_charger_slack, -12.0, 12.0))
        # 充电奖励降到极低水平，仅作方向引导
        charger_reward = low_battery_pressure * (0.03 * charger_progress + 0.015 * slack_improve)
        # 充电完成奖励大幅降低: 3.0 → 0.3 (相当于2.5格清扫)
        charge_event_reward = 0.3 * self.just_charged

        npc_penalty = -0.06 * float(np.clip((3.0 - self.nearest_npc_dist) / 3.0, 0.0, 1.0))
        revisit_penalty = -0.01 * float(np.clip(self.cur_visit_count - 1, 0.0, 3.0))
        stuck_penalty = -0.01 * float(np.clip(self.stuck_steps / 4.0, 0.0, 1.0))
        idle_penalty = -0.012 * float(np.clip(self.no_progress_steps / 30.0, 0.0, 1.0))
        step_penalty = -0.0015

        reward = (
            cleaning_reward
            + explore_reward
            + charger_reward
            + charge_event_reward
            + npc_penalty
            + revisit_penalty
            + stuck_penalty
            + idle_penalty
            + step_penalty
        )
        return float(np.clip(reward, -1.5, 1.5))
