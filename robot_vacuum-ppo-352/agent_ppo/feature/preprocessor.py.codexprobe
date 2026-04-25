#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
清扫大作战特征预处理器。
"""

import math

import numpy as np


def _clip_norm(v, vmax):
    if vmax <= 0:
        return 0.0
    return float(np.clip(float(v) / float(vmax), 0.0, 1.0))


def _get_pos(obj):
    pos = obj.get("pos", {}) if isinstance(obj, dict) else {}
    return float(pos.get("x", 0.0)), float(pos.get("z", 0.0))


class Preprocessor:
    """Feature preprocessor for Robot Vacuum PPO.

    488D feature layout:
    - 441: map_info flatten(21x21), normalized by /2.0
    -   5: battery_ratio, step_ratio, pos_x, pos_z, clean_ratio
    -   4: charger distances
    -  12: npc distances (4) + npc direction sin/cos pairs (8)
    -   8: last action one-hot
    -  10: behavior stats (coverage/repeat/charging/stuck/etc.)
    -   8: action priority scores (charge > evade > coverage)
    """

    GRID_SIZE = 128
    ACTION_NUM = 8
    MAX_ENTITY = 4
    MAX_DIST = 180.0
    FEATURE_DIM = 488
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

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.battery = 0
        self.battery_max = 1
        self.cur_pos = (0.0, 0.0)
        self.total_dirt = 1
        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self._legal_act = [1] * self.ACTION_NUM
        self._map_info = np.zeros((21, 21), dtype=np.float32)
        self._organs = []
        self._npcs = []
        self._terminated = False
        self._truncated = False
        self._fail_reason = ""
        self._step_cleaned = 0

        # Mild behavior tracking for reward shaping only (not in features)
        self._visit_counter = {}
        self._recent_positions = []
        self._history_size = 8
        self._was_stuck = False
        self._stuck_cnt = 0
        self._prev_min_charger_dist = None
        self._prev_min_npc_dist = None
        self._prev_battery_ratio = 1.0
        self._cur_battery_ratio = 1.0
        self._charge_success_cnt = 0
        self._charge_attempt_cnt = 0
        self._low_battery_trigger_cnt = 0
        self._low_battery_active = False
        self._last_charge_step = -1000
        self._repeat_visit_steps = 0
        self._near_npc_steps = 0

    def _extract_observation(self, env_obs):
        if not isinstance(env_obs, dict):
            return {}, False, False, ""

        if isinstance(env_obs.get("observation"), dict):
            obs = env_obs.get("observation", {})
        else:
            obs = env_obs

        terminated = bool(env_obs.get("terminated", False))
        truncated = bool(env_obs.get("truncated", False))
        fail_reason = str(env_obs.get("fail_reason", ""))

        if not fail_reason:
            fail_reason = str(obs.get("fail_reason", ""))
        if not fail_reason:
            fail_reason = str(obs.get("env_info", {}).get("fail_reason", ""))

        return obs, terminated, truncated, fail_reason

    def _update_movement_memory(self):
        cell = (int(round(self.cur_pos[0])), int(round(self.cur_pos[1])))
        visit_count = self._visit_counter.get(cell, 0) + 1
        self._visit_counter[cell] = visit_count
        if visit_count > 1:
            self._repeat_visit_steps += 1
        self._recent_positions.append(cell)
        if len(self._recent_positions) > self._history_size:
            self._recent_positions.pop(0)

    def _is_stuck(self):
        if len(self._recent_positions) < self._history_size:
            return False
        return len(set(self._recent_positions)) <= 2

    def pb2struct(self, env_obs, _last_action):
        obs, terminated, truncated, fail_reason = self._extract_observation(env_obs)
        frame_state = obs.get("frame_state", {})
        env_info = obs.get("env_info", {})
        hero = frame_state.get("heroes", {})

        self._terminated = terminated
        self._truncated = truncated
        self._fail_reason = fail_reason.lower()

        self.step_no = int(obs.get("step_no", 0))
        self.cur_pos = _get_pos(hero)

        self.battery = int(hero.get("battery", 0))
        self.battery_max = max(int(hero.get("battery_max", 1)), 1)
        self._cur_battery_ratio = _clip_norm(self.battery, self.battery_max)

        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero.get("dirt_cleaned", hero.get("score", 0)))
        self.total_dirt = max(int(env_info.get("total_dirt", 1)), 1)

        legal = obs.get("legal_action")
        if legal is None:
            legal = obs.get("legal_act")
        if isinstance(legal, (list, tuple)) and len(legal) >= self.ACTION_NUM:
            self._legal_act = [int(x) for x in list(legal)[: self.ACTION_NUM]]
        else:
            self._legal_act = [1] * self.ACTION_NUM

        map_info = obs.get("map_info")
        if map_info is not None:
            arr = np.array(map_info, dtype=np.float32)
            if arr.ndim == 2:
                out = np.zeros((21, 21), dtype=np.float32)
                h = min(arr.shape[0], 21)
                w = min(arr.shape[1], 21)
                out[:h, :w] = arr[:h, :w]
                self._map_info = out
            else:
                self._map_info = np.zeros((21, 21), dtype=np.float32)
        else:
            self._map_info = np.zeros((21, 21), dtype=np.float32)

        self._organs = frame_state.get("organs", []) or []
        self._npcs = frame_state.get("npcs", []) or []

        step_cleaned_cells = env_info.get("step_cleaned_cells", [])
        if isinstance(step_cleaned_cells, (list, tuple)):
            self._step_cleaned = len(step_cleaned_cells)
        else:
            self._step_cleaned = max(0, self.dirt_cleaned - self.last_dirt_cleaned)

        self._update_movement_memory()

    def _map_feature(self):
        return (self._map_info.flatten() / 2.0).astype(np.float32)

    def _global_feature(self):
        hx, hz = self.cur_pos
        battery_ratio = _clip_norm(self.battery, self.battery_max)
        step_ratio = _clip_norm(self.step_no, 1000.0)
        pos_x = _clip_norm(hx, self.GRID_SIZE)
        pos_z = _clip_norm(hz, self.GRID_SIZE)
        clean_ratio = _clip_norm(self.dirt_cleaned, self.total_dirt)
        return np.array([battery_ratio, step_ratio, pos_x, pos_z, clean_ratio], dtype=np.float32)

    def _entity_features(self):
        hx, hz = self.cur_pos

        chargers = []
        for organ in self._organs:
            ox, oz = _get_pos(organ)
            dist = math.sqrt((ox - hx) ** 2 + (oz - hz) ** 2)
            chargers.append(_clip_norm(dist, self.MAX_DIST))
        chargers = sorted(chargers)[: self.MAX_ENTITY]
        while len(chargers) < self.MAX_ENTITY:
            chargers.append(0.0)

        npc_items = []
        for npc in self._npcs:
            nx, nz = _get_pos(npc)
            dx = nx - hx
            dz = nz - hz
            dist = math.sqrt(dx**2 + dz**2)
            angle = math.atan2(dz, dx) if dist > 1e-6 else 0.0
            npc_items.append((_clip_norm(dist, self.MAX_DIST), math.sin(angle), math.cos(angle)))
        npc_items = sorted(npc_items, key=lambda x: x[0])[: self.MAX_ENTITY]

        npc_dists = [it[0] for it in npc_items]
        npc_dirs = []
        for _, sin_v, cos_v in npc_items:
            npc_dirs.extend([sin_v, cos_v])

        while len(npc_dists) < self.MAX_ENTITY:
            npc_dists.append(0.0)
        while len(npc_dirs) < self.MAX_ENTITY * 2:
            npc_dirs.append(0.0)

        return (
            np.array(chargers, dtype=np.float32),
            np.array(npc_dists, dtype=np.float32),
            np.array(npc_dirs, dtype=np.float32),
        )

    def _last_action_feature(self, last_action):
        onehot = np.zeros(self.ACTION_NUM, dtype=np.float32)
        if isinstance(last_action, int) and 0 <= last_action < self.ACTION_NUM:
            onehot[last_action] = 1.0
        return onehot

    def _behavior_feature(self):
        unique_visit_ratio = _clip_norm(len(self._visit_counter), max(self.total_dirt, 1))
        repeat_visit_ratio = _clip_norm(self._repeat_visit_steps, max(self.step_no, 1))
        min_charger_dist = self._min_charger_distance()
        min_npc_dist = self._min_npc_distance()
        battery_ratio = self._cur_battery_ratio
        low_battery_flag = 1.0 if battery_ratio < 0.40 else 0.0
        is_stuck = 1.0 if self._is_stuck() else 0.0
        charge_progress = 0.0
        if self._prev_min_charger_dist is not None and min_charger_dist < 1e8:
            charge_progress = float(np.clip((self._prev_min_charger_dist - min_charger_dist) / 20.0, -1.0, 1.0))
        battery_gain = float(np.clip(self._cur_battery_ratio - self._prev_battery_ratio, -1.0, 1.0))

        return np.array(
            [
                unique_visit_ratio,
                repeat_visit_ratio,
                _clip_norm(min_charger_dist, self.MAX_DIST),
                _clip_norm(min_npc_dist, self.MAX_DIST),
                low_battery_flag,
                is_stuck,
                _clip_norm(self._charge_success_cnt, 5.0),
                _clip_norm(self._stuck_cnt, 20.0),
                charge_progress,
                battery_gain,
            ],
            dtype=np.float32,
        )

    def get_legal_action(self):
        return list(self._legal_act)

    def feature_process(self, env_obs, last_action):
        self.pb2struct(env_obs, last_action)

        map_flat = self._map_feature()
        global_state = self._global_feature()
        charger_dist, npc_dist, npc_dir = self._entity_features()
        last_act_onehot = self._last_action_feature(last_action)
        behavior_state = self._behavior_feature()
        action_priority = self._action_priority_feature()

        feature = np.concatenate(
            [map_flat, global_state, charger_dist, npc_dist, npc_dir, last_act_onehot, behavior_state, action_priority],
            axis=0,
        ).astype(np.float32)

        if feature.shape[0] != self.FEATURE_DIM:
            fixed = np.zeros(self.FEATURE_DIM, dtype=np.float32)
            n = min(feature.shape[0], self.FEATURE_DIM)
            fixed[:n] = feature[:n]
            feature = fixed

        feature = np.nan_to_num(feature, nan=0.0, posinf=1.0, neginf=0.0)
        legal_action = self.get_legal_action()
        reward = self.reward_process()
        return feature, legal_action, reward

    def _min_charger_distance(self):
        hx, hz = self.cur_pos
        if not self._organs:
            return 1e9
        dists = []
        for organ in self._organs:
            ox, oz = _get_pos(organ)
            dists.append(math.sqrt((ox - hx) ** 2 + (oz - hz) ** 2))
        return float(min(dists)) if dists else 1e9

    def _nearest_charger_vector(self):
        hx, hz = self.cur_pos
        best_dx, best_dz = 0.0, 0.0
        best_dist = 1e9
        for organ in self._organs:
            ox, oz = _get_pos(organ)
            dx = ox - hx
            dz = oz - hz
            dist = math.sqrt(dx**2 + dz**2)
            if dist < best_dist:
                best_dx, best_dz, best_dist = dx, dz, dist
        return float(best_dx), float(best_dz), float(best_dist)

    def _min_npc_distance(self):
        hx, hz = self.cur_pos
        if not self._npcs:
            return 1e9
        dists = []
        for npc in self._npcs:
            nx, nz = _get_pos(npc)
            dists.append(math.sqrt((nx - hx) ** 2 + (nz - hz) ** 2))
        return float(min(dists)) if dists else 1e9

    def _nearest_npc_vector(self):
        hx, hz = self.cur_pos
        best_dx, best_dz = 0.0, 0.0
        best_dist = 1e9
        for npc in self._npcs:
            nx, nz = _get_pos(npc)
            dx = nx - hx
            dz = nz - hz
            dist = math.sqrt(dx**2 + dz**2)
            if dist < best_dist:
                best_dx, best_dz, best_dist = dx, dz, dist
        return float(best_dx), float(best_dz), float(best_dist)

    def _direction_scores(self, dx, dz, prefer_away=False):
        scores = np.zeros(self.ACTION_NUM, dtype=np.float32)
        norm = math.sqrt(dx**2 + dz**2)
        if norm < 1e-6:
            return scores

        tx = dx / norm
        tz = dz / norm
        if prefer_away:
            tx = -tx
            tz = -tz

        for i, (ax, az) in enumerate(self.ACTION_DELTAS):
            anorm = math.sqrt(ax**2 + az**2)
            dot = (ax / anorm) * tx + (az / anorm) * tz
            scores[i] = max(0.0, 0.5 * (dot + 1.0))
        return scores

    def _directional_coverage_scores(self):
        center = 10
        scores = np.zeros(self.ACTION_NUM, dtype=np.float32)
        for z in range(21):
            for x in range(21):
                val = int(self._map_info[z, x]) if self._map_info.ndim == 2 else 0
                if val != 2:
                    continue
                dx = x - center
                dz = z - center
                if dx == 0 and dz == 0:
                    continue
                dist = math.sqrt(dx**2 + dz**2)
                if dist < 1e-6:
                    continue
                for i, (ax, az) in enumerate(self.ACTION_DELTAS):
                    anorm = math.sqrt(ax**2 + az**2)
                    dot = (ax / anorm) * (dx / dist) + (az / anorm) * (dz / dist)
                    if dot > 0:
                        scores[i] += dot / (dist + 1.0)

        max_score = float(np.max(scores))
        if max_score > 1e-6:
            scores = scores / max_score
        return scores.astype(np.float32)

    def _action_priority_feature(self):
        battery_ratio = self._cur_battery_ratio
        charger_dx, charger_dz, charger_dist = self._nearest_charger_vector()
        npc_dx, npc_dz, npc_dist = self._nearest_npc_vector()

        charger_weight = float(np.clip((0.60 - battery_ratio) / 0.35, 0.0, 1.0))
        if charger_dist >= 1e8:
            charger_weight = 0.0

        evade_weight = float(np.clip((10.0 - npc_dist) / 7.0, 0.0, 1.0))
        coverage_weight = max(0.0, 0.25 * (1.0 - charger_weight) * (1.0 - 0.7 * evade_weight))

        charger_scores = self._direction_scores(charger_dx, charger_dz, prefer_away=False)
        evade_scores = self._direction_scores(npc_dx, npc_dz, prefer_away=True)
        coverage_scores = self._directional_coverage_scores()

        scores = (
            1.4 * charger_weight * charger_scores
            + 1.0 * evade_weight * evade_scores
            + coverage_weight * coverage_scores
        )
        scores = scores * np.array(self._legal_act, dtype=np.float32)

        max_score = float(np.max(scores))
        if max_score > 1e-6:
            scores = scores / max_score
        return scores.astype(np.float32)

    def reward_process(self):
        reward = 0.0

        # 1) Main cleaning reward
        cleaned_this_step = max(0, int(self._step_cleaned))
        reward += 1.05 * cleaned_this_step

        # 2) Time penalty to discourage inefficient wandering
        reward -= 0.015

        # 3) Coverage shaping (updated priority: P3)
        cur_cell = (int(round(self.cur_pos[0])), int(round(self.cur_pos[1])))
        visit_count = self._visit_counter.get(cur_cell, 1)
        if visit_count == 1:
            reward += 0.10  # Increased from 0.07
        elif visit_count > 6:
            reward -= 0.08 * min(visit_count - 6, 8)  # Increased penalty from 0.03

        unique_visit_ratio = len(self._visit_counter) / float(max(self.total_dirt, 1))
        reward += 0.03 * float(np.clip(unique_visit_ratio, 0.0, 1.0))

        # 4) Low-battery charging guidance (HIGHEST PRIORITY: P1)
        battery_ratio = self._cur_battery_ratio
        min_charger_dist = self._min_charger_distance()
        # Increased threshold from 0.45 to 0.55 for earlier charging motivation
        if battery_ratio < 0.55 and not self._low_battery_active:
            self._low_battery_trigger_cnt += 1
            self._low_battery_active = True
        elif battery_ratio >= 0.65:  # Increased hysteresis from 0.60 to 0.65
            self._low_battery_active = False

        if battery_ratio < 0.55:  # Updated from 0.45
            if min_charger_dist < 18.0:
                self._charge_attempt_cnt += 1
            if self._prev_min_charger_dist is not None:
                dist_delta = self._prev_min_charger_dist - min_charger_dist
                reward += 0.85 * float(np.clip(dist_delta / 6.0, -1.0, 1.0))  # Increased from 0.80
                if battery_ratio < 0.25 and dist_delta < 0:
                    reward -= 0.50 * float(np.clip(abs(dist_delta) / 6.0, 0.0, 1.0))  # Increased from 0.40
            if min_charger_dist < 12.0:
                reward += 0.60 * ((12.0 - min_charger_dist) / 12.0)
            if min_charger_dist > 20.0:
                reward -= 0.35 * float(np.clip((min_charger_dist - 20.0) / 30.0, 0.0, 1.0))

        battery_gain = self._cur_battery_ratio - self._prev_battery_ratio
        if battery_gain > 0.08 and (self.step_no - self._last_charge_step) > 10:
            self._charge_success_cnt += 1
            self._last_charge_step = self.step_no
            reward += 5.0  # Increased from 4.0

        # 5) NPC safety penalty (SECOND PRIORITY: P2)
        min_npc_dist = self._min_npc_distance()
        if min_npc_dist < 12.0:
            self._near_npc_steps += 1
        if min_npc_dist < 10.0:
            reward -= 1.50 * ((10.0 - min_npc_dist) / 10.0)  # Increased from 1.00
            if self._prev_min_npc_dist is not None:
                npc_escape_delta = min_npc_dist - self._prev_min_npc_dist
                reward += 0.60 * float(np.clip(npc_escape_delta / 4.0, -1.0, 1.0))  # Increased from 0.35
        
        # Extra penalty for very close NPC distance
        if min_npc_dist < 6.0:
            reward -= 2.00 * ((6.0 - min_npc_dist) / 6.0)

        # 6) Anti-stuck shaping
        is_stuck = self._is_stuck()
        if is_stuck:
            reward -= 0.18
            if not self._was_stuck:
                self._stuck_cnt += 1
        if self._was_stuck and not is_stuck:
            reward += 0.15
        self._was_stuck = is_stuck

        # 7) Terminal penalty
        if self._terminated and not self._truncated:
            if "collision" in self._fail_reason:
                reward -= 24.0
            elif "battery" in self._fail_reason:
                reward -= 22.0
            else:
                reward -= 10.0

        self._prev_min_charger_dist = min_charger_dist
        self._prev_min_npc_dist = min_npc_dist
        self._prev_battery_ratio = self._cur_battery_ratio
        return float(reward)

    def get_episode_metrics(self):
        step_denom = float(max(self.step_no, 1))
        charge_trigger_denom = float(max(self._low_battery_trigger_cnt, 1))
        return {
            "clean_ratio": round(_clip_norm(self.dirt_cleaned, self.total_dirt), 4),
            "coverage_rate": round(_clip_norm(len(self._visit_counter), max(self.total_dirt, 1)), 4),
            "repeat_visit_ratio": round(_clip_norm(self._repeat_visit_steps, step_denom), 4),
            "stuck_cnt": float(self._stuck_cnt),
            "charge_attempt_cnt": float(self._charge_attempt_cnt),
            "charge_success_cnt": float(self._charge_success_cnt),
            "charge_success_rate": round(
                float(self._charge_success_cnt) / charge_trigger_denom,
                4,
            ),
            "low_battery_trigger_cnt": float(self._low_battery_trigger_cnt),
            "near_npc_steps": float(self._near_npc_steps),
            "battery_ratio": round(float(self._cur_battery_ratio), 4),
        }
