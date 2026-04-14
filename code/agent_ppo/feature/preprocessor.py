#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
"""

import numpy as np


def _norm(v, v_max, v_min=0.0):
    """Linear normalize to [0, 1] with clipping."""
    v = float(np.clip(v, v_min, v_max))
    if v_max <= v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """
    输出 77D 特征：
      local_view   : 49D = centered 7x7 local map
      global_state : 20D = hand-crafted global statistics
      legal_action :  8D = env legal action mask
    """

    GRID_SIZE = 128
    VIEW_HALF = 10
    LOCAL_HALF = 3
    MAX_DIST = 181.0

    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.battery = 200
        self.battery_max = 200
        self.cur_pos = (0, 0)
        self.dirt_cleaned = 0
        self.last_dirt_cleaned = 0
        self.total_dirt = 1

        self._view_map = np.zeros((21, 21), dtype=np.float32)
        self._legal_act = [1] * 8

        # Only observed passable cells are set to 1.
        self.passable_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        self.observed_map = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)

        self.visited = set()
        self.visit_count = {}
        self.charger_positions = []
        self.npc_positions = []

        self._nearest_dirt_dist = 200.0
        self._last_nearest_dirt_dist = 200.0
        self.new_observed_cells = 0
        self.cur_revisit_count = 0

    def pb2struct(self, env_obs: dict, last_action: int):
        """Parse env observation and cache required state."""
        del last_action

        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        self.step_no = int(observation["step_no"])
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        self.battery = int(hero["battery"])
        self.battery_max = max(int(hero["battery_max"]), 1)

        self.last_dirt_cleaned = self.dirt_cleaned
        self.dirt_cleaned = int(hero["dirt_cleaned"])
        self.total_dirt = max(int(env_info["total_dirt"]), 1)

        self._legal_act = [int(x) for x in (observation.get("legal_action") or [1] * 8)]

        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.asarray(map_info, dtype=np.float32)
            self.new_observed_cells = self._update_passable(*self.cur_pos)
        else:
            self.new_observed_cells = 0

        self.cur_revisit_count = self.visit_count.get(self.cur_pos, 0)
        self.visit_count[self.cur_pos] = self.cur_revisit_count + 1
        self.visited.add(self.cur_pos)

        for organ in frame_state.get("organs", []):
            if organ.get("sub_type") == 1:
                cx = int(organ["pos"]["x"])
                cz = int(organ["pos"]["z"])
                if (cx, cz) not in self.charger_positions:
                    self.charger_positions.append((cx, cz))

        self.npc_positions = [
            (int(n["pos"]["x"]), int(n["pos"]["z"]))
            for n in frame_state.get("npcs", [])
        ]

    def _update_passable(self, hx: int, hz: int) -> int:
        """Merge current 21x21 observation into observed/passable maps."""
        view = self._view_map
        half = view.shape[0] // 2
        new_observed = 0
        for row in range(view.shape[0]):
            for col in range(view.shape[1]):
                gx = hx - half + col
                gz = hz - half + row
                if 0 <= gx < self.GRID_SIZE and 0 <= gz < self.GRID_SIZE:
                    if self.observed_map[gz, gx] == 0:
                        new_observed += 1
                    self.observed_map[gz, gx] = 1
                    self.passable_map[gz, gx] = 0 if int(view[row, col]) == 0 else 1
        return new_observed

    def _get_local_view_feature(self) -> np.ndarray:
        """Centered 7x7 local crop normalized to [0, 1]."""
        c = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[c - h: c + h + 1, c - h: c + h + 1]
        return (crop / 2.0).flatten().astype(np.float32)

    def _get_charger_feature(self) -> np.ndarray:
        """Top-2 nearest chargers: [dist_norm, dir_x, dir_z] * 2."""
        hx, hz = self.cur_pos
        feats = []

        sorted_c = sorted(
            self.charger_positions,
            key=lambda c: (c[0] - hx) ** 2 + (c[1] - hz) ** 2,
        )

        for idx in range(2):
            if idx < len(sorted_c):
                cx, cz = sorted_c[idx]
                dist = float(np.sqrt((cx - hx) ** 2 + (cz - hz) ** 2))
                dist_norm = _norm(dist, self.MAX_DIST)
                if dist > 1e-5:
                    dx_n = (cx - hx) / dist
                    dz_n = (cz - hz) / dist
                else:
                    dx_n = 0.0
                    dz_n = 0.0
            else:
                dist_norm = 1.0
                dx_n = 0.0
                dz_n = 0.0
            feats.extend([dist_norm, dx_n, dz_n])

        return np.asarray(feats, dtype=np.float32)

    def _get_npc_feature(self) -> np.ndarray:
        """Nearest NPC: [dist_norm, dir_x, dir_z]."""
        hx, hz = self.cur_pos
        if not self.npc_positions:
            return np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

        nx, nz = min(
            self.npc_positions,
            key=lambda n: (n[0] - hx) ** 2 + (n[1] - hz) ** 2,
        )
        dist = float(np.sqrt((nx - hx) ** 2 + (nz - hz) ** 2))
        dist_norm = _norm(dist, self.MAX_DIST)
        if dist > 1e-5:
            dx_n = (nx - hx) / dist
            dz_n = (nz - hz) / dist
        else:
            dx_n = 0.0
            dz_n = 0.0
        return np.asarray([dist_norm, dx_n, dz_n], dtype=np.float32)

    def _get_unvisited_ratio(self) -> float:
        """Ratio of local passable cells that have not been visited yet."""
        hx, hz = self.cur_pos
        c = self.VIEW_HALF
        h = self.LOCAL_HALF
        crop = self._view_map[c - h: c + h + 1, c - h: c + h + 1]

        passable_cnt = 0
        unvisited_cnt = 0
        for row in range(7):
            for col in range(7):
                if int(crop[row, col]) == 0:
                    continue
                passable_cnt += 1
                gx = hx - h + col
                gz = hz - h + row
                if (gx, gz) not in self.visited:
                    unvisited_cnt += 1

        return unvisited_cnt / max(passable_cnt, 1)

    def _calc_nearest_dirt_dist(self) -> float:
        coords = np.argwhere(self._view_map == 2)
        if len(coords) == 0:
            return 200.0
        center = self.VIEW_HALF
        dists = np.sqrt((coords[:, 0] - center) ** 2 + (coords[:, 1] - center) ** 2)
        return float(np.min(dists))

    def _get_global_state_feature(self) -> np.ndarray:
        """Construct the 20D global handcrafted feature vector."""
        hx, hz = self.cur_pos

        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)

        charger_feats = self._get_charger_feature()
        npc_feats = self._get_npc_feature()

        self._last_nearest_dirt_dist = self._nearest_dirt_dist
        self._nearest_dirt_dist = self._calc_nearest_dirt_dist()
        nearest_dirt_norm = _norm(self._nearest_dirt_dist, 180)
        dirt_approaching = 1.0 if self._nearest_dirt_dist < self._last_nearest_dirt_dist else 0.0

        unvisited_ratio = self._get_unvisited_ratio()
        is_low_battery = 1.0 if self.battery <= 50 else 0.0

        total_passable = max(int(np.count_nonzero(self.passable_map)), len(self.visited), 1)
        exploration_progress = min(len(self.visited) / total_passable, 1.0)

        feature = np.concatenate([
            np.asarray(
                [
                    step_norm,
                    battery_ratio,
                    cleaning_progress,
                    remaining_dirt,
                    pos_x_norm,
                    pos_z_norm,
                ],
                dtype=np.float32,
            ),
            charger_feats,
            npc_feats,
            np.asarray(
                [
                    nearest_dirt_norm,
                    dirt_approaching,
                    unvisited_ratio,
                    is_low_battery,
                    exploration_progress,
                ],
                dtype=np.float32,
            ),
        ])
        return feature.astype(np.float32)

    def reward_process(self) -> float:
        """
        Reward shaping keeps the original cleaning incentive,
        and adds late-stage efficiency pressure.
        """
        cleaned_this_step = max(0, self.dirt_cleaned - self.last_dirt_cleaned)
        cleaning_progress = self.dirt_cleaned / max(self.total_dirt, 1)

        cleaning_reward = 0.22 * cleaned_this_step
        explore_reward = 0.003 * min(self.new_observed_cells, 12) * max(0.0, 1.0 - cleaning_progress)
        approach_reward = 0.01 if cleaned_this_step == 0 and self._nearest_dirt_dist < self._last_nearest_dirt_dist else 0.0
        revisit_penalty = -0.0025 * min(3, self.cur_revisit_count) * (0.35 + cleaning_progress)
        step_penalty = -(0.001 + 0.002 * cleaning_progress)

        return cleaning_reward + explore_reward + approach_reward + revisit_penalty + step_penalty

    def get_legal_action(self) -> list:
        return list(self._legal_act)

    def feature_process(self, env_obs: dict, last_action: int):
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()
        global_state = self._get_global_state_feature()
        legal_action = self.get_legal_action()
        legal_arr = np.asarray(legal_action, dtype=np.float32)

        feature = np.concatenate([local_view, global_state, legal_arr]).astype(np.float32)
        reward = self.reward_process()
        return feature, legal_action, reward
