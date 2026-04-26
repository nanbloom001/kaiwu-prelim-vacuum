#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Feature preprocessor for Robot Vacuum.
"""

from typing import Any, List, Sequence, Set, Tuple

import numpy as np

Position = Tuple[int, int]


def _norm(v, v_max, v_min=0.0):
    """Linear normalize to [0, 1] with clipping."""
    v = float(np.clip(v, v_min, v_max))
    if v_max <= v_min:
        return 0.0
    return (v - v_min) / (v_max - v_min)


class Preprocessor:
    """
    输出 84D 特征：
      local_view   : 49D = centered 7x7 local map
      global_state : 27D = hand-crafted global statistics + episode config
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
        self.episode_max_step = 1000
        self.episode_charger_count = 4
        self.episode_battery_max = 200
        self.battery = 200
        self.battery_max = 200
        self.prev_battery = 200
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
        self._nearest_unarrived_charger_dist = self.MAX_DIST
        self._last_nearest_unarrived_charger_dist = self.MAX_DIST
        self.new_observed_cells = 0
        self.cur_revisit_count = 0
        self.charger_regions: List[Set[Position]] = []
        self.rewarded_arrived_charger_regions = set()
        self.new_charger_arrival_reward = 0.0
        self.charger_arrival_steps = {}

    def set_episode_config(self, max_step=None, robot_count=None, charger_count=None, battery_max=None):
        if max_step is not None:
            self.episode_max_step = max(1, int(max_step))
        if charger_count is not None:
            self.episode_charger_count = int(np.clip(charger_count, 1, 4))
        if battery_max is not None:
            self.episode_battery_max = int(np.clip(battery_max, 100, 999))

    def _dynamic_return_margin(self) -> float:
        charger_scarcity = (4.0 - float(self.episode_charger_count)) / 3.0
        low_capacity_factor = np.clip((260.0 - float(self.battery_max)) / 160.0, 0.0, 1.0)
        long_horizon_bonus = 2.0 if self.episode_max_step >= 1500 else 0.0
        return 28.0 + 6.0 * charger_scarcity + 4.0 * low_capacity_factor + long_horizon_bonus

    def pb2struct(self, env_obs: dict[str, Any], last_action: int):
        """Parse env observation and cache required state."""
        del last_action

        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        extra_info = self._get_dict(env_obs, "extra_info", {}) or self._get_dict(observation, "extra_info", {}) or {}
        global_frame_state = self._get_dict(extra_info, "frame_state", {})
        env_info = observation["env_info"]
        hero = frame_state["heroes"]

        self.step_no = int(observation["step_no"])
        self.cur_pos = (int(hero["pos"]["x"]), int(hero["pos"]["z"]))

        self.prev_battery = self.battery
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

        organs = self._merge_organ_states(
            self._get_dict(frame_state, "organs", []),
            self._get_dict(global_frame_state, "organs", []),
        )
        self.charger_regions = self._parse_charger_regions(organs)
        self.new_charger_arrival_reward = 0.0
        self._last_nearest_unarrived_charger_dist = self._nearest_unarrived_charger_dist
        for region in self.charger_regions:
            center = self._charger_region_center(region)
            if center not in self.charger_positions:
                self.charger_positions.append(center)
            region_key = self._charger_region_key(region)
            if self.cur_pos in region and region_key not in self.rewarded_arrived_charger_regions:
                arrival_order = len(self.rewarded_arrived_charger_regions)
                self.rewarded_arrived_charger_regions.add(region_key)
                self.charger_arrival_steps[region_key] = self.step_no
                if arrival_order == 0:
                    self.new_charger_arrival_reward += 0.18
                elif arrival_order == 1:
                    self.new_charger_arrival_reward += 0.24
                else:
                    self.new_charger_arrival_reward += 0.22
        self._nearest_unarrived_charger_dist = self._nearest_unarrived_charger_distance()

        self.npc_positions = self._merge_npc_positions(
            self._get_dict(frame_state, "npcs", []),
            self._get_dict(global_frame_state, "npcs", []),
        )

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

    def _in_bounds(self, pos: Position) -> bool:
        return 0 <= pos[0] < self.GRID_SIZE and 0 <= pos[1] < self.GRID_SIZE

    def _charger_region_key(self, region: Set[Position]) -> Position:
        return min(region)

    def _charger_region_center(self, region: Set[Position]) -> Position:
        xs = sorted(cell[0] for cell in region)
        zs = sorted(cell[1] for cell in region)
        return (xs[len(xs) // 2], zs[len(zs) // 2])

    def _get_dict(self, obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _parse_position(self, obj: Any) -> Position:
        x = int(self._safe_float(self._get_dict(obj, "x", 0), 0.0))
        z = int(self._safe_float(self._get_dict(obj, "z", 0), 0.0))
        return (x, z)

    def _merge_npc_positions(self, *npc_lists: Sequence[Any]) -> List[Position]:
        merged: List[Position] = []
        seen: Set[Position] = set()
        for npcs in npc_lists:
            if not isinstance(npcs, (list, tuple)):
                continue
            for npc in npcs:
                pos_obj = self._get_dict(npc, "pos", None)
                if pos_obj is None:
                    continue
                pos = self._parse_position(pos_obj)
                if pos in seen:
                    continue
                seen.add(pos)
                merged.append(pos)
        return merged

    def _organ_key(self, organ: Any) -> Tuple[Any, ...]:
        config_id = self._get_dict(organ, "config_id", None)
        if config_id not in (None, ""):
            return ("config_id", str(config_id))
        pos = self._parse_position(self._get_dict(organ, "pos", {}))
        sub_type = int(self._safe_float(self._get_dict(organ, "sub_type", 0), 0.0))
        w = int(self._safe_float(self._get_dict(organ, "w", 0), 0.0))
        h = int(self._safe_float(self._get_dict(organ, "h", 0), 0.0))
        return ("shape", sub_type, pos[0], pos[1], w, h)

    def _merge_organ_states(self, *organ_lists: Sequence[Any]) -> List[Any]:
        merged: List[Any] = []
        seen: Set[Tuple[Any, ...]] = set()
        for organs in organ_lists:
            if not isinstance(organs, (list, tuple)):
                continue
            for organ in organs:
                key = self._organ_key(organ)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(organ)
        return merged

    def _parse_charger_regions(self, organs: Sequence[Any]) -> List[Set[Position]]:
        regions: List[Set[Position]] = []
        for organ in organs:
            if int(self._safe_float(self._get_dict(organ, "sub_type", 0), 0.0)) != 1:
                continue
            pos = self._parse_position(self._get_dict(organ, "pos", {}))
            w = max(1, int(self._safe_float(self._get_dict(organ, "w", 3), 3.0)))
            h = max(1, int(self._safe_float(self._get_dict(organ, "h", 3), 3.0)))
            half_w = w // 2
            half_h = h // 2

            region: Set[Position] = set()
            for dx in range(w):
                for dz in range(h):
                    region.add((pos[0] + dx, pos[1] + dz))
            for dx in range(-half_w, half_w + 1):
                for dz in range(-half_h, half_h + 1):
                    region.add((pos[0] + dx, pos[1] + dz))

            region = {cell for cell in region if self._in_bounds(cell)}
            if region:
                regions.append(region)
        return regions

    def _nearest_charger_distance(self) -> float:
        hx, hz = self.cur_pos
        if not self.charger_positions:
            return self.MAX_DIST
        return float(min(
            np.sqrt((cx - hx) ** 2 + (cz - hz) ** 2)
            for cx, cz in self.charger_positions
        ))

    def _nearest_unarrived_charger_distance(self) -> float:
        hx, hz = self.cur_pos
        unarrived_centers = []
        for region in self.charger_regions:
            region_key = self._charger_region_key(region)
            if region_key in self.rewarded_arrived_charger_regions:
                continue
            unarrived_centers.append(self._charger_region_center(region))

        if not unarrived_centers:
            return self.MAX_DIST

        return float(min(
            np.sqrt((cx - hx) ** 2 + (cz - hz) ** 2)
            for cx, cz in unarrived_centers
        ))

    def _get_global_state_feature(self) -> np.ndarray:
        """Construct the 27D global handcrafted feature vector."""
        hx, hz = self.cur_pos

        step_norm = _norm(self.step_no, 2000)
        battery_ratio = _norm(self.battery, self.battery_max)
        cleaning_progress = _norm(self.dirt_cleaned, self.total_dirt)
        remaining_dirt = 1.0 - cleaning_progress
        pos_x_norm = _norm(hx, self.GRID_SIZE)
        pos_z_norm = _norm(hz, self.GRID_SIZE)
        max_step_norm = _norm(self.episode_max_step, 2000, 1)
        step_ratio = _norm(self.step_no, self.episode_max_step, 0)
        time_left_ratio = max(0.0, 1.0 - step_ratio)
        charger_count_norm = _norm(self.episode_charger_count, 4, 1)
        observed_charger_ratio = min(len(self.charger_positions) / max(self.episode_charger_count, 1), 1.0)
        battery_capacity_norm = _norm(self.battery_max, 999, 100)
        nearest_charger_dist = self._nearest_charger_distance()
        return_pressure = np.clip(nearest_charger_dist / max(float(self.battery), 1.0), 0.0, 1.5) / 1.5

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
                    max_step_norm,
                    step_ratio,
                    time_left_ratio,
                    charger_count_norm,
                    observed_charger_ratio,
                    battery_capacity_norm,
                    return_pressure,
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
        prev_battery_ratio = _norm(self.prev_battery, self.battery_max)
        cur_battery_ratio = _norm(self.battery, self.battery_max)
        charge_gain_ratio = max(0.0, cur_battery_ratio - prev_battery_ratio)
        nearest_charger_dist = self._nearest_charger_distance()
        charger_known = nearest_charger_dist < self.MAX_DIST
        is_on_charger = any(self.cur_pos in region for region in self.charger_regions)
        return_margin = self._dynamic_return_margin()
        is_starving = (
            charger_known
            and not is_on_charger
            and float(self.battery) <= nearest_charger_dist + return_margin
        )
        cleaning_reward = 0.22 * cleaned_this_step
        explore_reward = 0.003 * min(self.new_observed_cells, 12) * max(0.0, 1.0 - cleaning_progress)
        approach_reward = 0.01 if cleaned_this_step == 0 and self._nearest_dirt_dist < self._last_nearest_dirt_dist else 0.0
        fresh_path_reward = 0.0
        if self.cur_revisit_count == 0:
            fresh_path_reward = 0.012
        elif self.cur_revisit_count == 1:
            fresh_path_reward = 0.006
        fresh_path_reward *= max(0.4, 1.0 - 0.5 * cleaning_progress)
        unarrived_charger_progress_reward = 0.0
        if is_starving:
            cleaning_reward = 0.0
            explore_reward = 0.0
            approach_reward = 0.0
        # Shift charging preference earlier and punish staying in the dangerous low-battery zone.
        charger_scarcity = (4.0 - float(self.episode_charger_count)) / 3.0
        low_capacity_factor = np.clip((260.0 - float(self.battery_max)) / 160.0, 0.0, 1.0)
        target_low = np.clip(0.32 + 0.05 * charger_scarcity + 0.03 * low_capacity_factor, 0.30, 0.42)
        target_high = np.clip(0.58 + 0.04 * charger_scarcity + 0.03 * low_capacity_factor, 0.54, 0.68)
        if prev_battery_ratio < target_low:
            charge_timing_factor = prev_battery_ratio / max(target_low, 1e-6)
        elif prev_battery_ratio <= target_high:
            charge_timing_factor = 1.0
        else:
            charge_timing_factor = max(
                0.0,
                1.0 - (prev_battery_ratio - target_high) / max(1.0 - target_high, 1e-6),
            )
        charge_efficiency_reward = 1.20 * charge_gain_ratio * charge_timing_factor
        charge_event_reward = 0.0
        if charge_gain_ratio > 0.015:
            # Charging too late means the robot entered a dangerous state first.
            if prev_battery_ratio < 0.18:
                charge_event_reward -= 0.18 + 0.22 * (0.18 - prev_battery_ratio) / 0.18
            # Sweet zone: charge before risk becomes critical, while still using battery effectively.
            elif prev_battery_ratio <= 0.45:
                if prev_battery_ratio < 0.28:
                    charge_event_reward += 0.22 + 0.10 * (prev_battery_ratio - 0.18) / 0.10
                else:
                    charge_event_reward += 0.32
            # Charging too early wastes exploration time and battery utilization.
            elif prev_battery_ratio <= 0.60:
                charge_event_reward -= 0.08 * (prev_battery_ratio - 0.45) / 0.15
            else:
                charge_event_reward -= 0.16 + 0.24 * (prev_battery_ratio - 0.60) / 0.40
        low_battery_penalty = -0.045 * max(0.0, target_low - cur_battery_ratio) / max(target_low, 1e-6)
        critical_battery_penalty = -0.110 * max(0.0, 0.22 - cur_battery_ratio) / 0.22
        unarrived_progress = (
            self._last_nearest_unarrived_charger_dist - self._nearest_unarrived_charger_dist
        )
        if (
            charger_known
            and not is_starving
            and not is_on_charger
            and self._nearest_unarrived_charger_dist < self.MAX_DIST
            and cur_battery_ratio >= target_low + 0.05
            and unarrived_progress > 0.0
        ):
            unarrived_charger_progress_reward = 0.012 * np.clip(unarrived_progress / 2.0, 0.0, 1.5)
        revisit_penalty = -0.0040 * min(4, self.cur_revisit_count) * (0.40 + cleaning_progress)
        step_penalty = -(0.001 + 0.002 * cleaning_progress)
        if is_starving:
            step_penalty *= 2.5
        charger_arrival_reward = self.new_charger_arrival_reward * max(0.60, 1.0 - 0.35 * cleaning_progress)

        return (
            cleaning_reward
            + explore_reward
            + approach_reward
            + fresh_path_reward
            + unarrived_charger_progress_reward
            + charge_efficiency_reward
            + charge_event_reward
            + charger_arrival_reward
            + low_battery_penalty
            + critical_battery_penalty
            + revisit_penalty
            + step_penalty
        )

    def get_legal_action(self) -> list[int]:
        return list(self._legal_act)

    def feature_process(self, env_obs: dict[str, Any], last_action: int):
        self.pb2struct(env_obs, last_action)

        local_view = self._get_local_view_feature()
        global_state = self._get_global_state_feature()
        legal_action = self.get_legal_action()
        legal_arr = np.asarray(legal_action, dtype=np.float32)

        feature = np.concatenate([local_view, global_state, legal_arr]).astype(np.float32)
        reward = self.reward_process()
        return feature, legal_action, reward
