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
        self._charger_pos_set: Set[Position] = set()
        self.rewarded_arrived_charger_regions = set()
        self.new_charger_arrival_reward = 0.0
        self.charger_arrival_steps = {}
        self.candidate_charger_arrival_steps = {}
        self.confirmed_charger_arrival_steps = {}
        self.pending_arrivals = {}
        self.retro_canceled_arrivals = {}
        self.arrival_candidate_confirmed_count = 0
        self.arrival_confirmed_count = 0
        self.arrival_canceled_count = 0
        self.arrival_retro_canceled_count = 0
        self.arrival_confirm_to_fail_gap = -1
        self.arrival_confirm_reward_total = 0.0
        self.arrival_cancel_penalty_total = 0.0
        self.arrival_retro_cancel_penalty_total = 0.0
        self.charge_loop_frames = 0
        self.current_mode = ""
        self.should_charge = False

    def set_episode_config(self, max_step=None, robot_count=None, charger_count=None, battery_max=None):
        if max_step is not None:
            self.episode_max_step = max(1, int(max_step))
        if charger_count is not None:
            self.episode_charger_count = int(np.clip(charger_count, 1, 4))
        if battery_max is not None:
            self.episode_battery_max = int(np.clip(battery_max, 100, 999))

    def set_policy_context(self, target_mode=None, should_charge=None):
        if target_mode is not None:
            self.current_mode = str(target_mode)
        if should_charge is not None:
            self.should_charge = bool(should_charge)

    def pb2struct(self, env_obs: dict, last_action: int):
        """Parse env observation and cache required state."""
        del last_action

        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
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

        self.charger_regions = self._parse_charger_regions(frame_state.get("organs", []))
        self._charger_pos_set.clear()
        for region in self.charger_regions:
            self._charger_pos_set.update(region)
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
                self.pending_arrivals[region_key] = {
                    "region": region,
                    "arrival_order": arrival_order,
                    "arrival_step": self.step_no,
                    "base_reward": self._arrival_base_reward(arrival_order),
                    "start_dirt_cleaned": self.dirt_cleaned,
                    "start_observed_cells": int(np.count_nonzero(self.observed_map)),
                    "start_battery_ratio": _norm(self.battery, self.battery_max),
                    "left_region": False,
                    "candidate_confirmed": False,
                    "final_confirmed": False,
                    "candidate_step": -1,
                    "confirm_step": -1,
                    "candidate_reward_paid": 0.0,
                    "final_reward_paid": 0.0,
                    "cancel_reason": "",
                }
        self._nearest_unarrived_charger_dist = self._nearest_unarrived_charger_distance()

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

    def _arrival_base_reward(self, arrival_order: int) -> float:
        if arrival_order == 0:
            return 0.18
        if arrival_order == 1:
            return 0.15
        return 0.11

    def _arrival_early_discount(self, arrival_step: int) -> float:
        if arrival_step <= 100:
            return 0.60
        if arrival_step <= 150:
            return 0.80
        if arrival_step <= 220:
            return 0.92
        return 1.0

    def _arrival_confirm_params(self, arrival_order: int) -> dict:
        if arrival_order <= 0:
            return {
                "candidate_min_delay": 18,
                "candidate_deadline": 44,
                "final_min_delay": 64,
                "final_deadline": 144,
                "candidate_coverage": 3,
                "candidate_observed": 24,
                "candidate_left_coverage": 2,
                "final_coverage": 5,
                "final_observed": 32,
                "final_left_coverage": 3,
                "final_left_observed": 18,
                "final_battery_gain": 0.06,
                "candidate_scale": 0.40,
            }
        if arrival_order == 1:
            return {
                "candidate_min_delay": 22,
                "candidate_deadline": 52,
                "final_min_delay": 84,
                "final_deadline": 164,
                "candidate_coverage": 4,
                "candidate_observed": 30,
                "candidate_left_coverage": 2,
                "final_coverage": 7,
                "final_observed": 44,
                "final_left_coverage": 4,
                "final_left_observed": 22,
                "final_battery_gain": 0.08,
                "candidate_scale": 0.34,
            }
        return {
            "candidate_min_delay": 26,
            "candidate_deadline": 60,
            "final_min_delay": 100,
            "final_deadline": 188,
            "candidate_coverage": 5,
            "candidate_observed": 38,
            "candidate_left_coverage": 3,
            "final_coverage": 9,
            "final_observed": 56,
            "final_left_coverage": 5,
            "final_left_observed": 28,
            "final_battery_gain": 0.10,
            "candidate_scale": 0.24,
        }

    def _is_on_charger(self) -> bool:
        return self.cur_pos in self._charger_pos_set

    def _latest_confirmed_arrival_age(self) -> int:
        if not self.confirmed_charger_arrival_steps:
            return 10**9
        latest_confirm_step = max(int(step) for step in self.confirmed_charger_arrival_steps.values())
        return max(0, self.step_no - latest_confirm_step)

    def _is_charge_loop_failure(
        self,
        *,
        info: dict,
        charge_active: bool,
        left_region: bool,
        cleaned_this_step: int,
        charge_gain_ratio: float,
    ) -> bool:
        if not charge_active or left_region:
            return False

        age = self.step_no - int(info["arrival_step"])
        no_progress = self.new_observed_cells == 0 and cleaned_this_step == 0 and charge_gain_ratio < 0.01
        if age >= 80 and no_progress and self.charge_loop_frames >= 12:
            return True
        if self.charge_loop_frames >= 18 and self.cur_revisit_count >= 4:
            return True
        return False

    def _resolve_pending_arrivals(
        self,
        *,
        cleaning_progress: float,
        is_on_charger: bool,
        charge_active: bool,
        is_starving: bool,
        cleaned_this_step: int,
        charge_gain_ratio: float,
    ) -> float:
        reward = 0.0
        observed_cells = int(np.count_nonzero(self.observed_map))

        for region_key, info in list(self.pending_arrivals.items()):
            age = self.step_no - int(info["arrival_step"])
            params = self._arrival_confirm_params(int(info["arrival_order"]))
            region = info["region"]
            if not is_on_charger or self.cur_pos not in region:
                info["left_region"] = True

            if age < int(params["candidate_min_delay"]):
                continue

            coverage_gain = self.dirt_cleaned - int(info["start_dirt_cleaned"])
            observed_gain = observed_cells - int(info["start_observed_cells"])
            left_region = bool(info["left_region"])
            battery_ratio = _norm(self.battery, self.battery_max)
            battery_gain = battery_ratio - float(info["start_battery_ratio"])
            early_discount = self._arrival_early_discount(int(info["arrival_step"]))

            candidate_value = (
                coverage_gain >= int(params["candidate_coverage"])
                or observed_gain >= int(params["candidate_observed"])
                or (left_region and coverage_gain >= int(params["candidate_left_coverage"]))
            )
            final_value = (
                coverage_gain >= int(params["final_coverage"])
                or observed_gain >= int(params["final_observed"])
                or (
                    left_region
                    and (
                        coverage_gain >= int(params["final_left_coverage"])
                        or observed_gain >= int(params["final_left_observed"])
                        or battery_gain >= float(params["final_battery_gain"])
                    )
                )
            )
            charge_loop_failure = self._is_charge_loop_failure(
                info=info,
                charge_active=charge_active,
                left_region=left_region,
                cleaned_this_step=cleaned_this_step,
                charge_gain_ratio=charge_gain_ratio,
            )
            severe_risk = (
                is_starving
                or charge_loop_failure
                or battery_ratio <= 0.18
            )
            candidate_expired = age > int(params["candidate_deadline"]) and not bool(info["candidate_confirmed"])
            final_expired = age > int(params["final_deadline"]) and not bool(info["final_confirmed"])

            if severe_risk:
                if info["final_confirmed"]:
                    info["cancel_reason"] = "risk_after_confirm"
                    continue
                penalty = -max(0.03, float(info["candidate_reward_paid"]) + 0.02 * early_discount)
                reward += penalty
                self.arrival_canceled_count += 1
                self.arrival_cancel_penalty_total += penalty
                info["cancel_reason"] = "risk"
                del self.pending_arrivals[region_key]
                continue

            if not info["candidate_confirmed"]:
                if candidate_value and age >= int(params["candidate_min_delay"]):
                    candidate_reward = (
                        float(info["base_reward"])
                        * float(params["candidate_scale"])
                        * early_discount
                        * max(0.65, 1.0 - 0.25 * cleaning_progress)
                    )
                    if charge_active and not left_region:
                        candidate_reward *= 0.5
                    info["candidate_confirmed"] = True
                    info["candidate_step"] = self.step_no
                    info["candidate_reward_paid"] = candidate_reward
                    self.candidate_charger_arrival_steps[region_key] = self.step_no
                    self.arrival_candidate_confirmed_count += 1
                    reward += candidate_reward
                    continue

                if candidate_expired:
                    penalty = -0.02 * early_discount
                    reward += penalty
                    self.arrival_canceled_count += 1
                    self.arrival_cancel_penalty_total += penalty
                    info["cancel_reason"] = "candidate_expired"
                    del self.pending_arrivals[region_key]
                continue

            if info["final_confirmed"]:
                if (
                    charge_loop_failure
                    and age <= int(params["final_deadline"]) + 40
                ):
                    info["cancel_reason"] = "post_confirm_loop"
                continue

            if age < int(params["final_min_delay"]):
                continue

            if final_value and not severe_risk:
                full_reward = (
                    float(info["base_reward"])
                    * early_discount
                    * max(0.65, 1.0 - 0.25 * cleaning_progress)
                )
                remaining_reward = max(0.0, full_reward - float(info["candidate_reward_paid"]))
                if charge_active and not left_region:
                    remaining_reward *= 0.5
                info["final_confirmed"] = True
                info["confirm_step"] = self.step_no
                info["final_reward_paid"] = float(info["candidate_reward_paid"]) + remaining_reward
                self.arrival_confirmed_count += 1
                self.arrival_confirm_reward_total += info["final_reward_paid"]
                self.confirmed_charger_arrival_steps[region_key] = self.step_no
                reward += remaining_reward
                continue

            if final_expired:
                penalty = -max(0.02, float(info["candidate_reward_paid"]))
                reward += penalty
                self.arrival_canceled_count += 1
                self.arrival_cancel_penalty_total += penalty
                info["cancel_reason"] = "final_expired"
                del self.pending_arrivals[region_key]

        return reward

    def finalize_episode_rewards(
        self,
        *,
        result_str: str,
        final_mode: str,
        final_step: int,
        charge_fail_after_arrival: bool = False,
    ) -> float:
        reward = 0.0
        is_charge_fail = result_str == "FAIL" and (
            final_mode == "charge" or bool(charge_fail_after_arrival)
        )
        final_step = int(final_step)

        for region_key, info in list(self.pending_arrivals.items()):
            if not info["final_confirmed"]:
                paid = float(info["candidate_reward_paid"])
                if paid > 0.0:
                    penalty = -(paid + 0.02)
                else:
                    penalty = -0.02
                reward += penalty
                if is_charge_fail and paid > 0.0:
                    event_step = int(info["candidate_step"]) if int(info["candidate_step"]) >= 0 else int(info["arrival_step"])
                    gap = max(0, final_step - event_step)
                    self.arrival_retro_canceled_count += 1
                    self.arrival_confirm_to_fail_gap = (
                        gap
                        if self.arrival_confirm_to_fail_gap < 0
                        else min(self.arrival_confirm_to_fail_gap, gap)
                    )
                    self.retro_canceled_arrivals[region_key] = gap
                    self.arrival_retro_cancel_penalty_total += penalty
                else:
                    self.arrival_canceled_count += 1
                    self.arrival_cancel_penalty_total += penalty
                del self.pending_arrivals[region_key]
                continue

            confirm_step = int(info["confirm_step"])
            if is_charge_fail and 0 <= final_step - confirm_step <= 200:
                paid = float(info["final_reward_paid"])
                penalty = -(paid + 0.04)
                reward += penalty
                self.arrival_retro_canceled_count += 1
                self.arrival_confirm_to_fail_gap = (
                    final_step - confirm_step
                    if self.arrival_confirm_to_fail_gap < 0
                    else min(self.arrival_confirm_to_fail_gap, final_step - confirm_step)
                )
                self.retro_canceled_arrivals[region_key] = final_step - confirm_step
                self.arrival_retro_cancel_penalty_total += penalty

        self.pending_arrivals.clear()
        return reward

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
        unvisited_ratio = self._get_unvisited_ratio()
        prev_battery_ratio = _norm(self.prev_battery, self.battery_max)
        cur_battery_ratio = _norm(self.battery, self.battery_max)
        charge_gain_ratio = max(0.0, cur_battery_ratio - prev_battery_ratio)
        nearest_charger_dist = self._nearest_charger_distance()
        charger_known = nearest_charger_dist < self.MAX_DIST
        is_on_charger = self._is_on_charger()
        charge_active = self.current_mode == "charge" or self.should_charge
        early_phase = self.step_no <= 220
        charger_search_phase = early_phase and not self.charger_positions
        is_starving = (
            charger_known
            and not is_on_charger
            and float(self.battery) <= nearest_charger_dist + 22.0
        )
        cleaning_reward = 0.22 * cleaned_this_step
        explore_reward = 0.003 * min(self.new_observed_cells, 12) * max(0.0, 1.0 - cleaning_progress)
        approach_reward = 0.01 if cleaned_this_step == 0 and self._nearest_dirt_dist < self._last_nearest_dirt_dist else 0.0
        fresh_path_reward = 0.0
        if self.cur_revisit_count == 0:
            fresh_path_reward = 0.015
        elif self.cur_revisit_count == 1:
            fresh_path_reward = 0.006
        fresh_path_reward *= max(0.45, 1.0 - 0.45 * cleaning_progress)
        unarrived_charger_progress_reward = 0.0
        if is_starving:
            cleaning_reward = 0.0
            explore_reward = 0.0
            approach_reward = 0.0
        if charger_search_phase:
            explore_reward *= 0.75
        # Shift charging preference earlier and punish staying in the dangerous low-battery zone.
        charger_scarcity = (4.0 - float(self.episode_charger_count)) / 3.0
        low_capacity_factor = np.clip((260.0 - float(self.battery_max)) / 160.0, 0.0, 1.0)
        target_low = np.clip(0.32 + 0.05 * charger_scarcity + 0.03 * low_capacity_factor, 0.30, 0.42)
        target_high = np.clip(0.58 + 0.04 * charger_scarcity + 0.03 * low_capacity_factor, 0.54, 0.68)
        step_ratio = self.step_no / max(float(self.episode_max_step), 1.0)
        arrival_steps = sorted(int(step) for step in self.charger_arrival_steps.values())
        first_arrival_step = arrival_steps[0] if arrival_steps else -1
        distinct_arrival_count = len(arrival_steps)
        post_first_arrival_guard = (
            len(arrival_steps) == 1
            and first_arrival_step >= 0
            and self.step_no <= min(self.episode_max_step - 80, first_arrival_step + 280)
        )
        charge_risk_zone = (
            not charger_known
            or self.should_charge
            or charge_active
            or is_starving
            or cur_battery_ratio <= target_low + 0.04
        )
        safe_charge_window = (
            charger_known
            and not charge_risk_zone
            and step_ratio >= 0.55
            and cur_battery_ratio >= target_low + 0.12
        )
        transition_after_arrival = self._latest_confirmed_arrival_age() <= 45
        single_arrival_push_safe = (
            distinct_arrival_count != 1
            or (
                cur_battery_ratio >= target_low + 0.06
                and step_ratio >= 0.36
                and self._latest_confirmed_arrival_age() >= 44
            )
        )
        multi_arrival_push_safe = (
            distinct_arrival_count < 2
            or (
                cur_battery_ratio >= target_low + 0.12
                and step_ratio >= 0.64
                and self._latest_confirmed_arrival_age() >= 60
            )
        )
        phase_c_reward_enabled = (
            safe_charge_window
            and not transition_after_arrival
            and not charger_search_phase
            and not post_first_arrival_guard
        )
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
            elif safe_charge_window and not post_first_arrival_guard and prev_battery_ratio <= 0.60:
                charge_event_reward -= 0.08 * (prev_battery_ratio - 0.45) / 0.15
            elif safe_charge_window and not post_first_arrival_guard:
                charge_event_reward -= 0.16 + 0.24 * (prev_battery_ratio - 0.60) / 0.40
        low_battery_penalty = -0.035 * max(0.0, target_low - cur_battery_ratio) / max(target_low, 1e-6)
        critical_battery_penalty = -0.090 * max(0.0, 0.22 - cur_battery_ratio) / 0.22
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
            and single_arrival_push_safe
        ):
            unarrived_charger_progress_reward = 0.012 * np.clip(unarrived_progress / 2.0, 0.0, 1.5)
            if self._nearest_unarrived_charger_dist < 20.0:
                unarrived_charger_progress_reward += 0.012
            if self._nearest_unarrived_charger_dist < 12.0:
                unarrived_charger_progress_reward += 0.015
            if self._nearest_unarrived_charger_dist < 6.0:
                unarrived_charger_progress_reward += 0.010
            if early_phase:
                unarrived_charger_progress_reward *= 1.35
            if charger_search_phase:
                unarrived_charger_progress_reward *= 1.25
            if post_first_arrival_guard:
                unarrived_charger_progress_reward *= 1.36
            if distinct_arrival_count == 1:
                unarrived_charger_progress_reward *= 1.28
                if step_ratio >= 0.55:
                    unarrived_charger_progress_reward *= 1.20
                elif step_ratio >= 0.42:
                    unarrived_charger_progress_reward *= 1.12
            elif distinct_arrival_count >= 2:
                unarrived_charger_progress_reward *= 1.18
        late_multi_arrival_harvest = (
            distinct_arrival_count >= 2
            and step_ratio >= 0.64
            and not charge_active
            and not is_starving
            and cur_battery_ratio >= target_low + 0.12
            and multi_arrival_push_safe
        )
        late_multi_arrival_progress_reward = 0.0
        if late_multi_arrival_harvest:
            if cleaned_this_step > 0:
                late_multi_arrival_progress_reward += 0.030 * min(cleaned_this_step, 2)
                if self.cur_revisit_count == 0:
                    late_multi_arrival_progress_reward += 0.008
            if self.new_observed_cells > 0:
                late_multi_arrival_progress_reward += 0.003 * min(self.new_observed_cells, 8)
            if phase_c_reward_enabled:
                late_multi_arrival_progress_reward *= 1.15
        revisit_penalty = -0.0040 * min(4, self.cur_revisit_count) * (0.40 + cleaning_progress)
        single_charger_loop_penalty = 0.0
        if (
            distinct_arrival_count == 1
            and self.step_no >= max(first_arrival_step + 140, 320)
            and self.new_observed_cells == 0
            and cleaned_this_step == 0
            and self.cur_revisit_count >= 2
            and self._nearest_unarrived_charger_dist < self.MAX_DIST
        ):
            single_charger_loop_penalty -= 0.012
            if self._nearest_unarrived_charger_dist > 18.0:
                single_charger_loop_penalty -= 0.006
            if step_ratio >= 0.60:
                single_charger_loop_penalty -= 0.006
        single_charger_tail_penalty = 0.0
        if (
            distinct_arrival_count == 1
            and step_ratio >= 0.72
            and not charge_active
            and not is_starving
            and self.cur_revisit_count >= 1
        ):
            single_charger_tail_penalty -= 0.014
            if self.new_observed_cells == 0 and cleaned_this_step == 0:
                single_charger_tail_penalty -= 0.012
        if (
            distinct_arrival_count == 1
            and step_ratio >= 0.52
            and not charge_active
            and self.new_observed_cells == 0
            and cleaned_this_step == 0
            and self._nearest_unarrived_charger_dist < self.MAX_DIST
        ):
            single_charger_tail_penalty -= 0.008
        single_arrival_late_low_yield_penalty = 0.0
        if (
            distinct_arrival_count == 1
            and step_ratio >= 0.62
            and not charge_active
            and not is_starving
            and self.new_observed_cells == 0
            and cleaned_this_step == 0
        ):
            single_arrival_late_low_yield_penalty -= 0.012
            if self.cur_revisit_count >= 3:
                single_arrival_late_low_yield_penalty -= 0.008
        late_low_yield_penalty = 0.0
        if (
            late_multi_arrival_harvest
            and self.new_observed_cells == 0
            and cleaned_this_step == 0
            and self.cur_revisit_count >= 2
        ):
            late_low_yield_penalty -= 0.014
            if self.cur_revisit_count >= 4:
                late_low_yield_penalty -= 0.008
        loop_penalty = 0.0
        no_progress_penalty_scale = 1.0 if phase_c_reward_enabled else 0.35
        if charge_risk_zone:
            no_progress_penalty_scale = min(no_progress_penalty_scale, 0.18)
        if transition_after_arrival:
            no_progress_penalty_scale = min(no_progress_penalty_scale, 0.25)
        if post_first_arrival_guard:
            no_progress_penalty_scale = min(no_progress_penalty_scale, 0.20)
        if (
            not is_starving
            and self.cur_revisit_count >= 3
            and self.new_observed_cells == 0
            and cleaned_this_step == 0
        ):
            loop_penalty = -0.012 * no_progress_penalty_scale
        charge_loop_penalty = 0.0
        if (
            charge_active
            and self.cur_revisit_count >= 2
            and self.new_observed_cells == 0
            and cleaned_this_step == 0
            and charge_gain_ratio < 0.01
        ):
            charge_loop_penalty -= 0.016
            self.charge_loop_frames += 1
            if unvisited_ratio < 0.18:
                charge_loop_penalty -= 0.004
        else:
            self.charge_loop_frames = 0
        step_penalty = -(0.001 + 0.002 * cleaning_progress)
        if charge_active:
            step_penalty -= 0.0008
        elif phase_c_reward_enabled:
            step_penalty -= 0.0012
        elif post_first_arrival_guard:
            step_penalty -= 0.0004
        if is_starving:
            step_penalty *= 2.5
        charger_arrival_reward = self._resolve_pending_arrivals(
            cleaning_progress=cleaning_progress,
            is_on_charger=is_on_charger,
            charge_active=charge_active,
            is_starving=is_starving,
            cleaned_this_step=cleaned_this_step,
            charge_gain_ratio=charge_gain_ratio,
        )

        return (
            cleaning_reward
            + explore_reward
            + approach_reward
            + fresh_path_reward
            + unarrived_charger_progress_reward
            + charge_efficiency_reward
            + charge_event_reward
            + charger_arrival_reward
            + late_multi_arrival_progress_reward
            + low_battery_penalty
            + critical_battery_penalty
            + revisit_penalty
            + single_charger_loop_penalty
            + single_charger_tail_penalty
            + single_arrival_late_low_yield_penalty
            + late_low_yield_penalty
            + loop_penalty
            + charge_loop_penalty
            + step_penalty
        )

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
