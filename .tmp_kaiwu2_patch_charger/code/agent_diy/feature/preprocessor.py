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
    PATH_STYLE_NAMES = (
        "aggressive_explore",
        "balanced",
        "deep_clean",
        "safe_return",
        "escape",
    )
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
        self.hero_cleaned = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=np.uint8)
        self.npc_cleaned = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=np.uint8)
        self.blocked_cells = {}

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
        self._last_dist_map = None
        self._last_prev_x = None
        self._last_prev_z = None

        self.charge_mode = False
        self.on_charger = False
        self.last_on_charger = False
        self.goal = None
        self.goal_kind = None
        self.path = []
        self.last_plan_step = -999
        self.goal_set_step = -999
        self.goal_progress_step = -999
        self.last_charger_id = -1
        self.last_charger_rect = None
        self.post_charge_until_step = -999
        self.active_charger_id = -1
        self.active_charger_rect = None
        self.active_charger_spoke = None
        self.launch_lock_source_rect = None
        self.launch_lock_spoke = None
        self.charger_spoke_usage = {}
        self.explored_ratio = 0.0
        self.planner_mode = "explore"
        self.active_region_id = -1
        self.active_region_type = None
        self.active_region_entry = None
        self.active_region_anchor = None
        self.active_region_goal = None
        self.active_region_cells = []
        self.active_region_parent_charger = -1
        self.active_region_axis = 0
        self.active_region_sign = 1
        self.active_mouth_id = -1
        self.active_cover_sequence = []
        self.active_cover_index = 0
        self.active_cover_strip = 0
        self.completed_region_count = 0
        self.completed_region_ids = set()
        self.region_mask = np.full((Config.GRID_SIZE, Config.GRID_SIZE), -1, dtype=np.int16)
        self.transit_spine_mask = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        self.corridor_mask = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        self.room_mask = np.zeros((Config.GRID_SIZE, Config.GRID_SIZE), dtype=bool)
        self.regions = []
        self.mouths = []
        self.charger_halo_waste_steps = 0
        self.plan_churn_count = 0
        self.frontier_skip_steps = 0
        self.spine_transit_steps = 0

        self.pending_action = 0
        self.pending_prob = np.full(Config.ACTION_DIM, 1.0 / Config.ACTION_DIM, dtype=np.float32)
        self.current_path_style = 1
        self.current_path_style_name = "balanced"
        self.decision_event = True
        self.decision_step = 0
        self.current_decision_candidates = []
        self.candidate_feature = np.zeros((Config.CANDIDATE_FLAT_DIM,), dtype=np.float32)
        self.candidate_mask = np.zeros((Config.MAX_DECISION_CANDIDATES,), dtype=np.float32)
        self.teacher_candidate_idx = 0
        self.teacher_candidate_prob = np.zeros((Config.MAX_DECISION_CANDIDATES,), dtype=np.float32)
        self.teacher_path_style_idx = 1
        self.teacher_path_style_prob = np.zeros((Config.PATH_STYLE_DIM,), dtype=np.float32)
        self.teacher_candidate_type = None
        self.teacher_candidate_region_id = -1
        self.teacher_candidate_goal = None
        self.teacher_candidate_goal_kind = None
        self.teacher_candidate_charger_id = -1
        self.last_plan_risk = 0.0
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
        self._known_integral = np.zeros((Config.GRID_SIZE + 1, Config.GRID_SIZE + 1), dtype=np.float32)
        self.active_cover_strip_ids = []

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

    def get_candidate_feature(self):
        return self.candidate_feature.copy()

    def get_candidate_mask(self):
        return self.candidate_mask.copy()

    def get_teacher_candidate_index(self):
        return int(self.teacher_candidate_idx)

    def get_teacher_candidate_prob(self):
        return self.teacher_candidate_prob.copy()

    def get_teacher_path_style_index(self):
        return int(self.teacher_path_style_idx)

    def get_teacher_path_style_prob(self):
        return self.teacher_path_style_prob.copy()

    def is_decision_event(self):
        return bool(self.decision_event)

    def get_decision_span(self):
        return max(1, self.step_no - self.decision_step + 1)

    def get_blocked_cell_count(self):
        return int(len(self.blocked_cells))

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

    def _clear_decision_candidates(self):
        self.current_decision_candidates = []
        self.candidate_feature = np.zeros((Config.CANDIDATE_FLAT_DIM,), dtype=np.float32)
        self.candidate_mask = np.zeros((Config.MAX_DECISION_CANDIDATES,), dtype=np.float32)
        self.teacher_candidate_idx = 0
        self.teacher_candidate_prob = np.zeros((Config.MAX_DECISION_CANDIDATES,), dtype=np.float32)
        self.teacher_path_style_idx = 1
        self.teacher_path_style_prob = np.zeros((Config.PATH_STYLE_DIM,), dtype=np.float32)

    def _prepare_decision_candidates(self):
        self._clear_decision_candidates()
        dist = self._last_dist_map
        charger_dist = self._charger_dist_map
        candidates = []

        keep_plan = self._build_keep_plan_candidate()
        if keep_plan is not None:
            candidates.append(keep_plan)

        candidates.extend(self._build_clean_region_candidates(dist, charger_dist))
        candidates.extend(self._build_explore_mouth_candidates(dist, charger_dist))
        candidates.extend(self._build_charger_candidates(dist))

        fallback = self._build_frontier_candidate(dist, charger_dist)
        if fallback is not None:
            candidates.append(fallback)

        candidates = candidates[: Config.MAX_DECISION_CANDIDATES]
        self.current_decision_candidates = candidates
        if not candidates:
            self.candidate_mask[0] = 1.0
            self.teacher_candidate_prob[0] = 1.0
            self.teacher_path_style_prob[self.teacher_path_style_idx] = 1.0
            return

        for idx, candidate in enumerate(candidates):
            self.candidate_mask[idx] = 1.0
            feat = self._encode_candidate_feature(candidate)
            start = idx * Config.CANDIDATE_FEATURE_DIM
            self.candidate_feature[start : start + Config.CANDIDATE_FEATURE_DIM] = feat

        self.teacher_candidate_idx = self._match_teacher_candidate(candidates)
        self.teacher_candidate_prob[self.teacher_candidate_idx] = 1.0
        self.teacher_path_style_idx = self._teacher_style_for_candidate(candidates[self.teacher_candidate_idx])
        self.teacher_path_style_prob[self.teacher_path_style_idx] = 1.0
        self.current_path_style = int(self.teacher_path_style_idx)
        self.current_path_style_name = self.PATH_STYLE_NAMES[self.current_path_style]

    def _set_teacher_candidate_target(self, candidate_type=None, goal=None, goal_kind=None, region_id=-1, charger_id=-1):
        self.teacher_candidate_type = candidate_type
        self.teacher_candidate_region_id = int(region_id) if region_id is not None else -1
        self.teacher_candidate_goal = tuple(goal) if goal is not None else None
        self.teacher_candidate_goal_kind = goal_kind
        self.teacher_candidate_charger_id = int(charger_id) if charger_id is not None else -1

    def _candidate_matches_teacher_target(self, candidate):
        target_type = self.teacher_candidate_type
        if target_type is None or candidate.get("type") != target_type:
            return False

        if target_type == "clean_region":
            target_region = int(self.teacher_candidate_region_id)
            return target_region >= 0 and int(candidate.get("region_id", -1)) == target_region

        if target_type == "explore_mouth":
            target_region = int(self.teacher_candidate_region_id)
            return target_region >= 0 and int(candidate.get("region_id", -1)) == target_region

        if target_type == "charger":
            target_charger = int(self.teacher_candidate_charger_id)
            if target_charger >= 0 and int(candidate.get("charger_id", -1)) == target_charger:
                return True
            return self.teacher_candidate_goal is not None and tuple(candidate.get("goal", ())) == tuple(self.teacher_candidate_goal)

        if target_type == "fallback_frontier":
            if self.teacher_candidate_goal_kind is not None and candidate.get("goal_kind") != self.teacher_candidate_goal_kind:
                return False
            return self.teacher_candidate_goal is not None and tuple(candidate.get("goal", ())) == tuple(self.teacher_candidate_goal)

        if target_type == "keep_plan":
            return (
                self.teacher_candidate_goal is not None
                and tuple(candidate.get("goal", ())) == tuple(self.teacher_candidate_goal)
                and candidate.get("goal_kind") == self.teacher_candidate_goal_kind
            )
        return False

    def _build_keep_plan_candidate(self):
        if self.goal is None or not self._goal_still_valid():
            return None
        goal_dist = self._chebyshev(self.cur_pos, self.goal)
        candidate = {
            "type": "keep_plan",
            "goal": tuple(self.goal),
            "goal_kind": self.goal_kind or "frontier",
            "planner_mode": self.planner_mode,
            "region_id": int(self.active_region_id),
            "region_type": self.active_region_type,
            "entry": tuple(self.active_region_entry) if self.active_region_entry is not None else tuple(self.goal),
            "anchor": tuple(self.active_region_anchor) if self.active_region_anchor is not None else tuple(self.goal),
            "dirty_mass": float(self._region_total_mass(self._active_region_object()) if self._active_region_object() is not None else 0.0),
            "frontier_mass": float(self._frontier_gain(self.goal[0], self.goal[1])) if self.goal is not None else 0.0,
            "area": float(len(self.active_region_cells)),
            "depth": float(goal_dist),
            "goal_dist": float(goal_dist),
            "entry_dist": float(self._chebyshev(self.cur_pos, self.active_region_entry)) if self.active_region_entry is not None else float(goal_dist),
            "anchor_dist": float(self._chebyshev(self.cur_pos, self.active_region_anchor)) if self.active_region_anchor is not None else float(goal_dist),
            "charger_dist": float(self._current_charger_distance()),
            "return_budget": float(self._region_return_budget(self.goal, goal_dist, self._charger_dist_map)),
            "spine_ratio": 0.0,
            "risk": float(self._npc_zone_penalty(self.goal)),
            "blocked_risk": 1.0 if tuple(self.goal) in self.blocked_cells else 0.0,
            "npc_clean_ratio": float(self.npc_cleaned[self.goal[0], self.goal[1]]) if self._in_bounds(*self.goal) else 0.0,
            "score": float(Config.KEEP_PLAN_CANDIDATE_BONUS + self._goal_sticky_bonus(self.goal_kind, self.goal)),
        }
        return candidate

    def _build_clean_region_candidates(self, dist, charger_dist):
        rankings = self._rank_region_candidates(dist, charger_dist)
        out = []
        for score, region in rankings[: Config.MAX_CLEAN_REGION_CANDIDATES]:
            goal_kind, goal = self._pick_region_goal(region, dist, charger_dist)
            if goal is None:
                continue
            entry = tuple(region["entry"])
            anchor = tuple(region["anchor"])
            goal_dist = int(dist[goal[0], goal[1]]) if dist is not None else self._chebyshev(self.cur_pos, goal)
            out.append(
                {
                    "type": "clean_region",
                    "goal": tuple(goal),
                    "goal_kind": goal_kind or "dirty",
                    "planner_mode": "clean_region",
                    "region_id": int(region["id"]),
                    "region_type": region["type"],
                    "entry": entry,
                    "anchor": anchor,
                    "dirty_mass": float(region.get("dirty_mass", 0.0)),
                    "frontier_mass": float(region.get("frontier_mass", 0.0)),
                    "area": float(region.get("area", len(region.get("cells", [])))),
                    "depth": float(max(self._region_depth_progress(region, goal), 0.0)),
                    "goal_dist": float(max(goal_dist, 0)),
                    "entry_dist": float(max(int(dist[entry[0], entry[1]]) if dist is not None else self._chebyshev(self.cur_pos, entry), 0)),
                    "anchor_dist": float(max(int(dist[anchor[0], anchor[1]]) if dist is not None else self._chebyshev(self.cur_pos, anchor), 0)),
                    "charger_dist": float(self._charger_dist_at(goal, charger_dist)),
                    "return_budget": float(self._region_return_budget(goal, max(goal_dist, 0), charger_dist)),
                    "spine_ratio": self._region_spine_ratio(region),
                    "risk": float(region.get("risk_mean", 0.0)),
                    "blocked_risk": self._region_blocked_ratio(region),
                    "npc_clean_ratio": self._region_npc_clean_ratio(region),
                    "score": float(score),
                }
            )
        return out

    def _build_explore_mouth_candidates(self, dist, charger_dist):
        rankings = self._rank_explore_candidates(dist, charger_dist)
        out = []
        for score, goal, region in rankings[: Config.MAX_EXPLORE_MOUTH_CANDIDATES]:
            entry = tuple(region["entry"])
            anchor = tuple(region["anchor"])
            goal_dist = int(dist[goal[0], goal[1]]) if dist is not None else self._chebyshev(self.cur_pos, goal)
            out.append(
                {
                    "type": "explore_mouth",
                    "goal": tuple(goal),
                    "goal_kind": "frontier",
                    "planner_mode": "explore",
                    "region_id": int(region["id"]),
                    "region_type": region["type"],
                    "entry": entry,
                    "anchor": anchor,
                    "dirty_mass": float(region.get("dirty_mass", 0.0)),
                    "frontier_mass": float(region.get("frontier_mass", 0.0)),
                    "area": float(region.get("area", len(region.get("cells", [])))),
                    "depth": float(max(self._region_depth_progress(region, goal), 0.0)),
                    "goal_dist": float(max(goal_dist, 0)),
                    "entry_dist": float(max(int(dist[entry[0], entry[1]]) if dist is not None else self._chebyshev(self.cur_pos, entry), 0)),
                    "anchor_dist": float(max(int(dist[anchor[0], anchor[1]]) if dist is not None else self._chebyshev(self.cur_pos, anchor), 0)),
                    "charger_dist": float(self._charger_dist_at(goal, charger_dist)),
                    "return_budget": float(self._region_return_budget(goal, max(goal_dist, 0), charger_dist)),
                    "spine_ratio": self._region_spine_ratio(region),
                    "risk": float(region.get("risk_mean", 0.0)),
                    "blocked_risk": self._region_blocked_ratio(region),
                    "npc_clean_ratio": self._region_npc_clean_ratio(region),
                    "score": float(score),
                }
            )
        return out

    def _build_charger_candidates(self, dist):
        out = []
        rankings = self._rank_charger_candidates(dist)
        for score, charger, goal, goal_dist, spoke in rankings[: Config.MAX_CHARGER_CANDIDATES]:
            out.append(
                {
                    "type": "charger",
                    "goal": tuple(goal),
                    "goal_kind": "charger",
                    "planner_mode": "return_charge",
                    "region_id": -1,
                    "region_type": "charger",
                    "entry": tuple(goal),
                    "anchor": tuple(goal),
                    "dirty_mass": 0.0,
                    "frontier_mass": 0.0,
                    "area": float(charger["w"] * charger["h"]),
                    "depth": 0.0,
                    "goal_dist": float(max(goal_dist, 0)),
                    "entry_dist": float(max(goal_dist, 0)),
                    "anchor_dist": float(max(goal_dist, 0)),
                    "charger_dist": 0.0,
                    "return_budget": float(max(goal_dist, 0)),
                    "spine_ratio": 0.0,
                    "risk": float(self._npc_zone_penalty(goal)),
                    "blocked_risk": 1.0 if tuple(goal) in self.blocked_cells else 0.0,
                    "npc_clean_ratio": 0.0,
                    "score": float(score),
                    "charger_id": int(charger.get("id", -1)),
                    "charger_rect": {
                        "id": int(charger.get("id", -1)),
                        "x": int(charger["x"]),
                        "z": int(charger["z"]),
                        "w": int(charger["w"]),
                        "h": int(charger["h"]),
                    },
                    "charger_spoke": tuple(spoke),
                }
            )
        return out

    def _build_frontier_candidate(self, dist, charger_dist):
        goal_kind, goal = self._pick_best_frontier_goal(dist, charger_dist)
        if goal is None:
            goal_kind, goal = self._pick_best_dirty_goal(dist, charger_dist)
        if goal is None:
            return None
        goal_dist = int(dist[goal[0], goal[1]]) if dist is not None else self._chebyshev(self.cur_pos, goal)
        return {
            "type": "fallback_frontier",
            "goal": tuple(goal),
            "goal_kind": goal_kind or "frontier",
            "planner_mode": "explore",
            "region_id": -1,
            "region_type": "fallback",
            "entry": tuple(goal),
            "anchor": tuple(goal),
            "dirty_mass": float(self._dirty_density(goal[0], goal[1])),
            "frontier_mass": float(self._frontier_gain(goal[0], goal[1])),
            "area": 0.0,
            "depth": float(self._frontier_gain(goal[0], goal[1])),
            "goal_dist": float(max(goal_dist, 0)),
            "entry_dist": float(max(goal_dist, 0)),
            "anchor_dist": float(max(goal_dist, 0)),
            "charger_dist": float(self._charger_dist_at(goal, charger_dist)),
            "return_budget": float(self._region_return_budget(goal, max(goal_dist, 0), charger_dist)),
            "spine_ratio": float(self._spine_penalty(goal)),
            "risk": float(self._npc_zone_penalty(goal)),
            "blocked_risk": 1.0 if tuple(goal) in self.blocked_cells else 0.0,
            "npc_clean_ratio": float(self.npc_cleaned[goal[0], goal[1]]) if self._in_bounds(*goal) else 0.0,
            "score": float(self._frontier_gain(goal[0], goal[1])),
        }

    def _rank_region_candidates(self, dist, charger_dist):
        ranked = []
        for region in self.regions:
            dirty_mass, frontier_mass = self._region_current_masses(region)
            region["dirty_mass"] = float(dirty_mass)
            region["frontier_mass"] = float(frontier_mass)
            dirty_trigger = float(Config.REGION_DIRTY_TRIGGER)
            frontier_trigger = float(Config.REGION_FRONTIER_TRIGGER)
            if region["type"] == "bridge":
                dirty_trigger = max(dirty_trigger, float(Config.BRIDGE_REGION_DIRTY_TRIGGER))
                frontier_trigger = max(frontier_trigger, float(Config.BRIDGE_REGION_FRONTIER_TRIGGER))
            if dirty_mass < dirty_trigger and frontier_mass < frontier_trigger:
                continue
            entry = tuple(region["entry"])
            anchor = tuple(region["anchor"])
            entry_dist = int(dist[entry[0], entry[1]]) if dist is not None else self._chebyshev(self.cur_pos, entry)
            anchor_dist = int(dist[anchor[0], anchor[1]]) if dist is not None else self._chebyshev(self.cur_pos, anchor)
            if entry_dist < 0 and anchor_dist < 0:
                continue
            travel_cost = max(entry_dist, anchor_dist, 0)
            if self._region_return_budget(anchor, travel_cost, charger_dist) >= self.battery:
                continue
            score = (
                Config.REGION_VALUE_DIRTY_WEIGHT * float(dirty_mass)
                + Config.REGION_VALUE_FRONTIER_WEIGHT * float(frontier_mass)
                - Config.REGION_VALUE_DISTANCE_WEIGHT * float(max(entry_dist, 0))
                - Config.REGION_VALUE_RISK_WEIGHT * float(region["risk_mean"])
                - Config.REGION_VALUE_SPINE_PENALTY * float(max(0, len(region["mouths"]) - 1))
                - 1.1 * self._region_blocked_ratio(region)
                - Config.NPC_CLEAN_PENALTY * self._region_npc_clean_ratio(region)
            )
            if region["type"] == "tail":
                score += 5.2
            elif region["type"] == "room":
                score += 2.8
            else:
                score -= 6.0
            score -= float(self._spine_penalty(entry)) * Config.SPINE_DIRTY_PENALTY
            score -= self._post_charge_goal_penalty(entry)
            if self.active_region_id == int(region["id"]):
                score += 2.2
            ranked.append((float(score), region))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _rank_explore_candidates(self, dist, charger_dist):
        ranked = []
        for region in self.regions:
            dirty_mass, frontier_mass = self._region_current_masses(region)
            region["dirty_mass"] = float(dirty_mass)
            region["frontier_mass"] = float(frontier_mass)
            frontier_trigger = float(Config.REGION_FRONTIER_TRIGGER)
            if region["type"] == "bridge":
                frontier_trigger = max(frontier_trigger, float(Config.BRIDGE_REGION_FRONTIER_TRIGGER))
            if frontier_mass < frontier_trigger and dirty_mass <= 0.0:
                continue
            entry = tuple(region["entry"])
            d = int(dist[entry[0], entry[1]]) if dist is not None else self._chebyshev(self.cur_pos, entry)
            if d <= 0:
                continue
            if self._region_return_budget(entry, d, charger_dist) >= self.battery:
                continue
            goal_pos = None
            goal_gain = -1e9
            for pos in region["cells"]:
                x, z = pos
                cell_dist = int(dist[x, z]) if dist is not None else self._chebyshev(self.cur_pos, pos)
                if cell_dist <= 0 or not self._has_unknown_neighbor(pos):
                    continue
                gain = (
                    2.6 * self._frontier_gain(x, z)
                    + 0.12 * self._dirty_density(x, z)
                    - 0.08 * cell_dist
                    - 0.03 * self._npc_zone_penalty(pos)
                    - 0.15 * self._spine_penalty(pos)
                    - 0.60 * (1.0 if pos in self.blocked_cells else 0.0)
                )
                if gain > goal_gain:
                    goal_gain = gain
                    goal_pos = pos
            if goal_pos is None:
                goal_pos = entry
            score = (
                Config.EXPLORE_MOUTH_BONUS
                + Config.EXPLORE_ROOM_FRONTIER_BONUS * float(frontier_mass)
                + 0.18 * float(dirty_mass)
                - 0.22 * float(d)
                - 0.04 * float(region["risk_mean"])
                - 1.2 * self._region_blocked_ratio(region)
                - Config.NPC_CLEAN_PENALTY * self._region_npc_clean_ratio(region)
            )
            if region["type"] == "tail":
                score += 1.8
            elif region["type"] == "room":
                score += 1.2
            else:
                score -= Config.EXPLORE_BRIDGE_PENALTY
            if self.step_no <= Config.AGGRESSIVE_EDGE_STEPS:
                score += 0.65 * self._edge_bonus(goal_pos)
            ranked.append((float(score), tuple(goal_pos), region))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _rank_charger_candidates(self, dist):
        ranked = []
        for charger in self.chargers:
            spoke = self._select_charger_spoke(charger, dist)
            best_goal = self._choose_charger_goal_for_spoke(charger, dist, spoke)
            if best_goal is None:
                continue
            best_dist = int(dist[best_goal[0], best_goal[1]]) if dist is not None else self._chebyshev(self.cur_pos, best_goal)
            complexity = self._charger_retry_margin_for_goal(best_goal)
            score = -1.0 * float(best_dist) - 0.35 * complexity - 0.08 * self._npc_zone_penalty(best_goal)
            score -= Config.CHARGER_ZONE_REPEAT_GOAL_WEIGHT * self._charger_zone_repeat_penalty(best_goal, charger)
            score -= self._charger_spoke_usage_penalty(charger, spoke)
            score += 0.18 * self._score_charger_spoke(charger, spoke, dist)
            ranked.append((float(score), charger, tuple(best_goal), int(best_dist), tuple(spoke)))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked

    def _match_teacher_candidate(self, candidates):
        for idx, candidate in enumerate(candidates):
            if self._candidate_matches_teacher_target(candidate):
                return idx

        for idx, candidate in enumerate(candidates):
            if self.teacher_candidate_type not in {None, "keep_plan"} and candidate.get("type") == "keep_plan":
                continue
            if candidate["goal"] == self.goal and candidate["goal_kind"] == self.goal_kind and candidate["planner_mode"] == self.planner_mode:
                return idx

        if self.teacher_candidate_type not in {None, "keep_plan"}:
            for idx, candidate in enumerate(candidates):
                if candidate.get("type") != "keep_plan":
                    return idx
        return 0

    def _teacher_style_for_candidate(self, candidate):
        if self.stuck_chain > 0:
            return self.PATH_STYLE_NAMES.index("escape")
        candidate_type = candidate.get("type")
        if candidate_type == "charger" or self.charge_mode:
            return self.PATH_STYLE_NAMES.index("safe_return")
        if candidate_type == "clean_region":
            return self.PATH_STYLE_NAMES.index("deep_clean")
        if candidate_type == "explore_mouth" and (self.step_no <= Config.AGGRESSIVE_EDGE_STEPS or not self.chargers):
            return self.PATH_STYLE_NAMES.index("aggressive_explore")
        return self.PATH_STYLE_NAMES.index("balanced")

    def _encode_candidate_feature(self, candidate):
        candidate_type = candidate.get("type", "fallback_frontier")
        one_hot = [0.0] * 5
        index_map = {
            "keep_plan": 0,
            "clean_region": 1,
            "explore_mouth": 2,
            "charger": 3,
            "fallback_frontier": 4,
        }
        one_hot[index_map.get(candidate_type, 4)] = 1.0
        goal = candidate.get("goal", self.cur_pos)
        return np.array(
            one_hot
            + [
                _norm(candidate.get("dirty_mass", 0.0), 96.0),
                _norm(candidate.get("frontier_mass", 0.0), 96.0),
                _norm(candidate.get("area", 0.0), 512.0),
                _norm(candidate.get("depth", 0.0), 64.0),
                _norm(candidate.get("entry_dist", 0.0), Config.GRID_SIZE),
                _norm(candidate.get("anchor_dist", 0.0), Config.GRID_SIZE),
                _norm(candidate.get("goal_dist", 0.0), Config.GRID_SIZE),
                _norm(candidate.get("charger_dist", 0.0), Config.GRID_SIZE),
                _signed_norm(self.battery - candidate.get("return_budget", 0.0), 128.0),
                float(np.clip(candidate.get("spine_ratio", 0.0), 0.0, 1.0)),
                _norm(candidate.get("risk", 0.0), 40.0),
                float(np.clip(candidate.get("blocked_risk", 0.0), 0.0, 1.0)),
                float(np.clip(candidate.get("npc_clean_ratio", 0.0), 0.0, 1.0)),
                1.0 if candidate.get("region_id", -1) == self.active_region_id else 0.0,
                1.0 if candidate.get("goal_kind") == "dirty" else 0.0,
                1.0 if candidate.get("goal_kind") == "frontier" else 0.0,
                1.0 if candidate.get("goal_kind") == "charger" else 0.0,
                1.0 if candidate_type == "keep_plan" else 0.0,
                _signed_norm(candidate.get("score", 0.0), 32.0),
                self.explored_ratio,
                _norm(self.stuck_chain, 8),
                self._charge_slack(),
                _norm(self._frontier_gain(goal[0], goal[1]) if self._in_bounds(*goal) else 0.0, 16.0),
            ],
            dtype=np.float32,
        )

    def apply_decision(self, candidate_idx, path_style_idx):
        if not self.current_decision_candidates:
            self.teacher_candidate_idx = 0
            self.teacher_path_style_idx = 1
            return
        candidate_idx = int(np.clip(candidate_idx, 0, len(self.current_decision_candidates) - 1))
        path_style_idx = int(np.clip(path_style_idx, 0, Config.PATH_STYLE_DIM - 1))
        candidate = self.current_decision_candidates[candidate_idx]
        self.current_path_style = path_style_idx
        self.current_path_style_name = self.PATH_STYLE_NAMES[path_style_idx]
        self._commit_candidate(candidate)
        self.pending_action = self._compute_executor_action(self._legal_action)

    def _commit_candidate(self, candidate):
        dist = self._last_dist_map
        charger_dist = self._charger_dist_map
        goal_changed = candidate.get("goal_kind") != self.goal_kind or tuple(candidate.get("goal", self.cur_pos)) != self.goal

        candidate_type = candidate.get("type")
        if candidate_type == "keep_plan":
            pass
        elif candidate_type == "clean_region":
            self.active_charger_id = -1
            self.active_charger_rect = None
            self.active_charger_spoke = None
            region = self._region_by_id(candidate.get("region_id", -1))
            if region is not None:
                self.planner_mode = "clean_region"
                self._bind_active_region(region, preserve_goal=candidate.get("goal"))
                self.region_lock_kind = "dirty" if region["type"] != "bridge" else "frontier"
                self.region_lock_center = tuple(region["anchor"])
        elif candidate_type == "explore_mouth":
            self.active_charger_id = -1
            self.active_charger_rect = None
            self.active_charger_spoke = None
            region = self._region_by_id(candidate.get("region_id", -1))
            self.planner_mode = "explore"
            if region is not None:
                self.active_mouth_id = int(region["mouth_id"])
                self.region_lock_kind = "frontier"
                self.region_lock_center = tuple(region["entry"])
            else:
                self._clear_active_region()
        elif candidate_type == "charger":
            self.charge_mode = True
            self.planner_mode = "return_charge"
            self._clear_region_lock()
            self._clear_active_region()
            self._clear_launch_lock()
            charger_rect = candidate.get("charger_rect")
            self.active_charger_id = int(candidate.get("charger_id", -1))
            self.active_charger_rect = dict(charger_rect) if isinstance(charger_rect, dict) else None
            spoke = candidate.get("charger_spoke")
            self.active_charger_spoke = tuple(spoke) if spoke is not None else None
        else:
            self.planner_mode = "explore"
            self.active_charger_id = -1
            self.active_charger_rect = None
            self.active_charger_spoke = None
            self._clear_active_region()

        self.goal_kind = candidate.get("goal_kind")
        self.goal = tuple(candidate.get("goal", self.cur_pos))
        self.last_plan_step = self.step_no
        if goal_changed:
            self.plan_churn_count += 1
            self.goal_set_step = self.step_no
            self.goal_progress_step = self.step_no

        if self.goal is None:
            self.path = []
            self.last_plan_risk = float(self._npc_zone_penalty(self.cur_pos))
            return

        self.path = self._plan_weighted_path(self.goal, allow_unknown=True)
        if not self.path and self._last_prev_x is not None and self._last_prev_z is not None:
            self.path = self._reconstruct_path(self.goal, self._last_prev_x, self._last_prev_z)
        self.path = self.path[: Config.MAX_TRACKED_PATH]
        self.last_plan_risk = self._decision_risk_signature()

    def _compute_executor_action(self, legal_action):
        launch_act = self._launch_lock_action(legal_action)
        if launch_act is not None:
            self.pending_prob = np.zeros((Config.ACTION_DIM,), dtype=np.float32)
            self.pending_prob[launch_act] = 1.0
            return int(launch_act)

        commit_act = self._path_commit_action(legal_action)
        if commit_act is not None:
            self.pending_prob = np.zeros((Config.ACTION_DIM,), dtype=np.float32)
            self.pending_prob[commit_act] = 1.0
            return int(commit_act)

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

    def _region_by_id(self, region_id):
        region_id = int(region_id)
        if 0 <= region_id < len(self.regions):
            region = self.regions[region_id]
            if int(region.get("id", -1)) == region_id:
                return region
        return None

    def _region_spine_ratio(self, region):
        if region is None or not region.get("cells"):
            return 0.0
        total = float(len(region["cells"]))
        if total <= 0.0:
            return 0.0
        spine = 0.0
        for x, z in region["cells"]:
            if bool(self.transit_spine_mask[x, z]):
                spine += 1.0
        return float(spine / total)

    def _region_blocked_ratio(self, region):
        if region is None or not region.get("cells"):
            return 0.0
        total = float(len(region["cells"]))
        if total <= 0.0:
            return 0.0
        blocked = 0.0
        for pos in region["cells"]:
            if tuple(pos) in self.blocked_cells:
                blocked += 1.0
        return float(blocked / total)

    def _region_npc_clean_ratio(self, region):
        if region is None or not region.get("cells"):
            return 0.0
        total = float(len(region["cells"]))
        if total <= 0.0:
            return 0.0
        value = 0.0
        for x, z in region["cells"]:
            value += float(self.npc_cleaned[x, z])
        return float(value / total)

    def _charger_cells_for(self, charger):
        out = []
        for x in range(int(charger["x"]), int(charger["x"] + charger["w"])):
            for z in range(int(charger["z"]), int(charger["z"] + charger["h"])):
                if self._in_bounds(x, z):
                    out.append((x, z))
        return out

    def _charger_key(self, charger):
        if charger is None:
            return -1
        return int(charger.get("id", -1))

    def _battery_scaled_steps(self, base_steps, minimum=1):
        scale = float(self.battery_max) / 200.0 if float(self.battery_max) > 0.0 else 1.0
        scale = float(np.clip(scale, 0.6, 1.8))
        return max(int(minimum), int(round(float(base_steps) * scale)))

    def _charger_center(self, charger):
        return (
            float(int(charger["x"]) + (int(charger["w"]) - 1) / 2.0),
            float(int(charger["z"]) + (int(charger["h"]) - 1) / 2.0),
        )

    def _spoke_alignment(self, delta, spoke):
        if delta is None or spoke is None:
            return 0.0
        return float(np.clip((float(delta[0]) * float(spoke[0]) + float(delta[1]) * float(spoke[1])) / 2.0, -1.0, 1.0))

    def _charger_geometry_scale(self, pos):
        x, z = pos
        if not self._in_bounds(x, z):
            return float(Config.CHARGER_ZONE_REPEAT_CORNER_SCALE)

        blocked = set()
        for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            nz = z + dz
            if not self._in_bounds(nx, nz) or int(self.map_state[nx, nz]) == self.OBSTACLE:
                blocked.add((dx, dz))

        if (
            ((1, 0) in blocked and (0, 1) in blocked)
            or ((1, 0) in blocked and (0, -1) in blocked)
            or ((-1, 0) in blocked and (0, 1) in blocked)
            or ((-1, 0) in blocked and (0, -1) in blocked)
        ):
            return float(Config.CHARGER_ZONE_REPEAT_CORNER_SCALE)
        if blocked:
            return float(Config.CHARGER_ZONE_REPEAT_WALL_SCALE)
        return float(Config.CHARGER_ZONE_REPEAT_CENTER_SCALE)

    def _charger_zone_repeat_penalty(self, pos, charger=None):
        if charger is None:
            if self.charge_mode or self.goal_kind == "charger":
                charger = self.active_charger_rect
            elif self._in_post_charge_window():
                charger = self.last_charger_rect
        if charger is None or not self._in_bounds(*pos):
            return 0.0

        dist = self._dist_to_charger_rect(pos, charger)
        if dist > Config.POST_CHARGE_HALO_RADIUS:
            return 0.0

        x, z = pos
        depth_scale = (Config.POST_CHARGE_HALO_RADIUS + 1 - dist) / max(Config.POST_CHARGE_HALO_RADIUS + 1, 1)
        repeat_load = Config.CHARGER_ZONE_REPEAT_VISIT_WEIGHT * min(float(self.visit_count[x, z]), float(Config.MAX_VISIT_CLIP))
        repeat_load += Config.CHARGER_ZONE_REPEAT_TRANSIT_WEIGHT * min(float(self.clean_pass_count[x, z]), float(Config.MAX_TRANSIT_CLIP))
        repeat_load += Config.CHARGER_ZONE_REPEAT_DENSITY_WEIGHT * min(self._local_repeat_density(pos, radius=1), 20.0)
        return float(depth_scale * self._charger_geometry_scale(pos) * repeat_load)

    def _charger_spoke_usage_penalty(self, charger, spoke):
        if charger is None or spoke is None:
            return 0.0
        key = (self._charger_key(charger), int(spoke[0]), int(spoke[1]))
        return float(self.charger_spoke_usage.get(key, 0)) * float(Config.CHARGER_SPOKE_REUSE_PENALTY)

    def _choose_charger_goal_for_spoke(self, charger, dist, spoke):
        center_x, center_z = self._charger_center(charger)
        best_goal = None
        best_score = -1e9
        for cell in self._charger_cells_for(charger):
            d = int(dist[cell[0], cell[1]]) if dist is not None else self._chebyshev(self.cur_pos, cell)
            if d < 0:
                continue
            projection = (float(cell[0]) - center_x) * float(spoke[0]) + (float(cell[1]) - center_z) * float(spoke[1])
            score = 1.8 * projection - 0.16 * float(d)
            score -= Config.CHARGER_ZONE_REPEAT_GOAL_WEIGHT * self._charger_zone_repeat_penalty(cell, charger)
            score -= 0.18 * float(self.npc_cleaned[cell[0], cell[1]])
            if cell in self.blocked_cells:
                score -= 0.45
            if score > best_score:
                best_score = score
                best_goal = tuple(cell)
        return best_goal

    def _score_charger_spoke(self, charger, spoke, dist):
        center_x, center_z = self._charger_center(charger)
        score = -self._charger_spoke_usage_penalty(charger, spoke)
        reach = 0.0
        for step in range(1, Config.CHARGER_SPOKE_SCAN_RADIUS + 1):
            px = int(round(center_x + float(spoke[0]) * step))
            pz = int(round(center_z + float(spoke[1]) * step))
            if not self._in_bounds(px, pz):
                break
            if int(self.map_state[px, pz]) == self.OBSTACLE:
                break
            pos = (px, pz)
            reach += 1.0
            if int(self.map_state[px, pz]) == self.UNKNOWN:
                score += 1.15
            elif int(self.map_state[px, pz]) == self.DIRTY:
                score += 0.85
            score += 0.32 * self._frontier_gain(px, pz)
            score -= 0.10 * float(self.visit_count[px, pz])
            score -= 0.14 * float(self.clean_pass_count[px, pz])
            score -= 0.05 * float(self._npc_zone_penalty(pos))
            score -= Config.NPC_CLEAN_PENALTY * 0.30 * float(self.npc_cleaned[px, pz])
            if pos in self.blocked_cells:
                score -= 0.55
            score -= 0.30 * self._charger_zone_repeat_penalty(pos, charger)
            if dist is not None:
                d = int(dist[px, pz])
                if d >= 0:
                    score -= 0.015 * float(d)
        return float(score + 0.22 * reach)

    def _select_charger_spoke(self, charger, dist):
        best_spoke = None
        best_score = -1e9
        for spoke in self.ACTION_DELTAS:
            score = self._score_charger_spoke(charger, spoke, dist)
            if score > best_score:
                best_score = score
                best_spoke = tuple(spoke)
        return best_spoke if best_spoke is not None else (1, 0)

    def _clear_launch_lock(self):
        self.launch_lock_source_rect = None
        self.launch_lock_spoke = None

    def _launch_lock_active(self):
        if self.charge_mode or self.launch_lock_source_rect is None or self.launch_lock_spoke is None:
            return False
        cur_dist = self._dist_to_charger_rect(self.cur_pos, self.launch_lock_source_rect)
        if cur_dist >= self._battery_scaled_steps(Config.CHARGER_LAUNCH_MIN_DIST, minimum=3):
            self._clear_launch_lock()
            return False
        return True

    def _launch_lock_action(self, legal_action):
        if not self._launch_lock_active():
            return None

        cur_dist = self._dist_to_charger_rect(self.cur_pos, self.launch_lock_source_rect)
        best_forward = None
        best_forward_score = -1e9
        best_hold = None
        best_hold_score = -1e9

        for act, ok in enumerate(legal_action):
            if not ok:
                continue
            dx, dz = self.ACTION_DELTAS[act]
            next_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
            if not self._in_bounds(*next_pos):
                continue
            if self._npc_zone_hard_block(next_pos) or self._npc_dynamic_hard_block(next_pos):
                continue

            next_dist = self._dist_to_charger_rect(next_pos, self.launch_lock_source_rect)
            outward = float(next_dist - cur_dist)
            align = self._spoke_alignment((dx, dz), self.launch_lock_spoke)
            novelty = 1.0 / (1.0 + float(self.visit_count[next_pos[0], next_pos[1]]) + 1.2 * float(self.clean_pass_count[next_pos[0], next_pos[1]]))
            score = Config.CHARGER_LAUNCH_OUTWARD_WEIGHT * outward
            score += Config.CHARGER_SPOKE_ALIGN_WEIGHT * align
            score += Config.CHARGER_SPOKE_UNVISITED_BONUS * novelty
            score -= 0.12 * float(self._npc_zone_penalty(next_pos))
            score -= 0.45 * self._charger_zone_repeat_penalty(next_pos, self.launch_lock_source_rect)
            score -= 0.70 * (1.0 if next_pos in self.blocked_cells else 0.0)
            score -= 0.60 * float(self.npc_cleaned[next_pos[0], next_pos[1]])

            if outward > 0.0:
                if score > best_forward_score:
                    best_forward_score = score
                    best_forward = act
            else:
                score -= Config.CHARGER_LAUNCH_MONOTONIC_PENALTY * (1.0 if outward < 0.0 else 0.25)
                if score > best_hold_score:
                    best_hold_score = score
                    best_hold = act

        if best_forward is not None:
            return int(best_forward)
        return int(best_hold) if best_hold is not None else None

    def _charger_approach_bonus(self, cur_pos, next_pos):
        if not (self.charge_mode or self.goal_kind == "charger"):
            return 0.0
        if self.active_charger_rect is None or self.active_charger_spoke is None:
            return 0.0
        cur_dist = self._dist_to_charger_rect(cur_pos, self.active_charger_rect)
        next_dist = self._dist_to_charger_rect(next_pos, self.active_charger_rect)
        align = self._spoke_alignment(
            (next_pos[0] - cur_pos[0], next_pos[1] - cur_pos[1]),
            (-int(self.active_charger_spoke[0]), -int(self.active_charger_spoke[1])),
        )
        bonus = 0.45 * align * max(0.0, float(cur_dist - next_dist) + 0.25)
        bonus -= 0.18 * self._charger_zone_repeat_penalty(next_pos, self.active_charger_rect)
        return float(bonus)

    def _charger_retry_margin_for_goal(self, goal):
        path = self._plan_weighted_path(goal, allow_unknown=True)
        if not path:
            return float(self._battery_scaled_steps(Config.RETURN_CHARGE_BUFFER))
        turns = 0
        narrow = 0
        blocked = 0
        last_delta = None
        for prev, nxt in zip([self.cur_pos] + path[:-1], path):
            delta = (nxt[0] - prev[0], nxt[1] - prev[1])
            if last_delta is not None and delta != last_delta:
                turns += 1
            last_delta = delta
            if self._boundary_bonus(nxt) > 0.8:
                narrow += 1
            if nxt in self.blocked_cells:
                blocked += 1
        return (
            self._battery_scaled_steps(Config.RETURN_CHARGE_BUFFER)
            + Config.CHARGE_RETRY_MARGIN_SCALE * turns
            + Config.CHARGE_RETRY_MARGIN_NARROW * narrow
            + Config.CHARGE_RETRY_MARGIN_BLOCKED * blocked
        )

    def _edge_bonus(self, pos):
        x, z = pos
        margin = min(x, z, Config.GRID_SIZE - 1 - x, Config.GRID_SIZE - 1 - z)
        return float(np.clip((12.0 - margin) / 12.0, 0.0, 1.0))

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
        self.decision_event = False
        self._decay_blocked_cells()

        pos = hero.get("pos") or env_info.get("pos") or {}
        self.cur_pos = (
            int(pos.get("x", self.cur_pos[0])),
            int(pos.get("z", self.cur_pos[1])),
        )
        if self.goal is not None and self._chebyshev(self.cur_pos, self.goal) < self._chebyshev(self.last_pos, self.goal):
            self.goal_progress_step = self.step_no
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

        self._current_step_cleaned = set()

        map_info = observation.get("map_info")
        if map_info is not None:
            self._view_map = np.array(map_info, dtype=np.int8)
            self._integrate_local_view()
            self._update_stuck_state()

        cleaned_cells = env_info.get("step_cleaned_cells") or []
        for cell in cleaned_cells:
            x = int(cell.get("x", -1))
            z = int(cell.get("z", -1))
            if self._in_bounds(x, z):
                self.map_state[x, z] = self.CLEAN
                self._current_step_cleaned.add((x, z))
                self.hero_cleaned[x, z] = 1
                self.npc_cleaned[x, z] = 0

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
        self._update_runtime_counters()

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
        self.explored_ratio = float(np.mean(~unknown_mask))
        self._plannable_unknown_map = self.map_state != self.OBSTACLE
        self._plannable_known_map = clean_mask | dirty_mask

        self._known_integral = self._build_integral_grid(self._plannable_known_map)
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

    def _update_stuck_state(self):
        if self.cur_pos != self.last_pos or self.last_action < 0:
            self.stuck_chain = 0
            return

        self.stuck_chain = min(self.stuck_chain + 1, 8)
        attempted = self._attempted_next_pos()
        if attempted is not None:
            self.blocked_cells[attempted] = Config.BLOCKED_CELL_TTL
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

    def _attempted_next_pos(self):
        if self.last_action < 0 or self.last_action >= Config.ACTION_DIM:
            return None
        dx, dz = self.ACTION_DELTAS[int(self.last_action)]
        next_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
        if not self._in_bounds(*next_pos):
            return None
        return next_pos

    def _decay_blocked_cells(self):
        if not self.blocked_cells:
            return
        next_map = {}
        for pos, ttl in self.blocked_cells.items():
            ttl = int(ttl) - 1
            if ttl > 0:
                next_map[pos] = ttl
        self.blocked_cells = next_map

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
                prev = int(self.map_state[gx, gz])
                val = int(self._view_map[row, col])
                self.map_state[gx, gz] = val
                if prev == self.DIRTY and val == self.CLEAN and (gx, gz) not in self._current_step_cleaned:
                    self.npc_cleaned[gx, gz] = 1

    def _update_chargers(self, organs):
        self.chargers = []
        for organ in organs:
            if int(organ.get("sub_type", 0)) != 1:
                continue

            config_id = int(organ.get("config_id", len(self.chargers)))
            x = int(organ.get("pos", {}).get("x", 0))
            z = int(organ.get("pos", {}).get("z", 0))
            w = max(int(organ.get("w", 1)), 1)
            h = max(int(organ.get("h", 1)), 1)
            self.chargers.append({"id": config_id, "x": x, "z": z, "w": w, "h": h})

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

    def _charger_for_pos(self, pos):
        x, z = pos
        for charger in self.chargers:
            if charger["x"] <= x < charger["x"] + charger["w"] and charger["z"] <= z < charger["z"] + charger["h"]:
                return charger
        return None

    def _exit_charge_mode(self):
        self.charge_mode = False
        self.planner_mode = "explore"
        charger = self._charger_for_pos(self.cur_pos)
        if charger is not None:
            self.last_charger_id = int(charger.get("id", -1))
            self.last_charger_rect = dict(charger)
            self.post_charge_until_step = self.step_no + Config.POST_CHARGE_LOCK_STEPS
            spoke = self.active_charger_spoke if self.active_charger_spoke is not None else self._select_charger_spoke(charger, self._last_dist_map)
            self.launch_lock_source_rect = dict(charger)
            self.launch_lock_spoke = tuple(spoke) if spoke is not None else None
            if self.launch_lock_spoke is not None:
                key = (self.last_charger_id, int(self.launch_lock_spoke[0]), int(self.launch_lock_spoke[1]))
                self.charger_spoke_usage[key] = self.charger_spoke_usage.get(key, 0) + 1
        else:
            self._clear_launch_lock()
        self.goal = None
        self.goal_kind = None
        self.path = []
        self.goal_set_step = -999
        self.goal_progress_step = self.step_no
        self._clear_active_region()
        self.active_charger_id = -1
        self.active_charger_rect = None
        self.active_charger_spoke = None

    def _in_post_charge_window(self):
        return (
            not self.charge_mode
            and self.last_charger_rect is not None
            and self.step_no <= self.post_charge_until_step
        )

    def _post_charge_goal_penalty(self, pos):
        if not self._in_post_charge_window():
            return 0.0
        dist = self._dist_to_charger_rect(pos, self.last_charger_rect)
        if dist > Config.POST_CHARGE_HALO_RADIUS:
            return 0.0
        remain = Config.POST_CHARGE_HALO_RADIUS + 1 - dist
        scale = remain / max(Config.POST_CHARGE_HALO_RADIUS + 1, 1)
        penalty = Config.POST_CHARGE_HALO_PENALTY * scale
        penalty += Config.CHARGER_ZONE_REPEAT_GOAL_WEIGHT * self._charger_zone_repeat_penalty(pos, self.last_charger_rect)
        return float(penalty)

    def _post_charge_action_bonus(self, next_pos):
        if not self._in_post_charge_window():
            return 0.0
        cur_dist = self._dist_to_charger_rect(self.cur_pos, self.last_charger_rect)
        next_dist = self._dist_to_charger_rect(next_pos, self.last_charger_rect)
        bonus = Config.POST_CHARGE_OUTWARD_BONUS * float(next_dist - cur_dist)
        if self.launch_lock_spoke is not None:
            delta = (next_pos[0] - self.cur_pos[0], next_pos[1] - self.cur_pos[1])
            bonus += 0.85 * Config.CHARGER_SPOKE_ALIGN_WEIGHT * self._spoke_alignment(delta, self.launch_lock_spoke)
        novelty = 1.0 / (1.0 + float(self.visit_count[next_pos[0], next_pos[1]]) + 1.2 * float(self.clean_pass_count[next_pos[0], next_pos[1]]))
        bonus += 0.55 * Config.CHARGER_SPOKE_UNVISITED_BONUS * novelty
        bonus -= 0.45 * self._charger_zone_repeat_penalty(next_pos, self.last_charger_rect)
        if next_dist <= Config.POST_CHARGE_HALO_RADIUS:
            bonus -= 0.60 * float(Config.POST_CHARGE_HALO_RADIUS + 1 - next_dist)
        return float(bonus)

    def _explore_push(self):
        if self.charge_mode:
            return 0.0
        explore_gap = max(0.0, Config.EXPLORE_EARLY_RATIO - self.explored_ratio)
        ratio_push = explore_gap / max(Config.EXPLORE_EARLY_RATIO, 1e-6)
        slack = self._charge_slack()
        slack_push = max(0.0, slack - Config.EXPLORE_SLACK_THRESHOLD)
        slack_push /= max(1.0 - Config.EXPLORE_SLACK_THRESHOLD, 1e-6)
        push = max(ratio_push, 0.85 * slack_push)
        if not self.chargers:
            push = max(push, 0.55)
        return float(np.clip(push, 0.0, 1.0))

    def _goal_sticky_bonus(self, kind, goal):
        if goal is None or kind != self.goal_kind or goal != self.goal:
            return 0.0
        freshness = max(0.0, 1.0 - (self.step_no - self.goal_set_step) / max(Config.PLAN_COMMIT_STEPS, 1))
        return Config.GOAL_STICKY_BONUS * (0.70 + 0.30 * freshness)

    def _goal_still_valid(self):
        if self.goal is None:
            return False
        gx, gz = self.goal
        if not self._in_bounds(gx, gz):
            return False
        if self.goal_kind == "dirty":
            return self.map_state[gx, gz] == self.DIRTY
        if self.goal_kind == "frontier":
            return self._has_unknown_neighbor((gx, gz)) or self._frontier_gain(gx, gz) > 0
        if self.goal_kind == "charger":
            return not self.on_charger
        return True

    def _goal_making_progress(self):
        last_progress = max(self.goal_progress_step, self.goal_set_step)
        return self.step_no - last_progress <= Config.PLAN_PROGRESS_STALL_STEPS

    def _decision_risk_signature(self):
        current_risk = float(self._npc_zone_penalty(self.cur_pos))
        if not self.path:
            return current_risk
        lookahead = self.path[:4]
        if not lookahead:
            return current_risk
        future_risk = max(float(self._npc_zone_penalty(pos)) for pos in lookahead)
        return max(current_risk, future_risk)

    def _risk_spike_requires_replan(self):
        threshold = float(Config.DECISION_EVENT_RISK_DELTA)
        if threshold <= 0.0:
            return False
        return self._decision_risk_signature() - float(self.last_plan_risk) >= threshold

    def _path_head_advances_goal(self):
        if not self.path:
            return False
        next_pos = self.path[0]
        if self.charge_mode or self.goal_kind == "charger":
            return self._charger_dist_at(next_pos, self._charger_dist_map) < self._current_charger_distance()
        if self.goal is not None and self._chebyshev(next_pos, self.goal) < self._chebyshev(self.cur_pos, self.goal):
            return True
        if self.region_lock_kind in {"dirty", "frontier"} and self.region_lock_center is not None:
            return self._distance_to_region(next_pos, self.region_lock_kind, self.region_lock_center) < self._distance_to_region(
                self.cur_pos, self.region_lock_kind, self.region_lock_center
            )
        return False

    def _should_hold_plan(self):
        if not self.path or self.goal is None or self.stuck_chain > 0:
            return False
        if self._path_is_npc_unsafe() or not self._goal_still_valid():
            return False
        plan_age = self.step_no - self.last_plan_step
        if plan_age <= Config.PLAN_COMMIT_STEPS:
            return True
        if self._goal_making_progress() and self._path_head_advances_goal():
            return True
        if (
            self.region_lock_kind in {"dirty", "frontier"}
            and self.region_lock_center is not None
            and self._should_keep_region_lock()
            and self._distance_to_region(self.path[0], self.region_lock_kind, self.region_lock_center) == 0
            and self._goal_making_progress()
        ):
            return True
        return False

    def _adjacent_dirty_detour(self, path_next, legal_action):
        for act, ok in enumerate(legal_action):
            if not ok:
                continue
            dx, dz = self.ACTION_DELTAS[act]
            next_pos = (self.cur_pos[0] + dx, self.cur_pos[1] + dz)
            if next_pos == path_next or not self._in_bounds(*next_pos):
                continue
            if not self._is_local_move_passable(dx, dz):
                continue
            if self.map_state[next_pos[0], next_pos[1]] == self.DIRTY:
                return True
        return False

    def _path_commit_action(self, legal_action):
        if not self._should_hold_plan() or not self.path:
            return None
        next_pos = self.path[0]
        act = self._pos_to_action(next_pos)
        if act is None or not legal_action[act]:
            return None
        if self._npc_dynamic_hard_block(next_pos):
            return None
        if not self.charge_mode and self.goal_kind != "charger" and self._adjacent_dirty_detour(next_pos, legal_action):
            if self.planner_mode != "clean_region" or self._active_region_object() is None:
                return None
        return act

    def _clear_active_region(self):
        region = self._active_region_object()
        if region is not None and self.active_region_type in {"room", "tail"}:
            dirty_mass, frontier_mass = self._region_current_masses(region)
            region_id = int(region.get("id", -1))
            if dirty_mass <= 0.0 and frontier_mass <= 0.0 and region_id not in self.completed_region_ids:
                self.completed_region_count += 1
                self.completed_region_ids.add(region_id)
        self.active_region_id = -1
        self.active_region_type = None
        self.active_region_entry = None
        self.active_region_anchor = None
        self.active_region_goal = None
        self.active_region_cells = []
        self.active_region_parent_charger = -1
        self.active_region_axis = 0
        self.active_region_sign = 1
        self.active_mouth_id = -1
        self.active_cover_sequence = []
        self.active_cover_strip_ids = []
        self.active_cover_index = 0
        self.active_cover_strip = 0

    def _bind_active_region(self, region, preserve_goal=None):
        self.active_region_id = int(region["id"])
        self.active_region_type = region["type"]
        self.active_region_entry = tuple(region["entry"])
        self.active_region_anchor = tuple(region["anchor"])
        self.active_region_parent_charger = int(region["charger_id"])
        self.active_region_axis = int(region["axis"])
        self.active_region_sign = int(region["sign"])
        self.active_region_cells = list(region["cells"])
        self.active_cover_sequence, self.active_cover_strip_ids = self._build_cover_sequence(region)
        self.active_cover_index = 0
        self.active_cover_strip = 0
        if preserve_goal is not None and self._in_bounds(*preserve_goal) and int(self.region_mask[preserve_goal[0], preserve_goal[1]]) == self.active_region_id:
            self.active_region_goal = tuple(preserve_goal)
            self.active_cover_index = self._find_cover_index(preserve_goal)
            self.active_cover_strip = self._cover_strip_at_index(self.active_cover_index)
        else:
            self.active_region_goal = self._next_cover_goal(region) or tuple(region["anchor"])
        self.active_mouth_id = int(region["mouth_id"])

    def _nearest_charger_id(self, pos):
        best_id = -1
        best_dist = 10**9
        for charger in self.chargers:
            dist = self._dist_to_charger_rect(pos, charger)
            if dist < best_dist:
                best_dist = dist
                best_id = int(charger.get("id", -1))
        return best_id

    def _known_density(self, pos, radius=2):
        x, z = pos
        x0 = max(0, x - radius)
        x1 = min(Config.GRID_SIZE, x + radius + 1)
        z0 = max(0, z - radius)
        z1 = min(Config.GRID_SIZE, z + radius + 1)
        return int(self._window_sum(self._known_integral, x0, x1, z0, z1))

    def _dilate_mask(self, mask, radius):
        if radius <= 0 or not np.any(mask):
            return mask.copy()
        expanded = mask.copy()
        for x, z in np.argwhere(mask):
            x0 = max(0, int(x) - radius)
            x1 = min(Config.GRID_SIZE, int(x) + radius + 1)
            z0 = max(0, int(z) - radius)
            z1 = min(Config.GRID_SIZE, int(z) + radius + 1)
            expanded[x0:x1, z0:z1] = True
        return expanded

    def _spine_penalty(self, pos):
        x, z = int(pos[0]), int(pos[1])
        if not self._in_bounds(x, z):
            return 1.0
        if self._is_on_charger((x, z)):
            return 0.0
        return 1.0 if bool(self.transit_spine_mask[x, z]) else 0.0

    def _dynamic_return_buffer(self, pos, charger_dist=None):
        x, z = int(pos[0]), int(pos[1])
        buffer = float(self._battery_scaled_steps(Config.RETURN_CHARGE_BUFFER))
        if self._unknown_density((x, z), radius=2) >= 6:
            buffer += float(Config.DYNAMIC_RETURN_UNKNOWN_SURCHARGE)
        buffer += float(Config.DYNAMIC_RETURN_RISK_SCALE) * min(self._npc_zone_penalty((x, z)), 36.0)
        if self.planner_mode == "clean_region":
            buffer += float(Config.DYNAMIC_RETURN_REGION_SURCHARGE)
        if charger_dist is None or charger_dist < 0:
            if self._charger_dist_map is None or int(self._charger_dist_map[x, z]) < 0:
                buffer += float(Config.DYNAMIC_RETURN_HEURISTIC_SURCHARGE)
        return int(np.ceil(buffer))

    def _region_return_budget(self, pos, travel_cost, charger_dist):
        back_cost = self._charger_dist_at(pos, charger_dist)
        buffer = self._dynamic_return_buffer(pos, back_cost)
        return int(travel_cost + back_cost + buffer)

    def _region_depth_progress(self, region, pos):
        axis = int(region["axis"])
        sign = int(region["sign"])
        entry = tuple(region["entry"])
        value = int(pos[axis]) - int(entry[axis])
        return sign * value

    def _active_region_object(self):
        if 0 <= self.active_region_id < len(self.regions):
            region = self.regions[self.active_region_id]
            if int(region["id"]) == self.active_region_id:
                return region
        return None

    def _region_current_masses(self, region):
        if region is None:
            return 0.0, 0.0
        dirty_mass = 0.0
        frontier_mass = 0.0
        for x, z in region["cells"]:
            if int(self.map_state[x, z]) == self.DIRTY:
                dirty_mass += 1.0
            if self._has_unknown_neighbor((x, z)):
                frontier_mass += 1.0
        return float(dirty_mass), float(frontier_mass)

    def _region_active_thresholds(self, region):
        if region is None:
            return float(Config.ACTIVE_REGION_DIRTY_MIN), float(Config.ACTIVE_REGION_FRONTIER_MIN)
        region_type = region.get("type")
        if region_type == "room":
            return float(Config.ACTIVE_ROOM_DIRTY_MIN), float(Config.ACTIVE_ROOM_FRONTIER_MIN)
        if region_type == "tail":
            return float(Config.ACTIVE_TAIL_DIRTY_MIN), float(Config.ACTIVE_TAIL_FRONTIER_MIN)
        return float(Config.ACTIVE_REGION_DIRTY_MIN), float(Config.ACTIVE_REGION_FRONTIER_MIN)

    def _region_total_mass(self, region):
        dirty_mass, frontier_mass = self._region_current_masses(region)
        return float(dirty_mass + frontier_mass)

    def _active_region_force_hold(self, region):
        if region is None or self.charge_mode:
            return False
        total_mass = self._region_total_mass(region)
        if total_mass <= 0.0:
            return False
        if self.step_no - self.goal_set_step <= Config.ACTIVE_REGION_MIN_COMMIT_STEPS:
            return True
        if self._in_bounds(*self.cur_pos) and int(self.region_mask[self.cur_pos[0], self.cur_pos[1]]) == int(region["id"]):
            return total_mass >= 1.0
        return False

    def _region_entry_penalty(self, region, pos):
        if region is None:
            return 0.0
        dist = self._chebyshev(pos, tuple(region["entry"]))
        gap = Config.REGION_ENTRY_CLEAR_RADIUS - dist + 1
        if gap <= 0:
            return 0.0
        return float(Config.REGION_ENTRY_CLEAR_PENALTY * gap)

    def _region_room_value(self, region, pos):
        if region is None or region["type"] == "bridge":
            return 0.0
        x, z = pos
        bonus = Config.REGION_ROOM_DIRTY_DENSITY_WEIGHT * float(self._dirty_density(x, z))
        bonus += Config.REGION_ROOM_FRONTIER_DENSITY_WEIGHT * float(self._frontier_gain(x, z))
        if self._in_bounds(x, z) and int(self.region_mask[x, z]) == int(region["id"]) and self._spine_penalty(pos) <= 0.0:
            bonus += Config.CLEAN_ROOM_INTERIOR_BONUS
        return float(bonus)

    def _build_cover_sequence(self, region):
        if region is None or region.get("type") == "bridge":
            return [], []
        axis = int(region["axis"])
        sign = int(region["sign"])
        entry = tuple(region["entry"])
        strip_axis = 1 - axis
        strips = {}
        for pos in region["cells"]:
            strips.setdefault(int(pos[strip_axis]), []).append(tuple(pos))
        ordered_strips = sorted(strips.items(), key=lambda item: (abs(item[0] - entry[strip_axis]), item[0]))
        sequence = []
        strip_ids = []
        for strip_idx, (_, cells) in enumerate(ordered_strips):
            if len(cells) < Config.COVER_STRIP_MIN_CELLS and region["type"] == "room":
                continue
            reverse = bool(strip_idx % 2)
            cells = sorted(
                cells,
                key=lambda pos: (
                    sign * (int(pos[axis]) - int(entry[axis])),
                    abs(int(pos[strip_axis]) - int(entry[strip_axis])),
                ),
                reverse=not reverse,
            )
            for pos in cells:
                if self.map_state[pos[0], pos[1]] == self.OBSTACLE:
                    continue
                sequence.append(pos)
                strip_ids.append(strip_idx)
        return sequence, strip_ids

    def _find_cover_index(self, goal):
        if goal is None:
            return 0
        for idx, pos in enumerate(self.active_cover_sequence):
            if tuple(pos) == tuple(goal):
                return idx
        return 0

    def _cover_strip_at_index(self, idx):
        if 0 <= int(idx) < len(self.active_cover_strip_ids):
            return int(self.active_cover_strip_ids[int(idx)])
        return 0

    def _next_strip_cover_goal(self):
        if not self.active_cover_sequence or not self.active_cover_strip_ids:
            return None
        current_strip = int(self.active_cover_strip)
        for idx in range(self.active_cover_index + 1, len(self.active_cover_sequence)):
            strip_id = self._cover_strip_at_index(idx)
            if strip_id <= current_strip:
                continue
            pos = tuple(self.active_cover_sequence[idx])
            if self._cover_goal_has_work(pos):
                return pos
        return None

    def _cover_goal_has_work(self, pos):
        if not self._in_bounds(*pos):
            return False
        if int(self.map_state[pos[0], pos[1]]) == self.DIRTY:
            return True
        x, z = pos
        x0 = max(0, x - Config.COVER_TARGET_WORK_RADIUS)
        x1 = min(Config.GRID_SIZE, x + Config.COVER_TARGET_WORK_RADIUS + 1)
        z0 = max(0, z - Config.COVER_TARGET_WORK_RADIUS)
        z1 = min(Config.GRID_SIZE, z + Config.COVER_TARGET_WORK_RADIUS + 1)
        region_id = int(self.region_mask[x, z]) if self._in_bounds(x, z) else -1
        for nx in range(x0, x1):
            for nz in range(z0, z1):
                if int(self.region_mask[nx, nz]) != region_id:
                    continue
                if int(self.map_state[nx, nz]) == self.DIRTY or self._has_unknown_neighbor((nx, nz)):
                    return True
        return False

    def _next_cover_goal(self, region):
        if region is None:
            return None
        if not self.active_cover_sequence:
            self.active_cover_sequence, self.active_cover_strip_ids = self._build_cover_sequence(region)
            self.active_cover_index = 0
            self.active_cover_strip = 0
        lookahead_end = min(len(self.active_cover_sequence), self.active_cover_index + Config.COVER_SEQUENCE_LOOKAHEAD)
        for idx in range(self.active_cover_index, lookahead_end):
            pos = tuple(self.active_cover_sequence[idx])
            if self._cover_goal_has_work(pos):
                self.active_cover_index = idx
                self.active_cover_strip = self._cover_strip_at_index(idx)
                return pos
        for idx in range(self.active_cover_index, len(self.active_cover_sequence)):
            pos = tuple(self.active_cover_sequence[idx])
            if self._cover_goal_has_work(pos):
                self.active_cover_index = idx
                self.active_cover_strip = self._cover_strip_at_index(idx)
                return pos
        return None

    def _update_topology(self, dist=None, charger_dist=None):
        passable = self._plannable_known_map & (~self._npc_zone_hard_block_map) & (~self._npc_dynamic_hard_block_map)
        self.region_mask.fill(-1)
        self.transit_spine_mask[:] = False
        self.corridor_mask[:] = False
        self.room_mask[:] = False
        self.regions = []
        self.mouths = []

        if not np.any(passable):
            self._clear_active_region()
            return

        for x, z in np.argwhere(passable):
            pos = (int(x), int(z))
            openness = self._known_density(pos, radius=Config.TOPOLOGY_OPEN_RADIUS)
            if openness <= Config.TOPOLOGY_CORRIDOR_OPEN_THRESHOLD:
                self.corridor_mask[x, z] = True
            if openness >= Config.TOPOLOGY_ROOM_OPEN_THRESHOLD:
                self.room_mask[x, z] = True

        near_charger = np.zeros_like(passable, dtype=bool)
        for charger in self.chargers:
            x0 = max(0, int(charger["x"]) - 1)
            x1 = min(Config.GRID_SIZE, int(charger["x"] + charger["w"] + 1))
            z0 = max(0, int(charger["z"]) - 1)
            z1 = min(Config.GRID_SIZE, int(charger["z"] + charger["h"] + 1))
            near_charger[x0:x1, z0:z1] = True

        queue = deque()
        for sx, sz in self._charger_cells():
            if not self._in_bounds(sx, sz) or not passable[sx, sz]:
                continue
            if self.transit_spine_mask[sx, sz]:
                continue
            self.transit_spine_mask[sx, sz] = True
            queue.append((sx, sz))

        cardinals = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while queue:
            x, z = queue.popleft()
            for dx, dz in cardinals:
                nx = x + dx
                nz = z + dz
                if not self._in_bounds(nx, nz) or self.transit_spine_mask[nx, nz] or not passable[nx, nz]:
                    continue
                if near_charger[nx, nz] or self.corridor_mask[nx, nz]:
                    self.transit_spine_mask[nx, nz] = True
                    queue.append((nx, nz))

        self.transit_spine_mask = self._dilate_mask(self.transit_spine_mask, Config.TOPOLOGY_SPINE_HALO_RADIUS) & passable

        component_mask = passable & (~self.transit_spine_mask)
        visited = np.zeros_like(component_mask, dtype=bool)
        previous_entry = self.active_region_entry
        previous_goal = self.active_region_goal

        for sx, sz in np.argwhere(component_mask):
            sx = int(sx)
            sz = int(sz)
            if visited[sx, sz]:
                continue

            comp_queue = deque([(sx, sz)])
            visited[sx, sz] = True
            cells = []
            border_entries = set()
            dirty_mass = 0.0
            frontier_mass = 0.0
            risk_sum = 0.0
            min_x = max_x = sx
            min_z = max_z = sz

            while comp_queue:
                x, z = comp_queue.popleft()
                pos = (x, z)
                cells.append(pos)
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_z = min(min_z, z)
                max_z = max(max_z, z)
                risk_sum += self._npc_zone_penalty(pos)
                if int(self.map_state[x, z]) == self.DIRTY:
                    dirty_mass += 1.0
                if self._has_unknown_neighbor(pos):
                    frontier_mass += 1.0

                for dx, dz in cardinals:
                    nx = x + dx
                    nz = z + dz
                    if not self._in_bounds(nx, nz):
                        continue
                    if self.transit_spine_mask[nx, nz]:
                        border_entries.add(pos)
                        continue
                    if component_mask[nx, nz] and not visited[nx, nz]:
                        visited[nx, nz] = True
                        comp_queue.append((nx, nz))

            if not cells or not border_entries:
                continue

            if len(cells) < 8 and dirty_mass < 1.0 and frontier_mass < 1.0:
                continue

            if dist is not None:
                entry = min(border_entries, key=lambda p: (int(dist[p[0], p[1]]) if int(dist[p[0], p[1]]) >= 0 else 10**6, self._chebyshev(self.cur_pos, p)))
            else:
                entry = min(border_entries, key=lambda p: self._chebyshev(self.cur_pos, p))
            width = max_x - min_x + 1
            height = max_z - min_z + 1
            axis = 0 if width >= height else 1
            anchor = max(cells, key=lambda p: (self._chebyshev(entry, p), self._frontier_gain(p[0], p[1]), int(self.map_state[p[0], p[1]] == self.DIRTY)))
            sign = 1 if int(anchor[axis]) - int(entry[axis]) >= 0 else -1
            narrow = min(width, height)
            long_side = max(width, height)
            border_count = len(border_entries)

            if border_count >= 2 and narrow <= 4 and dirty_mass <= max(2.0, 0.10 * len(cells)) and frontier_mass <= Config.REGION_FRONTIER_TRIGGER:
                region_type = "bridge"
            elif narrow <= 4 and long_side >= Config.TAIL_REGION_MIN_DEPTH:
                region_type = "tail"
            else:
                region_type = "room"

            charger_id = self._nearest_charger_id(entry)
            region_id = len(self.regions)
            mouth_id = len(self.mouths)
            for x, z in cells:
                self.region_mask[x, z] = region_id

            region = {
                "id": region_id,
                "mouth_id": mouth_id,
                "type": region_type,
                "entry": entry,
                "anchor": anchor,
                "cells": cells,
                "area": len(cells),
                "dirty_mass": float(dirty_mass),
                "frontier_mass": float(frontier_mass),
                "risk_mean": float(risk_sum / max(len(cells), 1)),
                "bbox": (min_x, min_z, max_x, max_z),
                "axis": axis,
                "sign": sign,
                "charger_id": charger_id,
                "mouths": [tuple(p) for p in sorted(border_entries)],
            }
            self.regions.append(region)
            self.mouths.append(
                {
                    "id": mouth_id,
                    "region_id": region_id,
                    "center": entry,
                    "charger_id": charger_id,
                    "type": region_type,
                }
            )

        if previous_entry is not None and self._in_bounds(*previous_entry):
            region_id = int(self.region_mask[previous_entry[0], previous_entry[1]])
            if 0 <= region_id < len(self.regions):
                self._bind_active_region(self.regions[region_id], preserve_goal=previous_goal)
                return
        self._clear_active_region()

    def _region_still_viable(self, region, dist, charger_dist):
        if region is None:
            return False
        dirty_mass, frontier_mass = self._region_current_masses(region)
        region["dirty_mass"] = float(dirty_mass)
        region["frontier_mass"] = float(frontier_mass)

        min_dirty, min_frontier = self._region_active_thresholds(region)
        if region["type"] == "bridge" and not self._active_region_force_hold(region):
            min_dirty = max(min_dirty, float(Config.BRIDGE_REGION_DIRTY_TRIGGER))
            min_frontier = max(min_frontier, float(Config.BRIDGE_REGION_FRONTIER_TRIGGER))
        if not self._active_region_force_hold(region) and dirty_mass < min_dirty and frontier_mass < min_frontier:
            return False

        entry = tuple(region["entry"])
        anchor = tuple(region["anchor"])
        entry_dist = int(dist[entry[0], entry[1]]) if dist is not None else self._chebyshev(self.cur_pos, entry)
        anchor_dist = int(dist[anchor[0], anchor[1]]) if dist is not None else self._chebyshev(self.cur_pos, anchor)
        if entry_dist < 0 and anchor_dist < 0:
            return False

        target = anchor if dirty_mass >= frontier_mass else entry
        travel_cost = max(entry_dist, anchor_dist, 0)
        return self._region_return_budget(target, travel_cost, charger_dist) < self.battery

    def _pick_best_region(self, dist, charger_dist):
        preferred = []
        bridge = []
        for region in self.regions:
            dirty_mass, frontier_mass = self._region_current_masses(region)
            region["dirty_mass"] = float(dirty_mass)
            region["frontier_mass"] = float(frontier_mass)

            dirty_trigger = float(Config.REGION_DIRTY_TRIGGER)
            frontier_trigger = float(Config.REGION_FRONTIER_TRIGGER)
            if region["type"] == "bridge":
                dirty_trigger = max(dirty_trigger, float(Config.BRIDGE_REGION_DIRTY_TRIGGER))
                frontier_trigger = max(frontier_trigger, float(Config.BRIDGE_REGION_FRONTIER_TRIGGER))
            if dirty_mass < dirty_trigger and frontier_mass < frontier_trigger:
                continue

            entry = tuple(region["entry"])
            anchor = tuple(region["anchor"])
            entry_dist = int(dist[entry[0], entry[1]])
            anchor_dist = int(dist[anchor[0], anchor[1]])
            if entry_dist < 0 and anchor_dist < 0:
                continue

            travel_cost = max(entry_dist, anchor_dist, 0)
            if self._region_return_budget(anchor, travel_cost, charger_dist) >= self.battery:
                continue

            score = (
                Config.REGION_VALUE_DIRTY_WEIGHT * float(dirty_mass)
                + Config.REGION_VALUE_FRONTIER_WEIGHT * float(frontier_mass)
                - Config.REGION_VALUE_DISTANCE_WEIGHT * float(max(entry_dist, 0))
                - Config.REGION_VALUE_RISK_WEIGHT * float(region["risk_mean"])
                - Config.REGION_VALUE_SPINE_PENALTY * float(max(0, len(region["mouths"]) - 1))
            )
            if region["type"] == "tail":
                score += 5.2
            elif region["type"] == "room":
                score += 2.8
            else:
                score -= 6.0

            score -= float(self._spine_penalty(entry)) * Config.SPINE_DIRTY_PENALTY
            score -= self._post_charge_goal_penalty(entry)
            score -= 0.35 * self._region_entry_penalty(region, anchor)

            if self.active_region_id == int(region["id"]):
                score += 2.2
                if self._active_region_force_hold(region):
                    score += 2.4
            if self._in_bounds(*self.cur_pos) and int(self.region_mask[self.cur_pos[0], self.cur_pos[1]]) == int(region["id"]):
                score += 1.1

            bucket = bridge if region["type"] == "bridge" else preferred
            bucket.append((float(score), region))

        if preferred:
            preferred.sort(key=lambda item: item[0], reverse=True)
            return preferred[0][1]
        if bridge:
            bridge.sort(key=lambda item: item[0], reverse=True)
            return bridge[0][1]
        return None

    def _pick_region_goal(self, region, dist, charger_dist):
        dirty_mass, frontier_mass = self._region_current_masses(region)
        region["dirty_mass"] = float(dirty_mass)
        region["frontier_mass"] = float(frontier_mass)

        cover_goal = self._next_cover_goal(region)
        if cover_goal is not None:
            d = int(dist[cover_goal[0], cover_goal[1]])
            if d > 0 and self._region_return_budget(cover_goal, d, charger_dist) < self.battery:
                self.active_region_goal = tuple(cover_goal)
                if int(self.map_state[cover_goal[0], cover_goal[1]]) == self.DIRTY:
                    return "dirty", tuple(cover_goal)
                return "frontier", tuple(cover_goal)

        if self.active_region_goal is not None and self._in_bounds(*self.active_region_goal):
            gx, gz = self.active_region_goal
            if int(self.region_mask[gx, gz]) == int(region["id"]):
                d = int(dist[gx, gz])
                if d > 0 and self._region_return_budget((gx, gz), d, charger_dist) < self.battery:
                    if int(self.map_state[gx, gz]) == self.DIRTY:
                        return "dirty", (gx, gz)
                    if self._has_unknown_neighbor((gx, gz)):
                        return "frontier", (gx, gz)

        best_score = -1e9
        best_goal = None
        best_kind = None
        for pos in region["cells"]:
            x, z = pos
            d = int(dist[x, z])
            if d <= 0:
                continue
            is_dirty = int(self.map_state[x, z]) == self.DIRTY
            is_frontier = self._has_unknown_neighbor(pos)
            if not is_dirty and not is_frontier:
                continue
            if self._region_return_budget(pos, d, charger_dist) >= self.battery:
                continue

            depth_bonus = self._region_depth_progress(region, pos) * Config.CLEAN_REGION_FAR_PROGRESS
            if region["type"] == "tail":
                depth_bonus *= 1.55
            elif region["type"] == "room":
                depth_bonus *= 1.20

            score = (
                (10.0 if is_dirty else 0.0)
                + 1.10 * self._frontier_gain(x, z)
                + 0.18 * self._dirty_density(x, z)
                + depth_bonus
                + self._region_room_value(region, pos)
                - 0.14 * d
                - 0.28 * float(self.visit_count[x, z])
                - 0.42 * float(self.clean_pass_count[x, z])
                - 0.06 * self._npc_zone_penalty(pos)
                - Config.BLOCKED_CELL_PENALTY * (1.0 if pos in self.blocked_cells else 0.0)
                - Config.NPC_CLEAN_PENALTY * float(self.npc_cleaned[x, z])
                - Config.SPINE_DIRTY_PENALTY * self._spine_penalty(pos)
                - self._region_entry_penalty(region, pos)
            )
            if is_dirty and region["type"] != "bridge":
                score += 1.0
            if self.active_cover_sequence and self.active_cover_index < len(self.active_cover_sequence):
                target = self.active_cover_sequence[self.active_cover_index]
                next_strip_target = self._next_strip_cover_goal()
                if pos == tuple(target):
                    score += Config.ROOM_SWEEP_PERSIST_BONUS
                elif next_strip_target is not None and pos == tuple(next_strip_target):
                    score += Config.ROOM_SWEEP_NEXT_STRIP_BONUS
            score -= Config.ROOM_SWEEP_REPEAT_PENALTY * float(self.clean_pass_count[x, z])
            if self.active_region_goal == pos:
                score += Config.CLEAN_REGION_STAY_BONUS + Config.CLEAN_REGION_DEEP_STAY_BONUS
            if score > best_score:
                best_score = score
                best_goal = pos
                best_kind = "dirty" if is_dirty else "frontier"
        if best_goal is None:
            return None, None
        return best_kind, best_goal

    def _pick_topology_explore_goal(self, dist, charger_dist):
        preferred = []
        bridge = []
        for region in self.regions:
            dirty_mass, frontier_mass = self._region_current_masses(region)
            region["dirty_mass"] = float(dirty_mass)
            region["frontier_mass"] = float(frontier_mass)

            frontier_trigger = float(Config.REGION_FRONTIER_TRIGGER)
            if region["type"] == "bridge":
                frontier_trigger = max(frontier_trigger, float(Config.BRIDGE_REGION_FRONTIER_TRIGGER))
            if frontier_mass < frontier_trigger and dirty_mass <= 0.0:
                continue

            entry = tuple(region["entry"])
            d = int(dist[entry[0], entry[1]])
            if d <= 0:
                continue
            if self._region_return_budget(entry, d, charger_dist) >= self.battery:
                continue

            goal_pos = None
            goal_gain = -1e9
            for pos in region["cells"]:
                x, z = pos
                cell_dist = int(dist[x, z])
                if cell_dist <= 0:
                    continue
                if not self._has_unknown_neighbor(pos):
                    continue
                gain = (
                    2.6 * self._frontier_gain(x, z)
                    + 0.12 * self._dirty_density(x, z)
                    - 0.08 * cell_dist
                    - 0.03 * self._npc_zone_penalty(pos)
                    - 0.15 * self._spine_penalty(pos)
                )
                if gain > goal_gain:
                    goal_gain = gain
                    goal_pos = pos
            if goal_pos is None:
                goal_pos = entry

            score = (
                Config.EXPLORE_MOUTH_BONUS
                + Config.EXPLORE_ROOM_FRONTIER_BONUS * float(frontier_mass)
                + 0.18 * float(dirty_mass)
                - 0.22 * float(d)
                - 0.04 * float(region["risk_mean"])
                - Config.SPINE_DIRTY_PENALTY * 0.35 * self._spine_penalty(entry)
            )
            if region["type"] == "tail":
                score += 1.8
            elif region["type"] == "room":
                score += 1.2
            else:
                score -= Config.EXPLORE_BRIDGE_PENALTY
            score -= self._post_charge_goal_penalty(goal_pos)

            bucket = bridge if region["type"] == "bridge" else preferred
            bucket.append((float(score), goal_pos, region))

        candidates = preferred if preferred else bridge
        if not candidates:
            return None, None, None
        candidates.sort(key=lambda item: item[0], reverse=True)
        _, best_goal, best_region = candidates[0]
        return "frontier", best_goal, best_region

    def _update_runtime_counters(self):
        if self.chargers and len(self._current_step_cleaned) == 0:
            for charger in self.chargers:
                if self._dist_to_charger_rect(self.cur_pos, charger) <= Config.POST_CHARGE_HALO_RADIUS:
                    self.charger_halo_waste_steps += 1
                    break
        if self._in_bounds(*self.cur_pos) and bool(self.transit_spine_mask[self.cur_pos[0], self.cur_pos[1]]):
            if len(self._current_step_cleaned) == 0:
                self.spine_transit_steps += 1
        if self._has_unknown_neighbor(self.cur_pos) and self.planner_mode != "explore":
            self.frontier_skip_steps += 1

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
        if self.charge_mode and self.on_charger and self.battery >= int(self.battery_max * Config.CHARGE_EXIT_BATTERY_RATIO):
            self._exit_charge_mode()
        elif self._need_charge():
            self.charge_mode = True
            self.planner_mode = "return_charge"
            self._clear_region_lock()
            self._clear_active_region()

        if self._need_replan():
            self._replan()
            self.decision_event = True
            self.decision_step = self.step_no
            self._prepare_decision_candidates()
        else:
            self.decision_event = False
            self._clear_decision_candidates()

        launch_act = self._launch_lock_action(legal_action)
        if launch_act is not None:
            self.pending_prob = np.zeros((Config.ACTION_DIM,), dtype=np.float32)
            self.pending_prob[launch_act] = 1.0
            return int(launch_act)

        commit_act = self._path_commit_action(legal_action)
        if commit_act is not None:
            self.pending_prob = np.zeros((Config.ACTION_DIM,), dtype=np.float32)
            self.pending_prob[commit_act] = 1.0
            return int(commit_act)

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
        if self.charge_mode:
            return True
        charger_dist = self._current_charger_distance()
        return_buffer = self._dynamic_return_buffer(self.cur_pos, charger_dist)
        if self._in_post_charge_window():
            safe_floor = charger_dist + return_buffer + Config.CRITICAL_CHARGE_MARGIN
            if self.battery > safe_floor:
                return False
        if self.battery <= charger_dist + return_buffer:
            return True
        return self.battery / self.battery_max <= Config.LOW_BATTERY_RATIO

    def _need_replan(self):
        if not self.path:
            return True
        if self.stuck_chain > 0:
            return True
        if self.goal is None:
            return True
        if self.charge_mode and self.goal_kind != "charger":
            return True
        if self.goal_kind == "dirty":
            gx, gz = self.goal
            if not self._in_bounds(gx, gz) or self.map_state[gx, gz] != self.DIRTY:
                return True
        if self.goal_kind == "charger" and self.on_charger:
            return True
        if self._path_is_npc_unsafe():
            return True
        if self._risk_spike_requires_replan():
            return True
        if self.step_no - self.last_plan_step < Config.REPLAN_INTERVAL:
            return False
        return not self._should_hold_plan()

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
        if mass < self._region_min_mass(kind):
            return
        self.region_lock_kind = kind
        self.region_lock_center = (int(center[0]), int(center[1]))

    def _should_keep_region_lock(self):
        if self.charge_mode or self.region_lock_center is None or self.region_lock_kind is None:
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
        if center is None or kind not in {"dirty", "frontier"}:
            return 0
        return max(0, self._chebyshev(pos, center) - self._region_radius(kind))

    def _replan(self):
        dist, prev_x, prev_z = self._build_bfs_tree(allow_unknown=True)
        self._charger_dist_map = self._build_multi_source_dist(self._charger_cells(), allow_unknown=True)
        self._last_dist_map = dist
        self._last_prev_x = prev_x
        self._last_prev_z = prev_z
        self._update_topology(dist, self._charger_dist_map)

        goal_kind = None
        goal = None
        teacher_candidate_type = None
        teacher_region_id = -1
        teacher_charger_id = -1
        if self.charge_mode:
            self.planner_mode = "return_charge"
            self._clear_active_region()
            goal_kind, goal = self._pick_best_charger_goal(dist)
            teacher_candidate_type = "charger"
            charger = self._charger_for_pos(goal) if goal is not None else None
            teacher_charger_id = int(charger.get("id", -1)) if charger is not None else -1
        else:
            region = None
            if 0 <= self.active_region_id < len(self.regions):
                candidate = self.regions[self.active_region_id]
                if self._region_still_viable(candidate, dist, self._charger_dist_map):
                    region = candidate
            if region is None:
                region = self._pick_best_region(dist, self._charger_dist_map)

            if region is not None:
                self.planner_mode = "clean_region"
                self._bind_active_region(region, preserve_goal=self.active_region_goal)
                self.region_lock_kind = "dirty" if region["type"] != "bridge" else "frontier"
                self.region_lock_center = tuple(region["anchor"])
                goal_kind, goal = self._pick_region_goal(region, dist, self._charger_dist_map)
                if goal is None:
                    self._clear_active_region()
                    self._clear_region_lock()
                else:
                    teacher_candidate_type = "clean_region"
                    teacher_region_id = int(region["id"])
            if goal is None:
                self.planner_mode = "explore"
                self._clear_active_region()
                goal_kind, goal, explore_region = self._pick_topology_explore_goal(dist, self._charger_dist_map)
                if goal is not None and explore_region is not None:
                    self.active_mouth_id = int(explore_region["mouth_id"])
                    self.region_lock_kind = "frontier"
                    self.region_lock_center = tuple(explore_region["entry"])
                    teacher_candidate_type = "explore_mouth"
                    teacher_region_id = int(explore_region["id"])
                else:
                    self._clear_region_lock()
            if goal is None:
                goal_kind, goal = self._pick_best_frontier_goal(dist, self._charger_dist_map)
                if goal is not None:
                    teacher_candidate_type = "fallback_frontier"
            if goal is None:
                goal_kind, goal = self._pick_best_dirty_goal(dist, self._charger_dist_map)
                if goal is not None:
                    teacher_candidate_type = "fallback_frontier"
            if goal is None and self._charge_slack() <= Config.FALLBACK_CHARGE_SLACK:
                self.planner_mode = "return_charge"
                self._clear_region_lock()
                self._clear_active_region()
                goal_kind, goal = self._pick_best_charger_goal(dist)
                teacher_candidate_type = "charger"
                charger = self._charger_for_pos(goal) if goal is not None else None
                teacher_charger_id = int(charger.get("id", -1)) if charger is not None else -1

        goal_changed = goal_kind != self.goal_kind or goal != self.goal
        self._set_teacher_candidate_target(
            candidate_type=teacher_candidate_type,
            goal=goal,
            goal_kind=goal_kind,
            region_id=teacher_region_id,
            charger_id=teacher_charger_id,
        )
        self.goal_kind = goal_kind
        self.goal = goal
        self.last_plan_step = self.step_no
        if goal_changed:
            self.plan_churn_count += 1
            self.goal_set_step = self.step_no
            self.goal_progress_step = self.step_no

        if goal is None:
            self.path = []
            self.last_plan_risk = float(self._npc_zone_penalty(self.cur_pos))
            return

        path = self._plan_weighted_path(goal, allow_unknown=True)
        if not path:
            path = self._reconstruct_path(goal, prev_x, prev_z)
        self.path = path[: Config.MAX_TRACKED_PATH]
        self.last_plan_risk = self._decision_risk_signature()

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

    def _build_multi_source_dist(self, starts, allow_unknown):
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
                if dx != 0 and dz != 0 and not (plannable[x + dx, z] or plannable[x, z + dz]):
                    continue
                if zone_block[nx, nz] or dynamic_block[nx, nz]:
                    continue
                dist[nx, nz] = dist[x, z] + 1
                queue.append((nx, nz))

        return dist

    def _is_global_move_passable(self, cur_pos, next_pos, allow_unknown):
        nx, nz = next_pos
        plannable = self._plannable_unknown_map if allow_unknown else self._plannable_known_map
        if not self._in_bounds(nx, nz) or not plannable[nx, nz]:
            return False

        dx = nx - cur_pos[0]
        dz = nz - cur_pos[1]
        if dx != 0 and dz != 0:
            side_a = (cur_pos[0] + dx, cur_pos[1])
            side_b = (cur_pos[0], cur_pos[1] + dz)
            if not (plannable[side_a[0], side_a[1]] or plannable[side_b[0], side_b[1]]):
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
                if self._npc_zone_hard_block((nx, nz)) or self._npc_dynamic_hard_block((nx, nz)):
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
        explore_push = self._explore_push()
        repeat_scale = 1.0 - Config.EXPLORE_REPEAT_RELAX * explore_push
        charge_repeat_scale = Config.CHARGE_REPEAT_MULTIPLIER if self.goal_kind == "charger" or self.charge_mode else 1.0

        active_region = self._active_region_object() if self.planner_mode == "clean_region" else None
        in_region_now = (
            active_region is not None
            and self._in_bounds(*cur_pos)
            and int(self.region_mask[cur_pos[0], cur_pos[1]]) == int(active_region["id"])
        )
        in_region_next = (
            active_region is not None
            and self._in_bounds(nx, nz)
            and int(self.region_mask[nx, nz]) == int(active_region["id"])
        )
        room_clean_next = in_region_next and active_region is not None and active_region["type"] != "bridge"

        base = 1.0 + (0.05 if next_pos[0] != cur_pos[0] and next_pos[1] != cur_pos[1] else 0.0)
        recent_gap = self.step_no - int(self.last_visit_step[nx, nz]) if self._in_bounds(nx, nz) else 0
        recent_penalty = 0.0 if recent_gap > 20 else 0.08 * max(0, 20 - recent_gap)
        style_repeat = self._style_factor("repeat")
        style_transit = self._style_factor("transit")
        style_risk = self._style_factor("risk")
        style_boundary = self._style_factor("boundary")
        style_frontier = self._style_factor("frontier")
        style_dirty = self._style_factor("dirty")
        style_exit = self._style_factor("exit")
        repeat_penalty = 0.16 * style_repeat * charge_repeat_scale * repeat_scale * min(visit, float(Config.MAX_VISIT_CLIP))
        transit_penalty = 0.28 * style_transit * charge_repeat_scale * repeat_scale * min(transit, float(Config.MAX_TRANSIT_CLIP))
        transit_penalty += Config.BLOCKED_CELL_PENALTY * (1.0 if next_pos in self.blocked_cells else 0.0)
        transit_penalty += Config.NPC_CLEAN_PENALTY * float(self.npc_cleaned[nx, nz]) if self._in_bounds(nx, nz) else 0.0
        risk_penalty = 0.05 * style_risk * min(self._npc_zone_penalty(next_pos), 60.0)
        interior_penalty = self._interior_clean_penalty(next_pos)
        snake_bias = self._serpentine_bias(cur_pos, next_pos)
        boundary_repeat_scale = 1.0 - Config.EXPLORE_BOUNDARY_RELAX * explore_push
        boundary_bonus = self._boundary_bonus(next_pos, repeat_scale=boundary_repeat_scale)
        post_charge_penalty = 0.35 * self._post_charge_goal_penalty(next_pos)
        charger_zone_penalty = self._charger_zone_repeat_penalty(next_pos)
        topology_penalty = 0.0

        if self.planner_mode == "clean_region" and self._spine_penalty(next_pos) > 0.0:
            topology_penalty += 0.65 * Config.SPINE_DIRTY_PENALTY
        elif self.planner_mode == "explore" and self._spine_penalty(next_pos) > 0.0:
            topology_penalty -= 0.25 * Config.SPINE_TRANSIT_BONUS

        if room_clean_next:
            boundary_bonus *= Config.CLEAN_ROOM_BOUNDARY_SCALE
            topology_penalty -= 0.28 * self._region_room_value(active_region, next_pos)
            topology_penalty += self._region_entry_penalty(active_region, next_pos)
            if self._spine_penalty(next_pos) > 0.0:
                topology_penalty += Config.CLEAN_ROOM_SPINE_EXTRA_PENALTY
            else:
                topology_penalty -= Config.CLEAN_ROOM_INTERIOR_BONUS
            repeat_penalty += Config.ROOM_SWEEP_REPEAT_PENALTY * min(transit, float(Config.MAX_TRANSIT_CLIP))
        elif in_region_now and active_region is not None and active_region["type"] != "bridge" and not in_region_next:
            topology_penalty += style_exit * Config.CLEAN_REGION_STRONG_EXIT_PENALTY

        dirty_bonus = 0.95 * style_dirty if cell == self.DIRTY else 0.0
        frontier_bonus = 0.0
        if cell == self.UNKNOWN:
            frontier_bonus += 0.20 * style_frontier
        if self._has_unknown_neighbor(next_pos):
            frontier_bonus += 0.30 * style_frontier
        frontier_bonus += 0.05 * style_frontier * min(self._unknown_density(next_pos, radius=2), 8)

        if next_pos == self.last_pos:
            recent_penalty += 0.8

        slack = self._charge_slack()
        snake_scale = 1.0
        if self.goal_kind == "charger" or self.charge_mode:
            dirty_bonus *= slack
            frontier_bonus *= 0.65 * slack
            snake_scale = Config.CHARGE_SNAKE_SCALE * (0.35 + 0.65 * slack)
            topology_penalty -= self._charger_approach_bonus(cur_pos, next_pos)
        elif room_clean_next:
            snake_scale *= Config.CLEAN_ROOM_SNAKE_SCALE

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
            + post_charge_penalty
            + charger_zone_penalty
            + topology_penalty
            + snake_penalty
            - Config.BOUNDARY_BONUS_WEIGHT * style_boundary * boundary_bonus
        )
        cost -= dirty_bonus
        cost -= frontier_bonus
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
        margin = self.battery - charger_dist - self._dynamic_return_buffer(self.cur_pos, charger_dist)
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

    def _pick_best_dirty_goal(self, dist, charger_dist, region_center=None, restrict_region=False):
        dirty_cells = np.argwhere(self.map_state == self.DIRTY)
        if len(dirty_cells) == 0:
            return None, None

        candidates = self._cluster_cells(dirty_cells, Config.DIRTY_CLUSTER_RADIUS, self._dirty_density)
        candidates = self._filter_candidates_by_region(candidates, "dirty", region_center, restrict_region)
        if not candidates:
            return None, None

        explore_push = self._explore_push()
        distance_scale = 0.8 * (1.0 - Config.EXPLORE_DISTANCE_RELAX * explore_push)
        repeat_scale = 1.0 - Config.EXPLORE_REPEAT_RELAX * explore_push
        frontier_scale = 0.8 * (1.0 + Config.EXPLORE_FRONTIER_GAIN_SCALE * explore_push)
        best_score = -1e9
        best_goal = None
        best_region_anchor = None
        best_region_mass = 0.0
        for gx, gz in candidates:
            d = int(dist[gx, gz])
            if d <= 0:
                continue

            return_budget = self._region_return_budget((gx, gz), d, charger_dist)
            if return_budget >= self.battery:
                continue

            score = (
                18.0
                + 4.2 * self._dirty_density(gx, gz)
                + frontier_scale * self._frontier_gain(gx, gz)
                - distance_scale * d
                - 0.55 * repeat_scale * self.visit_count[gx, gz]
                - 0.75 * repeat_scale * self.clean_pass_count[gx, gz]
                - 0.45 * self._npc_zone_penalty((gx, gz))
                - Config.BLOCKED_CELL_PENALTY * (1.0 if (int(gx), int(gz)) in self.blocked_cells else 0.0)
                - Config.NPC_CLEAN_PENALTY * float(self.npc_cleaned[gx, gz])
            )
            region_anchor = region_center if region_center is not None else (gx, gz)
            region_mass = self._region_mass("dirty", region_anchor)
            score += Config.REGION_LOCK_VALUE_WEIGHT * min(region_mass, 18.0)
            score += self._goal_sticky_bonus("dirty", (int(gx), int(gz)))
            score -= self._post_charge_goal_penalty((int(gx), int(gz)))
            score -= Config.SPINE_DIRTY_PENALTY * self._spine_penalty((int(gx), int(gz)))

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

        explore_push = self._explore_push()
        gain_scale = 4.8 * (1.0 + Config.EXPLORE_FRONTIER_GAIN_SCALE * explore_push)
        distance_scale = 0.9 * (1.0 - Config.EXPLORE_DISTANCE_RELAX * explore_push)
        repeat_scale = 1.0 - Config.EXPLORE_REPEAT_RELAX * explore_push
        best_score = -1e9
        best_goal = None
        best_region_anchor = None
        best_region_mass = 0.0
        for gx, gz in candidates:
            d = int(dist[gx, gz])
            if d <= 0:
                continue

            return_budget = self._region_return_budget((gx, gz), d, charger_dist)
            if return_budget >= self.battery:
                continue

            unknown_gain = self._frontier_gain(gx, gz)
            score = (
                gain_scale * unknown_gain
                - distance_scale * d
                - 0.55 * repeat_scale * self.visit_count[gx, gz]
                - 0.85 * repeat_scale * self.clean_pass_count[gx, gz]
                - 0.55 * self._npc_zone_penalty((gx, gz))
                - Config.BLOCKED_CELL_PENALTY * (1.0 if (int(gx), int(gz)) in self.blocked_cells else 0.0)
                - Config.NPC_CLEAN_PENALTY * float(self.npc_cleaned[gx, gz])
            )
            region_anchor = region_center if region_center is not None else (gx, gz)
            region_mass = self._region_mass("frontier", region_anchor)
            score += Config.REGION_LOCK_VALUE_WEIGHT * min(region_mass, 22.0)
            score += self._goal_sticky_bonus("frontier", (int(gx), int(gz)))
            score -= self._post_charge_goal_penalty((int(gx), int(gz)))
            score -= Config.SPINE_DIRTY_PENALTY * 0.70 * self._spine_penalty((int(gx), int(gz)))
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
            charger_bias = 0.0
            if self._in_post_charge_window() and int(charger.get("id", -1)) == self.last_charger_id:
                charger_bias += Config.POST_CHARGE_HALO_PENALTY
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
                        + charger_bias
                        + 0.65 * float(self.npc_cleaned[gx, gz])
                        + (Config.BLOCKED_CELL_PENALTY if (gx, gz) in self.blocked_cells else 0.0)
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

    def _boundary_bonus(self, pos, repeat_scale=1.0):
        x, z = pos
        if not self._in_bounds(x, z):
            return -1.0

        wall_neighbors = 0.0
        open_neighbors = 0.0
        value_neighbors = 0.0
        cardinal_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for dx, dz in cardinal_dirs:
            nx = x + dx
            nz = z + dz
            neighbor = (nx, nz)
            if not self._in_bounds(nx, nz) or self.map_state[nx, nz] == self.OBSTACLE:
                wall_neighbors += 1
                continue
            if self._npc_zone_penalty(neighbor) >= Config.BOUNDARY_NPC_THRESHOLD:
                wall_neighbors += 1
                continue
            if self._is_repeat_wall_cell(neighbor):
                wall_neighbors += float(np.clip(repeat_scale, 0.0, 1.0))
                open_neighbors += float(np.clip(1.0 - repeat_scale, 0.0, 1.0))
                continue

            open_neighbors += 1
            if self.map_state[nx, nz] == self.DIRTY or self.map_state[nx, nz] == self.UNKNOWN:
                value_neighbors += 1
            elif self._has_unknown_neighbor(neighbor):
                value_neighbors += 1
            elif not self._is_repeat_wall_cell(neighbor):
                value_neighbors += 1

        bonus = 0.0
        if 1 <= wall_neighbors <= 2:
            bonus += 1.0 + 0.18 * wall_neighbors
        elif 2.5 <= wall_neighbors < 3.5:
            bonus += 0.45
        elif wall_neighbors >= 3.5:
            bonus -= 0.55

        if value_neighbors >= 2:
            bonus += 0.35
        elif value_neighbors == 0:
            bonus -= 0.35

        if open_neighbors <= 1:
            bonus -= 0.25

        if self._is_repeat_wall_cell(pos):
            bonus -= 0.45 * float(np.clip(repeat_scale, 0.0, 1.0))
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

    def _style_factor(self, key):
        style = self.current_path_style_name
        table = {
            "aggressive_explore": {
                "repeat": 0.72,
                "transit": 0.78,
                "risk": 0.85,
                "boundary": 1.18,
                "frontier": 1.25,
                "dirty": 1.00,
                "exit": 0.82,
            },
            "balanced": {
                "repeat": 1.0,
                "transit": 1.0,
                "risk": 1.0,
                "boundary": 1.0,
                "frontier": 1.0,
                "dirty": 1.0,
                "exit": 1.0,
            },
            "deep_clean": {
                "repeat": 0.92,
                "transit": 0.92,
                "risk": 1.05,
                "boundary": 0.82,
                "frontier": 0.78,
                "dirty": 1.18,
                "exit": 1.22,
            },
            "safe_return": {
                "repeat": 0.88,
                "transit": 1.35,
                "risk": 1.40,
                "boundary": 0.92,
                "frontier": 0.55,
                "dirty": 0.45,
                "exit": 0.70,
            },
            "escape": {
                "repeat": 0.70,
                "transit": 0.82,
                "risk": 1.55,
                "boundary": 0.72,
                "frontier": 0.52,
                "dirty": 0.50,
                "exit": 0.72,
            },
        }
        base = float(table.get(style, table["balanced"]).get(key, 1.0))
        bonus = {
            "aggressive_explore": float(Config.AGGRESSIVE_EXPLORE_STYLE_BONUS),
            "deep_clean": float(Config.DEEP_CLEAN_STYLE_BONUS),
            "safe_return": float(Config.SAFE_RETURN_STYLE_BONUS),
            "escape": float(Config.ESCAPE_STYLE_BONUS),
        }.get(style, 0.0)

        adjustment = 0.0
        if style == "aggressive_explore":
            adjust_map = {"frontier": 0.35, "boundary": 0.20, "repeat": -0.16, "risk": -0.08}
            adjustment = bonus * adjust_map.get(key, 0.0)
        elif style == "deep_clean":
            adjust_map = {"dirty": 0.30, "exit": 0.20, "repeat": -0.08}
            adjustment = bonus * adjust_map.get(key, 0.0)
        elif style == "safe_return":
            adjust_map = {"risk": 0.30, "transit": 0.22, "dirty": -0.18, "frontier": -0.12}
            adjustment = bonus * adjust_map.get(key, 0.0)
        elif style == "escape":
            adjust_map = {"risk": 0.32, "repeat": -0.18, "boundary": -0.10}
            adjustment = bonus * adjust_map.get(key, 0.0)

        return float(max(0.10, base + adjustment))

    def _score_actions(self, legal_action):
        scores = np.full((Config.ACTION_DIM,), -1e9, dtype=np.float32)
        charger_dist_now = self._current_charger_distance()
        goal_dist_now = self._chebyshev(self.cur_pos, self.goal) if self.goal is not None else 0
        region_dist_now = self._distance_to_region(self.cur_pos, self.region_lock_kind, self.region_lock_center)
        explore_push = self._explore_push()
        repeat_scale = 1.0 - Config.EXPLORE_REPEAT_RELAX * explore_push
        boundary_repeat_scale = 1.0 - Config.EXPLORE_BOUNDARY_RELAX * explore_push
        frontier_action_scale = 1.0 + Config.EXPLORE_FRONTIER_GAIN_SCALE * explore_push
        style_repeat = self._style_factor("repeat")
        style_boundary = self._style_factor("boundary")
        style_frontier = self._style_factor("frontier")
        style_dirty = self._style_factor("dirty")
        style_risk = self._style_factor("risk")
        style_exit = self._style_factor("exit")
        active_region = self._active_region_object() if self.planner_mode == "clean_region" else None
        active_region_mass = 0.0
        in_region_now = False
        if active_region is not None:
            active_region_mass = float(active_region.get("dirty_mass", 0.0)) + float(active_region.get("frontier_mass", 0.0))
            in_region_now = self._in_bounds(*self.cur_pos) and int(self.region_mask[self.cur_pos[0], self.cur_pos[1]]) == int(active_region["id"])

        for act, (dx, dz) in enumerate(self.ACTION_DELTAS):
            if not legal_action[act]:
                continue

            nx = self.cur_pos[0] + dx
            nz = self.cur_pos[1] + dz
            next_pos = (nx, nz)

            score = -1.6 * self._transition_cost(self.cur_pos, next_pos)
            if self._in_bounds(nx, nz):
                boundary_bonus = self._boundary_bonus(next_pos, repeat_scale=boundary_repeat_scale)
                snake_scale = 1.0 if not self.charge_mode else Config.CHARGE_SNAKE_SCALE * (0.35 + 0.65 * self._charge_slack())
                in_region_next = active_region is not None and int(self.region_mask[nx, nz]) == int(active_region["id"])
                room_clean_next = in_region_next and active_region is not None and active_region["type"] != "bridge"

                if room_clean_next:
                    boundary_bonus *= Config.CLEAN_ROOM_BOUNDARY_SCALE
                    if not self.charge_mode:
                        snake_scale *= Config.CLEAN_ROOM_SNAKE_SCALE

                if self.map_state[nx, nz] == self.DIRTY:
                    score += 8.0 * style_dirty
                score += 1.5 * style_frontier * frontier_action_scale * self._frontier_gain(nx, nz)
                score -= 0.45 * style_repeat * repeat_scale * self.visit_count[nx, nz]
                score -= 0.75 * style_repeat * repeat_scale * self.clean_pass_count[nx, nz]
                score -= 0.08 * style_risk * self._npc_zone_penalty(next_pos)
                score -= Config.BLOCKED_CELL_PENALTY * (1.0 if next_pos in self.blocked_cells else 0.0)
                score -= Config.NPC_CLEAN_PENALTY * float(self.npc_cleaned[nx, nz])
                score -= self._charger_zone_repeat_penalty(next_pos)
                score += Config.SNAKE_ACTION_WEIGHT * snake_scale * self._serpentine_bias(self.cur_pos, next_pos)
                score += Config.BOUNDARY_ACTION_WEIGHT * style_boundary * boundary_bonus
                if self.planner_mode == "explore" and self._spine_penalty(next_pos) > 0.0:
                    score += Config.SPINE_TRANSIT_BONUS
                    if self._in_bounds(*self.cur_pos) and bool(self.corridor_mask[self.cur_pos[0], self.cur_pos[1]]):
                        score += Config.EXPLORE_CORRIDOR_BOUNDARY_BONUS * max(0.0, boundary_bonus)
                if self.planner_mode == "clean_region" and active_region is not None:
                    if in_region_next:
                        score += Config.CLEAN_REGION_STAY_BONUS
                        score += 0.45 * self._region_room_value(active_region, next_pos)
                        score -= self._region_entry_penalty(active_region, next_pos)
                        if self.active_cover_sequence:
                            current_cover = self.active_cover_sequence[min(self.active_cover_index, len(self.active_cover_sequence) - 1)]
                            next_strip_target = self._next_strip_cover_goal()
                            if tuple(next_pos) == tuple(current_cover):
                                score += Config.ROOM_SWEEP_PERSIST_BONUS
                            elif next_strip_target is not None and tuple(next_pos) == tuple(next_strip_target):
                                score += Config.ROOM_SWEEP_NEXT_STRIP_BONUS
                        if self.active_region_entry is not None:
                            region = {
                                "axis": self.active_region_axis,
                                "sign": self.active_region_sign,
                                "entry": self.active_region_entry,
                            }
                            score += Config.CLEAN_REGION_SWEEP_BONUS * self._region_depth_progress(region, next_pos)
                        score -= Config.ROOM_SWEEP_REPEAT_PENALTY * float(self.clean_pass_count[nx, nz])
                    elif in_region_now:
                        exit_penalty = style_exit * Config.CLEAN_REGION_EXIT_PENALTY
                        if active_region_mass > 0.0 and active_region["type"] != "bridge":
                            exit_penalty += style_exit * Config.CLEAN_REGION_STRONG_EXIT_PENALTY
                        score -= exit_penalty
                    if self._spine_penalty(next_pos) > 0.0:
                        score -= Config.SPINE_DIRTY_PENALTY

            if next_pos == self.last_pos:
                score -= 2.0
            if self.stuck_chain > 0 and act == self.last_action:
                score -= 6.0

            if self.charge_mode:
                charger_dist_next = self._charger_dist_at(next_pos, self._charger_dist_map)
                score += 3.8 * (charger_dist_now - charger_dist_next)
                score += 1.35 * self._charger_approach_bonus(self.cur_pos, next_pos)
                if self._is_on_charger(next_pos):
                    score += 7.0
            elif self.goal is not None:
                score += 1.35 * (goal_dist_now - self._chebyshev(next_pos, self.goal))
                region_dist_next = self._distance_to_region(next_pos, self.region_lock_kind, self.region_lock_center)
                score += Config.REGION_LOCK_ACTION_WEIGHT * (region_dist_now - region_dist_next)
                if region_dist_next == 0 and self.region_lock_kind in {"dirty", "frontier"}:
                    score += 0.25
                score += self._post_charge_action_bonus(next_pos)
                if self.planner_mode == "explore" and self.active_region_entry is not None:
                    score += 0.60 * (self._chebyshev(self.cur_pos, self.active_region_entry) - self._chebyshev(next_pos, self.active_region_entry))

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
        explored_ratio = self.explored_ratio
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
