#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Feature preprocessor for robot_vacuum.
"""

from typing import Any, Optional

import numpy as np

from agent_ppo.feature.definition import Config
from agent_ppo.feature.planner import CoveragePlanner, PolicyInfo


class Preprocessor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.step_no = 0
        self.prev_clean_score = 0.0
        self.prev_known_cells = 0.0
        self.prev_target_distance = 0.0
        self.prev_charger_distance = 0.0
        self.prev_battery_ratio = 1.0
        self.last_policy_info: Optional[PolicyInfo] = None
        self.planner = CoveragePlanner()

    def feature_process(self, env_obs: Any, last_action: int):
        self.step_no += 1
        policy_info = self.planner.update(env_obs, last_action)
        self.last_policy_info = policy_info

        obs = self._extract_observation(env_obs)
        env_info = self._get(obs, "env_info", {})
        frame_state = self._get(obs, "frame_state", {})
        hero = self._extract_hero(frame_state)
        map_grid = self._extract_map_grid(obs)
        legal_action = np.asarray(policy_info.safe_action_mask, dtype=np.float32)

        local_feature = self._extract_local_view_feature(map_grid)
        global_feature = self._extract_global_feature(env_obs, env_info, hero, policy_info)
        feature = np.concatenate([local_feature, global_feature, legal_action], axis=0).astype(np.float32)
        if feature.shape[0] != Config.DIM_OF_OBSERVATION:
            feature = self._resize_feature(feature, Config.DIM_OF_OBSERVATION)

        reward = self._compute_reward(env_obs, env_info, hero, policy_info)
        return feature, legal_action.astype(np.float32), float(reward)

    def get_policy_output(self) -> Optional[PolicyInfo]:
        return self.last_policy_info

    def _extract_local_view_feature(self, map_grid: np.ndarray) -> np.ndarray:
        center = map_grid.shape[0] // 2
        radius = Config.LOCAL_VIEW_SIZE // 2
        patch = map_grid[center - radius : center + radius + 1, center - radius : center + radius + 1].astype(np.float32)
        patch = np.clip(patch / 2.0, 0.0, 1.0)
        return patch.reshape(-1).astype(np.float32)

    def _extract_global_feature(self, env_obs: Any, env_info: Any, hero: Any, policy_info: PolicyInfo) -> np.ndarray:
        step_no = self._safe_float(self._get_obs_level(env_obs, "step_no", None), 0.0)
        max_step = self._safe_float(self._get(env_info, "max_step", 1000), 1000.0)
        finished_steps = self._safe_float(self._get(env_info, "finished_steps", step_no), step_no)
        step_ratio = finished_steps / max(max_step, 1.0)

        clean_ratio = self._safe_float(self._get(env_info, "clean_score", self._get(hero, "score", 0)), 0.0) / max(
            self._safe_float(self._get(env_info, "total_dirt", 1), 1.0),
            1.0,
        )
        remaining_ratio = 1.0 - clean_ratio

        pos = self._get(hero, "pos", self._get(env_info, "pos", {}))
        x = self._safe_float(self._get(pos, "x", 0), 0.0)
        z = self._safe_float(self._get(pos, "z", 0), 0.0)
        x_norm = x / 127.0
        z_norm = z / 127.0

        charger_distance = min(policy_info.charger_distance / 128.0, 1.0)
        npc_distance = min(policy_info.nearest_npc_distance / 16.0, 1.0)
        target_distance = min(policy_info.target_distance / 128.0, 1.0)

        return np.asarray(
            [
                step_ratio,
                float(policy_info.battery_ratio),
                float(clean_ratio),
                float(max(0.0, remaining_ratio)),
                x_norm,
                z_norm,
                charger_distance,
                npc_distance,
                target_distance,
                float(policy_info.frontier_density),
                float(policy_info.local_dirty_ratio),
                float(policy_info.local_unknown_ratio),
            ],
            dtype=np.float32,
        )

    def _compute_reward(self, env_obs: Any, env_info: Any, hero: Any, policy_info: PolicyInfo) -> float:
        clean_score = self._safe_float(
            self._get(env_info, "clean_score", self._get(hero, "dirt_cleaned", self._get(hero, "score", 0))),
            0.0,
        )
        cleaned_delta = max(clean_score - self.prev_clean_score, 0.0)

        reward = 1.2 * cleaned_delta
        reward += 0.002 * min(max(float(policy_info.new_known_cells), 0.0), 40.0)

        if policy_info.should_charge:
            reward += 0.05 * np.clip(self.prev_charger_distance - float(policy_info.charger_distance), -2.0, 2.0)
        else:
            reward += 0.03 * np.clip(self.prev_target_distance - float(policy_info.target_distance), -2.0, 2.0)
            reward += 0.05 * float(policy_info.frontier_density)

        if policy_info.nearest_npc_distance < 3.0:
            reward -= 0.25 * (3.0 - float(policy_info.nearest_npc_distance))

        if policy_info.on_charger and self.prev_battery_ratio < 0.35:
            reward += 0.35

        terminated = bool(self._get_obs_level(env_obs, "terminated", False))
        truncated = bool(self._get_obs_level(env_obs, "truncated", False))
        if terminated and not truncated:
            if policy_info.battery <= 0:
                reward -= 4.0
            else:
                reward -= 8.0

        self.prev_clean_score = clean_score
        self.prev_known_cells += float(policy_info.new_known_cells)
        self.prev_target_distance = float(policy_info.target_distance)
        self.prev_charger_distance = float(policy_info.charger_distance)
        self.prev_battery_ratio = float(policy_info.battery_ratio)
        return float(reward)

    def _extract_observation(self, env_obs: Any) -> Any:
        if isinstance(env_obs, dict) and "observation" in env_obs:
            return env_obs.get("observation")
        return env_obs

    def _extract_hero(self, frame_state: Any) -> Any:
        heroes = self._get(frame_state, "heroes", {})
        if isinstance(heroes, (list, tuple)):
            return heroes[0] if heroes else {}
        return heroes

    def _extract_map_grid(self, obs: Any) -> np.ndarray:
        map_info = self._get(obs, "map_info", None)
        if map_info is None:
            return np.ones((21, 21), dtype=np.float32)
        arr = np.asarray(map_info, dtype=np.float32)
        if arr.ndim != 2:
            return np.ones((21, 21), dtype=np.float32)
        return arr

    def _resize_feature(self, feature: np.ndarray, target_dim: int) -> np.ndarray:
        out = np.zeros((target_dim,), dtype=np.float32)
        n = min(target_dim, feature.shape[0])
        out[:n] = feature[:n]
        return out

    @staticmethod
    def _get(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        if obj is None:
            return default
        return getattr(obj, key, default)

    @classmethod
    def _get_obs_level(cls, env_obs: Any, key: str, default: Any = None) -> Any:
        if isinstance(env_obs, dict) and key in env_obs:
            return env_obs.get(key, default)
        obs = cls._extract_observation_static(env_obs)
        return cls._get(obs, key, default)

    @classmethod
    def _extract_observation_static(cls, env_obs: Any) -> Any:
        if isinstance(env_obs, dict) and "observation" in env_obs:
            return env_obs.get("observation")
        return env_obs

    @staticmethod
    def _safe_float(v: Any, default: float) -> float:
        try:
            return float(v)
        except (TypeError, ValueError):
            return float(default)

