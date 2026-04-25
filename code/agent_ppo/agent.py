#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum PPO agent with planner-guided residual policy.
"""

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from agent_ppo.algorithm.algorithm import Algorithm, CoveragePlanner
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from kaiwudrl.interface.agent import BaseAgent


class Agent(BaseAgent):

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = torch.device("cpu")

        self.model = Model(self.device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.logger = logger
        self.monitor = monitor

        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
        self.preprocessor = Preprocessor()
        self.planner = CoveragePlanner()

        self.last_action = -1
        self.last_reward = 0.0

        super().__init__(agent_type, self.device, logger, monitor)

    def set_episode_config(self, max_step=None, robot_count=None, charger_count=None, battery_max=None):
        self.preprocessor.set_episode_config(
            max_step=max_step,
            robot_count=robot_count,
            charger_count=charger_count,
            battery_max=battery_max,
        )
        self.planner.set_episode_config(
            max_step=max_step,
            robot_count=robot_count,
            charger_count=charger_count,
            battery_max=battery_max,
        )

    def reset(self, env_obs):
        del env_obs
        self.preprocessor = Preprocessor()
        self.planner.reset()
        self.last_action = -1
        self.last_reward = 0.0

    def observation_process(self, env_obs):
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        self.last_reward = reward

        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        return obs_data, {}

    def action_process(self, act_data, is_stochastic: bool = True) -> int:
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def predict(self, list_obs_data: list) -> list:
        """Pure policy prediction on the env legal-action mask."""
        obs_data = list_obs_data[0]
        logits, value = self._run_model(obs_data.feature)

        legal_arr = np.asarray(obs_data.legal_action, dtype=np.float32)
        prob = self._legal_softmax(logits, legal_arr)
        action = self._sample(prob, use_max=False)
        d_action = self._sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
                policy_prob=list(prob),
                planner_prob=list(prob),
                mix_alpha=[1.0],
                action_mask=list(legal_arr),
            )
        ]

    def guided_predict(self, list_obs_data: list, policy_info, residual_alpha: float) -> list:
        """
        Planner-guided residual policy.
        final_prob = (1 - alpha) * planner_prob + alpha * policy_prob
        """
        obs_data = list_obs_data[0]
        logits, value = self._run_model(obs_data.feature)

        action_mask = self._build_action_mask(obs_data.legal_action, policy_info)
        policy_prob = self._legal_softmax(logits, action_mask)
        planner_prob = self._planner_prior(policy_info, action_mask)
        alpha = self._effective_residual_alpha(residual_alpha, policy_info)
        mix_prob = self._mix_prob(policy_prob, planner_prob, alpha, action_mask)

        action = self._sample(mix_prob, use_max=False)
        d_action = self._sample(mix_prob, use_max=True)
        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(mix_prob),
                value=value,
                policy_prob=list(policy_prob),
                planner_prob=list(planner_prob),
                mix_alpha=[alpha],
                action_mask=list(action_mask),
            )
        ]

    def exploit(self, env_obs) -> int:
        obs_data, _ = self.observation_process(env_obs)
        policy_info = self.planner.update(env_obs, self.last_action)
        self.preprocessor.set_policy_context(
            target_mode=getattr(policy_info, "target_mode", ""),
            should_charge=getattr(policy_info, "should_charge", False),
        )
        act_data = self.guided_predict(
            [obs_data],
            policy_info=policy_info,
            residual_alpha=Config.RESIDUAL_ALPHA_MAX,
        )[0]
        self.last_action = int(act_data.d_action[0])
        return self.last_action

    def learn(self, list_sample_data: list) -> dict:
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self.logger.info(f"load model {model_file_path} successfully")

    def _run_model(self, feature: list):
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(
            np.asarray([feature], dtype=np.float32)
        ).view(1, Config.DIM_OF_OBSERVATION).to(self.device)

        with torch.no_grad():
            rst = self.model(obs_tensor, inference=True)

        logits = rst[0].cpu().numpy()[0]
        value = rst[1].cpu().numpy()[0]
        return logits, value

    def _build_action_mask(self, legal_action, policy_info) -> np.ndarray:
        legal_arr = np.asarray(legal_action, dtype=np.float32)
        if policy_info is None or getattr(policy_info, "safe_action_mask", None) is None:
            return legal_arr

        safe_arr = np.asarray(policy_info.safe_action_mask, dtype=np.float32)
        action_mask = legal_arr * safe_arr
        if float(action_mask.sum()) <= 0.5:
            action_mask = legal_arr
        if float(action_mask.sum()) <= 0.5:
            action_mask = np.ones((Config.ACTION_NUM,), dtype=np.float32)
        return action_mask.astype(np.float32)

    def _effective_residual_alpha(self, residual_alpha: float, policy_info) -> float:
        alpha = float(np.clip(residual_alpha, 0.0, 1.0))
        if policy_info is None:
            return alpha
        charger_distance = float(getattr(policy_info, "charger_distance", 999.0))
        battery = float(getattr(policy_info, "battery", 0.0))
        target_mode = getattr(policy_info, "target_mode", "")
        should_charge = bool(getattr(policy_info, "should_charge", False))
        if (
            should_charge
            and target_mode == "charge"
            and np.isfinite(charger_distance)
            and charger_distance < 900.0
        ):
            if battery <= charger_distance + 16.0:
                return 0.0
            if battery <= charger_distance + 22.0:
                alpha = min(alpha, 0.001)
        if getattr(policy_info, "target_mode", "") == "charge":
            return min(alpha, Config.RESIDUAL_ALPHA_CHARGE_CAP)
        if getattr(policy_info, "target_mode", "") == "fallback":
            return min(alpha, Config.RESIDUAL_ALPHA_FALLBACK_CAP)
        return alpha

    def _planner_prior(self, policy_info, action_mask: np.ndarray) -> np.ndarray:
        if policy_info is None or getattr(policy_info, "action_scores", None) is None:
            return self._uniform_prob(action_mask)

        scores = np.asarray(policy_info.action_scores, dtype=np.float32)
        valid = action_mask > 0.5
        if not np.any(valid):
            return self._uniform_prob(action_mask)

        valid_scores = scores[valid]
        score_std = float(valid_scores.std())
        if not np.isfinite(score_std) or score_std < 1e-6:
            normalized = np.zeros_like(scores, dtype=np.float32)
        else:
            normalized = (scores - float(valid_scores.mean())) / (score_std + 1e-6)
        normalized = normalized / Config.PLANNER_PRIOR_TEMPERATURE
        return self._legal_softmax(normalized, action_mask)

    def _uniform_prob(self, action_mask: np.ndarray) -> np.ndarray:
        mask = np.asarray(action_mask, dtype=np.float64)
        if float(mask.sum()) <= 0.5:
            prob = np.full((Config.ACTION_NUM,), 1.0 / Config.ACTION_NUM, dtype=np.float64)
        else:
            prob = mask
        return self._normalize_prob(prob, action_mask)

    def _mix_prob(
        self,
        policy_prob: np.ndarray,
        planner_prob: np.ndarray,
        residual_alpha: float,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        alpha = float(np.clip(residual_alpha, 0.0, 1.0))
        mix_prob = (1.0 - alpha) * planner_prob + alpha * policy_prob
        return self._normalize_prob(mix_prob, action_mask)

    def _legal_softmax(self, logits: np.ndarray, legal_action: np.ndarray) -> np.ndarray:
        _w, _eps = 1e20, 1e-6
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _eps) * legal_action
        return self._normalize_prob(tmp, legal_action)

    def _sample(self, probs: np.ndarray, use_max: bool = False) -> int:
        probs = self._normalize_prob(probs)
        if use_max:
            return int(np.argmax(probs))
        cdf = np.cumsum(np.asarray(probs, dtype=np.float64))
        cdf[-1] = 1.0
        rand = float(np.random.random())
        return int(np.searchsorted(cdf, rand, side="right"))

    def _normalize_prob(self, probs: np.ndarray, action_mask: np.ndarray | None = None) -> np.ndarray:
        prob = np.asarray(probs, dtype=np.float64).reshape(-1)
        if prob.size != Config.ACTION_NUM:
            raise ValueError(f"Action probability size mismatch: {prob.size} vs {Config.ACTION_NUM}")

        prob = np.nan_to_num(prob, nan=0.0, posinf=0.0, neginf=0.0)
        prob = np.clip(prob, 0.0, None)

        if action_mask is not None:
            mask = np.asarray(action_mask, dtype=np.float64).reshape(-1)
            if mask.size != prob.size:
                raise ValueError(f"Action mask size mismatch: {mask.size} vs {prob.size}")
            mask = np.clip(mask, 0.0, 1.0)
            prob *= mask
        else:
            mask = np.ones_like(prob, dtype=np.float64)

        if float(prob.sum()) <= 1e-12:
            if float(mask.sum()) <= 1e-12:
                prob = np.full((Config.ACTION_NUM,), 1.0 / Config.ACTION_NUM, dtype=np.float64)
            else:
                prob = mask.copy()

        prob_sum = float(prob.sum(dtype=np.float64))
        if prob_sum <= 1e-12:
            prob = np.full((Config.ACTION_NUM,), 1.0 / Config.ACTION_NUM, dtype=np.float64)
        else:
            prob = prob / prob_sum

        # Keep probabilities numerically safe for downstream sampling.
        prob = np.clip(prob, 0.0, 1.0)
        tail = 1.0 - float(prob[:-1].sum(dtype=np.float64))
        prob[-1] = max(0.0, tail)

        final_sum = float(prob.sum(dtype=np.float64))
        if final_sum <= 1e-12:
            prob = np.full((Config.ACTION_NUM,), 1.0 / Config.ACTION_NUM, dtype=np.float64)
        else:
            prob = prob / final_sum

        return prob.astype(np.float32)
