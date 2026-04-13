#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Agent implementation for robot_vacuum.
"""

import glob
import os
from typing import Any, Optional, Tuple

import numpy as np
import torch

from common_python.config.config_control import CONFIG
from kaiwudrl.interface.agent import BaseAgent

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.feature.definition import ActData, Config, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        np.random.seed(0)

        self.device = torch.device(device if device is not None else "cpu")
        self.logger = logger
        self.monitor = monitor

        self.model = Model(device=self.device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            logger=self.logger,
            monitor=self.monitor,
        )
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_policy_output = None
        self._loaded_model_path: Optional[str] = None

        super().__init__(agent_type, self.device, logger, monitor)

    def reset(self, env_obs=None, *args, **kwargs):
        self.preprocessor.reset()
        self.last_action = -1
        self.last_policy_output = None

    def observation_process(self, env_obs, *args, **kwargs):
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        self.last_policy_output = self.preprocessor.get_policy_output()
        heuristic_action = None
        heuristic_scores = None
        if self.last_policy_output is not None:
            heuristic_action = [int(self.last_policy_output.chosen_action)]
            heuristic_scores = self.last_policy_output.action_scores.tolist()

        obs_data = ObsData(
            feature=feature.tolist(),
            legal_action=legal_action.tolist(),
            heuristic_action=heuristic_action,
            heuristic_scores=heuristic_scores,
        )
        remain_info = {"reward": [float(reward)]}
        return obs_data, remain_info

    def predict(self, list_obs_data, *args, **kwargs):
        if not isinstance(list_obs_data, list):
            list_obs_data = [list_obs_data]

        obs_data = list_obs_data[0]
        heuristic_action = getattr(obs_data, "heuristic_action", None)
        heuristic_scores = getattr(obs_data, "heuristic_scores", None)
        legal_action = np.asarray(obs_data.legal_action, dtype=np.float32)

        if heuristic_action is not None and heuristic_scores is not None:
            chosen_action = self._to_int_action(heuristic_action)
            scores = np.asarray(heuristic_scores, dtype=np.float32)
            prob = self._planner_scores_to_prob(scores, legal_action, chosen_action)
            greedy_action = int(np.argmax(prob))
            return [
                ActData(
                    action=[chosen_action],
                    d_action=[greedy_action],
                    prob=prob.tolist(),
                    value=[0.0],
                )
            ]

        feature = np.asarray(obs_data.feature, dtype=np.float32)
        logits, value, prob = self._run_model(feature, legal_action)
        greedy_action = self._legal_sample(prob, use_max=True)
        return [ActData(action=[greedy_action], d_action=[greedy_action], prob=prob.tolist(), value=[float(value)])]

    def exploit(self, data, *args, **kwargs):
        if isinstance(data, list):
            obs_data = data[0]
        elif hasattr(data, "feature") and hasattr(data, "legal_action"):
            obs_data = data
        else:
            obs_data, _ = self.observation_process(data)

        act_data = self.predict([obs_data])[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data, *args, **kwargs):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1", *args, **kwargs):
        ckpt_dir = self._resolve_ckpt_dir(path)
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_id = self._normalize_id_for_save(id, ckpt_dir)
        model_file_path = os.path.join(ckpt_dir, f"model.ckpt-{ckpt_id}.pkl")
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        self._log_info(f"save model {model_file_path} successfully")
        return True

    def load_model(self, path=None, id="1", *args, **kwargs):
        ckpt_dir = self._resolve_ckpt_dir(path)
        model_file_path = self._resolve_model_path_for_load(ckpt_dir, id)
        if model_file_path is None:
            self._log_warning(f"no model checkpoint found in {ckpt_dir}, skip load")
            return False
        if not os.path.exists(model_file_path):
            self._log_warning(f"model file {model_file_path} does not exist, skip load")
            return False
        if self._loaded_model_path == model_file_path:
            return True

        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        self._loaded_model_path = model_file_path
        self._log_info(f"load model {model_file_path} successfully")
        return True

    def action_process(self, act_data, is_stochastic=True, *args, **kwargs):
        raw_action = act_data.action if is_stochastic else act_data.d_action
        action = self._to_int_action(raw_action)
        action = max(0, min(Config.ACTION_NUM - 1, action))
        self.last_action = action
        return action

    def _planner_scores_to_prob(self, scores: np.ndarray, legal_action: np.ndarray, chosen_action: int) -> np.ndarray:
        legal = np.where(legal_action > 0.5, 1.0, 0.0).astype(np.float32)
        if legal.sum() <= 0:
            legal[:] = 1.0

        scores = scores.astype(np.float32).reshape(-1)
        if scores.shape[0] != Config.ACTION_NUM:
            fixed = np.full((Config.ACTION_NUM,), -1e9, dtype=np.float32)
            n = min(scores.shape[0], Config.ACTION_NUM)
            fixed[:n] = scores[:n]
            scores = fixed

        masked = scores.copy()
        masked[legal < 0.5] = -1e9
        if 0 <= chosen_action < Config.ACTION_NUM:
            masked[chosen_action] = max(masked[chosen_action], np.max(masked[legal > 0.5]) + 0.5)
        masked = masked - np.max(masked)
        exp_v = np.exp(masked) * legal
        denom = float(exp_v.sum())
        if denom <= 0.0:
            return legal / legal.sum()
        return exp_v / denom

    def _run_model(self, feature: np.ndarray, legal_action: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(feature, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)

        logits_np = logits.squeeze(0).detach().cpu().numpy()
        value_scalar = float(value.squeeze(0).detach().cpu().numpy()[0])
        legal_action = legal_action.astype(np.float32)
        if legal_action.shape[0] != Config.ACTION_NUM:
            fixed = np.ones(Config.ACTION_NUM, dtype=np.float32)
            n = min(Config.ACTION_NUM, legal_action.shape[0])
            fixed[:n] = legal_action[:n]
            legal_action = fixed

        prob = self._legal_softmax(logits_np, legal_action)
        return logits_np, value_scalar, prob

    def _legal_softmax(self, logits: np.ndarray, legal_action: np.ndarray) -> np.ndarray:
        legal = np.where(legal_action > 0.5, 1.0, 0.0).astype(np.float32)
        if legal.sum() <= 0:
            legal[:] = 1.0
        masked = logits.astype(np.float32).copy()
        masked[legal < 0.5] = -1e9
        masked = masked - np.max(masked)
        exp_v = np.exp(masked) * legal
        denom = float(exp_v.sum())
        if denom <= 0.0:
            return legal / legal.sum()
        return exp_v / denom

    def _legal_sample(self, probs: np.ndarray, use_max: bool = False) -> int:
        if use_max:
            return int(np.argmax(probs))
        return int(np.random.choice(len(probs), p=probs))

    def _resolve_ckpt_dir(self, path: Optional[str]) -> str:
        if path:
            return path
        return os.path.join(CONFIG.restore_dir, f"{CONFIG.app}_{CONFIG.algo}")

    def _resolve_model_path_for_load(self, ckpt_dir: str, ckpt_id: Any) -> Optional[str]:
        if ckpt_id in (None, "latest", "LATEST", "-1", -1):
            return self._find_latest_ckpt_file(ckpt_dir)
        return os.path.join(ckpt_dir, f"model.ckpt-{ckpt_id}.pkl")

    def _normalize_id_for_save(self, ckpt_id: Any, ckpt_dir: str) -> str:
        if ckpt_id in (None, "latest", "LATEST", "-1", -1):
            latest = self._find_latest_ckpt_id(ckpt_dir)
            return "0" if latest is None else str(latest + 1)
        return str(ckpt_id)

    def _find_latest_ckpt_file(self, ckpt_dir: str) -> Optional[str]:
        files = glob.glob(os.path.join(ckpt_dir, "model.ckpt-*.pkl"))
        if not files:
            return None
        files.sort(key=lambda f: self._extract_ckpt_id(f))
        return files[-1]

    def _find_latest_ckpt_id(self, ckpt_dir: str) -> Optional[int]:
        latest_file = self._find_latest_ckpt_file(ckpt_dir)
        return None if latest_file is None else self._extract_ckpt_id(latest_file)

    @staticmethod
    def _extract_ckpt_id(file_path: str) -> int:
        base = os.path.basename(file_path)
        suffix = base.replace("model.ckpt-", "").replace(".pkl", "")
        try:
            return int(suffix)
        except ValueError:
            return -1

    @staticmethod
    def _to_int_action(raw_action: Any) -> int:
        if isinstance(raw_action, np.ndarray):
            return 0 if raw_action.size == 0 else int(raw_action.reshape(-1)[0])
        if isinstance(raw_action, (list, tuple)):
            return 0 if not raw_action else int(raw_action[0])
        if torch.is_tensor(raw_action):
            return 0 if raw_action.numel() == 0 else int(raw_action.reshape(-1)[0].item())
        return int(raw_action)

    def _log_info(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)

    def _log_warning(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.warning(msg)


__all__ = ["Agent"]
