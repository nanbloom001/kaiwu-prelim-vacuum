#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Hybrid DIY robot-vacuum agent.
"""

import os
import time

import numpy as np
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

from agent_diy.algorithm.algorithm import Algorithm
from agent_diy.conf.conf import Config
from agent_diy.feature.definition import ActData, ObsData
from agent_diy.feature.preprocessor import Preprocessor
from agent_diy.model.model import Model
from kaiwudrl.interface.agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.model = Model(device).to(self.device)
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(),
            lr=Config.INIT_LEARNING_RATE_START,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)

        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0
        self.train_step = 0
        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs=None):
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0

    def observation_process(self, env_obs, preprocessor=None, extra_info=None):
        normalized_env_obs = self._normalize_env_obs(env_obs, preprocessor, extra_info)
        feature, legal_action, reward = self.preprocessor.feature_process(normalized_env_obs, self.last_action)
        self.last_reward = reward

        obs_data = ObsData(
            feature=list(feature),
            legal_action=list(legal_action),
            legal_act=list(legal_action),
            teacher_action=int(self.preprocessor.get_pending_action()),
            teacher_prob=list(self.preprocessor.get_pending_prob()),
            teacher_force=bool(self.preprocessor.should_force_teacher()),
            teacher_mix_bias=float(self.preprocessor.get_teacher_mix_bias()),
        )
        remain_info = {
            "goal_kind": self.preprocessor.goal_kind,
            "goal": self.preprocessor.goal,
            "reward": reward,
        }
        return obs_data, remain_info

    def predict(self, list_obs_data):
        if not list_obs_data:
            return []

        obs_data = list_obs_data[0]
        legal_action = getattr(obs_data, "legal_action", None)
        if legal_action is None:
            legal_action = getattr(obs_data, "legal_act", [1] * Config.ACTION_DIM)
        teacher_prob = getattr(obs_data, "teacher_prob", None)
        teacher_force = bool(getattr(obs_data, "teacher_force", False))
        teacher_mix_bias = float(getattr(obs_data, "teacher_mix_bias", 0.0) or 0.0)
        legal_arr = np.array(legal_action, dtype=np.float32)

        try:
            if teacher_force:
                return [self._teacher_fallback_act_data(obs_data, legal_arr)]

            logits, value = self._run_model(obs_data.feature)
            model_prob = self._legal_soft_max(logits, legal_arr)
            mixed_prob = self._mix_with_teacher(model_prob, teacher_prob, legal_arr, teacher_mix_bias)
            use_safe_greedy = self.train_step < Config.SAFE_GREEDY_WARMUP_STEPS or teacher_mix_bias > 0.0
            action = self._legal_sample(mixed_prob, use_max=use_safe_greedy)
            d_action = self._legal_sample(mixed_prob, use_max=True)
            return [
                ActData(
                    act=[action],
                    action=[action],
                    d_action=[d_action],
                    prob=list(mixed_prob),
                    probs=list(mixed_prob),
                    value=value,
                    values=value,
                )
            ]
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"predict fallback to teacher policy because: {exc}")
            return [self._teacher_fallback_act_data(obs_data, legal_arr)]

    def exploit(self, env_obs):
        if isinstance(env_obs, list):
            act_data = self.predict(env_obs)[0]
            return self.action_process(act_data, is_stochastic=False)

        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])[0]
        return self.action_process(act_data, is_stochastic=False)

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def learn(self, list_sample_data):
        result = self.algorithm.learn(list_sample_data)
        self.train_step = self.algorithm.train_step
        return result

    def save_model(self, path=None, id="1"):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "ckpt")

        os.makedirs(path, exist_ok=True)
        model_file_path = os.path.join(path, f"model.ckpt-{id}.pkl")
        checkpoint = {
            "state_dict": {k: v.detach().cpu() for k, v in self.model.state_dict().items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metadata": {
                "checkpoint_tag": Config.CHECKPOINT_TAG,
                "timestamp": time.time(),
                "feature_dim": Config.FEATURE_DIM,
                "train_step": self.algorithm.train_step,
            },
        }
        torch.save(checkpoint, model_file_path)
        if self.logger:
            self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "ckpt")

        model_file_path = os.path.join(path, f"model.ckpt-{id}.pkl")
        if not os.path.exists(model_file_path):
            if self.logger:
                self.logger.warning(f"model file {model_file_path} not found, skip loading")
            return

        checkpoint = torch.load(model_file_path, map_location=self.device)
        metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
        if not metadata or metadata.get("checkpoint_tag") != Config.CHECKPOINT_TAG:
            if self.logger:
                self.logger.warning(
                    f"skip loading {model_file_path}: checkpoint_tag="
                    f"{None if metadata is None else metadata.get('checkpoint_tag')}, expected={Config.CHECKPOINT_TAG}"
                )
            return

        state_dict = checkpoint.get("state_dict", checkpoint)
        try:
            self.model.load_state_dict(state_dict)
        except RuntimeError as exc:
            if self.logger:
                self.logger.warning(f"strict load failed for {model_file_path}: {exc}")
            current_state = self.model.state_dict()
            matched_state = {}
            for key, value in state_dict.items():
                if key in current_state and current_state[key].shape == value.shape:
                    matched_state[key] = value
            if not matched_state:
                return
            current_state.update(matched_state)
            self.model.load_state_dict(current_state)

        self.train_step = int(metadata.get("train_step", 0))
        self.algorithm.train_step = self.train_step

        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            try:
                self.optimizer.load_state_dict(optimizer_state)
            except Exception:
                pass

        if self.logger:
            self.logger.info(f"load model {model_file_path} successfully")

    def _normalize_env_obs(self, env_obs, preprocessor=None, extra_info=None):
        if isinstance(env_obs, dict) and "observation" in env_obs:
            return env_obs

        observation = dict(env_obs) if isinstance(env_obs, dict) else {}
        state = None
        if isinstance(preprocessor, dict):
            state = preprocessor
        if isinstance(extra_info, dict):
            state = extra_info

        if state is not None and "env_info" not in observation:
            observation["env_info"] = state
        if "frame_state" not in observation:
            observation["frame_state"] = observation.get("frame_state", {})
        return {"observation": observation}

    def _run_model(self, feature):
        self.model.set_eval_mode()
        obs_tensor = torch.tensor(np.array([feature], dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            logits, value = self.model(obs_tensor, inference=True)
        return logits.cpu().numpy()[0], value.cpu().numpy()[0]

    def _mix_with_teacher(self, model_prob, teacher_prob, legal_arr, teacher_mix_bias=0.0):
        teacher_mix = self._get_infer_teacher_mix(teacher_mix_bias)
        teacher_prob = self._normalize_teacher_prob(teacher_prob, legal_arr)
        mixed = (1.0 - teacher_mix) * model_prob + teacher_mix * teacher_prob
        return self._safe_normalize_prob(mixed, legal_arr)

    def _get_infer_teacher_mix(self, teacher_mix_bias=0.0):
        teacher_mix = max(self.algorithm.get_teacher_mix(), Config.INFER_TEACHER_MIX_FLOOR)
        if self.train_step < Config.STUDENT_WARMUP_STEPS:
            teacher_mix = max(teacher_mix, Config.INFER_TEACHER_MIX_WARMUP)
        teacher_mix = min(1.0, teacher_mix + teacher_mix_bias)
        return float(teacher_mix)

    def _normalize_teacher_prob(self, teacher_prob, legal_arr):
        if teacher_prob is None:
            return self._uniform_legal_prob(legal_arr)
        teacher_arr = np.clip(np.array(teacher_prob, dtype=np.float32), 0.0, None)
        return self._safe_normalize_prob(teacher_arr, legal_arr)

    def _uniform_legal_prob(self, legal_arr):
        if float(np.sum(legal_arr)) <= 0.0:
            return np.full(Config.ACTION_DIM, 1.0 / Config.ACTION_DIM, dtype=np.float32)
        return self._safe_normalize_prob(np.array(legal_arr, dtype=np.float32), legal_arr)

    def _legal_soft_max(self, logits, legal_action):
        penalty = 1e20
        eps = 1e-5
        masked = logits - penalty * (1.0 - legal_action)
        masked = np.clip(masked - np.max(masked, keepdims=True), -penalty, 1.0)
        masked = (np.exp(masked) + eps) * legal_action
        return self._safe_normalize_prob(masked, legal_action)

    def _legal_sample(self, probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        safe_probs = self._safe_normalize_prob(probs)
        cumulative = np.cumsum(safe_probs, dtype=np.float64)
        cumulative[-1] = 1.0
        sample = float(np.random.random())
        return int(np.searchsorted(cumulative, sample, side="right"))

    def _safe_normalize_prob(self, probs, legal_arr=None):
        arr = np.array(probs, dtype=np.float64)
        arr = np.clip(arr, 0.0, None)
        if legal_arr is not None:
            arr = arr * np.array(legal_arr, dtype=np.float64)

        total = float(np.sum(arr))
        if total <= 0.0:
            if legal_arr is None:
                return np.full(Config.ACTION_DIM, 1.0 / Config.ACTION_DIM, dtype=np.float32)
            legal = np.array(legal_arr, dtype=np.float64)
            legal_total = float(np.sum(legal))
            if legal_total <= 0.0:
                return np.full(Config.ACTION_DIM, 1.0 / Config.ACTION_DIM, dtype=np.float32)
            arr = legal / legal_total
        else:
            arr = arr / total

        positive_idx = np.where(arr > 0.0)[0]
        if positive_idx.size == 0:
            return np.full(Config.ACTION_DIM, 1.0 / Config.ACTION_DIM, dtype=np.float32)

        last_idx = int(positive_idx[-1])
        remainder = 1.0 - float(np.sum(arr[:last_idx])) - float(np.sum(arr[last_idx + 1 :]))
        arr[last_idx] = max(remainder, 0.0)

        final_total = float(np.sum(arr))
        if final_total <= 0.0:
            return np.full(Config.ACTION_DIM, 1.0 / Config.ACTION_DIM, dtype=np.float32)
        arr = arr / final_total
        return arr.astype(np.float32)

    def _teacher_fallback_act_data(self, obs_data, legal_arr):
        teacher_prob = self._normalize_teacher_prob(getattr(obs_data, "teacher_prob", None), legal_arr)
        teacher_action = getattr(obs_data, "teacher_action", None)
        if teacher_action is None or teacher_action < 0 or teacher_action >= Config.ACTION_DIM or legal_arr[teacher_action] <= 0:
            teacher_action = int(np.argmax(teacher_prob))
        value = np.zeros((1,), dtype=np.float32)
        return ActData(
            act=[teacher_action],
            action=[teacher_action],
            d_action=[teacher_action],
            prob=list(teacher_prob),
            probs=list(teacher_prob),
            value=value,
            values=value,
        )
