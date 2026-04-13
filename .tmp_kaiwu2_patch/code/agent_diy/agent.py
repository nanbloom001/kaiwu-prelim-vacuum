#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Hierarchical DIY robot-vacuum agent.
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
            candidate_feature=list(self.preprocessor.get_candidate_feature()),
            candidate_mask=list(self.preprocessor.get_candidate_mask()),
            decision_event=bool(self.preprocessor.is_decision_event()),
            teacher_action=int(self.preprocessor.get_pending_action()),
            teacher_prob=list(self.preprocessor.get_pending_prob()),
            teacher_force=bool(self.preprocessor.should_force_teacher()),
            teacher_mix_bias=float(self.preprocessor.get_teacher_mix_bias()),
            teacher_weight=float(self.preprocessor.get_teacher_weight()),
            policy_weight=float(self.preprocessor.get_policy_weight()),
            teacher_candidate=int(self.preprocessor.get_teacher_candidate_index()),
            teacher_candidate_prob=list(self.preprocessor.get_teacher_candidate_prob()),
            teacher_path_style=int(self.preprocessor.get_teacher_path_style_index()),
            teacher_path_style_prob=list(self.preprocessor.get_teacher_path_style_prob()),
        )
        remain_info = {
            "goal_kind": self.preprocessor.goal_kind,
            "goal": self.preprocessor.goal,
            "reward": reward,
            "planner_mode": self.preprocessor.planner_mode,
            "active_region_id": self.preprocessor.active_region_id,
            "active_region_type": self.preprocessor.active_region_type,
            "active_mouth_id": self.preprocessor.active_mouth_id,
            "charger_halo_waste_steps": self.preprocessor.charger_halo_waste_steps,
            "plan_churn_count": self.preprocessor.plan_churn_count,
            "frontier_skip_steps": self.preprocessor.frontier_skip_steps,
            "spine_transit_steps": self.preprocessor.spine_transit_steps,
            "blocked_cell_count": self.preprocessor.get_blocked_cell_count(),
            "stuck_chain": self.preprocessor.stuck_chain,
            "completed_region_count": self.preprocessor.completed_region_count,
            "decision_event": bool(self.preprocessor.is_decision_event()),
            "decision_span": int(self.preprocessor.get_decision_span()),
            "total_score": int(self.preprocessor.score),
            "charge_count": int(self.preprocessor.charge_count),
            "remaining_charge": int(self.preprocessor.battery),
        }
        return obs_data, remain_info

    def predict(self, list_obs_data):
        if not list_obs_data:
            return []

        obs_data = list_obs_data[0]
        legal_action = getattr(obs_data, "legal_action", None)
        if legal_action is None:
            legal_action = getattr(obs_data, "legal_act", [1] * Config.ACTION_DIM)
        teacher_force = bool(getattr(obs_data, "teacher_force", False))
        teacher_mix_bias = float(getattr(obs_data, "teacher_mix_bias", 0.0) or 0.0)
        decision_event = bool(getattr(obs_data, "decision_event", False))
        legal_arr = np.array(legal_action, dtype=np.float32)

        try:
            if not decision_event:
                return [self._executor_act_data(obs_data, legal_arr)]

            if Config.TEACHER_ONLY or teacher_force:
                self.preprocessor.apply_decision(
                    int(getattr(obs_data, "teacher_candidate", 0)),
                    int(getattr(obs_data, "teacher_path_style", 1)),
                )
                return [self._executor_act_data(obs_data, legal_arr, decision_event=True)]

            candidate_logits, style_logits, value = self._run_model(
                obs_data.feature,
                obs_data.candidate_feature,
                obs_data.candidate_mask,
            )
            candidate_prob = self._candidate_softmax(candidate_logits, np.asarray(obs_data.candidate_mask, dtype=np.float32))
            style_prob = self._style_softmax(style_logits)

            infer_teacher_mix = self._get_infer_teacher_mix(teacher_mix_bias)
            if infer_teacher_mix > 1e-6:
                candidate_prob = self._mix_with_teacher_candidate(
                    candidate_prob,
                    getattr(obs_data, "teacher_candidate_prob", None),
                    np.asarray(obs_data.candidate_mask, dtype=np.float32),
                    infer_teacher_mix,
                )
                style_prob = self._mix_with_teacher_style(style_prob, getattr(obs_data, "teacher_path_style_prob", None), infer_teacher_mix)

            use_safe_greedy = self.train_step < Config.SAFE_GREEDY_WARMUP_STEPS and infer_teacher_mix > 0.0
            decision_action = self._categorical_sample(candidate_prob, use_max=use_safe_greedy)
            path_style_action = self._categorical_sample(style_prob, use_max=use_safe_greedy)

            self.preprocessor.apply_decision(int(decision_action), int(path_style_action))
            executor = self._executor_act_data(
                obs_data,
                legal_arr,
                decision_event=True,
                decision_action=int(decision_action),
                decision_prob=candidate_prob,
                path_style_action=int(path_style_action),
                path_style_prob=style_prob,
                decision_value=value,
            )
            return [executor]
        except Exception as exc:
            if self.logger:
                self.logger.warning(f"predict fallback to teacher policy because: {exc}")
            self.preprocessor.apply_decision(
                int(getattr(obs_data, "teacher_candidate", 0)),
                int(getattr(obs_data, "teacher_path_style", 1)),
            )
            return [self._executor_act_data(obs_data, legal_arr, decision_event=True)]

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
        if Config.TEACHER_ONLY:
            return {
                "total_loss": 0.0,
                "reward": 0.0,
                "decision_policy_loss": 0.0,
                "style_policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy_loss": 0.0,
                "imitation_loss": 0.0,
                "teacher_mix": 1.0,
                "imitation_coef": 0.0,
                "teacher_weight": 1.0,
                "policy_weight": 0.0,
            }
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
                "candidate_dim": Config.CANDIDATE_FLAT_DIM,
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

    def _run_model(self, feature, candidate_feature, candidate_mask):
        self.model.set_eval_mode()
        state_tensor = torch.tensor(np.array([feature], dtype=np.float32), dtype=torch.float32, device=self.device)
        candidate_tensor = torch.tensor(np.array([candidate_feature], dtype=np.float32), dtype=torch.float32, device=self.device)
        mask_tensor = torch.tensor(np.array([candidate_mask], dtype=np.float32), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            candidate_logits, style_logits, value = self.model(state_tensor, candidate_tensor, mask_tensor, inference=True)
        return candidate_logits.cpu().numpy()[0], style_logits.cpu().numpy()[0], value.cpu().numpy()[0]

    def _mix_with_teacher_candidate(self, model_prob, teacher_prob, candidate_mask, teacher_mix):
        teacher_prob = self._normalize_teacher_candidate_prob(teacher_prob, candidate_mask)
        mixed = (1.0 - teacher_mix) * model_prob + teacher_mix * teacher_prob
        return self._safe_candidate_prob(mixed, candidate_mask)

    def _mix_with_teacher_style(self, model_prob, teacher_prob, teacher_mix):
        teacher_prob = self._normalize_teacher_style_prob(teacher_prob)
        mixed = (1.0 - teacher_mix) * model_prob + teacher_mix * teacher_prob
        return self._safe_style_prob(mixed)

    def _get_infer_teacher_mix(self, teacher_mix_bias=0.0):
        if self.train_step >= Config.STUDENT_WARMUP_STEPS:
            return 0.0
        teacher_mix = max(self.algorithm.get_teacher_mix(), Config.INFER_TEACHER_MIX_WARMUP)
        teacher_mix = max(teacher_mix, teacher_mix_bias)
        return float(np.clip(teacher_mix, Config.INFER_TEACHER_MIX_FLOOR, 1.0))

    def _candidate_softmax(self, logits, candidate_mask):
        penalty = 1e20
        eps = 1e-5
        masked = logits - penalty * (1.0 - candidate_mask)
        masked = np.clip(masked - np.max(masked, keepdims=True), -penalty, 1.0)
        masked = (np.exp(masked) + eps) * candidate_mask
        return self._safe_candidate_prob(masked, candidate_mask)

    def _style_softmax(self, logits):
        masked = np.clip(logits - np.max(logits, keepdims=True), -30.0, 20.0)
        return self._safe_style_prob(np.exp(masked))

    def _categorical_sample(self, probs, use_max=False):
        if use_max:
            return int(np.argmax(probs))
        safe_probs = np.array(probs, dtype=np.float64)
        cumulative = np.cumsum(safe_probs, dtype=np.float64)
        cumulative[-1] = 1.0
        sample = float(np.random.random())
        return int(np.searchsorted(cumulative, sample, side="right"))

    def _normalize_teacher_candidate_prob(self, teacher_prob, candidate_mask):
        if teacher_prob is None:
            return self._safe_candidate_prob(np.array(candidate_mask, dtype=np.float32), candidate_mask)
        teacher_arr = np.clip(np.array(teacher_prob, dtype=np.float32), 0.0, None)
        return self._safe_candidate_prob(teacher_arr, candidate_mask)

    def _normalize_teacher_style_prob(self, teacher_prob):
        if teacher_prob is None:
            return np.full(Config.PATH_STYLE_DIM, 1.0 / Config.PATH_STYLE_DIM, dtype=np.float32)
        teacher_arr = np.clip(np.array(teacher_prob, dtype=np.float32), 0.0, None)
        return self._safe_style_prob(teacher_arr)

    def _safe_candidate_prob(self, probs, candidate_mask=None):
        arr = np.array(probs, dtype=np.float64)
        arr = np.clip(arr, 0.0, None)
        if candidate_mask is not None:
            arr = arr * np.array(candidate_mask, dtype=np.float64)
        total = float(np.sum(arr))
        if total <= 0.0:
            if candidate_mask is None:
                return np.full(Config.MAX_DECISION_CANDIDATES, 1.0 / Config.MAX_DECISION_CANDIDATES, dtype=np.float32)
            mask = np.array(candidate_mask, dtype=np.float64)
            mask_total = float(np.sum(mask))
            if mask_total <= 0.0:
                return np.full(Config.MAX_DECISION_CANDIDATES, 1.0 / Config.MAX_DECISION_CANDIDATES, dtype=np.float32)
            arr = mask / mask_total
        else:
            arr = arr / total
        return arr.astype(np.float32)

    def _safe_style_prob(self, probs):
        arr = np.array(probs, dtype=np.float64)
        arr = np.clip(arr, 0.0, None)
        total = float(np.sum(arr))
        if total <= 0.0:
            return np.full(Config.PATH_STYLE_DIM, 1.0 / Config.PATH_STYLE_DIM, dtype=np.float32)
        return (arr / total).astype(np.float32)

    def _executor_act_data(
        self,
        obs_data,
        legal_arr,
        decision_event=False,
        decision_action=None,
        decision_prob=None,
        path_style_action=None,
        path_style_prob=None,
        decision_value=None,
    ):
        teacher_prob = self._normalize_teacher_action_prob(self.preprocessor.get_pending_prob(), legal_arr)
        teacher_action = int(self.preprocessor.get_pending_action())
        if teacher_action < 0 or teacher_action >= Config.ACTION_DIM or legal_arr[teacher_action] <= 0:
            teacher_action = int(np.argmax(teacher_prob))

        if decision_action is None:
            decision_action = int(getattr(obs_data, "teacher_candidate", 0) or 0)
        if decision_prob is None:
            decision_prob = self._normalize_teacher_candidate_prob(
                getattr(obs_data, "teacher_candidate_prob", None),
                np.asarray(getattr(obs_data, "candidate_mask", [1] * Config.MAX_DECISION_CANDIDATES), dtype=np.float32),
            )
        if path_style_action is None:
            path_style_action = int(getattr(obs_data, "teacher_path_style", 1) or 1)
        if path_style_prob is None:
            path_style_prob = self._normalize_teacher_style_prob(getattr(obs_data, "teacher_path_style_prob", None))
        if decision_value is None:
            decision_value = np.zeros((1,), dtype=np.float32)

        value = np.array(decision_value, dtype=np.float32).reshape(-1)[:1]
        return ActData(
            act=[teacher_action],
            action=[teacher_action],
            d_action=[teacher_action],
            prob=list(teacher_prob),
            probs=list(teacher_prob),
            value=value,
            values=value,
            decision_event=bool(decision_event),
            decision_action=[int(decision_action)],
            decision_prob=list(np.asarray(decision_prob, dtype=np.float32)),
            path_style_action=[int(path_style_action)],
            path_style_prob=list(np.asarray(path_style_prob, dtype=np.float32)),
            decision_value=value,
        )

    def _normalize_teacher_action_prob(self, teacher_prob, legal_arr):
        if teacher_prob is None:
            return self._uniform_legal_prob(legal_arr)
        teacher_arr = np.clip(np.array(teacher_prob, dtype=np.float32), 0.0, None)
        return self._safe_normalize_action_prob(teacher_arr, legal_arr)

    def _uniform_legal_prob(self, legal_arr):
        if float(np.sum(legal_arr)) <= 0.0:
            return np.full(Config.ACTION_DIM, 1.0 / Config.ACTION_DIM, dtype=np.float32)
        return self._safe_normalize_action_prob(np.array(legal_arr, dtype=np.float32), legal_arr)

    def _safe_normalize_action_prob(self, probs, legal_arr=None):
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
        return arr.astype(np.float32)
