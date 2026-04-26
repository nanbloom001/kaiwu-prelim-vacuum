#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Robot Vacuum PPO agent with planner-guided residual policy on top of win infra.
"""

import os

import numpy as np
import torch

from agent_ppo.algorithm.algorithm import Algorithm, CoveragePlanner
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from agent_ppo.utils.experiment_archive import ExperimentArchive, parse_checkpoint_id
from kaiwudrl.interface.agent import BaseAgent

try:
    from common_python.config.config_control import CONFIG as KAIWU_CONFIG
    from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
    from kaiwudrl.interface.remote_agent import RemoteAgent
except Exception:
    KAIWU_CONFIG = None
    KaiwuDRLDefine = None
    RemoteAgent = None


def _env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return int(default)
    return int(value)


def _configure_torch_runtime(service_name):
    is_learner = "learner" in (service_name or "")
    try:
        torch.set_num_threads(_env_int("KAIWU_LEARNER_CPU_THREADS", Config.LEARNER_CPU_THREADS) if is_learner else 1)
    except RuntimeError:
        pass
    try:
        torch.set_num_interop_threads(
            _env_int("KAIWU_LEARNER_CPU_INTEROP_THREADS", Config.LEARNER_CPU_INTEROP_THREADS)
            if is_learner
            else 1
        )
    except RuntimeError:
        pass

    if torch.cuda.is_available():
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _patch_remote_agent_batch_learn():
    if RemoteAgent is None or KAIWU_CONFIG is None or KaiwuDRLDefine is None:
        return
    if getattr(RemoteAgent, "_robot_vacuum_batch_tensor_patched", False):
        return

    original_learn = RemoteAgent.learn

    def patched_learn(self, list_sample_data, *args, **kwargs):
        if list_sample_data is None:
            return None

        is_learner_call = kwargs.get("framework") or KAIWU_CONFIG.svr_name == KaiwuDRLDefine.SERVER_LEARNER
        prefers_batch_tensor = getattr(self, "PREFER_BATCH_TENSOR_LEARN", False) or getattr(
            self.__class__, "PREFER_BATCH_TENSOR_LEARN", False
        )
        business_learn = getattr(self.__class__, "_business_learn", None)

        if is_learner_call and prefers_batch_tensor and business_learn is not None:
            kwargs.pop("framework", None)
            if isinstance(list_sample_data, (torch.Tensor, np.ndarray)):
                return business_learn(self, list_sample_data, *args, **kwargs)

        return original_learn(self, list_sample_data, *args, **kwargs)

    RemoteAgent.learn = patched_learn
    RemoteAgent._robot_vacuum_batch_tensor_patched = True


_patch_remote_agent_batch_learn()


class Agent(BaseAgent):
    PREFER_BATCH_TENSOR_LEARN = Config.LEARNER_PREFER_BATCH_TENSOR

    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)

        self.service_name = os.getenv("KAIWU_SERVICE_NAME", "")
        _configure_torch_runtime(self.service_name)
        self.device = self._resolve_device(device)
        self.model = Model(self.device).to(self.device)
        optimizer_kwargs = {
            "params": self.model.parameters(),
            "lr": Config.INIT_LEARNING_RATE_START,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }
        if _env_bool("KAIWU_LEARNER_USE_FOREACH_OPTIMIZER", Config.LEARNER_ALLOW_FOREACH_OPTIMIZER):
            optimizer_kwargs["foreach"] = True
        try:
            self.optimizer = torch.optim.Adam(**optimizer_kwargs)
        except (TypeError, RuntimeError):
            optimizer_kwargs.pop("foreach", None)
            self.optimizer = torch.optim.Adam(**optimizer_kwargs)
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(self.model, self.optimizer, self.device, self.logger, self.monitor)
        self.preprocessor = Preprocessor()
        self.planner = CoveragePlanner()
        self.archive = ExperimentArchive()
        self.last_action = -1
        self.last_reward = 0.0
        self.current_model_ref = {
            "path": None,
            "id": None,
            "checkpoint_id": None,
        }

        self._try_resume_checkpoint()
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

    def action_process(self, act_data, is_stochastic=True):
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def predict(self, list_obs_data, use_hard_override=False):
        """Fallback pure-policy prediction on the env legal-action mask."""
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

    def guided_predict(self, list_obs_data, policy_info, residual_alpha):
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

    def exploit(self, env_obs):
        obs_data, _ = self.observation_process(env_obs)
        policy_info = self.planner.update(env_obs, self.last_action)
        act_data = self.guided_predict(
            [obs_data],
            policy_info=policy_info,
            residual_alpha=Config.RESIDUAL_ALPHA_WARMUP_TARGET,
        )[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        state_dict_cpu = {k: v.clone().cpu() for k, v in self.model.state_dict().items()}
        torch.save(state_dict_cpu, model_file_path)
        checkpoint_id = parse_checkpoint_id(model_file_path) or str(id)
        self.current_model_ref = {
            "path": model_file_path,
            "id": str(id),
            "checkpoint_id": checkpoint_id,
        }
        self.archive.log_checkpoint(
            {
                "event": "checkpoint_saved",
                "path": model_file_path,
                "id": str(id),
                "checkpoint_id": checkpoint_id,
            }
        )
        self.archive.log_event(
            "checkpoint_saved",
            {
                "path": model_file_path,
                "id": str(id),
                "checkpoint_id": checkpoint_id,
            },
        )
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
        checkpoint_id = parse_checkpoint_id(model_file_path) or str(id)
        self.current_model_ref = {
            "path": model_file_path,
            "id": str(id),
            "checkpoint_id": checkpoint_id,
        }
        self.archive.log_event(
            "checkpoint_loaded",
            {
                "path": model_file_path,
                "id": str(id),
                "checkpoint_id": checkpoint_id,
            },
        )
        self.logger.info(f"load model {model_file_path} successfully")

    def _run_model(self, feature):
        self.model.set_eval_mode()
        obs_tensor = (
            torch.tensor(np.asarray([feature], dtype=np.float32))
            .view(1, Config.DIM_OF_OBSERVATION)
            .to(self.device)
        )

        with torch.no_grad():
            rst = self.model(obs_tensor, inference=True)

        logits = rst[0].cpu().numpy()[0]
        value = rst[1].cpu().numpy()[0]
        return logits, value

    def _build_action_mask(self, legal_action, policy_info):
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

    def _effective_residual_alpha(self, residual_alpha, policy_info):
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
            if battery <= charger_distance + 14.0:
                return 0.0
            if battery <= charger_distance + 20.0:
                alpha = min(alpha, 0.002)
        if getattr(policy_info, "target_mode", "") == "charge":
            return min(alpha, Config.RESIDUAL_ALPHA_CHARGE_CAP)
        if getattr(policy_info, "target_mode", "") == "fallback":
            return min(alpha, Config.RESIDUAL_ALPHA_FALLBACK_CAP)
        return alpha

    def _planner_prior(self, policy_info, action_mask):
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

    def _uniform_prob(self, action_mask):
        mask = np.asarray(action_mask, dtype=np.float64)
        if float(mask.sum()) <= 0.5:
            prob = np.full((Config.ACTION_NUM,), 1.0 / Config.ACTION_NUM, dtype=np.float64)
        else:
            prob = mask
        return self._normalize_prob(prob, action_mask)

    def _mix_prob(self, policy_prob, planner_prob, residual_alpha, action_mask):
        alpha = float(np.clip(residual_alpha, 0.0, 1.0))
        mix_prob = (1.0 - alpha) * planner_prob + alpha * policy_prob
        return self._normalize_prob(mix_prob, action_mask)

    def _legal_softmax(self, logits, legal_action):
        _w, _eps = 1e20, 1e-6
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _eps) * legal_action
        return self._normalize_prob(tmp, legal_action)

    def _sample(self, probs, use_max=False):
        probs = self._normalize_prob(probs)
        if use_max:
            return int(np.argmax(probs))
        cdf = np.cumsum(np.asarray(probs, dtype=np.float64))
        cdf[-1] = 1.0
        rand = float(np.random.random())
        return int(np.searchsorted(cdf, rand, side="right"))

    def _normalize_prob(self, probs, action_mask=None):
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

        prob = np.clip(prob, 0.0, 1.0)
        tail = 1.0 - float(prob[:-1].sum(dtype=np.float64))
        prob[-1] = max(0.0, tail)

        final_sum = float(prob.sum(dtype=np.float64))
        if final_sum <= 1e-12:
            prob = np.full((Config.ACTION_NUM,), 1.0 / Config.ACTION_NUM, dtype=np.float64)
        else:
            prob = prob / final_sum

        return prob.astype(np.float32)

    def _resolve_device(self, device):
        if device is None:
            resolved = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, torch.device):
            resolved = device
        else:
            resolved = torch.device(device)

        if resolved.type == "cuda" and not torch.cuda.is_available():
            resolved = torch.device("cpu")

        return resolved

    def _try_resume_checkpoint(self):
        if not Config.RESUME_CHECKPOINT:
            return

        resume_candidates = [
            os.path.join(os.path.dirname(__file__), "..", Config.RESUME_CHECKPOINT),
            os.path.join("/workspace/code", Config.RESUME_CHECKPOINT),
        ]
        for resume_path in resume_candidates:
            if not os.path.isfile(resume_path):
                continue
            try:
                state_dict = torch.load(resume_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                if self.logger:
                    self.logger.info(f"[RESUME] Loaded from {resume_path} on {self.device}")
                break
            except Exception as exc:
                if self.logger:
                    self.logger.warning(f"[RESUME] Failed to load {resume_path}: {exc}")
