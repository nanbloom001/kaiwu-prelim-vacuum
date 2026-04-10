#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum Agent.
清扫大作战 Agent 主类。
"""

import os
from pathlib import Path

import torch

import numpy as np
from common_python.config.config_control import CONFIG as KAIWU_CONFIG

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from kaiwudrl.common.utils.kaiwudrl_define import KaiwuDRLDefine
from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.utils.experiment_archive import ExperimentArchive, parse_checkpoint_id

try:
    from kaiwudrl.interface.remote_agent import RemoteAgent
except Exception:
    RemoteAgent = None


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _configure_torch_runtime(service_name: str, device) -> None:
    is_learner = "learner" in service_name
    if is_learner:
        # Learner has noticeable CPU-side fetch/collate work. Avoid pinning it to 1 thread.
        try:
            torch.set_num_threads(int(os.getenv("KAIWU_LEARNER_CPU_THREADS", str(Config.LEARNER_CPU_THREADS))))
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(
                int(os.getenv("KAIWU_LEARNER_CPU_INTEROP_THREADS", str(Config.LEARNER_CPU_INTEROP_THREADS)))
            )
        except RuntimeError:
            pass
    else:
        try:
            torch.set_num_threads(1)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    if device is not None and getattr(device, "type", "") == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def _patch_remote_agent_batch_learn() -> None:
    if RemoteAgent is None or getattr(RemoteAgent, "_robot_vacuum_batch_tensor_patched", False):
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
            if kwargs.get("framework"):
                del kwargs["framework"]
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
        self.device = device
        _configure_torch_runtime(self.service_name, self.device)
        self.model = Model(device).to(self.device)
        self.use_amp = (
            "learner" in self.service_name
            and self.device is not None
            and getattr(self.device, "type", "") == "cuda"
            and _env_flag("KAIWU_LEARNER_USE_AMP", Config.LEARNER_USE_AMP)
        )
        optimizer_kwargs = {
            "params": self.model.parameters(),
            "lr": Config.INIT_LEARNING_RATE_START,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
        }
        if _env_flag("KAIWU_LEARNER_USE_FOREACH_OPTIMIZER", Config.LEARNER_ALLOW_FOREACH_OPTIMIZER):
            optimizer_kwargs["foreach"] = True
        if (
            self.use_amp
            and _env_flag("KAIWU_LEARNER_USE_FUSED_OPTIMIZER", Config.LEARNER_ALLOW_FUSED_OPTIMIZER)
        ):
            optimizer_kwargs["fused"] = True
        try:
            self.optimizer = torch.optim.Adam(**optimizer_kwargs)
        except TypeError:
            optimizer_kwargs.pop("fused", None)
            optimizer_kwargs.pop("foreach", None)
            self.optimizer = torch.optim.Adam(**optimizer_kwargs)
        self.logger = logger
        self.monitor = monitor
        self.algorithm = Algorithm(
            self.model,
            self.optimizer,
            self.device,
            self.logger,
            self.monitor,
            use_amp=self.use_amp,
        )
        self.preprocessor = Preprocessor()
        self.archive = ExperimentArchive()
        self.last_action = -1
        self.last_reward = 0.0
        self.current_model_ref = {
            "path": None,
            "id": None,
            "checkpoint_id": None,
        }
        self.enable_load_model_cache = _env_flag("KAIWU_AGENT_LOAD_MODEL_CACHE", Config.AGENT_LOAD_MODEL_CACHE)
        self._last_loaded_model_path = None
        self._last_loaded_model_mtime_ns = None
        self._model_load_call_count = 0
        self._model_load_reload_count = 0
        self._model_load_cache_hit_count = 0

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        """Reset per-episode state.

        每局开始时重置 Agent 内部状态。
        """
        self.preprocessor = Preprocessor()
        self.last_action = -1
        self.last_reward = 0.0

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData (enhanced feature vector + legal action mask).

        将原始 env_obs 转换为 ObsData（69D 特征 + 合法动作掩码）。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        self.last_reward = reward

        obs_data = ObsData(
            feature=list(feature),
            legal_action=legal_action,
        )
        remain_info = {}
        return obs_data, remain_info

    def action_process(self, act_data, is_stochastic=True):
        """Extract int action from ActData and update last_action.

        从 ActData 中取出动作整数并更新 last_action。
        """
        action = act_data.action if is_stochastic else act_data.d_action
        self.last_action = int(action[0])
        return self.last_action

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时推理（随机采样动作）。
        """
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        logits, value = self._run_model(feature)
        logits = self._blend_policy_logits(logits)

        legal_arr = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits, legal_arr)
        action = self._legal_sample(prob, use_max=False)
        d_action = self._legal_sample(prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时推理（贪心）。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data])[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        """Delegate to Algorithm for PPO update.

        委托给 Algorithm 执行训练。
        """
        return self.algorithm.learn(list_sample_data)

    def save_model(self, path=None, id="1"):
        """Save model checkpoint.

        保存模型检查点。
        """
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
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        self._model_load_call_count += 1
        model_mtime_ns = self._get_model_mtime_ns(Path(model_file_path))
        should_reload = True
        if self.enable_load_model_cache:
            should_reload = self._should_reload_model(model_file_path, model_mtime_ns)

        if should_reload:
            self.model.load_state_dict(torch.load(model_file_path, map_location=self.device))
            self._last_loaded_model_path = model_file_path
            self._last_loaded_model_mtime_ns = model_mtime_ns
            self._model_load_reload_count += 1
        else:
            self._model_load_cache_hit_count += 1

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

    def get_runtime_metrics(self):
        return {
            "load_model_calls": self._model_load_call_count,
            "load_model_reloads": self._model_load_reload_count,
            "load_model_cache_hits": self._model_load_cache_hit_count,
        }

    def _get_model_mtime_ns(self, model_path: Path):
        try:
            return model_path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _should_reload_model(self, model_file_path, model_mtime_ns):
        if self._last_loaded_model_path != model_file_path:
            return True
        if model_mtime_ns is None:
            return True
        return self._last_loaded_model_mtime_ns != model_mtime_ns

    def _run_model(self, feature):
        """Gradient-free forward pass, returns (logits_np, value_np).

        无梯度推理，返回 (logits_np, value_np)。
        """
        self.model.set_eval_mode()
        obs_tensor = (
            torch.tensor(np.array([feature], dtype=np.float32)).view(1, Config.DIM_OF_OBSERVATION).to(self.device)
        )
        with torch.no_grad():
            rst = self.model(obs_tensor, inference=True)
        logits = rst[0].cpu().numpy()[0]
        value = rst[1].cpu().numpy()[0]
        return logits, value

    def _blend_policy_logits(self, logits):
        biases = self.preprocessor.get_action_biases()
        if biases is None:
            return logits

        mode = getattr(self.preprocessor, "current_mode", Config.MODE_NUM)
        if mode == self.preprocessor.MODE_CHARGE:
            bias_scale = 1.15
        elif mode == self.preprocessor.MODE_EVADE:
            bias_scale = 1.35
        else:
            bias_scale = 0.75

        invalid_pressure = float(np.clip(getattr(self.preprocessor, "invalid_move_ema", 0.0), 0.0, 1.0))
        revisit_pressure = float(
            np.clip((getattr(self.preprocessor, "cur_visit_count", 1) - 1) / 6.0, 0.0, 1.0)
        )
        adaptive_scale = bias_scale + 0.4 * invalid_pressure + 0.2 * revisit_pressure
        return logits + adaptive_scale * biases

    def _legal_soft_max(self, logits, legal_action):
        """Softmax with legal action masking.

        合法动作掩码下的 softmax。
        """
        _w, _e = 1e20, 1e-5
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return tmp / (np.sum(tmp, keepdims=True) * 1.00001)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution (argmax if use_max=True).

        按概率分布采样动作（use_max=True 时取 argmax）。
        """
        if use_max:
            return int(np.argmax(probs))
        return int(np.argmax(np.random.multinomial(1, probs, size=1)))
