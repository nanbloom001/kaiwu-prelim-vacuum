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
import json
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


_RUNTIME_PROBE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "KAIWU_EXPERIMENT_REPLAY_BUFFER_TYPE",
    "KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE",
    "KAIWU_EXPERIMENT_PREDICT_BATCH_SIZE",
    "KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE",
    "KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE",
    "KAIWU_SERVICE_NAME",
)

_RUNTIME_PROBE_CONFIG_KEYS = (
    "svr_name",
    "train_batch_size",
    "predict_batch_size",
    "proxy_batch_size",
    "send_sample_size",
    "replay_buffer_type",
    "reverb_sampler",
    "reverb_rate_limiter",
    "pytorch_read_data_from_reverb_type",
)


def _device_to_string(device):
    if device is None:
        return None
    return str(device)


def _get_model_param_device(model):
    if model is None or not hasattr(model, "parameters"):
        return None
    try:
        return str(next(model.parameters()).device)
    except (StopIteration, TypeError, AttributeError):
        return None


def _describe_runtime_value(value):
    if value is None:
        return {"type": "NoneType"}
    if torch.is_tensor(value):
        return {
            "type": "torch.Tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
        }
    if isinstance(value, np.ndarray):
        return {
            "type": "numpy.ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return {"type": type(value).__name__}


def _build_runtime_probe_payload(stage, service_name, requested_device, model, algorithm, use_amp, extra=None):
    payload = {
        "stage": stage,
        "service_name": service_name,
        "requested_device": _device_to_string(requested_device),
        "model_param_device": _get_model_param_device(model),
        "algorithm_device": _device_to_string(getattr(algorithm, "device", None)),
        "use_amp": use_amp,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "env": {key: os.getenv(key) for key in _RUNTIME_PROBE_ENV_KEYS},
        "config": {key: getattr(KAIWU_CONFIG, key, None) for key in _RUNTIME_PROBE_CONFIG_KEYS},
    }
    if torch.cuda.is_available():
        try:
            payload["torch_cuda_current_device"] = torch.cuda.current_device()
        except RuntimeError:
            payload["torch_cuda_current_device"] = None
    if extra:
        payload.update(extra)
    return payload


def _emit_runtime_probe_once(logger, seen_stages, stage, payload):
    if stage in seen_stages:
        return False
    seen_stages.add(stage)
    message = f"runtime_probe {json.dumps(payload, sort_keys=True, default=str)}"
    if logger is not None and hasattr(logger, "info"):
        logger.info(message)
    else:
        print(message, flush=True)
    return True


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
        if "learner" in self.service_name:
            compiled = False
            if _env_flag("KAIWU_LEARNER_TORCH_COMPILE", getattr(Config, "LEARNER_TORCH_COMPILE", True)):
                try:
                    self.model = torch.compile(self.model, mode="reduce-overhead")
                    compiled = True
                except RuntimeError:
                    pass
            if not compiled and _env_flag(
                "KAIWU_LEARNER_JIT_TRACE", getattr(Config, "LEARNER_JIT_TRACE", True)
            ):
                try:
                    _dummy = torch.randn(1, Config.DIM_OF_OBSERVATION, device=self.device)
                    self.model = torch.jit.trace(self.model, _dummy)
                    self.model.set_train_mode = lambda: self.model.train()
                    self.model.set_eval_mode = lambda: self.model.eval()
                except Exception:
                    pass
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
        use_fused = (
            self.use_amp
            and _env_flag("KAIWU_LEARNER_USE_FUSED_OPTIMIZER", Config.LEARNER_ALLOW_FUSED_OPTIMIZER)
        )
        use_foreach = (
            not use_fused
            and _env_flag("KAIWU_LEARNER_USE_FOREACH_OPTIMIZER", Config.LEARNER_ALLOW_FOREACH_OPTIMIZER)
        )
        if use_fused:
            optimizer_kwargs["fused"] = True
        elif use_foreach:
            optimizer_kwargs["foreach"] = True
        try:
            self.optimizer = torch.optim.Adam(**optimizer_kwargs)
        except (TypeError, RuntimeError):
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
        self._runtime_probe_stages = set()

        # Resume from checkpoint if configured in conf.py
        if Config.RESUME_CHECKPOINT:
            _resume_candidates = [
                os.path.join(os.path.dirname(__file__), "..", Config.RESUME_CHECKPOINT),
                os.path.join("/workspace/code", Config.RESUME_CHECKPOINT),
            ]
            for _resume_path in _resume_candidates:
                if os.path.isfile(_resume_path):
                    try:
                        state_dict = torch.load(_resume_path, map_location=self.device)
                        # Migrate checkpoint: expand conv1 from 3ch to 4ch (trajectory heatmap)
                        if 'local_encoder.0.weight' in state_dict:
                            w = state_dict['local_encoder.0.weight']
                            if w.shape[1] == 3 and Config.LOCAL_VIEW_CHANNELS == 4:
                                new_w = torch.zeros(w.shape[0], 4, w.shape[2], w.shape[3])
                                new_w[:, :3, :, :] = w
                                new_w[:, 3, :, :] = w.mean(dim=1)
                                state_dict['local_encoder.0.weight'] = new_w
                        self.model.load_state_dict(state_dict)
                        import sys
                        print(f"[RESUME] Loaded from {_resume_path}", file=sys.stderr, flush=True)
                        self.logger and self.logger.info(f"[RESUME] Loaded from {_resume_path}")
                        break
                    except Exception as e:
                        import sys
                        print(f"[RESUME] Failed: {e}", file=sys.stderr, flush=True)

        super().__init__(agent_type, device, logger, monitor)
        self._log_runtime_probe("init")

    def _log_runtime_probe(self, stage, extra=None):
        if stage in self._runtime_probe_stages:
            return False
        payload = _build_runtime_probe_payload(
            stage=stage,
            service_name=self.service_name,
            requested_device=self.device,
            model=self.model,
            algorithm=self.algorithm,
            use_amp=self.use_amp,
            extra=extra,
        )
        return _emit_runtime_probe_once(self.logger, self._runtime_probe_stages, stage, payload)

    def reset(self, env_obs):
        """Reset per-episode state.

        每局开始时重置 Agent 内部状态。
        """
        self.preprocessor = Preprocessor()
        self.preprocessor.expert.reset()
        self.last_action = -1
        self.last_reward = 0.0

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData (enhanced feature vector + legal action mask).

        将原始 env_obs 转换为 ObsData（69D 特征 + 合法动作掩码）。
        """
        feature, legal_action, reward, reward_components = self.preprocessor.feature_process(env_obs, self.last_action)
        self.last_reward = reward
        self.reward_components = reward_components

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

    def predict(self, list_obs_data, use_hard_override=False):
        """Stochastic inference for training (exploration).

        Training mode: Expert Logit Bias — soft guidance with correct PPO ratio.
        Evaluation mode (use_hard_override=True): Expert hard override for max survival.
        """
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        logits, value = self._run_model(feature)
        expert = self.preprocessor.expert
        self._last_expert_weight = 0.0  # default: no expert bias

        # Layer 1: NPC safety filter — block moves toward nearby NPCs (first!)
        filtered_legal = expert.filter_actions(self.preprocessor, legal_action)

        if use_hard_override:
            # Evaluation mode: hard expert override for max survival
            should_override, expert_action = expert.get_override(
                self.preprocessor, filtered_legal, last_action=self.last_action
            )
            if should_override:
                legal_arr = np.array(filtered_legal, dtype=np.float32)
                prob = self._legal_soft_max(logits, legal_arr)
                return [
                    ActData(
                        action=[expert_action],
                        d_action=[expert_action],
                        prob=list(prob),
                        value=value,
                    )
                ]
        else:
            # Training mode: soft expert logit bias with clean prob storage
            expert_bias = expert.get_logit_bias(
                self.preprocessor, filtered_legal, last_action=self.last_action
            )

            # Expert annealing: gradually reduce bias as training progresses
            if Config.EXPERT_ANNEAL_START_EPISODE > 0:
                episode = getattr(self, '_predict_episode_idx', 0)
                if episode >= Config.EXPERT_ANNEAL_START_EPISODE:
                    progress = min(
                        (episode - Config.EXPERT_ANNEAL_START_EPISODE)
                        / max(Config.EXPERT_ANNEAL_END_EPISODE - Config.EXPERT_ANNEAL_START_EPISODE, 1),
                        1.0,
                    )
                    scale = 1.0 - progress * (1.0 - Config.EXPERT_ANNEAL_MIN_SCALE)
                    expert_bias = expert_bias * scale

            self._last_expert_weight = float(max(0.0, np.max(expert_bias)))

        # Layer 2: Anti-stuck — random legal action if stuck too long
        # Skip anti-stuck during expert return_mode (Expert handles stuck via blocked cells + A*)
        if self.preprocessor.stuck_steps >= 10 and not expert.return_mode:
            legal_indices = [i for i, l in enumerate(filtered_legal) if l]
            if legal_indices:
                random_action = int(np.random.choice(legal_indices))
                prob = self._uniform_over_legal(filtered_legal)
                return [
                    ActData(
                        action=[random_action],
                        d_action=[random_action],
                        prob=prob,
                        value=value,
                    )
                ]

        # RL decision
        legal_arr = np.array(filtered_legal, dtype=np.float32)
        clean_prob = self._legal_soft_max(logits, legal_arr)

        if not use_hard_override and np.any(expert_bias > 0):
            # Training with bias: sample from biased distribution
            biased_logits = logits + np.array(expert_bias, dtype=np.float32)
            biased_prob = self._legal_soft_max(biased_logits, legal_arr)
            action = self._legal_sample(biased_prob, use_max=False)
            d_action = self._legal_sample(biased_prob, use_max=True)
        else:
            action = self._legal_sample(clean_prob, use_max=False)
            d_action = self._legal_sample(clean_prob, use_max=True)

        return [
            ActData(
                action=[action],
                d_action=[d_action],
                prob=list(clean_prob),
                value=value,
            )
        ]

    def exploit(self, env_obs):
        """Greedy inference for evaluation.

        评估时推理（贪心）。使用 Expert 硬覆盖保证最大存活率。
        """
        obs_data, _ = self.observation_process(env_obs)
        act_data = self.predict([obs_data], use_hard_override=True)[0]
        return self.action_process(act_data, is_stochastic=False)

    def learn(self, list_sample_data):
        """Delegate to Algorithm for PPO update.

        委托给 Algorithm 执行训练。
        """
        result = self.algorithm.learn(list_sample_data)
        self._log_runtime_probe(
            "learn",
            {
                "input": _describe_runtime_value(list_sample_data),
                "output": _describe_runtime_value(result),
            },
        )
        return result

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
        if path is None:
            from common_python.config.config_control import CONFIG as _CFG
            path = getattr(_CFG, "restore_dir", None)
            if path:
                path = f"{path}/{_CFG.app}_{_CFG.algo}"
            else:
                path = "/workspace/code/ckpt"
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
        self._log_runtime_probe("predict", {"input": _describe_runtime_value(obs_tensor)})
        with torch.no_grad():
            rst = self.model(obs_tensor, inference=True)
        logits = rst[0].cpu().numpy()[0]
        value = rst[1].cpu().numpy()[0]
        return logits, value

    def _uniform_over_legal(self, legal_action):
        """Uniform distribution over legal actions (for stable PPO ratio)."""
        n = max(sum(legal_action), 1)
        return [1.0 / n if x else 0.0 for x in legal_action]

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
