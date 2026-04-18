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
from agent_ppo.utils.policy_sampling import safe_sample_action, sanitize_policy_probs, uniform_over_legal

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
            if isinstance(list_sample_data, (list, tuple)) and list_sample_data:
                first = list_sample_data[0]
                if isinstance(first, (torch.Tensor, np.ndarray)):
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
        self.reward_components = {}
        self.rnn_state = None
        self._last_expert_weight = 0.0
        self._last_fallback_active = 0.0
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
        self._predict_fallback_count = 0
        self._predict_error_count = 0
        self._runtime_probe_stages = set()

        # Optional local bootstrap path. The default training entry now uses framework preload.
        use_local_resume_bootstrap = _env_flag(
            "KAIWU_USE_LOCAL_RESUME_BOOTSTRAP",
            Config.USE_LOCAL_RESUME_BOOTSTRAP,
        )
        if use_local_resume_bootstrap and Config.RESUME_CHECKPOINT:
            _resume_candidates = [
                os.path.join(os.path.dirname(__file__), "..", Config.RESUME_CHECKPOINT),
                os.path.join("/workspace/code", Config.RESUME_CHECKPOINT),
            ]
            for _resume_path in _resume_candidates:
                if os.path.isfile(_resume_path):
                    try:
                        state_dict = torch.load(_resume_path, map_location=self.device)
                        self._load_state_dict_compat(state_dict)
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
        self.reward_components = {}
        self.rnn_state = None
        self._last_expert_weight = 0.0
        self._last_fallback_active = 0.0

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData (enhanced feature vector + legal action mask).

        将原始 env_obs 转换为 ObsData（69D 特征 + 合法动作掩码）。
        """
        feature, legal_action, reward, reward_components = self.preprocessor.feature_process(env_obs, self.last_action)
        if isinstance(reward_components, dict):
            reward_payload = dict(reward_components)
            reward_payload.setdefault("reward_total", float(reward))
            reward_payload.setdefault("reward_clean", float(reward))
            reward_payload.setdefault("reward_survive", 0.0)
            reward_payload["fallback_mask"] = float(self._last_fallback_active)
            self.last_reward = reward_payload
            self.reward_components = reward_payload
        else:
            self.last_reward = float(reward)
            self.reward_components = {}

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
        expert = self.preprocessor.expert
        filtered_legal = expert.filter_actions(self.preprocessor, legal_action)
        legal_arr = np.array(filtered_legal, dtype=np.float32)

        try:
            outputs = self._run_model(feature)
            logits = outputs["policy_logits"]
            value_clean = float(outputs["value_clean"])
            value_survive = float(outputs["value_survive"])
            value_total = value_clean + value_survive
            mode_probs = self._sanitize_head_probs(outputs["mode_probs"])
            route_anchor_probs = self._sanitize_head_probs(outputs["route_anchor_probs"])
            target_probs = self._sanitize_head_probs(outputs["target_probs"])
        except Exception as exc:
            self._predict_error_count += 1
            self._predict_fallback_count += 1
            if self.logger:
                self.logger.warning(f"[PREDICT] inference fallback due to error: {exc}")
            return [self._build_safe_fallback_act_data(filtered_legal, 0.0, 0.0, 0.0)]

        self._last_expert_weight = 0.0
        self._last_fallback_active = 0.0

        clean_prob = self._legal_soft_max(logits, legal_arr)
        clean_prob, prob_fallback = sanitize_policy_probs(clean_prob, filtered_legal)
        if prob_fallback:
            self._predict_fallback_count += 1

        fallback = expert.get_emergency_fallback(
            self.preprocessor,
            filtered_legal,
            last_action=self.last_action,
        )
        if fallback.get("active") and fallback.get("action") is not None:
            self._last_fallback_active = 1.0

        if use_hard_override:
            if fallback.get("active") and fallback.get("action") is not None:
                expert_action = int(fallback["action"])
                return [
                    self._build_act_data(
                        action=expert_action,
                        d_action=expert_action,
                        prob=clean_prob,
                        value_total=value_total,
                        value_clean=value_clean,
                        value_survive=value_survive,
                        mode_probs=mode_probs,
                        route_anchor_probs=route_anchor_probs,
                        target_probs=target_probs,
                        return_action_logits=outputs["return_action_logits"],
                        aux_battery_risk=outputs["aux_battery_risk"],
                        aux_collision_risk=outputs["aux_collision_risk"],
                    )
                ]

        if self.preprocessor.stuck_steps >= 10 and not fallback.get("active"):
            legal_indices = [i for i, l in enumerate(filtered_legal) if l]
            if legal_indices:
                random_action = int(np.random.choice(legal_indices))
                prob = self._uniform_over_legal(filtered_legal)
                return [
                    self._build_act_data(
                        action=random_action,
                        d_action=random_action,
                        prob=prob,
                        value_total=value_total,
                        value_clean=value_clean,
                        value_survive=value_survive,
                        mode_probs=mode_probs,
                        route_anchor_probs=route_anchor_probs,
                        target_probs=target_probs,
                        return_action_logits=outputs["return_action_logits"],
                        aux_battery_risk=outputs["aux_battery_risk"],
                        aux_collision_risk=outputs["aux_collision_risk"],
                    )
                ]

        if fallback.get("active") and fallback.get("action") is not None:
            action = int(fallback["action"])
            d_action = action
        else:
            sampled = safe_sample_action(clean_prob, filtered_legal, use_max=False)
            greedy = safe_sample_action(clean_prob, filtered_legal, use_max=True)
            action = int(sampled["action"])
            d_action = int(greedy["action"])
            if sampled["used_fallback"] or greedy["used_fallback"]:
                self._predict_fallback_count += 1
                clean_prob = sampled["probs"]
        return [
            self._build_act_data(
                action=action,
                d_action=d_action,
                prob=clean_prob,
                value_total=value_total,
                value_clean=value_clean,
                value_survive=value_survive,
                mode_probs=mode_probs,
                route_anchor_probs=route_anchor_probs,
                target_probs=target_probs,
                return_action_logits=outputs["return_action_logits"],
                aux_battery_risk=outputs["aux_battery_risk"],
                aux_collision_risk=outputs["aux_collision_risk"],
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

    def _load_state_dict_compat(self, state_dict):
        model_state = self.model.state_dict()
        compatible = {}
        skipped = []
        for key, value in state_dict.items():
            if key not in model_state:
                skipped.append(key)
                continue
            if tuple(model_state[key].shape) != tuple(value.shape):
                skipped.append(key)
                continue
            compatible[key] = value
        missing, unexpected = self.model.load_state_dict(compatible, strict=False)
        if self.logger:
            self.logger.info(
                "[RESUME] compat_load matched=%d skipped=%d missing=%d unexpected=%d",
                len(compatible),
                len(skipped),
                len(missing),
                len(unexpected),
            )

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
            self._load_state_dict_compat(torch.load(model_file_path, map_location=self.device))
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
            "predict_fallback_count": self._predict_fallback_count,
            "predict_error_count": self._predict_error_count,
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
        """Gradient-free forward pass using the recurrent LTSPPO model."""
        self.model.set_eval_mode()
        obs_tensor = (
            torch.tensor(np.array([feature], dtype=np.float32)).view(1, Config.DIM_OF_OBSERVATION).to(self.device)
        )
        self._log_runtime_probe("predict", {"input": _describe_runtime_value(obs_tensor)})
        with torch.no_grad():
            rst = self.model(obs_tensor, rnn_state=self.rnn_state, inference=True)
        self.rnn_state = rst["next_rnn_state"]
        return {
            "policy_logits": rst["policy_logits"].cpu().numpy()[0],
            "mode_probs": rst["mode_probs"].cpu().numpy()[0],
            "route_anchor_probs": rst["route_anchor_probs"].cpu().numpy()[0],
            "target_probs": rst["target_probs"].cpu().numpy()[0],
            "return_action_logits": rst["return_action_logits"].cpu().numpy()[0],
            "value_clean": float(rst["value_clean"].cpu().numpy()[0][0]),
            "value_survive": float(rst["value_survive"].cpu().numpy()[0][0]),
            "aux_battery_risk": float(rst["aux_battery_risk"].cpu().numpy()[0][0]),
            "aux_collision_risk": float(rst["aux_collision_risk"].cpu().numpy()[0][0]),
        }

    def _uniform_over_legal(self, legal_action):
        """Uniform distribution over legal actions (for stable PPO ratio)."""
        return uniform_over_legal(legal_action)

    def _legal_soft_max(self, logits, legal_action):
        """Softmax with legal action masking.

        合法动作掩码下的 softmax。
        """
        logits = np.nan_to_num(np.asarray(logits, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        legal_action = np.where(np.asarray(legal_action, dtype=np.float32) > 0.5, 1.0, 0.0).astype(np.float32)
        if logits.size == 0:
            return self._uniform_over_legal(legal_action)
        if float(np.sum(legal_action)) <= 0.0:
            return self._uniform_over_legal(np.ones_like(logits, dtype=np.float32))
        _w, _e = 1e20, 1e-5
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        total = np.sum(tmp, keepdims=True)
        if float(total.squeeze()) <= 1e-8:
            return self._uniform_over_legal(legal_action)
        return (tmp / (total * 1.00001)).astype(np.float32).tolist()

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution (argmax if use_max=True).

        按概率分布采样动作（use_max=True 时取 argmax）。
        """
        sampled = safe_sample_action(probs, [1.0] * len(probs), use_max=use_max)
        if sampled["used_fallback"]:
            self._predict_fallback_count += 1
        return int(sampled["action"])

    @staticmethod
    def _sanitize_head_probs(probs):
        probs = np.nan_to_num(np.asarray(probs, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        if probs.size == 0:
            return np.array([1.0], dtype=np.float32)
        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs))
        if total <= 1e-8:
            probs = np.full(probs.shape, 1.0 / probs.size, dtype=np.float32)
        else:
            probs = probs / total
        return probs.astype(np.float32)

    def _build_act_data(
        self,
        action,
        d_action,
        prob,
        value_total,
        value_clean,
        value_survive,
        mode_probs,
        route_anchor_probs,
        target_probs,
        return_action_logits,
        aux_battery_risk,
        aux_collision_risk,
    ):
        mode = int(np.argmax(mode_probs))
        target = int(np.argmax(target_probs))
        route_anchor = int(np.argmax(route_anchor_probs))
        return ActData(
            action=[int(action)],
            d_action=[int(d_action)],
            prob=list(prob),
            value=np.array([value_total], dtype=np.float32),
            value_clean=np.array([value_clean], dtype=np.float32),
            value_survive=np.array([value_survive], dtype=np.float32),
            mode=np.array([mode], dtype=np.int64),
            mode_prob=np.array(mode_probs, dtype=np.float32),
            route_anchor=np.array([route_anchor], dtype=np.int64),
            route_anchor_prob=np.array(route_anchor_probs, dtype=np.float32),
            target=np.array([target], dtype=np.int64),
            target_prob=np.array(target_probs, dtype=np.float32),
            return_action_prob=np.nan_to_num(
                np.asarray(return_action_logits, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ),
            aux_battery_risk=np.array([aux_battery_risk], dtype=np.float32),
            aux_collision_risk=np.array([aux_collision_risk], dtype=np.float32),
        )

    def _build_safe_fallback_act_data(self, legal_action, value_total, value_clean, value_survive):
        safe = safe_sample_action([0.0] * len(legal_action), legal_action, use_max=True)
        neutral_mode = np.full((Config.MODE_NUM,), 1.0 / Config.MODE_NUM, dtype=np.float32)
        neutral_target = np.full((Config.TARGET_DIM,), 1.0 / Config.TARGET_DIM, dtype=np.float32)
        neutral_anchor = np.array([0.5, 0.5], dtype=np.float32)
        return self._build_act_data(
            action=safe["action"],
            d_action=safe["action"],
            prob=safe["probs"],
            value_total=value_total,
            value_clean=value_clean,
            value_survive=value_survive,
            mode_probs=neutral_mode,
            route_anchor_probs=neutral_anchor,
            target_probs=neutral_target,
            return_action_logits=np.zeros((Config.ACTION_NUM,), dtype=np.float32),
            aux_battery_risk=0.0,
            aux_collision_risk=0.0,
        )
