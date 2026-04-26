#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Robot Vacuum Agent.
清扫大作战 Agent 主类。

Ownership boundary:
- `train/.docker-compose.yaml` owns Linux framework hot-patches, startup patch injection, and env/TOML bridging.
- `code/agent_ppo/agent.py` owns only agent-local runtime behavior plus the in-process `RemoteAgent.learn` hook
  that depends on this module's business learn contract.
"""

import os
import json
import time
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


def _fallback_allowed_for_action(fallback, use_hard_override=False):
    """Gate eval-only safety fallbacks before they replace policy actions."""
    if not fallback.get("active") or fallback.get("action") is None:
        return False
    reason = str(fallback.get("reason") or "")
    if reason == "unsafe_slack_return" and not use_hard_override:
        return False
    return True


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
    """Configure process-local torch runtime knobs.

    Compose owns environment delivery; the agent owns how those env values are consumed inside the Python process.
    """
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
    """Keep the batch-tensor learner fast path owned by the agent runtime.

    This hook must stay in `agent.py` because it depends on `Agent.PREFER_BATCH_TENSOR_LEARN` and the current
    business `learn()` contract. Compose startup patches own framework file rewrites, but should not duplicate or
    relocate this in-process dispatch decision.
    """
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
        self._last_pre_return_bias_active = 0.0
        self._last_return_bias_active = 0.0
        self._last_fallback_active = 0.0
        self.reset_eval_override_summary()
        self.current_model_ref = {
            "path": None,
            "id": None,
            "checkpoint_id": None,
            "checkpoint_step": None,
            "global_step_since_resume": None,
        }
        self.enable_load_model_cache = _env_flag("KAIWU_AGENT_LOAD_MODEL_CACHE", Config.AGENT_LOAD_MODEL_CACHE)
        self._last_loaded_model_path = None
        self._last_loaded_model_mtime_ns = None
        self._model_load_call_count = 0
        self._model_load_reload_count = 0
        self._model_load_cache_hit_count = 0
        self._last_loaded_checkpoint_step = None
        self._last_real_model_reload_ts = 0.0
        self._load_model_transition_guard = False
        self._load_model_stage_transition_cooldown = False
        self._predict_fallback_count = 0
        self._predict_error_count = 0
        self._runtime_probe_stages = set()

        # Optional local bootstrap path. The default Linux training flow keeps resume/preload ownership in the
        # framework + compose startup path; this branch is only an explicit agent-local fallback.
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
        self._last_pre_return_bias_active = 0.0
        self._last_return_bias_active = 0.0
        self._last_fallback_active = 0.0
        self.reset_eval_override_summary()

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData (enhanced feature vector + legal action mask).

        将原始 env_obs 转换为 ObsData（69D 特征 + 合法动作掩码）。
        """
        runtime_payload = dict((env_obs or {}).get("runtime") or {})
        current_progress = self.current_model_ref.get("global_step_since_resume")
        if current_progress is None:
            current_progress = self.current_model_ref.get("checkpoint_step")
        try:
            runtime_payload.setdefault("global_step_since_resume", max(int(current_progress or 0), 0))
        except (TypeError, ValueError):
            runtime_payload.setdefault("global_step_since_resume", 0)
        obs_payload = dict(env_obs or {})
        obs_payload["runtime"] = runtime_payload
        feature, legal_action, reward, reward_components = self.preprocessor.feature_process(obs_payload, self.last_action)
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

        Training mode: planner-guided residual action mixing with correct PPO ratio.
        Evaluation mode (use_hard_override=True): preserve eval hard overrides for max survival.
        """
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action
        expert = self.preprocessor.expert
        filtered_legal = expert.filter_actions(self.preprocessor, legal_action)
        action_mask = self._build_action_mask(legal_action, filtered_legal)
        obs_data.legal_action = list(action_mask)
        legal_arr = np.array(action_mask, dtype=np.float32)
        guidance = expert.get_teacher_guidance(self.preprocessor, filtered_legal, last_action=self.last_action)
        signal = (guidance or {}).get("signal")
        if signal is None:
            signal = expert.get_charger_signal(self.preprocessor, filtered_legal, last_action=self.last_action)

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
        self._last_pre_return_bias_active = 0.0
        self._last_return_bias_active = 0.0
        self._last_fallback_active = 0.0
        policy_prob = np.asarray(self._legal_soft_max(logits, legal_arr), dtype=np.float32)
        planner_prob = self._planner_prior(guidance, signal, legal_arr)

        fallback = expert.get_emergency_fallback(
            self.preprocessor,
            filtered_legal,
            last_action=self.last_action,
        )
        fallback_action_allowed = _fallback_allowed_for_action(fallback, use_hard_override=use_hard_override)
        if fallback_action_allowed:
            self._last_fallback_active = 1.0
            self._last_eval_override_reason = str(fallback.get("reason") or "fallback")
            self._last_expert_weight = 0.0
            self._last_pre_return_bias_active = 0.0
            self._last_return_bias_active = 0.0
        else:
            self._last_eval_override_reason = None
        mix_alpha = self._effective_residual_alpha(guidance, signal, fallback_action_allowed)
        mix_prob = self._mix_prob(policy_prob, planner_prob, mix_alpha, legal_arr)
        mix_prob, prob_fallback = sanitize_policy_probs(mix_prob, filtered_legal)
        if prob_fallback:
            self._predict_fallback_count += 1

        if use_hard_override:
            self._eval_decision_count += 1
            if fallback_action_allowed:
                expert_action = int(fallback.get("action", 0))
                reason = str(fallback.get("reason") or "fallback")
                self._eval_override_count += 1
                self._eval_override_reason_counts[reason] = self._eval_override_reason_counts.get(reason, 0) + 1
                return [
                    self._build_act_data(
                        action=expert_action,
                        d_action=expert_action,
                        prob=mix_prob,
                        policy_prob=policy_prob,
                        planner_prob=planner_prob,
                        mix_alpha=mix_alpha,
                        action_mask=legal_arr,
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

        if self.preprocessor.stuck_steps >= 10 and not fallback_action_allowed:
            legal_indices = [i for i, l in enumerate(filtered_legal) if l]
            if legal_indices:
                random_action = int(np.random.choice(legal_indices))
                prob = self._uniform_over_legal(filtered_legal)
                return [
                    self._build_act_data(
                        action=random_action,
                        d_action=random_action,
                        prob=prob,
                        policy_prob=policy_prob,
                        planner_prob=planner_prob,
                        mix_alpha=mix_alpha,
                        action_mask=legal_arr,
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

        if fallback_action_allowed:
            action = int(fallback.get("action", 0))
            d_action = action
        else:
            sampled = safe_sample_action(mix_prob, filtered_legal, use_max=False)
            greedy = safe_sample_action(mix_prob, filtered_legal, use_max=True)
            action = int(sampled["action"])
            d_action = int(greedy["action"])
            if sampled["used_fallback"] or greedy["used_fallback"]:
                self._predict_fallback_count += 1
                mix_prob = sampled["probs"]
        return [
            self._build_act_data(
                action=action,
                d_action=d_action,
                prob=mix_prob,
                policy_prob=policy_prob,
                planner_prob=planner_prob,
                mix_alpha=mix_alpha,
                action_mask=legal_arr,
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
        checkpoint_step = self._parse_checkpoint_step(checkpoint_id)
        self.current_model_ref = {
            "path": model_file_path,
            "id": str(id),
            "checkpoint_id": checkpoint_id,
            "checkpoint_step": checkpoint_step,
            "global_step_since_resume": checkpoint_step,
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

    def set_load_model_context(self, transition_guard=False, stage_transition_cooldown=False):
        self._load_model_transition_guard = bool(transition_guard)
        self._load_model_stage_transition_cooldown = bool(stage_transition_cooldown)

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
        checkpoint_id = parse_checkpoint_id(model_file_path) or str(id)
        should_reload = True
        now_ts = time.time()
        if self.enable_load_model_cache:
            should_reload = self._should_reload_model(
                model_file_path,
                model_mtime_ns,
                checkpoint_id=checkpoint_id,
                now_ts=now_ts,
            )

        if should_reload:
            self._load_state_dict_compat(torch.load(model_file_path, map_location=self.device))
            self._last_loaded_model_path = model_file_path
            self._last_loaded_model_mtime_ns = model_mtime_ns
            self._last_loaded_checkpoint_step = self._parse_checkpoint_step(checkpoint_id)
            self._last_real_model_reload_ts = now_ts
            self._model_load_reload_count += 1
            self.current_model_ref = {
                "path": model_file_path,
                "id": str(id),
                "checkpoint_id": checkpoint_id,
                "checkpoint_step": self._last_loaded_checkpoint_step,
                "global_step_since_resume": self._last_loaded_checkpoint_step,
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
        else:
            self._model_load_cache_hit_count += 1
            if self.logger:
                self.logger.info(
                    "skip reload model %s checkpoint_id=%s cache_hit=1",
                    model_file_path,
                    checkpoint_id,
                )

    def _business_load_model(self, path=None, id="1"):
        return self.load_model(path=path, id=id)

    def get_runtime_metrics(self):
        return {
            "load_model_calls": self._model_load_call_count,
            "load_model_reloads": self._model_load_reload_count,
            "load_model_cache_hits": self._model_load_cache_hit_count,
            "last_loaded_checkpoint_step": self._last_loaded_checkpoint_step,
            "predict_fallback_count": self._predict_fallback_count,
            "predict_error_count": self._predict_error_count,
            "last_expert_weight": self._last_expert_weight,
            "last_pre_return_bias_active": self._last_pre_return_bias_active,
            "last_return_bias_active": self._last_return_bias_active,
            "eval_override_count": self._eval_override_count,
            "eval_override_rate": self._eval_override_count / max(self._eval_decision_count, 1),
            "eval_override_reason_counts": dict(self._eval_override_reason_counts),
        }

    def get_eval_override_summary(self):
        return {
            "eval_decision_count": int(self._eval_decision_count),
            "eval_override_count": int(self._eval_override_count),
            "eval_override_rate": round(self._eval_override_count / max(self._eval_decision_count, 1), 6),
            "eval_override_reason_counts": dict(self._eval_override_reason_counts),
            "last_eval_override_reason": self._last_eval_override_reason,
        }

    def reset_eval_override_summary(self):
        self._last_eval_override_reason = None
        self._eval_decision_count = 0
        self._eval_override_count = 0
        self._eval_override_reason_counts = {}

    def _get_model_mtime_ns(self, model_path: Path):
        try:
            return model_path.stat().st_mtime_ns
        except FileNotFoundError:
            return None

    def _parse_checkpoint_step(self, checkpoint_id):
        if checkpoint_id is None:
            return None
        text = str(checkpoint_id).strip()
        if not text.isdigit():
            return None
        try:
            return int(text)
        except ValueError:
            return None

    def _should_reload_model(self, model_file_path, model_mtime_ns, checkpoint_id=None, now_ts=None):
        now_ts = float(now_ts or time.time())
        checkpoint_step = self._parse_checkpoint_step(checkpoint_id)
        if checkpoint_step is not None and self._last_loaded_checkpoint_step is not None:
            step_gap = checkpoint_step - int(self._last_loaded_checkpoint_step)
            min_reload_gap = Config.AGENT_MIN_RELOAD_STEP_GAP
            if self._load_model_transition_guard or self._load_model_stage_transition_cooldown:
                min_reload_gap = max(min_reload_gap, int(Config.AGENT_TRANSITION_GUARD_RELOAD_STEP_GAP))
            reload_interval = max(int(Config.AGENT_MIN_RELOAD_INTERVAL_SECONDS), 0)
            if step_gap <= 0:
                return False
            if step_gap <= 500:
                return False
            if step_gap < min_reload_gap and (now_ts - float(self._last_real_model_reload_ts)) < reload_interval:
                return False
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

    @staticmethod
    def _default_residual_alpha():
        return float(
            np.clip(
                min(max(Config.RESIDUAL_ALPHA_START, Config.RESIDUAL_ALPHA_WARMUP_TARGET), Config.RESIDUAL_ALPHA_MAX),
                0.0,
                1.0,
            )
        )

    def _build_action_mask(self, legal_action, safe_action=None):
        base = np.where(np.asarray(legal_action, dtype=np.float32) > 0.5, 1.0, 0.0).astype(np.float32)
        if safe_action is not None:
            safe = np.where(np.asarray(safe_action, dtype=np.float32) > 0.5, 1.0, 0.0).astype(np.float32)
            if safe.shape == base.shape and float(np.sum(safe)) > 0.5:
                base = safe
        if float(np.sum(base)) <= 0.5:
            base = np.ones((Config.ACTION_NUM,), dtype=np.float32)
        return base.astype(np.float32)

    def _normalize_prob(self, prob, action_mask):
        mask = self._build_action_mask(action_mask)
        arr = np.nan_to_num(np.asarray(prob, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
        if arr.size != Config.ACTION_NUM:
            return np.asarray(self._uniform_over_legal(mask), dtype=np.float32)
        arr = np.clip(arr, 0.0, None) * mask
        total = float(np.sum(arr))
        if total <= 1e-8:
            return np.asarray(self._uniform_over_legal(mask), dtype=np.float32)
        return (arr / total).astype(np.float32)

    def _planner_prior(self, guidance, signal, action_mask):
        planner_action = -1
        planner_weight = 0.0
        payload = guidance or {}

        return_mask = float(payload.get("return_action_teacher_mask", 0.0))
        if return_mask > 0.0:
            planner_action = int(payload.get("return_action", -1))
            planner_weight = max(planner_weight, 1.0 + return_mask)

        action_mask_weight = max(
            float(payload.get("mode_teacher_mask", 0.0)),
            float(payload.get("target_teacher_mask", 0.0)),
            float(payload.get("route_anchor_teacher_mask", 0.0)),
        )
        if planner_action < 0:
            planner_action = int(payload.get("action", -1) if payload.get("action") is not None else -1)
            if planner_action >= 0:
                planner_weight = max(planner_weight, 0.75 + action_mask_weight)

        if planner_action < 0 and signal is not None:
            planner_action = int(signal.get("suggested_action", -1) if signal.get("suggested_action") is not None else -1)
            if planner_action >= 0:
                planner_weight = max(
                    planner_weight,
                    1.0 if bool(signal.get("return_action_reliable", False)) else 0.75,
                )

        mask = self._build_action_mask(action_mask)
        if planner_action < 0 or planner_action >= Config.ACTION_NUM or mask[planner_action] <= 0.5:
            return np.asarray(self._uniform_over_legal(mask), dtype=np.float32)

        scores = np.zeros((Config.ACTION_NUM,), dtype=np.float32)
        scores[planner_action] = planner_weight / max(float(Config.PLANNER_PRIOR_TEMPERATURE), 1e-6)
        scores = np.exp(scores - np.max(scores)) * mask
        return self._normalize_prob(scores, mask)

    def _effective_residual_alpha(self, guidance, signal, fallback_action_allowed):
        alpha = self._default_residual_alpha()
        if fallback_action_allowed:
            return float(min(alpha, Config.RESIDUAL_ALPHA_FALLBACK_CAP))
        payload = guidance or {}
        route_mode = str(payload.get("route_mode") or payload.get("mode") or "").strip().lower()
        if route_mode in {"contract", "return"}:
            return float(min(alpha, Config.RESIDUAL_ALPHA_CHARGE_CAP))
        if signal is not None:
            if bool(signal.get("return_action_reliable", False)):
                return float(min(alpha, Config.RESIDUAL_ALPHA_CHARGE_CAP))
            if not bool(signal.get("reachable", True)) or signal.get("suggested_action") is None:
                return float(min(alpha, Config.RESIDUAL_ALPHA_FALLBACK_CAP))
        return float(np.clip(alpha, 0.0, 1.0))

    def _mix_prob(self, policy_prob, planner_prob, mix_alpha, action_mask):
        alpha = float(np.clip(mix_alpha, 0.0, 1.0))
        mixed = (1.0 - alpha) * np.asarray(planner_prob, dtype=np.float32) + alpha * np.asarray(
            policy_prob,
            dtype=np.float32,
        )
        return self._normalize_prob(mixed, action_mask)

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
        policy_prob,
        planner_prob,
        mix_alpha,
        action_mask,
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
            policy_prob=np.asarray(policy_prob, dtype=np.float32),
            planner_prob=np.asarray(planner_prob, dtype=np.float32),
            mix_alpha=np.array([float(mix_alpha)], dtype=np.float32),
            action_mask=np.asarray(action_mask, dtype=np.float32),
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
        neutral_anchor = np.full((Config.ROUTE_ANCHOR_DIM,), 1.0 / Config.ROUTE_ANCHOR_DIM, dtype=np.float32)
        return self._build_act_data(
            action=safe["action"],
            d_action=safe["action"],
            prob=safe["probs"],
            policy_prob=np.asarray(safe["probs"], dtype=np.float32),
            planner_prob=np.asarray(safe["probs"], dtype=np.float32),
            mix_alpha=0.0,
            action_mask=np.asarray(legal_action, dtype=np.float32),
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
