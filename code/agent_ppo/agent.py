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

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import numpy as np

from agent_diy.agent import Agent as DIYRuleAgent
from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from kaiwudrl.interface.agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, agent_type="player", device=None, logger=None, monitor=None):
        torch.manual_seed(0)
        self.device = device
        self.model = Model(device).to(self.device)
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
        self.rule_agent = self._new_rule_agent()
        self.latest_rule_context = self._default_rule_context()
        self.last_env_charge_count = 0
        self.last_env_clean_score = 0
        self.last_action = -1
        self.last_reward = 0.0

        super().__init__(agent_type, device, logger, monitor)

    def reset(self, env_obs):
        """Reset per-episode state.

        每局开始时重置 Agent 内部状态。
        """
        self.preprocessor = Preprocessor()
        self.rule_agent = self._new_rule_agent()
        self.latest_rule_context = self._default_rule_context()
        self.last_env_charge_count = 0
        self.last_env_clean_score = 0
        self.last_action = -1
        self.last_reward = 0.0

    def _new_rule_agent(self):
        return DIYRuleAgent(agent_type="player", device=self.device, logger=self.logger, monitor=self.monitor)

    def _default_rule_context(self):
        return {
            "mode": "SAFE_EXPLORATION",
            "recommended_action": 0,
            "mode_actions": list(range(Config.ACTION_NUM)),
            "safe_energy": 0.0,
            "need_return": False,
            "early_charge": False,
            "emergency": False,
            "charge_count": 0,
            "target_charge_count": 0,
            "charge_deficit": 0,
            "charge_guard_active": False,
            "steps_since_last_charge": 0,
            "known_chargers": 0,
            "distance_to_charger": 128.0,
            "in_charger_safe_zone": False,
            "on_charger": False,
            "battery_ratio": 1.0,
            "danger_here": 0.0,
        }

    def _mode_one_hot(self, mode):
        mode_order = [
            "SAFE_EXPLORATION",
            "STRIPE_CLEANING",
            "DIRECT_DIRT_PICKUP",
            "CHARGER_TRANSFER",
            "RETURN_TO_CHARGER",
            "EARLY_CHARGE",
            "EMERGENCY_EVADE",
            "STUCK_RECOVERY",
        ]
        vec = np.zeros(len(mode_order), dtype=np.float32)
        if mode in mode_order:
            vec[mode_order.index(mode)] = 1.0
        return vec

    def _action_one_hot(self, action):
        vec = np.zeros(Config.ACTION_NUM, dtype=np.float32)
        if 0 <= int(action) < Config.ACTION_NUM:
            vec[int(action)] = 1.0
        return vec

    def _safe_get_charge_count(self, env_obs):
        try:
            return int(env_obs["observation"]["env_info"].get("charge_count", 0))
        except Exception:
            return 0

    def _safe_get_clean_score(self, env_obs):
        try:
            return int(env_obs["observation"]["env_info"].get("clean_score", 0))
        except Exception:
            return 0

    def _build_rule_context(self, env_obs, rule_action):
        ctx = self.rule_agent.get_hybrid_context(
            robot_pos=self.preprocessor.cur_pos,
            battery=self.preprocessor.battery,
        )
        ctx = dict(ctx)
        ctx["recommended_action"] = int(rule_action)
        return ctx

    def _apply_rule_mask(self, env_legal_action, ctx):
        env_mask = np.array(env_legal_action, dtype=np.float32)
        if env_mask.shape[0] != Config.ACTION_NUM:
            env_mask = np.ones(Config.ACTION_NUM, dtype=np.float32)

        mode_actions = [int(a) for a in ctx.get("mode_actions", []) if 0 <= int(a) < Config.ACTION_NUM]
        if mode_actions:
            rule_mask = np.zeros(Config.ACTION_NUM, dtype=np.float32)
            rule_mask[mode_actions] = 1.0
            combined = env_mask * rule_mask
            if combined.sum() > 0:
                env_mask = combined

        mode = ctx.get("mode", "SAFE_EXPLORATION")
        rule_action = int(ctx.get("recommended_action", 0))
        if mode in Config.RULE_FORCE_MODES and 0 <= rule_action < Config.ACTION_NUM and env_mask[rule_action] > 0:
            forced = np.zeros(Config.ACTION_NUM, dtype=np.float32)
            forced[rule_action] = 1.0
            env_mask = forced

        if env_mask.sum() <= 0:
            env_mask = np.array(env_legal_action, dtype=np.float32)
            if env_mask.sum() <= 0:
                env_mask = np.ones(Config.ACTION_NUM, dtype=np.float32)
        return env_mask.tolist()

    def _augment_feature(self, base_feature, legal_action, ctx):
        base_arr = np.array(base_feature, dtype=np.float32)
        local_view = base_arr[: 7 * 7]
        global_state = base_arr[7 * 7 : 7 * 7 + 12]
        battery_ratio = float(ctx.get("battery_ratio", 1.0))
        safe_energy = float(ctx.get("safe_energy", 0.0))
        battery_max = max(1.0, float(self.preprocessor.battery_max))
        safe_energy_ratio = safe_energy / battery_max
        battery_buffer_ratio = np.clip((float(self.preprocessor.battery) - safe_energy) / battery_max, -1.0, 1.0)
        charge_count_ratio = min(1.0, float(ctx.get("charge_count", 0)) / max(1.0, float(Config.ACTION_NUM + 16)))
        target_charge_ratio = min(1.0, float(ctx.get("target_charge_count", 0)) / max(1.0, float(Config.ACTION_NUM + 16)))
        charge_deficit_ratio = min(1.0, float(ctx.get("charge_deficit", 0)) / 12.0)
        steps_since_last_charge_norm = min(1.0, float(ctx.get("steps_since_last_charge", 0)) / 80.0)
        dist_to_charger_norm = min(1.0, float(ctx.get("distance_to_charger", 128.0)) / 128.0)
        known_chargers_norm = min(1.0, float(ctx.get("known_chargers", 0)) / 4.0)

        rule_scalars = np.array(
            [
                safe_energy_ratio,
                battery_buffer_ratio,
                charge_count_ratio,
                target_charge_ratio,
                charge_deficit_ratio,
                steps_since_last_charge_norm,
                dist_to_charger_norm,
                known_chargers_norm,
            ],
            dtype=np.float32,
        )
        rule_state = np.concatenate(
            [
                global_state,
                self._mode_one_hot(ctx.get("mode", "SAFE_EXPLORATION")),
                self._action_one_hot(ctx.get("recommended_action", 0)),
                rule_scalars,
            ]
        )
        legal_arr = np.array(legal_action, dtype=np.float32)
        return np.concatenate([local_view, rule_state, legal_arr]).astype(np.float32)

    def _shape_rule_reward(self, env_obs, ctx):
        reward = 0.0
        charge_count = self._safe_get_charge_count(env_obs)
        clean_score = self._safe_get_clean_score(env_obs)
        charge_gain = max(0, charge_count - self.last_env_charge_count)
        clean_gain = max(0, clean_score - self.last_env_clean_score)
        self.last_env_charge_count = charge_count
        self.last_env_clean_score = clean_score

        if charge_gain > 0:
            reward += 0.6 * charge_gain + 0.1 * min(5, int(ctx.get("charge_deficit", 0)))
        if ctx.get("charge_guard_active", False) and not ctx.get("in_charger_safe_zone", False):
            reward -= 0.03
        if ctx.get("need_return", False) and not ctx.get("on_charger", False):
            reward -= 0.02
        if (
            not ctx.get("need_return", False)
            and not ctx.get("early_charge", False)
            and ctx.get("in_charger_safe_zone", False)
            and ctx.get("distance_to_charger", 999) <= 4
            and ctx.get("battery_ratio", 0.0) >= 0.45
        ):
            reward -= 0.03
        if ctx.get("emergency", False):
            reward -= 0.05
        if clean_gain > 0 and ctx.get("in_charger_safe_zone", False):
            reward += 0.01 * clean_gain
        reward -= 0.002 * min(5, int(ctx.get("charge_deficit", 0)))
        return reward

    def _apply_rule_policy_overlay(self, prob, legal_arr):
        ctx = self.latest_rule_context or self._default_rule_context()
        mode = ctx.get("mode", "SAFE_EXPLORATION")
        rule_action = int(ctx.get("recommended_action", 0))
        if mode in Config.RULE_FORCE_MODES and 0 <= rule_action < Config.ACTION_NUM and legal_arr[rule_action] > 0:
            forced = np.zeros_like(prob, dtype=np.float32)
            forced[rule_action] = 1.0
            return forced

        if 0 <= rule_action < Config.ACTION_NUM and legal_arr[rule_action] > 0:
            rule_prior = np.zeros_like(prob, dtype=np.float32)
            rule_prior[rule_action] = 1.0
            prob = (1.0 - Config.RULE_GUIDE_BLEND) * prob + Config.RULE_GUIDE_BLEND * rule_prior
            prob = prob * legal_arr
        return self._normalize_prob(prob, legal_arr)

    def _normalize_prob(self, probs, legal_action):
        probs = np.asarray(probs, dtype=np.float64)
        legal = np.asarray(legal_action, dtype=np.float64)
        if probs.shape[0] != Config.ACTION_NUM:
            probs = np.ones(Config.ACTION_NUM, dtype=np.float64)
        if legal.shape[0] != Config.ACTION_NUM:
            legal = np.ones(Config.ACTION_NUM, dtype=np.float64)

        probs = np.clip(probs, 0.0, None) * legal
        total = probs.sum()
        if not np.isfinite(total) or total <= 0:
            probs = legal.copy()
            total = probs.sum()
        if total <= 0:
            probs = np.ones(Config.ACTION_NUM, dtype=np.float64)
            total = probs.sum()

        probs = probs / total
        # Guard against numerical issues in multinomial.
        probs = np.clip(probs, 0.0, 1.0)
        total = probs.sum()
        if total <= 0:
            probs = np.ones(Config.ACTION_NUM, dtype=np.float64) / Config.ACTION_NUM
        else:
            probs = probs / total
        probs[-1] = max(0.0, 1.0 - probs[:-1].sum())
        total = probs.sum()
        if total <= 0:
            probs = np.ones(Config.ACTION_NUM, dtype=np.float64) / Config.ACTION_NUM
        else:
            probs = probs / total
        return probs.astype(np.float32)

    def observation_process(self, env_obs):
        """Convert raw env_obs to ObsData (69D feature + legal action mask).

        将原始 env_obs 转换为 ObsData（69D 特征 + 合法动作掩码）。
        """
        feature, legal_action, reward = self.preprocessor.feature_process(env_obs, self.last_action)
        rule_action = self.rule_agent.act(env_obs)
        self.latest_rule_context = self._build_rule_context(env_obs, rule_action)
        legal_action = self._apply_rule_mask(legal_action, self.latest_rule_context)
        feature = self._augment_feature(feature, legal_action, self.latest_rule_context)
        self.last_reward = reward + self._shape_rule_reward(env_obs, self.latest_rule_context)

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
        if self.rule_agent is not None:
            self.rule_agent.last_action = self.last_action
        return self.last_action

    def predict(self, list_obs_data):
        """Stochastic inference for training (exploration).

        训练时推理（随机采样动作）。
        """
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        logits, value = self._run_model(feature)

        legal_arr = np.array(legal_action, dtype=np.float32)
        prob = self._legal_soft_max(logits, legal_arr)
        prob = self._apply_rule_policy_overlay(prob, legal_arr)
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
        self.logger.info(f"save model {model_file_path} successfully")

    def load_model(self, path=None, id="1"):
        """Load model checkpoint.

        加载模型检查点。
        """
        model_file_path = f"{path}/model.ckpt-{id}.pkl"
        try:
            loaded_state = torch.load(model_file_path, map_location=self.device)
            current_state = self.model.state_dict()
            compatible_state = {
                k: v for k, v in loaded_state.items()
                if k in current_state and tuple(current_state[k].shape) == tuple(v.shape)
            }
            self.model.load_state_dict(compatible_state, strict=False)
            dropped = len(loaded_state) - len(compatible_state)
            if dropped > 0:
                self.logger.info(f"load model {model_file_path} partially, dropped_incompatible_keys={dropped}")
            else:
                self.logger.info(f"load model {model_file_path} successfully")
        except Exception as exc:
            self.logger.info(f"skip loading model {model_file_path}, reason={exc}")

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

    def _legal_soft_max(self, logits, legal_action):
        """Softmax with legal action masking.

        合法动作掩码下的 softmax。
        """
        _w, _e = 1e20, 1e-5
        tmp = logits - _w * (1.0 - legal_action)
        tmp_max = np.max(tmp, keepdims=True)
        tmp = np.clip(tmp - tmp_max, -_w, 1)
        tmp = (np.exp(tmp) + _e) * legal_action
        return self._normalize_prob(tmp, legal_action)

    def _legal_sample(self, probs, use_max=False):
        """Sample action from probability distribution (argmax if use_max=True).

        按概率分布采样动作（use_max=True 时取 argmax）。
        """
        probs = self._normalize_prob(probs, np.ones(Config.ACTION_NUM, dtype=np.float32))
        if use_max:
            return int(np.argmax(probs))
        cdf = np.cumsum(probs, dtype=np.float64)
        cdf[-1] = 1.0
        r = float(np.random.random())
        return int(np.searchsorted(cdf, r, side="right"))
