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

from agent_ppo.algorithm.algorithm import Algorithm
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from kaiwudrl.interface.agent import BaseAgent
from agent_ppo.utils.experiment_archive import ExperimentArchive, parse_checkpoint_id


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
        self.archive = ExperimentArchive()
        self.last_action = -1
        self.last_reward = 0.0
        self.current_model_ref = {
            "path": None,
            "id": None,
            "checkpoint_id": None,
        }

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

        Training inference with expert override, NPC safety filter, and anti-stuck.
        """
        obs_data = list_obs_data[0]
        feature = obs_data.feature
        legal_action = obs_data.legal_action

        logits, value = self._run_model(feature)
        expert = self.preprocessor.expert

        # Layer 2: Expert strategic override (charging)
        should_override, expert_action = expert.get_override(self.preprocessor, legal_action)
        if should_override:
            prob = self._uniform_over_legal(legal_action)
            return [
                ActData(
                    action=[expert_action],
                    d_action=[expert_action],
                    prob=prob,
                    value=value,
                )
            ]

        # Layer 1: NPC safety filter — block moves toward nearby NPCs
        filtered_legal = expert.filter_actions(self.preprocessor, legal_action)

        # Layer 3: Anti-stuck — random legal action if stuck too long
        if self.preprocessor.stuck_steps >= 10:
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

        legal_arr = np.array(filtered_legal, dtype=np.float32)
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
