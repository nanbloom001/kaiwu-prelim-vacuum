#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Neural network model for robot_vacuum PPO baseline.
"""

import torch
import torch.nn as nn

from agent_ppo.feature.definition import Config


def _make_fc_layer(in_features: int, out_features: int) -> nn.Linear:
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """MLP backbone with actor/critic dual heads."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum_ppo"
        self.device = device

        self.backbone = nn.Sequential(
            _make_fc_layer(Config.DIM_OF_OBSERVATION, 128),
            nn.ReLU(),
            _make_fc_layer(128, 64),
            nn.ReLU(),
        )
        self.actor_head = _make_fc_layer(64, Config.ACTION_NUM)
        self.critic_head = _make_fc_layer(64, Config.VALUE_NUM)

    def forward(self, obs, inference=False):
        hidden = self.backbone(obs)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()

