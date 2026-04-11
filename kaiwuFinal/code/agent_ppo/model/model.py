#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Hybrid CNN + MLP policy network for Robot Vacuum.
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


def _make_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, gain=1.41421):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        local_dim, global_dim, scalar_dim, legal_dim = Config.FEATURE_SPLIT_SHAPE
        self.local_dim = local_dim
        self.global_dim = global_dim
        self.scalar_dim = scalar_dim
        self.legal_dim = legal_dim

        self.local_encoder = nn.Sequential(
            _make_conv(Config.LOCAL_VIEW_CHANNELS, 16),
            nn.ReLU(),
            _make_conv(16, 32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            _make_conv(32, 32),
            nn.ReLU(),
            nn.Flatten(),
        )
        local_flat_dim = 32 * 10 * 10
        self.local_proj = nn.Sequential(
            _make_fc(local_flat_dim, 256),
            nn.ReLU(),
        )

        self.global_encoder = nn.Sequential(
            _make_conv(Config.GLOBAL_MEMORY_CHANNELS, 16),
            nn.ReLU(),
            _make_conv(16, 16),
            nn.ReLU(),
            nn.Flatten(),
        )
        global_flat_dim = 16 * Config.GLOBAL_MEMORY_SIZE * Config.GLOBAL_MEMORY_SIZE
        self.global_proj = nn.Sequential(
            _make_fc(global_flat_dim, 64),
            nn.ReLU(),
        )

        self.scalar_proj = nn.Sequential(
            _make_fc(self.scalar_dim + self.legal_dim, 64),
            nn.ReLU(),
            _make_fc(64, 64),
            nn.ReLU(),
        )

        self.backbone = nn.Sequential(
            _make_fc(256 + 64 + 64, 256),
            nn.ReLU(),
            _make_fc(256, 128),
            nn.ReLU(),
        )

        self.actor_head = _make_fc(128, Config.ACTION_NUM, gain=0.01)
        self.critic_head = _make_fc(128, 1, gain=0.01)

    def forward(self, s, inference=False):
        x = s.to(torch.float32)
        local_map, global_memory, scalar_state, legal_state = self._split_obs(x)

        local_feature = self.local_proj(self.local_encoder(local_map))
        global_feature = self.global_proj(self.global_encoder(global_memory))
        scalar_feature = self.scalar_proj(torch.cat([scalar_state, legal_state], dim=1))

        fused = torch.cat([local_feature, global_feature, scalar_feature], dim=1)
        hidden = self.backbone(fused)
        logits = self.actor_head(hidden)
        value = self.critic_head(hidden)
        return [logits, value]

    def _split_obs(self, obs):
        begin = 0
        local_flat = obs[:, begin : begin + self.local_dim]
        begin += self.local_dim
        global_flat = obs[:, begin : begin + self.global_dim]
        begin += self.global_dim
        scalar_state = obs[:, begin : begin + self.scalar_dim]
        begin += self.scalar_dim
        legal_state = obs[:, begin : begin + self.legal_dim]

        local_map = local_flat.view(
            -1,
            Config.LOCAL_VIEW_CHANNELS,
            Config.LOCAL_VIEW_SIZE,
            Config.LOCAL_VIEW_SIZE,
        )
        global_memory = global_flat.view(
            -1,
            Config.GLOBAL_MEMORY_CHANNELS,
            Config.GLOBAL_MEMORY_SIZE,
            Config.GLOBAL_MEMORY_SIZE,
        )
        return local_map, global_memory, scalar_state, legal_state

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
