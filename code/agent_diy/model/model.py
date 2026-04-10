#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Hybrid policy/value model for the DIY vacuum agent.
"""

import torch
import torch.nn as nn

from agent_diy.conf.conf import Config


def _make_fc(in_dim, out_dim, gain=1.41421):
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum_diy_hybrid"
        self.device = device

        hidden = Config.HIDDEN_DIM
        self.backbone = nn.Sequential(
            _make_fc(Config.FEATURE_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            _make_fc(256, hidden),
            nn.GELU(),
            _make_fc(hidden, 128),
            nn.GELU(),
        )
        self.actor_head = _make_fc(128, Config.ACTION_DIM, gain=0.01)
        self.critic_head = _make_fc(128, 1, gain=1.0)

    def forward(self, s, inference=False):
        x = s.to(torch.float32)
        h = self.backbone(x)
        logits = self.actor_head(h)
        value = self.critic_head(h)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
