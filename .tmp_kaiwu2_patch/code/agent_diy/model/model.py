#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Hierarchical candidate policy/value model for the DIY vacuum agent.
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
        self.model_name = "robot_vacuum_diy_hierarchical"
        self.device = device

        hidden = Config.HIDDEN_DIM
        self.state_encoder = nn.Sequential(
            _make_fc(Config.FEATURE_DIM, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            _make_fc(256, hidden),
            nn.GELU(),
        )
        self.candidate_encoder = nn.Sequential(
            _make_fc(Config.CANDIDATE_FEATURE_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            _make_fc(hidden, hidden),
            nn.GELU(),
        )
        self.candidate_head = _make_fc(hidden * 2, 1, gain=0.01)
        self.style_head = nn.Sequential(
            _make_fc(hidden * 2, hidden),
            nn.GELU(),
            _make_fc(hidden, Config.PATH_STYLE_DIM, gain=0.01),
        )
        self.value_head = _make_fc(hidden * 2, 1, gain=1.0)

    def forward(self, state_feature, candidate_feature, candidate_mask, inference=False):
        state = state_feature.to(torch.float32)
        candidate = candidate_feature.to(torch.float32)
        mask = candidate_mask.to(torch.float32)

        batch_size = candidate.shape[0]
        candidate = candidate.view(batch_size, Config.MAX_DECISION_CANDIDATES, Config.CANDIDATE_FEATURE_DIM)

        state_ctx = self.state_encoder(state)
        candidate_ctx = self.candidate_encoder(candidate.view(-1, Config.CANDIDATE_FEATURE_DIM))
        candidate_ctx = candidate_ctx.view(batch_size, Config.MAX_DECISION_CANDIDATES, -1)

        expanded_state = state_ctx.unsqueeze(1).expand(-1, Config.MAX_DECISION_CANDIDATES, -1)
        candidate_joint = torch.cat([expanded_state, candidate_ctx], dim=-1)
        candidate_logits = self.candidate_head(candidate_joint).squeeze(-1)

        valid_mask = mask.unsqueeze(-1)
        pooled_candidate = (candidate_ctx * valid_mask).sum(dim=1)
        pooled_candidate = pooled_candidate / valid_mask.sum(dim=1).clamp_min(1.0)
        global_ctx = torch.cat([state_ctx, pooled_candidate], dim=-1)

        style_logits = self.style_head(global_ctx)
        value = self.value_head(global_ctx)
        return [candidate_logits, style_logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
