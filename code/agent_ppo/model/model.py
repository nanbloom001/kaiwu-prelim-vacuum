#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Actor-Critic MLP policy network for Robot Vacuum.
清扫大作战 Actor-Critic 策略网络。

网络结构：
  输入(77D)
    → FC(256) → LayerNorm → ReLU
    → FC(128) → LayerNorm → ReLU
    ┌─────────────────────┐
    ↓                     ↓
  Actor(8D)           Critic(1D)
  动作 logits         状态价值 V(s)

设计要点：
  - 正交初始化（orthogonal init）稳定训练早期梯度
  - LayerNorm 替换 BatchNorm，对单样本推理友好
  - Actor/Critic 头使用小 gain（0.01）避免初始输出饱和
"""

import torch
import torch.nn as nn

from agent_ppo.conf.conf import Config


def _make_fc(in_dim: int, out_dim: int, gain: float = 1.41421) -> nn.Linear:
    """创建正交初始化的线性层。"""
    layer = nn.Linear(in_dim, out_dim)
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)
    return layer


class ResBlock(nn.Module):
    """轻量残差块：两层 FC + LayerNorm，带跳连。用于共享骨干。"""

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = _make_fc(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.fc2 = _make_fc(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.act(self.norm1(self.fc1(x)))
        x = self.norm2(self.fc2(x))
        return self.act(x + residual)


class Model(nn.Module):
    """
    双头 MLP（Actor-Critic）策略网络。

    输入维度由 Config.DIM_OF_OBSERVATION 决定（当前 84D）。
    """

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum"
        self.device = device

        obs_dim = Config.DIM_OF_OBSERVATION
        act_num = Config.ACTION_NUM
        h1 = Config.HIDDEN_DIM_1
        h2 = Config.HIDDEN_DIM_2

        self.backbone = nn.Sequential(
            _make_fc(obs_dim, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            _make_fc(h1, h2),
            nn.LayerNorm(h2),
            nn.ReLU(),
        )
        self.res = ResBlock(h2)
        self.actor_head = _make_fc(h2, act_num, gain=0.01)
        self.critic_head = _make_fc(h2, 1, gain=0.01)

    def forward(self, s: torch.Tensor, inference: bool = False):
        x = s.to(torch.float32)
        h = self.backbone(x)
        h = self.res(h)
        logits = self.actor_head(h)
        value = self.critic_head(h)
        return [logits, value]

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
