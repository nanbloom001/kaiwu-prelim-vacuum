#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Entity-aware recurrent dual-critic LTSPPO model for Robot Vacuum.
"""

from __future__ import annotations

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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = _make_conv(channels, channels)
        self.conv2 = _make_conv(channels, channels)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return self.act(x + identity)


class Model(nn.Module):
    def __init__(self, device=None):
        super().__init__()
        self.model_name = "robot_vacuum_ltsppo"
        self.device = device

        (
            self.local_dim,
            self.global_dim,
            self.entity_dim,
            self.scalar_dim,
            self.action_history_dim,
        ) = Config.FEATURE_SPLIT_SHAPE

        # Local branch
        self.local_encoder = nn.Sequential(
            _make_conv(Config.LOCAL_VIEW_CHANNELS, 32),
            nn.ReLU(),
            ResidualBlock(32),
            ResidualBlock(32),
            _make_conv(32, 48, stride=2),
            nn.ReLU(),
            ResidualBlock(48),
            ResidualBlock(48),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            _make_fc(48, Config.LOCAL_EMBED_DIM),
            nn.ReLU(),
        )

        # Global branch
        self.global_encoder = nn.Sequential(
            _make_conv(Config.GLOBAL_MEMORY_CHANNELS, 24),
            nn.ReLU(),
            ResidualBlock(24),
            _make_conv(24, 32, stride=2),
            nn.ReLU(),
            ResidualBlock(32),
            ResidualBlock(32),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            _make_fc(32, Config.GLOBAL_EMBED_DIM),
            nn.ReLU(),
        )

        # Entity branch
        self.entity_token_encoder = nn.Sequential(
            _make_fc(Config.ENTITY_FEATURE_DIM, 32),
            nn.ReLU(),
            _make_fc(32, 32),
            nn.ReLU(),
        )
        self.entity_proj = nn.Sequential(
            _make_fc(64, Config.ENTITY_EMBED_DIM),
            nn.ReLU(),
        )

        # Scalar / action history
        self.scalar_proj = nn.Sequential(
            _make_fc(self.scalar_dim, 128),
            nn.ReLU(),
            _make_fc(128, Config.SCALAR_EMBED_DIM),
            nn.ReLU(),
        )
        self.action_history_proj = nn.Sequential(
            _make_fc(self.action_history_dim, 32),
            nn.ReLU(),
            _make_fc(32, Config.ACTION_HISTORY_EMBED_DIM),
            nn.ReLU(),
        )

        fused_dim = (
            Config.LOCAL_EMBED_DIM
            + Config.GLOBAL_EMBED_DIM
            + Config.ENTITY_EMBED_DIM
            + Config.SCALAR_EMBED_DIM
            + Config.ACTION_HISTORY_EMBED_DIM
        )
        self.fuse = nn.Sequential(
            _make_fc(fused_dim, Config.FUSED_HIDDEN_DIM),
            nn.ReLU(),
            _make_fc(Config.FUSED_HIDDEN_DIM, Config.PRE_RNN_DIM),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=Config.PRE_RNN_DIM,
            hidden_size=Config.RNN_HIDDEN_DIM,
            num_layers=1,
            batch_first=True,
        )

        # Mode and target
        self.mode_head = _make_fc(Config.RNN_HIDDEN_DIM, Config.MODE_NUM, gain=0.01)
        self.mode_embedding = nn.Parameter(torch.zeros(Config.MODE_NUM, Config.MODE_EMBED_DIM))
        nn.init.orthogonal_(self.mode_embedding)

        self.target_query = _make_fc(Config.RNN_HIDDEN_DIM, 32)
        self.target_key = _make_fc(32, 32)
        self.target_none_head = _make_fc(Config.RNN_HIDDEN_DIM, 1, gain=0.01)
        self.target_context_proj = nn.Sequential(
            _make_fc(32, Config.TARGET_CONTEXT_DIM),
            nn.ReLU(),
        )

        actor_in_dim = Config.RNN_HIDDEN_DIM + Config.TARGET_CONTEXT_DIM + Config.MODE_EMBED_DIM
        self.actor_head = nn.Sequential(
            _make_fc(actor_in_dim, 128),
            nn.ReLU(),
            _make_fc(128, Config.ACTION_NUM, gain=0.01),
        )
        self.clean_critic = nn.Sequential(
            _make_fc(Config.RNN_HIDDEN_DIM, 64),
            nn.ReLU(),
            _make_fc(64, 1, gain=0.01),
        )
        self.survive_critic = nn.Sequential(
            _make_fc(Config.RNN_HIDDEN_DIM, 64),
            nn.ReLU(),
            _make_fc(64, 1, gain=0.01),
        )
        self.aux_battery_risk = nn.Sequential(
            _make_fc(Config.RNN_HIDDEN_DIM, 64),
            nn.ReLU(),
            _make_fc(64, 1, gain=0.01),
        )
        self.aux_collision_risk = nn.Sequential(
            _make_fc(Config.RNN_HIDDEN_DIM, 64),
            nn.ReLU(),
            _make_fc(64, 1, gain=0.01),
        )

    def init_rnn_state(self, batch_size, device=None):
        dev = device or self.device or next(self.parameters()).device
        return torch.zeros(1, batch_size, Config.RNN_HIDDEN_DIM, device=dev, dtype=torch.float32)

    def forward(self, obs, rnn_state=None, inference=False):
        if obs.dim() == 2:
            return self.forward_step(obs, rnn_state=rnn_state)
        if obs.dim() == 3:
            return self.forward_sequence(obs, rnn_state=rnn_state)
        raise ValueError(f"Unsupported obs rank {obs.dim()}")

    def forward_step(self, obs, rnn_state=None):
        x = obs.to(torch.float32)
        pre_rnn, charger_tokens = self._encode_obs(x)
        seq_in = pre_rnn.unsqueeze(1)
        if rnn_state is None:
            rnn_state = self.init_rnn_state(seq_in.shape[0], device=seq_in.device)
        rnn_out, next_state = self.gru(seq_in, rnn_state)
        rnn_out = rnn_out[:, 0, :]
        return self._decode_heads(rnn_out, charger_tokens, next_state)

    def forward_sequence(self, obs_seq, rnn_state=None):
        bsz, seq_len, obs_dim = obs_seq.shape
        flat = obs_seq.reshape(bsz * seq_len, obs_dim).to(torch.float32)
        pre_rnn, charger_tokens = self._encode_obs(flat)
        pre_rnn = pre_rnn.view(bsz, seq_len, -1)
        charger_tokens = charger_tokens.view(bsz, seq_len, Config.CHARGER_SLOTS, -1)
        if rnn_state is None:
            rnn_state = self.init_rnn_state(bsz, device=pre_rnn.device)
        rnn_out, next_state = self.gru(pre_rnn, rnn_state)
        return self._decode_heads(
            rnn_out.reshape(bsz * seq_len, -1),
            charger_tokens.reshape(bsz * seq_len, Config.CHARGER_SLOTS, -1),
            next_state,
            reshape=(bsz, seq_len),
        )

    def _encode_obs(self, obs):
        local_map, global_memory, entity_state, scalar_state, action_history = self._split_obs(obs)

        local_feature = self.local_encoder(local_map)
        global_feature = self.global_encoder(global_memory)

        entity_tokens = self.entity_token_encoder(entity_state.reshape(-1, Config.ENTITY_FEATURE_DIM))
        entity_tokens = entity_tokens.view(-1, Config.ENTITY_SLOTS, 32)
        npc_tokens = entity_tokens[:, : Config.NPC_SLOTS, :]
        charger_tokens = entity_tokens[:, Config.NPC_SLOTS :, :]
        npc_summary = npc_tokens.mean(dim=1)
        charger_summary = charger_tokens.mean(dim=1)
        entity_feature = self.entity_proj(torch.cat([npc_summary, charger_summary], dim=1))

        scalar_feature = self.scalar_proj(scalar_state)
        action_hist_feature = self.action_history_proj(action_history)

        fused = torch.cat(
            [local_feature, global_feature, entity_feature, scalar_feature, action_hist_feature],
            dim=1,
        )
        pre_rnn = self.fuse(fused)
        return pre_rnn, charger_tokens

    def _decode_heads(self, rnn_out, charger_tokens, next_state, reshape=None):
        mode_logits = self.mode_head(rnn_out)
        mode_probs = torch.softmax(mode_logits, dim=-1)
        mode_context = mode_probs @ self.mode_embedding

        query = self.target_query(rnn_out).unsqueeze(1)
        keys = self.target_key(charger_tokens)
        target_logits_slots = torch.sum(query * keys, dim=-1)
        none_logit = self.target_none_head(rnn_out)
        target_logits = torch.cat([none_logit, target_logits_slots], dim=-1)
        target_probs = torch.softmax(target_logits, dim=-1)
        charger_probs = target_probs[:, 1:].unsqueeze(-1)
        target_context = self.target_context_proj(torch.sum(charger_tokens * charger_probs, dim=1))

        actor_input = torch.cat([rnn_out, target_context, mode_context], dim=-1)
        policy_logits = self.actor_head(actor_input)
        value_clean = self.clean_critic(rnn_out)
        value_survive = self.survive_critic(rnn_out)
        aux_battery_risk = self.aux_battery_risk(rnn_out)
        aux_collision_risk = self.aux_collision_risk(rnn_out)

        outputs = {
            "policy_logits": policy_logits,
            "mode_logits": mode_logits,
            "target_logits": target_logits,
            "value_clean": value_clean,
            "value_survive": value_survive,
            "aux_battery_risk": aux_battery_risk,
            "aux_collision_risk": aux_collision_risk,
            "mode_probs": mode_probs,
            "target_probs": target_probs,
            "next_rnn_state": next_state,
        }
        if reshape is not None:
            bsz, seq_len = reshape
            for key in (
                "policy_logits",
                "mode_logits",
                "target_logits",
                "value_clean",
                "value_survive",
                "aux_battery_risk",
                "aux_collision_risk",
                "mode_probs",
                "target_probs",
            ):
                outputs[key] = outputs[key].view(bsz, seq_len, -1)
        return outputs

    def _split_obs(self, obs):
        begin = 0
        local_flat = obs[:, begin : begin + self.local_dim]
        begin += self.local_dim
        global_flat = obs[:, begin : begin + self.global_dim]
        begin += self.global_dim
        entity_flat = obs[:, begin : begin + self.entity_dim]
        begin += self.entity_dim
        scalar_state = obs[:, begin : begin + self.scalar_dim]
        begin += self.scalar_dim
        action_history = obs[:, begin : begin + self.action_history_dim]

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
        entity_state = entity_flat.view(-1, Config.ENTITY_SLOTS, Config.ENTITY_FEATURE_DIM)
        return local_map, global_memory, entity_state, scalar_state, action_history

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()
