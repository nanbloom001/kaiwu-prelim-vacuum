#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Planner imitation and value regression for robot_vacuum.
"""

import os
import time
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from agent_ppo.feature.definition import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.device = device
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        self.logger = logger
        self.monitor = monitor

        self.var_beta = Config.BETA_START
        self.vf_coef = Config.VF_COEF
        self.last_report_monitor_time = 0.0
        self.train_step = 0

    def learn(self, list_sample_data: List[Any]) -> Dict[str, float]:
        if not list_sample_data:
            return {}

        obs = self._stack_field(list_sample_data, "obs", Config.DIM_OF_OBSERVATION)
        legal_action = self._stack_field(list_sample_data, "legal_action", Config.ACTION_NUM)
        act = self._stack_field(list_sample_data, "act", 1).long()
        reward = self._stack_field(list_sample_data, "reward", Config.VALUE_NUM)
        reward_sum = self._stack_field(list_sample_data, "reward_sum", Config.VALUE_NUM)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        logits, value_pred = self.model(obs)
        masked_logits = self._masked_logits(logits, legal_action)
        action_idx = act[:, 0].clamp(min=0, max=Config.ACTION_NUM - 1)

        policy_loss = F.cross_entropy(masked_logits, action_idx)
        value_loss = 0.5 * torch.square(reward_sum - value_pred).mean()
        prob = torch.softmax(masked_logits, dim=1)
        entropy_loss = (-prob * torch.log(prob.clamp(min=1e-9))).sum(dim=1).mean()

        total_loss = policy_loss + self.vf_coef * value_loss - self.var_beta * entropy_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        metrics = {
            "total_loss": float(total_loss.item()),
            "value_loss": float(value_loss.item()),
            "policy_loss": float(policy_loss.item()),
            "entropy_loss": float(entropy_loss.item()),
            "reward": float(reward.mean().item()),
        }
        self._report(metrics)
        return metrics

    def _masked_logits(self, logits: torch.Tensor, legal_action: torch.Tensor) -> torch.Tensor:
        legal = (legal_action > 0.5).float()
        legal_sum = legal.sum(dim=1, keepdim=True)
        legal = torch.where(legal_sum > 0, legal, torch.ones_like(legal))
        return logits + (legal - 1.0) * 1e9

    def _stack_field(self, samples: List[Any], field: str, dim: int) -> torch.Tensor:
        rows = []
        for sample in samples:
            value = getattr(sample, field)
            if not torch.is_tensor(value):
                value = torch.tensor(value, dtype=torch.float32)
            value = value.float().reshape(-1)
            if value.numel() < dim:
                pad = torch.zeros(dim - value.numel(), dtype=value.dtype, device=value.device)
                value = torch.cat([value, pad], dim=0)
            elif value.numel() > dim:
                value = value[:dim]
            rows.append(value)
        return torch.stack(rows, dim=0).to(self.device)

    def _report(self, metrics: Dict[str, float]) -> None:
        now = time.time()
        if now - self.last_report_monitor_time < 60:
            return

        if self.logger is not None:
            self.logger.info(
                "[train] total_loss:{:.4f} policy_loss:{:.4f} value_loss:{:.4f} entropy:{:.4f} reward:{:.4f}".format(
                    metrics["total_loss"],
                    metrics["policy_loss"],
                    metrics["value_loss"],
                    metrics["entropy_loss"],
                    metrics["reward"],
                )
            )

        if self.monitor is not None:
            self.monitor.put_data({os.getpid(): metrics})

        self.last_report_monitor_time = now

