#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Hierarchical PPO + imitation algorithm for the DIY vacuum agent.
"""

import os
import time

import numpy as np
import torch

from agent_diy.conf.conf import Config


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None):
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for group in optimizer.param_groups for p in group["params"]]
        self.device = device
        self.logger = logger
        self.monitor = monitor

        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF
        self.beta = Config.BETA_START

        self.train_step = 0
        self.last_report_time = 0

    def learn(self, list_sample_data):
        if not list_sample_data:
            return {"total_loss": 0.0}

        obs = self._stack_field(list_sample_data, "obs", torch.float32)
        candidate_feature = self._stack_field(list_sample_data, "candidate_feature", torch.float32)
        candidate_mask = self._stack_field(list_sample_data, "candidate_mask", torch.float32)
        old_action = self._stack_field(list_sample_data, "act", torch.long).view(-1, 1)
        old_prob = self._stack_field(list_sample_data, "prob", torch.float32)
        style_action = self._stack_field(list_sample_data, "style_act", torch.long).view(-1, 1)
        style_prob = self._stack_field(list_sample_data, "style_prob", torch.float32)
        old_value = self._stack_field(list_sample_data, "value", torch.float32).view(-1, 1)
        reward_sum = self._stack_field(list_sample_data, "reward_sum", torch.float32).view(-1, 1)
        advantage = self._stack_field(list_sample_data, "advantage", torch.float32).view(-1, 1)
        reward = self._stack_field(list_sample_data, "reward", torch.float32).view(-1, 1)
        teacher_prob = self._stack_field(list_sample_data, "teacher_prob", torch.float32)
        teacher_style_prob = self._stack_field(list_sample_data, "teacher_style_prob", torch.float32)
        teacher_weight = self._stack_field(list_sample_data, "teacher_weight", torch.float32).view(-1, 1)
        policy_weight = self._stack_field(list_sample_data, "policy_weight", torch.float32).view(-1, 1)

        advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-6)

        self.model.set_train_mode()
        self.optimizer.zero_grad()

        candidate_logits, style_logits, value_pred = self.model(obs, candidate_feature, candidate_mask)
        teacher_mix = self.get_teacher_mix()
        imitation_coef = self.get_imitation_coef()

        total_loss, info = self._compute_loss(
            candidate_logits=candidate_logits,
            style_logits=style_logits,
            value_pred=value_pred,
            candidate_mask=candidate_mask,
            old_action=old_action,
            old_prob=old_prob,
            style_action=style_action,
            style_prob=style_prob,
            old_value=old_value,
            reward_sum=reward_sum,
            advantage=advantage,
            teacher_prob=teacher_prob,
            teacher_style_prob=teacher_style_prob,
            imitation_coef=imitation_coef,
            teacher_weight=teacher_weight,
            policy_weight=policy_weight,
        )

        total_loss.backward()
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
        self.optimizer.step()
        self.train_step += 1

        results = {
            "total_loss": float(total_loss.item()),
            "reward": float(reward.mean().item()),
            "decision_policy_loss": round(info["decision_policy_loss"], 4),
            "style_policy_loss": round(info["style_policy_loss"], 4),
            "value_loss": round(info["value_loss"], 4),
            "entropy_loss": round(info["entropy_loss"], 4),
            "imitation_loss": round(info["imitation_loss"], 4),
            "teacher_mix": round(teacher_mix, 4),
            "imitation_coef": round(imitation_coef, 4),
            "teacher_weight": round(info["teacher_weight"], 4),
            "policy_weight": round(info["policy_weight"], 4),
        }

        now = time.time()
        if now - self.last_report_time >= 60:
            if self.logger:
                self.logger.info(
                    "decision_policy_loss: %.4f, style_policy_loss: %.4f, value_loss: %.4f, entropy_loss: %.4f, imitation_loss: %.4f, teacher_mix: %.4f, imitation_coef: %.4f"
                    % (
                        results["decision_policy_loss"],
                        results["style_policy_loss"],
                        results["value_loss"],
                        results["entropy_loss"],
                        results["imitation_loss"],
                        results["teacher_mix"],
                        results["imitation_coef"],
                    )
                )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_time = now

        return results

    def get_teacher_mix(self):
        return self._linear_decay(
            Config.TEACHER_MIX_START,
            Config.TEACHER_MIX_END,
            Config.TEACHER_MIX_DECAY_STEPS,
        )

    def get_imitation_coef(self):
        return self._linear_decay(
            Config.IMITATION_COEF_START,
            Config.IMITATION_COEF_END,
            Config.IMITATION_DECAY_STEPS,
        )

    def _linear_decay(self, start, end, steps):
        if steps <= 0:
            return float(end)
        ratio = min(float(self.train_step) / float(steps), 1.0)
        return float(start + (end - start) * ratio)

    def _compute_loss(
        self,
        candidate_logits,
        style_logits,
        value_pred,
        candidate_mask,
        old_action,
        old_prob,
        style_action,
        style_prob,
        old_value,
        reward_sum,
        advantage,
        teacher_prob,
        teacher_style_prob,
        imitation_coef,
        teacher_weight,
        policy_weight,
    ):
        candidate_prob = self._masked_softmax(candidate_logits, candidate_mask)
        teacher_prob = self._normalize_teacher_prob(teacher_prob, candidate_mask)
        style_prob_new = torch.softmax(style_logits, dim=1)
        teacher_style_prob = self._normalize_style_prob(teacher_style_prob)

        vp = value_pred.view(-1, 1)
        vp_clip = old_value + (vp - old_value).clamp(-self.clip_param, self.clip_param)
        value_loss = 0.5 * torch.maximum((reward_sum - vp) ** 2, (reward_sum - vp_clip) ** 2).mean()

        decision_entropy = self._weighted_mean(
            -(candidate_prob * torch.log(candidate_prob.clamp_min(1e-9))).sum(1, keepdim=True),
            policy_weight,
        )
        style_entropy = self._weighted_mean(
            -(style_prob_new * torch.log(style_prob_new.clamp_min(1e-9))).sum(1, keepdim=True),
            policy_weight,
        )
        entropy_loss = decision_entropy + Config.STYLE_ENTROPY_COEF * style_entropy

        decision_one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), Config.MAX_DECISION_CANDIDATES).float()
        new_decision_prob = (decision_one_hot * candidate_prob).sum(1, keepdim=True)
        old_decision_prob = (decision_one_hot * old_prob).sum(1, keepdim=True)
        decision_ratio = new_decision_prob / old_decision_prob.clamp_min(1e-9)
        decision_policy_loss = self._weighted_mean(
            torch.maximum(
                -decision_ratio * advantage,
                -decision_ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage,
            ),
            policy_weight,
        )

        style_one_hot = torch.nn.functional.one_hot(style_action[:, 0].long(), Config.PATH_STYLE_DIM).float()
        new_style_prob = (style_one_hot * style_prob_new).sum(1, keepdim=True)
        old_style_prob = (style_one_hot * style_prob).sum(1, keepdim=True)
        style_ratio = new_style_prob / old_style_prob.clamp_min(1e-9)
        style_policy_loss = self._weighted_mean(
            torch.maximum(
                -style_ratio * advantage,
                -style_ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * advantage,
            ),
            policy_weight,
        )

        decision_imitation = self._weighted_mean(
            -(teacher_prob * torch.log(candidate_prob.clamp_min(1e-9))).sum(1, keepdim=True),
            teacher_weight,
        )
        style_imitation = self._weighted_mean(
            -(teacher_style_prob * torch.log(style_prob_new.clamp_min(1e-9))).sum(1, keepdim=True),
            teacher_weight,
        )
        imitation_loss = decision_imitation + Config.STYLE_IMITATION_COEF * style_imitation

        total_loss = (
            self.vf_coef * value_loss
            + decision_policy_loss
            + Config.STYLE_POLICY_COEF * style_policy_loss
            - self.beta * entropy_loss
            + imitation_coef * imitation_loss
        )
        return total_loss, {
            "decision_policy_loss": float(decision_policy_loss.item()),
            "style_policy_loss": float(style_policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy_loss": float(entropy_loss.item()),
            "imitation_loss": float(imitation_loss.item()),
            "teacher_weight": float(teacher_weight.mean().item()),
            "policy_weight": float(policy_weight.mean().item()),
        }

    def _stack_field(self, list_sample_data, field, dtype):
        arr = np.stack([np.array(getattr(sample, field), dtype=np.float32) for sample in list_sample_data])
        if dtype == torch.long:
            arr = arr.astype(np.int64)
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def _masked_softmax(self, logits, mask):
        legal = (mask > 0.5).float()
        legal_sum = legal.sum(dim=1, keepdim=True)
        legal = torch.where(legal_sum > 0, legal, torch.ones_like(legal))
        label_max, _ = torch.max(logits * legal, dim=1, keepdim=True)
        masked_logits = logits - label_max
        masked_logits = masked_logits * legal
        masked_logits = masked_logits + 1e5 * (legal - 1.0)
        return torch.nn.functional.softmax(masked_logits, dim=1)

    def _normalize_teacher_prob(self, teacher_prob, candidate_mask):
        teacher_prob = torch.clamp(teacher_prob * candidate_mask, min=0.0)
        denom = teacher_prob.sum(dim=1, keepdim=True)
        fallback = candidate_mask / candidate_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        return torch.where(denom > 0.0, teacher_prob / denom.clamp_min(1e-9), fallback)

    def _normalize_style_prob(self, style_prob):
        style_prob = torch.clamp(style_prob, min=0.0)
        denom = style_prob.sum(dim=1, keepdim=True)
        fallback = torch.full_like(style_prob, 1.0 / Config.PATH_STYLE_DIM)
        return torch.where(denom > 0.0, style_prob / denom.clamp_min(1e-9), fallback)

    def _weighted_mean(self, value, weight):
        weight_sum = weight.sum()
        if float(weight_sum.item()) <= 1e-6:
            return value.new_tensor(0.0)
        return (value * weight).sum() / weight_sum.clamp_min(1e-6)
