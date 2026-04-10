#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Standard PPO algorithm for Robot Vacuum.
清扫大作战 PPO 算法。

Loss composition / 损失组成：
  total_loss = vf_coef * value_loss + policy_loss - beta * entropy_loss
"""

import os
import time

import numpy as np
import torch

from agent_ppo.conf.conf import Config
from agent_ppo.utils.experiment_archive import ExperimentArchive


class Algorithm:
    def __init__(self, model, optimizer, device=None, logger=None, monitor=None, use_amp=False):
        self.model = model
        self.optimizer = optimizer
        self.parameters = [p for pg in optimizer.param_groups for p in pg["params"]]
        self.device = device
        self.logger = logger
        self.monitor = monitor
        self.use_amp = use_amp and device is not None and getattr(device, "type", "") == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.clip_param = Config.CLIP_PARAM
        self.vf_coef = Config.VF_COEF
        self.var_beta = Config.BETA_START
        self.label_size = Config.ACTION_NUM
        self.archive = ExperimentArchive(service_name=os.getenv("KAIWU_SERVICE_NAME") or "learner")

        self.train_step = 0
        self.last_report_time = 0

    def learn(self, list_sample_data):
        """Training entry: perform one PPO gradient step on a batch of SampleData.

        训练入口：接收一批 SampleData，执行一步梯度更新。
        """
        (
            obs,
            legal_action,
            act,
            old_prob,
            old_value,
            reward_sum,
            advantage,
            reward,
        ) = self._unpack_train_batch(list_sample_data)

        self.model.set_train_mode()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            rst_list = self.model(obs)
            logits, value_pred = rst_list[0], rst_list[1]

            total_loss, info = self._compute_loss(
                logits=logits,
                value_pred=value_pred,
                legal_action=legal_action,
                old_action=act,
                old_prob=old_prob,
                old_value=old_value,
                reward_sum=reward_sum,
                advantage=advantage,
            )

        self.scaler.scale(total_loss).backward()

        if Config.USE_GRAD_CLIP:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.train_step += 1

        results = {"total_loss": total_loss.item()}

        # Periodic monitoring report
        # 定期上报监控
        now = time.time()
        if now - self.last_report_time >= 60:
            results["value_loss"] = round(info["value_loss"], 4)
            results["policy_loss"] = round(info["policy_loss"], 4)
            results["entropy_loss"] = round(info["entropy_loss"], 4)
            results["reward"] = round(reward.mean().item(), 4)
            results["train_step"] = self.train_step

            self.logger.info(
                f"policy_loss: {results['policy_loss']}, "
                f"value_loss: {results['value_loss']}, "
                f"entropy_loss: {results['entropy_loss']}"
            )
            self.archive.log_train_window(
                {
                    "record_type": "algorithm_window",
                    "train_step": self.train_step,
                    "total_loss": results["total_loss"],
                    "policy_loss": results["policy_loss"],
                    "value_loss": results["value_loss"],
                    "entropy_loss": results["entropy_loss"],
                    "reward_mean": results["reward"],
                }
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})

            self.last_report_time = now

        return results

    def _unpack_train_batch(self, batch_data):
        if isinstance(batch_data, (torch.Tensor, np.ndarray)):
            return self._unpack_batch_tensor(batch_data)
        return self._unpack_sample_objects(batch_data)

    def _unpack_batch_tensor(self, batch_data):
        if isinstance(batch_data, np.ndarray):
            batch_tensor = torch.from_numpy(batch_data)
        else:
            batch_tensor = batch_data

        if batch_tensor.dim() == 1:
            batch_tensor = batch_tensor.unsqueeze(0)

        if batch_tensor.dtype != torch.float32:
            batch_tensor = batch_tensor.to(torch.float32)

        to_device = {"device": self.device}
        if self.use_amp:
            to_device["non_blocking"] = True
        if self.device is not None and batch_tensor.device != self.device:
            batch_tensor = batch_tensor.to(**to_device)

        begin = 0

        obs = batch_tensor[:, begin : begin + Config.DIM_OF_OBSERVATION]
        begin += Config.DIM_OF_OBSERVATION

        legal_action = batch_tensor[:, begin : begin + Config.ACTION_NUM]
        begin += Config.ACTION_NUM

        act = batch_tensor[:, begin : begin + 1]
        begin += 1

        reward = batch_tensor[:, begin : begin + Config.VALUE_NUM]
        begin += Config.VALUE_NUM

        reward_sum = batch_tensor[:, begin : begin + Config.VALUE_NUM]
        begin += Config.VALUE_NUM

        begin += 1  # done

        old_value = batch_tensor[:, begin : begin + Config.VALUE_NUM]
        begin += Config.VALUE_NUM

        begin += Config.VALUE_NUM  # next_value

        advantage = batch_tensor[:, begin : begin + Config.VALUE_NUM]
        begin += Config.VALUE_NUM

        old_prob = batch_tensor[:, begin : begin + Config.ACTION_NUM]

        return (
            obs,
            legal_action,
            act,
            old_prob,
            old_value,
            reward_sum,
            advantage,
            reward,
        )

    def _unpack_sample_objects(self, list_sample_data):
        to_device = {"device": self.device}
        if self.use_amp:
            to_device["non_blocking"] = True

        obs = torch.stack([s.obs for s in list_sample_data]).to(**to_device)
        legal_action = torch.stack([s.legal_action for s in list_sample_data]).to(**to_device)
        act = torch.stack([s.act for s in list_sample_data]).to(**to_device).view(-1, 1)
        old_prob = torch.stack([s.prob for s in list_sample_data]).to(**to_device)
        old_value = torch.stack([s.value for s in list_sample_data]).to(**to_device)
        reward_sum = torch.stack([s.reward_sum for s in list_sample_data]).to(**to_device)
        advantage = torch.stack([s.advantage for s in list_sample_data]).to(**to_device)
        reward = torch.stack([s.reward for s in list_sample_data]).to(**to_device)

        return (
            obs,
            legal_action,
            act,
            old_prob,
            old_value,
            reward_sum,
            advantage,
            reward,
        )

    def _compute_loss(self, logits, value_pred, legal_action, old_action, old_prob, old_value, reward_sum, advantage):
        """Compute standard PPO loss (policy + value + entropy).

        计算标准 PPO 三项损失。
        """
        # Value loss (clipped)
        # 价值损失（裁剪）
        tdret = reward_sum.squeeze(-1) if reward_sum.dim() > 1 else reward_sum
        vp = value_pred.squeeze(-1) if value_pred.dim() > 1 else value_pred
        ov = old_value.squeeze(-1) if old_value.dim() > 1 else old_value

        vp_clip = ov + (vp - ov).clamp(-self.clip_param, self.clip_param)
        value_loss = (
            0.5
            * torch.maximum(
                (tdret - vp) ** 2,
                (tdret - vp_clip) ** 2,
            ).mean()
        )

        # Policy loss (PPO clip)
        # 策略损失（PPO clip）
        prob_dist = self._masked_softmax(logits, legal_action)
        entropy_loss = (-(prob_dist * torch.log(prob_dist.clamp(1e-9, 1))).sum(1)).mean()

        one_hot = torch.nn.functional.one_hot(old_action[:, 0].long(), self.label_size).float()
        new_prob = (one_hot * prob_dist).sum(1, keepdim=True)
        old_action_prob = (one_hot * old_prob).sum(1, keepdim=True)

        ratio = new_prob / old_action_prob.clamp(1e-9)

        adv = advantage.squeeze(-1) if advantage.dim() > 1 else advantage
        adv = adv.unsqueeze(-1)

        policy_loss = torch.maximum(
            -ratio * adv,
            -ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * adv,
        ).mean()

        # Total loss
        # 总损失
        total_loss = self.vf_coef * value_loss + policy_loss - self.var_beta * entropy_loss

        return total_loss, {
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
        }

    def _masked_softmax(self, logits, legal_action):
        """Apply legal action mask to logits before computing softmax.

        对 logits 应用合法动作掩码后计算 softmax。
        """
        label_max, _ = torch.max(logits * legal_action, dim=1, keepdim=True)
        logits = logits - label_max
        logits = logits * legal_action
        logits = logits + 1e5 * (legal_action - 1)
        return torch.nn.functional.softmax(logits, dim=1)
