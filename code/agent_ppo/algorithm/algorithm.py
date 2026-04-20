#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Sequence-aware PPO algorithm for LTSPPO.
"""

import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from agent_ppo.conf.conf import Config
from agent_ppo.utils.experiment_archive import ExperimentArchive


class Algorithm:
    SAMPLE_FIELD_ORDER = (
        "obs",
        "legal_action",
        "act",
        "reward_clean",
        "reward_survive",
        "done",
        "value_clean",
        "value_survive",
        "advantage_clean",
        "advantage_survive",
        "prob",
        "mode_teacher",
        "route_anchor_teacher",
        "target_teacher",
        "mode_teacher_mask",
        "route_anchor_teacher_mask",
        "target_teacher_mask",
        "return_action_teacher",
        "return_action_teacher_mask",
        "battery_risk_label",
        "collision_risk_label",
        "constraint_battery_process_cost",
        "fallback_mask",
        "expert_weight",
    )

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
        self.lambda_battery = Config.LAMBDA_BATTERY_INIT
        self.lambda_collision = Config.LAMBDA_COLLISION_INIT

        self.train_step = 0
        self.last_report_time = 0
        self.total_batch_count = 0
        self.nan_batch_count = 0
        self.consecutive_invalid_batches = 0
        self.last_finite_step = 0
        self.last_finite_metrics = {}

    @staticmethod
    def _tensor_summary(tensor):
        if tensor is None:
            return {"shape": None, "min": None, "max": None, "mean": None}
        detached = tensor.detach()
        if detached.numel() == 0:
            return {"shape": list(detached.shape), "min": None, "max": None, "mean": None}
        finite = detached[torch.isfinite(detached)]
        if finite.numel() == 0:
            return {"shape": list(detached.shape), "min": None, "max": None, "mean": None}
        finite = finite.to(dtype=torch.float32)
        return {
            "shape": list(detached.shape),
            "min": float(finite.min().item()),
            "max": float(finite.max().item()),
            "mean": float(finite.mean().item()),
        }

    def _first_nonfinite_tensor(self, mapping):
        for name, value in mapping.items():
            if value is None:
                continue
            if isinstance(value, dict):
                nested = self._first_nonfinite_tensor({f"{name}.{k}": v for k, v in value.items()})
                if nested is not None:
                    return nested
                continue
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value, device=self.device)
            if not torch.isfinite(tensor).all():
                return name, self._tensor_summary(tensor)
        return None

    def _gradients_are_finite(self):
        for param in self.parameters:
            if param.grad is None:
                continue
            if not torch.isfinite(param.grad).all():
                return False
        return True

    def _normalize_advantage(self, advantage, valid_mask, valid_count):
        adv_mean = (advantage * valid_mask).sum() / valid_count
        centered = advantage - adv_mean
        adv_var = ((centered ** 2) * valid_mask).sum() / valid_count
        adv_std = torch.sqrt(adv_var + Config.ADV_NORM_EPS)
        if float(adv_std.detach().item()) < Config.ADV_STD_MIN:
            return centered
        return centered / adv_std

    def _record_invalid_batch(self, reason, stats):
        self.nan_batch_count += 1
        self.consecutive_invalid_batches += 1
        self.optimizer.zero_grad(set_to_none=True)
        payload = {
            "reason": reason,
            "stats": stats,
            "nan_batch_count": self.nan_batch_count,
            "nan_skip_rate": self.nan_batch_count / max(self.total_batch_count, 1),
            "last_finite_step": self.last_finite_step,
            "consecutive_invalid_batches": self.consecutive_invalid_batches,
        }
        if self.logger:
            self.logger.warning(
                f"loss_invalid=1 reason={reason} stats={stats} "
                f"nan_batch_count={self.nan_batch_count} "
                f"nan_skip_rate={payload['nan_skip_rate']:.4f} "
                f"last_finite_step={self.last_finite_step} "
                f"consecutive_invalid_batches={self.consecutive_invalid_batches}"
            )
        if self.monitor:
            self.monitor.put_data(
                {
                    os.getpid(): {
                        "total_loss": float(self.last_finite_metrics.get("total_loss", 0.0)),
                        "loss_invalid": 1.0,
                        "nan_batch_count": float(self.nan_batch_count),
                        "nan_skip_rate": float(payload["nan_skip_rate"]),
                        "last_finite_step": float(self.last_finite_step),
                    }
                }
            )
        if self.consecutive_invalid_batches == Config.INVALID_BATCH_CONSECUTIVE_LIMIT:
            self.archive.log_event(
                "learner_invalid_batch_limit",
                {
                    "train_step": self.train_step,
                    "nan_batch_count": self.nan_batch_count,
                    "consecutive_invalid_batches": self.consecutive_invalid_batches,
                    "reason": reason,
                    "stats": stats,
                },
            )
        return {
            "total_loss": float(self.last_finite_metrics.get("total_loss", 0.0)),
            "loss_invalid": 1.0,
            "nan_batch_count": float(self.nan_batch_count),
            "nan_skip_rate": float(payload["nan_skip_rate"]),
            "last_finite_step": float(self.last_finite_step),
        }

    def _finalize_invalid_after_unscale(self, reason, stats):
        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.update()
        return self._record_invalid_batch(reason, stats)

    def _update_constraint_multipliers(self, battery_cost_mean, collision_cost_mean):
        if float(battery_cost_mean) > Config.BATTERY_PROCESS_COST_TARGET:
            self.lambda_battery = min(self.lambda_battery + Config.LAMBDA_BATTERY_UP, Config.LAMBDA_BATTERY_MAX)
        elif float(battery_cost_mean) < Config.BATTERY_PROCESS_COST_TARGET * 0.8:
            self.lambda_battery = max(self.lambda_battery - Config.LAMBDA_BATTERY_DOWN, Config.LAMBDA_BATTERY_MIN)

        if float(collision_cost_mean) > Config.COLLISION_PROCESS_COST_TARGET:
            self.lambda_collision = min(self.lambda_collision + Config.LAMBDA_COLLISION_UP, Config.LAMBDA_COLLISION_MAX)
        elif float(collision_cost_mean) < Config.COLLISION_PROCESS_COST_TARGET * 0.8:
            self.lambda_collision = max(self.lambda_collision - Config.LAMBDA_COLLISION_DOWN, Config.LAMBDA_COLLISION_MIN)

    def learn(self, list_sample_data):
        self.total_batch_count += 1
        batch = self._unpack_train_batch(list_sample_data)
        invalid = self._first_nonfinite_tensor(
            {
                "reward_clean": batch["reward_clean"],
                "reward_survive": batch["reward_survive"],
                "value_clean": batch["value_clean"],
                "value_survive": batch["value_survive"],
                "advantage_clean": batch["advantage_clean"],
                "advantage_survive": batch["advantage_survive"],
            }
        )
        if invalid is not None:
            name, stats = invalid
            return self._record_invalid_batch(f"batch:{name}", stats)

        self.model.set_train_mode()
        self.optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=self.use_amp):
            outputs = self.model(batch["obs"])
            invalid = self._first_nonfinite_tensor(
                {
                    "policy_logits": outputs.get("policy_logits"),
                    "mode_logits": outputs.get("mode_logits"),
                    "route_anchor_logits": outputs.get("route_anchor_logits"),
                    "target_logits": outputs.get("target_logits"),
                    "return_action_logits": outputs.get("return_action_logits"),
                    "value_clean_pred": outputs.get("value_clean"),
                    "value_survive_pred": outputs.get("value_survive"),
                }
            )
            if invalid is not None:
                name, stats = invalid
                return self._record_invalid_batch(f"model:{name}", stats)
            total_loss, info, loss_tensors = self._compute_loss(outputs, batch)
            invalid = self._first_nonfinite_tensor(loss_tensors)
            if invalid is not None:
                name, stats = invalid
                return self._record_invalid_batch(f"loss:{name}", stats)

        self.scaler.scale(total_loss).backward()
        self.scaler.unscale_(self.optimizer)
        if not self._gradients_are_finite():
            return self._finalize_invalid_after_unscale(
                "gradients",
                {"shape": None, "min": None, "max": None, "mean": None},
            )
        if Config.USE_GRAD_CLIP:
            torch.nn.utils.clip_grad_norm_(self.parameters, Config.GRAD_CLIP_RANGE)
            if not self._gradients_are_finite():
                return self._finalize_invalid_after_unscale(
                    "gradients_post_clip",
                    {"shape": None, "min": None, "max": None, "mean": None},
                )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.train_step += 1
        self.consecutive_invalid_batches = 0
        self.last_finite_step = self.train_step

        results = {"total_loss": float(total_loss.item())}
        self.last_finite_metrics = {
            "total_loss": results["total_loss"],
            "policy_loss": float(info["policy_loss"]),
            "value_clean_loss": float(info["value_clean_loss"]),
            "value_survive_loss": float(info["value_survive_loss"]),
            "entropy_loss": float(info["entropy_loss"]),
        }
        now = time.time()
        if now - self.last_report_time >= 60:
            results.update(
                {
                    "policy_loss": round(info["policy_loss"], 4),
                    "value_clean_loss": round(info["value_clean_loss"], 4),
                    "value_survive_loss": round(info["value_survive_loss"], 4),
                    "entropy_loss": round(info["entropy_loss"], 4),
                    "mode_teacher_loss": round(info["mode_teacher_loss"], 4),
                    "route_anchor_teacher_loss": round(info["route_anchor_teacher_loss"], 4),
                    "target_teacher_loss": round(info["target_teacher_loss"], 4),
                    "return_action_teacher_loss": round(info["return_action_teacher_loss"], 4),
                    "mode_teacher_active_rate": round(info["mode_teacher_active_rate"], 4),
                    "route_anchor_teacher_active_rate": round(info["route_anchor_teacher_active_rate"], 4),
                    "target_teacher_active_rate": round(info["target_teacher_active_rate"], 4),
                    "return_action_teacher_active_rate": round(info["return_action_teacher_active_rate"], 4),
                    "aux_battery_loss": round(info["aux_battery_loss"], 4),
                    "aux_collision_loss": round(info["aux_collision_loss"], 4),
                    "battery_process_cost_mean": round(info["battery_process_cost_mean"], 4),
                    "collision_process_cost_mean": round(info["collision_process_cost_mean"], 4),
                    "lambda_battery": round(info["lambda_battery"], 4),
                    "lambda_collision": round(info["lambda_collision"], 4),
                    "loss_invalid": 0.0,
                    "nan_batch_count": float(self.nan_batch_count),
                    "nan_skip_rate": float(self.nan_batch_count / max(self.total_batch_count, 1)),
                    "last_finite_step": float(self.last_finite_step),
                    "train_step": self.train_step,
                }
            )

            if self.logger:
                self.logger.info(
                    "policy_loss: %.4f, value_clean_loss: %.4f, value_survive_loss: %.4f, "
                    "entropy_loss: %.4f, mode_teacher_loss: %.4f, route_anchor_teacher_loss: %.4f, "
                    "target_teacher_loss: %.4f, return_action_teacher_loss: %.4f, "
                    "mode_teacher_active_rate: %.4f, route_anchor_teacher_active_rate: %.4f, "
                    "target_teacher_active_rate: %.4f, return_action_teacher_active_rate: %.4f, "
                    "battery_process_cost_mean: %.4f, collision_process_cost_mean: %.4f, "
                    "lambda_battery: %.4f, lambda_collision: %.4f, "
                    "loss_invalid: %.0f, nan_batch_count: %.0f, nan_skip_rate: %.4f, last_finite_step: %.0f"
                    % (
                        results["policy_loss"],
                        results["value_clean_loss"],
                        results["value_survive_loss"],
                        results["entropy_loss"],
                        results["mode_teacher_loss"],
                        results["route_anchor_teacher_loss"],
                        results["target_teacher_loss"],
                        results["return_action_teacher_loss"],
                        results["mode_teacher_active_rate"],
                        results["route_anchor_teacher_active_rate"],
                        results["target_teacher_active_rate"],
                        results["return_action_teacher_active_rate"],
                        results["battery_process_cost_mean"],
                        results["collision_process_cost_mean"],
                        results["lambda_battery"],
                        results["lambda_collision"],
                        results["loss_invalid"],
                        results["nan_batch_count"],
                        results["nan_skip_rate"],
                        results["last_finite_step"],
                    )
                )

            self.archive.log_train_window(
                {
                    "record_type": "algorithm_window",
                    "train_step": self.train_step,
                    "total_loss": results["total_loss"],
                    "policy_loss": results["policy_loss"],
                    "value_clean_loss": results["value_clean_loss"],
                    "value_survive_loss": results["value_survive_loss"],
                    "entropy_loss": results["entropy_loss"],
                    "mode_teacher_loss": results["mode_teacher_loss"],
                    "route_anchor_teacher_loss": results["route_anchor_teacher_loss"],
                    "target_teacher_loss": results["target_teacher_loss"],
                    "return_action_teacher_loss": results["return_action_teacher_loss"],
                    "mode_teacher_active_rate": results["mode_teacher_active_rate"],
                    "route_anchor_teacher_active_rate": results["route_anchor_teacher_active_rate"],
                    "target_teacher_active_rate": results["target_teacher_active_rate"],
                    "return_action_teacher_active_rate": results["return_action_teacher_active_rate"],
                    "aux_battery_loss": results["aux_battery_loss"],
                    "aux_collision_loss": results["aux_collision_loss"],
                    "battery_process_cost_mean": results["battery_process_cost_mean"],
                    "collision_process_cost_mean": results["collision_process_cost_mean"],
                    "lambda_battery": results["lambda_battery"],
                    "lambda_collision": results["lambda_collision"],
                    "loss_invalid": results["loss_invalid"],
                    "nan_batch_count": results["nan_batch_count"],
                    "nan_skip_rate": results["nan_skip_rate"],
                    "last_finite_step": results["last_finite_step"],
                }
            )
            if self.monitor:
                self.monitor.put_data({os.getpid(): results})
            self.last_report_time = now

        return results

    def _unpack_train_batch(self, list_sample_data):
        if isinstance(list_sample_data, (torch.Tensor, np.ndarray)):
            return self._unpack_flat_batch_tensor(list_sample_data)

        if isinstance(list_sample_data, (list, tuple)) and list_sample_data:
            first = list_sample_data[0]
            if isinstance(first, (torch.Tensor, np.ndarray)):
                if len(list_sample_data) == len(self.SAMPLE_FIELD_ORDER):
                    field_map = {
                        key: torch.as_tensor(value)
                        for key, value in zip(self.SAMPLE_FIELD_ORDER, list_sample_data)
                    }
                    return self._build_batch_from_field_map(field_map)

                stacked = torch.stack([torch.as_tensor(v) for v in list_sample_data], dim=0)
                return self._unpack_flat_batch_tensor(stacked)

        to_device = {"device": self.device}
        if self.use_amp:
            to_device["non_blocking"] = True

        obs = torch.stack([torch.as_tensor(s.obs, dtype=torch.float32) for s in list_sample_data]).to(**to_device)
        legal_action = torch.stack(
            [torch.as_tensor(s.legal_action, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        act = torch.stack([torch.as_tensor(s.act, dtype=torch.long) for s in list_sample_data]).to(**to_device)
        prob = torch.stack([torch.as_tensor(s.prob, dtype=torch.float32) for s in list_sample_data]).to(**to_device)
        reward_clean = torch.stack(
            [torch.as_tensor(s.reward_clean, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        reward_survive = torch.stack(
            [torch.as_tensor(s.reward_survive, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        done = torch.stack([torch.as_tensor(s.done, dtype=torch.float32) for s in list_sample_data]).to(**to_device)
        value_clean = torch.stack(
            [torch.as_tensor(s.value_clean, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        value_survive = torch.stack(
            [torch.as_tensor(s.value_survive, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        advantage_clean = torch.stack(
            [torch.as_tensor(s.advantage_clean, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        advantage_survive = torch.stack(
            [torch.as_tensor(s.advantage_survive, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        mode_teacher = torch.stack(
            [torch.as_tensor(s.mode_teacher, dtype=torch.long) for s in list_sample_data]
        ).to(**to_device)
        route_anchor_teacher = torch.stack(
            [torch.as_tensor(s.route_anchor_teacher, dtype=torch.long) for s in list_sample_data]
        ).to(**to_device)
        target_teacher = torch.stack(
            [torch.as_tensor(s.target_teacher, dtype=torch.long) for s in list_sample_data]
        ).to(**to_device)
        mode_teacher_mask = torch.stack(
            [torch.as_tensor(s.mode_teacher_mask, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        route_anchor_teacher_mask = torch.stack(
            [torch.as_tensor(s.route_anchor_teacher_mask, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        target_teacher_mask = torch.stack(
            [torch.as_tensor(s.target_teacher_mask, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        return_action_teacher = torch.stack(
            [torch.as_tensor(s.return_action_teacher, dtype=torch.long) for s in list_sample_data]
        ).to(**to_device)
        return_action_teacher_mask = torch.stack(
            [torch.as_tensor(s.return_action_teacher_mask, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        battery_risk_label = torch.stack(
            [torch.as_tensor(s.battery_risk_label, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        collision_risk_label = torch.stack(
            [torch.as_tensor(s.collision_risk_label, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        constraint_battery_process_cost = torch.stack(
            [torch.as_tensor(s.constraint_battery_process_cost, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        fallback_mask = torch.stack(
            [torch.as_tensor(s.fallback_mask, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)
        expert_weight = torch.stack(
            [torch.as_tensor(s.expert_weight, dtype=torch.float32) for s in list_sample_data]
        ).to(**to_device)

        return self._build_batch_from_field_map(
            {
                "obs": obs,
                "legal_action": legal_action,
                "act": act,
                "prob": prob,
                "reward_clean": reward_clean,
                "reward_survive": reward_survive,
                "done": done,
                "value_clean": value_clean,
                "value_survive": value_survive,
                "advantage_clean": advantage_clean,
                "advantage_survive": advantage_survive,
                "mode_teacher": mode_teacher,
                "route_anchor_teacher": route_anchor_teacher,
                "target_teacher": target_teacher,
                "mode_teacher_mask": mode_teacher_mask,
                "route_anchor_teacher_mask": route_anchor_teacher_mask,
                "target_teacher_mask": target_teacher_mask,
                "return_action_teacher": return_action_teacher,
                "return_action_teacher_mask": return_action_teacher_mask,
                "battery_risk_label": battery_risk_label,
                "collision_risk_label": collision_risk_label,
                "constraint_battery_process_cost": constraint_battery_process_cost,
                "fallback_mask": fallback_mask,
                "expert_weight": expert_weight,
            }
        )

    def _unpack_flat_batch_tensor(self, batch_data):
        batch_tensor = torch.as_tensor(batch_data)
        if batch_tensor.dim() == 1:
            batch_tensor = batch_tensor.unsqueeze(0)

        batch_tensor = batch_tensor.to(dtype=torch.float32, device=self.device)
        begin = 0

        field_map = {}
        field_map["obs"] = batch_tensor[:, begin : begin + Config.SAMPLE_OBS_DIM]
        begin += Config.SAMPLE_OBS_DIM

        field_map["legal_action"] = batch_tensor[:, begin : begin + Config.SAMPLE_LEGAL_ACTION_DIM]
        begin += Config.SAMPLE_LEGAL_ACTION_DIM

        field_map["act"] = batch_tensor[:, begin : begin + Config.SAMPLE_ACTION_DIM].to(dtype=torch.long)
        begin += Config.SAMPLE_ACTION_DIM

        field_map["reward_clean"] = batch_tensor[:, begin : begin + Config.SAMPLE_REWARD_DIM]
        begin += Config.SAMPLE_REWARD_DIM

        field_map["reward_survive"] = batch_tensor[:, begin : begin + Config.SAMPLE_REWARD_DIM]
        begin += Config.SAMPLE_REWARD_DIM

        field_map["done"] = batch_tensor[:, begin : begin + Config.SAMPLE_DONE_DIM]
        begin += Config.SAMPLE_DONE_DIM

        field_map["value_clean"] = batch_tensor[:, begin : begin + Config.SAMPLE_VALUE_DIM]
        begin += Config.SAMPLE_VALUE_DIM

        field_map["value_survive"] = batch_tensor[:, begin : begin + Config.SAMPLE_VALUE_DIM]
        begin += Config.SAMPLE_VALUE_DIM

        field_map["advantage_clean"] = batch_tensor[:, begin : begin + Config.SAMPLE_VALUE_DIM]
        begin += Config.SAMPLE_VALUE_DIM

        field_map["advantage_survive"] = batch_tensor[:, begin : begin + Config.SAMPLE_VALUE_DIM]
        begin += Config.SAMPLE_VALUE_DIM

        field_map["prob"] = batch_tensor[:, begin : begin + Config.SAMPLE_PROB_DIM]
        begin += Config.SAMPLE_PROB_DIM

        field_map["mode_teacher"] = batch_tensor[:, begin : begin + Config.SAMPLE_MODE_DIM].to(dtype=torch.long)
        begin += Config.SAMPLE_MODE_DIM

        field_map["route_anchor_teacher"] = batch_tensor[:, begin : begin + Config.SAMPLE_ROUTE_ANCHOR_DIM].to(dtype=torch.long)
        begin += Config.SAMPLE_ROUTE_ANCHOR_DIM

        field_map["target_teacher"] = batch_tensor[:, begin : begin + Config.SAMPLE_TARGET_DIM].to(dtype=torch.long)
        begin += Config.SAMPLE_TARGET_DIM

        field_map["mode_teacher_mask"] = batch_tensor[:, begin : begin + Config.SAMPLE_MODE_TEACHER_MASK_DIM]
        begin += Config.SAMPLE_MODE_TEACHER_MASK_DIM

        field_map["route_anchor_teacher_mask"] = batch_tensor[:, begin : begin + Config.SAMPLE_ROUTE_ANCHOR_MASK_DIM]
        begin += Config.SAMPLE_ROUTE_ANCHOR_MASK_DIM

        field_map["target_teacher_mask"] = batch_tensor[:, begin : begin + Config.SAMPLE_TARGET_TEACHER_MASK_DIM]
        begin += Config.SAMPLE_TARGET_TEACHER_MASK_DIM

        field_map["return_action_teacher"] = batch_tensor[:, begin : begin + Config.SAMPLE_RETURN_ACTION_DIM].to(dtype=torch.long)
        begin += Config.SAMPLE_RETURN_ACTION_DIM

        field_map["return_action_teacher_mask"] = batch_tensor[:, begin : begin + Config.SAMPLE_RETURN_ACTION_MASK_DIM]
        begin += Config.SAMPLE_RETURN_ACTION_MASK_DIM

        field_map["battery_risk_label"] = batch_tensor[:, begin : begin + Config.SAMPLE_AUX_LABEL_DIM]
        begin += Config.SAMPLE_AUX_LABEL_DIM

        field_map["collision_risk_label"] = batch_tensor[:, begin : begin + Config.SAMPLE_AUX_LABEL_DIM]
        begin += Config.SAMPLE_AUX_LABEL_DIM

        field_map["constraint_battery_process_cost"] = batch_tensor[:, begin : begin + Config.SAMPLE_CONSTRAINT_COST_DIM]
        begin += Config.SAMPLE_CONSTRAINT_COST_DIM

        field_map["fallback_mask"] = batch_tensor[:, begin : begin + Config.FALLBACK_MASK_DIM]
        begin += Config.FALLBACK_MASK_DIM

        field_map["expert_weight"] = batch_tensor[:, begin : begin + Config.EXPERT_WEIGHT_DIM]

        return self._build_batch_from_field_map(field_map)

    def _build_batch_from_field_map(self, field_map):
        to_device = {"device": self.device}
        if self.use_amp:
            to_device["non_blocking"] = True

        def _reshape_optional_sequence(tensor, batch_size, seq_len):
            if tensor.numel() == batch_size * seq_len:
                return tensor.view(batch_size, seq_len)
            if tensor.numel() % max(batch_size, 1) == 0:
                return tensor.view(batch_size, tensor.numel() // max(batch_size, 1))
            return tensor.view(batch_size, -1)

        obs = torch.as_tensor(field_map["obs"], dtype=torch.float32).to(**to_device)
        legal_action = torch.as_tensor(field_map["legal_action"], dtype=torch.float32).to(**to_device)
        act = torch.as_tensor(field_map["act"], dtype=torch.long).to(**to_device)
        prob = torch.as_tensor(field_map["prob"], dtype=torch.float32).to(**to_device)
        reward_clean = torch.as_tensor(field_map["reward_clean"], dtype=torch.float32).to(**to_device)
        reward_survive = torch.as_tensor(field_map["reward_survive"], dtype=torch.float32).to(**to_device)
        done = torch.as_tensor(field_map["done"], dtype=torch.float32).to(**to_device)
        value_clean = torch.as_tensor(field_map["value_clean"], dtype=torch.float32).to(**to_device)
        value_survive = torch.as_tensor(field_map["value_survive"], dtype=torch.float32).to(**to_device)
        advantage_clean = torch.as_tensor(field_map["advantage_clean"], dtype=torch.float32).to(**to_device)
        advantage_survive = torch.as_tensor(field_map["advantage_survive"], dtype=torch.float32).to(**to_device)
        mode_teacher = torch.as_tensor(field_map["mode_teacher"], dtype=torch.long).to(**to_device)
        route_anchor_teacher = torch.as_tensor(field_map["route_anchor_teacher"], dtype=torch.long).to(**to_device)
        target_teacher = torch.as_tensor(field_map["target_teacher"], dtype=torch.long).to(**to_device)
        mode_teacher_mask = torch.as_tensor(field_map["mode_teacher_mask"], dtype=torch.float32).to(**to_device)
        route_anchor_teacher_mask = torch.as_tensor(field_map["route_anchor_teacher_mask"], dtype=torch.float32).to(**to_device)
        target_teacher_mask = torch.as_tensor(field_map["target_teacher_mask"], dtype=torch.float32).to(**to_device)
        return_action_teacher = torch.as_tensor(field_map["return_action_teacher"], dtype=torch.long).to(**to_device)
        return_action_teacher_mask = torch.as_tensor(field_map["return_action_teacher_mask"], dtype=torch.float32).to(**to_device)
        battery_risk_label = torch.as_tensor(field_map["battery_risk_label"], dtype=torch.float32).to(**to_device)
        collision_risk_label = torch.as_tensor(field_map["collision_risk_label"], dtype=torch.float32).to(**to_device)
        constraint_battery_process_cost = torch.as_tensor(
            field_map["constraint_battery_process_cost"], dtype=torch.float32
        ).to(**to_device)
        fallback_mask = torch.as_tensor(field_map["fallback_mask"], dtype=torch.float32).to(**to_device)
        expert_weight = torch.as_tensor(field_map["expert_weight"], dtype=torch.float32).to(**to_device)

        batch_size = obs.shape[0]
        seq_len = Config.SEQ_CHUNK_LEN
        return {
            "obs": obs.view(batch_size, seq_len, Config.DIM_OF_OBSERVATION),
            "legal_action": legal_action.view(batch_size, seq_len, Config.ACTION_NUM),
            "act": act.view(batch_size, seq_len),
            "prob": prob.view(batch_size, seq_len, Config.ACTION_NUM),
            "reward_clean": reward_clean.view(batch_size, seq_len),
            "reward_survive": reward_survive.view(batch_size, seq_len),
            "done": done.view(batch_size, seq_len),
            "value_clean": value_clean.view(batch_size, seq_len),
            "value_survive": value_survive.view(batch_size, seq_len),
            "advantage_clean": advantage_clean.view(batch_size, seq_len),
            "advantage_survive": advantage_survive.view(batch_size, seq_len),
            "mode_teacher": mode_teacher.view(batch_size, seq_len),
            "route_anchor_teacher": route_anchor_teacher.view(batch_size, seq_len),
            "target_teacher": target_teacher.view(batch_size, seq_len),
            "mode_teacher_mask": mode_teacher_mask.view(batch_size, seq_len),
            "route_anchor_teacher_mask": route_anchor_teacher_mask.view(batch_size, seq_len),
            "target_teacher_mask": target_teacher_mask.view(batch_size, seq_len),
            "return_action_teacher": return_action_teacher.view(batch_size, seq_len),
            "return_action_teacher_mask": return_action_teacher_mask.view(batch_size, seq_len),
            "battery_risk_label": battery_risk_label.view(batch_size, seq_len),
            "collision_risk_label": collision_risk_label.view(batch_size, seq_len),
            "constraint_battery_process_cost": constraint_battery_process_cost.view(batch_size, seq_len),
            "fallback_mask": fallback_mask.view(batch_size, seq_len),
            "expert_weight": _reshape_optional_sequence(expert_weight, batch_size, seq_len),
        }

    @staticmethod
    def _battery_process_cost_mean(batch, learn_slice, valid_mask, valid_count):
        return (batch["constraint_battery_process_cost"][:, learn_slice] * valid_mask).sum() / valid_count

    def _compute_loss(self, outputs, batch):
        learn_slice = slice(Config.SEQ_BURN_IN, Config.SEQ_CHUNK_LEN)

        legal = batch["legal_action"][:, learn_slice, :]
        action = batch["act"][:, learn_slice]
        old_prob = batch["prob"][:, learn_slice, :]
        old_value_clean = batch["value_clean"][:, learn_slice]
        old_value_survive = batch["value_survive"][:, learn_slice]
        adv_clean = batch["advantage_clean"][:, learn_slice]
        adv_survive = batch["advantage_survive"][:, learn_slice]
        mode_teacher_mask = batch["mode_teacher_mask"][:, learn_slice]
        route_anchor_teacher_mask = batch["route_anchor_teacher_mask"][:, learn_slice]
        target_teacher_mask = batch["target_teacher_mask"][:, learn_slice]
        return_action_teacher_mask = batch["return_action_teacher_mask"][:, learn_slice]
        battery_label = batch["battery_risk_label"][:, learn_slice]
        collision_label = batch["collision_risk_label"][:, learn_slice]
        fallback_mask = batch["fallback_mask"][:, learn_slice]

        logits = outputs["policy_logits"][:, learn_slice, :]
        mode_logits = outputs["mode_logits"][:, learn_slice, :]
        route_anchor_logits = outputs["route_anchor_logits"][:, learn_slice, :]
        target_logits = outputs["target_logits"][:, learn_slice, :]
        return_action_logits = outputs["return_action_logits"][:, learn_slice, :]
        value_clean_pred = outputs["value_clean"][:, learn_slice, 0]
        value_survive_pred = outputs["value_survive"][:, learn_slice, 0]
        aux_battery = outputs["aux_battery_risk"][:, learn_slice, 0]
        aux_collision = outputs["aux_collision_risk"][:, learn_slice, 0]

        valid_mask = (1.0 - fallback_mask).to(torch.float32)
        valid_count = valid_mask.sum().clamp_min(1.0)

        adv_clean = adv_clean.clamp(-Config.ADV_CLEAN_CLIP, Config.ADV_CLEAN_CLIP)
        adv_survive = adv_survive.clamp(-Config.ADV_SURVIVE_CLIP, Config.ADV_SURVIVE_CLIP)
        returns_clean = (adv_clean + old_value_clean).clamp(-Config.RETURNS_CLEAN_CLIP, Config.RETURNS_CLEAN_CLIP)
        returns_survive = (adv_survive + old_value_survive).clamp(
            -Config.RETURNS_SURVIVE_CLIP,
            Config.RETURNS_SURVIVE_CLIP,
        )

        vp_clean_clip = old_value_clean + (value_clean_pred - old_value_clean).clamp(-self.clip_param, self.clip_param)
        vp_survive_clip = old_value_survive + (value_survive_pred - old_value_survive).clamp(
            -self.clip_param, self.clip_param
        )

        value_clean_loss = 0.5 * torch.maximum(
            (returns_clean - value_clean_pred) ** 2,
            (returns_clean - vp_clean_clip) ** 2,
        )
        value_survive_loss = 0.5 * torch.maximum(
            (returns_survive - value_survive_pred) ** 2,
            (returns_survive - vp_survive_clip) ** 2,
        )
        value_clean_loss = (value_clean_loss * valid_mask).sum() / valid_count
        value_survive_loss = (value_survive_loss * valid_mask).sum() / valid_count

        prob_dist = self._masked_softmax(logits.reshape(-1, Config.ACTION_NUM), legal.reshape(-1, Config.ACTION_NUM))
        prob_dist = prob_dist.view_as(logits)
        entropy = -(prob_dist * torch.log(prob_dist.clamp(1e-9, 1.0))).sum(dim=-1)
        entropy_loss = (entropy * valid_mask).sum() / valid_count

        chosen_idx = action.unsqueeze(-1)
        new_action_prob = prob_dist.gather(-1, chosen_idx).squeeze(-1)
        old_action_prob = old_prob.gather(-1, chosen_idx).squeeze(-1).clamp_min(1e-9)
        ratio = new_action_prob / old_action_prob

        adv_task = self._normalize_advantage(adv_clean, valid_mask, valid_count)
        adv_battery = self._normalize_advantage(adv_survive, valid_mask, valid_count)

        adv_actor = adv_task - self.lambda_battery * adv_battery
        adv_actor = self._normalize_advantage(adv_actor, valid_mask, valid_count)

        policy_loss = torch.maximum(
            -ratio * adv_actor,
            -ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param) * adv_actor,
        )
        policy_loss = (policy_loss * valid_mask).sum() / valid_count

        mode_teacher_active = mode_teacher_mask * valid_mask
        route_anchor_teacher_active = route_anchor_teacher_mask * valid_mask
        target_teacher_active = target_teacher_mask * valid_mask
        return_action_teacher_active = return_action_teacher_mask * valid_mask
        mode_teacher_raw_count = mode_teacher_active.sum()
        route_anchor_teacher_raw_count = route_anchor_teacher_active.sum()
        target_teacher_raw_count = target_teacher_active.sum()
        return_action_teacher_raw_count = return_action_teacher_active.sum()
        mode_teacher_count = mode_teacher_raw_count.clamp_min(1.0)
        route_anchor_teacher_count = route_anchor_teacher_raw_count.clamp_min(1.0)
        target_teacher_count = target_teacher_raw_count.clamp_min(1.0)
        return_action_teacher_count = return_action_teacher_raw_count.clamp_min(1.0)
        mode_teacher_active_rate = mode_teacher_raw_count / valid_count
        route_anchor_teacher_active_rate = route_anchor_teacher_raw_count / valid_count
        target_teacher_active_rate = target_teacher_raw_count / valid_count
        return_action_teacher_active_rate = return_action_teacher_raw_count / valid_count
        mode_teacher_loss = F.cross_entropy(
            mode_logits.reshape(-1, Config.MODE_NUM),
            batch["mode_teacher"][:, learn_slice].reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).view_as(mode_teacher_active)
        route_anchor_teacher_loss = F.cross_entropy(
            route_anchor_logits.reshape(-1, Config.ROUTE_ANCHOR_DIM),
            batch["route_anchor_teacher"][:, learn_slice].reshape(-1),
            reduction="none",
        ).view_as(route_anchor_teacher_active)
        target_teacher_loss = F.cross_entropy(
            target_logits.reshape(-1, Config.TARGET_DIM),
            batch["target_teacher"][:, learn_slice].reshape(-1),
            reduction="none",
        ).view_as(target_teacher_active)
        return_action_teacher_loss = F.cross_entropy(
            return_action_logits.reshape(-1, Config.ACTION_NUM),
            batch["return_action_teacher"][:, learn_slice].reshape(-1),
            ignore_index=-1,
            reduction="none",
        ).view_as(return_action_teacher_active)
        mode_teacher_loss = (mode_teacher_loss * mode_teacher_active).sum() / mode_teacher_count
        route_anchor_teacher_loss = (
            route_anchor_teacher_loss * route_anchor_teacher_active
        ).sum() / route_anchor_teacher_count
        target_teacher_loss = (target_teacher_loss * target_teacher_active).sum() / target_teacher_count
        return_action_teacher_loss = (
            return_action_teacher_loss * return_action_teacher_active
        ).sum() / return_action_teacher_count

        aux_battery_loss = F.binary_cross_entropy_with_logits(aux_battery, battery_label, reduction="none")
        aux_collision_loss = F.binary_cross_entropy_with_logits(aux_collision, collision_label, reduction="none")
        aux_battery_loss = (aux_battery_loss * valid_mask).sum() / valid_count
        aux_collision_loss = (aux_collision_loss * valid_mask).sum() / valid_count
        battery_process_cost_mean = self._battery_process_cost_mean(batch, learn_slice, valid_mask, valid_count)
        collision_process_cost_mean = (
            collision_label * Config.COLLISION_PROCESS_COST_SCALE * valid_mask
        ).sum() / valid_count

        effective_beta = self.var_beta
        entropy_value = float(entropy_loss.item())
        if entropy_value < Config.ENTROPY_FLOOR:
            floor_gap = Config.ENTROPY_FLOOR - entropy_value
            reference_scale = max(abs(value_clean_loss.item() + value_survive_loss.item()), 1.0)
            target_entropy_bonus = Config.ENTROPY_FLOOR_COEF * reference_scale * floor_gap
            effective_beta = max(self.var_beta, target_entropy_bonus / max(entropy_value, 0.01))

        total_loss = (
            policy_loss
            + self.vf_coef * (value_clean_loss + value_survive_loss)
            - effective_beta * entropy_loss
            + Config.MODE_TEACHER_WEIGHT * mode_teacher_loss
            + Config.ROUTE_ANCHOR_TEACHER_WEIGHT * route_anchor_teacher_loss
            + Config.TARGET_TEACHER_WEIGHT * target_teacher_loss
            + Config.RETURN_ACTION_TEACHER_WEIGHT * return_action_teacher_loss
            + Config.AUX_BATTERY_RISK_WEIGHT * aux_battery_loss
            + self.lambda_collision * aux_collision_loss
        )

        self._update_constraint_multipliers(
            battery_cost_mean=float(battery_process_cost_mean.item()),
            collision_cost_mean=float(collision_process_cost_mean.item()),
        )

        info = {
            "policy_loss": float(policy_loss.item()),
            "value_clean_loss": float(value_clean_loss.item()),
            "value_survive_loss": float(value_survive_loss.item()),
            "entropy_loss": float(entropy_loss.item()),
            "mode_teacher_loss": float(mode_teacher_loss.item()),
            "route_anchor_teacher_loss": float(route_anchor_teacher_loss.item()),
            "target_teacher_loss": float(target_teacher_loss.item()),
            "return_action_teacher_loss": float(return_action_teacher_loss.item()),
            "mode_teacher_active_rate": float(mode_teacher_active_rate.item()),
            "route_anchor_teacher_active_rate": float(route_anchor_teacher_active_rate.item()),
            "target_teacher_active_rate": float(target_teacher_active_rate.item()),
            "return_action_teacher_active_rate": float(return_action_teacher_active_rate.item()),
            "aux_battery_loss": float(aux_battery_loss.item()),
            "aux_collision_loss": float(aux_collision_loss.item()),
            "battery_process_cost_mean": float(battery_process_cost_mean.item()),
            "collision_process_cost_mean": float(collision_process_cost_mean.item()),
            "lambda_battery": float(self.lambda_battery),
            "lambda_collision": float(self.lambda_collision),
        }
        loss_tensors = {
            "returns_clean": returns_clean,
            "returns_survive": returns_survive,
            "policy_loss": policy_loss,
            "value_clean_loss": value_clean_loss,
            "value_survive_loss": value_survive_loss,
            "entropy_loss": entropy_loss,
            "mode_teacher_loss": mode_teacher_loss,
            "route_anchor_teacher_loss": route_anchor_teacher_loss,
            "target_teacher_loss": target_teacher_loss,
            "return_action_teacher_loss": return_action_teacher_loss,
            "aux_battery_loss": aux_battery_loss,
            "aux_collision_loss": aux_collision_loss,
            "total_loss": total_loss,
        }
        return total_loss, info, loss_tensors

    def _masked_softmax(self, logits, legal_action):
        masked = logits - 1e20 * (1.0 - legal_action)
        masked = masked - torch.max(masked, dim=-1, keepdim=True)[0]
        masked = torch.exp(masked).clamp_min(1e-8) * legal_action
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return masked / denom
