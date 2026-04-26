#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Data definitions and chunked recurrent PPO sample processing.
"""

from __future__ import annotations

import numpy as np

try:
    from common_python.utils.common_func import create_cls
except ModuleNotFoundError:
    def create_cls(name, **field_defaults):
        """Fallback lightweight record factory for host-side package imports."""

        field_names = tuple(field_defaults.keys())
        defaults = dict(field_defaults)

        def __init__(self, *args, **kwargs):
            if len(args) > len(field_names):
                raise TypeError(f"{name} expected at most {len(field_names)} positional arguments, got {len(args)}")
            unknown = set(kwargs) - set(field_names)
            if unknown:
                unknown_fields = ", ".join(sorted(unknown))
                raise TypeError(f"{name} got unexpected field(s): {unknown_fields}")

            values = dict(defaults)
            values.update(zip(field_names, args))
            values.update(kwargs)
            for field_name in field_names:
                setattr(self, field_name, values[field_name])

        def __repr__(self):
            rendered = ", ".join(f"{field_name}={getattr(self, field_name)!r}" for field_name in field_names)
            return f"{name}({rendered})"

        namespace = {
            "__init__": __init__,
            "__repr__": __repr__,
            "_fields": field_names,
            "_field_defaults": defaults,
        }
        namespace.update(defaults)
        return type(name, (), namespace)

from agent_ppo.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    policy_prob=None,
    planner_prob=None,
    mix_alpha=None,
    action_mask=None,
    value=None,
    value_clean=None,
    value_survive=None,
    mode=None,
    mode_prob=None,
    route_anchor=None,
    route_anchor_prob=None,
    target=None,
    target_prob=None,
    return_action_prob=None,
    aux_battery_risk=None,
    aux_collision_risk=None,
)


SampleData = create_cls(
    "SampleData",
    obs=Config.SAMPLE_OBS_DIM,
    legal_action=Config.SAMPLE_LEGAL_ACTION_DIM,
    act=Config.SAMPLE_ACTION_DIM,
    reward_clean=Config.SAMPLE_REWARD_DIM,
    reward_survive=Config.SAMPLE_REWARD_DIM,
    done=Config.SAMPLE_DONE_DIM,
    value_clean=Config.SAMPLE_VALUE_DIM,
    value_survive=Config.SAMPLE_VALUE_DIM,
    advantage_clean=Config.SAMPLE_VALUE_DIM,
    advantage_survive=Config.SAMPLE_VALUE_DIM,
    prob=Config.SAMPLE_PROB_DIM,
    planner_prob=Config.SAMPLE_PLANNER_PROB_DIM,
    mix_alpha=Config.SAMPLE_MIX_ALPHA_DIM,
    action_mask=Config.SAMPLE_ACTION_MASK_DIM,
    mode_teacher=Config.SAMPLE_MODE_DIM,
    route_anchor_teacher=Config.SAMPLE_ROUTE_ANCHOR_DIM,
    target_teacher=Config.SAMPLE_TARGET_DIM,
    mode_teacher_mask=Config.SAMPLE_MODE_TEACHER_MASK_DIM,
    route_anchor_teacher_mask=Config.SAMPLE_ROUTE_ANCHOR_MASK_DIM,
    target_teacher_mask=Config.SAMPLE_TARGET_TEACHER_MASK_DIM,
    return_action_teacher=Config.SAMPLE_RETURN_ACTION_DIM,
    return_action_teacher_mask=Config.SAMPLE_RETURN_ACTION_MASK_DIM,
    route_phase_action_teacher=Config.SAMPLE_ROUTE_PHASE_ACTION_DIM,
    route_phase_action_teacher_mask=Config.SAMPLE_ROUTE_PHASE_ACTION_MASK_DIM,
    battery_risk_label=Config.SAMPLE_AUX_LABEL_DIM,
    collision_risk_label=Config.SAMPLE_AUX_LABEL_DIM,
    constraint_battery_process_cost=Config.SAMPLE_CONSTRAINT_COST_DIM,
    fallback_mask=Config.FALLBACK_MASK_DIM,
    expert_weight=Config.EXPERT_WEIGHT_DIM,
)


def _as_array(value, dtype=np.float32):
    return np.asarray(value, dtype=dtype)


def _to_scalar_sequence(step_records, key):
    return np.array([float(rec[key]) for rec in step_records], dtype=np.float32)


def _to_int_sequence(step_records, key):
    return np.array([int(rec[key]) for rec in step_records], dtype=np.int64)


def _pad_seq(seq, target_len, pad_value=0.0, dtype=np.float32):
    arr = np.asarray(seq, dtype=dtype)
    if arr.shape[0] == target_len:
        return arr
    pad_shape = (target_len - arr.shape[0],) + arr.shape[1:]
    pad = np.full(pad_shape, pad_value, dtype=dtype)
    return np.concatenate([arr, pad], axis=0)


def _compute_gae_from_rewards(reward_seq, value_seq, done_seq):
    gae = 0.0
    adv = np.zeros_like(reward_seq, dtype=np.float32)
    ret = np.zeros_like(reward_seq, dtype=np.float32)
    gamma = Config.GAMMA
    lam = Config.LAMDA
    next_value = 0.0
    next_nonterminal = 0.0
    for t in reversed(range(len(reward_seq))):
        nonterminal = 1.0 - float(done_seq[t])
        delta = reward_seq[t] + gamma * next_value * next_nonterminal - value_seq[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        adv[t] = gae
        ret[t] = adv[t] + value_seq[t]
        next_value = value_seq[t]
        next_nonterminal = nonterminal
    return adv.astype(np.float32), ret.astype(np.float32)


def _normalize_action_mask(mask):
    arr = np.asarray(mask, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.ones((Config.ACTION_NUM,), dtype=np.float32)
    arr = np.where(arr > 0.5, 1.0, 0.0).astype(np.float32)
    if float(arr.sum()) <= 0.5:
        return np.ones((Config.ACTION_NUM,), dtype=np.float32)
    return arr


def _uniform_prob(action_mask):
    mask = _normalize_action_mask(action_mask)
    denom = float(mask.sum())
    if denom <= 0.5:
        return np.full((Config.ACTION_NUM,), 1.0 / Config.ACTION_NUM, dtype=np.float32)
    return (mask / denom).astype(np.float32)


def _normalize_prob(prob, action_mask):
    mask = _normalize_action_mask(action_mask)
    arr = np.nan_to_num(np.asarray(prob, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size != Config.ACTION_NUM:
        return _uniform_prob(mask)
    arr = np.clip(arr, 0.0, None) * mask
    total = float(arr.sum())
    if total <= 1e-8:
        return _uniform_prob(mask)
    return (arr / total).astype(np.float32)


def _planner_prior_from_record(record, action_mask):
    direct = record.get("planner_prob")
    if direct is not None:
        return _normalize_prob(direct, action_mask)

    planner_action = -1
    planner_weight = 0.0
    return_mask = float(record.get("return_action_teacher_mask", 0.0))
    if return_mask > 0.0:
        planner_action = int(record.get("return_action_teacher", -1))
        planner_weight = max(planner_weight, 1.0 + return_mask)

    route_mask = float(record.get("route_phase_action_teacher_mask", 0.0))
    if route_mask > 0.0 and planner_action < 0:
        planner_action = int(record.get("route_phase_action_teacher", -1))
        planner_weight = max(planner_weight, 0.75 + route_mask)

    mask = _normalize_action_mask(action_mask)
    if planner_action < 0 or planner_action >= Config.ACTION_NUM or mask[planner_action] <= 0.5:
        return _uniform_prob(mask)

    scores = np.zeros((Config.ACTION_NUM,), dtype=np.float32)
    scores[planner_action] = planner_weight / max(float(Config.PLANNER_PRIOR_TEMPERATURE), 1e-6)
    scores = np.exp(scores - np.max(scores)) * mask
    return _normalize_prob(scores, mask)


def _default_mix_alpha():
    alpha = float(
        np.clip(
            min(max(Config.RESIDUAL_ALPHA_START, Config.RESIDUAL_ALPHA_WARMUP_TARGET), Config.RESIDUAL_ALPHA_MAX),
            0.0,
            1.0,
        )
    )
    return alpha


def _mix_alpha_from_record(record):
    direct = record.get("mix_alpha")
    if direct is not None:
        arr = np.asarray(direct, dtype=np.float32).reshape(-1)
        if arr.size > 0:
            return float(np.clip(arr[0], 0.0, 1.0))

    alpha = _default_mix_alpha()
    if float(record.get("fallback_mask", 0.0)) > 0.5:
        return float(min(alpha, Config.RESIDUAL_ALPHA_FALLBACK_CAP))

    mode_teacher = int(record.get("mode_teacher", -1))
    mode_teacher_mask = float(record.get("mode_teacher_mask", 0.0))
    return_mask = float(record.get("return_action_teacher_mask", 0.0))
    charge_like_mode = mode_teacher_mask > 0.0 and mode_teacher in (3, 4)
    if return_mask > 0.0 or charge_like_mode:
        return float(min(alpha, Config.RESIDUAL_ALPHA_CHARGE_CAP))

    route_mask = float(record.get("route_phase_action_teacher_mask", 0.0))
    if route_mask <= 0.0 and mode_teacher_mask <= 0.0:
        return float(min(alpha, Config.RESIDUAL_ALPHA_FALLBACK_CAP))

    return alpha


def _teacher_scale(episode_idx):
    min_scale = float(getattr(Config, "TEACHER_MIN_SCALE", 0.0))
    if episode_idx <= Config.TEACHER_FORCE_UNTIL_EPISODE:
        return 1.0
    if episode_idx >= Config.TEACHER_ANNEAL_END_EPISODE:
        return min_scale
    span = max(Config.TEACHER_ANNEAL_END_EPISODE - Config.TEACHER_FORCE_UNTIL_EPISODE, 1)
    progress = (episode_idx - Config.TEACHER_FORCE_UNTIL_EPISODE) / span
    return float(max(min_scale, 1.0 - progress))


def sample_process(step_records, episode_idx=0):
    """Convert step records into fixed-size recurrent training chunks."""
    if not step_records:
        return []

    obs_seq = np.stack([_as_array(rec["obs"]) for rec in step_records], axis=0)
    legal_seq = np.stack([_normalize_action_mask(rec["legal_action"]) for rec in step_records], axis=0)
    act_seq = _to_int_sequence(step_records, "act")
    prob_seq = np.stack([_as_array(rec["prob"]) for rec in step_records], axis=0)
    planner_prob_seq = np.stack([_planner_prior_from_record(rec, legal_seq[idx]) for idx, rec in enumerate(step_records)], axis=0)
    mix_alpha_seq = np.array([_mix_alpha_from_record(rec) for rec in step_records], dtype=np.float32)
    done_seq = _to_scalar_sequence(step_records, "done")

    reward_clean_seq = _to_scalar_sequence(step_records, "reward_clean")
    reward_survive_seq = _to_scalar_sequence(step_records, "reward_survive")
    value_clean_seq = _to_scalar_sequence(step_records, "value_clean")
    value_survive_seq = _to_scalar_sequence(step_records, "value_survive")
    mode_teacher_seq = _to_int_sequence(step_records, "mode_teacher")
    route_anchor_teacher_seq = _to_int_sequence(step_records, "route_anchor_teacher")
    target_teacher_seq = _to_int_sequence(step_records, "target_teacher")
    mode_teacher_mask_seq = _to_scalar_sequence(step_records, "mode_teacher_mask")
    route_anchor_teacher_mask_seq = _to_scalar_sequence(step_records, "route_anchor_teacher_mask")
    target_teacher_mask_seq = _to_scalar_sequence(step_records, "target_teacher_mask")
    return_action_teacher_seq = _to_int_sequence(step_records, "return_action_teacher")
    return_action_teacher_mask_seq = _to_scalar_sequence(step_records, "return_action_teacher_mask")
    route_phase_action_teacher_seq = _to_int_sequence(step_records, "route_phase_action_teacher")
    route_phase_action_teacher_mask_seq = _to_scalar_sequence(step_records, "route_phase_action_teacher_mask")
    battery_risk_label_seq = _to_scalar_sequence(step_records, "battery_risk_label")
    collision_risk_label_seq = _to_scalar_sequence(step_records, "collision_risk_label")
    constraint_battery_process_cost_seq = _to_scalar_sequence(step_records, "constraint_battery_process_cost")
    fallback_mask_seq = _to_scalar_sequence(step_records, "fallback_mask")
    expert_weight_seq = _to_scalar_sequence(step_records, "expert_weight")

    adv_clean_seq, _ = _compute_gae_from_rewards(reward_clean_seq, value_clean_seq, done_seq)
    adv_survive_seq, _ = _compute_gae_from_rewards(reward_survive_seq, value_survive_seq, done_seq)

    teacher_scale = _teacher_scale(episode_idx)
    mode_teacher_mask_seq = mode_teacher_mask_seq * teacher_scale
    route_anchor_teacher_mask_seq = route_anchor_teacher_mask_seq * teacher_scale
    target_teacher_mask_seq = target_teacher_mask_seq * teacher_scale
    return_action_teacher_mask_seq = return_action_teacher_mask_seq * teacher_scale
    route_phase_action_teacher_mask_seq = route_phase_action_teacher_mask_seq * teacher_scale

    samples = []
    chunk_len = Config.SEQ_CHUNK_LEN
    stride = Config.SEQ_STRIDE

    start = 0
    while start < len(step_records):
        end = min(start + chunk_len, len(step_records))
        sl = slice(start, end)
        actual_len = end - start

        obs_chunk = _pad_seq(obs_seq[sl], chunk_len, dtype=np.float32)
        legal_chunk = _pad_seq(legal_seq[sl], chunk_len, dtype=np.float32)
        act_chunk = _pad_seq(act_seq[sl], chunk_len, dtype=np.int64)
        prob_chunk = _pad_seq(prob_seq[sl], chunk_len, dtype=np.float32)
        planner_prob_chunk = _pad_seq(planner_prob_seq[sl], chunk_len, dtype=np.float32)
        action_mask_chunk = _pad_seq(legal_seq[sl], chunk_len, pad_value=1.0, dtype=np.float32)
        mix_alpha_chunk = _pad_seq(mix_alpha_seq[sl], chunk_len, dtype=np.float32)
        done_chunk = _pad_seq(done_seq[sl], chunk_len, dtype=np.float32)
        reward_clean_chunk = _pad_seq(reward_clean_seq[sl], chunk_len, dtype=np.float32)
        reward_survive_chunk = _pad_seq(reward_survive_seq[sl], chunk_len, dtype=np.float32)
        value_clean_chunk = _pad_seq(value_clean_seq[sl], chunk_len, dtype=np.float32)
        value_survive_chunk = _pad_seq(value_survive_seq[sl], chunk_len, dtype=np.float32)
        adv_clean_chunk = _pad_seq(adv_clean_seq[sl], chunk_len, dtype=np.float32)
        adv_survive_chunk = _pad_seq(adv_survive_seq[sl], chunk_len, dtype=np.float32)
        mode_teacher_chunk = _pad_seq(mode_teacher_seq[sl], chunk_len, dtype=np.int64)
        route_anchor_teacher_chunk = _pad_seq(route_anchor_teacher_seq[sl], chunk_len, dtype=np.int64)
        target_teacher_chunk = _pad_seq(target_teacher_seq[sl], chunk_len, dtype=np.int64)
        mode_teacher_mask_chunk = _pad_seq(mode_teacher_mask_seq[sl], chunk_len, dtype=np.float32)
        route_anchor_teacher_mask_chunk = _pad_seq(route_anchor_teacher_mask_seq[sl], chunk_len, dtype=np.float32)
        target_teacher_mask_chunk = _pad_seq(target_teacher_mask_seq[sl], chunk_len, dtype=np.float32)
        return_action_teacher_chunk = _pad_seq(return_action_teacher_seq[sl], chunk_len, dtype=np.int64)
        return_action_teacher_mask_chunk = _pad_seq(return_action_teacher_mask_seq[sl], chunk_len, dtype=np.float32)
        route_phase_action_teacher_chunk = _pad_seq(route_phase_action_teacher_seq[sl], chunk_len, dtype=np.int64)
        route_phase_action_teacher_mask_chunk = _pad_seq(
            route_phase_action_teacher_mask_seq[sl], chunk_len, dtype=np.float32
        )
        battery_risk_chunk = _pad_seq(battery_risk_label_seq[sl], chunk_len, dtype=np.float32)
        collision_risk_chunk = _pad_seq(collision_risk_label_seq[sl], chunk_len, dtype=np.float32)
        constraint_battery_process_cost_chunk = _pad_seq(
            constraint_battery_process_cost_seq[sl], chunk_len, dtype=np.float32
        )
        fallback_mask_chunk = _pad_seq(fallback_mask_seq[sl], chunk_len, dtype=np.float32)
        expert_weight_chunk = _pad_seq(expert_weight_seq[sl], chunk_len, dtype=np.float32)

        if actual_len < chunk_len:
            done_chunk[actual_len:] = 1.0
            planner_prob_chunk[actual_len:] = np.full(
                (chunk_len - actual_len, Config.ACTION_NUM),
                1.0 / Config.ACTION_NUM,
                dtype=np.float32,
            )
            action_mask_chunk[actual_len:] = 1.0
            mix_alpha_chunk[actual_len:] = 0.0
            mode_teacher_mask_chunk[actual_len:] = 0.0
            route_anchor_teacher_mask_chunk[actual_len:] = 0.0
            target_teacher_mask_chunk[actual_len:] = 0.0
            return_action_teacher_mask_chunk[actual_len:] = 0.0
            route_phase_action_teacher_mask_chunk[actual_len:] = 0.0
            fallback_mask_chunk[actual_len:] = 1.0

        samples.append(
            SampleData(
                obs=obs_chunk.reshape(-1).astype(np.float32),
                legal_action=legal_chunk.reshape(-1).astype(np.float32),
                act=act_chunk.astype(np.int64),
                reward_clean=reward_clean_chunk.astype(np.float32),
                reward_survive=reward_survive_chunk.astype(np.float32),
                done=done_chunk.astype(np.float32),
                value_clean=value_clean_chunk.astype(np.float32),
                value_survive=value_survive_chunk.astype(np.float32),
                advantage_clean=adv_clean_chunk.astype(np.float32),
                advantage_survive=adv_survive_chunk.astype(np.float32),
                prob=prob_chunk.reshape(-1).astype(np.float32),
                planner_prob=planner_prob_chunk.reshape(-1).astype(np.float32),
                mix_alpha=mix_alpha_chunk.astype(np.float32),
                action_mask=action_mask_chunk.reshape(-1).astype(np.float32),
                mode_teacher=mode_teacher_chunk.astype(np.int64),
                route_anchor_teacher=route_anchor_teacher_chunk.astype(np.int64),
                target_teacher=target_teacher_chunk.astype(np.int64),
                mode_teacher_mask=mode_teacher_mask_chunk.astype(np.float32),
                route_anchor_teacher_mask=route_anchor_teacher_mask_chunk.astype(np.float32),
                target_teacher_mask=target_teacher_mask_chunk.astype(np.float32),
                return_action_teacher=return_action_teacher_chunk.astype(np.int64),
                return_action_teacher_mask=return_action_teacher_mask_chunk.astype(np.float32),
                route_phase_action_teacher=route_phase_action_teacher_chunk.astype(np.int64),
                route_phase_action_teacher_mask=route_phase_action_teacher_mask_chunk.astype(np.float32),
                battery_risk_label=battery_risk_chunk.astype(np.float32),
                collision_risk_label=collision_risk_chunk.astype(np.float32),
                constraint_battery_process_cost=constraint_battery_process_cost_chunk.astype(np.float32),
                fallback_mask=fallback_mask_chunk.astype(np.float32),
                expert_weight=np.array([float(expert_weight_chunk.max(initial=0.0))], dtype=np.float32),
            )
        )

        if end >= len(step_records):
            break
        start += stride

    return samples
