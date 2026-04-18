#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Dynamic curriculum policy helpers.
"""

from __future__ import annotations

import os
from copy import deepcopy
from typing import Any


STAGE_ORDER = ["warmup", "blend", "robust", "eval_hard"]
STAGE_INDEX = {name: idx for idx, name in enumerate(STAGE_ORDER)}
STAGE_PROFILE_WEIGHTS = {
    "warmup": (("anchor", 0.45), ("mild", 0.35), ("broad", 0.20)),
    "blend": (("anchor", 0.25), ("mild", 0.35), ("broad", 0.30), ("broad_eval", 0.10)),
    "robust": (("anchor", 0.10), ("mild", 0.25), ("broad", 0.40), ("broad_eval", 0.25)),
    "eval_hard": (("anchor", 0.05), ("mild", 0.15), ("broad", 0.35), ("broad_eval", 0.45)),
}
OBSERVATION_PROFILE_WEIGHTS = {
    "warmup": (("anchor", 0.55), ("mild", 0.35), ("broad", 0.10)),
    "blend": (("anchor", 0.35), ("mild", 0.40), ("broad", 0.20), ("broad_eval", 0.05)),
    "robust": (("anchor", 0.20), ("mild", 0.30), ("broad", 0.35), ("broad_eval", 0.15)),
    "eval_hard": STAGE_PROFILE_WEIGHTS["eval_hard"],
}
CONSERVATIVE_PROFILE_WEIGHTS = {
    "warmup": (("anchor", 0.60), ("mild", 0.35), ("broad", 0.05)),
    "blend": (("anchor", 0.45), ("mild", 0.35), ("broad", 0.15), ("broad_eval", 0.05)),
    "robust": (("anchor", 0.25), ("mild", 0.35), ("broad", 0.25), ("broad_eval", 0.15)),
    "eval_hard": (("anchor", 0.08), ("mild", 0.22), ("broad", 0.40), ("broad_eval", 0.30)),
}
PROFILE_KEYS = ("anchor", "mild", "broad", "broad_eval")

FAST_TRACK_MIN_EPISODES = 10
FULL_WINDOW_MIN_EPISODES = 40
MIN_STAGE_DWELL_STEPS = {
    "warmup": 3000,
    "blend": 5000,
    "robust": 8000,
    "eval_hard": 0,
}


def _metric(metrics: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not metrics:
        return float(default)
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _meets_s0_exit(global_step_since_resume: int, metrics: dict[str, Any] | None, learning: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    entropy_loss = _metric(learning, "entropy_loss", 9.9)
    entropy_trend_ratio = _metric(learning, "entropy_trend_ratio", 2.0)
    return (
        global_step_since_resume >= 3000
        and _metric(metrics, "win_rate") >= 0.60
        and _metric(metrics, "battery_fail_rate", 1.0) <= 0.20
        and _metric(metrics, "collision_fail_rate", 1.0) <= 0.10
        and _metric(metrics, "return_stall_rate", 1.0) <= 0.40
        and (entropy_loss <= 0.92 or entropy_trend_ratio <= 1.02)
    )


def _meets_fast_skip_blend(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    return (
        int(metrics.get("_count", 0)) >= FAST_TRACK_MIN_EPISODES
        and _metric(metrics, "win_rate") >= 0.72
        and _metric(metrics, "battery_fail_rate", 1.0) <= 0.12
        and _metric(metrics, "return_stall_rate", 1.0) <= 0.35
        and _metric(metrics, "wall_hugging_clean_floor_rate", 1.0) <= 0.06
        and _metric(metrics, "suboptimal_target_hold_rate", 1.0) <= 0.08
    )


def _meets_fast_skip_robust(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    return (
        _meets_fast_skip_blend(metrics)
        and _metric(metrics, "broad_win_rate", -1.0) >= 0.65
        and _metric(metrics, "battery_fail_rate", 1.0) <= 0.08
        and _metric(metrics, "planner_policy_divergence_rate", 1.0) <= 0.22
    )


def _meets_blend_to_robust(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    return (
        int(metrics.get("_count", 0)) >= FULL_WINDOW_MIN_EPISODES
        and _metric(metrics, "win_rate") >= 0.72
        and _metric(metrics, "battery_fail_rate", 1.0) <= 0.12
        and _metric(metrics, "collision_fail_rate", 1.0) <= 0.08
        and _metric(metrics, "return_stall_rate", 1.0) <= 0.35
        and _metric(metrics, "wall_hugging_clean_floor_rate", 1.0) <= 0.06
        and _metric(metrics, "stale_boundary_follow_rate", 1.0) <= 0.05
        and _metric(metrics, "suboptimal_target_hold_rate", 1.0) <= 0.08
        and _metric(metrics, "planner_policy_divergence_rate", 1.0) <= 0.28
    )


def _meets_robust_to_eval(metrics: dict[str, Any] | None, stage_entry: dict[str, Any] | None) -> bool:
    if not metrics or int(metrics.get("_count", 0)) < FULL_WINDOW_MIN_EPISODES:
        return False
    unknown_ok = True
    known_path_ok = True
    if stage_entry:
        entry_unknown = _metric(stage_entry, "avg_unknown_on_target_path_ratio", 0.0)
        current_unknown = _metric(metrics, "avg_unknown_on_target_path_ratio", entry_unknown)
        unknown_ok = current_unknown <= max(entry_unknown * 0.85, entry_unknown - 0.03)

        entry_known = _metric(stage_entry, "avg_all_charger_known_path_count", 0.0)
        current_known = _metric(metrics, "avg_all_charger_known_path_count", entry_known)
        known_path_ok = current_known >= max(entry_known * 1.10, entry_known + 0.10)

    return (
        _metric(metrics, "broad_win_rate", -1.0) >= 0.65
        and _metric(metrics, "battery_fail_rate", 1.0) <= 0.08
        and _metric(metrics, "collision_fail_rate", 1.0) <= 0.05
        and _metric(metrics, "suboptimal_target_hold_rate", 1.0) <= 0.05
        and unknown_ok
        and known_path_ok
    )


def should_regress_stage(
    current_stage: str,
    stage_entry_metrics: dict[str, Any] | None,
    current_metrics: dict[str, Any] | None,
    learning_metrics: dict[str, Any] | None,
) -> bool:
    if current_stage == "warmup" or not stage_entry_metrics or not current_metrics:
        return False

    battery_fail_now = _metric(current_metrics, "battery_fail_rate", 1.0)
    battery_fail_entry = _metric(stage_entry_metrics, "battery_fail_rate", battery_fail_now)
    stall_now = _metric(current_metrics, "return_stall_rate", 1.0)
    stall_entry = _metric(stage_entry_metrics, "return_stall_rate", stall_now)
    entropy_loss = _metric(learning_metrics, "entropy_loss", 0.0)
    env_total_score = _metric(learning_metrics, "env_total_score", 0.0)
    entry_env_total = _metric(stage_entry_metrics, "env_total_score", env_total_score)

    battery_regress = battery_fail_now > max(battery_fail_entry * 1.35, battery_fail_entry + 0.03)
    stall_regress = stall_now > max(stall_entry * 1.25, stall_entry + 0.05)
    score_regress = entropy_loss > 0.98 and env_total_score < entry_env_total * 0.92
    return battery_regress or stall_regress or score_regress


def choose_stage(
    current_stage: str,
    context: dict[str, Any],
    stage_entry_metrics: dict[str, Any] | None = None,
) -> str:
    global_step_since_resume = int(context.get("global_step_since_resume", 0))
    window_metrics = context.get("window_metrics")
    bootstrap_metrics = context.get("bootstrap_metrics")
    learning_metrics = context.get("learning_metrics")
    resume_fast_track = bool(context.get("resume_fast_track", False))

    stage = current_stage
    if stage == "warmup":
        if resume_fast_track and _meets_fast_skip_robust(bootstrap_metrics):
            return "robust"
        if resume_fast_track and _meets_fast_skip_blend(bootstrap_metrics):
            return "blend"
        if _meets_s0_exit(global_step_since_resume, bootstrap_metrics or window_metrics, learning_metrics):
            return "blend"
        return "warmup"
    if stage == "blend":
        if _meets_blend_to_robust(window_metrics):
            return "robust"
        return "blend"
    if stage == "robust":
        if _meets_robust_to_eval(window_metrics, stage_entry_metrics):
            return "eval_hard"
        return "robust"
    return "eval_hard"


def previous_stage(stage: str) -> str:
    idx = STAGE_INDEX.get(stage, 0)
    return STAGE_ORDER[max(idx - 1, 0)]


def profile_weights_for_stage(stage: str) -> tuple[tuple[str, float], ...]:
    return STAGE_PROFILE_WEIGHTS.get(stage, STAGE_PROFILE_WEIGHTS["warmup"])


def _weights_to_dict(weights: tuple[tuple[str, float], ...]) -> dict[str, float]:
    payload = {key: 0.0 for key in PROFILE_KEYS}
    for profile, weight in weights:
        payload[str(profile)] = float(weight)
    return payload


def _weights_to_tuple(weights: dict[str, float]) -> tuple[tuple[str, float], ...]:
    total = sum(max(float(weights.get(profile, 0.0)), 0.0) for profile in PROFILE_KEYS)
    if total <= 0:
        return STAGE_PROFILE_WEIGHTS["warmup"]
    normalized = {profile: max(float(weights.get(profile, 0.0)), 0.0) / total for profile in PROFILE_KEYS}
    return tuple((profile, normalized[profile]) for profile in PROFILE_KEYS if normalized[profile] > 0.0)


def _interpolate_weights(
    left: tuple[tuple[str, float], ...],
    right: tuple[tuple[str, float], ...],
    factor: float,
) -> tuple[tuple[str, float], ...]:
    factor = max(0.0, min(float(factor), 1.0))
    left_dict = _weights_to_dict(left)
    right_dict = _weights_to_dict(right)
    blended = {
        profile: left_dict[profile] * (1.0 - factor) + right_dict[profile] * factor
        for profile in PROFILE_KEYS
    }
    return _weights_to_tuple(blended)


def observation_phase_active(global_step_since_resume: int, stage: str) -> bool:
    if stage == "eval_hard":
        return False
    observation_steps = int(os.getenv("KAIWU_CURRICULUM_OBSERVATION_PHASE_STEPS", "5000") or "5000")
    return int(global_step_since_resume) < max(observation_steps, 0)


def _poor_behavior(stage: str, metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    battery_fail = _metric(metrics, "battery_fail_rate", 0.0)
    return_stall = _metric(metrics, "return_stall_rate", 0.0)
    planner_div = _metric(metrics, "planner_policy_divergence_rate", 0.0)
    thresholds = {
        "warmup": (0.22, 0.55, 0.82),
        "blend": (0.18, 0.48, 0.75),
        "robust": (0.12, 0.40, 0.60),
        "eval_hard": (0.10, 0.35, 0.50),
    }
    max_battery, max_stall, max_div = thresholds.get(stage, thresholds["warmup"])
    return battery_fail > max_battery or return_stall > max_stall or planner_div > max_div


def _strong_behavior(stage: str, metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    battery_fail = _metric(metrics, "battery_fail_rate", 1.0)
    return_stall = _metric(metrics, "return_stall_rate", 1.0)
    planner_div = _metric(metrics, "planner_policy_divergence_rate", 1.0)
    broad_win = _metric(metrics, "broad_win_rate", -1.0)
    avg_cps = _metric(metrics, "avg_clean_per_step", 0.0)
    thresholds = {
        "warmup": (0.15, 0.45, 0.72, 0.45, 0.50),
        "blend": (0.12, 0.40, 0.65, 0.50, 0.55),
        "robust": (0.08, 0.32, 0.50, 0.60, 0.62),
        "eval_hard": (0.08, 0.30, 0.45, 0.65, 0.65),
    }
    max_battery, max_stall, max_div, min_cps, min_broad = thresholds.get(stage, thresholds["warmup"])
    broad_ok = broad_win < 0.0 or broad_win >= min_broad
    return battery_fail <= max_battery and return_stall <= max_stall and planner_div <= max_div and avg_cps >= min_cps and broad_ok


def profile_plan_for_runtime(stage: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
    state = state or {}
    current_stage = str(stage or state.get("stage") or "warmup").strip().lower()
    if current_stage not in STAGE_INDEX:
        current_stage = "warmup"
    adaptive_enabled = str(os.getenv("KAIWU_CURRICULUM_ADAPTIVE_PROFILE_ENABLED", "1") or "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
    metrics = state.get("last_global_metrics") or state.get("last_bootstrap_metrics") or {}
    global_step_since_resume = int(state.get("global_step_since_resume", 0))
    observation_active = observation_phase_active(global_step_since_resume, current_stage)

    standard = profile_weights_for_stage(current_stage)
    observation = OBSERVATION_PROFILE_WEIGHTS.get(current_stage, standard)
    conservative = CONSERVATIVE_PROFILE_WEIGHTS.get(current_stage, observation)

    if not adaptive_enabled:
        selected = standard
        observation_active = False
    elif _poor_behavior(current_stage, metrics):
        selected = conservative
    elif observation_active and current_stage != "eval_hard":
        observation_steps = max(int(os.getenv("KAIWU_CURRICULUM_OBSERVATION_PHASE_STEPS", "5000") or "5000"), 1)
        release = min(global_step_since_resume / observation_steps, 1.0)
        if _strong_behavior(current_stage, metrics):
            release = min(release + 0.25, 1.0)
        selected = _interpolate_weights(observation, standard, release)
    else:
        selected = standard

    weight_dict = _weights_to_dict(selected)
    return {
        "weights": selected,
        "weight_map": weight_dict,
        "observation_phase_active": bool(observation_active),
        "tightened": bool(_poor_behavior(current_stage, metrics)),
    }


def stage_progress(stage: str, metrics: dict[str, Any] | None, learning: dict[str, Any] | None) -> float:
    if not metrics:
        return 0.0
    if stage == "warmup":
        ratios = [
            _metric(metrics, "win_rate") / 0.60,
            0.20 / max(_metric(metrics, "battery_fail_rate", 1.0), 1e-6),
            0.10 / max(_metric(metrics, "collision_fail_rate", 1.0), 1e-6),
            0.40 / max(_metric(metrics, "return_stall_rate", 1.0), 1e-6),
        ]
        entropy = _metric(learning, "entropy_loss", 9.9)
        ratios.append(0.92 / max(entropy, 1e-6))
        return max(0.0, min(min(ratios), 1.0))
    if stage == "blend":
        ratios = [
            _metric(metrics, "win_rate") / 0.72,
            0.12 / max(_metric(metrics, "battery_fail_rate", 1.0), 1e-6),
            0.35 / max(_metric(metrics, "return_stall_rate", 1.0), 1e-6),
            0.06 / max(_metric(metrics, "wall_hugging_clean_floor_rate", 1.0), 1e-6),
            0.08 / max(_metric(metrics, "suboptimal_target_hold_rate", 1.0), 1e-6),
        ]
        return max(0.0, min(min(ratios), 1.0))
    if stage == "robust":
        ratios = [
            _metric(metrics, "broad_win_rate", 0.0) / 0.65,
            0.08 / max(_metric(metrics, "battery_fail_rate", 1.0), 1e-6),
            0.05 / max(_metric(metrics, "suboptimal_target_hold_rate", 1.0), 1e-6),
        ]
        return max(0.0, min(min(ratios), 1.0))
    return 1.0


def snapshot_stage_entry_metrics(metrics: dict[str, Any] | None, learning_metrics: dict[str, Any] | None) -> dict[str, Any]:
    payload = deepcopy(metrics or {})
    if learning_metrics:
        payload["env_total_score"] = _metric(learning_metrics, "env_total_score", 0.0)
    return payload


def curriculum_gate_ratios(
    stage: str,
    metrics: dict[str, Any] | None,
    learning_metrics: dict[str, Any] | None,
    global_step_since_resume: int,
    entered_global_step: int = 0,
) -> dict[str, float]:
    del learning_metrics
    dwell_required = MIN_STAGE_DWELL_STEPS.get(stage, 0)
    dwell_progress = max(int(global_step_since_resume) - int(entered_global_step), 0)
    global_step_ratio = 1.0 if dwell_required <= 0 else dwell_progress / max(dwell_required, 1)

    stall_threshold = {
        "warmup": 0.40,
        "blend": 0.35,
    }.get(stage)
    if stall_threshold is None:
        return {
            "curriculum_gate_global_step_ratio": float(global_step_ratio),
            "curriculum_gate_return_stall_ratio": 1.0,
            "curriculum_gate_return_stall_ratio_raw": 1.0,
            "curriculum_return_stall_margin": 0.0,
        }

    current_stall = _metric(metrics, "return_stall_rate", 1.0)
    stall_ratio_raw = stall_threshold / max(current_stall, 1e-6)
    stall_ratio = min(stall_ratio_raw, 2.0)
    return {
        "curriculum_gate_global_step_ratio": float(global_step_ratio),
        "curriculum_gate_return_stall_ratio": float(stall_ratio),
        "curriculum_gate_return_stall_ratio_raw": float(stall_ratio_raw),
        "curriculum_return_stall_margin": float(current_stall - stall_threshold),
    }
