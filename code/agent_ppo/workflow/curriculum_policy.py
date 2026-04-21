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

from agent_ppo.conf.conf import Config


STAGE_ORDER = ["warmup", "blend", "robust", "eval_hard"]
STAGE_INDEX = {name: idx for idx, name in enumerate(STAGE_ORDER)}
STAGE_PROFILE_WEIGHTS = {
    "warmup": (("anchor", 0.45), ("mild", 0.40), ("broad", 0.15)),
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
DEGRADED_MAINLINE_PROFILE_WEIGHTS = {
    "warmup": (("anchor", 0.52), ("mild", 0.33), ("broad", 0.15)),
}
PROFILE_KEYS = ("anchor", "mild", "broad", "broad_eval")
S1_SURVIVAL_PROFILE_WEIGHTS = (("anchor", 0.60), ("mild", 0.30), ("broad", 0.10))
WARMUP_BATTERY_GUARD_WEIGHTS = (("anchor", 0.65), ("mild", 0.30), ("broad", 0.05))
TRANSITION_PROFILE_WEIGHTS = {
    "blend": (("anchor", 0.45), ("mild", 0.40), ("broad", 0.15)),
    "robust": (("anchor", 0.20), ("mild", 0.35), ("broad", 0.30), ("broad_eval", 0.15)),
}
TRANSITION_CONSERVATIVE_PROFILE_WEIGHTS = {
    "blend": (("anchor", 0.55), ("mild", 0.35), ("broad", 0.10)),
    "robust": (("anchor", 0.30), ("mild", 0.35), ("broad", 0.25), ("broad_eval", 0.10)),
}

FAST_TRACK_MIN_EPISODES = 10
FULL_WINDOW_MIN_EPISODES = 40
MIN_STAGE_DWELL_STEPS = {
    "warmup": 3000,
    "blend": 5000,
    "robust": 8000,
    "eval_hard": 0,
}
MAX_STAGE_WAIT_STEPS = {
    "warmup": Config.CURRICULUM_WARMUP_TIMEOUT_STEPS,
    "blend": Config.CURRICULUM_BLEND_TIMEOUT_STEPS,
    "robust": Config.CURRICULUM_ROBUST_TIMEOUT_STEPS,
    "eval_hard": 0,
}


def _metric(metrics: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not metrics:
        return float(default)
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _current_train_phase() -> str:
    return str(os.getenv("KAIWU_TRAIN_PHASE", "") or "").strip().lower()


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
        and _metric(metrics, "return_stall_rate", 1.0) <= 0.50
        and _metric(metrics, "planner_policy_divergence_rate", 1.0) <= 0.80
        and _metric(metrics, "zero_charge_battery_fail_rate", 1.0) <= Config.CURRICULUM_ZERO_CHARGE_FAIL_STRICT_GATE_MAX
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
    return choose_stage_decision(current_stage, context, stage_entry_metrics)["proposed_stage"]


def _meets_warmup_soft_gate(metrics: dict[str, Any] | None, global_step_since_resume: int) -> bool:
    if not metrics:
        return False
    return (
        int(global_step_since_resume) >= 12000
        and int(metrics.get("_count", 0)) >= FULL_WINDOW_MIN_EPISODES
        and _metric(metrics, "win_rate") >= 0.82
        and _metric(metrics, "battery_fail_rate", 1.0) <= 0.08
        and _metric(metrics, "collision_fail_rate", 1.0) <= 0.06
        and _metric(metrics, "avg_clean_per_step", 0.0) >= 0.68
        and 3.5 <= _metric(metrics, "avg_charge_count", 0.0) <= 6.5
        and _metric(metrics, "avg_coverage_efficiency_20", 0.0) >= 0.82
        and _metric(metrics, "return_stall_rate", 1.0) <= 0.65
        and _metric(metrics, "planner_policy_divergence_rate", 1.0) <= 0.92
    )


def _meets_blend_soft_gate(metrics: dict[str, Any] | None, dwell_steps: int) -> bool:
    if not metrics:
        return False
    return (
        int(dwell_steps) >= 18000
        and int(metrics.get("_count", 0)) >= FULL_WINDOW_MIN_EPISODES
        and _metric(metrics, "win_rate") >= 0.78
        and _metric(metrics, "battery_fail_rate", 1.0) <= 0.10
        and _metric(metrics, "avg_clean_per_step", 0.0) >= 0.72
        and _metric(metrics, "broad_win_rate", -1.0) >= 0.55
        and _metric(metrics, "avg_coverage_efficiency_20", 0.0) >= 0.84
        and _metric(metrics, "return_stall_rate", 1.0) <= 0.58
        and _metric(metrics, "planner_policy_divergence_rate", 1.0) <= 0.85
    )


def _meets_warmup_timeout_gate(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    return (
        _metric(metrics, "battery_fail_rate", 1.0) <= 0.18
        and _metric(metrics, "collision_fail_rate", 1.0) <= 0.08
        and _metric(metrics, "avg_clean_per_step", 0.0) >= 0.55
        and _metric(metrics, "win_rate", 0.0) >= 0.70
    )


def _meets_blend_timeout_gate(metrics: dict[str, Any] | None) -> bool:
    if not metrics:
        return False
    return (
        _metric(metrics, "battery_fail_rate", 1.0) <= 0.14
        and _metric(metrics, "collision_fail_rate", 1.0) <= 0.06
        and _metric(metrics, "avg_clean_per_step", 0.0) >= 0.62
        and _metric(metrics, "win_rate", 0.0) >= 0.72
    )


def choose_stage_decision(
    current_stage: str,
    context: dict[str, Any],
    stage_entry_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if _current_train_phase() == "s1_survival":
        return {"proposed_stage": "warmup", "promotion_reason": "phase_lock"}
    global_step_since_resume = int(context.get("global_step_since_resume", 0))
    window_metrics = context.get("window_metrics")
    bootstrap_metrics = context.get("bootstrap_metrics")
    learning_metrics = context.get("learning_metrics")
    resume_fast_track = bool(context.get("resume_fast_track", False))
    training_start_mode_raw = context.get("training_start_mode")
    preload_enabled = bool(context.get("preload_enabled", False))
    training_start_mode = str(training_start_mode_raw).strip().lower() if training_start_mode_raw is not None else ""
    stabilize_steps = int(Config.CURRICULUM_RESUME_STABILIZE_STEPS)
    entered_global_step = int(context.get("entered_global_step", 0))
    dwell_steps = max(global_step_since_resume - entered_global_step, 0)
    timeout_steps = int(MAX_STAGE_WAIT_STEPS.get(current_stage, 0))

    stage = current_stage
    if preload_enabled and training_start_mode and training_start_mode not in {"scratch", "random", "from_scratch", "fresh"} and global_step_since_resume < stabilize_steps:
        return {"proposed_stage": current_stage, "promotion_reason": None}
    if stage == "warmup":
        if resume_fast_track and _meets_fast_skip_robust(bootstrap_metrics):
            return {"proposed_stage": "robust", "promotion_reason": "strict_gate"}
        if resume_fast_track and _meets_fast_skip_blend(bootstrap_metrics):
            return {"proposed_stage": "blend", "promotion_reason": "strict_gate"}
        if _meets_s0_exit(global_step_since_resume, bootstrap_metrics or window_metrics, learning_metrics):
            return {"proposed_stage": "blend", "promotion_reason": "strict_gate"}
        if _meets_warmup_soft_gate(window_metrics, global_step_since_resume):
            return {"proposed_stage": "blend", "promotion_reason": "soft_gate"}
        if timeout_steps > 0 and dwell_steps >= timeout_steps and _meets_warmup_timeout_gate(window_metrics):
            return {"proposed_stage": "blend", "promotion_reason": "timeout_gate"}
        return {"proposed_stage": "warmup", "promotion_reason": None}
    if stage == "blend":
        if _meets_blend_to_robust(window_metrics):
            return {"proposed_stage": "robust", "promotion_reason": "strict_gate"}
        if _meets_blend_soft_gate(window_metrics, dwell_steps):
            return {"proposed_stage": "robust", "promotion_reason": "soft_gate"}
        if timeout_steps > 0 and dwell_steps >= timeout_steps and _meets_blend_timeout_gate(window_metrics):
            return {"proposed_stage": "robust", "promotion_reason": "timeout_gate"}
        return {"proposed_stage": "blend", "promotion_reason": None}
    if stage == "robust":
        if _meets_robust_to_eval(window_metrics, stage_entry_metrics):
            return {"proposed_stage": "eval_hard", "promotion_reason": "strict_gate"}
        return {"proposed_stage": "robust", "promotion_reason": None}
    return {"proposed_stage": "eval_hard", "promotion_reason": None}


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
    observation_steps = int(os.getenv("KAIWU_CURRICULUM_OBSERVATION_PHASE_STEPS", "8000") or "8000")
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
    if _current_train_phase() == "s1_survival":
        return {
            "weights": S1_SURVIVAL_PROFILE_WEIGHTS,
            "weight_map": _weights_to_dict(S1_SURVIVAL_PROFILE_WEIGHTS),
            "observation_phase_active": False,
            "tightened": True,
        }
    adaptive_enabled = str(os.getenv("KAIWU_CURRICULUM_ADAPTIVE_PROFILE_ENABLED", "1") or "1").strip().lower() not in {
        "0", "false", "no", "off"
    }
    metrics = state.get("last_global_metrics") or state.get("last_bootstrap_metrics") or {}
    global_step_since_resume = int(state.get("global_step_since_resume", 0))
    observation_active = observation_phase_active(global_step_since_resume, current_stage)
    training_start_mode_raw = state.get("training_start_mode")
    preload_enabled = bool(state.get("preload_enabled", False))
    training_start_mode = str(training_start_mode_raw).strip().lower() if training_start_mode_raw is not None else ""
    in_resume_stabilization = (
        preload_enabled
        and bool(training_start_mode)
        and training_start_mode not in {"scratch", "random", "from_scratch", "fresh"}
        and global_step_since_resume < int(Config.CURRICULUM_RESUME_STABILIZE_STEPS)
    )

    standard = profile_weights_for_stage(current_stage)
    observation = OBSERVATION_PROFILE_WEIGHTS.get(current_stage, standard)
    conservative = CONSERVATIVE_PROFILE_WEIGHTS.get(current_stage, observation)
    resume_stable = (("anchor", 0.60), ("mild", 0.30), ("broad", 0.10))
    resume_tight = (("anchor", 0.70), ("mild", 0.25), ("broad", 0.05))
    warmup_guard_steps = int(os.getenv("KAIWU_CURRICULUM_WARMUP_BATTERY_GUARD_STEPS", "10000") or "10000")

    stagnation_level = int((state or {}).get("curriculum_stagnation_level", 0) or 0)
    degraded_mainline = bool((state or {}).get("degraded_mainline", False))
    transition_target_stage = str((state or {}).get("transition_target_stage") or "").strip().lower()
    transition_entered_global_step = int((state or {}).get("transition_entered_global_step", 0) or 0)
    in_transition_guard = bool((state or {}).get("in_transition_guard", False))
    guard_elapsed = max(global_step_since_resume - transition_entered_global_step, 0)
    guard_steps = {
        "blend": int(Config.CURRICULUM_BLEND_GUARD_STEPS),
        "robust": int(Config.CURRICULUM_ROBUST_GUARD_STEPS),
    }.get(current_stage, 0)

    if in_transition_guard and transition_target_stage == current_stage and guard_elapsed < guard_steps:
        selected = TRANSITION_PROFILE_WEIGHTS.get(current_stage, standard)
        if _poor_behavior(current_stage, metrics) or stagnation_level > 0:
            selected = TRANSITION_CONSERVATIVE_PROFILE_WEIGHTS.get(current_stage, selected)
        return {
            "weights": selected,
            "weight_map": _weights_to_dict(selected),
            "observation_phase_active": False,
            "tightened": bool(_poor_behavior(current_stage, metrics) or stagnation_level > 0),
        }

    if not adaptive_enabled:
        selected = standard
        observation_active = False
    elif current_stage == "warmup" and global_step_since_resume < warmup_guard_steps:
        battery_fail = _metric(metrics, "battery_fail_rate", 1.0)
        avg_charge_count = _metric(metrics, "avg_charge_count", 0.0)
        zero_charge_fail = _metric(metrics, "zero_charge_battery_fail_rate", 1.0)
        if battery_fail <= 0.20 and 4.0 <= avg_charge_count <= 6.0 and zero_charge_fail <= 0.15:
            selected = standard
        else:
            selected = WARMUP_BATTERY_GUARD_WEIGHTS
        observation_active = False
    elif in_resume_stabilization:
        battery_fail = _metric(metrics, "battery_fail_rate", 0.0)
        collision_fail = _metric(metrics, "collision_fail_rate", 0.0)
        return_stall = _metric(metrics, "return_stall_rate", 0.0)
        if battery_fail > 0.22 or collision_fail > 0.06 or return_stall > 0.55:
            selected = resume_tight
        else:
            selected = resume_stable
        observation_active = True
    elif (
        _poor_behavior(current_stage, metrics)
        or stagnation_level > 0
        or degraded_mainline
        or _metric(metrics, "zero_charge_battery_fail_rate", 0.0) > Config.CURRICULUM_ZERO_CHARGE_FAIL_PROFILE_WARN
    ):
        if current_stage == "warmup" and (degraded_mainline or stagnation_level >= 2):
            selected = DEGRADED_MAINLINE_PROFILE_WEIGHTS.get(current_stage, conservative)
        else:
            selected = conservative
    elif observation_active and current_stage != "eval_hard":
        observation_steps = max(int(os.getenv("KAIWU_CURRICULUM_OBSERVATION_PHASE_STEPS", "8000") or "8000"), 1)
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
        "tightened": bool(
            _poor_behavior(current_stage, metrics)
            or stagnation_level > 0
            or degraded_mainline
            or _metric(metrics, "zero_charge_battery_fail_rate", 0.0) > Config.CURRICULUM_ZERO_CHARGE_FAIL_PROFILE_WARN
        ),
    }


def stagnation_status(
    stage: str,
    metrics: dict[str, Any] | None,
    global_step_since_resume: int,
    entered_global_step: int,
    stagnant_windows: int,
) -> tuple[int, list[str]]:
    if not metrics:
        return 0, []

    min_steps = {
        "warmup": Config.CURRICULUM_STAGNATION_WARMUP_MIN_STEPS,
        "blend": Config.CURRICULUM_STAGNATION_BLEND_MIN_STEPS,
        "robust": Config.CURRICULUM_STAGNATION_ROBUST_MIN_STEPS,
    }.get(stage, Config.CURRICULUM_STAGNATION_ROBUST_MIN_STEPS)
    if int(global_step_since_resume) - int(entered_global_step) < int(min_steps):
        return 0, []

    if _current_train_phase() == "s1_survival":
        reasons = []
        avg_cps = _metric(metrics, "avg_clean_per_step", 0.0) or 0.0
        zero_charge_fail = _metric(metrics, "zero_charge_battery_fail_rate", 0.0) or 0.0
        battery_positive_reward_rate = _metric(metrics, "battery_positive_reward_rate", 0.0) or 0.0
        if avg_cps < 0.46:
            reasons.append("collapse")
        if zero_charge_fail > 0.40:
            reasons.append("charge")
        if battery_positive_reward_rate > 0.20:
            reasons.append("reward")
        if len(reasons) < 2:
            return 0, reasons

        windows = int(stagnant_windows)
        if windows >= 8:
            return 3, reasons
        if windows >= 5:
            return 2, reasons
        if windows >= 3:
            return 1, reasons
        return 0, reasons

    thresholds = {
        "warmup": {"cps": 0.32, "planner": 0.55, "expand": 0.03, "stall": 0.45},
        "blend": {"cps": 0.38, "planner": 0.50, "expand": 0.03, "stall": 0.40},
        "robust": {"cps": 0.45, "planner": 0.42, "expand": 0.06, "stall": 0.32},
        "eval_hard": {"cps": 0.45, "planner": 0.35, "expand": 0.08, "stall": 0.28},
    }
    target = thresholds.get(stage, thresholds["robust"])
    reasons = []
    if _metric(metrics, "avg_clean_per_step", 0.0) < target["cps"]:
        reasons.append("cps")
    planner_metric = _metric(
        metrics,
        "reliable_planner_divergence_rate",
        _metric(metrics, "route_phase_planner_divergence_rate", 1.0),
    )
    if planner_metric > target["planner"]:
        reasons.append("planner")
    if _metric(metrics, "mode_usage_expand", 0.0) < target["expand"]:
        reasons.append("expand")
    stall_metric = _metric(
        metrics,
        "route_phase_return_stall_rate",
        _metric(metrics, "high_need_return_stall_rate", _metric(metrics, "return_stall_rate", 0.0)),
    )
    if stall_metric > target["stall"]:
        reasons.append("stall")
    if _metric(metrics, "zero_charge_battery_fail_rate", 0.0) > 0.55:
        reasons.append("charge")
    if _metric(metrics, "battery_positive_reward_rate", 0.0) > 0.20:
        reasons.append("reward")

    if len(reasons) < 2:
        return 0, reasons

    windows = int(stagnant_windows)
    if windows >= 8:
        return 3, reasons
    if windows >= 5:
        return 2, reasons
    if windows >= 3:
        return 1, reasons
    return 0, reasons


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
