#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Pure helpers for constraint-oriented training signals.
"""

from __future__ import annotations


def clip01(value: float) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    return value


def has_known_charge_route(all_charger_known_path_count: float, target_reliable: bool) -> bool:
    try:
        known = float(all_charger_known_path_count)
    except (TypeError, ValueError):
        known = 0.0
    return known >= 1.0 and bool(target_reliable)


def compute_slack_confidence(
    all_charger_known_path_count: float,
    target_reliable: bool,
    anchor_reliable: bool,
    unknown_on_target_path_ratio: float,
) -> float:
    confidence = 0.0
    try:
        known_count = float(all_charger_known_path_count)
    except (TypeError, ValueError):
        known_count = 0.0
    try:
        unknown_ratio = float(unknown_on_target_path_ratio)
    except (TypeError, ValueError):
        unknown_ratio = 1.0

    if known_count >= 1.0:
        confidence += 0.4
    if bool(target_reliable):
        confidence += 0.2
    if bool(anchor_reliable):
        confidence += 0.2
    if unknown_ratio < 0.15:
        confidence += 0.2
    return clip01(confidence)


def compute_charge_need_score(
    has_known_route: bool,
    charge_margin_now: float,
    battery_ratio: float,
    future_recoverability_score: float,
) -> float:
    recoverability_term = clip01(-float(future_recoverability_score))
    if has_known_route:
        margin_term = clip01((12.0 - float(charge_margin_now)) / 16.0)
        battery_term = clip01((0.35 - float(battery_ratio)) / 0.20)
        return max(margin_term, 0.7 * battery_term, 0.8 * recoverability_term)
    battery_term = clip01((0.30 - float(battery_ratio)) / 0.18)
    return max(0.8 * battery_term, 0.8 * recoverability_term)


def compute_route_phase_shadow_risk(
    *,
    min_recoverability: float,
    charger_slack: float,
    charge_margin_now: float,
    planner_topk_reachable_count: int,
    unknown_target_ratio: float,
    route_contract_pressure: float,
    recoverability_warn: float = 0.35,
    recoverability_span: float = 0.70,
    prepare_return_slack_threshold: float = 6.0,
    charge_margin_warn: float = 17.0,
    unknown_ratio_threshold: float = 0.20,
) -> float:
    recoverability_warn_term = clip01((float(recoverability_warn) - float(min_recoverability)) / max(float(recoverability_span), 1e-6))
    slack_warn_term = clip01(
        (float(prepare_return_slack_threshold) - float(charger_slack))
        / max(float(prepare_return_slack_threshold), 1.0)
    )
    margin_warn_term = clip01(
        (float(charge_margin_warn) - float(charge_margin_now))
        / max(float(charge_margin_warn), 1.0)
    )
    no_reachable_route_term = 1.0 if int(planner_topk_reachable_count) <= 0 else 0.0
    unknown_ratio_threshold = float(unknown_ratio_threshold)
    unknown_path_term = clip01(
        (float(unknown_target_ratio) - unknown_ratio_threshold)
        / max(1.0 - unknown_ratio_threshold, 1e-6)
    )
    unknown_route_term = max(no_reachable_route_term, unknown_path_term)
    route_pressure_term = clip01(route_contract_pressure)
    return max(
        1.00 * recoverability_warn_term,
        0.90 * slack_warn_term,
        0.75 * margin_warn_term,
        0.55 * unknown_route_term,
        0.65 * route_pressure_term,
    )


def compute_route_phase_reward_ready(
    *,
    current_mode: int,
    mode_contract: int,
    mode_return: int,
    route_phase_reliable_active: bool,
    return_action_reliable: bool,
    anchor_reliable: bool,
    target_reliable: bool,
    known_route_available: bool,
    route_phase_shadow_risk: float,
    route_phase_shadow_risk_threshold: float = 0.12,
) -> bool:
    route_phase_active = int(current_mode) in (int(mode_contract), int(mode_return))
    route_context_available = bool(
        route_phase_reliable_active
        or return_action_reliable
        or anchor_reliable
        or target_reliable
        or known_route_available
    )
    return bool(
        route_phase_active
        and route_context_available
        and float(route_phase_shadow_risk) >= float(route_phase_shadow_risk_threshold)
    )


def classify_battery_state(charge_need_score: float, safe_threshold: float = 0.12, critical_threshold: float = 0.28) -> str:
    score = float(charge_need_score)
    if score < float(safe_threshold):
        return "safe"
    if score < float(critical_threshold):
        return "planning"
    return "critical"


def compute_battery_process_cost_step(
    *,
    has_known_route: bool,
    charger_slack: float,
    slack_confidence: float,
    charge_margin_now: float,
    battery_ratio: float,
    future_recoverability_score: float,
    high_need_stall_indicator: float,
    safe_threshold: float = 0.12,
    critical_threshold: float = 0.28,
) -> tuple[float, float, str]:
    charge_need_score = compute_charge_need_score(
        has_known_route=has_known_route,
        charge_margin_now=charge_margin_now,
        battery_ratio=battery_ratio,
        future_recoverability_score=future_recoverability_score,
    )
    battery_state = classify_battery_state(
        charge_need_score,
        safe_threshold=safe_threshold,
        critical_threshold=critical_threshold,
    )
    if battery_state == "safe":
        return 0.0, charge_need_score, battery_state

    if has_known_route:
        slack_term = clip01(-float(charger_slack) / 12.0)
        recoverability_term = clip01(-float(future_recoverability_score))
        battery_term = clip01((0.35 - float(battery_ratio)) / 0.20)
        raw_battery_risk = (
            0.48 * slack_term
            + 0.22 * recoverability_term
            + 0.10 * battery_term
            + 0.20 * clip01(high_need_stall_indicator)
        )
        confidence_weight = clip01(slack_confidence)
    else:
        battery_term = clip01((0.30 - float(battery_ratio)) / 0.18)
        recoverability_term = clip01(-float(future_recoverability_score))
        charger_unknown_term = 1.0
        raw_battery_risk = 0.30 * battery_term + 0.20 * recoverability_term + 0.50 * charger_unknown_term
        confidence_weight = 0.70

    zone_weight = 0.70 if battery_state == "planning" else 1.0
    return zone_weight * confidence_weight * raw_battery_risk, charge_need_score, battery_state


def classify_battery_fail_severity(
    *,
    fail_reason: str,
    finished_steps: float,
    max_step: float,
    clean_per_step: float,
    all_charger_known_path_count: float,
    avg_unknown_on_target_path_ratio: float,
    remaining_charge: float,
    charge_count: float = 0.0,
) -> tuple[str | None, float]:
    if str(fail_reason) != "battery":
        return None, 0.0

    max_step = max(float(max_step), 1.0)
    progress_ratio = float(finished_steps) / max_step
    known_paths = float(all_charger_known_path_count)
    unknown_ratio = float(avg_unknown_on_target_path_ratio)
    clean_per_step = float(clean_per_step)
    remaining_charge = float(remaining_charge)
    charge_count = float(charge_count)

    if progress_ratio < 0.60 and (known_paths < 1.0 or unknown_ratio > 0.50):
        return "early_unrecoverable", 1.0
    if progress_ratio >= 0.94 and clean_per_step >= 0.72 and charge_count >= 1.0 and remaining_charge <= 0.0:
        return "late_near_completion", 0.25
    return "mid_recoverability_loss", 0.6


def compute_collision_process_cost_step(collision_risk_label: float, scale: float = 0.15) -> float:
    return float(scale) * clip01(collision_risk_label)
