#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import annotations

import os


SIMPLIFIED_CONTROL_STACK_PHASES = {
    "s1_survival_control_simplify_v1",
    "s1_survival_cps_align_v1",
}

CPS_ALIGN_PHASES = {"s1_survival_cps_align_v1"}


def control_stack_simplify_active(train_phase: str | None = None) -> bool:
    phase = train_phase
    if phase is None:
        phase = str(os.getenv("KAIWU_TRAIN_PHASE", "") or "").strip().lower()
    else:
        phase = str(train_phase or "").strip().lower()
    return phase in SIMPLIFIED_CONTROL_STACK_PHASES


def cps_align_active(train_phase: str | None = None) -> bool:
    phase = train_phase
    if phase is None:
        phase = str(os.getenv("KAIWU_TRAIN_PHASE", "") or "").strip().lower()
    else:
        phase = str(train_phase or "").strip().lower()
    return phase in CPS_ALIGN_PHASES


def evaluate_simplified_return_readiness(
    *,
    charger_slack: float,
    battery_ratio: float,
    future_recoverability_score: float,
    planner_multi_route_recoverability: float,
    margin: float,
    route_contract_pressure: float,
    known_path_count: int,
    total_charger: int,
    planner_topk_reachable_count: int,
    unknown_ratio: float,
    return_slack_threshold: float,
    return_battery_ratio: float,
    return_recoverability_threshold: float,
    prepare_return_slack_threshold: float,
    contract_battery_ratio: float,
    contract_recoverability_threshold: float,
    charge_margin_low: float,
    charge_margin_warn: float,
    contract_route_pressure_threshold: float,
    unknown_path_risk_battery_ratio: float,
    unknown_path_risk_threshold: float,
) -> dict[str, float | bool | int]:
    min_recoverability = min(
        float(future_recoverability_score),
        float(planner_multi_route_recoverability),
    )
    return_now = bool(
        float(charger_slack) <= float(return_slack_threshold)
        or float(battery_ratio) <= float(return_battery_ratio)
        or min_recoverability <= float(return_recoverability_threshold)
        or float(margin) <= float(charge_margin_low)
    )
    contract_hard_risk = bool(
        int(known_path_count) < min(int(total_charger), 2)
        and int(planner_topk_reachable_count) <= 0
        and float(battery_ratio) <= float(unknown_path_risk_battery_ratio)
        and float(unknown_ratio) >= float(unknown_path_risk_threshold)
    )
    primary_hits = 0
    primary_hits += int(float(charger_slack) <= float(prepare_return_slack_threshold))
    primary_hits += int(float(battery_ratio) <= float(contract_battery_ratio))
    primary_hits += int(min_recoverability <= float(contract_recoverability_threshold))
    primary_hits += int(float(margin) <= float(charge_margin_warn))
    route_pressure_hit = int(float(route_contract_pressure) >= float(contract_route_pressure_threshold))
    pre_return_ready = bool(
        contract_hard_risk
        or primary_hits >= 2
        or (primary_hits >= 1 and route_pressure_hit >= 1)
    )
    return {
        "return_now": return_now,
        "pre_return_ready": pre_return_ready,
        "primary_hits": primary_hits,
        "route_pressure_hit": route_pressure_hit,
        "hard_risk": contract_hard_risk,
        "min_recoverability": min_recoverability,
    }
