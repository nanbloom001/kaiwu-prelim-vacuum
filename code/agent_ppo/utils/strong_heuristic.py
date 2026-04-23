#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import annotations

import os


SLICE1_PHASE = "s1_survival_strong_heuristic_v1"
SLICE2A_PHASE = "s1_survival_strong_heuristic_slice2a_v1"
SLICE2A_FIXED8_PHASE = "s1_survival_strong_heuristic_slice2a_fixed8_v1"

SLICE2A_PHASES = {SLICE2A_PHASE, SLICE2A_FIXED8_PHASE}
STRONG_HEURISTIC_PHASES = {SLICE1_PHASE, *SLICE2A_PHASES}

LOGICAL_MODE_CLEAN = "clean"
LOGICAL_MODE_PRE_RETURN = "pre_return"
LOGICAL_MODE_RETURN = "return"
LOGICAL_MODE_EVADE = "evade"


def strong_heuristic_active(train_phase: str | None = None) -> bool:
    phase = train_phase
    if phase is None:
        phase = str(os.getenv("KAIWU_TRAIN_PHASE", "") or "").strip().lower()
    else:
        phase = str(train_phase or "").strip().lower()
    return phase in STRONG_HEURISTIC_PHASES


def strong_heuristic_slice2a_active(train_phase: str | None = None) -> bool:
    phase = train_phase
    if phase is None:
        phase = str(os.getenv("KAIWU_TRAIN_PHASE", "") or "").strip().lower()
    else:
        phase = str(train_phase or "").strip().lower()
    return phase in SLICE2A_PHASES


def logical_mode_to_training_mode(logical_mode: str, prep) -> int:
    mapping = {
        LOGICAL_MODE_CLEAN: getattr(prep, "MODE_EXPAND", 1),
        LOGICAL_MODE_PRE_RETURN: getattr(prep, "MODE_CONTRACT", 3),
        LOGICAL_MODE_RETURN: getattr(prep, "MODE_RETURN", 4),
        LOGICAL_MODE_EVADE: getattr(prep, "MODE_EVADE", 5),
    }
    return int(mapping.get(logical_mode, getattr(prep, "MODE_EXPAND", 1)))


def logical_mode_to_teacher_name(logical_mode: str) -> str:
    mapping = {
        LOGICAL_MODE_CLEAN: "expand",
        LOGICAL_MODE_PRE_RETURN: "contract",
        LOGICAL_MODE_RETURN: "return",
        LOGICAL_MODE_EVADE: "evade",
    }
    return mapping.get(logical_mode, "expand")


def evaluate_strong_heuristic_mode(
    *,
    current_mode: int,
    mode_return: int,
    nearest_npc_dist: float,
    on_charger: bool,
    battery: float,
    battery_ratio: float,
    charger_dist: float,
    charger_slack: float,
    margin: float,
    future_recoverability_score: float,
    planner_multi_route_recoverability: float,
    known_path_count: int,
    total_charger: int,
    unknown_ratio: float,
    route_contract_pressure: float,
    return_battery_ratio: float,
    return_slack_threshold: float,
    return_exit_battery_ratio: float,
    pre_return_battery_ratio: float,
    pre_return_slack_threshold: float,
    pre_return_recoverability_threshold: float,
    pre_return_unknown_ratio_threshold: float,
    pre_return_route_pressure_threshold: float,
    charge_margin_warn: float,
    evade_npc_distance: float,
) -> dict[str, float | str | bool]:
    min_recoverability = min(
        float(future_recoverability_score),
        float(planner_multi_route_recoverability),
    )

    if float(nearest_npc_dist) <= float(evade_npc_distance):
        return {
            "logical_mode": LOGICAL_MODE_EVADE,
            "hysteresis_active": False,
            "return_exit_ready": False,
            "min_recoverability": min_recoverability,
        }

    return_exit_ready = bool(on_charger and float(battery_ratio) >= float(return_exit_battery_ratio))
    if int(current_mode) == int(mode_return) and not return_exit_ready:
        return {
            "logical_mode": LOGICAL_MODE_RETURN,
            "hysteresis_active": True,
            "return_exit_ready": False,
            "min_recoverability": min_recoverability,
        }

    if (
        float(battery_ratio) <= float(return_battery_ratio)
        or float(charger_slack) <= float(return_slack_threshold)
        or float(battery) <= float(charger_dist) + float(margin)
    ):
        return {
            "logical_mode": LOGICAL_MODE_RETURN,
            "hysteresis_active": False,
            "return_exit_ready": return_exit_ready,
            "min_recoverability": min_recoverability,
        }

    if (
        float(battery_ratio) <= float(pre_return_battery_ratio)
        or float(charger_slack) <= float(pre_return_slack_threshold)
        or min_recoverability <= float(pre_return_recoverability_threshold)
        or (
            int(known_path_count) < min(max(int(total_charger), 1), 2)
            and float(unknown_ratio) >= float(pre_return_unknown_ratio_threshold)
        )
        or float(route_contract_pressure) >= float(pre_return_route_pressure_threshold)
        or float(margin) <= float(charge_margin_warn)
    ):
        return {
            "logical_mode": LOGICAL_MODE_PRE_RETURN,
            "hysteresis_active": False,
            "return_exit_ready": return_exit_ready,
            "min_recoverability": min_recoverability,
        }

    return {
        "logical_mode": LOGICAL_MODE_CLEAN,
        "hysteresis_active": False,
        "return_exit_ready": return_exit_ready,
        "min_recoverability": min_recoverability,
    }
