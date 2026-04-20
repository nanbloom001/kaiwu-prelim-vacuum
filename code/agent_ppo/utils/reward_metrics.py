#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Reward contribution helpers for window-level comparison.
"""

from __future__ import annotations

from typing import Mapping


POSITIVE_REWARD_KEYS = (
    "cleaning",
    "streak",
    "explore",
    "charge_route_progress_bonus",
    "return_progress_shaping_bonus",
    "necessary_charge_bonus",
    "frontier",
    "cps_bonus",
    "edge_follow_bonus",
    "charger_access_discovery_bonus",
    "charger_access_probe_bonus",
)

NEGATIVE_REWARD_KEYS = (
    "charge_detour_cost",
    "charge_interrupt_cost",
    "skip_needed_charge_penalty",
    "high_need_return_stall_penalty",
    "unnecessary_charge_penalty",
    "planner_alignment",
    "idle",
    "npc",
    "coverage_tangle_penalty",
)

CHARGING_POSITIVE_REWARD_KEYS = (
    "charge_route_progress_bonus",
    "return_progress_shaping_bonus",
    "necessary_charge_bonus",
    "charger_access_discovery_bonus",
    "charger_access_probe_bonus",
)

CHARGING_NEGATIVE_REWARD_KEYS = (
    "charge_detour_cost",
    "charge_interrupt_cost",
    "skip_needed_charge_penalty",
    "high_need_return_stall_penalty",
    "unnecessary_charge_penalty",
)


def _safe_float(mapping: Mapping[str, float], key: str) -> float:
    try:
        return float(mapping.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def compute_reward_contribution_payload(avg_reward_values: Mapping[str, float]) -> dict[str, float]:
    positive_total = sum(max(_safe_float(avg_reward_values, key), 0.0) for key in POSITIVE_REWARD_KEYS)
    negative_total = sum(max(-_safe_float(avg_reward_values, key), 0.0) for key in NEGATIVE_REWARD_KEYS)
    charging_positive_total = sum(max(_safe_float(avg_reward_values, key), 0.0) for key in CHARGING_POSITIVE_REWARD_KEYS)
    charging_negative_total = sum(max(-_safe_float(avg_reward_values, key), 0.0) for key in CHARGING_NEGATIVE_REWARD_KEYS)

    payload: dict[str, float] = {
        "reward_positive_total": positive_total,
        "reward_negative_total": negative_total,
        "reward_net_total": positive_total - negative_total,
        "reward_charging_positive_total": charging_positive_total,
        "reward_charging_negative_total": charging_negative_total,
        "reward_charging_net_total": charging_positive_total - charging_negative_total,
    }

    for key in POSITIVE_REWARD_KEYS:
        value = max(_safe_float(avg_reward_values, key), 0.0)
        payload[f"reward_positive_share_{key}"] = (value / positive_total) if positive_total > 1e-9 else 0.0

    for key in NEGATIVE_REWARD_KEYS:
        value = max(-_safe_float(avg_reward_values, key), 0.0)
        payload[f"reward_negative_share_{key}"] = (value / negative_total) if negative_total > 1e-9 else 0.0

    for key in CHARGING_POSITIVE_REWARD_KEYS:
        value = max(_safe_float(avg_reward_values, key), 0.0)
        payload[f"reward_charging_positive_share_{key}"] = (
            value / charging_positive_total if charging_positive_total > 1e-9 else 0.0
        )

    for key in CHARGING_NEGATIVE_REWARD_KEYS:
        value = max(-_safe_float(avg_reward_values, key), 0.0)
        payload[f"reward_charging_negative_share_{key}"] = (
            value / charging_negative_total if charging_negative_total > 1e-9 else 0.0
        )

    return payload
