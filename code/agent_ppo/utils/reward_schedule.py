#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Reward shaping schedules used to stabilize early training windows.
"""

from __future__ import annotations

from agent_ppo.conf.conf import Config


def _safe_step(global_step: float | int | None) -> int:
    try:
        return max(int(global_step or 0), 0)
    except (TypeError, ValueError):
        return 0


def _lerp(start: float, end: float, factor: float) -> float:
    factor = min(max(float(factor), 0.0), 1.0)
    return float(start + (end - start) * factor)


def _anneal(global_step: int, peak: float, final: float) -> float:
    warm_steps = max(int(Config.REWARD_SCHEDULE_WARM_STEPS), 0)
    total_steps = max(int(Config.REWARD_SCHEDULE_TOTAL_STEPS), warm_steps + 1)
    if not bool(Config.REWARD_SCHEDULE_ENABLED):
        return float(final)
    if global_step <= warm_steps:
        return float(peak)
    if global_step >= total_steps:
        return float(final)
    factor = (global_step - warm_steps) / max(total_steps - warm_steps, 1)
    return _lerp(float(peak), float(final), factor)


def get_reward_schedule(global_step: float | int | None) -> dict[str, float]:
    step = _safe_step(global_step)
    necessary_scale = _anneal(
        step,
        peak=Config.NECESSARY_CHARGE_BONUS_SCALE_PEAK,
        final=Config.NECESSARY_CHARGE_BONUS_SCALE,
    )
    battery_fail_scale = _anneal(
        step,
        peak=Config.BATTERY_FAIL_TASK_REWARD_SCALE_PEAK,
        final=Config.BATTERY_FAIL_TASK_REWARD_SCALE,
    )
    early_battery_fail_scale = _anneal(
        step,
        peak=Config.EARLY_BATTERY_FAIL_TASK_REWARD_SCALE_PEAK,
        final=Config.EARLY_BATTERY_FAIL_TASK_REWARD_SCALE,
    )
    return {
        "scheduled_necessary_charge_bonus_scale": necessary_scale,
        "scheduled_battery_fail_task_reward_scale": battery_fail_scale,
        "scheduled_early_battery_fail_task_reward_scale": early_battery_fail_scale,
    }
