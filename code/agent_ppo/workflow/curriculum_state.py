#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Shared curriculum state management for multi-helper aisrv training.

This module turns curriculum advancement from a per-helper local counter into a
shared state driven by globally aggregated helper signals.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_ppo.workflow.curriculum_policy import (
    STAGE_INDEX,
    choose_stage,
    previous_stage,
    should_regress_stage,
    snapshot_stage_entry_metrics,
    stage_progress,
)

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


SIGNAL_TTL_SECONDS = 20 * 60
STATE_VERSION = 1
FAST_SKIP_GLOBAL_EPISODES = 20
FULL_WINDOW_GLOBAL_EPISODES = 40
ADVANCE_CONFIRM_WINDOWS = 2
REGRESS_CONFIRM_WINDOWS = 2
MIN_STAGE_DWELL_STEPS = {
    "warmup": 3000,
    "blend": 5000,
    "robust": 8000,
    "eval_hard": 0,
}


def _now_ts() -> float:
    return float(time.time())


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _metric(payload: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if not payload:
        return float(default)
    try:
        value = payload.get(key, default)
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _weighted_average(signals: list[dict[str, Any]], key: str) -> float:
    weighted = 0.0
    total_weight = 0.0
    for signal in signals:
        metrics = signal.get("window_metrics") or {}
        count = float(metrics.get("_count", 0.0))
        if count <= 0:
            continue
        if key not in metrics or metrics.get(key) is None:
            continue
        weighted += float(metrics[key]) * count
        total_weight += count
    if total_weight <= 0:
        return 0.0
    return weighted / total_weight


def _weighted_average_from(signals: list[dict[str, Any]], field_name: str, metric_name: str) -> float:
    weighted = 0.0
    total_weight = 0.0
    for signal in signals:
        metrics = signal.get(field_name) or {}
        count = float(metrics.get("_count", 0.0))
        if count <= 0:
            continue
        value = metrics.get(metric_name)
        if value is None:
            continue
        weighted += float(value) * count
        total_weight += count
    if total_weight <= 0:
        return 0.0
    return weighted / total_weight


def _weighted_ratio(signals: list[dict[str, Any]], metric_name: str, default: float = -1.0) -> float:
    weighted = 0.0
    total_weight = 0.0
    for signal in signals:
        metrics = signal.get("window_metrics") or {}
        count = float(metrics.get("_count", 0.0))
        value = metrics.get(metric_name, default)
        if count <= 0 or value is None or float(value) < 0:
            continue
        weighted += float(value) * count
        total_weight += count
    if total_weight <= 0:
        return float(default)
    return weighted / total_weight


def _aggregate_metrics(signals: list[dict[str, Any]], field_name: str, min_episode_count: int) -> dict[str, Any] | None:
    active = [signal for signal in signals if (signal.get(field_name) or {}).get("_count", 0) > 0]
    total_count = sum(int((signal.get(field_name) or {}).get("_count", 0)) for signal in active)
    if total_count < min_episode_count:
        return None

    keys = [
        "win_rate",
        "avg_clean_score",
        "avg_finished_steps",
        "avg_charge_count",
        "avg_remaining_charge",
        "avg_invalid_move_rate",
        "avg_charge_efficiency",
        "avg_clean_per_step",
        "avg_expert_weight",
        "late_return_rate",
        "late_contract_rate",
        "anchor_switch_rate",
        "target_switch_rate",
        "diag_rate_all",
        "diag_rate_contract",
        "diag_rate_return",
        "return_progress_per_step",
        "return_efficiency_ratio",
        "return_stall_rate",
        "recoverability_score_avg",
        "recoverability_violation_rate",
        "wall_hugging_clean_floor_rate",
        "stale_boundary_follow_rate",
        "narrow_unknown_commit_rate",
        "missed_charge_opportunity_rate",
        "charger_nearby_not_charged_rate",
        "suboptimal_target_hold_rate",
        "planner_policy_divergence_rate",
        "avg_path_cross_count_50",
        "avg_coverage_efficiency_20",
        "avg_all_charger_known_path_count",
        "avg_unknown_on_target_path_ratio",
        "mode_usage_depart",
        "mode_usage_expand",
        "mode_usage_harvest",
        "mode_usage_contract",
        "mode_usage_return",
        "mode_usage_evade",
        "battery_fail_rate",
        "collision_fail_rate",
        "cps_win",
        "avg_charge_count_win",
        "avg_clean_score_win",
    ]
    payload: dict[str, Any] = {"_count": total_count}
    for key in keys:
        payload[key] = _weighted_average_from(active, field_name, key)
    payload["anchor_win_rate"] = _weighted_ratio(active, "anchor_win_rate")
    payload["mild_win_rate"] = _weighted_ratio(active, "mild_win_rate")
    payload["broad_win_rate"] = _weighted_ratio(active, "broad_win_rate")
    return payload


def _latest_learning_metrics(signals: list[dict[str, Any]]) -> dict[str, Any]:
    if not signals:
        return {}
    latest = max(signals, key=lambda signal: float(signal.get("updated_at_ts", 0.0)))
    return deepcopy(latest.get("learning_metrics") or {})


def _default_state() -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "stage": "warmup",
        "stage_version": 1,
        "updated_at_ts": 0.0,
        "entered_global_step": 0,
        "entered_wall_clock_ts": 0.0,
        "consecutive_pass_windows": 0,
        "consecutive_fail_windows": 0,
        "stage_entry_metrics": {},
        "last_global_metrics": {},
        "last_bootstrap_metrics": {},
        "last_learning_metrics": {},
        "global_episode_count": 0,
        "source_session_id": None,
    }


@dataclass
class SharedCurriculumStateStore:
    code_dir: Path

    def __post_init__(self):
        self.signal_dir = self.code_dir / "curriculum_signals"
        self.signal_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.code_dir / "curriculum_state.json"
        self.lock_path = self.code_dir / ".curriculum_state.lock"

    def signal_path(self, source_id: str) -> Path:
        safe = source_id.replace("/", "_").replace(":", "_")
        return self.signal_dir / f"{safe}.json"

    def write_signal(self, source_id: str, payload: dict[str, Any]) -> None:
        record = {
            "source_id": source_id,
            "updated_at_ts": _now_ts(),
            **deepcopy(payload),
        }
        _write_json(self.signal_path(source_id), record)

    def read_state(self) -> dict[str, Any]:
        return _read_json(self.state_path) or _default_state()

    def refresh_state(self) -> dict[str, Any]:
        if fcntl is None:
            return self._refresh_state_impl()
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                return self._refresh_state_impl()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _refresh_state_impl(self) -> dict[str, Any]:
        state = self.read_state()
        now = _now_ts()
        signals = []
        for path in self.signal_dir.glob("*.json"):
            signal = _read_json(path)
            if not signal:
                continue
            if now - float(signal.get("updated_at_ts", 0.0)) > SIGNAL_TTL_SECONDS:
                continue
            signals.append(signal)

        if not signals:
            state["updated_at_ts"] = now
            _write_json(self.state_path, state)
            return state

        bootstrap_metrics = _aggregate_metrics(signals, "bootstrap_metrics", FAST_SKIP_GLOBAL_EPISODES)
        window_metrics = _aggregate_metrics(signals, "window_metrics", FULL_WINDOW_GLOBAL_EPISODES)
        learning_metrics = _latest_learning_metrics(signals)
        global_step_since_resume = int(max(_metric(signal.get("runtime"), "global_step_since_resume", 0.0) for signal in signals))
        global_episode_count = int(sum(int((signal.get("window_metrics") or {}).get("_count", 0)) for signal in signals))
        resume_fast_track = bool(state.get("stage_version", 0) <= 1)

        context = {
            "global_step_since_resume": global_step_since_resume,
            "window_metrics": window_metrics,
            "bootstrap_metrics": bootstrap_metrics,
            "learning_metrics": learning_metrics,
            "resume_fast_track": resume_fast_track,
        }

        current_stage = state.get("stage", "warmup")
        proposed_stage = choose_stage(
            current_stage=current_stage,
            context=context,
            stage_entry_metrics=state.get("stage_entry_metrics"),
        )

        dwell_requirement = MIN_STAGE_DWELL_STEPS.get(current_stage, 0)
        dwell_satisfied = global_step_since_resume - int(state.get("entered_global_step", 0)) >= dwell_requirement

        if proposed_stage != current_stage and dwell_satisfied:
            pass_windows = int(state.get("consecutive_pass_windows", 0))
            if state.get("pending_stage") == proposed_stage:
                pass_windows += 1
            else:
                pass_windows = 1
            state["pending_stage"] = proposed_stage
            state["consecutive_pass_windows"] = pass_windows
            if pass_windows >= ADVANCE_CONFIRM_WINDOWS:
                current_stage = proposed_stage
                state["stage"] = current_stage
                state["stage_version"] = int(state.get("stage_version", 0)) + 1
                state["entered_global_step"] = global_step_since_resume
                state["entered_wall_clock_ts"] = now
                state["stage_entry_metrics"] = snapshot_stage_entry_metrics(window_metrics, learning_metrics)
                state["consecutive_pass_windows"] = 0
                state["pending_stage"] = None
        else:
            state["consecutive_pass_windows"] = 0
            state["pending_stage"] = None

        if should_regress_stage(
            current_stage=current_stage,
            stage_entry_metrics=state.get("stage_entry_metrics"),
            current_metrics=window_metrics,
            learning_metrics=learning_metrics,
        ):
            fail_windows = int(state.get("consecutive_fail_windows", 0)) + 1
            state["consecutive_fail_windows"] = fail_windows
            if fail_windows >= REGRESS_CONFIRM_WINDOWS and current_stage != "warmup":
                current_stage = previous_stage(current_stage)
                state["stage"] = current_stage
                state["stage_version"] = int(state.get("stage_version", 0)) + 1
                state["entered_global_step"] = global_step_since_resume
                state["entered_wall_clock_ts"] = now
                state["stage_entry_metrics"] = snapshot_stage_entry_metrics(window_metrics, learning_metrics)
                state["consecutive_fail_windows"] = 0
        else:
            state["consecutive_fail_windows"] = 0

        metrics_for_progress = window_metrics or bootstrap_metrics
        state["curriculum_stage_idx"] = STAGE_INDEX.get(state["stage"], 0)
        state["curriculum_progress"] = round(stage_progress(state["stage"], metrics_for_progress, learning_metrics), 4)
        state["last_global_metrics"] = deepcopy(window_metrics or {})
        state["last_bootstrap_metrics"] = deepcopy(bootstrap_metrics or {})
        state["last_learning_metrics"] = deepcopy(learning_metrics or {})
        state["global_episode_count"] = global_episode_count
        state["updated_at_ts"] = now
        _write_json(self.state_path, state)
        return state
