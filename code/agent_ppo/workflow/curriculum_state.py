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
import math
import os
import re
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_ppo.conf.conf import Config
from agent_ppo.utils.reward_metrics import compute_reward_contribution_payload
from agent_ppo.workflow.preload_checkpoint import (
    RESUME_CURRICULUM_SNAPSHOT_FILE,
    RESUME_LATEST_STATE_FILE,
)
from agent_ppo.workflow.state_layout import (
    RESUME_STATE_FILE,
    allow_legacy_resume_import,
    ensure_runtime_state_dirs,
    is_scratch_mode,
    legacy_curriculum_state_path,
    legacy_resume_curriculum_snapshot_path,
    legacy_resume_latest_state_path,
)
from agent_ppo.workflow.curriculum_policy import (
    STAGE_INDEX,
    choose_stage,
    choose_stage_decision,
    profile_plan_for_runtime,
    previous_stage,
    stagnation_status,
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
RECENT_EPISODE_KEEP = FULL_WINDOW_GLOBAL_EPISODES * 3
ADVANCE_CONFIRM_WINDOWS = 2
REGRESS_CONFIRM_WINDOWS = 2
MIN_STAGE_DWELL_STEPS = {
    "warmup": 3000,
    "blend": 5000,
    "robust": 8000,
    "eval_hard": 0,
}

RETURN_WINDOW_ALIAS_KEYS = (
    ("return_progress_per_step", "avg_return_progress_per_step"),
    ("return_efficiency_ratio", "avg_return_efficiency_ratio"),
    ("high_need_return_stall_rate", "avg_high_need_return_stall_rate"),
)

_LEARNER_LOG_STATE_PATTERNS = {
    "global_step": re.compile(r"global step is\s+(-?\d+(?:\.\d+)?)"),
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


def _profile_win_rate(records: list[dict[str, Any]], profiles: list[str]) -> float:
    subset = [record for record in records if record.get("profile") in profiles]
    if not subset:
        return -1.0
    return sum(1 for record in subset if record.get("result") == "completed") / len(subset)


def _aggregate_episode_records(records: list[dict[str, Any]], min_episode_count: int) -> dict[str, Any] | None:
    if len(records) < min_episode_count:
        return None

    def avg(key: str, default: float = 0.0) -> float:
        values = []
        for record in records:
            value = record.get(key, default)
            if value is None:
                continue
            values.append(float(value))
        if not values:
            return float(default)
        return sum(values) / len(values)

    wins = [record for record in records if record.get("result") == "completed"]
    battery_fails = [record for record in records if record.get("result") == "battery"]
    charged = [record for record in records if record.get("clean_per_charge_when_charged") is not None]
    reward_component_means = {
        f"avg_reward_{key}": avg(f"avg_reward_{key}")
        for key in (
            "cleaning",
            "streak",
            "explore",
            "frontier",
            "charger_access_discovery_bonus",
            "charger_access_probe_bonus",
            "idle",
            "npc",
            "planner_alignment",
            "charge_route_progress_bonus",
            "return_progress_shaping_bonus",
            "necessary_charge_bonus",
            "unnecessary_charge_penalty",
            "charge_detour_cost",
            "charge_interrupt_cost",
            "skip_needed_charge_penalty",
            "high_need_return_stall_penalty",
            "cps_bonus",
            "coverage_tangle_penalty",
            "edge_follow_bonus",
        )
    }
    payload = {
        "_count": len(records),
        "win_rate": sum(1 for record in records if record.get("result") == "completed") / len(records),
        "battery_fail_count": sum(1 for record in records if record.get("result") == "battery"),
        "battery_positive_reward_count": sum(
            1
            for record in records
            if record.get("result") == "battery" and float(record.get("total_reward", 0.0) or 0.0) > 0.0
        ),
        "avg_clean_score": avg("clean_score"),
        "avg_finished_steps": avg("finished_steps"),
        "avg_charge_count": avg("charge_count"),
        "avg_remaining_charge": avg("remaining_charge"),
        "avg_invalid_move_rate": avg("invalid_move_rate"),
        "avg_charge_efficiency": avg("charge_efficiency"),
        "avg_clean_per_charge_when_charged": (
            sum(float(record.get("clean_per_charge_when_charged", 0.0)) for record in charged) / len(charged)
        ) if charged else 0.0,
        "avg_clean_per_step": avg("clean_per_step"),
        "avg_expert_weight": avg("expert_weight"),
        "late_return_rate": avg("late_return_rate"),
        "late_contract_rate": avg("late_contract_rate"),
        "anchor_switch_rate": avg("anchor_switch_rate"),
        "target_switch_rate": avg("target_switch_rate"),
        "diag_rate_all": avg("diag_rate_all"),
        "diag_rate_contract": avg("diag_rate_contract"),
        "diag_rate_return": avg("diag_rate_return"),
        "return_progress_per_step": avg("return_progress_per_step"),
        "return_efficiency_ratio": avg("return_efficiency_ratio"),
        "return_stall_rate": avg("return_stall_rate"),
        "recoverability_score_avg": avg("recoverability_score_avg"),
        "recoverability_violation_rate": avg("recoverability_violation_rate"),
        "wall_hugging_clean_floor_rate": avg("wall_hugging_clean_floor_rate"),
        "stale_boundary_follow_rate": avg("stale_boundary_follow_rate"),
        "narrow_unknown_commit_rate": avg("narrow_unknown_commit_rate"),
        "missed_charge_opportunity_rate": avg("missed_charge_opportunity_rate"),
        "charger_nearby_not_charged_rate": avg("charger_nearby_not_charged_rate"),
        "suboptimal_target_hold_rate": avg("suboptimal_target_hold_rate"),
        "planner_policy_divergence_rate": avg("planner_policy_divergence_rate"),
        "avg_path_cross_count_50": avg("avg_path_cross_count_50"),
        "avg_coverage_efficiency_20": avg("avg_coverage_efficiency_20"),
        "avg_all_charger_known_path_count": avg("avg_all_charger_known_path_count"),
        "avg_unknown_on_target_path_ratio": avg("avg_unknown_on_target_path_ratio"),
        "avg_planner_topk_reachable_count": avg("avg_planner_topk_reachable_count"),
        "avg_planner_known_route_count_total": avg("avg_planner_known_route_count_total"),
        "avg_planner_best_target_route_diversity": avg("avg_planner_best_target_route_diversity"),
        "avg_planner_best_target_tangle_cost": avg("avg_planner_best_target_tangle_cost"),
        "avg_planner_best_target_edge_break_cost": avg("avg_planner_best_target_edge_break_cost"),
        "avg_planner_best_target_region_fragment_cost": avg("avg_planner_best_target_region_fragment_cost"),
        "avg_planner_multi_route_recoverability": avg("avg_planner_multi_route_recoverability"),
        "battery_process_cost_mean": avg("battery_process_cost_mean"),
        "collision_process_cost_mean": avg("collision_process_cost_mean"),
        "high_need_return_stall_rate": avg("high_need_return_stall_rate"),
        "avg_charge_need_score": avg("avg_charge_need_score"),
        "avg_slack_confidence": avg("avg_slack_confidence"),
        "battery_fail_severity_mean": avg("battery_fail_severity"),
        "mode_usage_depart": avg("mode_usage_depart"),
        "mode_usage_expand": avg("mode_usage_expand"),
        "mode_usage_harvest": avg("mode_usage_harvest"),
        "mode_usage_contract": avg("mode_usage_contract"),
        "mode_usage_return": avg("mode_usage_return"),
        "mode_usage_evade": avg("mode_usage_evade"),
        **reward_component_means,
        "battery_fail_rate": sum(1 for record in records if record.get("result") == "battery") / len(records),
        "collision_fail_rate": sum(1 for record in records if record.get("result") == "collision") / len(records),
        "zero_charge_battery_fail_rate": (
            sum(1 for record in battery_fails if float(record.get("charge_count", 0.0) or 0.0) <= 0.0) / len(battery_fails)
        ) if battery_fails else 0.0,
        "battery_positive_reward_rate": (
            sum(
                1
                for record in battery_fails
                if float(record.get("total_reward", 0.0) or 0.0) > 0.0
            ) / len(battery_fails)
        ) if battery_fails else 0.0,
        "cps_win": sum(float(record.get("clean_per_step", 0.0)) for record in wins) / len(wins) if wins else 0.0,
        "avg_charge_count_win": sum(float(record.get("charge_count", 0.0)) for record in wins) / len(wins) if wins else 0.0,
        "avg_clean_score_win": sum(float(record.get("clean_score", 0.0)) for record in wins) / len(wins) if wins else 0.0,
        "avg_charge_count_completed": sum(float(record.get("charge_count", 0.0)) for record in wins) / len(wins) if wins else 0.0,
        "avg_charge_count_battery_fail": (
            sum(float(record.get("charge_count", 0.0)) for record in battery_fails) / len(battery_fails)
        ) if battery_fails else 0.0,
        "anchor_win_rate": _profile_win_rate(records, ["anchor"]),
        "mild_win_rate": _profile_win_rate(records, ["mild"]),
        "broad_win_rate": _profile_win_rate(records, ["broad", "broad_eval"]),
    }
    payload.update(compute_reward_contribution_payload({
        key.replace("avg_reward_", ""): value for key, value in reward_component_means.items()
    }))
    for source_key, alias_key in RETURN_WINDOW_ALIAS_KEYS:
        payload[alias_key] = payload.get(source_key)
    return payload


def _merge_recent_episodes(state: dict[str, Any], signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for record in state.get("recent_episodes") or []:
        key = record.get("episode_key")
        if key:
            merged[str(key)] = deepcopy(record)

    for signal in signals:
        source_id = str(signal.get("source_id", "unknown"))
        signal_ts = float(signal.get("updated_at_ts", 0.0))
        for episode in signal.get("recent_episode_metrics") or []:
            local_ep = int(episode.get("episode_cnt_local") or 0)
            if local_ep <= 0:
                continue
            episode_key = f"{source_id}:{local_ep}"
            candidate = deepcopy(episode)
            candidate["episode_key"] = episode_key
            candidate["source_id"] = source_id
            candidate["completed_at_ts"] = float(candidate.get("completed_at_ts") or signal_ts)
            previous = merged.get(episode_key)
            if previous is None or float(candidate["completed_at_ts"]) >= float(previous.get("completed_at_ts", 0.0)):
                merged[episode_key] = candidate

    records = sorted(
        merged.values(),
        key=lambda item: (float(item.get("completed_at_ts", 0.0)), int(item.get("episode_cnt_local", 0))),
    )
    return records[-RECENT_EPISODE_KEEP:]


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
        "avg_clean_per_charge_when_charged",
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
        "avg_planner_topk_reachable_count",
        "avg_planner_known_route_count_total",
        "avg_planner_best_target_route_diversity",
        "avg_planner_best_target_tangle_cost",
        "avg_planner_best_target_edge_break_cost",
        "avg_planner_best_target_region_fragment_cost",
        "avg_planner_multi_route_recoverability",
        "battery_process_cost_mean",
        "collision_process_cost_mean",
        "high_need_return_stall_rate",
        "avg_charge_need_score",
        "avg_slack_confidence",
        "battery_fail_severity_mean",
        "mode_usage_depart",
        "mode_usage_expand",
        "mode_usage_harvest",
        "mode_usage_contract",
        "mode_usage_return",
        "mode_usage_evade",
        "avg_reward_cleaning",
        "avg_reward_streak",
        "avg_reward_explore",
        "avg_reward_frontier",
        "avg_reward_charger_access_discovery_bonus",
        "avg_reward_charger_access_probe_bonus",
        "avg_reward_idle",
        "avg_reward_npc",
        "avg_reward_planner_alignment",
        "avg_reward_charge_route_progress_bonus",
        "avg_reward_return_progress_shaping_bonus",
        "avg_reward_necessary_charge_bonus",
        "avg_reward_unnecessary_charge_penalty",
        "avg_reward_charge_detour_cost",
        "avg_reward_charge_interrupt_cost",
        "avg_reward_skip_needed_charge_penalty",
        "avg_reward_high_need_return_stall_penalty",
        "avg_reward_cps_bonus",
        "avg_reward_coverage_tangle_penalty",
        "avg_reward_edge_follow_bonus",
        "battery_fail_rate",
        "collision_fail_rate",
        "zero_charge_battery_fail_rate",
        "battery_fail_count",
        "battery_positive_reward_count",
        "battery_positive_reward_rate",
        "cps_win",
        "avg_charge_count_win",
        "avg_clean_score_win",
        "avg_charge_count_completed",
        "avg_charge_count_battery_fail",
        "reward_positive_total",
        "reward_negative_total",
        "reward_net_total",
        "reward_charging_positive_total",
        "reward_charging_negative_total",
        "reward_charging_net_total",
        "reward_positive_share_cleaning",
        "reward_positive_share_streak",
        "reward_positive_share_explore",
        "reward_positive_share_charge_route_progress_bonus",
        "reward_positive_share_return_progress_shaping_bonus",
        "reward_positive_share_necessary_charge_bonus",
        "reward_positive_share_frontier",
        "reward_positive_share_cps_bonus",
        "reward_positive_share_edge_follow_bonus",
        "reward_positive_share_charger_access_discovery_bonus",
        "reward_positive_share_charger_access_probe_bonus",
        "reward_negative_share_charge_detour_cost",
        "reward_negative_share_charge_interrupt_cost",
        "reward_negative_share_skip_needed_charge_penalty",
        "reward_negative_share_high_need_return_stall_penalty",
        "reward_negative_share_unnecessary_charge_penalty",
        "reward_negative_share_planner_alignment",
        "reward_negative_share_idle",
        "reward_negative_share_npc",
        "reward_negative_share_coverage_tangle_penalty",
        "reward_charging_positive_share_charge_route_progress_bonus",
        "reward_charging_positive_share_return_progress_shaping_bonus",
        "reward_charging_positive_share_necessary_charge_bonus",
        "reward_charging_positive_share_charger_access_discovery_bonus",
        "reward_charging_positive_share_charger_access_probe_bonus",
        "reward_charging_negative_share_high_need_return_stall_penalty",
        "reward_charging_negative_share_charge_detour_cost",
        "reward_charging_negative_share_charge_interrupt_cost",
        "reward_charging_negative_share_skip_needed_charge_penalty",
        "reward_charging_negative_share_unnecessary_charge_penalty",
    ]
    payload: dict[str, Any] = {"_count": total_count}
    for key in keys:
        if key in {"battery_fail_count", "battery_positive_reward_count"}:
            payload[key] = sum(int((signal.get(field_name) or {}).get(key, 0) or 0) for signal in active)
        else:
            payload[key] = _weighted_average_from(active, field_name, key)
    payload["anchor_win_rate"] = _weighted_ratio(active, "anchor_win_rate")
    payload["mild_win_rate"] = _weighted_ratio(active, "mild_win_rate")
    payload["broad_win_rate"] = _weighted_ratio(active, "broad_win_rate")
    for source_key, alias_key in RETURN_WINDOW_ALIAS_KEYS:
        payload[alias_key] = payload.get(source_key)
    return payload


def _latest_learner_log_metrics(code_dir: Path) -> dict[str, Any]:
    learner_log_dirs = [
        code_dir.parent / "train" / "log" / "learner",
        code_dir.parent / "log" / "learner",
    ]
    candidates: list[Path] = []
    for learner_log_dir in learner_log_dirs:
        if not learner_log_dir.exists():
            continue
        candidates.extend(learner_log_dir.glob("learner_train_pid*_log_*.log"))
    candidates = sorted(candidates, key=lambda path: path.stat().st_mtime)
    if not candidates:
        return {}
    try:
        lines = candidates[-1].read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    found: dict[str, Any] = {}
    for line in reversed(lines):
        for key, pattern in _LEARNER_LOG_STATE_PATTERNS.items():
            if key in found:
                continue
            match = pattern.search(line)
            if not match:
                continue
            try:
                found[key] = float(match.group(1))
            except (TypeError, ValueError):
                continue
        if len(found) == len(_LEARNER_LOG_STATE_PATTERNS):
            break
    return found


def _merge_with_learner_log_metrics(base: dict[str, Any] | None, learner_log_metrics: dict[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(base or {})
    if not learner_log_metrics:
        return merged
    if _is_finite_number(learner_log_metrics.get("global_step")):
        log_step = float(learner_log_metrics["global_step"])
        current_step = merged.get("global_step")
        if not _is_finite_number(current_step) or float(current_step) < log_step:
            merged["global_step"] = log_step
    return merged


_LEARNING_METRIC_PRIORITY_KEYS = (
    "global_step",
    "entropy_loss",
    "value_clean_loss",
    "value_survive_loss",
    "lambda_battery",
    "lambda_collision",
    "battery_process_cost_mean",
    "collision_process_cost_mean",
    "nan_batch_count",
    "nan_skip_rate",
    "last_finite_step",
)


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _has_finite_learning_metric(payload: dict[str, Any] | None) -> bool:
    if not payload:
        return False
    return any(_is_finite_number(payload.get(key)) for key in _LEARNING_METRIC_PRIORITY_KEYS)


def _merge_learning_metrics(base: dict[str, Any] | None, candidate: dict[str, Any] | None) -> dict[str, Any]:
    merged = deepcopy(base or {})
    if not candidate:
        return merged
    for key, value in candidate.items():
        if isinstance(value, (int, float)) and _is_finite_number(value):
            merged[key] = value
            continue
        if value is None:
            continue
        if isinstance(value, str):
            merged[key] = value
    return merged


def _best_learning_metrics(
    signals: list[dict[str, Any]],
    current_session_id: str | None,
    previous_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not signals:
        return deepcopy(previous_metrics or {})
    best_signal = None
    best_rank = None
    for signal in signals:
        payload = signal.get("learning_metrics") or {}
        session_match = 1 if current_session_id and str(signal.get("session_id")) == str(current_session_id) else 0
        has_finite = 1 if _has_finite_learning_metric(payload) else 0
        try:
            global_step = float(payload.get("global_step")) if payload.get("global_step") is not None else -1.0
        except (TypeError, ValueError):
            global_step = -1.0
        updated_at = float(signal.get("updated_at_ts", 0.0) or 0.0)
        rank = (session_match, has_finite, global_step, updated_at)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_signal = signal
    return _merge_learning_metrics(previous_metrics, (best_signal or {}).get("learning_metrics") or {})


def _resolve_resume_state_metadata_path(code_dir: Path) -> Path:
    override = str(os.getenv("KAIWU_RESUME_STATE_METADATA_PATH", "") or "").strip()
    if override:
        path = Path(override)
        return path if path.is_absolute() else (code_dir / path).resolve()
    layout = ensure_runtime_state_dirs(code_dir)
    prepared_path = layout.current.prepared_resume_dir / RESUME_STATE_FILE
    if prepared_path.exists():
        return prepared_path
    if allow_legacy_resume_import():
        return legacy_resume_latest_state_path(code_dir)
    return prepared_path


def _load_resume_bundle(code_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    state_path = _resolve_resume_state_metadata_path(code_dir)
    state_meta = _read_json(state_path)
    if not state_meta:
        return None, None
    snapshot_ref = state_meta.get("curriculum_state_snapshot_path")
    if snapshot_ref:
        snapshot_path = Path(str(snapshot_ref))
        if not snapshot_path.is_absolute():
            snapshot_path = (code_dir / snapshot_path).resolve()
    else:
        prepared_snapshot = ensure_runtime_state_dirs(code_dir).current.prepared_resume_dir / RESUME_CURRICULUM_SNAPSHOT_FILE
        snapshot_path = prepared_snapshot
        if not snapshot_path.exists() and allow_legacy_resume_import():
            snapshot_path = legacy_resume_curriculum_snapshot_path(code_dir)
    return state_meta, _read_json(snapshot_path)


def _default_state() -> dict[str, Any]:
    initial_stage = str(os.getenv("KAIWU_CURRICULUM_INITIAL_STAGE", "warmup") or "warmup").strip().lower()
    if initial_stage not in STAGE_INDEX:
        initial_stage = "blend"
    preload_enabled = str(os.getenv("KAIWU_PRELOAD_MODEL", "0") or "0").strip().lower() in {"1", "true", "on", "yes"}
    return {
        "version": STATE_VERSION,
        "stage": initial_stage,
        "stage_version": 1,
        "updated_at_ts": 0.0,
        "entered_global_step": 0,
        "entered_wall_clock_ts": 0.0,
        "consecutive_pass_windows": 0,
        "consecutive_fail_windows": 0,
        "last_promotion_reason": None,
        "promotion_timeout_steps": 0,
        "promotion_eligibility_snapshot": {},
        "stage_entry_metrics": {},
        "last_global_metrics": {},
        "last_bootstrap_metrics": {},
        "last_learning_metrics": {},
        "global_episode_count": 0,
        "global_step_since_resume": 0,
        "recent_episodes": [],
        "source_session_id": None,
        "restored_from_session_id": None,
        "initial_stage": initial_stage,
        "training_start_mode": str(os.getenv("KAIWU_TRAINING_START_MODE", "preload") or "preload").strip().lower(),
        "preload_enabled": bool(preload_enabled),
        "observation_phase_active": False,
        "curriculum_stagnation_level": 0,
        "curriculum_stagnation_reason": [],
        "stagnation_windows": 0,
        "invalid_for_promotion": False,
        "requires_reward_revision": False,
        "in_transition_guard": False,
        "transition_target_stage": None,
        "transition_entered_global_step": 0,
        "last_stage_transition_global_step": 0,
        "degraded_mainline": False,
        "degraded_mainline_windows": 0,
        "curriculum_profile_weights": {
            "anchor": 0.45,
            "mild": 0.35,
            "broad": 0.20,
            "broad_eval": 0.0,
        },
    }


@dataclass
class SharedCurriculumStateStore:
    code_dir: Path

    def __post_init__(self):
        self.layout = ensure_runtime_state_dirs(self.code_dir)
        self.signal_dir = self.layout.current.curriculum_signal_dir
        self.signal_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.layout.current.curriculum_state_path
        self.lock_path = self.layout.current.curriculum_state_lock_path

    def signal_path(self, source_id: str) -> Path:
        safe = source_id.replace("/", "_").replace(":", "_")
        return self.signal_dir / f"{safe}.json"

    def _run_signal_path(self, session_id: str | None, source_id: str) -> Path | None:
        if not session_id:
            return None
        safe = source_id.replace("/", "_").replace(":", "_")
        run_layout = self.layout.for_run(str(session_id))
        run_layout.curriculum_signal_dir.mkdir(parents=True, exist_ok=True)
        return run_layout.curriculum_signal_dir / f"{safe}.json"

    def _persist_run_state_view(self, state: dict[str, Any]) -> None:
        session_id = str(state.get("source_session_id") or "").strip()
        if not session_id:
            return
        run_layout = self.layout.for_run(session_id)
        run_layout.run_dir.mkdir(parents=True, exist_ok=True)
        _write_json(run_layout.curriculum_state_path, state)

    def write_signal(self, source_id: str, payload: dict[str, Any]) -> None:
        record = {
            "source_id": source_id,
            "updated_at_ts": _now_ts(),
            **deepcopy(payload),
        }
        _write_json(self.signal_path(source_id), record)
        run_signal_path = self._run_signal_path(record.get("session_id"), source_id)
        if run_signal_path is not None:
            _write_json(run_signal_path, record)

    def seed_initial_state(
        self,
        session_id: str,
        initial_stage: str,
        lite_benchmark_used: bool = False,
        lite_benchmark_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if fcntl is None:
            return self._seed_initial_state_impl(session_id, initial_stage, lite_benchmark_used, lite_benchmark_metrics)
        with self.lock_path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                return self._seed_initial_state_impl(session_id, initial_stage, lite_benchmark_used, lite_benchmark_metrics)
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _seed_initial_state_impl(
        self,
        session_id: str,
        initial_stage: str,
        lite_benchmark_used: bool = False,
        lite_benchmark_metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        state = self.read_state()
        if state.get("source_session_id") == session_id:
            return state
        state = _default_state()
        stage = str(initial_stage or state.get("stage") or "warmup").strip().lower()
        if stage not in STAGE_INDEX:
            stage = "warmup"
        training_start_mode = str(os.getenv("KAIWU_TRAINING_START_MODE", "preload") or "preload").strip().lower()
        preload_enabled = str(os.getenv("KAIWU_PRELOAD_MODEL", "0") or "0").strip().lower() in {"1", "true", "on", "yes"}
        if training_start_mode == "resume":
            resume_meta, resume_snapshot = _load_resume_bundle(self.code_dir)
            if resume_meta and resume_snapshot:
                restored = deepcopy(_default_state())
                for key in (
                    "stage",
                    "stage_version",
                    "entered_global_step",
                    "entered_wall_clock_ts",
                    "consecutive_pass_windows",
                    "consecutive_fail_windows",
                    "last_promotion_reason",
                    "promotion_timeout_steps",
                    "promotion_eligibility_snapshot",
                    "stage_entry_metrics",
                    "last_global_metrics",
                    "last_bootstrap_metrics",
                    "last_learning_metrics",
                    "global_episode_count",
                    "global_step_since_resume",
                    "initial_stage",
                    "observation_phase_active",
                    "curriculum_stagnation_level",
                    "curriculum_stagnation_reason",
                    "stagnation_windows",
                    "invalid_for_promotion",
                    "requires_reward_revision",
                    "in_transition_guard",
                    "transition_target_stage",
                    "transition_entered_global_step",
                    "curriculum_profile_weights",
                    "curriculum_progress",
                    "curriculum_stage_idx",
                    "last_stage_transition_global_step",
                    "degraded_mainline",
                    "degraded_mainline_windows",
                ):
                    if key in resume_snapshot:
                        restored[key] = deepcopy(resume_snapshot[key])
                restored["last_learning_metrics"] = _merge_learning_metrics(
                    restored.get("last_learning_metrics"),
                    resume_meta.get("last_learning_metrics"),
                )
                if _is_finite_number(resume_meta.get("global_step_since_resume")):
                    restored["global_step_since_resume"] = int(float(resume_meta.get("global_step_since_resume")))
                restored["source_session_id"] = session_id
                restored["restored_from_session_id"] = resume_meta.get("session_id") or resume_snapshot.get("source_session_id")
                restored["updated_at_ts"] = _now_ts()
                restored["training_start_mode"] = training_start_mode
                restored["preload_enabled"] = bool(preload_enabled)
                restored["recent_episodes"] = []
                if stage in STAGE_INDEX:
                    restored["initial_stage"] = str(resume_snapshot.get("initial_stage") or restored.get("initial_stage") or stage)
                state = restored
                _write_json(self.state_path, state)
                self._persist_run_state_view(state)
                return state
        state["stage"] = stage
        state["initial_stage"] = stage
        state["source_session_id"] = session_id
        state["lite_benchmark_used"] = bool(lite_benchmark_used)
        state["lite_benchmark_metrics"] = deepcopy(lite_benchmark_metrics or {})
        state["training_start_mode"] = training_start_mode
        state["preload_enabled"] = bool(preload_enabled)
        state["curriculum_progress"] = 0.0
        state["curriculum_stage_idx"] = STAGE_INDEX.get(stage, 0)
        state["updated_at_ts"] = _now_ts()
        _write_json(self.state_path, state)
        self._persist_run_state_view(state)
        return state

    def read_state(self) -> dict[str, Any]:
        state = _read_json(self.state_path)
        if state:
            return state
        if allow_legacy_resume_import() and not is_scratch_mode():
            legacy_state = _read_json(legacy_curriculum_state_path(self.code_dir))
            if legacy_state:
                return legacy_state
        return _default_state()

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
        learner_log_metrics = _latest_learner_log_metrics(self.code_dir)
        signals = []
        current_session_id = state.get("source_session_id")
        for path in self.signal_dir.glob("*.json"):
            signal = _read_json(path)
            if not signal:
                continue
            if now - float(signal.get("updated_at_ts", 0.0)) > SIGNAL_TTL_SECONDS:
                continue
            signal_session_id = signal.get("session_id")
            if current_session_id and signal_session_id and str(signal_session_id) != str(current_session_id):
                continue
            signals.append(signal)

        if not signals:
            state["last_learning_metrics"] = _merge_with_learner_log_metrics(
                state.get("last_learning_metrics"),
                learner_log_metrics,
            )
            state["updated_at_ts"] = now
            _write_json(self.state_path, state)
            self._persist_run_state_view(state)
            return state

        recent_episodes = _merge_recent_episodes(state, signals)
        bootstrap_metrics = _aggregate_episode_records(recent_episodes[-FAST_SKIP_GLOBAL_EPISODES:], FAST_SKIP_GLOBAL_EPISODES)
        window_metrics = _aggregate_episode_records(recent_episodes[-FULL_WINDOW_GLOBAL_EPISODES:], FULL_WINDOW_GLOBAL_EPISODES)
        if bootstrap_metrics is None:
            bootstrap_metrics = _aggregate_metrics(signals, "bootstrap_metrics", FAST_SKIP_GLOBAL_EPISODES)
        if window_metrics is None:
            window_metrics = _aggregate_metrics(signals, "window_metrics", FULL_WINDOW_GLOBAL_EPISODES)
        learning_metrics = _best_learning_metrics(
            signals,
            current_session_id=current_session_id,
            previous_metrics=state.get("last_learning_metrics"),
        )
        learning_metrics = _merge_with_learner_log_metrics(learning_metrics, learner_log_metrics)
        global_step_since_resume = int(max(_metric(signal.get("runtime"), "global_step_since_resume", 0.0) for signal in signals))
        global_episode_count = len(recent_episodes)
        resume_fast_track = bool(state.get("stage_version", 0) <= 1)

        context = {
            "global_step_since_resume": global_step_since_resume,
            "window_metrics": window_metrics,
            "bootstrap_metrics": bootstrap_metrics,
            "learning_metrics": learning_metrics,
            "resume_fast_track": resume_fast_track,
            "training_start_mode": state.get("training_start_mode"),
            "preload_enabled": state.get("preload_enabled"),
            "entered_global_step": int(state.get("entered_global_step", 0)),
        }

        current_stage = state.get("stage", "warmup")
        decision = choose_stage_decision(
            current_stage=current_stage,
            context=context,
            stage_entry_metrics=state.get("stage_entry_metrics"),
        )
        proposed_stage = decision["proposed_stage"]
        promotion_reason = decision.get("promotion_reason")
        initial_stage = str(state.get("initial_stage", os.getenv("KAIWU_CURRICULUM_INITIAL_STAGE", "warmup")) or "warmup")
        initial_blend_freeze_steps = int(os.getenv("KAIWU_CURRICULUM_INITIAL_BLEND_FREEZE_STEPS", "5000"))
        initial_stage_frozen = (
            current_stage == initial_stage == "blend"
            and int(state.get("stage_version", 1)) <= 1
            and global_step_since_resume < initial_blend_freeze_steps
        )
        if initial_stage_frozen and proposed_stage != current_stage:
            proposed_stage = current_stage

        dwell_requirement = MIN_STAGE_DWELL_STEPS.get(current_stage, 0)
        dwell_satisfied = global_step_since_resume - int(state.get("entered_global_step", 0)) >= dwell_requirement
        promotion_blocked = bool(state.get("invalid_for_promotion"))
        if promotion_reason == "timeout_gate":
            promotion_blocked = False
        last_transition_step = int(state.get("last_stage_transition_global_step", 0) or 0)
        stage_transition_cooldown_steps = int(Config.CURRICULUM_STAGE_TRANSITION_COOLDOWN_STEPS)
        stage_transition_in_cooldown = (
            last_transition_step > 0
            and max(global_step_since_resume - last_transition_step, 0) < max(stage_transition_cooldown_steps, 0)
        )

        if proposed_stage != current_stage and dwell_satisfied and not promotion_blocked and not stage_transition_in_cooldown:
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
                state["last_stage_transition_global_step"] = global_step_since_resume
                state["stage_entry_metrics"] = snapshot_stage_entry_metrics(window_metrics, learning_metrics)
                state["last_promotion_reason"] = promotion_reason
                state["promotion_timeout_steps"] = max(
                    global_step_since_resume - int(context.get("entered_global_step", 0)),
                    0,
                )
                state["promotion_eligibility_snapshot"] = {
                    "window_metrics": deepcopy(window_metrics or {}),
                    "bootstrap_metrics": deepcopy(bootstrap_metrics or {}),
                    "learning_metrics": deepcopy(learning_metrics or {}),
                }
                if current_stage in {"blend", "robust"}:
                    state["in_transition_guard"] = True
                    state["transition_target_stage"] = current_stage
                    state["transition_entered_global_step"] = global_step_since_resume
                else:
                    state["in_transition_guard"] = False
                    state["transition_target_stage"] = None
                    state["transition_entered_global_step"] = 0
                state["consecutive_pass_windows"] = 0
                state["pending_stage"] = None
        else:
            state["consecutive_pass_windows"] = 0
            state["pending_stage"] = None

        guard_regression_blocked = bool(state.get("in_transition_guard"))
        regress_now = should_regress_stage(
            current_stage=current_stage,
            stage_entry_metrics=state.get("stage_entry_metrics"),
            current_metrics=window_metrics,
            learning_metrics=learning_metrics,
        )
        if regress_now and not guard_regression_blocked and not stage_transition_in_cooldown:
            fail_windows = int(state.get("consecutive_fail_windows", 0)) + 1
            state["consecutive_fail_windows"] = fail_windows
            if fail_windows >= REGRESS_CONFIRM_WINDOWS and current_stage != "warmup":
                current_stage = previous_stage(current_stage)
                state["stage"] = current_stage
                state["stage_version"] = int(state.get("stage_version", 0)) + 1
                state["entered_global_step"] = global_step_since_resume
                state["entered_wall_clock_ts"] = now
                state["last_stage_transition_global_step"] = global_step_since_resume
                state["stage_entry_metrics"] = snapshot_stage_entry_metrics(window_metrics, learning_metrics)
                state["in_transition_guard"] = False
                state["transition_target_stage"] = None
                state["transition_entered_global_step"] = 0
                state["consecutive_fail_windows"] = 0
        else:
            state["consecutive_fail_windows"] = 0

        metrics_for_progress = window_metrics or bootstrap_metrics
        proposed_same_stage = proposed_stage == current_stage
        if (
            metrics_for_progress
            and proposed_same_stage
            and not should_regress_stage(
                current_stage=current_stage,
                stage_entry_metrics=state.get("stage_entry_metrics"),
                current_metrics=window_metrics,
                learning_metrics=learning_metrics,
            )
        ):
            state["stagnation_windows"] = int(state.get("stagnation_windows", 0)) + 1
        else:
            state["stagnation_windows"] = 0

        stagnation_level, stagnation_reason = stagnation_status(
            stage=current_stage,
            metrics=metrics_for_progress,
            global_step_since_resume=global_step_since_resume,
            entered_global_step=int(state.get("entered_global_step", 0)),
            stagnant_windows=int(state.get("stagnation_windows", 0)),
        )
        state["curriculum_stagnation_level"] = stagnation_level
        state["curriculum_stagnation_reason"] = stagnation_reason
        state["invalid_for_promotion"] = bool(stagnation_level >= 2)
        state["requires_reward_revision"] = bool(stagnation_level >= 3)
        degrade_flags = [
            _metric(metrics_for_progress, "battery_fail_rate", 0.0) >= 0.45,
            _metric(metrics_for_progress, "zero_charge_battery_fail_rate", 0.0) >= 0.55,
            _metric(metrics_for_progress, "planner_policy_divergence_rate", 0.0) >= 0.84,
            _metric(metrics_for_progress, "return_stall_rate", 0.0) >= 0.55,
        ] if metrics_for_progress else []
        degraded_windows = int(state.get("degraded_mainline_windows", 0))
        if sum(1 for flag in degrade_flags if flag) >= 2:
            degraded_windows += 1
        else:
            degraded_windows = 0
        state["degraded_mainline_windows"] = degraded_windows
        state["degraded_mainline"] = bool(
            degraded_windows >= int(Config.CURRICULUM_DEGRADED_MAINLINE_CONFIRM_WINDOWS)
        )
        if state["degraded_mainline"]:
            state["invalid_for_promotion"] = True
        if state.get("in_transition_guard"):
            target_stage = str(state.get("transition_target_stage") or "").strip().lower()
            guard_steps = {
                "blend": int(os.getenv("KAIWU_CURRICULUM_BLEND_GUARD_STEPS", "8000")),
                "robust": int(os.getenv("KAIWU_CURRICULUM_ROBUST_GUARD_STEPS", "12000")),
            }.get(target_stage, 0)
            if target_stage != state.get("stage") or global_step_since_resume - int(state.get("transition_entered_global_step", 0)) >= guard_steps:
                state["in_transition_guard"] = False
                state["transition_target_stage"] = None
                state["transition_entered_global_step"] = 0

        state["curriculum_stage_idx"] = STAGE_INDEX.get(state["stage"], 0)
        state["curriculum_progress"] = round(stage_progress(state["stage"], metrics_for_progress, learning_metrics), 4)
        state["last_global_metrics"] = deepcopy(window_metrics or {})
        state["last_bootstrap_metrics"] = deepcopy(bootstrap_metrics or {})
        state["last_learning_metrics"] = deepcopy(learning_metrics or {})
        state["global_episode_count"] = global_episode_count
        state["global_step_since_resume"] = global_step_since_resume
        state["recent_episodes"] = recent_episodes
        profile_plan = profile_plan_for_runtime(state["stage"], state)
        state["observation_phase_active"] = bool(profile_plan["observation_phase_active"])
        state["curriculum_profile_weights"] = deepcopy(profile_plan["weight_map"])
        state["updated_at_ts"] = now
        _write_json(self.state_path, state)
        self._persist_run_state_view(state)
        return state
