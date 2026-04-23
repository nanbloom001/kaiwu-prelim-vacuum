#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Checkpoint selection scoring for resume and submission candidates.

This module intentionally keeps the scoring logic pure and side-effect free so
it can be reused by training, offline analysis, and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _float_or(value: Any, default: float) -> float:
    parsed = _as_float(value)
    return float(default) if parsed is None else float(parsed)


def _score_higher(value: float | None, bad: float, good: float, neutral_if_missing: bool = True) -> float:
    if value is None:
        return 0.5 if neutral_if_missing else 0.0
    if good <= bad:
        return 0.0
    return _clamp01((float(value) - bad) / (good - bad))


def _score_lower(value: float | None, good: float, bad: float, neutral_if_missing: bool = True) -> float:
    if value is None:
        return 0.5 if neutral_if_missing else 0.0
    if bad <= good:
        return 0.0
    return _clamp01((bad - float(value)) / (bad - good))


def _score_band(
    value: float | None,
    outer_low: float,
    inner_low: float,
    inner_high: float,
    outer_high: float,
    neutral_if_missing: bool = True,
) -> float:
    if value is None:
        return 0.5 if neutral_if_missing else 0.0
    value = float(value)
    if inner_low <= value <= inner_high:
        return 1.0
    if value <= outer_low or value >= outer_high:
        return 0.0
    if value < inner_low:
        return _clamp01((value - outer_low) / max(inner_low - outer_low, 1e-6))
    return _clamp01((outer_high - value) / max(outer_high - inner_high, 1e-6))


@dataclass(frozen=True)
class MetricSpec:
    name: str
    source: str
    direction: str
    weight: float
    params: tuple[float, ...]
    neutral_if_missing: bool = True


def _metric_value(spec: MetricSpec, sources: dict[str, dict[str, Any]]) -> float | None:
    source_payload = sources.get(spec.source, {})
    return _as_float(source_payload.get(spec.name))


def _score_metric(spec: MetricSpec, sources: dict[str, dict[str, Any]]) -> float:
    value = _metric_value(spec, sources)
    if spec.direction == "higher":
        return _score_higher(value, *spec.params, neutral_if_missing=spec.neutral_if_missing)
    if spec.direction == "lower":
        return _score_lower(value, *spec.params, neutral_if_missing=spec.neutral_if_missing)
    if spec.direction == "band":
        return _score_band(value, *spec.params, neutral_if_missing=spec.neutral_if_missing)
    raise ValueError(f"unsupported metric direction: {spec.direction}")


def _score_category(specs: list[MetricSpec], sources: dict[str, dict[str, Any]], category_weight: float) -> float:
    if not specs:
        return 0.0
    weighted = 0.0
    total = 0.0
    for spec in specs:
        weighted += _score_metric(spec, sources) * spec.weight
        total += spec.weight
    if total <= 0.0:
        return 0.0
    return category_weight * weighted / total


RESUME_CATEGORY_SPECS: dict[str, tuple[float, list[MetricSpec]]] = {
    "safety": (
        30.0,
        [
            MetricSpec("win_rate", "window", "higher", 8.0, (0.55, 0.85)),
            MetricSpec("battery_fail_rate", "window", "lower", 7.0, (0.05, 0.25)),
            MetricSpec("collision_fail_rate", "window", "lower", 6.0, (0.02, 0.12)),
            MetricSpec("late_return_rate", "window", "lower", 4.0, (0.03, 0.20)),
            MetricSpec("route_phase_return_stall_rate", "window", "lower", 5.0, (0.15, 0.45)),
        ],
    ),
    "efficiency": (
        20.0,
        [
            MetricSpec("avg_clean_per_step", "window", "higher", 12.0, (0.45, 0.95)),
            MetricSpec("cps_win", "window", "higher", 8.0, (0.55, 1.05)),
        ],
    ),
    "behavior": (
        25.0,
        [
            MetricSpec("late_contract_rate", "window", "lower", 3.0, (0.02, 0.20)),
            MetricSpec("recoverability_violation_rate", "window", "lower", 3.0, (0.05, 0.30)),
            MetricSpec("wall_hugging_clean_floor_rate", "window", "lower", 3.0, (0.02, 0.10)),
            MetricSpec("stale_boundary_follow_rate", "window", "lower", 2.0, (0.01, 0.08)),
            MetricSpec("narrow_unknown_commit_rate", "window", "lower", 3.0, (0.03, 0.18)),
            MetricSpec("missed_charge_opportunity_rate", "window", "lower", 3.0, (0.0, 0.05)),
            MetricSpec("suboptimal_target_hold_rate", "window", "lower", 4.0, (0.02, 0.12)),
            MetricSpec("reliable_planner_divergence_rate", "window", "lower", 4.0, (0.12, 0.45)),
        ],
    ),
    "learning": (
        25.0,
        [
            MetricSpec("entropy_loss", "learning", "band", 8.0, (0.50, 0.65, 0.85, 1.05)),
            MetricSpec("value_clean_loss_trend_ratio", "learning", "lower", 6.0, (0.90, 1.10)),
            MetricSpec("value_survive_loss_trend_ratio", "learning", "lower", 4.0, (0.90, 1.10)),
            MetricSpec("mode_teacher_active_rate", "learning", "higher", 2.0, (0.20, 0.45)),
            MetricSpec("route_anchor_teacher_active_rate", "learning", "higher", 2.0, (0.40, 0.75)),
            MetricSpec("target_teacher_active_rate", "learning", "higher", 2.0, (0.40, 0.75)),
            MetricSpec("return_action_teacher_active_rate", "learning", "higher", 1.0, (0.00, 0.05)),
        ],
    ),
}


SUBMISSION_CATEGORY_SPECS: dict[str, tuple[float, list[MetricSpec]]] = {
    "completion": (
        35.0,
        [
            MetricSpec("completed_rate", "benchmark", "higher", 12.0, (0.60, 0.90)),
            MetricSpec("battery_fail_rate", "benchmark", "lower", 10.0, (0.03, 0.20)),
            MetricSpec("collision_fail_rate", "benchmark", "lower", 8.0, (0.01, 0.08)),
            MetricSpec("broad_win_rate", "benchmark", "higher", 5.0, (0.45, 0.75)),
        ],
    ),
    "efficiency": (
        25.0,
        [
            MetricSpec("avg_clean_per_step", "benchmark", "higher", 10.0, (0.45, 1.00)),
            MetricSpec("cps_win", "benchmark", "higher", 8.0, (0.55, 1.10)),
            MetricSpec("avg_clean_score_win", "benchmark", "higher", 3.0, (650.0, 1100.0)),
            MetricSpec("avg_remaining_charge", "benchmark", "higher", 4.0, (40.0, 200.0)),
        ],
    ),
    "stability": (
        20.0,
        [
            MetricSpec("late_return_rate", "benchmark", "lower", 6.0, (0.03, 0.18)),
            MetricSpec("route_phase_return_stall_rate", "benchmark", "lower", 8.0, (0.15, 0.45)),
            MetricSpec("recoverability_violation_rate", "benchmark", "lower", 6.0, (0.05, 0.25)),
        ],
    ),
    "behavior": (
        20.0,
        [
            MetricSpec("wall_hugging_clean_floor_rate", "benchmark", "lower", 3.0, (0.02, 0.10)),
            MetricSpec("stale_boundary_follow_rate", "benchmark", "lower", 2.0, (0.01, 0.08)),
            MetricSpec("narrow_unknown_commit_rate", "benchmark", "lower", 3.0, (0.03, 0.16)),
            MetricSpec("missed_charge_opportunity_rate", "benchmark", "lower", 3.0, (0.0, 0.05)),
            MetricSpec("suboptimal_target_hold_rate", "benchmark", "lower", 4.0, (0.02, 0.10)),
            MetricSpec("reliable_planner_divergence_rate", "benchmark", "lower", 5.0, (0.12, 0.40)),
        ],
    ),
}


_BENCHMARK_FALLBACK_METRICS = {
    "route_phase_return_stall_rate": "return_stall_rate",
    "reliable_planner_divergence_rate": "planner_policy_divergence_rate",
}


def _with_benchmark_fallback(metrics: dict[str, Any] | None) -> dict[str, Any] | None:
    if not metrics:
        return metrics
    payload = dict(metrics)
    for primary, fallback in _BENCHMARK_FALLBACK_METRICS.items():
        if payload.get(primary) is None and payload.get(fallback) is not None:
            payload[primary] = payload[fallback]
    return payload


def _resume_eligible(window_metrics: dict[str, Any], learning_metrics: dict[str, Any]) -> bool:
    clean_trend = _as_float(learning_metrics.get("value_clean_loss_trend_ratio"))
    return (
        int(_float_or(window_metrics.get("_count"), 0.0)) >= 20
        and
        _as_float(window_metrics.get("win_rate")) is not None
        and _float_or(window_metrics.get("win_rate"), 0.0) >= 0.55
        and _float_or(window_metrics.get("battery_fail_rate"), 1.0) <= 0.25
        and _float_or(window_metrics.get("collision_fail_rate"), 1.0) <= 0.12
        and _float_or(window_metrics.get("zero_charge_battery_fail_rate"), 1.0) <= 0.15
        and _float_or(
            window_metrics.get("reliable_planner_divergence_rate"),
            _float_or(window_metrics.get("route_phase_planner_divergence_rate"), 1.0),
        ) <= 0.60
        and _float_or(window_metrics.get("route_phase_return_stall_rate"), 1.0) <= 0.45
        and _float_or(learning_metrics.get("entropy_loss"), 0.85) <= 1.05
        and (clean_trend is None or clean_trend <= 1.15)
    )


def _submission_eligible(benchmark_metrics: dict[str, Any] | None) -> bool:
    benchmark_metrics = _with_benchmark_fallback(benchmark_metrics)
    if not benchmark_metrics:
        return False
    return (
        _float_or(benchmark_metrics.get("completed_rate"), 0.0) >= 0.60
        and _float_or(benchmark_metrics.get("battery_fail_rate"), 1.0) <= 0.20
        and _float_or(benchmark_metrics.get("collision_fail_rate"), 1.0) <= 0.08
    )


def _training_stability_bonus(learning_metrics: dict[str, Any]) -> float:
    specs = [
        MetricSpec("entropy_loss", "learning", "band", 30.0, (0.50, 0.65, 0.85, 1.05)),
        MetricSpec("value_clean_loss_trend_ratio", "learning", "lower", 25.0, (0.90, 1.10)),
        MetricSpec("value_survive_loss_trend_ratio", "learning", "lower", 15.0, (0.90, 1.10)),
        MetricSpec("env_total_score", "learning", "higher", 20.0, (700.0, 900.0)),
        MetricSpec("env_total_score_trend_ratio", "learning", "higher", 10.0, (0.92, 1.04)),
    ]
    return _score_category(specs, {"learning": learning_metrics}, 100.0)


def _benchmark_stability_bonus(benchmark_metrics: dict[str, Any] | None) -> float:
    benchmark_metrics = _with_benchmark_fallback(benchmark_metrics)
    if not benchmark_metrics:
        return 0.0
    specs = [
        MetricSpec("completed_rate", "benchmark", "higher", 40.0, (0.60, 0.90)),
        MetricSpec("battery_fail_rate", "benchmark", "lower", 30.0, (0.03, 0.20)),
        MetricSpec("collision_fail_rate", "benchmark", "lower", 20.0, (0.01, 0.08)),
        MetricSpec("broad_win_rate", "benchmark", "higher", 10.0, (0.45, 0.75)),
    ]
    return _score_category(specs, {"benchmark": benchmark_metrics}, 100.0)


def compute_legacy_robust_score(clean_scores: list[float], invalid_move_rate: float, fail_reason: str) -> float:
    if len(clean_scores) < 1:
        return float("-inf")
    scores = [float(v) for v in clean_scores]
    rolling_avg = sum(scores) / len(scores)
    sorted_scores = sorted(scores)
    if len(sorted_scores) == 1:
        p10 = sorted_scores[0]
    else:
        pos = (len(sorted_scores) - 1) * 0.10
        low = int(pos)
        high = min(low + 1, len(sorted_scores) - 1)
        weight = pos - low
        p10 = float(sorted_scores[low] * (1.0 - weight) + sorted_scores[high] * weight)
    return (
        rolling_avg
        + 3.0 * p10
        - 8.0 * float(invalid_move_rate)
        - 20.0 * (1.0 if fail_reason == "battery" else 0.0)
        - 30.0 * (1.0 if fail_reason == "collision" else 0.0)
    )


def compute_checkpoint_scores(
    window_metrics: dict[str, Any],
    learning_metrics: dict[str, Any] | None = None,
    benchmark_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    learning_metrics = learning_metrics or {}
    benchmark_metrics = _with_benchmark_fallback(benchmark_metrics or {})
    sources = {
        "window": window_metrics or {},
        "learning": learning_metrics,
        "benchmark": benchmark_metrics,
    }

    resume_breakdown = {}
    resume_readiness = 0.0
    for category_name, (category_weight, specs) in RESUME_CATEGORY_SPECS.items():
        score = _score_category(specs, sources, category_weight)
        resume_breakdown[f"resume_score_{category_name}"] = round(float(score), 4)
        resume_readiness += score

    submission_breakdown = {
        "submission_score_completion": 0.0,
        "submission_score_efficiency": 0.0,
        "submission_score_stability": 0.0,
        "submission_score_behavior": 0.0,
    }
    submission_score = 0.0
    if benchmark_metrics:
        for category_name, (category_weight, specs) in SUBMISSION_CATEGORY_SPECS.items():
            score = _score_category(specs, sources, category_weight)
            submission_breakdown[f"submission_score_{category_name}"] = round(float(score), 4)
            submission_score += score

    resume_eligible = _resume_eligible(window_metrics or {}, learning_metrics)
    submission_eligible = _submission_eligible(benchmark_metrics)
    training_bonus = _training_stability_bonus(learning_metrics)
    benchmark_bonus = _benchmark_stability_bonus(benchmark_metrics)

    if submission_eligible and benchmark_metrics:
        preservation = 0.45 * submission_score + 0.35 * resume_readiness + 0.20 * benchmark_bonus
    else:
        preservation = 0.70 * resume_readiness + 0.30 * training_bonus

    payload = {
        "resume_eligible": bool(resume_eligible),
        "submission_eligible": bool(submission_eligible),
        "resume_readiness_score": round(float(resume_readiness), 4),
        "submission_score": round(float(submission_score), 4),
        "training_stability_bonus": round(float(training_bonus), 4),
        "benchmark_stability_bonus": round(float(benchmark_bonus), 4),
        "checkpoint_preservation_score": round(float(preservation), 4),
    }
    payload.update(resume_breakdown)
    payload.update(submission_breakdown)
    return payload
