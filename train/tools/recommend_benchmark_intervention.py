#!/usr/bin/env python3
"""Rank benchmark-900 intervention candidates from failure taxonomy evidence.

The selector is data-only: it reads analyzer summaries and optional Wave 0 audit
findings, then emits deterministic JSON recommendations.  It does not edit
agent, reward, simulator, benchmark, or training behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CATEGORIES: tuple[str, ...] = (
    "global_position_signal_gap",
    "battery_safety",
    "missed_charge",
    "return_stall",
    "coverage_efficiency",
    "collision_stuck",
    "checkpoint_model_load",
    "representational_limit",
)
ALLOWED_MODIFICATION_CLASSES: tuple[str, ...] = (
    "P0_observe_only",
    "P1_information_additive",
    "P2_eval_only_safety",
    "R1_small_threshold",
    "R2_reward_positive",
    "R3_reward_penalty",
    "R4_light_refactor",
    "R5_architecture",
)
CLASS_PRIORITY: dict[str, int] = {
    "P0_observe_only": 80,
    "P1_information_additive": 76,
    "P2_eval_only_safety": 70,
    "R2_reward_positive": 58,
    "R1_small_threshold": 48,
    "R4_light_refactor": 36,
    "R3_reward_penalty": 24,
    "R5_architecture": 8,
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"input file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in {path}: {exc}") from None
    if not isinstance(payload, dict):
        raise SystemExit(f"JSON input must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    return int(_as_float(value, float(default)))


def _overall(summary: dict[str, Any]) -> dict[str, Any]:
    metadata = summary.get("benchmark_metadata") or {}
    overall = metadata.get("overall") or summary.get("overall") or {}
    return overall if isinstance(overall, dict) else {}


def _anomaly_summary(summary: dict[str, Any]) -> dict[str, Any]:
    anomaly = _overall(summary).get("anomaly_summary") or summary.get("anomaly_summary") or {}
    return anomaly if isinstance(anomaly, dict) else {}


def _failure_buckets(summary: dict[str, Any]) -> dict[str, int]:
    buckets = summary.get("failure_buckets") or {}
    if not isinstance(buckets, dict):
        return {}
    return {str(key): _as_int(value) for key, value in buckets.items()}


def _lever_reason_text(summary: dict[str, Any], lever: str) -> str:
    reasons: list[str] = []
    for item in summary.get("next_recommended_levers") or []:
        if not isinstance(item, dict) or item.get("lever") != lever:
            continue
        reasons.extend(str(reason) for reason in item.get("reasons") or [])
    return "; ".join(reasons)


def _per_map_count(summary: dict[str, Any], bucket: str) -> int:
    count = 0
    for item in summary.get("per_map") or []:
        if not isinstance(item, dict):
            continue
        buckets = item.get("failure_buckets") or {}
        if isinstance(buckets, dict) and _as_int(buckets.get(bucket)) > 0:
            count += 1
    return count


def _audit_opportunities(audit: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not audit:
        return []
    opportunities: list[dict[str, Any]] = []
    for item in audit.get("opportunity_ranking") or []:
        if isinstance(item, dict):
            opportunities.append(item)
    for report in audit.get("reports") or []:
        if not isinstance(report, dict):
            continue
        for item in report.get("opportunities") or []:
            if isinstance(item, dict):
                enriched = dict(item)
                enriched.setdefault("scope", report.get("scope"))
                opportunities.append(enriched)
    return opportunities


def _matching_audit(audit: dict[str, Any] | None, *needles: str) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    lowered = [needle.lower() for needle in needles]
    for item in _audit_opportunities(audit):
        haystack = " ".join(
            str(item.get(key) or "")
            for key in ("id", "title", "rationale", "scope", "intervention_class")
        ).lower()
        if any(needle in haystack for needle in lowered):
            matches.append(item)
    return matches


def _audit_evidence(items: list[dict[str, Any]], limit: int = 4) -> list[str]:
    evidence: list[str] = []
    for item in items:
        for entry in item.get("evidence") or []:
            text = str(entry)
            if text not in evidence:
                evidence.append(text)
            if len(evidence) >= limit:
                return evidence
    return evidence


def _score(modification_class: str, evidence_weight: float, risk_penalty: float = 0.0) -> float:
    return round(CLASS_PRIORITY[modification_class] + evidence_weight - risk_penalty, 3)


def _recommendation(
    category: str,
    title: str,
    modification_class: str,
    score: float,
    allowed_file_groups: list[str],
    existing_mechanisms_to_reuse: list[str],
    rationale: str,
    evidence_signals: list[str],
    next_wave_tasks: list[str],
    applicable: bool = True,
) -> dict[str, Any]:
    if category not in CATEGORIES:
        raise ValueError(f"unsupported category {category!r}")
    if modification_class not in ALLOWED_MODIFICATION_CLASSES:
        raise ValueError(f"unsupported modification class {modification_class!r}")
    return {
        "allowed_file_groups": allowed_file_groups,
        "applicable": bool(applicable),
        "category": category,
        "evidence_signals": evidence_signals,
        "existing_mechanisms_to_reuse": existing_mechanisms_to_reuse,
        "modification_class": modification_class,
        "next_wave_tasks": next_wave_tasks,
        "rationale": rationale,
        "score": score,
        "title": title,
    }


def build_recommendations(summary: dict[str, Any], audit: dict[str, Any] | None = None) -> dict[str, Any]:
    buckets = _failure_buckets(summary)
    overall = _overall(summary)
    anomaly = _anomaly_summary(summary)
    episode_count = _as_int(summary.get("episode_count") or overall.get("episode_count"))
    avg_clean_score = _as_float(overall.get("avg_clean_score"), _as_float(summary.get("avg_clean_score")))
    win_rate = _as_float(overall.get("broad_win_rate", overall.get("completed_rate")))
    battery_count = buckets.get("battery depletion", 0)
    collision_count = buckets.get("collision/stuck loop", 0)
    checkpoint_summary = (summary.get("benchmark_metadata") or {}).get("learner_log")
    checkpoint_issue = bool(isinstance(checkpoint_summary, dict) and checkpoint_summary.get("has_checkpoint_issue"))

    global_items = _matching_audit(audit, "global", "position", "additive-signal", "pure_positive")
    pure_positive_items = _matching_audit(audit, "pure_positive", "additive", "positive")
    global_evidence = _audit_evidence(global_items or pure_positive_items)
    global_signal_confirmed = bool(
        audit
        and audit.get("global_robot_positions_confirmed")
        and audit.get("global_charger_positions_confirmed")
        and audit.get("global_npc_positions_confirmed")
    )
    global_weight = 0.0
    if global_items:
        global_weight += 32.0
    if pure_positive_items:
        global_weight += 12.0
    if collision_count or battery_count or _as_float(anomaly.get("avg_revisit_on_clean_floor_rate")) > 0.05:
        global_weight += 10.0
    if global_signal_confirmed:
        global_weight += 4.0

    missed_reason = _lever_reason_text(summary, "missed charger")
    coverage_reason = _lever_reason_text(summary, "inefficient coverage")
    return_stall_rate = _as_float(overall.get("return_stall_rate"))
    revisit_rate = _as_float(anomaly.get("avg_revisit_on_clean_floor_rate"))
    low_value_revisit_rate = _as_float(anomaly.get("avg_low_value_revisit_rate"))
    no_progress_reward_rate = _as_float(anomaly.get("avg_positive_reward_while_no_progress_rate"))
    missed_charge_rate = _as_float(anomaly.get("avg_missed_charge_opportunity_rate"))
    clean_per_step_low_maps = sum(
        1
        for item in summary.get("per_map") or []
        if isinstance(item, dict) and _as_float(item.get("avg_clean_per_step"), 1.0) < 0.5
    )

    recommendations = [
        _recommendation(
            "global_position_signal_gap",
            "Expose and verify global-position signal coverage before behavior changes",
            "P1_information_additive",
            _score("P1_information_additive", global_weight),
            [
                "code/agent_ppo/feature/preprocessor.py diagnostics-only fields",
                "code/agent_ppo/feature/expert.py observation-grounded signal probes",
                "code/agent_ppo/eval/benchmark*.py diagnostic summaries",
                "train/tools fixed-observation and audit artifacts",
            ],
            [
                "Wave 0 source-to-consumer truth table",
                "benchmark issue-index/failure taxonomy",
                "fixed-observation comparator",
                "existing global robot/charger/NPC observation consumers",
            ],
            "Wave 0 confirms global position signals already exist, so the safest next candidate is additive coverage/freshness evidence rather than a penalty, controller rewrite, or architecture change.",
            [
                f"audit_global_positions_confirmed={global_signal_confirmed}",
                f"audit_opportunity_count={len(global_items)}",
                f"collision_stuck_failures={collision_count}",
                f"battery_depletion_failures={battery_count}",
            ]
            + global_evidence,
            [
                "Task 7/8 preflight: add non-control diagnostics for stale/missing charger, robot, and NPC signal paths if gaps are confirmed.",
                "Use fixed-observation comparison to prove additive fields do not change action/logit behavior.",
            ],
            applicable=bool(global_items or global_signal_confirmed),
        ),
        _recommendation(
            "battery_safety",
            "Bound battery-risk evidence before changing charge behavior",
            "P2_eval_only_safety",
            _score("P2_eval_only_safety", battery_count * 3.5 + _per_map_count(summary, "battery depletion") * 2.0),
            [
                "code/agent_ppo/feature/preprocessor.py battery/charger feature probes",
                "code/agent_ppo/feature/expert.py charging expert and safety filters",
                "code/agent_ppo/feature/reward_metrics.py battery attribution metrics",
                "code/agent_ppo/conf/conf.py reward/threshold configuration review",
                "code/agent_ppo/feature/constraint_utils.py safety constraints",
            ],
            [
                "battery and nearest-charger preprocessor features",
                "expert charging planner / safety filter",
                "reward_metrics battery attribution",
                "benchmark charge_timing_summary and failure buckets",
            ],
            "Battery depletion exists across multiple maps, but Wave 0 ordering favors eval-only safety evidence and existing charge mechanisms before threshold or reward edits.",
            [
                f"battery_depletion_failures={battery_count}",
                f"battery_fail_rate={overall.get('battery_fail_rate')}",
                f"battery_failure_maps={_per_map_count(summary, 'battery depletion')}",
            ],
            [
                "Task 7: select a bounded eval-time safety/charging candidate only after fixed-observation intended diffs are declared.",
                "Task 8: consider positive charging guidance only if safety diagnostics show inactive/late charge behavior.",
            ],
            applicable=battery_count > 0,
        ),
        _recommendation(
            "missed_charge",
            "Prefer positive charger-access reinforcement over penalties when missed-charge evidence is present",
            "R2_reward_positive",
            _score("R2_reward_positive", (30.0 if missed_reason else 0.0) + missed_charge_rate * 500.0),
            [
                "code/agent_ppo/feature/preprocessor.py charger accessibility features",
                "code/agent_ppo/feature/expert.py charging target selection",
                "code/agent_ppo/feature/reward_metrics.py positive charge attribution",
                "code/agent_ppo/conf/conf.py reward scale configuration",
            ],
            [
                "charger_candidates telemetry",
                "expert charging planner",
                "positive-only charger-access/progress reward hooks",
                "benchmark missed_charge_opportunity issue index",
            ],
            "Missed-charge evidence should map to positive guidance only after observe/eval gates; reward penalties are deliberately lower priority.",
            [f"avg_missed_charge_opportunity_rate={missed_charge_rate}", missed_reason or "missed_charge_reason=not_reported"],
            [
                "Task 8: test one positive-only charger-access/progress lever with fixed-observation reward diffs.",
                "Keep penalty candidates blocked unless positive/additive candidates are exhausted.",
            ],
            applicable=bool(missed_reason or missed_charge_rate > 0.0),
        ),
        _recommendation(
            "return_stall",
            "Diagnose return-stall loops through existing return metrics before thresholds",
            "P1_information_additive",
            _score("P1_information_additive", return_stall_rate * 120.0 + _as_float(overall.get("late_return_rate")) * 80.0),
            [
                "code/agent_ppo/feature/preprocessor.py return/charger diagnostic fields",
                "code/agent_ppo/feature/expert.py return-path planner probes",
                "code/agent_ppo/eval/benchmark*.py return_stall issue-index summaries",
            ],
            [
                "return_stall_rate analyzer metric",
                "late_return benchmark issue index",
                "planner target and nearest charger telemetry",
            ],
            "Return stall is high enough to merit additive diagnostics, but not a threshold or refactor before the safer signal-coverage path is exhausted.",
            [f"return_stall_rate={return_stall_rate}", f"late_return_rate={overall.get('late_return_rate')}", f"return_efficiency_ratio={overall.get('return_efficiency_ratio')}"],
            [
                "Task 7: if diagnostics prove stale return targeting, evaluate a bounded safety controller candidate.",
                "Record intended fixed-observation override diffs before any eval-only control safety change.",
            ],
            applicable=return_stall_rate > 0.0,
        ),
        _recommendation(
            "coverage_efficiency",
            "Improve coverage attribution with existing features/planner/reward hooks before network changes",
            "P1_information_additive",
            _score("P1_information_additive", clean_per_step_low_maps * 5.0 + revisit_rate * 80.0 + low_value_revisit_rate * 80.0 + no_progress_reward_rate * 70.0),
            [
                "code/agent_ppo/feature/preprocessor.py dirty/clean memory and coverage features",
                "code/agent_ppo/feature/expert.py target-selection planner evidence",
                "code/agent_ppo/feature/strong_heuristic.py planner diagnostics",
                "code/agent_ppo/feature/reward_metrics.py coverage reward attribution",
                "code/agent_ppo/eval/benchmark*.py coverage diagnostics",
            ],
            [
                "dirty-memory and clean-floor revisit signals",
                "strong_heuristic target selection",
                "reward attribution for positive reward while no progress",
                "per-map clean-per-step aggregates",
            ],
            "Coverage evidence points to revisit/no-progress attribution and planner diagnostics. Network/architecture is intentionally not recommended yet.",
            [
                f"low_clean_per_step_maps={clean_per_step_low_maps}",
                f"avg_revisit_on_clean_floor_rate={revisit_rate}",
                f"avg_low_value_revisit_rate={low_value_revisit_rate}",
                f"avg_positive_reward_while_no_progress_rate={no_progress_reward_rate}",
                coverage_reason or "coverage_reason=not_reported",
            ],
            [
                "Task 8: if diagnostics confirm reward misalignment, test one positive coverage-alignment lever before penalties.",
                "Keep architecture candidates blocked until diagnostics/safety/reward waves are exhausted.",
            ],
            applicable=bool(coverage_reason or clean_per_step_low_maps or revisit_rate > 0.0),
        ),
        _recommendation(
            "collision_stuck",
            "Treat collision/stuck loops as the dominant failure bucket but avoid immediate penalties",
            "P2_eval_only_safety",
            _score("P2_eval_only_safety", collision_count * 1.8 + _as_float(anomaly.get("loop_episode_rate")) * 80.0),
            [
                "code/agent_ppo/feature/preprocessor.py stuck/loop diagnostic features",
                "code/agent_ppo/feature/expert.py safety filters",
                "code/agent_ppo/feature/constraint_utils.py legal/safety constraints",
                "code/agent_ppo/eval/benchmark*.py collision and loop attribution",
            ],
            [
                "failure taxonomy collision/stuck bucket",
                "zero_progress_streak / position_repeat diagnostics",
                "expert safety filter",
                "benchmark collision/stuck issue index",
            ],
            "Collision/stuck is the largest bucket, but eval-only safety and diagnostics should precede reward penalties or light refactors.",
            [
                f"collision_stuck_failures={collision_count}",
                f"collision_fail_rate={overall.get('collision_fail_rate')}",
                f"loop_episode_rate={anomaly.get('loop_episode_rate')}",
                _lever_reason_text(summary, "collision/stuck loop") or "collision_reason=not_reported",
            ],
            [
                "Task 7: test bounded eval-time safety only if fixed-observation diffs declare override/action changes.",
                "Do not add collision penalties until safer candidates are exhausted.",
            ],
            applicable=collision_count > 0,
        ),
        _recommendation(
            "checkpoint_model_load",
            "Keep checkpoint/model-load correctness as eval-only validation unless failure logs appear",
            "P2_eval_only_safety",
            _score("P2_eval_only_safety", 25.0 if checkpoint_issue else 2.0),
            [
                "code/agent_ppo/agent.py checkpoint load diagnostics only",
                "code/agent_ppo/workflow/checkpoint_score.py checkpoint comparison evidence",
                "train/run_target_benchmark_900.sh manifest validation",
                "train/tools benchmark validators",
            ],
            [
                "benchmark manifest checkpoint field",
                "checkpoint_score.py",
                "learner log checkpoint/model-load scan",
                "Task 5 eval drift validators",
            ],
            "The Task 4 baseline does not report a concrete checkpoint-load failure, so this remains a low-ranked eval-only guard, not an algorithm intervention.",
            [f"checkpoint_issue={checkpoint_issue}", f"checkpoint={((summary.get('benchmark_metadata') or {}).get('checkpoint'))}"],
            [
                "Task 9: keep checkpoint metadata validation in promotion gates.",
                "Escalate only if analyzer learner-log scans show failed/mismatched model loads.",
            ],
            applicable=checkpoint_issue,
        ),
        _recommendation(
            "representational_limit",
            "Defer architecture changes until lower-risk diagnostics and reward/safety waves are exhausted",
            "R5_architecture",
            _score("R5_architecture", max(0.0, (500.0 - avg_clean_score) / 20.0), risk_penalty=15.0),
            [
                "code/agent_ppo/model/model.py only after lower intervention classes are terminal",
                "code/agent_ppo/conf/conf.py architecture dimensions only after Task 11 approval",
            ],
            [
                "Wave 0 modification-order guard",
                "fixed-window convergence comparison",
                "fixed-observation corpus and comparator",
            ],
            "Average score is far below target, but architecture is explicitly last-resort and must not preempt diagnostics, safety, or reward-alignment candidates.",
            [f"avg_clean_score={avg_clean_score}", f"win_rate={win_rate}", f"episode_count={episode_count}"],
            [
                "Task 11 only: revisit model capacity if all P0/P1/P2/R1/R2/R3/R4 candidates are terminal and evidence still indicates representation failure.",
            ],
            applicable=False,
        ),
    ]

    ordered = sorted(recommendations, key=lambda item: (-item["score"], CATEGORIES.index(item["category"])))
    for rank, item in enumerate(ordered, start=1):
        item["rank"] = rank

    return {
        "category_order_contract": list(CATEGORIES),
        "input_summary": {
            "avg_clean_score": avg_clean_score,
            "battery_depletion_failures": battery_count,
            "collision_stuck_failures": collision_count,
            "episode_count": episode_count,
            "win_rate": win_rate,
        },
        "recommendations": ordered,
        "schema_version": SCHEMA_VERSION,
        "selection_policy": {
            "deterministic": True,
            "modification_class_order": list(ALLOWED_MODIFICATION_CLASSES),
            "preference": "Prefer observe-only, information-additive, and eval-only safety mechanisms before thresholds, reward penalties, refactors, or architecture when relevant evidence exists.",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="Analyzer summary JSON from summarize_benchmark_failures.py")
    parser.add_argument("--audit", type=Path, help="Optional merged Wave 0 audit JSON")
    parser.add_argument("--output", type=Path, help="Optional path to also write deterministic JSON output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = _read_json(args.summary)
    audit = _read_json(args.audit) if args.audit else None
    payload = build_recommendations(summary, audit)
    text = json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        _write_json(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
