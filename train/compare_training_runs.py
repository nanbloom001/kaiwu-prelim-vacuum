#!/usr/bin/env python3
"""Compare training runs at fixed curriculum sampling points."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
CODE_DIR = REPO_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from agent_ppo.workflow.curriculum_state import (  # noqa: E402
    SAMPLE_WINDOW_EPISODES,
    _aggregate_episode_records,
)
from agent_ppo.workflow.state_layout import (  # noqa: E402
    CURRICULUM_STATE_FILE,
    COMPARISON_SAMPLES_FILE,
    RUN_SESSION_MANIFEST_FILE,
    runtime_state_layout,
)


SAMPLE_POINT_ORDER = [name for name, _ in SAMPLE_WINDOW_EPISODES]
SAMPLE_POINT_THRESHOLD = dict(SAMPLE_WINDOW_EPISODES)
CORE_METRICS = (
    "battery_positive_reward_rate",
    "zero_charge_battery_fail_rate",
    "battery_fail_rate",
    "avg_clean_per_step",
)
SECONDARY_METRICS = (
    "zero_charge_among_battery_fail_rate",
    "win_rate",
    "route_phase_return_stall_rate",
    "avg_reward_risk_release_reward",
    "avg_reward_charger_progress_arrival_bonus",
    "avg_reward_safe_return_progress_bonus",
    "avg_reward_clean_per_step_efficiency_bonus",
    "avg_reward_route_phase_risk_growth_penalty",
    "avg_reward_risk_growth_while_clean_penalty",
    "avg_reward_charge_opportunity_cost_penalty",
    "avg_reward_charge_reward_shadow_only_active",
    "return_entry_count",
    "readiness_supported_return_entry_count",
    "pre_return_readiness_hit_rate",
    "readiness_to_return_transition_rate",
    "direct_return_without_readiness_rate",
    "clean_floor_revisit_rate",
    "clean_floor_revisit_penalty_mean",
    "effective_coverage_bonus_mean",
    "expert_weight_nonzero_rate",
    "pre_return_bias_active_rate",
    "return_bias_active_rate",
    "mode_usage_contract",
    "planner_policy_divergence_rate",
    "reliable_planner_divergence_rate",
    "mode_teacher_active_rate",
    "route_anchor_teacher_active_rate",
    "target_teacher_active_rate",
    "return_action_teacher_active_rate",
    "route_phase_action_teacher_active_rate",
    "route_phase_policy_teacher_loss",
)

REQUIRED_LOCAL_METRIC_KEYS = (
    "clean_floor_revisit_rate",
    "clean_floor_revisit_penalty_mean",
    "effective_coverage_bonus_mean",
    "expert_weight_nonzero_rate",
    "pre_return_bias_active_rate",
    "return_bias_active_rate",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_run_dir(run_ref: str) -> Path:
    layout = runtime_state_layout(CODE_DIR)
    ref = str(run_ref).strip()
    if ref == "current":
        manifest = _read_json(layout.current.run_session_manifest_path)
        return layout.for_run(str(manifest["run_session_id"])).run_dir
    candidate = Path(ref)
    if candidate.exists():
        return candidate
    return layout.for_run(ref).run_dir


def _snapshot_sort_key(payload: dict, name: str) -> tuple[int, float, str]:
    return (
        int(payload.get("global_episode_count") or -1),
        float(payload.get("updated_at_ts") or 0.0),
        name,
    )


def _iter_curriculum_snapshots(run_dir: Path) -> list[tuple[str, dict]]:
    paths = [run_dir / CURRICULUM_STATE_FILE]
    paths.extend(sorted((run_dir / "resume" / "snapshots").glob("*.curriculum.json")))
    snapshots: list[tuple[str, dict]] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        snapshots.append((path.name, payload))
    snapshots.sort(key=lambda item: _snapshot_sort_key(item[1], item[0]))
    return snapshots


def _sample_record(sample_point: str, payload: dict, metrics: dict, source_name: str, derived_from: str) -> dict:
    episode_threshold = int(SAMPLE_POINT_THRESHOLD[sample_point])
    window_size = 10 if episode_threshold <= 20 else 20
    local_start = max(0, episode_threshold - window_size)
    local_end = episode_threshold
    local_metrics = _aggregate_episode_records(
        list(payload.get("recent_episodes") or [])[:episode_threshold][local_start:local_end],
        local_end - local_start,
    )
    return {
        "sample_point": sample_point,
        "episode_threshold": episode_threshold,
        "actual_global_episode_count": int(payload.get("global_episode_count") or 0),
        "global_step_since_resume": int(payload.get("global_step_since_resume") or 0),
        "captured_at_ts": float(payload.get("updated_at_ts") or 0.0),
        "window_origin": _payload_window_origin(payload),
        "resumed_from_session_id": payload.get("restored_from_session_id") or payload.get("resumed_from_session_id"),
        "metrics": metrics,
        "local_metrics": local_metrics,
        "local_window_size": int(window_size),
        "local_episode_start": int(local_start + 1),
        "local_episode_end": int(local_end),
        "learning_metrics": {
            key: (payload.get("last_learning_metrics") or {}).get(key)
            for key in (
                "mode_teacher_active_rate",
                "route_anchor_teacher_active_rate",
                "target_teacher_active_rate",
                "return_action_teacher_active_rate",
                "route_phase_action_teacher_active_rate",
                "mode_teacher_loss",
                "route_anchor_teacher_loss",
                "target_teacher_loss",
                "return_action_teacher_loss",
                "route_phase_policy_teacher_loss",
            )
            if (payload.get("last_learning_metrics") or {}).get(key) is not None
        },
        "source_name": source_name,
        "derived_from": derived_from,
    }


def _combined_metrics(sample: dict | None) -> dict | None:
    if not sample:
        return None
    merged = dict(sample.get("metrics") or {})
    merged.update(sample.get("learning_metrics") or {})
    return merged


def _combined_local_metrics(sample: dict | None) -> dict | None:
    if not sample:
        return None
    local_metrics = sample.get("local_metrics")
    if not local_metrics:
        return None
    merged = dict(local_metrics)
    merged.update(sample.get("learning_metrics") or {})
    return merged


def _sample_points_need_recompute(sample_points: dict[str, dict]) -> bool:
    for sample in (sample_points or {}).values():
        metrics = sample.get("metrics") or {}
        if "zero_charge_battery_fail_rate" in metrics and "zero_charge_among_battery_fail_rate" not in metrics:
            return True
    return False


def _payload_window_origin(payload: dict) -> str:
    explicit = str(payload.get("window_origin") or "").strip()
    if explicit:
        return explicit
    training_start_mode = str(payload.get("training_start_mode") or "").strip().lower()
    if training_start_mode == "resume" or payload.get("restored_from_session_id"):
        return "resumed_local"
    return "scratch_local"


def _maybe_attach_local_metrics(sample: dict, ordered_records: list[dict]) -> dict:
    updated = dict(sample)
    local_metrics = updated.get("local_metrics") or {}
    if local_metrics and all(key in local_metrics for key in REQUIRED_LOCAL_METRIC_KEYS):
        return updated
    episode_threshold = int(updated.get("episode_threshold") or 0)
    if episode_threshold <= 0 or len(ordered_records) < episode_threshold:
        return updated
    window_size = 10 if episode_threshold <= 20 else 20
    local_start = max(0, episode_threshold - window_size)
    local_end = episode_threshold
    local_metrics = _aggregate_episode_records(ordered_records[local_start:local_end], local_end - local_start)
    if local_metrics is None:
        return updated
    updated["local_metrics"] = local_metrics
    updated["local_window_size"] = int(window_size)
    updated["local_episode_start"] = int(local_start + 1)
    updated["local_episode_end"] = int(local_end)
    return updated


def load_run_samples(run_ref: str) -> dict:
    run_dir = _resolve_run_dir(run_ref)
    samples_payload = {}
    samples_path = run_dir / COMPARISON_SAMPLES_FILE
    if samples_path.exists():
        samples_payload = _read_json(samples_path)
    sample_points = dict(samples_payload.get("sample_points") or {})
    inferred_window_origin = str(samples_payload.get("window_origin") or "").strip()
    inferred_resumed_from_session_id = str(samples_payload.get("resumed_from_session_id") or "").strip()
    inferred_training_start_mode = str(samples_payload.get("training_start_mode") or "").strip()
    if _sample_points_need_recompute(sample_points):
        sample_points = {}

    for source_name, payload in _iter_curriculum_snapshots(run_dir):
        recent_episodes = list(payload.get("recent_episodes") or [])
        global_episode_count = int(payload.get("global_episode_count") or 0)
        window_origin = _payload_window_origin(payload)
        if not inferred_window_origin:
            inferred_window_origin = window_origin
        if not inferred_training_start_mode:
            inferred_training_start_mode = str(payload.get("training_start_mode") or "").strip()
        if not inferred_resumed_from_session_id:
            inferred_resumed_from_session_id = str(
                payload.get("restored_from_session_id") or payload.get("resumed_from_session_id") or ""
            ).strip()
        effective_episode_count = len(recent_episodes) if window_origin == "resumed_local" else global_episode_count
        if not recent_episodes or effective_episode_count <= 0:
            continue
        # Exact legacy reconstruction is only safe while snapshots still contain
        # the full prefix episode history from the start of the run.
        if window_origin != "resumed_local" and global_episode_count != len(recent_episodes):
            continue
        for sample_point, threshold in SAMPLE_WINDOW_EPISODES:
            if effective_episode_count < threshold or len(recent_episodes) < threshold:
                continue
            existing_sample = sample_points.get(sample_point)
            if existing_sample:
                sample_points[sample_point] = _maybe_attach_local_metrics(existing_sample, recent_episodes[:threshold])
                continue
            metrics = _aggregate_episode_records(recent_episodes[:threshold], threshold)
            if metrics is None:
                continue
            record_payload = dict(payload)
            record_payload["global_episode_count"] = int(effective_episode_count)
            record_payload["window_origin"] = window_origin
            if payload.get("restored_from_session_id"):
                record_payload["resumed_from_session_id"] = payload.get("restored_from_session_id")
            sample_points[sample_point] = _sample_record(
                sample_point,
                record_payload,
                metrics,
                source_name=source_name,
                derived_from=(
                    "resume_local_snapshot_recomputed"
                    if window_origin == "resumed_local"
                    else "legacy_snapshot_recomputed"
                )
                if not samples_payload.get("sample_points") or _sample_points_need_recompute(samples_payload.get("sample_points") or {})
                else "legacy_snapshot",
            )

    return {
        "run_session_id": run_dir.name,
        "run_dir": str(run_dir),
        "train_phase": str(samples_payload.get("train_phase") or ""),
        "primary_window_policy": "local",
        "training_start_mode": inferred_training_start_mode,
        "window_origin": inferred_window_origin,
        "resumed_from_session_id": inferred_resumed_from_session_id,
        "sample_points": sample_points,
    }


def _delta_dict(target_metrics: dict | None, baseline_metrics: dict | None, metric_names: tuple[str, ...]) -> dict[str, float | None]:
    deltas: dict[str, float | None] = {}
    for metric_name in metric_names:
        if not target_metrics or not baseline_metrics:
            deltas[metric_name] = None
            continue
        target_value = target_metrics.get(metric_name)
        baseline_value = baseline_metrics.get(metric_name)
        if target_value is None or baseline_value is None:
            deltas[metric_name] = None
            continue
        deltas[metric_name] = float(target_value) - float(baseline_value)
    return deltas


def _point_status(sample_point: str, target_metrics: dict | None, baseline_metrics: dict | None) -> str:
    if not target_metrics:
        return "missing_local"
    if not baseline_metrics:
        return "missing_baseline_local"
    target_bpr = float(target_metrics.get("battery_positive_reward_rate", 0.0) or 0.0)
    target_zero = float(target_metrics.get("zero_charge_battery_fail_rate", 0.0) or 0.0)
    target_fail = float(target_metrics.get("battery_fail_rate", 0.0) or 0.0)
    target_cps = float(target_metrics.get("avg_clean_per_step", 0.0) or 0.0)
    target_contract = float(target_metrics.get("mode_usage_contract", 0.0) or 0.0)
    baseline_bpr = float(baseline_metrics.get("battery_positive_reward_rate", 0.0) or 0.0)
    baseline_zero = float(baseline_metrics.get("zero_charge_battery_fail_rate", 0.0) or 0.0)
    baseline_cps = float(baseline_metrics.get("avg_clean_per_step", 0.0) or 0.0)

    if sample_point in {"bootstrap_10", "bootstrap_20"}:
        if target_cps < baseline_cps * 0.88 or (
            target_zero > baseline_zero + 0.05 and target_bpr > baseline_bpr - 0.05
        ):
            return "early_stop_warning"
        return "direction_normal"

    passes_main = (
        target_bpr <= 0.10
        and target_zero <= 0.15
        and target_fail <= 0.25
        and target_cps >= max(baseline_cps * 0.92, 0.75)
        and 0.02 <= target_contract <= 0.12
    )
    if sample_point == "global_40":
        return "main_pass" if passes_main else "main_fail"
    return "review_stable" if passes_main else "review_reversal"


def build_comparison_report(baseline_run: str, target_run: str, reference_run: str | None = None) -> dict:
    baseline = load_run_samples(baseline_run)
    target = load_run_samples(target_run)
    reference = load_run_samples(reference_run) if reference_run else None

    point_reports: dict[str, dict] = {}
    for sample_point in SAMPLE_POINT_ORDER:
        baseline_sample = baseline["sample_points"].get(sample_point)
        target_sample = target["sample_points"].get(sample_point)
        reference_sample = reference["sample_points"].get(sample_point) if reference else None
        baseline_metrics = _combined_local_metrics(baseline_sample)
        target_metrics = _combined_local_metrics(target_sample)
        reference_metrics = _combined_local_metrics(reference_sample)
        baseline_prefix_metrics = _combined_metrics(baseline_sample)
        target_prefix_metrics = _combined_metrics(target_sample)
        reference_prefix_metrics = _combined_metrics(reference_sample)
        point_reports[sample_point] = {
            "baseline": baseline_sample,
            "target": target_sample,
            "reference": reference_sample,
            "delta_vs_baseline": _delta_dict(target_metrics, baseline_metrics, CORE_METRICS + SECONDARY_METRICS),
            "delta_vs_reference": _delta_dict(target_metrics, reference_metrics, CORE_METRICS + SECONDARY_METRICS),
            "delta_vs_baseline_prefix": _delta_dict(target_prefix_metrics, baseline_prefix_metrics, CORE_METRICS + SECONDARY_METRICS),
            "delta_vs_reference_prefix": _delta_dict(target_prefix_metrics, reference_prefix_metrics, CORE_METRICS + SECONDARY_METRICS),
            "status": _point_status(sample_point, target_metrics, baseline_metrics),
        }

    return {
        "baseline_run": baseline["run_session_id"],
        "target_run": target["run_session_id"],
        "reference_run": reference["run_session_id"] if reference else None,
        "baseline_train_phase": baseline.get("train_phase") or "",
        "target_train_phase": target.get("train_phase") or "",
        "reference_train_phase": (reference.get("train_phase") or "") if reference else "",
        "baseline_window_origin": baseline.get("window_origin") or "",
        "target_window_origin": target.get("window_origin") or "",
        "reference_window_origin": (reference.get("window_origin") or "") if reference else "",
        "sample_points": point_reports,
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def print_report(report: dict) -> None:
    print(
        f"baseline={report['baseline_run']}[{report.get('baseline_train_phase') or '-'}]"
        f"({report.get('baseline_window_origin') or 'scratch_local'}) "
        f"target={report['target_run']}[{report.get('target_train_phase') or '-'}]"
        f"({report.get('target_window_origin') or 'scratch_local'}) "
        f"reference={report.get('reference_run') or '-'}"
    )
    print("point              status              bpr      zero      fail       cps  d_cps")
    for sample_point in SAMPLE_POINT_ORDER:
        payload = report["sample_points"][sample_point]
        target_metrics = ((payload.get("target") or {}).get("local_metrics") or {})
        delta = payload.get("delta_vs_baseline") or {}
        print(
            f"{sample_point:<18} {payload['status']:<18} "
            f"{_fmt(target_metrics.get('battery_positive_reward_rate')):>8} "
            f"{_fmt(target_metrics.get('zero_charge_battery_fail_rate')):>8} "
            f"{_fmt(target_metrics.get('battery_fail_rate')):>8} "
            f"{_fmt(target_metrics.get('avg_clean_per_step')):>8} "
            f"{_fmt(delta.get('avg_clean_per_step')):>8}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare training runs at fixed sampling points")
    parser.add_argument("baseline_run")
    parser.add_argument("target_run")
    parser.add_argument("--reference-run", default="")
    parser.add_argument("--json", action="store_true", help="Print full JSON report")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_comparison_report(
        baseline_run=args.baseline_run,
        target_run=args.target_run,
        reference_run=args.reference_run or None,
    )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
