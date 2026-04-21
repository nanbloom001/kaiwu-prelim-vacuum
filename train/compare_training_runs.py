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
    "win_rate",
    "route_phase_return_stall_rate",
    "mode_usage_contract",
    "planner_policy_divergence_rate",
    "reliable_planner_divergence_rate",
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
    return {
        "sample_point": sample_point,
        "episode_threshold": int(SAMPLE_POINT_THRESHOLD[sample_point]),
        "actual_global_episode_count": int(payload.get("global_episode_count") or 0),
        "global_step_since_resume": int(payload.get("global_step_since_resume") or 0),
        "captured_at_ts": float(payload.get("updated_at_ts") or 0.0),
        "metrics": metrics,
        "source_name": source_name,
        "derived_from": derived_from,
    }


def load_run_samples(run_ref: str) -> dict:
    run_dir = _resolve_run_dir(run_ref)
    samples_payload = {}
    samples_path = run_dir / COMPARISON_SAMPLES_FILE
    if samples_path.exists():
        samples_payload = _read_json(samples_path)
    sample_points = dict(samples_payload.get("sample_points") or {})

    for source_name, payload in _iter_curriculum_snapshots(run_dir):
        recent_episodes = list(payload.get("recent_episodes") or [])
        global_episode_count = int(payload.get("global_episode_count") or 0)
        if not recent_episodes or global_episode_count <= 0:
            continue
        # Exact legacy reconstruction is only safe while snapshots still contain
        # the full prefix episode history from the start of the run.
        if global_episode_count != len(recent_episodes):
            continue
        for sample_point, threshold in SAMPLE_WINDOW_EPISODES:
            if sample_point in sample_points or global_episode_count < threshold:
                continue
            metrics = _aggregate_episode_records(recent_episodes[:threshold], threshold)
            if metrics is None:
                continue
            sample_points[sample_point] = _sample_record(
                sample_point,
                payload,
                metrics,
                source_name=source_name,
                derived_from="legacy_snapshot",
            )

    return {
        "run_session_id": run_dir.name,
        "run_dir": str(run_dir),
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
        return "missing"
    if not baseline_metrics:
        return "missing_baseline"
    target_bpr = float(target_metrics.get("battery_positive_reward_rate", 0.0) or 0.0)
    target_zero = float(target_metrics.get("zero_charge_battery_fail_rate", 0.0) or 0.0)
    target_fail = float(target_metrics.get("battery_fail_rate", 0.0) or 0.0)
    target_cps = float(target_metrics.get("avg_clean_per_step", 0.0) or 0.0)
    baseline_bpr = float(baseline_metrics.get("battery_positive_reward_rate", 0.0) or 0.0)
    baseline_zero = float(baseline_metrics.get("zero_charge_battery_fail_rate", 0.0) or 0.0)
    baseline_cps = float(baseline_metrics.get("avg_clean_per_step", 0.0) or 0.0)

    if sample_point in {"bootstrap_10", "bootstrap_20"}:
        if target_cps < baseline_cps * 0.88 or (
            target_zero > baseline_zero + 0.10 and target_bpr > baseline_bpr - 0.05
        ):
            return "early_stop_warning"
        return "direction_normal"

    passes_main = (
        target_bpr <= 0.20
        and target_zero <= 0.40
        and target_fail <= 0.25
        and target_cps >= baseline_cps * 0.92
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
        baseline_metrics = (baseline_sample or {}).get("metrics")
        target_metrics = (target_sample or {}).get("metrics")
        reference_metrics = (reference_sample or {}).get("metrics")
        point_reports[sample_point] = {
            "baseline": baseline_sample,
            "target": target_sample,
            "reference": reference_sample,
            "delta_vs_baseline": _delta_dict(target_metrics, baseline_metrics, CORE_METRICS + SECONDARY_METRICS),
            "delta_vs_reference": _delta_dict(target_metrics, reference_metrics, CORE_METRICS + SECONDARY_METRICS),
            "status": _point_status(sample_point, target_metrics, baseline_metrics),
        }

    return {
        "baseline_run": baseline["run_session_id"],
        "target_run": target["run_session_id"],
        "reference_run": reference["run_session_id"] if reference else None,
        "sample_points": point_reports,
    }


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def print_report(report: dict) -> None:
    print(f"baseline={report['baseline_run']} target={report['target_run']} reference={report.get('reference_run') or '-'}")
    print("point              status              bpr      zero      fail       cps  d_cps")
    for sample_point in SAMPLE_POINT_ORDER:
        payload = report["sample_points"][sample_point]
        target_metrics = ((payload.get("target") or {}).get("metrics") or {})
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
