#!/usr/bin/env python3
"""Deterministic local stop gate for phase training rerun monitors.

This helper reads already-written runtime manifests and curriculum windows. It
never starts/stops Docker and never mutates runtime state; callers can use its
machine-readable JSON decision to decide whether an external cleanup command is
required.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_SESSION_PATH = REPO_ROOT / "code/runtime_state/current/run_session.json"
DEFAULT_CURRICULUM_STATE_PATH = REPO_ROOT / "code/runtime_state/current/curriculum_state.json"

BOOTSTRAP20_NODE = "bootstrap_20"
GLOBAL40_NODE = "global_40"
SEVERE_COLLAPSE_THRESHOLDS = {
    "battery_fail_rate_min": 0.90,
    "zero_charge_battery_fail_rate_min": 0.70,
    "win_rate_max": 0.10,
}
SNAPSHOT_KEYS = (
    "_count",
    "battery_fail_rate",
    "zero_charge_battery_fail_rate",
    "win_rate",
    "avg_clean_score",
    "avg_clean_per_step",
    "return_action_teacher_mask_nonzero_rate",
    "route_phase_teacher_from_critical_fallback_rate",
    "route_phase_teacher_from_anchor_or_target_rate",
    "route_phase_teacher_from_return_reliable_rate",
)

TERMINAL_DECISIONS = {
    "STOP_BINDING_MISMATCH",
    "STOP_BOOTSTRAP20_SEVERE_COLLAPSE",
    "STOP_GLOBAL40_ACCEPTED",
    "STOP_TIMEOUT_MISSING_WINDOWS",
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        return {}
    return {key: metrics.get(key) for key in SNAPSHOT_KEYS if key in metrics}


def _sample_windows(curriculum_state: dict[str, Any]) -> dict[str, Any]:
    sample_window_metrics = curriculum_state.get("sample_window_metrics")
    if isinstance(sample_window_metrics, dict):
        return sample_window_metrics
    return {}


def _window_metrics(curriculum_state: dict[str, Any], node: str) -> dict[str, Any] | None:
    sample_windows = _sample_windows(curriculum_state)
    metrics = sample_windows.get(node)
    if isinstance(metrics, dict):
        return metrics
    if node == BOOTSTRAP20_NODE:
        fallback = curriculum_state.get("last_bootstrap_metrics")
        return fallback if isinstance(fallback, dict) and fallback else None
    if node == GLOBAL40_NODE:
        fallback = curriculum_state.get("last_global_metrics")
        return fallback if isinstance(fallback, dict) and fallback else None
    return None


def is_severe_bootstrap20_collapse(metrics: dict[str, Any] | None) -> bool:
    battery_fail_rate = _as_float((metrics or {}).get("battery_fail_rate"))
    zero_charge_battery_fail_rate = _as_float((metrics or {}).get("zero_charge_battery_fail_rate"))
    win_rate = _as_float((metrics or {}).get("win_rate"))
    return bool(
        battery_fail_rate is not None
        and zero_charge_battery_fail_rate is not None
        and win_rate is not None
        and battery_fail_rate >= SEVERE_COLLAPSE_THRESHOLDS["battery_fail_rate_min"]
        and zero_charge_battery_fail_rate >= SEVERE_COLLAPSE_THRESHOLDS["zero_charge_battery_fail_rate_min"]
        and win_rate <= SEVERE_COLLAPSE_THRESHOLDS["win_rate_max"]
    )


def _binding_mismatches(
    run_session: dict[str, Any],
    curriculum_state: dict[str, Any],
    *,
    expected_run_session_id: str = "",
    expected_launch_label: str = "",
    expected_launch_instance_id: str = "",
) -> list[str]:
    mismatches: list[str] = []
    run_session_id = str(run_session.get("run_session_id") or "").strip()
    source_session_id = str(curriculum_state.get("source_session_id") or "").strip()
    launch_label = str(run_session.get("launch_label") or "").strip()
    launch_instance_id = str(run_session.get("launch_instance_id") or "").strip()

    if expected_run_session_id and run_session_id != expected_run_session_id:
        mismatches.append(f"run_session_id expected {expected_run_session_id!r}, got {run_session_id!r}")
    if expected_launch_label and launch_label != expected_launch_label:
        mismatches.append(f"launch_label expected {expected_launch_label!r}, got {launch_label!r}")
    if expected_launch_instance_id and launch_instance_id != expected_launch_instance_id:
        mismatches.append(
            f"launch_instance_id expected {expected_launch_instance_id!r}, got {launch_instance_id!r}"
        )
    if run_session_id and source_session_id and run_session_id != source_session_id:
        mismatches.append(f"curriculum source_session_id {source_session_id!r} does not match run_session_id {run_session_id!r}")
    if not bool(run_session.get("state_initialized")):
        mismatches.append("run_session manifest is not state_initialized")
    return mismatches


def _timeout_elapsed(now_ts: float | None, start_ts: float | None, timeout_seconds: float | None) -> bool:
    if now_ts is None or start_ts is None or timeout_seconds is None:
        return False
    return float(timeout_seconds) >= 0.0 and (float(now_ts) - float(start_ts)) >= float(timeout_seconds)


def _requires_cleanup(decision: str, active_container_names: list[str] | tuple[str, ...] | None) -> bool:
    if decision not in TERMINAL_DECISIONS:
        return False
    if active_container_names is None:
        return True
    return bool(active_container_names)


def evaluate_stop_gate(
    run_session: dict[str, Any],
    curriculum_state: dict[str, Any],
    *,
    expected_run_session_id: str = "",
    expected_launch_label: str = "",
    expected_launch_instance_id: str = "",
    now_ts: float | None = None,
    start_ts: float | None = None,
    timeout_seconds: float | None = None,
    active_container_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    sample_windows = _sample_windows(curriculum_state)
    sample_window_keys = sorted(sample_windows.keys())
    bootstrap20 = _window_metrics(curriculum_state, BOOTSTRAP20_NODE)
    global40 = _window_metrics(curriculum_state, GLOBAL40_NODE)
    mismatches = _binding_mismatches(
        run_session,
        curriculum_state,
        expected_run_session_id=expected_run_session_id,
        expected_launch_label=expected_launch_label,
        expected_launch_instance_id=expected_launch_instance_id,
    )

    if mismatches:
        decision = "STOP_BINDING_MISMATCH"
        reason = "; ".join(mismatches)
    elif is_severe_bootstrap20_collapse(bootstrap20):
        decision = "STOP_BOOTSTRAP20_SEVERE_COLLAPSE"
        reason = (
            "bootstrap_20 meets severe-collapse thresholds: "
            "battery_fail_rate >= 0.90, zero_charge_battery_fail_rate >= 0.70, win_rate <= 0.10"
        )
    elif isinstance(global40, dict) and global40:
        decision = "STOP_GLOBAL40_ACCEPTED"
        reason = "global_40 is present, source binding is valid, and no prior bootstrap_20 severe collapse is present"
    elif _timeout_elapsed(now_ts, start_ts, timeout_seconds):
        decision = "STOP_TIMEOUT_MISSING_WINDOWS"
        missing = [node for node, metrics in ((BOOTSTRAP20_NODE, bootstrap20), (GLOBAL40_NODE, global40)) if not metrics]
        reason = f"timeout elapsed before required windows were available: {missing or [GLOBAL40_NODE]!r}"
    else:
        decision = "WAITING_FOR_WINDOWS"
        reason = "binding is valid but no terminal stop-gate condition is present yet"

    run_session_id = str(run_session.get("run_session_id") or "").strip()
    source_session_id = str(curriculum_state.get("source_session_id") or "").strip()
    active_names = list(active_container_names) if active_container_names is not None else None

    return {
        "decision": decision,
        "reason": reason,
        "run_session_id": run_session_id,
        "source_session_id": source_session_id,
        "launch_label": str(run_session.get("launch_label") or "").strip(),
        "launch_instance_id": str(run_session.get("launch_instance_id") or "").strip(),
        "sample_window_keys": sample_window_keys,
        "requires_cleanup": _requires_cleanup(decision, active_names),
        "active_container_names": active_names or [],
        "metric_snapshot": {
            BOOTSTRAP20_NODE: _compact_metrics(bootstrap20),
            GLOBAL40_NODE: _compact_metrics(global40),
        },
        "thresholds": dict(SEVERE_COLLAPSE_THRESHOLDS),
        "binding_mismatches": mismatches,
        "timeout": {
            "now_ts": now_ts,
            "start_ts": start_ts,
            "timeout_seconds": timeout_seconds,
            "elapsed": None if now_ts is None or start_ts is None else float(now_ts) - float(start_ts),
        },
    }


def evaluate_stop_gate_from_paths(
    run_session_path: Path,
    curriculum_state_path: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    return evaluate_stop_gate(_read_json(run_session_path), _read_json(curriculum_state_path), **kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local training stop-gate state from runtime JSON files")
    parser.add_argument("--run-session", default=str(DEFAULT_RUN_SESSION_PATH), help="Path to run_session.json")
    parser.add_argument("--curriculum-state", default=str(DEFAULT_CURRICULUM_STATE_PATH), help="Path to curriculum_state.json")
    parser.add_argument("--expected-run-session-id", default="", help="Expected run_session_id for fresh binding")
    parser.add_argument("--expected-launch-label", default="", help="Expected KAIWU_PHASE_RUN_LABEL value")
    parser.add_argument("--expected-launch-instance-id", default="", help="Expected KAIWU_PHASE_RUN_LAUNCH_INSTANCE_ID value")
    parser.add_argument("--start-ts", type=float, default=None, help="Launch start timestamp for timeout decisions")
    parser.add_argument("--now-ts", type=float, default=None, help="Current timestamp for timeout decisions; defaults to time.time() when timeout is supplied")
    parser.add_argument("--timeout-seconds", type=float, default=None, help="Timeout in seconds for missing-window decisions")
    parser.add_argument("--active-container", action="append", default=None, help="Active container name; repeatable")
    parser.add_argument("--output", default="", help="Optional output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now_ts = args.now_ts
    if now_ts is None and args.timeout_seconds is not None:
        now_ts = time.time()
    report = evaluate_stop_gate_from_paths(
        Path(args.run_session),
        Path(args.curriculum_state),
        expected_run_session_id=args.expected_run_session_id,
        expected_launch_label=args.expected_launch_label,
        expected_launch_instance_id=args.expected_launch_instance_id,
        now_ts=now_ts,
        start_ts=args.start_ts,
        timeout_seconds=args.timeout_seconds,
        active_container_names=args.active_container,
    )
    if args.output:
        _write_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
