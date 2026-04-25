#!/usr/bin/env python3
"""Resumable benchmark-900 iteration controller.

This tool records the state machine for benchmark optimization iterations.  In
dry-run mode it only writes plans/state/evidence; it never starts Docker and it
never mutates algorithm files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = REPO_ROOT / "train" / "context" / "benchmark_iterations"
EVIDENCE_DIR = REPO_ROOT / ".sisyphus" / "evidence" / "benchmark-900"
BASELINE_SUMMARY_PATH = EVIDENCE_DIR / "task-4-baseline-summary.json"
WAVE0_AUDIT_PATH = EVIDENCE_DIR / "wave0" / "full-board-audit-merged.json"

STATE_SCHEMA_VERSION = 1
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
PRE_REWARD_CLASSES = {"P0_observe_only", "P1_information_additive", "P2_eval_only_safety"}
RISK_CLASS_PREFIXES = ("R1_", "R2_", "R3_", "R4_", "R5_")
TERMINAL_CANDIDATE_STATUSES = {"passed", "failed", "not_applicable"}
CONVERGENCE_DECISIONS = ("keep", "revert", "continue_to_global_80", "targeted_resume", "escalate")
SAMPLE_POINTS = (
    "bootstrap_10",
    "bootstrap_20",
    "global_40",
    "global_80",
    "global_120",
    "global_160",
    "global_200",
)

DEV_BENCHMARK_COMMAND = (
    "bash train/run_target_benchmark_900.sh --profile dev <checkpoint> --policy-mode eval --runner serial"
)
FULL_BENCHMARK_COMMAND_SERIAL = (
    "bash train/run_target_benchmark_900.sh --profile target <checkpoint> --policy-mode eval --runner serial"
)
FULL_BENCHMARK_COMMAND_PARALLEL_NOTE = (
    "bash train/run_target_benchmark_900.sh --profile target <checkpoint> --policy-mode eval --runner parallel"
)
TRAINING_COMMAND_TEMPLATE = "python3 train/run_training_phase.py s1_survival --seed-label <iteration_id>"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def _latest_state_path(state_dir: Path = STATE_DIR) -> Path | None:
    latest = state_dir / "latest_state.json"
    if latest.exists():
        return latest
    candidates = sorted(state_dir.glob("benchmark-iteration-state-*.json"))
    return candidates[-1] if candidates else None


def _load_latest_state(state_dir: Path = STATE_DIR) -> dict[str, Any] | None:
    path = _latest_state_path(state_dir)
    return _read_json(path) if path else None


def _opportunity_records(audit: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in audit.get("opportunity_ranking") or []:
        if not isinstance(item, dict):
            continue
        record = {
            "id": str(item.get("id") or ""),
            "title": str(item.get("title") or ""),
            "modification_class": str(item.get("intervention_class") or ""),
            "status": str(item.get("status") or "pending"),
            "source": "wave0_audit",
        }
        if record["id"] and record["modification_class"] in ALLOWED_MODIFICATION_CLASSES:
            records.append(record)
    return records


def _candidate_statuses(state: dict[str, Any], audit: dict[str, Any]) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for item in _opportunity_records(audit):
        status = item.get("status", "pending")
        if status in TERMINAL_CANDIDATE_STATUSES:
            statuses[item["id"]] = status

    for item in state.get("candidate_statuses") or []:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("id") or "")
        status = str(item.get("status") or "")
        if candidate_id and status in TERMINAL_CANDIDATE_STATUSES:
            statuses[candidate_id] = status

    for iteration in state.get("iterations") or []:
        if not isinstance(iteration, dict):
            continue
        candidate = iteration.get("candidate") or {}
        candidate_id = str(candidate.get("id") or iteration.get("candidate_id") or "")
        status = str(candidate.get("status") or iteration.get("status") or "")
        if candidate_id and status in TERMINAL_CANDIDATE_STATUSES:
            statuses[candidate_id] = status
    return statuses


def validate_modification_order(candidate: dict[str, Any], state: dict[str, Any], audit: dict[str, Any]) -> None:
    """Reject R1+ classes until all P0/P1/P2 opportunities are terminal."""

    modification_class = str(candidate.get("modification_class") or "")
    if modification_class not in ALLOWED_MODIFICATION_CLASSES:
        raise ValueError(f"unsupported modification_class {modification_class!r}")
    if not modification_class.startswith(RISK_CLASS_PREFIXES):
        return

    statuses = _candidate_statuses(state, audit)
    pending = [
        item["id"]
        for item in _opportunity_records(audit)
        if item["modification_class"] in PRE_REWARD_CLASSES and statuses.get(item["id"]) not in TERMINAL_CANDIDATE_STATUSES
    ]
    if pending:
        raise ValueError(
            f"{modification_class} is blocked until all P0/P1/P2 candidates are terminal; pending={pending}"
        )


def validate_numeric_guard(candidate: dict[str, Any]) -> None:
    """Reject numeric-only changes without one failure bucket and one fixed-observation diff."""

    if not bool(candidate.get("numeric_only")):
        return
    failure_buckets = [item for item in candidate.get("failure_buckets") or [] if str(item).strip()]
    fixed_observation_diffs = [item for item in candidate.get("fixed_observation_diffs") or [] if str(item).strip()]
    if len(failure_buckets) != 1 or len(fixed_observation_diffs) != 1:
        raise ValueError(
            "numeric-only parameter changes require exactly one failure bucket and one fixed-observation diff"
        )


def validate_sweep_guard(candidate: dict[str, Any]) -> None:
    """Reject grid/sweep candidates that test more than two values of one lever."""

    sweep = candidate.get("sweep")
    if not sweep:
        return
    if not isinstance(sweep, dict):
        raise ValueError("sweep must be a mapping with one lever and at most two values")
    levers = [key for key, value in sweep.items() if value is not None]
    if len(levers) != 1:
        raise ValueError("grid/sweep iterations may test exactly one lever")
    values = sweep.get(levers[0])
    if not isinstance(values, list) or len(values) > 2:
        raise ValueError("grid/sweep iterations may test no more than two values of one lever")


def validate_candidate(candidate: dict[str, Any], state: dict[str, Any], audit: dict[str, Any]) -> None:
    validate_modification_order(candidate, state, audit)
    validate_numeric_guard(candidate)
    validate_sweep_guard(candidate)


def _baseline_context(baseline_summary: dict[str, Any]) -> dict[str, Any]:
    metadata = baseline_summary.get("benchmark_metadata") or {}
    manifest = metadata.get("manifest") or {}
    overall = metadata.get("overall") or baseline_summary.get("overall") or {}
    return {
        "avg_clean_score": overall.get("avg_clean_score"),
        "checkpoint": metadata.get("checkpoint") or manifest.get("checkpoint") or "<checkpoint>",
        "failure_buckets": baseline_summary.get("failure_buckets") or {},
        "per_map_count": len(baseline_summary.get("per_map") or []),
        "readme_parallel_40_baseline_note": (
            "Task 4 accepted operational baseline is README parallel-40: maps 1-10 x4, "
            "3 chargers, 4 robots, 1000 steps, 150 battery, policy eval."
        ),
        "session_id": manifest.get("timestamp") or metadata.get("timestamp"),
        "win_rate": overall.get("broad_win_rate") or overall.get("completed_rate"),
        "wins": overall.get("completed_count"),
    }


def _command_templates(state: dict[str, Any]) -> dict[str, str]:
    baseline = state.get("baseline") or {}
    full_command = FULL_BENCHMARK_COMMAND_SERIAL
    if baseline.get("readme_parallel_40_baseline_note"):
        full_command = FULL_BENCHMARK_COMMAND_PARALLEL_NOTE
    return {
        "dev_slice": DEV_BENCHMARK_COMMAND,
        "full_benchmark": full_command,
        "full_benchmark_default_without_parallel_40_note": FULL_BENCHMARK_COMMAND_SERIAL,
        "training": TRAINING_COMMAND_TEMPLATE,
    }


def _planned_candidate(index: int, audit: dict[str, Any]) -> dict[str, Any]:
    opportunities = _opportunity_records(audit)
    if index - 1 < len(opportunities):
        opportunity = opportunities[index - 1]
    else:
        opportunity = {
            "id": f"hypothesis-slot-{index:04d}",
            "title": "Hypothesis slot reserved for downstream candidate selector",
            "modification_class": "P0_observe_only",
            "status": "pending",
            "source": "controller_placeholder",
        }
    return {
        "changed_files": [],
        "failure_buckets": [],
        "fixed_observation_diffs": [],
        "hypothesis": "TBD by candidate selector; controller reserves state-machine fields only.",
        "id": opportunity["id"],
        "modification_class": opportunity["modification_class"],
        "numeric_only": False,
        "source": opportunity["source"],
        "status": "planned",
        "sweep": None,
        "title": opportunity["title"],
    }


def _iteration_record(index: int, state: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    candidate = _planned_candidate(index, audit)
    validate_candidate(candidate, state, audit)
    iteration_id = f"iteration-{index:04d}"
    commands = _command_templates(state)
    return {
        "analyzer_summary": None,
        "benchmark_commands": {
            "dev_slice": commands["dev_slice"],
            "full": commands["full_benchmark"],
        },
        "candidate": candidate,
        "changed_files": [],
        "checkpoint": "<checkpoint>",
        "comparator_result": None,
        "convergence_gate_result": None,
        "dev_slice_result": None,
        "full_benchmark_result": None,
        "hypothesis": candidate["hypothesis"],
        "iteration_id": iteration_id,
        "keep_revert_decision": None,
        "next_action": "await_candidate_selector",
        "sample_points": list(SAMPLE_POINTS),
        "state": "planned",
        "training_command": commands["training"].replace("<iteration_id>", iteration_id),
    }


def _next_pending_iteration(state: dict[str, Any]) -> dict[str, Any] | None:
    for iteration in state.get("iterations") or []:
        if iteration.get("state") not in {"completed", "kept", "reverted"}:
            return iteration
    return None


def build_initial_state(baseline_summary: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    return {
        "allowed_modification_classes": list(ALLOWED_MODIFICATION_CLASSES),
        "baseline": _baseline_context(baseline_summary),
        "candidate_statuses": [],
        "command_templates": {},
        "dry_run": False,
        "guards": {
            "modification_order": "R1+ classes require all available P0/P1/P2 candidates terminal",
            "numeric_only": "exactly one failure bucket and one fixed-observation diff required",
            "sweep": "one lever, at most two values",
        },
        "iterations": [],
        "schema_version": STATE_SCHEMA_VERSION,
        "state_dir": _display_path(STATE_DIR),
        "success": False,
    }


def ensure_planned_iterations(state: dict[str, Any], audit: dict[str, Any], max_iterations: int) -> dict[str, Any]:
    if max_iterations < 0:
        raise ValueError("--max-iterations must be non-negative")
    while len(state.get("iterations") or []) < max_iterations:
        index = len(state["iterations"]) + 1
        state["iterations"].append(_iteration_record(index, state, audit))
    state["command_templates"] = _command_templates(state)
    return state


def _state_output_path(state: dict[str, Any], state_dir: Path) -> Path:
    session_id = str((state.get("baseline") or {}).get("session_id") or "unknown")
    return state_dir / f"benchmark-iteration-state-{session_id}.json"


def write_state_and_evidence(
    state: dict[str, Any], *, dry_run: bool, resume: bool, state_dir: Path = STATE_DIR, evidence_dir: Path = EVIDENCE_DIR
) -> dict[str, Path]:
    state = dict(state)
    state["dry_run"] = bool(dry_run)
    state["state_dir"] = _display_path(state_dir)
    state_path = _state_output_path(state, state_dir)
    latest_path = state_path.parent / "latest_state.json"
    _write_json(state_path, state)
    _write_json(latest_path, state)

    for iteration in state.get("iterations") or []:
        evidence_path = evidence_dir / f"{iteration['iteration_id']}.json"
        _write_json(evidence_path, iteration)

    dry_run_path = evidence_dir / "task-5-loop-dry-run.json"
    _write_json(
        dry_run_path,
        {
            "dry_run": bool(dry_run),
            "iteration_count": len(state.get("iterations") or []),
            "iterations": state.get("iterations") or [],
            "latest_state": _display_path(latest_path),
            "schema_version": STATE_SCHEMA_VERSION,
        },
    )

    resume_path = evidence_dir / "task-5-loop-resume.txt"
    pending = _next_pending_iteration(state)
    if resume:
        if pending:
            text = f"RESUME loaded {_display_path(latest_path)}; next pending: {pending['iteration_id']}\n"
        else:
            text = f"RESUME loaded {_display_path(latest_path)}; no pending iteration\n"
        _write_text(resume_path, text)
    return {"state": state_path, "latest": latest_path, "dry_run": dry_run_path, "resume": resume_path}


def run_controller(args: argparse.Namespace) -> dict[str, Any]:
    baseline_summary = _load_optional_json(args.baseline_summary)
    audit = _load_optional_json(args.wave0_audit)
    if args.resume:
        state = _load_latest_state(args.state_dir)
        if state is None:
            state = build_initial_state(baseline_summary, audit)
    else:
        state = build_initial_state(baseline_summary, audit)

    if args.stop_on_success and state.get("success"):
        paths = write_state_and_evidence(
            state, dry_run=args.dry_run, resume=args.resume, state_dir=args.state_dir, evidence_dir=EVIDENCE_DIR
        )
        return {"status": "success_already_recorded", "paths": paths, "state": state}

    max_iterations = int(args.max_iterations)
    if args.resume and max_iterations == 0:
        max_iterations = len(state.get("iterations") or [])
    state = ensure_planned_iterations(state, audit, max_iterations)
    paths = write_state_and_evidence(
        state, dry_run=args.dry_run, resume=args.resume, state_dir=args.state_dir, evidence_dir=EVIDENCE_DIR
    )
    pending = _next_pending_iteration(state)
    return {
        "dry_run": bool(args.dry_run),
        "iteration_count": len(state.get("iterations") or []),
        "next_pending_iteration": pending.get("iteration_id") if pending else None,
        "paths": {key: _display_path(value) for key, value in paths.items()},
        "resume": bool(args.resume),
        "status": "planned" if args.dry_run else "state_recorded",
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record resumable benchmark-900 iteration state.")
    parser.add_argument("--dry-run", action="store_true", help="Plan iterations without starting Docker or changing algorithms")
    parser.add_argument("--resume", action="store_true", help="Load latest durable iteration state before planning/reporting")
    parser.add_argument("--max-iterations", type=int, default=0, help="Ensure this many planned iteration records exist")
    parser.add_argument("--stop-on-success", action="store_true", help="Exit without adding work if state already records success")
    parser.add_argument("--baseline-summary", type=Path, default=BASELINE_SUMMARY_PATH)
    parser.add_argument("--wave0-audit", type=Path, default=WAVE0_AUDIT_PATH)
    parser.add_argument("--state-dir", type=Path, default=STATE_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dry_run:
        raise SystemExit("non-dry-run execution is intentionally blocked until downstream runner tasks wire Docker actions")
    result = run_controller(args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
