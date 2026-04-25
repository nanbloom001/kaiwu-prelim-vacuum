#!/usr/bin/env python3
"""Compare serial and parallel target benchmark result JSON files.

This tool is intentionally data-only: it reads finished benchmark artifacts and
never starts Docker, simulators, or benchmark runners.  Serial target results
remain canonical; parallel results are only an operational equivalence check when
a compatible serial result exists and the average score delta stays within the
requested tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / ".sisyphus/evidence/benchmark-900/task-10-serial-parallel.json"
NON_CANONICAL_TEXT = "parallel not canonical"
SERIAL_CANONICAL = "serial-only canonical"
PARALLEL_EQUIVALENT = "parallel operationally equivalent"


class ResultShapeError(ValueError):
    """Raised when a benchmark result cannot be normalized."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResultShapeError(f"missing result file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ResultShapeError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResultShapeError(f"result must be a JSON object: {path}")
    if isinstance(payload.get("benchmarks"), list):
        benchmarks = [row for row in payload["benchmarks"] if isinstance(row, dict)]
        if not benchmarks:
            raise ResultShapeError(f"benchmark collection has no object rows: {path}")
        payload = benchmarks[-1]
    return payload


def _as_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ResultShapeError(f"missing or non-numeric {label}") from exc


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ResultShapeError(f"missing or non-integer {label}") from exc


def _round_profile(rounds: Any) -> dict[str, str]:
    if not isinstance(rounds, dict) or not rounds:
        raise ResultShapeError("missing rounds profile")
    return {str(key): str(value) for key, value in sorted(rounds.items(), key=lambda item: str(item[0]))}


def _episode_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        return []
    return [row for row in episodes if isinstance(row, dict)]


def _distribution_from_episodes(episodes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    wins: dict[str, int] = defaultdict(int)
    for index, row in enumerate(episodes):
        key_value = row.get("map_id", row.get("map", row.get("round", f"episode_{index + 1}")))
        key = f"map_{key_value}" if isinstance(key_value, int) else str(key_value)
        grouped[key].append(_as_float(row.get("clean_score"), f"episodes[{index}].clean_score"))
        result = str(row.get("result", "")).lower()
        if result in {"win", "completed", "success", "done"} or row.get("completed") is True:
            wins[key] += 1
    return {
        key: {
            "episode_count": len(scores),
            "avg_clean_score": sum(scores) / len(scores),
            "win_episode_count": wins.get(key, 0),
        }
        for key, scores in sorted(grouped.items())
        if scores
    }


def _distribution_from_per_round(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_round = payload.get("per_round")
    if not isinstance(per_round, dict) or not per_round:
        return {}
    distribution: dict[str, dict[str, Any]] = {}
    for key, row in sorted(per_round.items(), key=lambda item: str(item[0])):
        if not isinstance(row, dict):
            raise ResultShapeError(f"per_round.{key} must be an object")
        distribution[str(key)] = {
            "episode_count": _as_int(row.get("episode_count"), f"per_round.{key}.episode_count"),
            "avg_clean_score": _as_float(row.get("avg_clean_score"), f"per_round.{key}.avg_clean_score"),
            "win_episode_count": _as_int(row.get("win_episode_count", 0), f"per_round.{key}.win_episode_count"),
        }
    return distribution


def normalize_result(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    profile = _round_profile(payload.get("rounds"))
    episodes = _episode_rows(payload)
    distribution = _distribution_from_episodes(episodes) if episodes else _distribution_from_per_round(payload)
    if not distribution:
        raise ResultShapeError("result has neither episodes nor per_round score distribution")

    total_episodes = sum(_as_int(row.get("episode_count"), f"distribution.{key}.episode_count") for key, row in distribution.items())
    overall_payload = payload.get("overall")
    overall = overall_payload if isinstance(overall_payload, dict) else {}
    average_score = overall.get("avg_clean_score")
    if average_score is None:
        weighted_sum = sum(row["avg_clean_score"] * row["episode_count"] for row in distribution.values())
        average_score = weighted_sum / total_episodes if total_episodes else 0.0
    win_count = sum(_as_int(row.get("win_episode_count", 0), f"distribution.{key}.win_episode_count") for key, row in distribution.items())

    return {
        "path": str(path),
        "timestamp": payload.get("timestamp"),
        "checkpoint": payload.get("checkpoint"),
        "policy_mode": payload.get("policy_mode"),
        "profile": profile,
        "distribution_kind": "per_map" if episodes else "per_round",
        "distribution": distribution,
        "episode_count": total_episodes,
        "win_episode_count": win_count,
        "average_score": round(_as_float(average_score, "overall.avg_clean_score"), 6),
    }


def compare_results(serial_path: Path, parallel_path: Path, max_avg_delta: float) -> dict[str, Any]:
    errors: list[str] = []
    serial: dict[str, Any] | None = None
    parallel: dict[str, Any] | None = None
    try:
        serial = normalize_result(serial_path)
    except ResultShapeError as exc:
        errors.append(f"serial result unavailable or invalid: {exc}")
    try:
        parallel = normalize_result(parallel_path)
    except ResultShapeError as exc:
        errors.append(f"parallel result unavailable or invalid: {exc}")

    if serial is None or parallel is None:
        report: dict[str, Any] = {
            "serial_path": str(serial_path),
            "parallel_path": str(parallel_path),
            "max_avg_delta": max_avg_delta,
            "profile_compatible": False,
            "status": "blocked_serial_comparison_unavailable" if serial is None else "blocked_parallel_comparison_unavailable",
            "canonical_runner_decision": SERIAL_CANONICAL,
            "message": f"{NON_CANONICAL_TEXT}: serial comparison unavailable; {SERIAL_CANONICAL} remains required" if serial is None else f"{NON_CANONICAL_TEXT}: parallel comparison unavailable; {SERIAL_CANONICAL} remains required",
            "errors": errors,
        }
        if serial is not None:
            report["serial"] = {
                "episode_count": serial["episode_count"],
                "win_episode_count": serial["win_episode_count"],
                "average_score": serial["average_score"],
                "distribution_kind": serial["distribution_kind"],
                "profile": serial["profile"],
            }
        if parallel is not None:
            report["parallel"] = {
                "episode_count": parallel["episode_count"],
                "win_episode_count": parallel["win_episode_count"],
                "average_score": parallel["average_score"],
                "distribution_kind": parallel["distribution_kind"],
                "profile": parallel["profile"],
            }
        return report

    same_profile = serial["profile"] == parallel["profile"]
    same_distribution_keys = set(serial["distribution"]) == set(parallel["distribution"])
    same_episode_count = serial["episode_count"] == parallel["episode_count"]
    profile_compatible = same_profile and same_distribution_keys and same_episode_count
    avg_delta = abs(serial["average_score"] - parallel["average_score"])

    distribution_deltas: dict[str, dict[str, Any]] = {}
    for key in sorted(set(serial["distribution"]) | set(parallel["distribution"])):
        serial_row = serial["distribution"].get(key)
        parallel_row = parallel["distribution"].get(key)
        if serial_row is None or parallel_row is None:
            distribution_deltas[key] = {"compatible": False, "reason": "missing distribution key on one side"}
            continue
        distribution_deltas[key] = {
            "compatible": True,
            "serial_avg_clean_score": round(float(serial_row["avg_clean_score"]), 6),
            "parallel_avg_clean_score": round(float(parallel_row["avg_clean_score"]), 6),
            "delta": round(abs(float(serial_row["avg_clean_score"]) - float(parallel_row["avg_clean_score"])), 6),
            "serial_episode_count": serial_row["episode_count"],
            "parallel_episode_count": parallel_row["episode_count"],
        }

    accepted = profile_compatible and avg_delta <= max_avg_delta
    reasons: list[str] = []
    if not same_profile:
        reasons.append("round profile differs")
    if not same_distribution_keys:
        reasons.append("score distribution keys differ")
    if not same_episode_count:
        reasons.append("episode counts differ")
    if avg_delta > max_avg_delta:
        reasons.append(f"average score delta {avg_delta:.6g} exceeds tolerance {max_avg_delta:.6g}")

    return {
        "serial_path": str(serial_path),
        "parallel_path": str(parallel_path),
        "max_avg_delta": max_avg_delta,
        "profile_compatible": profile_compatible,
        "compatibility_checks": {
            "same_round_profile": same_profile,
            "same_distribution_keys": same_distribution_keys,
            "same_episode_count": same_episode_count,
        },
        "serial": {
            "episode_count": serial["episode_count"],
            "win_episode_count": serial["win_episode_count"],
            "average_score": serial["average_score"],
            "distribution_kind": serial["distribution_kind"],
            "profile": serial["profile"],
        },
        "parallel": {
            "episode_count": parallel["episode_count"],
            "win_episode_count": parallel["win_episode_count"],
            "average_score": parallel["average_score"],
            "distribution_kind": parallel["distribution_kind"],
            "profile": parallel["profile"],
        },
        "average_delta": round(avg_delta, 6),
        "per_map_deltas": distribution_deltas,
        "canonical_runner_decision": SERIAL_CANONICAL,
        "parallel_equivalence_decision": PARALLEL_EQUIVALENT if accepted else NON_CANONICAL_TEXT,
        "status": "accepted" if accepted else "rejected",
        "message": (
            f"{PARALLEL_EQUIVALENT}: serial/parallel target benchmark results are equivalent within tolerance; "
            f"{SERIAL_CANONICAL} remains the success authority"
        )
        if accepted
        else f"{NON_CANONICAL_TEXT}: {', '.join(reasons) or 'comparison rejected'}; {SERIAL_CANONICAL} remains required",
        "errors": errors,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("serial_result", type=Path, help="Serial benchmark result.json")
    parser.add_argument("parallel_result", type=Path, help="Parallel benchmark result.json")
    parser.add_argument("--max-avg-delta", type=float, default=25.0, help="Maximum allowed average clean-score delta")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    report = compare_results(args.serial_result, args.parallel_result, args.max_avg_delta)
    text = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True)
    print(text)
    if args.output:
        _write_json(args.output, report)
    if report["status"] == "accepted":
        return 0
    print(report["message"], file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
