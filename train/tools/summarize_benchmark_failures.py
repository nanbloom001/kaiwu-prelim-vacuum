#!/usr/bin/env python3
"""Summarize benchmark episode failures into taxonomy buckets.

The tool is intentionally data-only: it reads serial or parallel benchmark result
JSON, optional episode JSONL records, optional learner logs, and nearby runtime
manifests.  Missing optional telemetry is represented as ``"unavailable"`` so
older benchmark artifacts remain analyzable.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

UNAVAILABLE = "unavailable"
LEVER_ORDER = (
    "battery depletion",
    "missed charger",
    "inefficient coverage",
    "collision/stuck loop",
    "poor coordination",
    "timeout",
    "checkpoint issue",
)
CHECKPOINT_PATTERNS = (
    "checkpoint",
    "model load",
    "load model",
    "restore model",
    "resume checkpoint",
    "failed to load",
    "not found",
)


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"result file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON in result file {path}: {exc}") from None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value == UNAVAILABLE or value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int | None = None) -> int | None:
    number = _as_float(value, None)
    if number is None:
        return default
    return int(number)


def _rate(numerator: float, denominator: float) -> float | str:
    if denominator <= 0:
        return UNAVAILABLE
    return round(numerator / denominator, 6)


def _avg(values: list[float]) -> float | str:
    return round(mean(values), 6) if values else UNAVAILABLE


def _max(values: list[float]) -> float | str:
    return round(max(values), 6) if values else UNAVAILABLE


def _normalize_benchmark(data: Any, source_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(data, dict):
        raise SystemExit(f"result file must contain a JSON object: {source_path}")
    metadata: dict[str, Any] = {
        "source_path": str(source_path),
        "source_kind": "session_result" if "episodes" in data else "summary_collection",
    }
    if isinstance(data.get("benchmarks"), list):
        benchmarks = [item for item in data["benchmarks"] if isinstance(item, dict)]
        if not benchmarks:
            raise SystemExit(f"result file contains no benchmark entries: {source_path}")
        selected = benchmarks[-1]
        metadata.update(
            {
                "source_kind": "benchmark_collection",
                "benchmark_count": len(benchmarks),
                "selected_benchmark_index": len(benchmarks) - 1,
            }
        )
        return selected, metadata
    return data, metadata


def _session_dir_from_result(result_path: Path) -> Path | None:
    if result_path.name == "result.json":
        return result_path.parent
    return None


def _candidate_manifest_paths(result_path: Path, explicit: Path | None) -> list[Path]:
    paths: list[Path] = []
    if explicit is not None:
        paths.append(explicit)
    session_dir = _session_dir_from_result(result_path)
    if session_dir is not None:
        paths.append(session_dir / "manifest.json")
    return paths


def _load_manifest(result_path: Path, explicit: Path | None) -> dict[str, Any] | str:
    for path in _candidate_manifest_paths(result_path, explicit):
        if path.is_file():
            loaded = _load_json(path)
            return loaded if isinstance(loaded, dict) else {"manifest_value": loaded}
    return UNAVAILABLE


def _episode_dir(result_path: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    session_dir = _session_dir_from_result(result_path)
    if session_dir is not None and (session_dir / "episodes").is_dir():
        return session_dir / "episodes"
    return None


def _episode_key(round_name: Any, map_id: Any) -> str:
    return f"{round_name or 'unknown_round'}_map{map_id if map_id is not None else 'unknown'}"


def _episode_log_name(episode: dict[str, Any]) -> str | None:
    step_log = episode.get("step_log")
    if isinstance(step_log, str) and step_log:
        return Path(step_log).name
    round_name = episode.get("round") or "unknown_round"
    map_id = episode.get("map_id")
    if map_id is None:
        return None
    return f"{round_name}_map{map_id}.jsonl"


def _load_episode_records(episodes: list[dict[str, Any]], episode_dir: Path | None) -> dict[str, list[dict[str, Any]]]:
    if episode_dir is None or not episode_dir.is_dir():
        return {}
    records_by_key: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        name = _episode_log_name(episode)
        if not name:
            continue
        path = episode_dir / name
        if not path.is_file():
            continue
        rows: list[dict[str, Any]] = []
        for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                rows.append({"_parse_error": f"line {line_no}"})
                continue
            if isinstance(row, dict):
                rows.append(row)
        records_by_key[_episode_key(episode.get("round"), episode.get("map_id"))] = rows
    return records_by_key


def _learner_log_summary(paths: list[Path]) -> dict[str, Any] | str:
    if not paths:
        return UNAVAILABLE
    matched: list[dict[str, Any]] = []
    missing: list[str] = []
    for path in paths:
        if not path.is_file():
            missing.append(str(path))
            continue
        for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
            lower = line.lower()
            if any(pattern in lower for pattern in CHECKPOINT_PATTERNS):
                matched.append({"path": str(path), "line": line_no, "message": line.strip()[:500]})
    return {
        "checked_paths": [str(path) for path in paths],
        "missing_paths": missing,
        "checkpoint_or_model_load_messages": matched,
        "has_checkpoint_issue": any(_is_checkpoint_issue(item["message"]) for item in matched),
    }


def _is_checkpoint_issue(text: Any) -> bool:
    lower = str(text or "").lower()
    if not lower:
        return False
    if not any(pattern in lower for pattern in CHECKPOINT_PATTERNS):
        return False
    benign = ("loaded", "success", "resolved", "using checkpoint")
    failure = ("fail", "error", "missing", "not found", "exception", "traceback", "unable")
    return any(token in lower for token in failure) and not any(token in lower for token in benign)


def _records_available(records: list[dict[str, Any]]) -> bool:
    return bool(records) and not all("_parse_error" in rec for rec in records)


def _record_values(records: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _as_float(record.get(key), None)
        if value is not None:
            values.append(value)
    return values


def _count_positive(records: list[dict[str, Any]], key: str) -> int:
    return sum(1 for record in records if (_as_float(record.get(key), 0.0) or 0.0) > 0.0)


def _battery_metrics(episode: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    battery_values = _record_values(records, "battery")
    battery_end = episode.get("remaining_charge", UNAVAILABLE)
    if battery_end == UNAVAILABLE and battery_values:
        battery_end = battery_values[-1]
    battery_min = _min_or_unavailable(battery_values)
    return {"battery_end": battery_end, "battery_min": battery_min}


def _min_or_unavailable(values: list[float]) -> float | str:
    return round(min(values), 6) if values else UNAVAILABLE


def _time_on_charger(records: list[dict[str, Any]]) -> int | str:
    if not _records_available(records):
        return UNAVAILABLE
    return _count_positive(records, "charger_nearby_not_charged") + sum(
        1 for record in records if (_as_float(record.get("nearest_charger_dist"), 999999.0) or 999999.0) <= 1.0
    )


def _return_override_rate(episode: dict[str, Any], records: list[dict[str, Any]]) -> float | str:
    for key in ("diag_rate_return", "return_override_rate"):
        if key in episode:
            value = _as_float(episode.get(key), None)
            if value is not None:
                return round(value, 6)
    if not _records_available(records):
        return UNAVAILABLE
    return_markers = 0
    for record in records:
        mode = str(record.get("mode", "")).lower()
        if mode in {"return", "returning", "charge", "charging", "4", "5"}:
            return_markers += 1
        elif (_as_float(record.get("charge_margin_now"), 999999.0) or 999999.0) <= 20:
            return_markers += 1
    return _rate(return_markers, len(records))


def _stuck_revisit_signals(episode: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    anomaly_value = episode.get("anomaly_summary")
    anomaly: dict[str, Any] = anomaly_value if isinstance(anomaly_value, dict) else {}
    if _records_available(records):
        repeat_16 = _record_values(records, "position_repeat_16")
        zero_streak = _record_values(records, "zero_progress_streak")
        revisit = _record_values(records, "revisit_pressure")
        return {
            "available": True,
            "max_position_repeat_16": _max(repeat_16),
            "max_zero_progress_streak": _max(zero_streak),
            "avg_revisit_pressure": _avg(revisit),
            "loop_suspect_steps": _count_positive(records, "loop_suspect"),
            "low_value_revisit_rate": anomaly.get("low_value_revisit_rate", UNAVAILABLE),
            "loop_episode_detected": anomaly.get("loop_episode_detected", UNAVAILABLE),
            "corner_loop_detected": anomaly.get("corner_loop_detected", UNAVAILABLE),
        }
    if anomaly:
        return {
            "available": True,
            "max_position_repeat_16": UNAVAILABLE,
            "max_zero_progress_streak": UNAVAILABLE,
            "avg_revisit_pressure": UNAVAILABLE,
            "loop_suspect_steps": UNAVAILABLE,
            "low_value_revisit_rate": anomaly.get("low_value_revisit_rate", UNAVAILABLE),
            "loop_episode_detected": anomaly.get("loop_episode_detected", UNAVAILABLE),
            "corner_loop_detected": anomaly.get("corner_loop_detected", UNAVAILABLE),
        }
    return {"available": False, "reason": UNAVAILABLE}


def _collision_npc_events(episode: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    result = str(episode.get("result", "")).lower()
    nearest = _record_values(records, "nearest_npc_dist")
    contested = _count_positive(records, "target_charger_contested") if _records_available(records) else UNAVAILABLE
    return {
        "collision_result": result == "collision",
        "nearest_npc_min": _min_or_unavailable(nearest),
        "target_charger_contested_steps": contested,
        "invalid_move_count": episode.get("invalid_move_count", UNAVAILABLE),
        "invalid_move_rate": episode.get("invalid_move_rate", UNAVAILABLE),
    }


def _timeout_uncleaned_area(episode: dict[str, Any], round_def: dict[str, Any] | None) -> dict[str, Any]:
    steps = _as_float(episode.get("steps"), None)
    max_step = _as_float((round_def or {}).get("max_step"), None)
    result = str(episode.get("result", "")).lower()
    timeout = result in {"timeout", "truncated"} or (steps is not None and max_step is not None and steps >= max_step and result != "completed")
    dirt_ratio = _as_float(episode.get("dirt_ratio"), None)
    uncleaned_ratio: float | str = UNAVAILABLE if dirt_ratio is None else round(max(0.0, 1.0 - dirt_ratio), 6)
    return {"timeout": timeout, "uncleaned_ratio": uncleaned_ratio, "max_step": max_step if max_step is not None else UNAVAILABLE}


def _round_defs(benchmark: dict[str, Any], manifest: dict[str, Any] | str) -> dict[str, dict[str, Any]]:
    rounds: dict[str, dict[str, Any]] = {}
    source_rounds = manifest.get("rounds") if isinstance(manifest, dict) else None
    if isinstance(source_rounds, list):
        for item in source_rounds:
            if isinstance(item, dict) and item.get("name"):
                rounds[str(item["name"])] = item
    benchmark_rounds = benchmark.get("rounds")
    if isinstance(benchmark_rounds, dict):
        for name, desc in benchmark_rounds.items():
            rounds.setdefault(str(name), {"name": str(name), "desc": desc})
            parsed = _parse_round_desc(str(desc))
            rounds[str(name)].update({key: value for key, value in parsed.items() if key not in rounds[str(name)]})
    return rounds


def _parse_round_desc(desc: str) -> dict[str, int]:
    numbers = [int(value) for value in re.findall(r"(\d+)\s*(?:chargers?|robots?|steps?|battery)", desc)]
    keys = ("charger_count", "robot_count", "max_step", "battery_max")
    return {key: numbers[index] for index, key in enumerate(keys) if index < len(numbers)}


def _clean_per_step(episode: dict[str, Any]) -> float | str:
    clean = _as_float(episode.get("clean_score"), None)
    steps = _as_float(episode.get("steps"), None)
    if clean is None or steps is None or steps <= 0:
        return UNAVAILABLE
    return round(clean / steps, 6)



def _infer_fail_reason(episode: dict[str, Any], round_def: dict[str, Any] | None) -> Any:
    raw = episode.get("result", UNAVAILABLE)
    result = str(raw or "").lower()
    if result not in {"", "unknown", "none", UNAVAILABLE}:
        return raw
    remaining = _as_float(episode.get("remaining_charge"), None)
    if remaining is not None and remaining <= 0.0:
        return "battery"
    steps = _as_float(episode.get("steps"), None)
    max_step = _as_float((round_def or {}).get("max_step"), None)
    if steps is not None and max_step is not None and steps >= max_step:
        return "timeout"
    clean = _as_float(episode.get("clean_score"), None)
    if clean is not None and clean <= 0.0 and steps is not None and steps <= 0.0:
        return "error"
    return raw

def _failure_bucket(episode: dict[str, Any], taxonomy: dict[str, Any], checkpoint_issue: bool) -> str:
    result = str(taxonomy.get("fail_reason", episode.get("result", "unknown")) or "unknown").lower()
    if checkpoint_issue or result in {"error", "model_error", "checkpoint"}:
        return "checkpoint issue"
    if result == "battery":
        if taxonomy.get("zero_charge_battery_fail") is True:
            return "battery depletion"
        if taxonomy.get("charge_count") == 0:
            return "missed charger"
        return "battery depletion"
    if result == "collision":
        return "collision/stuck loop"
    timeout = taxonomy.get("timeout_uncleaned_area", {}).get("timeout") if isinstance(taxonomy.get("timeout_uncleaned_area"), dict) else False
    if timeout:
        return "timeout"
    stuck = taxonomy.get("stuck_revisit_signals", {})
    if isinstance(stuck, dict):
        max_zero = _as_float(stuck.get("max_zero_progress_streak"), 0.0) or 0.0
        max_repeat = _as_float(stuck.get("max_position_repeat_16"), 0.0) or 0.0
        if max_zero >= 10 or max_repeat >= 0.5 or stuck.get("loop_episode_detected") is True:
            return "collision/stuck loop"
    if result == "completed":
        cps = _as_float(taxonomy.get("clean_per_step"), None)
        if cps is not None and cps < 0.5:
            return "inefficient coverage"
        return "completed"
    return result or "unknown"


def _lever_reasons(episode_taxonomies: list[dict[str, Any]], failure_counts: Counter[str], checkpoint_issue: bool) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {lever: [] for lever in LEVER_ORDER}
    if checkpoint_issue or failure_counts.get("checkpoint issue", 0):
        reasons["checkpoint issue"].append("checkpoint/model-load issue found in result or learner log")
    if failure_counts.get("battery depletion", 0):
        reasons["battery depletion"].append(f"{failure_counts['battery depletion']} battery-depletion episodes")
    missed = [ep for ep in episode_taxonomies if ep.get("fail_reason") == "battery" and ep.get("charge_count") == 0]
    missed += [ep for ep in episode_taxonomies if _episode_has_positive(ep, "missed_charge_opportunity")]
    if missed:
        reasons["missed charger"].append(f"{len(missed)} episodes with zero charges or missed-charge telemetry")
    inefficient = [ep for ep in episode_taxonomies if (_as_float(ep.get("clean_per_step"), 999.0) or 999.0) < 0.5]
    if inefficient:
        reasons["inefficient coverage"].append(f"{len(inefficient)} episodes below 0.5 clean-per-step")
    collision = failure_counts.get("collision/stuck loop", 0)
    stuck = [ep for ep in episode_taxonomies if _stuck_flag(ep)]
    if collision or stuck:
        reasons["collision/stuck loop"].append(f"{collision} collision/stuck failures; {len(stuck)} episodes with loop/revisit signals")
    poor_coord = [ep for ep in episode_taxonomies if _episode_has_positive(ep, "target_charger_contested_steps")]
    if poor_coord:
        reasons["poor coordination"].append(f"{len(poor_coord)} episodes with charger/NPC contention telemetry")
    if failure_counts.get("timeout", 0) or any(_timeout_flag(ep) for ep in episode_taxonomies):
        reasons["timeout"].append("timeout or max-step uncleaned-area episodes detected")
    return {lever: value for lever, value in reasons.items() if value}


def _episode_has_positive(ep: dict[str, Any], nested_key: str) -> bool:
    for section in ("stuck_revisit_signals", "collision_npc_events"):
        value = ep.get(section)
        if isinstance(value, dict) and (_as_float(value.get(nested_key), 0.0) or 0.0) > 0.0:
            return True
    return False


def _stuck_flag(ep: dict[str, Any]) -> bool:
    stuck = ep.get("stuck_revisit_signals")
    if not isinstance(stuck, dict):
        return False
    return bool(stuck.get("loop_episode_detected")) or (_as_float(stuck.get("max_zero_progress_streak"), 0.0) or 0.0) >= 10


def _timeout_flag(ep: dict[str, Any]) -> bool:
    timeout = ep.get("timeout_uncleaned_area")
    return bool(timeout.get("timeout")) if isinstance(timeout, dict) else False


def _sort_levers(reasons: dict[str, list[str]], failure_counts: Counter[str]) -> list[dict[str, Any]]:
    priority = {lever: index for index, lever in enumerate(LEVER_ORDER)}
    return [
        {"lever": lever, "reasons": reasons[lever], "failure_count": failure_counts.get(lever, 0)}
        for lever in sorted(reasons, key=lambda item: (-failure_counts.get(item, 0), priority.get(item, 999)))
    ]


def _aggregate_per_map(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        grouped[str(episode.get("map", UNAVAILABLE))].append(episode)
    rows: list[dict[str, Any]] = []
    for map_id in sorted(grouped, key=lambda item: (item == UNAVAILABLE, _as_int(item, 10**9) or 10**9, item)):
        items = grouped[map_id]
        scores = [_as_float(item.get("clean_score"), None) for item in items]
        cps = [_as_float(item.get("clean_per_step"), None) for item in items]
        scores_f = [value for value in scores if value is not None]
        cps_f = [value for value in cps if value is not None]
        bucket_counts = Counter(str(item.get("failure_bucket", "unknown")) for item in items)
        rows.append(
            {
                "map": _as_int(map_id, None) if map_id != UNAVAILABLE else UNAVAILABLE,
                "episode_count": len(items),
                "avg_clean_score": _avg(scores_f),
                "avg_clean_per_step": _avg(cps_f),
                "completed_count": bucket_counts.get("completed", 0),
                "failure_buckets": dict(sorted(bucket_counts.items())),
            }
        )
    return rows


def summarize_result(
    result_path: Path,
    episode_dir: Path | None = None,
    learner_logs: list[Path] | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    benchmark, metadata = _normalize_benchmark(_load_json(result_path), result_path)
    episodes_raw = benchmark.get("episodes")
    if not isinstance(episodes_raw, list):
        raise SystemExit(f"result file has no episodes list: {result_path}")
    episodes = [episode for episode in episodes_raw if isinstance(episode, dict)]
    manifest = _load_manifest(result_path, manifest_path)
    rounds = _round_defs(benchmark, manifest)
    records_by_key = _load_episode_records(episodes, _episode_dir(result_path, episode_dir))
    learner_log = _learner_log_summary(learner_logs or [])
    checkpoint_issue = _checkpoint_issue_present(benchmark, learner_log)

    episode_taxonomies: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    unavailable_fields: Counter[str] = Counter()
    for episode in episodes:
        key = _episode_key(episode.get("round"), episode.get("map_id"))
        records = records_by_key.get(key, [])
        battery = _battery_metrics(episode, records)
        round_name = str(episode.get("round", UNAVAILABLE))
        round_def = rounds.get(round_name)
        fail_reason = _infer_fail_reason(episode, round_def)
        clean_score = episode.get("clean_score", UNAVAILABLE)
        clean_per_step = _clean_per_step(episode)
        taxonomy: dict[str, Any] = {
            "episode_id": episode.get("episode_id") or key,
            "score": clean_score,
            "map": episode.get("map_id", UNAVAILABLE),
            "round": round_name,
            "clean_score": clean_score,
            "clean_per_step": clean_per_step,
            "fail_reason": fail_reason,
            "fail_step": episode.get("steps", UNAVAILABLE),
            "battery_end": battery["battery_end"],
            "battery_min": battery["battery_min"],
            "charge_count": episode.get("charge_count", UNAVAILABLE),
            "zero_charge_battery_fail": str(fail_reason).lower() == "battery" and (_as_float(episode.get("charge_count"), 0.0) or 0.0) <= 0.0,
            "time_on_charger": _time_on_charger(records),
            "return_override_rate": _return_override_rate(episode, records),
            "stuck_revisit_signals": _stuck_revisit_signals(episode, records),
            "collision_npc_events": _collision_npc_events(episode, records),
            "timeout_uncleaned_area": _timeout_uncleaned_area({**episode, "result": fail_reason}, round_def),
            "model_load_checkpoint_issue": checkpoint_issue,
            "episode_jsonl_records": len(records) if records else UNAVAILABLE,
        }
        taxonomy["failure_bucket"] = _failure_bucket(episode, taxonomy, checkpoint_issue)
        for field, value in taxonomy.items():
            if value == UNAVAILABLE or (isinstance(value, dict) and value.get("reason") == UNAVAILABLE):
                unavailable_fields[field] += 1
        failure_counts.update([str(taxonomy["failure_bucket"])])
        episode_taxonomies.append(taxonomy)

    lever_reasons = _lever_reasons(episode_taxonomies, failure_counts, checkpoint_issue)
    per_map = _aggregate_per_map(episode_taxonomies)
    summary = {
        "schema_version": 1,
        "generated_by": "train/tools/summarize_benchmark_failures.py",
        "source": metadata,
        "benchmark_metadata": {
            "timestamp": benchmark.get("timestamp", UNAVAILABLE),
            "checkpoint": benchmark.get("checkpoint", UNAVAILABLE),
            "policy_mode": benchmark.get("policy_mode", UNAVAILABLE),
            "git_commit": benchmark.get("git_commit", UNAVAILABLE),
            "execution": benchmark.get("execution", UNAVAILABLE),
            "overall": benchmark.get("overall", UNAVAILABLE),
            "per_round": benchmark.get("per_round", UNAVAILABLE),
            "manifest": manifest,
            "learner_log": learner_log,
        },
        "episode_count": len(episode_taxonomies),
        "episodes": episode_taxonomies,
        "per_map": per_map,
        "per_map_table": per_map,
        "failure_buckets": dict(sorted(failure_counts.items())),
        "next_recommended_levers": _sort_levers(lever_reasons, failure_counts),
        "unavailable_field_counts": dict(sorted(unavailable_fields.items())),
    }
    return summary


def _checkpoint_issue_present(benchmark: dict[str, Any], learner_log: dict[str, Any] | str) -> bool:
    if _is_checkpoint_issue(benchmark.get("checkpoint")):
        return True
    if isinstance(learner_log, dict) and learner_log.get("has_checkpoint_issue"):
        return True
    return any(str(ep.get("result", "")).lower() in {"model_error", "checkpoint"} for ep in benchmark.get("episodes", []) if isinstance(ep, dict))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize serial/parallel benchmark failures into taxonomy JSON.")
    parser.add_argument("result", type=Path, help="Benchmark result JSON or eval_results/eval_parallel_results collection JSON")
    parser.add_argument("--episode-dir", type=Path, help="Optional directory containing per-episode JSONL files")
    parser.add_argument("--learner-log", action="append", default=[], type=Path, help="Optional learner log path; repeatable")
    parser.add_argument("--manifest", type=Path, help="Optional runtime manifest JSON path")
    parser.add_argument("--output", type=Path, help="Write summary JSON to this path")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = summarize_result(args.result, args.episode_dir, args.learner_log, args.manifest)
    if args.output:
        _write_json(args.output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
