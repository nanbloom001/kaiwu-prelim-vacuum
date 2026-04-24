#!/usr/bin/env python3
"""
Analyze fixed holdout benchmark runner output into JSON and optional Markdown.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


CLASSIFICATION_ORDER = [
    "charger_unknown",
    "return_too_late",
    "optimistic_route_budget",
    "npc_or_path_blocked",
    "repeated_invalid_move",
    "high_score_battery_death",
    "late_battery_death",
    "unknown",
]

CLASSIFICATION_LABELS = {
    "charger_unknown": "Charger unknown / first charger not found",
    "return_too_late": "Known charger but return too late",
    "optimistic_route_budget": "Optimistic route budget / negative slack",
    "npc_or_path_blocked": "NPC or path blocked",
    "repeated_invalid_move": "Repeated invalid move / stuck pattern",
    "high_score_battery_death": "High-score battery death",
    "late_battery_death": "Late or high-step battery death",
    "unknown": "Unknown",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze holdout benchmark output.")
    parser.add_argument("--input", required=True, help="Path to runner JSON output.")
    parser.add_argument("--output-md", default=None, help="Optional path to write Markdown summary.")
    parser.add_argument(
        "--archive-run-dir",
        action="append",
        default=[],
        help="Optional train/archive/<run_id> directory for merged episode/death_replay enrichment.",
    )
    parser.add_argument(
        "--death-replay",
        action="append",
        default=[],
        help="Optional death replay JSON/JSONL file or directory. Can be repeated.",
    )
    return parser.parse_args()


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def coerce_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: object, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_int_sort_key(value: str) -> tuple[int, object]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, value)


def normalize_fail_reason(episode: dict[str, object]) -> str:
    for key in ("fail_reason", "done_reason", "status"):
        value = episode.get(key)
        if value:
            return str(value).lower()
    return "unknown"


def is_completed(reason: str) -> bool:
    normalized = str(reason or "").lower()
    if not normalized:
        return False
    return "completed" in normalized or normalized in {"success", "finished", "done"}


def is_battery_failure(reason: str) -> bool:
    normalized = str(reason or "").lower()
    if not normalized:
        return False
    if "battery" in normalized:
        return True
    return "out" in normalized and "battery" in normalized


def is_collision_failure(reason: str) -> bool:
    normalized = str(reason or "").lower()
    if not normalized:
        return False
    if "collision" in normalized:
        return True
    if "npc" in normalized and any(token in normalized for token in ("hit", "collision", "collide")):
        return True
    return False


def normalize_episode_id(episode: dict[str, object], fallback_index: int) -> str:
    for key in ("episode_id", "id"):
        value = episode.get(key)
        if value not in (None, ""):
            return str(value)
    return f"episode-{fallback_index:04d}"


def build_metrics(episodes: list[dict[str, object]]) -> dict[str, object]:
    if not episodes:
        return {
            "status": "NO_EPISODES",
            "episode_count": 0,
            "avg_clean_score": 0.0,
            "score_p10": 0.0,
            "score_p50": 0.0,
            "score_p90": 0.0,
            "completed_rate": 0.0,
            "battery_fail_rate": 0.0,
            "collision_fail_rate": 0.0,
            "avg_clean_per_step": 0.0,
            "avg_finished_steps": 0.0,
            "avg_charge_count": 0.0,
            "avg_remaining_charge": 0.0,
        }

    clean_scores = [coerce_float(ep.get("clean_score", ep.get("score", 0.0))) for ep in episodes]
    finished_steps = [coerce_float(ep.get("finished_steps", ep.get("step", 0.0))) for ep in episodes]
    charge_counts = [coerce_float(ep.get("charge_count", 0.0)) for ep in episodes]
    remaining_charge = [coerce_float(ep.get("remaining_charge", ep.get("battery", 0.0))) for ep in episodes]
    clean_per_step = []
    for score, steps in zip(clean_scores, finished_steps):
        clean_per_step.append(score / steps if steps > 0 else 0.0)

    fail_reasons = [normalize_fail_reason(ep) for ep in episodes]
    return {
        "status": "OK",
        "episode_count": len(episodes),
        "avg_clean_score": round(mean(clean_scores), 4),
        "score_p10": round(percentile(clean_scores, 0.10), 4),
        "score_p50": round(percentile(clean_scores, 0.50), 4),
        "score_p90": round(percentile(clean_scores, 0.90), 4),
        "completed_rate": round(mean([1.0 if is_completed(reason) else 0.0 for reason in fail_reasons]), 4),
        "battery_fail_rate": round(mean([1.0 if is_battery_failure(reason) else 0.0 for reason in fail_reasons]), 4),
        "collision_fail_rate": round(mean([1.0 if is_collision_failure(reason) else 0.0 for reason in fail_reasons]), 4),
        "avg_clean_per_step": round(mean(clean_per_step), 6),
        "avg_finished_steps": round(mean(finished_steps), 4),
        "avg_charge_count": round(mean(charge_counts), 4),
        "avg_remaining_charge": round(mean(remaining_charge), 4),
    }


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def extend_unique_episodes(target: list[dict[str, object]], extra_rows: list[dict[str, object]]) -> None:
    existing_ids = {str(row.get("episode_id")) for row in target if row.get("episode_id") is not None}
    for row in extra_rows:
        row_episode_id = row.get("episode_id")
        if row_episode_id is not None and str(row_episode_id) in existing_ids:
            continue
        target.append(row)
        if row_episode_id is not None:
            existing_ids.add(str(row_episode_id))


def load_archive_episode_rows(archive_run_dir: Path, warnings: list[dict[str, object]]) -> list[dict[str, object]]:
    merged_path = archive_run_dir / "ai" / "episode_summary.jsonl"
    if merged_path.exists():
        return load_jsonl(merged_path)

    rows: list[dict[str, object]] = []
    for path in sorted((archive_run_dir / "ai" / "streams").glob("episode_summary.*.jsonl")):
        try:
            rows.extend(load_jsonl(path))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(
                {
                    "path": str(path),
                    "warning": f"Failed to read archive episode summary JSONL: {type(exc).__name__}: {exc}",
                }
            )
    return rows


def collect_replay_paths(cli_paths: list[str], archive_run_dirs: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for raw_path in cli_paths:
        path = Path(raw_path).resolve()
        if path.is_dir():
            for candidate in sorted(path.rglob("*.jsonl")):
                paths.append(candidate)
            for candidate in sorted(path.rglob("*.json")):
                paths.append(candidate)
            continue
        paths.append(path)
    for archive_run_dir in archive_run_dirs:
        for candidate in sorted((archive_run_dir / "ai" / "streams").glob("death_replay.*.jsonl")):
            paths.append(candidate)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def load_replay_index(candidate_paths: list[Path], warnings: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    replay_index: dict[str, dict[str, object]] = {}
    for path in candidate_paths:
        if not path.exists():
            warnings.append({"path": str(path), "warning": "Replay path does not exist."})
            continue
        try:
            if path.suffix.lower() == ".jsonl":
                rows = load_jsonl(path)
            else:
                payload = load_json(path)
                rows = payload if isinstance(payload, list) else [payload]
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            warnings.append(
                {
                    "path": str(path),
                    "warning": f"Failed to read replay evidence: {type(exc).__name__}: {exc}",
                }
            )
            continue

        for row in rows:
            if not isinstance(row, dict):
                continue
            episode_id = row.get("episode_id")
            if episode_id is None:
                continue
            replay_index[str(episode_id)] = row
    return replay_index


def infer_explicit_replay_path(episode: dict[str, object], replay: dict[str, object] | None) -> str | None:
    for source in (episode, replay or {}):
        value = source.get("death_replay_path")
        if value:
            return str(value)
    return None


def load_inline_replay(path_text: str, warnings: list[dict[str, object]]) -> dict[str, object] | None:
    path = Path(path_text).resolve()
    if not path.exists():
        warnings.append({"path": str(path), "warning": "Episode death_replay_path is missing; JSON fields were used without replay enrichment."})
        return None
    try:
        if path.suffix.lower() == ".jsonl":
            rows = load_jsonl(path)
            return rows[-1] if rows else None
        payload = load_json(path)
        return payload if isinstance(payload, dict) else None
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        warnings.append(
            {
                "path": str(path),
                "warning": f"Episode death_replay_path could not be read; JSON fields were used without replay enrichment: {type(exc).__name__}: {exc}",
            }
        )
        return None


def snapshot_example(episode_id: str, episode: dict[str, object], replay_path: str | None) -> dict[str, object]:
    return {
        "episode_id": episode_id,
        "map_id": episode.get("map_id"),
        "fail_reason": normalize_fail_reason(episode),
        "clean_score": round(coerce_float(episode.get("clean_score", episode.get("total_score", 0.0))), 4),
        "finished_steps": coerce_int(episode.get("finished_steps", episode.get("step", 0))),
        "remaining_charge": round(coerce_float(episode.get("remaining_charge", episode.get("battery", 0.0))), 4),
        "charger_arrived_count": coerce_int(episode.get("charger_arrived_count", 0)),
        "charger_first_arrival_step": coerce_int(episode.get("charger_first_arrival_step", -1), -1),
        "invalid_move_rate": round(coerce_float(episode.get("invalid_move_rate", 0.0)), 4),
        "death_replay_path": replay_path,
    }


def get_trajectory(replay: dict[str, object] | None) -> list[dict[str, object]]:
    if not replay:
        return []
    trajectory = replay.get("trajectory")
    if not isinstance(trajectory, list):
        return []
    return [row for row in trajectory if isinstance(row, dict)]


def text_blob(*parts: object) -> str:
    rendered = []
    for part in parts:
        if part in (None, ""):
            continue
        rendered.append(str(part).lower())
    return " | ".join(rendered)


def classify_failure(episode: dict[str, object], replay: dict[str, object] | None) -> tuple[str, list[str]]:
    fail_reason = normalize_fail_reason(episode)
    clean_score = coerce_float(episode.get("clean_score", episode.get("total_score", 0.0)))
    finished_steps = coerce_float(episode.get("finished_steps", episode.get("step", 0.0)))
    max_step = max(1.0, coerce_float(episode.get("max_step", 1000.0), 1000.0))
    invalid_move_rate = coerce_float(episode.get("invalid_move_rate", 0.0))
    charger_arrived_count = coerce_int(episode.get("charger_arrived_count", 0))
    charger_first_arrival_step = coerce_int(episode.get("charger_first_arrival_step", -1), -1)
    remaining_charge = coerce_float(episode.get("remaining_charge", episode.get("battery", 0.0)))
    text = text_blob(
        episode.get("fail_reason"),
        episode.get("done_reason"),
        episode.get("status"),
        episode.get("result"),
        replay.get("result") if replay else None,
    )

    trajectory = get_trajectory(replay)
    invalid_snapshots = sum(1 for row in trajectory if row.get("invalid_move"))
    low_npc_distance = min(
        (
            coerce_float(row.get("nearest_npc_dist"), default=9999.0)
            for row in trajectory
            if row.get("nearest_npc_dist") is not None
        ),
        default=9999.0,
    )
    min_charger_slack = min(
        (coerce_float(row.get("charger_slack")) for row in trajectory if row.get("charger_slack") is not None),
        default=9999.0,
    )
    attempted_charge = any(
        str(row.get("mode", "")).lower() == "charge" or bool(row.get("should_charge")) for row in trajectory
    )

    reasons: list[str] = []

    if not is_battery_failure(fail_reason):
        if is_collision_failure(fail_reason) or "npc" in text or "collision" in text or "blocked" in text or "path" in text:
            reasons.append("non-battery failure text mentions npc/path/collision")
            return "npc_or_path_blocked", reasons
        if invalid_move_rate >= 0.25:
            reasons.append(f"invalid_move_rate={invalid_move_rate:.3f}")
            return "repeated_invalid_move", reasons
        return "unknown", reasons

    if "charger unknown" in text or "charger not found" in text or "first charger not found" in text:
        reasons.append("failure text explicitly says charger was not found")
        return "charger_unknown", reasons
    if charger_arrived_count <= 0 and charger_first_arrival_step < 0 and not attempted_charge:
        reasons.append("no charger arrivals and no replay evidence of charge-return mode")
        return "charger_unknown", reasons
    if invalid_move_rate >= 0.25 or invalid_snapshots >= 4:
        reasons.append(f"invalid_move_rate={invalid_move_rate:.3f}, invalid_snapshots={invalid_snapshots}")
        return "repeated_invalid_move", reasons
    if "npc" in text or "blocked" in text or "path" in text or low_npc_distance <= 1.5:
        if low_npc_distance <= 1.5:
            reasons.append(f"nearest_npc_dist={low_npc_distance:.2f}")
        else:
            reasons.append("failure text mentions npc/path blockage")
        return "npc_or_path_blocked", reasons
    if min_charger_slack <= -3.0:
        reasons.append(f"charger_slack reached {min_charger_slack:.2f}")
        return "optimistic_route_budget", reasons
    if charger_arrived_count > 0 and attempted_charge:
        reasons.append(
            f"charger_arrived_count={charger_arrived_count}, charger_first_arrival_step={charger_first_arrival_step}, charge mode seen"
        )
        return "return_too_late", reasons
    if clean_score > 900.0:
        reasons.append(f"clean_score={clean_score:.1f} > 900")
        return "high_score_battery_death", reasons
    if finished_steps >= max_step * 0.85 or finished_steps >= 850:
        reasons.append(f"finished_steps={finished_steps:.0f}, max_step={max_step:.0f}, remaining_charge={remaining_charge:.1f}")
        return "late_battery_death", reasons
    return "unknown", reasons


def build_failure_classification(
    episodes: list[dict[str, object]],
    replay_index: dict[str, dict[str, object]],
    replay_warnings: list[dict[str, object]],
) -> dict[str, object]:
    categories = {
        code: {"label": CLASSIFICATION_LABELS[code], "count": 0, "examples": []}
        for code in CLASSIFICATION_ORDER
    }
    analyzed_failures = 0
    analyzed_episode_ids: list[str] = []

    for index, episode in enumerate(episodes, start=1):
        fail_reason = normalize_fail_reason(episode)
        if is_completed(fail_reason):
            continue

        episode_id = normalize_episode_id(episode, index)
        analyzed_failures += 1
        analyzed_episode_ids.append(episode_id)

        replay = replay_index.get(episode_id)
        explicit_replay_path = infer_explicit_replay_path(episode, replay)
        if replay is None and explicit_replay_path:
            replay = load_inline_replay(explicit_replay_path, replay_warnings)
            if replay is not None:
                replay_index[episode_id] = replay

        category_code, reasons = classify_failure(episode, replay)
        bucket = categories[category_code]
        bucket["count"] += 1
        if len(bucket["examples"]) < 3:
            example = snapshot_example(episode_id, episode, explicit_replay_path)
            if reasons:
                example["reasons"] = reasons
            bucket["examples"].append(example)

    return {
        "status": "NO_EPISODES" if not episodes else "OK",
        "analyzed_failure_count": analyzed_failures,
        "analyzed_episode_ids": analyzed_episode_ids,
        "categories": categories,
    }


def derive_risks(
    payload: dict[str, object],
    combined: dict[str, object],
    replay_warnings: list[dict[str, object]],
) -> list[dict[str, object]]:
    risks = list(payload.get("risks") or [])
    mutation_guard = payload.get("model_mutation_guard") or {}
    if mutation_guard.get("mutation_detected"):
        risks.append(
            {
                "code": "MODEL_MUTATION_DETECTED",
                "severity": "error",
                "message": "Tracked model/checkpoint files drifted between pre/post snapshots.",
            }
        )
    if combined.get("status") == "NO_EPISODES":
        risks.append(
            {
                "code": "NO_EPISODES",
                "severity": "warning",
                "message": "Runner output contains no executed episodes yet; decision inputs are contract-only.",
            }
        )
    for warning in replay_warnings:
        risks.append(
            {
                "code": "MISSING_OR_UNREADABLE_DEATH_REPLAY",
                "severity": "warning",
                "message": f"{warning.get('warning')} Path: {warning.get('path')}",
            }
        )
    return risks


def dominant_category(failure_classification: dict[str, object]) -> tuple[str, dict[str, object]]:
    categories = failure_classification.get("categories") or {}
    winner_code = "unknown"
    winner_bucket = categories.get("unknown") or {"count": 0, "examples": []}
    for code in CLASSIFICATION_ORDER:
        bucket = categories.get(code) or {"count": 0, "examples": []}
        if bucket.get("count", 0) > winner_bucket.get("count", 0):
            winner_code = code
            winner_bucket = bucket
    return winner_code, winner_bucket


def build_next_step(
    payload: dict[str, object],
    input_path: Path,
    combined: dict[str, object],
    failure_classification: dict[str, object],
    replay_warnings: list[dict[str, object]],
    archive_run_dirs: list[Path],
    cli_replay_paths: list[Path],
) -> dict[str, object]:
    evidence_paths = [str(input_path)]
    for key in ("detail_log_schema_path", "detail_log_dir"):
        value = payload.get(key)
        if value:
            evidence_paths.append(str(value))
    for path in archive_run_dirs:
        evidence_paths.append(str(path))
    for path in cli_replay_paths:
        evidence_paths.append(str(path))
    for warning in replay_warnings:
        path = warning.get("path")
        if path:
            evidence_paths.append(str(path))
    evidence_paths = list(dict.fromkeys(evidence_paths))

    if combined.get("episode_count", 0) == 0:
        return {
            "status": "NEED_MORE_DATA",
            "recommendation": "Implement or connect a safe inference-only runtime path, then rerun the fixed 2x10 holdout baseline so real episodes and replay-backed failures exist for diagnosis.",
            "optimization_level": "infrastructure",
            "evidence_paths": evidence_paths,
        }

    top_code, top_bucket = dominant_category(failure_classification)
    avg_clean_score = coerce_float(combined.get("avg_clean_score", 0.0))
    high_score_bucket = ((failure_classification.get("categories") or {}).get("high_score_battery_death") or {}).get("count", 0)
    late_bucket = ((failure_classification.get("categories") or {}).get("late_battery_death") or {}).get("count", 0)

    if high_score_bucket <= 0 and avg_clean_score <= 900.0 and top_bucket.get("count", 0) <= 0:
        return {
            "status": "NEED_MORE_DATA",
            "recommendation": "Collect more replay-backed holdout failures before choosing a PPO-side optimization; current evidence volume is too weak and scores are not yet above the >900 threshold.",
            "optimization_level": "data",
            "evidence_paths": evidence_paths,
        }

    recommendation_map = {
        "charger_unknown": (
            "Prioritize charger-discovery diagnostics on failed maps and verify the agent can enter charge-return mode before battery enters the terminal zone.",
            "targeted",
        ),
        "return_too_late": (
            "Prioritize earlier return-to-charge decisions because the agent has charger knowledge but still starts recovery too late.",
            "targeted",
        ),
        "optimistic_route_budget": (
            "Prioritize route-budget calibration because replay evidence shows negative charger slack before death.",
            "targeted",
        ),
        "npc_or_path_blocked": (
            "Prioritize blocked-route and NPC-interference analysis because the dominant failures cluster around path obstruction.",
            "targeted",
        ),
        "repeated_invalid_move": (
            "Prioritize invalid-move/stuck diagnostics because the dominant failures show repeated ineffective actions before termination.",
            "targeted",
        ),
        "high_score_battery_death": (
            "Prioritize battery-death mitigation over coverage gains because strong-cleaning episodes are still dying before safely returning.",
            "high",
        ),
        "late_battery_death": (
            "Prioritize late-episode battery safeguards because deaths cluster near the step horizon rather than from early catastrophic errors.",
            "targeted",
        ),
        "unknown": (
            "Collect additional replay-backed failures and richer death traces because the current evidence does not isolate a single dominant root cause.",
            "data",
        ),
    }

    recommendation, optimization_level = recommendation_map.get(top_code, recommendation_map["unknown"])
    if high_score_bucket > 0 and top_code != "high_score_battery_death":
        recommendation, optimization_level = recommendation_map["high_score_battery_death"]
    elif late_bucket > 0 and top_code == "unknown":
        recommendation, optimization_level = recommendation_map["late_battery_death"]

    return {
        "status": "ACTIONABLE",
        "recommendation": recommendation,
        "optimization_level": optimization_level,
        "evidence_paths": evidence_paths,
    }


def render_markdown(result: dict[str, object]) -> str:
    combined = result["combined"]
    per_map = result["per_map"]
    failure_classification = result["failure_classification"]
    categories = failure_classification.get("categories") or {}
    next_step = result["next_step"]
    replay_warnings = result.get("missing_replay_warnings") or []
    lines = [
        "# Holdout Benchmark Analysis",
        "",
        f"- Run ID: `{result['run_id']}`",
        f"- Checkpoint: `{(result.get('checkpoint') or {}).get('path')}`",
        f"- Maps: `{result['maps']}`",
        f"- Episodes per map: `{result['episodes_per_map']}`",
        f"- Combined status: `{combined['status']}`",
        "",
        "## Combined",
        "",
        f"- Episode count: `{combined['episode_count']}`",
        f"- Avg clean score: `{combined['avg_clean_score']}`",
        f"- P10 / P50 / P90: `{combined['score_p10']}` / `{combined['score_p50']}` / `{combined['score_p90']}`",
        f"- Completed / Battery fail / Collision fail: `{combined['completed_rate']}` / `{combined['battery_fail_rate']}` / `{combined['collision_fail_rate']}`",
        "",
        "## Per Map",
        "",
        "| Map | Status | Episodes | Avg Score | P10 | P90 | Battery Fail | Collision Fail |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for map_id in sorted(per_map, key=safe_int_sort_key):
        row = per_map[map_id]
        lines.append(
            "| {map_id} | {status} | {episode_count} | {avg_clean_score} | {score_p10} | {score_p90} | {battery_fail_rate} | {collision_fail_rate} |".format(
                map_id=map_id,
                **row,
            )
        )

    lines.extend([
        "",
        "## Failure Classification",
        "",
        f"- Analyzed failure count: `{failure_classification.get('analyzed_failure_count', 0)}`",
        "",
        "| Category | Count | Example |",
        "| --- | ---: | --- |",
    ])
    for code in CLASSIFICATION_ORDER:
        bucket = categories.get(code) or {"count": 0, "examples": []}
        examples = bucket.get("examples") or []
        if examples:
            first = examples[0]
            example_text = (
                f"{first.get('episode_id')} map={first.get('map_id')} score={first.get('clean_score')} "
                f"steps={first.get('finished_steps')} replay={first.get('death_replay_path')}"
            )
        else:
            example_text = "-"
        lines.append(f"| {bucket.get('label', code)} | {bucket.get('count', 0)} | {example_text} |")

    lines.extend([
        "",
        "## Next Step",
        "",
        f"- Status: `{next_step.get('status')}`",
        f"- Optimization level: `{next_step.get('optimization_level')}`",
        f"- Recommendation: {next_step.get('recommendation')}",
        "- Evidence paths:",
    ])
    for path in next_step.get("evidence_paths") or []:
        lines.append(f"  - `{path}`")

    risks = result.get("risks") or []
    lines.extend(["", "## Risks", ""])
    if risks:
        for risk in risks:
            lines.append(f"- `{risk.get('severity', 'info')}` `{risk.get('code', 'UNKNOWN')}`: {risk.get('message', '')}")
    else:
        lines.append("- None")

    lines.extend(["", "## Missing Replay Warnings", ""])
    if replay_warnings:
        for warning in replay_warnings:
            lines.append(f"- `{warning.get('path')}`: {warning.get('warning')}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).resolve()
    payload = load_json(input_path)

    replay_warnings: list[dict[str, object]] = []
    archive_run_dirs = [Path(path).resolve() for path in args.archive_run_dir]
    cli_replay_paths = collect_replay_paths(args.death_replay, archive_run_dirs)

    episodes = list(payload.get("episodes") or [])
    for archive_run_dir in archive_run_dirs:
        if not archive_run_dir.exists():
            replay_warnings.append({"path": str(archive_run_dir), "warning": "Archive run dir does not exist."})
            continue
        extend_unique_episodes(episodes, load_archive_episode_rows(archive_run_dir, replay_warnings))

    replay_index = load_replay_index(cli_replay_paths, replay_warnings)

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for index, episode in enumerate(episodes, start=1):
        episode = dict(episode)
        episode.setdefault("episode_id", normalize_episode_id(episode, index))
        map_id = str(episode.get("map_id", "unknown"))
        grouped[map_id].append(episode)

    contract = payload.get("contract") or {}
    configured_maps = [str(map_id) for map_id in (contract.get("maps") or payload.get("maps") or [])]
    for map_id in configured_maps:
        grouped.setdefault(map_id, [])

    combined = build_metrics(episodes)
    per_map = {map_id: build_metrics(map_episodes) for map_id, map_episodes in sorted(grouped.items(), key=lambda item: safe_int_sort_key(item[0]))}
    failure_classification = build_failure_classification(episodes, replay_index, replay_warnings)
    next_step = build_next_step(
        payload=payload,
        input_path=input_path,
        combined=combined,
        failure_classification=failure_classification,
        replay_warnings=replay_warnings,
        archive_run_dirs=archive_run_dirs,
        cli_replay_paths=cli_replay_paths,
    )

    result = {
        "run_id": payload.get("run_id"),
        "checkpoint": payload.get("checkpoint"),
        "maps": contract.get("maps") or payload.get("maps") or [],
        "episodes_per_map": contract.get("episodes_per_map") or payload.get("episodes_per_map"),
        "fixed_config": contract.get("fixed_config") or {},
        "combined": combined,
        "per_map": per_map,
        "failure_classification": failure_classification,
        "missing_replay_warnings": replay_warnings,
        "next_step": next_step,
        "risks": derive_risks(payload, combined, replay_warnings),
        "decision_inputs": payload.get("decision_inputs") or {},
    }

    if args.output_md:
        output_md = Path(args.output_md).resolve()
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(result), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
