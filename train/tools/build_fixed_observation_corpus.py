#!/usr/bin/env python3
"""Build and validate fixed-observation comparison corpora.

The corpus is intentionally data-only.  Before Task 4 produces a stable
baseline session, this tool can validate checked-in fixtures or existing
benchmark episode JSONL files.  After Task 4, point ``build`` at
``train/eval_logs/<session_id>`` or ``train/eval_parallel_logs/<session_id>``
to materialize deterministic comparison nodes from ``episodes/*.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, cast

REPO_ROOT = Path(__file__).resolve().parents[2]
REQUIRED_MAPS = tuple(range(1, 11))
REQUIRED_TAGS = (
    "start",
    "early_clean",
    "charger_visible",
    "low_battery",
    "pre_return",
    "on_charger",
    "route_stall",
    "npc_near",
    "no_progress",
)
OBSERVATION_SECTIONS = ("features", "guidance", "teacher_mask", "action_or_logit", "override", "raw_fields")
DEFAULT_MAX_NODES_PER_TAG_MAP = 1
EPISODE_NAME_RE = re.compile(r"(?P<round>round_\d+)_map(?P<map_id>\d+)\.jsonl$")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _has_charger_candidates(record: dict[str, Any]) -> bool:
    candidates = record.get("charger_candidates")
    if isinstance(candidates, list) and candidates:
        return True
    return "nearest_charger_dist" in record or "charge_margin_now" in record


def infer_tags(record: dict[str, Any], first_step: int | None = None) -> list[str]:
    """Infer stable comparison tags from available benchmark step fields."""
    tags: list[str] = []
    step = _as_int(record.get("step"), 0)
    if step <= 1 or (first_step is not None and step == first_step):
        tags.append("start")
    if step <= 80 and (_as_float(record.get("cleaned_this_step")) > 0 or _as_float(record.get("dirt_cleaned")) > 0):
        tags.append("early_clean")
    if _has_charger_candidates(record):
        tags.append("charger_visible")
    battery = _as_float(record.get("battery"), 0.0)
    battery_max = max(_as_float(record.get("battery_max"), 0.0), 1.0)
    if battery / battery_max <= 0.30:
        tags.append("low_battery")
    mode = str(record.get("mode", "")).lower()
    if mode in {"1", "return", "returning", "charge", "charging"} or _as_float(record.get("charge_margin_now"), 9999.0) <= 20:
        tags.append("pre_return")
    if _as_float(record.get("nearest_charger_dist"), 9999.0) <= 1 or _as_float(record.get("charger_nearby_not_charged")) > 0:
        tags.append("on_charger")
    if _as_float(record.get("zero_progress_streak")) >= 5 or _as_float(record.get("target_progress_delta"), 1.0) < 0:
        tags.append("route_stall")
    if _as_float(record.get("nearest_npc_dist"), 9999.0) <= 10:
        tags.append("npc_near")
    if _as_float(record.get("zero_progress_streak")) >= 3 or (
        _as_float(record.get("cleaned_this_step"), 1.0) <= 0 and _as_float(record.get("target_progress_delta"), 1.0) <= 0
    ):
        tags.append("no_progress")
    return [tag for tag in REQUIRED_TAGS if tag in tags]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    """Project a benchmark step record into comparison sections."""
    feature_keys = (
        "x",
        "z",
        "battery",
        "battery_max",
        "dirt_cleaned",
        "total_dirt",
        "nearest_charger_dist",
        "nearest_npc_dist",
        "zero_progress_streak",
        "local_frontier_density",
        "position_repeat_8",
        "position_repeat_16",
    )
    guidance_keys = (
        "mode",
        "route_anchor",
        "target",
        "charger_slack",
        "future_recoverability_score",
        "anchor_return_dist",
        "planner_suggested_action",
        "planner_target_gap",
        "planner_unknown_path_ratio",
        "planner_signal_reachable",
        "target_selection_gap",
        "selected_target_rank",
    )
    override_keys = (
        "path_source",
        "fallback_to_chebyshev",
        "missed_charge_opportunity",
        "charger_nearby_not_charged",
        "target_charger_contested",
        "retarget_event",
    )
    raw_keys = (
        "step",
        "reward",
        "total_reward",
        "cleaned_this_step",
        "dirty_adjacent",
        "wall_adjacent",
        "charger_candidates",
    )
    action_keys = ("action", "sampled_action", "greedy_action", "policy_logits", "logits", "return_action_logits")
    teacher_keys = ("legal_action", "legal_act", "planner_suggested_action_legal", "action_vs_planner_match")

    def pick(keys: Iterable[str]) -> dict[str, Any]:
        return {key: record[key] for key in keys if key in record}

    return {
        "features": pick(feature_keys),
        "guidance": pick(guidance_keys),
        "teacher_mask": pick(teacher_keys),
        "action_or_logit": pick(action_keys),
        "override": pick(override_keys),
        "raw_fields": pick(raw_keys),
    }


def iter_episode_records(session_dir: Path) -> Iterable[tuple[str, int, Path, dict[str, Any], int | None]]:
    episodes_dir = session_dir / "episodes"
    if not episodes_dir.is_dir():
        raise FileNotFoundError(f"missing episodes directory: {episodes_dir}")
    for jsonl_path in sorted(episodes_dir.glob("*.jsonl")):
        match = EPISODE_NAME_RE.search(jsonl_path.name)
        round_name = match.group("round") if match else "unknown_round"
        map_id = int(match.group("map_id")) if match else 0
        first_step: int | None = None
        for line_no, line in enumerate(jsonl_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            step = _as_int(record.get("step"), line_no)
            first_step = step if first_step is None else first_step
            yield round_name, map_id, jsonl_path, record, first_step


def build_corpus(session_dir: Path, max_nodes_per_tag_map: int = DEFAULT_MAX_NODES_PER_TAG_MAP) -> dict[str, Any]:
    selected: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    source_files: set[str] = set()
    for round_name, map_id, jsonl_path, record, first_step in iter_episode_records(session_dir):
        source_files.add(str(jsonl_path.relative_to(session_dir)))
        for tag in infer_tags(record, first_step):
            key = (map_id, tag)
            if len(selected[key]) >= max_nodes_per_tag_map:
                continue
            node_id = f"map{map_id:02d}-{round_name}-step{_as_int(record.get('step')):04d}-{tag}"
            selected[key].append(
                {
                    "node_id": node_id,
                    "map_id": map_id,
                    "round": round_name,
                    "step": _as_int(record.get("step")),
                    "tags": [tag],
                    "source": str(jsonl_path.relative_to(session_dir)),
                    "observation": normalize_record(record),
                }
            )

    nodes = [node for key in sorted(selected) for node in selected[key]]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_by": "train/tools/build_fixed_observation_corpus.py",
        "materialization_status": "materialized_from_episode_jsonl",
        "source_session": str(session_dir),
        "source_files": sorted(source_files),
        "required_maps": list(REQUIRED_MAPS),
        "required_tags": list(REQUIRED_TAGS),
        "nodes": sorted(nodes, key=lambda node: (node["map_id"], node["round"], node["step"], node["tags"][0])),
        "notes": [
            "Task 4 baseline artifacts are not required for fixture validation.",
            "Target corpus materialization should use Task 4 baseline eval_logs/<session_id>/episodes/*.jsonl once available.",
        ],
    }
    payload["coverage"] = coverage_summary(payload)
    return payload


def validate_corpus(payload: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    nodes = payload.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append("nodes must be a non-empty list")
        nodes = [] if not isinstance(nodes, list) else nodes
    seen: set[str] = set()
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            errors.append(f"nodes[{idx}] is not an object")
            continue
        node_id = node.get("node_id")
        if not node_id:
            errors.append(f"nodes[{idx}] missing node_id")
        elif str(node_id) in seen:
            errors.append(f"duplicate node_id: {node_id}")
        seen.add(str(node_id))
        if node.get("map_id") not in REQUIRED_MAPS:
            errors.append(f"{node_id or idx} has unsupported map_id {node.get('map_id')!r}")
        tags = node.get("tags")
        if not isinstance(tags, list) or not tags:
            errors.append(f"{node_id or idx} missing non-empty tags")
            tags = []
        for tag in tags:
            if tag not in REQUIRED_TAGS:
                errors.append(f"{node_id or idx} has unsupported tag {tag!r}")
        observation = node.get("observation")
        if not isinstance(observation, dict):
            errors.append(f"{node_id or idx} missing observation object")
            continue
        for section in OBSERVATION_SECTIONS:
            if section not in observation:
                errors.append(f"{node_id or idx} missing observation.{section}")
            elif not isinstance(observation[section], (dict, list, int, float, str, bool, type(None))):
                errors.append(f"{node_id or idx} observation.{section} is not JSON-compatible")
    return errors


def coverage_summary(payload: dict[str, Any]) -> dict[str, Any]:
    raw_nodes = payload.get("nodes")
    nodes = cast(list[dict[str, Any]], raw_nodes if isinstance(raw_nodes, list) else [])
    maps = sorted({int(node["map_id"]) for node in nodes if isinstance(node, dict) and node.get("map_id") in REQUIRED_MAPS})
    tags = sorted({tag for node in nodes if isinstance(node, dict) for tag in node.get("tags", []) if tag in REQUIRED_TAGS})
    node_count_by_tag = {tag: 0 for tag in REQUIRED_TAGS}
    node_count_by_map = {str(map_id): 0 for map_id in REQUIRED_MAPS}
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if node.get("map_id") in REQUIRED_MAPS:
            node_count_by_map[str(node["map_id"])] += 1
        for tag in node.get("tags", []):
            if tag in node_count_by_tag:
                node_count_by_tag[tag] += 1
    errors = validate_corpus(payload)
    missing_maps = [map_id for map_id in REQUIRED_MAPS if map_id not in maps]
    missing_tags = [tag for tag in REQUIRED_TAGS if tag not in tags]
    return {
        "schema_valid": not errors,
        "validation_errors": errors,
        "node_count": len(nodes),
        "maps_present": maps,
        "missing_maps": missing_maps,
        "tags_present": tags,
        "missing_tags": missing_tags,
        "node_count_by_map": node_count_by_map,
        "node_count_by_tag": node_count_by_tag,
        "target_task4_requirement": {
            "min_nodes": 60,
            "all_maps_required": list(REQUIRED_MAPS),
            "all_tags_required": list(REQUIRED_TAGS),
            "materialization_waits_for_task4_baseline": True,
        },
        "coverage_ok_for_fixture_or_existing_artifact": not errors and not missing_maps and not missing_tags,
    }


def cmd_build(args: argparse.Namespace) -> int:
    payload = build_corpus(Path(args.source), max_nodes_per_tag_map=args.max_nodes_per_tag_map)
    _write_json(Path(args.output), payload)
    print(json.dumps({"output": args.output, "coverage": payload["coverage"]}, indent=2, sort_keys=True))
    return 0 if payload["coverage"]["schema_valid"] else 1


def cmd_validate(args: argparse.Namespace) -> int:
    payload = _load_json(Path(args.input))
    report = coverage_summary(payload)
    if args.output:
        _write_json(Path(args.output), report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["schema_valid"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or validate fixed-observation comparison corpora.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="Materialize corpus nodes from eval session episodes/*.jsonl")
    build.add_argument("--source", required=True, help="Path to train/eval_logs/<session_id> or train/eval_parallel_logs/<session_id>")
    build.add_argument("--output", required=True, help="Output corpus JSON path")
    build.add_argument("--max-nodes-per-tag-map", type=int, default=DEFAULT_MAX_NODES_PER_TAG_MAP)
    build.set_defaults(func=cmd_build)

    validate = subparsers.add_parser("validate", help="Validate a corpus JSON and emit coverage summary")
    validate.add_argument("--input", required=True, help="Input corpus JSON path")
    validate.add_argument("--output", help="Optional coverage report JSON path")
    validate.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
