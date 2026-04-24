#!/usr/bin/env python3
"""
Safe fixed-contract holdout benchmark runner for maps [4, 7].

T2 intentionally implements the auditable dry-run contract and artifact/schema
generation without invoking training/runtime code paths that could mutate model
artifacts. Real episode execution is intentionally deferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path


ALLOWED_HOLDOUT_MAPS = [4, 7]
FIXED_CONFIG = {
    "robot_count": 4,
    "charger_count": 3,
    "max_step": 1000,
    "battery_max": 150,
    "map_random": False,
}
DETAIL_LOG_FIELDS = [
    "episode_id",
    "map_id",
    "step",
    "action",
    "planner_mode",
    "planner_target",
    "battery",
    "charger_distance",
    "return_slack",
    "reward_components",
    "fail_reason",
    "death_replay_path",
    "checkpoint_id",
]
CHECKPOINT_RELATIVE_PATHS = [
    "code/latest_model.pkl",
    "code/model.ckpt-resume.pkl",
    "code/model.ckpt-resume.meta.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed [4,7] holdout benchmark contract.")
    parser.add_argument("--maps", default="4,7", help="Comma-separated holdout maps. Only 4,7 is allowed in T2.")
    parser.add_argument("--episodes-per-map", type=int, default=10, help="Fixed holdout episodes per map.")
    parser.add_argument(
        "--checkpoint",
        default="code/model.ckpt-resume.pkl",
        help="Checkpoint path recorded in the benchmark contract.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSON path. Defaults to train/context/HOLDOUT_BENCHMARK_YYYYMMDD_HHMM.json",
    )
    parser.add_argument(
        "--detail-log-dir",
        default=None,
        help="Detail log directory. Defaults to train/holdout_detail_logs/<run_id>",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write the benchmark plan and schema without execution.")
    return parser.parse_args()


def utc_iso_from_timestamp(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest().upper()


def snapshot_file(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "size": None,
            "sha256": None,
            "mtime_utc": None,
        }
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size": int(stat.st_size),
        "sha256": sha256_file(path),
        "mtime_utc": utc_iso_from_timestamp(stat.st_mtime),
    }


def collect_model_snapshots(repo_root: Path) -> dict[str, dict[str, object]]:
    snapshots: dict[str, dict[str, object]] = {}
    for relative_path in CHECKPOINT_RELATIVE_PATHS:
        snapshots[relative_path] = snapshot_file((repo_root / relative_path).resolve())
    return snapshots


def compare_snapshots(before: dict[str, dict[str, object]], after: dict[str, dict[str, object]]) -> dict[str, object]:
    drifted = []
    for relative_path, before_entry in before.items():
        after_entry = after.get(relative_path)
        if after_entry != before_entry:
            drifted.append(relative_path)
    return {
        "before": before,
        "after": after,
        "mutation_detected": bool(drifted),
        "drifted_paths": drifted,
    }


def parse_maps(raw_maps: str) -> list[int]:
    values = []
    for chunk in raw_maps.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except ValueError as exc:
            raise ValueError(f"Invalid map id '{text}'. Expected comma-separated integers.") from exc
    if not values:
        raise ValueError("At least one map must be provided.")
    return values


def load_training_maps(train_env_conf: Path) -> list[int]:
    content = train_env_conf.read_text(encoding="utf-8")
    match = re.search(r"^\s*map\s*=\s*\[(.*?)\]", content, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise ValueError(f"Could not parse training maps from {train_env_conf}")
    raw_values = match.group(1)
    maps = []
    for token in raw_values.split(","):
        text = token.strip()
        if not text:
            continue
        maps.append(int(text))
    return maps


def validate_contract(requested_maps: list[int], episodes_per_map: int, training_maps: list[int]) -> list[str]:
    errors = []
    invalid_maps = sorted({map_id for map_id in requested_maps if map_id not in ALLOWED_HOLDOUT_MAPS})
    leaked_maps = sorted({map_id for map_id in requested_maps if map_id in training_maps})
    if invalid_maps:
        errors.append(
            "Holdout map contract only allows [4, 7]; got invalid map(s): {maps}".format(maps=invalid_maps)
        )
    if leaked_maps:
        errors.append(
            "Training-map leakage detected. Requested holdout map(s) overlap training maps {training_maps}: {leaks}".format(
                training_maps=training_maps,
                leaks=leaked_maps,
            )
        )
    if sorted(set(training_maps)) != [1, 2, 3, 5, 6, 8, 9, 10]:
        errors.append(
            "Unexpected training-map source. Expected [1, 2, 3, 5, 6, 8, 9, 10], got {maps}".format(
                maps=training_maps
            )
        )
    if episodes_per_map <= 0:
        errors.append("--episodes-per-map must be positive.")
    return errors


def build_default_output_path(repo_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    return repo_root / "train" / "context" / f"HOLDOUT_BENCHMARK_{stamp}.json"


def build_run_id() -> str:
    return f"holdout-benchmark-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def build_sample_schema(run_id: str, checkpoint_id: str, detail_log_dir: Path) -> dict[str, object]:
    return {
        "run_id": run_id,
        "schema_version": 1,
        "description": "Per-step holdout detail log schema for AI-friendly analysis.",
        "required_fields": DETAIL_LOG_FIELDS,
        "sample_record": {
            "episode_id": f"{run_id}-map4-ep01",
            "map_id": 4,
            "step": 1,
            "action": {"id": 0, "name": "MOVE_UP"},
            "planner_mode": "coverage",
            "planner_target": {"kind": "cell", "x": 12, "y": 8},
            "battery": 150,
            "charger_distance": 18,
            "return_slack": 12,
            "reward_components": {"cleaning": 0.0, "charge": 0.0, "penalty": 0.0},
            "fail_reason": None,
            "death_replay_path": None,
            "checkpoint_id": checkpoint_id,
        },
        "notes": {
            "detail_log_dir": str(detail_log_dir),
            "jsonl_pattern": "episode_<episode_id>.jsonl",
            "degrade_gracefully": True,
        },
    }


def infer_checkpoint_id(checkpoint_path: Path) -> str:
    return checkpoint_path.stem.replace(".", "-")


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    train_env_conf = repo_root / "code" / "agent_ppo" / "conf" / "train_env_conf.toml"

    try:
        requested_maps = parse_maps(args.maps)
        training_maps = load_training_maps(train_env_conf)
        errors = validate_contract(requested_maps, args.episodes_per_map, training_maps)
        if errors:
            raise ValueError("; ".join(errors))
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    run_id = build_run_id()
    checkpoint_path = (repo_root / args.checkpoint).resolve() if not Path(args.checkpoint).is_absolute() else Path(args.checkpoint)
    output_path = build_default_output_path(repo_root) if args.output is None else Path(args.output)
    if not output_path.is_absolute():
        output_path = (repo_root / output_path).resolve()
    detail_log_dir = Path(args.detail_log_dir) if args.detail_log_dir else repo_root / "train" / "holdout_detail_logs" / run_id
    if not detail_log_dir.is_absolute():
        detail_log_dir = (repo_root / detail_log_dir).resolve()

    before_snapshots = collect_model_snapshots(repo_root)
    checkpoint_id = infer_checkpoint_id(checkpoint_path)
    schema_payload = build_sample_schema(run_id, checkpoint_id, detail_log_dir)

    detail_log_dir.mkdir(parents=True, exist_ok=True)
    schema_path = detail_log_dir / "schema.json"
    schema_path.write_text(json.dumps(schema_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    after_snapshots = collect_model_snapshots(repo_root)
    mutation_guard = compare_snapshots(before_snapshots, after_snapshots)

    contract = {
        "maps": requested_maps,
        "episodes_per_map": int(args.episodes_per_map),
        "fixed_config": deepcopy(FIXED_CONFIG),
        "training_maps_source": {
            "path": str(train_env_conf),
            "maps": training_maps,
            "training_map_exclusion_verified": not any(map_id in training_maps for map_id in requested_maps),
        },
    }
    checkpoint = {
        "path": str(checkpoint_path),
        "checkpoint_id": checkpoint_id,
        "requested_relative_path": args.checkpoint,
        "snapshot": snapshot_file(checkpoint_path),
        "tracked_artifacts": before_snapshots,
    }
    episodes_planned = [
        {
            "episode_id": f"{run_id}-map{map_id}-ep{episode_index:02d}",
            "map_id": map_id,
            "planned_index": episode_index,
            "checkpoint_id": checkpoint_id,
            "detail_log_path": str(detail_log_dir / f"episode_{run_id}-map{map_id}-ep{episode_index:02d}.jsonl"),
            "status": "PLANNED" if args.dry_run else "NOT_EXECUTED",
        }
        for map_id in requested_maps
        for episode_index in range(1, args.episodes_per_map + 1)
    ]

    result = {
        "run_id": run_id,
        "status": "DRY_RUN" if args.dry_run else "NOT_IMPLEMENTED",
        "contract": contract,
        "checkpoint": checkpoint,
        "detail_log_dir": str(detail_log_dir),
        "detail_log_schema_path": str(schema_path),
        "detail_log_schema": schema_payload,
        "model_mutation_guard": mutation_guard,
        "episodes": [] if args.dry_run else [],
        "episodes_planned": episodes_planned,
        "risks": [
            {
                "code": "MODEL_MUTATION_GUARD",
                "severity": "error" if mutation_guard["mutation_detected"] else "info",
                "message": (
                    "Tracked model artifacts changed during benchmark setup."
                    if mutation_guard["mutation_detected"]
                    else "Tracked model artifacts remained unchanged during benchmark setup."
                ),
            },
            {
                "code": "REAL_EXECUTION_DEFERRED",
                "severity": "warning",
                "message": "T2 intentionally does not execute real holdout episodes. T3 should extend this runner with a safe runtime path.",
            },
        ],
        "decision_inputs": {
            "checkpoint_id": checkpoint_id,
            "maps": requested_maps,
            "episodes_per_map": int(args.episodes_per_map),
            "fixed_config": deepcopy(FIXED_CONFIG),
            "planned_episode_count": len(episodes_planned),
        },
    }

    if not args.dry_run:
        result["failure"] = {
            "reason": "REAL_EXECUTION_UNSUPPORTED_IN_T2",
            "message": "Non-dry-run execution is intentionally unsupported in T2 to avoid hidden training/runtime side effects. Use --dry-run in T2; T3 will add real execution.",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "output": str(output_path)}, ensure_ascii=True))
        return 3

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
