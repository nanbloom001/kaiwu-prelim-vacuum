#!/usr/bin/env python3
# pyright: reportArgumentType=false, reportGeneralTypeIssues=false
"""
Safe fixed-contract all-map benchmark runner for maps [1..10].

The runner keeps the auditable dry-run contract and can launch the real
inference-only Docker benchmark. Real execution is sharded by default so all
maps can run in parallel across AISRV workers, with maps 4 and 7 receiving the
full holdout episode count and training maps receiving a smaller screen count.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import cast


ALLOWED_HOLDOUT_MAPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
HOLDOUT_SPECIAL_MAPS = [4, 7]
TRAINING_MAPS_EPISODES = 1
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
    parser = argparse.ArgumentParser(description="Run the fixed all-map benchmark contract.")
    parser.add_argument("--maps", default="1,2,3,4,5,6,7,8,9,10", help="Comma-separated maps to benchmark.")
    parser.add_argument(
        "--episodes-per-map",
        type=int,
        default=8,
        help="Episodes per holdout map (4,7). Training maps always get 1.",
    )
    parser.add_argument(
        "--training-episodes",
        type=int,
        default=TRAINING_MAPS_EPISODES,
        help="Episodes per training map (default 1).",
    )
    parser.add_argument(
        "--sharded",
        action="store_true",
        help="Compatibility flag. Real benchmark execution is sharded by default unless --serial is set.",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run benchmark on aisrv-1 only. Intended for debugging; slower than the default sharded mode.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=2,
        help="Shard count used for sharded benchmark execution. Defaults to 2.",
    )
    parser.add_argument(
        "--aisrv-count",
        type=int,
        default=None,
        help="AISRV worker count for dynamic benchmark mode. Defaults to --shard-count.",
    )
    parser.add_argument(
        "--workers-per-aisrv",
        type=int,
        default=int(os.getenv("KAIWU_PARALLEL_ENV_PER_AISRV", "4") or "4"),
        help="Parallel env/gamecore workers per AISRV for benchmark execution. Defaults to KAIWU_PARALLEL_ENV_PER_AISRV or 4.",
    )
    parser.add_argument(
        "--envs-per-aisrv",
        type=int,
        default=None,
        help="Framework env/gamecore processes per AISRV. Defaults to --workers-per-aisrv.",
    )
    parser.add_argument(
        "--scheduler",
        choices=("dynamic", "static"),
        default="dynamic",
        help="Benchmark scheduler. dynamic uses a shared task queue; static keeps legacy per-AISRV shard assignments.",
    )
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
    parser.add_argument(
        "--aggregate-shards-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--shard-root",
        default=None,
        help=argparse.SUPPRESS,
    )
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


def validate_contract(
    requested_maps: list[int],
    episodes_per_map: int,
    training_maps: list[int],
    training_episodes: int = TRAINING_MAPS_EPISODES,
) -> list[str]:
    errors = []
    invalid_maps = sorted({map_id for map_id in requested_maps if map_id not in ALLOWED_HOLDOUT_MAPS})
    if invalid_maps:
        errors.append(
            "All-map benchmark contract only allows maps [1,2,3,4,5,6,7,8,9,10]; got invalid map(s): {maps}".format(
                maps=invalid_maps
            )
        )
    if episodes_per_map <= 0:
        errors.append("--episodes-per-map must be positive.")
    if training_episodes <= 0:
        errors.append("--training-episodes must be positive.")
    return errors


def build_task_list(maps: list[int], episodes_per_map: int, training_episodes: int = 1) -> list[tuple[int, int]]:
    """Build ordered task list: training maps get training_episodes each, special holdout maps get episodes_per_map."""
    tasks = []
    for map_id in sorted(maps):
        if map_id in HOLDOUT_SPECIAL_MAPS:
            for ep_idx in range(episodes_per_map):
                tasks.append((map_id, ep_idx))
        else:
            for ep_idx in range(training_episodes):
                tasks.append((map_id, ep_idx))
    return tasks


def validate_sharding(sharded: bool, shard_count: int) -> list[str]:
    if not sharded:
        return []
    if shard_count <= 0:
        return ["--shard-count must be positive when --sharded is enabled."]
    return []


def validate_workers_per_aisrv(workers_per_aisrv: int) -> list[str]:
    if workers_per_aisrv <= 0:
        return ["--workers-per-aisrv must be positive."]
    return []


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


def build_episode_plan(
    requested_maps: list[int],
    episodes_per_map: int,
    training_episodes: int,
    run_id: str,
    checkpoint_id: str,
    detail_log_dir: Path,
    dry_run: bool,
) -> list[dict[str, object]]:
    episodes_planned = []
    task_list = build_task_list(
        maps=requested_maps,
        episodes_per_map=episodes_per_map,
        training_episodes=training_episodes,
    )
    for planned_index, (map_id, zero_based_episode_index) in enumerate(task_list, start=1):
        episode_index = zero_based_episode_index + 1
        episode_id = f"{run_id}-map{map_id}-ep{episode_index:02d}"
        episodes_planned.append(
            {
                "episode_id": episode_id,
                "map_id": map_id,
                "ep_idx": episode_index,
                "planned_index": planned_index,
                "checkpoint_id": checkpoint_id,
                "detail_log_path": str(detail_log_dir / f"episode_{episode_id}.jsonl"),
                "status": "PLANNED" if dry_run else "NOT_EXECUTED",
            }
        )
    return episodes_planned


def expected_episode_pairs(
    requested_maps: list[int],
    episodes_per_map: int,
    training_episodes: int,
) -> set[tuple[int, int]]:
    return {
        (map_id, zero_based_episode_index + 1)
        for map_id, zero_based_episode_index in build_task_list(
            maps=requested_maps,
            episodes_per_map=episodes_per_map,
            training_episodes=training_episodes,
        )
    }


def expected_episode_counts_by_map(
    requested_maps: list[int],
    episodes_per_map: int,
    training_episodes: int,
) -> dict[int, int]:
    return {
        map_id: episodes_per_map if map_id in HOLDOUT_SPECIAL_MAPS else training_episodes
        for map_id in requested_maps
    }


def build_benchmark_metadata(
    requested_maps: list[int],
    episodes_per_map: int,
    training_episodes: int,
) -> dict[str, object]:
    return {
        "benchmark_mode": "all_maps",
        "holdout_special_maps": list(HOLDOUT_SPECIAL_MAPS),
        "holdout_episodes_per_map": episodes_per_map,
        "training_episodes_per_map": training_episodes,
        "total_episodes": len(
            build_task_list(
                maps=requested_maps,
                episodes_per_map=episodes_per_map,
                training_episodes=training_episodes,
            )
        ),
    }


def build_shard_assignment_payload(
    shard_index: int,
    shard_count: int,
    episodes: list[dict[str, object]],
    requested_maps: list[int],
    episodes_per_map: int,
    training_episodes: int,
    run_id: str,
) -> dict[str, object]:
    return {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "episodes": episodes,
        "maps": list(requested_maps),
        "episodes_per_map": episodes_per_map,
        "training_episodes_per_map": training_episodes,
        "benchmark_metadata": build_benchmark_metadata(
            requested_maps=requested_maps,
            episodes_per_map=episodes_per_map,
            training_episodes=training_episodes,
        ),
        "run_id": run_id,
    }


def build_shard_assignments(
    episodes_planned: list[dict[str, object]],
    shard_count: int,
    requested_maps: list[int],
    episodes_per_map: int,
    training_episodes: int,
    run_id: str,
    strategy: str | None = None,
) -> list[dict[str, object]]:
    effective_strategy = strategy or infer_shard_strategy(
        requested_maps=requested_maps,
        shard_count=shard_count,
    )
    shards = [[] for _ in range(shard_count)]
    if effective_strategy == "map_partition":
        map_to_shard = {map_id: idx for idx, map_id in enumerate(requested_maps)}
        for episode in episodes_planned:
            shards[map_to_shard[int(episode["map_id"])]].append(episode)
    else:
        for episode_position, episode in enumerate(episodes_planned):
            shards[episode_position % shard_count].append(episode)
    return [
        build_shard_assignment_payload(
            shard_index=shard_index,
            shard_count=shard_count,
            episodes=episodes,
            requested_maps=requested_maps,
            episodes_per_map=episodes_per_map,
            training_episodes=training_episodes,
            run_id=run_id,
        )
        for shard_index, episodes in enumerate(shards)
    ]


def infer_shard_strategy(requested_maps: list[int], shard_count: int) -> str:
    if shard_count == len(requested_maps) and len(set(requested_maps)) == len(requested_maps):
        return "map_partition"
    return "round_robin_episode"


def build_sharding_metadata(
    shard_assignments: list[dict[str, object]],
    shard_count: int,
    run_id: str,
    strategy: str,
) -> dict[str, object]:
    return {
        "enabled": True,
        "strategy": strategy,
        "shard_count": shard_count,
        "run_id": run_id,
        "assignments": shard_assignments,
    }


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def shard_result_path(results_dir: Path, shard_index: int) -> Path:
    return results_dir / f"shard_{shard_index}.json"


def shard_done_marker_path(done_dir: Path, shard_index: int) -> Path:
    return done_dir / f".done_shard_{shard_index}"


def locate_expected_shard_paths(shard_root: Path, shard_count: int) -> list[dict[str, object]]:
    results_dir = shard_root / "results"
    done_dir = shard_root / "done"
    return [
        {
            "shard_index": shard_index,
            "result_path": shard_result_path(results_dir, shard_index),
            "done_path": shard_done_marker_path(done_dir, shard_index),
        }
        for shard_index in range(shard_count)
    ]


def prepare_sharded_runtime_paths(repo_root: Path) -> dict[str, Path]:
    holdout_shards_dir = repo_root / "code" / "holdout_shards"
    assignments_dir = holdout_shards_dir / "assignments"
    done_marker = repo_root / "code" / ".benchmark_done"
    result_path = repo_root / "code" / "holdout_result.json"

    if holdout_shards_dir.exists():
        shutil.rmtree(holdout_shards_dir)
    assignments_dir.mkdir(parents=True, exist_ok=True)

    for target in (done_marker, result_path):
        if target.exists():
            target.unlink()

    return {
        "holdout_shards_dir": holdout_shards_dir,
        "assignments_dir": assignments_dir,
        "done_marker": done_marker,
        "result_path": result_path,
    }


def resolve_aisrv_hostnames(benchmark_project: str, shard_count: int) -> list[tuple[int, str, str]]:
    import subprocess

    resolved = []
    for shard_number in range(1, shard_count + 1):
        container_name = f"{benchmark_project}-aisrv-{shard_number}"
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.Config.Hostname}}", container_name],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to resolve hostname for container {container_name}: {result.stderr.strip() or result.stdout.strip()}"
            )
        hostname = result.stdout.strip()
        if not hostname:
            raise RuntimeError(f"Failed to resolve hostname for container {container_name}: docker inspect returned empty output")
        resolved.append((shard_number - 1, container_name, hostname))
    return resolved


def write_shard_assignment_files(
    assignments_dir: Path,
    shard_assignments: list[dict[str, object]],
    resolved_hostnames: list[tuple[int, str, str]],
) -> list[dict[str, object]]:
    manifest = []
    for shard_index, container_name, hostname in resolved_hostnames:
        payload = deepcopy(shard_assignments[shard_index])
        payload["container_name"] = container_name
        payload["hostname"] = hostname
        assignment_path = assignments_dir / f"{hostname}.json"
        atomic_write_json(assignment_path, cast(dict[str, object], payload))
        manifest.append({
            "shard_index": shard_index,
            "container_name": container_name,
            "hostname": hostname,
            "assignment_path": str(assignment_path),
        })
    return manifest


def write_indexed_shard_assignment_files(
    assignments_dir: Path,
    shard_assignments: list[dict[str, object]],
) -> list[dict[str, object]]:
    manifest = []
    for assignment in shard_assignments:
        shard_index = int(assignment["shard_index"])
        assignment_path = assignments_dir / f"shard_{shard_index}.json"
        atomic_write_json(assignment_path, cast(dict[str, object], deepcopy(assignment)))
        manifest.append(
            {
                "shard_index": shard_index,
                "assignment_path": str(assignment_path),
            }
        )
    return manifest


def wait_for_all_shard_markers(expected_paths: list[dict[str, object]], max_wait: int = 3600, poll_interval: int = 10) -> bool:
    elapsed = 0
    while elapsed < max_wait:
        if all(cast(Path, entry["done_path"]).exists() for entry in expected_paths):
            return True
        time.sleep(poll_interval)
        elapsed += poll_interval
    return all(cast(Path, entry["done_path"]).exists() for entry in expected_paths)


def load_strict_json_file(path: Path, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {label} {path}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"Failed to read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {label} {path}, got {type(payload).__name__}")
    return payload


def aggregate_episode_list(episodes: list[dict[str, object]]) -> dict[str, object]:
    if not episodes:
        return {
            "episode_count": 0,
            "avg_clean_score": 0.0,
            "completed_rate": 0.0,
            "battery_fail_rate": 0.0,
            "collision_fail_rate": 0.0,
        }

    wins = [ep for ep in episodes if ep["result"] == "completed"]
    fails_battery = [ep for ep in episodes if ep["result"] == "battery"]
    fails_collision = [ep for ep in episodes if ep["result"] == "collision"]

    scores = [float(ep["clean_score"]) for ep in episodes]
    sorted_scores = sorted(scores)

    return {
        "episode_count": len(episodes),
        "win_episode_count": len(wins),
        "avg_clean_score": round(sum(scores) / len(scores), 1),
        "score_p10": sorted_scores[max(0, int(len(sorted_scores) * 0.1) - 1)] if sorted_scores else 0.0,
        "score_p50": sorted_scores[len(sorted_scores) // 2] if sorted_scores else 0.0,
        "score_p90": sorted_scores[min(len(sorted_scores) - 1, int(len(sorted_scores) * 0.9))] if sorted_scores else 0.0,
        "min_clean_score": min(scores) if scores else 0.0,
        "max_clean_score": max(scores) if scores else 0.0,
        "completed_rate": round(len(wins) / len(episodes), 4),
        "battery_fail_rate": round(len(fails_battery) / len(episodes), 4),
        "collision_fail_rate": round(len(fails_collision) / len(episodes), 4),
        "avg_steps": round(sum(float(ep["steps"]) for ep in episodes) / len(episodes), 1),
        "avg_charge_count": round(sum(float(ep["charge_count"]) for ep in episodes) / len(episodes), 2),
        "avg_invalid_move_rate": round(sum(float(ep["invalid_move_rate"]) for ep in episodes) / len(episodes), 4),
        "avg_dirt_ratio": round(sum(float(ep["dirt_ratio"]) for ep in episodes) / len(episodes), 4),
        "avg_total_reward": round(sum(float(ep["total_reward"]) for ep in episodes) / len(episodes), 1),
    }


def aggregate_results(episode_results: list[dict[str, object]], requested_maps: list[int]) -> dict[str, object]:
    per_map = {}
    for map_id in requested_maps:
        map_eps = [ep for ep in episode_results if int(ep["map_id"]) == map_id]
        per_map[f"map{map_id}"] = aggregate_episode_list(map_eps)
    overall = aggregate_episode_list(episode_results)
    return {"per_map": per_map, "overall": overall}


def validate_episode_identity(episode: dict[str, object], shard_index: int, source_path: Path) -> tuple[int, int]:
    if "map_id" not in episode or "ep_idx" not in episode:
        raise ValueError(
            f"Shard {shard_index} result {source_path} contains episode missing required (map_id, ep_idx) key fields"
        )
    try:
        map_id = int(episode["map_id"])
        ep_idx = int(episode["ep_idx"])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Shard {shard_index} result {source_path} has non-integer episode key values: "
            f"map_id={episode.get('map_id')} ep_idx={episode.get('ep_idx')}"
        ) from exc
    return map_id, ep_idx


def validate_common_shard_payload(
    payload: dict[str, object],
    shard_index: int,
    expected_shards: int,
    requested_maps: list[int],
    episodes_per_map: int,
    base_payload: dict[str, object] | None,
    run_id: str | None,
    source_path: Path,
) -> None:
    execution = payload.get("execution")
    if not isinstance(execution, dict):
        raise ValueError(f"Shard {shard_index} result {source_path} is missing execution metadata")
    if execution.get("mode") != "sharded":
        raise ValueError(f"Shard {shard_index} result {source_path} is not marked execution.mode='sharded'")
    if int(execution.get("shard_index", -1)) != shard_index:
        raise ValueError(f"Shard {shard_index} result {source_path} reported mismatched execution.shard_index")
    if int(execution.get("shard_count", -1)) != expected_shards:
        raise ValueError(f"Shard {shard_index} result {source_path} reported mismatched execution.shard_count")
    if sorted(int(map_id) for map_id in payload.get("maps") or []) != sorted(requested_maps):
        raise ValueError(f"Shard {shard_index} result {source_path} reported unexpected maps")
    if int(payload.get("episodes_per_map", -1)) != episodes_per_map:
        raise ValueError(f"Shard {shard_index} result {source_path} reported unexpected episodes_per_map")
    shard_run_id = execution.get("run_id")
    if run_id is not None and shard_run_id not in (None, run_id):
        raise ValueError(f"Shard {shard_index} result {source_path} reported unexpected execution.run_id={shard_run_id}")
    if base_payload is None:
        return
    for key in ("schema_version", "checkpoint", "contract", "round_def", "maps", "episodes_per_map"):
        if payload.get(key) != base_payload.get(key):
            raise ValueError(f"Shard {shard_index} result {source_path} mismatched shard-invariant field '{key}'")


def canonicalize_shard_assignments(shard_assignments: list[dict[str, object]], run_id: str | None) -> list[dict[str, object]]:
    canonical = []
    for assignment in shard_assignments:
        episodes = assignment.get("episodes") or []
        canonical.append(
            {
                "shard_index": int(assignment["shard_index"]),
                "shard_count": int(assignment["shard_count"]),
                "maps": [int(map_id) for map_id in cast(list[object], assignment.get("maps") or [])],
                "episodes_per_map": int(assignment["episodes_per_map"]),
                "training_episodes_per_map": int(assignment.get("training_episodes_per_map", TRAINING_MAPS_EPISODES)),
                "benchmark_metadata": assignment.get("benchmark_metadata"),
                "run_id": run_id if run_id is not None else assignment.get("run_id"),
                "episodes": [
                    {
                        "map_id": int(cast(dict[str, object], episode)["map_id"]),
                        "ep_idx": int(cast(dict[str, object], episode)["ep_idx"]),
                    }
                    for episode in cast(list[object], episodes)
                ],
            }
        )
    return canonical


def shard_elapsed_seconds(payloads: list[dict[str, object]]) -> list[dict[str, object]]:
    elapsed = []
    for payload in payloads:
        execution = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
        elapsed.append(
            {
                "shard_index": int(cast(dict[str, object], execution).get("shard_index", -1)),
                "elapsed_seconds": round(float(payload.get("elapsed_seconds", 0.0) or 0.0), 1),
            }
        )
    return sorted(elapsed, key=lambda item: int(item["shard_index"]))


def aggregate_sharded_results(
    shard_root: Path,
    shard_count: int,
    requested_maps: list[int],
    episodes_per_map: int,
    training_episodes: int,
    shard_assignments: list[dict[str, object]],
    output_path: Path,
    run_id: str | None,
    wait_for_done: bool,
    max_wait: int = 3600,
    poll_interval: int = 10,
) -> dict[str, object]:
    expected_paths = locate_expected_shard_paths(shard_root=shard_root, shard_count=shard_count)
    if wait_for_done and not wait_for_all_shard_markers(expected_paths, max_wait=max_wait, poll_interval=poll_interval):
        missing_done = [
            str(cast(Path, entry["done_path"]))
            for entry in expected_paths
            if not cast(Path, entry["done_path"]).exists()
        ]
        raise ValueError(f"Timed out waiting for shard done markers: {missing_done}")

    shard_payloads: list[dict[str, object]] = []
    source_files: list[str] = []
    all_episodes: list[dict[str, object]] = []
    seen_pairs: set[tuple[int, int]] = set()
    base_payload: dict[str, object] | None = None

    for entry in expected_paths:
        shard_index = cast(int, entry["shard_index"])
        done_path = cast(Path, entry["done_path"])
        result_path = cast(Path, entry["result_path"])
        if not done_path.exists():
            raise ValueError(f"Missing shard done marker for shard {shard_index}: {done_path}")
        if not result_path.exists():
            raise ValueError(f"Missing shard result for shard {shard_index}: {result_path}")

        payload = load_strict_json_file(result_path, label=f"shard_{shard_index}")
        validate_common_shard_payload(
            payload=payload,
            shard_index=shard_index,
            expected_shards=shard_count,
            requested_maps=requested_maps,
            episodes_per_map=episodes_per_map,
            base_payload=base_payload,
            run_id=run_id,
            source_path=result_path,
        )
        episodes = payload.get("episodes")
        if not isinstance(episodes, list):
            raise ValueError(f"Shard {shard_index} result {result_path} must contain an episodes list")

        for raw_episode in episodes:
            if not isinstance(raw_episode, dict):
                raise ValueError(f"Shard {shard_index} result {result_path} contains a non-object episode entry")
            episode = dict(raw_episode)
            pair = validate_episode_identity(episode=episode, shard_index=shard_index, source_path=result_path)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate episode key detected across shards: map_id={pair[0]} ep_idx={pair[1]}")
            expected_pairs = expected_episode_pairs(
                requested_maps=requested_maps,
                episodes_per_map=episodes_per_map,
                training_episodes=training_episodes,
            )
            if pair not in expected_pairs:
                raise ValueError(
                    f"Unexpected episode key in shard {shard_index} result {result_path}: map_id={pair[0]} ep_idx={pair[1]}"
                )
            seen_pairs.add(pair)
            all_episodes.append(episode)

        shard_payloads.append(payload)
        source_files.append(str(result_path))
        if base_payload is None:
            base_payload = payload

    expected_pairs = expected_episode_pairs(
        requested_maps=requested_maps,
        episodes_per_map=episodes_per_map,
        training_episodes=training_episodes,
    )
    missing_pairs = sorted(expected_pairs - seen_pairs)
    unexpected_pairs = sorted(seen_pairs - expected_pairs)
    if missing_pairs:
        raise ValueError(f"Missing expected episodes from shard results: {missing_pairs}")
    if unexpected_pairs:
        raise ValueError(f"Unexpected episodes found in shard results: {unexpected_pairs}")

    per_map_counts = {map_id: 0 for map_id in requested_maps}
    for episode in all_episodes:
        per_map_counts[int(episode["map_id"])] += 1
    expected_counts = expected_episode_counts_by_map(
        requested_maps=requested_maps,
        episodes_per_map=episodes_per_map,
        training_episodes=training_episodes,
    )
    wrong_counts = {map_id: count for map_id, count in per_map_counts.items() if count != expected_counts[map_id]}
    if wrong_counts:
        raise ValueError(f"Per-map episode counts do not match expected counts {expected_counts}: {wrong_counts}")

    ordered_episodes = sorted(all_episodes, key=lambda ep: (int(ep["map_id"]), int(ep["ep_idx"])))
    aggregated_metrics = aggregate_results(ordered_episodes, requested_maps)
    assert base_payload is not None
    execution_run_id = run_id or cast(dict[str, object], base_payload.get("execution") or {}).get("run_id")

    execution = {
        "mode": "sharded",
        "run_id": execution_run_id,
        "expected_shards": shard_count,
        "completed_shards": len(shard_payloads),
        "shard_assignments": canonicalize_shard_assignments(shard_assignments, run_id=execution_run_id),
        "source_files": source_files,
        "errors": [],
    }

    per_shard_elapsed = shard_elapsed_seconds(shard_payloads)
    wall_elapsed_seconds = round(
        max((float(item["elapsed_seconds"]) for item in per_shard_elapsed), default=0.0),
        1,
    )
    sum_shard_elapsed_seconds = round(
        sum(float(item["elapsed_seconds"]) for item in per_shard_elapsed),
        1,
    )
    aggregate_payload: dict[str, object] = {
        "run_id": execution["run_id"],
        "schema_version": base_payload.get("schema_version"),
        "timestamp": base_payload.get("timestamp"),
        "checkpoint": base_payload.get("checkpoint"),
        "benchmark_mode": "all_maps",
        "holdout_special_maps": list(HOLDOUT_SPECIAL_MAPS),
        "holdout_episodes_per_map": episodes_per_map,
        "training_episodes_per_map": training_episodes,
        "total_episodes": len(expected_pairs),
        "elapsed_seconds": wall_elapsed_seconds,
        "wall_elapsed_seconds": wall_elapsed_seconds,
        "sum_shard_elapsed_seconds": sum_shard_elapsed_seconds,
        "per_shard_elapsed_seconds": per_shard_elapsed,
        "contract": base_payload.get("contract"),
        "benchmark_metadata": build_benchmark_metadata(
            requested_maps=requested_maps,
            episodes_per_map=episodes_per_map,
            training_episodes=training_episodes,
        ),
        "round_def": base_payload.get("round_def"),
        "maps": base_payload.get("maps"),
        "episodes_per_map": base_payload.get("episodes_per_map"),
        "overall": aggregated_metrics["overall"],
        "per_map": aggregated_metrics["per_map"],
        "episodes": ordered_episodes,
        "execution": execution,
    }
    atomic_write_json(output_path, aggregate_payload)
    return aggregate_payload


def verify_aisrv_env_count_config(subprocess_module, container: str, expected: int, attempts: int = 30) -> bool:
    target = "/data/projects/robot_vacuum/kaiwudrl/conf/kaiwudrl/configure.toml"
    command = (
        "python3 - <<'PY'\n"
        "from pathlib import Path\n"
        f"path = Path('{target}')\n"
        "value = ''\n"
        "if path.exists():\n"
        "    for line in path.read_text().splitlines():\n"
        "        if line.strip().split('=', 1)[0].strip() == 'aisrv_connect_to_kaiwu_env_count':\n"
        "            value = line.split('=', 1)[1].strip().strip('\"')\n"
        "print(value)\n"
        "PY"
    )
    for _ in range(attempts):
        result = subprocess_module.run(
            ["docker", "exec", container, "sh", "-lc", command],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip() == str(expected):
            return True
        time.sleep(1)
    return False


def _launch_benchmark_via_docker(
    repo_root: Path,
    requested_maps: list[int],
    episodes_per_map: int,
    training_episodes: int,
    checkpoint_path: Path,
    output_path: Path,
    run_id: str,
    sharded: bool = False,
    shard_count: int = 1,
    workers_per_aisrv: int = 1,
    shard_assignments: list[dict[str, object]] | None = None,
    scheduler: str = "dynamic",
) -> int:
    """Launch the holdout benchmark via docker compose benchmark overlay.

    This stops training containers (port conflict), starts benchmark containers
    with KAIWU_BENCHMARK_MODE=1, waits for completion, copies the result, then
    optionally restarts training containers.
    """
    import subprocess

    maps_str = ",".join(str(m) for m in requested_maps)
    planned_episode_count = len(
        build_task_list(
            maps=requested_maps,
            episodes_per_map=episodes_per_map,
            training_episodes=training_episodes,
        )
    )
    compose_file = str(repo_root / "train" / ".docker-compose.yaml")
    benchmark_overlay = str(repo_root / "train" / ".docker-compose.benchmark.yaml")

    # Compute checkpoint path relative to /workspace/code (container mount)
    code_dir = repo_root / "code"
    try:
        checkpoint_relative = checkpoint_path.relative_to(code_dir)
        checkpoint_container = str(checkpoint_relative).replace("\\", "/")
    except ValueError:
        checkpoint_container = "model.ckpt-resume.pkl"

    env = dict(os.environ)
    env.update({
        "KAIWU_BENCHMARK_MODE": "1",
        "KAIWU_BENCHMARK_PARALLEL_MODE": "1" if scheduler == "dynamic" else "0",
        "KAIWU_BENCHMARK_SCHEDULER": scheduler,
        "KAIWU_BENCHMARK_MAPS": maps_str,
        "KAIWU_BENCHMARK_EPISODES_PER_MAP": str(episodes_per_map),
        "KAIWU_BENCHMARK_TRAINING_EPISODES_PER_MAP": str(training_episodes),
        "KAIWU_BENCHMARK_CHECKPOINT": checkpoint_container,
        "KAIWU_BENCHMARK_SHARDED": "1" if sharded else "0",
        "KAIWU_BENCHMARK_SHARD_COUNT": str(shard_count if sharded else 1),
        "KAIWU_BENCHMARK_WORKERS_PER_AISRV": str(workers_per_aisrv),
        "KAIWU_BENCHMARK_WORKER_COUNT": str(shard_count if sharded else 1),
        "KAIWU_BENCHMARK_ENVS_PER_WORKER": str(workers_per_aisrv),
        "KAIWU_BENCHMARK_RUNTIME_DIR": "/workspace/code/holdout_shards/dynamic",
        "KAIWU_AISRV_NUM": str(shard_count if sharded else 1),
        "KAIWU_PARALLEL_ENV_PER_AISRV": str(workers_per_aisrv),
        "KAIWU_GAMECORE_NUM": str((shard_count if sharded else 1) * workers_per_aisrv),
    })
    train_project = "kaiwu-train"
    benchmark_project = "kaiwu-train"  # Must match training project name for framework service discovery

    runtime_paths = None
    if sharded:
        runtime_paths = prepare_sharded_runtime_paths(repo_root)
        if shard_assignments is None:
            raise RuntimeError("Sharded launch requires shard assignments.")
        indexed_assignment_manifest = write_indexed_shard_assignment_files(
            assignments_dir=runtime_paths["assignments_dir"],
            shard_assignments=shard_assignments,
        )
        print(
            json.dumps(
                {
                    "status": "SHARDED_INDEX_ASSIGNMENTS_READY",
                    "run_id": run_id,
                    "assignments": indexed_assignment_manifest,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    print(f"[HOLDOUT-BENCH] Starting benchmark via docker compose...", flush=True)
    print(
        f"  maps={maps_str} episodes_per_map={episodes_per_map} checkpoint={checkpoint_container} "
        f"scheduler={scheduler} sharded={int(sharded)} aisrv_count={shard_count} envs_per_aisrv={workers_per_aisrv} "
        f"gamecore_num={(shard_count if sharded else 1) * workers_per_aisrv}",
        flush=True,
    )

    # Step 1: Stop training containers (port conflict)
    training_was_running = False
    try:
        check = subprocess.run(
            ["docker", "compose", "-p", train_project, "-f", compose_file, "ps", "-q"],
            capture_output=True, text=True, timeout=30,
        )
        if check.returncode == 0 and check.stdout.strip():
            training_was_running = True
            print(f"[HOLDOUT-BENCH] Stopping training containers (port conflict)...", flush=True)
            subprocess.run(
                ["docker", "compose", "-p", train_project, "-f", compose_file, "down"],
                capture_output=True, text=True, timeout=120,
            )
            print("[HOLDOUT-BENCH] Training containers stopped.", flush=True)
    except Exception as exc:
        print(f"[HOLDOUT-BENCH] Warning: could not check/stop training: {exc}", flush=True)

    try:
        # Step 2: Start benchmark containers
        up_cmd = [
            "docker", "compose",
            "-p", benchmark_project,
            "-f", compose_file,
            "-f", benchmark_overlay,
            "--profile", "distributed",
            "up", "-d",
        ]
        print(f"  Running: {' '.join(up_cmd)}", flush=True)
        result = subprocess.run(up_cmd, capture_output=True, text=True, timeout=120, env=env)
        if result.returncode != 0:
            print(f"[HOLDOUT-BENCH] docker compose up failed:", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return 4

        aisrv_container = f"{benchmark_project}-aisrv-1"

        if sharded:
            assert runtime_paths is not None
            resolved_hostnames = resolve_aisrv_hostnames(benchmark_project=benchmark_project, shard_count=shard_count)
            assignment_manifest = write_shard_assignment_files(
                assignments_dir=runtime_paths["assignments_dir"],
                shard_assignments=shard_assignments,
                resolved_hostnames=resolved_hostnames,
            )
            print(
                json.dumps(
                    {
                        "status": "SHARDED_ASSIGNMENTS_READY",
                        "run_id": run_id,
                        "assignments": assignment_manifest,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )

        if scheduler == "dynamic":
            if not verify_aisrv_env_count_config(subprocess, aisrv_container, workers_per_aisrv):
                print(
                    "[HOLDOUT-BENCH] aisrv_connect_to_kaiwu_env_count did not reach "
                    f"{workers_per_aisrv} inside {aisrv_container}",
                    file=sys.stderr,
                )
                return 8

        # Step 3: Wait for completion marker in aisrv container
        if scheduler == "dynamic":
            print(f"[HOLDOUT-BENCH] Waiting for dynamic benchmark completion...", flush=True)
            max_wait = 3600
            poll_interval = 10
            elapsed = 0
            benchmark_completed = False
            while elapsed < max_wait:
                try:
                    check = subprocess.run(
                        ["docker", "exec", aisrv_container, "test", "-f", "/workspace/code/.benchmark_done"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if check.returncode == 0:
                        print(f"[HOLDOUT-BENCH] Dynamic benchmark completed after ~{elapsed}s", flush=True)
                        benchmark_completed = True
                        break
                    progress = subprocess.run(
                        [
                            "docker", "exec", aisrv_container, "sh", "-lc",
                            "find /workspace/code/holdout_shards/dynamic/tasks/completed -maxdepth 1 -name '*.json' 2>/dev/null | wc -l",
                        ],
                        capture_output=True, text=True, timeout=10,
                    )
                    if progress.returncode == 0:
                        print(
                            f"[HOLDOUT-BENCH] dynamic progress completed={progress.stdout.strip()}/"
                            f"{planned_episode_count} elapsed={elapsed}s",
                            flush=True,
                        )
                except Exception:
                    pass
                time.sleep(poll_interval)
                elapsed += poll_interval

            if benchmark_completed:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(repo_root / "code" / "holdout_result.json", output_path)
                print(f"[HOLDOUT-BENCH] Dynamic result copied to {output_path}", flush=True)
                try:
                    payload = json.loads(output_path.read_text(encoding="utf-8"))
                    execution = payload.get("execution") if isinstance(payload.get("execution"), dict) else {}
                    observed_workers = int(cast(dict[str, object], execution).get("observed_worker_count", 0) or 0)
                    expected_workers = min(int(cast(dict[str, object], execution).get("logical_worker_count", 0) or 0), planned_episode_count)
                    if expected_workers > 0 and observed_workers < expected_workers:
                        print(
                            "[HOLDOUT-BENCH] Dynamic benchmark completed but did not expose requested concurrency: "
                            f"observed_worker_count={observed_workers}, expected={expected_workers}",
                            file=sys.stderr,
                        )
                        return 9
                except Exception as exc:
                    print(f"[HOLDOUT-BENCH] Warning: could not validate dynamic worker count: {exc}", flush=True)
            else:
                print("[HOLDOUT-BENCH] Dynamic benchmark did not complete before timeout.", file=sys.stderr)
        elif sharded:
            assert runtime_paths is not None
            shard_root = runtime_paths["holdout_shards_dir"]
            print(
                f"[HOLDOUT-BENCH] Waiting for shard done markers under {shard_root / 'done'}...",
                flush=True,
            )
            try:
                aggregate_sharded_results(
                    shard_root=shard_root,
                    shard_count=shard_count,
                    requested_maps=requested_maps,
                    episodes_per_map=episodes_per_map,
                    training_episodes=training_episodes,
                    shard_assignments=shard_assignments or [],
                    output_path=runtime_paths["result_path"],
                    run_id=run_id,
                    wait_for_done=True,
                )
                benchmark_completed = True
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(runtime_paths["result_path"], output_path)
                print(f"[HOLDOUT-BENCH] Sharded result copied to {output_path}", flush=True)
            except ValueError as exc:
                benchmark_completed = False
                print(f"[HOLDOUT-BENCH] Sharded aggregation failed: {exc}", file=sys.stderr)
        else:
            print(f"[HOLDOUT-BENCH] Waiting for benchmark to complete (container: {aisrv_container})...", flush=True)

            max_wait = 3600  # 1 hour max
            poll_interval = 10
            elapsed = 0
            benchmark_completed = False
            while elapsed < max_wait:
                try:
                    check = subprocess.run(
                        ["docker", "exec", aisrv_container, "test", "-f", "/workspace/code/.benchmark_done"],
                        capture_output=True, text=True, timeout=10,
                    )
                    if check.returncode == 0:
                        print(f"[HOLDOUT-BENCH] Benchmark completed after ~{elapsed}s", flush=True)
                        benchmark_completed = True
                        break
                except Exception:
                    pass

                # Check if container is still running
                try:
                    ps = subprocess.run(
                        ["docker", "inspect", "-f", "{{.State.Running}}", aisrv_container],
                        capture_output=True, text=True, timeout=10,
                    )
                    if ps.returncode != 0 or "true" not in ps.stdout.lower():
                        print(f"[HOLDOUT-BENCH] Container {aisrv_container} exited.", file=sys.stderr)
                        # Check logs for errors
                        try:
                            logs = subprocess.run(
                                ["docker", "logs", "--tail", "50", aisrv_container],
                                capture_output=True, text=True, timeout=30,
                            )
                            if logs.stdout:
                                print(f"[HOLDOUT-BENCH] Container logs (last 50 lines):", file=sys.stderr)
                                print(logs.stdout[-3000:], file=sys.stderr)
                        except Exception:
                            pass
                        break
                except Exception:
                    pass

                time.sleep(poll_interval)
                elapsed += poll_interval

            # Step 4: Copy result from container
            try:
                result_container_path = "/workspace/code/holdout_result.json"
                copy_cmd = ["docker", "cp", f"{aisrv_container}:{result_container_path}", str(output_path)]
                subprocess.run(copy_cmd, capture_output=True, text=True, timeout=30)
                print(f"[HOLDOUT-BENCH] Result copied to {output_path}", flush=True)
            except Exception as exc:
                print(f"[HOLDOUT-BENCH] Failed to copy result: {exc}", file=sys.stderr)

        benchmark_exit_code = 0 if output_path.exists() else 5
        if sharded and benchmark_exit_code != 0:
            print(f"[HOLDOUT-BENCH] Result file not found at {output_path}", file=sys.stderr)

        # Step 5: Tear down benchmark containers
        print("[HOLDOUT-BENCH] Cleaning up benchmark containers...", flush=True)
        try:
            down_cmd = [
                "docker", "compose",
                "-p", benchmark_project,
                "-f", compose_file,
                "-f", benchmark_overlay,
                "--profile", "distributed",
                "down",
            ]
            subprocess.run(down_cmd, capture_output=True, text=True, timeout=60, env=env)
        except Exception:
            pass

        # Step 6: Restart training containers if they were running
        if training_was_running:
            print("[HOLDOUT-BENCH] Restarting training containers...", flush=True)
            try:
                # Remove old containers first (process_stop.done issue)
                subprocess.run(
                    ["docker", "rm", "-f",
                     "kaiwu-train-learner-1", "kaiwu-train-aisrv-1", "kaiwu-train-aisrv-2"],
                    capture_output=True, text=True, timeout=30,
                )
                subprocess.run(
                    ["docker", "compose", "-p", train_project, "-f", compose_file,
                     "--profile", "distributed", "up", "-d"],
                    capture_output=True, text=True, timeout=120,
                )
                print("[HOLDOUT-BENCH] Training containers restarted.", flush=True)
            except Exception as exc:
                print(f"[HOLDOUT-BENCH] Warning: could not restart training: {exc}", flush=True)

        if benchmark_exit_code == 0:
            print(json.dumps({"status": "COMPLETED", "output": str(output_path)}, ensure_ascii=True))
            return 0
        else:
            print(f"[HOLDOUT-BENCH] Result file not found at {output_path}", file=sys.stderr)
            return benchmark_exit_code

    except Exception as exc:
        print(f"[HOLDOUT-BENCH] Unexpected error: {exc}", file=sys.stderr)
        # Try to restart training even on failure
        if training_was_running:
            try:
                subprocess.run(
                    ["docker", "compose", "-p", train_project, "-f", compose_file,
                     "--profile", "distributed", "up", "-d"],
                    capture_output=True, text=True, timeout=120,
                )
            except Exception:
                pass
        return 6


def main() -> int:
    args = parse_args()
    if args.aisrv_count is not None:
        args.shard_count = int(args.aisrv_count)
    if args.envs_per_aisrv is not None:
        args.workers_per_aisrv = int(args.envs_per_aisrv)
    repo_root = Path(__file__).resolve().parents[1]
    train_env_conf = repo_root / "code" / "agent_ppo" / "conf" / "train_env_conf.toml"
    effective_sharded = not bool(args.serial)
    if args.sharded:
        effective_sharded = True

    try:
        requested_maps = parse_maps(args.maps)
        shard_strategy = infer_shard_strategy(
            requested_maps=requested_maps,
            shard_count=int(args.shard_count),
        )
        training_maps = load_training_maps(train_env_conf)
        errors = validate_contract(requested_maps, args.episodes_per_map, training_maps, args.training_episodes)
        errors.extend(validate_sharding(effective_sharded, args.shard_count))
        errors.extend(validate_workers_per_aisrv(int(args.workers_per_aisrv)))
        if not effective_sharded and args.scheduler == "dynamic":
            errors.append("--scheduler dynamic requires sharded/AISRV assignment mode; use --scheduler static with --serial.")
        planned_task_count = len(
            build_task_list(
                maps=requested_maps,
                episodes_per_map=int(args.episodes_per_map),
                training_episodes=int(args.training_episodes),
            )
        )
        if effective_sharded and int(args.shard_count) > planned_task_count:
            errors.append("--shard-count cannot exceed the planned episode count.")
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
    shard_root = Path(args.shard_root) if args.shard_root else repo_root / "code" / "holdout_shards"
    if not shard_root.is_absolute():
        shard_root = (repo_root / shard_root).resolve()

    if args.aggregate_shards_only:
        shard_assignments = build_shard_assignments(
            episodes_planned=build_episode_plan(
                requested_maps=requested_maps,
                episodes_per_map=int(args.episodes_per_map),
                training_episodes=int(args.training_episodes),
                run_id="aggregate-shards-only",
                checkpoint_id="aggregate-shards-only",
                detail_log_dir=repo_root / "train" / "holdout_detail_logs" / "aggregate-shards-only",
                dry_run=True,
            ),
            shard_count=int(args.shard_count),
            requested_maps=requested_maps,
            episodes_per_map=int(args.episodes_per_map),
            training_episodes=int(args.training_episodes),
            run_id="aggregate-shards-only",
            strategy=shard_strategy,
        )
        try:
            aggregate_sharded_results(
                shard_root=shard_root,
                shard_count=int(args.shard_count),
                requested_maps=requested_maps,
                episodes_per_map=int(args.episodes_per_map),
                training_episodes=int(args.training_episodes),
                shard_assignments=shard_assignments,
                output_path=output_path,
                run_id=None,
                wait_for_done=False,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 7
        print(json.dumps({"status": "AGGREGATED", "output": str(output_path)}, ensure_ascii=True))
        return 0

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
        "benchmark_mode": "all_maps",
        "holdout_special_maps": list(HOLDOUT_SPECIAL_MAPS),
        "holdout_episodes_per_map": int(args.episodes_per_map),
        "training_episodes_per_map": int(args.training_episodes),
        "total_episodes": len(
            build_task_list(
                maps=requested_maps,
                episodes_per_map=int(args.episodes_per_map),
                training_episodes=int(args.training_episodes),
            )
        ),
        "fixed_config": deepcopy(FIXED_CONFIG),
        "training_maps_source": {
            "path": str(train_env_conf),
            "maps": training_maps,
            "expected_training_maps": [1, 2, 3, 5, 6, 8, 9, 10],
            "matches_expected_training_maps": sorted(set(training_maps)) == [1, 2, 3, 5, 6, 8, 9, 10],
        },
    }
    checkpoint = {
        "path": str(checkpoint_path),
        "checkpoint_id": checkpoint_id,
        "requested_relative_path": args.checkpoint,
        "snapshot": snapshot_file(checkpoint_path),
        "tracked_artifacts": before_snapshots,
    }
    episodes_planned = build_episode_plan(
        requested_maps=requested_maps,
        episodes_per_map=int(args.episodes_per_map),
        training_episodes=int(args.training_episodes),
        run_id=run_id,
        checkpoint_id=checkpoint_id,
        detail_log_dir=detail_log_dir,
        dry_run=bool(args.dry_run),
    )
    shard_assignments = (
        build_shard_assignments(
            episodes_planned=episodes_planned,
            shard_count=int(args.shard_count),
            requested_maps=requested_maps,
            episodes_per_map=int(args.episodes_per_map),
            training_episodes=int(args.training_episodes),
            run_id=run_id,
            strategy=shard_strategy,
        )
        if effective_sharded
        else []
    )

    risks: list[dict[str, object]] = [
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
            "code": "REAL_EXECUTION_VIA_DOCKER",
            "severity": "info" if not args.dry_run else "warning",
            "message": (
                "Real execution will launch benchmark via docker compose overlay."
                if not args.dry_run
                else "Dry-run: no real episodes executed."
            ),
        },
    ]
    decision_inputs: dict[str, object] = {
        "checkpoint_id": checkpoint_id,
        "maps": requested_maps,
        "episodes_per_map": int(args.episodes_per_map),
        "benchmark_mode": "all_maps",
        "holdout_special_maps": list(HOLDOUT_SPECIAL_MAPS),
        "holdout_episodes_per_map": int(args.episodes_per_map),
        "training_episodes_per_map": int(args.training_episodes),
        "total_episodes": len(episodes_planned),
        "fixed_config": deepcopy(FIXED_CONFIG),
        "planned_episode_count": len(episodes_planned),
        "scheduler": args.scheduler,
        "workers_per_aisrv": int(args.workers_per_aisrv),
        "envs_per_aisrv": int(args.workers_per_aisrv),
        "planned_aisrv_num": int(args.shard_count) if effective_sharded else 1,
        "planned_gamecore_num": (int(args.shard_count) if effective_sharded else 1) * int(args.workers_per_aisrv),
    }

    result: dict[str, object] = {
        "run_id": run_id,
        "status": "DRY_RUN" if args.dry_run else "LAUNCHING",
        "benchmark_mode": "all_maps",
        "holdout_special_maps": list(HOLDOUT_SPECIAL_MAPS),
        "holdout_episodes_per_map": int(args.episodes_per_map),
        "training_episodes_per_map": int(args.training_episodes),
        "total_episodes": len(episodes_planned),
        "contract": contract,
        "checkpoint": checkpoint,
        "detail_log_dir": str(detail_log_dir),
        "detail_log_schema_path": str(schema_path),
        "detail_log_schema": schema_payload,
        "model_mutation_guard": mutation_guard,
        "episodes": [] if args.dry_run else [],
        "episodes_planned": episodes_planned,
        "risks": risks,
        "decision_inputs": decision_inputs,
    }
    if effective_sharded:
        result["sharding"] = build_sharding_metadata(
            shard_assignments=shard_assignments,
            shard_count=int(args.shard_count),
            run_id=run_id,
            strategy=shard_strategy,
        )
        decision_inputs.update(
            {
                "sharded": True,
                "scheduler": args.scheduler,
                "shard_count": int(args.shard_count),
                "aisrv_count": int(args.shard_count),
                "shard_strategy": shard_strategy,
                "workers_per_aisrv": int(args.workers_per_aisrv),
                "envs_per_aisrv": int(args.workers_per_aisrv),
                "planned_aisrv_num": int(args.shard_count),
                "planned_gamecore_num": int(args.shard_count) * int(args.workers_per_aisrv),
            }
        )
        risks.append(
            {
                "code": "SHARDED_BENCHMARK_ASSIGNMENTS",
                "severity": "warning" if args.dry_run else "info",
                "message": (
                    "Dry-run: deterministic shard assignments recorded without launching docker."
                    if args.dry_run
                    else "Sharded benchmark launch will prepare indexed assignments before startup and hostname assignments after containers are resolvable."
                    if args.scheduler == "static"
                    else "Dynamic benchmark launch will prepare AISRV identity assignments and use a shared task queue."
                ),
            }
        )

    if not args.dry_run:
        # Real execution: launch docker compose benchmark, wait for result, copy output.
        benchmark_status = _launch_benchmark_via_docker(
            repo_root=repo_root,
            requested_maps=requested_maps,
            episodes_per_map=int(args.episodes_per_map),
            training_episodes=int(args.training_episodes),
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            run_id=run_id,
            sharded=effective_sharded,
            shard_count=int(args.shard_count),
            workers_per_aisrv=int(args.workers_per_aisrv),
            shard_assignments=shard_assignments,
            scheduler=args.scheduler,
        )
        return benchmark_status

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
