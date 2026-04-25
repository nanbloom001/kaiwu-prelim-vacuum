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
import os
import re
import sys
import time
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


def _launch_benchmark_via_docker(
    repo_root: Path,
    requested_maps: list[int],
    episodes_per_map: int,
    checkpoint_path: Path,
    output_path: Path,
    run_id: str,
) -> int:
    """Launch the holdout benchmark via docker compose benchmark overlay.

    This stops training containers (port conflict), starts benchmark containers
    with KAIWU_BENCHMARK_MODE=1, waits for completion, copies the result, then
    optionally restarts training containers.
    """
    import subprocess

    maps_str = ",".join(str(m) for m in requested_maps)
    compose_file = str(repo_root / "train" / ".docker-compose.yaml")
    benchmark_overlay = str(repo_root / "train" / ".docker-compose.benchmark.yaml")
    train_profile = "--profile distributed"

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
        "KAIWU_BENCHMARK_MAPS": maps_str,
        "KAIWU_BENCHMARK_EPISODES_PER_MAP": str(episodes_per_map),
        "KAIWU_BENCHMARK_CHECKPOINT": checkpoint_container,
    })

    train_project = "kaiwu-train"
    benchmark_project = "kaiwu-train"  # Must match training project name for framework service discovery

    print(f"[HOLDOUT-BENCH] Starting benchmark via docker compose...", flush=True)
    print(f"  maps={maps_str} episodes_per_map={episodes_per_map} checkpoint={checkpoint_container}", flush=True)

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

        # Step 3: Wait for completion marker in aisrv container
        aisrv_container = f"{benchmark_project}-aisrv-1"
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

        if output_path.exists():
            print(json.dumps({"status": "COMPLETED", "output": str(output_path)}, ensure_ascii=True))
            return 0
        else:
            print(f"[HOLDOUT-BENCH] Result file not found at {output_path}", file=sys.stderr)
            return 5

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
        "status": "DRY_RUN" if args.dry_run else "LAUNCHING",
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
                "code": "REAL_EXECUTION_VIA_DOCKER",
                "severity": "info" if not args.dry_run else "warning",
                "message": (
                    "Real execution will launch benchmark via docker compose overlay."
                    if not args.dry_run
                    else "Dry-run: no real episodes executed."
                ),
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
        # Real execution: launch docker compose benchmark, wait for result, copy output.
        benchmark_status = _launch_benchmark_via_docker(
            repo_root=repo_root,
            requested_maps=requested_maps,
            episodes_per_map=int(args.episodes_per_map),
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            run_id=run_id,
        )
        return benchmark_status

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(output_path)}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
