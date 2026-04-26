#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Launch training with a phase-specific env overlay without mutating train/.env.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
import uuid
from pathlib import Path


DEFAULT_SERVICES = ["learner", "aisrv", "gamecore", "backup_model", "pushgateway"]


def load_env_file(path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    if not path.exists():
        raise FileNotFoundError(path)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        payload[key.strip()] = value.strip()
    return payload


def merge_env_dicts(*payloads: dict[str, str] | None) -> dict[str, str]:
    merged: dict[str, str] = {}
    for payload in payloads:
        if not payload:
            continue
        for key, value in payload.items():
            if value is None:
                continue
            merged[str(key)] = str(value)
    return merged


def write_env_file(path: Path, payload: dict[str, str]) -> None:
    lines = [f"{key}={value}" for key, value in sorted(payload.items())]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_phase_env_file(
    base_env_path: Path,
    phase_env_path: Path,
    output_path: Path,
    extra_overrides: dict[str, str] | None = None,
) -> dict[str, str]:
    merged = merge_env_dicts(
        load_env_file(base_env_path),
        load_env_file(phase_env_path),
        extra_overrides or {},
    )
    write_env_file(output_path, merged)
    return merged


def build_compose_command(
    env_file: Path,
    compose_file: Path,
    profile: str,
    project: str,
    services: list[str],
) -> list[str]:
    return [
        "docker",
        "compose",
        "--env-file",
        str(env_file),
        "-p",
        project,
        "-f",
        str(compose_file),
        "--profile",
        profile,
        "up",
        "-d",
        "--force-recreate",
        *services,
    ]


def build_training_mode_overrides(start_mode: str, resume_bundle_dir: str = "") -> dict[str, str]:
    mode = str(start_mode or "scratch").strip().lower()
    if mode == "scratch":
        return {
            "KAIWU_TRAINING_START_MODE": "scratch",
            "KAIWU_PRELOAD_MODEL": "0",
            "KAIWU_PRELOAD_MODEL_DIR": "",
            "KAIWU_PRELOAD_MODEL_ID": "",
            "KAIWU_RESUME_BUNDLE_DIR": "",
            "KAIWU_RESUME_RUN_ID": "",
            "KAIWU_CURRICULUM_INITIAL_STAGE": "warmup",
        }
    if mode == "resume":
        bundle_dir = str(resume_bundle_dir or "").strip()
        if not bundle_dir:
            raise ValueError("--resume-bundle-dir is required when --start-mode=resume")
        return {
            "KAIWU_TRAINING_START_MODE": "resume",
            "KAIWU_PRELOAD_MODEL": "0",
            "KAIWU_PRELOAD_MODEL_DIR": "",
            "KAIWU_PRELOAD_MODEL_ID": "",
            "KAIWU_RESUME_BUNDLE_DIR": bundle_dir,
            "KAIWU_RESUME_RUN_ID": "",
            "KAIWU_CURRICULUM_INITIAL_STAGE": "warmup",
        }
    raise ValueError(f"unsupported start mode: {start_mode}")


def build_launch_instance_id(phase: str) -> str:
    phase_slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in str(phase or "phase"))
    phase_slug = phase_slug.strip("-") or "phase"
    return f"{phase_slug}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:12]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start a training phase using an env overlay")
    parser.add_argument("phase", help="Phase name, e.g. s1_survival")
    parser.add_argument("--seed-label", default="", help="Optional label such as a / b for repeated runs")
    parser.add_argument("--train-dir", default=str(Path(__file__).resolve().parent), help="Train directory root")
    parser.add_argument("--profile", default="distributed", help="Docker compose profile")
    parser.add_argument("--project", default="kaiwu-train", help="Docker compose project name")
    parser.add_argument(
        "--services",
        nargs="*",
        default=DEFAULT_SERVICES,
        help="Compose services to recreate",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print merged env and compose command only")
    parser.add_argument("--start-mode", choices=["scratch", "resume"], default="scratch", help="Training start mode")
    parser.add_argument("--resume-bundle-dir", default="", help="Explicit resume bundle directory when start mode is resume")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    train_dir = Path(args.train_dir).resolve()
    base_env_path = train_dir / ".env"
    phase_env_path = train_dir / "phases" / f"{args.phase}.env"
    compose_file = train_dir / ".docker-compose.yaml"

    extra_overrides = build_training_mode_overrides(args.start_mode, args.resume_bundle_dir)
    extra_overrides["KAIWU_TRAIN_PHASE"] = args.phase
    extra_overrides["KAIWU_PHASE_RUN_LAUNCH_INSTANCE_ID"] = build_launch_instance_id(args.phase)
    if args.seed_label:
        extra_overrides["KAIWU_PHASE_RUN_LABEL"] = f"{args.phase}_{args.seed_label}"

    with tempfile.TemporaryDirectory(prefix=f"{args.phase}_") as tmp_dir:
        env_file = Path(tmp_dir) / f"{args.phase}.env"
        merged = build_phase_env_file(
            base_env_path=base_env_path,
            phase_env_path=phase_env_path,
            output_path=env_file,
            extra_overrides=extra_overrides,
        )
        command = build_compose_command(
            env_file=env_file,
            compose_file=compose_file,
            profile=args.profile,
            project=args.project,
            services=args.services,
        )
        if args.dry_run:
            print(f"phase={args.phase}")
            print(f"env_file={env_file}")
            for key in (
                "KAIWU_TRAIN_PHASE",
                "KAIWU_PHASE_RUN_LAUNCH_INSTANCE_ID",
                "KAIWU_TRAINING_START_MODE",
                "KAIWU_PRELOAD_MODEL",
                "KAIWU_RESUME_BUNDLE_DIR",
                "KAIWU_CURRICULUM_INITIAL_STAGE",
                "KAIWU_PHASE_RUN_LABEL",
                "KAIWU_ENV_FIXED_DIFFICULTY",
                "KAIWU_TRAIN_MAPS",
                "KAIWU_TRAIN_MAP_RANDOM",
                "KAIWU_TRAIN_ROBOT_COUNT",
                "KAIWU_TRAIN_CHARGER_COUNT",
                "KAIWU_TRAIN_MAX_STEP",
                "KAIWU_TRAIN_BATTERY_MAX",
                "KAIWU_BENCHMARK_POLICY_MODE",
                "KAIWU_BENCHMARK_MAPS",
                "KAIWU_BENCHMARK_ROUNDS_JSON",
            ):
                if key in merged:
                    print(f"{key}={merged[key]}")
            print("command=" + " ".join(command))
            return 0
        subprocess.run(command, cwd=str(train_dir), check=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
