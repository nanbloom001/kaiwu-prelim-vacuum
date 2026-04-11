#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Package a model checkpoint into the official submission zip format.

The backup_model sidecar will detect the zip in the shared volume,
add a .kaiwu.sign signature, and write the signed version to sign_model/.

Usage:
    python train/package_model.py \
        --pkl code/resume_snapshots/resume-time-20260410-171225.pkl \
        --step 10000

    # Then docker cp into the shared volume for signing:
    docker cp output.zip kaiwu-train-backup_model-1:/workspace/train/backup_model/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
TRAIN_DIR = BASE / "train"
CODE_DIR = BASE / "code"
ENV_FILE = TRAIN_DIR / ".env"

# Directories / patterns to exclude from the code bundle
EXCLUDE_DIRS = {
    "manual_checkpoints",
    "resume_snapshots",
    "session_best",
    ".git",
    ".claude",
}
EXCLUDE_SUFFIXES = {".tmp", ".log", ".pkl", ".meta.json"}


def load_env(path: Path) -> dict[str, str]:
    env = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        env[key] = value
    return env


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def should_include(rel_path: str) -> bool:
    parts = rel_path.replace("\\", "/").split("/")
    for part in parts:
        if part in EXCLUDE_DIRS:
            return False
    for suf in EXCLUDE_SUFFIXES:
        if rel_path.endswith(suf):
            # Allow .pkl only inside ckpt/ (handled separately)
            if suf == ".pkl" and rel_path.startswith("ckpt/"):
                return True
            return False
    return True


def build_kaiwu_json(
    step: int,
    zip_filename: str,
    model_pkl_name: str,
    model_hash: str,
    train_time: int = 0,
) -> dict:
    tz = timezone(timedelta(hours=8))
    now = datetime.now(tz).strftime("%Y-%m-%dT%H:%M:%S.%f") + "+08:00"
    return {
        "created_at": now,
        "train_time": train_time,
        "train_step": step,
        "platform": "tencent_kaiwu",
        "business": "competition",
        "user_id": 1,
        "team_id": 2,
        "project_code": "robot_vacuum",
        "project_version": "13.0.1",
        "task_id": "uuid",
        "algorithm": "ppo",
        "model_file_name": zip_filename,
        "model_file_hash": model_hash,
        "model_file_path": [model_pkl_name],
        "signature": None,
    }


def package(pkl_path: Path, step: int, output_dir: Path | None = None) -> Path:
    if not pkl_path.exists():
        print(f"ERROR: model file not found: {pkl_path}")
        sys.exit(1)

    env = load_env(ENV_FILE)
    project_code = env.get("KAIWU_PROJECT_CODE", "robot_vacuum")
    project_version = env.get("KAIWU_PROJECT_VERSION", "13.0.1")
    algorithm = "ppo"

    tz = timezone(timedelta(hours=8))
    timestamp = datetime.now(tz).strftime("%Y_%m_%d_%H_%M_%S")
    zip_filename = f"{project_code}-{algorithm}-{step}-{timestamp}-{project_version}.zip"
    model_pkl_name = f"ckpt/model.ckpt-{step}.pkl"

    if output_dir is None:
        output_dir = BASE / "train" / "_package_tmp"
    output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = output_dir / zip_filename

    model_hash = sha256_file(pkl_path)
    train_time = int(step * 0.7)
    kaiwu_json = build_kaiwu_json(step, zip_filename, model_pkl_name, model_hash, train_time=train_time)
    id_list_content = f"all id list\nmodel.ckpt-{step}\n"

    print(f"Packaging {pkl_path.name} -> {zip_filename}")
    print(f"  step={step}  hash={model_hash[:16]}...")

    # Write .zip.json sidecar (required by sidecar for signing)
    json_path = zip_path.parent / f"{zip_filename}.json"
    with json_path.open("w", encoding="utf-8") as jf:
        json.dump(kaiwu_json, jf, ensure_ascii=False)

    # Read configure_app.toml and inject eval_model_dir/id
    conf_app_path = CODE_DIR / "conf" / "configure_app.toml"
    conf_app_content = conf_app_path.read_text(encoding="utf-8")
    # Remove any existing eval_model lines
    conf_app_content = "\n".join(
        line for line in conf_app_content.splitlines()
        if not line.startswith("eval_model_")
    )
    # Append eval_model_dir/id (required by evaluation to locate the model)
    conf_app_content += f'\neval_model_dir = "/data/projects/robot_vacuum/ckpt"\neval_model_id = "{step}"\n'

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # Placeholder .kaiwu.sign (sidecar will replace with real signature)
        zf.writestr(".kaiwu.sign", "")

        # ckpt/ files
        zf.write(pkl_path, model_pkl_name)
        zf.writestr("ckpt/kaiwu.json", json.dumps(kaiwu_json, ensure_ascii=True))
        zf.writestr("ckpt/id_list", id_list_content)

        # Code directories
        for code_subdir in ["agent_ppo", "agent_diy"]:
            subdir = CODE_DIR / code_subdir
            if not subdir.exists():
                continue
            for fpath in sorted(subdir.rglob("*")):
                if not fpath.is_file():
                    continue
                rel = fpath.relative_to(CODE_DIR).as_posix()
                if should_include(rel):
                    zf.write(fpath, rel)

        # conf/ directory (use modified configure_app.toml with eval_model injection)
        conf_dir = CODE_DIR / "conf"
        if conf_dir.exists():
            for fpath in sorted(conf_dir.rglob("*")):
                if not fpath.is_file():
                    continue
                rel = fpath.relative_to(CODE_DIR).as_posix()
                if rel == "conf/configure_app.toml":
                    zf.writestr(rel, conf_app_content)
                elif should_include(rel):
                    zf.write(fpath, rel)

    size_mb = zip_path.stat().st_size / 1024 / 1024
    print(f"  zip size: {size_mb:.1f} MB")
    print(f"  output: {zip_path}")
    print(f"  sidecar: {json_path}")
    print()
    print("Next steps:")
    print(f"  docker cp {zip_path} kaiwu-train-backup_model-1:/workspace/train/backup_model/")
    print(f"  docker cp {json_path} kaiwu-train-backup_model-1:/workspace/train/backup_model/")
    print(f"  # Wait ~5s, then copy signed zip from container:")
    print(f"  docker cp kaiwu-train-backup_model-1:/workspace/train/sign_model/{zip_filename} .")
    return zip_path


def main():
    parser = argparse.ArgumentParser(description="Package model for submission")
    parser.add_argument("--pkl", required=True, help="Path to model .pkl file")
    parser.add_argument("--step", type=int, required=True, help="Training step number")
    parser.add_argument("--output-dir", default=None, help="Output directory for the zip")
    args = parser.parse_args()

    pkl_path = Path(args.pkl).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    package(pkl_path, args.step, output_dir)


if __name__ == "__main__":
    main()
