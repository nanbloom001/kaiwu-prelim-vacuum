#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Helpers for preload-compatible checkpoint discovery and selection.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from agent_ppo.workflow.state_layout import (
    CURRICULUM_STATE_SNAPSHOT_FILE,
    LATEST_PRELOAD_FILE,
    RESUME_CHECKPOINT_FILE,
    RESUME_META_FILE,
    RESUME_STATE_FILE,
    allow_legacy_resume_import,
    ensure_runtime_state_dirs,
    is_resume_mode,
    is_scratch_mode,
    latest_preload_file_candidates,
    latest_resume_checkpoint_file_candidates,
    latest_resume_state_file_candidates,
    legacy_resume_curriculum_snapshot_path,
    legacy_resume_snapshots_dir,
    resolve_resume_bundle_dir,
    runtime_state_layout,
)

PRELOAD_RELATIVE_DIR = "runtime_state/preload_cache"
RESUME_LATEST_META_FILE = RESUME_META_FILE
RESUME_LATEST_STATE_FILE = RESUME_STATE_FILE
RESUME_CURRICULUM_SNAPSHOT_FILE = CURRICULUM_STATE_SNAPSHOT_FILE


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _normalize_path(code_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    raw = Path(str(value))
    if raw.is_absolute():
        return raw
    return (code_dir / raw).resolve()


def _fallback_resume_candidates(code_dir: Path) -> list[Path]:
    candidates = []
    for latest_resume in latest_resume_checkpoint_file_candidates(code_dir):
        if latest_resume.exists():
            candidates.append(latest_resume)
    legacy_latest = code_dir / "model.ckpt-resume.pkl"
    if legacy_latest.exists():
        candidates.append(legacy_latest)
    snapshot_dirs = [
        legacy_resume_snapshots_dir(code_dir),
        *sorted(
            (runtime_state_layout(code_dir).state_root / "runs").glob("*/resume/snapshots"),
            key=lambda item: item.stat().st_mtime if item.exists() else 0.0,
            reverse=True,
        ),
    ]
    for snapshot_dir in snapshot_dirs:
        if not snapshot_dir.exists():
            continue
        candidates.extend(sorted(snapshot_dir.glob("resume-time-*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True))
        candidates.extend(sorted(snapshot_dir.glob("resume-episode-*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True))
        candidates.extend(sorted(snapshot_dir.glob("resume-step-*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True))
    return candidates


def latest_preload_metadata_path(code_dir: Path) -> Path:
    return runtime_state_layout(code_dir).preload_latest_metadata_path


def latest_resume_metadata_path(code_dir: Path) -> Path:
    bundle_dir = resolve_resume_bundle_dir(code_dir, os.environ)
    if bundle_dir is not None:
        return bundle_dir / RESUME_LATEST_META_FILE
    return runtime_state_layout(code_dir).current.prepared_resume_dir / RESUME_LATEST_META_FILE


def latest_resume_state_path(code_dir: Path) -> Path:
    bundle_dir = resolve_resume_bundle_dir(code_dir, os.environ)
    if bundle_dir is not None:
        return bundle_dir / RESUME_LATEST_STATE_FILE
    return runtime_state_layout(code_dir).current.prepared_resume_dir / RESUME_LATEST_STATE_FILE


def latest_resume_curriculum_snapshot_path(code_dir: Path) -> Path:
    bundle_dir = resolve_resume_bundle_dir(code_dir, os.environ)
    if bundle_dir is not None:
        return bundle_dir / RESUME_CURRICULUM_SNAPSHOT_FILE
    return runtime_state_layout(code_dir).current.prepared_resume_dir / RESUME_CURRICULUM_SNAPSHOT_FILE


def resolve_latest_preload(code_dir: Path) -> dict[str, Any] | None:
    payload = None
    for candidate in latest_preload_file_candidates(code_dir):
        payload = _read_json(candidate)
        if payload:
            break
    if not payload:
        return None
    checkpoint_path = _normalize_path(code_dir, payload.get("checkpoint_path"))
    checkpoint_dir = _normalize_path(code_dir, payload.get("checkpoint_dir"))
    if checkpoint_path is None or checkpoint_dir is None or not checkpoint_path.exists():
        return None
    checkpoint_id = payload.get("checkpoint_id")
    if checkpoint_id in (None, ""):
        checkpoint_id = payload.get("global_step")
    return {
        "enabled": True,
        "checkpoint_id": str(checkpoint_id),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_dir_relative": str(payload.get("checkpoint_dir_relative") or PRELOAD_RELATIVE_DIR),
        "global_step": int(payload.get("global_step") or 0),
        "episode_cnt": int(payload.get("episode_cnt") or 0),
        "clean_score": float(payload.get("clean_score") or 0.0),
        "saved_at": payload.get("saved_at"),
    }


def resolve_latest_resume_state(code_dir: Path, env: dict[str, str] | None = None) -> dict[str, Any] | None:
    env = env or os.environ
    for metadata_path in latest_resume_state_file_candidates(code_dir, env):
        payload = _read_json(metadata_path)
        if not payload:
            continue
        bundle_dir = metadata_path.parent
        checkpoint_path = _normalize_path(code_dir, payload.get("checkpoint_path")) or (bundle_dir / RESUME_CHECKPOINT_FILE)
        if not checkpoint_path.exists():
            legacy_checkpoint = code_dir / "model.ckpt-resume.pkl"
            if allow_legacy_resume_import(env) and legacy_checkpoint.exists():
                checkpoint_path = legacy_checkpoint
        curriculum_snapshot_path = _normalize_path(
            code_dir, payload.get("curriculum_state_snapshot_path")
        ) or (bundle_dir / CURRICULUM_STATE_SNAPSHOT_FILE)
        if not curriculum_snapshot_path.exists():
            legacy_curriculum = legacy_resume_curriculum_snapshot_path(code_dir)
            if allow_legacy_resume_import(env) and legacy_curriculum.exists():
                curriculum_snapshot_path = legacy_curriculum
        return {
            **payload,
            "state_metadata_path": str(metadata_path),
            "checkpoint_path": str(checkpoint_path),
            "curriculum_state_snapshot_path": str(curriculum_snapshot_path),
        }
    return None


def resolve_training_preload(code_dir: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    env = env or os.environ
    training_start_mode = str(env.get("KAIWU_TRAINING_START_MODE", "preload") or "preload").strip().lower()
    if is_scratch_mode(env):
        return {"enabled": False, "training_start_mode": training_start_mode}
    if is_resume_mode(env):
        resume_state = resolve_latest_resume_state(code_dir, env)
        if not resume_state:
            raise FileNotFoundError(
                "resume start mode requires a valid explicit resume bundle via "
                "KAIWU_RESUME_BUNDLE_DIR or KAIWU_RESUME_RUN_ID"
            )
        checkpoint_path = Path(str(resume_state.get("checkpoint_path") or "")).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
        checkpoint_dir = checkpoint_path.parent
        checkpoint_id = str(resume_state.get("checkpoint_id") or resume_state.get("global_step") or "0")
        try:
            checkpoint_dir_relative = str(checkpoint_dir.relative_to(code_dir))
        except ValueError:
            checkpoint_dir_relative = str(checkpoint_dir)
        return {
            "enabled": True,
            "checkpoint_id": checkpoint_id,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint_dir_relative": checkpoint_dir_relative,
            "training_start_mode": training_start_mode,
            "resume_state_metadata_path": resume_state["state_metadata_path"],
            "curriculum_state_snapshot_path": resume_state["curriculum_state_snapshot_path"],
            "global_step": int(resume_state.get("global_step") or 0),
        }
    mode = str(env.get("KAIWU_PRELOAD_MODEL", "")).strip().lower()
    manual_dir = str(env.get("KAIWU_PRELOAD_MODEL_DIR", "")).strip()
    manual_id = str(env.get("KAIWU_PRELOAD_MODEL_ID", "")).strip()

    if mode in {"0", "false", "off", "no"}:
        return {"enabled": False}

    if manual_dir and manual_id:
        checkpoint_dir = _normalize_path(code_dir, manual_dir)
        checkpoint_path = checkpoint_dir / f"model.ckpt-{manual_id}.pkl" if checkpoint_dir else None
        if checkpoint_path and checkpoint_path.exists():
            resolved = {
                "enabled": True,
                "checkpoint_id": str(manual_id),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_dir_relative": manual_dir,
                "training_start_mode": training_start_mode,
            }
            return resolved

    latest = resolve_latest_preload(code_dir)
    if latest and mode in {"", "1", "true", "on", "auto", "latest"}:
        return {
            **latest,
            "training_start_mode": training_start_mode,
        }

    return {"enabled": False, "training_start_mode": training_start_mode}


def _snapshot_state_sidecar_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".state.json")


def _snapshot_curriculum_sidecar_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".curriculum.json")


def seed_preload_from_resume(code_dir: Path, checkpoint_ref: str | None, checkpoint_id: str = "0") -> dict[str, Any] | None:
    checkpoint = _normalize_path(code_dir, (checkpoint_ref or "").strip())
    if checkpoint is None or not checkpoint.exists():
        fallback = _fallback_resume_candidates(code_dir)
        checkpoint = fallback[0] if fallback else None
    if checkpoint is None or not checkpoint.exists():
        return None
    layout = ensure_runtime_state_dirs(code_dir)
    ckpt_dir = layout.preload_cache_dir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    target = ckpt_dir / f"model.ckpt-{checkpoint_id}.pkl"
    if not target.exists() or checkpoint.resolve() != target.resolve():
        shutil.copy2(checkpoint, target)
    payload = {
        "enabled": True,
        "checkpoint_id": str(checkpoint_id),
        "checkpoint_path": str(target),
        "checkpoint_dir": str(ckpt_dir),
        "checkpoint_dir_relative": PRELOAD_RELATIVE_DIR,
        "global_step": int(checkpoint_id) if str(checkpoint_id).isdigit() else 0,
        "episode_cnt": 0,
        "clean_score": 0.0,
        "saved_at": None,
        "seeded_from_resume": str(checkpoint),
    }
    latest_preload_metadata_path(code_dir).write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    prepared_dir = layout.current.prepared_resume_dir
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_checkpoint = prepared_dir / RESUME_CHECKPOINT_FILE
    shutil.copy2(checkpoint, prepared_checkpoint)
    source_state_path = _snapshot_state_sidecar_path(checkpoint)
    target_state_path = prepared_dir / RESUME_LATEST_STATE_FILE
    if source_state_path.exists():
        shutil.copy2(source_state_path, target_state_path)
        try:
            state_payload = json.loads(target_state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            state_payload = None
        if isinstance(state_payload, dict):
            source_curriculum_path = _normalize_path(code_dir, state_payload.get("curriculum_state_snapshot_path"))
            target_curriculum_path = prepared_dir / RESUME_CURRICULUM_SNAPSHOT_FILE
            state_payload["checkpoint_path"] = str(prepared_checkpoint)
            if source_curriculum_path and source_curriculum_path.exists():
                shutil.copy2(source_curriculum_path, target_curriculum_path)
                state_payload["curriculum_state_snapshot_path"] = str(target_curriculum_path)
            target_state_path.write_text(
                json.dumps(state_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    else:
        source_curriculum_sidecar = _snapshot_curriculum_sidecar_path(checkpoint)
        if source_curriculum_sidecar.exists():
            shutil.copy2(source_curriculum_sidecar, prepared_dir / RESUME_CURRICULUM_SNAPSHOT_FILE)
    prepared_meta = {
        "checkpoint_path": str(prepared_checkpoint),
        "seeded_from_resume": str(checkpoint),
        "prepared_at": None,
        "checkpoint_id": str(checkpoint_id),
    }
    (prepared_dir / RESUME_LATEST_META_FILE).write_text(
        json.dumps(prepared_meta, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def resolve_benchmark_checkpoint(code_dir: Path, explicit_checkpoint: str | None, config_resume_checkpoint: str | None) -> str:
    explicit = (explicit_checkpoint or "").strip()
    if explicit:
        return explicit
    latest = resolve_latest_preload(code_dir)
    if latest:
        return latest["checkpoint_path"]
    configured = (config_resume_checkpoint or "").strip()
    if configured and Path(configured).exists():
        return configured
    for fallback in latest_resume_checkpoint_file_candidates(code_dir):
        if fallback.exists():
            return str(fallback)
    return str(code_dir / "model.ckpt-resume.pkl")
