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


PRELOAD_RELATIVE_DIR = "agent_ppo/ckpt"
LATEST_PRELOAD_FILE = "latest_preload.json"


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
    latest_resume = code_dir / "model.ckpt-resume.pkl"
    if latest_resume.exists():
        candidates.append(latest_resume)
    snapshot_dir = code_dir / "resume_snapshots"
    if snapshot_dir.exists():
        candidates.extend(sorted(snapshot_dir.glob("resume-time-*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True))
        candidates.extend(sorted(snapshot_dir.glob("resume-episode-*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True))
    return candidates


def latest_preload_metadata_path(code_dir: Path) -> Path:
    return code_dir / PRELOAD_RELATIVE_DIR / LATEST_PRELOAD_FILE


def resolve_latest_preload(code_dir: Path) -> dict[str, Any] | None:
    payload = _read_json(latest_preload_metadata_path(code_dir))
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


def resolve_training_preload(code_dir: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    env = env or os.environ
    training_start_mode = str(env.get("KAIWU_TRAINING_START_MODE", "preload") or "preload").strip().lower()
    if training_start_mode in {"scratch", "random", "from_scratch", "fresh"}:
        return {"enabled": False, "training_start_mode": training_start_mode}
    mode = str(env.get("KAIWU_PRELOAD_MODEL", "")).strip().lower()
    manual_dir = str(env.get("KAIWU_PRELOAD_MODEL_DIR", "")).strip()
    manual_id = str(env.get("KAIWU_PRELOAD_MODEL_ID", "")).strip()

    if mode in {"0", "false", "off", "no"}:
        return {"enabled": False}

    if manual_dir and manual_id:
        checkpoint_dir = _normalize_path(code_dir, manual_dir)
        checkpoint_path = checkpoint_dir / f"model.ckpt-{manual_id}.pkl" if checkpoint_dir else None
        if checkpoint_path and checkpoint_path.exists():
            return {
                "enabled": True,
                "checkpoint_id": str(manual_id),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_dir": str(checkpoint_dir),
                "checkpoint_dir_relative": manual_dir,
            }

    latest = resolve_latest_preload(code_dir)
    if latest and mode in {"", "1", "true", "on", "auto", "latest"}:
        return latest

    return {"enabled": False}


def seed_preload_from_resume(code_dir: Path, checkpoint_ref: str | None, checkpoint_id: str = "0") -> dict[str, Any] | None:
    checkpoint = _normalize_path(code_dir, (checkpoint_ref or "").strip())
    if checkpoint is None or not checkpoint.exists():
        fallback = _fallback_resume_candidates(code_dir)
        checkpoint = fallback[0] if fallback else None
    if checkpoint is None or not checkpoint.exists():
        return None
    ckpt_dir = code_dir / PRELOAD_RELATIVE_DIR
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
    return payload


def resolve_benchmark_checkpoint(code_dir: Path, explicit_checkpoint: str | None, config_resume_checkpoint: str | None) -> str:
    explicit = (explicit_checkpoint or "").strip()
    if explicit:
        return explicit
    latest = resolve_latest_preload(code_dir)
    if latest:
        return latest["checkpoint_path"]
    configured = (config_resume_checkpoint or "").strip()
    if configured:
        return configured
    fallback = code_dir / "model.ckpt-resume.pkl"
    return str(fallback)
