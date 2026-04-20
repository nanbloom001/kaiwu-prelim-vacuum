#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Unified runtime-state layout for scratch/resume training.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


STATE_ROOT_ENV = "KAIWU_STATE_ROOT"
RESUME_BUNDLE_DIR_ENV = "KAIWU_RESUME_BUNDLE_DIR"
RESUME_RUN_ID_ENV = "KAIWU_RESUME_RUN_ID"
ALLOW_LEGACY_RESUME_IMPORT_ENV = "KAIWU_ALLOW_LEGACY_RESUME_IMPORT"

DEFAULT_STATE_ROOT_RELATIVE = "runtime_state"
CURRENT_DIR_NAME = "current"
RUNS_DIR_NAME = "runs"
MANUAL_SAVES_DIR_NAME = "manual_saves"
PRELOAD_CACHE_DIR_NAME = "preload_cache"
PREPARED_RESUME_DIR_NAME = "prepared_resume"
LOCKS_DIR_NAME = "locks"

RUN_SESSION_MANIFEST_FILE = "run_session.json"
CURRICULUM_STATE_FILE = "curriculum_state.json"
CURRICULUM_STATE_SNAPSHOT_FILE = "curriculum_state.snapshot.json"
CURRICULUM_SIGNALS_DIR_NAME = "curriculum_signals"
RUN_SESSION_LOCK_FILE = "run_session.lock"
CURRICULUM_STATE_LOCK_FILE = "curriculum_state.lock"
LATEST_PRELOAD_FILE = "latest_preload.json"
RESUME_CHECKPOINT_FILE = "model.pkl"
RESUME_META_FILE = "resume.meta.json"
RESUME_STATE_FILE = "resume.state.json"
MANUAL_SAVE_MANIFEST_FILE = "manifest.json"

LEGACY_PRELOAD_RELATIVE_DIR = "agent_ppo/ckpt"
LEGACY_RESUME_CHECKPOINT_FILE = "model.ckpt-resume.pkl"
LEGACY_RESUME_META_FILE = "model.ckpt-resume.meta.json"
LEGACY_RESUME_STATE_FILE = "model.ckpt-resume.state.json"
LEGACY_CURRICULUM_STATE_SNAPSHOT_FILE = "curriculum_state.resume_snapshot.json"
SCRATCH_MODES = {"scratch", "random", "from_scratch", "fresh"}


def _normalize_path(base: Path, raw: str | None) -> Path:
    if not raw:
        return base / DEFAULT_STATE_ROOT_RELATIVE
    target = Path(str(raw))
    return target if target.is_absolute() else (base / target).resolve()


def resolve_state_root(code_dir: Path, env: dict[str, str] | None = None) -> Path:
    env = env or os.environ
    return _normalize_path(code_dir, env.get(STATE_ROOT_ENV))


def training_start_mode(env: dict[str, str] | None = None) -> str:
    env = env or os.environ
    return str(env.get("KAIWU_TRAINING_START_MODE", "preload") or "preload").strip().lower()


def is_scratch_mode(env: dict[str, str] | None = None) -> bool:
    return training_start_mode(env) in SCRATCH_MODES


def is_resume_mode(env: dict[str, str] | None = None) -> bool:
    return training_start_mode(env) == "resume"


def allow_legacy_resume_import(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    return str(env.get(ALLOW_LEGACY_RESUME_IMPORT_ENV, "0") or "0").strip().lower() in {"1", "true", "on", "yes"}


@dataclass(frozen=True)
class CurrentStateLayout:
    root: Path
    current_dir: Path
    run_session_manifest_path: Path
    curriculum_state_path: Path
    curriculum_signal_dir: Path
    locks_dir: Path
    run_session_lock_path: Path
    curriculum_state_lock_path: Path
    prepared_resume_dir: Path


@dataclass(frozen=True)
class RunStateLayout:
    root: Path
    run_session_id: str
    run_dir: Path
    run_manifest_path: Path
    curriculum_state_path: Path
    curriculum_signal_dir: Path
    resume_dir: Path
    resume_latest_dir: Path
    resume_latest_checkpoint_path: Path
    resume_latest_meta_path: Path
    resume_latest_state_path: Path
    resume_latest_curriculum_snapshot_path: Path
    resume_snapshots_dir: Path
    session_best_dir: Path
    manual_checkpoints_dir: Path


@dataclass(frozen=True)
class RuntimeStateLayout:
    code_dir: Path
    state_root: Path
    current: CurrentStateLayout
    preload_cache_dir: Path
    preload_latest_metadata_path: Path
    manual_saves_dir: Path
    manual_saves_latest_index_path: Path

    def for_run(self, run_session_id: str) -> RunStateLayout:
        run_dir = self.state_root / RUNS_DIR_NAME / str(run_session_id)
        resume_dir = run_dir / "resume"
        resume_latest_dir = resume_dir / "latest"
        return RunStateLayout(
            root=self.state_root,
            run_session_id=str(run_session_id),
            run_dir=run_dir,
            run_manifest_path=run_dir / RUN_SESSION_MANIFEST_FILE,
            curriculum_state_path=run_dir / CURRICULUM_STATE_FILE,
            curriculum_signal_dir=run_dir / CURRICULUM_SIGNALS_DIR_NAME,
            resume_dir=resume_dir,
            resume_latest_dir=resume_latest_dir,
            resume_latest_checkpoint_path=resume_latest_dir / RESUME_CHECKPOINT_FILE,
            resume_latest_meta_path=resume_latest_dir / RESUME_META_FILE,
            resume_latest_state_path=resume_latest_dir / RESUME_STATE_FILE,
            resume_latest_curriculum_snapshot_path=resume_latest_dir / CURRICULUM_STATE_SNAPSHOT_FILE,
            resume_snapshots_dir=resume_dir / "snapshots",
            session_best_dir=run_dir / "session_best",
            manual_checkpoints_dir=run_dir / "manual_checkpoints",
        )


def runtime_state_layout(code_dir: Path, env: dict[str, str] | None = None) -> RuntimeStateLayout:
    state_root = resolve_state_root(code_dir, env)
    current_dir = state_root / CURRENT_DIR_NAME
    locks_dir = current_dir / LOCKS_DIR_NAME
    current = CurrentStateLayout(
        root=state_root,
        current_dir=current_dir,
        run_session_manifest_path=current_dir / RUN_SESSION_MANIFEST_FILE,
        curriculum_state_path=current_dir / CURRICULUM_STATE_FILE,
        curriculum_signal_dir=current_dir / CURRICULUM_SIGNALS_DIR_NAME,
        locks_dir=locks_dir,
        run_session_lock_path=locks_dir / RUN_SESSION_LOCK_FILE,
        curriculum_state_lock_path=locks_dir / CURRICULUM_STATE_LOCK_FILE,
        prepared_resume_dir=current_dir / PREPARED_RESUME_DIR_NAME,
    )
    preload_cache_dir = state_root / PRELOAD_CACHE_DIR_NAME
    manual_saves_dir = state_root / MANUAL_SAVES_DIR_NAME
    return RuntimeStateLayout(
        code_dir=code_dir,
        state_root=state_root,
        current=current,
        preload_cache_dir=preload_cache_dir,
        preload_latest_metadata_path=preload_cache_dir / LATEST_PRELOAD_FILE,
        manual_saves_dir=manual_saves_dir,
        manual_saves_latest_index_path=manual_saves_dir / "latest.json",
    )


def ensure_runtime_state_dirs(code_dir: Path, run_session_id: str | None = None, env: dict[str, str] | None = None) -> RuntimeStateLayout:
    layout = runtime_state_layout(code_dir, env)
    for path in (
        layout.state_root,
        layout.current.current_dir,
        layout.current.curriculum_signal_dir,
        layout.current.locks_dir,
        layout.current.prepared_resume_dir,
        layout.preload_cache_dir,
        layout.manual_saves_dir,
    ):
        path.mkdir(parents=True, exist_ok=True)
    if run_session_id:
        run_layout = layout.for_run(run_session_id)
        for path in (
            run_layout.run_dir,
            run_layout.curriculum_signal_dir,
            run_layout.resume_latest_dir,
            run_layout.resume_snapshots_dir,
            run_layout.session_best_dir,
            run_layout.manual_checkpoints_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)
    return layout


def legacy_resume_latest_checkpoint_path(code_dir: Path) -> Path:
    return code_dir / LEGACY_RESUME_CHECKPOINT_FILE


def legacy_resume_latest_meta_path(code_dir: Path) -> Path:
    return code_dir / LEGACY_RESUME_META_FILE


def legacy_resume_latest_state_path(code_dir: Path) -> Path:
    return code_dir / LEGACY_RESUME_STATE_FILE


def legacy_resume_curriculum_snapshot_path(code_dir: Path) -> Path:
    return code_dir / LEGACY_CURRICULUM_STATE_SNAPSHOT_FILE


def legacy_resume_snapshots_dir(code_dir: Path) -> Path:
    return code_dir / "resume_snapshots"


def legacy_curriculum_state_path(code_dir: Path) -> Path:
    return code_dir / CURRICULUM_STATE_FILE


def legacy_curriculum_signal_dir(code_dir: Path) -> Path:
    return code_dir / CURRICULUM_SIGNALS_DIR_NAME


def legacy_run_session_manifest_path(code_dir: Path) -> Path:
    return code_dir / ".current_run_session.json"


def legacy_run_session_lock_path(code_dir: Path) -> Path:
    return code_dir / ".current_run_session.lock"


def legacy_curriculum_state_lock_path(code_dir: Path) -> Path:
    return code_dir / ".curriculum_state.lock"


def legacy_preload_cache_dir(code_dir: Path) -> Path:
    return code_dir / LEGACY_PRELOAD_RELATIVE_DIR


def resolve_resume_bundle_dir(code_dir: Path, env: dict[str, str] | None = None) -> Path | None:
    env = env or os.environ
    explicit_bundle = str(env.get(RESUME_BUNDLE_DIR_ENV, "") or "").strip()
    if explicit_bundle:
        bundle = Path(explicit_bundle)
        return bundle if bundle.is_absolute() else (code_dir / bundle).resolve()

    resume_run_id = str(env.get(RESUME_RUN_ID_ENV, "") or "").strip()
    layout = runtime_state_layout(code_dir, env)
    if resume_run_id:
        return layout.for_run(resume_run_id).resume_latest_dir
    return None


def latest_preload_file_candidates(code_dir: Path, env: dict[str, str] | None = None) -> list[Path]:
    layout = runtime_state_layout(code_dir, env)
    return [
        layout.preload_latest_metadata_path,
        legacy_preload_cache_dir(code_dir) / LATEST_PRELOAD_FILE,
    ]


def latest_resume_state_file_candidates(code_dir: Path, env: dict[str, str] | None = None) -> list[Path]:
    candidates: list[Path] = []
    bundle_dir = resolve_resume_bundle_dir(code_dir, env)
    if bundle_dir is not None:
        candidates.append(bundle_dir / RESUME_STATE_FILE)
    override = str((env or os.environ).get("KAIWU_RESUME_STATE_METADATA_PATH", "") or "").strip()
    if override:
        override_path = Path(override)
        candidates.insert(0, override_path if override_path.is_absolute() else (code_dir / override_path).resolve())
    if allow_legacy_resume_import(env):
        candidates.append(legacy_resume_latest_state_path(code_dir))
    return candidates


def latest_resume_checkpoint_file_candidates(code_dir: Path, env: dict[str, str] | None = None) -> list[Path]:
    candidates: list[Path] = []
    bundle_dir = resolve_resume_bundle_dir(code_dir, env)
    if bundle_dir is not None:
        candidates.append(bundle_dir / RESUME_CHECKPOINT_FILE)
    if allow_legacy_resume_import(env):
        candidates.append(legacy_resume_latest_checkpoint_path(code_dir))
    return candidates


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except IsADirectoryError:
        shutil.rmtree(path, ignore_errors=True)


def clear_current_runtime_state(code_dir: Path, env: dict[str, str] | None = None, clear_legacy_current: bool = False) -> RuntimeStateLayout:
    layout = ensure_runtime_state_dirs(code_dir, env=env)
    _unlink_if_exists(layout.current.run_session_manifest_path)
    _unlink_if_exists(layout.current.curriculum_state_path)
    shutil.rmtree(layout.current.curriculum_signal_dir, ignore_errors=True)
    shutil.rmtree(layout.current.prepared_resume_dir, ignore_errors=True)
    layout.current.curriculum_signal_dir.mkdir(parents=True, exist_ok=True)
    layout.current.prepared_resume_dir.mkdir(parents=True, exist_ok=True)
    if clear_legacy_current:
        _unlink_if_exists(legacy_curriculum_state_path(code_dir))
        shutil.rmtree(legacy_curriculum_signal_dir(code_dir), ignore_errors=True)
        _unlink_if_exists(legacy_run_session_manifest_path(code_dir))
        _unlink_if_exists(legacy_run_session_lock_path(code_dir))
        _unlink_if_exists(legacy_curriculum_state_lock_path(code_dir))
    return layout


def prepare_resume_bundle_in_current(code_dir: Path, env: dict[str, str] | None = None) -> Path:
    env = env or os.environ
    bundle_dir = resolve_resume_bundle_dir(code_dir, env)
    if bundle_dir is None:
        raise FileNotFoundError("resume start mode requires KAIWU_RESUME_BUNDLE_DIR or KAIWU_RESUME_RUN_ID")
    source_bundle = bundle_dir.resolve()
    checkpoint_path = source_bundle / RESUME_CHECKPOINT_FILE
    state_path = source_bundle / RESUME_STATE_FILE
    curriculum_path = source_bundle / CURRICULUM_STATE_SNAPSHOT_FILE
    missing = [str(path.name) for path in (checkpoint_path, state_path, curriculum_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"resume bundle is incomplete: missing {', '.join(missing)} in {source_bundle}")
    layout = clear_current_runtime_state(code_dir, env=env, clear_legacy_current=False)
    target_bundle = layout.current.prepared_resume_dir
    shutil.copy2(checkpoint_path, target_bundle / RESUME_CHECKPOINT_FILE)
    shutil.copy2(state_path, target_bundle / RESUME_STATE_FILE)
    shutil.copy2(curriculum_path, target_bundle / CURRICULUM_STATE_SNAPSHOT_FILE)
    source_manifest = source_bundle / RUN_SESSION_MANIFEST_FILE
    if source_manifest.exists():
        shutil.copy2(source_manifest, target_bundle / RUN_SESSION_MANIFEST_FILE)
    elif env.get(RESUME_RUN_ID_ENV):
        run_manifest = runtime_state_layout(code_dir, env).for_run(str(env.get(RESUME_RUN_ID_ENV))).run_manifest_path
        if run_manifest.exists():
            shutil.copy2(run_manifest, target_bundle / RUN_SESSION_MANIFEST_FILE)
    meta_payload = {
        "prepared_from": str(source_bundle),
        "prepared_at": None,
        "checkpoint_path": str(target_bundle / RESUME_CHECKPOINT_FILE),
        "curriculum_state_snapshot_path": str(target_bundle / CURRICULUM_STATE_SNAPSHOT_FILE),
    }
    (target_bundle / RESUME_META_FILE).write_text(
        json.dumps(meta_payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_bundle


__all__ = [
    "STATE_ROOT_ENV",
    "RESUME_BUNDLE_DIR_ENV",
    "RESUME_RUN_ID_ENV",
    "ALLOW_LEGACY_RESUME_IMPORT_ENV",
    "DEFAULT_STATE_ROOT_RELATIVE",
    "CURRENT_DIR_NAME",
    "RUNS_DIR_NAME",
    "MANUAL_SAVES_DIR_NAME",
    "PRELOAD_CACHE_DIR_NAME",
    "LOCKS_DIR_NAME",
    "RUN_SESSION_MANIFEST_FILE",
    "CURRICULUM_STATE_FILE",
    "CURRICULUM_STATE_SNAPSHOT_FILE",
    "CURRICULUM_SIGNALS_DIR_NAME",
    "RUN_SESSION_LOCK_FILE",
    "CURRICULUM_STATE_LOCK_FILE",
    "LATEST_PRELOAD_FILE",
    "RESUME_CHECKPOINT_FILE",
    "RESUME_META_FILE",
    "RESUME_STATE_FILE",
    "MANUAL_SAVE_MANIFEST_FILE",
    "LEGACY_PRELOAD_RELATIVE_DIR",
    "CurrentStateLayout",
    "RunStateLayout",
    "RuntimeStateLayout",
    "training_start_mode",
    "is_scratch_mode",
    "is_resume_mode",
    "allow_legacy_resume_import",
    "resolve_state_root",
    "runtime_state_layout",
    "ensure_runtime_state_dirs",
    "legacy_resume_latest_checkpoint_path",
    "legacy_resume_latest_meta_path",
    "legacy_resume_latest_state_path",
    "legacy_resume_curriculum_snapshot_path",
    "legacy_resume_snapshots_dir",
    "legacy_curriculum_state_path",
    "legacy_curriculum_signal_dir",
    "legacy_run_session_manifest_path",
    "legacy_run_session_lock_path",
    "legacy_curriculum_state_lock_path",
    "legacy_preload_cache_dir",
    "resolve_resume_bundle_dir",
    "latest_preload_file_candidates",
    "latest_resume_state_file_candidates",
    "latest_resume_checkpoint_file_candidates",
    "clear_current_runtime_state",
    "prepare_resume_bundle_in_current",
]
