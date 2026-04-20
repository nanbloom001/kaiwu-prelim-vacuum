#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Helpers for scratch/resume startup isolation.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from agent_ppo.workflow.state_layout import is_scratch_mode


def _clear_dir_contents(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)


def clear_external_framework_state(project_code: str, algorithm: str) -> None:
    """Remove stale framework/modelpool assets that can leak across scratch runs."""
    ckpt_root = Path(f"/data/ckpt/{project_code}_{algorithm}")
    if ckpt_root.exists():
        for pattern in ("model.ckpt-*.pkl", "model.ckpt-*.tar.gz", "id_list"):
            for path in ckpt_root.glob(pattern):
                if path.is_file() or path.is_symlink():
                    path.unlink(missing_ok=True)
        for name in ("process_stop.done", "process_stop.meta.json"):
            (ckpt_root / name).unlink(missing_ok=True)
        for name in ("models", "models_new", "plugins", "convert_models_aisrv"):
            target = ckpt_root / name
            target.mkdir(parents=True, exist_ok=True)
            _clear_dir_contents(target)
    Path("/data/projects/sigterm_pids").unlink(missing_ok=True)


def maybe_clear_external_state_for_scratch(env: dict[str, str] | None = None) -> bool:
    env = env or os.environ
    if not is_scratch_mode(env):
        return False
    project_code = str(env.get("KAIWU_PROJECT_CODE", "robot_vacuum") or "robot_vacuum").strip()
    algorithm = str(env.get("KAIWU_ALGORITHM", "ppo") or "ppo").strip()
    clear_external_framework_state(project_code, algorithm)
    return True


__all__ = [
    "clear_external_framework_state",
    "maybe_clear_external_state_for_scratch",
]
