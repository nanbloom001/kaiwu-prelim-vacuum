#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from __future__ import annotations

import ast
import re
from pathlib import Path

SPAWN_PATCH_MARKER = "_COPILOT_ZMQ_SPAWN_PATCH_APPLIED"
START_METHOD_PATCH_MARKER = "_COPILOT_GLOBAL_SPAWN_START_METHOD_APPLIED"
ZMQ_SPAWN_TARGETS = (
    Path("kaiwudrl/common/replay_buffer/zmq_replay_buffer.py"),
    Path("kaiwudrl/server/learner/learner_server.py"),
)
ENTRYPOINT_SPAWN_TARGETS = (
    Path("kaiwudrl/server/learner/learner.py"),
    Path("kaiwudrl/server/aisrv/aisrv.py"),
)

_ENCODING_RE = re.compile(r"coding[:=]\s*([-\w.]+)")
_SPAWN_CONTEXT_SYMBOLS = (
    "Process",
    "Queue",
    "SimpleQueue",
    "JoinableQueue",
    "Event",
    "Lock",
    "RLock",
    "Semaphore",
    "BoundedSemaphore",
    "Condition",
    "Barrier",
    "Pool",
    "Array",
    "Value",
)


def _find_insertion_line(source: str) -> int:
    insert_after = 0
    lines = source.splitlines(keepends=True)

    if lines and lines[0].startswith("#!"):
        insert_after = 1

    for index, line in enumerate(lines[:2]):
        if _ENCODING_RE.search(line):
            insert_after = max(insert_after, index + 1)

    try:
        module = ast.parse(source)
    except Exception:
        return insert_after

    body = list(module.body)
    body_index = 0
    if body:
        first_node = body[0]
        if isinstance(first_node, ast.Expr) and isinstance(getattr(first_node, "value", None), ast.Constant):
            if isinstance(first_node.value.value, str):
                insert_after = max(insert_after, first_node.end_lineno or first_node.lineno)
                body_index = 1

    while body_index < len(body):
        node = body[body_index]
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            insert_after = max(insert_after, node.end_lineno or node.lineno)
            body_index += 1
            continue
        break

    return insert_after


def _build_spawn_prelude() -> str:
    lines = [
        "import multiprocessing as _copilot_mp",
        f"if not globals().get('{SPAWN_PATCH_MARKER}', False):",
        '    _copilot_spawn_ctx = _copilot_mp.get_context("spawn")',
    ]
    for symbol in _SPAWN_CONTEXT_SYMBOLS:
        lines.append(f"    if hasattr(_copilot_spawn_ctx, '{symbol}'):")
        lines.append(f"        setattr(_copilot_mp, '{symbol}', getattr(_copilot_spawn_ctx, '{symbol}'))")
    lines.append(f"    {SPAWN_PATCH_MARKER} = True")
    return "\n".join(lines) + "\n\n"


def _build_start_method_prelude() -> str:
    lines = [
        "import multiprocessing as _copilot_mp",
        "import sys as _copilot_sys",
        f"if not globals().get('{START_METHOD_PATCH_MARKER}', False):",
        '    if _copilot_sys.platform.startswith("linux") and _copilot_mp.get_start_method(allow_none=True) != "spawn":',
        '        _copilot_mp.set_start_method("spawn", force=True)',
        f"    {START_METHOD_PATCH_MARKER} = True",
    ]
    return "\n".join(lines) + "\n\n"


def inject_spawn_context_prelude(source: str) -> tuple[str, bool]:
    """Inject a spawn-context prelude before regular imports.

    The prelude must run before the target module imports multiprocessing
    symbols such as Process or Queue; injecting it near the top of the file
    allows subsequent imports in that module to resolve to spawn-context
    implementations on Linux.
    """
    if SPAWN_PATCH_MARKER in source:
        return source, False

    lines = source.splitlines(keepends=True)
    insert_after = _find_insertion_line(source)
    offset = sum(len(line) for line in lines[:insert_after])
    updated = source[:offset] + _build_spawn_prelude() + source[offset:]
    return updated, True


def inject_spawn_start_method_prelude(source: str) -> tuple[str, bool]:
    """Inject a global spawn start-method prelude into an entrypoint module."""
    if START_METHOD_PATCH_MARKER in source:
        return source, False

    lines = source.splitlines(keepends=True)
    insert_after = _find_insertion_line(source)
    offset = sum(len(line) for line in lines[:insert_after])
    updated = source[:offset] + _build_start_method_prelude() + source[offset:]
    return updated, True


def patch_file_with_spawn_context(target: Path | str) -> bool:
    """Patch a framework file in place if it has not been patched yet."""
    target_path = Path(target)
    if not target_path.exists():
        return False

    source = target_path.read_text()
    if SPAWN_PATCH_MARKER in source:
        return False

    updated, changed = inject_spawn_context_prelude(source)
    if changed:
        target_path.write_text(updated)
    return changed


def patch_file_with_spawn_start_method(target: Path | str) -> bool:
    """Patch an entrypoint file to force the multiprocessing start method to spawn."""
    target_path = Path(target)
    if not target_path.exists():
        return False

    source = target_path.read_text()
    if START_METHOD_PATCH_MARKER in source:
        return False

    updated, changed = inject_spawn_start_method_prelude(source)
    if changed:
        target_path.write_text(updated)
    return changed


def patch_zmq_entrypoints(root: Path | str, replay_buffer_type: str | None) -> list[Path]:
    """Patch learner and AISRV entrypoints to use spawn when ZMQ is enabled."""
    if (replay_buffer_type or "").strip().lower() != "zmq":
        return []

    root_path = Path(root)
    patched_paths = []
    for relative_path in ENTRYPOINT_SPAWN_TARGETS:
        target_path = root_path / relative_path
        if patch_file_with_spawn_start_method(target_path):
            patched_paths.append(target_path)
    return patched_paths


def patch_zmq_runtime_files(root: Path | str, replay_buffer_type: str | None) -> list[Path]:
    """Patch ZMQ runtime files only when the configured replay buffer is ZMQ."""
    if (replay_buffer_type or "").strip().lower() != "zmq":
        return []

    root_path = Path(root)
    patched_paths = []
    for relative_path in ZMQ_SPAWN_TARGETS:
        target_path = root_path / relative_path
        if patch_file_with_spawn_context(target_path):
            patched_paths.append(target_path)
    return patched_paths