#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Lightweight benchmark bootstrap for curriculum initial stage selection.
"""

from __future__ import annotations

import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from agent_ppo.conf.conf import Config
from agent_ppo.workflow.preload_checkpoint import resolve_training_preload
from agent_ppo.workflow.state_layout import is_scratch_mode, training_start_mode
from agent_ppo.workflow.state_layout import runtime_state_layout


LITE_BENCH_FILE = "latest_lite_benchmark.json"
LITE_ROUNDS = [
    {
        "name": "lite_round_1",
        "desc": "4 chargers / 2 robots / 700 steps / 200 battery",
        "charger_count": 4,
        "robot_count": 2,
        "max_step": 700,
        "battery_max": 200,
    },
    {
        "name": "lite_round_2",
        "desc": "2 chargers / 3 robots / 1000 steps / 200 battery",
        "charger_count": 2,
        "robot_count": 3,
        "max_step": 1000,
        "battery_max": 200,
    },
]
LITE_MAPS = [1, 5]


def lite_benchmark_metadata_path(code_dir: Path) -> Path:
    return runtime_state_layout(code_dir).preload_cache_dir / LITE_BENCH_FILE


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _current_checkpoint(code_dir: Path, logger=None) -> dict[str, Any]:
    env = dict(os.environ)
    current_mode = training_start_mode(env)
    if is_scratch_mode(env):
        return {
            "enabled": False,
            "checkpoint_path": "",
            "checkpoint_id": None,
            "training_start_mode": current_mode,
        }
    try:
        preload = resolve_training_preload(code_dir, env)
    except FileNotFoundError as exc:
        if logger is not None:
            logger.warning(
                "[LITE_BENCH] preload resolution failed in start_mode=%s; fallback to checkpoint-disabled payload: %s",
                current_mode,
                exc,
            )
        return {
            "enabled": False,
            "checkpoint_path": "",
            "checkpoint_id": None,
            "training_start_mode": current_mode,
            "resolution_error": str(exc),
        }
    if preload.get("enabled"):
        return preload
    checkpoint_path = str(getattr(Config, "RESUME_CHECKPOINT", "") or "").strip()
    return {
        "enabled": bool(checkpoint_path),
        "checkpoint_path": checkpoint_path,
        "checkpoint_id": None,
        "training_start_mode": current_mode,
    }


def _mode_enabled(mode: str, checkpoint_payload: dict[str, Any]) -> bool:
    mode = (mode or "resume_only").strip().lower()
    if not Config.CURRICULUM_LITE_BENCH_ENABLED:
        return False
    if mode == "off":
        return False
    if mode == "all":
        return True
    if mode == "resume_only":
        return bool(checkpoint_payload.get("checkpoint_path"))
    return False


def _checkpoint_mtime_ns(checkpoint_path: str | None) -> int:
    if not checkpoint_path:
        return 0
    try:
        return int(Path(checkpoint_path).stat().st_mtime_ns)
    except OSError:
        return 0


def _lite_cache_signature() -> dict[str, Any]:
    return {
        "rounds": deepcopy(LITE_ROUNDS),
        "maps": list(LITE_MAPS),
        "policy_mode": Config.CURRICULUM_LITE_BENCH_POLICY_MODE,
        "schema_version": 1,
    }


def _recommended_initial_stage(metrics: dict[str, Any]) -> str:
    if (
        float(metrics.get("completed_rate", 0.0)) >= 0.70
        and float(metrics.get("battery_fail_rate", 1.0)) <= 0.10
        and float(metrics.get("collision_fail_rate", 1.0)) <= 0.05
        and float(metrics.get("broad_win_rate", 0.0)) >= 0.65
        and float(metrics.get("return_stall_rate", 1.0)) <= 0.40
    ):
        return "robust"
    if (
        float(metrics.get("completed_rate", 0.0)) >= 0.55
        and float(metrics.get("battery_fail_rate", 1.0)) <= 0.22
        and float(metrics.get("collision_fail_rate", 1.0)) <= 0.10
    ):
        return "blend"
    if (
        float(metrics.get("completed_rate", 0.0)) >= 0.90
        and float(metrics.get("battery_fail_rate", 1.0)) <= 0.05
        and float(metrics.get("collision_fail_rate", 1.0)) <= 0.05
    ):
        return "blend"
    return "warmup"


def resolve_cached_lite_benchmark(code_dir: Path, checkpoint_path: str | None) -> dict[str, Any] | None:
    payload = _read_json(lite_benchmark_metadata_path(code_dir))
    if not payload:
        return None
    cached_path = str(payload.get("checkpoint_path") or "")
    if checkpoint_path and cached_path != checkpoint_path:
        return None
    if int(payload.get("checkpoint_mtime_ns") or 0) != _checkpoint_mtime_ns(checkpoint_path):
        return None
    if payload.get("cache_signature") != _lite_cache_signature():
        return None
    return payload


def wait_for_lite_benchmark_result(code_dir: Path, checkpoint_path: str | None, timeout_seconds: int) -> dict[str, Any] | None:
    deadline = time.time() + max(int(timeout_seconds), 1)
    while time.time() < deadline:
        cached = resolve_cached_lite_benchmark(code_dir, checkpoint_path)
        if cached:
            return cached
        time.sleep(2.0)
    return None


def maybe_run_lite_benchmark(env, agent, usr_conf, logger) -> dict[str, Any] | None:
    code_dir = Path("/workspace/code")
    checkpoint_payload = _current_checkpoint(code_dir, logger=logger)
    checkpoint_path = str(checkpoint_payload.get("checkpoint_path") or "")
    if not _mode_enabled(Config.CURRICULUM_LITE_BENCH_MODE, checkpoint_payload):
        return None

    cached = resolve_cached_lite_benchmark(code_dir, checkpoint_path)
    if cached:
        return cached

    aisrv_index = str(os.getenv("KAIWU_AISRV_INDEX", "1") or "1").strip()
    if aisrv_index not in {"", "1"}:
        return wait_for_lite_benchmark_result(code_dir, checkpoint_path, Config.CURRICULUM_LITE_BENCH_TIMEOUT_SECONDS)

    from agent_ppo.eval import benchmark as benchmark_mod

    session_dir = benchmark_mod.EVAL_LOG_BASE / f"lite_bootstrap_{time.strftime('%Y%m%d-%H%M%S')}"
    session_dir.mkdir(parents=True, exist_ok=True)
    step_log = benchmark_mod._create_step_logger(session_dir)
    base_env_conf = benchmark_mod._extract_base_env_conf(usr_conf)
    results = []
    idx = 0
    total = len(LITE_ROUNDS) * len(LITE_MAPS)

    policy_mode_prev = os.environ.get("KAIWU_BENCHMARK_POLICY_MODE")
    os.environ["KAIWU_BENCHMARK_POLICY_MODE"] = Config.CURRICULUM_LITE_BENCH_POLICY_MODE
    try:
        for round_def in LITE_ROUNDS:
            for map_id in LITE_MAPS:
                idx += 1
                env_conf = deepcopy(base_env_conf)
                env_conf["map"] = [map_id]
                env_conf["map_random"] = False
                env_conf["robot_count"] = round_def["robot_count"]
                env_conf["charger_count"] = round_def["charger_count"]
                env_conf["max_step"] = round_def["max_step"]
                env_conf["battery_max"] = round_def["battery_max"]
                wrapped_conf = benchmark_mod._wrap_env_conf(usr_conf, env_conf)
                results.append(
                    benchmark_mod._run_eval_episode(
                        env=env,
                        agent=agent,
                        usr_conf=wrapped_conf,
                        round_name=round_def["name"],
                        map_id=map_id,
                        round_def=round_def,
                        logger=logger,
                        step_log=step_log,
                        idx=idx,
                        total=total,
                        session_dir=session_dir,
                    )
                )
    finally:
        if policy_mode_prev is None:
            os.environ.pop("KAIWU_BENCHMARK_POLICY_MODE", None)
        else:
            os.environ["KAIWU_BENCHMARK_POLICY_MODE"] = policy_mode_prev
        for handler in step_log.handlers[:]:
            handler.close()
            step_log.removeHandler(handler)

    aggregated = benchmark_mod._aggregate_results(results)
    overall = aggregated["overall"]
    payload = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_id": checkpoint_payload.get("checkpoint_id"),
        "checkpoint_mtime_ns": _checkpoint_mtime_ns(checkpoint_path),
        "cache_signature": _lite_cache_signature(),
        "completed_rate": float(overall.get("completed_rate", overall.get("win_rate", 0.0))),
        "battery_fail_rate": float(overall.get("battery_fail_rate", 0.0)),
        "collision_fail_rate": float(overall.get("collision_fail_rate", 0.0)),
        "broad_win_rate": float(overall.get("broad_win_rate", 0.0)),
        "return_stall_rate": float(overall.get("return_stall_rate", 0.0)),
        "recommended_initial_stage": _recommended_initial_stage(overall),
        "episode_count": int(overall.get("episode_count", len(results))),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "used": True,
        "source": "fresh",
    }
    _write_json(lite_benchmark_metadata_path(code_dir), payload)
    logger.info(
        "[LITE_BENCH] completed_rate=%.2f battery_fail=%.2f return_stall=%.2f broad_win=%.2f stage=%s",
        payload["completed_rate"],
        payload["battery_fail_rate"],
        payload["return_stall_rate"],
        payload["broad_win_rate"],
        payload["recommended_initial_stage"],
    )
    return payload
