#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Parallel benchmark evaluation module.

This runner keeps the normal training workflow untouched and uses a shared
runtime directory plus multiple aisrv workers to execute benchmark episodes in
parallel. It supports:

- Multiple aisrv workers via docker-compose scaling
- Dynamic task queue with atomic file claims
- Per-worker heartbeat and stale-claim recovery
- Optional per-aisrv multi-slot execution when workflow exposes multiple
  env/agent handles
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from copy import deepcopy
from pathlib import Path

from agent_ppo.conf.conf import Config


HEARTBEAT_INTERVAL_SECONDS = 5.0
STALE_WORKER_SECONDS = 30.0
POLL_SECONDS = 2.0
RESULTS_FILE_DEFAULT = "/workspace/train/eval_parallel_results.json"
DEFAULT_ALL_MAPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
DEFAULT_ROUNDS = [
    {
        "name": "round_1",
        "desc": "4 chargers / 3 robots / 1000 steps / 200 battery",
        "charger_count": 4,
        "robot_count": 3,
        "max_step": 1000,
        "battery_max": 200,
    },
    {
        "name": "round_2",
        "desc": "3 chargers / 4 robots / 1200 steps / 300 battery",
        "charger_count": 3,
        "robot_count": 4,
        "max_step": 1200,
        "battery_max": 300,
    },
    {
        "name": "round_3",
        "desc": "2 chargers / 4 robots / 1600 steps / 200 battery",
        "charger_count": 2,
        "robot_count": 4,
        "max_step": 1600,
        "battery_max": 200,
    },
    {
        "name": "round_4",
        "desc": "2 chargers / 4 robots / 2000 steps / 200 battery",
        "charger_count": 2,
        "robot_count": 4,
        "max_step": 2000,
        "battery_max": 200,
    },
]


def _benchmark_api():
    from agent_ppo.eval import benchmark as benchmark_mod

    return benchmark_mod


def _rounds_and_maps():
    try:
        benchmark_mod = _benchmark_api()
        return benchmark_mod.ROUNDS, benchmark_mod.ALL_MAPS
    except ModuleNotFoundError:
        return DEFAULT_ROUNDS, DEFAULT_ALL_MAPS


def run_parallel_benchmark(envs, agents, usr_conf, logger):
    """Run fixed benchmark scenarios with multiple aisrv workers."""
    requested_slots = _env_int("KAIWU_BENCHMARK_ENVS_PER_WORKER", 1)
    worker_count = _env_int("KAIWU_BENCHMARK_WORKER_COUNT", _env_int("KAIWU_AISRV_NUM", 1))
    worker_id = os.getenv("KAIWU_AISRV_INDEX", "").strip() or "1"
    scheduler = os.getenv("KAIWU_BENCHMARK_SCHEDULER", "dynamic").strip() or "dynamic"
    checkpoint = os.getenv("KAIWU_BENCHMARK_CHECKPOINT", "").strip() or Config.RESUME_CHECKPOINT
    session_id = os.getenv("KAIWU_BENCHMARK_SESSION_ID", "").strip() or time.strftime("%Y%m%d-%H%M%S")
    benchmark_mod = _benchmark_api()
    runtime_dir = Path(
        os.getenv("KAIWU_BENCHMARK_RUNTIME_DIR", "").strip()
        or f"/workspace/train/benchmark_runtime/{session_id}"
    )
    results_file = Path(os.getenv("KAIWU_BENCHMARK_RESULTS_FILE", "").strip() or RESULTS_FILE_DEFAULT)
    rounds, maps = _rounds_and_maps()
    total_episodes = len(rounds) * len(maps)

    slot_count = determine_effective_slot_count(requested_slots, envs, agents)
    coordinator = worker_id == "1"

    logger.info("[PBENCH] ========== Parallel Evaluation Start ==========")
    logger.info(
        f"[PBENCH] session={session_id} worker={worker_id}/{worker_count} "
        f"scheduler={scheduler} checkpoint={checkpoint} "
        f"requested_slots={requested_slots} effective_slots={slot_count}"
    )
    logger.info(
        f"[PBENCH] available_env_handles={len(envs)} available_agent_handles={len(agents)} "
        f"total_episodes={total_episodes} runtime_dir={runtime_dir}"
    )

    if scheduler != "dynamic":
        raise ValueError(f"Unsupported benchmark scheduler: {scheduler}")

    if slot_count <= 0:
        raise RuntimeError("No env/agent handles available for parallel benchmark worker")

    _ensure_runtime_layout(runtime_dir)
    base_env_conf = benchmark_mod._extract_base_env_conf(usr_conf)

    if coordinator:
        _initialize_session(
            runtime_dir=runtime_dir,
            session_id=session_id,
            checkpoint=checkpoint,
            worker_count=worker_count,
            requested_slots=requested_slots,
            effective_slots=slot_count,
            available_env_handles=len(envs),
            available_agent_handles=len(agents),
            scheduler=scheduler,
            base_env_conf=base_env_conf,
        )
    else:
        _wait_for_manifest(runtime_dir, logger)

    stop_event = threading.Event()
    heartbeat_thread = threading.Thread(
        target=_heartbeat_loop,
        args=(runtime_dir, worker_id, slot_count, stop_event),
        daemon=True,
    )
    heartbeat_thread.start()

    slot_threads = []
    for slot_index in range(slot_count):
        worker_thread = threading.Thread(
            target=_slot_worker_loop,
            args=(
                runtime_dir,
                worker_id,
                slot_index,
                envs[slot_index],
                agents[slot_index],
                usr_conf,
                checkpoint,
                logger,
            ),
            daemon=True,
        )
        slot_threads.append(worker_thread)
        worker_thread.start()

    done_path = runtime_dir / "done.json"
    try:
        while True:
            if done_path.exists():
                break

            completed_count = _count_json_files(runtime_dir / "tasks" / "completed")
            if coordinator and completed_count >= total_episodes:
                aggregated = _finalize_parallel_benchmark(
                    runtime_dir=runtime_dir,
                    session_id=session_id,
                    checkpoint=checkpoint,
                    worker_count=worker_count,
                    requested_slots=requested_slots,
                    effective_slots=slot_count,
                    scheduler=scheduler,
                    total_episodes=total_episodes,
                    results_file=results_file,
                    logger=logger,
                )
                _atomic_write_json(
                    done_path,
                    {
                        "session_id": session_id,
                        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                        "checkpoint": checkpoint,
                        "overall": aggregated["overall"],
                    },
                )
                logger.info("[PBENCH] Completion marker written to %s", done_path)
                break

            if not any(thread.is_alive() for thread in slot_threads):
                if coordinator:
                    logger.warning("[PBENCH] All local worker threads exited before completion")
                else:
                    logger.info("[PBENCH] Local worker threads exited; waiting for coordinator")
                if not coordinator:
                    _wait_for_done(done_path, logger)
                break

            time.sleep(POLL_SECONDS)
    finally:
        stop_event.set()
        for thread in slot_threads:
            thread.join(timeout=1.0)
        heartbeat_thread.join(timeout=1.0)


def determine_effective_slot_count(requested_slots, envs, agents):
    """Return the safe per-worker slot count for this workflow invocation."""
    available_envs = len(envs or [])
    available_agents = len(agents or [])
    return max(1, min(int(requested_slots), available_envs, available_agents))


def build_parallel_tasks():
    """Build the deterministic benchmark task list."""
    tasks = []
    idx = 0
    rounds, maps = _rounds_and_maps()
    for round_def in rounds:
        for map_id in maps:
            idx += 1
            tasks.append(
                {
                    "task_id": f"{idx:04d}-{round_def['name']}-map{map_id}",
                    "idx": idx,
                    "total": len(rounds) * len(maps),
                    "round_name": round_def["name"],
                    "map_id": map_id,
                    "round_def": deepcopy(round_def),
                    "requeue_count": 0,
                }
            )
    return tasks


def recover_stale_claims(runtime_dir: Path, worker_timeout_seconds: float = STALE_WORKER_SECONDS):
    """Move abandoned claimed tasks back into the pending queue."""
    now = time.time()
    requeued = []
    claimed_root = runtime_dir / "tasks" / "claimed"
    for claimed_path in sorted(claimed_root.glob("*/*.json")):
        owner = claimed_path.parent.name
        heartbeat_path = runtime_dir / "workers" / f"{owner}.json"
        heartbeat = _read_json(heartbeat_path) if heartbeat_path.exists() else {}
        updated_at = float(heartbeat.get("updated_at", 0.0) or 0.0)
        if updated_at and now - updated_at <= worker_timeout_seconds:
            continue

        task_payload = _read_json(claimed_path)
        task_payload["requeue_count"] = int(task_payload.get("requeue_count", 0)) + 1
        task_payload["claimed_by"] = None
        task_payload["claimed_at"] = None

        pending_path = runtime_dir / "tasks" / "pending" / claimed_path.name
        _atomic_write_json(pending_path, task_payload)
        try:
            claimed_path.unlink()
        except FileNotFoundError:
            continue
        requeued.append(task_payload["task_id"])
    return requeued


def _initialize_session(
    runtime_dir: Path,
    session_id: str,
    checkpoint: str,
    worker_count: int,
    requested_slots: int,
    effective_slots: int,
    available_env_handles: int,
    available_agent_handles: int,
    scheduler: str,
    base_env_conf,
):
    manifest_path = runtime_dir / "manifest.json"
    if manifest_path.exists():
        return

    _ensure_runtime_layout(runtime_dir)

    tasks = build_parallel_tasks()
    for task in tasks:
        _atomic_write_json(runtime_dir / "tasks" / "pending" / f"{task['task_id']}.json", task)

    rounds, maps = _rounds_and_maps()
    benchmark_mod = _benchmark_api()
    manifest = {
        "timestamp": session_id,
        "created_at": time.time(),
        "checkpoint": checkpoint,
        "git_commit": benchmark_mod._get_git_commit(),
        "rounds": rounds,
        "maps": maps,
        "total_episodes": len(tasks),
        "execution": {
            "mode": "parallel",
            "worker_count": worker_count,
            "scheduler": scheduler,
            "requested_envs_per_worker": requested_slots,
            "effective_envs_per_worker": effective_slots,
            "available_env_handles": available_env_handles,
            "available_agent_handles": available_agent_handles,
            "compose_project": os.getenv("COMPOSE_PROJECT_NAME", ""),
            "runtime_dir": str(runtime_dir),
        },
        "base_env_conf": deepcopy(base_env_conf),
    }
    _atomic_write_json(manifest_path, manifest)


def _slot_worker_loop(runtime_dir, worker_id, slot_index, env, agent, usr_conf, checkpoint, logger):
    slot_name = f"aisrv-{worker_id}-slot-{slot_index + 1}"
    slot_logger = _create_worker_step_logger(runtime_dir, worker_id, slot_index)

    _benchmark_api()._load_benchmark_checkpoint(agent, checkpoint, logger)
    logger.info("[PBENCH] %s loaded checkpoint %s", slot_name, checkpoint)

    while True:
        done_path = runtime_dir / "done.json"
        if done_path.exists():
            return

        task_payload = _claim_next_task(runtime_dir, worker_id, slot_name)
        if task_payload is None:
            requeued = recover_stale_claims(runtime_dir)
            if requeued:
                logger.warning("[PBENCH] %s requeued stale tasks: %s", slot_name, ",".join(requeued))

            manifest = _read_json(runtime_dir / "manifest.json")
            completed_count = _count_json_files(runtime_dir / "tasks" / "completed")
            total_episodes = int(manifest.get("total_episodes", 0))
            if total_episodes and completed_count >= total_episodes:
                return
            time.sleep(POLL_SECONDS)
            continue

        try:
            result = _execute_task(runtime_dir, env, agent, usr_conf, task_payload, logger, slot_logger)
            _complete_task(runtime_dir, worker_id, task_payload, result)
        except Exception as exc:
            logger.exception("[PBENCH] %s failed task %s: %s", slot_name, task_payload["task_id"], exc)
            _release_claim(runtime_dir, worker_id, task_payload, error=str(exc))
            time.sleep(POLL_SECONDS)


def _execute_task(runtime_dir, env, agent, usr_conf, task_payload, logger, slot_logger):
    manifest = _read_json(runtime_dir / "manifest.json")
    base_env_conf = deepcopy(manifest.get("base_env_conf", {}))
    env_conf = deepcopy(base_env_conf)
    env_conf["map"] = [task_payload["map_id"]]
    env_conf["map_random"] = False
    env_conf["robot_count"] = task_payload["round_def"]["robot_count"]
    env_conf["charger_count"] = task_payload["round_def"]["charger_count"]
    env_conf["max_step"] = task_payload["round_def"]["max_step"]
    env_conf["battery_max"] = task_payload["round_def"]["battery_max"]
    wrapped_conf = _benchmark_api()._wrap_env_conf(usr_conf, env_conf)

    result = _benchmark_api()._run_eval_episode(
        env=env,
        agent=agent,
        usr_conf=wrapped_conf,
        round_name=task_payload["round_name"],
        map_id=task_payload["map_id"],
        round_def=task_payload["round_def"],
        logger=logger,
        step_log=slot_logger,
        idx=task_payload["idx"],
        total=task_payload["total"],
        session_dir=runtime_dir,
    )
    return _normalize_episode_result(result, task_payload)


def _finalize_parallel_benchmark(
    runtime_dir: Path,
    session_id: str,
    checkpoint: str,
    worker_count: int,
    requested_slots: int,
    effective_slots: int,
    scheduler: str,
    total_episodes: int,
    results_file: Path,
    logger,
):
    completed_dir = runtime_dir / "tasks" / "completed"
    episode_results = []
    for completed_path in sorted(completed_dir.glob("*.json")):
        payload = _read_json(completed_path)
        episode_results.append(payload["episode_result"])

    benchmark_mod = _benchmark_api()
    aggregated = benchmark_mod._aggregate_results(episode_results)
    manifest = _read_json(runtime_dir / "manifest.json")

    snapshot = {
        "version": 3,
        "timestamp": session_id,
        "checkpoint": checkpoint,
        "git_commit": manifest.get("git_commit", benchmark_mod._get_git_commit()),
        "elapsed_seconds": round(time.time() - float(manifest.get("created_at", time.time())), 1),
        "rounds": {r["name"]: r["desc"] for r in benchmark_mod.ROUNDS},
        "per_round": aggregated["per_round"],
        "overall": aggregated["overall"],
        "episodes": episode_results,
        "execution": {
            "mode": "parallel",
            "worker_count": worker_count,
            "scheduler": scheduler,
            "requested_envs_per_worker": requested_slots,
            "effective_envs_per_worker": effective_slots,
            "compose_project": os.getenv("COMPOSE_PROJECT_NAME", ""),
            "completed_task_count": len(episode_results),
            "total_episodes": total_episodes,
        },
    }

    benchmark_mod._save_results(results_file, snapshot)
    _atomic_write_json(runtime_dir / "result.json", snapshot)
    logger.info(
        "[PBENCH] Parallel benchmark complete: WR=%s CS=%s (%s/%s)",
        aggregated["overall"]["win_rate"],
        aggregated["overall"]["avg_clean_score"],
        aggregated["overall"]["win_episode_count"],
        aggregated["overall"]["episode_count"],
    )
    return aggregated


def _claim_next_task(runtime_dir: Path, worker_id: str, slot_name: str):
    pending_dir = runtime_dir / "tasks" / "pending"
    claimed_dir = runtime_dir / "tasks" / "claimed" / worker_id
    claimed_dir.mkdir(parents=True, exist_ok=True)

    for pending_path in sorted(pending_dir.glob("*.json")):
        claimed_path = claimed_dir / pending_path.name
        try:
            os.replace(pending_path, claimed_path)
        except FileNotFoundError:
            continue

        payload = _read_json(claimed_path)
        payload["claimed_by"] = slot_name
        payload["claimed_at"] = time.time()
        _atomic_write_json(claimed_path, payload)
        return payload
    return None


def _complete_task(runtime_dir: Path, worker_id: str, task_payload, result):
    claimed_path = runtime_dir / "tasks" / "claimed" / worker_id / f"{task_payload['task_id']}.json"
    completed_path = runtime_dir / "tasks" / "completed" / f"{task_payload['task_id']}.json"
    payload = deepcopy(task_payload)
    payload["completed_at"] = time.time()
    payload["episode_result"] = result
    _atomic_write_json(completed_path, payload)
    try:
        claimed_path.unlink()
    except FileNotFoundError:
        pass


def _release_claim(runtime_dir: Path, worker_id: str, task_payload, error: str):
    claimed_path = runtime_dir / "tasks" / "claimed" / worker_id / f"{task_payload['task_id']}.json"
    pending_path = runtime_dir / "tasks" / "pending" / f"{task_payload['task_id']}.json"
    payload = deepcopy(task_payload)
    payload["last_error"] = error
    payload["requeue_count"] = int(payload.get("requeue_count", 0)) + 1
    payload["claimed_by"] = None
    payload["claimed_at"] = None
    _atomic_write_json(pending_path, payload)
    try:
        claimed_path.unlink()
    except FileNotFoundError:
        pass


def _heartbeat_loop(runtime_dir: Path, worker_id: str, slot_count: int, stop_event: threading.Event):
    worker_path = runtime_dir / "workers" / f"{worker_id}.json"
    while not stop_event.is_set():
        payload = {
            "worker_id": worker_id,
            "slot_count": slot_count,
            "updated_at": time.time(),
            "hostname": os.getenv("HOSTNAME", ""),
            "pid": os.getpid(),
        }
        _atomic_write_json(worker_path, payload)
        stop_event.wait(HEARTBEAT_INTERVAL_SECONDS)


def _wait_for_manifest(runtime_dir: Path, logger):
    manifest_path = runtime_dir / "manifest.json"
    while not manifest_path.exists():
        logger.info("[PBENCH] Waiting for coordinator manifest at %s", manifest_path)
        time.sleep(POLL_SECONDS)


def _wait_for_done(done_path: Path, logger):
    while not done_path.exists():
        logger.info("[PBENCH] Waiting for completion marker at %s", done_path)
        time.sleep(POLL_SECONDS)


def _ensure_runtime_layout(runtime_dir: Path):
    for path in (
        runtime_dir,
        runtime_dir / "episodes",
        runtime_dir / "logs",
        runtime_dir / "workers",
        runtime_dir / "tasks" / "pending",
        runtime_dir / "tasks" / "claimed",
        runtime_dir / "tasks" / "completed",
    ):
        path.mkdir(parents=True, exist_ok=True)


def _create_worker_step_logger(runtime_dir: Path, worker_id: str, slot_index: int) -> logging.Logger:
    logger_name = f"parallel_benchmark.{runtime_dir.name}.w{worker_id}.s{slot_index + 1}"
    log = logging.getLogger(logger_name)
    log.setLevel(logging.DEBUG)
    log.propagate = False
    if log.handlers:
        return log

    log_file = runtime_dir / "logs" / f"worker-{worker_id}-slot-{slot_index + 1}.log"
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(handler)
    return log


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _count_json_files(directory: Path):
    if not directory.exists():
        return 0
    return sum(1 for _ in directory.glob("*.json"))


def _atomic_write_json(path, data):
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    with tmp.open("w", encoding="utf-8") as file_obj:
        json.dump(data, file_obj, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _normalize_episode_result(result, task_payload):
    payload = dict(result or {})
    payload.setdefault("round", task_payload["round_name"])
    payload.setdefault("map_id", task_payload["map_id"])
    payload.setdefault("result", "error")
    payload.setdefault("clean_score", 0.0)
    payload.setdefault("steps", 0.0)
    payload.setdefault("charge_count", 0.0)
    payload.setdefault("remaining_charge", 0.0)
    payload.setdefault("total_reward", 0.0)
    payload.setdefault("dirt_cleaned", 0)
    payload.setdefault("total_dirt", 0)
    payload.setdefault("dirt_ratio", 0.0)
    payload.setdefault("invalid_move_count", 0)
    payload.setdefault("invalid_move_rate", 0.0)
    payload.setdefault("step_log", "")
    return payload


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return int(default)
    return int(value)
