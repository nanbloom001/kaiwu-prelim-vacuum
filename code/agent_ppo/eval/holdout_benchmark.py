#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# pyright: reportMissingImports=false, reportImplicitRelativeImport=false, reportMissingTypeArgument=false, reportReturnType=false
"""
Inference-only holdout benchmark for maps [4, 7].

Adapted from linux-LTSPPO-charge-constraint eval/benchmark.py but self-contained
for win_YJY branch. Runs inside the aisrv container via KAIWU_BENCHMARK_MODE=1
environment variable. Does NOT send data to learner, does NOT save checkpoints,
does NOT pollute training curriculum or episode counters.

Fixed holdout contract:
  - maps = [4, 7]
  - episodes_per_map = configurable (default 10)
  - max_step = 1000, battery_max = 150, robot_count = 4, charger_count = 3
  - map_random = false
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
from typing import TypedDict

import numpy as np
import torch

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import ActData, ObsData
from agent_ppo.utils.experiment_archive import infer_fail_reason, parse_checkpoint_id

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEMA_VERSION = 2
AI_SUMMARY_SCHEMA_VERSION = 1
FINAL_WINDOW_MAX_ROWS = 50
EVIDENCE_WINDOW_RADIUS = 5
LOW_SLACK_THRESHOLD = -3.0
NPC_NEAR_DISTANCE = 2.5
SCHEMA_CONTRACT = {
    "schema_version": SCHEMA_VERSION,
    "maps": [4, 7],
    "episodes_per_map": "configurable",
    "round_def": {
        "name": "holdout",
        "desc": "3 chargers / 4 robots / 1000 steps / 150 battery",
        "charger_count": 3,
        "robot_count": 4,
        "max_step": 1000,
        "battery_max": 150,
    },
}
EVAL_LOG_BASE = Path("/workspace/code/eval_logs")

# Fixed holdout config
HOLDOUT_ROUNDS = [
    {
        "name": "holdout",
        "desc": "3 chargers / 4 robots / 1000 steps / 150 battery",
        "charger_count": 3,
        "robot_count": 4,
        "max_step": 1000,
        "battery_max": 150,
    },
]


class EpisodeAssignment(TypedDict):
    map_id: int
    ep_idx: int


class ShardAssignment(TypedDict):
    shard_index: int
    shard_count: int
    episodes: list[EpisodeAssignment]
    maps: list[int]
    episodes_per_map: int
    run_id: str
    assignment_path: str


# ---------------------------------------------------------------------------
# Public API — called from train_workflow.py
# ---------------------------------------------------------------------------

def run_holdout_benchmark(env, agent, usr_conf, logger, envs=None, agents=None, process_index=None) -> dict[str, object]:
    """
    Run fixed holdout benchmark and save structured results.

    This function is called inside the aisrv container when
    KAIWU_BENCHMARK_MODE=1 is set. It must NOT call agent.learn(),
    agent.save_model(), or any training-side effect.

    Args:
        env: Framework environment (gamecore connection).
        agent: Framework agent (model inference).
        usr_conf: Base user config from train_env_conf.toml.
        logger: Framework logger.

    Returns:
        dict with 'overall', 'per_map', 'episodes', 'checkpoint'.
    """
    if _env_flag("KAIWU_BENCHMARK_PARALLEL_MODE") or os.getenv("KAIWU_BENCHMARK_SCHEDULER", "").strip().lower() == "dynamic":
        return _run_dynamic_holdout_benchmark(
            env=env,
            agent=agent,
            usr_conf=usr_conf,
            logger=logger,
            envs=envs,
            agents=agents,
            process_index=process_index,
        )

    base_env_conf = _extract_base_env_conf(usr_conf)
    checkpoint_path = os.getenv(
        "KAIWU_BENCHMARK_CHECKPOINT", "model.ckpt-resume.pkl"
    ).strip()
    maps_str = os.getenv("KAIWU_BENCHMARK_MAPS", "4,7").strip()
    requested_maps = [int(m.strip()) for m in maps_str.split(",") if m.strip()]
    episodes_per_map = int(os.getenv("KAIWU_BENCHMARK_EPISODES_PER_MAP", "10").strip())
    sharded_mode = _env_flag("KAIWU_BENCHMARK_SHARDED")
    requested_workers_per_aisrv = _env_int("KAIWU_BENCHMARK_WORKERS_PER_AISRV", 1)

    round_def = HOLDOUT_ROUNDS[0]  # single fixed round
    total_episodes = len(requested_maps) * episodes_per_map
    session_id = time.strftime("%Y%m%d-%H%M%S")
    hostname = socket.gethostname()
    shard_assignment: ShardAssignment | None = None
    current_shard: ShardAssignment | None = None
    execution: dict[str, object] = {"mode": "single"}

    if sharded_mode:
        shard_assignment = _load_shard_assignment(
            hostname=hostname,
            requested_maps=requested_maps,
            episodes_per_map=episodes_per_map,
        )
        if shard_assignment is None:
            logger.info(f"[HOLDOUT-BENCH] host={hostname} has no shard assignment; skipping extra AISRV worker")
            return {"skipped": True, "reason": "no_shard_assignment", "hostname": hostname}
        current_shard = shard_assignment
        total_episodes = len(shard_assignment["episodes"])
        execution = {
            "mode": "sharded",
            "shard_index": shard_assignment["shard_index"],
            "shard_count": shard_assignment["shard_count"],
            "hostname": hostname,
            "run_id": shard_assignment["run_id"],
            "assignment_path": shard_assignment["assignment_path"],
            "requested_maps": list(shard_assignment["maps"]),
            "episodes_per_map": shard_assignment["episodes_per_map"],
            "episodes_run": list(shard_assignment["episodes"]),
            "episode_count": total_episodes,
        }

    logger.info(f"[HOLDOUT-BENCH] ========== Holdout Benchmark Start ==========")
    logger.info(f"[HOLDOUT-BENCH] checkpoint={checkpoint_path}")
    logger.info(f"[HOLDOUT-BENCH] maps={requested_maps} episodes_per_map={episodes_per_map} total={total_episodes}")
    if sharded_mode:
        if current_shard is None:
            raise RuntimeError("Sharded benchmark requested without a valid shard assignment")
        logger.info(f"[HOLDOUT-BENCH] sharded assignment host={hostname} shard={current_shard['shard_index']}/{current_shard['shard_count']} episodes={total_episodes} run_id={current_shard['run_id']}")

    # Create session dir for logs
    session_dir = EVAL_LOG_BASE / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    step_log = _create_step_logger(session_dir)

    # Run episodes
    episode_results = []
    idx = 0
    t_start = time.time()

    assigned_episodes: list[EpisodeAssignment] = (
        shard_assignment["episodes"]
        if shard_assignment is not None
        else [
            {"map_id": map_id, "ep_idx": ep_idx}
            for map_id in requested_maps
            for ep_idx in range(1, episodes_per_map + 1)
        ]
    )
    worker_pairs, worker_downgrade_reason = _build_worker_pairs(
        primary_env=env,
        primary_agent=agent,
        envs=envs,
        agents=agents,
        requested_workers=requested_workers_per_aisrv,
        episode_count=len(assigned_episodes),
    )
    execution["workers_per_aisrv_requested"] = requested_workers_per_aisrv
    execution["workers_per_aisrv_effective"] = len(worker_pairs)
    execution["env_count"] = len(_as_list(envs, env))
    execution["agent_count"] = len(_as_list(agents, agent))
    execution["worker_mode"] = "threaded_env_agent_pairs" if len(worker_pairs) > 1 else "serial"
    if worker_downgrade_reason:
        execution["worker_downgrade_reason"] = worker_downgrade_reason

    active_agents = []
    seen_agent_ids = set()
    for _, worker_agent in worker_pairs:
        agent_id = id(worker_agent)
        if agent_id in seen_agent_ids:
            continue
        seen_agent_ids.add(agent_id)
        active_agents.append(worker_agent)

    loaded_checkpoint = None
    for worker_agent in active_agents:
        loaded_checkpoint = _load_benchmark_checkpoint(worker_agent, checkpoint_path, logger)
    if loaded_checkpoint is None:
        loaded_checkpoint = _load_benchmark_checkpoint(agent, checkpoint_path, logger)

    # Write manifest after worker resolution so concurrency metadata is captured.
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": session_id,
        "checkpoint": loaded_checkpoint,
        "maps": requested_maps,
        "episodes_per_map": episodes_per_map,
        "round_def": round_def,
        "total_episodes": total_episodes,
        "execution": execution,
    }
    _atomic_write_json(session_dir / "manifest.json", manifest)

    if len(worker_pairs) <= 1:
        worker_env, worker_agent = worker_pairs[0]
        for assignment in assigned_episodes:
            idx += 1
            result = _run_assigned_episode(
                assignment=assignment,
                env=worker_env,
                agent=worker_agent,
                base_env_conf=base_env_conf,
                usr_conf=usr_conf,
                idx=idx,
                total=total_episodes,
                session_dir=session_dir,
                logger=logger,
                step_log=step_log,
                round_def=round_def,
                worker_index=0,
            )
            episode_results.append(result)
    else:
        indexed_assignments = [
            (position + 1, assignment)
            for position, assignment in enumerate(assigned_episodes)
        ]
        assignment_batches = _partition_indexed_assignments(indexed_assignments, len(worker_pairs))
        logger.info(
            f"[HOLDOUT-BENCH] using {len(worker_pairs)} env/agent workers inside AISRV "
            f"for {len(assigned_episodes)} assigned episodes"
        )
        with ThreadPoolExecutor(max_workers=len(worker_pairs)) as executor:
            futures = []
            for worker_index, ((worker_env, worker_agent), batch) in enumerate(zip(worker_pairs, assignment_batches)):
                if not batch:
                    continue
                futures.append(
                    executor.submit(
                        _run_assignment_batch,
                        batch=batch,
                        env=worker_env,
                        agent=worker_agent,
                        base_env_conf=base_env_conf,
                        usr_conf=usr_conf,
                        total=total_episodes,
                        session_dir=session_dir,
                        logger=logger,
                        step_log=step_log,
                        round_def=round_def,
                        worker_index=worker_index,
                    )
                )
            for future in as_completed(futures):
                episode_results.extend(future.result())
        episode_results.sort(key=lambda item: (int(item.get("map_id", 0)), int(item.get("ep_idx", 0))))

    elapsed = time.time() - t_start

    # Aggregate
    aggregated = _aggregate_results(episode_results, requested_maps)

    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": session_id,
        "checkpoint": loaded_checkpoint,
        "elapsed_seconds": round(elapsed, 1),
        "contract": {
            "schema_version": SCHEMA_VERSION,
            "maps": requested_maps,
            "episodes_per_map": episodes_per_map,
            "round_def": round_def,
            "fixed_config": {
                "map_random": False,
                "max_step": round_def["max_step"],
                "battery_max": round_def["battery_max"],
                "robot_count": round_def["robot_count"],
                "charger_count": round_def["charger_count"],
            },
            "checkpoint": loaded_checkpoint,
        },
        "round_def": round_def,
        "maps": requested_maps,
        "episodes_per_map": episodes_per_map,
        "overall": aggregated["overall"],
        "per_map": aggregated["per_map"],
        "episodes": episode_results,
        "execution": execution,
    }

    _atomic_write_json(session_dir / "result.json", snapshot)

    ai_summary = _build_ai_summary(snapshot)
    _atomic_write_json(session_dir / "ai_summary.json", ai_summary)

    if sharded_mode:
        current_shard = shard_assignment
        if current_shard is None:
            raise RuntimeError("Sharded benchmark requested without a valid shard assignment")
        shard_results_dir = Path("/workspace/code/holdout_shards/results")
        shard_done_dir = Path("/workspace/code/holdout_shards/done")
        shard_results_dir.mkdir(parents=True, exist_ok=True)
        shard_done_dir.mkdir(parents=True, exist_ok=True)

        shard_result_path = shard_results_dir / f"shard_{current_shard['shard_index']}.json"
        _atomic_write_json(shard_result_path, snapshot)

        done_marker = shard_done_dir / f".done_shard_{current_shard['shard_index']}"
        _ = done_marker.write_text(
            json.dumps({
                "session_id": session_id,
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "checkpoint": loaded_checkpoint,
                "overall": aggregated["overall"],
                "execution": execution,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
    else:
        # Also write to /workspace/code/holdout_result.json for external pickup
        _atomic_write_json(Path("/workspace/code/holdout_result.json"), snapshot)

        # Write completion marker
        done_marker = Path("/workspace/code/.benchmark_done")
        _ = done_marker.write_text(
            json.dumps({
                "session_id": session_id,
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "checkpoint": loaded_checkpoint,
                "overall": aggregated["overall"],
            }, ensure_ascii=False),
            encoding="utf-8",
        )

    for handler in step_log.handlers[:]:
        handler.close()
        step_log.removeHandler(handler)

    _print_summary(aggregated, logger, elapsed)
    logger.info(f"[HOLDOUT-BENCH] Completion marker written to {done_marker}")

    return aggregated


# ---------------------------------------------------------------------------
# Dynamic parallel benchmark runner
# ---------------------------------------------------------------------------

def _run_dynamic_holdout_benchmark(env, agent, usr_conf, logger, envs=None, agents=None, process_index=None):
    base_env_conf = _extract_base_env_conf(usr_conf)
    checkpoint_path = os.getenv(
        "KAIWU_BENCHMARK_CHECKPOINT", "model.ckpt-resume.pkl"
    ).strip()
    maps_str = os.getenv("KAIWU_BENCHMARK_MAPS", "4,7").strip()
    requested_maps = [int(m.strip()) for m in maps_str.split(",") if m.strip()]
    episodes_per_map = _env_int("KAIWU_BENCHMARK_EPISODES_PER_MAP", 4)
    configured_envs_per_aisrv = _env_int(
        "KAIWU_BENCHMARK_ENVS_PER_WORKER",
        _env_int("KAIWU_BENCHMARK_WORKERS_PER_AISRV", _env_int("KAIWU_PARALLEL_ENV_PER_AISRV", 1)),
    )
    aisrv_worker_count = _env_int("KAIWU_BENCHMARK_WORKER_COUNT", _env_int("KAIWU_AISRV_NUM", 1))
    hostname = socket.gethostname()
    shard_assignment = _load_shard_assignment(
        hostname=hostname,
        requested_maps=requested_maps,
        episodes_per_map=episodes_per_map,
    )
    if shard_assignment is None:
        logger.info(f"[HOLDOUT-DYN] host={hostname} has no AISRV assignment; skipping")
        return {"skipped": True, "reason": "no_dynamic_assignment", "hostname": hostname}

    aisrv_worker_id = int(shard_assignment["shard_index"]) + 1
    normalized_process_index = _normalize_process_index(process_index)
    if normalized_process_index >= configured_envs_per_aisrv:
        logger.info(
            f"[HOLDOUT-DYN] process_index={normalized_process_index} exceeds "
            f"configured_envs_per_aisrv={configured_envs_per_aisrv}; skipping"
        )
        return {"skipped": True, "reason": "process_index_out_of_range", "process_index": normalized_process_index}

    session_id = str(shard_assignment["run_id"])
    logical_worker_count = max(aisrv_worker_count, 1) * max(configured_envs_per_aisrv, 1)
    logical_worker_id = (aisrv_worker_id - 1) * max(configured_envs_per_aisrv, 1) + normalized_process_index + 1
    worker_id = str(logical_worker_id)
    coordinator = aisrv_worker_id == 1 and normalized_process_index == 0
    runtime_dir = Path(os.getenv("KAIWU_BENCHMARK_RUNTIME_DIR", "").strip() or "/workspace/code/holdout_shards/dynamic")
    result_path = Path("/workspace/code/holdout_result.json")
    done_marker = Path("/workspace/code/.benchmark_done")
    round_def = HOLDOUT_ROUNDS[0]
    total_episodes = len(requested_maps) * episodes_per_map

    logger.info("[HOLDOUT-DYN] ========== Dynamic Holdout Benchmark Start ==========")
    logger.info(
        f"[HOLDOUT-DYN] session={session_id} worker={logical_worker_id}/{logical_worker_count} "
        f"aisrv={aisrv_worker_id}/{aisrv_worker_count} process_index={normalized_process_index} "
        f"envs_per_aisrv={configured_envs_per_aisrv} checkpoint={checkpoint_path}"
    )
    logger.info(
        f"[HOLDOUT-DYN] visible_env_handles={len(_as_list(envs, env))} "
        f"visible_agent_handles={len(_as_list(agents, agent))} total_episodes={total_episodes}"
    )

    if coordinator:
        _initialize_dynamic_runtime(
            runtime_dir=runtime_dir,
            session_id=session_id,
            checkpoint_path=checkpoint_path,
            requested_maps=requested_maps,
            episodes_per_map=episodes_per_map,
            round_def=round_def,
            aisrv_worker_count=aisrv_worker_count,
            configured_envs_per_aisrv=configured_envs_per_aisrv,
            logical_worker_count=logical_worker_count,
            base_env_conf=base_env_conf,
        )
    else:
        _wait_for_dynamic_manifest(runtime_dir, logger)

    session_dir = EVAL_LOG_BASE / f"{session_id}-worker{logical_worker_id:02d}"
    session_dir.mkdir(parents=True, exist_ok=True)
    step_log = _create_step_logger(session_dir)

    assigned_episode_count = total_episodes
    worker_pairs, worker_downgrade_reason = _build_worker_pairs(
        primary_env=env,
        primary_agent=agent,
        envs=envs,
        agents=agents,
        requested_workers=configured_envs_per_aisrv,
        episode_count=assigned_episode_count,
    )
    active_agents = []
    seen_agent_ids = set()
    for _, worker_agent in worker_pairs:
        agent_id = id(worker_agent)
        if agent_id in seen_agent_ids:
            continue
        seen_agent_ids.add(agent_id)
        active_agents.append(worker_agent)

    loaded_checkpoint = None
    for worker_agent in active_agents:
        loaded_checkpoint = _load_benchmark_checkpoint(worker_agent, checkpoint_path, logger)
    if loaded_checkpoint is None:
        loaded_checkpoint = _load_benchmark_checkpoint(agent, checkpoint_path, logger)

    _write_dynamic_heartbeat(
        runtime_dir=runtime_dir,
        worker_id=worker_id,
        logical_worker_id=logical_worker_id,
        aisrv_worker_id=aisrv_worker_id,
        process_index=normalized_process_index,
        slot_count=len(worker_pairs),
        env_count=len(_as_list(envs, env)),
        agent_count=len(_as_list(agents, agent)),
        worker_downgrade_reason=worker_downgrade_reason,
    )

    t_start = time.time()
    if len(worker_pairs) <= 1:
        _dynamic_slot_loop(
            runtime_dir=runtime_dir,
            worker_id=worker_id,
            slot_index=0,
            env=worker_pairs[0][0],
            agent=worker_pairs[0][1],
            base_env_conf=base_env_conf,
            usr_conf=usr_conf,
            session_dir=session_dir,
            logger=logger,
            step_log=step_log,
            round_def=round_def,
        )
    else:
        logger.info(f"[HOLDOUT-DYN] using {len(worker_pairs)} visible env/agent slots in this workflow process")
        with ThreadPoolExecutor(max_workers=len(worker_pairs)) as executor:
            futures = [
                executor.submit(
                    _dynamic_slot_loop,
                    runtime_dir=runtime_dir,
                    worker_id=worker_id,
                    slot_index=slot_index,
                    env=worker_env,
                    agent=worker_agent,
                    base_env_conf=base_env_conf,
                    usr_conf=usr_conf,
                    session_dir=session_dir,
                    logger=logger,
                    step_log=step_log,
                    round_def=round_def,
                )
                for slot_index, (worker_env, worker_agent) in enumerate(worker_pairs)
            ]
            for future in as_completed(futures):
                future.result()

    if coordinator:
        snapshot = _finalize_dynamic_runtime(
            runtime_dir=runtime_dir,
            session_id=session_id,
            checkpoint=loaded_checkpoint,
            requested_maps=requested_maps,
            episodes_per_map=episodes_per_map,
            round_def=round_def,
            result_path=result_path,
            done_marker=done_marker,
            elapsed_seconds=round(time.time() - t_start, 1),
            logger=logger,
        )
    else:
        snapshot = _wait_for_dynamic_done(done_marker, logger)

    for handler in step_log.handlers[:]:
        handler.close()
        step_log.removeHandler(handler)

    return snapshot


def _initialize_dynamic_runtime(
    runtime_dir,
    session_id,
    checkpoint_path,
    requested_maps,
    episodes_per_map,
    round_def,
    aisrv_worker_count,
    configured_envs_per_aisrv,
    logical_worker_count,
    base_env_conf,
):
    for subdir in ("tasks/pending", "tasks/claimed", "tasks/completed", "workers"):
        (runtime_dir / subdir).mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": session_id,
        "checkpoint": checkpoint_path,
        "maps": requested_maps,
        "episodes_per_map": episodes_per_map,
        "round_def": round_def,
        "total_episodes": len(requested_maps) * episodes_per_map,
        "execution": {
            "mode": "dynamic",
            "scheduler": "dynamic",
            "aisrv_worker_count": aisrv_worker_count,
            "configured_envs_per_aisrv": configured_envs_per_aisrv,
            "logical_worker_count": logical_worker_count,
            "runtime_dir": str(runtime_dir),
        },
        "base_env_conf": deepcopy(base_env_conf),
    }
    _atomic_write_json(runtime_dir / "manifest.json", manifest)

    task_index = 0
    for map_id in requested_maps:
        for ep_idx in range(1, episodes_per_map + 1):
            task_index += 1
            task = {
                "task_id": f"{task_index:04d}-map{map_id}-ep{ep_idx:02d}",
                "idx": task_index,
                "total": len(requested_maps) * episodes_per_map,
                "map_id": map_id,
                "ep_idx": ep_idx,
            }
            _atomic_write_json(runtime_dir / "tasks" / "pending" / f"{task['task_id']}.json", task)


def _wait_for_dynamic_manifest(runtime_dir, logger):
    deadline = time.time() + _env_int("KAIWU_BENCHMARK_ASSIGNMENT_WAIT_SECONDS", 180)
    manifest_path = runtime_dir / "manifest.json"
    while time.time() <= deadline:
        if manifest_path.is_file():
            return
        time.sleep(1.0)
    raise FileNotFoundError(f"Dynamic benchmark manifest not found: {manifest_path}")


def _write_dynamic_heartbeat(
    runtime_dir,
    worker_id,
    logical_worker_id,
    aisrv_worker_id,
    process_index,
    slot_count,
    env_count,
    agent_count,
    worker_downgrade_reason,
):
    payload = {
        "worker_id": worker_id,
        "logical_worker_id": logical_worker_id,
        "aisrv_worker_id": aisrv_worker_id,
        "process_index": process_index,
        "slot_count": slot_count,
        "visible_env_handles": env_count,
        "visible_agent_handles": agent_count,
        "updated_at": time.time(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
    }
    if worker_downgrade_reason:
        payload["worker_downgrade_reason"] = worker_downgrade_reason
    _atomic_write_json(runtime_dir / "workers" / f"{worker_id}.json", payload)


def _dynamic_slot_loop(runtime_dir, worker_id, slot_index, env, agent, base_env_conf, usr_conf, session_dir, logger, step_log, round_def):
    slot_name = f"{worker_id}.{slot_index}"
    while True:
        task = _claim_dynamic_task(runtime_dir, worker_id, slot_name)
        if task is None:
            manifest = _read_json(runtime_dir / "manifest.json")
            total_episodes = int(manifest.get("total_episodes", 0) or 0)
            if total_episodes and _count_json_files(runtime_dir / "tasks" / "completed") >= total_episodes:
                return
            if not any((runtime_dir / "tasks" / "pending").glob("*.json")):
                return
            time.sleep(1.0)
            continue

        try:
            result = _run_assigned_episode(
                assignment={"map_id": int(task["map_id"]), "ep_idx": int(task["ep_idx"])},
                env=env,
                agent=agent,
                base_env_conf=base_env_conf,
                usr_conf=usr_conf,
                idx=int(task["idx"]),
                total=int(task["total"]),
                session_dir=session_dir,
                logger=logger,
                step_log=step_log,
                round_def=round_def,
                worker_index=int(worker_id),
            )
            _complete_dynamic_task(runtime_dir, worker_id, task, result)
        except Exception as exc:
            logger.exception(f"[HOLDOUT-DYN] slot={slot_name} failed task={task.get('task_id')}: {exc}")
            _release_dynamic_task(runtime_dir, worker_id, task, str(exc))
            time.sleep(1.0)


def _claim_dynamic_task(runtime_dir, worker_id, slot_name):
    pending_dir = runtime_dir / "tasks" / "pending"
    claimed_dir = runtime_dir / "tasks" / "claimed" / worker_id
    claimed_dir.mkdir(parents=True, exist_ok=True)
    for pending_path in sorted(pending_dir.glob("*.json")):
        claimed_path = claimed_dir / pending_path.name
        try:
            os.replace(pending_path, claimed_path)
        except FileNotFoundError:
            continue
        task = _read_json(claimed_path)
        task["claimed_by"] = slot_name
        task["claimed_at"] = time.time()
        _atomic_write_json(claimed_path, task)
        return task
    return None


def _complete_dynamic_task(runtime_dir, worker_id, task, result):
    claimed_path = runtime_dir / "tasks" / "claimed" / worker_id / f"{task['task_id']}.json"
    completed_path = runtime_dir / "tasks" / "completed" / f"{task['task_id']}.json"
    payload = deepcopy(task)
    payload["completed_at"] = time.time()
    payload["episode_result"] = result
    _atomic_write_json(completed_path, payload)
    try:
        claimed_path.unlink()
    except FileNotFoundError:
        pass


def _release_dynamic_task(runtime_dir, worker_id, task, error):
    claimed_path = runtime_dir / "tasks" / "claimed" / worker_id / f"{task['task_id']}.json"
    pending_path = runtime_dir / "tasks" / "pending" / f"{task['task_id']}.json"
    payload = deepcopy(task)
    payload["last_error"] = error
    payload["requeue_count"] = int(payload.get("requeue_count", 0) or 0) + 1
    payload["claimed_by"] = None
    payload["claimed_at"] = None
    _atomic_write_json(pending_path, payload)
    try:
        claimed_path.unlink()
    except FileNotFoundError:
        pass


def _finalize_dynamic_runtime(
    runtime_dir,
    session_id,
    checkpoint,
    requested_maps,
    episodes_per_map,
    round_def,
    result_path,
    done_marker,
    elapsed_seconds,
    logger,
):
    manifest = _read_json(runtime_dir / "manifest.json")
    total_episodes = int(manifest.get("total_episodes", 0) or 0)
    deadline = time.time() + _env_int("KAIWU_BENCHMARK_MAX_WAIT_SECONDS", 3600)
    completed_dir = runtime_dir / "tasks" / "completed"
    while time.time() <= deadline:
        completed_count = _count_json_files(completed_dir)
        if completed_count >= total_episodes:
            break
        logger.info(f"[HOLDOUT-DYN] coordinator waiting completed={completed_count}/{total_episodes}")
        time.sleep(5.0)

    completed_count = _count_json_files(completed_dir)
    if completed_count < total_episodes:
        raise TimeoutError(f"Dynamic benchmark timed out completed={completed_count}/{total_episodes}")

    episode_results = []
    for completed_path in sorted(completed_dir.glob("*.json")):
        payload = _read_json(completed_path)
        episode_results.append(payload["episode_result"])
    episode_results.sort(key=lambda item: (int(item.get("map_id", 0)), int(item.get("ep_idx", 0))))
    aggregated = _aggregate_results(episode_results, requested_maps)
    workers = [_read_json(path) for path in sorted((runtime_dir / "workers").glob("*.json"))]
    execution = deepcopy(manifest.get("execution") or {})
    execution.update(
        {
            "mode": "dynamic",
            "completed_task_count": len(episode_results),
            "total_episodes": total_episodes,
            "observed_worker_count": len(workers),
            "workers": workers,
        }
    )
    snapshot = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": session_id,
        "checkpoint": checkpoint,
        "elapsed_seconds": elapsed_seconds,
        "contract": {
            "schema_version": SCHEMA_VERSION,
            "maps": requested_maps,
            "episodes_per_map": episodes_per_map,
            "round_def": round_def,
            "fixed_config": {
                "map_random": False,
                "max_step": round_def["max_step"],
                "battery_max": round_def["battery_max"],
                "robot_count": round_def["robot_count"],
                "charger_count": round_def["charger_count"],
            },
            "checkpoint": checkpoint,
        },
        "round_def": round_def,
        "maps": requested_maps,
        "episodes_per_map": episodes_per_map,
        "overall": aggregated["overall"],
        "per_map": aggregated["per_map"],
        "episodes": episode_results,
        "execution": execution,
    }
    _atomic_write_json(runtime_dir / "result.json", snapshot)
    _atomic_write_json(result_path, snapshot)
    _ = done_marker.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "checkpoint": checkpoint,
                "overall": aggregated["overall"],
                "execution": execution,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    _print_summary(aggregated, logger, elapsed_seconds)
    return snapshot


def _wait_for_dynamic_done(done_marker, logger):
    deadline = time.time() + _env_int("KAIWU_BENCHMARK_MAX_WAIT_SECONDS", 3600)
    while time.time() <= deadline:
        if done_marker.is_file():
            try:
                return json.loads(Path("/workspace/code/holdout_result.json").read_text(encoding="utf-8"))
            except Exception:
                return {"status": "done", "done_marker": str(done_marker)}
        time.sleep(2.0)
    raise TimeoutError(f"Dynamic benchmark done marker not found: {done_marker}")


def _normalize_process_index(process_index):
    try:
        return max(int(process_index), 0)
    except (TypeError, ValueError):
        return 0


def _count_json_files(path):
    return len(list(path.glob("*.json"))) if path.exists() else 0


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def _as_list(value, fallback):
    if value is None:
        return [fallback]
    if isinstance(value, (list, tuple)):
        return list(value) or [fallback]
    return [value]


def _build_worker_pairs(primary_env, primary_agent, envs, agents, requested_workers, episode_count):
    env_list = _as_list(envs, primary_env)
    agent_list = _as_list(agents, primary_agent)
    max_workers = min(
        max(1, int(requested_workers)),
        len(env_list),
        len(agent_list),
        max(1, int(episode_count)),
    )
    if max_workers <= 1:
        reason = None
        if int(requested_workers) > 1:
            reason = (
                f"requested={requested_workers}, env_count={len(env_list)}, "
                f"agent_count={len(agent_list)}, episode_count={episode_count}"
            )
        return [(primary_env, primary_agent)], reason

    pairs = [(env_list[index], agent_list[index]) for index in range(max_workers)]
    return pairs, None


def _partition_indexed_assignments(indexed_assignments, worker_count):
    batches = [[] for _ in range(worker_count)]
    for position, item in enumerate(indexed_assignments):
        batches[position % worker_count].append(item)
    return batches


def _run_assigned_episode(
    assignment,
    env,
    agent,
    base_env_conf,
    usr_conf,
    idx,
    total,
    session_dir,
    logger,
    step_log,
    round_def,
    worker_index,
):
    map_id = int(assignment["map_id"])
    ep_idx = int(assignment["ep_idx"])
    env_conf = deepcopy(base_env_conf)
    env_conf["map"] = [map_id]
    env_conf["map_random"] = False
    env_conf["robot_count"] = round_def["robot_count"]
    env_conf["charger_count"] = round_def["charger_count"]
    env_conf["max_step"] = round_def["max_step"]
    env_conf["battery_max"] = round_def["battery_max"]

    wrapped_conf = _wrap_env_conf(usr_conf, env_conf)
    result = _run_eval_episode(
        env=env,
        agent=agent,
        usr_conf=wrapped_conf,
        map_id=map_id,
        ep_idx=ep_idx,
        idx=idx,
        total=total,
        session_dir=session_dir,
        logger=logger,
        step_log=step_log,
        round_def=round_def,
    )
    result["worker_index"] = worker_index
    return result


def _run_assignment_batch(
    batch,
    env,
    agent,
    base_env_conf,
    usr_conf,
    total,
    session_dir,
    logger,
    step_log,
    round_def,
    worker_index,
):
    results = []
    for idx, assignment in batch:
        results.append(
            _run_assigned_episode(
                assignment=assignment,
                env=env,
                agent=agent,
                base_env_conf=base_env_conf,
                usr_conf=usr_conf,
                idx=idx,
                total=total,
                session_dir=session_dir,
                logger=logger,
                step_log=step_log,
                round_def=round_def,
                worker_index=worker_index,
            )
        )
    return results


def _run_eval_episode(
    env, agent, usr_conf, map_id, ep_idx, idx, total,
    session_dir, logger, step_log, round_def,
):
    ep_label = f"[{idx}/{total}] map{map_id}/ep{ep_idx}"
    logger.info(
        f"[HOLDOUT-BENCH] {ep_label} START | "
        f"robots={round_def['robot_count']} chargers={round_def['charger_count']} "
        f"steps={round_def['max_step']} battery={round_def['battery_max']}"
    )

    # Inject runtime metadata before reset
    env_obs = _inject_agent_runtime(env.reset(usr_conf), agent)

    # Check for disaster recovery
    try:
        from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
        if handle_disaster_recovery(env_obs, logger):
            logger.warning(f"[HOLDOUT-BENCH] {ep_label} SKIP (disaster recovery)")
            return _empty_episode_result(map_id, ep_idx, "error")
    except ImportError:
        pass

    agent.reset(env_obs)
    obs_data, _ = agent.observation_process(env_obs)

    # Set episode config on preprocessor/planner
    agent.set_episode_config(
        max_step=round_def["max_step"],
        robot_count=round_def["robot_count"],
        charger_count=round_def["charger_count"],
        battery_max=round_def["battery_max"],
    )

    # Per-episode JSONL log
    ep_log_path = session_dir / "episodes" / f"map{map_id}_ep{ep_idx:02d}.jsonl"
    ep_log_path.parent.mkdir(parents=True, exist_ok=True)
    ep_log_file = open(ep_log_path, "w", encoding="utf-8")

    fm = agent.preprocessor
    step_records = []
    done = False
    step = 0
    total_reward = 0.0
    terminated = False
    truncated = False

    while not done:
        last_action_before = agent.last_action
        # Use guided_predict for planner-guided inference (exploit mode)
        policy_info = agent.planner.update(env_obs, agent.last_action)
        act_data = agent.guided_predict(
            [obs_data],
            policy_info=policy_info,
            residual_alpha=Config.RESIDUAL_ALPHA_MAX,
        )[0]
        act = agent.action_process(act_data, is_stochastic=False)
        selected_action = int(act)

        decision_context, decision_missing = _extract_decision_context(
            fm=fm,
            obs_data=obs_data,
            policy_info=policy_info,
            act_data=act_data,
            selected_action=selected_action,
            step=step,
            pos_before=getattr(fm, "cur_pos", None),
            last_action=last_action_before,
        )

        _, env_obs = env.step(act)
        env_obs = _inject_agent_runtime(env_obs, agent)

        step += 1

        # Check disaster recovery mid-episode
        recovery_hit = False
        try:
            from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
            if handle_disaster_recovery(env_obs, logger):
                recovery_hit = True
                terminated = bool(env_obs.get("terminated", False))
                truncated = bool(env_obs.get("truncated", False))
        except ImportError:
            terminated = bool(env_obs.get("terminated", False))
            truncated = bool(env_obs.get("truncated", False))

        if not terminated and not truncated:
            terminated = bool(env_obs.get("terminated", False))
            truncated = bool(env_obs.get("truncated", False))

        done = recovery_hit or terminated or truncated

        try:
            obs_data, _ = agent.observation_process(env_obs)
        except Exception as exc:
            logger.warning(f"[HOLDOUT-BENCH] {ep_label} observation_process failed: {exc}")

        reward, reward_components = _extract_reward_value(getattr(agent, "last_reward", 0.0))
        total_reward += reward

        outcome_state, outcome_missing = _extract_outcome_state(
            fm=fm,
            env_obs=env_obs,
            obs_data=obs_data,
            policy_info=policy_info,
            act_data=act_data,
            selected_action=selected_action,
            reward=reward,
            total_reward=total_reward,
            terminated=terminated,
            truncated=truncated,
            decision_context=decision_context,
        )

        step_missing_signals = _merge_missing_signals(decision_missing, outcome_missing)
        charger_slack = _safe_float(outcome_state.get("charger_slack"), None)
        nearest_charger_dist = _safe_float(outcome_state.get("nearest_charger_dist"), None)
        nearest_npc_dist = _safe_float(outcome_state.get("nearest_npc_dist"), None)

        # Build step record
        step_rec = {
            "step": step,
            "action": selected_action,
            "reward": round(reward, 4),
            "reward_components": reward_components,
            "total_reward": round(total_reward, 4),
            "battery": _safe_int(outcome_state.get("battery"), 0),
            "battery_max": _safe_int(outcome_state.get("battery_max"), 0),
            "dirt_cleaned": _safe_int(outcome_state.get("dirt_cleaned"), 0),
            "total_dirt": _safe_int(outcome_state.get("total_dirt"), 0),
            "mode": _safe_int(outcome_state.get("current_mode"), -1),
            "charger_slack": round(charger_slack, 2) if charger_slack is not None else None,
            "nearest_charger_dist": round(nearest_charger_dist, 1) if nearest_charger_dist is not None else None,
            "nearest_npc_dist": round(nearest_npc_dist, 1) if nearest_npc_dist is not None else None,
            "invalid_move_count": _safe_int(outcome_state.get("invalid_move_count"), 0),
            "is_diag_action": 1.0 if selected_action in (1, 3, 5, 7) else 0.0,
            # Guidance/planner info
            "guidance": _extract_guidance(policy_info, act_data),
            "decision_context": decision_context,
            "outcome_state": outcome_state,
            "missing_signals": step_missing_signals,
        }

        step_records.append(step_rec)
        ep_log_file.write(json.dumps(step_rec, ensure_ascii=False) + "\n")

        if step % 100 == 0 or done:
            step_log.info(
                f"{ep_label} step={step} bat={step_rec['battery']}/{step_rec['battery_max']} "
                f"dirt={step_rec['dirt_cleaned']}/{step_rec['total_dirt']} "
                f"mode={step_rec['mode']} act={selected_action} reward={reward:.3f}"
            )

    ep_log_file.close()

    # Extract episode-level result
    observation = env_obs.get("observation") or {}
    env_info = observation.get("env_info") or {}
    hero = (observation.get("frame_state") or {}).get("heroes") or {}
    extra_info = env_obs.get("extra_info") or observation.get("extra_info") or {}

    fail_reason = infer_fail_reason(
        terminated=terminated,
        truncated=truncated,
        battery=hero.get("battery"),
        extra_info=extra_info,
    )
    clean_score = float(env_info.get("clean_score", 0))
    finished_steps = float(env_info.get("finished_steps", step))
    charge_count = float(env_info.get("charge_count", 0))
    remaining_charge = float(env_info.get("remaining_charge", hero.get("battery", 0)))

    episode_id = f"map{map_id}_ep{ep_idx:02d}"
    result = {
        "episode_id": episode_id,
        "map_id": map_id,
        "ep_idx": ep_idx,
        "result": fail_reason,
        "fail_reason": fail_reason,
        "done_reason": fail_reason,
        "status": "completed" if fail_reason == "completed" else "failed",
        "clean_score": clean_score,
        "steps": finished_steps,
        "finished_steps": finished_steps,
        "max_step": round_def["max_step"],
        "battery_max": round_def["battery_max"],
        "robot_count": round_def["robot_count"],
        "charger_count": round_def["charger_count"],
        "charge_count": charge_count,
        "remaining_charge": remaining_charge,
        "total_reward": round(total_reward, 3),
        "dirt_cleaned": int(getattr(fm, "dirt_cleaned", 0)),
        "total_dirt": int(getattr(fm, "total_dirt", 0)),
        "dirt_ratio": round(
            getattr(fm, "dirt_cleaned", 0) / max(getattr(fm, "total_dirt", 1), 1), 4
        ),
        "invalid_move_count": int(getattr(fm, "invalid_move_count", 0)),
        "invalid_move_rate": round(
            getattr(fm, "invalid_move_count", 0) / max(step, 1), 4
        ),
        "step_log": str(ep_log_path.relative_to(session_dir)),
        "field_availability": _build_field_availability(step_records),
        "missing_signals": sorted({signal for rec in step_records for signal in rec.get("missing_signals", [])}),
    }
    result.update(_build_episode_diagnostics(step_records, total_reward=total_reward))

    result_str = "COMPLETED" if fail_reason == "completed" else f"FAIL({fail_reason})"
    logger.info(
        f"[HOLDOUT-BENCH] {ep_label} {result_str} | score={clean_score:.0f} "
        f"dirt={result['dirt_cleaned']}/{result['total_dirt']} steps={int(finished_steps)} "
        f"charges={int(charge_count)} bat_left={int(remaining_charge)} "
        f"reward={total_reward:.1f}"
    )
    return result


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default=None):
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _first_value(value, default=None):
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return value.reshape(-1)[0]
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return default
        return value[0]
    return value


def _count_positive_mask(values):
    if values is None:
        return 0
    try:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    except Exception:
        return 0
    if arr.size == 0:
        return 0
    return int(np.count_nonzero(arr > 0.0))


def _topk_summary(values, k=3):
    if values is None:
        return []
    try:
        arr = np.asarray(values, dtype=np.float32).reshape(-1)
    except Exception:
        return []
    if arr.size == 0:
        return []
    limit = min(int(k), int(arr.size))
    indices = np.argsort(-arr)[:limit]
    return [
        {"action": int(idx), "prob": round(float(arr[idx]), 4)}
        for idx in indices
    ]


def _top1_summary(values):
    topk = _topk_summary(values, k=1)
    return topk[0] if topk else None


def _extract_reward_value(raw_reward):
    if isinstance(raw_reward, dict):
        components = {}
        total = 0.0
        for key, value in raw_reward.items():
            numeric = _safe_float(value, None)
            if numeric is None:
                continue
            components[str(key)] = round(numeric, 4)
            total += numeric
        return round(total, 4), (components or None)
    return _safe_float(raw_reward, 0.0) or 0.0, None


def _extract_policy_info(policy_info):
    missing = []
    info = {
        "target_mode": getattr(policy_info, "target_mode", None),
        "target_pos": list(getattr(policy_info, "target_pos", [])) if getattr(policy_info, "target_pos", None) is not None else None,
        "target_distance": _safe_float(getattr(policy_info, "target_distance", None)),
        "battery": _safe_float(getattr(policy_info, "battery", None)),
        "battery_ratio": _safe_float(getattr(policy_info, "battery_ratio", None)),
        "charger_distance": _safe_float(getattr(policy_info, "charger_distance", None)),
        "charger_slack": _safe_float(getattr(policy_info, "charger_slack", None)),
        "nearest_npc_distance": _safe_float(getattr(policy_info, "nearest_npc_distance", None)),
        "frontier_density": _safe_float(getattr(policy_info, "frontier_density", None)),
        "local_dirty_ratio": _safe_float(getattr(policy_info, "local_dirty_ratio", None)),
        "local_unknown_ratio": _safe_float(getattr(policy_info, "local_unknown_ratio", None)),
        "new_known_cells": _safe_int(getattr(policy_info, "new_known_cells", None)),
        "on_charger": bool(getattr(policy_info, "on_charger", False)),
        "should_charge": bool(getattr(policy_info, "should_charge", False)),
        "policy_top1": _top1_summary(getattr(policy_info, "action_scores", None)),
        "policy_topk": _topk_summary(getattr(policy_info, "action_scores", None), k=3),
    }
    if info["policy_top1"] is None:
        missing.append("policy_info.action_scores")
    if info["target_mode"] is None:
        missing.append("policy_info.target_mode")
    return info, missing


def _extract_action_diagnostics(act_data):
    missing = []
    policy_top1 = _top1_summary(getattr(act_data, "policy_prob", None))
    planner_top1 = _top1_summary(getattr(act_data, "planner_prob", None))
    action_prob = getattr(act_data, "prob", None)
    diagnostics = {
        "selected_action": _safe_int(_first_value(getattr(act_data, "d_action", None), None)),
        "model_action": _safe_int(_first_value(getattr(act_data, "action", None), None)),
        "action_entropy": _action_entropy(action_prob),
        "policy_top1": policy_top1,
        "planner_top1": planner_top1,
        "policy_topk": _topk_summary(getattr(act_data, "policy_prob", None), k=3),
        "planner_topk": _topk_summary(getattr(act_data, "planner_prob", None), k=3),
        "mix_alpha": _safe_float(_first_value(getattr(act_data, "mix_alpha", None), None)),
        "action_mask_valid_count": _safe_int(_count_positive_mask(getattr(act_data, "action_mask", None)), 0),
    }
    if policy_top1 is None:
        missing.append("act_data.policy_prob")
    if planner_top1 is None:
        missing.append("act_data.planner_prob")
    return diagnostics, missing


def _extract_preprocessor_state(fm):
    missing = []

    def capture(name, attr_name=None, *, default=None, caster=None):
        source_name = attr_name or name
        value = getattr(fm, source_name, None)
        if value is None:
            missing.append(f"preprocessor.{source_name}")
            return default
        if caster is not None:
            try:
                return caster(value)
            except Exception:
                missing.append(f"preprocessor.{source_name}")
                return default
        return value

    cur_pos = capture("cur_pos", default=None, caster=lambda v: list(v))
    state = {
        "step_no": _safe_int(capture("step_no", default=None), None),
        "battery": _safe_int(capture("battery", default=None), None),
        "battery_max": _safe_int(capture("battery_max", default=None), None),
        "prev_battery": _safe_int(capture("prev_battery", default=None), None),
        "dirt_cleaned": _safe_int(capture("dirt_cleaned", default=None), None),
        "total_dirt": _safe_int(capture("total_dirt", default=None), None),
        "current_mode": _safe_int(capture("current_mode", default=None), None),
        "cur_pos": cur_pos,
        "new_observed_cells": _safe_int(capture("new_observed_cells", default=None), None),
        "cur_revisit_count": _safe_int(capture("cur_revisit_count", default=None), None),
        "nearest_charger_dist": _safe_float(getattr(fm, "_nearest_charger_distance", lambda: None)() if hasattr(fm, "_nearest_charger_distance") else None),
        "nearest_npc_dist": _safe_float(_derive_nearest_npc_dist(fm)),
        "nearest_unarrived_charger_dist": _safe_float(getattr(fm, "_nearest_unarrived_charger_dist", None)),
        "observed_cells_count": _safe_int(int(np.count_nonzero(getattr(fm, "observed_map", []))) if getattr(fm, "observed_map", None) is not None else None, None),
        "observed_total_cells": _safe_int(int(np.size(getattr(fm, "observed_map", []))) if getattr(fm, "observed_map", None) is not None else None, None),
        "known_charger_count": _safe_int(len(getattr(fm, "charger_positions", []) or []), None),
        "charger_arrived_count": _safe_int(len(getattr(fm, "charger_arrival_steps", {}) or {}), None),
        "charger_arrival_steps": sorted(
            int(step_no)
            for step_no in (getattr(fm, "charger_arrival_steps", {}) or {}).values()
            if _safe_int(step_no) is not None
        ),
    }
    if state["cur_pos"] is None:
        missing.append("preprocessor.cur_pos")
    if state["current_mode"] is None:
        missing.append("preprocessor.current_mode")
    return state, missing


def _derive_nearest_npc_dist(fm):
    cur_pos = getattr(fm, "cur_pos", None)
    npc_positions = getattr(fm, "npc_positions", None)
    if cur_pos is None or not npc_positions:
        return None
    try:
        hx, hz = cur_pos
        return min(float(np.sqrt((nx - hx) ** 2 + (nz - hz) ** 2)) for nx, nz in npc_positions)
    except Exception:
        return None


def _extract_decision_context(fm, obs_data, policy_info, act_data, selected_action, step=None, pos_before=None, last_action=None):
    preprocessor_state, pre_missing = _extract_preprocessor_state(fm)
    policy_snapshot, policy_missing = _extract_policy_info(policy_info)
    action_diag, action_missing = _extract_action_diagnostics(act_data)
    missing = _merge_missing_signals(pre_missing, policy_missing, action_missing)
    legal_action_count = _safe_int(_count_positive_mask(getattr(obs_data, "legal_action", None)), 0)
    safe_action_count = _safe_int(_count_positive_mask(getattr(policy_info, "safe_action_mask", None)), legal_action_count)
    context = {
        "step": _safe_int(step, None),
        "pos_before": list(pos_before) if pos_before is not None else preprocessor_state.get("cur_pos"),
        "last_action": _safe_int(last_action, None),
        "selected_action": _safe_int(selected_action),
        "chosen_action": _safe_int(getattr(policy_info, "chosen_action", None), _safe_int(selected_action)),
        "policy_action": _safe_int(_first_value(getattr(act_data, "action", None), None), _safe_int(selected_action)),
        "greedy_action": _safe_int(getattr(policy_info, "greedy_action", None), _safe_int(selected_action)),
        "mix_alpha": _safe_float(_first_value(getattr(act_data, "mix_alpha", None), None)),
        "planner_match": bool(_safe_int(selected_action) == _safe_int(getattr(policy_info, "chosen_action", None), _safe_int(selected_action))),
        "action_entropy": action_diag.get("action_entropy"),
        "legal_action_count": legal_action_count,
        "safe_action_count": safe_action_count,
        "target_mode": policy_snapshot.get("target_mode"),
        "should_charge": policy_snapshot.get("should_charge"),
        "charger_distance": policy_snapshot.get("charger_distance"),
        "charger_slack": policy_snapshot.get("charger_slack"),
        "battery": policy_snapshot.get("battery"),
        "battery_ratio": policy_snapshot.get("battery_ratio"),
        "on_charger": policy_snapshot.get("on_charger"),
        "nearest_npc_distance": policy_snapshot.get("nearest_npc_distance"),
        "frontier_density": policy_snapshot.get("frontier_density"),
        "local_unknown_ratio": policy_snapshot.get("local_unknown_ratio"),
        "local_dirty_ratio": policy_snapshot.get("local_dirty_ratio"),
        "new_known_cells": policy_snapshot.get("new_known_cells"),
        "preprocessor_state": preprocessor_state,
        "policy_snapshot": policy_snapshot,
        "action_diagnostics": action_diag,
        "policy_top1": policy_snapshot.get("policy_top1"),
        "planner_top1": action_diag.get("planner_top1"),
    }
    context["missing_signals"] = list(missing)
    return context, missing


def _extract_outcome_state(fm, env_obs, obs_data, policy_info, act_data, selected_action, reward, total_reward, terminated, truncated, decision_context):
    preprocessor_state, pre_missing = _extract_preprocessor_state(fm)
    policy_snapshot, policy_missing = _extract_policy_info(policy_info)
    action_diag, action_missing = _extract_action_diagnostics(act_data)
    missing = _merge_missing_signals(pre_missing, policy_missing, action_missing)

    pre_battery = _safe_float((decision_context or {}).get("preprocessor_state", {}).get("battery"), None)
    post_battery = _safe_float(preprocessor_state.get("battery"), None)
    cleaned_delta = None
    prev_cleaned = _safe_int((decision_context or {}).get("preprocessor_state", {}).get("dirt_cleaned"), None)
    post_cleaned = _safe_int(preprocessor_state.get("dirt_cleaned"), None)
    if prev_cleaned is not None and post_cleaned is not None:
        cleaned_delta = post_cleaned - prev_cleaned
    battery_delta = None
    if pre_battery is not None and post_battery is not None:
        battery_delta = round(post_battery - pre_battery, 4)
    pos_after = preprocessor_state.get("cur_pos")
    done = bool(terminated or truncated)

    outcome = {
        "selected_action": _safe_int(selected_action),
        "reward": round(_safe_float(reward, 0.0) or 0.0, 4),
        "total_reward": round(_safe_float(total_reward, 0.0) or 0.0, 4),
        "done": done,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "battery": preprocessor_state.get("battery"),
        "battery_max": preprocessor_state.get("battery_max"),
        "battery_delta": battery_delta,
        "pos_after": pos_after,
        "cleaned_delta": cleaned_delta,
        "dirt_cleaned": preprocessor_state.get("dirt_cleaned"),
        "total_dirt": preprocessor_state.get("total_dirt"),
        "cur_pos": preprocessor_state.get("cur_pos"),
        "step_no": preprocessor_state.get("step_no"),
        "cur_revisit_count": preprocessor_state.get("cur_revisit_count"),
        "observed_cells_count": preprocessor_state.get("observed_cells_count"),
        "observed_total_cells": preprocessor_state.get("observed_total_cells"),
        "known_charger_count": preprocessor_state.get("known_charger_count"),
        "charger_arrived_count": preprocessor_state.get("charger_arrived_count"),
        "charger_arrival_steps": preprocessor_state.get("charger_arrival_steps"),
        "nearest_charger_dist": policy_snapshot.get("charger_distance"),
        "charger_slack": policy_snapshot.get("charger_slack"),
        "nearest_npc_dist": policy_snapshot.get("nearest_npc_distance"),
        "invalid_move_count": _safe_int(getattr(fm, "invalid_move_count", None), None),
        "should_charge": policy_snapshot.get("should_charge"),
        "target_mode": policy_snapshot.get("target_mode"),
        "on_charger": policy_snapshot.get("on_charger"),
        "is_diag_action": 1.0 if selected_action in (1, 3, 5, 7) else 0.0,
        "policy_top1": policy_snapshot.get("policy_top1"),
        "planner_top1": action_diag.get("planner_top1"),
        "policy_topk": policy_snapshot.get("policy_topk"),
        "planner_topk": action_diag.get("planner_topk"),
        "post_observation_state": preprocessor_state,
    }
    if battery_delta is None:
        missing.append("battery_delta")
    if cleaned_delta is None:
        missing.append("cleaned_delta")
    if outcome["nearest_npc_dist"] is None:
        missing.append("policy_info.nearest_npc_distance")
    if outcome["nearest_charger_dist"] is None:
        missing.append("policy_info.charger_distance")
    if outcome["invalid_move_count"] is None:
        missing.append("preprocessor.invalid_move_count")
    outcome["missing_signals"] = list(missing)
    return outcome, missing


def _merge_missing_signals(*groups):
    merged = []
    seen = set()
    for group in groups:
        if not group:
            continue
        for item in group:
            if item in seen:
                continue
            seen.add(item)
            merged.append(item)
    return merged


def _action_entropy(probs):
    try:
        arr = np.asarray(probs, dtype=np.float64).reshape(-1)
    except Exception:
        return None
    if arr.size == 0:
        return None
    total = float(np.sum(arr))
    if total <= 0.0:
        return None
    arr = arr / total
    arr = arr[arr > 0.0]
    if arr.size == 0:
        return None
    return round(float(-np.sum(arr * np.log(arr))), 4)


def _build_field_availability(step_records):
    fields = ["decision_context", "outcome_state", "missing_signals", "policy_top1", "planner_top1", "battery_delta"]
    availability = {field: False for field in fields}
    for rec in step_records:
        for field in fields:
            if rec.get(field) is not None:
                availability[field] = True
    return availability


def _summarize_rate(numerator, denominator):
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 4)


def _build_reward_attribution_lite(step_records, total_reward):
    component_totals = {}
    component_step_count = 0
    for step_rec in step_records:
        reward_components = step_rec.get("reward_components")
        if not isinstance(reward_components, dict) or not reward_components:
            continue
        component_step_count += 1
        for key, value in reward_components.items():
            numeric = _safe_float(value, None)
            if numeric is None:
                continue
            component_totals[str(key)] = component_totals.get(str(key), 0.0) + numeric
    if not component_totals:
        return {
            "available": False,
            "reason": "scalar_reward_only",
            "total_reward": round(_safe_float(total_reward, 0.0) or 0.0, 4),
        }
    sorted_components = sorted(
        component_totals.items(),
        key=lambda item: (-abs(item[1]), item[0]),
    )
    return {
        "available": True,
        "steps_with_components": component_step_count,
        "component_totals": {
            key: round(value, 4)
            for key, value in sorted_components[:8]
        },
        "total_reward": round(_safe_float(total_reward, 0.0) or 0.0, 4),
    }


def _build_anomaly_summary_lite(step_records):
    if not step_records:
        return {
            "available": False,
            "reason": "no_step_records",
            "low_slack_rate": None,
            "no_clean_step_rate": None,
            "revisit_rate": None,
            "planner_policy_mismatch_rate": None,
            "npc_near_rate": None,
            "loop_suspect_rate": None,
            "observed_step_counts": {},
            "missing_signals": [
                "charger_slack",
                "cleaned_delta",
                "cur_revisit_count",
                "planner_match",
                "nearest_npc_dist",
            ],
        }

    low_slack_hits = 0
    low_slack_seen = 0
    no_clean_hits = 0
    no_clean_seen = 0
    revisit_hits = 0
    revisit_seen = 0
    mismatch_hits = 0
    mismatch_seen = 0
    npc_near_hits = 0
    npc_near_seen = 0
    loop_hits = 0
    loop_seen = 0
    current_action_streak = 0
    previous_action = None
    missing_signals = set()

    for step_rec in step_records:
        action = _safe_int(step_rec.get("action"), None)
        if action is not None and previous_action == action:
            current_action_streak += 1
        elif action is not None:
            current_action_streak = 1
            previous_action = action

        decision = step_rec.get("decision_context") or {}
        outcome = step_rec.get("outcome_state") or {}
        post_state = outcome.get("post_observation_state") or {}

        charger_slack = _safe_float(step_rec.get("charger_slack"), None)
        if charger_slack is None:
            missing_signals.add("charger_slack")
        else:
            low_slack_seen += 1
            if charger_slack <= LOW_SLACK_THRESHOLD:
                low_slack_hits += 1

        cleaned_delta = _safe_float(outcome.get("cleaned_delta"), None)
        if cleaned_delta is None:
            missing_signals.add("cleaned_delta")
        else:
            no_clean_seen += 1
            if cleaned_delta <= 0.0:
                no_clean_hits += 1

        revisit_count = _safe_int(post_state.get("cur_revisit_count"), None)
        if revisit_count is None:
            missing_signals.add("cur_revisit_count")
        else:
            revisit_seen += 1
            if revisit_count > 0:
                revisit_hits += 1

        planner_match = decision.get("planner_match")
        if planner_match is None:
            missing_signals.add("planner_match")
        else:
            mismatch_seen += 1
            if not bool(planner_match):
                mismatch_hits += 1

        nearest_npc_dist = _safe_float(step_rec.get("nearest_npc_dist"), None)
        if nearest_npc_dist is None:
            missing_signals.add("nearest_npc_dist")
        else:
            npc_near_seen += 1
            if nearest_npc_dist <= NPC_NEAR_DISTANCE:
                npc_near_hits += 1

        if action is None and revisit_count is None:
            missing_signals.add("loop_suspect_inputs")
        else:
            loop_seen += 1
            if (revisit_count is not None and revisit_count >= 3) or current_action_streak >= 4:
                loop_hits += 1

    observed_step_counts = {
        "low_slack_rate": low_slack_seen,
        "no_clean_step_rate": no_clean_seen,
        "revisit_rate": revisit_seen,
        "planner_policy_mismatch_rate": mismatch_seen,
        "npc_near_rate": npc_near_seen,
        "loop_suspect_rate": loop_seen,
    }
    return {
        "available": any(count > 0 for count in observed_step_counts.values()),
        "low_slack_rate": _summarize_rate(low_slack_hits, low_slack_seen),
        "no_clean_step_rate": _summarize_rate(no_clean_hits, no_clean_seen),
        "revisit_rate": _summarize_rate(revisit_hits, revisit_seen),
        "planner_policy_mismatch_rate": _summarize_rate(mismatch_hits, mismatch_seen),
        "npc_near_rate": _summarize_rate(npc_near_hits, npc_near_seen),
        "loop_suspect_rate": _summarize_rate(loop_hits, loop_seen),
        "observed_step_counts": observed_step_counts,
        "missing_signals": sorted(missing_signals),
    }


def _compact_step_row(step_rec):
    decision = step_rec.get("decision_context") or {}
    outcome = step_rec.get("outcome_state") or {}
    post_state = outcome.get("post_observation_state") or {}
    return {
        "step": _safe_int(step_rec.get("step"), 0),
        "action": _safe_int(step_rec.get("action"), -1),
        "battery": _safe_int(step_rec.get("battery"), None),
        "charger_slack": _safe_float(step_rec.get("charger_slack"), None),
        "target_mode": decision.get("target_mode") or outcome.get("target_mode"),
        "should_charge": bool(decision.get("should_charge", False)),
        "on_charger": bool(outcome.get("on_charger", False)),
        "known_charger_count": _safe_int(outcome.get("known_charger_count"), None),
        "charger_arrived_count": _safe_int(outcome.get("charger_arrived_count"), None),
        "invalid_move_count": _safe_int(step_rec.get("invalid_move_count"), None),
        "cur_revisit_count": _safe_int(post_state.get("cur_revisit_count"), None),
        "cur_pos": post_state.get("cur_pos") or outcome.get("cur_pos"),
    }


def _window_rows(step_records, center_step, radius=EVIDENCE_WINDOW_RADIUS):
    if center_step is None:
        return []
    start_step = max(1, int(center_step) - int(radius))
    end_step = int(center_step) + int(radius)
    return [
        _compact_step_row(rec)
        for rec in step_records
        if start_step <= _safe_int(rec.get("step"), 0) <= end_step
    ]


def _build_episode_diagnostics(step_records, total_reward=0.0):
    if not step_records:
        return {
            "charger_known_first_step": None,
            "charger_known_final": None,
            "known_charger_count_final": 0,
            "charger_arrived_count": 0,
            "charger_first_arrival_step": None,
            "charger_arrival_steps": [],
            "first_should_charge_step": None,
            "attempted_charge_step_count": 0,
            "first_return_mode_step": None,
            "min_battery": None,
            "min_battery_step": None,
            "min_charger_slack": None,
            "max_negative_charger_slack": 0.0,
            "action_histogram": {},
            "last_actions": [],
            "repeat_action_max_streak": 0,
            "revisit_ratio": 0.0,
            "max_revisit_count": 0,
            "unique_cells_visited": 0,
            "observed_ratio_final": None,
            "final_window": [],
            "evidence_windows": {
                "first_low_slack_window": [],
                "first_should_charge_window": [],
                "first_missed_charge_window": [],
                "first_loop_window": [],
                "last_failure_window": [],
            },
            "reward_attribution_lite": {
                "available": False,
                "reason": "scalar_reward_only",
                "total_reward": round(_safe_float(total_reward, 0.0) or 0.0, 4),
            },
            "anomaly_summary_lite": _build_anomaly_summary_lite(step_records),
        }

    action_histogram = {}
    last_actions = []
    repeat_action_max_streak = 0
    current_streak = 0
    previous_action = None
    revisit_steps = 0
    max_revisit_count = 0
    unique_cells = set()
    first_loop_step = None
    first_should_charge_step = None
    first_return_mode_step = None
    first_missed_charge_step = None
    first_low_slack_step = None
    charger_known_first_step = None
    min_battery = None
    min_battery_step = None
    min_charger_slack = None
    known_charger_count_final = None
    charger_arrival_steps = []
    observed_ratio_final = None
    attempted_charge_step_count = 0

    for step_rec in step_records:
        step_no = _safe_int(step_rec.get("step"), 0)
        action = _safe_int(step_rec.get("action"), -1)
        battery = _safe_int(step_rec.get("battery"), None)
        charger_slack = _safe_float(step_rec.get("charger_slack"), None)
        decision = step_rec.get("decision_context") or {}
        outcome = step_rec.get("outcome_state") or {}
        post_state = outcome.get("post_observation_state") or {}
        known_charger_count = _safe_int(outcome.get("known_charger_count"), None)
        charger_known = None if known_charger_count is None else known_charger_count > 0
        should_charge = bool(decision.get("should_charge", False))
        target_mode = str(decision.get("target_mode") or outcome.get("target_mode") or "").lower()
        on_charger = bool(outcome.get("on_charger", False))
        revisit_count = _safe_int(post_state.get("cur_revisit_count"), 0) or 0

        action_histogram[str(action)] = action_histogram.get(str(action), 0) + 1
        last_actions.append(action)
        if len(last_actions) > 10:
            last_actions = last_actions[-10:]

        if previous_action == action:
            current_streak += 1
        else:
            current_streak = 1
            previous_action = action
        repeat_action_max_streak = max(repeat_action_max_streak, current_streak)

        if revisit_count > 0:
            revisit_steps += 1
        max_revisit_count = max(max_revisit_count, revisit_count)
        if first_loop_step is None and (revisit_count >= 3 or current_streak >= 4):
            first_loop_step = step_no

        cur_pos = post_state.get("cur_pos") or outcome.get("cur_pos")
        if isinstance(cur_pos, (list, tuple)) and len(cur_pos) >= 2:
            unique_cells.add((int(cur_pos[0]), int(cur_pos[1])))

        if charger_known_first_step is None and charger_known:
            charger_known_first_step = step_no
        if first_should_charge_step is None and should_charge:
            first_should_charge_step = step_no
        if target_mode == "charge":
            attempted_charge_step_count += 1
            if first_return_mode_step is None:
                first_return_mode_step = step_no
        if first_missed_charge_step is None and should_charge and target_mode != "charge" and not on_charger:
            first_missed_charge_step = step_no

        if battery is not None and (min_battery is None or battery < min_battery):
            min_battery = battery
            min_battery_step = step_no
        if charger_slack is not None:
            if min_charger_slack is None or charger_slack < min_charger_slack:
                min_charger_slack = charger_slack
            if first_low_slack_step is None and charger_slack <= -3.0:
                first_low_slack_step = step_no

        known_charger_count_final = known_charger_count
        raw_arrival_steps = outcome.get("charger_arrival_steps")
        if isinstance(raw_arrival_steps, list):
            charger_arrival_steps = [
                int(value)
                for value in raw_arrival_steps
                if _safe_int(value) is not None
            ]

        observed_cells_count = _safe_float(post_state.get("observed_cells_count"), None)
        observed_total_cells = _safe_float(post_state.get("observed_total_cells"), None)
        if observed_cells_count is not None and observed_total_cells not in (None, 0):
            observed_ratio_final = round(observed_cells_count / max(observed_total_cells, 1.0), 4)

    charger_arrival_steps = sorted(set(charger_arrival_steps))
    charger_arrived_count = len(charger_arrival_steps)
    charger_first_arrival_step = charger_arrival_steps[0] if charger_arrival_steps else None
    charger_known_final = None if known_charger_count_final is None else known_charger_count_final > 0
    max_negative_charger_slack = 0.0
    if min_charger_slack is not None and min_charger_slack < 0.0:
        max_negative_charger_slack = round(min_charger_slack, 4)

    return {
        "charger_known_first_step": charger_known_first_step,
        "charger_known_final": charger_known_final,
        "known_charger_count_final": _safe_int(known_charger_count_final, 0),
        "charger_arrived_count": charger_arrived_count,
        "charger_first_arrival_step": charger_first_arrival_step,
        "charger_arrival_steps": charger_arrival_steps,
        "first_should_charge_step": first_should_charge_step,
        "attempted_charge_step_count": attempted_charge_step_count,
        "first_return_mode_step": first_return_mode_step,
        "min_battery": min_battery,
        "min_battery_step": min_battery_step,
        "min_charger_slack": round(min_charger_slack, 4) if min_charger_slack is not None else None,
        "max_negative_charger_slack": max_negative_charger_slack,
        "action_histogram": action_histogram,
        "last_actions": last_actions,
        "repeat_action_max_streak": repeat_action_max_streak,
        "revisit_ratio": round(revisit_steps / max(len(step_records), 1), 4),
        "max_revisit_count": max_revisit_count,
        "unique_cells_visited": len(unique_cells),
        "observed_ratio_final": observed_ratio_final,
        "final_window": [_compact_step_row(rec) for rec in step_records[-FINAL_WINDOW_MAX_ROWS:]],
        "evidence_windows": {
            "first_low_slack_window": _window_rows(step_records, first_low_slack_step),
            "first_should_charge_window": _window_rows(step_records, first_should_charge_step),
            "first_missed_charge_window": _window_rows(step_records, first_missed_charge_step),
            "first_loop_window": _window_rows(step_records, first_loop_step),
            "last_failure_window": _window_rows(step_records, _safe_int(step_records[-1].get("step"), len(step_records))),
        },
        "reward_attribution_lite": _build_reward_attribution_lite(step_records, total_reward),
        "anomaly_summary_lite": _build_anomaly_summary_lite(step_records),
    }


def _build_ai_summary(snapshot):
    episodes = list(snapshot.get("episodes") or [])
    failure_counts = {}
    aggregated_missing = set()
    example_evidence_windows = []

    for episode in episodes:
        fail_reason = str(episode.get("fail_reason") or episode.get("done_reason") or episode.get("result") or "unknown").lower()
        if fail_reason != "completed":
            failure_counts[fail_reason] = failure_counts.get(fail_reason, 0) + 1

        for signal in episode.get("missing_signals") or []:
            aggregated_missing.add(str(signal))
        reward_attribution = episode.get("reward_attribution_lite") or {}
        for signal in reward_attribution.get("missing_signals") or []:
            aggregated_missing.add(str(signal))
        anomaly_summary = episode.get("anomaly_summary_lite") or {}
        for signal in anomaly_summary.get("missing_signals") or []:
            aggregated_missing.add(str(signal))

        if len(example_evidence_windows) >= 3:
            continue
        evidence_windows = episode.get("evidence_windows") or {}
        non_empty_windows = {
            key: value
            for key, value in evidence_windows.items()
            if isinstance(value, list) and value
        }
        if not non_empty_windows:
            continue
        example_evidence_windows.append(
            {
                "episode_id": episode.get("episode_id"),
                "map_id": episode.get("map_id"),
                "fail_reason": fail_reason,
                "clean_score": _safe_float(episode.get("clean_score"), 0.0),
                "windows": non_empty_windows,
            }
        )

    top_failure_modes = [
        {"failure_mode": key, "count": value}
        for key, value in sorted(failure_counts.items(), key=lambda item: (-item[1], item[0]))[:5]
    ]
    recommended_next_analysis = []
    dominant_failure = top_failure_modes[0]["failure_mode"] if top_failure_modes else None
    if dominant_failure == "battery":
        recommended_next_analysis.append("Inspect low-slack and missed-charge evidence windows for earlier return triggers.")
    elif dominant_failure == "collision":
        recommended_next_analysis.append("Inspect NPC-near and planner/policy mismatch steps around collision failures.")
    elif dominant_failure:
        recommended_next_analysis.append(f"Review evidence windows for the dominant failure mode `{dominant_failure}`.")
    if aggregated_missing:
        recommended_next_analysis.append("Treat AI conclusions as partial where optional signals are missing; prefer fields present in `anomaly_summary_lite` and episode windows.")
    else:
        recommended_next_analysis.append("AI conclusions can use episode windows plus lite anomaly rates without waiting for replay-only enrichment.")

    return {
        "schema_version": AI_SUMMARY_SCHEMA_VERSION,
        "timestamp": snapshot.get("timestamp"),
        "checkpoint": snapshot.get("checkpoint"),
        "overall": snapshot.get("overall") or {},
        "per_map": snapshot.get("per_map") or {},
        "top_failure_modes": top_failure_modes,
        "missing_signals": sorted(aggregated_missing),
        "example_evidence_windows": example_evidence_windows,
        "recommended_next_analysis": recommended_next_analysis,
    }


def _extract_guidance(policy_info, act_data) -> dict[str, object]:
    """Summarize planner guidance from policy_info and act_data."""
    policy_snapshot, _ = _extract_policy_info(policy_info)
    action_diag, _ = _extract_action_diagnostics(act_data)
    return {
        "target_mode": policy_snapshot.get("target_mode"),
        "target_pos": policy_snapshot.get("target_pos"),
        "should_charge": policy_snapshot.get("should_charge"),
        "policy_top1": policy_snapshot.get("policy_top1"),
        "planner_top1": action_diag.get("planner_top1"),
        "mix_alpha": action_diag.get("mix_alpha"),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_results(episode_results, requested_maps):
    """Aggregate by map and overall."""
    per_map = {}
    for map_id in requested_maps:
        map_eps = [ep for ep in episode_results if ep["map_id"] == map_id]
        per_map[f"map{map_id}"] = _aggregate_episode_list(map_eps)

    overall = _aggregate_episode_list(episode_results)
    return {"per_map": per_map, "overall": overall}


def _aggregate_episode_list(episodes):
    if not episodes:
        return {
            "episode_count": 0,
            "avg_clean_score": 0.0,
            "completed_rate": 0.0,
            "battery_fail_rate": 0.0,
            "collision_fail_rate": 0.0,
        }

    wins = [ep for ep in episodes if ep["result"] == "completed"]
    fails_battery = [ep for ep in episodes if ep["result"] == "battery"]
    fails_collision = [ep for ep in episodes if ep["result"] == "collision"]

    scores = [ep["clean_score"] for ep in episodes]
    sorted_scores = sorted(scores)

    return {
        "episode_count": len(episodes),
        "win_episode_count": len(wins),
        "avg_clean_score": round(sum(scores) / len(scores), 1),
        "score_p10": sorted_scores[max(0, int(len(sorted_scores) * 0.1) - 1)] if sorted_scores else 0.0,
        "score_p50": sorted_scores[len(sorted_scores) // 2] if sorted_scores else 0.0,
        "score_p90": sorted_scores[min(len(sorted_scores) - 1, int(len(sorted_scores) * 0.9))] if sorted_scores else 0.0,
        "min_clean_score": min(scores) if scores else 0.0,
        "max_clean_score": max(scores) if scores else 0.0,
        "completed_rate": round(len(wins) / len(episodes), 4),
        "battery_fail_rate": round(len(fails_battery) / len(episodes), 4),
        "collision_fail_rate": round(len(fails_collision) / len(episodes), 4),
        "avg_steps": round(sum(ep["steps"] for ep in episodes) / len(episodes), 1),
        "avg_charge_count": round(sum(ep["charge_count"] for ep in episodes) / len(episodes), 2),
        "avg_invalid_move_rate": round(sum(ep["invalid_move_rate"] for ep in episodes) / len(episodes), 4),
        "avg_dirt_ratio": round(sum(ep["dirt_ratio"] for ep in episodes) / len(episodes), 4),
        "avg_total_reward": round(sum(ep["total_reward"] for ep in episodes) / len(episodes), 1),
    }


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_benchmark_checkpoint(agent, checkpoint, logger):
    """Load a specific checkpoint for benchmark evaluation."""
    resolved = str(checkpoint or "").strip()
    if resolved in {"", "latest"}:
        agent.load_model(id="latest")
        ref = getattr(agent, "current_model_ref", {}) or {}
        return {
            "path": ref.get("path") or resolved or "latest",
            "checkpoint_id": ref.get("checkpoint_id"),
        }

    if not os.path.isabs(resolved):
        resolved = os.path.join("/workspace/code", resolved)

    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Benchmark checkpoint not found: {resolved}")

    logger.info(f"[HOLDOUT-BENCH] Loading checkpoint: {resolved}")
    state_dict = torch.load(resolved, map_location=agent.device)
    agent.model.load_state_dict(state_dict)
    checkpoint_id = parse_checkpoint_id(resolved) or os.path.basename(resolved)

    if hasattr(agent, "current_model_ref"):
        agent.current_model_ref = {
            "path": resolved,
            "id": "benchmark",
            "checkpoint_id": checkpoint_id,
        }
    logger.info(f"[HOLDOUT-BENCH] Loaded state_dict from {resolved}")
    return {"path": resolved, "checkpoint_id": checkpoint_id}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_base_env_conf(usr_conf):
    if isinstance(usr_conf, dict) and isinstance(usr_conf.get("env_conf"), dict):
        return deepcopy(usr_conf["env_conf"])
    return deepcopy(usr_conf)


def _env_flag(name):
    return str(os.getenv(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name, default):
    value = os.getenv(name)
    if value in (None, ""):
        return int(default)
    return int(value)


def _load_shard_assignment(hostname, requested_maps, episodes_per_map) -> ShardAssignment | None:
    assignments_dir = Path("/workspace/code/holdout_shards/assignments")
    shard_count = int(os.getenv("KAIWU_BENCHMARK_SHARD_COUNT", "1").strip() or "1")
    aisrv_index_raw = os.getenv("KAIWU_AISRV_INDEX", "").strip()
    candidate_paths = []

    if aisrv_index_raw:
        try:
            shard_index = int(aisrv_index_raw) - 1
        except ValueError as exc:
            raise ValueError(f"KAIWU_AISRV_INDEX must be an integer, got {aisrv_index_raw!r}") from exc
        if shard_index < 0:
            raise ValueError(f"KAIWU_AISRV_INDEX must be >= 1, got {aisrv_index_raw!r}")
        if shard_index >= shard_count:
            return None
        candidate_paths.append(assignments_dir / f"shard_{shard_index}.json")

    candidate_paths.append(assignments_dir / f"{hostname}.json")

    deduped_paths = []
    seen_paths = set()
    for path in candidate_paths:
        if path in seen_paths:
            continue
        deduped_paths.append(path)
        seen_paths.add(path)

    assignment_path = None
    max_wait_seconds = int(os.getenv("KAIWU_BENCHMARK_ASSIGNMENT_WAIT_SECONDS", "180").strip() or "180")
    deadline = time.time() + max_wait_seconds
    while time.time() <= deadline:
        for candidate_path in deduped_paths:
            if candidate_path.is_file():
                assignment_path = candidate_path
                break
        if assignment_path is not None:
            break
        time.sleep(1.0)

    if assignment_path is None:
        raise FileNotFoundError(
            f"Holdout shard assignment not found for host {hostname}; checked: "
            f"{[str(path) for path in deduped_paths]}"
        )

    try:
        assignment = json.loads(assignment_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Holdout shard assignment is not valid JSON: {assignment_path}") from exc

    if not isinstance(assignment, dict):
        raise ValueError(f"Holdout shard assignment must be a JSON object: {assignment_path}")

    required_fields = ["shard_index", "shard_count", "episodes", "maps", "episodes_per_map", "run_id"]
    missing_fields = [field for field in required_fields if field not in assignment]
    if missing_fields:
        raise ValueError(f"Holdout shard assignment missing fields {missing_fields}: {assignment_path}")

    if not isinstance(assignment["maps"], list):
        raise ValueError(f"Holdout shard assignment maps must be a JSON array: {assignment_path}")
    if not isinstance(assignment["episodes"], list):
        raise ValueError(f"Holdout shard assignment episodes must be a JSON array: {assignment_path}")

    shard_index = int(assignment["shard_index"])
    shard_count = int(assignment["shard_count"])
    assignment_maps = [int(map_id) for map_id in assignment["maps"]]
    assignment_eps_per_map = int(assignment["episodes_per_map"])
    run_id = str(assignment["run_id"] or "").strip()
    raw_episodes = list(assignment["episodes"])

    if shard_index < 0:
        raise ValueError(f"Holdout shard_index must be >= 0: {assignment_path}")
    if shard_count <= 0:
        raise ValueError(f"Holdout shard_count must be > 0: {assignment_path}")
    if shard_index >= shard_count:
        raise ValueError(f"Holdout shard_index must be < shard_count: {assignment_path}")
    if not assignment_maps:
        raise ValueError(f"Holdout shard assignment maps must be non-empty: {assignment_path}")
    if assignment_eps_per_map <= 0:
        raise ValueError(f"Holdout shard assignment episodes_per_map must be > 0: {assignment_path}")
    if not run_id:
        raise ValueError(f"Holdout shard assignment run_id must be non-empty: {assignment_path}")
    if not raw_episodes:
        raise ValueError(f"Holdout shard assignment episodes must be non-empty: {assignment_path}")
    if len(set(assignment_maps)) != len(assignment_maps):
        raise ValueError(f"Holdout shard assignment maps must be unique: {assignment_path}")
    if set(assignment_maps) != set(requested_maps):
        raise ValueError(
            f"Holdout shard assignment maps {assignment_maps} do not match requested maps {list(requested_maps)}: {assignment_path}"
        )
    if assignment_eps_per_map != int(episodes_per_map):
        raise ValueError(
            f"Holdout shard assignment episodes_per_map {assignment_eps_per_map} does not match requested {episodes_per_map}: {assignment_path}"
        )

    normalized_episodes = []
    seen = set()
    allowed_maps = set(assignment_maps)
    for item in raw_episodes:
        if isinstance(item, dict):
            map_id = item.get("map_id")
            ep_idx = item.get("ep_idx")
            if map_id is None or ep_idx is None:
                raise ValueError(
                    f"Holdout shard assignment episode dict must include map_id and ep_idx: {assignment_path}"
                )
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            map_id, ep_idx = item
        else:
            raise ValueError(
                f"Holdout shard assignment episodes must contain map_id/ep_idx pairs: {assignment_path}"
            )

        map_id = int(map_id)
        ep_idx = int(ep_idx)
        pair = (map_id, ep_idx)

        if map_id not in allowed_maps:
            raise ValueError(f"Holdout shard episode map_id {map_id} not declared in maps: {assignment_path}")
        if ep_idx < 1 or ep_idx > assignment_eps_per_map:
            raise ValueError(
                f"Holdout shard episode ep_idx {ep_idx} out of range 1..{assignment_eps_per_map}: {assignment_path}"
            )
        if pair in seen:
            raise ValueError(f"Holdout shard assignment contains duplicate episode pair {pair}: {assignment_path}")

        seen.add(pair)
        normalized_episodes.append({"map_id": map_id, "ep_idx": ep_idx})

    return {
        "shard_index": shard_index,
        "shard_count": shard_count,
        "episodes": normalized_episodes,
        "maps": assignment_maps,
        "episodes_per_map": assignment_eps_per_map,
        "run_id": run_id,
        "assignment_path": str(assignment_path),
    }


def _wrap_env_conf(usr_conf, env_conf):
    if isinstance(usr_conf, dict) and "env_conf" in usr_conf:
        wrapped = deepcopy(usr_conf)
        wrapped["env_conf"] = deepcopy(env_conf)
        return wrapped
    return deepcopy(env_conf)


def _inject_agent_runtime(env_obs, agent):
    payload = dict(env_obs or {})
    runtime = dict(payload.get("runtime") or {})
    ref = getattr(agent, "current_model_ref", {}) or {}
    runtime.setdefault("global_step_since_resume", int(ref.get("global_step_since_resume") or 0))
    runtime.setdefault("checkpoint_global_step", int(ref.get("checkpoint_step") or 0))
    payload["runtime"] = runtime
    return payload


def _create_step_logger(session_dir):
    log = logging.getLogger(f"holdout_benchmark.{session_dir.name}")
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(session_dir / "benchmark.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(fh)
    return log


def _print_summary(aggregated, logger, elapsed):
    overall = aggregated["overall"]
    per_map_parts = []
    for name, metrics in aggregated["per_map"].items():
        per_map_parts.append(
            f"{name}: CS={metrics['avg_clean_score']:.0f} "
            f"completed={metrics['completed_rate']:.0%} "
            f"bat_fail={metrics['battery_fail_rate']:.0%}"
        )
    logger.info(f"[HOLDOUT-BENCH] ========== Done ({elapsed:.0f}s) ==========")
    logger.info(
        f"[HOLDOUT-BENCH] Overall CS={overall['avg_clean_score']:.0f} "
        f"({overall['win_episode_count']}/{overall['episode_count']}) "
        f"completed={overall['completed_rate']:.0%} "
        f"bat_fail={overall['battery_fail_rate']:.0%} "
        f"col_fail={overall['collision_fail_rate']:.0%} "
        f"p10={overall.get('score_p10', 0):.0f} p90={overall.get('score_p90', 0):.0f}"
    )
    for part in per_map_parts:
        logger.info(f"[HOLDOUT-BENCH]   {part}")


def _empty_episode_result(map_id, ep_idx, fail_reason):
    return {
        "episode_id": f"map{map_id}_ep{ep_idx:02d}",
        "map_id": map_id,
        "ep_idx": ep_idx,
        "result": fail_reason,
        "fail_reason": fail_reason,
        "done_reason": fail_reason,
        "status": "completed" if fail_reason == "completed" else "failed",
        "clean_score": 0,
        "steps": 0,
        "finished_steps": 0,
        "max_step": HOLDOUT_ROUNDS[0]["max_step"],
        "battery_max": HOLDOUT_ROUNDS[0]["battery_max"],
        "robot_count": HOLDOUT_ROUNDS[0]["robot_count"],
        "charger_count": HOLDOUT_ROUNDS[0]["charger_count"],
        "charge_count": 0,
        "remaining_charge": 0,
        "total_reward": 0.0,
        "dirt_cleaned": 0,
        "total_dirt": 0,
        "dirt_ratio": 0.0,
        "invalid_move_count": 0,
        "invalid_move_rate": 0.0,
        "step_log": "",
        "field_availability": {},
        "missing_signals": [],
        "charger_known_first_step": None,
        "charger_known_final": None,
        "known_charger_count_final": 0,
        "charger_arrived_count": 0,
        "charger_first_arrival_step": None,
        "charger_arrival_steps": [],
        "first_should_charge_step": None,
        "attempted_charge_step_count": 0,
        "first_return_mode_step": None,
        "min_battery": None,
        "min_battery_step": None,
        "min_charger_slack": None,
        "max_negative_charger_slack": 0.0,
        "action_histogram": {},
        "last_actions": [],
        "repeat_action_max_streak": 0,
        "revisit_ratio": 0.0,
        "max_revisit_count": 0,
        "unique_cells_visited": 0,
        "observed_ratio_final": None,
        "final_window": [],
        "evidence_windows": {
            "first_low_slack_window": [],
            "first_should_charge_window": [],
            "first_missed_charge_window": [],
            "first_loop_window": [],
            "last_failure_window": [],
        },
        "reward_attribution_lite": {
            "available": False,
            "reason": "scalar_reward_only",
            "total_reward": 0.0,
        },
        "anomaly_summary_lite": _build_anomaly_summary_lite([]),
    }


def _atomic_write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
