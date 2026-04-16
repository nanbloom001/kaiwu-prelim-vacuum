#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Standalone benchmark evaluation module.

Runs fixed scenarios against a model checkpoint and saves structured results.
Completely independent from the training loop — no data sent to learner,
no curriculum interference, no episode count pollution.

Scenario configs are defined in ROUNDS below — easy to modify.
Each round runs on ALL maps (1-10).

Triggered via KAIWU_BENCHMARK_MODE=1 environment variable.
"""

import json
import logging
import os
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from agent_ppo.conf.conf import Config
from agent_ppo.utils.experiment_archive import infer_fail_reason
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery

# ---------------------------------------------------------------------------
# Scenario definitions — modify here to change test configs
# ---------------------------------------------------------------------------
ALL_MAPS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

ROUNDS = [
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

# Log output base directory (inside container)
EVAL_LOG_BASE = Path("/workspace/code/eval_logs")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_benchmark(env, agent, usr_conf, logger):
    """
    Run all evaluation rounds and save results + detailed logs.

    Args:
        env: Framework environment (gamecore connection).
        agent: Framework agent (model inference).
        usr_conf: Base user config from train_env_conf.toml.
        logger: Framework logger.

    Returns:
        dict: Overall benchmark results.
    """
    base_env_conf = _extract_base_env_conf(usr_conf)
    checkpoint = os.getenv("KAIWU_BENCHMARK_CHECKPOINT", "").strip() or Config.RESUME_CHECKPOINT
    session_id = time.strftime("%Y%m%d-%H%M%S")

    total_episodes = len(ROUNDS) * len(ALL_MAPS)
    logger.info(f"[BENCHMARK] ========== Evaluation Start ==========")
    logger.info(f"[BENCHMARK] checkpoint={checkpoint}")
    logger.info(f"[BENCHMARK] rounds={len(ROUNDS)} maps={len(ALL_MAPS)} total={total_episodes}")
    for r in ROUNDS:
        logger.info(f"[BENCHMARK]   {r['name']}: {r['desc']}")

    _load_benchmark_checkpoint(agent, checkpoint, logger)

    # Prepare session log directory
    session_dir = EVAL_LOG_BASE / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    step_log = _create_step_logger(session_dir)

    # Write scenario manifest
    manifest = {
        "timestamp": session_id,
        "checkpoint": checkpoint,
        "git_commit": _get_git_commit(),
        "rounds": ROUNDS,
        "maps": ALL_MAPS,
        "total_episodes": total_episodes,
    }
    _atomic_write_json(session_dir / "manifest.json", manifest)

    episode_results = []
    idx = 0
    t_start = time.time()

    for round_def in ROUNDS:
        for map_id in ALL_MAPS:
            idx += 1
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
                round_name=round_def["name"],
                map_id=map_id,
                round_def=round_def,
                logger=logger,
                step_log=step_log,
                idx=idx,
                total=total_episodes,
                session_dir=session_dir,
            )
            episode_results.append(result)

    elapsed = time.time() - t_start

    # Aggregate
    aggregated = _aggregate_results(episode_results)

    # Save results JSON
    snapshot = {
        "version": 3,
        "timestamp": session_id,
        "checkpoint": checkpoint,
        "git_commit": _get_git_commit(),
        "elapsed_seconds": round(elapsed, 1),
        "rounds": {r["name"]: r["desc"] for r in ROUNDS},
        "per_round": aggregated["per_round"],
        "overall": aggregated["overall"],
        "episodes": episode_results,
    }
    _save_results(Path("/workspace/code") / "eval_results.json", snapshot)
    _atomic_write_json(session_dir / "result.json", snapshot)

    # Close step logger
    for handler in step_log.handlers[:]:
        handler.close()
        step_log.removeHandler(handler)

    _print_summary(aggregated, logger, elapsed)
    done_marker = Path("/workspace/code/.benchmark_done")
    done_marker.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "timestamp": time.strftime("%Y%m%d-%H%M%S"),
                "checkpoint": checkpoint,
                "overall": aggregated["overall"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(f"[BENCHMARK] Completion marker written to {done_marker}")
    return aggregated


# ---------------------------------------------------------------------------
# Episode runner with detailed logging
# ---------------------------------------------------------------------------

def _run_eval_episode(env, agent, usr_conf, round_name, map_id, round_def,
                      logger, step_log, idx, total, session_dir):
    """Run one evaluation episode with per-step detailed logging."""
    ep_label = f"[{idx}/{total}] {round_name}/map{map_id}"
    logger.info(f"[BENCHMARK] {ep_label} START | robots={round_def['robot_count']} "
                f"chargers={round_def['charger_count']} steps={round_def['max_step']} "
                f"battery={round_def['battery_max']}")

    env_obs = env.reset(usr_conf)
    if handle_disaster_recovery(env_obs, logger):
        logger.warning(f"[BENCHMARK] {ep_label} SKIP (disaster recovery)")
        return {"round": round_name, "map_id": map_id, "result": "error",
                "clean_score": 0, "steps": 0, "charge_count": 0}

    agent.reset(env_obs)
    obs_data, _ = agent.observation_process(env_obs)

    # Per-episode step log file
    ep_log_path = session_dir / "episodes" / f"{round_name}_map{map_id}.jsonl"
    ep_log_path.parent.mkdir(parents=True, exist_ok=True)
    ep_log_file = open(ep_log_path, "w", encoding="utf-8")

    fm = agent.preprocessor
    step_records = []
    done = False
    step = 0
    total_reward = 0.0

    while not done:
        act_data = agent.predict([obs_data])[0]
        act = agent.action_process(act_data)
        action_idx = int(np.argmax(act_data.action)) if hasattr(act_data, "action") else -1

        env_reward, env_obs = env.step(act)
        if handle_disaster_recovery(env_obs, logger):
            break

        terminated = env_obs["terminated"]
        truncated = env_obs["truncated"]
        frame_no = env_obs.get("frame_no", step)
        step += 1
        done = terminated or truncated

        reward_scalar = float(agent.last_reward)
        total_reward += reward_scalar

        # Collect per-step diagnostics
        step_rec = {
            "step": step,
            "action": action_idx,
            "reward": round(reward_scalar, 4),
            "total_reward": round(total_reward, 4),
            "battery": fm.battery,
            "battery_max": fm.battery_max,
            "dirt_cleaned": fm.dirt_cleaned,
            "total_dirt": fm.total_dirt,
            "mode": fm.current_mode,
            "charger_slack": round(fm.charger_slack, 2),
            "nearest_npc_dist": round(fm.nearest_npc_dist, 1),
            "invalid_move_count": fm.invalid_move_count,
        }

        # Write to episode JSONL file
        ep_log_file.write(json.dumps(step_rec, ensure_ascii=False) + "\n")

        # Log milestone steps (every 100 steps, or near end)
        if step % 100 == 0 or done:
            step_log.info(
                f"{ep_label} step={step} bat={fm.battery}/{fm.battery_max} "
                f"dirt={fm.dirt_cleaned}/{fm.total_dirt} mode={fm.current_mode} "
                f"slack={fm.charger_slack:.1f} npc={fm.nearest_npc_dist:.0f} "
                f"act={action_idx} reward={reward_scalar:.3f}"
            )

        if not done:
            obs_data, _ = agent.observation_process(env_obs)

    ep_log_file.close()

    # Extract final results
    observation = env_obs.get("observation") or {}
    env_info = observation.get("env_info") or {}
    hero = (observation.get("frame_state") or {}).get("heroes") or {}
    extra_info = env_obs.get("extra_info") or observation.get("extra_info") or {}

    fail_reason = infer_fail_reason(
        terminated=terminated, truncated=truncated,
        battery=hero.get("battery"), extra_info=extra_info,
    )
    clean_score = float(env_info.get("clean_score", 0))
    finished_steps = float(env_info.get("finished_steps", step))
    charge_count = float(env_info.get("charge_count", 0))
    remaining_charge = float(env_info.get("remaining_charge", hero.get("battery", 0)))

    result = {
        "round": round_name,
        "map_id": map_id,
        "result": fail_reason,
        "clean_score": clean_score,
        "steps": finished_steps,
        "charge_count": charge_count,
        "remaining_charge": remaining_charge,
        "total_reward": round(total_reward, 3),
        "dirt_cleaned": fm.dirt_cleaned,
        "total_dirt": fm.total_dirt,
        "dirt_ratio": round(fm.dirt_cleaned / max(fm.total_dirt, 1), 4),
        "invalid_move_count": fm.invalid_move_count,
        "invalid_move_rate": round(fm.invalid_move_count / max(step, 1), 4),
        "step_log": str(ep_log_path.relative_to(session_dir)),
    }

    result_str = "WIN" if fail_reason == "completed" else f"FAIL({fail_reason})"
    logger.info(
        f"[BENCHMARK] {ep_label} {result_str} | score={clean_score:.0f} "
        f"dirt={fm.dirt_cleaned}/{fm.total_dirt} steps={int(finished_steps)} "
        f"charges={int(charge_count)} bat_left={int(remaining_charge)} "
        f"reward={total_reward:.1f}"
    )

    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _aggregate_results(episode_results):
    """Aggregate by round and overall."""
    per_round = {}
    for r in episode_results:
        name = r["round"]
        per_round.setdefault(name, []).append(r)

    round_metrics = {}
    for name, eps in per_round.items():
        wins = [e for e in eps if e["result"] == "completed"]
        fails_battery = [e for e in eps if e["result"] == "battery"]
        fails_collision = [e for e in eps if e["result"] == "collision"]
        round_metrics[name] = {
            "win_rate": round(len(wins) / len(eps), 4) if eps else 0,
            "avg_clean_score": round(sum(e["clean_score"] for e in eps) / len(eps), 1),
            "avg_steps": round(sum(e["steps"] for e in eps) / len(eps), 1),
            "avg_charge_count": round(sum(e["charge_count"] for e in eps) / len(eps), 2),
            "avg_dirt_ratio": round(sum(e["dirt_ratio"] for e in eps) / len(eps), 4),
            "battery_fail_rate": round(len(fails_battery) / len(eps), 4) if eps else 0,
            "collision_fail_rate": round(len(fails_collision) / len(eps), 4) if eps else 0,
            "avg_invalid_move_rate": round(sum(e["invalid_move_rate"] for e in eps) / len(eps), 4),
            "episode_count": len(eps),
            "win_episode_count": len(wins),
        }

    all_eps = episode_results
    wins = [e for e in all_eps if e["result"] == "completed"]
    overall = {
        "win_rate": round(len(wins) / len(all_eps), 4) if all_eps else 0,
        "avg_clean_score": round(sum(e["clean_score"] for e in all_eps) / len(all_eps), 1) if all_eps else 0,
        "avg_steps": round(sum(e["steps"] for e in all_eps) / len(all_eps), 1) if all_eps else 0,
        "avg_charge_count": round(sum(e["charge_count"] for e in all_eps) / len(all_eps), 2) if all_eps else 0,
        "episode_count": len(all_eps),
        "win_episode_count": len(wins),
    }

    return {"per_round": round_metrics, "overall": overall}


# ---------------------------------------------------------------------------
# Log helpers
# ---------------------------------------------------------------------------

def _create_step_logger(session_dir: Path) -> logging.Logger:
    """Create a dedicated logger for eval step-level output."""
    log = logging.getLogger(f"benchmark.{session_dir.name}")
    log.setLevel(logging.DEBUG)
    fh = logging.FileHandler(session_dir / "benchmark.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(fh)
    return log


def _print_summary(aggregated, logger, elapsed):
    """Print benchmark summary."""
    o = aggregated["overall"]
    parts = " | ".join(
        f"{n}: WR={m['win_rate']:.0%} CS={m['avg_clean_score']:.0f} "
        f"bat_fail={m['battery_fail_rate']:.0%} col_fail={m['collision_fail_rate']:.0%}"
        for n, m in aggregated["per_round"].items()
    )
    logger.info(f"[BENCHMARK] ========== Done ({elapsed:.0f}s) ==========")
    logger.info(f"[BENCHMARK] Overall WR={o['win_rate']:.0%} CS={o['avg_clean_score']:.0f} "
                f"({o['win_episode_count']}/{o['episode_count']})")
    logger.info(f"[BENCHMARK] {parts}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_base_env_conf(usr_conf):
    if isinstance(usr_conf, dict) and isinstance(usr_conf.get("env_conf"), dict):
        return deepcopy(usr_conf["env_conf"])
    return deepcopy(usr_conf)


def _load_benchmark_checkpoint(agent, checkpoint, logger):
    """Load a specific checkpoint for benchmark evaluation."""
    resolved = checkpoint
    if not os.path.isabs(resolved):
        resolved = os.path.join("/workspace/code", resolved)

    if not os.path.isfile(resolved):
        logger.warning(f"[BENCHMARK] Checkpoint not found: {resolved}, using default")
        agent.load_model(id="latest")
        return

    logger.info(f"[BENCHMARK] Loading checkpoint: {resolved}")
    state_dict = torch.load(resolved, map_location=agent.device)
    agent.model.load_state_dict(state_dict)
    if hasattr(agent, "current_model_ref"):
        agent.current_model_ref = {
            "path": resolved,
            "id": "benchmark",
            "checkpoint_id": os.path.basename(resolved),
        }
    logger.info(f"[BENCHMARK] Loaded state_dict from {resolved}")


def _wrap_env_conf(usr_conf, env_conf):
    if isinstance(usr_conf, dict) and "env_conf" in usr_conf:
        wrapped = deepcopy(usr_conf)
        wrapped["env_conf"] = deepcopy(env_conf)
        return wrapped
    return deepcopy(env_conf)


def _save_results(results_file, snapshot):
    results = {"version": 3, "benchmarks": []}
    if results_file.exists():
        try:
            existing = json.loads(results_file.read_text(encoding="utf-8"))
            results["benchmarks"] = existing.get("benchmarks", [])
        except Exception:
            pass
    results["benchmarks"].append(snapshot)
    _atomic_write_json(results_file, results)


def _atomic_write_json(path, data):
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _get_git_commit():
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd="/workspace/code",
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"
