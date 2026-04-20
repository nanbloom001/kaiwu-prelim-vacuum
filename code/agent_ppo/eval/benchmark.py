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
from agent_ppo.workflow.preload_checkpoint import resolve_benchmark_checkpoint
from agent_ppo.utils.experiment_archive import infer_fail_reason, parse_checkpoint_id
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

SCHEMA_VERSION = 4
TOPK = 2
EVIDENCE_RADIUS = 5
REWARD_COMPONENT_KEYS = (
    "cleaning",
    "streak",
    "explore",
    "frontier",
    "recoverability",
    "charge",
    "npc",
    "stuck",
    "idle",
    "anchor_consistency",
    "sticky_anchor_penalty",
    "return_progress",
    "diag_efficiency",
    "return_stall",
    "late_contract",
    "astar_potential",
    "charge_margin_pressure",
    "unknown_path_risk",
    "missed_charge_penalty",
    "planner_alignment",
    "cleaning_context_scale",
    "cps_bonus",
)
ANOMALY_KEYS = (
    "revisit_on_clean_floor",
    "redundant_clean_path",
    "low_value_revisit",
    "loop_suspect",
    "corner_loop_suspect",
    "no_clean_no_return_progress",
    "positive_reward_while_no_progress",
    "positive_clean_reward_while_revisiting",
    "positive_explore_reward_while_revisiting",
    "positive_frontier_reward_while_revisiting",
    "wall_hugging_clean_floor",
    "stale_boundary_follow",
    "narrow_unknown_commit",
    "missed_charge_opportunity",
    "charger_nearby_not_charged",
    "suboptimal_target_hold",
    "planner_policy_divergence",
    "charger_contested",
)
ISSUE_INDEX_CONFIG = {
    "wall_hugging": {
        "predicate": lambda ep: ep.get("anomaly_summary", {}).get("wall_hugging_clean_floor_rate", 0.0) >= 0.1,
        "primary_metrics": [
            "wall_hugging_clean_floor_rate",
            "stale_boundary_follow_rate",
            "avg_wall_follow_streak",
        ],
        "primary_attribution_key": "reward_attribution.wall_hugging",
    },
    "corner_loop": {
        "predicate": lambda ep: bool(ep.get("anomaly_summary", {}).get("corner_loop_detected")),
        "primary_metrics": [
            "corner_loop_rate",
            "positive_reward_while_no_progress_rate",
            "revisit_on_clean_floor_rate",
        ],
        "primary_attribution_key": "reward_attribution.corner_loop",
    },
    "loop": {
        "predicate": lambda ep: bool(ep.get("anomaly_summary", {}).get("loop_episode_detected")),
        "primary_metrics": [
            "loop_suspect_rate",
            "position_repeat_16_avg",
            "avg_zero_progress_streak",
        ],
        "primary_attribution_key": "reward_attribution.loop",
    },
    "low_value_revisit": {
        "predicate": lambda ep: ep.get("anomaly_summary", {}).get("low_value_revisit_rate", 0.0) >= 0.1,
        "primary_metrics": [
            "low_value_revisit_rate",
            "redundant_clean_path_rate",
            "positive_reward_while_no_progress_rate",
        ],
        "primary_attribution_key": "reward_attribution.low_value_revisit",
    },
    "narrow_unknown_commit": {
        "predicate": lambda ep: ep.get("anomaly_summary", {}).get("narrow_unknown_commit_rate", 0.0) >= 0.05,
        "primary_metrics": [
            "narrow_unknown_commit_rate",
            "avg_narrow_passage_rate",
            "avg_fallback_to_chebyshev_rate",
        ],
        "primary_attribution_key": "reward_attribution.narrow_unknown_commit",
    },
    "missed_charge_opportunity": {
        "predicate": lambda ep: ep.get("anomaly_summary", {}).get("missed_charge_opportunity_rate", 0.0) > 0.0,
        "primary_metrics": [
            "missed_charge_opportunity_rate",
            "charger_nearby_not_charged_rate",
            "avg_charge_margin_now",
        ],
        "primary_attribution_key": "reward_attribution.missed_charge_opportunity",
    },
    "late_return": {
        "predicate": lambda ep: ep.get("late_return_rate", 0.0) > 0.0,
        "primary_metrics": [
            "late_return_rate",
            "return_stall_rate",
            "return_efficiency_ratio",
        ],
        "primary_attribution_key": "reward_attribution.return_stall_window",
    },
    "late_contract": {
        "predicate": lambda ep: ep.get("late_contract_rate", 0.0) > 0.0,
        "primary_metrics": [
            "late_contract_rate",
            "recoverability_violation_rate",
            "return_progress_per_step",
        ],
        "primary_attribution_key": "reward_attribution.battery_fail_trajectory",
    },
    "return_stall": {
        "predicate": lambda ep: ep.get("return_stall_rate", 0.0) >= 0.5,
        "primary_metrics": [
            "return_stall_rate",
            "return_progress_per_step",
            "diag_rate_return",
        ],
        "primary_attribution_key": "reward_attribution.return_stall_window",
    },
    "target_selection": {
        "predicate": lambda ep: ep.get("anomaly_summary", {}).get("suboptimal_target_hold_rate", 0.0) >= 0.05,
        "primary_metrics": [
            "suboptimal_target_hold_rate",
            "avg_target_selection_gap",
            "target_switch_rate",
        ],
        "primary_attribution_key": "reward_attribution.suboptimal_target_hold",
    },
    "charger_contested": {
        "predicate": lambda ep: ep.get("anomaly_summary", {}).get("charger_contested_rate", 0.0) >= 0.05,
        "primary_metrics": [
            "charger_contested_rate",
            "avg_target_charger_robot_count",
            "avg_robot_on_target_path_count",
        ],
        "primary_attribution_key": "reward_attribution.charger_contested",
    },
    "battery_fail": {
        "predicate": lambda ep: ep.get("result") == "battery",
        "primary_metrics": [
            "battery_fail_rate",
            "late_return_rate",
            "recoverability_violation_rate",
        ],
        "primary_attribution_key": "reward_attribution.battery_fail_trajectory",
    },
}


def _benchmark_policy_mode():
    mode = os.getenv("KAIWU_BENCHMARK_POLICY_MODE", "eval").strip().lower()
    return mode if mode in {"train", "eval"} else "eval"


def _configured_maps():
    raw = os.getenv("KAIWU_BENCHMARK_MAPS", "").strip()
    if not raw:
        return list(ALL_MAPS)
    maps = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        maps.append(int(item))
    return maps or list(ALL_MAPS)


def _configured_rounds():
    raw = os.getenv("KAIWU_BENCHMARK_ROUNDS_JSON", "").strip()
    if not raw:
        return deepcopy(ROUNDS)
    rounds = json.loads(raw)
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("KAIWU_BENCHMARK_ROUNDS_JSON must be a non-empty JSON list")
    return rounds


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
    requested_checkpoint = resolve_benchmark_checkpoint(
        Path("/workspace/code"),
        os.getenv("KAIWU_BENCHMARK_CHECKPOINT", "").strip(),
        Config.RESUME_CHECKPOINT,
    )
    policy_mode = _benchmark_policy_mode()
    rounds = _configured_rounds()
    maps = _configured_maps()
    session_id = time.strftime("%Y%m%d-%H%M%S")

    total_episodes = len(rounds) * len(maps)
    logger.info("[BENCHMARK] ========== Evaluation Start ==========")
    logger.info(f"[BENCHMARK] checkpoint={requested_checkpoint}")
    logger.info(f"[BENCHMARK] policy_mode={policy_mode}")
    logger.info(f"[BENCHMARK] rounds={len(rounds)} maps={len(maps)} total={total_episodes}")
    for round_def in rounds:
        logger.info(f"[BENCHMARK]   {round_def['name']}: {round_def['desc']}")

    loaded_checkpoint = _load_benchmark_checkpoint(agent, requested_checkpoint, logger)

    session_dir = EVAL_LOG_BASE / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    step_log = _create_step_logger(session_dir)

    manifest = {
        "version": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "timestamp": session_id,
        "checkpoint": loaded_checkpoint["checkpoint"],
        "policy_mode": policy_mode,
        "git_commit": _get_git_commit(),
        "rounds": rounds,
        "maps": maps,
        "total_episodes": total_episodes,
    }
    _atomic_write_json(session_dir / "manifest.json", manifest)

    episode_results = []
    idx = 0
    t_start = time.time()

    for round_def in rounds:
        for map_id in maps:
            idx += 1
            env_conf = deepcopy(base_env_conf)
            env_conf["map"] = [map_id]
            env_conf["map_random"] = False
            env_conf["robot_count"] = round_def["robot_count"]
            env_conf["charger_count"] = round_def["charger_count"]
            env_conf["max_step"] = round_def["max_step"]
            env_conf["battery_max"] = round_def["battery_max"]

            wrapped_conf = _wrap_env_conf(usr_conf, env_conf)
            episode_results.append(
                _run_eval_episode(
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
            )

    elapsed = time.time() - t_start
    aggregated = _aggregate_results(episode_results)

    snapshot = {
        "version": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "timestamp": session_id,
        "checkpoint": loaded_checkpoint["checkpoint"],
        "policy_mode": policy_mode,
        "git_commit": _get_git_commit(),
        "elapsed_seconds": round(elapsed, 1),
        "rounds": {round_def["name"]: round_def["desc"] for round_def in rounds},
        "per_round": aggregated["per_round"],
        "overall": aggregated["overall"],
        "execution": {
            "mode": "single",
            "policy_mode": policy_mode,
        },
        "episodes": episode_results,
    }
    ai_summary = _build_ai_summary(snapshot)

    _save_results(Path("/workspace/code") / "eval_results.json", snapshot)
    _atomic_write_json(session_dir / "result.json", snapshot)
    _atomic_write_json(session_dir / "ai_summary.json", ai_summary)

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
                "checkpoint": loaded_checkpoint["checkpoint"],
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

def _run_eval_episode(env, agent, usr_conf, round_name, map_id, round_def, logger, step_log, idx, total, session_dir):
    """Run one evaluation episode with per-step detailed logging."""
    ep_label = f"[{idx}/{total}] {round_name}/map{map_id}"
    logger.info(
        f"[BENCHMARK] {ep_label} START | robots={round_def['robot_count']} "
        f"chargers={round_def['charger_count']} steps={round_def['max_step']} "
        f"battery={round_def['battery_max']}"
    )

    env_obs = _inject_agent_runtime(env.reset(usr_conf), agent)
    if handle_disaster_recovery(env_obs, logger):
        logger.warning(f"[BENCHMARK] {ep_label} SKIP (disaster recovery)")
        return {
            "episode_id": f"{round_name}_map{map_id}",
            "round": round_name,
            "map_id": map_id,
            "result": "error",
            "clean_score": 0,
            "steps": 0,
            "charge_count": 0,
            "summary": {"result": "error"},
            "diagnostics": _sequence_diagnostics([]),
            "anomaly_summary": _anomaly_summary([]),
            "reward_attribution": _reward_attribution([]),
            "evidence_windows": _extract_evidence_windows([]),
        }

    agent.reset(env_obs)
    obs_data, _ = agent.observation_process(env_obs)

    ep_log_path = session_dir / "episodes" / f"{round_name}_map{map_id}.jsonl"
    ep_log_path.parent.mkdir(parents=True, exist_ok=True)
    ep_log_file = open(ep_log_path, "w", encoding="utf-8")

    fm = agent.preprocessor
    step_records = []
    recent_positions = []
    done = False
    step = 0
    total_reward = 0.0
    zero_progress_streak = 0
    policy_mode = _benchmark_policy_mode()
    use_eval_policy = policy_mode == "eval"
    terminated = False
    truncated = False

    while not done:
        act_data = agent.predict([obs_data], use_hard_override=use_eval_policy)[0]
        act = agent.action_process(act_data, is_stochastic=not use_eval_policy)
        action_idx = int(np.asarray(getattr(act_data, "action", [-1])).reshape(-1)[0])
        selected_action = int(act)

        _, env_obs = env.step(act)
        env_obs = _inject_agent_runtime(env_obs, agent)
        if handle_disaster_recovery(env_obs, logger):
            terminated = bool(env_obs.get("terminated", False))
            truncated = bool(env_obs.get("truncated", False))
            done = True
            break

        terminated = env_obs["terminated"]
        truncated = env_obs["truncated"]
        step += 1
        done = terminated or truncated

        reward_payload = _normalize_reward_payload(getattr(agent, "last_reward", 0.0))
        reward_scalar = float(reward_payload["reward_total"])
        total_reward += reward_scalar

        cur_pos = tuple(int(v) for v in getattr(fm, "cur_pos", (0, 0)))
        prev_anchor_dist = float(step_records[-1]["anchor_return_dist"]) if step_records else float(
            getattr(fm, "anchor_return_dist", 0.0)
        )
        prev_recoverability = float(step_records[-1]["future_recoverability_score"]) if step_records else float(
            getattr(fm, "future_recoverability_score", 0.0)
        )
        prev_slack = float(step_records[-1]["charger_slack"]) if step_records else float(getattr(fm, "charger_slack", 0.0))

        anchor_return_dist = round(float(getattr(fm, "anchor_return_dist", 0.0)), 4)
        future_recoverability = round(float(getattr(fm, "future_recoverability_score", 0.0)), 4)
        charger_slack = round(float(getattr(fm, "charger_slack", 0.0)), 2)
        cleaned_this_step = int(getattr(fm, "cleaned_this_step", 0))
        dirty_adjacent = int(getattr(fm, "dirty_adjacent", 0))
        wall_adjacent = int(getattr(fm, "wall_adjacent", 0))
        local_frontier_density = round(float(getattr(fm, "local_frontier_density", 0.0)), 4)
        cur_visit_count = int(getattr(fm, "cur_visit_count", 0))
        revisit_pressure = float(np.clip((cur_visit_count - 1) / 6.0, 0.0, 1.0))
        target_progress_delta = round(prev_anchor_dist - anchor_return_dist, 4)
        recoverability_delta = round(future_recoverability - prev_recoverability, 4)
        charger_slack_delta = round(charger_slack - prev_slack, 4)
        no_progress_step = cleaned_this_step == 0 and target_progress_delta <= 0.0 and recoverability_delta <= 0.0
        zero_progress_streak = zero_progress_streak + 1 if no_progress_step else 0

        recent_positions.append(cur_pos)
        position_repeat_8 = round(_repeat_ratio(recent_positions, 8), 4)
        position_repeat_16 = round(_repeat_ratio(recent_positions, 16), 4)
        small_cycle_4 = 1.0 if _has_cycle(recent_positions, 4) else 0.0
        small_cycle_8 = 1.0 if _has_cycle(recent_positions, 8) else 0.0
        same_region_streak = _same_region_streak(recent_positions)
        corner_proximity = round(float(np.clip(getattr(fm, "wall_adjacent", 0) / 4.0, 0.0, 1.0)), 4)
        path_cross_count_50 = _path_cross_count(recent_positions, 50)
        local_unique_cells_20 = len(set(recent_positions[-20:]))
        coverage_efficiency_20 = _coverage_efficiency(step_records + [{"dirt_cleaned": int(fm.dirt_cleaned)}], 20)
        wall_follow_streak = 0
        for rec in reversed(step_records):
            if rec.get("wall_adjacent", 0) > 0:
                wall_follow_streak += 1
            else:
                break
        if wall_adjacent > 0:
            wall_follow_streak += 1

        guidance = fm._get_guidance() if hasattr(fm, "_get_guidance") else {}
        charger_path = guidance.get("charger_path") or []
        path_source = _path_source(guidance)
        fallback_to_chebyshev = 1.0 if path_source == "fallback_chebyshev" else 0.0
        planner_suggested_action = int(guidance.get("suggested_action", -1) if guidance.get("suggested_action") is not None else -1)
        planner_suggested_action_legal = 1.0 if guidance.get("suggested_action_legal", False) else 0.0
        planner_action_margin = round(float(guidance.get("action_margin", 0.0)), 4)
        planner_target_gap = round(float(guidance.get("target_gap", 0.0)), 4)
        planner_unknown_path_ratio = round(float(guidance.get("unknown_path_ratio", 0.0)), 4)
        planner_target_stable = 1.0 if guidance.get("target_stable", False) else 0.0
        planner_anchor_stable = 1.0 if guidance.get("anchor_stable", False) else 0.0
        planner_signal_reachable = 1.0 if guidance.get("reachable", False) else 0.0
        planner_action_match = planner_suggested_action < 0 or planner_suggested_action == selected_action
        decision_route_anchor = (
            int(np.asarray(getattr(act_data, "route_anchor", 0)).reshape(-1)[0]) if hasattr(act_data, "route_anchor") else 0
        )
        decision_target = (
            int(np.asarray(getattr(act_data, "target", 0)).reshape(-1)[0]) if hasattr(act_data, "target") else 0
        )
        nearest_charger_center = (
            tuple(fm.sorted_charger_candidates[0]["center"]) if getattr(fm, "sorted_charger_candidates", None) else None
        )
        target_center = tuple(guidance.get("charger_target")) if guidance.get("charger_target") is not None else None
        target_charger_robot_count = _count_npcs_near(target_center, fm, radius=2)
        nearest_charger_robot_count = _count_npcs_near(nearest_charger_center, fm, radius=2)
        robot_on_target_path_count = _count_npcs_on_path(charger_path, fm, radius=1)
        target_charger_contested = 1.0 if target_charger_robot_count > 0 else 0.0
        local_passage_width = _local_passage_width(fm, selected_action)
        narrow_passage_flag = 1.0 if 0 < local_passage_width <= 2 else 0.0
        nearest_charger_dist = float(getattr(fm, "nearest_charger_dist", 0.0))
        charge_margin_now = round(float(charger_slack), 4)
        reserve = max(8.0, 0.04 * max(float(getattr(fm, "battery_max", 1)), 1.0))
        min_margin_any_charger = 0.0
        if getattr(fm, "sorted_charger_candidates", None):
            best_dist = min(float(cand.get("astar_dist", float("inf"))) for cand in fm.sorted_charger_candidates)
            min_margin_any_charger = round(float(getattr(fm, "battery", 0.0) - best_dist - reserve), 4)
        missed_charge_opportunity = (
            nearest_charger_dist <= 1.0
            and float(getattr(fm, "battery", 0.0)) <= 0.35 * max(float(getattr(fm, "battery_max", 1.0)), 1.0)
            and not bool(getattr(fm, "just_charged", False))
        )
        charger_nearby_not_charged = (
            nearest_charger_dist <= 2.0
            and float(getattr(fm, "battery", 0.0)) <= 0.25 * max(float(getattr(fm, "battery_max", 1.0)), 1.0)
            and not bool(getattr(fm, "just_charged", False))
        )
        target_selection_gap = 0.0
        selected_target_rank = int(decision_target)
        best_astar_charger_idx = 0
        best_cheb_charger_idx = 0
        charger_candidates = _charger_snapshot(getattr(fm, "sorted_charger_candidates", []), limit=4)
        if getattr(fm, "sorted_charger_candidates", None):
            best = fm.sorted_charger_candidates[0]
            best_astar_charger_idx = 1
            best_cheb_cand = min(fm.sorted_charger_candidates, key=lambda cand: float(cand.get("dist", float("inf"))))
            best_cheb_charger_idx = next(
                (idx for idx, cand in enumerate(fm.sorted_charger_candidates, start=1) if cand["center"] == best_cheb_cand["center"]),
                0,
            )
            if 1 <= decision_target <= len(fm.sorted_charger_candidates):
                current_cand = fm.sorted_charger_candidates[decision_target - 1]
                target_selection_gap = round(float(current_cand.get("astar_dist", 0.0) - best.get("astar_dist", 0.0)), 4)
        all_charger_known_path_count = 1 if charger_path else 0
        unknown_on_target_path_ratio = planner_unknown_path_ratio
        retarget_event = 1.0 if step_records and decision_target > 0 and int(step_records[-1].get("target", 0)) != decision_target else 0.0

        action_probs = _to_prob_array(getattr(act_data, "prob", []))
        mode_probs = _to_prob_array(getattr(act_data, "mode_prob", []))
        target_probs = _to_prob_array(getattr(act_data, "target_prob", []))
        anchor_probs = _to_prob_array(getattr(act_data, "route_anchor_prob", []))
        action_topk = _topk_summary(action_probs, TOPK)
        mode_topk = _topk_summary(mode_probs, TOPK)
        target_topk = _topk_summary(target_probs, TOPK)
        anchor_topk = _topk_summary(anchor_probs, TOPK)

        anomalies = _compute_step_anomalies(
            cur_visit_count=cur_visit_count,
            cleaned_this_step=cleaned_this_step,
            dirty_adjacent=dirty_adjacent,
            reward_total=reward_scalar,
            reward_clean=float(reward_payload["reward_clean"]),
            reward_explore=float(reward_payload["reward_explore"]),
            reward_frontier=float(reward_payload["reward_frontier"]),
            current_mode=int(getattr(fm, "current_mode", -1)),
            target_progress_delta=target_progress_delta,
            recoverability_delta=recoverability_delta,
            position_repeat_8=position_repeat_8,
            position_repeat_16=position_repeat_16,
            small_cycle_4=small_cycle_4,
            small_cycle_8=small_cycle_8,
            corner_proximity=corner_proximity,
            zero_progress_streak=zero_progress_streak,
            wall_follow_streak=wall_follow_streak,
            wall_adjacent=wall_adjacent,
            local_frontier_density=local_frontier_density,
            narrow_passage_flag=narrow_passage_flag,
            unknown_on_target_path_ratio=unknown_on_target_path_ratio,
            missed_charge_opportunity=missed_charge_opportunity,
            charger_nearby_not_charged=charger_nearby_not_charged,
            planner_action_match=planner_action_match,
            target_selection_gap=target_selection_gap,
            target_charger_contested=target_charger_contested > 0,
        )

        step_rec = {
            "step": step,
            "x": cur_pos[0],
            "z": cur_pos[1],
            "action": selected_action,
            "sampled_action": action_idx,
            "greedy_action": int(np.asarray(getattr(act_data, "d_action", [selected_action])).reshape(-1)[0]),
            "reward": round(reward_scalar, 4),
            "total_reward": round(total_reward, 4),
            "battery": int(fm.battery),
            "battery_max": int(fm.battery_max),
            "dirt_cleaned": int(fm.dirt_cleaned),
            "total_dirt": int(fm.total_dirt),
            "mode": int(fm.current_mode),
            "route_anchor": decision_route_anchor,
            "target": decision_target,
            "charger_slack": charger_slack,
            "future_recoverability_score": future_recoverability,
            "anchor_return_dist": anchor_return_dist,
            "is_diag_action": 1.0 if selected_action in (1, 3, 5, 7) else 0.0,
            "nearest_npc_dist": round(float(fm.nearest_npc_dist), 1),
            "invalid_move_count": int(fm.invalid_move_count),
            "cur_visit_count": cur_visit_count,
            "revisit_pressure": round(revisit_pressure, 4),
            "cleaned_this_step": cleaned_this_step,
            "dirty_adjacent": dirty_adjacent,
            "wall_adjacent": wall_adjacent,
            "local_frontier_density": local_frontier_density,
            "zero_progress_streak": zero_progress_streak,
            "position_repeat_8": position_repeat_8,
            "position_repeat_16": position_repeat_16,
            "small_cycle_4": small_cycle_4,
            "small_cycle_8": small_cycle_8,
            "same_region_streak": same_region_streak,
            "wall_follow_streak": wall_follow_streak,
            "local_unique_cells_20": local_unique_cells_20,
            "path_cross_count_50": path_cross_count_50,
            "coverage_efficiency_20": coverage_efficiency_20,
            "corner_proximity": corner_proximity,
            "target_progress_delta": target_progress_delta,
            "recoverability_delta": recoverability_delta,
            "charger_slack_delta": charger_slack_delta,
            "local_passage_width": local_passage_width,
            "narrow_passage_flag": narrow_passage_flag,
            "path_source": path_source,
            "fallback_to_chebyshev": fallback_to_chebyshev,
            "planner_suggested_action": planner_suggested_action,
            "planner_suggested_action_legal": planner_suggested_action_legal,
            "planner_action_margin": planner_action_margin,
            "planner_target_gap": planner_target_gap,
            "planner_unknown_path_ratio": planner_unknown_path_ratio,
            "planner_target_stable": planner_target_stable,
            "planner_anchor_stable": planner_anchor_stable,
            "planner_signal_reachable": planner_signal_reachable,
            "action_vs_planner_match": 1.0 if planner_action_match else 0.0,
            "charge_margin_now": charge_margin_now,
            "min_margin_any_charger": min_margin_any_charger,
            "nearest_charger_dist": round(nearest_charger_dist, 4),
            "missed_charge_opportunity": 1.0 if missed_charge_opportunity else 0.0,
            "charger_nearby_not_charged": 1.0 if charger_nearby_not_charged else 0.0,
            "target_selection_gap": target_selection_gap,
            "selected_target_rank": selected_target_rank,
            "best_astar_charger_idx": best_astar_charger_idx,
            "best_cheb_charger_idx": best_cheb_charger_idx,
            "all_charger_known_path_count": all_charger_known_path_count,
            "unknown_on_target_path_ratio": unknown_on_target_path_ratio,
            "target_charger_robot_count": target_charger_robot_count,
            "nearest_charger_robot_count": nearest_charger_robot_count,
            "robot_on_target_path_count": robot_on_target_path_count,
            "target_charger_contested": target_charger_contested,
            "retarget_event": retarget_event,
            "charger_candidates": charger_candidates,
            "reward_clean": round(float(reward_payload["reward_clean"]), 4),
            "reward_survive": round(float(reward_payload["reward_survive"]), 4),
            "reward_cleaning": round(float(reward_payload["reward_cleaning"]), 4),
            "reward_streak": round(float(reward_payload["reward_streak"]), 4),
            "reward_explore": round(float(reward_payload["reward_explore"]), 4),
            "reward_frontier": round(float(reward_payload["reward_frontier"]), 4),
            "reward_recoverability": round(float(reward_payload["reward_recoverability"]), 4),
            "reward_charge": round(float(reward_payload["reward_charge"]), 4),
            "reward_npc": round(float(reward_payload["reward_npc"]), 4),
            "reward_stuck": round(float(reward_payload["reward_stuck"]), 4),
            "reward_idle": round(float(reward_payload["reward_idle"]), 4),
            "reward_anchor_consistency": round(float(reward_payload["reward_anchor_consistency"]), 4),
            "reward_sticky_anchor_penalty": round(float(reward_payload["reward_sticky_anchor_penalty"]), 4),
            "reward_return_progress": round(float(reward_payload["reward_return_progress"]), 4),
            "reward_diag_efficiency": round(float(reward_payload["reward_diag_efficiency"]), 4),
            "reward_return_stall": round(float(reward_payload["reward_return_stall"]), 4),
            "reward_late_contract": round(float(reward_payload["reward_late_contract"]), 4),
            "reward_astar_potential": round(float(reward_payload["reward_astar_potential"]), 4),
            "reward_charge_margin_pressure": round(float(reward_payload["reward_charge_margin_pressure"]), 4),
            "reward_unknown_path_risk": round(float(reward_payload["reward_unknown_path_risk"]), 4),
            "reward_missed_charge_penalty": round(float(reward_payload["reward_missed_charge_penalty"]), 4),
            "reward_planner_alignment": round(float(reward_payload["reward_planner_alignment"]), 4),
            "reward_cleaning_context_scale": round(float(reward_payload["reward_cleaning_context_scale"]), 4),
            "reward_cps_bonus": round(float(reward_payload["reward_cps_bonus"]), 4),
            "action_top1": action_topk[0]["index"],
            "action_prob_top1": action_topk[0]["prob"],
            "action_top2": action_topk[1]["index"],
            "action_prob_top2": action_topk[1]["prob"],
            "action_entropy": _entropy(action_probs),
            "mode_top1": mode_topk[0]["index"],
            "mode_prob_top1": mode_topk[0]["prob"],
            "mode_top2": mode_topk[1]["index"],
            "mode_prob_top2": mode_topk[1]["prob"],
            "target_top1": target_topk[0]["index"],
            "target_prob_top1": target_topk[0]["prob"],
            "target_top2": target_topk[1]["index"],
            "target_prob_top2": target_topk[1]["prob"],
            "route_anchor_top1": anchor_topk[0]["index"],
            "route_anchor_prob_top1": anchor_topk[0]["prob"],
            "route_anchor_top2": anchor_topk[1]["index"],
            "route_anchor_prob_top2": anchor_topk[1]["prob"],
            "state": {
                "battery": int(fm.battery),
                "battery_max": int(fm.battery_max),
                "charger_slack": charger_slack,
                "future_recoverability_score": future_recoverability,
                "anchor_return_dist": anchor_return_dist,
                "nearest_npc_dist": round(float(fm.nearest_npc_dist), 1),
                "cur_visit_count": cur_visit_count,
                "dirty_adjacent": dirty_adjacent,
                "cleaned_this_step": cleaned_this_step,
                "wall_adjacent": wall_adjacent,
                "local_frontier_density": local_frontier_density,
                "local_passage_width": local_passage_width,
                "charge_margin_now": charge_margin_now,
                "min_margin_any_charger": min_margin_any_charger,
                "target_charger_robot_count": target_charger_robot_count,
                "all_charger_known_path_count": all_charger_known_path_count,
                "unknown_on_target_path_ratio": unknown_on_target_path_ratio,
            },
            "decision": {
                "action": selected_action,
                "policy_mode": policy_mode,
                "mode": int(fm.current_mode),
                "route_anchor": decision_route_anchor,
                "target": decision_target,
                "selected_target_rank": selected_target_rank,
                "best_astar_charger_idx": best_astar_charger_idx,
                "best_cheb_charger_idx": best_cheb_charger_idx,
                "target_selection_gap": target_selection_gap,
                "action_top1": action_topk[0],
                "action_top2": action_topk[1],
                "mode_top1": mode_topk[0],
                "mode_top2": mode_topk[1],
                "target_top1": target_topk[0],
                "target_top2": target_topk[1],
                "route_anchor_top1": anchor_topk[0],
                "route_anchor_top2": anchor_topk[1],
                "action_entropy": _entropy(action_probs),
                "planner_suggested_action": planner_suggested_action,
                "planner_suggested_action_legal": planner_suggested_action_legal,
                "planner_action_margin": planner_action_margin,
                "planner_target_gap": planner_target_gap,
                "planner_unknown_path_ratio": planner_unknown_path_ratio,
                "planner_target_stable": planner_target_stable,
                "planner_anchor_stable": planner_anchor_stable,
                "planner_signal_reachable": planner_signal_reachable,
                "action_vs_planner_match": 1.0 if planner_action_match else 0.0,
                "path_source": path_source,
                "all_charger_known_path_count": all_charger_known_path_count,
            },
            "reward_breakdown": {
                "total": round(float(reward_payload["reward_total"]), 4),
                "clean": round(float(reward_payload["reward_clean"]), 4),
                "survive": round(float(reward_payload["reward_survive"]), 4),
                "components": {key: round(float(reward_payload[f"reward_{key}"]), 4) for key in REWARD_COMPONENT_KEYS},
            },
            "behavior": {
                "is_diag_action": 1.0 if selected_action in (1, 3, 5, 7) else 0.0,
                "invalid_move_count": int(fm.invalid_move_count),
                "zero_progress_streak": int(zero_progress_streak),
                "position_repeat_8": position_repeat_8,
                "position_repeat_16": position_repeat_16,
                "small_cycle_4": small_cycle_4,
                "small_cycle_8": small_cycle_8,
                "wall_follow_streak": wall_follow_streak,
                "local_unique_cells_20": local_unique_cells_20,
                "path_cross_count_50": path_cross_count_50,
                "coverage_efficiency_20": coverage_efficiency_20,
                "corner_proximity": corner_proximity,
                "target_progress_delta": target_progress_delta,
                "recoverability_delta": recoverability_delta,
                "charger_slack_delta": charger_slack_delta,
                "local_passage_width": local_passage_width,
                "narrow_passage_flag": narrow_passage_flag,
                "target_charger_robot_count": target_charger_robot_count,
                "nearest_charger_robot_count": nearest_charger_robot_count,
                "robot_on_target_path_count": robot_on_target_path_count,
                "target_charger_contested": target_charger_contested,
            },
            "anomalies": anomalies,
        }

        ep_log_file.write(json.dumps(step_rec, ensure_ascii=False) + "\n")
        step_records.append(step_rec)

        if step % 100 == 0 or done:
            step_log.info(
                f"{ep_label} step={step} bat={fm.battery}/{fm.battery_max} "
                f"dirt={fm.dirt_cleaned}/{fm.total_dirt} mode={fm.current_mode} "
                f"slack={fm.charger_slack:.1f} npc={fm.nearest_npc_dist:.0f} "
                f"act={selected_action} reward={reward_scalar:.3f}"
            )

        if not done:
            obs_data, _ = agent.observation_process(env_obs)

    ep_log_file.close()

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
    diagnostics = _sequence_diagnostics(step_records)
    anomaly_summary = _anomaly_summary(step_records)
    reward_attribution = _reward_attribution(step_records)
    evidence_windows = _extract_evidence_windows(step_records)
    episode_id = f"{round_name}_map{map_id}"

    result = {
        "episode_id": episode_id,
        "round": round_name,
        "map_id": map_id,
        "result": fail_reason,
        "clean_score": clean_score,
        "steps": finished_steps,
        "charge_count": charge_count,
        "remaining_charge": remaining_charge,
        "total_reward": round(total_reward, 3),
        "dirt_cleaned": int(fm.dirt_cleaned),
        "total_dirt": int(fm.total_dirt),
        "dirt_ratio": round(fm.dirt_cleaned / max(fm.total_dirt, 1), 4),
        "invalid_move_count": int(fm.invalid_move_count),
        "invalid_move_rate": round(fm.invalid_move_count / max(step, 1), 4),
        "late_return_rate": round(diagnostics["late_return_rate"], 4),
        "late_contract_rate": round(diagnostics["late_contract_rate"], 4),
        "anchor_switch_rate": round(diagnostics["anchor_switch_rate"], 4),
        "target_switch_rate": round(diagnostics["target_switch_rate"], 4),
        "diag_rate_all": round(diagnostics["diag_rate_all"], 4),
        "diag_rate_contract": round(diagnostics["diag_rate_contract"], 4),
        "diag_rate_return": round(diagnostics["diag_rate_return"], 4),
        "return_progress_per_step": round(diagnostics["return_progress_per_step"], 4),
        "return_efficiency_ratio": round(diagnostics["return_efficiency_ratio"], 4),
        "return_stall_rate": round(diagnostics["return_stall_rate"], 4),
        "recoverability_score_avg": round(diagnostics["recoverability_score_avg"], 4),
        "recoverability_violation_rate": round(diagnostics["recoverability_violation_rate"], 4),
        "mode_usage_depart": round(diagnostics["mode_usage_depart"], 4),
        "mode_usage_expand": round(diagnostics["mode_usage_expand"], 4),
        "mode_usage_harvest": round(diagnostics["mode_usage_harvest"], 4),
        "mode_usage_contract": round(diagnostics["mode_usage_contract"], 4),
        "mode_usage_return": round(diagnostics["mode_usage_return"], 4),
        "mode_usage_evade": round(diagnostics["mode_usage_evade"], 4),
        "summary": {
            "result": fail_reason,
            "clean_score": clean_score,
            "steps": finished_steps,
            "charge_count": charge_count,
            "remaining_charge": remaining_charge,
            "total_reward": round(total_reward, 3),
        },
        "diagnostics": diagnostics,
        "anomaly_summary": anomaly_summary,
        "reward_attribution": reward_attribution,
        "evidence_windows": evidence_windows,
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
    grouped = {}
    for episode in episode_results:
        grouped.setdefault(episode["round"], []).append(episode)

    per_round = {}
    for round_name, episodes in grouped.items():
        wins = [ep for ep in episodes if ep["result"] == "completed"]
        fails_battery = [ep for ep in episodes if ep["result"] == "battery"]
        fails_collision = [ep for ep in episodes if ep["result"] == "collision"]
        per_round[round_name] = {
            "win_rate": round(len(wins) / len(episodes), 4) if episodes else 0.0,
            "avg_clean_score": round(sum(ep["clean_score"] for ep in episodes) / len(episodes), 1) if episodes else 0.0,
            "avg_steps": round(sum(ep["steps"] for ep in episodes) / len(episodes), 1) if episodes else 0.0,
            "avg_charge_count": round(sum(ep["charge_count"] for ep in episodes) / len(episodes), 2) if episodes else 0.0,
            "avg_dirt_ratio": round(sum(ep["dirt_ratio"] for ep in episodes) / len(episodes), 4) if episodes else 0.0,
            "battery_fail_rate": round(len(fails_battery) / len(episodes), 4) if episodes else 0.0,
            "collision_fail_rate": round(len(fails_collision) / len(episodes), 4) if episodes else 0.0,
            "avg_invalid_move_rate": round(sum(ep["invalid_move_rate"] for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "late_return_rate": round(sum(ep.get("late_return_rate", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "late_contract_rate": round(sum(ep.get("late_contract_rate", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "anchor_switch_rate": round(sum(ep.get("anchor_switch_rate", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "target_switch_rate": round(sum(ep.get("target_switch_rate", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "diag_rate_all": round(sum(ep.get("diag_rate_all", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "diag_rate_contract": round(sum(ep.get("diag_rate_contract", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "diag_rate_return": round(sum(ep.get("diag_rate_return", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "return_progress_per_step": round(
                sum(ep.get("return_progress_per_step", 0.0) for ep in episodes) / len(episodes), 4
            )
            if episodes
            else 0.0,
            "return_efficiency_ratio": round(
                sum(ep.get("return_efficiency_ratio", 0.0) for ep in episodes) / len(episodes), 4
            )
            if episodes
            else 0.0,
            "return_stall_rate": round(sum(ep.get("return_stall_rate", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "recoverability_score_avg": round(
                sum(ep.get("recoverability_score_avg", 0.0) for ep in episodes) / len(episodes), 4
            )
            if episodes
            else 0.0,
            "recoverability_violation_rate": round(
                sum(ep.get("recoverability_violation_rate", 0.0) for ep in episodes) / len(episodes), 4
            )
            if episodes
            else 0.0,
            "mode_usage_depart": round(sum(ep.get("mode_usage_depart", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "mode_usage_expand": round(sum(ep.get("mode_usage_expand", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "mode_usage_harvest": round(sum(ep.get("mode_usage_harvest", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "mode_usage_contract": round(sum(ep.get("mode_usage_contract", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "mode_usage_return": round(sum(ep.get("mode_usage_return", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "mode_usage_evade": round(sum(ep.get("mode_usage_evade", 0.0) for ep in episodes) / len(episodes), 4)
            if episodes
            else 0.0,
            "anomaly_summary": _aggregate_anomaly_summary(episodes),
            "reward_attribution": _aggregate_reward_attribution(episodes),
            "episode_count": len(episodes),
            "win_episode_count": len(wins),
        }

    wins = [ep for ep in episode_results if ep["result"] == "completed"]
    fails_battery = [ep for ep in episode_results if ep["result"] == "battery"]
    fails_collision = [ep for ep in episode_results if ep["result"] == "collision"]
    overall = {
        "win_rate": round(len(wins) / len(episode_results), 4) if episode_results else 0.0,
        "completed_rate": round(len(wins) / len(episode_results), 4) if episode_results else 0.0,
        "broad_win_rate": round(len(wins) / len(episode_results), 4) if episode_results else 0.0,
        "avg_clean_score": round(sum(ep["clean_score"] for ep in episode_results) / len(episode_results), 1)
        if episode_results
        else 0.0,
        "avg_steps": round(sum(ep["steps"] for ep in episode_results) / len(episode_results), 1) if episode_results else 0.0,
        "avg_charge_count": round(sum(ep["charge_count"] for ep in episode_results) / len(episode_results), 2)
        if episode_results
        else 0.0,
        "battery_fail_rate": round(len(fails_battery) / len(episode_results), 4) if episode_results else 0.0,
        "collision_fail_rate": round(len(fails_collision) / len(episode_results), 4) if episode_results else 0.0,
        "avg_invalid_move_rate": round(sum(ep["invalid_move_rate"] for ep in episode_results) / len(episode_results), 4)
        if episode_results
        else 0.0,
        "late_return_rate": round(sum(ep.get("late_return_rate", 0.0) for ep in episode_results) / len(episode_results), 4)
        if episode_results
        else 0.0,
        "late_contract_rate": round(
            sum(ep.get("late_contract_rate", 0.0) for ep in episode_results) / len(episode_results), 4
        )
        if episode_results
        else 0.0,
        "anchor_switch_rate": round(
            sum(ep.get("anchor_switch_rate", 0.0) for ep in episode_results) / len(episode_results), 4
        )
        if episode_results
        else 0.0,
        "target_switch_rate": round(
            sum(ep.get("target_switch_rate", 0.0) for ep in episode_results) / len(episode_results), 4
        )
        if episode_results
        else 0.0,
        "diag_rate_all": round(sum(ep.get("diag_rate_all", 0.0) for ep in episode_results) / len(episode_results), 4)
        if episode_results
        else 0.0,
        "diag_rate_contract": round(
            sum(ep.get("diag_rate_contract", 0.0) for ep in episode_results) / len(episode_results), 4
        )
        if episode_results
        else 0.0,
        "diag_rate_return": round(
            sum(ep.get("diag_rate_return", 0.0) for ep in episode_results) / len(episode_results), 4
        )
        if episode_results
        else 0.0,
        "return_progress_per_step": round(
            sum(ep.get("return_progress_per_step", 0.0) for ep in episode_results) / len(episode_results), 4
        )
        if episode_results
        else 0.0,
        "return_efficiency_ratio": round(
            sum(ep.get("return_efficiency_ratio", 0.0) for ep in episode_results) / len(episode_results), 4
        )
        if episode_results
        else 0.0,
        "return_stall_rate": round(
            sum(ep.get("return_stall_rate", 0.0) for ep in episode_results) / len(episode_results), 4
        )
        if episode_results
        else 0.0,
        "recoverability_score_avg": round(
            sum(ep.get("recoverability_score_avg", 0.0) for ep in episode_results) / len(episode_results), 4
        )
        if episode_results
        else 0.0,
        "recoverability_violation_rate": round(
            sum(ep.get("recoverability_violation_rate", 0.0) for ep in episode_results) / len(episode_results), 4
        )
        if episode_results
        else 0.0,
        "anomaly_summary": _aggregate_anomaly_summary(episode_results),
        "reward_attribution": _aggregate_reward_attribution(episode_results),
        "episode_count": len(episode_results),
        "win_episode_count": len(wins),
    }

    return {"per_round": per_round, "overall": overall}


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
    overall = aggregated["overall"]
    parts = " | ".join(
        f"{name}: WR={metrics['win_rate']:.0%} CS={metrics['avg_clean_score']:.0f} "
        f"bat_fail={metrics['battery_fail_rate']:.0%} col_fail={metrics['collision_fail_rate']:.0%}"
        for name, metrics in aggregated["per_round"].items()
    )
    logger.info(f"[BENCHMARK] ========== Done ({elapsed:.0f}s) ==========")
    logger.info(
        f"[BENCHMARK] Overall WR={overall['win_rate']:.0%} CS={overall['avg_clean_score']:.0f} "
        f"({overall['win_episode_count']}/{overall['episode_count']}) "
        f"late_return={overall.get('late_return_rate', 0.0):.0%} "
        f"target_switch={overall.get('target_switch_rate', 0.0):.0%}"
    )
    anomaly = overall.get("anomaly_summary", {})
    logger.info(
        f"[BENCHMARK] Anomaly loop_ep={anomaly.get('loop_episode_rate', 0.0):.0%} "
        f"corner_loop_ep={anomaly.get('corner_loop_episode_rate', 0.0):.0%} "
        f"clean_floor_revisit={anomaly.get('avg_revisit_on_clean_floor_rate', 0.0):.0%} "
        f"pos_reward_no_progress={anomaly.get('avg_positive_reward_while_no_progress_rate', 0.0):.0%}"
    )
    logger.info(f"[BENCHMARK] {parts}")


def _sequence_diagnostics(step_records):
    if not step_records:
        return {
            "late_return_rate": 0.0,
            "late_contract_rate": 0.0,
            "anchor_switch_rate": 0.0,
            "target_switch_rate": 0.0,
            "diag_rate_all": 0.0,
            "diag_rate_contract": 0.0,
            "diag_rate_return": 0.0,
            "return_progress_per_step": 0.0,
            "return_efficiency_ratio": 0.0,
            "return_stall_rate": 0.0,
            "recoverability_score_avg": 0.0,
            "recoverability_violation_rate": 0.0,
            "mode_usage_depart": 0.0,
            "mode_usage_expand": 0.0,
            "mode_usage_harvest": 0.0,
            "mode_usage_contract": 0.0,
            "mode_usage_return": 0.0,
            "mode_usage_evade": 0.0,
        }

    modes = [int(rec.get("mode", -1)) for rec in step_records]
    targets = [int(rec.get("target", 0)) for rec in step_records]
    anchors = [int(rec.get("route_anchor", 0)) for rec in step_records]
    slacks = [float(rec.get("charger_slack", 0.0)) for rec in step_records]
    recoverability = [float(rec.get("future_recoverability_score", 0.0)) for rec in step_records]
    anchor_dists = [float(rec.get("anchor_return_dist", 0.0)) for rec in step_records]
    diag_actions = [float(rec.get("is_diag_action", 0.0)) for rec in step_records]
    total = float(len(step_records))

    target_steps = [target for target in targets if target > 0]
    target_switches = sum(1 for lhs, rhs in zip(target_steps, target_steps[1:]) if lhs != rhs)
    target_switch_rate = target_switches / max(len(target_steps) - 1, 1)
    anchor_steps = [anchor for anchor in anchors if anchor > 0]
    anchor_switches = sum(1 for lhs, rhs in zip(anchor_steps, anchor_steps[1:]) if lhs != rhs)
    anchor_switch_rate = anchor_switches / max(len(anchor_steps) - 1, 1)

    first_contract_idx = next((idx for idx, mode in enumerate(modes) if mode in (3, 4)), None)
    late_contract_rate = 1.0 if first_contract_idx is not None and recoverability[first_contract_idx] < 0.0 else 0.0
    first_return_idx = next((idx for idx, mode in enumerate(modes) if mode == 4), None)
    late_return_rate = 1.0 if first_return_idx is None else (1.0 if slacks[first_return_idx] < 0.0 else 0.0)

    contract_steps = [idx for idx, mode in enumerate(modes) if mode == 3]
    return_steps = [idx for idx, mode in enumerate(modes) if mode == 4]
    diag_rate_all = sum(diag_actions) / total
    diag_rate_contract = sum(diag_actions[idx] for idx in contract_steps) / max(len(contract_steps), 1)
    diag_rate_return = sum(diag_actions[idx] for idx in return_steps) / max(len(return_steps), 1)

    route_phase_steps = [idx for idx, mode in enumerate(modes) if mode in (3, 4)]
    progress_deltas = [
        anchor_dists[prev_idx] - anchor_dists[cur_idx]
        for prev_idx, cur_idx in zip(route_phase_steps, route_phase_steps[1:])
    ]
    return_progress_per_step = float(sum(progress_deltas) / max(len(progress_deltas), 1))
    return_efficiency_ratio = float(
        (anchor_dists[route_phase_steps[0]] / max(len(route_phase_steps), 1))
        if route_phase_steps and anchor_dists[route_phase_steps[0]] > 0.0
        else 0.0
    )
    return_stall_rate = float(sum(1 for delta in progress_deltas if delta <= 0.0) / max(len(progress_deltas), 1))

    return {
        "late_return_rate": float(late_return_rate),
        "late_contract_rate": float(late_contract_rate),
        "anchor_switch_rate": float(anchor_switch_rate),
        "target_switch_rate": float(target_switch_rate),
        "diag_rate_all": float(diag_rate_all),
        "diag_rate_contract": float(diag_rate_contract),
        "diag_rate_return": float(diag_rate_return),
        "return_progress_per_step": float(return_progress_per_step),
        "return_efficiency_ratio": float(return_efficiency_ratio),
        "return_stall_rate": float(return_stall_rate),
        "recoverability_score_avg": float(sum(recoverability) / max(len(recoverability), 1)),
        "recoverability_violation_rate": float(sum(1 for value in recoverability if value < 0.0) / max(len(recoverability), 1)),
        "mode_usage_depart": sum(1 for mode in modes if mode == 0) / total,
        "mode_usage_expand": sum(1 for mode in modes if mode == 1) / total,
        "mode_usage_harvest": sum(1 for mode in modes if mode == 2) / total,
        "mode_usage_contract": sum(1 for mode in modes if mode == 3) / total,
        "mode_usage_return": sum(1 for mode in modes if mode == 4) / total,
        "mode_usage_evade": sum(1 for mode in modes if mode == 5) / total,
    }


def _normalize_reward_payload(reward):
    if isinstance(reward, dict):
        reward_total = float(
            reward.get(
                "reward_total",
                reward.get("total", reward.get("reward_clean", 0.0) + reward.get("reward_survive", 0.0)),
            )
        )
        payload = {
            "reward_total": reward_total,
            "reward_clean": float(reward.get("reward_clean", reward.get("clean", 0.0))),
            "reward_survive": float(reward.get("reward_survive", reward.get("survive", 0.0))),
        }
        for key in REWARD_COMPONENT_KEYS:
            payload[f"reward_{key}"] = float(reward.get(key, 0.0))
        return payload

    reward_scalar = float(reward)
    payload = {
        "reward_total": reward_scalar,
        "reward_clean": reward_scalar,
        "reward_survive": 0.0,
    }
    for key in REWARD_COMPONENT_KEYS:
        payload[f"reward_{key}"] = 0.0
    return payload


def _to_prob_array(value):
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.float32)
    arr = np.clip(arr, 0.0, None)
    total = float(arr.sum())
    if total > 0.0:
        arr = arr / total
    return arr


def _topk_summary(probs, k=2):
    if probs.size == 0:
        return [{"index": -1, "prob": 0.0} for _ in range(k)]
    idxs = np.argsort(probs)[::-1][:k]
    result = [{"index": int(idx), "prob": round(float(probs[idx]), 4)} for idx in idxs]
    while len(result) < k:
        result.append({"index": -1, "prob": 0.0})
    return result


def _entropy(probs):
    if probs.size == 0:
        return 0.0
    clipped = np.clip(probs, 1e-8, 1.0)
    return round(float(-(clipped * np.log(clipped)).sum()), 4)


def _repeat_ratio(positions, window):
    if not positions:
        return 0.0
    cur = positions[-1]
    history = positions[max(0, len(positions) - 1 - window) : -1]
    if not history:
        return 0.0
    return sum(1 for pos in history if pos == cur) / len(history)


def _has_cycle(positions, lag):
    return len(positions) > lag and positions[-1] == positions[-1 - lag]


def _same_region_streak(positions, radius=1):
    if not positions:
        return 0
    cur = positions[-1]
    streak = 0
    for pos in reversed(positions[:-1]):
        if max(abs(pos[0] - cur[0]), abs(pos[1] - cur[1])) <= radius:
            streak += 1
        else:
            break
    return streak


def _path_cross_count(positions, window=50):
    if not positions:
        return 0
    subset = positions[-window:]
    return max(len(subset) - len(set(subset)), 0)


def _coverage_efficiency(step_records, window=20):
    if not step_records:
        return 0.0
    start_idx = max(len(step_records) - window, 0)
    start_cleaned = int(step_records[start_idx]["dirt_cleaned"]) if step_records else 0
    end_cleaned = int(step_records[-1]["dirt_cleaned"]) if step_records else 0
    span = max(len(step_records) - start_idx, 1)
    return round(float(max(end_cleaned - start_cleaned, 0)) / float(span), 4)


def _npc_positions(fm):
    positions = []
    for npc in getattr(fm, "_npcs", []) or []:
        pos = npc.get("pos") or {}
        positions.append((int(pos.get("x", 0)), int(pos.get("z", 0))))
    return positions


def _count_npcs_near(center, fm, radius=2):
    if center is None:
        return 0
    cx, cz = center
    return sum(1 for x, z in _npc_positions(fm) if max(abs(x - cx), abs(z - cz)) <= radius)


def _count_npcs_on_path(path, fm, radius=1):
    if not path:
        return 0
    npc_positions = _npc_positions(fm)
    touched = 0
    for npc_x, npc_z in npc_positions:
        if any(max(abs(px - npc_x), abs(pz - npc_z)) <= radius for px, pz in path):
            touched += 1
    return touched


def _local_passage_width(fm, action):
    deltas = getattr(fm, "ACTION_DELTAS", None)
    passable_map = getattr(fm, "passable_map", None)
    cur_pos = getattr(fm, "cur_pos", None)
    if deltas is None or passable_map is None or cur_pos is None or action is None or action < 0:
        return 0
    if not (0 <= action < len(deltas)):
        return 0
    dx, dz = deltas[action]
    hx, hz = cur_pos
    nx, nz = hx + dx, hz + dz
    if not (0 <= nx < passable_map.shape[0] and 0 <= nz < passable_map.shape[1]):
        return 0
    if float(passable_map[nx, nz]) < 0.5:
        return 0
    pdx, pdz = -int(np.sign(dz)), int(np.sign(dx))
    width = 1
    for sign in (-1, 1):
        tx = nx + sign * pdx
        tz = nz + sign * pdz
        if 0 <= tx < passable_map.shape[0] and 0 <= tz < passable_map.shape[1] and float(passable_map[tx, tz]) >= 0.5:
            width += 1
    return width


def _charger_snapshot(candidates, limit=4):
    snapshot = []
    for cand in (candidates or [])[:limit]:
        snapshot.append(
            {
                "center": list(cand.get("center", (0, 0))),
                "dist": round(float(cand.get("dist", 0.0)), 4),
                "astar_dist": round(float(cand.get("astar_dist", 0.0)), 4),
                "priority": round(float(cand.get("priority", 0.0)), 4),
                "reachable": round(float(cand.get("reachable", 0.0)), 4),
            }
        )
    return snapshot


def _path_source(signal):
    path = signal.get("charger_path") or []
    if path:
        return "astar"
    charger_dist = float(signal.get("charger_dist", float("inf")))
    if np.isfinite(charger_dist):
        return "fallback_chebyshev"
    return "unreachable"


def _compute_step_anomalies(
    *,
    cur_visit_count,
    cleaned_this_step,
    dirty_adjacent,
    reward_total,
    reward_clean,
    reward_explore,
    reward_frontier,
    current_mode,
    target_progress_delta,
    recoverability_delta,
    position_repeat_8,
    position_repeat_16,
    small_cycle_4,
    small_cycle_8,
    corner_proximity,
    zero_progress_streak,
    wall_follow_streak,
    wall_adjacent,
    local_frontier_density,
    narrow_passage_flag,
    unknown_on_target_path_ratio,
    missed_charge_opportunity,
    charger_nearby_not_charged,
    planner_action_match,
    target_selection_gap,
    target_charger_contested,
):
    revisit_on_clean_floor = cur_visit_count >= 2 and cleaned_this_step == 0 and dirty_adjacent == 0
    redundant_clean_path = cur_visit_count >= 3 and cleaned_this_step == 0 and reward_clean <= 0.0 and current_mode != 4
    no_clean_no_return_progress = cleaned_this_step == 0 and target_progress_delta <= 0.0 and recoverability_delta <= 0.0
    low_value_revisit = cur_visit_count >= 2 and no_clean_no_return_progress and reward_total >= 0.0
    loop_suspect = position_repeat_8 >= 0.5 and zero_progress_streak >= 4
    corner_loop_suspect = loop_suspect and corner_proximity >= 0.5 and dirty_adjacent == 0
    wall_hugging_clean_floor = wall_adjacent > 0 and wall_follow_streak >= 4 and cleaned_this_step == 0 and dirty_adjacent == 0
    stale_boundary_follow = wall_hugging_clean_floor and local_frontier_density <= 0.05
    narrow_unknown_commit = narrow_passage_flag > 0 and unknown_on_target_path_ratio >= 0.1
    suboptimal_target_hold = target_selection_gap > 0.5
    return {
        "revisit_on_clean_floor": 1.0 if revisit_on_clean_floor else 0.0,
        "redundant_clean_path": 1.0 if redundant_clean_path else 0.0,
        "low_value_revisit": 1.0 if low_value_revisit else 0.0,
        "loop_suspect": 1.0 if loop_suspect else 0.0,
        "corner_loop_suspect": 1.0 if corner_loop_suspect else 0.0,
        "no_clean_no_return_progress": 1.0 if no_clean_no_return_progress else 0.0,
        "positive_reward_while_no_progress": 1.0 if (no_clean_no_return_progress and reward_total > 0.0) else 0.0,
        "positive_clean_reward_while_revisiting": 1.0 if (revisit_on_clean_floor and reward_clean > 0.0) else 0.0,
        "positive_explore_reward_while_revisiting": 1.0 if (revisit_on_clean_floor and reward_explore > 0.0) else 0.0,
        "positive_frontier_reward_while_revisiting": 1.0 if (revisit_on_clean_floor and reward_frontier > 0.0) else 0.0,
        "wall_hugging_clean_floor": 1.0 if wall_hugging_clean_floor else 0.0,
        "stale_boundary_follow": 1.0 if stale_boundary_follow else 0.0,
        "narrow_unknown_commit": 1.0 if narrow_unknown_commit else 0.0,
        "missed_charge_opportunity": 1.0 if missed_charge_opportunity else 0.0,
        "charger_nearby_not_charged": 1.0 if charger_nearby_not_charged else 0.0,
        "suboptimal_target_hold": 1.0 if suboptimal_target_hold else 0.0,
        "planner_policy_divergence": 1.0 if not planner_action_match else 0.0,
        "charger_contested": 1.0 if target_charger_contested else 0.0,
    }


def _anomaly_summary(step_records):
    total = max(len(step_records), 1)

    def _rate(key):
        return round(sum(float(rec.get("anomalies", {}).get(key, 0.0)) for rec in step_records) / total, 4)

    loop_episode_detected = _rate("loop_suspect") >= 0.15
    corner_loop_detected = _rate("corner_loop_suspect") >= 0.10
    return {
        "revisit_on_clean_floor_rate": _rate("revisit_on_clean_floor"),
        "redundant_clean_path_rate": _rate("redundant_clean_path"),
        "low_value_revisit_rate": _rate("low_value_revisit"),
        "wall_hugging_clean_floor_rate": _rate("wall_hugging_clean_floor"),
        "stale_boundary_follow_rate": _rate("stale_boundary_follow"),
        "narrow_unknown_commit_rate": _rate("narrow_unknown_commit"),
        "missed_charge_opportunity_rate": _rate("missed_charge_opportunity"),
        "charger_nearby_not_charged_rate": _rate("charger_nearby_not_charged"),
        "suboptimal_target_hold_rate": _rate("suboptimal_target_hold"),
        "planner_policy_divergence_rate": _rate("planner_policy_divergence"),
        "charger_contested_rate": _rate("charger_contested"),
        "loop_suspect_rate": _rate("loop_suspect"),
        "corner_loop_rate": _rate("corner_loop_suspect"),
        "no_clean_no_return_progress_rate": _rate("no_clean_no_return_progress"),
        "positive_reward_while_no_progress_rate": _rate("positive_reward_while_no_progress"),
        "positive_explore_reward_while_revisiting_rate": _rate("positive_explore_reward_while_revisiting"),
        "positive_frontier_reward_while_revisiting_rate": _rate("positive_frontier_reward_while_revisiting"),
        "loop_episode_detected": bool(loop_episode_detected),
        "corner_loop_detected": bool(corner_loop_detected),
        "position_repeat_16_avg": round(sum(float(rec.get("position_repeat_16", 0.0)) for rec in step_records) / total, 4),
        "avg_zero_progress_streak": round(sum(float(rec.get("zero_progress_streak", 0.0)) for rec in step_records) / total, 4),
        "avg_wall_follow_streak": round(sum(float(rec.get("wall_follow_streak", 0.0)) for rec in step_records) / total, 4),
        "avg_narrow_passage_rate": round(sum(float(rec.get("narrow_passage_flag", 0.0)) for rec in step_records) / total, 4),
        "avg_fallback_to_chebyshev_rate": round(sum(float(rec.get("fallback_to_chebyshev", 0.0)) for rec in step_records) / total, 4),
        "avg_target_selection_gap": round(sum(float(rec.get("target_selection_gap", 0.0)) for rec in step_records) / total, 4),
        "avg_target_charger_robot_count": round(sum(float(rec.get("target_charger_robot_count", 0.0)) for rec in step_records) / total, 4),
        "avg_robot_on_target_path_count": round(sum(float(rec.get("robot_on_target_path_count", 0.0)) for rec in step_records) / total, 4),
        "avg_charge_margin_now": round(sum(float(rec.get("charge_margin_now", 0.0)) for rec in step_records) / total, 4),
        "avg_path_cross_count_50": round(sum(float(rec.get("path_cross_count_50", 0.0)) for rec in step_records) / total, 4),
        "avg_coverage_efficiency_20": round(sum(float(rec.get("coverage_efficiency_20", 0.0)) for rec in step_records) / total, 4),
    }


def _reward_attribution(step_records):
    return {
        "wall_hugging": _subset_reward_attribution(step_records, lambda rec: rec.get("anomalies", {}).get("wall_hugging_clean_floor", 0.0) > 0.0),
        "corner_loop": _subset_reward_attribution(step_records, lambda rec: rec.get("anomalies", {}).get("corner_loop_suspect", 0.0) > 0.0),
        "loop": _subset_reward_attribution(step_records, lambda rec: rec.get("anomalies", {}).get("loop_suspect", 0.0) > 0.0),
        "revisit_on_clean_floor": _subset_reward_attribution(
            step_records, lambda rec: rec.get("anomalies", {}).get("revisit_on_clean_floor", 0.0) > 0.0
        ),
        "low_value_revisit": _subset_reward_attribution(
            step_records, lambda rec: rec.get("anomalies", {}).get("low_value_revisit", 0.0) > 0.0
        ),
        "no_clean_no_return_progress": _subset_reward_attribution(
            step_records, lambda rec: rec.get("anomalies", {}).get("no_clean_no_return_progress", 0.0) > 0.0
        ),
        "narrow_unknown_commit": _subset_reward_attribution(
            step_records, lambda rec: rec.get("anomalies", {}).get("narrow_unknown_commit", 0.0) > 0.0
        ),
        "missed_charge_opportunity": _subset_reward_attribution(
            step_records, lambda rec: rec.get("anomalies", {}).get("missed_charge_opportunity", 0.0) > 0.0
        ),
        "suboptimal_target_hold": _subset_reward_attribution(
            step_records, lambda rec: rec.get("anomalies", {}).get("suboptimal_target_hold", 0.0) > 0.0
        ),
        "charger_contested": _subset_reward_attribution(
            step_records, lambda rec: rec.get("anomalies", {}).get("charger_contested", 0.0) > 0.0
        ),
        "battery_fail_trajectory": _subset_reward_attribution(step_records[-12:] if step_records else [], lambda rec: True),
        "return_stall_window": _subset_reward_attribution(
            step_records, lambda rec: rec.get("mode", -1) == 4 and rec.get("target_progress_delta", 0.0) <= 0.0
        ),
    }


def _subset_reward_attribution(step_records, predicate):
    subset = [rec for rec in step_records if predicate(rec)]
    if not subset:
        return {
            "sample_count": 0,
            "reward_total_mean": 0.0,
            "reward_clean_mean": 0.0,
            "reward_survive_mean": 0.0,
            "top_positive_reward_terms": [],
            "top_negative_reward_terms": [],
            "avg_mode_distribution": {},
            "avg_action_entropy": 0.0,
            "avg_target_progress_delta": 0.0,
            "avg_recoverability_delta": 0.0,
        }

    component_means = {}
    for key in REWARD_COMPONENT_KEYS:
        component_means[f"reward_{key}"] = float(sum(rec.get(f"reward_{key}", 0.0) for rec in subset) / len(subset))

    top_positive = sorted(
        ((label, value) for label, value in component_means.items() if value > 0.0),
        key=lambda item: item[1],
        reverse=True,
    )[:3]
    top_negative = sorted(
        ((label, value) for label, value in component_means.items() if value < 0.0),
        key=lambda item: item[1],
    )[:3]
    mode_counts = {}
    for rec in subset:
        mode = str(int(rec.get("mode", -1)))
        mode_counts[mode] = mode_counts.get(mode, 0) + 1

    return {
        "sample_count": len(subset),
        "reward_total_mean": round(sum(rec.get("reward", 0.0) for rec in subset) / len(subset), 4),
        "reward_clean_mean": round(sum(rec.get("reward_clean", 0.0) for rec in subset) / len(subset), 4),
        "reward_survive_mean": round(sum(rec.get("reward_survive", 0.0) for rec in subset) / len(subset), 4),
        "top_positive_reward_terms": [[label, round(value, 4)] for label, value in top_positive],
        "top_negative_reward_terms": [[label, round(value, 4)] for label, value in top_negative],
        "avg_mode_distribution": {label: round(count / len(subset), 4) for label, count in sorted(mode_counts.items())},
        "avg_action_entropy": round(sum(rec.get("action_entropy", 0.0) for rec in subset) / len(subset), 4),
        "avg_target_progress_delta": round(sum(rec.get("target_progress_delta", 0.0) for rec in subset) / len(subset), 4),
        "avg_recoverability_delta": round(sum(rec.get("recoverability_delta", 0.0) for rec in subset) / len(subset), 4),
    }


def _extract_evidence_windows(step_records):
    return {
        "first_late_return_window": _slice_window(
            step_records,
            next((idx for idx, rec in enumerate(step_records) if rec.get("mode") == 4 and rec.get("charger_slack", 0.0) < 0.0), None),
        ),
        "first_wall_hugging_window": _slice_window(
            step_records,
            next((idx for idx, rec in enumerate(step_records) if rec.get("anomalies", {}).get("wall_hugging_clean_floor", 0.0) > 0.0), None),
        ),
        "first_missed_charge_window": _slice_window(
            step_records,
            next((idx for idx, rec in enumerate(step_records) if rec.get("anomalies", {}).get("missed_charge_opportunity", 0.0) > 0.0), None),
        ),
        "first_narrow_unknown_commit_window": _slice_window(
            step_records,
            next((idx for idx, rec in enumerate(step_records) if rec.get("anomalies", {}).get("narrow_unknown_commit", 0.0) > 0.0), None),
        ),
        "first_target_retarget_window": _slice_window(
            step_records,
            next((idx for idx, rec in enumerate(step_records) if rec.get("retarget_event", 0.0) > 0.0), None),
        ),
        "first_corner_loop_window": _slice_window(
            step_records,
            next((idx for idx, rec in enumerate(step_records) if rec.get("anomalies", {}).get("corner_loop_suspect", 0.0) > 0.0), None),
        ),
        "last_battery_fail_window": _slice_window(step_records, len(step_records) - 1 if step_records else None),
    }


def _slice_window(step_records, center_idx, radius=EVIDENCE_RADIUS):
    if center_idx is None or not step_records:
        return []
    start = max(center_idx - radius, 0)
    end = min(center_idx + radius + 1, len(step_records))
    window = []
    for rec in step_records[start:end]:
        top_positive = sorted(
            ((f"reward_{key}", rec.get(f"reward_{key}", 0.0)) for key in REWARD_COMPONENT_KEYS if rec.get(f"reward_{key}", 0.0) > 0.0),
            key=lambda item: item[1],
            reverse=True,
        )[:2]
        top_negative = sorted(
            ((f"reward_{key}", rec.get(f"reward_{key}", 0.0)) for key in REWARD_COMPONENT_KEYS if rec.get(f"reward_{key}", 0.0) < 0.0),
            key=lambda item: item[1],
        )[:2]
        window.append(
            {
                "step": rec.get("step"),
                "mode": rec.get("mode"),
                "action": rec.get("action"),
                "charger_slack": rec.get("charger_slack"),
                "target_progress_delta": rec.get("target_progress_delta"),
                "reward_total": rec.get("reward"),
                "top_positive_reward_terms": [[label, round(value, 4)] for label, value in top_positive],
                "top_negative_reward_terms": [[label, round(value, 4)] for label, value in top_negative],
                "anomalies": rec.get("anomalies", {}),
            }
        )
    return window


def _aggregate_anomaly_summary(episodes):
    if not episodes:
        return {
            "loop_episode_rate": 0.0,
            "corner_loop_episode_rate": 0.0,
            "avg_revisit_on_clean_floor_rate": 0.0,
            "avg_redundant_clean_path_rate": 0.0,
            "avg_low_value_revisit_rate": 0.0,
            "avg_positive_reward_while_no_progress_rate": 0.0,
            "battery_fail_loop_episode_rate": 0.0,
            "battery_fail_corner_loop_rate": 0.0,
            "completed_loop_episode_rate": 0.0,
        }

    battery_eps = [ep for ep in episodes if ep.get("result") == "battery"]
    completed_eps = [ep for ep in episodes if ep.get("result") == "completed"]
    anomaly_summaries = [ep.get("anomaly_summary", {}) for ep in episodes]
    return {
        "loop_episode_rate": round(sum(1 for summary in anomaly_summaries if summary.get("loop_episode_detected")) / len(episodes), 4),
        "corner_loop_episode_rate": round(
            sum(1 for summary in anomaly_summaries if summary.get("corner_loop_detected")) / len(episodes), 4
        ),
        "avg_wall_hugging_clean_floor_rate": round(
            sum(summary.get("wall_hugging_clean_floor_rate", 0.0) for summary in anomaly_summaries) / len(episodes), 4
        ),
        "avg_stale_boundary_follow_rate": round(
            sum(summary.get("stale_boundary_follow_rate", 0.0) for summary in anomaly_summaries) / len(episodes), 4
        ),
        "avg_revisit_on_clean_floor_rate": round(
            sum(summary.get("revisit_on_clean_floor_rate", 0.0) for summary in anomaly_summaries) / len(episodes), 4
        ),
        "avg_redundant_clean_path_rate": round(
            sum(summary.get("redundant_clean_path_rate", 0.0) for summary in anomaly_summaries) / len(episodes), 4
        ),
        "avg_low_value_revisit_rate": round(
            sum(summary.get("low_value_revisit_rate", 0.0) for summary in anomaly_summaries) / len(episodes), 4
        ),
        "avg_positive_reward_while_no_progress_rate": round(
            sum(summary.get("positive_reward_while_no_progress_rate", 0.0) for summary in anomaly_summaries) / len(episodes),
            4,
        ),
        "avg_narrow_unknown_commit_rate": round(
            sum(summary.get("narrow_unknown_commit_rate", 0.0) for summary in anomaly_summaries) / len(episodes), 4
        ),
        "avg_missed_charge_opportunity_rate": round(
            sum(summary.get("missed_charge_opportunity_rate", 0.0) for summary in anomaly_summaries) / len(episodes), 4
        ),
        "avg_suboptimal_target_hold_rate": round(
            sum(summary.get("suboptimal_target_hold_rate", 0.0) for summary in anomaly_summaries) / len(episodes), 4
        ),
        "avg_charger_contested_rate": round(
            sum(summary.get("charger_contested_rate", 0.0) for summary in anomaly_summaries) / len(episodes), 4
        ),
        "battery_fail_loop_episode_rate": round(
            sum(1 for ep in battery_eps if ep.get("anomaly_summary", {}).get("loop_episode_detected")) / max(len(battery_eps), 1),
            4,
        ),
        "battery_fail_corner_loop_rate": round(
            sum(1 for ep in battery_eps if ep.get("anomaly_summary", {}).get("corner_loop_detected")) / max(len(battery_eps), 1),
            4,
        ),
        "completed_loop_episode_rate": round(
            sum(1 for ep in completed_eps if ep.get("anomaly_summary", {}).get("loop_episode_detected")) / max(len(completed_eps), 1),
            4,
        ),
    }


def _aggregate_reward_attribution(episodes):
    keys = (
        "wall_hugging",
        "corner_loop",
        "loop",
        "revisit_on_clean_floor",
        "low_value_revisit",
        "no_clean_no_return_progress",
        "narrow_unknown_commit",
        "missed_charge_opportunity",
        "suboptimal_target_hold",
        "charger_contested",
        "battery_fail_trajectory",
        "return_stall_window",
    )
    aggregated = {}
    for key in keys:
        attrs = [ep.get("reward_attribution", {}).get(key, {}) for ep in episodes]
        attrs = [attr for attr in attrs if attr.get("sample_count", 0) > 0]
        if not attrs:
            aggregated[key] = {
                "sample_count": 0,
                "reward_total_mean": 0.0,
                "reward_clean_mean": 0.0,
                "reward_survive_mean": 0.0,
                "top_positive_reward_terms": [],
                "top_negative_reward_terms": [],
            }
            continue

        component_scores = {}
        sample_count = sum(attr["sample_count"] for attr in attrs)
        for attr in attrs:
            for label, value in attr.get("top_positive_reward_terms", []):
                component_scores[label] = component_scores.get(label, 0.0) + float(value)
            for label, value in attr.get("top_negative_reward_terms", []):
                component_scores[label] = component_scores.get(label, 0.0) + float(value)

        top_positive = sorted(
            ((label, value) for label, value in component_scores.items() if value > 0.0),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        top_negative = sorted(
            ((label, value) for label, value in component_scores.items() if value < 0.0),
            key=lambda item: item[1],
        )[:3]
        aggregated[key] = {
            "sample_count": sample_count,
            "reward_total_mean": round(sum(attr["reward_total_mean"] for attr in attrs) / len(attrs), 4),
            "reward_clean_mean": round(sum(attr["reward_clean_mean"] for attr in attrs) / len(attrs), 4),
            "reward_survive_mean": round(sum(attr["reward_survive_mean"] for attr in attrs) / len(attrs), 4),
            "top_positive_reward_terms": [[label, round(value, 4)] for label, value in top_positive],
            "top_negative_reward_terms": [[label, round(value, 4)] for label, value in top_negative],
        }
    return aggregated


def _build_ai_summary(snapshot):
    episodes = snapshot.get("episodes", [])
    issue_index = {}
    for issue, config in ISSUE_INDEX_CONFIG.items():
        matched = [ep for ep in episodes if config["predicate"](ep)]
        issue_index[issue] = {
            "detected": bool(matched),
            "episode_count": len(matched),
            "example_episode_ids": [ep.get("episode_id") for ep in matched[:3]],
            "primary_metrics": config["primary_metrics"],
            "primary_attribution_key": config["primary_attribution_key"],
        }

    ranked = sorted(
        ((issue, meta["episode_count"]) for issue, meta in issue_index.items() if meta["episode_count"] > 0),
        key=lambda item: item[1],
        reverse=True,
    )
    evidence = {}
    for issue, meta in issue_index.items():
        if not meta["example_episode_ids"]:
            continue
        first_id = meta["example_episode_ids"][0]
        episode = next((item for item in episodes if item.get("episode_id") == first_id), None)
        if episode is None:
            continue
        evidence[issue] = episode.get("evidence_windows", {})

    return {
        "version": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "timestamp": snapshot.get("timestamp"),
        "checkpoint": snapshot.get("checkpoint"),
        "overall": snapshot.get("overall", {}),
        "top_anomalies": [{"issue": issue, "episode_count": count} for issue, count in ranked[:5]],
        "issue_index": issue_index,
        "round_summaries": snapshot.get("per_round", {}),
        "reward_attribution": snapshot.get("overall", {}).get("reward_attribution", {}),
        "example_evidence_windows": evidence,
    }


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _extract_base_env_conf(usr_conf):
    if isinstance(usr_conf, dict) and isinstance(usr_conf.get("env_conf"), dict):
        return deepcopy(usr_conf["env_conf"])
    return deepcopy(usr_conf)


def _load_benchmark_checkpoint(agent, checkpoint, logger):
    """Load a specific checkpoint for benchmark evaluation."""
    resolved = str(checkpoint or "").strip()
    if resolved in {"", "latest"}:
        agent.load_model(id="latest")
        checkpoint_ref = getattr(agent, "current_model_ref", {}) or {}
        return {
            "checkpoint": str(checkpoint_ref.get("path") or resolved or "latest"),
            "checkpoint_id": checkpoint_ref.get("checkpoint_id"),
            "checkpoint_step": checkpoint_ref.get("checkpoint_step"),
        }
    if not os.path.isabs(resolved):
        resolved = os.path.join("/workspace/code", resolved)

    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"Benchmark checkpoint not found: {resolved}")

    logger.info(f"[BENCHMARK] Loading checkpoint: {resolved}")
    state_dict = torch.load(resolved, map_location=agent.device)
    agent.model.load_state_dict(state_dict)
    checkpoint_id = parse_checkpoint_id(resolved) or os.path.basename(resolved)
    checkpoint_step = int(checkpoint_id) if str(checkpoint_id).isdigit() else 0
    if hasattr(agent, "current_model_ref"):
        agent.current_model_ref = {
            "path": resolved,
            "id": "benchmark",
            "checkpoint_id": checkpoint_id,
            "checkpoint_step": checkpoint_step,
            "global_step_since_resume": checkpoint_step,
        }
    logger.info(f"[BENCHMARK] Loaded state_dict from {resolved}")
    return {
        "checkpoint": resolved,
        "checkpoint_id": checkpoint_id,
        "checkpoint_step": checkpoint_step,
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
    checkpoint_ref = getattr(agent, "current_model_ref", {}) or {}
    runtime.setdefault("global_step_since_resume", int(checkpoint_ref.get("global_step_since_resume") or 0))
    runtime.setdefault("checkpoint_global_step", int(checkpoint_ref.get("checkpoint_step") or 0))
    payload["runtime"] = runtime
    return payload


def _save_results(results_file, snapshot):
    results = {"version": SCHEMA_VERSION, "schema_version": SCHEMA_VERSION, "benchmarks": []}
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
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _get_git_commit():
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd="/workspace/code",
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"
