#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for Robot Vacuum.
"""

import os
import time
import json
from collections import deque
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import sample_process
from agent_ppo.workflow.checkpoint_score import compute_checkpoint_scores, compute_legacy_robust_score
from agent_ppo.workflow.curriculum_policy import (
    STAGE_INDEX,
    profile_weights_for_stage,
    stage_progress,
)
from agent_ppo.workflow.curriculum_state import SharedCurriculumStateStore
from agent_ppo.utils.experiment_archive import ExperimentArchive, infer_fail_reason
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or value == "":
        return int(default)
    return int(value)


class PerfWindow:
    def __init__(self):
        self.values = {}

    def add(self, name, duration_ms, count=1):
        stats = self.values.setdefault(name, {"total_ms": 0.0, "count": 0})
        stats["total_ms"] += float(duration_ms)
        stats["count"] += int(count)

    def flush(self, prefix):
        payload = {}
        for name, stats in self.values.items():
            payload[f"{prefix}_{name}_total_ms"] = round(stats["total_ms"], 4)
            payload[f"{prefix}_{name}_count"] = int(stats["count"])
            if stats["count"] > 0:
                payload[f"{prefix}_{name}_avg_ms"] = round(stats["total_ms"] / stats["count"], 4)
        self.values = {}
        return payload


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env = envs[0]
    agent = agents[0]

    if os.getenv("KAIWU_BENCHMARK_PARALLEL_MODE", "").strip() in ("1", "true"):
        from agent_ppo.eval.benchmark_parallel import run_parallel_benchmark

        usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
        run_parallel_benchmark(envs, agents, usr_conf, logger)
        return

    # Benchmark mode: run fixed eval scenarios, save results, exit
    if os.getenv("KAIWU_BENCHMARK_MODE", "").strip() in ("1", "true"):
        aisrv_index = os.getenv("KAIWU_AISRV_INDEX", "").strip()
        if aisrv_index not in ("", "1"):
            logger.info(f"[BENCHMARK] Skipping on aisrv-{aisrv_index}, only aisrv-1 runs benchmark")
            marker = Path("/workspace/code/.benchmark_done")
            while not marker.exists():
                time.sleep(5)
            logger.info("[BENCHMARK] aisrv-1 benchmark complete, exiting")
            return
        from agent_ppo.eval.benchmark import run_benchmark
        usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
        run_benchmark(env, agent, usr_conf, logger)
        return

    archive = ExperimentArchive(service_name=os.getenv("KAIWU_SERVICE_NAME") or "aisrv")

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    archive.ensure_run(
        {
            "workflow_started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "algorithm": os.getenv("KAIWU_ALGORITHM") or "ppo",
            "usr_conf": usr_conf,
            "training_strategy": {
                "policy_blending": "model_logits_plus_heuristic_bias",
                "curriculum": "anchor_mild_broad_randomization",
                "selection_target": "checkpoint_preservation_score_v2",
            },
        }
    )
    archive.log_event(
        "run_start",
        {
            "algorithm": os.getenv("KAIWU_ALGORITHM") or "ppo",
            "usr_conf": usr_conf,
        },
    )

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        config_sampler=EnvConfigSampler(usr_conf),
        logger=logger,
        monitor=monitor,
        archive=archive,
    )
    send_perf = PerfWindow()
    last_send_perf_report_time = 0.0

    while True:
        for g_data in episode_runner.run_episodes():
            send_begin = time.perf_counter()
            agent.send_sample_data(g_data)
            send_perf.add("send_sample_data", (time.perf_counter() - send_begin) * 1000.0)
            send_perf.add("samples_sent", 0.0, count=len(g_data))
            g_data.clear()
            now = time.time()
            if now - last_send_perf_report_time >= _env_int(
                "KAIWU_PERF_STAT_WINDOW_SECONDS", Config.PERF_STAT_WINDOW_SECONDS
            ):
                payload = {
                    "record_type": "aisrv_send_window",
                    "episode_cnt": episode_runner.episode_cnt,
                }
                payload.update(send_perf.flush("workflow"))
                runtime_metrics = agent.get_runtime_metrics() if hasattr(agent, "get_runtime_metrics") else {}
                for key, value in runtime_metrics.items():
                    payload[f"agent_{key}"] = value
                archive.log_train_window(payload)
                last_send_perf_report_time = now


class EnvConfigSampler:
    def __init__(self, usr_conf):
        self.base_usr_conf = deepcopy(usr_conf)
        if isinstance(self.base_usr_conf, dict) and isinstance(self.base_usr_conf.get("env_conf"), dict):
            self.env_key = "env_conf"
            self.base_env_conf = deepcopy(self.base_usr_conf["env_conf"])
        else:
            self.env_key = None
            self.base_env_conf = deepcopy(self.base_usr_conf)

        self.maps = list(self.base_env_conf.get("map") or [1, 2, 3])
        self.base_map_random = bool(self.base_env_conf.get("map_random", True))
        self.base_robot_count = int(self.base_env_conf.get("robot_count", 1))
        self.base_charger_count = int(self.base_env_conf.get("charger_count", 4))
        self.base_max_step = int(self.base_env_conf.get("max_step", 1000))
        self.base_battery_max = int(self.base_env_conf.get("battery_max", 200))
        self.rng = np.random.default_rng(seed=20260409)

    def sample(self, stage):
        profile = self._pick_profile(stage)
        env_conf = deepcopy(self.base_env_conf)

        if profile == "anchor":
            env_conf["map_random"] = self.base_map_random
            env_conf["map"] = list(self.maps)
        else:
            env_conf["map_random"] = True
            env_conf["map"] = self._sample_maps(profile)
            env_conf["robot_count"] = self._sample_robot_count(profile)
            env_conf["charger_count"] = self._sample_charger_count(profile)
            env_conf["max_step"] = self._sample_max_step(profile)
            env_conf["battery_max"] = self._sample_battery_max(profile)

        sampled_usr_conf = self._wrap_env_conf(env_conf)
        meta = {
            "stage": stage,
            "profile": profile,
            "env_conf": deepcopy(env_conf),
        }
        return sampled_usr_conf, meta

    def _pick_profile(self, stage):
        draw = self.rng.random()
        cumulative = 0.0
        for profile, weight in profile_weights_for_stage(stage):
            cumulative += float(weight)
            if draw < cumulative:
                return profile
        return profile_weights_for_stage(stage)[-1][0]

    def _sample_robot_count(self, profile):
        if profile == "mild":
            return self._sample_near(self.base_robot_count, 1, 4, (-1, 0, 1))
        return int(self.rng.integers(1, 5))

    def _sample_charger_count(self, profile):
        if profile == "mild":
            return self._sample_near(self.base_charger_count, 1, 4, (-1, 0, 1))
        return int(self.rng.integers(1, 5))

    def _sample_max_step(self, profile):
        if profile == "broad_eval":
            if self.rng.random() < 0.50:
                return 2000
            return int(self.rng.choice([1100, 1400, 1700, 2000]))
        if profile == "mild":
            return self._sample_near(self.base_max_step, 400, 2000, (-250, -100, 0, 100, 250), quant=50)
        levels = [500, 700, 900, 1100, 1400, 1700, 2000]
        return int(self.rng.choice(levels))

    def _sample_battery_max(self, profile):
        if profile == "broad_eval":
            levels = [120, 160, 200, 200, 200, 260, 320, 420]
            return int(self.rng.choice(levels))
        if profile == "mild":
            return self._sample_near(self.base_battery_max, 100, 999, (-80, -40, 0, 40, 80, 120), quant=20)
        levels = [120, 160, 200, 260, 320, 420, 560, 720]
        return int(self.rng.choice(levels))

    def _sample_maps(self, profile):
        if len(self.maps) <= 1 or profile == "mild":
            return list(self.maps)
        subset_size = int(self.rng.integers(1, len(self.maps) + 1))
        indices = self.rng.choice(len(self.maps), size=subset_size, replace=False)
        return [self.maps[int(idx)] for idx in sorted(indices)]

    def _sample_near(self, base_value, v_min, v_max, deltas, quant=1):
        candidate = int(base_value + self.rng.choice(deltas))
        candidate = int(np.clip(candidate, v_min, v_max))
        if quant > 1:
            candidate = int(round(candidate / quant) * quant)
            candidate = int(np.clip(candidate, v_min, v_max))
        return candidate

    def _wrap_env_conf(self, env_conf):
        if self.env_key is None:
            return deepcopy(env_conf)
        wrapped = deepcopy(self.base_usr_conf)
        wrapped[self.env_key] = deepcopy(env_conf)
        return wrapped


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, config_sampler, logger, monitor, archive):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.config_sampler = config_sampler
        self.logger = logger
        self.monitor = monitor
        self.archive = archive
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.last_perf_stat_time = 0
        self.perf_window = PerfWindow()

        self.episode_history = deque(maxlen=Config.MONITOR_WINDOW)
        self.death_trajectory_buffer = []
        self.DEATH_TRAJ_LENGTH = 20
        self.config_stats = {}

        self.best_avg_score = 0.0
        self.best_robust_score = float("-inf")
        self.best_resume_readiness_score = float("-inf")
        self.best_submission_score = float("-inf")
        self.best_preservation_score = float("-inf")
        self.last_checkpoint_scores = {
            "resume_readiness_score": 0.0,
            "submission_score": 0.0,
            "checkpoint_preservation_score": 0.0,
            "resume_eligible": False,
            "submission_eligible": False,
        }
        self.last_clean_score = 0.0
        self.is_new_best = False
        self.per_map_scores: dict[str, list[float]] = {}
        self.latest_training_metrics = {}
        self.latest_learning_metrics = {}
        self.learning_metric_history = deque(maxlen=8)
        self.resume_global_step_base = None
        self.resume_fast_track = True
        self.code_path = resolve_shared_code_dir()
        self.curriculum_store = SharedCurriculumStateStore(self.code_path)
        self.signal_source_id = f"aisrv-{os.getenv('KAIWU_AISRV_INDEX', 'x')}-pid-{os.getpid()}"
        self.code_dir = str(self.code_path)
        self.session_id = time.strftime("%Y%m%d-%H%M%S")
        self.session_dir = self.code_path / "session_best" / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.best_score_file = self.session_dir / "best_score.json"
        self.logger.info(f"[SESSION] New training session: {self.session_id}")
        self.resume_snapshot_dir = self.code_path / "resume_snapshots"
        self.resume_snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.resume_latest_path = self.code_path / "model.ckpt-resume.pkl"
        self.resume_latest_meta_path = self.code_path / "model.ckpt-resume.meta.json"
        self.latest_model_path = self.code_path / "latest_model.pkl"
        self.manual_ckpt_dir = self.code_path / "manual_checkpoints"
        self.manual_ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.last_episode_snapshot_episode = 0
        self.last_time_snapshot_at = time.time()
        self.last_latest_sync_episode = 0
        self.save_interval = _env_int("KAIWU_SAVE_MODEL_INTERVAL_EPISODES", Config.SAVE_MODEL_INTERVAL_EPISODES)
        self.resume_episode_interval = _env_int(
            "KAIWU_RESUME_EPISODE_SNAPSHOT_INTERVAL", Config.RESUME_EPISODE_SNAPSHOT_INTERVAL
        )
        self.latest_sync_interval = _env_int(
            "KAIWU_RESUME_LATEST_SYNC_INTERVAL_EPISODES", Config.RESUME_LATEST_SYNC_INTERVAL_EPISODES
        )
        self.time_save_interval_seconds = _env_int(
            "KAIWU_RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS", Config.RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS
        )
        self.keep_episode_snapshots = Config.KEEP_EPISODE_RESUME_SNAPSHOTS
        self.keep_time_snapshots = Config.KEEP_TIME_RESUME_SNAPSHOTS
        self.keep_best_snapshots = Config.KEEP_BEST_RESUME_SNAPSHOTS

    def _persist_best_score(self):
        data = {
            "best_robust_score": self.best_robust_score,
            "best_avg_score": self.best_avg_score,
            "best_resume_readiness_score": self.best_resume_readiness_score,
            "best_submission_score": self.best_submission_score,
            "best_preservation_score": self.best_preservation_score,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "episode_cnt": self.episode_cnt,
        }
        tmp = self.best_score_file.parent / f".{self.best_score_file.name}.{os.getpid()}.{time.time_ns()}.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
        os.replace(tmp, self.best_score_file)

    def _update_manifest(self):
        manifest_path = self.code_path / "session_best" / "manifest.json"
        entries = {}
        if manifest_path.exists():
            try:
                entries = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                entries = {}
        entries[self.session_id] = {
            "best_robust_score": round(self.best_robust_score, 4),
            "best_avg_score": round(self.best_avg_score, 4),
            "best_resume_readiness_score": round(self.best_resume_readiness_score, 4),
            "best_submission_score": round(self.best_submission_score, 4),
            "best_preservation_score": round(self.best_preservation_score, 4),
            "episode_cnt": self.episode_cnt,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = manifest_path.parent / f".manifest.{os.getpid()}.{time.time_ns()}.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=True, indent=2)
        os.replace(tmp, manifest_path)

    def _save_best_model(self, clean_score, score_payload):
        best_path = self.session_dir / "best_model.pkl"
        state_dict = {k: v.clone().cpu() for k, v in self.agent.model.state_dict().items()}
        torch.save(state_dict, best_path)
        self._persist_best_score()
        self._update_manifest()
        self.logger.info(
            f"[BEST] session={self.session_id} ep={self.episode_cnt} avg={self.best_avg_score:.2f} "
            f"robust={self.best_robust_score:.2f} resume={score_payload['resume_readiness_score']:.2f} "
            f"preserve={score_payload['checkpoint_preservation_score']:.2f} score={clean_score:.1f} saved to {best_path}"
        )
        self.archive.log_checkpoint(
            {
                "episode_cnt": self.episode_cnt,
                "clean_score": round(float(clean_score), 4),
                "best_path": str(best_path),
                **score_payload,
            }
        )

    def _snapshot_state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.agent.model.state_dict().items()}

    def _write_state_dict(self, path, state_dict):
        tmp_path = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        torch.save(state_dict, tmp_path)
        os.replace(tmp_path, path)

    def _write_resume_meta(self, payload):
        tmp_path = self.resume_latest_meta_path.parent / f".{self.resume_latest_meta_path.name}.{os.getpid()}.{time.time_ns()}.tmp"
        with tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, self.resume_latest_meta_path)

    def _prune_snapshots(self, prefix, keep_count):
        files = sorted(self.resume_snapshot_dir.glob(f"{prefix}-*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True)
        for old_path in files[keep_count:]:
            old_path.unlink(missing_ok=True)

    def _save_resume_artifacts(self, trigger, clean_score, with_named_snapshot=False):
        state_dict = self._snapshot_state_dict()
        self._write_state_dict(self.latest_model_path, state_dict)

        meta = {
            "trigger": trigger,
            "episode_cnt": self.episode_cnt,
            "clean_score": round(float(clean_score), 4),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pid": os.getpid(),
        }
        self._write_state_dict(self.resume_latest_path, state_dict)
        self._write_resume_meta(meta)
        self.archive.log_event(
            "resume_checkpoint_refreshed",
            {
                "trigger": trigger,
                "episode_cnt": self.episode_cnt,
                "clean_score": round(float(clean_score), 4),
                "path": str(self.resume_latest_path),
            },
        )

        if with_named_snapshot:
            if trigger == "best":
                snapshot_name = f"best-ep{self.episode_cnt:06d}-score{int(round(clean_score)):05d}.pkl"
                snapshot_path = self.session_dir / snapshot_name
                self._write_state_dict(snapshot_path, state_dict)
                self.archive.log_event(
                    "resume_snapshot_saved",
                    {
                        "trigger": trigger,
                        "episode_cnt": self.episode_cnt,
                        "clean_score": round(float(clean_score), 4),
                        "path": str(snapshot_path),
                    },
                )
                self.logger.info(
                    f"[SNAPSHOT] trigger={trigger} ep={self.episode_cnt} "
                    f"score={clean_score:.1f} path={snapshot_path}"
                )
                return
            if trigger == "episode":
                snapshot_name = f"resume-episode-ep{self.episode_cnt:06d}.pkl"
                keep_count = self.keep_episode_snapshots
                prune_prefix = "resume-episode"
            else:
                snapshot_name = f"resume-time-{time.strftime('%Y%m%d-%H%M%S')}.pkl"
                keep_count = self.keep_time_snapshots
                prune_prefix = "resume-time"
            snapshot_path = self.resume_snapshot_dir / snapshot_name
            self._write_state_dict(snapshot_path, state_dict)
            self._prune_snapshots(prune_prefix, keep_count)
            self.archive.log_event(
                "resume_snapshot_saved",
                {
                    "trigger": trigger,
                    "episode_cnt": self.episode_cnt,
                    "clean_score": round(float(clean_score), 4),
                    "path": str(snapshot_path),
                },
            )
            self.logger.info(
                f"[SNAPSHOT] trigger={trigger} ep={self.episode_cnt} "
                f"score={clean_score:.1f} path={snapshot_path}"
            )

    def _maybe_save_progress_snapshots(self):
        if self.episode_cnt - self.last_latest_sync_episode >= self.latest_sync_interval:
            self._save_resume_artifacts("latest", self.last_clean_score, with_named_snapshot=False)
            self.last_latest_sync_episode = self.episode_cnt

        if self.episode_cnt - self.last_episode_snapshot_episode >= self.resume_episode_interval:
            self._save_resume_artifacts("episode", self.last_clean_score, with_named_snapshot=True)
            self.last_episode_snapshot_episode = self.episode_cnt

        now = time.time()
        if now - self.last_time_snapshot_at >= self.time_save_interval_seconds:
            self._save_resume_artifacts("time", self.last_clean_score, with_named_snapshot=True)
            self.last_time_snapshot_at = now

    def _window_metrics(self, episodes=None):
        """Compute rolling-window metrics from episode_history."""
        buf = list(episodes or self.episode_history)
        n = len(buf)
        if n == 0:
            return {}
        wins = [ep for ep in buf if ep["result"] == "completed"]
        w = len(wins)
        return {
            "_count": n,
            "win_rate": w / n,
            "avg_clean_score": sum(ep["clean_score"] for ep in buf) / n,
            "avg_finished_steps": sum(ep["finished_steps"] for ep in buf) / n,
            "avg_charge_count": sum(ep["charge_count"] for ep in buf) / n,
            "avg_remaining_charge": sum(ep["remaining_charge"] for ep in buf) / n,
            "avg_invalid_move_rate": sum(ep["invalid_move_rate"] for ep in buf) / n,
            "avg_charge_efficiency": sum(ep["charge_efficiency"] for ep in buf) / n,
            "avg_clean_per_step": sum(ep["clean_per_step"] for ep in buf) / n,
            "avg_expert_weight": sum(ep["expert_weight"] for ep in buf) / n,
            "late_return_rate": sum(ep.get("late_return_rate", 0.0) for ep in buf) / n,
            "late_contract_rate": sum(ep.get("late_contract_rate", 0.0) for ep in buf) / n,
            "anchor_switch_rate": sum(ep.get("anchor_switch_rate", 0.0) for ep in buf) / n,
            "target_switch_rate": sum(ep.get("target_switch_rate", 0.0) for ep in buf) / n,
            "diag_rate_all": sum(ep.get("diag_rate_all", 0.0) for ep in buf) / n,
            "diag_rate_contract": sum(ep.get("diag_rate_contract", 0.0) for ep in buf) / n,
            "diag_rate_return": sum(ep.get("diag_rate_return", 0.0) for ep in buf) / n,
            "return_progress_per_step": sum(ep.get("return_progress_per_step", 0.0) for ep in buf) / n,
            "return_efficiency_ratio": sum(ep.get("return_efficiency_ratio", 0.0) for ep in buf) / n,
            "return_stall_rate": sum(ep.get("return_stall_rate", 0.0) for ep in buf) / n,
            "recoverability_score_avg": sum(ep.get("recoverability_score_avg", 0.0) for ep in buf) / n,
            "recoverability_violation_rate": sum(ep.get("recoverability_violation_rate", 0.0) for ep in buf) / n,
            "wall_hugging_clean_floor_rate": sum(ep.get("wall_hugging_clean_floor_rate", 0.0) for ep in buf) / n,
            "stale_boundary_follow_rate": sum(ep.get("stale_boundary_follow_rate", 0.0) for ep in buf) / n,
            "narrow_unknown_commit_rate": sum(ep.get("narrow_unknown_commit_rate", 0.0) for ep in buf) / n,
            "missed_charge_opportunity_rate": sum(ep.get("missed_charge_opportunity_rate", 0.0) for ep in buf) / n,
            "charger_nearby_not_charged_rate": sum(ep.get("charger_nearby_not_charged_rate", 0.0) for ep in buf) / n,
            "suboptimal_target_hold_rate": sum(ep.get("suboptimal_target_hold_rate", 0.0) for ep in buf) / n,
            "planner_policy_divergence_rate": sum(ep.get("planner_policy_divergence_rate", 0.0) for ep in buf) / n,
            "avg_path_cross_count_50": sum(ep.get("avg_path_cross_count_50", 0.0) for ep in buf) / n,
            "avg_coverage_efficiency_20": sum(ep.get("avg_coverage_efficiency_20", 0.0) for ep in buf) / n,
            "avg_all_charger_known_path_count": sum(ep.get("avg_all_charger_known_path_count", 0.0) for ep in buf) / n,
            "avg_unknown_on_target_path_ratio": sum(ep.get("avg_unknown_on_target_path_ratio", 0.0) for ep in buf) / n,
            "mode_usage_depart": sum(ep.get("mode_usage_depart", 0.0) for ep in buf) / n,
            "mode_usage_expand": sum(ep.get("mode_usage_expand", 0.0) for ep in buf) / n,
            "mode_usage_harvest": sum(ep.get("mode_usage_harvest", 0.0) for ep in buf) / n,
            "mode_usage_contract": sum(ep.get("mode_usage_contract", 0.0) for ep in buf) / n,
            "mode_usage_return": sum(ep.get("mode_usage_return", 0.0) for ep in buf) / n,
            "mode_usage_evade": sum(ep.get("mode_usage_evade", 0.0) for ep in buf) / n,
            "battery_fail_rate": sum(1 for ep in buf if ep["result"] == "battery") / n,
            "collision_fail_rate": sum(1 for ep in buf if ep["result"] == "collision") / n,
            "cps_win": (sum(ep["clean_per_step"] for ep in wins) / w) if w else 0.0,
            "avg_charge_count_win": (sum(ep["charge_count"] for ep in wins) / w) if w else 0.0,
            "avg_clean_score_win": (sum(ep["clean_score"] for ep in wins) / w) if w else 0.0,
            "anchor_win_rate": self._profile_win_rate(buf, "anchor"),
            "mild_win_rate": self._profile_win_rate(buf, "mild"),
            "broad_win_rate": self._profile_win_rate(buf, ["broad", "broad_eval"]),
        }

    @staticmethod
    def _profile_win_rate(buf, profiles):
        if isinstance(profiles, str):
            profiles = [profiles]
        subset = [ep for ep in buf if ep["profile"] in profiles]
        if not subset:
            return -1.0
        return sum(1 for ep in subset if ep["result"] == "completed") / len(subset)

    def _curriculum_progress_payload(self):
        """Compute curriculum stage and progress towards advancement."""
        state = self.curriculum_store.read_state()
        stage = state.get("stage", "warmup")
        progress = float(state.get("curriculum_progress", 0.0))
        return {
            "curriculum_stage_idx": STAGE_INDEX.get(stage, 0),
            "curriculum_progress": round(min(progress, 1.0), 4),
        }

    def _get_curriculum_metrics(self):
        return self._window_metrics(list(self.episode_history)[-Config.CURRICULUM_WINDOW:]) if len(self.episode_history) >= Config.CURRICULUM_WINDOW else None

    def _get_bootstrap_curriculum_metrics(self):
        min_count = min(10, len(self.episode_history))
        if min_count < 10:
            return None
        return self._window_metrics(list(self.episode_history)[-min_count:])

    def _get_curriculum_context(self):
        return {
            "window_metrics": self._get_curriculum_metrics(),
            "bootstrap_metrics": self._get_bootstrap_curriculum_metrics(),
            "learning_metrics": deepcopy(self.latest_learning_metrics),
            "global_step_since_resume": self._global_step_since_resume(),
            "resume_fast_track": self.resume_fast_track,
        }

    def _emit_curriculum_signal(self):
        payload = {
            "window_metrics": self._get_curriculum_metrics() or {},
            "bootstrap_metrics": self._get_bootstrap_curriculum_metrics() or {},
            "learning_metrics": deepcopy(self.latest_learning_metrics),
            "runtime": {
                "global_step_since_resume": self._global_step_since_resume(),
            },
            "session_id": self.session_id,
            "episode_cnt_local": self.episode_cnt,
        }
        self.curriculum_store.write_signal(self.signal_source_id, payload)
        return self.curriculum_store.refresh_state()

    def _global_step_since_resume(self):
        current = _extract_numeric_metric(self.latest_training_metrics, "learner", "global_step", default=0.0)
        if self.resume_global_step_base is None:
            self.resume_global_step_base = int(current)
        return max(int(current) - int(self.resume_global_step_base), 0)

    def _refresh_learning_metrics(self, training_metrics):
        snapshot = {
            "global_step": _extract_numeric_metric(training_metrics, "learner", "global_step")
            if _extract_numeric_metric(training_metrics, "learner", "global_step") is not None
            else _extract_numeric_metric(training_metrics, "basic", "train_global_step", default=0.0),
            "entropy_loss": _extract_numeric_metric(training_metrics, "learner", "entropy_loss")
            if _extract_numeric_metric(training_metrics, "learner", "entropy_loss") is not None
            else _extract_numeric_metric(training_metrics, "algorithm", "entropy_loss"),
            "value_clean_loss": _extract_numeric_metric(training_metrics, "learner", "value_clean_loss"),
            "value_survive_loss": _extract_numeric_metric(training_metrics, "learner", "value_survive_loss"),
            "mode_teacher_active_rate": _extract_numeric_metric(training_metrics, "learner", "mode_teacher_active_rate"),
            "route_anchor_teacher_active_rate": _extract_numeric_metric(training_metrics, "learner", "route_anchor_teacher_active_rate"),
            "target_teacher_active_rate": _extract_numeric_metric(training_metrics, "learner", "target_teacher_active_rate"),
            "return_action_teacher_active_rate": _extract_numeric_metric(training_metrics, "learner", "return_action_teacher_active_rate"),
            "env_total_score": _extract_numeric_metric(training_metrics, "env", "total_score"),
        }
        self.latest_training_metrics = deepcopy(training_metrics)
        self.learning_metric_history.append(snapshot)

        prev_clean = _history_average(self.learning_metric_history, "value_clean_loss", exclude_latest=True)
        prev_survive = _history_average(self.learning_metric_history, "value_survive_loss", exclude_latest=True)
        prev_entropy = _history_average(self.learning_metric_history, "entropy_loss", exclude_latest=True)
        prev_env = _history_average(self.learning_metric_history, "env_total_score", exclude_latest=True)

        self.latest_learning_metrics = {
            **snapshot,
            "value_clean_loss_trend_ratio": _ratio(snapshot.get("value_clean_loss"), prev_clean),
            "value_survive_loss_trend_ratio": _ratio(snapshot.get("value_survive_loss"), prev_survive),
            "entropy_trend_ratio": _ratio(snapshot.get("entropy_loss"), prev_entropy),
            "env_total_score_trend_ratio": _ratio(snapshot.get("env_total_score"), prev_env),
        }

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None and training_metrics is not False:
                    self._refresh_learning_metrics(training_metrics)
                    self.logger.info(f"training_metrics: {training_metrics}")
                    window_payload = {
                        "record_type": "workflow_window",
                        "episode_cnt": self.episode_cnt,
                        "rolling_episode_total": len(self.episode_history),
                    }
                    for group_name, group_metrics in training_metrics.items():
                        if not isinstance(group_metrics, dict):
                            continue
                        for key, value in group_metrics.items():
                            window_payload[f"{group_name}_{key}"] = value
                    self.archive.log_train_window(window_payload)
                    if now - self.last_perf_stat_time >= _env_int(
                        "KAIWU_PERF_STAT_WINDOW_SECONDS", Config.PERF_STAT_WINDOW_SECONDS
                    ):
                        window_payload.update(self.perf_window.flush("episode"))
                        runtime_metrics = self.agent.get_runtime_metrics() if hasattr(self.agent, "get_runtime_metrics") else {}
                        for key, value in runtime_metrics.items():
                            window_payload[f"agent_{key}"] = value
                        self.last_perf_stat_time = now

            curriculum_state = self.curriculum_store.refresh_state()
            sampled_usr_conf, sampled_meta = self.config_sampler.sample(
                curriculum_state.get("stage", "warmup")
            )
            sampled_env_conf = deepcopy(sampled_meta["env_conf"])
            env_obs = self.env.reset(sampled_usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                self.archive.log_event(
                    "disaster_recovery",
                    {
                        "stage": "env_reset",
                        "train_stage": sampled_meta["stage"],
                        "train_profile": sampled_meta["profile"],
                        "sampled_env_conf": sampled_env_conf,
                    },
                )
                continue

            self.agent.reset(env_obs)

            load_begin = time.perf_counter()
            self.agent.load_model(id="latest")
            self.perf_window.add("load_model", (time.perf_counter() - load_begin) * 1000.0)

            obs_begin = time.perf_counter()
            obs_data, _ = self.agent.observation_process(env_obs)
            self.perf_window.add("observation_process", (time.perf_counter() - obs_begin) * 1000.0)

            step_records = []
            self.episode_cnt += 1
            self.agent._predict_episode_idx = self.episode_cnt
            done = False
            step = 0
            total_reward = 0.0

            self.logger.info(
                f"Episode {self.episode_cnt} start "
                f"stage={sampled_meta['stage']} global_progress={curriculum_state.get('curriculum_progress', 0.0):.2f} "
                f"profile={sampled_meta['profile']} env={sampled_env_conf}"
            )

            while not done:
                predict_begin = time.perf_counter()
                act_data = self.agent.predict([obs_data])[0]
                self.perf_window.add("predict", (time.perf_counter() - predict_begin) * 1000.0)
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    self.archive.log_event(
                        "disaster_recovery",
                        {
                            "stage": "env_step",
                            "episode_cnt": self.episode_cnt,
                            "train_stage": sampled_meta["stage"],
                            "train_profile": sampled_meta["profile"],
                            "sampled_env_conf": sampled_env_conf,
                        },
                    )
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                frame_no = env_obs["frame_no"]
                step += 1
                done = terminated or truncated

                # Record death trajectory snapshot
                fm = self.agent.preprocessor
                self.death_trajectory_buffer.append({
                    "step": step, "battery": fm.battery, "battery_max": fm.battery_max,
                    "charger_slack": fm.charger_slack, "nearest_npc_dist": fm.nearest_npc_dist,
                    "mode": fm.current_mode, "action": self.agent.last_action,
                })
                if len(self.death_trajectory_buffer) > self.DEATH_TRAJ_LENGTH:
                    self.death_trajectory_buffer.pop(0)

                next_obs_begin = time.perf_counter()
                next_obs_data, _ = self.agent.observation_process(env_obs)
                self.perf_window.add("observation_process", (time.perf_counter() - next_obs_begin) * 1000.0)
                next_obs_data.frame_no = frame_no

                reward_payload = self._normalize_reward_payload(getattr(self.agent, "last_reward", 0.0))
                reward_total = float(reward_payload["reward_total"])
                total_reward += reward_total
                guidance = self.agent.preprocessor._get_guidance() if hasattr(self.agent.preprocessor, "_get_guidance") else {}
                battery_ratio = float(getattr(self.agent.preprocessor, "battery", 0.0)) / max(
                    float(getattr(self.agent.preprocessor, "battery_max", 1.0)), 1.0
                )
                wall_adjacent = int(getattr(self.agent.preprocessor, "wall_adjacent", 0))
                dirty_adjacent = int(getattr(self.agent.preprocessor, "dirty_adjacent", 0))
                cleaned_this_step = int(getattr(self.agent.preprocessor, "cleaned_this_step", 0))
                local_frontier_density = float(getattr(self.agent.preprocessor, "local_frontier_density", 0.0))
                wall_follow_streak = 0
                for rec in reversed(step_records):
                    if rec.get("wall_adjacent", 0) > 0:
                        wall_follow_streak += 1
                    else:
                        break
                if wall_adjacent > 0:
                    wall_follow_streak += 1
                wall_hugging_clean_floor = (
                    wall_adjacent > 0 and wall_follow_streak >= 4 and cleaned_this_step == 0 and dirty_adjacent == 0
                )
                stale_boundary_follow = wall_hugging_clean_floor and local_frontier_density <= 0.05
                local_passage_width = float(guidance.get("local_passage_width", 0.0))
                narrow_unknown_commit = bool(self.agent.preprocessor._is_narrow_unknown_commit(guidance, battery_ratio))
                missed_charge_opportunity = bool(self.agent.preprocessor._is_missed_charge_opportunity(guidance, battery_ratio))
                charger_nearby_not_charged = bool(
                    self.agent.preprocessor._is_charger_nearby_not_charged(guidance, battery_ratio)
                )
                suboptimal_target_hold = bool(self.agent.preprocessor._is_suboptimal_target_hold(guidance))
                planner_suggested_action = int(guidance.get("suggested_action", -1) if guidance.get("suggested_action") is not None else -1)
                selected_action = int(np.asarray(act_data.action).reshape(-1)[0])
                planner_action_match = planner_suggested_action < 0 or planner_suggested_action == selected_action
                all_charger_known_path_count = int(guidance.get("all_charger_known_path_count", 0))
                unknown_on_target_path_ratio = float(guidance.get("unknown_path_ratio", 0.0))
                target_selection_gap = float(guidance.get("target_gap", 0.0))
                charge_margin_now = float(guidance.get("margin", getattr(self.agent.preprocessor, "charger_slack", 0.0)))

                final_reward = 0.0
                if done:
                    final_reward = self._handle_episode_end(
                        env_obs=env_obs,
                        terminated=terminated,
                        truncated=truncated,
                        step=step,
                        total_reward=total_reward,
                        sampled_env_conf=sampled_env_conf,
                        sampled_meta=sampled_meta,
                        step_records=step_records,
                    )

                step_records.append(
                    {
                        "obs": np.array(obs_data.feature, dtype=np.float32),
                        "legal_action": np.array(obs_data.legal_action, dtype=np.float32),
                        "act": int(np.asarray(act_data.action).reshape(-1)[0]),
                        "prob": np.array(act_data.prob, dtype=np.float32).reshape(-1),
                        "done": float(done),
                        "reward_clean": float(reward_payload["reward_clean"]),
                        "reward_survive": float(reward_payload["reward_survive"]),
                        "value_clean": float(self._scalar_from_any(getattr(act_data, "value_clean", None))),
                        "value_survive": float(self._scalar_from_any(getattr(act_data, "value_survive", None))),
                        "mode": int(self._scalar_from_any(getattr(act_data, "mode", -1), default=-1)),
                        "target": int(self._scalar_from_any(getattr(act_data, "target", 0), default=0)),
                        "charger_slack": float(getattr(self.agent.preprocessor, "charger_slack", 0.0)),
                        "mode_teacher": int(reward_payload["mode_teacher"]),
                        "route_anchor_teacher": int(reward_payload["route_anchor_teacher"]),
                        "target_teacher": int(reward_payload["target_teacher"]),
                        "mode_teacher_mask": float(reward_payload["mode_teacher_mask"]),
                        "route_anchor_teacher_mask": float(reward_payload["route_anchor_teacher_mask"]),
                        "target_teacher_mask": float(reward_payload["target_teacher_mask"]),
                        "return_action_teacher": int(reward_payload["return_action_teacher"]),
                        "return_action_teacher_mask": float(reward_payload["return_action_teacher_mask"]),
                        "battery_risk_label": float(reward_payload["battery_risk_label"]),
                        "collision_risk_label": float(reward_payload["collision_risk_label"]),
                        "fallback_mask": float(reward_payload["fallback_mask"]),
                        "expert_weight": float(getattr(self.agent, "_last_expert_weight", 0.0)),
                        "route_anchor": int(self._scalar_from_any(getattr(act_data, "route_anchor", 0), default=0)),
                        "future_recoverability_score": float(getattr(self.agent.preprocessor, "future_recoverability_score", 0.0)),
                        "anchor_return_dist": float(getattr(self.agent.preprocessor, "anchor_return_dist", 0.0)),
                        "is_diag_action": 1.0 if int(np.asarray(act_data.action).reshape(-1)[0]) in (1, 3, 5, 7) else 0.0,
                        "wall_adjacent": wall_adjacent,
                        "dirty_adjacent": dirty_adjacent,
                        "cleaned_this_step": cleaned_this_step,
                        "local_frontier_density": local_frontier_density,
                        "same_region_streak": float(getattr(self.agent.preprocessor, "same_region_streak", 0.0)),
                        "path_cross_count_50": float(getattr(self.agent.preprocessor, "path_cross_count_50", 0.0)),
                        "coverage_efficiency_20": float(getattr(self.agent.preprocessor, "coverage_efficiency_20", 1.0)),
                        "wall_hugging_clean_floor": 1.0 if wall_hugging_clean_floor else 0.0,
                        "stale_boundary_follow": 1.0 if stale_boundary_follow else 0.0,
                        "narrow_unknown_commit": 1.0 if narrow_unknown_commit else 0.0,
                        "missed_charge_opportunity": 1.0 if missed_charge_opportunity else 0.0,
                        "charger_nearby_not_charged": 1.0 if charger_nearby_not_charged else 0.0,
                        "suboptimal_target_hold": 1.0 if suboptimal_target_hold else 0.0,
                        "planner_policy_divergence": 0.0 if planner_action_match else 1.0,
                        "planner_suggested_action": planner_suggested_action,
                        "planner_action_margin": float(guidance.get("action_margin", 0.0)),
                        "all_charger_known_path_count": float(all_charger_known_path_count),
                        "unknown_on_target_path_ratio": float(unknown_on_target_path_ratio),
                        "target_selection_gap": float(target_selection_gap),
                        "charge_margin_now": float(charge_margin_now),
                        "local_passage_width": float(local_passage_width),
                    }
                )

                if done:
                    step_records[-1]["reward_survive"] = float(step_records[-1]["reward_survive"]) + float(final_reward)

                    if self.is_new_best:
                        self._save_best_model(self.last_clean_score, self.last_checkpoint_scores)
                        self._save_resume_artifacts("best", self.last_clean_score, with_named_snapshot=True)

                    self._maybe_save_progress_snapshots()

                    if self.episode_cnt % self.save_interval == 0:
                        self.agent.save_model(path=str(self.manual_ckpt_dir), id=str(self.episode_cnt))
                        self.archive.log_event(
                            "framework_checkpoint_saved",
                            {
                                "episode_cnt": self.episode_cnt,
                                "checkpoint_id": str(self.episode_cnt),
                                "path": str(self.manual_ckpt_dir / f"model.ckpt-{self.episode_cnt}.pkl"),
                            },
                        )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        self.monitor.put_data({os.getpid(): self._build_monitor_payload(total_reward + final_reward)})
                        self.last_report_monitor_time = now

                    if step_records:
                        sample_process_begin = time.perf_counter()
                        collector = sample_process(step_records, episode_idx=self.episode_cnt)
                        self.perf_window.add("sample_process", (time.perf_counter() - sample_process_begin) * 1000.0)
                        self.perf_window.add("episodes_yielded", 0.0, count=1)
                        self.perf_window.add("samples_built", 0.0, count=len(collector))
                        yield collector
                    break

                obs_data = next_obs_data

    def _handle_episode_end(self, env_obs, terminated, truncated, step, total_reward, sampled_env_conf, sampled_meta, step_records):
        observation = env_obs.get("observation") or {}
        frame_state = observation.get("frame_state") or {}
        env_info = observation.get("env_info") or {}
        hero = frame_state.get("heroes") or {}
        fm = self.agent.preprocessor

        total_score = float(env_info.get("total_score", 0))
        clean_score = float(env_info.get("clean_score", total_score))
        battery = hero.get("battery")
        extra_info = env_obs.get("extra_info") or observation.get("extra_info") or {}

        fail_reason = infer_fail_reason(
            terminated=terminated,
            truncated=truncated,
            battery=battery,
            extra_info=extra_info,
        )

        cleaning_ratio = fm.dirt_cleaned / max(fm.total_dirt, 1)
        self.last_clean_score = clean_score
        _base_bonus = {
            "completed": 1.5,
            "battery": -3.0,
            "collision": -8.0,
            "unknown": -4.0,
        }.get(fail_reason, -4.0)

        if fail_reason in ("battery", "collision", "unknown"):
            actual_max_step = max(int(env_info.get("max_step", sampled_env_conf.get("max_step", step))), 1)
            remaining_ratio = max(0.0, 1.0 - float(step) / actual_max_step)
            outcome_bonus = _base_bonus * (1.0 + 1.5 * remaining_ratio)
        else:
            outcome_bonus = _base_bonus

        efficiency_bonus = 0.5 * cleaning_ratio + 0.5 * min(clean_score / max(step, 1), 1.0)
        final_reward = outcome_bonus + efficiency_bonus
        result_str = "WIN" if fail_reason == "completed" else "FAIL"

        # Death trajectory logging
        if fail_reason in ("battery", "collision"):
            traj_parts = []
            for s in self.death_trajectory_buffer:
                traj_parts.append(
                    f"s{s['step']}:bat={s['battery']}/{s['battery_max']}"
                    f" slack={s['charger_slack']:.1f} npc={s['nearest_npc_dist']:.0f}"
                    f" mode={s['mode']} act={s['action']}"
                )
            self.logger.info(
                f"[DEATH_TRAJ] ep:{self.episode_cnt} reason:{fail_reason} "
                f"traj=[{' | '.join(traj_parts)}]"
            )
            self.death_trajectory_buffer.clear()
        elif fail_reason == "completed":
            self.death_trajectory_buffer.clear()

        invalid_move_rate = fm.invalid_move_count / max(step, 1)
        charge_count = float(env_info.get("charge_count", 0))
        finished_steps = float(env_info.get("finished_steps", step))
        remaining_charge = float(env_info.get("remaining_charge", battery or 0))
        charge_efficiency = clean_score / max(charge_count, 1.0)
        clean_per_step = clean_score / max(finished_steps, 1.0)
        diagnostics = self._episode_sequence_diagnostics(step_records)

        self.episode_history.append({
            "result": fail_reason,
            "clean_score": clean_score,
            "finished_steps": finished_steps,
            "charge_count": charge_count,
            "remaining_charge": remaining_charge,
            "invalid_move_rate": invalid_move_rate,
            "charge_efficiency": charge_efficiency,
            "clean_per_step": clean_per_step,
            "expert_weight": getattr(self.agent, '_last_expert_weight', 0.0),
            "profile": sampled_meta['profile'],
            "total_reward": total_reward + final_reward,
            "late_return_rate": diagnostics["late_return_rate"],
            "late_contract_rate": diagnostics["late_contract_rate"],
            "anchor_switch_rate": diagnostics["anchor_switch_rate"],
            "target_switch_rate": diagnostics["target_switch_rate"],
            "diag_rate_all": diagnostics["diag_rate_all"],
            "diag_rate_contract": diagnostics["diag_rate_contract"],
            "diag_rate_return": diagnostics["diag_rate_return"],
            "return_progress_per_step": diagnostics["return_progress_per_step"],
            "return_efficiency_ratio": diagnostics["return_efficiency_ratio"],
            "return_stall_rate": diagnostics["return_stall_rate"],
            "recoverability_score_avg": diagnostics["recoverability_score_avg"],
            "recoverability_violation_rate": diagnostics["recoverability_violation_rate"],
            "wall_hugging_clean_floor_rate": diagnostics["wall_hugging_clean_floor_rate"],
            "stale_boundary_follow_rate": diagnostics["stale_boundary_follow_rate"],
            "narrow_unknown_commit_rate": diagnostics["narrow_unknown_commit_rate"],
            "missed_charge_opportunity_rate": diagnostics["missed_charge_opportunity_rate"],
            "charger_nearby_not_charged_rate": diagnostics["charger_nearby_not_charged_rate"],
            "suboptimal_target_hold_rate": diagnostics["suboptimal_target_hold_rate"],
            "planner_policy_divergence_rate": diagnostics["planner_policy_divergence_rate"],
            "avg_path_cross_count_50": diagnostics["avg_path_cross_count_50"],
            "avg_coverage_efficiency_20": diagnostics["avg_coverage_efficiency_20"],
            "avg_all_charger_known_path_count": diagnostics["avg_all_charger_known_path_count"],
            "avg_unknown_on_target_path_ratio": diagnostics["avg_unknown_on_target_path_ratio"],
            "mode_usage_depart": diagnostics["mode_usage_depart"],
            "mode_usage_expand": diagnostics["mode_usage_expand"],
            "mode_usage_harvest": diagnostics["mode_usage_harvest"],
            "mode_usage_contract": diagnostics["mode_usage_contract"],
            "mode_usage_return": diagnostics["mode_usage_return"],
            "mode_usage_evade": diagnostics["mode_usage_evade"],
        })
        curriculum_state = self._emit_curriculum_signal()

        map_id = extra_info.get("map_id") or extra_info.get("map_code") or "?"
        actual_robot_count = int(env_info.get("npc_count", sampled_env_conf.get("robot_count", 1)))
        actual_charger_count = int(env_info.get("total_charger", sampled_env_conf.get("charger_count", 4)))
        self.logger.info(
            f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} "
            f"result:{result_str} final_bonus:{final_reward:.2f} "
            f"total_reward:{total_reward:.3f} clean_score:{clean_score:.1f} "
            f"dirt_cleaned:{fm.dirt_cleaned}/{fm.total_dirt} "
            f"invalid_move_rate:{invalid_move_rate:.3f} "
            f"profile:{sampled_meta['profile']} "
            f"map:{map_id} chargers:{actual_charger_count} robots:{actual_robot_count}"
        )

        self.is_new_best = False

        # Log curriculum metrics periodically
        cur_metrics = self._get_curriculum_metrics()
        if cur_metrics is not None and self.episode_cnt % 10 == 0:
            self.logger.info(
                f"[CURRICULUM] ep:{self.episode_cnt} local_stage:{sampled_meta['stage']} "
                f"global_stage:{curriculum_state.get('stage', 'warmup')} "
                f"progress:{curriculum_state.get('curriculum_progress', 0.0):.2f} "
                f"win_rate:{cur_metrics['win_rate']:.2f} "
                f"avg_cs:{cur_metrics['avg_clean_score']:.0f} avg_cc:{cur_metrics['avg_charge_count']:.1f} "
                f"battery_fail:{cur_metrics['battery_fail_rate']:.2f} "
                f"return_stall:{cur_metrics['return_stall_rate']:.2f}"
            )

        # Per-map score tracking for generalization monitoring
        if map_id != "?":
            self.per_map_scores.setdefault(str(map_id), []).append(clean_score)
            if len(self.per_map_scores[str(map_id)]) > 30:
                self.per_map_scores[str(map_id)] = self.per_map_scores[str(map_id)][-30:]

        # Log cross-map variance every 10 episodes
        if self.episode_cnt % 10 == 0 and len(self.per_map_scores) >= 3:
            map_avgs = {}
            for mid, scores in sorted(self.per_map_scores.items()):
                if len(scores) >= 3:
                    map_avgs[mid] = round(sum(scores[-10:]) / len(scores[-10:]), 1)
            if len(map_avgs) >= 3:
                avg_vals = list(map_avgs.values())
                variance = float(np.std(avg_vals))
                min_avg = min(avg_vals)
                spread = max(avg_vals) - min_avg
                self.logger.info(
                    f"[MAP_STATS] maps:{map_avgs} "
                    f"variance:{variance:.1f} min_avg:{min_avg:.1f} spread:{spread:.1f}"
                )

        scores = [ep["clean_score"] for ep in self.episode_history]
        robust_score = float("-inf")
        if len(scores) >= 20:
            rolling_avg = sum(scores) / len(scores)
            robust_score = compute_legacy_robust_score(scores, invalid_move_rate, fail_reason)
            if rolling_avg > self.best_avg_score:
                self.best_avg_score = rolling_avg
            if robust_score > self.best_robust_score:
                self.best_robust_score = robust_score

        window_metrics = self._window_metrics()
        try:
            checkpoint_scores = compute_checkpoint_scores(window_metrics, self.latest_learning_metrics)
        except Exception as exc:
            self.logger.warning(f"[CHECKPOINT_SCORE] ep:{self.episode_cnt} fallback due to error: {exc}")
            checkpoint_scores = {
                "resume_eligible": False,
                "submission_eligible": False,
                "resume_readiness_score": 0.0,
                "submission_score": 0.0,
                "training_stability_bonus": 0.0,
                "benchmark_stability_bonus": 0.0,
                "checkpoint_preservation_score": 0.0,
            }
        self.last_checkpoint_scores = checkpoint_scores
        self.best_resume_readiness_score = max(
            self.best_resume_readiness_score, checkpoint_scores["resume_readiness_score"]
        )
        self.best_submission_score = max(
            self.best_submission_score, checkpoint_scores["submission_score"]
        )
        if checkpoint_scores["resume_eligible"] and (
            checkpoint_scores["checkpoint_preservation_score"] > self.best_preservation_score
        ):
            self.best_preservation_score = checkpoint_scores["checkpoint_preservation_score"]
            self.is_new_best = True

        checkpoint_ref = getattr(self.agent, "current_model_ref", {}) or {}
        actual_battery_max = int(hero.get("battery_max", env_info.get("battery_max", sampled_env_conf.get("battery_max", 200))))
        actual_max_step = int(env_info.get("max_step", sampled_env_conf.get("max_step", step)))
        episode_payload = {
            "episode_id": self.episode_cnt,
            "checkpoint_id": checkpoint_ref.get("checkpoint_id"),
            "checkpoint_path": checkpoint_ref.get("path"),
            "map_id": extra_info.get("map_id") or extra_info.get("map_code"),
            "result": result_str.lower(),
            "fail_reason": fail_reason,
            "total_score": total_score,
            "clean_score": clean_score,
            "finished_steps": finished_steps,
            "charge_count": charge_count,
            "remaining_charge": remaining_charge,
            "total_reward": round(total_reward + final_reward, 4),
            "robot_count": actual_robot_count,
            "charger_count": actual_charger_count,
            "battery_max": actual_battery_max,
            "max_step": actual_max_step,
            "map_random": int(env_info.get("map_random", sampled_env_conf.get("map_random", 0))),
            "train_stage": sampled_meta["stage"],
            "train_profile": sampled_meta["profile"],
            "invalid_move_rate": round(invalid_move_rate, 4),
            "charge_efficiency": round(charge_efficiency, 4),
            "clean_per_step": round(clean_per_step, 4),
            "legacy_robust_score": round(float(robust_score), 4) if robust_score != float("-inf") else None,
            "resume_readiness_score": checkpoint_scores["resume_readiness_score"],
            "submission_score": checkpoint_scores["submission_score"],
            "checkpoint_preservation_score": checkpoint_scores["checkpoint_preservation_score"],
            "resume_eligible": checkpoint_scores["resume_eligible"],
            "mode": int(getattr(fm, "current_mode", -1)),
            "sampled_env_conf": deepcopy(sampled_env_conf),
        }
        self.archive.log_episode_summary(episode_payload)
        self.archive.log_event("episode_end", episode_payload)
        if fail_reason == "battery":
            self.archive.log_event("battery_fail", episode_payload)
        elif fail_reason == "collision":
            self.archive.log_event("collision_fail", episode_payload)

        # Per-config failure rate tracking
        battery_bin = min(int(actual_battery_max / 100) * 100, 800)
        max_step_bin = min(int(actual_max_step / 500) * 500, 2000)
        config_key = f"r{actual_robot_count}_c{actual_charger_count}_b{battery_bin}_s{max_step_bin}"
        if config_key not in self.config_stats:
            self.config_stats[config_key] = {"total": 0, "battery": 0, "collision": 0, "completed": 0}
        stats = self.config_stats[config_key]
        stats["total"] += 1
        stats[fail_reason] = stats.get(fail_reason, 0) + 1

        if self.episode_cnt % 50 == 0 and len(self.config_stats) >= 3:
            high_death = []
            for ck, cs in sorted(self.config_stats.items()):
                if cs["total"] >= 3:
                    dr = (cs.get("battery", 0) + cs.get("collision", 0)) / cs["total"]
                    if dr > 0.3:
                        high_death.append(f"{ck}:{dr:.0%}({cs['total']}ep)")
            if high_death:
                self.logger.info(f"[CONFIG_RISK] {' '.join(high_death)}")

        return final_reward

    @staticmethod
    def _scalar_from_any(value, default=0.0):
        if value is None:
            return float(default)
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return float(default)
        return float(arr[0])

    def _normalize_reward_payload(self, reward):
        if isinstance(reward, dict):
            reward_clean = float(reward.get("reward_clean", reward.get("clean", 0.0)))
            reward_survive = float(
                reward.get("reward_survive", reward.get("survive", reward.get("total", reward_clean)))
            )
            reward_total = float(reward.get("reward_total", reward.get("total", reward_clean + reward_survive)))
            return {
                "reward_clean": reward_clean,
                "reward_survive": reward_survive,
                "reward_total": reward_total,
                "mode_teacher": int(reward.get("mode_teacher", -1)),
                "route_anchor_teacher": int(reward.get("route_anchor_teacher", 0)),
                "target_teacher": int(reward.get("target_teacher", 0)),
                "mode_teacher_mask": float(reward.get("mode_teacher_mask", 0.0)),
                "route_anchor_teacher_mask": float(reward.get("route_anchor_teacher_mask", 0.0)),
                "target_teacher_mask": float(reward.get("target_teacher_mask", 0.0)),
                "return_action_teacher": int(reward.get("return_action_teacher", -1)),
                "return_action_teacher_mask": float(reward.get("return_action_teacher_mask", 0.0)),
                "battery_risk_label": float(reward.get("battery_risk_label", 0.0)),
                "collision_risk_label": float(reward.get("collision_risk_label", 0.0)),
                "fallback_mask": float(reward.get("fallback_mask", 0.0)),
            }

        reward_scalar = float(reward)
        return {
            "reward_clean": reward_scalar,
            "reward_survive": 0.0,
            "reward_total": reward_scalar,
            "mode_teacher": -1,
            "route_anchor_teacher": 0,
            "target_teacher": 0,
            "mode_teacher_mask": 0.0,
            "route_anchor_teacher_mask": 0.0,
            "target_teacher_mask": 0.0,
            "return_action_teacher": -1,
            "return_action_teacher_mask": 0.0,
            "battery_risk_label": 0.0,
            "collision_risk_label": 0.0,
            "fallback_mask": 0.0,
        }

    @staticmethod
    def _episode_sequence_diagnostics(step_records):
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
                "wall_hugging_clean_floor_rate": 0.0,
                "stale_boundary_follow_rate": 0.0,
                "narrow_unknown_commit_rate": 0.0,
                "missed_charge_opportunity_rate": 0.0,
                "charger_nearby_not_charged_rate": 0.0,
                "suboptimal_target_hold_rate": 0.0,
                "planner_policy_divergence_rate": 0.0,
                "avg_path_cross_count_50": 0.0,
                "avg_coverage_efficiency_20": 0.0,
                "avg_all_charger_known_path_count": 0.0,
                "avg_unknown_on_target_path_ratio": 0.0,
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

        target_steps = [t for t in targets if t > 0]
        target_switches = sum(1 for a, b in zip(target_steps, target_steps[1:]) if a != b)
        target_switch_rate = target_switches / max(len(target_steps) - 1, 1)
        anchor_steps = [a for a in anchors if a > 0]
        anchor_switches = sum(1 for a, b in zip(anchor_steps, anchor_steps[1:]) if a != b)
        anchor_switch_rate = anchor_switches / max(len(anchor_steps) - 1, 1)

        first_contract_idx = next((idx for idx, mode in enumerate(modes) if mode in (3, 4)), None)
        late_contract_rate = 1.0 if first_contract_idx is not None and recoverability[first_contract_idx] < 0.0 else 0.0
        first_return_idx = next((idx for idx, mode in enumerate(modes) if mode == 4), None)
        if first_return_idx is None:
            late_return_rate = 1.0
        else:
            late_return_rate = 1.0 if slacks[first_return_idx] < 0.0 else 0.0

        contract_steps = [idx for idx, mode in enumerate(modes) if mode == 3]
        return_steps = [idx for idx, mode in enumerate(modes) if mode == 4]
        diag_rate_all = sum(diag_actions) / total
        diag_rate_contract = sum(diag_actions[idx] for idx in contract_steps) / max(len(contract_steps), 1)
        diag_rate_return = sum(diag_actions[idx] for idx in return_steps) / max(len(return_steps), 1)
        route_phase_steps = [idx for idx, mode in enumerate(modes) if mode in (3, 4)]
        progress_deltas = []
        stall_count = 0
        for prev_idx, cur_idx in zip(route_phase_steps, route_phase_steps[1:]):
            progress = anchor_dists[prev_idx] - anchor_dists[cur_idx]
            progress_deltas.append(progress)
            if progress <= 0.0:
                stall_count += 1
        return_progress_per_step = float(sum(progress_deltas) / max(len(progress_deltas), 1))
        return_efficiency_ratio = float(
            (anchor_dists[route_phase_steps[0]] / max(len(route_phase_steps), 1))
            if route_phase_steps and anchor_dists[route_phase_steps[0]] > 0.0
            else 0.0
        )
        return_stall_rate = float(stall_count / max(len(progress_deltas), 1))
        wall_hugging_clean_floor_rate = float(
            sum(float(rec.get("wall_hugging_clean_floor", 0.0)) for rec in step_records) / total
        )
        stale_boundary_follow_rate = float(
            sum(float(rec.get("stale_boundary_follow", 0.0)) for rec in step_records) / total
        )
        narrow_unknown_commit_rate = float(
            sum(float(rec.get("narrow_unknown_commit", 0.0)) for rec in step_records) / total
        )
        missed_charge_opportunity_rate = float(
            sum(float(rec.get("missed_charge_opportunity", 0.0)) for rec in step_records) / total
        )
        charger_nearby_not_charged_rate = float(
            sum(float(rec.get("charger_nearby_not_charged", 0.0)) for rec in step_records) / total
        )
        suboptimal_target_hold_rate = float(
            sum(float(rec.get("suboptimal_target_hold", 0.0)) for rec in step_records) / total
        )
        planner_policy_divergence_rate = float(
            sum(float(rec.get("planner_policy_divergence", 0.0)) for rec in step_records) / total
        )

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
            "recoverability_violation_rate": float(sum(1 for x in recoverability if x < 0.0) / max(len(recoverability), 1)),
            "wall_hugging_clean_floor_rate": wall_hugging_clean_floor_rate,
            "stale_boundary_follow_rate": stale_boundary_follow_rate,
            "narrow_unknown_commit_rate": narrow_unknown_commit_rate,
            "missed_charge_opportunity_rate": missed_charge_opportunity_rate,
            "charger_nearby_not_charged_rate": charger_nearby_not_charged_rate,
            "suboptimal_target_hold_rate": suboptimal_target_hold_rate,
            "planner_policy_divergence_rate": planner_policy_divergence_rate,
            "avg_path_cross_count_50": float(sum(float(rec.get("path_cross_count_50", 0.0)) for rec in step_records) / total),
            "avg_coverage_efficiency_20": float(sum(float(rec.get("coverage_efficiency_20", 0.0)) for rec in step_records) / total),
            "avg_all_charger_known_path_count": float(
                sum(float(rec.get("all_charger_known_path_count", 0.0)) for rec in step_records) / total
            ),
            "avg_unknown_on_target_path_ratio": float(
                sum(float(rec.get("unknown_on_target_path_ratio", 0.0)) for rec in step_records) / total
            ),
            "mode_usage_depart": sum(1 for mode in modes if mode == 0) / total,
            "mode_usage_expand": sum(1 for mode in modes if mode == 1) / total,
            "mode_usage_harvest": sum(1 for mode in modes if mode == 2) / total,
            "mode_usage_contract": sum(1 for mode in modes if mode == 3) / total,
            "mode_usage_return": sum(1 for mode in modes if mode == 4) / total,
            "mode_usage_evade": sum(1 for mode in modes if mode == 5) / total,
        }

    def _build_monitor_payload(self, reward):
        m = self._window_metrics()
        if not m:
            return {"reward": reward, "episode_cnt": self.episode_cnt}
        payload = {
            "reward": reward,
            "episode_cnt": self.episode_cnt,
            "avg_episode_steps": round(m["avg_finished_steps"], 2),
            "avg_charge_count": round(m["avg_charge_count"], 2),
            "avg_cleaned_cells": round(m["avg_clean_score"], 2),
            "avg_remaining_charge": round(m["avg_remaining_charge"], 2),
            "avg_invalid_move_rate": round(m["avg_invalid_move_rate"], 4),
            "avg_charge_efficiency": round(m["avg_charge_efficiency"], 4),
            "avg_clean_per_step": round(m["avg_clean_per_step"], 4),
            "battery_fail_rate": round(m["battery_fail_rate"], 4),
            "collision_fail_rate": round(m["collision_fail_rate"], 4),
            "completed_rate": round(m["win_rate"], 4),
            "cps_win": round(m["cps_win"], 4),
            "avg_charge_count_win": round(m["avg_charge_count_win"], 2),
            "avg_clean_score_win": round(m["avg_clean_score_win"], 2),
            "avg_expert_weight": round(m["avg_expert_weight"], 2),
            "late_return_rate": round(m["late_return_rate"], 4),
            "late_contract_rate": round(m["late_contract_rate"], 4),
            "anchor_switch_rate": round(m["anchor_switch_rate"], 4),
            "target_switch_rate": round(m["target_switch_rate"], 4),
            "diag_rate_all": round(m["diag_rate_all"], 4),
            "diag_rate_contract": round(m["diag_rate_contract"], 4),
            "diag_rate_return": round(m["diag_rate_return"], 4),
            "return_progress_per_step": round(m["return_progress_per_step"], 4),
            "return_efficiency_ratio": round(m["return_efficiency_ratio"], 4),
            "return_stall_rate": round(m["return_stall_rate"], 4),
            "recoverability_score_avg": round(m["recoverability_score_avg"], 4),
            "recoverability_violation_rate": round(m["recoverability_violation_rate"], 4),
            "wall_hugging_clean_floor_rate": round(m["wall_hugging_clean_floor_rate"], 4),
            "stale_boundary_follow_rate": round(m["stale_boundary_follow_rate"], 4),
            "narrow_unknown_commit_rate": round(m["narrow_unknown_commit_rate"], 4),
            "missed_charge_opportunity_rate": round(m["missed_charge_opportunity_rate"], 4),
            "charger_nearby_not_charged_rate": round(m["charger_nearby_not_charged_rate"], 4),
            "suboptimal_target_hold_rate": round(m["suboptimal_target_hold_rate"], 4),
            "planner_policy_divergence_rate": round(m["planner_policy_divergence_rate"], 4),
            "avg_path_cross_count_50": round(m["avg_path_cross_count_50"], 4),
            "avg_coverage_efficiency_20": round(m["avg_coverage_efficiency_20"], 4),
            "avg_all_charger_known_path_count": round(m["avg_all_charger_known_path_count"], 4),
            "avg_unknown_on_target_path_ratio": round(m["avg_unknown_on_target_path_ratio"], 4),
            "mode_usage_depart": round(m["mode_usage_depart"], 4),
            "mode_usage_expand": round(m["mode_usage_expand"], 4),
            "mode_usage_harvest": round(m["mode_usage_harvest"], 4),
            "mode_usage_contract": round(m["mode_usage_contract"], 4),
            "mode_usage_return": round(m["mode_usage_return"], 4),
            "mode_usage_evade": round(m["mode_usage_evade"], 4),
            "anchor_win_rate": round(m["anchor_win_rate"], 4) if m["anchor_win_rate"] >= 0 else -1,
            "mild_win_rate": round(m["mild_win_rate"], 4) if m["mild_win_rate"] >= 0 else -1,
            "broad_win_rate": round(m["broad_win_rate"], 4) if m["broad_win_rate"] >= 0 else -1,
            "resume_readiness_score": round(self.last_checkpoint_scores["resume_readiness_score"], 4),
            "submission_score": round(self.last_checkpoint_scores["submission_score"], 4),
            "checkpoint_preservation_score": round(self.last_checkpoint_scores["checkpoint_preservation_score"], 4),
            "resume_eligible": 1.0 if self.last_checkpoint_scores["resume_eligible"] else 0.0,
        }
        payload.update(self._curriculum_progress_payload())
        return payload


def _extract_numeric_metric(payload, group, key, default=None):
    if not isinstance(payload, dict):
        return default
    group_payload = payload.get(group, {})
    if not isinstance(group_payload, dict):
        return default
    try:
        value = group_payload.get(key, default)
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _history_average(history, key, exclude_latest=False):
    items = list(history)
    if exclude_latest and items:
        items = items[:-1]
    values = []
    for item in items:
        try:
            value = item.get(key)
            if value is None:
                continue
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    if not values:
        return None
    return sum(values) / len(values)


def _ratio(current, previous):
    try:
        current = float(current)
    except (TypeError, ValueError):
        return 1.0
    if previous in (None, 0.0):
        return 1.0
    try:
        previous = float(previous)
    except (TypeError, ValueError):
        return 1.0
    if abs(previous) < 1e-6:
        return 1.0
    return current / previous


def resolve_shared_code_dir():
    shared_candidates = [Path("/workspace/code")]
    env_shared_code_dir = os.getenv("KAIWU_SHARED_CODE_DIR")
    if env_shared_code_dir:
        shared_candidates.append(Path(env_shared_code_dir))
    for candidate in shared_candidates:
        if candidate.exists():
            return candidate.resolve()
    return Path(__file__).resolve().parents[2]
