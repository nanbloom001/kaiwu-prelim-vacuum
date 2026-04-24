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
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from agent_ppo.algorithm.algorithm import CoveragePlanner
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from agent_ppo.utils.experiment_archive import ExperimentArchive, infer_fail_reason
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env = envs[0]
    agent = agents[0]
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
                "policy_blending": "coverage_planner_residual_policy",
                "curriculum": "win_env_sampler_with_yjy_defaults",
                "selection_target": "clean_score_with_robustness",
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

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()


class ResidualScheduler:
    """Adaptive scheduler that gradually lets PPO take more control."""

    def __init__(self):
        self.alpha = Config.RESIDUAL_ALPHA_START
        self.ema_score = None
        self.best_ema = float("-inf")
        self.stale_episodes = 0
        self.episode_cnt = 0

    def action_alpha(self, target_mode):
        alpha = self.alpha
        if target_mode == "charge":
            alpha = min(alpha, Config.RESIDUAL_ALPHA_CHARGE_CAP)
        elif target_mode == "fallback":
            alpha = min(alpha, Config.RESIDUAL_ALPHA_FALLBACK_CAP)
        return float(np.clip(alpha, 0.0, Config.RESIDUAL_ALPHA_MAX))

    def update(self, episode_score, cleaning_ratio):
        self.episode_cnt += 1

        decay = Config.RESIDUAL_SCORE_EMA_DECAY
        if self.ema_score is None:
            self.ema_score = float(episode_score)
        else:
            self.ema_score = decay * self.ema_score + (1.0 - decay) * float(episode_score)

        warmup_t = min(1.0, self.episode_cnt / max(Config.RESIDUAL_WARMUP_EPISODES, 1))
        warmup_alpha = (
            Config.RESIDUAL_ALPHA_START
            + (Config.RESIDUAL_ALPHA_WARMUP_TARGET - Config.RESIDUAL_ALPHA_START) * warmup_t
        )
        self.alpha = max(self.alpha, warmup_alpha)

        if self.ema_score > self.best_ema + Config.RESIDUAL_SCORE_IMPROVE:
            self.best_ema = self.ema_score
            self.stale_episodes = 0
        else:
            self.stale_episodes += 1

        if (
            self.stale_episodes >= Config.RESIDUAL_PLATEAU_PATIENCE
            and self.ema_score >= Config.RESIDUAL_PLATEAU_SCORE
        ):
            bonus = Config.RESIDUAL_ALPHA_STEP * (1.0 + 0.5 * max(0.0, cleaning_ratio - 0.85))
            self.alpha = min(Config.RESIDUAL_ALPHA_MAX, self.alpha + bonus)
            self.stale_episodes = 0
        elif self.best_ema - self.ema_score >= Config.RESIDUAL_SCORE_DROP:
            self.alpha = max(
                Config.RESIDUAL_ALPHA_START,
                self.alpha - 0.5 * Config.RESIDUAL_ALPHA_STEP,
            )
            self.stale_episodes = 0

        return self.alpha


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

    def sample(self, episode_idx):
        stage = self._stage_name(episode_idx)
        profile = "fixed"
        env_conf = deepcopy(self.base_env_conf)
        env_conf["map_random"] = True
        env_conf["map"] = list(self.maps)
        env_conf["robot_count"] = self.base_robot_count
        env_conf["charger_count"] = self.base_charger_count
        env_conf["max_step"] = self.base_max_step
        env_conf["battery_max"] = self.base_battery_max

        sampled_usr_conf = self._wrap_env_conf(env_conf)
        meta = {
            "stage": stage,
            "profile": profile,
            "env_conf": deepcopy(env_conf),
        }
        return sampled_usr_conf, meta

    def _stage_name(self, episode_idx):
        if episode_idx <= 40:
            return "warmup"
        if episode_idx <= 200:
            return "blend"
        if episode_idx <= 400:
            return "robust"
        return "eval_hard"

    def _pick_profile(self, stage):
        draw = self.rng.random()
        if stage == "warmup":
            if draw < 0.70:
                return "anchor"
            if draw < 0.92:
                return "mild"
            return "broad"
        if stage == "blend":
            if draw < 0.35:
                return "anchor"
            if draw < 0.75:
                return "mild"
            return "broad"
        if stage == "robust":
            if draw < 0.10:
                return "anchor"
            if draw < 0.45:
                return "mild"
            return "broad"
        # eval_hard
        if draw < 0.05:
            return "anchor"
        if draw < 0.25:
            return "mild"
        return "broad_eval"

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
        self.planner = CoveragePlanner()
        self.scheduler = ResidualScheduler()
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0
        self.current_mix_alpha = Config.RESIDUAL_ALPHA_START
        self.current_target_mode = "explore"
        self.current_episode_invalid_moves = 0
        self.last_charger_arrived_count = 0
        self.last_charger_first_arrival_step = -1
        self.last_charger_second_arrival_step = -1
        self.last_charger_third_arrival_step = -1

        self.failure_counts = {
            "battery": 0,
            "collision": 0,
            "completed": 0,
            "unknown": 0,
        }

        self.rolling_charge_total = 0.0
        self.rolling_cleaned_total = 0.0
        self.rolling_finished_steps = 0.0
        self.rolling_remaining_charge_total = 0.0
        self.rolling_invalid_move_rate_total = 0.0
        self.rolling_charge_efficiency_total = 0.0
        self.rolling_clean_per_step_total = 0.0
        self.rolling_episode_total = 0

        self.death_trajectory_buffer = []
        self.DEATH_TRAJ_LENGTH = 20
        self.config_stats = {}

        self.score_window = []
        self.best_avg_score = 0.0
        self.best_robust_score = float("-inf")
        self.last_clean_score = 0.0
        self.is_new_best = False
        self.per_map_scores: dict[str, list[float]] = {}
        self.code_path = resolve_shared_code_dir()
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
        self.save_interval = Config.SAVE_MODEL_INTERVAL_EPISODES
        self.resume_episode_interval = Config.RESUME_EPISODE_SNAPSHOT_INTERVAL
        self.latest_sync_interval = Config.RESUME_LATEST_SYNC_INTERVAL_EPISODES
        self.time_save_interval_seconds = Config.RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS
        self.keep_episode_snapshots = Config.KEEP_EPISODE_RESUME_SNAPSHOTS
        self.keep_time_snapshots = Config.KEEP_TIME_RESUME_SNAPSHOTS
        self.keep_best_snapshots = Config.KEEP_BEST_RESUME_SNAPSHOTS

    def _persist_best_score(self):
        data = {
            "best_robust_score": self.best_robust_score,
            "best_avg_score": self.best_avg_score,
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
            "episode_cnt": self.episode_cnt,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        tmp = manifest_path.parent / f".manifest.{os.getpid()}.{time.time_ns()}.tmp"
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=True, indent=2)
        os.replace(tmp, manifest_path)

    def _save_best_model(self, clean_score):
        best_path = self.session_dir / "best_model.pkl"
        state_dict = {k: v.clone().cpu() for k, v in self.agent.model.state_dict().items()}
        torch.save(state_dict, best_path)
        self._persist_best_score()
        self._update_manifest()
        self.logger.info(
            f"[BEST] session={self.session_id} ep={self.episode_cnt} avg={self.best_avg_score:.2f} "
            f"robust={self.best_robust_score:.2f} score={clean_score:.1f} saved to {best_path}"
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

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics: {training_metrics}")
                    window_payload = {
                        "record_type": "workflow_window",
                        "episode_cnt": self.episode_cnt,
                        "rolling_episode_total": self.rolling_episode_total,
                    }
                    for group_name, group_metrics in training_metrics.items():
                        if not isinstance(group_metrics, dict):
                            continue
                        for key, value in group_metrics.items():
                            window_payload[f"{group_name}_{key}"] = value
                    self.archive.log_train_window(window_payload)

            sampled_usr_conf, sampled_meta = self.config_sampler.sample(self.episode_cnt + 1)
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
            self.planner.reset()
            episode_max_step = int(sampled_env_conf.get("max_step", 1000))
            episode_robot_count = int(sampled_env_conf.get("robot_count", 4))
            episode_charger_count = int(sampled_env_conf.get("charger_count", 4))
            episode_battery_max = int(sampled_env_conf.get("battery_max", 200))
            self.agent.set_episode_config(
                max_step=episode_max_step,
                robot_count=episode_robot_count,
                charger_count=episode_charger_count,
                battery_max=episode_battery_max,
            )
            self.planner.set_episode_config(
                max_step=episode_max_step,
                robot_count=episode_robot_count,
                charger_count=episode_charger_count,
                battery_max=episode_battery_max,
            )
            self.agent.load_model(id="latest")

            obs_data, _ = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0
            last_alpha = self.scheduler.action_alpha(self.current_target_mode)
            self.current_episode_invalid_moves = 0

            self.logger.info(
                f"Episode {self.episode_cnt} start "
                f"stage={sampled_meta['stage']} profile={sampled_meta['profile']} "
                f"alpha={last_alpha:.3f} max_step={episode_max_step} "
                f"robot_count={episode_robot_count} charger_count={episode_charger_count} "
                f"battery_max={episode_battery_max} env={sampled_env_conf}"
            )

            while not done:
                pre_step_obs = env_obs.get("observation") or {}
                pre_frame_state = pre_step_obs.get("frame_state") or {}
                pre_hero = pre_frame_state.get("heroes") or {}
                pre_pos = pre_hero.get("pos") or {}
                prev_pos = (int(pre_pos.get("x", 0)), int(pre_pos.get("z", 0)))

                policy_info = self.planner.update(env_obs, self.agent.last_action)
                self.current_target_mode = policy_info.target_mode
                last_alpha = self.scheduler.action_alpha(self.current_target_mode)
                act_data = self.agent.guided_predict(
                    [obs_data],
                    policy_info=policy_info,
                    residual_alpha=last_alpha,
                )[0]
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

                post_step_obs = env_obs.get("observation") or {}
                post_frame_state = post_step_obs.get("frame_state") or {}
                post_hero = post_frame_state.get("heroes") or {}
                post_pos = post_hero.get("pos") or {}
                next_pos = (int(post_pos.get("x", 0)), int(post_pos.get("z", 0)))
                if next_pos == prev_pos:
                    self.current_episode_invalid_moves += 1

                # Record death trajectory snapshot
                fm = self.agent.preprocessor
                self.death_trajectory_buffer.append({
                    "step": step,
                    "battery": getattr(fm, "battery", 0),
                    "battery_max": getattr(fm, "battery_max", 0),
                    "charger_slack": float(getattr(policy_info, "charger_slack", 0.0)),
                    "nearest_npc_dist": float(getattr(policy_info, "nearest_npc_distance", 999.0)),
                    "mode": self.current_target_mode,
                    "action": self.agent.last_action,
                })
                if len(self.death_trajectory_buffer) > self.DEATH_TRAJ_LENGTH:
                    self.death_trajectory_buffer.pop(0)

                next_obs_data, _ = self.agent.observation_process(env_obs)
                next_obs_data.frame_no = frame_no

                reward_scalar = float(self.agent.last_reward)
                total_reward += reward_scalar

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
                    )

                reward_arr = np.array([reward_scalar], dtype=np.float32)
                value_arr = act_data.value.flatten()[: Config.VALUE_NUM]

                collector.append(
                    SampleData(
                        obs=np.array(obs_data.feature, dtype=np.float32),
                        legal_action=np.array(act_data.action_mask, dtype=np.float32),
                        act=np.array(act_data.action),
                        reward=reward_arr,
                        done=np.array([float(done)]),
                        reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                        value=value_arr,
                        next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                        advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                        prob=np.array(act_data.prob, dtype=np.float32),
                        planner_prob=np.array(act_data.planner_prob, dtype=np.float32),
                        mix_alpha=np.array(act_data.mix_alpha, dtype=np.float32),
                    )
                )

                if done:
                    collector[-1].reward = collector[-1].reward + np.array([final_reward], dtype=np.float32)
                    cleaning_ratio = self.agent.preprocessor.dirt_cleaned / max(self.agent.preprocessor.total_dirt, 1)
                    self.current_mix_alpha = self.scheduler.update(self.last_clean_score, cleaning_ratio)
                    self.logger.info(
                        f"[RESIDUAL] ep={self.episode_cnt} mode={self.current_target_mode} "
                        f"alpha={last_alpha:.3f}->{self.current_mix_alpha:.3f} "
                        f"score={self.last_clean_score:.1f} clean_ratio={cleaning_ratio:.3f}"
                    )

                    if self.is_new_best:
                        self._save_best_model(self.last_clean_score)
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

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = next_obs_data

    def _handle_episode_end(self, env_obs, terminated, truncated, step, total_reward, sampled_env_conf, sampled_meta):
        observation = env_obs.get("observation") or {}
        frame_state = observation.get("frame_state") or {}
        env_info = observation.get("env_info") or {}
        hero = frame_state.get("heroes") or {}
        fm = self.agent.preprocessor

        total_score = float(env_info.get("total_score", 0))
        clean_score = float(env_info.get("clean_score", total_score))
        battery = hero.get("battery")
        extra_info = env_obs.get("extra_info") or observation.get("extra_info") or {}
        arrival_steps = sorted(int(step_no) for step_no in getattr(fm, "charger_arrival_steps", {}).values())
        charger_arrived_count = len(arrival_steps)
        first_arrival_step = arrival_steps[0] if charger_arrived_count >= 1 else -1
        second_arrival_step = arrival_steps[1] if charger_arrived_count >= 2 else -1
        third_arrival_step = arrival_steps[2] if charger_arrived_count >= 3 else -1
        self.last_charger_arrived_count = charger_arrived_count
        self.last_charger_first_arrival_step = first_arrival_step
        self.last_charger_second_arrival_step = second_arrival_step
        self.last_charger_third_arrival_step = third_arrival_step

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
        invalid_move_rate = self.current_episode_invalid_moves / max(step, 1)
        charge_count = float(env_info.get("charge_count", 0))
        finished_steps = float(env_info.get("finished_steps", step))
        remaining_charge = float(env_info.get("remaining_charge", battery or 0))
        charge_efficiency = clean_score / max(charge_count, 1.0)
        clean_per_step = clean_score / max(finished_steps, 1.0)

        self.failure_counts.setdefault(fail_reason, 0)
        self.failure_counts[fail_reason] += 1
        self.rolling_episode_total += 1
        self.rolling_charge_total += charge_count
        self.rolling_cleaned_total += clean_score
        self.rolling_finished_steps += finished_steps
        self.rolling_remaining_charge_total += remaining_charge
        self.rolling_invalid_move_rate_total += invalid_move_rate
        self.rolling_charge_efficiency_total += charge_efficiency
        self.rolling_clean_per_step_total += clean_per_step

        map_id = extra_info.get("map_id") or extra_info.get("map_code") or "?"
        actual_robot_count = int(env_info.get("npc_count", sampled_env_conf.get("robot_count", 1)))
        actual_charger_count = int(env_info.get("total_charger", sampled_env_conf.get("charger_count", 4)))
        self.logger.info(
            f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} "
            f"result:{result_str} final_bonus:{final_reward:.2f} "
            f"total_reward:{total_reward:.3f} clean_score:{clean_score:.1f} "
            f"dirt_cleaned:{fm.dirt_cleaned}/{fm.total_dirt} "
            f"invalid_move_rate:{invalid_move_rate:.3f} "
            f"charger_arrivals:{charger_arrived_count} "
            f"arrival_steps:[{first_arrival_step},{second_arrival_step},{third_arrival_step}] "
            f"profile:{sampled_meta['profile']} "
            f"map:{map_id} chargers:{actual_charger_count} robots:{actual_robot_count}"
        )

        self.score_window.append(clean_score)
        if len(self.score_window) > 30:
            self.score_window.pop(0)
        self.is_new_best = False

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

        if len(self.score_window) >= 20:
            rolling_avg = sum(self.score_window) / len(self.score_window)
            robust_score = (
                rolling_avg
                + 3.0 * _percentile(self.score_window, 0.10)
                - 8.0 * invalid_move_rate
                - 20.0 * (1.0 if fail_reason == "battery" else 0.0)
                - 30.0 * (1.0 if fail_reason == "collision" else 0.0)
            )
            if rolling_avg > self.best_avg_score:
                self.best_avg_score = rolling_avg
            if fail_reason == "completed" and robust_score > self.best_robust_score:
                self.best_robust_score = robust_score
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
            "charger_arrived_count": charger_arrived_count,
            "charger_first_arrival_step": first_arrival_step,
            "charger_second_arrival_step": second_arrival_step,
            "charger_third_arrival_step": third_arrival_step,
            "mode": self.current_target_mode,
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

    def _build_monitor_payload(self, reward):
        avg_episode_steps = self.rolling_finished_steps / self.rolling_episode_total if self.rolling_episode_total else 0.0
        avg_charge_count = self.rolling_charge_total / self.rolling_episode_total if self.rolling_episode_total else 0.0
        avg_cleaned_cells = self.rolling_cleaned_total / self.rolling_episode_total if self.rolling_episode_total else 0.0
        avg_remaining_charge = (
            self.rolling_remaining_charge_total / self.rolling_episode_total if self.rolling_episode_total else 0.0
        )
        avg_invalid_move_rate = (
            self.rolling_invalid_move_rate_total / self.rolling_episode_total if self.rolling_episode_total else 0.0
        )
        avg_charge_efficiency = (
            self.rolling_charge_efficiency_total / self.rolling_episode_total if self.rolling_episode_total else 0.0
        )
        avg_clean_per_step = (
            self.rolling_clean_per_step_total / self.rolling_episode_total if self.rolling_episode_total else 0.0
        )
        battery_fail_rate = self.failure_counts["battery"] / self.rolling_episode_total if self.rolling_episode_total else 0.0
        collision_fail_rate = (
            self.failure_counts["collision"] / self.rolling_episode_total if self.rolling_episode_total else 0.0
        )
        completed_rate = self.failure_counts["completed"] / self.rolling_episode_total if self.rolling_episode_total else 0.0

        return {
            "reward": reward,
            "episode_cnt": self.episode_cnt,
            "avg_episode_steps": avg_episode_steps,
            "avg_charge_count": avg_charge_count,
            "avg_cleaned_cells": avg_cleaned_cells,
            "avg_remaining_charge": avg_remaining_charge,
            "avg_invalid_move_rate": round(avg_invalid_move_rate, 4),
            "avg_charge_efficiency": round(avg_charge_efficiency, 4),
            "avg_clean_per_step": round(avg_clean_per_step, 4),
            "battery_fail_rate": round(battery_fail_rate, 4),
            "collision_fail_rate": round(collision_fail_rate, 4),
            "completed_rate": round(completed_rate, 4),
            "mix_alpha": round(self.current_mix_alpha, 4),
            "charger_arrived_count": self.last_charger_arrived_count,
            "charger_first_arrival_step": self.last_charger_first_arrival_step,
            "charger_second_arrival_step": self.last_charger_second_arrival_step,
            "charger_third_arrival_step": self.last_charger_third_arrival_step,
        }


def _percentile(values, q):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    weight = pos - low
    return float(values[low] * (1.0 - weight) + values[high] * weight)


def resolve_shared_code_dir():
    shared_candidates = [Path("/workspace/code")]
    env_shared_code_dir = os.getenv("KAIWU_SHARED_CODE_DIR")
    if env_shared_code_dir:
        shared_candidates.append(Path(env_shared_code_dir))
    for candidate in shared_candidates:
        if candidate.exists():
            return candidate.resolve()
    return Path(__file__).resolve().parents[2]
