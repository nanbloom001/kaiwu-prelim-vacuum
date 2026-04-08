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

import numpy as np

from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from agent_ppo.utils.experiment_archive import ExperimentArchive, infer_fail_reason
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    last_save_model_time = time.time()
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
        logger=logger,
        monitor=monitor,
        archive=archive,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 1800:
                agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor, archive):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.archive = archive
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

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
        self.rolling_episode_total = 0

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

            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                self.archive.log_event("disaster_recovery", {"stage": "env_reset"})
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, _ = self.agent.observation_process(env_obs)

            collector = []
            self.episode_cnt += 1
            done = False
            step = 0
            total_reward = 0.0

            self.logger.info(f"Episode {self.episode_cnt} start")

            while not done:
                act_data = self.agent.predict([obs_data])[0]
                act = self.agent.action_process(act_data)

                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    self.archive.log_event("disaster_recovery", {"stage": "env_step", "episode_cnt": self.episode_cnt})
                    break

                terminated = env_obs["terminated"]
                truncated = env_obs["truncated"]
                frame_no = env_obs["frame_no"]
                step += 1
                done = terminated or truncated

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
                    )

                reward_arr = np.array([reward_scalar], dtype=np.float32)
                value_arr = act_data.value.flatten()[: Config.VALUE_NUM]

                collector.append(
                    SampleData(
                        obs=np.array(obs_data.feature, dtype=np.float32),
                        legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                        act=np.array(act_data.action),
                        reward=reward_arr,
                        done=np.array([float(done)]),
                        reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                        value=value_arr,
                        next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                        advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                        prob=np.array(act_data.prob, dtype=np.float32),
                    )
                )

                if done:
                    collector[-1].reward = collector[-1].reward + np.array([final_reward], dtype=np.float32)

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        self.monitor.put_data({os.getpid(): self._build_monitor_payload(total_reward + final_reward)})
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = next_obs_data

    def _handle_episode_end(self, env_obs, terminated, truncated, step, total_reward):
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
        if truncated:
            final_reward = 4.0 + 0.015 * clean_score + 4.0 * cleaning_ratio
            result_str = "WIN"
        else:
            fail_penalty = -3.0 if fail_reason == "collision" else -2.0
            final_reward = fail_penalty + 0.01 * clean_score + 2.0 * cleaning_ratio
            result_str = "FAIL"

        self.failure_counts.setdefault(fail_reason, 0)
        self.failure_counts[fail_reason] += 1
        self.rolling_episode_total += 1
        self.rolling_charge_total += float(env_info.get("charge_count", 0))
        self.rolling_cleaned_total += clean_score
        self.rolling_finished_steps += float(env_info.get("finished_steps", step))
        self.rolling_remaining_charge_total += float(env_info.get("remaining_charge", battery or 0))

        self.logger.info(
            f"[GAMEOVER] ep:{self.episode_cnt} steps:{step} "
            f"result:{result_str} final_bonus:{final_reward:.2f} "
            f"total_reward:{total_reward:.3f} clean_score:{clean_score:.1f} "
            f"dirt_cleaned:{fm.dirt_cleaned}/{fm.total_dirt}"
        )

        checkpoint_ref = getattr(self.agent, "current_model_ref", {}) or {}
        episode_payload = {
            "episode_id": self.episode_cnt,
            "checkpoint_id": checkpoint_ref.get("checkpoint_id"),
            "checkpoint_path": checkpoint_ref.get("path"),
            "map_id": extra_info.get("map_id") or extra_info.get("map_code"),
            "result": result_str.lower(),
            "fail_reason": fail_reason,
            "total_score": total_score,
            "clean_score": clean_score,
            "finished_steps": env_info.get("finished_steps", step),
            "charge_count": env_info.get("charge_count", 0),
            "remaining_charge": env_info.get("remaining_charge", battery),
            "total_reward": round(total_reward + final_reward, 4),
        }
        self.archive.log_episode_summary(episode_payload)
        self.archive.log_event("episode_end", episode_payload)
        if fail_reason == "battery":
            self.archive.log_event("battery_fail", episode_payload)
        elif fail_reason == "collision":
            self.archive.log_event("collision_fail", episode_payload)

        return final_reward

    def _build_monitor_payload(self, reward):
        avg_episode_steps = self.rolling_finished_steps / self.rolling_episode_total if self.rolling_episode_total else 0.0
        avg_charge_count = self.rolling_charge_total / self.rolling_episode_total if self.rolling_episode_total else 0.0
        avg_cleaned_cells = self.rolling_cleaned_total / self.rolling_episode_total if self.rolling_episode_total else 0.0
        avg_remaining_charge = (
            self.rolling_remaining_charge_total / self.rolling_episode_total if self.rolling_episode_total else 0.0
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
            "battery_fail_rate": round(battery_fail_rate, 4),
            "collision_fail_rate": round(collision_fail_rate, 4),
            "completed_rate": round(completed_rate, 4),
        }
