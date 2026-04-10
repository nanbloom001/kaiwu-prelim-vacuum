#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Training workflow for the hybrid DIY vacuum agent.
"""

import os
import time

import numpy as np

from agent_diy.conf.conf import Config
from agent_diy.feature.definition import SampleData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    env = envs[0]
    agent = agents[0]
    last_save_model_time = time.time()

    usr_conf = read_usr_conf("agent_diy/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_diy/conf/train_env_conf.toml")
        return

    episode_runner = EpisodeRunner(
        env=env,
        agent=agent,
        usr_conf=usr_conf,
        logger=logger,
        monitor=monitor,
    )

    while True:
        for g_data in episode_runner.run_episodes():
            if g_data:
                agent.send_sample_data(g_data)
                g_data.clear()

            now = time.time()
            if now - last_save_model_time >= Config.SAVE_MODEL_INTERVAL_SEC:
                agent.save_model(id="latest")
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_get_training_metrics_time = 0

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_get_training_metrics_time >= 60:
                training_metrics = get_training_metrics()
                self.last_get_training_metrics_time = now
                if training_metrics is not None:
                    self.logger.info(f"training_metrics: {training_metrics}")

            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            obs_data, _ = self.agent.observation_process(env_obs)
            act_data_list = self.agent.predict([obs_data])
            if not act_data_list:
                self.logger.warning("predict returned empty action list at episode start, restart episode")
                continue
            act_data = act_data_list[0]

            collector = []
            self.episode_cnt += 1
            step = 0
            done = False
            total_reward = 0.0

            self.logger.info(f"DIY Episode {self.episode_cnt} start")

            while not done:
                act = self.agent.action_process(act_data)
                env_reward, env_obs = self.env.step(act)
                if handle_disaster_recovery(env_obs, self.logger):
                    break

                terminated = bool(env_obs["terminated"])
                truncated = bool(env_obs["truncated"])
                done = terminated or truncated
                step += 1

                next_obs_data, _ = self.agent.observation_process(env_obs)
                raw_reward_scalar = float(self.agent.last_reward)
                reward_scalar = raw_reward_scalar * Config.REWARD_SCALE
                total_reward += raw_reward_scalar

                if done:
                    next_value = np.zeros((1,), dtype=np.float32)
                    next_act_data = None
                else:
                    next_act_data_list = self.agent.predict([next_obs_data])
                    if not next_act_data_list:
                        self.logger.warning("predict returned empty action list mid-episode, terminate current episode")
                        break
                    next_act_data = next_act_data_list[0]
                    next_value = np.array(next_act_data.value, dtype=np.float32).reshape(-1)[:1]

                frame = SampleData(
                    obs=np.array(obs_data.feature, dtype=np.float32),
                    legal_action=np.array(obs_data.legal_action, dtype=np.float32),
                    act=np.array(act_data.action, dtype=np.int32),
                    prob=np.array(act_data.prob, dtype=np.float32),
                    reward=np.array([reward_scalar], dtype=np.float32),
                    value=np.array(act_data.value, dtype=np.float32).reshape(-1)[:1],
                    done=np.array([float(done)], dtype=np.float32),
                    reward_sum=np.zeros((1,), dtype=np.float32),
                    next_value=np.zeros((1,), dtype=np.float32),
                    advantage=np.zeros((1,), dtype=np.float32),
                    teacher_action=np.array([int(obs_data.teacher_action)], dtype=np.int32),
                    teacher_prob=np.array(obs_data.teacher_prob, dtype=np.float32),
                )
                collector.append(frame)

                if len(collector) >= Config.SAMPLE_CHUNK_SIZE and not done:
                    yield sample_process(collector, bootstrap_value=next_value)
                    collector = []

                if done:
                    env_info = env_obs["observation"]["env_info"]
                    total_score = int(env_info.get("total_score", 0))
                    dirt_cleaned = max(int(env_info.get("clean_score", total_score)), 0)
                    total_dirt = max(int(env_info.get("total_dirt", max(total_score, 1))), 1)

                    if truncated:
                        final_bonus = 4.0 + 6.0 * (dirt_cleaned / total_dirt)
                    else:
                        final_bonus = -3.0
                    train_final_bonus = final_bonus * Config.FINAL_BONUS_SCALE
                    collector[-1].reward = collector[-1].reward + np.array([train_final_bonus], dtype=np.float32)

                    if self.logger:
                        self.logger.info(
                            f"[DIY GAMEOVER] ep:{self.episode_cnt} steps:{step} "
                            f"terminated:{terminated} truncated:{truncated} score:{total_score} "
                            f"total_reward:{total_reward + final_bonus:.3f}"
                        )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        self.monitor.put_data(
                            {
                                os.getpid(): {
                                    "reward": total_reward + final_bonus,
                                    "episode_cnt": self.episode_cnt,
                                }
                            }
                        )
                        self.last_report_monitor_time = now

                    if collector:
                        yield sample_process(collector, bootstrap_value=np.zeros((1,), dtype=np.float32))
                    break

                obs_data = next_obs_data
                act_data = next_act_data
