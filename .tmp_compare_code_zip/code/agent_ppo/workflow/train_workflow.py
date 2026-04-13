#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Training workflow for the robot_vacuum planner-assisted baseline.
"""

import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from agent_ppo.feature.definition import SampleData, sample_process
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    if not envs or not agents:
        if logger is not None:
            logger.error("workflow init failed: envs or agents is empty")
        return

    env = envs[0]
    primary_agent = agents[0]

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        if logger is not None:
            logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    runner = EpisodeRunner(env=env, agent=primary_agent, usr_conf=usr_conf, logger=logger, monitor=monitor)

    last_save_model_time = time.time()
    while True:
        for g_data in runner.run_episodes():
            if not g_data:
                continue
            primary_agent.send_sample_data(g_data)
            g_data.clear()

            now = time.time()
            if now - last_save_model_time >= 600:
                primary_agent.save_model()
                last_save_model_time = now


class EpisodeRunner:
    def __init__(self, env, agent, usr_conf, logger, monitor):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.episode_cnt = 0
        self.last_report_monitor_time = 0.0
        self.last_get_training_metrics_time = 0.0

    def run_episodes(self):
        while True:
            self._maybe_log_training_metrics()

            env_obs = self.env.reset(usr_conf=self.usr_conf)
            if self._should_skip_env_obs(env_obs):
                continue

            self.agent.reset(env_obs)
            self.agent.load_model(id="latest")

            collector: List[SampleData] = []
            done = False
            step = 0
            total_reward = 0.0
            self.episode_cnt += 1

            obs_data, _ = self.agent.observation_process(env_obs)
            if self.logger is not None:
                self.logger.info(f"episode {self.episode_cnt} start")

            while not done:
                act_data = self.agent.predict(list_obs_data=[obs_data])[0]
                act = self.agent.action_process(act_data, is_stochastic=True)

                env_reward, next_env_obs = self.env.step(act)
                if self._should_skip_env_obs(next_env_obs):
                    break

                terminated = bool(next_env_obs.get("terminated", False)) if isinstance(next_env_obs, dict) else False
                truncated = bool(next_env_obs.get("truncated", False)) if isinstance(next_env_obs, dict) else False
                done = terminated or truncated

                next_obs_data, next_remain_info = self.agent.observation_process(next_env_obs)
                step_reward = self._extract_step_reward(next_remain_info, env_reward)
                total_reward += step_reward

                frame = SampleData(
                    obs=np.asarray(obs_data.feature, dtype=np.float32),
                    legal_action=np.asarray(obs_data.legal_action, dtype=np.float32),
                    act=np.asarray([act_data.action[0]], dtype=np.float32),
                    reward=np.asarray([step_reward], dtype=np.float32),
                    done=np.asarray([float(done)], dtype=np.float32),
                    reward_sum=np.zeros((1,), dtype=np.float32),
                    value=np.asarray(act_data.value, dtype=np.float32).reshape(-1)[:1],
                    next_value=np.zeros((1,), dtype=np.float32),
                    advantage=np.zeros((1,), dtype=np.float32),
                    prob=np.asarray(act_data.prob, dtype=np.float32),
                )
                collector.append(frame)

                if done:
                    self._maybe_report_episode_monitor(total_reward, step + 1)
                    if collector:
                        collector = sample_process(collector)
                        yield collector
                    break

                obs_data = next_obs_data
                step += 1

    def _maybe_log_training_metrics(self):
        now = time.time()
        if now - self.last_get_training_metrics_time < 60:
            return
        self.last_get_training_metrics_time = now
        metrics = get_training_metrics()
        if metrics is not None and self.logger is not None:
            self.logger.info(f"training_metrics is {metrics}")

    def _maybe_report_episode_monitor(self, total_reward: float, episode_steps: int) -> None:
        now = time.time()
        if now - self.last_report_monitor_time < 60:
            return
        if self.monitor is not None:
            monitor_data = {
                "reward": round(total_reward, 4),
                "episode_steps": int(episode_steps),
                "episode_cnt": int(self.episode_cnt),
                "active_agents": 1,
            }
            self.monitor.put_data({os.getpid(): monitor_data})
        self.last_report_monitor_time = now

    def _extract_step_reward(self, remain_info: Optional[Dict[str, Any]], env_reward: Any) -> float:
        if isinstance(remain_info, dict):
            reward = remain_info.get("reward", None)
            if isinstance(reward, (list, tuple, np.ndarray)) and len(reward) > 0:
                return float(np.asarray(reward, dtype=np.float32).reshape(-1)[0])
            if reward is not None:
                try:
                    return float(reward)
                except (TypeError, ValueError):
                    pass
        try:
            return float(env_reward)
        except (TypeError, ValueError):
            return 0.0

    def _should_skip_env_obs(self, env_obs: Any) -> bool:
        if env_obs is None:
            return True
        if isinstance(env_obs, dict) and "extra_info" in env_obs:
            try:
                return bool(handle_disaster_recovery(env_obs, self.logger))
            except Exception:
                if self.logger is not None:
                    self.logger.exception("handle_disaster_recovery failed")
                return True
        return False
