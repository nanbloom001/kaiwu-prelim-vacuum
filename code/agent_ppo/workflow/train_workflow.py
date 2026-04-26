#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Training workflow for Robot Vacuum PPO with planner-guided residual policy.
"""

import os
import threading
import time

import numpy as np

from agent_ppo.algorithm.algorithm import CoveragePlanner
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import SampleData, sample_process
from common_python.utils.workflow_disaster_recovery import handle_disaster_recovery
from tools.metrics_utils import get_training_metrics
from tools.train_env_conf_validate import read_usr_conf


class ResidualScheduler:
    """Adaptive scheduler that gradually lets PPO take more control."""

    def __init__(self):
        self.alpha = Config.RESIDUAL_ALPHA_START
        self.ema_score = None
        self.best_ema = float("-inf")
        self.stale_episodes = 0
        self.episode_cnt = 0

    def action_alpha(self, target_mode: str) -> float:
        alpha = self.alpha
        if target_mode == "charge":
            alpha = min(alpha, Config.RESIDUAL_ALPHA_CHARGE_CAP)
        elif target_mode == "fallback":
            alpha = min(alpha, Config.RESIDUAL_ALPHA_FALLBACK_CAP)
        return float(np.clip(alpha, 0.0, Config.RESIDUAL_ALPHA_MAX))

    def update(self, episode_score: float, cleaning_ratio: float) -> float:
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


class EpisodeRunner:
    """Single-agent episode runner."""

    def __init__(self, env, agent, usr_conf, logger, monitor, agent_id: int = 0):
        self.env = env
        self.agent = agent
        self.usr_conf = usr_conf
        self.logger = logger
        self.monitor = monitor
        self.agent_id = agent_id

        self.planner = CoveragePlanner()
        self.scheduler = ResidualScheduler()

        self.episode_cnt = 0
        self.last_report_monitor_time = 0
        self.last_training_metrics_time = 0
        self.local_predict_cnt = 0
        self.local_frame_cnt = 0
        self.local_yield_cnt = 0

    def _early_termination_penalty(self, finished_steps: int, max_steps: int) -> float:
        completion_ratio = finished_steps / max(max_steps, 1)
        if completion_ratio >= 0.95:
            return 0.0
        if completion_ratio >= 0.85:
            return -0.08
        if completion_ratio >= 0.70:
            return -0.18
        if completion_ratio >= 0.50:
            return -0.35
        return -0.55

    def _parse_obs(self, env_obs: dict) -> dict:
        obs = env_obs["observation"]
        fs = obs["frame_state"]
        hero = fs["heroes"]
        env_info = obs.get("env_info", {})
        return {
            "hero_pos": (int(hero["pos"]["x"]), int(hero["pos"]["z"])),
            "battery": int(hero["battery"]),
            "battery_max": max(int(hero["battery_max"]), 1),
            "dirt_cleaned": int(hero["dirt_cleaned"]),
            "legal_act": [int(x) for x in (obs.get("legal_action") or [1] * 8)],
            "map_info": obs.get("map_info"),
            "npcs": fs.get("npcs", []),
            "organs": fs.get("organs", []),
            "total_dirt": max(int(env_info.get("total_dirt", 1)), 1),
            "total_score": int(env_info.get("total_score", 0)),
        }

    def run_episodes(self):
        while True:
            now = time.time()
            if now - self.last_training_metrics_time >= 60:
                metrics = get_training_metrics()
                self.last_training_metrics_time = now
                if metrics:
                    self.logger.info(
                        f"[Agent{self.agent_id}] training_metrics: framework={metrics}, "
                        f"local={{'episode_cnt': {self.episode_cnt}, 'predict_cnt': {self.local_predict_cnt}, "
                        f"'frame_cnt': {self.local_frame_cnt}, 'yield_cnt': {self.local_yield_cnt}}}"
                    )

            env_obs = self.env.reset(self.usr_conf)
            if handle_disaster_recovery(env_obs, self.logger):
                continue

            self.agent.reset(env_obs)
            self.planner.reset()
            env_conf = self.usr_conf.get("env_conf", self.usr_conf) if isinstance(self.usr_conf, dict) else {}
            episode_max_step = int(env_conf.get("max_step", 1000))
            episode_charger_count = int(env_conf.get("charger_count", 4))
            episode_battery_max = int(env_conf.get("battery_max", 200))
            self.agent.set_episode_config(
                max_step=episode_max_step,
                charger_count=episode_charger_count,
                battery_max=episode_battery_max,
            )
            self.planner.set_episode_config(
                max_step=episode_max_step,
                charger_count=episode_charger_count,
                battery_max=episode_battery_max,
            )
            self.episode_cnt += 1
            collector = []
            done = False
            step = 0
            total_reward = 0.0
            last_mode = "explore"
            last_alpha = self.scheduler.action_alpha(last_mode)

            obs_data, _ = self.agent.observation_process(env_obs)
            self.logger.info(
                f"[Agent{self.agent_id}] Episode {self.episode_cnt} start "
                f"alpha={last_alpha:.3f} max_step={episode_max_step} "
                f"charger_count={episode_charger_count} "
                f"battery_max={episode_battery_max}"
            )

            while not done:
                policy_info = self.planner.update(env_obs, self.agent.last_action)
                last_mode = policy_info.target_mode
                self.agent.preprocessor.set_policy_context(
                    target_mode=last_mode,
                    should_charge=getattr(policy_info, "should_charge", False),
                )
                last_alpha = self.scheduler.action_alpha(last_mode)

                act_data = self.agent.guided_predict(
                    [obs_data],
                    policy_info=policy_info,
                    residual_alpha=last_alpha,
                )[0]
                self.local_predict_cnt += 1
                final_act = self.agent.action_process(act_data, is_stochastic=True)
                self.agent.last_action = final_act

                env_reward, env_obs = self.env.step(final_act)
                del env_reward
                if handle_disaster_recovery(env_obs, self.logger):
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

                frame = SampleData(
                    obs=np.asarray(obs_data.feature, dtype=np.float32),
                    legal_action=np.asarray(act_data.action_mask, dtype=np.float32),
                    act=np.asarray([final_act], dtype=np.int64),
                    reward=np.asarray([reward_scalar], dtype=np.float32),
                    done=np.asarray([float(done)], dtype=np.float32),
                    reward_sum=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    value=act_data.value.flatten()[:Config.VALUE_NUM],
                    next_value=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    advantage=np.zeros(Config.VALUE_NUM, dtype=np.float32),
                    prob=np.asarray(act_data.prob, dtype=np.float32),
                    planner_prob=np.asarray(act_data.planner_prob, dtype=np.float32),
                    mix_alpha=np.asarray(act_data.mix_alpha, dtype=np.float32),
                )
                collector.append(frame)
                self.local_frame_cnt += 1

                if done:
                    final_parsed = self._parse_obs(env_obs)
                    fm = self.agent.preprocessor
                    cleaning_ratio = fm.dirt_cleaned / max(fm.total_dirt, 1)
                    episode_score = float(final_parsed["total_score"])
                    arrival_steps = sorted(fm.charger_arrival_steps.values())
                    candidate_arrival_steps = sorted(fm.candidate_charger_arrival_steps.values())
                    confirmed_arrival_steps = sorted(fm.confirmed_charger_arrival_steps.values())
                    charger_arrived_count = len(arrival_steps)
                    arrival_candidate_confirmed_count = int(fm.arrival_candidate_confirmed_count)
                    arrival_confirmed_count = int(fm.arrival_confirmed_count)
                    arrival_canceled_count = int(fm.arrival_canceled_count)
                    arrival_retro_canceled_count = int(fm.arrival_retro_canceled_count)
                    pending_arrival_count = len(fm.pending_arrivals)
                    first_arrival_step = arrival_steps[0] if charger_arrived_count >= 1 else -1
                    second_arrival_step = arrival_steps[1] if charger_arrived_count >= 2 else -1
                    third_arrival_step = arrival_steps[2] if charger_arrived_count >= 3 else -1
                    first_candidate_confirmed_step = (
                        candidate_arrival_steps[0] if candidate_arrival_steps else -1
                    )
                    first_confirmed_arrival_step = (
                        confirmed_arrival_steps[0] if confirmed_arrival_steps else -1
                    )
                    first_confirmed_arrival_is_early = int(
                        first_confirmed_arrival_step > 0 and first_confirmed_arrival_step <= 150
                    )
                    arrival_confirm_to_fail_gap = int(fm.arrival_confirm_to_fail_gap)
                    score_ratio = episode_score / 2000.0
                    quality_bonus = 0.0
                    if truncated:
                        if last_mode == "charge":
                            quality_bonus += 0.05

                        if charger_arrived_count >= 3:
                            quality_bonus += 0.36
                        elif charger_arrived_count == 2:
                            quality_bonus += 0.18

                        if (
                            charger_arrived_count >= 2
                            and second_arrival_step > 0
                            and second_arrival_step <= 500
                        ):
                            quality_bonus += 0.10

                        if (
                            charger_arrived_count >= 3
                            and third_arrival_step > 0
                            and third_arrival_step <= 700
                        ):
                            quality_bonus += 0.08

                        if charger_arrived_count == 1 and episode_score < 900:
                            quality_bonus -= 0.22

                        if charger_arrived_count == 1 and episode_score < 820:
                            quality_bonus -= 0.32

                        if charger_arrived_count == 1 and episode_score < 760:
                            quality_bonus -= 0.18

                        final_reward = 2.5 * cleaning_ratio + 1.5 * score_ratio + quality_bonus
                        result_str = "WIN"
                    else:
                        if charger_arrived_count >= 3:
                            quality_bonus += 0.08
                        elif charger_arrived_count == 2:
                            quality_bonus += 0.05

                        if charger_arrived_count == 1:
                            quality_bonus -= 0.08
                            if step <= 450:
                                quality_bonus -= 0.12

                        if last_mode == "charge":
                            quality_bonus -= 0.20
                            if step <= 250:
                                quality_bonus -= 0.15

                        early_termination_penalty = self._early_termination_penalty(
                            finished_steps=step,
                            max_steps=episode_max_step,
                        )
                        final_reward = (
                            -2.5
                            - 0.5 * max(0.0, 0.9 - cleaning_ratio)
                            + quality_bonus
                            + early_termination_penalty
                        )
                        result_str = "FAIL"
                    if truncated:
                        early_termination_penalty = 0.0

                    charge_fail_after_arrival = int(not truncated and arrival_confirmed_count > 0)
                    go23_terminal_adjust = fm.finalize_episode_rewards(
                        result_str=result_str,
                        final_mode=last_mode,
                        final_step=step,
                        charge_fail_after_arrival=bool(charge_fail_after_arrival),
                    )
                    final_reward += go23_terminal_adjust

                    collector[-1].reward = (
                        collector[-1].reward + np.asarray([final_reward], dtype=np.float32)
                    )

                    new_alpha = self.scheduler.update(episode_score, cleaning_ratio)
                    self.logger.info(
                        f"[Agent{self.agent_id}][GAMEOVER] "
                        f"ep={self.episode_cnt} steps={step} result={result_str} "
                        f"mode={last_mode} alpha={last_alpha:.3f}->{new_alpha:.3f} "
                        f"max_step={episode_max_step} "
                        f"charger_count={episode_charger_count} "
                        f"battery_max={episode_battery_max} "
                        f"quality_bonus={quality_bonus:.3f} "
                        f"early_termination_penalty={early_termination_penalty:.3f} "
                        f"score={episode_score:.1f} reward={total_reward + final_reward:.3f} "
                        f"dirt={fm.dirt_cleaned}/{fm.total_dirt} "
                        f"charger_arrivals={charger_arrived_count} "
                        f"arrival_steps=[{first_arrival_step},{second_arrival_step},{third_arrival_step}] "
                        f"arrival_candidate_confirmed={arrival_candidate_confirmed_count} "
                        f"arrival_confirmed={arrival_confirmed_count} "
                        f"arrival_canceled={arrival_canceled_count} "
                        f"arrival_retro_canceled={arrival_retro_canceled_count} "
                        f"arrival_pending={pending_arrival_count} "
                        f"first_candidate_confirmed_step={first_candidate_confirmed_step} "
                        f"first_confirmed_arrival_step={first_confirmed_arrival_step} "
                        f"arrival_confirm_to_fail_gap={arrival_confirm_to_fail_gap} "
                        f"first_confirmed_arrival_is_early={first_confirmed_arrival_is_early} "
                        f"charge_loop_frames={fm.charge_loop_frames} "
                        f"charge_fail_after_arrival={charge_fail_after_arrival} "
                        f"go23_terminal_adjust={go23_terminal_adjust:.3f}"
                    )

                    now = time.time()
                    if now - self.last_report_monitor_time >= 60 and self.monitor:
                        monitor_payload = {
                            "reward": total_reward + final_reward,
                            "episode_cnt": self.episode_cnt,
                            "mix_alpha": new_alpha,
                            "score": episode_score,
                            "quality_bonus": quality_bonus,
                            "early_termination_penalty": early_termination_penalty,
                            "local_predict_cnt": self.local_predict_cnt,
                            "local_frame_cnt": self.local_frame_cnt,
                            "local_yield_cnt": self.local_yield_cnt,
                            "max_step_target_1000": 1000 if episode_max_step == 1000 else 0,
                            "finished_steps_actual_1000": step if episode_max_step == 1000 else 0,
                            "max_step_target_2000": 2000 if episode_max_step == 2000 else 0,
                            "finished_steps_actual_2000": step if episode_max_step == 2000 else 0,
                            "charger_arrived_count": charger_arrived_count,
                            "charger_first_arrival_step": first_arrival_step,
                            "charger_second_arrival_step": second_arrival_step,
                            "charger_third_arrival_step": third_arrival_step,
                            "arrival_candidate_confirmed_count": arrival_candidate_confirmed_count,
                            "arrival_confirmed_count": arrival_confirmed_count,
                            "arrival_canceled_count": arrival_canceled_count,
                            "arrival_retro_canceled_count": arrival_retro_canceled_count,
                            "arrival_pending_count": pending_arrival_count,
                            "charger_first_candidate_confirmed_step": first_candidate_confirmed_step,
                            "charger_first_confirmed_arrival_step": first_confirmed_arrival_step,
                            "arrival_confirm_to_fail_gap": arrival_confirm_to_fail_gap,
                            "first_confirmed_arrival_is_early": first_confirmed_arrival_is_early,
                            "charge_loop_frames": fm.charge_loop_frames,
                            "charge_fail_after_arrival": charge_fail_after_arrival,
                            "go23_terminal_adjust": go23_terminal_adjust,
                        }
                        self.monitor.put_data({
                            os.getpid(): monitor_payload
                        })
                        self.last_report_monitor_time = now

                    if collector:
                        collector = sample_process(collector)
                        self.local_yield_cnt += len(collector)
                        yield collector
                    break

                obs_data = next_obs_data


def workflow(envs, agents, logger=None, monitor=None, *args, **kwargs):
    del args, kwargs

    usr_conf = read_usr_conf("agent_ppo/conf/train_env_conf.toml", logger)
    if usr_conf is None:
        logger.error("usr_conf is None, please check agent_ppo/conf/train_env_conf.toml")
        return

    n_agents = min(len(envs), len(agents), Config.NUM_AGENTS)
    logger.info(f"Starting {n_agents} parallel agent(s) in this workflow process")

    last_save_time = [time.time()]
    save_lock = threading.Lock()

    def run_agent(idx: int):
        runner = EpisodeRunner(
            env=envs[idx],
            agent=agents[idx],
            usr_conf=usr_conf,
            logger=logger,
            monitor=monitor,
            agent_id=idx,
        )
        for g_data in runner.run_episodes():
            agents[idx].send_sample_data(g_data)
            g_data.clear()

            if idx == 0:
                with save_lock:
                    now = time.time()
                    if now - last_save_time[0] >= 1800:
                        agents[0].save_model()
                        last_save_time[0] = now

    if n_agents == 1:
        run_agent(0)
        return

    threads = [
        threading.Thread(target=run_agent, args=(i,), daemon=True, name=f"Agent-{i}")
        for i in range(n_agents)
    ]
    for thread in threads:
        thread.start()
        logger.info(f"Thread {thread.name} started")
    for thread in threads:
        thread.join()
