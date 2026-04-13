#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Data definitions and GAE computation for robot_vacuum PPO baseline.
"""

import numpy as np

from common_python.utils.common_func import create_cls


class Config:
    DIM_OF_OBSERVATION = 69
    ACTION_NUM = 8
    VALUE_NUM = 1

    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 3e-4
    BETA_START = 1e-3
    CLIP_PARAM = 0.2
    VF_COEF = 1.0
    GRAD_CLIP_RANGE = 0.5

    MAP_OBS_SIZE = 21
    LOCAL_VIEW_SIZE = 7


ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_action=None,
    heuristic_action=None,
    heuristic_scores=None,
)

ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)


def sample_process(list_sample_data):
    if not list_sample_data:
        return list_sample_data

    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = np.asarray(list_sample_data[i + 1].value, dtype=np.float32)
    list_sample_data[-1].next_value = np.zeros((Config.VALUE_NUM,), dtype=np.float32)

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gae = np.zeros((Config.VALUE_NUM,), dtype=np.float32)
    gamma = float(Config.GAMMA)
    lamda = float(Config.LAMDA)

    for sample in reversed(list_sample_data):
        value = np.asarray(sample.value, dtype=np.float32)
        reward = np.asarray(sample.reward, dtype=np.float32)
        next_value = np.asarray(sample.next_value, dtype=np.float32)
        done = float(np.asarray(sample.done, dtype=np.float32).reshape(-1)[0])
        not_done = 1.0 - done

        delta = reward + gamma * next_value * not_done - value
        gae = delta + gamma * lamda * not_done * gae

        sample.advantage = gae.astype(np.float32)
        sample.reward_sum = (gae + value).astype(np.float32)
