#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Data definitions and GAE utilities for the hybrid DIY agent.
"""

import numpy as np

from common_python.utils.common_func import create_cls

from agent_diy.conf.conf import Config


ObsData = create_cls(
    "ObsData",
    feature=None,
    legal_action=None,
    legal_act=None,
    teacher_action=None,
    teacher_prob=None,
    teacher_force=None,
    teacher_mix_bias=None,
    teacher_weight=None,
    policy_weight=None,
)


ActData = create_cls(
    "ActData",
    act=None,
    action=None,
    d_action=None,
    prob=None,
    probs=None,
    value=None,
    values=None,
)


SampleData = create_cls(
    "SampleData",
    obs=Config.FEATURE_DIM,
    legal_action=Config.ACTION_DIM,
    act=1,
    prob=Config.ACTION_DIM,
    reward=1,
    value=1,
    done=1,
    reward_sum=1,
    next_value=1,
    advantage=1,
    teacher_action=1,
    teacher_prob=Config.ACTION_DIM,
    teacher_weight=1,
    policy_weight=1,
)


def reward_shaping(*args, **kwargs):
    return 0.0


def sample_process(list_sample_data, bootstrap_value=None):
    if not list_sample_data:
        return list_sample_data

    for idx in range(len(list_sample_data) - 1):
        list_sample_data[idx].next_value = np.array(list_sample_data[idx + 1].value, dtype=np.float32)

    if bootstrap_value is None:
        bootstrap_value = np.zeros_like(np.array(list_sample_data[-1].value, dtype=np.float32))
    list_sample_data[-1].next_value = np.array(bootstrap_value, dtype=np.float32)

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gae = np.zeros((1,), dtype=np.float32)
    gamma = Config.GAMMA
    lamda = Config.LAMDA

    for sample in reversed(list_sample_data):
        value = np.array(sample.value, dtype=np.float32).reshape(-1)
        reward = np.array(sample.reward, dtype=np.float32).reshape(-1)
        next_value = np.array(sample.next_value, dtype=np.float32).reshape(-1)
        done = float(np.array(sample.done, dtype=np.float32).reshape(-1)[0])

        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * lamda * (1.0 - done) * gae
        sample.advantage = gae.astype(np.float32)
        sample.reward_sum = (gae + value).astype(np.float32)
