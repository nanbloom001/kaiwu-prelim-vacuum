#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definition and GAE computation for Robot Vacuum.
清扫大作战数据类定义与 GAE 计算。
"""

import numpy as np

try:
    from common_python.utils.common_func import create_cls
except ModuleNotFoundError:
    def create_cls(name, **field_defaults):
        field_names = tuple(field_defaults.keys())
        defaults = dict(field_defaults)

        def __init__(self, *args, **kwargs):
            if len(args) > len(field_names):
                raise TypeError(f"{name} expected at most {len(field_names)} positional arguments, got {len(args)}")
            unknown = set(kwargs) - set(field_names)
            if unknown:
                unknown_fields = ", ".join(sorted(unknown))
                raise TypeError(f"{name} got unexpected field(s): {unknown_fields}")

            values = dict(defaults)
            values.update(zip(field_names, args))
            values.update(kwargs)
            for field_name in field_names:
                setattr(self, field_name, values[field_name])

        namespace = {"__init__": __init__}
        namespace.update(defaults)
        return type(name, (), namespace)

from agent_ppo.conf.conf import Config


# feature     : 84D 特征向量（local_view 49 + global_state 27 + legal_action 8）
# legal_action: 8D  合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

ActData = create_cls(
    "ActData",
    action=None,
    d_action=None,
    prob=None,
    value=None,
    policy_prob=None,
    planner_prob=None,
    mix_alpha=None,
    action_mask=None,
)

# ── 训练样本数据 ──────────────────────────────────────────────────────────────
# 字段值为 int 时框架按维度自动处理
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
    planner_prob=Config.ACTION_NUM,
    mix_alpha=Config.VALUE_NUM,
)


def sample_process(list_sample_data: list) -> list:
    for index in range(len(list_sample_data) - 1):
        list_sample_data[index].next_value = list_sample_data[index + 1].value

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data: list):
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    gae = 0.0

    for sample in reversed(list_sample_data):
        delta = float(sample.reward) + gamma * float(sample.next_value) - float(sample.value)
        gae = delta + gamma * lamda * gae
        sample.advantage = np.array([gae], dtype=np.float32)
        sample.reward_sum = np.array([gae + float(sample.value)], dtype=np.float32)
