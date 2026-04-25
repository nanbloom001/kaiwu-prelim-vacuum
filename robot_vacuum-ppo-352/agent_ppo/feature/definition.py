#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definition and GAE computation for Robot Vacuum.
清扫大作战数据类定义与 GAE 计算。
"""

import numpy as np
from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


# ── 观测数据 ──────────────────────────────────────────────────────────────────
# feature     : 77D 特征向量（local_view 49 + global_state 20 + legal_action 8）
# legal_action: 8D  合法动作掩码
ObsData = create_cls("ObsData", feature=None, legal_action=None)

# ── 动作数据 ──────────────────────────────────────────────────────────────────
# action  : 随机采样动作（训练时用）
# d_action: 贪心动作（评估时用）
# prob    : 完整的 8D 动作概率分布
# value   : Critic 估计的状态价值
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
    obs          = Config.DIM_OF_OBSERVATION,   # 77D 特征向量
    legal_action = Config.ACTION_NUM,            # 8D  合法动作掩码
    act          = 1,                            # 执行的动作索引
    reward       = Config.VALUE_NUM,             # 1D  即时奖励
    reward_sum   = Config.VALUE_NUM,             # 1D  GAE TD-λ 回报（训练 Critic 目标）
    done         = 1,                            # 是否终止
    value        = Config.VALUE_NUM,             # 1D  价值估计 V(s)
    next_value   = Config.VALUE_NUM,             # 1D  下一状态价值 V(s')
    advantage    = Config.VALUE_NUM,             # 1D  GAE 优势函数
    prob         = Config.ACTION_NUM,            # 8D  动作概率分布（供重要性比率计算）
    planner_prob = Config.ACTION_NUM,            # 8D  规则规划器先验分布
    mix_alpha    = Config.VALUE_NUM,             # 1D  residual policy 混合系数
)


def sample_process(list_sample_data: list) -> list:
    """
    对一局轨迹执行后处理：填充 next_value 并计算 GAE。

    调用时机：episode 结束后，yield 给 learner 之前。
    """
    # 填充 next_value：第 i 步的 next_value = 第 i+1 步的 value
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value
    # 最后一帧的 next_value 保持 0（终止状态）

    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data: list):
    """
    使用 GAE(λ) 反向遍历计算优势函数与 TD-λ 回报。

    公式：
      δ_t   = r_t + γ·V(s_{t+1}) - V(s_t)
      A_t   = δ_t + γλ·A_{t+1}           （反向累积）
      G_t   = A_t + V(s_t)               （TD-λ return，用于训练 Critic）
    """
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    gae   = 0.0

    for sample in reversed(list_sample_data):
        delta = float(sample.reward) + gamma * float(sample.next_value) - float(sample.value)
        gae   = delta + gamma * lamda * gae
        sample.advantage  = np.array([gae], dtype=np.float32)
        sample.reward_sum = np.array([gae + float(sample.value)], dtype=np.float32)
