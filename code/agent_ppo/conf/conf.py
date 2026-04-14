#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright 漏 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Robot Vacuum PPO agent.
"""


class Config:

    # local_view(7x7=49) + global_state(20) + legal_action(8) = 77D
    FEATURES = [49, 20, 8]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    ACTION_NUM = 8
    VALUE_NUM = 1

    GAMMA = 0.99
    LAMDA = 0.95

    INIT_LEARNING_RATE_START = 3e-4
    BETA_START = 0.005
    BETA_END = 0.0015
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    HIDDEN_DIM_1 = 256
    HIDDEN_DIM_2 = 128

    NUM_AGENTS = 10

    # Planner-guided residual PPO
    RESIDUAL_ALPHA_START = 0.18
    RESIDUAL_ALPHA_WARMUP_TARGET = 0.32
    RESIDUAL_ALPHA_MAX = 0.72
    RESIDUAL_ALPHA_CHARGE_CAP = 0.24
    RESIDUAL_ALPHA_FALLBACK_CAP = 0.12
    RESIDUAL_WARMUP_EPISODES = 80
    RESIDUAL_SCORE_EMA_DECAY = 0.92
    RESIDUAL_SCORE_IMPROVE = 8.0
    RESIDUAL_SCORE_DROP = 70.0
    RESIDUAL_PLATEAU_PATIENCE = 12
    RESIDUAL_PLATEAU_SCORE = 1650.0
    RESIDUAL_ALPHA_STEP = 0.05
    PLANNER_PRIOR_TEMPERATURE = 1.35

    # Behavior cloning regularization against planner, decayed as alpha rises
    BC_COEF_START = 0.60
    BC_COEF_MIN = 0.05
