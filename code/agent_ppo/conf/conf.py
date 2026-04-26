#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Robot Vacuum PPO agent.
"""


class Config:

    # local_view(7x7=49) + global_state(27) + legal_action(8) = 84D
    FEATURES = [49, 27, 8]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    ACTION_NUM = 8
    VALUE_NUM = 1

    GAMMA = 0.99
    LAMDA = 0.95

    INIT_LEARNING_RATE_START = 3e-4
    BETA_START = 0.004
    BETA_END = 0.0018
    CLIP_PARAM = 0.2
    VF_COEF = 0.5
    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    HIDDEN_DIM_1 = 256
    HIDDEN_DIM_2 = 128

    NUM_AGENTS = 10

    # Linux runtime / benchmark bridge fields retained by infrastructure.
    RESUME_CHECKPOINT = "/workspace/code/runtime_state/current/prepared_resume/model.pkl"
    PREPARE_RETURN_SLACK_THRESHOLD = 6.0

    # Planner-guided residual PPO
    RESIDUAL_ALPHA_START = 0.10
    RESIDUAL_ALPHA_WARMUP_TARGET = 0.18
    RESIDUAL_ALPHA_MAX = 0.45
    RESIDUAL_ALPHA_CHARGE_CAP = 0.010
    RESIDUAL_ALPHA_FALLBACK_CAP = 0.006
    RESIDUAL_WARMUP_EPISODES = 240
    RESIDUAL_SCORE_EMA_DECAY = 0.92
    RESIDUAL_SCORE_IMPROVE = 10.0
    RESIDUAL_SCORE_DROP = 50.0
    RESIDUAL_PLATEAU_PATIENCE = 16
    RESIDUAL_PLATEAU_SCORE = 1820.0
    RESIDUAL_ALPHA_STEP = 0.015
    PLANNER_PRIOR_TEMPERATURE = 0.54

    # Behavior cloning regularization against planner, decayed as alpha rises
    BC_COEF_START = 1.10
    BC_COEF_MIN = 0.28
