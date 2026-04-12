#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for Robot Vacuum PPO agent.
"""


class Config:
    # Feature layout
    LOCAL_VIEW_SIZE = 21
    LOCAL_VIEW_CHANNELS = 3
    GLOBAL_MEMORY_SIZE = 8
    GLOBAL_MEMORY_CHANNELS = 3
    SCALAR_DIM = 74  # 39 raw + 26 extra (4 NPC×3 + 4 charger×3 - NPC#1 - charger#1 already in raw + 8 dir_dirty) + 9 one-hot

    FEATURES = [
        LOCAL_VIEW_CHANNELS * LOCAL_VIEW_SIZE * LOCAL_VIEW_SIZE,
        GLOBAL_MEMORY_CHANNELS * GLOBAL_MEMORY_SIZE * GLOBAL_MEMORY_SIZE,
        SCALAR_DIM,
        8,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # Action space: 8 directional moves
    ACTION_NUM = 8
    MODE_NUM = 3

    # Single-head value
    VALUE_NUM = 1

    # PPO hyperparameters
    GAMMA = 0.99
    LAMDA = 0.95

    INIT_LEARNING_RATE_START = 0.0001
    BETA_START = 0.008
    CLIP_PARAM = 0.2
    VF_COEF = 0.5

    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # Training snapshot / resume strategy
    SAVE_MODEL_INTERVAL_EPISODES = 50
    RESUME_LATEST_SYNC_INTERVAL_EPISODES = 20
    RESUME_EPISODE_SNAPSHOT_INTERVAL = 50
    RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS = 15 * 60
    KEEP_EPISODE_RESUME_SNAPSHOTS = 8
    KEEP_TIME_RESUME_SNAPSHOTS = 6
    KEEP_BEST_RESUME_SNAPSHOTS = 5

    # Resume control: None = train from scratch; filename = resume from that checkpoint
    RESUME_CHECKPOINT = None  # e.g. "model.ckpt-resume.pkl"
