#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Configuration for the LTSPPO Robot Vacuum agent.
"""

import os


class Config:
    # ------------------------------------------------------------------
    # Observation layout
    # ------------------------------------------------------------------
    LOCAL_VIEW_SIZE = 21
    LOCAL_VIEW_CHANNELS = 8
    GLOBAL_MEMORY_SIZE = 16
    GLOBAL_MEMORY_CHANNELS = 6

    ENTITY_SLOTS = 8          # 4 NPC + 4 charger
    NPC_SLOTS = 4
    CHARGER_SLOTS = 4
    ENTITY_FEATURE_DIM = 7
    ENTITY_DIM = ENTITY_SLOTS * ENTITY_FEATURE_DIM

    SCALAR_DIM = 88
    ACTION_HISTORY_DIM = 16   # last action one-hot + recent 4-step histogram

    FEATURES = [
        LOCAL_VIEW_CHANNELS * LOCAL_VIEW_SIZE * LOCAL_VIEW_SIZE,
        GLOBAL_MEMORY_CHANNELS * GLOBAL_MEMORY_SIZE * GLOBAL_MEMORY_SIZE,
        ENTITY_DIM,
        SCALAR_DIM,
        ACTION_HISTORY_DIM,
    ]
    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURES)
    DIM_OF_OBSERVATION = FEATURE_LEN

    # ------------------------------------------------------------------
    # Action / heads
    # ------------------------------------------------------------------
    ACTION_NUM = 8
    MODE_NUM = 4
    TARGET_DIM = CHARGER_SLOTS + 1  # none + top-4 charger candidates
    VALUE_NUM = 1

    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    # ------------------------------------------------------------------
    # Recurrent training
    # ------------------------------------------------------------------
    USE_RECURRENT_POLICY = True
    RNN_HIDDEN_DIM = 192
    SEQ_CHUNK_LEN = 16
    SEQ_BURN_IN = 4
    SEQ_LEARN_LEN = SEQ_CHUNK_LEN - SEQ_BURN_IN
    SEQ_STRIDE = 12

    SAMPLE_OBS_DIM = DIM_OF_OBSERVATION * SEQ_CHUNK_LEN
    SAMPLE_LEGAL_ACTION_DIM = ACTION_NUM * SEQ_CHUNK_LEN
    SAMPLE_ACTION_DIM = SEQ_CHUNK_LEN
    SAMPLE_REWARD_DIM = SEQ_CHUNK_LEN
    SAMPLE_DONE_DIM = SEQ_CHUNK_LEN
    SAMPLE_PROB_DIM = ACTION_NUM * SEQ_CHUNK_LEN
    SAMPLE_MODE_DIM = SEQ_CHUNK_LEN
    SAMPLE_TARGET_DIM = SEQ_CHUNK_LEN
    SAMPLE_MODE_TEACHER_MASK_DIM = SEQ_CHUNK_LEN
    SAMPLE_TARGET_TEACHER_MASK_DIM = SEQ_CHUNK_LEN
    SAMPLE_VALUE_DIM = SEQ_CHUNK_LEN
    SAMPLE_AUX_LABEL_DIM = SEQ_CHUNK_LEN
    EXPERT_WEIGHT_DIM = 1
    FALLBACK_MASK_DIM = SEQ_CHUNK_LEN

    # ------------------------------------------------------------------
    # Model dimensions
    # ------------------------------------------------------------------
    LOCAL_EMBED_DIM = 192
    GLOBAL_EMBED_DIM = 128
    ENTITY_EMBED_DIM = 128
    SCALAR_EMBED_DIM = 96
    ACTION_HISTORY_EMBED_DIM = 32
    FUSED_HIDDEN_DIM = 256
    PRE_RNN_DIM = 192
    TARGET_CONTEXT_DIM = 64
    MODE_EMBED_DIM = 16

    # ------------------------------------------------------------------
    # PPO hyperparameters
    # ------------------------------------------------------------------
    GAMMA = 0.99
    LAMDA = 0.95
    INIT_LEARNING_RATE_START = 0.00003
    BETA_START = 0.015
    CLIP_PARAM = 0.15
    VF_COEF = 0.5
    ENTROPY_FLOOR = 0.15
    ENTROPY_FLOOR_COEF = 0.10
    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # Teacher / auxiliary losses
    MODE_TEACHER_WEIGHT = 0.10
    TARGET_TEACHER_WEIGHT = 0.10
    AUX_BATTERY_RISK_WEIGHT = 0.05
    AUX_COLLISION_RISK_WEIGHT = 0.05
    TEACHER_FORCE_UNTIL_EPISODE = 160
    TEACHER_ANNEAL_END_EPISODE = 600
    USE_EXPERT_GRADIENT_ISOLATION = True

    # ------------------------------------------------------------------
    # Reward shaping / observation memories
    # ------------------------------------------------------------------
    ASTAR_POTENTIAL_ALPHA = 0.35
    ASTAR_POTENTIAL_BATTERY_THRESHOLD = 0.88
    TRAJECTORY_LENGTH = 50
    TRAJECTORY_DECAY = 0.02
    ACTION_HISTORY_WINDOW = 4
    RETURN_SLACK_THRESHOLD = 4.0
    RETURN_BATTERY_RATIO = 0.25
    PREPARE_RETURN_SLACK_THRESHOLD = 12.0
    PREPARE_RETURN_BATTERY_RATIO = 0.45
    EXPLORE_REWARD_SCALE = 0.03
    EXPLORE_REWARD_CAP = 4.0
    FRONTIER_REWARD_SCALE = 0.15
    FRONTIER_LOW_BATTERY_RATIO = 0.35
    FRONTIER_CRITICAL_BATTERY_RATIO = 0.20
    TEACHER_TARGET_STABILITY_WINDOW = 3
    TEACHER_UNKNOWN_PATH_RATIO_MAX = 0.20
    TEACHER_TARGET_MARGIN_MIN = 3.0

    # ------------------------------------------------------------------
    # Expert fallback
    # ------------------------------------------------------------------
    EXPERT_BIAS_MIN = 0.0
    EXPERT_BIAS_MAX = 0.0
    EXPERT_ANNEAL_START_EPISODE = 0
    EXPERT_ANNEAL_END_EPISODE = 0
    EXPERT_ANNEAL_MIN_SCALE = 0.0
    EXPERT_EMERGENCY_BATTERY_RATIO = 0.08
    EXPERT_RELIABLE_SLACK_BUFFER = 12.0
    EXPERT_RELIABLE_RETURN_RATIO = 0.45
    EXPERT_RELIABLE_PREPARE_RETURN_RATIO = 0.65

    # ------------------------------------------------------------------
    # Learner runtime tuning
    # ------------------------------------------------------------------
    LEARNER_CPU_THREADS = 4
    LEARNER_CPU_INTEROP_THREADS = 2
    LEARNER_USE_AMP = True
    LEARNER_ALLOW_FOREACH_OPTIMIZER = True
    LEARNER_ALLOW_FUSED_OPTIMIZER = True
    LEARNER_PREFER_BATCH_TENSOR = True
    LEARNER_JIT_TRACE = False
    LEARNER_TORCH_COMPILE = False
    AGENT_LOAD_MODEL_CACHE = True
    PERF_STAT_WINDOW_SECONDS = 60

    # ------------------------------------------------------------------
    # Dynamic curriculum advancement
    # ------------------------------------------------------------------
    CURRICULUM_WINDOW = 20
    MONITOR_WINDOW = 20
    CURRICULUM_ADVANCE_WIN_RATE = 0.80
    CURRICULUM_ADVANCE_AVG_CS = 800
    CURRICULUM_ADVANCE_CHARGE = 2.5
    CURRICULUM_HOLD_WIN_RATE = 0.70

    # ------------------------------------------------------------------
    # Training snapshot / resume strategy
    # ------------------------------------------------------------------
    SAVE_MODEL_INTERVAL_EPISODES = 50
    RESUME_LATEST_SYNC_INTERVAL_EPISODES = 20
    RESUME_EPISODE_SNAPSHOT_INTERVAL = 50
    RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS = 10 * 60
    KEEP_EPISODE_RESUME_SNAPSHOTS = 16
    KEEP_TIME_RESUME_SNAPSHOTS = 12
    KEEP_BEST_RESUME_SNAPSHOTS = 10

    RESUME_CHECKPOINT = "saved_models/v6-ltsppo-ep188/model.ckpt-resume.pkl"

    # Optional helper for external scripts
    LTSPPO_BRANCH_NAME = os.getenv("KAIWU_LTSPPO_BRANCH_NAME", "linux-LTSPPO")
