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
    LOCAL_VIEW_CHANNELS = 4  # was 3; 4th channel = trajectory heatmap
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

    INIT_LEARNING_RATE_START = 0.00005
    BETA_START = 0.012
    CLIP_PARAM = 0.15
    VF_COEF = 0.5
    # Entropy floor: proportional scaling to prevent policy collapse
    # When entropy < ENTROPY_FLOOR, bonus scales as COEF * value_loss * gap
    ENTROPY_FLOOR = 0.15  # trigger protection earlier (was 0.5)
    ENTROPY_FLOOR_COEF = 0.05  # proportional coef: bonus = 5% * value_loss * gap (was 3.0 absolute)

    LABEL_SIZE_LIST = [ACTION_NUM]
    LEGAL_ACTION_SIZE_LIST = LABEL_SIZE_LIST.copy()

    USE_GRAD_CLIP = True
    GRAD_CLIP_RANGE = 0.5

    # A* potential-based reward shaping (Ng et al. 1999)
    ASTAR_POTENTIAL_ALPHA = 0.35              # was 0.25; stronger per-step signal
    ASTAR_POTENTIAL_BATTERY_THRESHOLD = 0.80  # was 0.65; activate earlier so model learns charger paths

    # Trajectory heatmap
    TRAJECTORY_LENGTH = 50
    TRAJECTORY_DECAY = 0.02

    # Expert soft bias range
    EXPERT_BIAS_MIN = 5.0
    EXPERT_BIAS_MAX = 15.0

    # Expert annealing: gradually reduce expert influence
    EXPERT_ANNEAL_START_EPISODE = 50    # begin reducing bias at ep 50
    EXPERT_ANNEAL_END_EPISODE = 300     # full anneal by ep 300
    EXPERT_ANNEAL_MIN_SCALE = 0.2       # retain 20% bias after full anneal

    # Gradient isolation for expert-overridden samples
    EXPERT_WEIGHT_DIM = 1
    USE_EXPERT_GRADIENT_ISOLATION = True

    # Learner runtime tuning
    LEARNER_CPU_THREADS = 4
    LEARNER_CPU_INTEROP_THREADS = 2
    LEARNER_USE_AMP = True
    LEARNER_ALLOW_FOREACH_OPTIMIZER = True
    LEARNER_ALLOW_FUSED_OPTIMIZER = True
    LEARNER_PREFER_BATCH_TENSOR = True
    LEARNER_JIT_TRACE = True
    LEARNER_TORCH_COMPILE = True
    AGENT_LOAD_MODEL_CACHE = True
    PERF_STAT_WINDOW_SECONDS = 60

    # Dynamic curriculum advancement
    CURRICULUM_WINDOW = 20           # rolling window for curriculum metrics
    MONITOR_WINDOW = 20              # rolling window for monitor dashboard (same as curriculum)
    CURRICULUM_ADVANCE_WIN_RATE = 0.80  # advance: WinRate >= 80% (was 0.90)
    CURRICULUM_ADVANCE_AVG_CS = 800     # advance: avg CleanScore >= 800 (was 850)
    CURRICULUM_ADVANCE_CHARGE = 2.5     # advance: avg ChargeCount >= 2.5 (was 3.0)
    CURRICULUM_HOLD_WIN_RATE = 0.70     # hold stage if WinRate < 70%

    # Training snapshot / resume strategy
    SAVE_MODEL_INTERVAL_EPISODES = 50
    RESUME_LATEST_SYNC_INTERVAL_EPISODES = 20
    RESUME_EPISODE_SNAPSHOT_INTERVAL = 50
    RESUME_TIME_SNAPSHOT_INTERVAL_SECONDS = 10 * 60
    KEEP_EPISODE_RESUME_SNAPSHOTS = 16
    KEEP_TIME_RESUME_SNAPSHOTS = 12
    KEEP_BEST_RESUME_SNAPSHOTS = 10

    # Resume control: None = train from scratch; filename = resume from that checkpoint
    RESUME_CHECKPOINT = "saved_models/v52-step70000/model.ckpt-resume.pkl"  # v52 Phase 2 best: anchor 100% WR, CPS 0.879
