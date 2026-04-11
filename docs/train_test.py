#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

train_test helper aligned with docs/plan方案.md phases.

- Default algorithm: diy (hybrid rule + hierarchical planning + PPO fine-tune)
- Phase presets map directly to the staged plan (Phase1–Phase3 + fast smoke)
- CLI overrides let you tweak env vars without editing code

Examples:
    python docs/train_test.py --algo diy --preset phase2
    python docs/train_test.py --algo ppo --preset fast --env train_batch_size=8
    python docs/train_test.py --dry-run
"""

from typing import Dict
import argparse
import sys

from kaiwudrl.common.utils.train_test_utils import run_train_test

# Supported algorithms (see conf/algo_conf_robot_vacuum.toml)
algorithm_name_list = ["ppo", "diy"]

# Lightweight defaults (safe for quick smoke tests)
BASE_ENV_VARS: Dict[str, str] = {
    "replay_buffer_capacity": "10",
    "preload_ratio": "0.2",
    "train_batch_size": "2",
    "dump_model_freq": "1",
}

# Phase presets derived from docs/plan方案.md
PLAN_PRESETS: Dict[str, Dict[str, str]] = {
    # Phase 1: basic perception + A*/BFS skeleton
    "phase1": {
        "replay_buffer_capacity": "4000",
        "preload_ratio": "0.45",
        "train_batch_size": "24",
        "dump_model_freq": "15",
    },
    # Phase 2: add region manager, return/unstuck, mode queue, lookahead trigger
    "phase2": {
        "replay_buffer_capacity": "20000",
        "preload_ratio": "0.55",
        "train_batch_size": "64",
        "dump_model_freq": "30",
    },
    # Phase 3: adaptive reward shaping + PPO fine-tuning
    "phase3": {
        "replay_buffer_capacity": "60000",
        "preload_ratio": "0.7",
        "train_batch_size": "128",
        "dump_model_freq": "60",
    },
    # High-intensity preset for charge-focused policy hardening
    "phase4_charge": {
        "replay_buffer_capacity": "120000",
        "preload_ratio": "0.75",
        "train_batch_size": "192",
        "dump_model_freq": "80",
    },
    # Fast local smoke (keep tiny defaults)
    "fast": BASE_ENV_VARS,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run train_test with plan-based presets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algo",
        choices=algorithm_name_list,
        default="diy",
        help="algorithm to run (diy matches the hybrid plan)",
    )
    parser.add_argument(
        "--preset",
        choices=list(PLAN_PRESETS.keys()),
        default="phase3",
        help="training preset mapped to docs/plan方案.md phases",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="manual env var override, can be specified multiple times",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved configuration and exit",
    )
    return parser.parse_args()


def build_env_vars(args) -> Dict[str, str]:
    env = dict(BASE_ENV_VARS)
    env.update(PLAN_PRESETS.get(args.preset, {}))

    for item in args.env:
        if "=" not in item:
            raise ValueError(f"Invalid --env '{item}', expected KEY=VALUE")
        key, value = item.split("=", 1)
        env[key.strip()] = value.strip()
    return env


if __name__ == "__main__":
    args = parse_args()
    env_vars = build_env_vars(args)

    if args.dry_run:
        print(f"algorithm_name: {args.algo}")
        print("env_vars:")
        for k in sorted(env_vars.keys()):
            print(f"  {k} = {env_vars[k]}")
        sys.exit(0)

    run_train_test(
        algorithm_name=args.algo,
        algorithm_name_list=algorithm_name_list,
        env_vars=env_vars,
    )
