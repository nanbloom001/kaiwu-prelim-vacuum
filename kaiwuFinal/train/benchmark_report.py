#!/usr/bin/env python3
"""
Summarize archived checkpoint robustness for Robot Vacuum PPO runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build a compact robustness report from archived training runs.")
    parser.add_argument("--run-dir", required=True, help="Path to train/archive/<run_id>")
    parser.add_argument("--top-k", type=int, default=5, help="How many checkpoints to print")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    code_root = repo_root / "code"
    if str(code_root) not in sys.path:
        sys.path.insert(0, str(code_root))

    from agent_ppo.utils.archive_analysis import build_checkpoint_analysis, load_merged_episode_summary

    run_dir = Path(args.run_dir).resolve()
    episodes = load_merged_episode_summary(run_dir)
    analysis = build_checkpoint_analysis(episodes)

    leaderboard = analysis.get("leaderboard") or []
    summary = {
        "run_dir": str(run_dir),
        "best_checkpoint_id": analysis.get("best_checkpoint_id"),
        "checkpoint_count": analysis.get("checkpoint_count", 0),
        "top_checkpoints": leaderboard[: max(args.top_k, 1)],
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
