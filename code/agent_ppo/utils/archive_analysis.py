#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright (c) 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Archive analysis helpers for checkpoint ranking and post-training diagnosis.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def checkpoint_key(raw_checkpoint_id):
    return str(raw_checkpoint_id or "bootstrap")


def _mean(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _percentile(values, q):
    values = sorted(float(v) for v in values if v is not None)
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    weight = pos - low
    return float(values[low] * (1.0 - weight) + values[high] * weight)


def build_checkpoint_analysis(episodes):
    grouped = defaultdict(list)
    for episode in episodes:
        grouped[checkpoint_key(episode.get("checkpoint_id"))].append(episode)

    real_checkpoints = [key for key in grouped if key != "bootstrap"]
    if real_checkpoints:
        grouped = {key: value for key, value in grouped.items() if key != "bootstrap"}

    leaderboard = []
    for checkpoint_id, rows in grouped.items():
        clean_scores = [float(row.get("clean_score", row.get("total_score", 0))) for row in rows]
        charge_counts = [float(row.get("charge_count", 0)) for row in rows]
        finished_steps = [float(row.get("finished_steps", 0)) for row in rows]
        remaining_charge = [float(row.get("remaining_charge", 0)) for row in rows]

        battery_fail_rate = _mean([1.0 if row.get("fail_reason") == "battery" else 0.0 for row in rows])
        collision_fail_rate = _mean([1.0 if row.get("fail_reason") == "collision" else 0.0 for row in rows])
        completed_rate = _mean([1.0 if row.get("fail_reason") == "completed" else 0.0 for row in rows])

        map_scores = defaultdict(list)
        for row in rows:
            map_id = str(row.get("map_id", "unknown"))
            map_scores[map_id].append(float(row.get("clean_score", row.get("total_score", 0))))

        avg_clean_score = _mean(clean_scores)
        score_p90 = _percentile(clean_scores, 0.90)
        ranking_score = (
            avg_clean_score
            + 0.15 * score_p90
            + 20.0 * completed_rate
            - 35.0 * battery_fail_rate
            - 20.0 * collision_fail_rate
        )

        leaderboard.append(
            {
                "checkpoint_id": checkpoint_id,
                "episode_count": len(rows),
                "avg_clean_score": round(avg_clean_score, 4),
                "score_p90": round(score_p90, 4),
                "avg_charge_count": round(_mean(charge_counts), 4),
                "avg_finished_steps": round(_mean(finished_steps), 4),
                "avg_remaining_charge": round(_mean(remaining_charge), 4),
                "battery_fail_rate": round(battery_fail_rate, 4),
                "collision_fail_rate": round(collision_fail_rate, 4),
                "completed_rate": round(completed_rate, 4),
                "ranking_score": round(ranking_score, 4),
                "per_map_avg_score": {map_id: round(_mean(scores), 4) for map_id, scores in sorted(map_scores.items())},
            }
        )

    leaderboard.sort(key=lambda item: item["ranking_score"], reverse=True)
    return {
        "checkpoint_count": len(leaderboard),
        "best_checkpoint_id": leaderboard[0]["checkpoint_id"] if leaderboard else None,
        "leaderboard": leaderboard,
    }


def write_checkpoint_reports(run_dir, analysis):
    run_dir = Path(run_dir)
    ai_dir = run_dir / "ai"
    human_dir = run_dir / "human"
    ai_dir.mkdir(parents=True, exist_ok=True)
    human_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = ai_dir / "checkpoint_metrics.json"
    metrics_path.write_text(json.dumps(analysis, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    leaderboard = analysis.get("leaderboard") or []
    lines = [
        "# Checkpoint Ranking",
        "",
        f"- Best checkpoint: `{analysis.get('best_checkpoint_id')}`",
        f"- Checkpoint count: `{analysis.get('checkpoint_count', 0)}`",
        "",
        "| Rank | Checkpoint | Avg Clean Score | P90 | Battery Fail | Collision Fail | Completed | Avg Charge | Score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, row in enumerate(leaderboard[:20], start=1):
        lines.append(
            "| {rank} | {checkpoint_id} | {avg_clean_score:.2f} | {score_p90:.2f} | "
            "{battery_fail_rate:.2%} | {collision_fail_rate:.2%} | {completed_rate:.2%} | "
            "{avg_charge_count:.2f} | {ranking_score:.2f} |".format(
                rank=idx,
                **row,
            )
        )

    report_path = human_dir / "checkpoint_ranking.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return metrics_path, report_path


def load_merged_episode_summary(run_dir):
    run_dir = Path(run_dir)
    merged_path = run_dir / "ai" / "episode_summary.jsonl"
    if merged_path.exists():
        rows = []
        for line in merged_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
        return rows

    rows = []
    for path in sorted((run_dir / "ai" / "streams").glob("episode_summary.*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Rank checkpoints from archived Robot Vacuum runs.")
    parser.add_argument("--run-dir", required=True, help="Path to train/archive/<run_id>")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    episodes = load_merged_episode_summary(run_dir)
    analysis = build_checkpoint_analysis(episodes)
    metrics_path, report_path = write_checkpoint_reports(run_dir, analysis)
    print(json.dumps({"metrics_path": str(metrics_path), "report_path": str(report_path)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
