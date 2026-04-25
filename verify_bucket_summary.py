#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Bucketed validation summary for Kaiwu training logs.

从 aisrv 训练日志中提取 Episode start / GAMEOVER 信息，
按配置桶汇总 success rate、score、finished_steps。
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean


START_RE = re.compile(
    r"Episode (?P<ep>\d+) start .*?max_step=(?P<max_step>\d+)"
    r"(?: .*?robot_count=(?P<robot_count>\d+))?"
    r"(?: .*?charger_count=(?P<charger_count>\d+))?"
    r"(?: .*?battery_max=(?P<battery_max>\d+))?"
)

END_RE = re.compile(
    r"\[GAMEOVER\] ep=(?P<ep>\d+) steps=(?P<steps>\d+) result=(?P<result>\w+)"
    r".*?mode=(?P<mode>\w+)"
    r".*?max_step=(?P<max_step>\d+)"
    r"(?: .*?robot_count=(?P<robot_count>\d+))?"
    r"(?: .*?charger_count=(?P<charger_count>\d+))?"
    r"(?: .*?battery_max=(?P<battery_max>\d+))?"
    r".*?score=(?P<score>[0-9.]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize bucketed training/eval results from aisrv logs.")
    parser.add_argument(
        "log_dir",
        nargs="?",
        default="train/log/aisrv",
        help="Directory containing aisrv logs. Default: train/log/aisrv",
    )
    return parser.parse_args()


def coerce_int(value, default=-1):
    if value is None or value == "":
        return default
    return int(value)


def iter_log_lines(log_dir: Path):
    for path in sorted(log_dir.glob("aisrv_kaiwu_rl_helper_pid*.log")):
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield path, line


def parse_message(line: str) -> str:
    try:
        record = json.loads(line)
        return str(record.get("message", ""))
    except Exception:
        return line


def build_episode_records(log_dir: Path):
    episodes = {}
    for path, line in iter_log_lines(log_dir):
        msg = parse_message(line)
        m = START_RE.search(msg)
        if m:
            key = (path.name, int(m.group("ep")))
            episodes.setdefault(key, {})
            episodes[key].update(
                {
                    "source_log": path.name,
                    "episode": int(m.group("ep")),
                    "max_step": int(m.group("max_step")),
                    "robot_count": coerce_int(m.group("robot_count")),
                    "charger_count": coerce_int(m.group("charger_count")),
                    "battery_max": coerce_int(m.group("battery_max")),
                }
            )
            continue

        m = END_RE.search(msg)
        if m:
            key = (path.name, int(m.group("ep")))
            episodes.setdefault(key, {})
            episodes[key].update(
                {
                    "source_log": path.name,
                    "episode": int(m.group("ep")),
                    "max_step": int(m.group("max_step")),
                    "robot_count": coerce_int(m.group("robot_count")),
                    "charger_count": coerce_int(m.group("charger_count")),
                    "battery_max": coerce_int(m.group("battery_max")),
                    "steps": int(m.group("steps")),
                    "result": m.group("result"),
                    "mode": m.group("mode"),
                    "score": float(m.group("score")),
                }
            )
    return [ep for ep in episodes.values() if "steps" in ep]


def render_table(rows):
    if not rows:
        return ""
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(
        "  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)).rstrip()
        for row in rows
    )


def summarize(records):
    bucketed = defaultdict(list)
    for r in records:
        bucket = (
            r.get("robot_count", -1),
            r.get("charger_count", -1),
            r.get("max_step", -1),
            r.get("battery_max", -1),
        )
        bucketed[bucket].append(r)

    rows = [[
        "robot",
        "charger",
        "max_step",
        "battery",
        "n",
        "succ",
        "avg_score",
        "avg_steps",
        "min_score",
        "min_steps",
    ]]

    for bucket in sorted(bucketed):
        rs = bucketed[bucket]
        succ = sum(1 for r in rs if r["result"] == "WIN")
        rows.append([
            bucket[0],
            bucket[1],
            bucket[2],
            bucket[3],
            len(rs),
            f"{succ}/{len(rs)}",
            f"{mean(r['score'] for r in rs):.2f}",
            f"{mean(r['steps'] for r in rs):.2f}",
            f"{min(r['score'] for r in rs):.2f}",
            min(r["steps"] for r in rs),
        ])
    return render_table(rows)


def summarize_failures(records):
    failures = [r for r in records if r["result"] != "WIN"]
    if not failures:
        return "No failures."

    rows = [[
        "log",
        "ep",
        "robot",
        "charger",
        "max_step",
        "battery",
        "steps",
        "score",
        "mode",
    ]]
    for r in sorted(failures, key=lambda item: (item["max_step"], item["steps"], item["score"])):
        rows.append([
            r["source_log"],
            r["episode"],
            r.get("robot_count", -1),
            r.get("charger_count", -1),
            r.get("max_step", -1),
            r.get("battery_max", -1),
            r["steps"],
            f"{r['score']:.2f}",
            r["mode"],
        ])
    return render_table(rows)


def main() -> int:
    args = parse_args()
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        print(f"[ERROR] log_dir not found: {log_dir}")
        return 1

    records = build_episode_records(log_dir)
    if not records:
        print(f"[ERROR] no completed episodes found in: {log_dir}")
        return 1

    print(f"log_dir: {log_dir}")
    print(f"episodes: {len(records)}")
    print()
    print("Bucket summary")
    print(summarize(records))
    print()
    print("Failures")
    print(summarize_failures(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
