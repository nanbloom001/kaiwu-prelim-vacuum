#!/usr/bin/env python3
"""data_fetch 瓶颈对照实验：逐步叠加优化配置，隔离归因。

用法:
  python3 run_datafetch_benchmark.py

默认跑 4 组（A/B/C/D），每组 180 秒，共 ~12 分钟。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
TRAIN_DIR = BASE / "train"
COMPOSE_FILE = TRAIN_DIR / ".docker-compose.yaml"
PROJECT_NAME = "kaiwu-train"
LEARNER_CONTAINER = "kaiwu-train-learner-1"
CONTAINER_LEARNER_LOG = "/data/projects/robot_vacuum/log/learner.log"
RESULT_PATH = TRAIN_DIR / "context" / "DATAFETCH_BENCHMARK_RESULTS.json"

LEARNER_PATTERN = re.compile(
    r"global step is (?P<step>\d+), train once cost time is (?P<total>[\d.]+) ms "
    r"\(data_fetch: (?P<fetch>[\d.]+) ms, real_train: (?P<train>[\d.]+) ms\).*"
    r"sample_production_and_consumption_ratio is (?P<ratio>[\d.]+), "
    r"replay buffer monitor is \{'buffer_utilization': '(?P<buffer>[^']+)'\}"
)

TIME_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

SERVICES_TO_RECREATE = ["learner", "aisrv", "gamecore"]

# Baseline config: 8 envs, current params
BASELINE_ENVS = 8


@dataclass
class Trial:
    name: str
    description: str
    cache_multiplier: int = 4
    reverb_workers: int = 4
    extra_patches: list = field(default_factory=list)


DEFAULT_TRIALS = [
    Trial("A-baseline", "当前配置: cache=4, workers=4"),
    Trial("B-bigger-buffer", "cache=16, workers=4", cache_multiplier=16),
    Trial("C-bigger-buffer-more-workers", "cache=16, workers=8",
          cache_multiplier=16, reverb_workers=8),
    Trial("D-all-config", "cache=32, workers=8",
          cache_multiplier=32, reverb_workers=8),
]


def run_cmd(cmd, env=None, timeout=180):
    result = subprocess.run(cmd, cwd=BASE, env=env, capture_output=True, text=True, timeout=timeout, check=True)
    return result.stdout


def get_compose_container_state(service_name, env=None):
    output = run_cmd(
        ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE), "ps", "-a", "--format", "json", service_name],
        env=env, timeout=30,
    ).strip()
    if not output:
        return None
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    state = json.loads(lines[0])
    return {"service": service_name, "name": state.get("Name") or "", "state": state.get("State") or ""}


def get_container_logs(container_name, tail=120):
    return run_cmd(["docker", "logs", "--tail", str(tail), container_name], timeout=30)


def wait_for_services(timeout_seconds=180, env=None):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        learner = get_compose_container_state("learner", env=env)
        aisrv = get_compose_container_state("aisrv", env=env)
        if learner and learner["state"] == "running" and aisrv and aisrv["state"] == "running":
            return
        for s in (learner, aisrv):
            if s and s["state"] == "exited":
                logs = get_container_logs(s["name"])
                raise RuntimeError(f"{s['service']} exited: {s['state']}\n{logs}")
        time.sleep(3)
    raise RuntimeError("learner/aisrv did not reach running state")


def restart_with_trial(trial: Trial):
    env = os.environ.copy()
    # Fixed params
    env["KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE"] = "2"
    env["KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE"] = "2048"
    env["KAIWU_EXPERIMENT_DUMP_MODEL_FREQ"] = "200"
    env["KAIWU_EXPERIMENT_REVERB_RATE_LIMITER"] = "MinSize"
    env["KAIWU_PERF_STAT_WINDOW_SECONDS"] = "60"
    # Environment scaling
    env["KAIWU_GAMECORE_NUM"] = str(BASELINE_ENVS)
    env["KAIWU_AISRV_NUM"] = "2"
    env["KAIWU_PARALLEL_ENV_PER_AISRV"] = "4"
    env["KAIWU_AISRV_GPU1_NUM"] = "1"
    env["KAIWU_AISRV_GPU2_NUM"] = "1"
    env["KAIWU_AISRV_GPU3_NUM"] = "0"
    # Trial-specific overrides
    env["KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER"] = str(trial.cache_multiplier)
    env["KAIWU_REVERB_NUM_WORKERS_PER_ITERATOR"] = str(trial.reverb_workers)

    run_cmd(
        ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE),
         "up", "-d", "--remove-orphans", "--force-recreate"] + SERVICES_TO_RECREATE,
        env=env, timeout=300,
    )
    wait_for_services(env=env)


def collect_rows(since_timestamp):
    rows = []
    log_content = run_cmd(
        ["docker", "exec", LEARNER_CONTAINER, "cat", CONTAINER_LEARNER_LOG],
        timeout=30,
    )
    for line in log_content.splitlines():
        match = LEARNER_PATTERN.search(line)
        if not match:
            continue
        time_match = TIME_PREFIX.match(line)
        if not time_match:
            continue
        ts_str = time_match.group(1)
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        if dt.timestamp() < since_timestamp:
            continue
        rows.append({
            "time": ts_str,
            "step": int(match.group("step")),
            "total_ms": float(match.group("total")),
            "fetch_ms": float(match.group("fetch")),
            "train_ms": float(match.group("train")),
            "ratio": float(match.group("ratio")),
            "buffer_utilization": match.group("buffer"),
        })
    rows.sort(key=lambda item: item["time"])
    return rows


def summarize(trial: Trial, rows):
    if len(rows) < 2:
        return {
            "name": trial.name, "description": trial.description,
            "cache_multiplier": trial.cache_multiplier,
            "reverb_workers": trial.reverb_workers,
            "status": "insufficient_data", "row_count": len(rows), "rows": rows,
        }

    def mean(key):
        return round(sum(r[key] for r in rows) / len(rows), 2)

    first, last = rows[0], rows[-1]
    first_ts = datetime.strptime(first["time"], "%Y-%m-%d %H:%M:%S")
    last_ts = datetime.strptime(last["time"], "%Y-%m-%d %H:%M:%S")
    elapsed = max((last_ts - first_ts).total_seconds() / 60.0, 1e-6)
    step_per_min = round((last["step"] - first["step"]) / elapsed, 2)
    fetch_sorted = sorted(r["fetch_ms"] for r in rows)
    p95_idx = min(len(fetch_sorted) - 1, max(0, int(len(fetch_sorted) * 0.95) - 1))

    return {
        "name": trial.name, "description": trial.description,
        "cache_multiplier": trial.cache_multiplier,
        "reverb_workers": trial.reverb_workers,
        "status": "ok", "row_count": len(rows),
        "step_per_min": step_per_min,
        "mean_total_ms": mean("total_ms"),
        "mean_fetch_ms": mean("fetch_ms"),
        "mean_train_ms": mean("train_ms"),
        "p95_fetch_ms": round(fetch_sorted[p95_idx], 2),
        "mean_ratio": mean("ratio"),
        "first_buffer": first["buffer_utilization"],
        "last_buffer": last["buffer_utilization"],
        "rows": rows,
    }


def stop_services():
    run_cmd(
        ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE), "stop"] + SERVICES_TO_RECREATE,
        timeout=120,
    )


def main():
    parser = argparse.ArgumentParser(description="data_fetch bottleneck benchmark")
    parser.add_argument("--duration-seconds", type=int, default=180)
    parser.add_argument("--skip-to", type=str, default=None,
                        help="Skip to a specific trial by name (e.g. 'B-bigger-buffer')")
    args = parser.parse_args()

    trials = DEFAULT_TRIALS
    if args.skip_to:
        idx = next((i for i, t in enumerate(trials) if t.name == args.skip_to), None)
        if idx is None:
            print(f"Trial '{args.skip_to}' not found. Available: {[t.name for t in trials]}")
            return
        trials = trials[idx:]

    summaries = []
    try:
        for trial in trials:
            print(f"\n{'='*60}")
            print(f"[{trial.name}] {trial.description}")
            print(f"  cache_multiplier={trial.cache_multiplier}, reverb_workers={trial.reverb_workers}")
            print(f"{'='*60}")
            since = time.time()
            restart_with_trial(trial)
            print(f"  Services up, collecting for {args.duration_seconds}s...")
            time.sleep(args.duration_seconds)
            rows = collect_rows(since)
            summary = summarize(trial, rows)
            summaries.append(summary)
            print(f"  Result: {summary['status']}, rows={summary['row_count']}, "
                  f"step/min={summary.get('step_per_min', 'N/A')}, "
                  f"mean_fetch={summary.get('mean_fetch_ms', 'N/A')}ms, "
                  f"mean_ratio={summary.get('mean_ratio', 'N/A')}, "
                  f"buffer={summary.get('last_buffer', 'N/A')}")
    finally:
        stop_services()

    # Comparison table
    print(f"\n{'='*60}")
    print("COMPARISON TABLE")
    print(f"{'='*60}")
    print(f"{'Trial':<35} {'rows':>5} {'step/min':>9} {'fetch':>8} {'p95fetch':>9} {'train':>8} {'ratio':>7}")
    print("-" * 85)
    for s in summaries:
        print(f"{s['name']:<35} {s['row_count']:>5} "
              f"{s.get('step_per_min', 'N/A'):>9} "
              f"{s.get('mean_fetch_ms', 'N/A'):>7}ms "
              f"{s.get('p95_fetch_ms', 'N/A'):>8}ms "
              f"{s.get('mean_train_ms', 'N/A'):>7}ms "
              f"{s.get('mean_ratio', 'N/A'):>7}")

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "baseline_envs": BASELINE_ENVS,
        "duration_seconds": args.duration_seconds,
        "experiments": summaries,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nResults: {RESULT_PATH}")


if __name__ == "__main__":
    main()
