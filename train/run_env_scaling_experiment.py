#!/usr/bin/env python3
"""Run environment scaling experiments: vary total env count and measure training throughput."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


BASE = Path(__file__).resolve().parents[1]
TRAIN_DIR = BASE / "train"
COMPOSE_FILE = TRAIN_DIR / ".docker-compose.yaml"
PROJECT_NAME = "kaiwu-train"
LEARNER_CONTAINER = "kaiwu-train-learner-1"
CONTAINER_LEARNER_LOG = "/data/projects/robot_vacuum/log/learner.log"
RESULT_PATH = TRAIN_DIR / "context" / "ENV_SCALING_RESULTS.json"

LEARNER_PATTERN = re.compile(
    r"global step is (?P<step>\d+), train once cost time is (?P<total>[\d.]+) ms "
    r"\(data_fetch: (?P<fetch>[\d.]+) ms, real_train: (?P<train>[\d.]+) ms\).*"
    r"sample_production_and_consumption_ratio is (?P<ratio>[\d.]+), "
    r"replay buffer monitor is \{'buffer_utilization': '(?P<buffer>[^']+)'\}"
)

TIME_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")

SERVICES_TO_RECREATE = ["learner", "aisrv", "gamecore"]

# Fixed experiment parameters
FIXED_REVERB_TYPE = 2
FIXED_BATCH_SIZE = 2048
FIXED_DUMP_MODEL_FREQ = 200
FIXED_CACHE_MULTIPLIER = 4
FIXED_RATE_LIMITER = "MinSize"
FIXED_PERF_WINDOW = "60"


def env_to_config(total_envs: int) -> dict[str, str]:
    """Map total environment count to docker-compose env vars.

    Strategy: keep PARALLEL_ENV_PER_AISRV as large as possible (up to 4),
    then scale AISRV_NUM. GAMECORE_NUM = total_envs.
    """
    if total_envs <= 0:
        raise ValueError(f"total_envs must be > 0, got {total_envs}")

    parallel = min(total_envs, 4)
    aisrv_num = math.ceil(total_envs / parallel)
    gamecore_num = total_envs

    # GPU assignment: spread AISRV instances across GPU 1, 2, 3
    if aisrv_num <= 1:
        gpu1_num, gpu2_num, gpu3_num = aisrv_num, 0, 0
    elif aisrv_num <= 2:
        gpu1_num, gpu2_num, gpu3_num = 1, aisrv_num - 1, 0
    else:
        gpu1_num = 1
        gpu2_num = 1
        gpu3_num = aisrv_num - 2

    return {
        "KAIWU_GAMECORE_NUM": str(gamecore_num),
        "KAIWU_AISRV_NUM": str(aisrv_num),
        "KAIWU_PARALLEL_ENV_PER_AISRV": str(parallel),
        "KAIWU_AISRV_GPU1_NUM": str(gpu1_num),
        "KAIWU_AISRV_GPU2_NUM": str(gpu2_num),
        "KAIWU_AISRV_GPU3_NUM": str(gpu3_num),
    }


@dataclass
class Experiment:
    total_envs: int
    aisrv_num: int
    parallel: int
    gamecore_num: int
    duration_seconds: int


def run_cmd(cmd, env=None, timeout=180):
    result = subprocess.run(cmd, cwd=BASE, env=env, capture_output=True, text=True, timeout=timeout, check=True)
    return result.stdout


def get_compose_container_state(service_name, env=None):
    cmd = ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE), "ps", "-a", "--format", "json", service_name]
    output = run_cmd(cmd, env=env, timeout=30).strip()
    if not output:
        return None
    lines = [line for line in output.splitlines() if line.strip()]
    if not lines:
        return None
    state = json.loads(lines[0])
    return {
        "service": service_name,
        "name": state.get("Name") or "",
        "state": state.get("State") or "",
        "status": state.get("Status") or "",
    }


def get_container_logs(container_name, tail=120):
    return run_cmd(["docker", "logs", "--tail", str(tail), container_name], timeout=30)


def wait_for_services(timeout_seconds=180, env=None):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        learner_state = get_compose_container_state("learner", env=env)
        aisrv_state = get_compose_container_state("aisrv", env=env)

        if (learner_state and learner_state["state"] == "running"
                and aisrv_state and aisrv_state["state"] == "running"):
            return

        for state in (learner_state, aisrv_state):
            if state and state["state"] == "exited":
                logs = get_container_logs(state["name"])
                raise RuntimeError(
                    f"{state['service']} exited during startup: {state['status']}\n"
                    f"container={state['name']}\n{logs}"
                )
        time.sleep(3)
    raise RuntimeError("learner/aisrv did not reach running state in time")


def restart_with_config(experiment: Experiment):
    env = os.environ.copy()
    config = env_to_config(experiment.total_envs)
    for key, value in config.items():
        env[key] = value

    # Fixed training parameters
    env["KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE"] = str(FIXED_REVERB_TYPE)
    env["KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE"] = str(FIXED_BATCH_SIZE)
    env["KAIWU_EXPERIMENT_DUMP_MODEL_FREQ"] = str(FIXED_DUMP_MODEL_FREQ)
    env["KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER"] = str(FIXED_CACHE_MULTIPLIER)
    env["KAIWU_EXPERIMENT_REVERB_RATE_LIMITER"] = FIXED_RATE_LIMITER
    env["KAIWU_PERF_STAT_WINDOW_SECONDS"] = FIXED_PERF_WINDOW

    run_cmd(
        ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE),
         "up", "-d", "--remove-orphans", "--force-recreate"] + SERVICES_TO_RECREATE,
        env=env,
        timeout=300,
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


def summarize(experiment: Experiment, rows):
    if len(rows) < 2:
        return {
            "total_envs": experiment.total_envs,
            "aisrv_num": experiment.aisrv_num,
            "parallel": experiment.parallel,
            "gamecore_num": experiment.gamecore_num,
            "reverb_type": FIXED_REVERB_TYPE,
            "batch_size": FIXED_BATCH_SIZE,
            "status": "insufficient_data",
            "row_count": len(rows),
            "rows": rows,
        }

    def mean(key):
        return round(sum(item[key] for item in rows) / len(rows), 2)

    first, last = rows[0], rows[-1]
    first_ts = datetime.strptime(first["time"], "%Y-%m-%d %H:%M:%S")
    last_ts = datetime.strptime(last["time"], "%Y-%m-%d %H:%M:%S")
    elapsed_minutes = max((last_ts - first_ts).total_seconds() / 60.0, 1e-6)
    step_per_min = round((last["step"] - first["step"]) / elapsed_minutes, 2)
    fetch_values = sorted(item["fetch_ms"] for item in rows)
    p95_index = min(len(fetch_values) - 1, max(0, int(len(fetch_values) * 0.95) - 1))

    return {
        "total_envs": experiment.total_envs,
        "aisrv_num": experiment.aisrv_num,
        "parallel": experiment.parallel,
        "gamecore_num": experiment.gamecore_num,
        "reverb_type": FIXED_REVERB_TYPE,
        "batch_size": FIXED_BATCH_SIZE,
        "status": "ok",
        "row_count": len(rows),
        "step_per_min": step_per_min,
        "mean_total_ms": mean("total_ms"),
        "mean_fetch_ms": mean("fetch_ms"),
        "mean_train_ms": mean("train_ms"),
        "mean_ratio": mean("ratio"),
        "p95_fetch_ms": round(fetch_values[p95_index], 2),
        "first_buffer": first["buffer_utilization"],
        "last_buffer": last["buffer_utilization"],
        "rows": rows,
    }


def parse_matrix(matrix_text, duration_seconds):
    experiments = []
    for item in matrix_text.split(","):
        item = item.strip()
        if not item:
            continue
        total_envs = int(item)
        config = env_to_config(total_envs)
        aisrv_num = int(config["KAIWU_AISRV_NUM"])
        parallel = int(config["KAIWU_PARALLEL_ENV_PER_AISRV"])
        gamecore_num = int(config["KAIWU_GAMECORE_NUM"])
        experiments.append(Experiment(
            total_envs=total_envs,
            aisrv_num=aisrv_num,
            parallel=parallel,
            gamecore_num=gamecore_num,
            duration_seconds=duration_seconds,
        ))
    return experiments


def stop_services():
    run_cmd(
        ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE), "stop"] + SERVICES_TO_RECREATE,
        timeout=120,
    )


def main():
    parser = argparse.ArgumentParser(description="Environment scaling experiment")
    parser.add_argument("--duration-seconds", type=int, default=180)
    parser.add_argument("--matrix", default="4,6,8,12", help="Comma-separated env counts, e.g. 4,6,8,12")
    args = parser.parse_args()

    experiments = parse_matrix(args.matrix, args.duration_seconds)
    summaries = []
    try:
        for experiment in experiments:
            print(f"\n{'='*60}")
            print(f"Starting experiment: {experiment.total_envs} envs "
                  f"({experiment.aisrv_num} AISRV x {experiment.parallel} parallel, "
                  f"{experiment.gamecore_num} gamecore)")
            print(f"{'='*60}")
            since_timestamp = time.time()
            restart_with_config(experiment)
            print(f"Services up, sleeping {experiment.duration_seconds}s...")
            time.sleep(experiment.duration_seconds)
            rows = collect_rows(since_timestamp)
            summary = summarize(experiment, rows)
            summaries.append(summary)
            print(f"Result: {summary['status']}, row_count={summary['row_count']}, "
                  f"step_per_min={summary.get('step_per_min', 'N/A')}, "
                  f"mean_fetch_ms={summary.get('mean_fetch_ms', 'N/A')}, "
                  f"mean_ratio={summary.get('mean_ratio', 'N/A')}")
    finally:
        stop_services()

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "fixed_params": {
            "reverb_type": FIXED_REVERB_TYPE,
            "batch_size": FIXED_BATCH_SIZE,
            "dump_model_freq": FIXED_DUMP_MODEL_FREQ,
            "cache_multiplier": FIXED_CACHE_MULTIPLIER,
            "rate_limiter": FIXED_RATE_LIMITER,
            "perf_window_seconds": FIXED_PERF_WINDOW,
        },
        "experiments": summaries,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nResults written to {RESULT_PATH}")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
