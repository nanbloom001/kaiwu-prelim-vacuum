#!/usr/bin/env python3
"""Run replay stability experiments by recreating learner/aisrv with overrides."""

from __future__ import annotations

import argparse
import json
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
RESULT_PATH = TRAIN_DIR / "context" / "REPLAY_STABILITY_RESULTS.json"

LEARNER_PATTERN = re.compile(
    r"global step is (?P<step>\d+), train once cost time is (?P<total>[\d.]+) ms "
    r"\(data_fetch: (?P<fetch>[\d.]+) ms, real_train: (?P<train>[\d.]+) ms\).*"
    r"sample_production_and_consumption_ratio is (?P<ratio>[\d.]+), "
    r"replay buffer monitor is \{'buffer_utilization': '(?P<buffer>[^']+)'\}"
)


@dataclass
class Experiment:
    name: str
    reverb_type: int
    batch_size: int
    duration_seconds: int


def run_cmd(cmd, env=None, timeout=180):
    result = subprocess.run(cmd, cwd=BASE, env=env, capture_output=True, text=True, timeout=timeout, check=True)
    return result.stdout


def get_compose_container_state(service_name):
    output = run_cmd(
        ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE), "ps", "-a", "--format", "json", service_name],
        timeout=30,
    ).strip()
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


def wait_for_services(timeout_seconds=180):
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        learner_state = get_compose_container_state("learner")
        aisrv_state = get_compose_container_state("aisrv")

        if learner_state and learner_state["state"] == "running" and aisrv_state and aisrv_state["state"] == "running":
            return

        for state in (learner_state, aisrv_state):
            if state and state["state"] == "exited":
                logs = get_container_logs(state["name"])
                raise RuntimeError(
                    f"{state['service']} exited during startup: {state['status']}\n"
                    f"container={state['name']}\n{logs}"
                )
        time.sleep(3)
    raise RuntimeError("learner/aisrv did not reach Up state in time")


def restart_with_overrides(experiment: Experiment):
    env = os.environ.copy()
    env["KAIWU_PYTORCH_READ_DATA_FROM_REVERB_TYPE"] = str(experiment.reverb_type)
    env["KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE"] = str(experiment.batch_size)
    env["KAIWU_EXPERIMENT_DUMP_MODEL_FREQ"] = "200"
    env["KAIWU_EXPERIMENT_REPLAY_BUFFER_CACHE_MULTIPLIER"] = "4"
    env["KAIWU_EXPERIMENT_REVERB_RATE_LIMITER"] = "MinSize"
    env["KAIWU_PERF_STAT_WINDOW_SECONDS"] = "60"
    run_cmd(
        ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE), "up", "-d", "--force-recreate", "learner", "aisrv"],
        env=env,
        timeout=240,
    )
    wait_for_services()


TIME_PREFIX = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


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
        rows.append(
            {
                "time": ts_str,
                "step": int(match.group("step")),
                "total_ms": float(match.group("total")),
                "fetch_ms": float(match.group("fetch")),
                "train_ms": float(match.group("train")),
                "ratio": float(match.group("ratio")),
                "buffer_utilization": match.group("buffer"),
            }
        )
    rows.sort(key=lambda item: item["time"])
    return rows


def summarize(experiment: Experiment, rows):
    if len(rows) < 2:
        return {
            "name": experiment.name,
            "reverb_type": experiment.reverb_type,
            "batch_size": experiment.batch_size,
            "status": "insufficient_data",
            "row_count": len(rows),
            "rows": rows,
        }

    def mean(key):
        return round(sum(item[key] for item in rows) / len(rows), 2)

    first = rows[0]
    last = rows[-1]
    first_ts = datetime.strptime(first["time"][:19], "%Y-%m-%d %H:%M:%S")
    last_ts = datetime.strptime(last["time"][:19], "%Y-%m-%d %H:%M:%S")
    elapsed_minutes = max((last_ts - first_ts).total_seconds() / 60.0, 1e-6)
    step_per_min = round((last["step"] - first["step"]) / elapsed_minutes, 2)
    fetch_values = sorted(item["fetch_ms"] for item in rows)
    p95_index = min(len(fetch_values) - 1, max(0, int(len(fetch_values) * 0.95) - 1))

    return {
        "name": experiment.name,
        "reverb_type": experiment.reverb_type,
        "batch_size": experiment.batch_size,
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
        dataset, batch = item.split("-")
        reverb_type = 1 if dataset == "v1" else 2
        experiments.append(Experiment(name=item, reverb_type=reverb_type, batch_size=int(batch), duration_seconds=duration_seconds))
    return experiments


def stop_services():
    run_cmd(
        ["docker", "compose", "-p", PROJECT_NAME, "-f", str(COMPOSE_FILE), "stop", "learner", "aisrv"],
        timeout=120,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-seconds", type=int, default=180)
    parser.add_argument("--matrix", default="v1-2048,v2-2048", help="Comma-separated experiments such as v1-2048,v2-1536")
    args = parser.parse_args()

    experiments = parse_matrix(args.matrix, args.duration_seconds)
    summaries = []
    try:
        for experiment in experiments:
            since_timestamp = time.time()
            restart_with_overrides(experiment)
            time.sleep(experiment.duration_seconds)
            rows = collect_rows(since_timestamp)
            summaries.append(summarize(experiment, rows))
    finally:
        stop_services()

    payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": summaries,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(RESULT_PATH)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
