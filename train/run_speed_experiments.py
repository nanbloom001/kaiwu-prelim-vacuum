#!/usr/bin/env python3
"""
Training speed bottleneck diagnosis experiments.
Runs 3 tests (5 min each) and collects metrics from learner log.

Usage: cd train && python3 run_speed_experiments.py
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ENV_FILE = Path(__file__).parent / ".env"
COMPOSE_FILE = Path(__file__).parent / ".docker-compose.yaml"
RESULTS_FILE = Path(__file__).parent / "context" / "SPEED_EXPERIMENT_RESULTS.json"
TEST_DURATION = 300  # 5 minutes per test

TESTS = [
    {
        "name": "T1_baseline",
        "desc": "8gc, batch=2048, Uniform, dump=100, send=10000",
        "env": {
            "KAIWU_GAMECORE_NUM": "8",
            "KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE": "2048",
            "KAIWU_EXPERIMENT_REVERB_SAMPLER": "Uniform",
            "KAIWU_EXPERIMENT_DUMP_MODEL_FREQ": "100",
            "KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE": "10000",
        },
    },
    {
        "name": "T2_win_params",
        "desc": "8gc, batch=4096, Fifo, dump=500, send=4096",
        "env": {
            "KAIWU_GAMECORE_NUM": "8",
            "KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE": "4096",
            "KAIWU_EXPERIMENT_REVERB_SAMPLER": "Fifo",
            "KAIWU_EXPERIMENT_DUMP_MODEL_FREQ": "500",
            "KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE": "4096",
        },
    },
    {
        "name": "T3_win_params_16gc",
        "desc": "16gc, batch=4096, Fifo, dump=500, send=4096",
        "env": {
            "KAIWU_GAMECORE_NUM": "16",
            "KAIWU_EXPERIMENT_TRAIN_BATCH_SIZE": "4096",
            "KAIWU_EXPERIMENT_REVERB_SAMPLER": "Fifo",
            "KAIWU_EXPERIMENT_DUMP_MODEL_FREQ": "500",
            "KAIWU_EXPERIMENT_SEND_SAMPLE_SIZE": "4096",
        },
    },
]


def run(cmd, **kwargs):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True, **kwargs)


def update_env(overrides):
    """Update .env file with test-specific values."""
    lines = ENV_FILE.read_text().splitlines()
    keys_to_update = set(overrides.keys())
    new_lines = []
    updated_keys = set()

    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in keys_to_update:
                new_lines.append(f"{key}={overrides[key]}")
                updated_keys.add(key)
                continue
        new_lines.append(line)

    # Add any keys that weren't in the file
    for key in keys_to_update - updated_keys:
        new_lines.append(f"{key}={overrides[key]}")

    ENV_FILE.write_text("\n".join(new_lines) + "\n")


def wait_for_training(timeout=120):
    """Wait until learner is producing training logs."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            result = subprocess.run(
                ["docker", "exec", "kaiwu-train-learner-1",
                 "cat", "/data/projects/robot_vacuum/log/learner.log"],
                capture_output=True, text=True, timeout=10,
            )
            if "train once cost time" in result.stdout:
                lines = result.stdout.strip().splitlines()
                metric_lines = [l for l in lines if "train once cost time" in l]
                if len(metric_lines) >= 2:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def collect_metrics():
    """Extract metrics from learner log."""
    result = subprocess.run(
        ["docker", "exec", "kaiwu-train-learner-1",
         "cat", "/data/projects/robot_vacuum/log/learner.log"],
        capture_output=True, text=True, timeout=15,
    )

    pattern = (
        r'train count is (\d+).*?'
        r'cost time is ([\d.]+) ms.*?'
        r'data_fetch: ([\d.]+) ms.*?'
        r'real_train: ([\d.]+) ms.*?'
        r"buffer_utilization': '(\d+)/(\d+)'"
    )
    matches = re.findall(pattern, result.stdout)
    if len(matches) < 3:
        return None

    recent = matches[-5:]  # Last 5 minutes of data
    counts = [int(m[0]) for m in recent]
    total_costs = [float(m[1]) for m in recent]
    data_fetches = [float(m[2]) for m in recent]
    real_trains = [float(m[3]) for m in recent]

    elapsed_min = len(recent)  # Each metric line is ~1 minute apart
    steps_per_sec = (counts[-1] - counts[0]) / (elapsed_min * 60) if elapsed_min > 1 else 0

    df_sorted = sorted(data_fetches)
    rt_sorted = sorted(real_trains)

    # Also get save cycle timing
    save_pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+.*?save model.*?successfully'
    save_matches = re.findall(save_pattern, result.stdout)
    save_cycle_s = 0
    if len(save_matches) >= 2:
        from datetime import datetime
        recent_saves = save_matches[-5:]
        t1 = datetime.strptime(recent_saves[0], "%Y-%m-%d %H:%M:%S")
        t2 = datetime.strptime(recent_saves[-1], "%Y-%m-%d %H:%M:%S")
        save_cycle_s = (t2 - t1).total_seconds() / (len(recent_saves) - 1)

    return {
        "steps_per_sec": round(steps_per_sec, 2),
        "steps_per_min": counts[-1] - counts[0],
        "data_fetch_avg": round(sum(data_fetches) / len(data_fetches), 1),
        "data_fetch_p50": round(df_sorted[len(df_sorted) // 2], 1),
        "data_fetch_p95": round(df_sorted[int(len(df_sorted) * 0.95)], 1),
        "data_fetch_max": round(max(data_fetches), 1),
        "real_train_avg": round(sum(real_trains) / len(real_trains), 1),
        "real_train_max": round(max(real_trains), 1),
        "total_cost_avg": round(sum(total_costs) / len(total_costs), 1),
        "save_cycle_s": round(save_cycle_s, 1),
        "data_fetch_pct": round(sum(data_fetches) / sum(total_costs) * 100, 1),
        "n_samples": len(recent),
    }


def run_test(test):
    """Run a single test: update env, restart, wait, collect."""
    print(f"\n{'='*60}")
    print(f"  {test['name']}: {test['desc']}")
    print(f"{'='*60}")

    # Update .env
    update_env(test["env"])
    print(f"  Updated .env: {test['env']}")

    # Restart
    print("  Stopping containers...")
    run("cd train && docker compose -f .docker-compose.yaml --profile distributed down", timeout=60)

    print("  Starting containers...")
    run("cd train && docker compose -f .docker-compose.yaml --profile distributed up -d", timeout=120)

    # Wait for training to start
    print("  Waiting for training to start...")
    if not wait_for_training():
        print("  ERROR: Training did not start within timeout!")
        return None

    # Wait for test duration
    print(f"  Running test for {TEST_DURATION//60} minutes...")
    time.sleep(TEST_DURATION)

    # Collect metrics
    print("  Collecting metrics...")
    metrics = collect_metrics()
    if metrics:
        print(f"  Results: {json.dumps(metrics, indent=2)}")
    else:
        print("  ERROR: Could not collect metrics!")
    return metrics


def main():
    os.chdir(Path(__file__).parent.parent)  # project root

    results = {}
    for test in TESTS:
        metrics = run_test(test)
        if metrics:
            results[test["name"]] = {
                "desc": test["desc"],
                "params": test["env"],
                "metrics": metrics,
            }

    # Save results
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {RESULTS_FILE}")

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")
    print(f"  {'Test':<25} {'steps/s':>8} {'df_avg':>8} {'df_p95':>8} {'rt_avg':>8} {'df%':>6}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
    for name, data in results.items():
        m = data["metrics"]
        print(f"  {name:<25} {m['steps_per_sec']:>8.2f} {m['data_fetch_avg']:>7.1f}ms "
              f"{m['data_fetch_p95']:>7.1f}ms {m['real_train_avg']:>7.1f}ms {m['data_fetch_pct']:>5.1f}%")


if __name__ == "__main__":
    main()
