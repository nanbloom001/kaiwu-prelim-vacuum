# Benchmark Evaluation System

## Overview

The benchmark system runs fixed evaluation scenarios against saved model checkpoints, producing detailed logs and structured JSON results. It is **completely independent** from training — no data is sent to the learner, no curriculum interference, no episode count pollution.

This allows you to **measure the effect of code changes**: run benchmark before and after a change, compare the results quantitatively.

---

## Quick Start

```bash
cd train

# Stop training first (if running)
docker compose -f .docker-compose.yaml --profile distributed down

# Run benchmark with default checkpoint (conf.py RESUME_CHECKPOINT)
bash run_benchmark.sh

# Run benchmark with a specific checkpoint
bash run_benchmark.sh saved_models/v52-step70000/model.ckpt-resume.pkl

# Auto-restart training after benchmark
RESTART=1 bash run_benchmark.sh

# View results
python3 compare_benchmarks.py
python3 compare_benchmarks.py latest
python3 compare_benchmarks.py 0 1   # compare two runs
```

---

## How It Works

1. **Trigger**: `KAIWU_BENCHMARK_MODE=1` environment variable tells the workflow to run eval instead of training
2. **Stack**: Same Docker stack starts (learner sits idle, aisrv runs benchmark)
3. **Episodes**: 40 episodes run (4 rounds × 10 maps), each with fixed deterministic configs
4. **Logs**: Detailed per-step logs saved to `eval_logs/{session_id}/`
5. **Results**: JSON summary saved to `eval_results.json` (appends each run)
6. **Cleanup**: Stack stops after completion

---

## Test Scenarios (4 Rounds × 10 Maps = 40 Episodes)

| Round | Chargers | Robots | Max Steps | Battery | Description |
|-------|----------|--------|-----------|---------|-------------|
| round_1 | 4 | 3 | 1000 | 200 | Standard difficulty |
| round_2 | 3 | 4 | 1200 | 300 | More robots, more time |
| round_3 | 2 | 4 | 1600 | 200 | Few chargers, long game |
| round_4 | 2 | 4 | 2000 | 200 | Few chargers, very long game |

Each round runs on **all 10 public maps** (map 1-10), no randomness.

### Modifying Scenarios

Edit `code/agent_ppo/eval/benchmark.py`, find the `ROUNDS` list at the top:

```python
ROUNDS = [
    {
        "name": "round_1",
        "desc": "4 chargers / 3 robots / 1000 steps / 200 battery",
        "charger_count": 4,
        "robot_count": 3,
        "max_step": 1000,
        "battery_max": 200,
    },
    # ... add or modify rounds here
]
```

Also update `ROUND_DISPLAY` in `train/compare_benchmarks.py` for display labels.

---

## Output Files

### eval_results.json (host: `train/eval_results.json`)

Appended after each benchmark run. Contains all historical results:

```json
{
  "version": 3,
  "benchmarks": [
    {
      "timestamp": "20260415-213000",
      "checkpoint": "saved_models/v52-step70000/model.ckpt-resume.pkl",
      "git_commit": "abc1234",
      "elapsed_seconds": 450,
      "per_round": {
        "round_1": { "win_rate": 0.9, "avg_clean_score": 850, ... },
        "round_2": { "win_rate": 0.7, "avg_clean_score": 700, ... },
        ...
      },
      "overall": { "win_rate": 0.65, "avg_clean_score": 650, ... },
      "episodes": [ ... ]   // per-episode details
    }
  ]
}
```

### eval_logs/{session_id}/ (host: `train/eval_logs/`)

Each benchmark run creates a timestamped subfolder:

```
eval_logs/
└── 20260415-213000/
    ├── manifest.json              # scenario configs used
    ├── result.json                # full results (copy)
    ├── benchmark.log              # step-level progress log
    └── episodes/
        ├── round_1_map1.jsonl     # per-step details for each episode
        ├── round_1_map2.jsonl
        ├── ...
        └── round_4_map10.jsonl    # 40 files total
```

### Per-Step Log Format (episodes/*.jsonl)

Each line is one step:

```json
{
  "step": 100,
  "action": 3,
  "reward": 0.512,
  "total_reward": 45.3,
  "battery": 156,
  "battery_max": 200,
  "dirt_cleaned": 450,
  "total_dirt": 6800,
  "mode": 0,
  "charger_slack": 12.5,
  "nearest_npc_dist": 8.0,
  "invalid_move_count": 0
}
```

Use these to diagnose specific failures (battery death trajectory, collision patterns, charging behavior).

---

## Comparison Tool

```bash
# Show all historical benchmarks
python3 compare_benchmarks.py

# Show latest with per-round breakdown
python3 compare_benchmarks.py latest

# Compare two benchmarks side by side with deltas
python3 compare_benchmarks.py 0 1
```

Output example:
```
  Round              Metric        A        B    Delta
  R1: 4C/3R/1000    WR       0.900    1.000   +0.100
  R1: 4C/3R/1000    CS     850.000  920.000  +70.000
  R3: 2C/4R/1600    WR       0.300    0.500   +0.200
  OVERALL            WR       0.525    0.675   +0.150

  Verdict: IMPROVED
```

---

## A/B Testing Workflow

To determine if a code change actually improves the model:

1. **Save current checkpoint as baseline**
   ```bash
   cp code/model.ckpt-resume.pkl code/saved_models/baseline_v1.pkl
   ```

2. **Run benchmark against baseline**
   ```bash
   bash run_benchmark.sh saved_models/baseline_v1.pkl
   ```

3. **Make code changes** (reward shaping, architecture, hyperparameters, etc.)

4. **Train with new code** from the same starting checkpoint
   ```bash
   # Update RESUME_CHECKPOINT in conf.py to same baseline
   docker compose up -d --force-recreate
   # ... let it train for a while ...
   docker compose down
   ```

5. **Run benchmark against the new checkpoint**
   ```bash
   bash run_benchmark.sh model.ckpt-resume.pkl
   ```

6. **Compare**
   ```bash
   python3 compare_benchmarks.py 0 1
   ```

If WR and CS improve → code change was effective. If they regress → revert.

---

## Architecture

```
train_workflow.py                  eval/benchmark.py
┌──────────────────────┐          ┌──────────────────────┐
│ workflow() entry     │          │ run_benchmark()      │
│                      │          │                      │
│ if BENCHMARK_MODE:   │──4 lines─│ 40 fixed episodes    │
│   → benchmark.run()  │          │ No learner data      │
│   return             │          │ Detailed step logs   │
│                      │          │ JSON results         │
│ Normal training...   │          └──────────────────────┘
└──────────────────────┘

run_benchmark.sh          compare_benchmarks.py
┌──────────────────────┐  ┌──────────────────────┐
│ Stops training       │  │ Reads eval_results   │
│ Sets BENCHMARK_MODE  │  │ Per-round breakdown  │
│ Starts stack         │  │ Side-by-side compare │
│ Waits & copies logs  │  │ Verdict: IMPROVED/   │
│ Optional restart     │  │   REGRESSED/STABLE   │
└──────────────────────┘  └──────────────────────┘
```

---

## Key Files

| File | Purpose |
|------|---------|
| `code/agent_ppo/eval/benchmark.py` | Core benchmark logic, scenario definitions |
| `code/agent_ppo/workflow/train_workflow.py` | 4-line mode switch in workflow() |
| `train/run_benchmark.sh` | Shell entry point (start/stop/wait/copy) |
| `train/compare_benchmarks.py` | Results comparison tool |
| `train/eval_results.json` | Historical benchmark results |
| `train/eval_logs/` | Per-session detailed logs |
